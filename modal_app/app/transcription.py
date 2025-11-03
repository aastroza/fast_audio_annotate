"""Batch transcription with VAD segmentation in Modal."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import modal
import numpy as np

from .common import AUDIO_DIR, MODEL_DIR, RESULTS_DIR, app, audio_volume, model_cache, results_volume

# Audio processing constants
TARGET_LUFS = -23.0
ASR_SAMPLE_RATE = 16000

DEFAULT_SERVER_VAD = {
    "aggressiveness": 2,
    "frame_ms": 30,
    "min_speech_ms": 150,
    "min_silence_ms": 300,
    "max_chunk_ms": 30000,
    "padding_ms": 200,
}


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""

    aggressiveness: int = DEFAULT_SERVER_VAD["aggressiveness"]
    frame_ms: int = DEFAULT_SERVER_VAD["frame_ms"]
    min_speech_ms: int = DEFAULT_SERVER_VAD["min_speech_ms"]
    min_silence_ms: int = DEFAULT_SERVER_VAD["min_silence_ms"]
    max_chunk_ms: int = DEFAULT_SERVER_VAD["max_chunk_ms"]
    padding_ms: int = DEFAULT_SERVER_VAD["padding_ms"]


@dataclass
class WordTranscription:
    """Word-level transcription with timestamps."""

    start: Optional[float]
    end: Optional[float]
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text}


@dataclass
class SegmentTranscription:
    """Segment transcription with optional word timestamps."""

    start: float
    end: float
    text: str
    words: Optional[List[WordTranscription]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {"start": self.start, "end": self.end, "text": self.text}
        if self.words is not None:
            data["words"] = [word.to_dict() for word in self.words]
        return data


@dataclass
class FileTranscription:
    """Complete file transcription with segments."""

    audio_path: str
    segments: List[SegmentTranscription]
    language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "language": self.language,
            "segments": [segment.to_dict() for segment in self.segments],
        }


def loudness_normalize(y: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Normalize loudness to target_lufs using ITU-R BS.1770."""
    import pyloudnorm as pyln

    y = y.astype(np.float32, copy=False)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)
    y_norm = pyln.normalize.loudness(y, loudness, target_lufs)
    peak = np.max(np.abs(y_norm)) + 1e-8
    if peak > 1.0:
        y_norm = y_norm / peak
    return y_norm


def ensure_mono(y: np.ndarray) -> np.ndarray:
    """Convert stereo to mono if needed."""
    if y.ndim == 1:
        return y
    return np.mean(y, axis=0)


def resample_to(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    import resampy

    if orig_sr == target_sr:
        return y.astype(np.float32, copy=False)
    return resampy.resample(y, orig_sr, target_sr).astype(np.float32, copy=False)


def float_to_pcm16(y: np.ndarray) -> bytes:
    """Convert float audio to 16-bit PCM bytes."""
    y = np.clip(y, -1.0, 1.0)
    pcm16 = (y * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def vad_chunk_boundaries(y16: bytes, sr: int, cfg: VADConfig) -> List[Tuple[int, int]]:
    """
    Detect speech boundaries using WebRTC VAD.

    Args:
        y16: 16-bit PCM audio bytes
        sr: Sample rate (must be 8000, 16000, 32000, or 48000)
        cfg: VAD configuration

    Returns:
        List of (start_sample, end_sample) tuples
    """
    import webrtcvad

    vad = webrtcvad.Vad(cfg.aggressiveness)
    frame_len = int(sr * cfg.frame_ms / 1000)
    min_speech_frames = int(cfg.min_speech_ms / cfg.frame_ms)
    min_silence_frames = int(cfg.min_silence_ms / cfg.frame_ms)
    max_chunk_frames = int(cfg.max_chunk_ms / cfg.frame_ms)
    pad_frames = int(cfg.padding_ms / cfg.frame_ms)

    total_samples = len(y16) // 2
    samples = np.frombuffer(y16, dtype=np.int16)

    def frame_bytes(i: int) -> bytes:
        start = i * frame_len
        end = min(start + frame_len, total_samples)
        return samples[start:end].tobytes()

    boundaries: List[Tuple[int, int]] = []
    in_speech = False
    speech_start_frame = 0
    silence_run = 0
    speech_run = 0
    i = 0

    while i * frame_len < total_samples:
        fb = frame_bytes(i)
        if len(fb) < frame_len * 2:
            fb = fb + b"\x00" * (frame_len * 2 - len(fb))

        is_speech = vad.is_speech(fb, sr)
        if is_speech:
            speech_run += 1
            silence_run = 0
            if not in_speech and speech_run >= min_speech_frames:
                in_speech = True
                speech_start_frame = max(0, i - pad_frames)
        else:
            silence_run += 1
            if in_speech and silence_run >= min_silence_frames:
                end_frame = i + pad_frames
                start_f = speech_start_frame
                while end_frame - start_f > max_chunk_frames:
                    boundaries.append((start_f * frame_len, (start_f + max_chunk_frames) * frame_len))
                    start_f += max_chunk_frames
                boundaries.append((start_f * frame_len, end_frame * frame_len))
                in_speech = False
                speech_run = 0

        if in_speech and (i - speech_start_frame) >= max_chunk_frames:
            boundaries.append((speech_start_frame * frame_len, i * frame_len))
            in_speech = False
            speech_run = 0
            silence_run = 0

        i += 1

    if in_speech:
        end_frame = i
        start_f = speech_start_frame
        while end_frame - start_f > max_chunk_frames:
            boundaries.append((start_f * frame_len, (start_f + max_chunk_frames) * frame_len))
            start_f += max_chunk_frames
        boundaries.append((start_f * frame_len, end_frame * frame_len))

    return boundaries


def make_segments_from_vad(
    y: np.ndarray,
    sr: int,
    server_vad: Optional[Dict[str, Any]] = None,
) -> List[Tuple[np.ndarray, float, float]]:
    """
    Segment audio using VAD.

    Args:
        y: Audio waveform
        sr: Sample rate
        server_vad: VAD configuration override

    Returns:
        List of (audio_segment, start_time, end_time) tuples
    """
    cfg = VADConfig(**({} if server_vad is None else server_vad))

    y = ensure_mono(y)
    y = loudness_normalize(y, sr, TARGET_LUFS)

    y16k = resample_to(y, sr, ASR_SAMPLE_RATE)
    pcm16 = float_to_pcm16(y16k)

    boundaries = vad_chunk_boundaries(pcm16, ASR_SAMPLE_RATE, cfg)

    segments: List[Tuple[np.ndarray, float, float]] = []
    n = len(y16k)
    for start_samp, end_samp in boundaries:
        start_samp = max(0, min(n, start_samp))
        end_samp = max(0, min(n, end_samp))
        if end_samp <= start_samp:
            continue
        seg = y16k[start_samp:end_samp]
        start_t = start_samp / ASR_SAMPLE_RATE
        end_t = end_samp / ASR_SAMPLE_RATE
        segments.append((seg.astype(np.float32, copy=False), start_t, end_t))

    if not segments:
        segments = [(y16k, 0.0, len(y16k) / ASR_SAMPLE_RATE)]
    return segments


@app.cls(
    gpu="L40S",
    timeout=60 * 10,
    scaledown_window=5,
    max_containers=10,
    volumes={MODEL_DIR: model_cache},
)
class WhisperModel:
    """Remote Whisper inference with GPU acceleration."""

    model_name: str = modal.parameter()
    language: str = modal.parameter(default="")  # Empty string = auto-detect
    return_word_timestamps: bool = modal.parameter(default=False)
    batch_size: int = modal.parameter(default=8)

    @modal.enter()
    def load_model(self) -> None:
        """Load Whisper model on container startup."""
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        torch_dtype = torch.bfloat16

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to("cuda")

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device="cuda",
        )

        self.generate_kwargs: Dict[str, str] = {"task": "transcribe"}
        if self.language:
            self.generate_kwargs["language"] = self.language

    @modal.batched(max_batch_size=64, wait_ms=1000)
    def transcribe_segments(self, audio_segments: List[np.ndarray]) -> List[Dict[str, object]]:
        """
        Run batched Whisper inference on pre-chunked audio segments.

        Args:
            audio_segments: List of audio arrays (16kHz, mono, float32)

        Returns:
            List of transcription results with text and optional word timestamps
        """
        if not audio_segments:
            return []

        print(f"Received {len(audio_segments)} segments for transcription")
        print(f"First segment type: {type(audio_segments[0])}")
        
        # Convert all segments to numpy arrays if needed
        # Must maintain same length as input for Modal batching
        processed_segments = []
        for i, seg in enumerate(audio_segments):
            try:
                if isinstance(seg, list):
                    # Convert list to numpy array
                    seg = np.array(seg, dtype=np.float32)
                elif not isinstance(seg, np.ndarray):
                    # Convert other types to numpy array
                    seg = np.array(seg, dtype=np.float32)
                
                # Ensure 1D
                if len(seg.shape) > 1:
                    seg = seg.flatten()
                    
                processed_segments.append(seg)
            except Exception as e:
                print(f"Error processing segment {i}: {e}")
                print(f"Segment type: {type(seg)}")
                print(f"Segment content: {seg[:10] if hasattr(seg, '__len__') and len(seg) > 10 else seg}")
                # Add empty segment to maintain length consistency
                processed_segments.append(np.array([], dtype=np.float32))

        # Check if we have any valid segments
        if not processed_segments:
            print("No valid segments found after processing")
            return []

        print(f"Successfully processed {len(processed_segments)} segments")

        # Process segments individually but in batch using pipeline
        kwargs: Dict[str, object] = {
            "batch_size": min(len(processed_segments), self.batch_size),
            "return_timestamps": "word" if self.return_word_timestamps else False,
            "generate_kwargs": self.generate_kwargs,
        }

        # Convert list of arrays to format expected by pipeline
        # Pipeline expects each audio as separate input
        results = []
        batch_size = max(1, kwargs["batch_size"])  # Ensure batch_size is at least 1
        
        for i in range(0, len(processed_segments), batch_size):
            batch = processed_segments[i:i + batch_size]
            
            # Process each segment in the batch
            batch_results = []
            for audio_array in batch:
                try:
                    # Skip empty segments but still add a result
                    if len(audio_array) == 0:
                        batch_results.append({"text": "", "chunks": []})
                        continue
                    
                    # Process single audio segment
                    result = self.pipeline(
                        audio_array,
                        return_timestamps="word" if self.return_word_timestamps else False,
                        generate_kwargs=self.generate_kwargs,
                    )
                    batch_results.append(result)
                except Exception as e:
                    print(f"Error transcribing segment: {e}")
                    # Add empty result to maintain batch consistency
                    batch_results.append({"text": "", "chunks": []})
            
            results.extend(batch_results)

        # Ensure we return exactly the same number of results as inputs
        if len(results) != len(audio_segments):
            print(f"WARNING: Result count mismatch! Input: {len(audio_segments)}, Output: {len(results)}")
            # Pad with empty results if needed
            while len(results) < len(audio_segments):
                results.append({"text": "", "chunks": []})
            # Truncate if too many results
            results = results[:len(audio_segments)]

        return results


@app.function(
    volumes={
        AUDIO_DIR: audio_volume,
        RESULTS_DIR: results_volume,
    },
    timeout=60 * 60,  # 1 hour for large files
)
def transcribe_audio_file(
    audio_path: str,
    model_name: str,
    language: str = "",
    return_word_timestamps: bool = False,
    use_vad: bool = True,
    vad_config: Optional[Dict[str, Any]] = None,
) -> FileTranscription:
    """
    Transcribe a single audio file from Modal Volume.

    Args:
        audio_path: Path to audio file in Modal Volume (relative to AUDIO_DIR)
        model_name: Whisper model name (e.g., "openai/whisper-large-v3")
        language: Language code (empty string for auto-detect)
        return_word_timestamps: Whether to return word-level timestamps
        use_vad: Whether to use VAD segmentation
        vad_config: VAD configuration override

    Returns:
        FileTranscription with segments
    """
    import soundfile as sf

    # Load audio from Modal Volume
    full_path = Path(AUDIO_DIR) / audio_path
    if not full_path.exists():
        raise ValueError(f"Audio file not found in Modal Volume: {audio_path}")

    print(f"Loading audio: {audio_path}")
    
    # Try loading with soundfile first, fallback to ffmpeg for unsupported formats
    try:
        waveform, sample_rate = sf.read(str(full_path), always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
    except sf.LibsndfileError:
        # Fallback to ffmpeg for formats not supported by soundfile (like WebM)
        print(f"soundfile failed, using ffmpeg for: {audio_path}")
        import subprocess
        
        # Use ffmpeg to convert to WAV in memory
        cmd = [
            "ffmpeg", "-i", str(full_path), 
            "-f", "wav", "-acodec", "pcm_s16le", 
            "-ar", "16000", "-ac", "1", "-"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            # Load the WAV data from ffmpeg output
            import io
            waveform, sample_rate = sf.read(io.BytesIO(result.stdout), always_2d=False)
            waveform = np.asarray(waveform, dtype=np.float32)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Could not load audio file {audio_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error processing audio file {audio_path}: {e}")

    # Segment audio with VAD or process as single segment
    if use_vad:
        print(f"Segmenting audio with VAD...")
        segments = make_segments_from_vad(waveform, sample_rate, server_vad=vad_config)
        print(f"Created {len(segments)} segments")
    else:
        normalized = loudness_normalize(ensure_mono(waveform), sample_rate, TARGET_LUFS)
        resampled = resample_to(normalized, sample_rate, ASR_SAMPLE_RATE)
        segments = [(resampled, 0.0, len(resampled) / ASR_SAMPLE_RATE)]

    # Get Whisper model instance
    whisper_model = WhisperModel(
        model_name=model_name,
        language=language,
        return_word_timestamps=return_word_timestamps,
    )

    # Run batch transcription
    flat_audio = [seg for seg, _, _ in segments]
    print(f"Transcribing {len(flat_audio)} segments...")
    batch_out = whisper_model.transcribe_segments.remote(flat_audio)

    # Collect results
    collected_segments: List[SegmentTranscription] = []
    detected_language: Optional[str] = None

    for (_, start_t, end_t), out in zip(segments, batch_out):
        # Handle different output formats from the pipeline
        if isinstance(out, str):
            # Pipeline returned just the text
            text = out.strip()
            words = None
        elif isinstance(out, dict):
            # Pipeline returned a dictionary with metadata
            text = str(out.get("text", "")).strip()
            words: Optional[List[WordTranscription]] = None

            if return_word_timestamps and "chunks" in out:
                words = []
                for chunk in out.get("chunks", []):
                    timestamp = chunk.get("timestamp") or (None, None)
                    start, end = timestamp
                    if start is not None:
                        start = float(start) + start_t
                    if end is not None:
                        end = float(end) + start_t
                    words.append(WordTranscription(start=start, end=end, text=str(chunk.get("text", ""))))

            if detected_language is None:
                detected_language = str(out.get("language")) if out.get("language") else None
        else:
            # Unknown format, try to convert to string
            text = str(out).strip()
            words = None

        collected_segments.append(SegmentTranscription(start=float(start_t), end=float(end_t), text=text, words=words))

    return FileTranscription(audio_path=audio_path, segments=collected_segments, language=detected_language)


@app.function(
    volumes={
        AUDIO_DIR: audio_volume,
        RESULTS_DIR: results_volume,
    },
    timeout=60 * 60 * 2,  # 2 hours for batch processing
)
def batch_transcribe(
    model_name: str,
    language: str = "",
    return_word_timestamps: bool = False,
    use_vad: bool = True,
    vad_config: Optional[Dict[str, Any]] = None,
    audio_files: Optional[List[str]] = None,
) -> List[str]:
    """
    Batch transcribe all audio files in Modal Volume.

    Args:
        model_name: Whisper model name
        language: Language code (empty string for auto-detect)
        return_word_timestamps: Whether to return word-level timestamps
        use_vad: Whether to use VAD segmentation
        vad_config: VAD configuration override
        audio_files: Specific files to transcribe (None = all files)

    Returns:
        List of result file paths
    """
    audio_dir = Path(AUDIO_DIR)
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get list of audio files
    if audio_files is None:
        audio_extensions = {".webm", ".mp3", ".wav", ".ogg", ".m4a", ".flac"}
        all_files = []
        for ext in audio_extensions:
            all_files.extend(audio_dir.rglob(f"*{ext}"))
        audio_files = [str(p.relative_to(audio_dir)) for p in all_files]

    if not audio_files:
        print("No audio files found")
        return []

    print(f"Batch transcribing {len(audio_files)} files...")

    # Process files in parallel using .map()
    transcription_results = list(
        transcribe_audio_file.map(
            audio_files,
            [model_name] * len(audio_files),
            [language] * len(audio_files),
            [return_word_timestamps] * len(audio_files),
            [use_vad] * len(audio_files),
            [vad_config] * len(audio_files),
        )
    )

    # Save results
    result_paths = []
    for result in transcription_results:
        if result is None:
            continue

        # Create result file path
        audio_stem = Path(result.audio_path).stem
        result_file = results_dir / f"{audio_stem}_transcription.json"

        # Ensure parent directory exists
        result_file.parent.mkdir(parents=True, exist_ok=True)

        # Save transcription
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        result_paths.append(str(result_file.relative_to(results_dir)))
        print(f"Saved: {result_file.name}")

    # Commit results to volume
    results_volume.commit()

    print(f"Batch transcription complete: {len(result_paths)} files")

    return result_paths
