"""Whisper transcription helpers for preprocessing audio datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union
import numpy as np

TARGET_LUFS = -23.0
ASR_SAMPLE_RATE = 16000

DEFAULT_CHUNKING_STRATEGY: Union[str, Dict[str, Any]] = "auto"

DEFAULT_SERVER_VAD = {
    "aggressiveness": 2,
    "frame_ms": 30,
    "min_speech_ms": 150,
    "min_silence_ms": 300,
    "max_chunk_ms": 30000,
    "padding_ms": 200,
}


def loudness_normalize(y: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Normalize loudness to ``target_lufs`` using ITU-R BS.1770."""

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
    if y.ndim == 1:
        return y
    return np.mean(y, axis=0)


def resample_to(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    import resampy

    if orig_sr == target_sr:
        return y.astype(np.float32, copy=False)
    return resampy.resample(y, orig_sr, target_sr).astype(np.float32, copy=False)


def float_to_pcm16(y: np.ndarray) -> bytes:
    y = np.clip(y, -1.0, 1.0)
    pcm16 = (y * 32767.0).astype(np.int16)
    return pcm16.tobytes()


@dataclass
class VADConfig:
    aggressiveness: int = DEFAULT_SERVER_VAD["aggressiveness"]
    frame_ms: int = DEFAULT_SERVER_VAD["frame_ms"]
    min_speech_ms: int = DEFAULT_SERVER_VAD["min_speech_ms"]
    min_silence_ms: int = DEFAULT_SERVER_VAD["min_silence_ms"]
    max_chunk_ms: int = DEFAULT_SERVER_VAD["max_chunk_ms"]
    padding_ms: int = DEFAULT_SERVER_VAD["padding_ms"]


def vad_chunk_boundaries(y16: bytes, sr: int, cfg: VADConfig) -> List[Tuple[int, int]]:
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


@dataclass
class WordTranscription:
    start: Optional[float]
    end: Optional[float]
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text}


@dataclass
class SegmentTranscription:
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
    audio_path: Path
    segments: List[SegmentTranscription]
    language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_path": str(self.audio_path),
            "language": self.language,
            "segments": [segment.to_dict() for segment in self.segments],
        }


class WhisperTranscriber:
    """Utility class that wraps a Whisper pipeline with VAD-based chunking."""

    def __init__(
        self,
        model_name: str,
        *,
        language: Optional[str] = None,
        return_word_timestamps: bool = False,
        chunking_strategy: Union[str, Dict[str, Any]] = DEFAULT_CHUNKING_STRATEGY,
        batch_size: int = 8,
    ) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.model_name = model_name
        self.language = language
        self.return_word_timestamps = return_word_timestamps
        self.chunking_strategy = chunking_strategy
        self.batch_size = batch_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)

        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
        )

        self.generate_kwargs: Dict[str, Any] = {"task": "transcribe"}
        if language:
            self.generate_kwargs["language"] = language

    @staticmethod
    def load_audio(audio_path: Path) -> Tuple[np.ndarray, int]:
        import soundfile as sf

        waveform, sample_rate = sf.read(str(audio_path), always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        return waveform, int(sample_rate)

    def _run_pipeline(self, audio_segments: List[np.ndarray]) -> List[Dict[str, Any]]:
        if not audio_segments:
            return []

        kwargs: Dict[str, Any] = {
            "sampling_rate": ASR_SAMPLE_RATE,
            "batch_size": min(len(audio_segments), self.batch_size),
            "return_timestamps": "word" if self.return_word_timestamps else False,
            "generate_kwargs": self.generate_kwargs,
        }
        result = self.pipeline(audio_segments, **kwargs)
        if isinstance(result, dict):
            return [result]
        return list(result)

    def _segment_audio(self, waveform: np.ndarray, sr: int) -> List[Tuple[np.ndarray, float, float]]:
        if isinstance(self.chunking_strategy, str) and self.chunking_strategy == "auto":
            return make_segments_from_vad(waveform, sr, server_vad=None)
        if isinstance(self.chunking_strategy, dict):
            return make_segments_from_vad(waveform, sr, server_vad=self.chunking_strategy)

        normalized = loudness_normalize(ensure_mono(waveform), sr, TARGET_LUFS)
        resampled = resample_to(normalized, sr, ASR_SAMPLE_RATE)
        return [(resampled, 0.0, len(resampled) / ASR_SAMPLE_RATE)]

    def transcribe_file(self, audio_path: Path) -> FileTranscription:
        waveform, sample_rate = self.load_audio(audio_path)
        segments = self._segment_audio(waveform, sample_rate)

        flat_audio = [seg for seg, _, _ in segments]
        batch_out = self._run_pipeline(flat_audio)

        collected_segments: List[SegmentTranscription] = []
        detected_language: Optional[str] = None

        for (_, start_t, end_t), out in zip(segments, batch_out):
            text = out.get("text", "").strip()
            words: Optional[List[WordTranscription]] = None
            if self.return_word_timestamps and "chunks" in out:
                words = []
                for chunk in out.get("chunks", []):
                    timestamp = chunk.get("timestamp") or (None, None)
                    start, end = timestamp
                    if start is not None:
                        start = float(start) + start_t
                    if end is not None:
                        end = float(end) + start_t
                    words.append(WordTranscription(start=start, end=end, text=chunk.get("text", "")))
            if detected_language is None:
                detected_language = out.get("language")
            collected_segments.append(
                SegmentTranscription(start=float(start_t), end=float(end_t), text=text, words=words)
            )

        return FileTranscription(audio_path=audio_path, segments=collected_segments, language=detected_language)


def transcribe_directory(
    audio_files: Iterable[Path],
    transcriber: WhisperTranscriber,
) -> Iterator[FileTranscription]:
    for audio_path in audio_files:
        yield transcriber.transcribe_file(audio_path)
