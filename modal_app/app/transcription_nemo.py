"""NeMo ASR with integrated VAD for batch transcription in Modal."""
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal
import numpy as np

from .common import AUDIO_DIR, MODEL_DIR, RESULTS_DIR, audio_volume, model_cache, nemo_image, results_volume

# Create separate app for NeMo transcription
nemo_app = modal.App("fast-audio-annotate-nemo", image=nemo_image)

# NeMo ASR configuration
ASR_SAMPLE_RATE = 16000


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


@nemo_app.cls(
    gpu="L40S",
    timeout=60 * 30,  # 30 minutes
    scaledown_window=5,
    max_containers=10,
    volumes={MODEL_DIR: model_cache},
)
class NeMoASRModel:
    """Remote NeMo ASR inference with integrated VAD."""

    model_name: str = modal.parameter()
    language: str = modal.parameter(default="es")  # Spanish by default
    vad_model_name: str = modal.parameter(default="vad_multilingual_marblenet")

    @modal.enter()
    def load_models(self) -> None:
        """Load NeMo ASR and VAD models on container startup."""
        import logging

        import nemo.collections.asr as nemo_asr
        import torch

        # Silence chatty logs from nemo
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        # Set compute dtype
        self._COMPUTE_DTYPE = torch.bfloat16

        # Load ASR model
        print(f"Loading ASR model: {self.model_name}")
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
        self.asr_model.to(self._COMPUTE_DTYPE)
        self.asr_model.eval()

        # Configure decoding strategy
        if hasattr(self.asr_model, 'cfg') and hasattr(self.asr_model.cfg, 'decoding'):
            if self.asr_model.cfg.decoding.strategy != "beam":
                self.asr_model.cfg.decoding.strategy = "greedy_batch"
                self.asr_model.change_decoding_strategy(self.asr_model.cfg.decoding)

        # Load VAD model for feature extraction
        print(f"Loading VAD model: {self.vad_model_name}")
        self.vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(self.vad_model_name)
        self.vad_model.to(self._COMPUTE_DTYPE)
        self.vad_model.eval()

        print("Models loaded successfully")

    @modal.method()
    def extract_features(self, audio_filepath: str) -> Dict[str, Any]:
        """
        Extract audio features using NeMo VAD model.

        Args:
            audio_filepath: Path to audio file in Modal Volume

        Returns:
            Dictionary with feature path and metadata
        """
        import soundfile as sf
        import torch

        # Load audio
        full_path = Path(AUDIO_DIR) / audio_filepath
        if not full_path.exists():
            raise ValueError(f"Audio file not found: {audio_filepath}")

        audio_array, sample_rate = sf.read(str(full_path), always_2d=False)
        audio_array = np.asarray(audio_array, dtype=np.float32)

        # Ensure mono
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=0)

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).to(self.vad_model.device)
        audio_length = torch.tensor([len(audio_array)]).to(self.vad_model.device)

        # Extract features using VAD preprocessor
        with torch.autocast("cuda", enabled=False, dtype=self._COMPUTE_DTYPE), torch.inference_mode(), torch.no_grad():
            processed_signal, processed_signal_length = self.vad_model.preprocessor(
                input_signal=audio_tensor, length=audio_length
            )

        # Save features to temporary file
        features = processed_signal.squeeze(0)[:, :processed_signal_length].cpu()

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pt") as f:
            torch.save(features, f)
            feature_path = f.name

        return {
            "feature_path": feature_path,
            "audio_filepath": audio_filepath,
            "duration": len(audio_array) / sample_rate,
        }

    @modal.method()
    def run_vad(
        self,
        feature_path: str,
        vad_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, float]]:
        """
        Run VAD on extracted features to detect speech segments.

        Args:
            feature_path: Path to saved feature tensor
            vad_config: VAD configuration parameters

        Returns:
            List of speech segments with start/end times
        """
        import torch

        # Default VAD config
        default_config = {
            "window_length_in_sec": 0.15,
            "shift_length_in_sec": 0.01,
            "smoothing": "median",
            "overlap": 0.5,
            "postprocessing": {
                "onset": 0.5,
                "offset": 0.5,
                "min_duration_on": 0.1,
                "min_duration_off": 0.3,
                "filter_speech_first": True,
            },
        }

        if vad_config:
            default_config.update(vad_config)

        config = default_config

        # Load features
        features = torch.load(feature_path)

        # Setup VAD test data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Create manifest for VAD
            manifest_data = {"audio_filepath": "dummy", "duration": 1000000, "offset": 0, "label": "infer"}
            f.write(json.dumps(manifest_data) + "\n")
            manifest_path = f.name

        test_config = {
            "vad_stream": True,
            "manifest_filepath": manifest_path,
            "labels": ["infer"],
            "num_workers": 1,
            "shuffle": False,
            "window_length_in_sec": config["window_length_in_sec"],
            "shift_length_in_sec": config["shift_length_in_sec"],
        }

        self.vad_model.setup_test_data(test_data_config=test_config, use_feat=True)

        # Run VAD inference
        vad_probs = []
        with torch.autocast("cuda", enabled=False, dtype=self._COMPUTE_DTYPE), torch.inference_mode(), torch.no_grad():
            # Process features through VAD model
            features_tensor = features.unsqueeze(0).to(self.vad_model.device)
            features_length = torch.tensor([features.shape[1]]).to(self.vad_model.device)

            log_probs = self.vad_model(processed_signal=features_tensor, processed_signal_length=features_length)
            probs = torch.softmax(log_probs, dim=-1)
            if len(probs.shape) == 3:
                probs = probs.squeeze(0)
            pred = probs[:, 1]  # Speech probability
            vad_probs = pred.cpu().tolist()

        # Convert VAD probabilities to segments using thresholding
        segments = self._vad_probs_to_segments(
            vad_probs,
            frame_length=config["shift_length_in_sec"],
            onset=config["postprocessing"]["onset"],
            offset=config["postprocessing"]["offset"],
            min_duration_on=config["postprocessing"]["min_duration_on"],
            min_duration_off=config["postprocessing"]["min_duration_off"],
        )

        # Cleanup
        os.unlink(feature_path)
        os.unlink(manifest_path)

        return segments

    def _vad_probs_to_segments(
        self,
        vad_probs: List[float],
        frame_length: float,
        onset: float,
        offset: float,
        min_duration_on: float,
        min_duration_off: float,
    ) -> List[Dict[str, float]]:
        """
        Convert VAD probabilities to speech segments.

        Args:
            vad_probs: List of VAD probabilities per frame
            frame_length: Duration of each frame in seconds
            onset: Threshold to start speech segment
            offset: Threshold to end speech segment
            min_duration_on: Minimum speech duration in seconds
            min_duration_off: Minimum silence duration in seconds

        Returns:
            List of segments with start/end times
        """
        segments = []
        in_speech = False
        speech_start = 0

        min_frames_on = int(min_duration_on / frame_length)
        min_frames_off = int(min_duration_off / frame_length)

        silence_run = 0
        current_segment_start = 0

        for i, prob in enumerate(vad_probs):
            time = i * frame_length

            if not in_speech:
                if prob >= onset:
                    # Start of potential speech
                    current_segment_start = i
                    in_speech = True
                    silence_run = 0
            else:
                if prob < offset:
                    silence_run += 1
                    if silence_run >= min_frames_off:
                        # End of speech segment
                        segment_length = i - current_segment_start - silence_run
                        if segment_length >= min_frames_on:
                            start_time = current_segment_start * frame_length
                            end_time = (i - silence_run) * frame_length
                            segments.append({"start": start_time, "end": end_time})
                        in_speech = False
                        silence_run = 0
                else:
                    silence_run = 0

        # Handle final segment
        if in_speech:
            segment_length = len(vad_probs) - current_segment_start
            if segment_length >= min_frames_on:
                start_time = current_segment_start * frame_length
                end_time = len(vad_probs) * frame_length
                segments.append({"start": start_time, "end": end_time})

        return segments

    @modal.method()
    def transcribe_segments(
        self,
        audio_filepath: str,
        segments: List[Dict[str, float]],
    ) -> List[SegmentTranscription]:
        """
        Transcribe audio segments using NeMo ASR.

        Args:
            audio_filepath: Path to audio file in Modal Volume
            segments: List of segments with start/end times

        Returns:
            List of segment transcriptions
        """
        import soundfile as sf
        import torch

        # Load full audio
        full_path = Path(AUDIO_DIR) / audio_filepath
        audio_array, sample_rate = sf.read(str(full_path), always_2d=False)
        audio_array = np.asarray(audio_array, dtype=np.float32)

        # Ensure mono
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=0)

        # Extract segment audio and save to temporary files
        temp_files = []
        for seg in segments:
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            segment_audio = audio_array[start_sample:end_sample]

            with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".wav") as f:
                sf.write(f, segment_audio, sample_rate)
                temp_files.append(f.name)

        # Transcribe segments in batch
        transcriptions = []
        if temp_files:
            with torch.autocast("cuda", enabled=False, dtype=self._COMPUTE_DTYPE), torch.inference_mode(), torch.no_grad():
                if "canary" in self.model_name.lower():
                    # Canary v2 uses pnc, v1 uses nopnc
                    pnc = "pnc" if "v2" in self.model_name.lower() else "nopnc"
                    batch_transcriptions = self.asr_model.transcribe(
                        temp_files,
                        batch_size=min(len(temp_files), 8),
                        verbose=False,
                        pnc=pnc,
                        num_workers=1,
                        source_lang=self.language,
                        target_lang=self.language,
                    )
                else:
                    batch_transcriptions = self.asr_model.transcribe(
                        temp_files, batch_size=min(len(temp_files), 8), num_workers=1
                    )

            # Process transcriptions
            if isinstance(batch_transcriptions, tuple) and len(batch_transcriptions) == 2:
                batch_transcriptions = batch_transcriptions[0]

            for seg, trans in zip(segments, batch_transcriptions):
                text = trans.text if hasattr(trans, "text") else str(trans)
                transcriptions.append(
                    SegmentTranscription(
                        start=seg["start"],
                        end=seg["end"],
                        text=text.strip(),
                        words=None,  # NeMo doesn't provide word timestamps by default
                    )
                )

        # Cleanup temp files
        for temp_file in temp_files:
            os.unlink(temp_file)

        return transcriptions


@nemo_app.function(
    volumes={
        AUDIO_DIR: audio_volume,
        RESULTS_DIR: results_volume,
    },
    timeout=60 * 60,  # 1 hour for large files
)
def transcribe_audio_file_nemo(
    audio_path: str,
    model_name: str,
    language: str = "es",
    vad_model_name: str = "vad_multilingual_marblenet",
    vad_config: Optional[Dict[str, Any]] = None,
) -> FileTranscription:
    """
    Transcribe a single audio file using NeMo ASR + VAD.

    Args:
        audio_path: Path to audio file in Modal Volume (relative to AUDIO_DIR)
        model_name: NeMo model name (e.g., "nvidia/parakeet-tdt-0.6b" or "nvidia/canary-1b")
        language: Language code for transcription
        vad_model_name: NeMo VAD model name
        vad_config: VAD configuration override

    Returns:
        FileTranscription with segments
    """
    print(f"Transcribing with NeMo: {audio_path}")
    print(f"ASR Model: {model_name}")
    print(f"VAD Model: {vad_model_name}")

    # Initialize NeMo model
    nemo_model = NeMoASRModel(
        model_name=model_name,
        language=language,
        vad_model_name=vad_model_name,
    )

    # Step 1: Extract audio features
    print("Extracting audio features...")
    feature_data = nemo_model.extract_features.remote(audio_path)

    # Step 2: Run VAD to detect speech segments
    print("Running VAD to detect speech segments...")
    segments = nemo_model.run_vad.remote(
        feature_path=feature_data["feature_path"],
        vad_config=vad_config,
    )

    print(f"Detected {len(segments)} speech segments")

    # If no segments detected, use full audio
    if not segments:
        print("No speech detected, using full audio duration")
        segments = [{"start": 0.0, "end": feature_data["duration"]}]

    # Step 3: Transcribe detected segments
    print("Transcribing speech segments...")
    segment_transcriptions = nemo_model.transcribe_segments.remote(audio_path, segments)

    return FileTranscription(
        audio_path=audio_path,
        segments=segment_transcriptions,
        language=language,
    )


@nemo_app.function(
    volumes={
        AUDIO_DIR: audio_volume,
        RESULTS_DIR: results_volume,
    },
    timeout=60 * 60 * 2,  # 2 hours for batch processing
)
def batch_transcribe_nemo(
    model_name: str,
    language: str = "es",
    vad_model_name: str = "vad_multilingual_marblenet",
    vad_config: Optional[Dict[str, Any]] = None,
    audio_files: Optional[List[str]] = None,
) -> List[str]:
    """
    Batch transcribe all audio files using NeMo ASR + VAD.

    Args:
        model_name: NeMo ASR model name
        language: Language code for transcription
        vad_model_name: NeMo VAD model name
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

    print(f"Batch transcribing {len(audio_files)} files with NeMo ASR + VAD...")

    # Process files in parallel using .map()
    transcription_results = list(
        transcribe_audio_file_nemo.map(
            audio_files,
            [model_name] * len(audio_files),
            [language] * len(audio_files),
            [vad_model_name] * len(audio_files),
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
