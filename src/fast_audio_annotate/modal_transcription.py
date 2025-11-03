"""Modal-backed Whisper transcription utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import modal
import numpy as np

from .transcription import (
    ASR_SAMPLE_RATE,
    TARGET_LUFS,
    FileTranscription,
    SegmentTranscription,
    WordTranscription,
    ensure_mono,
    loudness_normalize,
    make_segments_from_vad,
    resample_to,
)


MODEL_DIR = "/model"


@dataclass
class ModalSettings:
    """Configuration for instantiating the remote Whisper model."""

    model_name: str
    language: Optional[str]
    return_word_timestamps: bool
    batch_size: int


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.47.1",
        "hf-transfer==0.1.8",
        "huggingface_hub==0.27.0",
        "librosa==0.10.2",
        "soundfile==0.12.1",
        "accelerate==1.2.1",
        "datasets==3.2.0",
        "pyloudnorm==0.1.1",
        "webrtcvad==2.0.10",
        "resampy==0.4.3",
        "numpy==1.26.4",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": MODEL_DIR})
)

model_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

app = modal.App(
    "fast-audio-annotate-transcriber",
    image=image,
    volumes={MODEL_DIR: model_cache},
)


@app.cls(gpu="a10g", max_containers=10)
class WhisperModel:
    """Remote Whisper inference that mirrors the local :class:`WhisperTranscriber`."""

    def __init__(
        self,
        settings: ModalSettings,
    ) -> None:
        self.settings = settings

    @modal.enter()
    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        torch_dtype = torch.float16

        self.processor = AutoProcessor.from_pretrained(self.settings.model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.settings.model_name,
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

        self.generate_kwargs: Dict[str, Union[str, bool]] = {"task": "transcribe"}
        if self.settings.language:
            self.generate_kwargs["language"] = self.settings.language

    @modal.batched(max_batch_size=64, wait_ms=1000)
    def transcribe_segments(self, audio_segments: List[np.ndarray]) -> List[Dict[str, object]]:
        """Run batched Whisper inference on pre-chunked audio segments."""

        if not audio_segments:
            return []

        kwargs: Dict[str, object] = {
            "sampling_rate": ASR_SAMPLE_RATE,
            "batch_size": min(len(audio_segments), self.settings.batch_size),
            "return_timestamps": "word" if self.settings.return_word_timestamps else False,
            "generate_kwargs": self.generate_kwargs,
        }

        result = self.pipeline(audio_segments, **kwargs)
        if isinstance(result, dict):
            return [result]
        return list(result)


class ModalWhisperTranscriber:
    """Client wrapper that delegates Whisper inference to Modal."""

    def __init__(
        self,
        model_name: str,
        *,
        language: Optional[str] = None,
        return_word_timestamps: bool = False,
        chunking_strategy: Union[str, Dict[str, object]] = "auto",
        batch_size: int = 8,
    ) -> None:
        self.chunking_strategy = chunking_strategy
        self.settings = ModalSettings(
            model_name=model_name,
            language=language,
            return_word_timestamps=return_word_timestamps,
            batch_size=batch_size,
        )
        self._model = WhisperModel(self.settings)

    @staticmethod
    def load_audio(audio_path: Path) -> Tuple[np.ndarray, int]:
        import soundfile as sf

        waveform, sample_rate = sf.read(str(audio_path), always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        return waveform, int(sample_rate)

    def _segment_audio(self, waveform: np.ndarray, sr: int) -> List[Tuple[np.ndarray, float, float]]:
        if isinstance(self.chunking_strategy, str) and self.chunking_strategy == "auto":
            return make_segments_from_vad(waveform, sr, server_vad=None)
        if isinstance(self.chunking_strategy, dict):
            return make_segments_from_vad(waveform, sr, server_vad=self.chunking_strategy)

        normalized = loudness_normalize(ensure_mono(waveform), sr, TARGET_LUFS)
        resampled = resample_to(normalized, sr, ASR_SAMPLE_RATE)
        return [(resampled, 0.0, len(resampled) / ASR_SAMPLE_RATE)]

    def _collect_segments(
        self,
        segments: List[Tuple[np.ndarray, float, float]],
        batch_out: List[Dict[str, object]],
    ) -> List[SegmentTranscription]:
        collected_segments: List[SegmentTranscription] = []

        for (_, start_t, end_t), out in zip(segments, batch_out):
            text = str(out.get("text", "")).strip()
            words: Optional[List[WordTranscription]] = None
            if self.settings.return_word_timestamps and "chunks" in out:
                words = []
                for chunk in out.get("chunks", []):
                    timestamp = chunk.get("timestamp") or (None, None)
                    start, end = timestamp
                    if start is not None:
                        start = float(start) + start_t
                    if end is not None:
                        end = float(end) + start_t
                    words.append(
                        WordTranscription(start=start, end=end, text=str(chunk.get("text", "")))
                    )

            collected_segments.append(
                SegmentTranscription(start=float(start_t), end=float(end_t), text=text, words=words)
            )

        return collected_segments

    def transcribe_file(self, audio_path: Path) -> FileTranscription:
        waveform, sample_rate = self.load_audio(audio_path)
        segments = self._segment_audio(waveform, sample_rate)

        flat_audio = [seg for seg, _, _ in segments]
        batch_out = self._model.transcribe_segments.remote(flat_audio)

        collected_segments = self._collect_segments(segments, batch_out)
        detected_language: Optional[str] = None
        if batch_out:
            first = batch_out[0]
            detected_language = str(first.get("language")) if first.get("language") else None

        return FileTranscription(audio_path=audio_path, segments=collected_segments, language=detected_language)


def transcribe_directory(
    audio_files: Iterable[Path],
    transcriber: ModalWhisperTranscriber,
) -> Iterator[FileTranscription]:
    for audio_path in audio_files:
        yield transcriber.transcribe_file(audio_path)

