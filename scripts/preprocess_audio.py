#!/usr/bin/env python3
"""Preprocess audio files with Whisper and export segment-level transcripts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fast_audio_annotate.config import AppConfig, parse_app_config
from fast_audio_annotate.metadata import iter_audio_files
from fast_audio_annotate.transcription import WhisperTranscriber


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="./config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--audio-folder", dest="audio_folder", help="Override the audio folder defined in the config.")
    parser.add_argument("--output", dest="output_dir", help="Directory where JSON transcripts will be written.")
    parser.add_argument("--model", dest="model_name", help="Whisper model name to use.")
    parser.add_argument("--language", dest="language", help="Force a transcription language (e.g. 'es').")
    parser.add_argument("--batch-size", type=int, default=8, help="Maximum batch size for Whisper inference.")
    parser.add_argument("--word-timestamps", action="store_true", help="Include word-level timestamps in the output.")
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD-based chunking and transcribe each file as a single block.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing transcript files instead of skipping them.",
    )
    return parser.parse_args()


def resolve_audio_directory(args: argparse.Namespace, config: AppConfig) -> Path:
    if args.audio_folder:
        return Path(args.audio_folder)
    return config.audio_path


def resolve_output_directory(args: argparse.Namespace, audio_dir: Path) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return audio_dir / "transcriptions"


def resolve_model_name(args: argparse.Namespace, config: AppConfig) -> str:
    return args.model_name or config.whisper_model


def resolve_language(args: argparse.Namespace, config: AppConfig) -> Optional[str]:
    if args.language:
        return None if args.language.lower() == "auto" else args.language
    return config.transcription_language


def main() -> None:
    args = parse_args()
    config = parse_app_config(args.config)

    audio_dir = resolve_audio_directory(args, config)
    output_dir = resolve_output_directory(args, audio_dir)
    model_name = resolve_model_name(args, config)
    language = resolve_language(args, config)
    chunking_strategy = "none" if args.no_vad else "auto"

    audio_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcriber = WhisperTranscriber(
        model_name,
        language=language,
        return_word_timestamps=args.word_timestamps,
        chunking_strategy=chunking_strategy,
        batch_size=args.batch_size,
    )

    audio_files = list(iter_audio_files(audio_dir))
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    for audio_path in audio_files:
        relative_path = audio_path.relative_to(audio_dir)
        output_path = output_dir / relative_path.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not args.overwrite:
            print(f"Skipping {relative_path} (transcript already exists)")
            continue

        print(f"Transcribing {relative_path} with {model_name}...")
        result = transcriber.transcribe_file(audio_path)

        data = result.to_dict()
        data.update(
            {
                "relative_audio_path": relative_path.as_posix(),
                "model": model_name,
                "language": result.language or language,
                "duration": max((segment.end for segment in result.segments), default=0.0),
            }
        )

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

        print(f"Saved transcript to {output_path}")


if __name__ == "__main__":
    main()
