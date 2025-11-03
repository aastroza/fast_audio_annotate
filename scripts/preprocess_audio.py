#!/usr/bin/env python3
"""Preprocess audio files with Whisper and persist segment-level transcripts."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fast_audio_annotate.config import AppConfig, parse_app_config
from fast_audio_annotate.metadata import iter_audio_files
from fast_audio_annotate.transcription import WhisperTranscriber

from db_backend import DatabaseBackend


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
    parser.add_argument(
        "--database-url",
        dest="database_url",
        help="Override the database URL defined in the config or environment.",
    )
    return parser.parse_args()


def resolve_audio_directory(args: argparse.Namespace, config: AppConfig) -> Path:
    if args.audio_folder:
        return Path(args.audio_folder)
    return config.audio_path


def resolve_database_url(args: argparse.Namespace, config: AppConfig) -> Optional[str]:
    if args.database_url:
        return args.database_url
    return (
        config.database_url
        or os.environ.get("DATABASE_URL")
        or os.environ.get("NEON_DATABASE_URL")
    )


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


def get_username() -> str:
    return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


def main() -> None:
    args = parse_args()
    config = parse_app_config(args.config)

    audio_dir = resolve_audio_directory(args, config)
    output_dir = resolve_output_directory(args, audio_dir)
    model_name = resolve_model_name(args, config)
    language = resolve_language(args, config)
    chunking_strategy = "none" if args.no_vad else "auto"
    database_url = resolve_database_url(args, config)

    audio_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcriber = WhisperTranscriber(
        model_name,
        language=language,
        return_word_timestamps=args.word_timestamps,
        chunking_strategy=chunking_strategy,
        batch_size=args.batch_size,
    )

    db_backend = DatabaseBackend(audio_dir / "annotations.db", database_url)
    print(f"Using database backend: {db_backend.backend_label()}")
    username = get_username()

    audio_files = list(iter_audio_files(audio_dir))
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    for audio_path in audio_files:
        relative_path = audio_path.relative_to(audio_dir)
        relative_audio_path = relative_path.as_posix()
        output_path = output_dir / relative_path.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        existing_clip_count = db_backend.count_clips(relative_audio_path)
        transcript_exists = output_path.exists()

        if (transcript_exists or existing_clip_count) and not args.overwrite:
            reason_bits = []
            if transcript_exists:
                reason_bits.append("transcript already exists")
            if existing_clip_count:
                reason_bits.append(f"{existing_clip_count} clips already in database")
            reason = " and ".join(reason_bits)
            print(f"Skipping {relative_path} ({reason})")
            continue

        if existing_clip_count and args.overwrite:
            print(
                f"Clearing {existing_clip_count} existing clips for {relative_path} from database..."
            )
            db_backend.delete_clips_for_audio(relative_audio_path)

        print(f"Transcribing {relative_path} with {model_name}...")
        result = transcriber.transcribe_file(audio_path)

        data = result.to_dict()
        data.update(
            {
                "relative_audio_path": relative_audio_path,
                "model": model_name,
                "language": result.language or language,
                "duration": max((segment.end for segment in result.segments), default=0.0),
            }
        )

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

        print(f"Saved transcript to {output_path}")

        timestamp = datetime.now().isoformat()
        inserted = 0

        for segment in result.segments:
            clip_values = {
                "audio_path": relative_audio_path,
                "start_timestamp": float(segment.start),
                "end_timestamp": float(segment.end),
                "text": segment.text,
                "username": username,
                "timestamp": timestamp,
                "marked": False,
            }

            db_backend.create_clip(clip_values)
            inserted += 1

        print(
            f"Stored {inserted} clip{'s' if inserted != 1 else ''} for {relative_path} in database"
        )


if __name__ == "__main__":
    main()
