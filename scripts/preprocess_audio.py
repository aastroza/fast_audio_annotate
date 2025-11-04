#!/usr/bin/env python3
"""Transcribe audio with OpenAI Whisper and store segment-level results."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".oga",
    ".opus",
    ".wav",
    ".webm",
}

from db_backend import DatabaseBackend  # noqa: E402  pylint: disable=wrong-import-position


@dataclass(frozen=True)
class TranscriptionTask:
    """Container with the metadata required to process an audio file."""

    audio_path: Path
    relative_audio_path: str
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-folder",
        dest="audio_folder",
        help="Path to the directory containing audio files (defaults to config.yaml or ./audio).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Optional YAML configuration file with defaults.",
    )
    parser.add_argument(
        "--database-url",
        dest="database_url",
        help="Postgres/Neon connection string (overrides config/env).",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        help="OpenAI Whisper model to use (defaults to config or whisper-1).",
    )
    parser.add_argument(
        "--language",
        dest="language",
        help="Language hint for transcription (e.g. es, en).",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        help="Directory where verbose JSON transcripts will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate clips even if they already exist in the database.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without modifying the database (still writes JSON if enabled).",
    )
    parser.add_argument(
        "--transcribe-workers",
        type=int,
        dest="transcribe_workers",
        help="Number of parallel workers used for transcription (defaults to min(available CPU, 4)).",
    )
    parser.add_argument(
        "--db-workers",
        type=int,
        dest="db_workers",
        help="Number of parallel workers used for database inserts (defaults to min(available CPU, 4)).",
    )
    return parser.parse_args()


def load_simple_yaml(path: str) -> Dict[str, Any]:
    """Read a tiny subset of YAML consisting of key: value pairs.

    The project config file uses a very small subset of YAML, so we can parse it
    without adding PyYAML as a dependency. Only the keys required by this script
    are returned.
    """

    config_path = Path(path)
    if not config_path.exists():
        return {}

    data: Dict[str, Any] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if value.lower() in {"null", "none", ""}:
            data[key] = None
        else:
            data[key] = value
    return data


def resolve_audio_directory(args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    candidate = args.audio_folder or config.get("audio_folder") or "audio"
    audio_dir = Path(candidate)
    if not audio_dir.is_absolute():
        audio_dir = (ROOT_DIR / audio_dir).resolve()
    return audio_dir


def resolve_model_name(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    return args.model_name or config.get("whisper_model") or "whisper-1"


def resolve_language(args: argparse.Namespace, config: Dict[str, Any]) -> Optional[str]:
    language = args.language or config.get("transcription_language")
    if isinstance(language, str) and language.lower() == "auto":
        return None
    return language


def resolve_database_url(args: argparse.Namespace, config: Dict[str, Any]) -> Optional[str]:
    return (
        args.database_url
        or config.get("database_url")
        or os.environ.get("DATABASE_URL")
        or os.environ.get("NEON_DATABASE_URL")
    )


def resolve_output_directory(args: argparse.Namespace, audio_dir: Path) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            return (ROOT_DIR / output_dir).resolve()
        return output_dir
    return audio_dir / "transcriptions"


def iter_audio_files(audio_dir: Path) -> Iterator[Path]:
    for path in sorted(audio_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def determine_worker_count(requested: Optional[int], task_count: int) -> int:
    """Return a sensible worker count given a user hint and pending tasks."""

    if requested is not None and requested > 0:
        return requested

    if task_count <= 0:
        return 1

    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count, task_count))


def create_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required to use Whisper API.")
    return OpenAI(api_key=api_key)


def coerce_segment_value(segment: Any, key: str, default: Any) -> Any:
    if isinstance(segment, dict):
        return segment.get(key, default)
    return getattr(segment, key, default)


def normalize_segments(transcription: Any) -> List[Dict[str, Any]]:
    raw_segments: Optional[Iterable[Any]] = None
    if isinstance(transcription, dict):
        raw_segments = transcription.get("segments")
    else:
        raw_segments = getattr(transcription, "segments", None)

    segments: List[Dict[str, Any]] = []
    if raw_segments:
        for index, segment in enumerate(raw_segments):
            start = coerce_segment_value(segment, "start", None)
            end = coerce_segment_value(segment, "end", None)
            text = coerce_segment_value(segment, "text", "")
            if start is None or end is None:
                continue
            segments.append(
                {
                    "id": coerce_segment_value(segment, "id", index),
                    "start": float(start),
                    "end": float(end),
                    "text": str(text).strip(),
                }
            )

    text_value = (
        transcription.get("text")
        if isinstance(transcription, dict)
        else getattr(transcription, "text", "")
    )

    if not segments and text_value:
        segments.append({"id": 0, "start": 0.0, "end": 0.0, "text": str(text_value).strip()})

    return segments


def transcription_to_payload(
    transcription: Any,
    relative_audio_path: str,
    model_name: str,
    language_hint: Optional[str],
) -> Dict[str, Any]:
    segments = normalize_segments(transcription)
    language_value = (
        transcription.get("language")
        if isinstance(transcription, dict)
        else getattr(transcription, "language", None)
    )
    text_value = (
        transcription.get("text")
        if isinstance(transcription, dict)
        else getattr(transcription, "text", "")
    )

    return {
        "model": model_name,
        "language": language_value or language_hint,
        "relative_audio_path": relative_audio_path,
        "text": text_value,
        "duration": max((segment["end"] for segment in segments), default=0.0),
        "segments": segments,
    }


def write_transcript_json(output_path: Path, payload: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def process_transcription_task(
    task: TranscriptionTask, model_name: str, language: Optional[str]
) -> Dict[str, Any]:
    """Run Whisper on ``task`` and return the serialized payload."""

    print(f"Transcribing {task.relative_audio_path} with {model_name} via OpenAI Whisper...")
    client = create_openai_client()
    transcription = transcribe_audio(client, task.audio_path, model_name, language)
    payload = transcription_to_payload(
        transcription,
        task.relative_audio_path,
        model_name,
        language,
    )
    write_transcript_json(task.output_path, payload)
    print(f"Saved transcript to {task.output_path}")
    return payload


def store_segments(
    db_backend: DatabaseBackend,
    relative_audio_path: str,
    segments: List[Dict[str, Any]],
    username: str,
    timestamp: str,
) -> int:
    """Persist ``segments`` for an audio file and return the number inserted."""

    clip_values = [
        {
            "audio_path": relative_audio_path,
            "start_timestamp": segment["start"],
            "end_timestamp": segment["end"],
            "text": segment["text"],
            "username": username,
            "timestamp": timestamp,
            "marked": False,
            "human_reviewed": False,
        }
        for segment in segments
    ]
    return db_backend.create_clips_bulk(clip_values)


def transcribe_audio(
    client: OpenAI,
    audio_path: Path,
    model_name: str,
    language: Optional[str],
) -> Any:
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"],
    }
    if language:
        kwargs["language"] = language

    with audio_path.open("rb") as audio_file:
        return client.audio.transcriptions.create(file=audio_file, **kwargs)


def get_username() -> str:
    return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


def main() -> None:
    args = parse_args()
    config_data = load_simple_yaml(args.config)

    audio_dir = resolve_audio_directory(args, config_data)
    output_dir = resolve_output_directory(args, audio_dir)
    model_name = resolve_model_name(args, config_data)
    language = resolve_language(args, config_data)
    database_url = resolve_database_url(args, config_data)

    audio_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_backend = DatabaseBackend(audio_dir / "annotations.db", database_url)
    print(f"Using database backend: {db_backend.backend_label()}")

    username = get_username()
    audio_files = list(iter_audio_files(audio_dir))
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    tasks: List[TranscriptionTask] = []
    for audio_path in audio_files:
        relative_path = audio_path.relative_to(audio_dir)
        relative_audio_path = relative_path.as_posix()
        output_path = output_dir / relative_path.with_suffix(".json")

        existing_clip_count = db_backend.count_clips(relative_audio_path)
        if existing_clip_count and not args.overwrite:
            print(
                f"Skipping {relative_audio_path} ({existing_clip_count} clips already stored)"
            )
            continue

        if existing_clip_count and args.overwrite and not args.dry_run:
            print(f"Clearing {existing_clip_count} existing clips for {relative_audio_path}...")
            db_backend.delete_clips_for_audio(relative_audio_path)

        tasks.append(
            TranscriptionTask(
                audio_path=audio_path,
                relative_audio_path=relative_audio_path,
                output_path=output_path,
            )
        )

    if not tasks:
        print("All audio files are already processed. Nothing to do.")
        return

    transcribe_workers = determine_worker_count(args.transcribe_workers, len(tasks))
    print(
        f"Processing {len(tasks)} audio file{'s' if len(tasks) != 1 else ''} "
        f"with {transcribe_workers} transcription worker{'s' if transcribe_workers != 1 else ''}."
    )

    db_executor: Optional[ThreadPoolExecutor] = None
    if not args.dry_run:
        db_workers = determine_worker_count(args.db_workers, len(tasks))
        db_executor = ThreadPoolExecutor(max_workers=db_workers)

    db_futures: Dict[Future[int], str] = {}

    try:
        with ThreadPoolExecutor(max_workers=transcribe_workers) as transcription_executor:
            future_to_task = {
                transcription_executor.submit(
                    process_transcription_task,
                    task,
                    model_name,
                    language,
                ): task
                for task in tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    payload = future.result()
                except Exception as exc:  # pragma: no cover - network errors
                    print(f"Failed to transcribe {task.relative_audio_path}: {exc}")
                    continue

                if args.dry_run or not db_executor:
                    continue

                segments = payload.get("segments", [])
                if not segments:
                    print(f"Stored 0 clips for {task.relative_audio_path} in database")
                    continue

                timestamp = datetime.now().isoformat()
                db_future = db_executor.submit(
                    store_segments,
                    db_backend,
                    task.relative_audio_path,
                    segments,
                    username,
                    timestamp,
                )
                db_futures[db_future] = task.relative_audio_path

        if db_executor:
            for future in as_completed(db_futures):
                relative_audio_path = db_futures[future]
                try:
                    inserted = future.result()
                except Exception as exc:  # pragma: no cover - database errors
                    print(f"Failed to store clips for {relative_audio_path}: {exc}")
                else:
                    print(
                        f"Stored {inserted} clip{'s' if inserted != 1 else ''} "
                        f"for {relative_audio_path} in database"
                    )
    finally:
        if db_executor:
            db_executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
