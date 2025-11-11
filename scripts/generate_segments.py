#!/usr/bin/env python3
"""Pre-generate trimmed audio segments for all clips in the database."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import textwrap
from typing import Iterable, Optional

import yaml

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db_backend import ClipRecord, DatabaseBackend
from src.fast_audio_annotate.config import AppConfig
from src.fast_audio_annotate.segments import (
    SegmentGenerationError,
    generate_segment,
    locate_ffmpeg,
    remove_previous_segment,
)


DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_PADDING_SECONDS = 2.0
DEFAULT_SEGMENT_SUBDIR = "segments"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-clip audio segments ahead of time.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            The script iterates over every clip stored in the annotations database and
            creates an audio snippet that includes a configurable amount of context
            around each clip. Existing snippets are reused unless --force is
            provided. All generated files live inside the <audio_folder>/segments
            directory so they can be served by the annotation UI.
            """
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML configuration file (default: %(default)s).",
    )
    parser.add_argument(
        "--audio-folder",
        help="Override the audio_folder from the configuration file.",
    )
    parser.add_argument(
        "--database-url",
        help="Override the database URL from the configuration file.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING_SECONDS,
        help="Seconds of context to include before and after each clip.",
    )
    parser.add_argument(
        "--segment-subdir",
        default=DEFAULT_SEGMENT_SUBDIR,
        help="Directory (relative to the audio folder) where segments are stored.",
    )
    parser.add_argument(
        "--ffmpeg-binary",
        help="Path to the ffmpeg executable (defaults to auto-detection).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate segments even if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the actions without creating or updating any files.",
    )
    return parser.parse_args(argv)


def load_config(path: Path) -> AppConfig:
    data = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    return AppConfig(**data)


@dataclass
class SegmentJob:
    clip: ClipRecord
    source_path: Path


def collect_jobs(db: DatabaseBackend, audio_root: Path) -> list[SegmentJob]:
    clips = db.fetch_all_clips()
    jobs: list[SegmentJob] = []

    for clip in clips:
        source_path = audio_root / clip.audio_path
        if not source_path.exists():
            print(f"⚠️  Skipping clip {clip.id}: audio source missing ({source_path}).")
            continue
        jobs.append(SegmentJob(clip=clip, source_path=source_path))

    return jobs


def run(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    config_path = Path(args.config)
    app_config = load_config(config_path)

    if args.audio_folder:
        app_config.audio_folder = args.audio_folder
    if args.database_url:
        app_config.database_url = args.database_url

    if app_config.audio_folder.startswith(("http://", "https://")):
        print("Error: audio_folder points to a remote location. Download the audio locally first.")
        return 1

    audio_root = Path(app_config.audio_folder)
    if not audio_root.is_absolute():
        # Always resolve relative paths from the project root, not from current working directory
        audio_root = (ROOT_DIR / audio_root).resolve()

    sqlite_path = audio_root / "annotations.db"
    
    # Resolve database URL from config, environment variables, or default to SQLite
    database_url = (
        app_config.database_url
        or os.environ.get("DATABASE_URL")
        or os.environ.get("NEON_DATABASE_URL")
    )
    
    db_backend = DatabaseBackend(sqlite_path, database_url)

    ffmpeg_binary = args.ffmpeg_binary or locate_ffmpeg()
    if ffmpeg_binary is None:
        print("Error: ffmpeg binary not found. Provide --ffmpeg-binary or install ffmpeg.")
        return 1

    jobs = collect_jobs(db_backend, audio_root)
    if not jobs:
        print("No clips found in the database. Nothing to do.")
        return 0

    generated = 0
    skipped = 0
    failures = 0

    for job in jobs:
        clip = job.clip
        existing_path = clip.segment_path
        if (
            not args.force
            and existing_path
            and (audio_root / existing_path).exists()
            and clip.segment_start_timestamp is not None
            and clip.segment_end_timestamp is not None
        ):
            skipped += 1
            continue

        if args.dry_run:
            print(
                f"Would generate segment for clip {clip.id} ({clip.audio_path}:"
                f" {clip.start_timestamp:.2f}s–{clip.end_timestamp:.2f}s)."
            )
            skipped += 1
            continue

        remove_previous_segment(audio_root, existing_path if args.force else None)

        try:
            result = generate_segment(
                clip_id=clip.id,
                audio_root=audio_root,
                source_path=job.source_path,
                segment_dir_name=args.segment_subdir,
                clip_start=clip.start_timestamp,
                clip_end=clip.end_timestamp,
                padding=args.padding,
                ffmpeg_binary=ffmpeg_binary,
            )
        except SegmentGenerationError as exc:
            print(f"❌ Failed to generate segment for clip {clip.id}: {exc}")
            failures += 1
            continue

        db_backend.upsert_clip_segment(
            clip.id,
            result.relative_path,
            result.start_timestamp,
            result.end_timestamp,
        )
        generated += 1

    total = len(jobs)
    print(
        f"Processed {total} clips: {generated} generated, {skipped} reused, {failures} failed."
    )

    return 0 if failures == 0 else 2


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
