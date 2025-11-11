"""Utilities for generating short audio segments for clip review."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import shutil
import subprocess


@dataclass
class SegmentGenerationResult:
    """Result of generating (or reusing) an audio segment for a clip."""

    relative_path: str
    start_timestamp: float
    end_timestamp: float


class SegmentGenerationError(RuntimeError):
    """Raised when a segment cannot be generated."""


def compute_segment_window(
    clip_start: float,
    clip_end: float,
    *,
    padding: float,
    lower_bound: float = 0.0,
    minimum_duration: float = 0.5,
) -> tuple[float, float]:
    """Return the desired segment window surrounding the clip."""

    start = max(lower_bound, clip_start - padding)
    end = clip_end + padding
    if end <= start:
        end = start + max(clip_end - clip_start, minimum_duration)
    return start, end


def generate_segment(
    *,
    clip_id: int,
    audio_root: Path,
    source_path: Path,
    segment_dir_name: str,
    clip_start: float,
    clip_end: float,
    padding: float,
    ffmpeg_binary: Optional[str] = None,
) -> SegmentGenerationResult:
    """Generate an audio segment file for the given clip."""

    if ffmpeg_binary is None:
        raise SegmentGenerationError("ffmpeg binary is required to pre-generate segments")

    if not source_path.exists():
        raise SegmentGenerationError(f"Audio source missing: {source_path}")

    segment_start, segment_end = compute_segment_window(
        clip_start,
        clip_end,
        padding=padding,
        lower_bound=0.0,
    )
    segment_duration = segment_end - segment_start

    # Create subdirectory for each audio file (without extension)
    audio_name = source_path.stem  # Gets filename without extension
    segment_subdir = audio_root / segment_dir_name / audio_name
    segment_subdir.mkdir(parents=True, exist_ok=True)

    suffix = source_path.suffix or ".webm"
    segment_filename = (
        f"clip_{clip_id}_{int(segment_start * 1000):010d}_{int(segment_end * 1000):010d}{suffix}"
    )
    output_path = segment_subdir / segment_filename

    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{segment_start:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{segment_duration:.3f}",
        "-c",
        "copy",
        str(output_path),
    ]

    try:
        subprocess.run(command, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:  # pragma: no cover - runtime
        raise SegmentGenerationError(str(exc)) from exc

    relative_path = str(Path(segment_dir_name) / audio_name / output_path.name).replace("\\", "/")

    return SegmentGenerationResult(
        relative_path=relative_path,
        start_timestamp=segment_start,
        end_timestamp=segment_end,
    )


def remove_previous_segment(audio_root: Path, relative_path: Optional[str]) -> None:
    """Remove a previously generated segment file if it exists."""

    if not relative_path:
        return

    candidate = audio_root / relative_path
    try:
        candidate.unlink()
    except FileNotFoundError:
        return
    except OSError:
        # Ignore failures when removing stale files.
        pass


def locate_ffmpeg() -> Optional[str]:
    """Return the path to the ffmpeg binary if present."""

    return shutil.which("ffmpeg")
