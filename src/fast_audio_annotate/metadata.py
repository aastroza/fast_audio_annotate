"""Metadata loading and audio file helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import json

AUDIO_EXTENSIONS = {".webm", ".mp3", ".wav", ".ogg", ".m4a", ".flac"}


def looks_like_audio_file(name: str) -> bool:
    try:
        return Path(name).suffix.lower() in AUDIO_EXTENSIONS
    except Exception:
        return False


def iter_audio_files(directory: Path, *, recursive: bool = True) -> Iterator[Path]:
    """Yield audio files contained in *directory*."""

    if not directory.exists():
        return iter(())

    pattern = "**/*" if recursive else "*"
    files = [
        entry
        for entry in directory.glob(pattern)
        if entry.is_file() and looks_like_audio_file(entry.name)
    ]
    return iter(sorted(files))


def _extract_audio_path(entry: Dict[str, Any]) -> Optional[str]:
    path_keys = ("audio_path", "path", "file", "filename", "name")
    for key in path_keys:
        value = entry.get(key)
        if value:
            return str(value)
    return None


def _normalize_audio_path(audio_path: str, base_dir: Path) -> str:
    path_obj = Path(audio_path)
    try:
        path_obj = path_obj.relative_to(base_dir)
    except ValueError:
        pass
    return path_obj.as_posix()


def parse_metadata_payload(payload: Any, base_dir: Path) -> dict[str, dict]:
    metadata_map: dict[str, dict] = {}

    def store_entry(audio_path: Optional[str], metadata: Any) -> None:
        if not audio_path:
            return
        normalized_path = _normalize_audio_path(audio_path, base_dir)
        if isinstance(metadata, dict):
            metadata_map[normalized_path] = dict(metadata)
        else:
            metadata_map[normalized_path] = {"value": metadata}

    def handle_dict_entry(entry: Dict[str, Any]) -> None:
        audio_path = _extract_audio_path(entry)
        if not audio_path:
            return
        path_keys = {"audio_path", "path", "file", "filename", "name"}
        metadata = {k: v for k, v in entry.items() if k not in path_keys}
        store_entry(audio_path, metadata)

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                handle_dict_entry(item)
        return metadata_map

    if isinstance(payload, dict):
        list_keys = {"audios", "audio_files", "files"}
        for key, value in payload.items():
            if key in list_keys and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        handle_dict_entry(item)
                continue

            if isinstance(value, dict) and _extract_audio_path(value):
                handle_dict_entry(value)
                continue

            if looks_like_audio_file(str(key)):
                store_entry(str(key), value)

    return metadata_map


def load_audio_metadata_from_file(
    audio_folder: Path,
    db_backend: Any,
    metadata_filename: str = "metadata.json",
) -> None:
    """Load metadata from ``metadata_filename`` and synchronise with the database."""

    metadata_file = audio_folder / metadata_filename
    if not metadata_file.exists():
        return

    try:
        with metadata_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # pragma: no cover - logging aid
        print(f"Warning: Failed to load metadata from {metadata_file}: {exc}")
        return

    metadata_map = parse_metadata_payload(payload, audio_folder)

    if not metadata_map:
        print(f"Warning: No metadata entries found in {metadata_file}")
        return

    db_backend.sync_audio_metadata(metadata_map)
    print(f"Loaded metadata for {len(metadata_map)} audio files from {metadata_file}")
