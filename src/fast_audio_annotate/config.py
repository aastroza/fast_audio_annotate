"""Configuration helpers for the Fast Audio Annotate project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import simple_parsing as sp
import yaml


@dataclass
class AppConfig:
    """Application configuration parsed from CLI arguments and YAML file."""

    audio_folder: str = sp.field(
        positional=True,
        default="audio",
        help="Directory that contains the audio files and annotations database.",
    )
    title: str = "Audio Annotation Tool"
    description: str = "Annotate audio clips with transcriptions"
    max_history: int = 10
    database_url: Optional[str] = sp.field(
        default=None,
        help="Optional database URL (e.g. Neon Postgres). Overrides the default SQLite file.",
    )
    metadata_filename: str = "metadata.json"
    whisper_model: str = "openai/whisper-large-v3"
    transcription_language: Optional[str] = None

    @property
    def audio_path(self) -> Path:
        return Path(self.audio_folder)


def parse_app_config(config_path: str = "./config.yaml") -> AppConfig:
    """Parse the :class:`AppConfig` from CLI arguments and an optional YAML file."""

    defaults: dict[str, object] = {}
    yaml_path = Path(config_path)
    if yaml_path.exists():
        with yaml_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Invalid configuration file structure in {config_path}")
            defaults = loaded

    parser = sp.ArgumentParser(add_config_path_arg=False)
    parser.add_arguments(AppConfig, dest="config", default=AppConfig(**defaults))
    parsed = parser.parse_args()
    return parsed.config
