"""Database backend helpers for SQLite and Postgres/Neon."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import json
import psycopg
import sqlite3

from psycopg.types.json import Jsonb


@dataclass
class ClipRecord:
    """Data structure representing a single clip annotation."""

    id: int
    audio_path: str
    start_timestamp: float
    end_timestamp: float
    text: str
    username: str
    timestamp: str
    marked: bool
    human_reviewed: bool


@dataclass
class AudioMetadataRecord:
    """Metadata attached to an audio file."""

    audio_path: str
    metadata: Dict[str, Any]


class DatabaseBackend:
    """Abstraction layer for clips storage.

    The app defaults to SQLite, but can connect to a remote Postgres/Neon
    database when a connection string is provided.
    """

    def __init__(self, sqlite_path: Path, database_url: Optional[str] = None) -> None:
        self.sqlite_path = sqlite_path
        self.database_url = database_url
        self.backend = self._determine_backend()

        if self.backend == "sqlite":
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_sqlite()
        elif self.backend == "postgres":
            self._init_postgres()
        else:
            raise ValueError(f"Unsupported database backend: {self.backend}")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def backend_label(self) -> str:
        """Return a human readable name for the configured backend."""

        if self.backend == "sqlite":
            return str(self.sqlite_path)
        return self.database_url or "postgres"

    def fetch_clips(self, audio_path: Optional[str]) -> List[ClipRecord]:
        """Return all clips for a given audio file."""

        if not audio_path:
            return []

        if self.backend == "sqlite":
            with self._sqlite_conn() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, audio_path, start_timestamp, end_timestamp, text,
                           username, timestamp, marked, human_reviewed
                    FROM clips
                    WHERE audio_path = ?
                    ORDER BY start_timestamp
                    """,
                    (str(audio_path),),
                )
                rows = cursor.fetchall()
        else:
            with self._postgres_cursor() as cur:
                cur.execute(
                    """
                    SELECT id, audio_path, start_timestamp, end_timestamp, text,
                           username, timestamp, marked, human_reviewed
                    FROM clips
                    WHERE audio_path = %s
                    ORDER BY start_timestamp
                    """,
                    (str(audio_path),),
                )
                rows = cur.fetchall()

        return [self._row_to_clip(row) for row in rows]

    def fetch_all_clips(self) -> List[ClipRecord]:
        """Return all clips from the database."""

        if self.backend == "sqlite":
            with self._sqlite_conn() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, audio_path, start_timestamp, end_timestamp, text,
                           username, timestamp, marked, human_reviewed
                    FROM clips
                    ORDER BY id
                    """
                )
                rows = cursor.fetchall()
        else:
            with self._postgres_cursor() as cur:
                cur.execute(
                    """
                    SELECT id, audio_path, start_timestamp, end_timestamp, text,
                           username, timestamp, marked, human_reviewed
                    FROM clips
                    ORDER BY id
                    """
                )
                rows = cur.fetchall()

        return [self._row_to_clip(row) for row in rows]

    def fetch_random_clip(self, include_marked: bool = False, include_reviewed: bool = False) -> Optional[ClipRecord]:
        """Return a random clip filtered by review/mark flags."""

        where_clauses: List[str] = []
        params: List[Any] = []

        if not include_marked:
            where_clauses.append("marked = ?" if self.backend == "sqlite" else "marked = %s")
            params.append(False)
        if not include_reviewed:
            where_clauses.append("human_reviewed = ?" if self.backend == "sqlite" else "human_reviewed = %s")
            params.append(False)

        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)

        if self.backend == "sqlite":
            with self._sqlite_conn() as conn:
                cursor = conn.execute(
                    f"""
                    SELECT id, audio_path, start_timestamp, end_timestamp, text,
                           username, timestamp, marked, human_reviewed
                    FROM clips{where_sql}
                    ORDER BY RANDOM()
                    LIMIT 1
                    """,
                    tuple(params),
                )
                row = cursor.fetchone()
        else:
            with self._postgres_cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, audio_path, start_timestamp, end_timestamp, text,
                           username, timestamp, marked, human_reviewed
                    FROM clips{where_sql}
                    ORDER BY RANDOM()
                    LIMIT 1
                    """,
                    tuple(params),
                )
                row = cur.fetchone()

        return self._row_to_clip(row) if row else None

    def count_clips(self, audio_path: Optional[str] = None) -> int:
        """Return the number of clips, optionally filtered by audio path."""

        if self.backend == "sqlite":
            with self._sqlite_conn() as conn:
                if audio_path is None:
                    cursor = conn.execute("SELECT COUNT(*) FROM clips")
                    return cursor.fetchone()[0]
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM clips WHERE audio_path = ?", (str(audio_path),)
                )
                return cursor.fetchone()[0]

        with self._postgres_cursor() as cur:
            if audio_path is None:
                cur.execute("SELECT COUNT(*) FROM clips")
                return cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FROM clips WHERE audio_path = %s", (str(audio_path),)
            )
            return cur.fetchone()[0]

    def get_clip(self, clip_id: int) -> Optional[ClipRecord]:
        """Return a clip by primary key."""

        if self.backend == "sqlite":
            with self._sqlite_conn() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, audio_path, start_timestamp, end_timestamp, text,
                           username, timestamp, marked, human_reviewed
                    FROM clips
                    WHERE id = ?
                    """,
                    (clip_id,),
                )
                row = cursor.fetchone()
        else:
            with self._postgres_cursor() as cur:
                cur.execute(
                    """
                    SELECT id, audio_path, start_timestamp, end_timestamp, text,
                           username, timestamp, marked, human_reviewed
                    FROM clips
                    WHERE id = %s
                    """,
                    (clip_id,),
                )
                row = cur.fetchone()

        return self._row_to_clip(row) if row else None

    def create_clip(self, values: dict) -> ClipRecord:
        """Insert a new clip and return the created record."""

        audio_path = str(values["audio_path"])
        params = (
            audio_path,
            float(values["start_timestamp"]),
            float(values["end_timestamp"]),
            values.get("text", ""),
            values["username"],
            values["timestamp"],
            bool(values.get("marked", False)),
            bool(values.get("human_reviewed", False)),
        )

        if self.backend == "sqlite":
            with self._sqlite_conn() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO clips (
                        audio_path, start_timestamp, end_timestamp, text,
                        username, timestamp, marked, human_reviewed
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    params,
                )
                clip_id = cursor.lastrowid
            return self.get_clip(int(clip_id))

        with self._postgres_cursor() as cur:
            cur.execute(
                """
                INSERT INTO clips (
                    audio_path, start_timestamp, end_timestamp, text,
                    username, timestamp, marked, human_reviewed
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                params,
            )
            clip_id = cur.fetchone()[0]
        return self.get_clip(int(clip_id))

    def update_clip(self, clip_id: int, updates: dict) -> None:
        """Update a clip with the provided values."""

        if not updates:
            return

        keys = list(updates.keys())
        values = [updates[key] for key in keys]

        if self.backend == "sqlite":
            assignments = ", ".join(f"{key} = ?" for key in keys)
            with self._sqlite_conn() as conn:
                conn.execute(
                    f"UPDATE clips SET {assignments} WHERE id = ?",
                    (*values, clip_id),
                )
            return

        assignments = ", ".join(f"{key} = %s" for key in keys)
        with self._postgres_cursor() as cur:
            cur.execute(
                f"UPDATE clips SET {assignments} WHERE id = %s",
                (*values, clip_id),
            )

    def delete_clip(self, clip_id: int) -> None:
        """Delete a clip by id."""

        if self.backend == "sqlite":
            with self._sqlite_conn() as conn:
                conn.execute("DELETE FROM clips WHERE id = ?", (clip_id,))
            return

        with self._postgres_cursor() as cur:
            cur.execute("DELETE FROM clips WHERE id = %s", (clip_id,))

    def delete_clips_for_audio(self, audio_path: str) -> None:
        """Delete all clips associated with ``audio_path``."""

        if self.backend == "sqlite":
            with self._sqlite_conn() as conn:
                conn.execute("DELETE FROM clips WHERE audio_path = ?", (str(audio_path),))
            return

        with self._postgres_cursor() as cur:
            cur.execute("DELETE FROM clips WHERE audio_path = %s", (str(audio_path),))

    def sync_audio_metadata(self, metadata_map: Mapping[str, Any]) -> None:
        """Insert or update metadata for the provided audio files."""

        if not metadata_map:
            return

        def _normalize_metadata(metadata: Any) -> Dict[str, Any]:
            if isinstance(metadata, Mapping):
                return dict(metadata)
            return {"value": metadata}

        if self.backend == "sqlite":
            records = [
                (
                    str(Path(audio_path)),
                    json.dumps(_normalize_metadata(metadata), ensure_ascii=False),
                )
                for audio_path, metadata in metadata_map.items()
            ]
            with self._sqlite_conn() as conn:
                conn.executemany(
                    """
                    INSERT INTO audio_metadata (audio_path, metadata_json)
                    VALUES (?, ?)
                    ON CONFLICT(audio_path) DO UPDATE SET
                        metadata_json = excluded.metadata_json
                    """,
                    records,
                )
            return

        with self._postgres_cursor() as cur:
            for audio_path, metadata in metadata_map.items():
                cur.execute(
                    """
                    INSERT INTO audio_metadata (audio_path, metadata_json)
                    VALUES (%s, %s)
                    ON CONFLICT (audio_path) DO UPDATE SET
                        metadata_json = EXCLUDED.metadata_json
                    """,
                    (str(Path(audio_path)), Jsonb(_normalize_metadata(metadata))),
                )

    def fetch_audio_metadata(self, audio_path: str) -> Optional[AudioMetadataRecord]:
        """Return metadata stored for an audio file."""

        if self.backend == "sqlite":
            with self._sqlite_conn() as conn:
                cursor = conn.execute(
                    "SELECT audio_path, metadata_json FROM audio_metadata WHERE audio_path = ?",
                    (str(audio_path),),
                )
                row = cursor.fetchone()
        else:
            with self._postgres_cursor() as cur:
                cur.execute(
                    "SELECT audio_path, metadata_json FROM audio_metadata WHERE audio_path = %s",
                    (str(audio_path),),
                )
                row = cur.fetchone()

        if not row:
            return None

        if isinstance(row, sqlite3.Row):
            audio_path_value = str(row["audio_path"])
            metadata_raw = row["metadata_json"]
        else:
            audio_path_value, metadata_raw = row

        if isinstance(metadata_raw, str):
            metadata_value = json.loads(metadata_raw)
        else:
            metadata_value = metadata_raw

        if isinstance(metadata_value, dict):
            metadata_dict = dict(metadata_value)
        else:
            metadata_dict = {"value": metadata_value}

        return AudioMetadataRecord(audio_path=audio_path_value, metadata=metadata_dict)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _determine_backend(self) -> str:
        if not self.database_url:
            return "sqlite"

        lowered = self.database_url.lower()
        if lowered.startswith("postgres://") or lowered.startswith("postgresql://"):
            return "postgres"
        if lowered.startswith("sqlite://"):
            # Allow overriding sqlite through a URL style string.
            self.sqlite_path = Path(self.database_url.split("sqlite://", 1)[1])
            return "sqlite"
        return "sqlite"

    def _init_sqlite(self) -> None:
        with self._sqlite_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS clips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audio_path TEXT NOT NULL,
                    start_timestamp REAL NOT NULL,
                    end_timestamp REAL NOT NULL,
                    text TEXT NOT NULL DEFAULT '',
                    username TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    marked BOOLEAN NOT NULL DEFAULT 0,
                    human_reviewed BOOLEAN NOT NULL DEFAULT 0
                )
                """
            )
            try:
                conn.execute(
                    "ALTER TABLE clips ADD COLUMN human_reviewed BOOLEAN NOT NULL DEFAULT 0"
                )
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_metadata (
                    audio_path TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL
                )
                """
            )

    def _init_postgres(self) -> None:
        with self._postgres_cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS clips (
                    id BIGSERIAL PRIMARY KEY,
                    audio_path TEXT NOT NULL,
                    start_timestamp DOUBLE PRECISION NOT NULL,
                    end_timestamp DOUBLE PRECISION NOT NULL,
                    text TEXT NOT NULL DEFAULT '',
                    username TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    marked BOOLEAN NOT NULL DEFAULT FALSE,
                    human_reviewed BOOLEAN NOT NULL DEFAULT FALSE
                )
                """
            )
            cur.execute(
                """
                ALTER TABLE clips
                ADD COLUMN IF NOT EXISTS human_reviewed BOOLEAN NOT NULL DEFAULT FALSE
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS audio_metadata (
                    audio_path TEXT PRIMARY KEY,
                    metadata_json JSONB NOT NULL
                )
                """
            )

    @contextmanager
    def _sqlite_conn(self):
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def _postgres_cursor(self):
        conn = psycopg.connect(self.database_url)
        try:
            with conn.cursor() as cur:
                yield cur
                conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row_to_clip(row: Iterable) -> ClipRecord:
        """Convert a database row to a ClipRecord."""

        if isinstance(row, sqlite3.Row):
            data = {key: row[key] for key in row.keys()}
        else:
            # psycopg returns tuples by default ordered as queried
            (
                clip_id,
                audio_path,
                start_timestamp,
                end_timestamp,
                text,
                username,
                timestamp,
                marked,
                human_reviewed,
            ) = row
            data = {
                "id": clip_id,
                "audio_path": audio_path,
                "start_timestamp": float(start_timestamp),
                "end_timestamp": float(end_timestamp),
                "text": text,
                "username": username,
                "timestamp": timestamp,
                "marked": bool(marked),
                "human_reviewed": bool(human_reviewed),
            }

        return ClipRecord(
            id=int(data["id"]),
            audio_path=str(data["audio_path"]),
            start_timestamp=float(data["start_timestamp"]),
            end_timestamp=float(data["end_timestamp"]),
            text=data["text"],
            username=data["username"],
            timestamp=data["timestamp"],
            marked=bool(data["marked"]),
            human_reviewed=bool(data.get("human_reviewed", False)),
        )
