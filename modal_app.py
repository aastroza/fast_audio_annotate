"""FastHTML Audio Annotation Tool - Modal deployment with full feature set."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import fasthtml.common as fh
import modal
from dotenv import load_dotenv
from starlette.responses import FileResponse, Response

from db_backend import ClipRecord, DatabaseBackend

# ---------------------------------------------------------------------------
# Environment and configuration
# ---------------------------------------------------------------------------
load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fast_audio_annotate.config import AppConfig, parse_app_config  # noqa: E402
from fast_audio_annotate.metadata import load_audio_metadata_from_file  # noqa: E402


config: AppConfig = parse_app_config()

# Database setup mirrors ``main.py`` including Neon/Postgres support
DATABASE_URL = (
    config.database_url
    or os.environ.get("DATABASE_URL")
    or os.environ.get("NEON_DATABASE_URL")
)
db_backend = DatabaseBackend(config.audio_path / "annotations.db", DATABASE_URL)

# Load metadata so the interface can surface it just like ``main.py``
load_audio_metadata_from_file(config.audio_path, db_backend, config.metadata_filename)

# Review workflow constants
CLIP_PADDING_SECONDS = 1.5

# ---------------------------------------------------------------------------
# FastHTML setup
# ---------------------------------------------------------------------------
fasthtml_app, rt = fh.fast_app(
    hdrs=(
        fh.Link(rel="stylesheet", href="/styles.css"),
        fh.Script(src="https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.min.js"),
        fh.Script(src="https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/regions.min.js"),
        fh.Script(src="https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/timeline.min.js"),
    ),
    pico=False,
    debug=True,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_username(contributor_name: str = "") -> str:
    """Return the username for audit purposes, preferring contributor name."""

    if contributor_name and contributor_name.strip():
        return contributor_name.strip()
    return os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"


def get_contributor_stats() -> dict:
    """Get statistics about contributors."""

    try:
        return db_backend.get_contributor_stats()
    except Exception as exc:  # pragma: no cover - defensive guardrail
        print(f"Error getting contributor stats: {exc}")
        return {"total_contributors": 0, "total_contributions": 0, "contributors": []}


def get_audio_metadata(audio_path: Optional[str]) -> Optional[dict]:
    """Fetch metadata for an audio file from the database."""

    if not audio_path:
        return None

    record = db_backend.fetch_audio_metadata(str(audio_path))
    return record.metadata if record else None


def render_audio_metadata_panel(metadata: Optional[dict]) -> fh.Node:
    """Render a panel summarizing audio metadata."""

    if not metadata:
        body = fh.Div(
            "No metadata available for this audio file.",
            style="color: #666; font-style: italic;",
        )
    else:
        entries: list[fh.Node] = []
        for key, value in sorted(metadata.items(), key=lambda item: str(item[0])):
            if isinstance(value, (dict, list)):
                formatted = json.dumps(value, ensure_ascii=False, indent=2)
                value_node = fh.Pre(
                    formatted,
                    style=(
                        "margin: 0; white-space: pre-wrap; background: #f1f3f5; padding: 8px; "
                        "border-radius: 4px; flex: 1; font-family: 'Fira Code', monospace; font-size: 13px;"
                    ),
                )
            else:
                value_node = fh.Span(str(value))

            entries.append(
                fh.Div(
                    fh.Span(f"{key}:", style="font-weight: 600; min-width: 120px;"),
                    value_node,
                    style="display: flex; gap: 8px; align-items: flex-start;",
                )
            )

        body = fh.Div(
            *entries,
            style="display: flex; flex-direction: column; gap: 6px;",
        )

    return fh.Div(
        fh.H4("Audio Metadata", style="margin-bottom: 10px; color: #343a40;"),
        body,
        cls="audio-metadata-panel",
        style=(
            "margin-bottom: 20px; padding: 15px; background: #ffffff; border: 1px solid #dee2e6; "
            "border-radius: 8px;"
        ),
    )


def select_random_clip() -> Optional[ClipRecord]:
    """Pick a random clip that still needs human review."""

    return db_backend.fetch_random_clip()


def get_clip(clip_id: Optional[str]) -> Optional[ClipRecord]:
    """Return a clip by id, or ``None`` if unavailable."""

    if not clip_id:
        return None
    try:
        return db_backend.get_clip(int(clip_id))
    except (TypeError, ValueError):
        return None


def compute_display_window(start: float, end: float, duration: Optional[float] = None) -> tuple[float, float]:
    """Return the playback window that surrounds the clip with a safety margin."""

    padded_start = max(0.0, start - CLIP_PADDING_SECONDS)
    padded_end = end + CLIP_PADDING_SECONDS
    if duration is not None:
        padded_end = min(duration, padded_end)
    return padded_start, padded_end


def render_clip_editor(clip: ClipRecord) -> fh.Node:
    """Render the editor for a single clip."""

    metadata = get_audio_metadata(clip.audio_path)
    padded_start, padded_end = compute_display_window(clip.start_timestamp, clip.end_timestamp)
    duration = clip.end_timestamp - clip.start_timestamp

    instructions = fh.Div(
        fh.H3("How to review this clip", style="margin-bottom: 8px; color: #0d6efd;"),
        fh.P(
            "Listen carefully to the highlighted audio, correct the transcription so it matches the speech exactly, "
            "and adjust the start/end times if they need a better cut. Use the buttons below to either save your progress, "
            "mark the clip as reviewed, or report an issue if the audio is unusable."
        ),
        style="margin-bottom: 18px; background: #f8f9fa; padding: 16px; border-radius: 8px; border: 1px solid #dee2e6;",
    )

    clip_info = fh.Div(
        fh.Div(
            fh.Strong("Audio file:"),
            fh.Span(f" {clip.audio_path}"),
            style="margin-bottom: 4px;",
        ),
        fh.Div(
            fh.Strong("Clip window:"),
            fh.Span(f" {clip.start_timestamp:.2f}s – {clip.end_timestamp:.2f}s ({duration:.2f}s long)"),
            style="margin-bottom: 4px;",
        ),
        fh.Div(
            fh.Strong("Last updated by:"),
            fh.Span(f" {clip.username} at {clip.timestamp}"),
        ),
        style="margin-bottom: 16px; display: flex; flex-direction: column; gap: 4px;",
    )

    form_inputs = fh.Div(
        fh.Input(type="hidden", name="clip_id", value=str(clip.id)),
        fh.Div(
            fh.Div(
                fh.Label("Start (seconds)", style="display: block; margin-bottom: 4px; font-weight: 600;"),
                fh.Input(
                    type="number",
                    name="start_time",
                    value=f"{clip.start_timestamp:.2f}",
                    step="0.01",
                    min="0",
                    id="start-time-input",
                    style="width: 100%; padding: 10px; border: 1px solid #ced4da; border-radius: 6px; font-size: 14px;",
                ),
            ),
            fh.Div(
                fh.Label("End (seconds)", style="display: block; margin-bottom: 4px; font-weight: 600;"),
                fh.Input(
                    type="number",
                    name="end_time",
                    value=f"{clip.end_timestamp:.2f}",
                    step="0.01",
                    min="0",
                    id="end-time-input",
                    style="width: 100%; padding: 10px; border: 1px solid #ced4da; border-radius: 6px; font-size: 14px;",
                ),
            ),
            style="display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap;",
        ),
        fh.Div(
            fh.Label("Transcription", style="display: block; margin-bottom: 6px; font-weight: 600; font-size: 16px;"),
            fh.Textarea(
                clip.text or "",
                name="transcription",
                id="transcription-input",
                rows="6",
                placeholder="Type the corrected transcription here...",
                style="width: 100%; padding: 12px; border: 1px solid #ced4da; border-radius: 6px; font-size: 15px; resize: vertical;",
            ),
            style="margin-bottom: 16px;",
        ),
        fh.Div(
            fh.Label("Your name (optional)", style="display: block; margin-bottom: 6px; font-weight: 600; font-size: 16px; color: #495057;"),
            fh.Input(
                value=clip.username if hasattr(clip, "username") and clip.username and clip.username != "unknown" else "",
                name="contributor_name",
                id="contributor-name-input",
                placeholder="Enter your name to be credited as a contributor...",
                style="width: 100%; padding: 10px; border: 1px solid #ced4da; border-radius: 6px; font-size: 14px;",
            ),
            fh.Div(
                "💡 Your name will be used to credit your contributions to this project!",
                style="font-size: 12px; color: #6c757d; margin-top: 4px; font-style: italic;",
            ),
            style="margin-bottom: 20px;",
        ),
        id="clip-form",
    )

    actions = fh.Div(
        fh.Button(
            "➡️ Next clip",
            cls="next-btn",
            hx_post="/next_clip",
            hx_include="#clip-form input, #clip-form textarea",
            hx_target="#main-content",
            hx_swap="outerHTML",
            hx_indicator="#loading-next",
            style="padding: 12px 18px; border-radius: 6px; background: #ffc107; color: #000; border: none; font-size: 15px; cursor: pointer;",
        ),
        fh.Button(
            "✅ Finish review",
            cls="complete-btn",
            hx_post="/complete_clip",
            hx_include="#clip-form input, #clip-form textarea",
            hx_target="#main-content",
            hx_swap="outerHTML",
            hx_indicator="#loading-complete",
            style="padding: 12px 18px; border-radius: 6px; background: #0d6efd; color: white; border: none; font-size: 15px; cursor: pointer;",
        ),
        fh.Button(
            "🚩 Report issue",
            cls="flag-btn",
            hx_post="/flag_clip",
            hx_include="#clip-form input, #clip-form textarea",
            hx_confirm="Report this clip as problematic?",
            hx_target="#main-content",
            hx_swap="outerHTML",
            hx_indicator="#loading-flag",
            style="padding: 12px 18px; border-radius: 6px; background: #dc3545; color: white; border: none; font-size: 15px; cursor: pointer;",
        ),
        fh.Div(
            "🔄 Loading next clip...",
            id="loading-next",
            cls="htmx-indicator",
            style="padding: 8px 12px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; color: #856404; font-size: 14px;",
        ),
        fh.Div(
            "✅ Completing review...",
            id="loading-complete",
            cls="htmx-indicator",
            style="padding: 8px 12px; background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 4px; color: #0c5460; font-size: 14px;",
        ),
        fh.Div(
            "🚩 Reporting issue...",
            id="loading-flag",
            cls="htmx-indicator",
            style="padding: 8px 12px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px; color: #721c24; font-size: 14px;",
        ),
        style="display: flex; gap: 12px; flex-wrap: wrap; align-items: center;",
    )

    waveform = fh.Div(
        fh.Div(
            fh.Div(
                "Current Time: ",
                fh.Span("0.00", id="current-time", style="font-weight: bold; color: #0d6efd;"),
                " s",
                style="font-size: 16px; margin-bottom: 12px;",
            ),
            fh.Div(
                "Hotkeys: [",
                fh.Span("Q", style="font-weight: 600; color: #198754;"),
                "] start • [",
                fh.Span("W", style="font-weight: 600; color: #dc3545;"),
                "] end • [",
                fh.Span("Space", style="font-weight: 600; color: #0d6efd;"),
                "] play/pause",
                style="color: #6c757d; font-size: 14px;",
            ),
            style="margin-bottom: 16px;",
        ),
        fh.Div(id="waveform", style="width: 100%; height: 140px; background: #f1f3f5; border-radius: 8px; margin-bottom: 12px;"),
        fh.Div(id="timeline", style="width: 100%; margin-bottom: 16px;"),
        fh.Div(
            fh.Button("▶ Play", id="play-btn", cls="control-btn", style="padding: 10px 18px; font-size: 15px;"),
            fh.Button("⏸ Pause", id="pause-btn", cls="control-btn", style="padding: 10px 18px; font-size: 15px;"),
            fh.Button("⏹ Stop", id="stop-btn", cls="control-btn", style="padding: 10px 18px; font-size: 15px;"),
            fh.Label("Speed:", style="margin-left: 12px; font-weight: 600;"),
            fh.Select(
                fh.Option("0.75x", value="0.75"),
                fh.Option("1x", value="1", selected=True),
                fh.Option("1.25x", value="1.25"),
                fh.Option("1.5x", value="1.5"),
                fh.Option("2x", value="2"),
                id="speed-select",
                style="padding: 8px; border-radius: 6px; border: 1px solid #ced4da;",
            ),
            style="display: flex; align-items: center; gap: 10px; justify-content: center;",
        ),
        style="margin-bottom: 24px;",
    )

    metadata_panel = render_audio_metadata_panel(metadata)

    return fh.Div(
        instructions,
        clip_info,
        waveform,
        form_inputs,
        actions,
        metadata_panel,
        id="main-content",
        data_audio_path=str(clip.audio_path),
        data_clip_start=f"{clip.start_timestamp:.2f}",
        data_clip_end=f"{clip.end_timestamp:.2f}",
        data_display_start=f"{padded_start:.2f}",
        data_display_end=f"{padded_end:.2f}",
    )


def render_empty_state() -> fh.Node:
    """Render a friendly message when no clips are available."""

    return fh.Div(
        fh.H2("All caught up!", style="text-align: center; color: #198754;"),
        fh.P(
            "There are no clips waiting for human review right now. Please check back later.",
            style="text-align: center; font-size: 16px; color: #6c757d;",
        ),
        id="main-content",
        style="max-width: 640px; margin: 60px auto; background: white; padding: 32px; border-radius: 12px;",
    )


def render_main_content(clip: Optional[ClipRecord]) -> fh.Node:
    """Render the main content area."""

    if clip:
        return render_clip_editor(clip)
    return render_empty_state()


def render_contributor_stats() -> fh.Node:
    """Render a panel showing contributor statistics."""

    try:
        stats = get_contributor_stats()
        if stats["total_contributors"] == 0:
            return fh.Div(
                fh.H4("🙏 Contributors", style="margin-bottom: 10px; color: #343a40;"),
                fh.P(
                    "Be the first to contribute! Enter your name when reviewing clips to be credited.",
                    style="color: #6c757d; font-style: italic;",
                ),
                cls="contributor-stats-panel",
                style=(
                    "margin-bottom: 20px; padding: 15px; background: #f8f9fa; border: 1px solid #e9ecef; "
                    "border-radius: 8px;"
                ),
            )

        top_contributors = stats["contributors"][:5]
        contributor_list: list[fh.Node] = []
        for i, contributor in enumerate(top_contributors):
            rank_emoji = ["🥇", "🥈", "🥉", "🏅", "⭐"][i] if i < 5 else "✨"
            contributor_list.append(
                fh.Div(
                    fh.Span(f"{rank_emoji} {contributor['name']}", style="font-weight: 600;"),
                    fh.Span(
                        f" - {contributor['contributions']} contributions",
                        style="color: #6c757d; margin-left: 8px;",
                    ),
                    style="margin-bottom: 4px;",
                )
            )

        return fh.Div(
            fh.H4("🙏 Contributors", style="margin-bottom: 10px; color: #343a40;"),
            fh.Div(
                fh.P(
                    f"Total contributors: {stats['total_contributors']} | Total contributions: {stats['total_contributions']}",
                    style="margin-bottom: 12px; font-weight: 500; color: #495057;",
                ),
                *contributor_list,
                style="margin-bottom: 8px;",
            ),
            fh.P(
                "Thank you to everyone who has contributed to improving this dataset! 🎉",
                style="color: #198754; font-style: italic; margin-bottom: 0; font-size: 14px;",
            ),
            cls="contributor-stats-panel",
            style=(
                "margin-bottom: 20px; padding: 15px; background: #f8f9fa; border: 1px solid #e9ecef; "
                "border-radius: 8px;"
            ),
        )
    except Exception as exc:  # pragma: no cover - defensive guardrail
        print(f"Error rendering contributor stats: {exc}")
        return fh.Div()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@rt("/")
def index():
    """Main entry point for the crowdsourced clip review interface."""

    clip = select_random_clip()
    main_content = render_main_content(clip)
    contributor_stats = render_contributor_stats()

    return fh.Titled(
        config.title,
        fh.Div(
            fh.H1("Clip review"),
            contributor_stats,
            main_content,
            cls="container",
        ),
        fh.Script(
            """
            let wavesurfer = null;
            let wsRegions = null;
            let currentRegion = null;

            function initWaveSurfer() {
                if (wavesurfer) {
                    wavesurfer.destroy();
                    wavesurfer = null;
                }

                const mainContent = document.getElementById('main-content');
                if (!mainContent) {
                    return;
                }

                const audioPath = mainContent.dataset.audioPath;
                const clipStart = parseFloat(mainContent.dataset.clipStart || '0');
                const clipEnd = parseFloat(mainContent.dataset.clipEnd || '0');
                const displayStart = parseFloat(mainContent.dataset.displayStart || clipStart);
                const displayEnd = parseFloat(mainContent.dataset.displayEnd || clipEnd);

                if (!audioPath) {
                    return;
                }

                wavesurfer = WaveSurfer.create({
                    container: '#waveform',
                    waveColor: '#4F4A85',
                    progressColor: '#383351',
                    height: 140,
                    barWidth: 2,
                    barGap: 1,
                    barRadius: 2,
                    responsive: true,
                });

                wsRegions = wavesurfer.registerPlugin(WaveSurfer.Regions.create());
                wavesurfer.registerPlugin(WaveSurfer.Timeline.create({
                    container: '#timeline',
                }));

                wavesurfer.load('/"""
            + f"{config.audio_folder}"
            + """/' + audioPath);

                wavesurfer.on('ready', () => {
                    wsRegions.clearRegions();
                    currentRegion = wsRegions.addRegion({
                        start: clipStart,
                        end: clipEnd,
                        color: 'rgba(13, 110, 253, 0.3)',
                        drag: true,
                        resize: true,
                    });

                    currentRegion.on('update', () => {
                        const startInput = document.getElementById('start-time-input');
                        const endInput = document.getElementById('end-time-input');
                        if (startInput) startInput.value = currentRegion.start.toFixed(2);
                        if (endInput) endInput.value = currentRegion.end.toFixed(2);
                    });

                    const viewDuration = Math.max(0.5, displayEnd - displayStart);
                    const pxPerSec = Math.max(120, 900 / viewDuration);
                    const timeline = document.getElementById('timeline');
                    if (timeline) {
                        timeline.style.width = '100%';
                    }

                    const container = document.getElementById('waveform');
                    if (container) {
                        container.scrollLeft = displayStart * pxPerSec;
                    }

                    wavesurfer.setTime(displayStart);
                    wavesurfer.playPause();
                });

                wavesurfer.on('audioprocess', () => {
                    const currentTime = wavesurfer.getCurrentTime();
                    const timeDisplay = document.getElementById('current-time');
                    if (timeDisplay) {
                        timeDisplay.textContent = currentTime.toFixed(2);
                    }
                });

                wavesurfer.on('seek', () => {
                    const currentTime = wavesurfer.getCurrentTime();
                    const timeDisplay = document.getElementById('current-time');
                    if (timeDisplay) {
                        timeDisplay.textContent = currentTime.toFixed(2);
                    }
                });

                document.getElementById('play-btn')?.addEventListener('click', () => wavesurfer.play());
                document.getElementById('pause-btn')?.addEventListener('click', () => wavesurfer.pause());
                document.getElementById('stop-btn')?.addEventListener('click', () => {
                    wavesurfer.stop();
                    const timeDisplay = document.getElementById('current-time');
                    if (timeDisplay) {
                        timeDisplay.textContent = '0.00';
                    }
                });

                document.getElementById('speed-select')?.addEventListener('change', (event) => {
                    const speed = parseFloat(event.target.value);
                    wavesurfer.setPlaybackRate(speed);
                });

                document.addEventListener('keydown', (event) => {
                    if (event.target && ['INPUT', 'TEXTAREA'].includes(event.target.tagName)) {
                        return;
                    }

                    if (event.key === 'q' || event.key === 'Q') {
                        const currentTime = wavesurfer.getCurrentTime();
                        const startInput = document.getElementById('start-time-input');
                        if (startInput) {
                            startInput.value = currentTime.toFixed(2);
                        }
                        if (currentRegion) {
                            currentRegion.setOptions({ start: currentTime });
                        }
                    }

                    if (event.key === 'w' || event.key === 'W') {
                        const currentTime = wavesurfer.getCurrentTime();
                        const endInput = document.getElementById('end-time-input');
                        if (endInput) {
                            endInput.value = currentTime.toFixed(2);
                        }
                        if (currentRegion) {
                            currentRegion.setOptions({ end: currentTime });
                        }
                    }

                    if (event.code === 'Space') {
                        event.preventDefault();
                        wavesurfer.playPause();
                    }
                });
            }

            function teardownWaveSurfer() {
                if (wavesurfer) {
                    wavesurfer.destroy();
                    wavesurfer = null;
                }
            }

            document.body.addEventListener('htmx:beforeSwap', (event) => {
                if (event.target.id === 'main-content') {
                    teardownWaveSurfer();
                }
            });

            document.body.addEventListener('htmx:afterSwap', (event) => {
                if (event.target.id === 'main-content') {
                    initWaveSurfer();
                }
            });

            document.addEventListener('DOMContentLoaded', initWaveSurfer);
            """
        ),
    )


@rt("/next_clip", methods=["POST"])
def next_clip(
    clip_id: str = "",
    transcription: str = "",
    start_time: str = "0",
    end_time: str = "0",
    contributor_name: str = "",
):
    """Save progress and fetch the next clip."""

    clip = get_clip(clip_id)
    if clip:
        updates = {
            "text": transcription,
            "timestamp": datetime.now().isoformat(),
            "username": get_username(contributor_name),
        }
        try:
            start = float(start_time)
            end = float(end_time)
            if start >= 0 and end > start:
                updates["start_timestamp"] = start
                updates["end_timestamp"] = end
        except ValueError:
            pass
        db_backend.update_clip(clip.id, updates)

    next_candidate = select_random_clip()
    return render_main_content(next_candidate)


@rt("/complete_clip", methods=["POST"])
def complete_clip(
    clip_id: str = "",
    transcription: str = "",
    start_time: str = "0",
    end_time: str = "0",
    contributor_name: str = "",
):
    """Mark a clip as reviewed and fetch the next clip."""

    clip = get_clip(clip_id)
    if clip:
        updates = {
            "text": transcription,
            "timestamp": datetime.now().isoformat(),
            "username": get_username(contributor_name),
            "human_reviewed": True,
        }
        try:
            start = float(start_time)
            end = float(end_time)
            if start >= 0 and end > start:
                updates["start_timestamp"] = start
                updates["end_timestamp"] = end
        except ValueError:
            pass
        db_backend.update_clip(clip.id, updates)

    next_candidate = select_random_clip()
    return render_main_content(next_candidate)


@rt("/flag_clip", methods=["POST"])
def flag_clip(
    clip_id: str = "",
    transcription: str = "",
    start_time: str = "0",
    end_time: str = "0",
    contributor_name: str = "",
):
    """Mark a clip as problematic so it disappears from the review queue."""

    clip = get_clip(clip_id)
    if clip:
        updates = {
            "text": transcription,
            "timestamp": datetime.now().isoformat(),
            "username": get_username(contributor_name),
            "marked": True,
        }
        try:
            start = float(start_time)
            end = float(end_time)
            if start >= 0 and end > start:
                updates["start_timestamp"] = start
                updates["end_timestamp"] = end
        except ValueError:
            pass
        db_backend.update_clip(clip.id, updates)

    next_candidate = select_random_clip()
    return render_main_content(next_candidate)


@rt("/styles.css")
def get_styles():
    """Serve the CSS file."""

    css_path = Path("styles.css")
    if css_path.exists():
        return FileResponse(str(css_path), media_type="text/css")
    return Response("/* Styles not found */", media_type="text/css")


@rt(f"/{config.audio_folder}/{{audio_name:path}}")
def get_audio(audio_name: str):
    """Serve audio files with security checks."""

    if ".." in audio_name or audio_name.startswith("/"):
        return Response("Invalid path", status_code=400)

    valid_exts = (".webm", ".mp3", ".wav", ".ogg", ".m4a", ".flac")
    if not audio_name.lower().endswith(valid_exts):
        return Response("Invalid file type", status_code=400)

    audio_path = Path(config.audio_folder) / audio_name

    try:
        audio_dir = Path(config.audio_folder).resolve()
        resolved_path = audio_path.resolve()
        if not str(resolved_path).startswith(str(audio_dir)):
            return Response("Access denied", status_code=403)
    except Exception:
        return Response("Invalid path", status_code=400)

    if audio_path.exists():
        return FileResponse(
            str(audio_path),
            headers={"Cache-Control": "public, max-age=3600"},
        )
    return Response("Audio not found", status_code=404)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fh.serve(app=fasthtml_app, host="localhost", port=5001)
else:
    app = modal.App(name="audio-annotation-tool")

    data_volume = modal.Volume.from_name("audio-annotation-data", create_if_missing=True)

    requirements_path = ROOT_DIR / "requirements.txt"
    base_image = modal.Image.debian_slim(python_version="3.12")
    if requirements_path.exists():
        modal_image = base_image.pip_install_from_requirements(requirements_path)
    else:  # pragma: no cover - fallback when requirements are missing
        modal_image = base_image.pip_install("python-fasthtml==0.12.33")

    @app.function(
        image=modal_image,
        volumes={"/data": data_volume},
        allow_concurrent_inputs=100,
        secrets=[modal.Secret.from_dotenv()],
    )
    @modal.asgi_app()
    def serve():
        return fasthtml_app
