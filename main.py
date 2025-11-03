"""FastHTML Audio Annotation Tool - Crowdsourced clip review interface."""
from fasthtml.common import *
from starlette.responses import FileResponse, Response
from pathlib import Path
from typing import Optional
import os
import sys
import json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fast_audio_annotate.config import AppConfig, parse_app_config
from fast_audio_annotate.metadata import iter_audio_files, load_audio_metadata_from_file

from db_backend import ClipRecord, DatabaseBackend

config: AppConfig = parse_app_config()

# Database setup
database_url = (
    config.database_url
    or os.environ.get("DATABASE_URL")
    or os.environ.get("NEON_DATABASE_URL")
)
db_backend = DatabaseBackend(config.audio_path / "annotations.db", database_url)


load_audio_metadata_from_file(config.audio_path, db_backend, config.metadata_filename)

# Constants for the review workflow
CLIP_PADDING_SECONDS = 1.5

# Initialize FastHTML app with custom styles and scripts
app, rt = fast_app(
    hdrs=(
        Link(rel='stylesheet', href='/styles.css'),
        # WaveSurfer.js and plugins
        Script(src='https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.min.js'),
        Script(src='https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/regions.min.js'),
        Script(src='https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/timeline.min.js'),
    ),
    pico=False,
    debug=True
)


# Helper functions
def get_username() -> str:
    """Return the username for audit purposes."""
    return os.environ.get('USER') or os.environ.get('USERNAME') or 'unknown'


def get_audio_metadata(audio_path: Optional[str]) -> Optional[dict]:
    """Fetch metadata for an audio file from the database."""

    if not audio_path:
        return None

    record = db_backend.fetch_audio_metadata(str(audio_path))
    return record.metadata if record else None


def render_audio_metadata_panel(metadata: Optional[dict]):
    """Render a panel summarizing audio metadata."""

    if not metadata:
        body = Div(
            "No metadata available for this audio file.",
            style="color: #666; font-style: italic;",
        )
    else:
        entries = []
        for key, value in sorted(metadata.items(), key=lambda item: str(item[0])):
            if isinstance(value, (dict, list)):
                formatted_value = json.dumps(value, ensure_ascii=False, indent=2)
                value_node = Pre(
                    formatted_value,
                    style=(
                        "margin: 0; white-space: pre-wrap; background: #f1f3f5; padding: 8px; "
                        "border-radius: 4px; flex: 1; font-family: 'Fira Code', monospace; font-size: 13px;"
                    ),
                )
            else:
                value_node = Span(str(value))

            entries.append(
                Div(
                    Span(f"{key}:", style="font-weight: 600; min-width: 120px;"),
                    value_node,
                    style="display: flex; gap: 8px; align-items: flex-start;",
                )
            )

        body = Div(
            *entries,
            style="display: flex; flex-direction: column; gap: 6px;"
        )

    return Div(
        H4("Audio Metadata", style="margin-bottom: 10px; color: #343a40;"),
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


def render_clip_editor(clip: ClipRecord) -> Div:
    """Render the editor for a single clip."""

    metadata = get_audio_metadata(clip.audio_path)
    padded_start, padded_end = compute_display_window(clip.start_timestamp, clip.end_timestamp)
    duration = clip.end_timestamp - clip.start_timestamp

    instructions = Div(
        H3("How to review this clip", style="margin-bottom: 8px; color: #0d6efd;"),
        P(
            "Listen carefully to the highlighted audio, correct the transcription so it matches the speech exactly, "
            "and adjust the start/end times if they need a better cut. Use the buttons below to either save your progress, "
            "mark the clip as reviewed, or report an issue if the audio is unusable."
        ),
        style="margin-bottom: 18px; background: #f8f9fa; padding: 16px; border-radius: 8px; border: 1px solid #dee2e6;"
    )

    clip_info = Div(
        Div(
            Strong("Audio file:"),
            Span(f" {clip.audio_path}"),
            style="margin-bottom: 4px;"
        ),
        Div(
            Strong("Clip window:"),
            Span(f" {clip.start_timestamp:.2f}s – {clip.end_timestamp:.2f}s ({duration:.2f}s long)"),
            style="margin-bottom: 4px;"
        ),
        Div(
            Strong("Last updated by:"),
            Span(f" {clip.username} at {clip.timestamp}"),
        ),
        style="margin-bottom: 16px; display: flex; flex-direction: column; gap: 4px;"
    )

    form_inputs = Div(
        Input(type="hidden", name="clip_id", value=str(clip.id)),
        Div(
            Div(
                Label("Start (seconds)", style="display: block; margin-bottom: 4px; font-weight: 600;"),
                Input(
                    type="number",
                    name="start_time",
                    value=f"{clip.start_timestamp:.2f}",
                    step="0.01",
                    min="0",
                    id="start-time-input",
                    style="width: 100%; padding: 10px; border: 1px solid #ced4da; border-radius: 6px; font-size: 14px;",
                ),
            ),
            Div(
                Label("End (seconds)", style="display: block; margin-bottom: 4px; font-weight: 600;"),
                Input(
                    type="number",
                    name="end_time",
                    value=f"{clip.end_timestamp:.2f}",
                    step="0.01",
                    min="0",
                    id="end-time-input",
                    style="width: 100%; padding: 10px; border: 1px solid #ced4da; border-radius: 6px; font-size: 14px;",
                ),
            ),
            style="display: flex; gap: 16px; margin-bottom: 16px; flex-wrap: wrap;"
        ),
        Div(
            Label("Transcription", style="display: block; margin-bottom: 6px; font-weight: 600; font-size: 16px;"),
            Textarea(
                clip.text or "",
                name="transcription",
                id="transcription-input",
                rows="6",
                placeholder="Type the corrected transcription here...",
                style="width: 100%; padding: 12px; border: 1px solid #ced4da; border-radius: 6px; font-size: 15px; resize: vertical;",
            ),
            style="margin-bottom: 20px;"
        ),
        id="clip-form"
    )

    actions = Div(
        Button(
            "💾 Save progress",
            cls="save-btn",
            hx_post="/save_clip",
            hx_include="#clip-form input, #clip-form textarea",
            hx_target="#main-content",
            hx_swap="outerHTML",
            style="padding: 12px 18px; border-radius: 6px; background: #198754; color: white; border: none; font-size: 15px; cursor: pointer;"
        ),
        Button(
            "✅ Finish review",
            cls="complete-btn",
            hx_post="/complete_clip",
            hx_include="#clip-form input, #clip-form textarea",
            hx_target="#main-content",
            hx_swap="outerHTML",
            style="padding: 12px 18px; border-radius: 6px; background: #0d6efd; color: white; border: none; font-size: 15px; cursor: pointer;"
        ),
        Button(
            "🚩 Report issue",
            cls="flag-btn",
            hx_post="/flag_clip",
            hx_include="#clip-form input, #clip-form textarea",
            hx_confirm="Report this clip as problematic?",
            hx_target="#main-content",
            hx_swap="outerHTML",
            style="padding: 12px 18px; border-radius: 6px; background: #dc3545; color: white; border: none; font-size: 15px; cursor: pointer;"
        ),
        style="display: flex; gap: 12px; flex-wrap: wrap;"
    )

    waveform = Div(
        Div(
            Div(
                "Current Time: ",
                Span("0.00", id="current-time", style="font-weight: bold; color: #0d6efd;"),
                " s",
                style="font-size: 16px; margin-bottom: 12px;"
            ),
            Div(
                "Hotkeys: [",
                Span("Q", style="font-weight: 600; color: #198754;"),
                "] start • [",
                Span("W", style="font-weight: 600; color: #dc3545;"),
                "] end • [",
                Span("Space", style="font-weight: 600; color: #0d6efd;"),
                "] play/pause",
                style="color: #6c757d; font-size: 14px;"
            ),
            style="margin-bottom: 16px;"
        ),
        Div(id="waveform", style="width: 100%; height: 140px; background: #f1f3f5; border-radius: 8px; margin-bottom: 12px;"),
        Div(id="timeline", style="width: 100%; margin-bottom: 16px;"),
        Div(
            Button("▶ Play", id="play-btn", cls="control-btn", style="padding: 10px 18px; font-size: 15px;"),
            Button("⏸ Pause", id="pause-btn", cls="control-btn", style="padding: 10px 18px; font-size: 15px;"),
            Button("⏹ Stop", id="stop-btn", cls="control-btn", style="padding: 10px 18px; font-size: 15px;"),
            Label("Speed:", style="margin-left: 12px; font-weight: 600;"),
            Select(
                Option("0.75x", value="0.75"),
                Option("1x", value="1", selected=True),
                Option("1.25x", value="1.25"),
                Option("1.5x", value="1.5"),
                Option("2x", value="2"),
                id="speed-select",
                style="padding: 8px; border-radius: 6px; border: 1px solid #ced4da;"
            ),
            style="display: flex; align-items: center; gap: 10px; justify-content: center;"
        ),
        style="margin-bottom: 24px;"
    )

    metadata_panel = render_audio_metadata_panel(metadata)

    return Div(
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
        data_display_end=f"{padded_end:.2f}"
    )


def render_empty_state() -> Div:
    """Render a friendly message when no clips are available."""

    return Div(
        H2("All caught up!", style="text-align: center; color: #198754;"),
        P(
            "There are no clips waiting for human review right now. Please check back later.",
            style="text-align: center; font-size: 16px; color: #6c757d;"
        ),
        id="main-content",
        style="max-width: 640px; margin: 60px auto; background: white; padding: 32px; border-radius: 12px;"
    )


def render_main_content(clip: Optional[ClipRecord]) -> Div:
    """Render the main content area."""
    if clip:
        return render_clip_editor(clip)
    return render_empty_state()


# Routes
@rt("/")
def index():
    """Main entry point for the crowdsourced clip review interface."""
    clip = select_random_clip()
    main_content = render_main_content(clip)

    return Titled(
        config.title,
        Div(
            H1("Clip review"),
            main_content,
            cls="container"
        ),
        Script("""
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

                wavesurfer.load('/""" + f"{config.audio_folder}" + """/' + audioPath);

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
                    wavesurfer.zoomTo(pxPerSec);
                    wavesurfer.setTime(displayStart);
                });

                const updateCurrentTime = () => {
                    const timeDisplay = document.getElementById('current-time');
                    if (timeDisplay && wavesurfer) {
                        timeDisplay.textContent = wavesurfer.getCurrentTime().toFixed(2);
                    }
                };

                wavesurfer.on('audioprocess', updateCurrentTime);
                wavesurfer.on('pause', updateCurrentTime);

                const startInput = document.getElementById('start-time-input');
                const endInput = document.getElementById('end-time-input');

                if (startInput) {
                    startInput.addEventListener('input', (event) => {
                        if (!currentRegion) return;
                        const value = parseFloat(event.target.value);
                        if (!Number.isNaN(value)) {
                            currentRegion.setOptions({ start: value });
                        }
                    });
                }

                if (endInput) {
                    endInput.addEventListener('input', (event) => {
                        if (!currentRegion) return;
                        const value = parseFloat(event.target.value);
                        if (!Number.isNaN(value)) {
                            currentRegion.setOptions({ end: value });
                        }
                    });
                }

                const playButton = document.getElementById('play-btn');
                const pauseButton = document.getElementById('pause-btn');
                const stopButton = document.getElementById('stop-btn');
                const speedSelect = document.getElementById('speed-select');

                if (playButton) {
                    playButton.addEventListener('click', () => {
                        const start = Math.max(0, (currentRegion ? currentRegion.start : clipStart) - 0.2);
                        const end = currentRegion ? currentRegion.end + 0.2 : clipEnd;
                        wavesurfer.play(start, end);
                    });
                }

                if (pauseButton) {
                    pauseButton.addEventListener('click', () => wavesurfer && wavesurfer.pause());
                }

                if (stopButton) {
                    stopButton.addEventListener('click', () => {
                        if (!wavesurfer) return;
                        wavesurfer.stop();
                        wavesurfer.setTime(displayStart);
                    });
                }

                if (speedSelect) {
                    speedSelect.addEventListener('change', (event) => {
                        const rate = parseFloat(event.target.value);
                        if (!Number.isNaN(rate) && wavesurfer) {
                            wavesurfer.setPlaybackRate(rate);
                        }
                    });
                }

                document.addEventListener('keydown', (event) => {
                    if (!wavesurfer) return;
                    if (event.target && ['INPUT', 'TEXTAREA'].includes(event.target.tagName)) {
                        return;
                    }
                    if (event.code === 'Space') {
                        event.preventDefault();
                        if (wavesurfer.isPlaying()) {
                            wavesurfer.pause();
                        } else {
                            const start = Math.max(0, (currentRegion ? currentRegion.start : clipStart) - 0.2);
                            const end = currentRegion ? currentRegion.end + 0.2 : clipEnd;
                            wavesurfer.play(start, end);
                        }
                    }
                    if (event.key.toLowerCase() === 'q' && currentRegion) {
                        event.preventDefault();
                        const time = wavesurfer.getCurrentTime();
                        currentRegion.setOptions({ start: time });
                        const startInputEl = document.getElementById('start-time-input');
                        if (startInputEl) startInputEl.value = time.toFixed(2);
                    }
                    if (event.key.toLowerCase() === 'w' && currentRegion) {
                        event.preventDefault();
                        const time = wavesurfer.getCurrentTime();
                        currentRegion.setOptions({ end: time });
                        const endInputEl = document.getElementById('end-time-input');
                        if (endInputEl) endInputEl.value = time.toFixed(2);
                    }
                });
            }

            document.addEventListener('DOMContentLoaded', initWaveSurfer);
            document.body.addEventListener('htmx:afterSwap', (event) => {
                if (event.target.id === 'main-content') {
                    initWaveSurfer();
                }
            });
        """)
    )


@rt("/save_clip", methods=["POST"])
def save_clip(clip_id: str = "", start_time: str = "0", end_time: str = "0", transcription: str = ""):
    """Persist current progress for the active clip."""
    clip = get_clip(clip_id)
    if clip:
        try:
            start = float(start_time)
            end = float(end_time)
            if start >= 0 and end > start:
                updates = {
                    'start_timestamp': start,
                    'end_timestamp': end,
                    'text': transcription,
                    'timestamp': datetime.now().isoformat(),
                    'username': get_username(),
                }
                db_backend.update_clip(clip.id, updates)
                clip = get_clip(clip_id)
        except ValueError:
            pass
    return render_main_content(clip)


@rt("/complete_clip", methods=["POST"])
def complete_clip(clip_id: str = "", start_time: str = "0", end_time: str = "0", transcription: str = ""):
    """Finalize a clip as human reviewed and move to another task."""
    clip = get_clip(clip_id)
    if clip:
        try:
            start = float(start_time)
            end = float(end_time)
            if start >= 0 and end > start:
                updates = {
                    'start_timestamp': start,
                    'end_timestamp': end,
                    'text': transcription,
                    'timestamp': datetime.now().isoformat(),
                    'username': get_username(),
                    'human_reviewed': True,
                    'marked': False,
                }
                db_backend.update_clip(clip.id, updates)
        except ValueError:
            pass
    next_clip = select_random_clip()
    return render_main_content(next_clip)


@rt("/flag_clip", methods=["POST"])
def flag_clip(clip_id: str = "", transcription: str = "", start_time: str = "0", end_time: str = "0"):
    """Mark a clip as problematic so it disappears from the review queue."""
    clip = get_clip(clip_id)
    if clip:
        updates = {
            'text': transcription,
            'timestamp': datetime.now().isoformat(),
            'username': get_username(),
            'marked': True,
        }
        try:
            start = float(start_time)
            end = float(end_time)
            if start >= 0 and end > start:
                updates['start_timestamp'] = start
                updates['end_timestamp'] = end
        except ValueError:
            pass
        db_backend.update_clip(clip.id, updates)
    next_clip = select_random_clip()
    return render_main_content(next_clip)


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
    # Check for path traversal attempts
    if ".." in audio_name or audio_name.startswith("/"):
        return Response("Invalid path", status_code=400)

    # Validate file extension
    valid_exts = ('.webm', '.mp3', '.wav', '.ogg', '.m4a', '.flac')
    if not audio_name.lower().endswith(valid_exts):
        return Response("Invalid file type", status_code=400)

    audio_path = Path(config.audio_folder) / audio_name

    # Ensure the resolved path is within audio directory
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
            headers={"Cache-Control": "public, max-age=3600"}
        )
    return Response("Audio not found", status_code=404)


# Print startup info
if __name__ == "__main__":
    print(f"Starting {config.title}")
    print("Configuration:")
    print(f"  - Audio folder: {config.audio_folder}")
    print(f"  - Database: {db_backend.backend_label()}")
    print(f"  - Annotating as: {get_username()}")

    audio_files = [path for path in iter_audio_files(config.audio_path)]
    print(f"  - Total audio files: {len(audio_files)}")

    total_clips = db_backend.count_clips()
    print(f"  - Total clips: {total_clips}")

    try:
        serve(host="localhost", port=5001)
    except KeyboardInterrupt:
        print("\nShutting down...")
        print("Goodbye!")
