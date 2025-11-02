"""FastHTML Audio Annotation Tool - For transcription purposes."""
from fasthtml.common import *
from starlette.responses import FileResponse, Response
from pathlib import Path
from typing import Optional
import os
import sys
import json
from datetime import datetime


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fast_audio_annotate.config import AppConfig, parse_app_config
from fast_audio_annotate.metadata import iter_audio_files, load_audio_metadata_from_file

from db_backend import DatabaseBackend

config: AppConfig = parse_app_config()

# Database setup
database_url = (
    config.database_url
    or os.environ.get("DATABASE_URL")
    or os.environ.get("NEON_DATABASE_URL")
)
db_backend = DatabaseBackend(config.audio_path / "annotations.db", database_url)


load_audio_metadata_from_file(config.audio_path, db_backend, config.metadata_filename)

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

# State management
class AppState:
    def __init__(self):
        self.current_audio = None  # Path to current audio file
        self.current_clip_index = 0  # Index of current clip being edited
        self.audio_duration = 0  # Duration of current audio in seconds
        self.history = []

state = AppState()

# Helper functions
def get_audio_files():
    """Get all audio files from the configured directory (deduplicated)."""
    audio_dir = config.audio_path
    files = [path.relative_to(audio_dir) for path in iter_audio_files(audio_dir)]
    return sorted(dict.fromkeys(files))


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


def get_username():
    """Get current username."""
    return os.environ.get('USER') or os.environ.get('USERNAME') or 'unknown'

def get_clips_for_audio(audio_path):
    """Get all clips for a specific audio file, sorted by start time."""
    return db_backend.fetch_clips(str(audio_path) if audio_path else None)

def auto_generate_clip(audio_path, last_end_time=0):
    """Auto-generate a 10-second clip starting from last_end_time."""
    start = last_end_time
    end = start + 10.0

    return db_backend.create_clip({
        'audio_path': str(audio_path),
        'start_timestamp': start,
        'end_timestamp': end,
        'text': '',
        'username': get_username(),
        'timestamp': datetime.now().isoformat(),
        'marked': False,
    })

def get_current_clip():
    """Get the current clip being edited, auto-generating if needed."""
    if not state.current_audio:
        return None

    audio_clips = get_clips_for_audio(state.current_audio)

    # If no clips exist, create the first one
    if not audio_clips:
        new_clip = auto_generate_clip(state.current_audio, 0)
        state.current_clip_index = 0
        return new_clip

    # Ensure current_clip_index is valid
    if state.current_clip_index >= len(audio_clips):
        state.current_clip_index = len(audio_clips) - 1
    elif state.current_clip_index < 0:
        state.current_clip_index = 0

    return audio_clips[state.current_clip_index]

def get_progress_stats():
    """Calculate progress statistics."""
    audio_files = get_audio_files()
    total_audio = len(audio_files)

    if not state.current_audio:
        return {
            'total_audio': total_audio,
            'current_audio_index': 0,
            'total_clips': 0,
            'current_clip_num': 0,
            'marked_clips': 0
        }

    # Find current audio index
    try:
        current_audio_index = audio_files.index(Path(state.current_audio)) + 1
    except (ValueError, AttributeError):
        current_audio_index = 1

    # Count clips for current audio
    audio_clips = get_clips_for_audio(state.current_audio)
    total_clips = len(audio_clips)
    current_clip_num = state.current_clip_index + 1
    marked_clips = len([c for c in audio_clips if c.marked])

    return {
        'total_audio': total_audio,
        'current_audio_index': current_audio_index,
        'total_clips': total_clips,
        'current_clip_num': current_clip_num,
        'marked_clips': marked_clips
    }

def render_main_content():
    """Render the main content area (for HTMX swapping)."""
    audio_files = get_audio_files()
    current_clip = get_current_clip()
    stats = get_progress_stats()
    metadata = get_audio_metadata(state.current_audio)

    return Div(
        # Audio file selector
        Div(
            Label("Audio File:", style="margin-right: 10px; font-weight: 600;"),
            Select(
                *[Option(str(audio), value=str(audio), selected=(str(audio) == state.current_audio))
                  for audio in audio_files],
                name="audio_select",
                hx_post="/select_audio",
                hx_target="#main-content",
                hx_swap="outerHTML",
                hx_trigger="change",
                style="flex: 1; padding: 8px 12px; border-radius: 6px; border: 2px solid #007bff; background: white; font-size: 14px;"
            ),
            style="display: flex; align-items: center; margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;"
        ),

        # Progress section
        Div(
            Div(
                f"Audio {stats['current_audio_index']} of {stats['total_audio']} | ",
                f"Clip {stats['current_clip_num']} of {stats['total_clips']} | ",
                f"Reviewed: {stats['marked_clips']}",
                cls="progress"
            ),
        ),

        # Current audio filename and playback info
        Div(f"File: {state.current_audio}", cls="progress", style="font-weight: 500; margin-bottom: 10px;"),

        render_audio_metadata_panel(metadata),

        # Current time display and hotkeys info
        Div(
            Div(
                "Current Time: ",
                Span("0.00", id="current-time", style="font-weight: bold; color: #007bff;"),
                " seconds",
                style="font-size: 16px; margin-bottom: 10px;"
            ),
            Div(
                "Hotkeys: ",
                Span("[", style="color: #666;"),
                Span("Q", style="font-weight: bold; color: #28a745;"),
                Span("] Set Start Time | [", style="color: #666;"),
                Span("W", style="font-weight: bold; color: #dc3545;"),
                Span("] Set End Time | [", style="color: #666;"),
                Span("Space", style="font-weight: bold; color: #007bff;"),
                Span("] Play/Pause", style="color: #666;"),
                style="font-size: 14px; color: #666; margin-bottom: 15px;"
            ),
            style="padding: 10px; background: #f8f9fa; border-radius: 6px; margin-bottom: 15px;"
        ),

        # Waveform container
        Div(
            id="waveform",
            style="width: 100%; height: 128px; margin-bottom: 15px; background: #f0f0f0; border-radius: 4px;"
        ),

        # Timeline container
        Div(
            id="timeline",
            style="width: 100%; margin-bottom: 20px;"
        ),

        # Playback controls
        Div(
            Button("▶ Play Clip", id="play-btn", cls="control-btn", style="padding: 12px 24px; font-size: 16px;"),
            Button("⏸ Pause", id="pause-btn", cls="control-btn", style="padding: 12px 24px; font-size: 16px;"),
            Button("⏹ Stop", id="stop-btn", cls="control-btn", style="padding: 12px 24px; font-size: 16px;"),
            Label("Speed:", style="margin-left: 20px; font-weight: 500;"),
            Select(
                Option("0.5x", value="0.5"),
                Option("0.75x", value="0.75"),
                Option("1x", value="1", selected=True),
                Option("1.25x", value="1.25"),
                Option("1.5x", value="1.5"),
                Option("2x", value="2"),
                id="speed-select",
                style="padding: 8px; margin-left: 5px;"
            ),
            style="display: flex; align-items: center; justify-content: center; gap: 10px; margin-bottom: 25px; padding: 15px; background: #f8f9fa; border-radius: 8px;"
        ),

        # Current clip editor
        Div(
            H3(f"Clip {stats['current_clip_num']}", style="margin-bottom: 15px; color: #007bff;"),

            # Timestamp controls
            Div(
                Div(
                    Label("Start (seconds):", style="display: block; margin-bottom: 5px; font-weight: 500;"),
                    Input(
                        type="number",
                        name="start_time",
                        value=f"{current_clip.start_timestamp:.2f}" if current_clip else "0.00",
                        step="0.01",
                        min="0",
                        id="start-time-input",
                        style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px;"
                    ),
                    style="flex: 1;"
                ),
                Div(
                    Label("End (seconds):", style="display: block; margin-bottom: 5px; font-weight: 500;"),
                    Input(
                        type="number",
                        name="end_time",
                        value=f"{current_clip.end_timestamp:.2f}" if current_clip else "10.00",
                        step="0.01",
                        min="0",
                        id="end-time-input",
                        style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px;"
                    ),
                    style="flex: 1;"
                ),
                Button(
                    "Update Times",
                    hx_post="/update_times",
                    hx_include="#start-time-input, #end-time-input",
                    hx_target="#main-content",
                    hx_swap="outerHTML",
                    cls="update-btn",
                    style="align-self: flex-end; padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;"
                ),
                style="display: flex; gap: 15px; margin-bottom: 20px;"
            ),

            # Transcription text area
            Div(
                Label("Transcription:", style="display: block; margin-bottom: 8px; font-weight: 500; font-size: 16px;"),
                Textarea(
                    current_clip.text if current_clip and current_clip.text else "",
                    name="transcription",
                    id="transcription-input",
                    rows="6",
                    placeholder="Enter transcription for this clip...",
                    style="width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 4px; font-family: inherit; font-size: 14px; resize: vertical;"
                ),
                style="margin-bottom: 15px;"
            ),

            # Mark as reviewed
            Div(
                Label(
                    Input(
                        type="checkbox",
                        name="marked",
                        id="marked-input",
                        checked=current_clip.marked if current_clip else False
                    ),
                    " Mark as reviewed",
                    style="display: flex; align-items: center; gap: 8px; font-size: 14px; cursor: pointer;"
                ),
                style="margin-bottom: 20px;"
            ),

            # Save button
            Button(
                "💾 Save Clip",
                hx_post="/save_current_clip",
                hx_include="#transcription-input, #marked-input",
                hx_target="#main-content",
                hx_swap="outerHTML",
                cls="save-btn",
                style="width: 100%; padding: 12px; background: #28a745; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: 600;"
            ),

            style="padding: 20px; background: #f9f9f9; border: 2px solid #007bff; border-radius: 8px; margin-bottom: 20px;"
        ),

        # Navigation controls
        Div(
            Button(
                "← Previous Clip",
                cls="nav-btn",
                hx_post="/prev_clip",
                hx_target="#main-content",
                hx_swap="outerHTML",
                style="flex: 1;"
            ),
            Button(
                "Delete Clip",
                hx_post="/delete_current_clip",
                hx_target="#main-content",
                hx_swap="outerHTML",
                hx_confirm="Delete this clip?",
                cls="delete-btn",
                style="background: #dc3545; color: white; padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer;"
            ),
            Button(
                "Next Clip →",
                cls="nav-btn",
                hx_post="/next_clip",
                hx_target="#main-content",
                hx_swap="outerHTML",
                style="flex: 1;"
            ),
            cls="nav-controls",
            style="display: flex; gap: 10px; justify-content: center;"
        ),

        id="main-content",
        # Add data attributes for WaveSurfer initialization
        **{
            'data-audio-path': str(state.current_audio),
            'data-clip-start': str(current_clip.start_timestamp if current_clip else 0),
            'data-clip-end': str(current_clip.end_timestamp if current_clip else 10),
        }
    )

@rt("/")
def index():
    """Main audio annotation interface."""
    audio_files = get_audio_files()

    if not audio_files:
        return Titled(config.title,
            Div(
                H2("No Audio Files Found", style="text-align: center; margin-bottom: 20px;"),
                P(f"Please add audio files to the '{config.audio_folder}/' directory.",
                  style="text-align: center; color: #666;"),
                P("Supported formats: .webm, .mp3, .wav, .ogg, .m4a, .flac",
                  style="text-align: center; color: #999; font-size: 14px;"),
                style="max-width: 800px; margin: 2rem auto; padding: 2rem; background: white; border-radius: 8px;"
            )
        )

    # Set default audio if none selected
    if not state.current_audio:
        state.current_audio = str(audio_files[0])
        state.current_clip_index = 0

    return Titled(config.title,
        Div(
            render_main_content(),
            cls="container"
        ),

        # WaveSurfer.js initialization script
        Script("""
            let wavesurfer = null;
            let wsRegions = null;
            let currentRegion = null;

            function initWaveSurfer() {
                // Destroy existing instance if any
                if (wavesurfer) {
                    wavesurfer.destroy();
                }

                // Get data from main-content element
                const mainContent = document.getElementById('main-content');
                if (!mainContent) return;

                const audioPath = mainContent.dataset.audioPath;
                const clipStart = parseFloat(mainContent.dataset.clipStart);
                const clipEnd = parseFloat(mainContent.dataset.clipEnd);

                // Initialize WaveSurfer
                wavesurfer = WaveSurfer.create({
                    container: '#waveform',
                    waveColor: '#4F4A85',
                    progressColor: '#383351',
                    height: 128,
                    barWidth: 2,
                    barGap: 1,
                    barRadius: 2,
                    responsive: true,
                });

                // Add regions plugin
                wsRegions = wavesurfer.registerPlugin(WaveSurfer.Regions.create());

                // Add timeline plugin
                wavesurfer.registerPlugin(WaveSurfer.Timeline.create({
                    container: '#timeline',
                }));

                // Load audio file
                wavesurfer.load('""" + f"/{config.audio_folder}/" + """' + audioPath);

                wavesurfer.on('ready', () => {
                    // Clear any existing regions
                    wsRegions.clearRegions();

                    // Add current clip region
                    currentRegion = wsRegions.addRegion({
                        start: clipStart,
                        end: clipEnd,
                        color: 'rgba(0, 123, 255, 0.3)',
                        drag: true,
                        resize: true,
                    });

                    // Update input fields when region is dragged/resized
                    currentRegion.on('update', () => {
                        const startInput = document.getElementById('start-time-input');
                        const endInput = document.getElementById('end-time-input');
                        if (startInput) startInput.value = currentRegion.start.toFixed(2);
                        if (endInput) endInput.value = currentRegion.end.toFixed(2);
                    });
                });

                // Update region when input fields change
                const startInput = document.getElementById('start-time-input');
                const endInput = document.getElementById('end-time-input');

                if (startInput) {
                    startInput.addEventListener('input', (e) => {
                        if (currentRegion) {
                            const start = parseFloat(e.target.value) || 0;
                            const end = currentRegion.end;
                            currentRegion.setOptions({ start, end });
                        }
                    });
                }

                if (endInput) {
                    endInput.addEventListener('input', (e) => {
                        if (currentRegion) {
                            const start = currentRegion.start;
                            const end = parseFloat(e.target.value) || 10;
                            currentRegion.setOptions({ start, end });
                        }
                    });
                }

                // Playback controls - play only current clip
                const playBtn = document.getElementById('play-btn');
                const pauseBtn = document.getElementById('pause-btn');
                const stopBtn = document.getElementById('stop-btn');
                const speedSelect = document.getElementById('speed-select');

                if (playBtn) {
                    playBtn.addEventListener('click', () => {
                        if (currentRegion) {
                            currentRegion.play();
                        }
                    });
                }

                if (pauseBtn) {
                    pauseBtn.addEventListener('click', () => {
                        wavesurfer.pause();
                    });
                }

                if (stopBtn) {
                    stopBtn.addEventListener('click', () => {
                        wavesurfer.stop();
                    });
                }

                if (speedSelect) {
                    speedSelect.addEventListener('change', (e) => {
                        wavesurfer.setPlaybackRate(parseFloat(e.target.value));
                    });
                }

                // Update current time display
                const updateCurrentTime = () => {
                    const currentTimeEl = document.getElementById('current-time');
                    if (currentTimeEl && wavesurfer) {
                        const currentTime = wavesurfer.getCurrentTime();
                        currentTimeEl.textContent = currentTime.toFixed(2);
                    }
                };

                // Update time display during playback
                wavesurfer.on('timeupdate', updateCurrentTime);
                wavesurfer.on('seeking', updateCurrentTime);

                // Keyboard shortcuts
                document.addEventListener('keydown', (e) => {
                    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

                    switch(e.key) {
                        case ' ':
                            e.preventDefault();
                            if (currentRegion) {
                                if (wavesurfer.isPlaying()) {
                                    wavesurfer.pause();
                                } else {
                                    currentRegion.play();
                                }
                            }
                            break;
                        case 'ArrowLeft':
                            e.preventDefault();
                            wavesurfer.skip(-2);
                            break;
                        case 'ArrowRight':
                            e.preventDefault();
                            wavesurfer.skip(2);
                            break;
                        case 'q':
                        case 'Q':
                            e.preventDefault();
                            // Set start time to current playback position
                            if (wavesurfer && currentRegion) {
                                const currentTime = wavesurfer.getCurrentTime();
                                const startInput = document.getElementById('start-time-input');
                                if (startInput) {
                                    startInput.value = currentTime.toFixed(2);
                                    // Update the region
                                    const endTime = currentRegion.end;
                                    currentRegion.setOptions({ start: currentTime, end: endTime });
                                    
                                    // Send update to server
                                    fetch('/set_start_time', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                                        body: `time=${currentTime.toFixed(2)}`
                                    });
                                }
                            }
                            break;
                        case 'w':
                        case 'W':
                            e.preventDefault();
                            // Set end time to current playback position
                            if (wavesurfer && currentRegion) {
                                const currentTime = wavesurfer.getCurrentTime();
                                const endInput = document.getElementById('end-time-input');
                                if (endInput) {
                                    endInput.value = currentTime.toFixed(2);
                                    // Update the region
                                    const startTime = currentRegion.start;
                                    currentRegion.setOptions({ start: startTime, end: currentTime });
                                    
                                    // Send update to server
                                    fetch('/set_end_time', {
                                        method: 'POST',
                                        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                                        body: `time=${currentTime.toFixed(2)}`
                                    });
                                }
                            }
                            break;
                    }
                });
            }

            // Initialize on page load
            document.addEventListener('DOMContentLoaded', initWaveSurfer);

            // Re-initialize after HTMX swap
            document.body.addEventListener('htmx:afterSwap', function(event) {
                if (event.detail.target.id === 'main-content') {
                    initWaveSurfer();
                }
            });
        """)
    )

@rt("/select_audio", methods=["POST"])
def select_audio(audio_select: str = ''):
    """Switch to a different audio file."""
    if audio_select:
        state.current_audio = audio_select
        state.current_clip_index = 0
    return render_main_content()

@rt("/save_current_clip", methods=["POST"])
def save_current_clip(transcription: str = "", marked: str = ""):
    """Save the current clip's transcription and marked status."""
    current_clip = get_current_clip()
    if current_clip:
        db_backend.update_clip(
            current_clip.id,
            {
                'text': transcription,
                'marked': marked == "on",
                'timestamp': datetime.now().isoformat(),
            },
        )
    return render_main_content()

@rt("/update_times", methods=["POST"])
def update_times(start_time: str = "0", end_time: str = "10"):
    """Update the current clip's start and end times."""
    current_clip = get_current_clip()
    if current_clip:
        try:
            start = float(start_time)
            end = float(end_time)
            if start >= 0 and end > start:
                db_backend.update_clip(
                    current_clip.id,
                    {
                        'start_timestamp': start,
                        'end_timestamp': end,
                        'timestamp': datetime.now().isoformat(),
                    },
                )
        except ValueError:
            pass
    return render_main_content()

@rt("/set_start_time", methods=["POST"])
def set_start_time(time: str = "0"):
    """Set the start time for the current clip."""
    current_clip = get_current_clip()
    if current_clip:
        try:
            start_time = float(time)
            if start_time >= 0 and start_time < current_clip.end_timestamp:
                db_backend.update_clip(
                    current_clip.id,
                    {
                        'start_timestamp': start_time,
                        'timestamp': datetime.now().isoformat(),
                    },
                )
        except ValueError:
            pass
    return render_main_content()

@rt("/set_end_time", methods=["POST"])
def set_end_time(time: str = "10"):
    """Set the end time for the current clip."""
    current_clip = get_current_clip()
    if current_clip:
        try:
            end_time = float(time)
            if end_time > current_clip.start_timestamp:
                db_backend.update_clip(
                    current_clip.id,
                    {
                        'end_timestamp': end_time,
                        'timestamp': datetime.now().isoformat(),
                    },
                )
        except ValueError:
            pass
    return render_main_content()

@rt("/prev_clip", methods=["POST"])
def prev_clip():
    """Navigate to previous clip, or stay at first clip."""
    if state.current_clip_index > 0:
        state.current_clip_index -= 1
    return render_main_content()

@rt("/next_clip", methods=["POST"])
def next_clip():
    """Navigate to next clip, auto-generating if needed."""
    audio_clips = get_clips_for_audio(state.current_audio)

    # If we're at the last clip, generate a new one
    if state.current_clip_index >= len(audio_clips) - 1:
        last_clip = audio_clips[-1] if audio_clips else None
        last_end = last_clip.end_timestamp if last_clip else 0
        auto_generate_clip(state.current_audio, last_end)

    # Move to next clip
    state.current_clip_index += 1

    return render_main_content()

@rt("/delete_current_clip", methods=["POST"])
def delete_current_clip():
    """Delete the current clip."""
    current_clip = get_current_clip()
    if current_clip:
        db_backend.delete_clip(current_clip.id)
        # Adjust index if needed
        audio_clips = get_clips_for_audio(state.current_audio)
        if state.current_clip_index >= len(audio_clips) and state.current_clip_index > 0:
            state.current_clip_index -= 1
    return render_main_content()

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
    except:
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
    print(f"Configuration:")
    print(f"  - Audio folder: {config.audio_folder}")
    print(f"  - Database: {db_backend.backend_label()}")
    print(f"  - Annotating as: {get_username()}")

    audio_files = get_audio_files()
    print(f"  - Total audio files: {len(audio_files)}")

    total_clips = db_backend.count_clips()
    print(f"  - Total clips: {total_clips}")

    try:
        serve(host="localhost", port=5001)
    except KeyboardInterrupt:
        print("\nShutting down...")
        print("Goodbye!")
