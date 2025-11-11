"""FastHTML Audio Annotation Tool - Crowdsourced clip review interface."""
from fasthtml.common import *
from starlette.responses import FileResponse, Response
from pathlib import Path
from typing import Optional
import os
import sys
import json
from datetime import datetime

try:
    import modal
except ImportError:  # pragma: no cover - optional dependency
    modal = None

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fast_audio_annotate.config import AppConfig, parse_app_config
from fast_audio_annotate.metadata import iter_audio_files, load_audio_metadata_from_file
from fast_audio_annotate.segments import compute_segment_window

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

# Runtime helpers for audio segments
# Segment constants
SEGMENT_PADDING_SECONDS = 5.0
SEGMENT_SUBDIR_NAME = "segments"
AUDIO_FOLDER_IS_REMOTE = config.audio_folder.startswith(("http://", "https://"))

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

fasthtml_serve = serve


# Helper functions
def get_username(contributor_name: str = "") -> str:
    """Return the username for audit purposes, preferring contributor name."""
    if contributor_name and contributor_name.strip():
        return contributor_name.strip()
    return os.environ.get('USER') or os.environ.get('USERNAME') or 'unknown'


def get_contributor_stats() -> dict:
    """Get statistics about contributors."""
    try:
        # Get contributor counts from database
        stats = db_backend.get_contributor_stats()
        return stats
    except Exception as e:
        print(f"Error getting contributor stats: {e}")
        return {"total_contributors": 0, "total_contributions": 0, "contributors": []}


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
    clip = db_backend.fetch_random_clip()
    return ensure_clip_segment(clip)


def get_clip(clip_id: Optional[str]) -> Optional[ClipRecord]:
    """Return a clip by id, or ``None`` if unavailable."""

    if not clip_id:
        return None
    try:
        clip = db_backend.get_clip(int(clip_id))
        return ensure_clip_segment(clip)
    except (TypeError, ValueError):
        return None


def compute_display_window(
    start: float,
    end: float,
    *,
    lower_bound: float = 0.0,
    upper_bound: Optional[float] = None,
) -> tuple[float, float]:
    """Return the playback window that surrounds the clip with a safety margin."""

    padded_start = max(lower_bound, start - CLIP_PADDING_SECONDS)
    padded_end = end + CLIP_PADDING_SECONDS

    if upper_bound is not None:
        padded_end = min(upper_bound, padded_end)

    if padded_end <= padded_start:
        minimal_length = max(end - start, 0.5)
        padded_end = padded_start + minimal_length
        if upper_bound is not None:
            padded_end = min(upper_bound, padded_end)

    return padded_start, padded_end


def ensure_clip_segment(clip: Optional[ClipRecord]) -> Optional[ClipRecord]:
    """Attach stored segment metadata to ``clip`` if available."""

    if clip is None:
        return None

    # When segment metadata is missing, synthesize a reasonable default window so the
    # review UI still offers a focused playback range using the original audio file.
    if clip.segment_start_timestamp is None or clip.segment_end_timestamp is None:
        fallback_start, fallback_end = compute_segment_window(
            clip.start_timestamp,
            clip.end_timestamp,
            padding=SEGMENT_PADDING_SECONDS,
            lower_bound=0.0,
        )
        clip.segment_start_timestamp = fallback_start
        clip.segment_end_timestamp = fallback_end

    if not clip.segment_path:
        return clip

    if AUDIO_FOLDER_IS_REMOTE:
        return clip

    audio_root = config.audio_path
    segment_path = audio_root / clip.segment_path
    if segment_path.exists():
        return clip

    # If the stored segment path is stale, drop it so the frontend falls back to the
    # original audio file without crashing.
    clip.segment_path = None
    return clip


def render_clip_editor(clip: ClipRecord) -> Div:
    """Render the editor for a single clip."""

    clip = ensure_clip_segment(clip)
    metadata = get_audio_metadata(clip.audio_path)
    segment_offset = clip.segment_start_timestamp or 0.0
    segment_end = clip.segment_end_timestamp
    padded_start, padded_end = compute_display_window(
        clip.start_timestamp,
        clip.end_timestamp,
        lower_bound=segment_offset,
        upper_bound=segment_end,
    )
    relative_clip_start = max(0.0, clip.start_timestamp - segment_offset)
    relative_clip_end = max(relative_clip_start, clip.end_timestamp - segment_offset)
    relative_display_start = max(0.0, padded_start - segment_offset)
    relative_display_end = max(relative_display_start, padded_end - segment_offset)
    segment_duration = None
    if segment_end is not None:
        segment_duration = max(0.0, segment_end - segment_offset)
    audio_path_for_playback = clip.segment_path or clip.audio_path
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

    clip_info_entries = [
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
    ]

    if clip.segment_start_timestamp is not None and clip.segment_end_timestamp is not None:
        segment_context = clip.segment_end_timestamp - clip.segment_start_timestamp
        clip_info_entries.append(
            Div(
                Strong("Segment window:"),
                Span(
                    f" {clip.segment_start_timestamp:.2f}s – {clip.segment_end_timestamp:.2f}s "
                    f"({segment_context:.2f}s total)"
                ),
                style="margin-bottom: 4px;"
            )
        )

    clip_info_entries.append(
        Div(
            Strong("Last updated by:"),
            Span(f" {clip.username} at {clip.timestamp}"),
        )
    )

    clip_info = Div(
        *clip_info_entries,
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
            style="margin-bottom: 16px;"
        ),
        Div(
            Label("Your name (optional)", style="display: block; margin-bottom: 6px; font-weight: 600; font-size: 16px; color: #495057;"),
            Input(
                value=clip.username if hasattr(clip, 'username') and clip.username and clip.username != 'unknown' else "",
                name="contributor_name",
                id="contributor-name-input",
                placeholder="Enter your name to be credited as a contributor...",
                style="width: 100%; padding: 10px; border: 1px solid #ced4da; border-radius: 6px; font-size: 14px;",
            ),
            Div(
                "💡 Your name will be used to credit your contributions to this project!",
                style="font-size: 12px; color: #6c757d; margin-top: 4px; font-style: italic;"
            ),
            style="margin-bottom: 20px;"
        ),
        id="clip-form"
    )

    actions = Div(
        Button(
            "➡️ Next clip",
            cls="next-btn",
            hx_post="/next_clip",
            hx_include="#clip-form input, #clip-form textarea",
            hx_target="#main-content",
            hx_swap="outerHTML",
            hx_indicator="#loading-next",
            style="padding: 12px 18px; border-radius: 6px; background: #ffc107; color: #000; border: none; font-size: 15px; cursor: pointer;"
        ),
        Button(
            "✅ Finish review",
            cls="complete-btn",
            hx_post="/complete_clip",
            hx_include="#clip-form input, #clip-form textarea",
            hx_target="#main-content",
            hx_swap="outerHTML",
            hx_indicator="#loading-complete",
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
            hx_indicator="#loading-flag",
            style="padding: 12px 18px; border-radius: 6px; background: #dc3545; color: white; border: none; font-size: 15px; cursor: pointer;"
        ),
        # Loading indicators
        Div(
            "🔄 Loading next clip...",
            id="loading-next",
            cls="htmx-indicator",
            style="padding: 8px 12px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; color: #856404; font-size: 14px;"
        ),
        Div(
            "✅ Completing review...",
            id="loading-complete", 
            cls="htmx-indicator",
            style="padding: 8px 12px; background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 4px; color: #0c5460; font-size: 14px;"
        ),
        Div(
            "🚩 Reporting issue...",
            id="loading-flag",
            cls="htmx-indicator", 
            style="padding: 8px 12px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px; color: #721c24; font-size: 14px;"
        ),
        style="display: flex; gap: 12px; flex-wrap: wrap; align-items: center;"
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
        data_audio_path=str(audio_path_for_playback),
        data_original_audio_path=str(clip.audio_path),
        data_clip_start=f"{clip.start_timestamp:.2f}",
        data_clip_end=f"{clip.end_timestamp:.2f}",
        data_display_start=f"{padded_start:.2f}",
        data_display_end=f"{padded_end:.2f}",
        data_segment_offset=f"{segment_offset:.2f}",
        data_segment_duration=(
            f"{segment_duration:.2f}" if segment_duration is not None else ""
        )
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


def render_contributor_stats() -> Div:
    """Render a panel showing contributor statistics."""
    try:
        stats = get_contributor_stats()
        
        if stats["total_contributors"] == 0:
            return Div(
                H4("🙏 Contributors", style="margin-bottom: 10px; color: #343a40;"),
                P("Be the first to contribute! Enter your name when reviewing clips to be credited.",
                  style="color: #6c757d; font-style: italic;"),
                cls="contributor-stats-panel",
                style=(
                    "margin-bottom: 20px; padding: 15px; background: #f8f9fa; border: 1px solid #e9ecef; "
                    "border-radius: 8px;"
                ),
            )
        
        # Show top contributors (limit to top 5)
        top_contributors = stats["contributors"][:5]
        
        contributor_list = []
        for i, contributor in enumerate(top_contributors):
            rank_emoji = ["🥇", "🥈", "🥉", "🏅", "⭐"][i] if i < 5 else "✨"
            contributor_list.append(
                Div(
                    Span(f"{rank_emoji} {contributor['name']}", style="font-weight: 600;"),
                    Span(f" - {contributor['contributions']} contributions", style="color: #6c757d; margin-left: 8px;"),
                    style="margin-bottom: 4px;"
                )
            )
        
        return Div(
            H4("🙏 Contributors", style="margin-bottom: 10px; color: #343a40;"),
            Div(
                P(f"Total contributors: {stats['total_contributors']} | Total contributions: {stats['total_contributions']}",
                  style="margin-bottom: 12px; font-weight: 500; color: #495057;"),
                *contributor_list,
                style="margin-bottom: 8px;"
            ),
            P("Thank you to everyone who has contributed to improving this dataset! 🎉",
              style="color: #198754; font-style: italic; margin-bottom: 0; font-size: 14px;"),
            cls="contributor-stats-panel",
            style=(
                "margin-bottom: 20px; padding: 15px; background: #f8f9fa; border: 1px solid #e9ecef; "
                "border-radius: 8px;"
            ),
        )
    except Exception as e:
        print(f"Error rendering contributor stats: {e}")
        return Div()  # Return empty div on error


# Routes
@rt("/")
def index():
    """Main entry point for the crowdsourced clip review interface."""
    clip = select_random_clip()
    main_content = render_main_content(clip)
    contributor_stats = render_contributor_stats()

    return Titled(
        config.title,
        Div(
            H1("Clip review"),
            contributor_stats,
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
                const segmentOffset = parseFloat(mainContent.dataset.segmentOffset || '0');
                const segmentDuration = parseFloat(mainContent.dataset.segmentDuration || '0');
                const clipStartAbsolute = parseFloat(mainContent.dataset.clipStart || '0');
                const clipEndAbsolute = parseFloat(mainContent.dataset.clipEnd || '0');
                const displayStartAbsolute = parseFloat(mainContent.dataset.displayStart || clipStartAbsolute);
                const displayEndAbsolute = parseFloat(mainContent.dataset.displayEnd || clipEndAbsolute);

                const clipStart = Math.max(0, clipStartAbsolute - segmentOffset);
                const clipEnd = Math.max(clipStart, clipEndAbsolute - segmentOffset);
                let displayStart = Math.max(0, displayStartAbsolute - segmentOffset);
                let displayEnd = Math.max(displayStart, displayEndAbsolute - segmentOffset);

                if (!Number.isNaN(segmentDuration) && segmentDuration > 0) {
                    displayStart = Math.min(displayStart, segmentDuration);
                    displayEnd = Math.min(displayEnd, segmentDuration);
                }

                if (!audioPath) {
                    return;
                }

                const startInput = document.getElementById('start-time-input');
                const endInput = document.getElementById('end-time-input');

                const clampRelativeTime = (value) => {
                    let result = Math.max(0, value);
                    if (!Number.isNaN(segmentDuration) && segmentDuration > 0) {
                        result = Math.min(result, segmentDuration);
                    }
                    return result;
                };

                const updateInputsFromRegion = () => {
                    if (!currentRegion) return;
                    const absoluteStart = segmentOffset + currentRegion.start;
                    const absoluteEnd = segmentOffset + currentRegion.end;
                    if (startInput) startInput.value = absoluteStart.toFixed(2);
                    if (endInput) endInput.value = absoluteEnd.toFixed(2);
                };

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

                // Construct audio URL - use full URL if config.audio_folder is a URL, otherwise use local path
                const audioFolder = '""" + f"{config.audio_folder}" + """';
                const audioUrl = audioFolder.startsWith('http') 
                    ? audioFolder + '/' + audioPath
                    : '/' + audioFolder + '/' + audioPath;
                
                wavesurfer.load(audioUrl);

                wavesurfer.on('ready', () => {
                    wsRegions.clearRegions();
                    currentRegion = wsRegions.addRegion({
                        start: clipStart,
                        end: clipEnd,
                        color: 'rgba(13, 110, 253, 0.3)',
                        drag: true,
                        resize: true,
                    });

                    currentRegion.on('update', updateInputsFromRegion);
                    updateInputsFromRegion();

                    const viewDuration = Math.max(0.5, displayEnd - displayStart);
                    const pxPerSec = Math.max(120, 900 / viewDuration);
                    wavesurfer.zoomTo(pxPerSec);
                    wavesurfer.setTime(displayStart);
                });

                const updateCurrentTime = () => {
                    const timeDisplay = document.getElementById('current-time');
                    if (timeDisplay && wavesurfer) {
                        const absoluteTime = segmentOffset + wavesurfer.getCurrentTime();
                        timeDisplay.textContent = absoluteTime.toFixed(2);
                    }
                };

                wavesurfer.on('audioprocess', updateCurrentTime);
                wavesurfer.on('pause', updateCurrentTime);

                if (startInput) {
                    startInput.addEventListener('input', (event) => {
                        if (!currentRegion) return;
                        const value = parseFloat(event.target.value);
                        if (!Number.isNaN(value)) {
                            const desiredStart = clampRelativeTime(value - segmentOffset);
                            currentRegion.setOptions({ start: Math.min(desiredStart, currentRegion.end) });
                        }
                    });
                }

                if (endInput) {
                    endInput.addEventListener('input', (event) => {
                        if (!currentRegion) return;
                        const value = parseFloat(event.target.value);
                        if (!Number.isNaN(value)) {
                            const desiredEnd = clampRelativeTime(value - segmentOffset);
                            currentRegion.setOptions({ end: Math.max(desiredEnd, currentRegion.start) });
                        }
                    });
                }

                const playButton = document.getElementById('play-btn');
                const pauseButton = document.getElementById('pause-btn');
                const stopButton = document.getElementById('stop-btn');
                const speedSelect = document.getElementById('speed-select');

                if (playButton) {
                    playButton.addEventListener('click', () => {
                        let start = Math.max(0, (currentRegion ? currentRegion.start : clipStart) - 0.2);
                        let end = currentRegion ? currentRegion.end + 0.2 : clipEnd;
                        if (!Number.isNaN(segmentDuration) && segmentDuration > 0) {
                            start = Math.max(0, Math.min(start, segmentDuration));
                            end = Math.max(start, Math.min(end, segmentDuration));
                        }
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
                            let start = Math.max(0, (currentRegion ? currentRegion.start : clipStart) - 0.2);
                            let end = currentRegion ? currentRegion.end + 0.2 : clipEnd;
                            if (!Number.isNaN(segmentDuration) && segmentDuration > 0) {
                                start = Math.max(0, Math.min(start, segmentDuration));
                                end = Math.max(start, Math.min(end, segmentDuration));
                            }
                            wavesurfer.play(start, end);
                        }
                    }
                    if (event.key.toLowerCase() === 'q' && currentRegion) {
                        event.preventDefault();
                        const time = clampRelativeTime(wavesurfer.getCurrentTime());
                        currentRegion.setOptions({ start: Math.min(time, currentRegion.end) });
                        updateInputsFromRegion();
                    }
                    if (event.key.toLowerCase() === 'w' && currentRegion) {
                        event.preventDefault();
                        const time = clampRelativeTime(wavesurfer.getCurrentTime());
                        currentRegion.setOptions({ end: Math.max(time, currentRegion.start) });
                        updateInputsFromRegion();
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


@rt("/next_clip", methods=["POST"])
def next_clip(clip_id: str = "", start_time: str = "0", end_time: str = "0", transcription: str = "", contributor_name: str = ""):
    """Move to the next random clip without completing the current one."""
    # Optionally save current progress before moving to next clip
    current_clip = get_clip(clip_id)
    if current_clip:
        try:
            start = float(start_time)
            end = float(end_time)
            if start >= 0 and end > start:
                updates = {
                    'start_timestamp': start,
                    'end_timestamp': end,
                    'text': transcription,
                    'timestamp': datetime.now().isoformat(),
                    'username': get_username(contributor_name),
                }
                db_backend.update_clip(current_clip.id, updates)
        except ValueError:
            pass
    
    # Get a new random clip
    next_clip = select_random_clip()
    return render_main_content(next_clip)


@rt("/complete_clip", methods=["POST"])
def complete_clip(clip_id: str = "", start_time: str = "0", end_time: str = "0", transcription: str = "", contributor_name: str = ""):
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
                    'username': get_username(contributor_name),
                    'human_reviewed': True,
                    'marked': False,
                }
                db_backend.update_clip(clip.id, updates)
        except ValueError:
            pass
    next_clip = select_random_clip()
    return render_main_content(next_clip)


@rt("/flag_clip", methods=["POST"])
def flag_clip(clip_id: str = "", transcription: str = "", start_time: str = "0", end_time: str = "0", contributor_name: str = ""):
    """Mark a clip as problematic so it disappears from the review queue."""
    clip = get_clip(clip_id)
    if clip:
        updates = {
            'text': transcription,
            'timestamp': datetime.now().isoformat(),
            'username': get_username(contributor_name),
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


# Only create local audio route if audio_folder is a local path (not a URL)
if not AUDIO_FOLDER_IS_REMOTE:
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


def get_asgi_app():
    """Return the FastHTML ASGI app instance for external servers."""

    return app


if modal is not None:
    modal_app = modal.App("fast-audio-annotate")

    requirements_file = ROOT_DIR / "requirements.txt"
    image_builder = modal.Image.debian_slim(python_version="3.12")
    if requirements_file.exists():
        if hasattr(image_builder, "pip_install_from_requirements"):
            modal_image = image_builder.pip_install_from_requirements(str(requirements_file))
        else:  # Fallback for older modal clients
            with requirements_file.open("r", encoding="utf-8") as handle:
                packages = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
            for package in packages:
                image_builder = image_builder.pip_install(package)
            modal_image = image_builder
    else:
        modal_image = image_builder.pip_install("python-fasthtml==0.12.33")

    @modal_app.function(image=modal_image)
    @modal.asgi_app()
    def serve():
        """Expose the FastHTML app as an ASGI application on Modal."""

        return get_asgi_app()


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
        fasthtml_serve(host="localhost", port=5001)
    except KeyboardInterrupt:
        print("\nShutting down...")
        print("Goodbye!")
