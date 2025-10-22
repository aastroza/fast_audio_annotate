"""FastHTML Audio Annotation Tool - For transcription purposes."""
from fasthtml.common import *
from starlette.responses import FileResponse, Response
from pathlib import Path
import os
from datetime import datetime
from urllib.parse import urlencode
from dataclasses import dataclass
import simple_parsing as sp
import json

@dataclass
class Config:
    audio_folder: str = sp.field(positional=True, help="The folder containing the audio files and annotations.db")
    title: str = "Audio Transcription Tool"
    description: str = "Annotate audio clips with transcriptions"
    max_history: int = 10

config = sp.parse(Config, config_path="./config.yaml")

# Database setup
db = None

class Clip:
    id: int
    audio_path: str
    start_timestamp: float
    end_timestamp: float
    text: str
    username: str
    timestamp: str
    marked: bool = False

clips = None

def switch_folder(new_folder: str):
    """Switch to a different data folder."""
    global config, db, clips, state

    folder_path = get_folder_path(new_folder)
    if not folder_path:
        print(f"Warning: Could not find folder path for {new_folder}")
        return

    config.audio_folder = folder_path
    db = database(f'{config.audio_folder}/annotations.db')
    clips = db.create(Clip, pk='id')

    state.current_index = 0
    state.current_audio = None
    state.history.clear()

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
        self.current_index = 0
        self.current_audio = None
        self.history = []

state = AppState()

# Helper functions
def get_audio_files():
    """Get all audio files from the configured directory."""
    audio_dir = Path(config.audio_folder)
    audio_files = []
    if audio_dir.exists():
        for ext in ['.webm', '.mp3', '.wav', '.ogg', '.m4a', '.flac']:
            audio_files.extend(audio_dir.rglob(f"*{ext}"))
            audio_files.extend(audio_dir.rglob(f"*{ext.upper()}"))
    return sorted([audio.relative_to(audio_dir) for audio in audio_files])

def find_annotation_folders(search_dir: Path = None):
    """Find all folders containing annotations.db files."""
    if search_dir is None:
        search_dir = Path(".")

    annotation_folders = []
    try:
        for item in search_dir.iterdir():
            if item.is_dir() and (item / "annotations.db").exists():
                try:
                    rel_path = item.relative_to(search_dir)
                    annotation_folders.append({
                        "name": str(rel_path),
                        "path": str(item)
                    })
                except ValueError:
                    continue
    except (PermissionError, OSError):
        pass

    return sorted(annotation_folders, key=lambda x: x["name"])

def get_available_folders():
    """Get all available annotation folders."""
    if hasattr(config, 'audio_folder') and config.audio_folder and Path(config.audio_folder).exists():
        current_folder = Path(config.audio_folder)
        parent_dir = current_folder.parent
        annotation_folders = find_annotation_folders(parent_dir)
        return [f["name"] for f in annotation_folders]

    for search_path in [Path("data"), Path(".")]:
        if search_path.exists():
            annotation_folders = find_annotation_folders(search_path)
            if annotation_folders:
                return [f["name"] for f in annotation_folders]

    return []

def get_folder_path(folder_name: str):
    """Get the full path for a folder name."""
    if hasattr(config, 'audio_folder') and config.audio_folder and Path(config.audio_folder).exists():
        current_folder = Path(config.audio_folder)
        parent_dir = current_folder.parent
        annotation_folders = find_annotation_folders(parent_dir)
        for folder in annotation_folders:
            if folder["name"] == folder_name:
                return folder["path"]

    for search_path in [Path("data"), Path(".")]:
        if search_path.exists():
            annotation_folders = find_annotation_folders(search_path)
            for folder in annotation_folders:
                if folder["name"] == folder_name:
                    return folder["path"]

    return None

def get_username():
    """Get current username."""
    return os.environ.get('USER') or os.environ.get('USERNAME') or 'unknown'

def get_current_audio():
    """Get current audio file based on state."""
    audio_files = get_audio_files()
    if not audio_files:
        return None
    if 0 <= state.current_index < len(audio_files):
        return audio_files[state.current_index]
    return None

def get_clips_for_audio(audio_path):
    """Get all clips for a specific audio file."""
    return clips("audio_path=?", (str(audio_path),))

def get_progress_stats():
    """Calculate progress statistics."""
    audio_files = get_audio_files()
    total = len(audio_files)

    # Count how many audio files have at least one clip
    annotated_audio = set(c.audio_path for c in clips())
    annotated_count = len(annotated_audio)

    # Count total clips and marked clips
    all_clips = clips()
    total_clips = len(all_clips)
    marked_clips = len([c for c in all_clips if c.marked])

    return {
        'total_audio': total,
        'annotated_audio': annotated_count,
        'total_clips': total_clips,
        'marked_clips': marked_clips,
        'remaining_audio': total - annotated_count,
        'percentage': round(100 * annotated_count / total) if total > 0 else 0
    }

def index_of_audio(audio_name: str) -> int:
    """Return index of an audio file in the list."""
    audio_files = get_audio_files()
    for i, p in enumerate(audio_files):
        if str(p) == audio_name:
            return i
    return -1

@rt("/")
def index():
    """Main audio annotation interface."""
    # Check if we have a valid audio_folder
    if not hasattr(config, 'audio_folder') or not config.audio_folder or not Path(config.audio_folder).exists():
        available_folders = get_available_folders()
        return Titled(config.title,
            Div(
                H2("Select a Folder to Start Annotating", style="text-align: center; margin-bottom: 30px;"),
                Div(
                    "Please select a folder containing an annotations.db file to begin.",
                    style="text-align: center; margin-bottom: 30px; color: #666;"
                ),
                Div(
                    *[Div(
                        Span(f"📁 {folder}", style="flex: 1;"),
                        Button(
                            "Select",
                            hx_post="/switch_folder",
                            hx_vals=f"js:{{folder_select: '{folder}'}}",
                            hx_target="body",
                            hx_swap="outerHTML",
                            style="padding: 6px 12px; border-radius: 4px; border: 1px solid #007bff; background: #007bff; color: white; cursor: pointer;"
                        ),
                        style="display: flex; align-items: center; justify-content: space-between; padding: 10px; margin-bottom: 5px; border: 1px solid #ddd; border-radius: 4px; background: white;"
                    ) for folder in available_folders[:10]],
                    style="max-width: 600px; margin: 0 auto;"
                ),
                style="max-width: 800px; margin: 2rem auto; padding: 2rem; background: white; border-radius: 8px;"
            )
        )

    audio_files = get_audio_files()
    if not audio_files:
        return Titled(config.title,
            Div(f"No audio files found in {config.audio_folder}/ directory",
                style="max-width: 800px; margin: 2rem auto; padding: 2rem; background: white; border-radius: 8px;")
        )

    current_audio = get_current_audio()
    if not current_audio:
        state.current_index = 0
        current_audio = get_current_audio()

    state.current_audio = str(current_audio)
    audio_clips = get_clips_for_audio(current_audio) if current_audio else []
    stats = get_progress_stats()

    return Titled(config.title,
        Div(
            # Folder selection
            Div(
                Label("Choose Folder:", style="margin-right: 10px; font-weight: 600;"),
                Select(
                    *[Option(folder, value=folder, selected=(get_folder_path(folder) == config.audio_folder))
                      for folder in get_available_folders()],
                    name="folder_select",
                    hx_post="/switch_folder",
                    hx_target="body",
                    hx_swap="outerHTML",
                    hx_trigger="change",
                    style="padding: 8px 12px; border-radius: 6px; border: 2px solid #007bff; background: white; font-size: 14px; min-width: 300px;"
                ),
                style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;"
            ),

            # Progress section
            Div(
                Div(
                    f"Audio {state.current_index + 1} of {stats['total_audio']} | ",
                    f"Annotated: {stats['annotated_audio']}/{stats['total_audio']} ({stats['percentage']}%) | ",
                    f"Total Clips: {stats['total_clips']} | ",
                    f"Marked: {stats['marked_clips']}",
                    cls="progress"
                ),
                Div(
                    Div(style=f"width: {stats['percentage']}%", cls="progress-fill"),
                    cls="progress-bar"
                ),
            ),

            # Current audio info
            Div(f"Current: {current_audio}", cls="progress", style="font-weight: 500; margin-bottom: 10px;"),

            # Navigation controls
            Div(
                Button(
                    "← Previous Audio",
                    cls="nav-btn",
                    hx_post="/prev_audio",
                    hx_target="body",
                    hx_swap="outerHTML",
                    disabled=state.current_index == 0
                ),
                Button(
                    "Next Audio →",
                    cls="nav-btn",
                    hx_post="/next_audio",
                    hx_target="body",
                    hx_swap="outerHTML",
                    disabled=state.current_index >= len(audio_files) - 1
                ),
                cls="nav-controls",
                style="margin-bottom: 20px; display: flex; gap: 10px; justify-content: center;"
            ),

            # Waveform container
            Div(
                id="waveform",
                style="width: 100%; height: 128px; margin-bottom: 20px; background: #f0f0f0; border-radius: 4px;"
            ),

            # Timeline container
            Div(
                id="timeline",
                style="width: 100%; margin-bottom: 20px;"
            ),

            # Playback controls
            Div(
                Button("▶ Play", id="play-btn", cls="control-btn", style="padding: 10px 20px; font-size: 16px;"),
                Button("⏸ Pause", id="pause-btn", cls="control-btn", style="padding: 10px 20px; font-size: 16px;"),
                Button("⏹ Stop", id="stop-btn", cls="control-btn", style="padding: 10px 20px; font-size: 16px;"),
                Label("Speed:", style="margin-left: 20px;"),
                Select(
                    Option("0.5x", value="0.5"),
                    Option("0.75x", value="0.75"),
                    Option("1x", value="1", selected=True),
                    Option("1.25x", value="1.25"),
                    Option("1.5x", value="1.5"),
                    Option("2x", value="2"),
                    id="speed-select",
                    style="padding: 5px; margin-left: 5px;"
                ),
                style="display: flex; align-items: center; justify-content: center; gap: 10px; margin-bottom: 30px; padding: 15px; background: #f8f9fa; border-radius: 8px;"
            ),

            # Clips section
            Div(
                H3("Clips", style="margin-bottom: 15px;"),
                Div(
                    *[render_clip(clip) for clip in audio_clips],
                    id="clips-list",
                    style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; border-radius: 4px; padding: 10px; background: white;"
                ) if audio_clips else Div(
                    "No clips yet. Click and drag on the waveform to create a clip.",
                    id="clips-list",
                    style="padding: 20px; text-align: center; color: #666; border: 1px dashed #ccc; border-radius: 4px; background: #fafafa;"
                ),
                style="margin-bottom: 20px;"
            ),

            cls="container"
        ),

        # WaveSurfer.js initialization script
        Script(f"""
            let wavesurfer;
            let wsRegions;

            document.addEventListener('DOMContentLoaded', function() {{
                // Initialize WaveSurfer
                wavesurfer = WaveSurfer.create({{
                    container: '#waveform',
                    waveColor: '#4F4A85',
                    progressColor: '#383351',
                    height: 128,
                    barWidth: 2,
                    barGap: 1,
                    barRadius: 2,
                    responsive: true,
                }});

                // Add regions plugin
                wsRegions = wavesurfer.registerPlugin(WaveSurfer.Regions.create());

                // Add timeline plugin
                wavesurfer.registerPlugin(WaveSurfer.Timeline.create({{
                    container: '#timeline',
                }}));

                // Load audio file
                wavesurfer.load('/{config.audio_folder}/{current_audio}');

                // Load existing clips as regions
                const existingClips = {json.dumps([{"id": c.id, "start": c.start_timestamp, "end": c.end_timestamp, "text": c.text} for c in audio_clips])};

                wavesurfer.on('ready', () => {{
                    existingClips.forEach(clip => {{
                        wsRegions.addRegion({{
                            id: 'clip-' + clip.id,
                            start: clip.start,
                            end: clip.end,
                            color: 'rgba(0, 123, 255, 0.3)',
                            drag: true,
                            resize: true,
                        }});
                    }});
                }});

                // Handle region creation
                wsRegions.enableDragSelection({{
                    color: 'rgba(0, 200, 0, 0.3)',
                }});

                wsRegions.on('region-created', (region) => {{
                    console.log('Region created:', region.start, region.end);
                }});

                // Handle region update (drag/resize)
                wsRegions.on('region-updated', (region) => {{
                    const clipId = region.id.replace('clip-', '');
                    if (clipId && clipId !== region.id) {{
                        // Update existing clip
                        htmx.ajax('POST', '/update_clip_times', {{
                            values: {{
                                clip_id: clipId,
                                start: region.start,
                                end: region.end
                            }},
                            swap: 'none'
                        }});
                    }}
                }});

                // Handle region click (play that region)
                wsRegions.on('region-clicked', (region, e) => {{
                    e.stopPropagation();
                    region.play();
                }});

                // Playback controls
                document.getElementById('play-btn').addEventListener('click', () => {{
                    wavesurfer.play();
                }});

                document.getElementById('pause-btn').addEventListener('click', () => {{
                    wavesurfer.pause();
                }});

                document.getElementById('stop-btn').addEventListener('click', () => {{
                    wavesurfer.stop();
                }});

                document.getElementById('speed-select').addEventListener('change', (e) => {{
                    wavesurfer.setPlaybackRate(parseFloat(e.target.value));
                }});

                // Handle double-click on waveform to create clip
                wsRegions.on('region-double-clicked', (region, e) => {{
                    e.stopPropagation();
                    // Get clip ID if this is an existing clip
                    const clipId = region.id.replace('clip-', '');
                    if (clipId && clipId !== region.id) {{
                        // This is an existing clip - open edit form
                        const clipElement = document.getElementById('clip-' + clipId);
                        if (clipElement) {{
                            const editBtn = clipElement.querySelector('[data-action="edit"]');
                            if (editBtn) editBtn.click();
                        }}
                    }} else {{
                        // This is a new region - create clip
                        htmx.ajax('POST', '/create_clip', {{
                            values: {{
                                start: region.start,
                                end: region.end
                            }},
                            target: 'body',
                            swap: 'outerHTML'
                        }});
                    }}
                }});

                // Keyboard shortcuts
                document.addEventListener('keydown', (e) => {{
                    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

                    switch(e.key) {{
                        case ' ':
                            e.preventDefault();
                            wavesurfer.playPause();
                            break;
                        case 'ArrowLeft':
                            e.preventDefault();
                            wavesurfer.skip(-2);
                            break;
                        case 'ArrowRight':
                            e.preventDefault();
                            wavesurfer.skip(2);
                            break;
                    }}
                }});
            }});
        """)
    )

def render_clip(clip):
    """Render a single clip element."""
    return Div(
        Div(
            Div(
                Strong(f"[{clip.start_timestamp:.2f}s - {clip.end_timestamp:.2f}s]"),
                style="margin-bottom: 5px; color: #007bff;"
            ),
            Div(
                clip.text if clip.text else "(no transcription)",
                style="margin-bottom: 10px; " + ("color: #666; font-style: italic;" if not clip.text else "")
            ),
            Div(
                Button(
                    "▶ Play",
                    hx_post=f"/play_clip/{clip.id}",
                    hx_swap="none",
                    cls="clip-btn",
                    style="padding: 4px 8px; margin-right: 5px; font-size: 12px;"
                ),
                Button(
                    "✏️ Edit",
                    hx_get=f"/edit_clip/{clip.id}",
                    hx_target=f"#clip-{clip.id}",
                    hx_swap="outerHTML",
                    data_action="edit",
                    cls="clip-btn",
                    style="padding: 4px 8px; margin-right: 5px; font-size: 12px;"
                ),
                Button(
                    "🗑️ Delete",
                    hx_post=f"/delete_clip/{clip.id}",
                    hx_target="body",
                    hx_swap="outerHTML",
                    hx_confirm="Delete this clip?",
                    cls="clip-btn",
                    style="padding: 4px 8px; font-size: 12px; background: #dc3545; color: white;"
                ),
                (Span("⚑ MARKED", style="margin-left: 10px; color: #dc3545; font-weight: 600;") if clip.marked else None),
                style="display: flex; align-items: center;"
            ),
        ),
        id=f"clip-{clip.id}",
        style="padding: 12px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 4px; background: #f9f9f9;"
    )

@rt("/edit_clip/{clip_id:int}")
def edit_clip_form(clip_id: int):
    """Show edit form for a clip."""
    clip = clips[clip_id]
    if not clip:
        return Div("Clip not found")

    return Div(
        Form(
            Div(
                Strong(f"[{clip.start_timestamp:.2f}s - {clip.end_timestamp:.2f}s]"),
                style="margin-bottom: 10px; color: #007bff;"
            ),
            Textarea(
                clip.text,
                name="text",
                rows="3",
                style="width: 100%; padding: 8px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 4px; font-family: inherit;",
                placeholder="Enter transcription..."
            ),
            Div(
                Label(
                    Input(type="checkbox", name="marked", checked=clip.marked),
                    " Mark as problematic",
                    style="margin-bottom: 10px; display: flex; align-items: center; gap: 5px;"
                )
            ),
            Div(
                Button(
                    "💾 Save",
                    type="submit",
                    cls="clip-btn",
                    style="padding: 4px 12px; margin-right: 5px; font-size: 12px; background: #28a745; color: white;"
                ),
                Button(
                    "❌ Cancel",
                    hx_get=f"/cancel_edit/{clip_id}",
                    hx_target=f"#clip-{clip_id}",
                    hx_swap="outerHTML",
                    cls="clip-btn",
                    style="padding: 4px 12px; font-size: 12px;"
                ),
                style="display: flex; gap: 5px;"
            ),
            hx_post=f"/save_clip/{clip_id}",
            hx_target="body",
            hx_swap="outerHTML",
        ),
        id=f"clip-{clip_id}",
        style="padding: 12px; margin-bottom: 10px; border: 2px solid #007bff; border-radius: 4px; background: #fff;"
    )

@rt("/cancel_edit/{clip_id:int}")
def cancel_edit(clip_id: int):
    """Cancel editing and show clip normally."""
    clip = clips[clip_id]
    return render_clip(clip)

@rt("/save_clip/{clip_id:int}", methods=["POST"])
def save_clip(clip_id: int, text: str = "", marked: str = ""):
    """Save clip changes."""
    clip = clips[clip_id]
    if clip:
        clips.update({
            'text': text,
            'marked': marked == "on",
            'timestamp': datetime.now().isoformat()
        }, clip_id)
    return index()

@rt("/create_clip", methods=["POST"])
def create_clip(start: float, end: float):
    """Create a new clip."""
    current_audio = state.current_audio
    if current_audio:
        clips.insert({
            'audio_path': current_audio,
            'start_timestamp': float(start),
            'end_timestamp': float(end),
            'text': '',
            'username': get_username(),
            'timestamp': datetime.now().isoformat(),
            'marked': False
        })
    return index()

@rt("/update_clip_times", methods=["POST"])
def update_clip_times(clip_id: int, start: float, end: float):
    """Update clip timestamps after drag/resize."""
    clip = clips.get(int(clip_id))
    if clip:
        clips.update({
            'start_timestamp': float(start),
            'end_timestamp': float(end),
            'timestamp': datetime.now().isoformat()
        }, int(clip_id))
    return Response("OK")

@rt("/delete_clip/{clip_id:int}", methods=["POST"])
def delete_clip(clip_id: int):
    """Delete a clip."""
    clips.delete(clip_id)
    return index()

@rt("/play_clip/{clip_id:int}", methods=["POST"])
def play_clip(clip_id: int):
    """Play a specific clip (handled client-side via region)."""
    return Response("OK")

@rt("/prev_audio", methods=["POST"])
def prev_audio():
    """Navigate to previous audio file."""
    if state.current_index > 0:
        state.current_index -= 1
    return index()

@rt("/next_audio", methods=["POST"])
def next_audio():
    """Navigate to next audio file."""
    audio_files = get_audio_files()
    if state.current_index < len(audio_files) - 1:
        state.current_index += 1
    return index()

@rt("/switch_folder", methods=["POST"])
def switch_folder_endpoint(folder_select: str = ''):
    """Switch to a different data folder."""
    if folder_select and folder_select in get_available_folders():
        switch_folder(folder_select)
        print(f"Switched to folder: {folder_select}")
    return index()

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

# Initialize database
if hasattr(config, 'audio_folder') and config.audio_folder:
    db = database(f'{config.audio_folder}/annotations.db')
    clips = db.create(Clip, pk='id')

# Print startup info
if __name__ == "__main__":
    print(f"Starting {config.title}")
    print(f"Configuration:")
    print(f"  - Audio folder: {config.audio_folder}")
    print(f"  - Database: {config.audio_folder}/annotations.db")
    print(f"  - Annotating as: {get_username()}")

    audio_files = get_audio_files()
    print(f"  - Total audio files: {len(audio_files)}")

    stats = get_progress_stats()
    print(f"  - Audio files with clips: {stats['annotated_audio']}")
    print(f"  - Total clips: {stats['total_clips']}")

    try:
        serve(host="localhost", port=5001)
    except KeyboardInterrupt:
        print("\nShutting down...")
        print("Goodbye!")
