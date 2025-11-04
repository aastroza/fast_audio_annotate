"""
FastHTML Audio Annotation Tool - Simple Modal Deployment
"""
from pathlib import Path
import fasthtml.common as fh
import modal
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# Create the FastHTML app object and its rt decorator
fasthtml_app, rt = fh.fast_app(
    hdrs=(
        fh.Style("""
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 10px; background: #f5f5f5; }
            .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .htmx-indicator { display: none; animation: fadeIn 0.3s ease-in; }
            .htmx-request .htmx-indicator { display: inline-block; animation: pulse 1.5s infinite; }
            @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
            .next-btn:hover { background: #e0a800 !important; transform: translateY(-1px); box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
            .complete-btn:hover { background: #0b5ed7 !important; transform: translateY(-1px); box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
            .flag-btn:hover { background: #bb2d3b !important; transform: translateY(-1px); box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
            .htmx-request button { opacity: 0.7; pointer-events: none; }
        """),
    ),
    pico=False
)

@dataclass
class ClipRecord:
    id: int
    audio_path: str
    start_timestamp: float
    end_timestamp: float
    text: str
    username: str
    timestamp: str
    marked: bool
    human_reviewed: bool

# Database functions
def init_db():
    """Initialize the database."""
    db_path = "/data/annotations.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audio_path TEXT NOT NULL,
                start_timestamp REAL NOT NULL,
                end_timestamp REAL NOT NULL,
                text TEXT,
                username TEXT DEFAULT 'unknown',
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                marked BOOLEAN DEFAULT FALSE,
                human_reviewed BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Create sample clips if none exist
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM clips")
        if cursor.fetchone()[0] == 0:
            sample_clips = [
                ("sample1.mp3", 0.0, 5.0, "This is a sample transcription that needs review", "system"),
                ("sample2.mp3", 10.0, 15.0, "Another sample clip for testing purposes", "system"),
                ("sample3.mp3", 20.0, 25.0, "Test audio clip number three", "system"),
            ]
            
            for audio_path, start, end, text, username in sample_clips:
                conn.execute("""
                    INSERT INTO clips (audio_path, start_timestamp, end_timestamp, text, username)
                    VALUES (?, ?, ?, ?, ?)
                """, (audio_path, start, end, text, username))
        
        conn.commit()

def get_random_clip() -> Optional[ClipRecord]:
    """Get a random unreviewed clip."""
    with sqlite3.connect("/data/annotations.db") as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM clips 
            WHERE marked = FALSE AND human_reviewed = FALSE 
            ORDER BY RANDOM() LIMIT 1
        """)
        row = cursor.fetchone()
        if row:
            return ClipRecord(**dict(row))
    return None

def get_clip(clip_id: int) -> Optional[ClipRecord]:
    """Get a specific clip by ID."""
    with sqlite3.connect("/data/annotations.db") as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM clips WHERE id = ?", (clip_id,))
        row = cursor.fetchone()
        if row:
            return ClipRecord(**dict(row))
    return None

def update_clip(clip_id: int, updates: dict):
    """Update a clip."""
    set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
    values = list(updates.values()) + [clip_id]
    
    with sqlite3.connect("/data/annotations.db") as conn:
        cursor = conn.cursor()
        cursor.execute(f"UPDATE clips SET {set_clause} WHERE id = ?", values)
        conn.commit()

def get_username(contributor_name: str = "") -> str:
    """Get username for attribution."""
    if contributor_name and contributor_name.strip():
        return contributor_name.strip()
    return "anonymous"

# Initialize database
init_db()

# Routes
@rt("/")
def home():
    """Main page."""
    clip = get_random_clip()
    if not clip:
        return fh.Titled(
            "Audio Annotation Tool",
            fh.Div(
                fh.H1("Audio Annotation Tool"),
                fh.Div(
                    fh.H2("All caught up!", style="text-align: center; color: #198754;"),
                    fh.P("No clips waiting for review.", style="text-align: center; color: #6c757d;"),
                    style="text-align: center; padding: 40px;"
                ),
                cls="container"
            )
        )
    
    return fh.Titled(
        "Audio Annotation Tool",
        fh.Div(
            fh.H1("Audio Annotation Tool"),
            
            # Instructions
            fh.Div(
                fh.H3("How to review this clip", style="margin-bottom: 8px; color: #0d6efd;"),
                fh.P("Listen to the audio, adjust times if needed, and correct the transcription."),
                style="margin-bottom: 16px; padding: 12px; background: #e3f2fd; border-radius: 6px;"
            ),
            
            # Clip info
            fh.Div(
                fh.Div(fh.Strong("Audio file: "), fh.Span(clip.audio_path), style="margin-bottom: 4px;"),
                fh.Div(fh.Strong("Segment: "), fh.Span(f"{clip.start_timestamp:.2f}s - {clip.end_timestamp:.2f}s"), style="margin-bottom: 8px;"),
                style="margin-bottom: 16px; font-size: 14px;"
            ),
            
            # Audio player (placeholder - you'll need to add actual audio files)
            fh.Div(
                fh.P(f"🎵 Audio: {clip.audio_path} ({clip.start_timestamp:.2f}s - {clip.end_timestamp:.2f}s)", 
                     style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 6px; color: #6c757d;"),
                style="margin-bottom: 16px;"
            ),
            
            # Form
            fh.Form(
                fh.Input(value=str(clip.id), name="clip_id", type="hidden"),
                
                fh.Div(
                    fh.Div(
                        fh.Label("Start time (seconds)", style="display: block; margin-bottom: 6px; font-weight: 600;"),
                        fh.Input(
                            value=f"{clip.start_timestamp:.2f}",
                            name="start_time",
                            type="number",
                            step="0.01",
                            style="width: 100%; padding: 8px; border: 1px solid #ced4da; border-radius: 4px;",
                        ),
                        style="flex: 1; margin-right: 8px;"
                    ),
                    fh.Div(
                        fh.Label("End time (seconds)", style="display: block; margin-bottom: 6px; font-weight: 600;"),
                        fh.Input(
                            value=f"{clip.end_timestamp:.2f}",
                            name="end_time",
                            type="number",
                            step="0.01",
                            style="width: 100%; padding: 8px; border: 1px solid #ced4da; border-radius: 4px;",
                        ),
                        style="flex: 1;"
                    ),
                    style="display: flex; margin-bottom: 16px;"
                ),
                
                fh.Div(
                    fh.Label("Transcription", style="display: block; margin-bottom: 6px; font-weight: 600;"),
                    fh.Textarea(
                        clip.text or "",
                        name="transcription",
                        rows="4",
                        placeholder="Type the corrected transcription here...",
                        style="width: 100%; padding: 8px; border: 1px solid #ced4da; border-radius: 4px;",
                    ),
                    style="margin-bottom: 16px;"
                ),
                
                fh.Div(
                    fh.Label("Your name (optional)", style="display: block; margin-bottom: 6px; font-weight: 600;"),
                    fh.Input(
                        name="contributor_name",
                        placeholder="Enter your name to be credited...",
                        style="width: 100%; padding: 8px; border: 1px solid #ced4da; border-radius: 4px;",
                    ),
                    style="margin-bottom: 20px;"
                ),
                
                fh.Div(
                    fh.Button(
                        "➡️ Next clip",
                        formaction="/next_clip",
                        style="padding: 10px 16px; background: #ffc107; color: #000; border: none; border-radius: 4px; margin-right: 8px; cursor: pointer;"
                    ),
                    fh.Button(
                        "✅ Finish review",
                        formaction="/complete_clip", 
                        style="padding: 10px 16px; background: #0d6efd; color: white; border: none; border-radius: 4px; margin-right: 8px; cursor: pointer;"
                    ),
                    fh.Button(
                        "🚩 Report issue",
                        formaction="/flag_clip",
                        onclick="return confirm('Report this clip as problematic?')",
                        style="padding: 10px 16px; background: #dc3545; color: white; border: none; border-radius: 4px; cursor: pointer;"
                    ),
                    style="display: flex; gap: 8px;"
                ),
                
                method="post"
            ),
            
            cls="container"
        )
    )

@rt("/next_clip", methods=["POST"])
def next_clip(clip_id: str = "", start_time: str = "0", end_time: str = "0", transcription: str = "", contributor_name: str = ""):
    """Save progress and move to next clip."""
    if clip_id:
        try:
            updates = {
                'start_timestamp': float(start_time),
                'end_timestamp': float(end_time),
                'text': transcription,
                'timestamp': datetime.now().isoformat(),
                'username': get_username(contributor_name),
            }
            update_clip(int(clip_id), updates)
        except ValueError:
            pass
    
    return fh.RedirectResponse("/", status_code=303)

@rt("/complete_clip", methods=["POST"])
def complete_clip(clip_id: str = "", start_time: str = "0", end_time: str = "0", transcription: str = "", contributor_name: str = ""):
    """Complete review and move to next clip."""
    if clip_id:
        try:
            updates = {
                'start_timestamp': float(start_time),
                'end_timestamp': float(end_time),
                'text': transcription,
                'timestamp': datetime.now().isoformat(),
                'username': get_username(contributor_name),
                'human_reviewed': True,
                'marked': False,
            }
            update_clip(int(clip_id), updates)
        except ValueError:
            pass
    
    return fh.RedirectResponse("/", status_code=303)

@rt("/flag_clip", methods=["POST"])
def flag_clip(clip_id: str = "", transcription: str = "", contributor_name: str = ""):
    """Flag clip as problematic."""
    if clip_id:
        updates = {
            'text': transcription,
            'timestamp': datetime.now().isoformat(),
            'username': get_username(contributor_name),
            'marked': True,
        }
        update_clip(int(clip_id), updates)
    
    return fh.RedirectResponse("/", status_code=303)

if __name__ == "__main__":  # if invoked with `python`, run locally
    fh.serve(app=fasthtml_app, host="localhost", port=5001)
else:  # create a modal app for deployment
    app = modal.App(name="audio-annotation-tool")
    
    # Create volume for persistent data
    data_volume = modal.Volume.from_name("audio-annotation-data", create_if_missing=True)

    @app.function(
        image=modal.Image.debian_slim().pip_install_from_requirements(
            Path(__file__).parent / "requirements.txt"
        ),
        volumes={"/data": data_volume},
        allow_concurrent_inputs=100,
    )
    @modal.asgi_app()
    def serve():
        return fasthtml_app