# Fast Audio Annotate

A FastHTML audio transcription annotation tool - Simple, visual audio annotation with waveform display built with FastHTML and WaveSurfer.js.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastHTML](https://img.shields.io/badge/FastHTML-latest-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## Features

- **Visual waveform display** - Interactive audio waveform with zoom and navigation
- **Clip-based annotation** - Create clips by selecting regions on the waveform
- **Transcription editing** - Add text transcriptions for each audio clip
- **Flexible playback** - Play entire audio, individual clips, or arbitrary segments
- **Variable speed** - Adjust playback speed (0.5x to 2x)
- **Drag & resize clips** - Visually adjust clip boundaries on the waveform
- **Mark problematic clips** - Flag clips that have issues
- **Multi-user support** - Tracks username and timestamp for each annotation
- **SQLite database** - Persistent storage with efficient queries
- **Multiple audio formats** - Supports .webm, .mp3, .wav, .ogg, .m4a, .flac
- **HTMX-powered** - Dynamic updates without full page reloads

## Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/fast_audio_annotate.git
cd fast_audio_annotate
pip install .

# Configure (edit config.yaml)
# Place audio files in audio/ folder

# Run with uv (recommended)
uv run python main.py

# Or with regular Python
python main.py
```

Open browser to `http://localhost:5001`

## How to Annotate

1. **Load Audio**: Navigate between audio files using Previous/Next buttons
2. **Create Clips**: Click and drag on the waveform to select a region, then double-click to create a clip
3. **Add Transcription**: Click "Edit" on any clip to add transcription text
4. **Adjust Boundaries**: Drag the edges of existing clips to adjust start/end times
5. **Play Clips**: Click on a clip region to play just that segment
6. **Mark Issues**: Use the checkbox when editing to flag problematic clips

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play/Pause audio |
| `←/→` | Skip backward/forward 2 seconds |
| Click & Drag | Create new clip region |
| Double-click region | Open edit form for clip |
| Click region | Play that clip |

## Configuration

Edit `config.yaml`:

```yaml
title: "Audio Transcription Tool"
description: "Annotate audio clips with transcriptions"
audio_folder: "audio"  # Folder containing audio files to annotate
max_history: 10  # Number of undo operations to keep
```

## Database Schema

Annotations are stored in SQLite database (`annotations.db`):

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| audio_path | TEXT | Audio filename (relative path) |
| start_timestamp | FLOAT | Clip start time in seconds |
| end_timestamp | FLOAT | Clip end time in seconds |
| text | TEXT | Transcription text for the clip |
| username | TEXT | System username |
| timestamp | TEXT | ISO format timestamp |
| marked | BOOLEAN | Flag for problematic clips |

## Project Structure

```
main.py              # FastHTML application
config.yaml          # User configuration
styles.css           # Custom CSS styles
audio/               # Audio files folder
  annotations.db     # SQLite database (created automatically)
pyproject.toml       # Project metadata and dependencies
```

## Supported Audio Formats

- WebM (.webm)
- MP3 (.mp3)
- WAV (.wav)
- OGG (.ogg)
- M4A (.m4a)
- FLAC (.flac)

## Technology Stack

- **FastHTML** - Python web framework with HTMX integration
- **WaveSurfer.js v7** - Audio waveform visualization
- **Regions Plugin** - Interactive region selection and editing
- **Timeline Plugin** - Time ruler for audio navigation
- **SQLite** - Lightweight database for annotations
- **HTMX** - Dynamic HTML updates without JavaScript

## Development

The app uses FastHTML's single-file pattern for simplicity:
- Database models and routes in `main.py`
- SQLite for persistence
- HTMX for dynamic updates
- WaveSurfer.js for client-side audio handling
- Clean state management with server-side session

## Exporting Annotations

You can export annotations directly from the SQLite database:

```python
import sqlite3
import json

# Connect to database
conn = sqlite3.connect('audio/annotations.db')
cursor = conn.cursor()

# Get all clips
cursor.execute('SELECT * FROM clip')
clips = cursor.fetchall()

# Export to JSON
with open('annotations.json', 'w') as f:
    json.dump(clips, f, indent=2)
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Built with [FastHTML](https://github.com/AnswerDotAI/fasthtml) - The fast, Pythonic way to create web applications
- Audio visualization powered by [WaveSurfer.js](https://wavesurfer.xyz/)
