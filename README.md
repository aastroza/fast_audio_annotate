# Fast Audio Annotate

A FastHTML audio transcription annotation tool - Streamlined, clip-focused audio annotation with waveform display built with FastHTML and WaveSurfer.js.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastHTML](https://img.shields.io/badge/FastHTML-latest-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## Features

- **Clip-focused workflow** - Navigate and edit one clip at a time
- **Visual waveform display** - Interactive audio waveform with zoom and navigation
- **Auto-generation** - Automatically creates 10-second clips as you navigate
- **Precise timestamp editing** - Numeric input fields for exact clip boundaries
- **Dedicated transcription area** - Large text area for comfortable transcription editing
- **Visual clip adjustment** - Drag and resize clip boundaries on the waveform
- **Clip-only playback** - Play button always plays the current clip, not the full audio
- **Variable speed** - Adjust playback speed (0.5x to 2x)
- **Mark problematic clips** - Flag clips that have issues
- **Multi-user support** - Tracks username and timestamp for each annotation
- **SQLite database** - Persistent storage with efficient queries
- **Multiple audio formats** - Supports .webm, .mp3, .wav, .ogg, .m4a, .flac
- **HTMX-powered** - Dynamic updates without full page reloads
- **Audio metadata sync** - Optionally ingest per-audio metadata from `metadata.json`

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/fast_audio_annotate.git
cd fast_audio_annotate

# Install dependencies
pip install -r requirements.txt

# Place audio files in audio/ folder
mkdir -p audio
cp your-audio-files.webm audio/

# Run the web interface
python main.py
```

Open browser to `http://localhost:5001`

**For Modal transcription setup**, see [INSTALL.md](INSTALL.md) for detailed installation instructions.

## How to Annotate

### Basic Workflow

1. **Select Audio**: Choose an audio file from the dropdown
2. **Navigate Clips**: Use Previous/Next buttons to move between clips
   - If no clip exists, a new 10-second clip is automatically created
   - Clips are always sorted by start time
3. **Adjust Timestamps**:
   - Type exact values in Start/End input fields
   - Or drag the clip boundaries on the waveform
   - Click "Update Times" to apply numeric changes
4. **Add Transcription**: Type or paste transcription text in the large text area
5. **Save**: Click "Save Clip" to store your changes
6. **Repeat**: Click "Next Clip" to move forward (auto-generates new clips as needed)

### Key Features

#### Single-Clip Focus
The interface always displays exactly one clip at a time, making it easy to focus on transcribing without distraction.

#### Auto-Generation
When you click "Next Clip" at the end of the audio, the tool automatically creates a new 10-second clip starting from where the last clip ended.

#### Precise Control
- **Numeric inputs**: Enter exact timestamps (e.g., 12.45 seconds)
- **Visual adjustment**: Drag clip boundaries on the waveform
- **Real-time sync**: Changes in inputs update the waveform and vice versa

#### Clip-Only Playback
The "Play Clip" button only plays the current clip region, not the entire audio file. This lets you quickly review your clip boundaries.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play/Pause current clip |
| `←/→` | Skip backward/forward 2 seconds |

## Interface Overview

```
┌─────────────────────────────────────────┐
│ Audio File: [Dropdown Selector]         │  ← Select audio file
├─────────────────────────────────────────┤
│ Audio 1 of 5 | Clip 3 of 12 | Marked: 1│  ← Progress stats
├─────────────────────────────────────────┤
│ [Waveform Display]                      │  ← Visual waveform
│ [Timeline]                              │
├─────────────────────────────────────────┤
│ [▶ Play Clip] [⏸ Pause] [⏹ Stop]       │  ← Playback controls
├─────────────────────────────────────────┤
│ Clip 3                                  │
│ Start: [12.45] End: [22.45] [Update]   │  ← Precise timestamps
│                                         │
│ Transcription:                          │
│ ┌─────────────────────────────────────┐ │
│ │ [Large text area for transcription] │ │  ← Main editing area
│ │                                     │ │
│ └─────────────────────────────────────┘ │
│ ☐ Mark as problematic                  │
│ [💾 Save Clip]                          │
├─────────────────────────────────────────┤
│ [← Previous] [Delete] [Next Clip →]    │  ← Clip navigation
└─────────────────────────────────────────┘
```

## Configuration

Edit `config.yaml`:

```yaml
title: "Audio Transcription Tool"
description: "Annotate audio clips with transcriptions"
audio_folder: "audio"  # Folder containing audio files to annotate
max_history: 10  # Number of undo operations to keep
database_url: null  # Optional Postgres/Neon connection string
metadata_filename: "metadata.json"  # Metadata file to ingest on startup
whisper_model: "openai/whisper-large-v3"  # Default Whisper model for preprocessing
transcription_language: null  # Force a transcription language or leave null for auto-detect
```

If `database_url` is omitted (or set to `null`), the app stores annotations in
`audio/annotations.db` using SQLite. Provide a Postgres connection string to use
an external database instead.

### Preprocessing audios with Whisper (Modal)

The tool includes **cloud-based Whisper transcription** using [Modal](https://modal.com/), which automatically:
- ✅ **Segments audio using VAD** (Voice Activity Detection) with configurable parameters
- ✅ **Runs on cloud GPUs** (L40S by default) - no local GPU needed
- ✅ **Generates word-level timestamps** (optional)
- ✅ **Batch parallel processing** across multiple GPU workers
- ✅ **Saves segments to database** with precise start/end times

**Two architectures available:**

#### Full Modal Architecture (Recommended)

Everything runs in Modal - no local dependencies except Modal CLI:

```bash
# First-time setup
pip install modal
modal setup

cd modal_app

# 1. Upload audio files to Modal Volume
modal run.py::stage_data --audio-folder ./audio

# 2. Batch transcribe all files (parallel processing)
modal run.py::batch_transcription --language es --word-timestamps

# 3. Download results
modal volume get transcription-results ./results/
```

See **[modal_app/README.md](modal_app/README.md)** for complete documentation.

#### Hybrid Architecture (Legacy)

Local VAD processing + Modal transcription:

```bash
# Install local dependencies
pip install -r requirements.txt

# Quick test with single file
modal run modal_run.py::test_transcription --audio-file ./audio/example.webm

# Batch processing with database storage
python scripts/preprocess_audio.py --audio-folder ./audio --language es
```

See **[MODAL_TRANSCRIPTION_GUIDE.md](MODAL_TRANSCRIPTION_GUIDE.md)** for detailed documentation.

## Using Neon (Optional)

You can host the annotation database on [Neon](https://neon.com/) by supplying
the Neon connection string. Either set the value in `config.yaml` or export one
of the following environment variables before running the app:

```bash
export NEON_DATABASE_URL="postgresql://user:password@host/db?sslmode=require"
# or
export DATABASE_URL="postgresql://user:password@host/db?sslmode=require"
```

The application will automatically detect the Postgres URL and manage the
`clips` table on startup. Neon requires SSL connections, which are supported by
the bundled `psycopg` driver. Refer to the [Neon Python guide](https://neon.com/docs/guides/python)
for detailed instructions on creating a project and retrieving your connection
string.

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

### Audio Metadata Table

When a `metadata.json` file is present, the app stores its contents in the
`audio_metadata` table:

| Column | Type | Description |
|--------|------|-------------|
| audio_path | TEXT | Audio filename (relative path, matches `clips.audio_path`) |
| metadata_json | JSON / TEXT | Serialized metadata payload |

This table lets you join metadata with clip annotations using the shared
`audio_path` value.

## Audio Metadata File (Optional)

To associate metadata with audio files, place a `metadata.json` file inside the
audio directory specified in `config.yaml` (default: `audio/`). Each entry should
map to an audio filename (relative to the audio directory). Examples of accepted
structures:

```json
{
  "session_01.webm": {
    "speaker": "Alice",
    "language": "es-MX",
    "notes": "First interview"
  },
  "nested/session_02.wav": {
    "speaker": "Bob",
    "topic": "Customer feedback"
  }
}
```

or

```json
{
  "audios": [
    {
      "audio_path": "session_03.webm",
      "speaker": "Clara",
      "duration": 612.4
    },
    {
      "file": "nested/session_04.wav",
      "language": "es-AR",
      "tags": ["support", "priority"]
    }
  ]
}
```

The application automatically loads this file on startup and upserts the records
into the `audio_metadata` table. Metadata is displayed alongside the waveform and
is available for downstream exports or integrations.

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

## Workflow Tips

### Efficient Transcription
1. Load audio file and start with clip 1
2. Play the clip to hear the audio
3. Adjust start/end times if needed
4. Type the transcription
5. Click "Save Clip"
6. Press "Next Clip" to auto-advance (creates new clip if at the end)

### Handling Long Audio
The tool automatically segments long audio files into manageable 10-second clips. You can:
- Adjust clip lengths as needed (shorter for dense dialogue, longer for sparse speech)
- Delete unnecessary clips (silence, music, etc.)
- Navigate freely between clips

### Quality Control
- Use the "Mark as problematic" checkbox for difficult clips
- Review marked clips later by checking the stats counter
- Delete and regenerate clips if needed

## Exporting Annotations

You can export annotations directly from the SQLite database:

```python
import sqlite3
import json

# Connect to database
conn = sqlite3.connect('audio/annotations.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Get all clips
cursor.execute('SELECT * FROM clip ORDER BY audio_path, start_timestamp')
clips = [dict(row) for row in cursor.fetchall()]

# Export to JSON
with open('annotations.json', 'w') as f:
    json.dump(clips, f, indent=2)

print(f"Exported {len(clips)} clips to annotations.json")
```

### Export Format

The exported JSON will have this structure:

```json
[
  {
    "id": 1,
    "audio_path": "interview.webm",
    "start_timestamp": 0.0,
    "end_timestamp": 10.0,
    "text": "Hello, welcome to the interview.",
    "username": "user",
    "timestamp": "2025-01-15T10:30:00",
    "marked": false
  },
  ...
]
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Built with [FastHTML](https://github.com/AnswerDotAI/fasthtml) - The fast, Pythonic way to create web applications
- Audio visualization powered by [WaveSurfer.js](https://wavesurfer.xyz/)
