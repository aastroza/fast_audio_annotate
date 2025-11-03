# Modal Transcription Guide

This project uses Modal to run Whisper transcriptions in the cloud with automatic segmentation using VAD (Voice Activity Detection).

## Features

- ✅ **Modal Transcription**: Uses cloud L40S GPUs (configurable)
- ✅ **VAD Segmentation**: Automatically splits audio into segments with timestamps
- ✅ **Word Timestamps**: Optionally captures word-level timestamps
- ✅ **Automatic Batching**: Processes multiple segments in parallel
- ✅ **Model Caching**: Models are cached in Modal Volume for fast loading

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Modal:
```bash
modal setup
```

3. (Optional) Configure your Hugging Face token if using private models:
```bash
export HF_TOKEN=your_token_here
```

## Basic Usage

### Quick Test (Single File)

Test Modal transcription with a single audio file:

```bash
# Auto-detect first audio file in ./audio folder
modal run modal_run.py::test_transcription

# Specify a specific file
modal run modal_run.py::test_transcription --audio-file ./audio/example.webm

# With word timestamps
modal run modal_run.py::test_transcription --word-timestamps

# Different model
modal run modal_run.py::test_transcription --model openai/whisper-large-v3-turbo
```

### Transcribe Single File (Save Results)

```bash
# Transcribe and save to JSON
modal run modal_run.py::transcribe_file --audio-file ./audio/example.webm

# Save to specific output path
modal run modal_run.py::transcribe_file \
  --audio-file ./audio/example.webm \
  --output results/transcription.json

# With word timestamps
modal run modal_run.py::transcribe_file \
  --audio-file ./audio/example.webm \
  --word-timestamps
```

### Batch Processing (Multiple Files)

For processing entire directories with database storage:

```bash
python scripts/preprocess_audio.py \
  --audio-folder ./audio \
  --model openai/whisper-large-v3 \
  --language es \
  --batch-size 8
```

This will:
- Transcribe all audio files in `./audio`
- Save JSON results to `./audio/transcriptions/`
- Store segments in SQLite database (`./audio/annotations.db`)

### Additional Options

**With word timestamps:**
```bash
modal run modal_run.py::transcribe_file \
  --audio-file ./audio/example.webm \
  --word-timestamps
```

**Without VAD segmentation:**
```bash
modal run modal_run.py::transcribe_file \
  --audio-file ./audio/example.webm \
  --no-vad
```

**Auto-detect language:**
```bash
modal run modal_run.py::transcribe_file \
  --audio-file ./audio/example.webm \
  --language auto
```

## Configuration Options

### Command-line Arguments

- `--config`: Path to YAML configuration file (default: `./config.yaml`)
- `--audio-folder`: Folder containing audio files
- `--output`: Directory to save transcription JSONs (default: `{audio-folder}/transcriptions`)
- `--model`: Whisper model to use (default: from config.yaml)
- `--language`: Transcription language (e.g., `es`, `en`, `auto`)
- `--batch-size`: Batch size for inference (default: 8)
- `--word-timestamps`: Include word-level timestamps
- `--no-vad`: Disable VAD segmentation
- `--overwrite`: Overwrite existing transcriptions
- `--database-url`: Database URL (to use Postgres instead of SQLite)
- `--modal`: Force Modal usage (default)
- `--no-modal`: Run Whisper locally instead of Modal

## Output Format

### Transcription JSON

Each audio file generates a JSON with this structure:

```json
{
  "audio_path": "path/to/audio.wav",
  "relative_audio_path": "audio.wav",
  "model": "openai/whisper-large-v3",
  "language": "es",
  "duration": 120.5,
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Hello, this is a test.",
      "words": [
        {
          "start": 0.0,
          "end": 0.5,
          "text": "Hello"
        },
        {
          "start": 0.6,
          "end": 1.0,
          "text": "this"
        }
      ]
    }
  ]
}
```

### Database

Segments are also saved to the database with these fields:

- `audio_path`: Relative audio path
- `start_timestamp`: Segment start time (seconds)
- `end_timestamp`: Segment end time (seconds)
- `text`: Segment transcription
- `username`: User who created the transcription
- `timestamp`: Creation timestamp
- `marked`: Boolean flag to mark clips

## Advanced Configuration

### Customize VAD Parameters

You can adjust VAD segmentation by editing the parameters in `src/fast_audio_annotate/transcription.py`:

```python
DEFAULT_SERVER_VAD = {
    "aggressiveness": 2,        # 0-3, higher = more aggressive
    "frame_ms": 30,             # Frame size in ms
    "min_speech_ms": 150,       # Minimum speech to start segment
    "min_silence_ms": 300,      # Minimum silence to end segment
    "max_chunk_ms": 30000,      # Maximum chunk size (30s)
    "padding_ms": 200,          # Padding at start/end of each segment
}
```

### Change GPU in Modal

Edit `src/fast_audio_annotate/modal_transcription.py`:

```python
@app.cls(gpu="L40S", timeout=60*10, scaledown_window=5, max_containers=10)
```

GPU options:
- `"L4"`: Budget GPU, 24GB VRAM
- `"L40S"`: Recommended, 48GB VRAM (default)
- `"A100"`: Powerful GPU, 40GB/80GB VRAM
- `"H100"`: Most powerful GPU, 80GB VRAM

### Use PostgreSQL Database

For production, you can use Neon or any PostgreSQL:

```bash
export NEON_DATABASE_URL="postgresql://user:pass@host/db"

python scripts/preprocess_audio.py \
  --audio-folder ./audio
```

Or specify directly:

```bash
python scripts/preprocess_audio.py \
  --audio-folder ./audio \
  --database-url "postgresql://user:pass@host/db"
```

## Troubleshooting

### Error: "No module named 'modal'"

```bash
pip install modal==1.2.1
```

### Error: Modal authentication

```bash
modal setup
# Follow the instructions to authenticate
```

### Segments are too long

Adjust `max_chunk_ms` in `DEFAULT_SERVER_VAD` to a lower value (e.g., 15000 for 15s)

### Segments are too short

Increase `min_speech_ms` to require more speech time before creating a segment

### Model not found

Make sure the model exists on Hugging Face:
- `openai/whisper-large-v3`
- `openai/whisper-large-v3-turbo`
- `openai/whisper-medium`
- `openai/whisper-small`

## Architecture

```
┌─────────────────────────────────────────────┐
│  Local: preprocess_audio.py                │
│  - Load audio                               │
│  - Apply VAD to create segments            │
│  - Send segments to Modal                  │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│  Modal: WhisperModel                        │
│  - GPU: L40S                                │
│  - Batched inference (up to 64 samples)    │
│  - Returns transcriptions with timestamps  │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│  Local: Save results                        │
│  - JSON with segments                       │
│  - Database (SQLite/PostgreSQL)             │
└─────────────────────────────────────────────┘
```

## Comparison: Local vs Modal

| Aspect              | Local (--no-modal) | Modal (default)     |
|---------------------|--------------------|---------------------|
| GPU required        | ✅ Yes             | ❌ No               |
| Speed               | Depends on HW      | Fast (L40S)         |
| Cost                | $0 (your HW)       | ~$1.20/hour GPU     |
| Parallelization     | Limited            | ✅ Auto-scaling     |
| Setup               | CUDA, drivers, etc | `modal setup`       |

## Usage Examples

### Example 1: Basic Spanish transcription

```bash
python scripts/preprocess_audio.py \
  --audio-folder ./podcasts \
  --model openai/whisper-large-v3 \
  --language es
```

### Example 2: Detailed analysis with word timestamps

```bash
python scripts/preprocess_audio.py \
  --audio-folder ./interviews \
  --model openai/whisper-large-v3-turbo \
  --language es \
  --word-timestamps \
  --batch-size 16
```

### Example 3: Re-process with different model

```bash
python scripts/preprocess_audio.py \
  --audio-folder ./audio \
  --model openai/whisper-medium \
  --overwrite
```

## Contributing

To report issues or contribute, visit the repository on GitHub.

## License

See LICENSE file.
