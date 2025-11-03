# Modal Transcription Architecture

Complete cloud-based audio transcription using Modal's serverless infrastructure.

## 🆕 Two ASR Options Available

This project now supports **two transcription approaches**:

### Option 1: NeMo ASR + Integrated VAD (⭐ **Recommended for Spanish**)

- ✅ **Native VAD**: Uses NVIDIA NeMo's integrated Voice Activity Detection
- ✅ **Faster**: Parakeet models are lighter and faster than Whisper Large
- ✅ **Better Spanish**: Optimized multilingual models (Canary, Parakeet)
- ✅ **Precise Segmentation**: VAD shares feature extraction with ASR
- 📖 **Guide**: See [`NEMO_GUIDE.md`](NEMO_GUIDE.md) for complete documentation

```bash
# Quick start with NeMo
modal run modal_app/run_nemo.py::batch_transcription
```

### Option 2: Whisper + WebRTC VAD (Original)

- ✅ **Word Timestamps**: Provides word-level timestamps
- ✅ **Many Models**: Access to full Whisper model family
- ✅ **Proven**: Stable, well-tested implementation
- 📖 **Guide**: See below for Whisper documentation

```bash
# Quick start with Whisper
modal run modal_app/run.py::batch_transcription --language es
```

## Architecture Overview

This implementation runs **ALL processing in Modal** (not hybrid):

1. **Stage Data**: Upload audio files to Modal Volume
2. **Process in Modal**: Audio loading, VAD segmentation, and transcription all happen in Modal containers
3. **Batch Parallel**: Multiple GPU workers process files in parallel using `.map()`
4. **Results Storage**: Transcriptions saved to Modal Volume

## Directory Structure

```
modal_app/
├── app/
│   ├── __init__.py
│   ├── common.py              # Shared Modal config (images, volumes)
│   ├── stage_data.py          # Upload audio files to Modal Volume
│   ├── transcription.py       # Whisper + WebRTC VAD (original)
│   └── transcription_nemo.py  # 🆕 NeMo ASR + integrated VAD
├── run.py                     # Whisper entrypoints
├── run_nemo.py                # 🆕 NeMo entrypoints
├── vad_config.yaml            # 🆕 NeMo VAD configuration template
├── NEMO_GUIDE.md              # 🆕 Complete NeMo documentation
└── README.md                  # This file
```

## Quick Start

### 1. Setup Modal

```bash
# Install Modal
pip install modal

# Authenticate
modal setup
```

### 2. Stage Your Audio Files

Upload local audio files to Modal Volume:

```bash
modal run modal_app/run.py::stage_data --audio-folder ./audio
```

This uploads all audio files (`.webm`, `.mp3`, `.wav`, `.ogg`, `.m4a`, `.flac`) to Modal's persistent storage.

### 3. List Files in Modal Volume

Verify uploaded files:

```bash
modal run modal_app/run.py::list_files
```

### 4. Transcribe Single File

Test with a single file:

```bash
# Basic transcription (auto-detect language)
modal run modal_app/run.py::transcribe_single --audio-file example.webm

# With language and word timestamps
modal run modal_app/run.py::transcribe_single \
  --audio-file example.webm \
  --language es \
  --word-timestamps
```

### 5. Batch Transcription

Process all files in parallel:

```bash
# Basic batch transcription
modal run modal_app/run.py::batch_transcription

# With custom settings
modal run modal_app/run.py::batch_transcription \
  --model openai/whisper-large-v3 \
  --language es \
  --word-timestamps \
  --vad-aggressiveness 2
```

## Entrypoints

### `stage_data`

Upload audio files to Modal Volume.

```bash
modal run modal_app/run.py::stage_data --audio-folder ./audio
```

**Arguments:**
- `--audio-folder`: Path to local audio folder (default: `./audio`)

### `list_files`

List all audio files in Modal Volume.

```bash
modal run modal_app/run.py::list_files
```

### `transcribe_single`

Transcribe a single audio file.

```bash
modal run modal_app/run.py::transcribe_single --audio-file example.webm
```

**Arguments:**
- `--audio-file`: Path to audio file in Modal Volume (required)
- `--model`: Whisper model name (default: `openai/whisper-large-v3`)
- `--language`: Language code (e.g., `es`, `en`). Empty for auto-detect
- `--word-timestamps`: Return word-level timestamps (flag)
- `--no-vad`: Disable VAD segmentation (flag)

### `batch_transcription`

Batch transcribe all audio files.

```bash
modal run modal_app/run.py::batch_transcription --language es
```

**Arguments:**
- `--model`: Whisper model name (default: `openai/whisper-large-v3`)
- `--language`: Language code. Empty for auto-detect
- `--word-timestamps`: Return word-level timestamps (flag)
- `--no-vad`: Disable VAD segmentation (flag)
- `--vad-aggressiveness`: VAD aggressiveness 0-3 (default: 2)
- `--vad-min-speech-ms`: Minimum speech duration in ms (default: 150)
- `--vad-min-silence-ms`: Minimum silence duration in ms (default: 300)
- `--vad-max-chunk-ms`: Maximum chunk duration in ms (default: 30000)

## VAD Configuration

Voice Activity Detection (VAD) automatically segments audio:

- **Aggressiveness**: 0 (least) to 3 (most aggressive)
  - 0: Detects more speech (may include noise)
  - 3: Only clear speech (may miss quiet parts)
- **Min Speech**: Minimum speech duration to start a segment (default: 150ms)
- **Min Silence**: Minimum silence to end a segment (default: 300ms)
- **Max Chunk**: Maximum segment length (default: 30000ms = 30s)
- **Padding**: Extra audio around segments (default: 200ms)

## GPU Options

Edit `modal_app/app/transcription.py` to change GPU type:

```python
@app.cls(
    gpu="L40S",  # Options: L4, L40S, A100, H100
    timeout=60 * 10,
    scaledown_window=5,
    max_containers=10,  # Parallel workers
    volumes={MODEL_DIR: model_cache},
)
class WhisperModel:
    ...
```

## Modal Volumes

Three persistent volumes are used:

1. **hf-hub-cache**: Hugging Face model cache
2. **audio-files**: Uploaded audio files
3. **transcription-results**: Transcription JSON files

## Output Format

Results are saved as JSON files in Modal Volume at `/results/`:

```json
{
  "audio_path": "example.webm",
  "language": "es",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hola, bienvenidos.",
      "words": [
        {"start": 0.0, "end": 0.5, "text": "Hola"},
        {"start": 0.6, "end": 1.8, "text": "bienvenidos"}
      ]
    }
  ]
}
```

## Downloading Results

To download results from Modal Volume:

```bash
# List volumes
modal volume ls

# Download entire results volume
modal volume get transcription-results results/
```

## Advantages Over Hybrid Architecture

**Old Hybrid Architecture:**
- ❌ Local audio loading (requires local dependencies)
- ❌ Local VAD processing (CPU-bound, slow)
- ✅ Modal transcription (GPU-accelerated)

**New Full Modal Architecture:**
- ✅ Modal audio loading (no local dependencies needed)
- ✅ Modal VAD processing (faster, scalable)
- ✅ Modal transcription (GPU-accelerated)
- ✅ Batch parallel processing (multiple GPUs)
- ✅ Persistent storage (Modal Volumes)
- ✅ No local GPU/dependencies required

## Example Workflow

```bash
# 1. Upload audio files
modal run modal_app/run.py::stage_data --audio-folder ./audio

# 2. Verify uploaded files
modal run modal_app/run.py::list_files

# 3. Test single file
modal run modal_app/run.py::transcribe_single \
  --audio-file example.webm \
  --language es

# 4. Batch process all files
modal run modal_app/run.py::batch_transcription \
  --model openai/whisper-large-v3 \
  --language es \
  --word-timestamps

# 5. Download results
modal volume get transcription-results ./results/
```

## Troubleshooting

### No audio files found

Make sure you've uploaded files first:
```bash
modal run modal_app/run.py::stage_data --audio-folder ./audio
```

### GPU out of memory

Reduce batch size in `modal_app/app/transcription.py`:
```python
batch_size: int = modal.parameter(default=4)  # Reduce from 8 to 4
```

### VAD creating too many/few segments

Adjust VAD parameters:
```bash
# More aggressive (fewer segments)
modal run modal_app/run.py::batch_transcription --vad-aggressiveness 3

# Less aggressive (more segments)
modal run modal_app/run.py::batch_transcription --vad-aggressiveness 1
```

## Cost Optimization

- **GPU Type**: L4 is cheapest, L40S/A100/H100 are faster
- **Scaledown Window**: Containers stay warm for 5 minutes (set in `@app.cls`)
- **Max Containers**: Limits parallel workers (default: 10)
- **Batch Size**: Higher = faster but more memory (default: 8)

## Comparison with Reference Code

This implementation follows the architecture pattern from `aastroza-open_asr_leaderboard_cl`:

| Feature | Reference Code | This Implementation |
|---------|---------------|---------------------|
| Data staging | ✅ `stage_data.py` | ✅ `stage_data.py` |
| Modal Volumes | ✅ 3 volumes | ✅ 3 volumes |
| Batch processing | ✅ `.map()` | ✅ `.map()` |
| GPU inference | ✅ Modal class | ✅ `WhisperModel` |
| VAD segmentation | ❌ Not included | ✅ WebRTC VAD |
| Word timestamps | ✅ Optional | ✅ Optional |
| Entrypoints | ✅ `@app.local_entrypoint()` | ✅ `@app.local_entrypoint()` |

**Key Addition:** This implementation adds VAD-based audio segmentation, which was requested by the user.
