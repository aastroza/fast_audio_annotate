# Installation Guide

## Architecture Overview

This project uses a **hybrid architecture** for transcription:

```
┌─────────────────────────────────────────┐
│ YOUR COMPUTER (Local)                   │
├─────────────────────────────────────────┤
│ 1. Load audio files (soundfile)        │
│ 2. Apply VAD segmentation (webrtcvad)  │
│ 3. Send segments to Modal  ─────────┐  │
└─────────────────────────────────────┼───┘
                                      │
                                      ▼
┌─────────────────────────────────────────┐
│ MODAL (Cloud GPU)                       │
├─────────────────────────────────────────┤
│ 4. Transcribe with Whisper             │
│ 5. Return transcriptions  ──────────┐   │
└─────────────────────────────────────┼───┘
                                      │
                                      ▼
┌─────────────────────────────────────────┐
│ YOUR COMPUTER (Local)                   │
├─────────────────────────────────────────┤
│ 6. Combine results                      │
│ 7. Save to database/JSON                │
└─────────────────────────────────────────┘
```

**This means you need audio processing libraries installed locally**, even though transcription happens on Modal's GPUs.

## Quick Install

### Windows

#### Option 1: Using pip (Recommended)

```powershell
# Install dependencies
pip install -r requirements.txt
```

**If webrtcvad fails to install**, you need Visual Studio Build Tools:

1. Download and install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
2. During installation, select "Desktop development with C++"
3. Restart your terminal
4. Try again: `pip install webrtcvad`

#### Option 2: Using pre-built wheels (Faster)

If you have issues compiling `webrtcvad`, try installing a pre-built wheel:

```powershell
# For Python 3.12 on Windows x64
pip install https://github.com/wiseman/py-webrtcvad/releases/download/2.0.10/webrtcvad-2.0.10-cp312-cp312-win_amd64.whl
```

Then install the rest:
```powershell
pip install -r requirements.txt
```

### macOS

```bash
# Install system dependencies (if using Homebrew)
brew install portaudio

# Install Python dependencies
pip install -r requirements.txt
```

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get install portaudio19-dev python3-dev

# Install Python dependencies
pip install -r requirements.txt
```

## Setup Modal

After installing dependencies:

```bash
# Install Modal (if not already installed)
pip install modal

# Authenticate with Modal
modal setup
```

Follow the prompts to create/login to your Modal account.

## Verify Installation

Test that everything works:

```bash
# Test Modal connection
modal run modal_run.py::test_transcription --audio-file ./audio/example.webm
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'soundfile'`

**Cause**: Local audio processing dependencies not installed.

**Fix**:
```powershell
pip install soundfile librosa pyloudnorm resampy
```

### `webrtcvad` won't install on Windows

**Cause**: Missing C++ compiler.

**Fix Options**:

1. **Install Visual Studio Build Tools** (Recommended):
   - Download: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
   - Select "Desktop development with C++"
   - Restart terminal after installation

2. **Use pre-built wheel** (Easier):
   ```powershell
   pip install https://github.com/wiseman/py-webrtcvad/releases/download/2.0.10/webrtcvad-2.0.10-cp312-cp312-win_amd64.whl
   ```

3. **Skip VAD** (if you don't need segmentation):
   ```bash
   modal run modal_run.py::transcribe_file --audio-file ./audio/example.webm --no-vad
   ```

### Modal authentication issues

```bash
# Re-authenticate
modal token new

# Verify connection
modal app list
```

### Audio file format not supported

Supported formats: `.wav`, `.mp3`, `.webm`, `.ogg`, `.m4a`, `.flac`

If you have other formats, convert first:
```bash
ffmpeg -i input.mp4 output.wav
```

## What Gets Installed Where

### Local (Your Computer)
- `soundfile` - Load audio files
- `librosa` - Audio processing
- `webrtcvad` - Voice Activity Detection
- `pyloudnorm` - Audio normalization
- `resampy` - Resampling
- `modal` - Modal client library

### Modal (Cloud)
- `torch` - PyTorch for GPU
- `transformers` - Whisper models
- `accelerate` - GPU acceleration
- All audio processing libraries (for consistency)

The Modal image is built automatically on first run and cached for subsequent runs.

## Next Steps

After installation:

1. **Test with a single file**:
   ```bash
   modal run modal_run.py::test_transcription --audio-file ./audio/example.webm
   ```

2. **Transcribe and save**:
   ```bash
   modal run modal_run.py::transcribe_file --audio-file ./audio/example.webm
   ```

3. **Batch process directory**:
   ```bash
   python scripts/preprocess_audio.py --audio-folder ./audio
   ```

See [MODAL_TRANSCRIPTION_GUIDE.md](MODAL_TRANSCRIPTION_GUIDE.md) for complete usage documentation.
