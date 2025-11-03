"""Modal entrypoints for Fast Audio Annotate transcription."""
import argparse
from pathlib import Path

from app.common import app
from app.stage_data import list_audio_files, upload_single_file
from app.transcription import batch_transcribe, transcribe_audio_file


@app.local_entrypoint()
def stage_data(*args):
    """
    Upload audio files from local directory to Modal Volume.

    Usage:
        modal run modal_app/run.py::stage_data --audio-folder ./audio
    """
    parser = argparse.ArgumentParser(description="Upload audio files to Modal Volume")
    parser.add_argument(
        "--audio-folder",
        type=str,
        default="./audio",
        help="Path to local audio folder (default: ./audio)",
    )
    parsed_args = parser.parse_args(args)

    audio_folder = Path(parsed_args.audio_folder).resolve()

    if not audio_folder.exists():
        print(f"Error: Audio folder does not exist: {audio_folder}")
        return

    # Find all audio files locally
    audio_extensions = {".webm", ".mp3", ".wav", ".ogg", ".m4a", ".flac"}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_folder.rglob(f"*{ext}"))

    if not audio_files:
        print(f"No audio files found in {audio_folder}")
        return

    print(f"Found {len(audio_files)} audio files to upload from: {audio_folder}")

    # Upload files to Modal Volume
    uploaded_files = []
    for local_path in audio_files:
        rel_path = local_path.relative_to(audio_folder)
        print(f"Uploading: {rel_path}")
        
        # Read file content locally
        with open(local_path, 'rb') as f:
            file_content = f.read()
        
        # Upload to Modal Volume
        upload_single_file.remote(str(rel_path), file_content)
        uploaded_files.append(str(rel_path))

    print(f"\nSuccessfully uploaded {len(uploaded_files)} files:")
    for file_path in uploaded_files:
        print(f"  - {file_path}")


@app.local_entrypoint()
def list_files(*args):
    """
    List all audio files in Modal Volume.

    Usage:
        modal run modal_app/run.py::list_files
    """
    print("Listing audio files in Modal Volume...")
    files = list_audio_files.remote()

    if not files:
        print("No audio files found in Modal Volume")
        print("Upload files with: modal run modal_app/run.py::stage_data --audio-folder ./audio")
    else:
        print(f"\nFound {len(files)} audio files:")
        for file_path in files:
            print(f"  - {file_path}")


@app.local_entrypoint()
def transcribe_single(*args):
    """
    Transcribe a single audio file from Modal Volume.

    Usage:
        modal run modal_app/run.py::transcribe_single --audio-file example.webm
        modal run modal_app/run.py::transcribe_single --audio-file example.webm --language es --word-timestamps
    """
    parser = argparse.ArgumentParser(description="Transcribe single audio file")
    parser.add_argument(
        "--audio-file",
        type=str,
        required=True,
        help="Path to audio file in Modal Volume (relative path)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3",
        help="Whisper model name (default: openai/whisper-large-v3)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="",
        help="Language code (e.g., 'es', 'en'). Leave empty for auto-detect.",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Return word-level timestamps",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD segmentation (transcribe as single segment)",
    )
    parsed_args = parser.parse_args(args)

    print(f"Transcribing: {parsed_args.audio_file}")
    print(f"Model: {parsed_args.model}")
    print(f"Language: {parsed_args.language or 'auto-detect'}")
    print(f"Word timestamps: {parsed_args.word_timestamps}")
    print(f"VAD: {not parsed_args.no_vad}")

    result = transcribe_audio_file.remote(
        audio_path=parsed_args.audio_file,
        model_name=parsed_args.model,
        language=parsed_args.language,
        return_word_timestamps=parsed_args.word_timestamps,
        use_vad=not parsed_args.no_vad,
    )

    print(f"\nTranscription complete!")
    print(f"Language: {result.language}")
    print(f"Segments: {len(result.segments)}")
    print("\nSegments:")
    for i, seg in enumerate(result.segments, 1):
        print(f"  [{i}] {seg.start:.2f}s - {seg.end:.2f}s: {seg.text}")

    return result


@app.local_entrypoint()
def batch_transcription(*args):
    """
    Batch transcribe all audio files in Modal Volume.

    Usage:
        modal run modal_app/run.py::batch_transcription --model openai/whisper-large-v3
        modal run modal_app/run.py::batch_transcription --model openai/whisper-large-v3 --language es --word-timestamps
    """
    parser = argparse.ArgumentParser(description="Batch transcribe audio files")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3",
        help="Whisper model name (default: openai/whisper-large-v3)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="",
        help="Language code (e.g., 'es', 'en'). Leave empty for auto-detect.",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Return word-level timestamps",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD segmentation (transcribe as single segment)",
    )
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="VAD aggressiveness (0-3, default: 2)",
    )
    parser.add_argument(
        "--vad-min-speech-ms",
        type=int,
        default=150,
        help="Minimum speech duration in ms (default: 150)",
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=300,
        help="Minimum silence duration in ms (default: 300)",
    )
    parser.add_argument(
        "--vad-max-chunk-ms",
        type=int,
        default=30000,
        help="Maximum chunk duration in ms (default: 30000)",
    )
    parsed_args = parser.parse_args(args)

    # Build VAD config
    vad_config = None
    if not parsed_args.no_vad:
        vad_config = {
            "aggressiveness": parsed_args.vad_aggressiveness,
            "frame_ms": 30,
            "min_speech_ms": parsed_args.vad_min_speech_ms,
            "min_silence_ms": parsed_args.vad_min_silence_ms,
            "max_chunk_ms": parsed_args.vad_max_chunk_ms,
            "padding_ms": 200,
        }

    print(f"Batch transcription starting...")
    print(f"Model: {parsed_args.model}")
    print(f"Language: {parsed_args.language or 'auto-detect'}")
    print(f"Word timestamps: {parsed_args.word_timestamps}")
    print(f"VAD: {not parsed_args.no_vad}")
    if vad_config:
        print(f"VAD config: aggressiveness={vad_config['aggressiveness']}, "
              f"min_speech={vad_config['min_speech_ms']}ms, "
              f"min_silence={vad_config['min_silence_ms']}ms, "
              f"max_chunk={vad_config['max_chunk_ms']}ms")

    result_paths = batch_transcribe.remote(
        model_name=parsed_args.model,
        language=parsed_args.language,
        return_word_timestamps=parsed_args.word_timestamps,
        use_vad=not parsed_args.no_vad,
        vad_config=vad_config,
    )

    print(f"\nBatch transcription complete!")
    print(f"Processed {len(result_paths)} files")
    print("\nResults saved to Modal Volume at /results/:")
    for path in result_paths:
        print(f"  - {path}")

    print("\nTo download results, use Modal Volume download or export functionality.")

    return result_paths
