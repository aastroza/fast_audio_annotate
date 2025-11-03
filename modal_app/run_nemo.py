"""Modal entrypoints for NeMo ASR transcription with integrated VAD."""
import argparse
from pathlib import Path

from app.stage_data import list_audio_files, upload_single_file
from app.transcription_nemo import batch_transcribe_nemo, nemo_app, transcribe_audio_file_nemo


@nemo_app.local_entrypoint()
def stage_data(*args):
    """
    Upload audio files from local directory to Modal Volume.

    Usage:
        modal run modal_app/run_nemo.py::stage_data --audio-folder ./audio
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
        with open(local_path, "rb") as f:
            file_content = f.read()

        # Upload to Modal Volume
        upload_single_file.remote(str(rel_path), file_content)
        uploaded_files.append(str(rel_path))

    print(f"\nSuccessfully uploaded {len(uploaded_files)} files:")
    for file_path in uploaded_files:
        print(f"  - {file_path}")


@nemo_app.local_entrypoint()
def list_files(*args):
    """
    List all audio files in Modal Volume.

    Usage:
        modal run modal_app/run_nemo.py::list_files
    """
    print("Listing audio files in Modal Volume...")
    files = list_audio_files.remote()

    if not files:
        print("No audio files found in Modal Volume")
        print("Upload files with: modal run modal_app/run_nemo.py::stage_data --audio-folder ./audio")
    else:
        print(f"\nFound {len(files)} audio files:")
        for file_path in files:
            print(f"  - {file_path}")


@nemo_app.local_entrypoint()
def transcribe_single(*args):
    """
    Transcribe a single audio file using NeMo ASR + VAD.

    Usage:
        modal run modal_app/run_nemo.py::transcribe_single --audio-file example.webm
        modal run modal_app/run_nemo.py::transcribe_single --audio-file example.webm --model nvidia/canary-1b --language es
    """
    parser = argparse.ArgumentParser(description="Transcribe single audio file with NeMo ASR")
    parser.add_argument(
        "--audio-file",
        type=str,
        required=True,
        help="Path to audio file in Modal Volume (relative path)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/parakeet-tdt-0.6b",
        help="NeMo ASR model name (default: nvidia/parakeet-tdt-0.6b)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="es",
        help="Language code (e.g., 'es', 'en'). Default: 'es'",
    )
    parser.add_argument(
        "--vad-model",
        type=str,
        default="vad_multilingual_marblenet",
        help="NeMo VAD model name (default: vad_multilingual_marblenet)",
    )
    parser.add_argument(
        "--vad-onset",
        type=float,
        default=0.5,
        help="VAD onset threshold (default: 0.5)",
    )
    parser.add_argument(
        "--vad-offset",
        type=float,
        default=0.5,
        help="VAD offset threshold (default: 0.5)",
    )
    parser.add_argument(
        "--vad-min-duration-on",
        type=float,
        default=0.1,
        help="Minimum speech duration in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--vad-min-duration-off",
        type=float,
        default=0.3,
        help="Minimum silence duration in seconds (default: 0.3)",
    )
    parsed_args = parser.parse_args(args)

    # Build VAD config
    vad_config = {
        "window_length_in_sec": 0.15,
        "shift_length_in_sec": 0.01,
        "smoothing": "median",
        "overlap": 0.5,
        "postprocessing": {
            "onset": parsed_args.vad_onset,
            "offset": parsed_args.vad_offset,
            "min_duration_on": parsed_args.vad_min_duration_on,
            "min_duration_off": parsed_args.vad_min_duration_off,
            "filter_speech_first": True,
        },
    }

    print(f"Transcribing with NeMo ASR + VAD: {parsed_args.audio_file}")
    print(f"ASR Model: {parsed_args.model}")
    print(f"VAD Model: {parsed_args.vad_model}")
    print(f"Language: {parsed_args.language}")
    print(f"VAD config: onset={vad_config['postprocessing']['onset']}, "
          f"offset={vad_config['postprocessing']['offset']}, "
          f"min_duration_on={vad_config['postprocessing']['min_duration_on']}s, "
          f"min_duration_off={vad_config['postprocessing']['min_duration_off']}s")

    result = transcribe_audio_file_nemo.remote(
        audio_path=parsed_args.audio_file,
        model_name=parsed_args.model,
        language=parsed_args.language,
        vad_model_name=parsed_args.vad_model,
        vad_config=vad_config,
    )

    print(f"\nTranscription complete!")
    print(f"Language: {result.language}")
    print(f"Segments: {len(result.segments)}")
    print("\nSegments:")
    for i, seg in enumerate(result.segments, 1):
        print(f"  [{i}] {seg.start:.2f}s - {seg.end:.2f}s: {seg.text}")

    return result


@nemo_app.local_entrypoint()
def batch_transcription(*args):
    """
    Batch transcribe all audio files using NeMo ASR + integrated VAD.

    This uses NVIDIA NeMo's native VAD capabilities which are more accurate
    than WebRTC VAD and better integrated with the ASR models.

    Usage:
        # Using default Parakeet model
        modal run modal_app/run_nemo.py::batch_transcription

        # Using Canary model for multilingual support
        modal run modal_app/run_nemo.py::batch_transcription --model nvidia/canary-1b

        # Custom VAD thresholds
        modal run modal_app/run_nemo.py::batch_transcription --vad-onset 0.7 --vad-offset 0.3
    """
    parser = argparse.ArgumentParser(description="Batch transcribe audio files with NeMo ASR + VAD")
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/parakeet-tdt-0.6b",
        help="NeMo ASR model name (default: nvidia/parakeet-tdt-0.6b)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="es",
        help="Language code (e.g., 'es', 'en'). Default: 'es'",
    )
    parser.add_argument(
        "--vad-model",
        type=str,
        default="vad_multilingual_marblenet",
        help="NeMo VAD model name (default: vad_multilingual_marblenet)",
    )
    parser.add_argument(
        "--vad-onset",
        type=float,
        default=0.5,
        help="VAD onset threshold (0-1, default: 0.5). Higher = more conservative speech detection",
    )
    parser.add_argument(
        "--vad-offset",
        type=float,
        default=0.5,
        help="VAD offset threshold (0-1, default: 0.5). Higher = segments end sooner",
    )
    parser.add_argument(
        "--vad-min-duration-on",
        type=float,
        default=0.1,
        help="Minimum speech duration in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--vad-min-duration-off",
        type=float,
        default=0.3,
        help="Minimum silence duration in seconds (default: 0.3)",
    )
    parser.add_argument(
        "--vad-window-length",
        type=float,
        default=0.15,
        help="VAD window length in seconds (default: 0.15)",
    )
    parser.add_argument(
        "--vad-shift-length",
        type=float,
        default=0.01,
        help="VAD shift length in seconds (default: 0.01)",
    )
    parsed_args = parser.parse_args(args)

    # Build VAD config
    vad_config = {
        "window_length_in_sec": parsed_args.vad_window_length,
        "shift_length_in_sec": parsed_args.vad_shift_length,
        "smoothing": "median",
        "overlap": 0.5,
        "postprocessing": {
            "onset": parsed_args.vad_onset,
            "offset": parsed_args.vad_offset,
            "min_duration_on": parsed_args.vad_min_duration_on,
            "min_duration_off": parsed_args.vad_min_duration_off,
            "filter_speech_first": True,
        },
    }

    print(f"Batch transcription with NeMo ASR + VAD starting...")
    print(f"ASR Model: {parsed_args.model}")
    print(f"VAD Model: {parsed_args.vad_model}")
    print(f"Language: {parsed_args.language}")
    print(f"\nVAD Configuration:")
    print(f"  Window length: {vad_config['window_length_in_sec']}s")
    print(f"  Shift length: {vad_config['shift_length_in_sec']}s")
    print(f"  Onset threshold: {vad_config['postprocessing']['onset']}")
    print(f"  Offset threshold: {vad_config['postprocessing']['offset']}")
    print(f"  Min speech duration: {vad_config['postprocessing']['min_duration_on']}s")
    print(f"  Min silence duration: {vad_config['postprocessing']['min_duration_off']}s")

    result_paths = batch_transcribe_nemo.remote(
        model_name=parsed_args.model,
        language=parsed_args.language,
        vad_model_name=parsed_args.vad_model,
        vad_config=vad_config,
    )

    print(f"\nBatch transcription complete!")
    print(f"Processed {len(result_paths)} files")
    print("\nResults saved to Modal Volume at /results/:")
    for path in result_paths:
        print(f"  - {path}")

    print("\nTo download results:")
    print("  modal volume get transcription-results /results ./local_results")

    return result_paths
