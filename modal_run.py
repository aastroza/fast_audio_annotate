"""Modal entrypoints for fast_audio_annotate transcription."""
import argparse
from pathlib import Path

from src.fast_audio_annotate.modal_transcription import app, ModalWhisperTranscriber
from src.fast_audio_annotate.metadata import iter_audio_files


@app.local_entrypoint()
def test_transcription(*args):
    """
    Test Modal transcription with a single audio file.

    Usage:
        modal run modal_run.py::test_transcription
        modal run modal_run.py::test_transcription --audio-file ./audio/example.webm
        modal run modal_run.py::test_transcription --model openai/whisper-large-v3-turbo
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Path to audio file to transcribe (auto-detects first file if not specified)",
    )
    parser.add_argument(
        "--audio-folder",
        type=str,
        default="./audio",
        help="Folder containing audio files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3",
        help="Whisper model to use",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="es",
        help="Language code (e.g., 'es', 'en') or 'auto' for auto-detect",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Include word-level timestamps",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD segmentation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    cfg = parser.parse_args(args=args)

    print("="*80)
    print("MODAL TRANSCRIPTION TEST")
    print("="*80)

    # Find audio file
    if cfg.audio_file:
        audio_path = Path(cfg.audio_file)
        if not audio_path.exists():
            print(f"\n❌ Error: File {audio_path} does not exist")
            return
    else:
        audio_dir = Path(cfg.audio_folder)
        if not audio_dir.exists():
            print(f"\n❌ Error: Directory {audio_dir} does not exist")
            print(f"   Create '{cfg.audio_folder}' and add audio files")
            return

        audio_files = list(iter_audio_files(audio_dir))
        if not audio_files:
            print(f"\n❌ Error: No audio files found in {audio_dir}")
            print("   Supported formats: .wav, .mp3, .webm, .ogg, .m4a, .flac")
            return

        audio_path = audio_files[0]

    # Configuration
    language = None if cfg.language.lower() == "auto" else cfg.language
    chunking_strategy = "none" if cfg.no_vad else "auto"

    print(f"\nConfiguration:")
    print(f"  Model: {cfg.model}")
    print(f"  Language: {cfg.language}")
    print(f"  VAD: {'Disabled' if cfg.no_vad else 'Enabled (auto)'}")
    print(f"  Word timestamps: {'Enabled' if cfg.word_timestamps else 'Disabled'}")
    print(f"  Batch size: {cfg.batch_size}")

    # Create transcriber
    print("\nInitializing transcriber with Modal...")
    transcriber = ModalWhisperTranscriber(
        model_name=cfg.model,
        language=language,
        return_word_timestamps=cfg.word_timestamps,
        chunking_strategy=chunking_strategy,
        batch_size=cfg.batch_size,
    )
    print("✓ Transcriber initialized")

    print(f"\nTest file: {audio_path}")

    # Transcribe
    print("\nTranscribing (this may take a few minutes on first run)...")
    print("  - Modal will download the model if not cached")
    print("  - GPU will start automatically")
    if not cfg.no_vad:
        print("  - Audio will be segmented using VAD")

    try:
        result = transcriber.transcribe_file(audio_path)

        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)

        print(f"\nDetected language: {result.language or 'N/A'}")
        print(f"Number of segments: {len(result.segments)}")

        print("\nSegments:")
        for i, segment in enumerate(result.segments, 1):
            duration = segment.end - segment.start
            print(f"\n  [{i}] {segment.start:.2f}s - {segment.end:.2f}s (duration: {duration:.2f}s)")
            print(f"      \"{segment.text}\"")

            if segment.words:
                print(f"      Words: {len(segment.words)}")
                if cfg.word_timestamps:
                    for word in segment.words[:5]:  # Show first 5 words
                        print(f"        - {word.start:.2f}s: {word.text}")
                    if len(segment.words) > 5:
                        print(f"        ... and {len(segment.words) - 5} more words")

        # Full text
        full_text = " ".join(s.text for s in result.segments)
        print(f"\nFull text:\n  {full_text}")

        print("\n" + "="*80)
        print("✓ Test completed successfully")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error during transcription:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\nTo process multiple files, use:")
    print("  python scripts/preprocess_audio.py --audio-folder ./audio")


@app.local_entrypoint()
def transcribe_file(*args):
    """
    Transcribe a single audio file and save results.

    Usage:
        modal run modal_run.py::transcribe_file --audio-file ./audio/example.webm --output results.json
        modal run modal_run.py::transcribe_file --audio-file ./audio/example.webm --model openai/whisper-large-v3-turbo
    """
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio-file",
        type=str,
        required=True,
        help="Path to audio file to transcribe",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: same as audio file with .json extension)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3",
        help="Whisper model to use",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="es",
        help="Language code (e.g., 'es', 'en') or 'auto' for auto-detect",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Include word-level timestamps",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD segmentation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    cfg = parser.parse_args(args=args)

    # Validate input file
    audio_path = Path(cfg.audio_file)
    if not audio_path.exists():
        print(f"❌ Error: File {audio_path} does not exist")
        return

    # Determine output path
    if cfg.output:
        output_path = Path(cfg.output)
    else:
        output_path = audio_path.with_suffix(".json")

    # Configuration
    language = None if cfg.language.lower() == "auto" else cfg.language
    chunking_strategy = "none" if cfg.no_vad else "auto"

    print(f"Transcribing: {audio_path}")
    print(f"Model: {cfg.model}")
    print(f"Language: {cfg.language}")
    print(f"VAD: {'Disabled' if cfg.no_vad else 'Enabled'}")

    # Create transcriber
    transcriber = ModalWhisperTranscriber(
        model_name=cfg.model,
        language=language,
        return_word_timestamps=cfg.word_timestamps,
        chunking_strategy=chunking_strategy,
        batch_size=cfg.batch_size,
    )

    # Transcribe
    print("\nTranscribing...")
    result = transcriber.transcribe_file(audio_path)

    # Save results
    data = result.to_dict()
    data.update({
        "model": cfg.model,
        "language": result.language or cfg.language,
        "duration": max((segment.end for segment in result.segments), default=0.0),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Transcription saved to: {output_path}")
    print(f"  Segments: {len(result.segments)}")
    print(f"  Duration: {data['duration']:.2f}s")
