"""Stage audio files to Modal Volume for processing."""
from pathlib import Path
from typing import List

from .common import AUDIO_DIR, app, audio_volume


@app.function(
    volumes={AUDIO_DIR: audio_volume},
    timeout=60 * 5,  # 5 minutes per file upload
)
def upload_single_file(rel_path: str, file_content: bytes) -> str:
    """
    Upload a single audio file to Modal Volume.

    Args:
        rel_path: Relative path for the file in Modal Volume
        file_content: Binary content of the file

    Returns:
        The relative path of the uploaded file
    """
    # Create Modal Volume directory structure
    modal_audio_dir = Path(AUDIO_DIR)
    modal_path = modal_audio_dir / rel_path

    # Create parent directories
    modal_path.parent.mkdir(parents=True, exist_ok=True)

    # Write file content to Modal Volume
    with open(modal_path, 'wb') as f:
        f.write(file_content)

    # Commit changes to Modal Volume
    audio_volume.commit()

    print(f"Successfully uploaded: {rel_path}")
    return rel_path


@app.function(
    volumes={AUDIO_DIR: audio_volume},
    timeout=60 * 30,  # 30 minutes for large uploads
)
def upload_audio_files(audio_folder: str) -> List[str]:
    """
    Upload audio files from local directory to Modal Volume.
    DEPRECATED: Use upload_single_file instead.

    Args:
        audio_folder: Path to local audio folder

    Returns:
        List of uploaded file paths in Modal Volume
    """
    import shutil

    audio_folder_path = Path(audio_folder).resolve()

    if not audio_folder_path.exists():
        raise ValueError(f"Audio folder does not exist: {audio_folder}")

    # Supported audio formats
    audio_extensions = {".webm", ".mp3", ".wav", ".ogg", ".m4a", ".flac"}

    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_folder_path.rglob(f"*{ext}"))

    if not audio_files:
        raise ValueError(f"No audio files found in {audio_folder}")

    print(f"Found {len(audio_files)} audio files to upload")

    # Create Modal Volume directory structure
    modal_audio_dir = Path(AUDIO_DIR)
    modal_audio_dir.mkdir(parents=True, exist_ok=True)

    uploaded_paths = []

    for local_path in audio_files:
        # Preserve relative path structure
        rel_path = local_path.relative_to(audio_folder_path)
        modal_path = modal_audio_dir / rel_path

        # Create parent directories
        modal_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file to Modal Volume
        shutil.copy2(str(local_path), str(modal_path))

        uploaded_paths.append(str(rel_path))
        print(f"Uploaded: {rel_path}")

    # Commit changes to Modal Volume
    audio_volume.commit()

    print(f"Successfully uploaded {len(uploaded_paths)} files to Modal Volume")

    return uploaded_paths

    for local_path in audio_files:
        # Preserve relative path structure
        rel_path = local_path.relative_to(audio_folder_path)
        modal_path = modal_audio_dir / rel_path

        # Create parent directories
        modal_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file to Modal Volume
        shutil.copy2(str(local_path), str(modal_path))

        uploaded_paths.append(str(rel_path))
        print(f"Uploaded: {rel_path}")

    # Commit changes to Modal Volume
    audio_volume.commit()

    print(f"Successfully uploaded {len(uploaded_paths)} files to Modal Volume")

    return uploaded_paths


@app.function(
    volumes={AUDIO_DIR: audio_volume},
)
def list_audio_files() -> List[str]:
    """
    List all audio files in Modal Volume.

    Returns:
        List of audio file paths in Modal Volume
    """
    audio_dir = Path(AUDIO_DIR)

    if not audio_dir.exists():
        return []

    audio_extensions = {".webm", ".mp3", ".wav", ".ogg", ".m4a", ".flac"}

    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.rglob(f"*{ext}"))

    # Return relative paths
    rel_paths = [str(p.relative_to(audio_dir)) for p in audio_files]
    rel_paths.sort()

    return rel_paths
