"""Common Modal configuration for Fast Audio Annotate transcription."""
import modal

MODEL_DIR = "/model"
AUDIO_DIR = "/audio"
RESULTS_DIR = "/results"

# Modal image with Whisper + WebRTC VAD
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": MODEL_DIR,
        }
    )
    # Install build tools for webrtcvad compilation
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "build-essential",
        "clang",
    )
    .pip_install(
        "torch==2.7.1",
        "transformers==4.48.1",
        "accelerate==1.3.0",
        "evaluate==0.4.3",
        "librosa==0.11.0",
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.32.4",
        "datasets[audio]==4.0.0",
        "soundfile==0.13.1",
        "jiwer==4.0.0",
        "pyloudnorm==0.1.1",
        "webrtcvad==2.0.10",
        "resampy==0.4.3",
    )
    .entrypoint([])
)

# Modal image with NeMo ASR + integrated VAD
nemo_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": MODEL_DIR,
            "CXX": "g++",
            "CC": "g++",
        }
    )
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.7.1",
        "evaluate==0.4.3",
        "librosa==0.11.0",
        "soundfile==0.13.1",
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.32.4",
        "cuda-python==12.8.0",
        "nemo_toolkit[asr]==2.3.1",
    )
    .entrypoint([])
)

# Modal Volumes for persistent storage
model_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
audio_volume = modal.Volume.from_name("audio-files", create_if_missing=True)
results_volume = modal.Volume.from_name("transcription-results", create_if_missing=True)

# Modal App
app = modal.App(
    "fast-audio-annotate",
    image=image,
)
