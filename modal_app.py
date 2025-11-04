"""Modal deployment entrypoint that reuses the main Fast Audio Annotate app."""
from __future__ import annotations

from pathlib import Path

import main as fast_audio_main

# The FastHTML application defined in ``main.py`` already contains the full UI,
# database integration (including Neon via ``DatabaseBackend``) and routes.
# Reuse that exact app for both local and Modal deployments so behaviour stays
# consistent between environments.
fasthtml_app = fast_audio_main.get_asgi_app()

# Attempt to import Modal so we can expose the ASGI app when available.
try:  # pragma: no cover - optional dependency
    import modal
except ImportError:  # pragma: no cover - optional dependency
    modal = None  # type: ignore[assignment]

# Reuse the Modal app/function defined in ``main.py`` when possible.  That file
# already sets up the container image and dependency installation logic.
app = getattr(fast_audio_main, "modal_app", None)
serve = getattr(fast_audio_main, "serve", None)

if modal is not None and (app is None or serve is None):
    # ``main.py`` was imported in an environment with Modal available but the
    # Modal app wasn't created (for example if the module was imported before
    # Modal was installed).  Fall back to defining the deployment hooks here so
    # ``modal run modal_app.py::serve`` continues to work as expected.
    app = modal.App("fast-audio-annotate")

    requirements_file = Path(__file__).with_name("requirements.txt")
    image_builder = modal.Image.debian_slim(python_version="3.12")
    if requirements_file.exists():
        modal_image = image_builder.pip_install_from_requirements(str(requirements_file))
    else:  # pragma: no cover - sanity fallback when requirements are missing
        modal_image = image_builder.pip_install("python-fasthtml==0.12.33")

    @app.function(image=modal_image)
    @modal.asgi_app()
    def serve():  # type: ignore[no-redef]
        """Expose the FastHTML application to Modal."""

        return fasthtml_app


if __name__ == "__main__":
    # Allow ``python modal_app.py`` to run the exact same UI locally.
    fast_audio_main.fasthtml_serve(host="0.0.0.0", port=5001)
