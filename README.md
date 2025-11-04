# Fast Audio Annotate

Fast Audio Annotate is a lightweight FastHTML interface built for collective audio transcription review.

## Workflow

1. **Prepare your audio**: drop files into the configured folder (default: `audio/`) and, if needed, add a `metadata.json` with extra context.
2. **Run it locally**: install dependencies with `pip install -r requirements.txt` and launch `python main.py` to serve the app at `http://localhost:5001`.
3. **Draft with Whisper**: use the included scripts to generate initial transcriptions before asking the community for help.
4. **Deploy and share**: deploy the app using your preferred platform (see [deployment options](https://github.com/AnswerDotAI/fh-deploy)) and share the link so your collaborators can correct and approve clips from anywhere.

The tool keeps track of every contribution, renders waveforms with WaveSurfer.js, and focuses on being simple so anyone can jump in and improve the transcripts.

## Installation

1. **Clone and enter the project**
   ```bash
   git clone https://github.com/aastroza/fast_audio_annotate.git
   cd fast_audio_annotate
   ```
2. **Create an isolated environment (optional but recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Prepare the audio directory**
   * Place your `.wav`/`.mp3` files under the folder configured by `audio_folder` (defaults to `audio/`).
   * Optionally create `metadata.json` (see [Configuration](#configuration)) before starting the server so metadata is ingested on boot.

## Configuration

You can customize defaults through `config.yaml` (parsed on startup) or CLI arguments (`python main.py --help` shows available flags). The most relevant options are:

| Key | Default | Description |
| --- | --- | --- |
| `audio_folder` | `audio` | Directory containing source audio and the local annotations database. For deployed apps, use a full URL to cloud storage (e.g., `https://storage.googleapis.com/your-bucket-name`). |
| `metadata_filename` | `metadata.json` | File automatically scanned to populate clip metadata. |
| `database_url` | `null` | PostgreSQL/Neon URL to use instead of the local SQLite database. |
| `title` / `description` | App defaults | Copy shown in the page header. |
| `whisper_model` | `openai/whisper-large-v3` | Base model used by preprocessing scripts when generating drafts. |

### Remote Audio Storage

When deploying the app to a cloud platform (instead of running locally), you'll need to host your audio files in cloud storage and configure CORS (Cross-Origin Resource Sharing) to allow browser access:

#### Google Cloud Storage Example

1. **Upload your audio files** to a Google Cloud Storage bucket
2. **Configure CORS** to allow your deployed app's domain to access the files:
   ```bash
   # Create a cors.json file
   echo '[{"origin": ["https://your-app-domain.com"], "method": ["GET"], "responseHeader": ["Content-Type"], "maxAgeSeconds": 3600}]' > cors.json
   
   # Apply CORS configuration to your bucket
   gsutil cors set cors.json gs://your-bucket-name
   ```
3. **Update config.yaml** to point to your bucket:
   ```yaml
   audio_folder: "https://storage.googleapis.com/your-bucket-name"
   ```

#### Other Storage Providers

Similar CORS configuration is needed for other cloud storage providers:
- **AWS S3**: Configure CORS in the S3 bucket settings
- **Azure Blob Storage**: Set CORS rules in the storage account
- **Cloudflare R2**: Configure CORS policies in the R2 dashboard

Make sure your storage allows `GET` requests from your deployed app's origin domain.

## Environment variables

Environment variables can be provided directly or through a `.env` file (automatically loaded by `python-dotenv`). Relevant keys:

- `DATABASE_URL`: PostgreSQL connection string used to store clips remotely.
- `NEON_DATABASE_URL`: Alternate name for the same setting when using Neon-hosted Postgres.
- `USER` / `USERNAME`: Optional contributor name fallback that is logged when reviewers skip the explicit form field.

If neither `DATABASE_URL` nor `NEON_DATABASE_URL` is set, the application stores everything in `annotations.db` inside the audio directory.

## Interface overview

The review UI is designed to make correcting clips fast while providing useful context:

- **Randomized queue**: every reviewer receives a random pending clip; empty queues render a friendly "All caught up" message.
- **Waveform player**: WaveSurfer.js provides playback, zoom, keyboard shortcuts (Space/Q/W), speed controls, and draggable region handles.
- **Timing inputs**: numeric fields mirror the selected region so you can fine-tune start/end times manually if needed.
- **Transcription editor**: a multiline text area seeded with the current draft ready for corrections.
- **Contributor credit**: reviewers can enter their name, and the sidebar highlights top contributors with contribution counts.
- **Metadata panel**: contextual information from `metadata.json` (or your database) is rendered alongside the clip so annotators know what they are hearing.
- **HTMX actions**: the buttons at the bottom save work, mark clips as reviewed, or flag problematic audio without reloading the page.

## Data flow and storage

- Each audio file is split into clips and stored in the `clips` table managed by `db_backend.py`.
- By default the app initializes a SQLite database (`annotations.db`) under the audio folder; setting `DATABASE_URL`/`NEON_DATABASE_URL` switches to PostgreSQL while keeping the same schema.
- When reviewers interact with the interface, every action updates the clip record with the latest timestamps, text, reviewer name, and review status.
- Flagging a clip flips its `marked` flag so it leaves the active queue, while completing a clip marks it as `human_reviewed` and ready for export.
- `metadata.json` entries are loaded into the database at startup so additional fields (speaker, language, tags, etc.) can be displayed in the sidebar.