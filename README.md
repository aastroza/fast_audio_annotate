# Fast Audio Annotation

![Interface preview](fast_annotate.png)

A FastHTML-based tool for annotating audio files with timestamped transcription clips. Load any audio file directly in the browser, mark precise start/end timestamps, and document the spoken content for each segment.

## Features

- **Waveform navigation** powered by [WaveSurfer.js](https://wavesurfer-js.org/) with timeline overlay
- **Clip builder** to capture start/end timestamps from playback and attach transcription text
- **Region editing** by dragging handles on the highlighted clip selection
- **Segment playback controls** including quick rewind/forward and ad-hoc segment playback
- **In-browser workflow** – audio stays on the client, annotations live in the current session

## Getting Started

```bash
# Install dependencies
pip install .

# Run the FastHTML app
python main.py
```

Open your browser to `http://localhost:5001`.

## Usage

1. **Choose an audio file** using the file picker. The waveform and timeline will load automatically.
2. **Navigate the audio** with the play/pause, stop, and skip buttons.
3. **Set clip boundaries** by capturing the current time as the start or end marker, or by typing values manually.
4. **Provide the transcription text** for the selected time span and click **Add Clip**.
5. **Edit or delete clips** at any time from the clip table. Editing reopens the clip with a draggable waveform region.
6. **Replay any segment** by entering custom start/end values in the quick playback panel.

> Clips are stored in memory for the active browser tab. Export functionality can be added by copying the table data or extending the app to persist JSON.

## Development Notes

- Built with [FastHTML](https://github.com/AnswerDotAI/fasthtml) and vanilla JavaScript.
- Waveform rendering uses CDN-hosted WaveSurfer.js plus its Regions and Timeline plugins.
- Styling lives in `styles.css`; tweak colors, spacing, or responsive behaviour there.

## License

MIT License – see [LICENSE](LICENSE).
