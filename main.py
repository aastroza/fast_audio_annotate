"""FastHTML audio annotation tool for creating timestamped transcription clips."""
from fasthtml.common import *
from starlette.responses import FileResponse

app, rt = fast_app(
    hdrs=(
        Link(rel="stylesheet", href="/styles.css"),
    ),
    pico=False,
)


@rt("/")
def index():
    """Render the audio annotation interface."""
    scripts = (
        Script(src="https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.min.js"),
        Script(src="https://unpkg.com/wavesurfer.js@7/dist/plugins/timeline.min.js"),
        Script(src="https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.min.js"),
        Script(
            """
const formatTime = (time) => {
    if (!Number.isFinite(time)) return '0:00.000';
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    const ms = Math.round((time % 1) * 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
};

document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('audio-file');
    const audioName = document.getElementById('audio-name');
    const playPauseBtn = document.getElementById('play-pause');
    const stopBtn = document.getElementById('stop');
    const rewindBtn = document.getElementById('rewind');
    const forwardBtn = document.getElementById('forward');
    const segmentStartBtn = document.getElementById('segment-start');
    const segmentEndBtn = document.getElementById('segment-end');
    const addClipBtn = document.getElementById('add-clip');
    const clearClipBtn = document.getElementById('clear-clip');
    const startInput = document.getElementById('clip-start');
    const endInput = document.getElementById('clip-end');
    const textInput = document.getElementById('clip-text');
    const clipsBody = document.getElementById('clips-body');
    const clipStatus = document.getElementById('clip-status');
    const segmentPlayBtn = document.getElementById('play-segment');
    const segmentStartInput = document.getElementById('segment-start-input');
    const segmentEndInput = document.getElementById('segment-end-input');

    let clipId = 0;
    let clips = [];
    let editingId = null;

    const wavesurfer = WaveSurfer.create({
        container: '#waveform',
        height: 180,
        waveColor: '#96b4ff',
        progressColor: '#0d6efd',
        cursorColor: '#0d6efd',
    });

    const regions = wavesurfer.registerPlugin(WaveSurfer.Regions.create({
        dragSelection: { slop: 5 },
    }));

    wavesurfer.registerPlugin(WaveSurfer.Timeline.create({
        container: '#timeline',
        primaryLabelInterval: 1,
        secondaryLabelInterval: 0.5,
    }));

    let activeRegion = null;

    const setActiveRegion = (start, end) => {
        if (!Number.isFinite(start) || !Number.isFinite(end)) return;
        if (activeRegion) activeRegion.remove();
        activeRegion = regions.addRegion({
            start,
            end,
            color: 'rgba(13, 110, 253, 0.15)',
            drag: true,
            resize: true,
        });
        activeRegion.on('update-end', () => {
            startInput.value = activeRegion.start.toFixed(3);
            endInput.value = activeRegion.end.toFixed(3);
        });
    };

    const renderClips = () => {
        clipsBody.innerHTML = '';
        if (!clips.length) {
            clipsBody.innerHTML = '<tr><td colspan="5" class="empty">No clips yet. Define a clip using the controls above.</td></tr>';
            return;
        }
        clips
            .sort((a, b) => a.start - b.start)
            .forEach((clip) => {
                const row = document.createElement('tr');
                if (clip.id === editingId) row.classList.add('editing');
                row.innerHTML = `
                    <td>${formatTime(clip.start)}</td>
                    <td>${formatTime(clip.end)}</td>
                    <td>${formatTime(clip.end - clip.start)}</td>
                    <td class="transcription">${clip.text.replace(/</g, '&lt;')}</td>
                    <td class="actions">
                        <button type="button" data-action="play">Play</button>
                        <button type="button" data-action="edit">Edit</button>
                        <button type="button" data-action="delete">Delete</button>
                    </td>
                `;
                row.dataset.id = clip.id;
                clipsBody.appendChild(row);
            });
    };

    const resetForm = () => {
        editingId = null;
        startInput.value = '';
        endInput.value = '';
        textInput.value = '';
        clipStatus.textContent = 'Define a new clip by setting start/end times and entering transcription text.';
        clipStatus.className = 'status';
        if (activeRegion) {
            activeRegion.remove();
            activeRegion = null;
        }
        addClipBtn.textContent = 'Add Clip';
        clearClipBtn.classList.add('hidden');
    };

    const parseInputTime = (input) => {
        const value = parseFloat(input.value);
        return Number.isFinite(value) ? value : null;
    };

    const validateTimes = (start, end) => {
        if (start === null || end === null) {
            clipStatus.textContent = 'Both start and end times are required.';
            clipStatus.className = 'status error';
            return false;
        }
        if (start < 0 || end <= start) {
            clipStatus.textContent = 'End time must be greater than start time.';
            clipStatus.className = 'status error';
            return false;
        }
        clipStatus.textContent = '';
        clipStatus.className = 'status';
        return true;
    };

    fileInput.addEventListener('change', () => {
        const file = fileInput.files?.[0];
        if (!file) return;
        const url = URL.createObjectURL(file);
        wavesurfer.load(url);
        audioName.textContent = file.name;
        resetForm();
        clips = [];
        renderClips();
    });

    playPauseBtn.addEventListener('click', () => {
        wavesurfer.playPause();
    });

    wavesurfer.on('play', () => {
        playPauseBtn.textContent = 'Pause';
    });

    wavesurfer.on('pause', () => {
        playPauseBtn.textContent = 'Play';
    });

    stopBtn.addEventListener('click', () => {
        wavesurfer.stop();
    });

    rewindBtn.addEventListener('click', () => {
        const current = wavesurfer.getCurrentTime();
        wavesurfer.setTime(Math.max(current - 5, 0));
    });

    forwardBtn.addEventListener('click', () => {
        const current = wavesurfer.getCurrentTime();
        const duration = wavesurfer.getDuration();
        wavesurfer.setTime(Math.min(current + 5, duration));
    });

    segmentStartBtn.addEventListener('click', () => {
        const time = wavesurfer.getCurrentTime();
        startInput.value = time.toFixed(3);
        const end = parseInputTime(endInput);
        if (end !== null && end > time) setActiveRegion(time, end);
    });

    segmentEndBtn.addEventListener('click', () => {
        const time = wavesurfer.getCurrentTime();
        endInput.value = time.toFixed(3);
        const start = parseInputTime(startInput);
        if (start !== null && time > start) setActiveRegion(start, time);
    });

    startInput.addEventListener('change', () => {
        const start = parseInputTime(startInput);
        const end = parseInputTime(endInput);
        if (start !== null && end !== null && end > start) setActiveRegion(start, end);
    });

    endInput.addEventListener('change', () => {
        const start = parseInputTime(startInput);
        const end = parseInputTime(endInput);
        if (start !== null && end !== null && end > start) setActiveRegion(start, end);
    });

    addClipBtn.addEventListener('click', () => {
        const start = parseInputTime(startInput);
        const end = parseInputTime(endInput);
        if (!validateTimes(start, end)) return;
        if (!textInput.value.trim()) {
            clipStatus.textContent = 'Transcription text is required.';
            clipStatus.className = 'status error';
            return;
        }

        if (editingId === null) {
            clips.push({
                id: clipId++,
                start,
                end,
                text: textInput.value.trim(),
            });
        } else {
            clips = clips.map((clip) =>
                clip.id === editingId ? { ...clip, start, end, text: textInput.value.trim() } : clip,
            );
        }

        renderClips();
        resetForm();
    });

    clearClipBtn.addEventListener('click', () => {
        resetForm();
    });

    clipsBody.addEventListener('click', (event) => {
        const button = event.target.closest('button');
        if (!button) return;
        const row = button.closest('tr');
        const id = Number(row?.dataset.id);
        const clip = clips.find((c) => c.id === id);
        if (!clip) return;

        const action = button.dataset.action;
        if (action === 'play') {
            wavesurfer.play(clip.start, clip.end);
            setActiveRegion(clip.start, clip.end);
        }
        if (action === 'edit') {
            editingId = clip.id;
            startInput.value = clip.start.toFixed(3);
            endInput.value = clip.end.toFixed(3);
            textInput.value = clip.text;
            clipStatus.textContent = `Editing clip ${formatTime(clip.start)}–${formatTime(clip.end)}`;
            clipStatus.className = 'status';
            setActiveRegion(clip.start, clip.end);
            addClipBtn.textContent = 'Save Changes';
            clearClipBtn.classList.remove('hidden');
        }
        if (action === 'delete') {
            clips = clips.filter((c) => c.id !== id);
            if (editingId === id) resetForm();
            renderClips();
        }
    });

    segmentPlayBtn.addEventListener('click', () => {
        const start = parseFloat(segmentStartInput.value);
        const end = parseFloat(segmentEndInput.value);
        if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
            segmentPlayBtn.classList.add('shake');
            setTimeout(() => segmentPlayBtn.classList.remove('shake'), 300);
            return;
        }
        wavesurfer.play(start, end);
    });

    wavesurfer.on('ready', () => {
        segmentStartInput.value = '0.000';
        segmentEndInput.value = wavesurfer.getDuration().toFixed(3);
        playPauseBtn.textContent = 'Play';
    });

    resetForm();
    renderClips();
});
            """,
            type="module",
        ),
    )

    return Titled(
        "Audio Annotation Tool",
        Main(
            Div(
                H1("Audio Clip Annotator"),
                P(
                    "Load an audio file, define timestamped clips, and record transcription text for each segment.",
                    cls="intro",
                ),
                Div(
                    Label(
                        "Audio file",
                        Input(type="file", id="audio-file", accept="audio/*"),
                        cls="file-label",
                    ),
                    Span("No file selected", id="audio-name", cls="file-name"),
                    cls="file-picker",
                ),
                Div(
                    Div(id="waveform", cls="waveform"),
                    Div(id="timeline", cls="timeline"),
                    cls="waveform-shell",
                ),
                Div(
                    Button("Play", id="play-pause", cls="control primary"),
                    Button("Stop", id="stop", cls="control"),
                    Button("⏪ 5s", id="rewind", cls="control"),
                    Button("5s ⏩", id="forward", cls="control"),
                    cls="playback-controls",
                ),
                Fieldset(
                    Legend("Clip builder"),
                    Div(
                        Button("Set start", id="segment-start", cls="control"),
                        Button("Set end", id="segment-end", cls="control"),
                        Label("Start", Input(id="clip-start", type="number", step="0.001", min="0", placeholder="seconds")),
                        Label("End", Input(id="clip-end", type="number", step="0.001", min="0", placeholder="seconds")),
                        cls="clip-time-controls",
                    ),
                    Textarea(id="clip-text", rows=3, placeholder="Transcribed content for this clip..."),
                    Div(
                        Button("Add Clip", id="add-clip", cls="control primary"),
                        Button("Cancel edit", id="clear-clip", cls="control danger hidden"),
                        Span("", id="clip-status", cls="status"),
                        cls="clip-actions",
                    ),
                    cls="clip-builder",
                ),
                Fieldset(
                    Legend("Quick segment playback"),
                    Div(
                        Label("Start", Input(id="segment-start-input", type="number", step="0.001", min="0")),
                        Label("End", Input(id="segment-end-input", type="number", step="0.001", min="0")),
                        Button("Play segment", id="play-segment", cls="control"),
                        cls="segment-controls",
                    ),
                ),
                Div(
                    H2("Clips"),
                    Table(
                        Thead(Tr(Th("Start"), Th("End"), Th("Duration"), Th("Transcription"), Th("Actions"))),
                        Tbody(id="clips-body"),
                        cls="clips-table",
                    ),
                    cls="clips-section",
                ),
                cls="app-shell",
            ),
        ),
        scripts=scripts,
    )


@rt("/styles.css")
def styles():
    return FileResponse("styles.css")


if __name__ == "__main__":
    serve()
