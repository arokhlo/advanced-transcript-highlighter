# TellMe Podcast – WordSync Studio

A local Flask application that transcribes English audio/video, highlights each spoken word, displays captions over video, and optionally shows Persian translation below the English subtitle in right-to-left direction.

## Final features

- English transcription with word-level timestamps
- Current word highlighted over the video
- English subtitle above Persian subtitle
- Persian subtitle displayed RTL and right-aligned
- Translation toggle button
- Playback-speed slider from 0.5× to 2×
- Transcript panel with search and click-to-seek
- TXT, SRT and JSON export
- Fast CPU defaults using `tiny.en`

## Windows Git Bash installation

```bash
cd "C:/path/to/advanced-transcript-highlighter"
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Important

Persian translation uses `deep-translator` and requires internet access. English transcription runs locally after the Whisper model has been downloaded.

For a more accurate but slower model:

```bash
export WHISPER_MODEL=small.en
python app.py
```
# advanced-transcript-highlighter
