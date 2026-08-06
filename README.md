# WordSync Studio: YouTube Fullscreen Final

This version supports:

- Local audio upload
- Local video upload
- Public YouTube URL transcription
- Word-by-word English highlighting
- Optional Persian translation
- English and Persian subtitles over local videos
- English and Persian subtitles over YouTube videos
- Custom full-screen mode that keeps both subtitle layers visible

## Install

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python app.py
```

On Windows Command Prompt, activate with:

```cmd
.venv\Scripts\activate.bat
```

## Render start command

```bash
gunicorn app:app --worker-class sync --workers 1 --timeout 600 --bind 0.0.0.0:$PORT
```

Set the Render health-check path to:

```text
/health
```

## Important full-screen behaviour

Use the application's **Full screen** button. The native YouTube full-screen button is disabled so that the English and Persian HTML subtitle layers remain inside the element that enters full-screen mode.
