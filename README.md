# WordSync YouTube update

Install: `python -m pip install -r requirements.txt`

Render start command: `gunicorn app:app --worker-class sync --workers 1 --timeout 600 --bind 0.0.0.0:$PORT`

Supports public single-video YouTube URLs. Audio is downloaded temporarily, transcribed, then deleted. YouTube may block cloud-hosted requests or require cookies.
