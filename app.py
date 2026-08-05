from __future__ import annotations

import logging
import os
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from faster_whisper import WhisperModel
from werkzeug.utils import secure_filename

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

app = Flask(__name__)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny.en").strip() or "tiny.en"
DEVICE = os.getenv("WHISPER_DEVICE", "cpu").strip() or "cpu"
COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE",
    "int8" if DEVICE == "cpu" else "float16",
).strip()
CPU_THREADS = max(1, int(os.getenv("WHISPER_CPU_THREADS", "1")))
WORKERS = max(1, int(os.getenv("WHISPER_WORKERS", "1")))
TRANSLATION_BATCH_LIMIT = max(1, int(os.getenv("TRANSLATION_BATCH_LIMIT", "20")))

ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac",
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".mpeg", ".mpg",
}

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("wordsync")

_whisper_model: WhisperModel | None = None
_model_lock = threading.Lock()
_transcription_lock = threading.Lock()


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    with _model_lock:
        if _whisper_model is None:
            logger.info(
                "Loading Whisper model=%s device=%s compute_type=%s threads=%s workers=%s",
                MODEL_SIZE,
                DEVICE,
                COMPUTE_TYPE,
                CPU_THREADS,
                WORKERS,
            )
            _whisper_model = WhisperModel(
                MODEL_SIZE,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=CPU_THREADS,
                num_workers=WORKERS,
            )
            logger.info("Whisper model loaded successfully.")

    return _whisper_model


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def remove_temp_file(path: str | None) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Could not remove temporary file: %s", path, exc_info=True)


def json_error(message: str, status_code: int):
    return jsonify(error=message), status_code


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_name=MODEL_SIZE,
        device_name=DEVICE,
        cpu_threads=CPU_THREADS,
        max_upload_mb=MAX_UPLOAD_MB,
    )


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        model=MODEL_SIZE,
        device=DEVICE,
        model_loaded=_whisper_model is not None,
    ), 200


@app.get("/api/status")
def status():
    return jsonify(
        ready=True,
        model=MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
        workers=WORKERS,
        model_loaded=_whisper_model is not None,
        translation_available=GoogleTranslator is not None,
        max_upload_mb=MAX_UPLOAD_MB,
    )


@app.post("/api/transcribe")
def transcribe():
    media = request.files.get("media")
    if media is None or not media.filename:
        return json_error("No media file was supplied.", 400)

    safe_name = secure_filename(media.filename)
    if not safe_name:
        return json_error("The uploaded filename is invalid.", 400)
    if not allowed_file(safe_name):
        return json_error(
            "Unsupported file type. Please upload a common audio or video file.",
            400,
        )

    suffix = Path(safe_name).suffix.lower()
    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="wordsync_") as temp_file:
            media.save(temp_file)
            temp_path = temp_file.name

        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return json_error("The uploaded file is empty.", 400)

        logger.info(
            "Starting transcription: filename=%s size_bytes=%s",
            safe_name,
            os.path.getsize(temp_path),
        )

        model = get_whisper_model()

        with _transcription_lock:
            segments, info = model.transcribe(
                temp_path,
                language="en",
                beam_size=1,
                best_of=1,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 350,
                    "speech_pad_ms": 120,
                },
                condition_on_previous_text=False,
                temperature=0.0,
            )

            words: list[dict[str, float | str]] = []
            duration = 0.0

            for segment in segments:
                duration = max(duration, float(segment.end))
                for word in segment.words or []:
                    text = (word.word or "").strip()
                    if not text:
                        continue
                    words.append(
                        {
                            "text": text,
                            "start": round(float(word.start), 3),
                            "end": round(float(word.end), 3),
                            "probability": round(
                                float(getattr(word, "probability", 0.0) or 0.0),
                                4,
                            ),
                        }
                    )

        if not words:
            return json_error("No clear English speech was detected in this file.", 422)

        logger.info(
            "Transcription completed: filename=%s words=%s duration=%s",
            safe_name,
            len(words),
            round(duration, 3),
        )

        return jsonify(
            language=info.language,
            language_probability=round(
                float(getattr(info, "language_probability", 0.0) or 0.0),
                4,
            ),
            duration=round(duration, 3),
            model=MODEL_SIZE,
            device=DEVICE,
            words=words,
        )

    except MemoryError:
        logger.exception("Transcription failed because the server ran out of memory.")
        return json_error(
            "The server ran out of memory. Try a shorter or smaller file.",
            507,
        )
    except Exception as exc:
        logger.exception("Transcription failed.")
        return json_error(f"Transcription failed: {exc}", 500)
    finally:
        remove_temp_file(temp_path)


@lru_cache(maxsize=4096)
def translate_cached(text: str, target: str) -> str:
    if GoogleTranslator is None:
        raise RuntimeError(
            "Translation support is not installed. Run: python -m pip install deep-translator"
        )

    clean_text = text.strip()
    if not clean_text:
        return ""

    result = GoogleTranslator(source="en", target=target).translate(clean_text)
    return str(result or "")


@app.post("/api/translate")
def translate():
    payload: dict[str, Any] = request.get_json(silent=True) or {}
    texts = payload.get("texts")
    target = str(payload.get("target", "fa")).strip() or "fa"

    if not isinstance(texts, list) or not texts:
        return json_error("No text was supplied for translation.", 400)
    if len(texts) > TRANSLATION_BATCH_LIMIT:
        return json_error(
            f"Send no more than {TRANSLATION_BATCH_LIMIT} segments in one batch.",
            400,
        )
    if GoogleTranslator is None:
        return json_error(
            "Translation support is not installed. Run: python -m pip install deep-translator",
            503,
        )

    cleaned = [str(item or "").strip() for item in texts]

    try:
        translations = [translate_cached(text, target) if text else "" for text in cleaned]
        return jsonify(translations=translations, target=target)
    except Exception as exc:
        logger.exception("Translation failed.")
        return json_error(f"Translation failed: {exc}", 502)


@app.errorhandler(404)
def not_found(_error):
    return json_error("The requested endpoint was not found.", 404)


@app.errorhandler(405)
def method_not_allowed(_error):
    return json_error("This HTTP method is not allowed for this endpoint.", 405)


@app.errorhandler(413)
def file_too_large(_error):
    return json_error(
        f"The uploaded file must be smaller than {MAX_UPLOAD_MB} MB.",
        413,
    )


@app.errorhandler(500)
def internal_server_error(_error):
    logger.exception("Unhandled server error.")
    return json_error("An unexpected server error occurred.", 500)


if __name__ == "__main__":
    local_port = int(os.getenv("PORT", "5000"))
    app.run(
        host="127.0.0.1",
        port=local_port,
        debug=False,
        threaded=False,
    )
