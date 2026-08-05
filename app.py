from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from faster_whisper import WhisperModel
from werkzeug.utils import secure_filename

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny.en")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE",
    "int8" if DEVICE == "cpu" else "float16",
)
CPU_THREADS = int(
    os.getenv("WHISPER_CPU_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))
)
WORKERS = int(os.getenv("WHISPER_WORKERS", "1"))

model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    cpu_threads=CPU_THREADS,
    num_workers=WORKERS,
)


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_name=MODEL_SIZE,
        device_name=DEVICE,
        cpu_threads=CPU_THREADS,
    )


@app.get("/api/status")
def status():
    return jsonify(
        ready=True,
        model=MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
        translation_available=GoogleTranslator is not None,
    )


@app.post("/api/transcribe")
def transcribe():
    media = request.files.get("media")
    if media is None or not media.filename:
        return jsonify(error="No media file was supplied."), 400

    suffix = Path(secure_filename(media.filename)).suffix or ".media"
    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            media.save(temp_file)
            temp_path = temp_file.name

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
                text = word.word.strip()
                if text:
                    words.append(
                        {
                            "text": text,
                            "start": round(float(word.start), 3),
                            "end": round(float(word.end), 3),
                            "probability": round(float(word.probability), 4),
                        }
                    )

        return jsonify(
            language=info.language,
            language_probability=round(float(info.language_probability), 4),
            duration=round(duration, 3),
            model=MODEL_SIZE,
            device=DEVICE,
            words=words,
        )
    except Exception as exc:
        app.logger.exception("Transcription failed")
        return jsonify(error=f"Transcription failed: {exc}"), 500
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


@lru_cache(maxsize=2048)
def translate_cached(text: str, target: str) -> str:
    if GoogleTranslator is None:
        raise RuntimeError(
            "Translation support is not installed. Run: python -m pip install deep-translator"
        )
    return GoogleTranslator(source="en", target=target).translate(text)


@app.post("/api/translate")
def translate():
    payload = request.get_json(silent=True) or {}
    texts = payload.get("texts")
    target = str(payload.get("target", "fa")).strip() or "fa"

    if not isinstance(texts, list) or not texts:
        return jsonify(error="No text was supplied for translation."), 400
    if len(texts) > 40:
        return jsonify(error="Send no more than 40 segments in one translation batch."), 400
    if GoogleTranslator is None:
        return jsonify(
            error="Translation support is not installed. Run: python -m pip install deep-translator"
        ), 503

    cleaned = [str(item).strip() for item in texts]

    try:
        translator = GoogleTranslator(source="en", target=target)
        non_empty = [text for text in cleaned if text]
        translated_non_empty: list[str] = []

        if non_empty:
            try:
                result = translator.translate_batch(non_empty)
                if not isinstance(result, list) or len(result) != len(non_empty):
                    raise RuntimeError("The translation service returned an incomplete batch.")
                translated_non_empty = [str(item or "") for item in result]
            except Exception:
                # Some networks reject batch translation. Fall back to cached single requests.
                translated_non_empty = [translate_cached(text, target) for text in non_empty]

        iterator = iter(translated_non_empty)
        translations = [next(iterator) if text else "" for text in cleaned]
        return jsonify(translations=translations, target=target)
    except Exception as exc:
        app.logger.exception("Translation failed")
        return jsonify(error=f"Translation failed: {exc}"), 502


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(error="The uploaded file is larger than 500 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
