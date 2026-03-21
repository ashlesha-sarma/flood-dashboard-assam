"""FloodSense Assam Flask entry point."""

import logging
import os
import threading

from flask import Flask, jsonify, render_template
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

BACKGROUND_TRAINING = os.environ.get("FLOODSENSE_BACKGROUND_TRAINING", "").lower() in {
    "1",
    "true",
    "yes",
}

_training_done = False
_training_error = None
_training_thread = None
_training_state_lock = threading.Lock()


def _run_training():
    global _training_done, _training_error
    try:
        logger.info("Model initialization started.")
        from services.ml_engine import _ensure_trained

        _ensure_trained()
        _training_done = True
        _training_error = None
        logger.info("Model initialization complete.")
    except Exception as exc:
        _training_error = str(exc)
        logger.exception("Model initialization failed: %s", exc)


def _start_background_training():
    global _training_thread
    with _training_state_lock:
        if _training_done:
            return
        if _training_thread and _training_thread.is_alive():
            return
        _training_thread = threading.Thread(target=_run_training, daemon=True)
        _training_thread.start()


def _ensure_ready():
    if _training_done:
        return True
    if _training_error:
        return False
    if BACKGROUND_TRAINING:
        _start_background_training()
        return False
    _run_training()
    return _training_done


def _err(msg, status=500):
    logger.error("API %d: %s", status, msg)
    return jsonify({"error": True, "message": msg}), status


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"ok": True, "training": not _training_done and not _training_error})


@app.route("/api/districts")
def api_districts():
    if not _ensure_ready():
        if _training_error:
            return _err(f"Model training failed: {_training_error}")
        return jsonify({"training": True, "districts": []}), 202

    try:
        from services.ml_engine import get_all_districts_risk

        return jsonify({"districts": get_all_districts_risk()})
    except Exception as exc:
        logger.exception("District API failed: %s", exc)
        return _err(str(exc))


@app.route("/api/metrics")
def api_metrics():
    if not _ensure_ready():
        if _training_error:
            return _err(f"Model training failed: {_training_error}")
        return jsonify({"training": True}), 202

    try:
        from services.ml_engine import get_model_metrics

        return jsonify(get_model_metrics())
    except Exception as exc:
        logger.exception("Metrics API failed: %s", exc)
        return _err(str(exc))


@app.route("/api/historical")
def api_historical():
    try:
        from services.ml_engine import get_historical_summary

        return jsonify(get_historical_summary())
    except Exception as exc:
        logger.exception("Historical API failed: %s", exc)
        return _err(str(exc))


if __name__ == "__main__":
    app.run(debug=False, port=5000)
