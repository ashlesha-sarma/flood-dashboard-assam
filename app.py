"""FloodSense Assam — Flask entry point.
FIX: Models are trained in a background thread on startup so the server
     accepts requests immediately. /api/districts returns a 202 with a
     'training' flag while training is in progress; the JS retries.
"""
import logging
import threading
from flask import Flask, jsonify, render_template

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s',
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Background training state ─────────────────────────────────────────
_training_done  = False
_training_error = None


def _train_in_background():
    global _training_done, _training_error
    try:
        logger.info("Background training started…")
        from services.ml_engine import _ensure_trained
        _ensure_trained()
        _training_done = True
        logger.info("Background training complete.")
    except Exception as exc:
        _training_error = str(exc)
        logger.exception("Background training failed: %s", exc)


# Start training immediately when the module loads (works for both
# `python app.py` and gunicorn --preload).
_thread = threading.Thread(target=_train_in_background, daemon=True)
_thread.start()


# ── Error helper ──────────────────────────────────────────────────────
def _err(msg, status=500):
    logger.error("API %d: %s", status, msg)
    return jsonify({'error': True, 'message': msg}), status


# ── Routes ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/districts')
def api_districts():
    """Return all district risk predictions.
    While models are still training, returns HTTP 202 with training=True
    so the frontend knows to retry after a short delay.
    """
    if _training_error:
        return _err(f"Model training failed: {_training_error}")

    if not _training_done:
        # Still training — tell the client to retry
        return jsonify({'training': True, 'districts': []}), 202

    try:
        from services.ml_engine import get_all_districts_risk
        return jsonify({'districts': get_all_districts_risk()})
    except Exception as exc:
        logger.exception(exc)
        return _err(str(exc))


@app.route('/api/metrics')
def api_metrics():
    if not _training_done:
        return jsonify({'training': True}), 202
    try:
        from services.ml_engine import get_model_metrics
        return jsonify(get_model_metrics())
    except Exception as exc:
        logger.exception(exc)
        return _err(str(exc))


@app.route('/api/historical')
def api_historical():
    try:
        from services.ml_engine import get_historical_summary
        return jsonify(get_historical_summary())
    except Exception as exc:
        logger.exception(exc)
        return _err(str(exc))


if __name__ == '__main__':
    app.run(debug=False, port=5000)