"""FloodSense Assam — Flask entry point"""
import logging, os
from flask import Flask, jsonify, render_template, request

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s — %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

def _err(msg, status=500):
    return jsonify({'error': True, 'message': msg}), status

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/districts')
def api_districts():
    try:
        from services.ml_engine import get_all_districts_risk
        return jsonify({'districts': get_all_districts_risk()})
    except Exception as e:
        logger.exception(e); return _err(str(e))

@app.route('/api/district/<name>')
def api_district(name):
    try:
        from services.ml_engine import predict_district
        return jsonify(predict_district(name))
    except Exception as e:
        logger.exception(e); return _err(str(e))

@app.route('/api/metrics')
def api_metrics():
    try:
        from services.ml_engine import get_model_metrics
        return jsonify(get_model_metrics())
    except Exception as e:
        logger.exception(e); return _err(str(e))

@app.route('/api/historical')
def api_historical():
    try:
        from services.ml_engine import get_historical_summary
        return jsonify(get_historical_summary())
    except Exception as e:
        logger.exception(e); return _err(str(e))

if __name__ == '__main__':
    logger.info("Pre-training models...")
    from services.ml_engine import _ensure_trained
    _ensure_trained()
    logger.info("Models ready — starting server")
    app.run(debug=False, port=5000)
