# FloodSense Assam

District-level flood risk and crop-impact dashboard for Assam, built with Flask, scikit-learn, and Leaflet.

## Project Layout

```text
floodsense-assam/
|-- app.py
|-- requirements.txt
|-- Procfile
|-- runtime.txt
|-- services/
|   |-- __init__.py
|   |-- data_pipeline.py
|   `-- ml_engine.py
|-- static/
|   |-- css/
|   |-- data/
|   `-- js/
`-- templates/
    `-- index.html
```

## Local Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

The app now trains lazily on first API access instead of training during module import. If you want the old non-blocking `202 training` behavior locally, set:

```bash
set FLOODSENSE_BACKGROUND_TRAINING=1
python app.py
```

## Production-Safe Model Workflow

Best practice is to train once before deployment and save the artifacts:

```bash
python -m services.ml_engine
```

That generates `services/model_artifacts.joblib`. On Render or Gunicorn, the app will load that file instead of retraining on startup.

If the artifact file is missing, the app will train once on first request and then save it.

## Render Deployment

1. Push this repository to GitHub.
2. In Render, create a new Web Service from the repo.
3. Use these settings:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn --workers 1 --threads 4 app:app
```

4. Set environment variable `PYTHON_VERSION=3.11.9`.
5. Optional: set `FLOODSENSE_BACKGROUND_TRAINING=1` only if you want `/api/districts` to return `202` while a background thread warms models. Leave it unset for the simplest production behavior.

## Notes

- Keep `.git/` if you are deploying from GitHub. Only exclude it if you are manually uploading a source archive.
- `flask-cors` is enabled in `app.py`.
- `GET /health` returns a simple readiness payload for uptime checks.

## API

- `GET /`
- `GET /health`
- `GET /api/districts`
- `GET /api/metrics`
- `GET /api/historical`
