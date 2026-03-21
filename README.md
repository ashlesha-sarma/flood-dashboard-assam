# FloodSense Assam

A district-level flood risk and agricultural damage predictor for all 33 districts of Assam, India. Built with Gradient Boosting, Random Forest regression, and SHAP-style feature attribution — visualised on an interactive Leaflet choropleth map.

## Live Demo
→ [Deploy on HuggingFace Spaces or Render]

## ML Architecture

| Component | Model | Purpose |
|-----------|-------|---------|
| Risk classification | Gradient Boosting (60 estimators) | Predict low/moderate/high flood risk per district per day |
| Crop damage | Random Forest Regressor | Estimate hectares of farmland at risk |
| Explainability | Feature contribution scores | SHAP-style "why this prediction" per district |

### Features engineered
- River level (m) + percentage to danger level
- 24h level change + 3-day trend
- 1-day, 3-day, 7-day district rainfall (mm)
- Upstream catchment rainfall proxy
- Monsoon season binary flag

### Evaluation (held-out 20% test set)
| Metric | Gradient Boosting | Threshold Baseline |
|--------|------------------|--------------------|
| F1 (flood class) | ~0.90 | ~0.60 |
| F1 Macro | ~0.88 | ~0.55 |

## Quick Start

```bash
git clone https://github.com/yourname/floodsense-assam
cd floodsense-assam
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

Models train automatically on first run (~20 seconds).

## Data Sources

| Source | Data |
|--------|------|
| IMD Open Data Portal | District rainfall patterns |
| CWC Flood Forecasting | River level thresholds |
| ASDMA FRIMS | Annual flood impact statistics (2014–2023) |
| OpenStreetMap | Base map tiles |

## Project Structure

```
floodsense/
├── app.py                    # Flask API (4 endpoints)
├── services/
│   ├── data_pipeline.py      # Feature engineering + synthetic data generation
│   └── ml_engine.py          # Model training, inference, SHAP attribution
├── static/
│   ├── css/main.css          # Dark editorial UI
│   ├── js/app.js             # Map, charts, detail panel
│   └── data/assam_districts.geojson
├── templates/index.html      # Single-page dashboard
└── requirements.txt
```

## API

| Endpoint | Description |
|----------|-------------|
| `GET /api/districts` | Risk prediction for all 33 districts |
| `GET /api/district/<n>` | Full detail: SHAP, forecast, probabilities |
| `GET /api/metrics` | Model evaluation + feature importance |
| `GET /api/historical` | ASDMA annual flood impact data |

## Tech Stack
Python · Flask · scikit-learn · NumPy · Pandas · Leaflet.js · Chart.js

## License
MIT
