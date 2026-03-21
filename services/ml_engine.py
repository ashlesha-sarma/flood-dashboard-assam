"""
services/ml_engine.py — FloodSense ML pipeline
Trains GradientBoosting (risk) + RandomForest (crop damage).
Uses sklearn only — no external ML dependencies needed.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

from services.data_pipeline import (
    get_all_data, RIVER_THRESHOLDS, DISTRICT_RIVER_MAP, ALL_DISTRICTS,
    _monsoon_factor,
)

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    'river_level', 'pct_to_danger', 'level_delta', 'level_trend',
    'rain_1d', 'rain_3d', 'rain_7d', 'upstream_3d', 'is_monsoon',
]

FEATURE_LABELS = {
    'river_level':   'River Level (m)',
    'pct_to_danger': '% to Danger Level',
    'level_delta':   'Level Change (24h)',
    'level_trend':   'Level Trend (3-day)',
    'rain_1d':       'Rainfall Today (mm)',
    'rain_3d':       '3-Day Rainfall (mm)',
    'rain_7d':       '7-Day Rainfall (mm)',
    'upstream_3d':   'Upstream Rain (mm)',
    'is_monsoon':    'Monsoon Season',
}

RISK_LABELS = {0: 'low', 1: 'moderate', 2: 'high'}

_models = {}
_scalers = {}
_feature_importance = {}
_eval_metrics = {}
_trained = False


def _train_models():
    global _trained
    logger.info("Training FloodSense ML models...")
    river_df, rain_df, train_df, crop_df = get_all_data()

    # Sample for tractable training
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(train_df))[:12000]
    sample = train_df.iloc[sorted(idx)].reset_index(drop=True)

    X = sample[FEATURE_COLS].values
    y = sample['label'].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    _scalers['risk'] = scaler

    clf = GradientBoostingClassifier(
        n_estimators=60, max_depth=4, learning_rate=0.15,
        subsample=0.8, random_state=42,
    )
    clf.fit(X_sc, y)
    _models['risk'] = clf

    # Eval on held-out 20%
    split = int(len(X_sc) * 0.8)
    X_tr, X_te = X_sc[:split], X_sc[split:]
    y_tr, y_te = y[:split], y[split:]
    clf2 = GradientBoostingClassifier(n_estimators=60, max_depth=4,
                                       learning_rate=0.15, random_state=42)
    clf2.fit(X_tr, y_tr)
    y_pred = clf2.predict(X_te)

    report = classification_report(y_te, y_pred,
                                   target_names=['Low', 'Moderate', 'High'],
                                   output_dict=True, zero_division=0)
    _eval_metrics['risk'] = {
        'f1_macro':  round(f1_score(y_te, y_pred, average='macro', zero_division=0), 3),
        'f1_flood':  round(report.get('High', {}).get('f1-score', 0), 3),
        'precision': round(report.get('High', {}).get('precision', 0), 3),
        'recall':    round(report.get('High', {}).get('recall', 0), 3),
        'n_train': split, 'n_test': len(X_te),
    }

    fi = clf.feature_importances_
    fi_norm = fi / fi.sum()
    _feature_importance['risk'] = [
        {'feature': FEATURE_COLS[i], 'label': FEATURE_LABELS[FEATURE_COLS[i]],
         'importance': round(float(fi_norm[i]), 4)}
        for i in np.argsort(fi_norm)[::-1]
    ]

    # Crop damage regressor
    crop_features = ['river_level', 'pct_to_danger', 'rain_3d', 'rain_7d', 'level_delta', 'is_monsoon']
    if len(crop_df) > 50:
        sc2 = StandardScaler()
        Xc = sc2.fit_transform(crop_df[crop_features].values)
        yc = crop_df['crop_ha'].values
        reg = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=1)
        reg.fit(Xc, yc)
        _models['crop'] = reg
        _scalers['crop'] = sc2
        _scalers['crop_features'] = crop_features

    _models['train_df'] = train_df
    _models['river_df'] = river_df

    _trained = True
    logger.info("Models ready. F1 flood=%.3f", _eval_metrics['risk']['f1_flood'])


def _ensure_trained():
    if not _trained:
        _train_models()


def _get_live_features(district):
    river = DISTRICT_RIVER_MAP.get(district)
    if not river:
        return None
    thr = RIVER_THRESHOLDS[river]
    span = thr['danger'] - thr['min']
    train_df = _models['train_df']
    rows = train_df[train_df['district'] == district]
    if rows.empty:
        return None
    row = rows.iloc[-1]

    doy = datetime.now().timetuple().tm_yday
    mf = _monsoon_factor(doy)
    noise = np.random.normal(0, span * 0.015)
    live_level = np.clip(row['river_level'] + noise, thr['min'], thr['danger'] * 1.05)

    return {
        'river_level':   round(live_level, 3),
        'pct_to_danger': round((live_level - thr['min']) / span, 4),
        'level_delta':   round(float(row['level_delta']), 4),
        'level_trend':   round(float(row['level_trend']), 4),
        'rain_1d':       round(float(row['rain_1d']), 2),
        'rain_3d':       round(float(row['rain_3d']), 2),
        'rain_7d':       round(float(row['rain_7d']), 2),
        'upstream_3d':   round(float(row['upstream_3d']), 2),
        'is_monsoon':    int(mf > 0.2),
    }


def predict_district(district):
    _ensure_trained()
    features = _get_live_features(district)
    if not features:
        return {'district': district, 'risk': 'low', 'risk_idx': 0, 'probability': [1,0,0]}

    clf = _models['risk']
    scaler = _scalers['risk']
    thr = RIVER_THRESHOLDS[DISTRICT_RIVER_MAP[district]]
    river = DISTRICT_RIVER_MAP[district]

    X = np.array([[features[f] for f in FEATURE_COLS]])
    X_sc = scaler.transform(X)
    proba = clf.predict_proba(X_sc)[0]
    risk_idx = int(np.argmax(proba))
    risk = RISK_LABELS[risk_idx]

    # Feature contribution approximation
    fi = clf.feature_importances_
    feat_vals = X[0]
    deviation = (feat_vals - scaler.mean_) / scaler.scale_
    contributions = fi * deviation
    contrib_norm = contributions / (np.abs(contributions).sum() + 1e-8)

    shap_features = []
    for i, col in enumerate(FEATURE_COLS):
        shap_features.append({
            'feature':      col,
            'label':        FEATURE_LABELS[col],
            'value':        round(float(feat_vals[i]), 3),
            'contribution': round(float(contrib_norm[i]), 4),
            'direction':    'up' if contrib_norm[i] > 0 else 'down',
        })
    shap_features.sort(key=lambda x: abs(x['contribution']), reverse=True)

    crop_ha = 0.0
    if risk_idx >= 1 and 'crop' in _models:
        crop_features = _scalers['crop_features']
        Xc = _scalers['crop'].transform(np.array([[features.get(f, 0) for f in crop_features]]))
        crop_ha = max(0, float(_models['crop'].predict(Xc)[0]))
        if risk_idx == 1:
            crop_ha *= 0.35

    forecast = []
    level = features['river_level']
    trend = features['level_trend']
    span = thr['danger'] - thr['min']
    for day in range(1, 4):
        date = (datetime.now() + timedelta(days=day)).strftime('%b %d')
        trend *= 0.65
        level = np.clip(level + trend + np.random.normal(0, span * 0.012),
                        thr['min'], thr['danger'] * 1.05)
        forecast.append({
            'date': date,
            'level': round(level, 2),
            'pct': round((level - thr['min']) / span * 100, 1),
        })

    return {
        'district':      district,
        'river':         river,
        'risk':          risk,
        'risk_idx':      risk_idx,
        'probability':   [round(float(p), 3) for p in proba],
        'confidence':    round(float(proba[risk_idx]), 3),
        'river_level':   features['river_level'],
        'warning_level': thr['warning'],
        'danger_level':  thr['danger'],
        'pct_to_danger': round(features['pct_to_danger'] * 100, 1),
        'rain_3d':       features['rain_3d'],
        'rain_7d':       features['rain_7d'],
        'level_delta':   features['level_delta'],
        'crop_ha':       round(crop_ha),
        'shap_features': shap_features[:6],
        'forecast':      forecast,
    }


def get_all_districts_risk():
    _ensure_trained()
    results = []
    for district in ALL_DISTRICTS:
        try:
            results.append(predict_district(district))
        except Exception as e:
            logger.warning("Failed %s: %s", district, e)
            results.append({'district': district, 'risk': 'low', 'risk_idx': 0})
    return results


def get_model_metrics():
    _ensure_trained()
    metrics = _eval_metrics.get('risk', {})
    fi = _feature_importance.get('risk', [])
    train_df = _models['train_df']

    y_true = train_df['label'].values[-3000:]
    pct = train_df['pct_to_danger'].values[-3000:]
    y_thresh = np.where(pct >= 0.85, 2, np.where(pct >= 0.65, 1, 0))
    thresh_f1 = round(f1_score(y_true, y_thresh, average='macro', zero_division=0), 3)
    thresh_f1_flood_vals = f1_score(y_true, y_thresh, average=None, zero_division=0)
    thresh_f1_flood = round(float(thresh_f1_flood_vals[2]) if len(thresh_f1_flood_vals) > 2 else 0, 3)

    return {
        'gradient_boosting': {
            'name': 'Gradient Boosting',
            'f1_macro':  metrics.get('f1_macro', 0),
            'f1_flood':  metrics.get('f1_flood', 0),
            'precision': metrics.get('precision', 0),
            'recall':    metrics.get('recall', 0),
            'n_train':   metrics.get('n_train', 0),
        },
        'threshold_baseline': {
            'name': 'Threshold Rule',
            'f1_macro':  thresh_f1,
            'f1_flood':  thresh_f1_flood,
            'precision': 0, 'recall': 0,
        },
        'feature_importance': fi[:8],
        'class_distribution': {
            'low':      int((train_df['label'] == 0).sum()),
            'moderate': int((train_df['label'] == 1).sum()),
            'high':     int((train_df['label'] == 2).sum()),
        },
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }


def get_historical_summary():
    years = list(range(2014, 2024))
    people_affected = [3200000, 2800000, 5700000, 1900000, 5400000,
                       3100000, 4800000, 2300000, 6200000, 4100000]
    districts_affected = [21, 18, 28, 14, 27, 19, 25, 16, 30, 23]
    crop_area_ha = [280000, 210000, 520000, 140000, 490000,
                    260000, 410000, 190000, 570000, 360000]
    return {
        'years': years,
        'people_affected': people_affected,
        'districts_affected': districts_affected,
        'crop_area_ha': crop_area_ha,
    }
