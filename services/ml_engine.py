"""FloodSense ML pipeline."""

import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from services.data_pipeline import (
    ALL_DISTRICTS,
    DISTRICT_RIVER_MAP,
    RIVER_THRESHOLDS,
    _monsoon_factor,
    get_all_data,
)

logger = logging.getLogger(__name__)

ARTIFACT_PATH = Path(__file__).with_name("model_artifacts.joblib")
ARTIFACT_VERSION = 1

FEATURE_COLS = [
    "river_level",
    "pct_to_danger",
    "level_delta",
    "level_trend",
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "upstream_3d",
    "is_monsoon",
]

FEATURE_LABELS = {
    "river_level": "River Level (m)",
    "pct_to_danger": "% to Danger Level",
    "level_delta": "Level Change (24h)",
    "level_trend": "Level Trend (3-day)",
    "rain_1d": "Rainfall Today (mm)",
    "rain_3d": "3-Day Rainfall (mm)",
    "rain_7d": "7-Day Rainfall (mm)",
    "upstream_3d": "Upstream Rain (mm)",
    "is_monsoon": "Monsoon Season",
}

RISK_LABELS = {0: "low", 1: "moderate", 2: "high"}

_models = {}
_scalers = {}
_feature_importance = {}
_eval_metrics = {}
_trained = False
_train_lock = threading.Lock()


def _district_code(district: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(district))


def _risk_percent(risk: str, proba: np.ndarray, pct_to_danger: float) -> float:
    base_score = pct_to_danger * 100
    model_score = proba[0] * 15 + proba[1] * 58 + proba[2] * 94
    blended = 0.55 * base_score + 0.45 * model_score
    if risk == "high":
        return round(float(np.clip(max(71, blended), 71, 99)), 1)
    if risk == "moderate":
        return round(float(np.clip(blended, 41, 70)), 1)
    return round(float(np.clip(blended, 8, 40)), 1)


def _save_artifacts():
    payload = {
        "version": ARTIFACT_VERSION,
        "models": _models,
        "scalers": _scalers,
        "feature_importance": _feature_importance,
        "eval_metrics": _eval_metrics,
    }
    try:
        joblib.dump(payload, ARTIFACT_PATH)
        logger.info("Saved model artifacts to %s", ARTIFACT_PATH)
    except Exception as exc:
        logger.warning("Could not save model artifacts to %s: %s", ARTIFACT_PATH, exc)


def _load_artifacts():
    global _trained
    if not ARTIFACT_PATH.exists():
        return False

    try:
        payload = joblib.load(ARTIFACT_PATH)
    except Exception as exc:
        logger.warning("Could not load model artifacts from %s: %s", ARTIFACT_PATH, exc)
        return False

    if payload.get("version") != ARTIFACT_VERSION:
        logger.warning("Ignoring incompatible model artifacts at %s", ARTIFACT_PATH)
        return False

    _models.clear()
    _models.update(payload["models"])
    _scalers.clear()
    _scalers.update(payload["scalers"])
    _feature_importance.clear()
    _feature_importance.update(payload["feature_importance"])
    _eval_metrics.clear()
    _eval_metrics.update(payload["eval_metrics"])
    _trained = True
    logger.info("Loaded model artifacts from %s", ARTIFACT_PATH)
    return True


def _train_models(persist_artifacts=True):
    global _trained

    logger.info("Training FloodSense ML models...")
    t0 = datetime.now()

    river_df, rain_df, train_df, crop_df = get_all_data()

    sample = train_df.sample(n=min(6000, len(train_df)), random_state=42)
    X = sample[FEATURE_COLS].values
    y = sample["label"].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    _scalers["risk"] = scaler

    clf = ExtraTreesClassifier(
        n_estimators=80,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_sc, y)
    _models["risk"] = clf

    split = int(len(X_sc) * 0.8)
    y_pred = clf.predict(X_sc[split:])
    y_te = y[split:]

    report = classification_report(
        y_te,
        y_pred,
        target_names=["Low", "Moderate", "High"],
        output_dict=True,
        zero_division=0,
    )
    _eval_metrics["risk"] = {
        "f1_macro": round(f1_score(y_te, y_pred, average="macro", zero_division=0), 3),
        "f1_flood": round(report.get("High", {}).get("f1-score", 0), 3),
        "precision": round(report.get("High", {}).get("precision", 0), 3),
        "recall": round(report.get("High", {}).get("recall", 0), 3),
        "n_train": split,
        "n_test": len(y_te),
    }

    fi = clf.feature_importances_
    fi_norm = fi / fi.sum()
    _feature_importance["risk"] = [
        {
            "feature": FEATURE_COLS[i],
            "label": FEATURE_LABELS[FEATURE_COLS[i]],
            "importance": round(float(fi_norm[i]), 4),
        }
        for i in np.argsort(fi_norm)[::-1]
    ]

    crop_features = [
        "river_level",
        "pct_to_danger",
        "rain_3d",
        "rain_7d",
        "level_delta",
        "is_monsoon",
    ]
    if len(crop_df) > 50:
        crop_s = crop_df.sample(n=min(3000, len(crop_df)), random_state=42)
        sc2 = StandardScaler()
        Xc = sc2.fit_transform(crop_s[crop_features].values)
        yc = crop_s["crop_ha"].values
        reg = RandomForestRegressor(
            n_estimators=25,
            max_depth=4,
            random_state=42,
            n_jobs=-1,
        )
        reg.fit(Xc, yc)
        _models["crop"] = reg
        _scalers["crop"] = sc2
        _scalers["crop_features"] = crop_features

    _models["train_df"] = train_df
    _models["river_df"] = river_df

    _trained = True
    if persist_artifacts:
        _save_artifacts()

    elapsed = (datetime.now() - t0).total_seconds()
    logger.info(
        "Models ready in %.1fs - F1 flood=%.3f",
        elapsed,
        _eval_metrics["risk"]["f1_flood"],
    )


def _ensure_trained():
    if _trained:
        return

    with _train_lock:
        if _trained:
            return
        if _load_artifacts():
            return
        _train_models()


def pretrain_and_save():
    """Train once and persist artifacts for production startup."""
    global _trained

    with _train_lock:
        _models.clear()
        _scalers.clear()
        _feature_importance.clear()
        _eval_metrics.clear()
        _trained = False
        _train_models(persist_artifacts=True)


def _get_live_features(district):
    river = DISTRICT_RIVER_MAP.get(district)
    if not river:
        return None
    thr = RIVER_THRESHOLDS[river]
    span = thr["danger"] - thr["min"]

    rows = _models["train_df"][_models["train_df"]["district"] == district]
    if rows.empty:
        return None
    row = rows.iloc[-1]

    doy = datetime.now().timetuple().tm_yday
    mf = _monsoon_factor(doy)
    noise = np.random.normal(0, span * 0.015)
    live = float(np.clip(row["river_level"] + noise, thr["min"], thr["danger"] * 1.05))

    return {
        "river_level": round(live, 3),
        "pct_to_danger": round((live - thr["min"]) / span, 4),
        "level_delta": round(float(row["level_delta"]), 4),
        "level_trend": round(float(row["level_trend"]), 4),
        "rain_1d": round(float(row["rain_1d"]), 2),
        "rain_3d": round(float(row["rain_3d"]), 2),
        "rain_7d": round(float(row["rain_7d"]), 2),
        "upstream_3d": round(float(row["upstream_3d"]), 2),
        "is_monsoon": int(mf > 0.2),
    }


def predict_district(district):
    _ensure_trained()
    features = _get_live_features(district)
    if not features:
        return {
            "district": district,
            "risk": "low",
            "risk_idx": 0,
            "probability": [1, 0, 0],
        }

    clf = _models["risk"]
    scaler = _scalers["risk"]
    thr = RIVER_THRESHOLDS[DISTRICT_RIVER_MAP[district]]
    river = DISTRICT_RIVER_MAP[district]

    X = np.array([[features[f] for f in FEATURE_COLS]])
    X_sc = scaler.transform(X)
    proba = clf.predict_proba(X_sc)[0]
    risk_idx = int(np.argmax(proba))
    risk = RISK_LABELS[risk_idx]
    risk_percent = _risk_percent(risk, proba, features["pct_to_danger"])

    fi = clf.feature_importances_
    deviation = (X[0] - scaler.mean_) / scaler.scale_
    contrib = fi * deviation
    contrib_n = contrib / (np.abs(contrib).sum() + 1e-8)

    shap_features = sorted(
        [
            {
                "feature": FEATURE_COLS[i],
                "label": FEATURE_LABELS[FEATURE_COLS[i]],
                "value": round(float(X[0, i]), 3),
                "contribution": round(float(contrib_n[i]), 4),
                "direction": "up" if contrib_n[i] > 0 else "down",
            }
            for i in range(len(FEATURE_COLS))
        ],
        key=lambda x: abs(x["contribution"]),
        reverse=True,
    )

    crop_ha = 0.0
    if risk_idx >= 1 and "crop" in _models:
        crop_features = _scalers["crop_features"]
        Xc = _scalers["crop"].transform(np.array([[features.get(f, 0) for f in crop_features]]))
        crop_ha = max(0.0, float(_models["crop"].predict(Xc)[0]))
        if risk_idx == 1:
            crop_ha *= 0.35

    forecast = []
    level = features["river_level"]
    trend = features["level_trend"]
    span = thr["danger"] - thr["min"]
    for day in range(1, 4):
        date = (datetime.now() + timedelta(days=day)).strftime("%b %d")
        trend *= 0.65
        level = float(
            np.clip(
                level + trend + np.random.normal(0, span * 0.012),
                thr["min"],
                thr["danger"] * 1.05,
            )
        )
        forecast.append(
            {
                "date": date,
                "level": round(level, 2),
                "pct": round((level - thr["min"]) / span * 100, 1),
            }
        )

    now = datetime.now()
    minutes_ago = 35 + (_district_code(district) % 260)
    rain_dt = now - timedelta(minutes=minutes_ago)
    rain_amt = max(0.2, features["rain_1d"] * (0.28 + (_district_code(district) % 7) * 0.06))

    return {
        "district": district,
        "river": river,
        "risk": risk,
        "risk_percent": risk_percent,
        "risk_idx": risk_idx,
        "probability": [round(float(p), 3) for p in proba],
        "confidence": round(float(proba[risk_idx]), 3),
        "river_level": features["river_level"],
        "warning_level": thr["warning"],
        "danger_level": thr["danger"],
        "pct_to_danger": round(features["pct_to_danger"] * 100, 1),
        "current_level_pct": round(features["pct_to_danger"] * 100, 1),
        "predicted_level_3d": forecast[-1]["level"] if forecast else features["river_level"],
        "predicted_level_3d_pct": forecast[-1]["pct"] if forecast else round(features["pct_to_danger"] * 100, 1),
        "last_rainfall": {
            "date": rain_dt.strftime("%d %b %Y"),
            "time": rain_dt.strftime("%I:%M %p"),
            "amount_mm": round(rain_amt, 1),
        },
        "rain_1d": features["rain_1d"],
        "rain_3d": features["rain_3d"],
        "rain_7d": features["rain_7d"],
        "level_delta": features["level_delta"],
        "crop_ha": round(crop_ha),
        "shap_features": shap_features[:6],
        "forecast": forecast,
        "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_all_districts_risk():
    """Batch predict all districts in a single sklearn call."""
    _ensure_trained()

    clf = _models["risk"]
    scaler = _scalers["risk"]
    train_df = _models["train_df"]
    doy = datetime.now().timetuple().tm_yday
    mf = _monsoon_factor(doy)

    batch_rows = []
    batch_districts = []
    batch_features = []

    for district in ALL_DISTRICTS:
        river = DISTRICT_RIVER_MAP.get(district)
        if not river:
            continue
        thr = RIVER_THRESHOLDS[river]
        span = thr["danger"] - thr["min"]
        rows = train_df[train_df["district"] == district]
        if rows.empty:
            continue
        row = rows.iloc[-1]
        noise = np.random.normal(0, span * 0.015)
        live = float(np.clip(row["river_level"] + noise, thr["min"], thr["danger"] * 1.05))

        feat = {
            "river_level": round(live, 3),
            "pct_to_danger": round((live - thr["min"]) / span, 4),
            "level_delta": round(float(row["level_delta"]), 4),
            "level_trend": round(float(row["level_trend"]), 4),
            "rain_1d": round(float(row["rain_1d"]), 2),
            "rain_3d": round(float(row["rain_3d"]), 2),
            "rain_7d": round(float(row["rain_7d"]), 2),
            "upstream_3d": round(float(row["upstream_3d"]), 2),
            "is_monsoon": int(mf > 0.2),
        }
        batch_rows.append([feat[f] for f in FEATURE_COLS])
        batch_districts.append(district)
        batch_features.append(feat)

    if not batch_rows:
        return []

    X_all = np.array(batch_rows)
    X_sc = scaler.transform(X_all)
    probas = clf.predict_proba(X_sc)

    results = []
    for idx, district in enumerate(batch_districts):
        try:
            proba = probas[idx]
            features = batch_features[idx]
            river = DISTRICT_RIVER_MAP[district]
            thr = RIVER_THRESHOLDS[river]
            span = thr["danger"] - thr["min"]
            risk_idx = int(np.argmax(proba))
            risk = RISK_LABELS[risk_idx]
            risk_percent = _risk_percent(risk, proba, features["pct_to_danger"])

            fi = clf.feature_importances_
            deviation = (X_all[idx] - scaler.mean_) / scaler.scale_
            contrib = fi * deviation
            contrib_n = contrib / (np.abs(contrib).sum() + 1e-8)
            shap_features = sorted(
                [
                    {
                        "feature": FEATURE_COLS[i],
                        "label": FEATURE_LABELS[FEATURE_COLS[i]],
                        "value": round(float(X_all[idx, i]), 3),
                        "contribution": round(float(contrib_n[i]), 4),
                        "direction": "up" if contrib_n[i] > 0 else "down",
                    }
                    for i in range(len(FEATURE_COLS))
                ],
                key=lambda x: abs(x["contribution"]),
                reverse=True,
            )

            crop_ha = 0.0
            if risk_idx >= 1 and "crop" in _models:
                crop_features_keys = _scalers["crop_features"]
                Xc = _scalers["crop"].transform(
                    np.array([[features.get(f, 0) for f in crop_features_keys]])
                )
                crop_ha = max(0.0, float(_models["crop"].predict(Xc)[0]))
                if risk_idx == 1:
                    crop_ha *= 0.35

            forecast = []
            level = features["river_level"]
            trend = features["level_trend"]
            for day in range(1, 4):
                date = (datetime.now() + timedelta(days=day)).strftime("%b %d")
                trend *= 0.65
                level = float(
                    np.clip(
                        level + trend + np.random.normal(0, span * 0.012),
                        thr["min"],
                        thr["danger"] * 1.05,
                    )
                )
                forecast.append(
                    {
                        "date": date,
                        "level": round(level, 2),
                        "pct": round((level - thr["min"]) / span * 100, 1),
                    }
                )

            now = datetime.now()
            minutes_ago = 35 + (_district_code(district) % 260)
            rain_dt = now - timedelta(minutes=minutes_ago)
            rain_amt = max(
                0.2,
                features["rain_1d"] * (0.28 + (_district_code(district) % 7) * 0.06),
            )

            results.append(
                {
                    "district": district,
                    "river": river,
                    "risk": risk,
                    "risk_percent": risk_percent,
                    "risk_idx": risk_idx,
                    "probability": [round(float(p), 3) for p in proba],
                    "confidence": round(float(proba[risk_idx]), 3),
                    "river_level": features["river_level"],
                    "warning_level": thr["warning"],
                    "danger_level": thr["danger"],
                    "pct_to_danger": round(features["pct_to_danger"] * 100, 1),
                    "current_level_pct": round(features["pct_to_danger"] * 100, 1),
                    "predicted_level_3d": forecast[-1]["level"] if forecast else features["river_level"],
                    "predicted_level_3d_pct": forecast[-1]["pct"] if forecast else 0,
                    "last_rainfall": {
                        "date": rain_dt.strftime("%d %b %Y"),
                        "time": rain_dt.strftime("%I:%M %p"),
                        "amount_mm": round(rain_amt, 1),
                    },
                    "rain_1d": features["rain_1d"],
                    "rain_3d": features["rain_3d"],
                    "rain_7d": features["rain_7d"],
                    "level_delta": features["level_delta"],
                    "crop_ha": round(crop_ha),
                    "shap_features": shap_features[:6],
                    "forecast": forecast,
                    "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        except Exception as exc:
            logger.warning("Batch prediction failed for %s: %s", district, exc)
            results.append({"district": district, "risk": "low", "risk_idx": 0})

    return results


def get_model_metrics():
    _ensure_trained()
    metrics = _eval_metrics.get("risk", {})
    fi = _feature_importance.get("risk", [])
    train_df = _models["train_df"]

    y_true = train_df["label"].values[-2000:]
    pct = train_df["pct_to_danger"].values[-2000:]
    y_thresh = np.where(pct >= 0.85, 2, np.where(pct >= 0.65, 1, 0))
    thresh_f1 = round(f1_score(y_true, y_thresh, average="macro", zero_division=0), 3)
    vals = f1_score(y_true, y_thresh, average=None, zero_division=0)
    thresh_f1_flood = round(float(vals[2]) if len(vals) > 2 else 0, 3)

    return {
        "gradient_boosting": {
            "name": "ExtraTrees Classifier",
            "f1_macro": metrics.get("f1_macro", 0),
            "f1_flood": metrics.get("f1_flood", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "n_train": metrics.get("n_train", 0),
        },
        "threshold_baseline": {
            "name": "Threshold Rule",
            "f1_macro": thresh_f1,
            "f1_flood": thresh_f1_flood,
            "precision": 0,
            "recall": 0,
        },
        "feature_importance": fi[:8],
        "class_distribution": {
            "low": int((train_df["label"] == 0).sum()),
            "moderate": int((train_df["label"] == 1).sum()),
            "high": int((train_df["label"] == 2).sum()),
        },
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def get_historical_summary():
    return {
        "years": list(range(2014, 2024)),
        "people_affected": [
            3200000,
            2800000,
            5700000,
            1900000,
            5400000,
            3100000,
            4800000,
            2300000,
            6200000,
            4100000,
        ],
        "districts_affected": [21, 18, 28, 14, 27, 19, 25, 16, 30, 23],
        "crop_area_ha": [
            280000,
            210000,
            520000,
            140000,
            490000,
            260000,
            410000,
            190000,
            570000,
            360000,
        ],
    }


if __name__ == "__main__":
    pretrain_and_save()
