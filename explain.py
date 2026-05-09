"""
Per-prediction SHAP explanations for v3/v4 model.
Cached TreeExplainer — first call ~2s, subsequent calls <10ms.
v4 wraps a StackingClassifier, so we drill down to the inner XGBoost.
"""
import os
import numpy as np
import pandas as pd
from functools import lru_cache

from trainer import ALL_FEATURES, add_features, clip_features, CLIP


def _find_xgb(estimator):
    """Drill into a calibrated/stacked estimator until we find an XGBoost tree."""
    # Bare XGBoost (or any tree with feature_importances_ + DMatrix-compatible API)
    if hasattr(estimator, "get_booster"):
        return estimator
    # StackingClassifier — find the first XGB base learner
    if hasattr(estimator, "estimators_"):
        for base in estimator.estimators_:
            xgb = _find_xgb(base)
            if xgb is not None:
                return xgb
    return None


@lru_cache(maxsize=1)
def _load_explainer():
    import shap, joblib
    # Prefer v4, fall back to v3
    pkl = "stress_model_v4.pkl" if os.path.exists("stress_model_v4.pkl") else "stress_model_v3.pkl"
    model = joblib.load(pkl)

    # Drill into CalibratedClassifierCV → first fold → its inner estimator
    inner = model.calibrated_classifiers_[0].estimator if hasattr(model, "calibrated_classifiers_") else model
    xgb = _find_xgb(inner)
    if xgb is None:
        raise RuntimeError(f"Could not locate an XGBoost tree inside {pkl} for SHAP")

    explainer = shap.TreeExplainer(xgb)
    return model, explainer


def _build_row(sample: dict) -> pd.DataFrame:
    defaults = {
        "study_hours_per_day": 5.5, "sleep_hours_per_day": 7.0,
        "social_hours_per_day": 1.5, "physical_activity_hours_per_day": 1.0,
        "gpa_norm": 0.7, "screen_time_hours": 2.0, "extracurricular_hours": 1.0,
    }
    row = {k: float(sample.get(k, defaults.get(k, 0.0))) for k in defaults}
    df = pd.DataFrame([row])
    # Clip base features
    for col, (lo, hi) in CLIP.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    # Engineer features
    df = add_features(df)
    return df[ALL_FEATURES]


def explain_prediction(sample: dict, top_n: int = 5) -> dict:
    model, explainer = _load_explainer()
    X = _build_row(sample)

    shap_vals = explainer.shap_values(X)

    # Handle both old (list per class) and new (3D array) SHAP APIs
    if isinstance(shap_vals, list):
        # Old API: list of [n_samples, n_features] per class
        sv_high = np.array(shap_vals[2])[0]   # toward High class
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        # New API: (n_samples, n_features, n_classes)
        sv_high = shap_vals[0, :, 2]
    else:
        # Fallback: single 2D array, take as-is
        sv_high = np.array(shap_vals)[0] if shap_vals.ndim > 1 else np.array(shap_vals)

    contribs = sorted(
        [
            {
                "feature":   feat,
                "shap":      round(float(s), 4),
                "value":     round(float(X.iloc[0][feat]), 3),
                "direction": "toward_high_risk" if s > 0 else "toward_low_risk",
            }
            for feat, s in zip(ALL_FEATURES, sv_high)
        ],
        key=lambda c: abs(c["shap"]),
        reverse=True,
    )[:top_n]

    proba = model.predict_proba(X)[0]
    pred  = int(model.predict(X)[0])

    return {
        "prediction":    pred,
        "probabilities": {
            "low":    round(float(proba[0]), 4),
            "medium": round(float(proba[1]), 4),
            "high":   round(float(proba[2]), 4),
        },
        "contributions": contribs,
    }
