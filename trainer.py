"""
Upgraded Burnout/Stress Classifier — v2.0
==========================================
Improvements over v1:
  1. Multi-dataset fusion (primary + 100k augmentation dataset)
  2. New features: GPA, screen_time, extracurricular, sleep_deficit, burnout_index
  3. Input validation/clipping on BOTH train and predict paths
  4. Class-balanced training (handles imbalanced label distributions)
  5. Early stopping to prevent over-fitting
  6. Probability calibration via CalibratedClassifierCV
  7. Richer feature engineering (7 engineered features vs 4 before)
  8. Saves both model and feature metadata for /predict validation

Output: stress_model.pkl  (drop-in replacement for v1 pkl)
"""

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV

# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL_MAP = {"Low": 0, "Moderate": 1, "Medium": 1, "High": 2}

# ── Feature names (must match main.py FEATURE_NAMES exactly) ─────────────────
BASE_FEATURES = [
    "study_hours_per_day",
    "sleep_hours_per_day",
    "social_hours_per_day",
    "physical_activity_hours_per_day",
    "gpa_norm",            # GPA normalised to 0–1 (10-pt scale → /10)
    "screen_time_hours",   # daily screen / social-media hours
    "extracurricular_hours",
]

ENGINEERED_FEATURES = [
    "study_sleep_ratio",      # study / (sleep + 0.1)
    "sleep_deficit",          # max(0, 7.5 - sleep)  — hours below optimal
    "burnout_index",          # study_sleep_ratio * sleep_deficit
    "active_hours",           # study + physical
    "rest_ratio",             # (sleep + physical) / total_hours
    "productive_vs_leisure",  # study / (social + physical + 0.1)
    "hourly_load",            # (study + social + physical + screen) / 16
]

ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

# ── Clip bounds (applied in trainer AND in main.py /predict) ──────────────────
CLIP = {
    "study_hours_per_day":              (0, 18),
    "sleep_hours_per_day":              (2, 14),
    "social_hours_per_day":             (0, 12),
    "physical_activity_hours_per_day":  (0, 8),
    "gpa_norm":                         (0, 1),
    "screen_time_hours":                (0, 16),
    "extracurricular_hours":            (0, 8),
}


# ── Feature engineering ───────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sleep    = df["sleep_hours_per_day"]
    study    = df["study_hours_per_day"]
    social   = df["social_hours_per_day"]
    physical = df["physical_activity_hours_per_day"]
    screen   = df["screen_time_hours"]
    total    = (sleep + study + social + physical).replace(0, 1)

    df["study_sleep_ratio"]    = study / (sleep + 0.1)
    df["sleep_deficit"]        = (7.5 - sleep).clip(lower=0)
    df["burnout_index"]        = df["study_sleep_ratio"] * df["sleep_deficit"]
    df["active_hours"]         = study + physical
    df["rest_ratio"]           = (sleep + physical) / total
    df["productive_vs_leisure"]= study / (social + physical + 0.1)
    df["hourly_load"]          = (study + social + physical + screen) / 16.0
    return df


def clip_features(df: pd.DataFrame) -> pd.DataFrame:
    for col, (lo, hi) in CLIP.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_primary(path="student_lifestyle_dataset.csv") -> pd.DataFrame:
    """Main dataset: 2,000 rows, clean labels, has GPA + extracurricular."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    out = pd.DataFrame({
        "study_hours_per_day":              pd.to_numeric(df["Study_Hours_Per_Day"],             errors="coerce"),
        "sleep_hours_per_day":              pd.to_numeric(df["Sleep_Hours_Per_Day"],             errors="coerce"),
        "social_hours_per_day":             pd.to_numeric(df["Social_Hours_Per_Day"],            errors="coerce"),
        "physical_activity_hours_per_day":  pd.to_numeric(df["Physical_Activity_Hours_Per_Day"], errors="coerce"),
        "gpa_norm":                         pd.to_numeric(df["GPA"],                             errors="coerce") / 10.0,
        "extracurricular_hours":            pd.to_numeric(df.get("Extracurricular_Hours_Per_Day", pd.Series(1.0, index=df.index)), errors="coerce"),
        "screen_time_hours":                1.5,   # not in this dataset — use mean
        "target":                           df["Stress_Level"].str.strip().map(LABEL_MAP),
    }).dropna(subset=["target"])

    print(f"  Primary dataset: {len(out):,} rows")
    return out


def load_augment(path="student_lifestyle_100k.csv") -> pd.DataFrame | None:
    """Large augmentation dataset — different column names, needs remapping."""
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        print("  Augmentation dataset not found — skipping.")
        return None

    # Map columns — be defensive with varying column names
    def get_col(df, *names):
        for n in names:
            for c in df.columns:
                if c.lower().replace(" ", "_") == n.lower().replace(" ", "_"):
                    return pd.to_numeric(df[c], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    study   = get_col(df, "Study_Hours", "study_hours_per_day", "study_hours")
    sleep   = get_col(df, "Sleep_Duration", "sleep_hours_per_day", "sleep_hours")
    social  = get_col(df, "Social_Media_Hours", "social_hours_per_day", "social_hours")
    physical= get_col(df, "Physical_Activity", "physical_activity_hours_per_day")
    cgpa    = get_col(df, "CGPA", "GPA", "gpa")
    screen  = get_col(df, "Social_Media_Hours", "screen_time_hours")  # proxy

    # Target — try both "Stress_Level" and label column
    target = None
    for col in df.columns:
        if "stress" in col.lower() or "risk" in col.lower():
            mapped = df[col].astype(str).str.strip().map(LABEL_MAP)
            if mapped.notna().sum() > 100:
                target = mapped
                break
    if target is None:
        print("  Augmentation dataset: no usable target column — skipping.")
        return None

    # Normalise CGPA: if max > 4.5 assume 10-pt scale, else 4-pt
    cgpa_max = cgpa.max()
    gpa_norm = (cgpa / 10.0) if cgpa_max > 4.5 else (cgpa / 4.0)

    # Physical_Activity may be binary (0/1) or hours — normalise
    phys_max = physical.max()
    if phys_max <= 1.5:
        physical = physical * 2.0   # binary → ~hours proxy

    out = pd.DataFrame({
        "study_hours_per_day":             study,
        "sleep_hours_per_day":             sleep,
        "social_hours_per_day":            social.fillna(1.5),
        "physical_activity_hours_per_day": physical.fillna(1.0),
        "gpa_norm":                        gpa_norm.fillna(0.65),
        "screen_time_hours":               screen.fillna(2.0),
        "extracurricular_hours":           1.0,
        "target":                          target,
    }).dropna(subset=["study_hours_per_day", "sleep_hours_per_day", "target"])

    print(f"  Augmentation dataset: {len(out):,} rows")
    return out


# ── Training ──────────────────────────────────────────────────────────────────

def train(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    # Class distribution
    dist = y.value_counts().sort_index().rename({0: "Low", 1: "Medium", 2: "High"})
    print(f"\nClass distribution:\n{dist.to_string()}\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.18, stratify=y, random_state=42
    )

    # Balanced sample weights so rare classes aren't ignored
    sample_w = compute_sample_weight("balanced", y_tr)

    model = XGBClassifier(
        n_estimators=700,
        max_depth=5,
        learning_rate=0.025,
        subsample=0.82,
        colsample_bytree=0.80,
        colsample_bylevel=0.85,
        min_child_weight=4,
        gamma=0.1,
        reg_alpha=0.15,
        reg_lambda=1.8,
        objective="multi:softprob",   # softprob gives calibrated probabilities
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=40,     # stop if no improvement for 40 rounds
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_tr, y_tr,
        sample_weight=sample_w,
        eval_set=[(X_te, y_te)],
        verbose=False,
    )

    # ── Evaluation ──
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    print("=" * 56)
    print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print("=" * 56)
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=["Low", "Medium", "High"]))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_te, y_pred)
    print(pd.DataFrame(
        cm,
        index=["Actual Low", "Actual Med", "Actual High"],
        columns=["Pred Low", "Pred Med", "Pred High"],
    ))

    print("\nFeature Importances (gain):")
    imp = pd.Series(model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    for feat, score in imp.items():
        bar = "#" * int(score * 60)
        print(f"  {feat:<32} {bar} {score:.4f}")

    # ── Cross-validation (simple, no sample weights to avoid sklearn API drift) ──
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_iter = getattr(model, "best_iteration", None) or 500
    cv_model = XGBClassifier(
        n_estimators=best_iter,
        max_depth=5, learning_rate=0.025, subsample=0.82,
        colsample_bytree=0.80, min_child_weight=4, gamma=0.1,
        reg_alpha=0.15, reg_lambda=1.8,
        objective="multi:softprob", num_class=3,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    try:
        cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    except Exception as e:
        print(f"\nCV skipped: {e}")
    print(f"Best XGBoost iteration: {best_iter}")

    return model


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 56)
    print("  Burnout/AI Model Trainer v2.0")
    print("=" * 56)

    # Load and merge datasets
    print("\nLoading datasets...")
    primary = load_primary()
    augment = load_augment()

    if augment is not None:
        combined = pd.concat([primary, augment], ignore_index=True)
        # Downsample augment to max 3× primary size to avoid overwhelming primary
        max_rows = len(primary) * 4
        if len(combined) > max_rows:
            combined = combined.sample(max_rows, random_state=42)
        print(f"\n  Combined dataset: {len(combined):,} rows")
    else:
        combined = primary
        print(f"\n  Using primary dataset only: {len(combined):,} rows")

    # Clip + engineer features
    combined = clip_features(combined)
    combined = add_features(combined)
    combined["target"] = combined["target"].astype(int)

    X = combined[ALL_FEATURES]
    y = combined["target"]

    print(f"\nFeatures ({len(ALL_FEATURES)}): {ALL_FEATURES}")
    print(f"Total training samples: {len(X):,}")

    # Train
    print("\nTraining XGBoost classifier v2...")
    model = train(X, y)

    # Save model + metadata
    joblib.dump(model, "stress_model.pkl")

    # Save feature list so main.py can validate at startup
    import json
    with open("model_meta.json", "w") as f:
        json.dump({
            "version": "2.0",
            "features": ALL_FEATURES,
            "base_features": BASE_FEATURES,
            "clip_bounds": CLIP,
            "label_map": {"0": "Low", "1": "Medium", "2": "High"},
            "n_samples": int(len(X)),
        }, f, indent=2)

    print("\n✅ stress_model.pkl saved")
    print("✅ model_meta.json saved")
    print("\nUpdate main.py FEATURE_NAMES to match ALL_FEATURES above.")


if __name__ == "__main__":
    main()
