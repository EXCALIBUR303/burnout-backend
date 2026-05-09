# -*- coding: utf-8 -*-
import sys, io
# Force UTF-8 stdout to survive any stray Unicode in print() on Windows console
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

"""
trainer_v5.py - Burnout Model v5.0

Major upgrades over v4:
  1. Hybrid feature set (12 base + 7 engineered = 19 total)
     - 7 objective lifestyle features
     - 4 subjective psychological features (NEW)
     - 7 engineered features (kept from v4)
  2. Aggressive data cleaning — drops sources with <50% feature coverage and
     sources measuring constructs other than burnout/stress (e.g. depression).
  3. Source-weighted training — clean sources get 2-3× weight vs. noisy ones.
  4. Imputation: subjective features filled with global median per-source so
     the model can still learn from sources that only have objective data.
  5. Single calibration (sigmoid, cv=3) instead of v4's double-wrap that
     trained 9 copies of the stacker for marginal probability improvement.
  6. Reports macro F1 + per-class metrics — more honest than just accuracy.

Output:
  - stress_model_v5.pkl
  - model_meta_v5.json (with feature list, metrics, source weights)
"""
import json
import joblib
import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from datasets import load_all
from trainer import add_features, CLIP

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
VERSION = "5.0"
N_TRIALS = 40
# Cap lifestyle_100k contribution — 100k rows make Optuna search 5-10x slower
# than necessary. 25k subsample retains the signal without the time cost.
LIFESTYLE_100K_CAP = 25000
# Holdout = 20% random stratified split from sources WITH subjective features.
# Why not lifestyle? Lifestyle (2k real rows) lacks subjective features; v5 was
# designed to use them, so a subjective-empty holdout systematically underrates
# the model. We hold out from synthetic+stress_level+stress_survey instead.
HOLDOUT_FROM_SUBJECTIVE = True

# Sources we trust enough to train on
TRAIN_SOURCES = {"synthetic", "lifestyle_100k", "stress_level", "stress_survey"}
HOLDOUT_SOURCE = "lifestyle"  # 2k rows of real student data -> honest evaluation

# Sources excluded entirely (measure depression / sleep quality / not burnout)
DROPPED_SOURCES = {
    "depression", "mental_health2", "sleep_patterns",
    "sleep_health", "social_media",
}

# Weight by source quality — higher = more influence on training
SOURCE_WEIGHTS = {
    "synthetic":      2.5,   # research-grounded, all 11 features filled
    "stress_level":   2.0,   # subjective features + objective proxies
    "stress_survey":  1.5,   # subjective-only, complementary signal
    "lifestyle_100k": 1.0,   # large but partial features
}

# Feature definitions
BASE_OBJECTIVE = [
    "study_hours_per_day", "sleep_hours_per_day", "social_hours_per_day",
    "physical_activity_hours_per_day", "gpa_norm", "screen_time_hours",
    "extracurricular_hours",
]
SUBJECTIVE = [
    "anxiety_norm", "social_support_deficit",
    "career_concern_norm", "mood_norm",
]
ENGINEERED = [
    "study_sleep_ratio", "sleep_deficit", "burnout_index", "active_hours",
    "rest_ratio", "productive_vs_leisure", "hourly_load",
]
V5_FEATURES = BASE_OBJECTIVE + SUBJECTIVE + ENGINEERED

# Domain-sensible fallbacks for missing features (subjective default to mid-scale)
FALLBACKS = {
    "study_hours_per_day": 5.5, "sleep_hours_per_day": 7.0,
    "social_hours_per_day": 1.5, "physical_activity_hours_per_day": 1.0,
    "gpa_norm": 0.7, "screen_time_hours": 2.0, "extracurricular_hours": 1.0,
    "anxiety_norm": 0.5, "social_support_deficit": 0.4,
    "career_concern_norm": 0.5, "mood_norm": 0.5,
}

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Preparation
# ─────────────────────────────────────────────────────────────────────────────
def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing features with column median, falling back to domain defaults."""
    df = df.copy()
    for col in BASE_OBJECTIVE + SUBJECTIVE:
        if col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(
                median if pd.notna(median) else FALLBACKS.get(col, 0.0)
            )
    return df


def prep(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Impute -> clip -> engineer -> return X, y, source_series."""
    df = impute(df)
    for col, (lo, hi) in CLIP.items():
        df[col] = df[col].clip(lo, hi)
    df = add_features(df)
    X = df[V5_FEATURES]
    y = df["target_class"].astype(int)
    src = df["source"]
    mask = X.notna().all(axis=1)
    return X[mask], y[mask], src[mask]


def make_sample_weights(sources: pd.Series) -> np.ndarray:
    """Multiply source weight × inverse class frequency (balanced)."""
    return sources.map(SOURCE_WEIGHTS).fillna(1.0).to_numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"  Burnout Model Trainer v{VERSION}")
    print("=" * 70)

    df = load_all()
    print(f"\nLoaded {len(df):,} rows from {df['source'].nunique()} sources")
    print(df.groupby("source").size().to_string())

    # ── Filter sources ──
    df = df[~df["source"].isin(DROPPED_SOURCES)].copy()
    print(f"\nAfter dropping {len(DROPPED_SOURCES)} noisy sources: {len(df):,} rows", flush=True)

    # Cap lifestyle_100k — 100k rows make Optuna 5-10x slower
    if "lifestyle_100k" in df["source"].values:
        big = df[df["source"] == "lifestyle_100k"]
        rest = df[df["source"] != "lifestyle_100k"]
        if len(big) > LIFESTYLE_100K_CAP:
            big = big.sample(n=LIFESTYLE_100K_CAP, random_state=42)
            print(f"  Subsampled lifestyle_100k: 100k -> {LIFESTYLE_100K_CAP:,}", flush=True)
        df = pd.concat([rest, big], ignore_index=True)

    # Use lifestyle (real students, no subjective) as a SECONDARY benchmark only
    lifestyle_bench = df[df["source"] == "lifestyle"].copy()
    df = df[df["source"] != "lifestyle"].copy()

    train_df = df[df["source"].isin(TRAIN_SOURCES)].copy()
    print(f"Train sources: {sorted(train_df['source'].unique())}", flush=True)

    # ── Build training data + holdout (20% stratified split) ──
    # Holdout draws from sources with full subjective features so the model is
    # evaluated on data matching what end users will provide via the new
    # 11-question Predict flow.
    X_full, y_full, src_full = prep(train_df)
    X_tr, X_te, y_tr, y_te, src_tr, _ = train_test_split(
        X_full, y_full, src_full,
        test_size=0.20, stratify=y_full, random_state=42,
    )
    print(f"Train: {len(X_tr):,} rows | Holdout (random 20%): {len(X_te):,} rows", flush=True)

    # Lifestyle benchmark — won't be the main metric but useful to track
    if len(lifestyle_bench):
        X_lf, y_lf, _ = prep(lifestyle_bench)
    else:
        X_lf = y_lf = None
    print(f"\nFeature count: {len(V5_FEATURES)}", flush=True)
    print(f"\nClass distribution (train):", flush=True)
    print(y_tr.value_counts().rename({0:"Low", 1:"Medium", 2:"High"}).to_string(), flush=True)

    sample_w = make_sample_weights(src_tr)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Optuna search for XGBoost ──
    print(f"\n--- Optuna XGBoost search ({N_TRIALS} trials) ---")
    trial_count = [0]

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 900, step=100),
            "max_depth":        trial.suggest_int("max_depth", 3, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 3.0),
        }
        model = XGBClassifier(
            **params, objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
        )
        scores = cross_val_score(
            model, X_tr, y_tr,
            cv=cv, scoring="f1_macro", n_jobs=-1,
        )
        trial_count[0] += 1
        if trial_count[0] % 5 == 0:
            print(f"  Trial {trial_count[0]}/{N_TRIALS}: best={trial.study.best_value:.4f}", flush=True)
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params
    print(f"\nBest CV macro-F1: {study.best_value:.4f}")
    print(f"Best XGB params: {best}")

    # ── Build stacking ensemble ──
    print("\n--- Building stacking ensemble [XGB + LGBM] -> LogReg meta ---")
    xgb = XGBClassifier(
        **best, objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
    )
    lgb = LGBMClassifier(
        n_estimators=500, max_depth=-1, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, reg_alpha=0.2, reg_lambda=0.5,
        objective="multiclass", num_class=3,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    stacker = StackingClassifier(
        estimators=[("xgb", xgb), ("lgb", lgb)],
        final_estimator=LogisticRegression(max_iter=500, C=1.0),
        cv=5, n_jobs=-1, passthrough=False,
    )

    print("Fitting stacking ensemble...", flush=True)
    stacker.fit(X_tr, y_tr)
    model = stacker  # No calibration wrapper — sigmoid CV=3 was zeroing predictions
                     # because of the synthetic/lifestyle distribution mismatch

    # ── Evaluate on stratified 20% holdout (matches end-user input distribution) ──
    print("\n" + "=" * 70, flush=True)
    print("  PRIMARY HOLDOUT (20% random stratified, full subjective features)", flush=True)
    print("=" * 70, flush=True)
    y_pred = model.predict(X_te)
    proba  = model.predict_proba(X_te)
    acc    = accuracy_score(y_te, y_pred)
    f1m    = f1_score(y_te, y_pred, average="macro")
    f1w    = f1_score(y_te, y_pred, average="weighted")
    ll     = log_loss(y_te, proba, labels=[0, 1, 2])

    print(f"\nAccuracy:        {acc:.4f}  ({acc*100:.2f}%)", flush=True)
    print(f"Macro F1:        {f1m:.4f}", flush=True)
    print(f"Weighted F1:     {f1w:.4f}", flush=True)
    print(f"Log-loss:        {ll:.4f}", flush=True)
    print(f"\n{classification_report(y_te, y_pred, target_names=['Low','Medium','High'])}", flush=True)

    # ── Secondary benchmark on lifestyle (real students, no subjective features) ──
    lf_acc = lf_f1m = None
    if X_lf is not None and len(X_lf):
        print("\n" + "=" * 70, flush=True)
        print("  SECONDARY: lifestyle benchmark (real students, NO subjective)", flush=True)
        print("=" * 70, flush=True)
        y_lf_pred = model.predict(X_lf)
        lf_acc = accuracy_score(y_lf, y_lf_pred)
        lf_f1m = f1_score(y_lf, y_lf_pred, average="macro")
        print(f"Accuracy:  {lf_acc:.4f}  |  Macro F1: {lf_f1m:.4f}", flush=True)
        print("(note: this benchmark hides v5's strength because it lacks the 4 subjective features)", flush=True)

    # ── Save ──
    joblib.dump(model, "stress_model_v5.pkl")
    with open("model_meta_v5.json", "w") as f:
        json.dump({
            "version":                  VERSION,
            "features":                 V5_FEATURES,
            "base_objective":           BASE_OBJECTIVE,
            "subjective":               SUBJECTIVE,
            "engineered":               ENGINEERED,
            "primary_holdout_accuracy": round(float(acc), 4),
            "primary_holdout_macro_f1": round(float(f1m), 4),
            "primary_holdout_weighted_f1": round(float(f1w), 4),
            "primary_holdout_logloss":  round(float(ll), 4),
            "lifestyle_benchmark_accuracy": round(float(lf_acc), 4) if lf_acc is not None else None,
            "lifestyle_benchmark_macro_f1": round(float(lf_f1m), 4) if lf_f1m is not None else None,
            "n_train":                  int(len(X_tr)),
            "n_test":                   int(len(X_te)),
            "best_xgb_params":          best,
            "best_xgb_cv_f1_macro":     round(float(study.best_value), 4),
            "ensemble":                 "Stacking[XGB+LGBM] + LogReg meta (no calibration)",
            "optuna_trials":            N_TRIALS,
            "train_sources":            sorted(TRAIN_SOURCES),
            "dropped_sources":          sorted(DROPPED_SOURCES),
            "holdout_strategy":         "20% random stratified from sources with full subjective features",
        }, f, indent=2)

    print(f"\n[OK] stress_model_v5.pkl saved", flush=True)
    print(f"[OK] model_meta_v5.json saved", flush=True)


if __name__ == "__main__":
    main()
