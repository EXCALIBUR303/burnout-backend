"""v3 trainer: multi-source, calibrated, honestly held out."""
import json, joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from datasets import load_all
from trainer import ALL_FEATURES, add_features, clip_features, CLIP

VERSION = "3.0"

def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing base features with global median (per-column)."""
    df = df.copy()
    base = ['study_hours_per_day','sleep_hours_per_day','social_hours_per_day',
            'physical_activity_hours_per_day','gpa_norm','screen_time_hours','extracurricular_hours']
    for col in base:
        if col in df.columns:
            median = df[col].median()
            # Use 7.5 for sleep, 1.5 for social, etc. as domain sensible fallbacks if median is NaN
            fallbacks = {
                'study_hours_per_day': 5.5, 'sleep_hours_per_day': 7.0,
                'social_hours_per_day': 1.5, 'physical_activity_hours_per_day': 1.0,
                'gpa_norm': 0.7, 'screen_time_hours': 2.0, 'extracurricular_hours': 1.0,
            }
            df[col] = df[col].fillna(median if pd.notna(median) else fallbacks.get(col, 0.0))
    return df

def main():
    print("=" * 56)
    print("  Burnout Model Trainer v3.0")
    print("=" * 56)

    df = load_all()
    print(f"\nLoaded {len(df):,} rows from {df['source'].nunique()} sources")
    print(df.groupby("source").size().to_string())

    # Honesty split: hold out lifestyle (rule-based labels) for evaluation
    train_df = df[df["source"] != "lifestyle"].copy()
    test_df  = df[df["source"] == "lifestyle"].copy()

    # Exclude lifestyle_100k from training: it has near-zero feature-target
    # correlations (the labels appear synthetically generated independently of
    # features), so including it introduces 100k rows of label noise that
    # overwhelms the signal from the smaller real-world sources and causes
    # the model to predict only Low stress everywhere.
    train_df = train_df[train_df["source"] != "lifestyle_100k"].copy()

    print(f"\nTrain: {len(train_df):,} rows | Holdout (lifestyle): {len(test_df):,} rows")

    # Impute missing values, clip, engineer features
    for split in [train_df, test_df]:
        split = impute_features(split)  # in-place on copy

    train_df = impute_features(train_df)
    test_df  = impute_features(test_df)

    for col, (lo, hi) in CLIP.items():
        train_df[col] = train_df[col].clip(lo, hi)
        test_df[col]  = test_df[col].clip(lo, hi)

    train_df = add_features(train_df)
    test_df  = add_features(test_df)

    X_tr = train_df[ALL_FEATURES]
    y_tr = train_df["target_class"].astype(int)
    X_te = test_df[ALL_FEATURES]
    y_te = test_df["target_class"].astype(int)

    # Drop rows where any feature is still NaN after imputation
    mask_tr = X_tr.notna().all(axis=1)
    mask_te = X_te.notna().all(axis=1)
    X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]
    X_te, y_te = X_te[mask_te], y_te[mask_te]
    print(f"\nAfter NaN drop: train={len(X_tr):,}, test={len(X_te):,}")

    print("\nClass distribution (train):")
    print(y_tr.value_counts().rename({0:"Low",1:"Medium",2:"High"}).to_string())

    # Base XGBoost (lighter than v2 — more data compensates)
    base = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=6,
        reg_alpha=0.2, reg_lambda=2.0,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
    )

    print("\nFitting base XGBoost...")
    sw = compute_sample_weight("balanced", y_tr)
    base.fit(X_tr, y_tr, sample_weight=sw)

    print("Wrapping in CalibratedClassifierCV (isotonic, cv=5)...")
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    # Pass sample_weight so calibration folds also respect class balance
    model.fit(X_tr, y_tr, sample_weight=sw)

    print("\n=== HONESTY HOLDOUT (lifestyle — rule-based labels) ===")
    y_pred = model.predict(X_te)
    proba  = model.predict_proba(X_te)
    acc    = accuracy_score(y_te, y_pred)
    ll     = log_loss(y_te, proba, labels=[0, 1, 2])
    print(f"Accuracy: {acc:.4f} | Log-loss: {ll:.4f}")
    print(classification_report(y_te, y_pred, target_names=["Low","Medium","High"]))

    if acc > 0.95:
        print("WARNING: Holdout accuracy still >0.95 — possible residual leakage from 100k dataset.")
    elif acc < 0.40:
        print("WARNING: Holdout accuracy <0.40 — model weaker than majority-class baseline.")
    else:
        print("OK: Holdout accuracy in realistic range (real generalisation, not memorisation).")

    # Average feature importances across calibration folds
    fold_imps = [c.estimator.feature_importances_ for c in model.calibrated_classifiers_]
    avg_imp = np.mean(fold_imps, axis=0)
    imp_series = pd.Series(avg_imp, index=ALL_FEATURES).sort_values(ascending=False)
    print("\nFeature importances (averaged across CV folds):")
    for feat, imp in imp_series.items():
        bar = "#" * int(imp * 50)
        print(f"  {feat:<32} {bar} {imp:.4f}")

    joblib.dump(model, "stress_model_v3.pkl")
    with open("model_meta_v3.json", "w") as f:
        json.dump({
            "version": VERSION,
            "features": ALL_FEATURES,
            "honesty_holdout_accuracy": round(float(acc), 4),
            "honesty_holdout_logloss":  round(float(ll), 4),
            "n_train": int(len(X_tr)),
            "n_test":  int(len(X_te)),
            "sources_trained": sorted(train_df["source"].unique().tolist()),
            "calibration": "isotonic, cv=5",
        }, f, indent=2)
    print("\n[OK] stress_model_v3.pkl saved")
    print("[OK] model_meta_v3.json saved")

if __name__ == "__main__":
    main()
