"""trainer_v4.py - Optuna-tuned XGBoost + LightGBM stacking, calibrated."""
import json, joblib
import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight

from datasets import load_all
from trainer import ALL_FEATURES, add_features, CLIP
from trainer_v3 import impute_features

VERSION = "4.0"
N_TRIALS = 60

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _prep(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = impute_features(df)
    for col, (lo, hi) in CLIP.items():
        df[col] = df[col].clip(lo, hi)
    df = add_features(df)
    X = df[ALL_FEATURES]
    y = df["target_class"].astype(int)
    mask = X.notna().all(axis=1)
    return X[mask], y[mask]


def main():
    print("=" * 60)
    print("  Burnout Model Trainer v4.0 - Optuna + Stacking + Calibration")
    print("=" * 60)

    df = load_all()
    print(f"\nLoaded {len(df):,} rows from {df['source'].nunique()} sources")

    train_df = df[df["source"] != "lifestyle"].copy()
    test_df  = df[df["source"] == "lifestyle"].copy()
    # Keep parity with v3: also exclude lifestyle_100k from training
    train_df = train_df[train_df["source"] != "lifestyle_100k"].copy()

    X_tr, y_tr = _prep(train_df)
    X_te, y_te = _prep(test_df)
    print(f"\nTrain: {len(X_tr):,} | Holdout: {len(X_te):,}")

    sw = compute_sample_weight("balanced", y_tr)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n--- Optuna search ({N_TRIALS} trials) ---")
    trial_count = [0]

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
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
        scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy", n_jobs=-1)
        trial_count[0] += 1
        if trial_count[0] % 10 == 0:
            print(f"  Trial {trial_count[0]}/{N_TRIALS}: best={trial.study.best_value:.4f}")
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_params
    print(f"\nBest CV accuracy: {study.best_value:.4f}")
    print(f"Best params: {best}")

    print("\n--- Building stacking ensemble ---")
    xgb = XGBClassifier(
        **best, objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
    )
    lgb = LGBMClassifier(
        n_estimators=400, max_depth=-1, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, reg_alpha=0.2, reg_lambda=0.5,
        objective="multiclass", num_class=3,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    stacker = StackingClassifier(
        estimators=[("xgb", xgb), ("lgb", lgb)],
        final_estimator=LogisticRegression(max_iter=500),  # multinomial is default in sklearn 1.5+
        cv=3, n_jobs=-1,
    )

    print("Fitting stacking ensemble...")
    stacker.fit(X_tr, y_tr)

    print("Wrapping in CalibratedClassifierCV (isotonic, cv=3)...")
    model = CalibratedClassifierCV(stacker, method="isotonic", cv=3)
    model.fit(X_tr, y_tr)

    print("\n=== HONESTY HOLDOUT (lifestyle) ===")
    y_pred = model.predict(X_te)
    proba  = model.predict_proba(X_te)
    acc    = accuracy_score(y_te, y_pred)
    ll     = log_loss(y_te, proba, labels=[0, 1, 2])
    print(f"Accuracy: {acc:.4f}  |  Log-loss: {ll:.4f}")
    print(classification_report(y_te, y_pred, target_names=["Low","Medium","High"]))

    joblib.dump(model, "stress_model_v4.pkl")
    with open("model_meta_v4.json", "w") as f:
        json.dump({
            "version":                  VERSION,
            "features":                 ALL_FEATURES,
            "honesty_holdout_accuracy": round(float(acc), 4),
            "honesty_holdout_logloss":  round(float(ll), 4),
            "n_train":                  int(len(X_tr)),
            "n_test":                   int(len(X_te)),
            "best_xgb_params":          best,
            "best_xgb_cv_accuracy":     round(float(study.best_value), 4),
            "ensemble":                 "Stacking[XGB+LGBM]+LogReg, isotonic-calibrated",
            "optuna_trials":            N_TRIALS,
        }, f, indent=2)
    print("\n[OK] stress_model_v4.pkl + model_meta_v4.json saved")

if __name__ == "__main__":
    main()
