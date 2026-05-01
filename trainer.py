"""
Trains a 3-class burnout/stress classifier and saves it as stress_model.pkl.

Output labels match main.py: {0: "Low", 1: "Medium", 2: "High"}
Input features match the /predict endpoint:
    study_hours_per_day, sleep_hours_per_day,
    social_hours_per_day, physical_activity_hours_per_day
"""

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_FEATURES = [
    "study_hours_per_day",
    "sleep_hours_per_day",
    "social_hours_per_day",
    "physical_activity_hours_per_day",
]

ALL_FEATURES = BASE_FEATURES + [
    "study_sleep_ratio",
    "active_hours",
    "rest_ratio",
    "productive_vs_leisure",
]

LABEL_MAP = {"Low": 0, "Moderate": 1, "Medium": 1, "High": 2}


# ── Feature engineering ──────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sleep = df["sleep_hours_per_day"]
    study = df["study_hours_per_day"]
    social = df["social_hours_per_day"]
    physical = df["physical_activity_hours_per_day"]
    total = (sleep + study + social + physical).replace(0, 1)

    df["study_sleep_ratio"]       = study / (sleep + 0.1)
    df["active_hours"]            = study + physical
    df["rest_ratio"]              = (sleep + physical) / total
    df["productive_vs_leisure"]   = study / (social + physical + 0.1)
    return df


# ── Dataset loader ───────────────────────────────────────────────────────────

def load_dataset(path="student_lifestyle_dataset.csv") -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    out = pd.DataFrame({
        "study_hours_per_day":             df["Study_Hours_Per_Day"],
        "sleep_hours_per_day":             df["Sleep_Hours_Per_Day"],
        "social_hours_per_day":            df["Social_Hours_Per_Day"],
        "physical_activity_hours_per_day": df["Physical_Activity_Hours_Per_Day"],
        "target":                          df["Stress_Level"].str.strip().map(LABEL_MAP),
    }).dropna(subset=["target"])

    out["study_hours_per_day"]             = out["study_hours_per_day"].clip(0, 18)
    out["sleep_hours_per_day"]             = out["sleep_hours_per_day"].clip(2, 14)
    out["social_hours_per_day"]            = out["social_hours_per_day"].clip(0, 12)
    out["physical_activity_hours_per_day"] = out["physical_activity_hours_per_day"].clip(0, 8)

    out = add_features(out)
    out["target"] = out["target"].astype(int)

    return out[ALL_FEATURES], out["target"]


# ── Training ─────────────────────────────────────────────────────────────────

def train(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.05,
        reg_alpha=0.1,
        reg_lambda=1.5,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluation ──
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 52)
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print("=" * 52 + "\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(
        cm,
        index=["Actual Low", "Actual Medium", "Actual High"],
        columns=["Pred Low", "Pred Medium", "Pred High"],
    ))

    print("\nFeature Importances:")
    imp = pd.Series(model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
    for feat, score in imp.items():
        bar = "#" * int(score * 50)
        print(f"  {feat:<30} {bar} {score:.4f}")

    # ── Cross-validation ──
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    return model


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    print("Loading student_lifestyle_dataset.csv...")
    X, y = load_dataset()

    label_names = {0: "Low", 1: "Medium", 2: "High"}
    dist = y.value_counts().sort_index().rename(label_names)
    print(f"Total samples : {len(X):,}")
    print(f"Class distribution:\n{dist.to_string()}\n")

    print("Training XGBoost classifier...")
    model = train(X, y)

    joblib.dump(model, "stress_model.pkl")
    print("\n[OK] Model saved to stress_model.pkl")


if __name__ == "__main__":
    main()
