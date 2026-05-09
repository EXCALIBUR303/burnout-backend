"""retrain.py - augment training set with user feedback rows + retrain v4."""
import sqlite3, joblib, json
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

from datasets import load_all
from trainer import ALL_FEATURES, add_features, CLIP
from trainer_v3 import impute_features

LABEL_TO_CLASS = {"Low": 0, "Medium": 1, "High": 2, "low": 0, "medium": 1, "high": 2}


def fetch_feedback_rows(db_path: str = "burnout.db") -> pd.DataFrame:
    """Pull predictions JOIN feedback where the user gave a corrected label."""
    conn = sqlite3.connect(db_path)
    try:
        rows = pd.read_sql_query("""
            SELECT p.study, p.sleep, p.social, p.physical, p.result,
                   f.actual_label, f.accurate
            FROM predictions p
            JOIN feedback f ON f.prediction_id = p.id
        """, conn)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()
    if rows.empty:
        return pd.DataFrame()
    # Use actual_label if user provided it, else if accurate=1 use the prediction itself
    rows["true_label"] = rows.apply(
        lambda r: r["actual_label"] if pd.notna(r["actual_label"]) and r["actual_label"]
                  else (r["result"] if r["accurate"] == 1 else None),
        axis=1,
    )
    rows = rows.dropna(subset=["true_label"])
    rows["target_class"] = rows["true_label"].map(LABEL_TO_CLASS)
    rows = rows.dropna(subset=["target_class"])
    return rows


def augment_with_feedback(base_df: pd.DataFrame, fb: pd.DataFrame) -> pd.DataFrame:
    if fb.empty:
        return base_df
    aug = pd.DataFrame({
        "study_hours_per_day":             fb["study"],
        "sleep_hours_per_day":             fb["sleep"],
        "social_hours_per_day":            fb["social"],
        "physical_activity_hours_per_day": fb["physical"],
        "gpa_norm":                        np.nan,
        "screen_time_hours":               np.nan,
        "extracurricular_hours":           np.nan,
        "target_continuous":               fb["target_class"] / 2.0,
        "target_class":                    fb["target_class"].astype(int),
        "source":                          "user_feedback",
    })
    print(f"  Augmenting with {len(aug)} feedback rows")
    return pd.concat([base_df, aug], ignore_index=True)


def main():
    print("=" * 60)
    print("  v4 Retrain Pipeline (with user feedback)")
    print("=" * 60)
    base = load_all()
    fb = fetch_feedback_rows()
    df = augment_with_feedback(base, fb)
    if fb.empty:
        print("No feedback yet. Run trainer_v4.py instead.")
        return

    print(f"\nTotal training rows after augmentation: {len(df):,}")
    print(df.groupby("source").size().to_string())

    # Reuse trainer_v4 logic by invoking it programmatically:
    # Save augmented df as a temporary CSV is not needed - just import trainer_v4 main flow
    # but we override load_all in-place
    import trainer_v4
    original_load = trainer_v4.load_all
    trainer_v4.load_all = lambda: df
    try:
        trainer_v4.main()
        # Rename output to mark as retrained
        import shutil
        shutil.copy("stress_model_v4.pkl", "stress_model_v4_retrained.pkl")
        shutil.copy("model_meta_v4.json", "model_meta_v4_retrained.json")
        print("\nSaved retrained artifacts.")
    finally:
        trainer_v4.load_all = original_load


if __name__ == "__main__":
    main()
