"""
datasets.py — Unified multi-dataset loader for burnout/stress ML model (v3).

Each loader reads a CSV, maps columns to UNIFIED_COLUMNS, derives target_class
from target_continuous, and returns a DataFrame (or None if the file is missing).
"""

import os
import numpy as np
import pandas as pd


def _safe_csv(path: str) -> pd.DataFrame | None:
    """Read CSV if it exists, else return None."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _to_class(c: float) -> int:
    """Bin a single continuous value in [0,1] into 0/1/2."""
    if c < 0.33:
        return 0
    if c < 0.66:
        return 1
    return 2

UNIFIED_COLUMNS = [
    "study_hours_per_day",
    "sleep_hours_per_day",
    "social_hours_per_day",
    "physical_activity_hours_per_day",
    "gpa_norm",
    "screen_time_hours",
    "extracurricular_hours",
    # ── Subjective features (NEW in v5) ──
    "anxiety_norm",            # 0-1, higher = more anxious
    "social_support_deficit",  # 0-1, higher = LESS support (risk)
    "career_concern_norm",     # 0-1, higher = more worried
    "mood_norm",               # 0-1, higher = more stressed
    # ── Targets ──
    "target_continuous",
    "target_class",
    "source",
]


def _derive_target_class(series: pd.Series) -> pd.Series:
    """Bin continuous target into 3 classes: <0.33→0, <0.66→1, else→2."""
    return pd.cut(
        series,
        bins=[-0.001, 0.33, 0.66, 1.001],
        labels=[0, 1, 2],
    ).astype(int)


def _finalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Add source column, derive target_class, ensure all UNIFIED_COLUMNS exist,
    drop rows without a target, and return df[UNIFIED_COLUMNS]."""
    df = df.copy()
    df["source"] = source

    # Derive target_class from target_continuous
    df = df.dropna(subset=["target_continuous"])
    df["target_class"] = _derive_target_class(df["target_continuous"])

    # Fill any missing unified columns with NaN
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")

    return df[UNIFIED_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_lifestyle(path: str = "student_lifestyle_dataset.csv") -> pd.DataFrame | None:
    """2000-row student lifestyle dataset."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    stress_map = {"Low": 0.15, "Moderate": 0.5, "Medium": 0.5, "High": 0.85}

    out = pd.DataFrame()
    out["study_hours_per_day"] = pd.to_numeric(df["Study_Hours_Per_Day"], errors="coerce")
    out["sleep_hours_per_day"] = pd.to_numeric(df["Sleep_Hours_Per_Day"], errors="coerce")
    out["social_hours_per_day"] = pd.to_numeric(df["Social_Hours_Per_Day"], errors="coerce")
    out["physical_activity_hours_per_day"] = pd.to_numeric(
        df["Physical_Activity_Hours_Per_Day"], errors="coerce"
    )
    out["gpa_norm"] = pd.to_numeric(df["GPA"], errors="coerce") / 10.0
    out["extracurricular_hours"] = (
        pd.to_numeric(df["Extracurricular_Hours_Per_Day"], errors="coerce")
        if "Extracurricular_Hours_Per_Day" in df.columns
        else float("nan")
    )
    out["target_continuous"] = df["Stress_Level"].map(stress_map)

    return _finalize(out, "lifestyle")


def load_lifestyle_100k(path: str = "student_lifestyle_100k.csv") -> pd.DataFrame | None:
    """100k-row student lifestyle dataset with numeric Stress_Level (2–10)."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    out = pd.DataFrame()
    out["study_hours_per_day"] = pd.to_numeric(df["Study_Hours"], errors="coerce")
    out["sleep_hours_per_day"] = pd.to_numeric(df["Sleep_Duration"], errors="coerce")
    # Social_Media_Hours used as screen_time proxy
    out["screen_time_hours"] = pd.to_numeric(df["Social_Media_Hours"], errors="coerce")
    # Physical_Activity is in minutes/day → convert to hours
    out["physical_activity_hours_per_day"] = (
        pd.to_numeric(df["Physical_Activity"], errors="coerce") / 60.0
    )
    out["gpa_norm"] = pd.to_numeric(df["CGPA"], errors="coerce") / 10.0
    # Stress_Level 2–10 → normalize to [0, 1]
    stress_num = pd.to_numeric(df["Stress_Level"], errors="coerce")
    out["target_continuous"] = (stress_num - 2) / 8

    return _finalize(out, "lifestyle_100k")


def load_sleep_health(path: str = "Sleep_health_and_lifestyle_dataset.csv") -> pd.DataFrame | None:
    """Sleep health & lifestyle dataset."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    out = pd.DataFrame()
    out["sleep_hours_per_day"] = pd.to_numeric(df["Sleep Duration"], errors="coerce")
    # Physical Activity Level is minutes/week → convert to hours/day
    out["physical_activity_hours_per_day"] = (
        pd.to_numeric(df["Physical Activity Level"], errors="coerce") / (7 * 60)
    )
    # Stress Level 3–8 → normalize: (val - 1) / 7
    stress_num = pd.to_numeric(df["Stress Level"], errors="coerce")
    out["target_continuous"] = (stress_num - 1) / 7

    return _finalize(out, "sleep_health")


def load_stress_level(path: str = "StressLevelDataset.csv") -> pd.DataFrame | None:
    """
    Student stress level dataset (1100 rows, 21 columns of survey ratings).

    UPGRADE in v5: extracts all 4 subjective features (anxiety, social support,
    career concern, mood proxy) — previously only 3 of 20 columns were used.
    """
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    stress_map = {0: 0.15, 1: 0.5, 2: 0.85}

    out = pd.DataFrame()
    # ── Objective lifestyle proxies (from Likert ratings) ──
    # study_load (0–5 rating) → scale to approximate study hours: * 1.2
    out["study_hours_per_day"] = pd.to_numeric(df["study_load"], errors="coerce") * 1.2
    # sleep_quality (0–5, higher=better) → approximate sleep hours
    sleep_q = pd.to_numeric(df["sleep_quality"], errors="coerce")
    out["sleep_hours_per_day"] = 4 + (10 - sleep_q) * 0.4
    out["extracurricular_hours"] = (
        pd.to_numeric(df["extracurricular_activities"], errors="coerce") * 1.5
    )

    # ── Subjective features (NEW in v5) ──
    # anxiety_level: 0-21 scale → normalize to 0-1
    out["anxiety_norm"] = (
        pd.to_numeric(df["anxiety_level"], errors="coerce") / 21.0
    ).clip(0, 1)
    # social_support: 0-3 scale where higher = MORE support
    # We need DEFICIT (higher = WORSE), so invert: (3 - val) / 3
    out["social_support_deficit"] = (
        (3 - pd.to_numeric(df["social_support"], errors="coerce")) / 3.0
    ).clip(0, 1)
    # future_career_concerns: 0-5 scale, higher = more worried (already a deficit)
    out["career_concern_norm"] = (
        pd.to_numeric(df["future_career_concerns"], errors="coerce") / 5.0
    ).clip(0, 1)
    # No direct mood field — derive from headache + breathing_problem (0-5 each)
    # as an embodied-stress proxy: physical symptoms ≈ subjective stress
    headache = pd.to_numeric(df["headache"], errors="coerce") / 5.0
    breathing = pd.to_numeric(df["breathing_problem"], errors="coerce") / 5.0
    out["mood_norm"] = ((headache + breathing) / 2.0).clip(0, 1)

    out["target_continuous"] = pd.to_numeric(df["stress_level"], errors="coerce").map(stress_map)

    return _finalize(out, "stress_level")


def load_social_media(path: str = "Students Social Media Addiction.csv") -> pd.DataFrame | None:
    """Students Social Media Addiction dataset."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    out = pd.DataFrame()
    out["screen_time_hours"] = pd.to_numeric(df["Avg_Daily_Usage_Hours"], errors="coerce")
    out["sleep_hours_per_day"] = pd.to_numeric(df["Sleep_Hours_Per_Night"], errors="coerce")
    # Mental_Health_Score range 4–9, higher = worse → normalize: (val - 4) / 5
    mhs = pd.to_numeric(df["Mental_Health_Score"], errors="coerce")
    out["target_continuous"] = (mhs - 4) / 5

    return _finalize(out, "social_media")


def load_depression(path: str = "student_depression_dataset.csv") -> pd.DataFrame | None:
    """Student depression dataset (27k rows)."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    sleep_map = {
        "'Less than 5 hours'": 4.0,
        "'5-6 hours'": 5.5,
        "'7-8 hours'": 7.5,
        "'More than 8 hours'": 8.5,
    }
    depression_map = {0: 0.2, 1: 0.75}

    out = pd.DataFrame()
    out["study_hours_per_day"] = pd.to_numeric(df["Work/Study Hours"], errors="coerce")
    out["gpa_norm"] = pd.to_numeric(df["CGPA"], errors="coerce") / 10.0
    out["sleep_hours_per_day"] = df["Sleep Duration"].map(sleep_map)
    dep = pd.to_numeric(df["Depression"], errors="coerce")
    out["target_continuous"] = dep.map(depression_map)

    return _finalize(out, "depression")


def load_mental_health2(path: str = "student_mental_health_dataset.csv") -> pd.DataFrame | None:
    """Student mental health dataset (Stress_Levels float 20–96)."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    out = pd.DataFrame()
    out["study_hours_per_day"] = pd.to_numeric(df["Study_Hours"], errors="coerce")
    # Stress_Levels float 20–96 → normalize: val / 100
    stress = pd.to_numeric(df["Stress_Levels"], errors="coerce")
    out["target_continuous"] = stress / 100

    return _finalize(out, "mental_health2")


def load_sleep_patterns(path: str = "student_sleep_patterns.csv") -> pd.DataFrame | None:
    """Student sleep patterns dataset."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    out = pd.DataFrame()
    out["study_hours_per_day"] = pd.to_numeric(df["Study_Hours"], errors="coerce")
    out["sleep_hours_per_day"] = pd.to_numeric(df["Sleep_Duration"], errors="coerce")
    out["screen_time_hours"] = pd.to_numeric(df["Screen_Time"], errors="coerce")
    # Physical_Activity is minutes/day → convert to hours
    out["physical_activity_hours_per_day"] = (
        pd.to_numeric(df["Physical_Activity"], errors="coerce") / 60.0
    )
    # Sleep_Quality 1–10, high = good → invert for burnout proxy: (10 - val) / 9
    sq = pd.to_numeric(df["Sleep_Quality"], errors="coerce")
    out["target_continuous"] = (10 - sq) / 9

    return _finalize(out, "sleep_patterns")


def load_stress_survey(path: str = "Stress_Dataset.csv") -> pd.DataFrame | None:
    """
    Stress_Dataset.csv: 843 rows of 1-5 scale survey items.

    UPGRADE in v5: extracts subjective features (anxiety, social isolation,
    career confidence, mood) from individual survey items rather than just
    averaging everything into the target. Now provides 4/4 subjective coverage.
    """
    df = _safe_csv(path)
    if df is None:
        return None

    def _find_col(keyword: str):
        for c in df.columns:
            if keyword.lower() in c.lower():
                return c
        return None

    def _norm(col_name: str):
        """1-5 Likert → 0-1, with NaN-safe handling."""
        if col_name is None or col_name not in df.columns:
            return pd.Series(float("nan"), index=df.index)
        s = pd.to_numeric(df[col_name], errors="coerce")
        return ((s - 1) / 4).clip(0, 1)

    # ── Build subjective features from specific items ──
    anxiety = pd.concat([
        _norm(_find_col("anxiety or tension")),
        _norm(_find_col("rapid heartbeat")),
    ], axis=1).mean(axis=1)

    social_isolation = _norm(_find_col("lonely or isolated"))

    confidence_lack = pd.concat([
        _norm(_find_col("lack confidence in your academic performanc")),
        _norm(_find_col("overwhelmed with your academic")),
    ], axis=1).mean(axis=1)

    mood_low = pd.concat([
        _norm(_find_col("sadness or low mood")),
        _norm(_find_col("irritated easily")),
    ], axis=1).mean(axis=1)

    # ── Build target from broader symptom set ──
    symptom_keywords = [
        "experienced stress", "rapid heartbeat", "anxiety or tension",
        "sleep problems", "headaches", "irritated easily",
        "trouble concentrating", "sadness or low mood", "lonely or isolated",
        "overwhelmed with your academic", "lack confidence in your academic performanc",
    ]
    symptom_cols: list[str] = []
    for kw in symptom_keywords:
        for c in df.columns:
            if kw.lower() in c.lower() and c not in symptom_cols:
                symptom_cols.append(c)
                break
    if len(symptom_cols) < 5:
        return None

    sym = df[symptom_cols].apply(pd.to_numeric, errors="coerce")
    cont = ((sym.mean(axis=1) - 1) / 4).clip(0, 1)

    out = pd.DataFrame({
        "study_hours_per_day":             np.nan,
        "sleep_hours_per_day":             np.nan,
        "social_hours_per_day":            np.nan,
        "physical_activity_hours_per_day": np.nan,
        "gpa_norm":                        np.nan,
        "screen_time_hours":               np.nan,
        "extracurricular_hours":           np.nan,
        "anxiety_norm":                    anxiety,
        "social_support_deficit":          social_isolation,
        "career_concern_norm":             confidence_lack,
        "mood_norm":                       mood_low,
        "target_continuous":               cont,
    })
    out["target_class"] = out["target_continuous"].apply(
        lambda c: _to_class(c) if pd.notna(c) else np.nan
    )
    out["source"] = "stress_survey"
    return out.dropna(subset=["target_continuous"])[UNIFIED_COLUMNS]


def load_synthetic(path: str = "synthetic_burnout_25k.csv") -> pd.DataFrame | None:
    """
    25,000 research-grounded synthetic samples from synthetic_burnout.py v2.
    Includes 4 subjective features (anxiety, social_support_deficit, career_concern, mood).
    Falls back to the older 5k file if the 25k one isn't generated yet.
    """
    df = _safe_csv(path)
    if df is None:
        df = _safe_csv("synthetic_burnout_5k.csv")
        if df is None:
            return None
    cls_map = {"Low": 0.15, "Moderate": 0.5, "Medium": 0.5, "High": 0.85}
    out = pd.DataFrame({
        "study_hours_per_day":             pd.to_numeric(df["Study_Hours_Per_Day"], errors="coerce"),
        "sleep_hours_per_day":             pd.to_numeric(df["Sleep_Hours_Per_Day"], errors="coerce"),
        "social_hours_per_day":            pd.to_numeric(df["Social_Hours_Per_Day"], errors="coerce"),
        "physical_activity_hours_per_day": pd.to_numeric(df["Physical_Activity_Hours_Per_Day"], errors="coerce"),
        "gpa_norm":                        pd.to_numeric(df["GPA"], errors="coerce") / 10.0,
        "screen_time_hours":               pd.to_numeric(df["Screen_Time_Hours_Per_Day"], errors="coerce"),
        "extracurricular_hours":           pd.to_numeric(df["Extracurricular_Hours_Per_Day"], errors="coerce"),
        "target_continuous":               df["Stress_Level"].map(cls_map),
    })
    # Subjective columns — present in 25k file, absent in old 5k file
    for col_in, col_out in [
        ("Anxiety_Norm",            "anxiety_norm"),
        ("Social_Support_Deficit",  "social_support_deficit"),
        ("Career_Concern_Norm",     "career_concern_norm"),
        ("Mood_Norm",               "mood_norm"),
    ]:
        if col_in in df.columns:
            out[col_out] = pd.to_numeric(df[col_in], errors="coerce")
        else:
            out[col_out] = float("nan")

    out["target_class"] = out["target_continuous"].apply(
        lambda c: _to_class(c) if pd.notna(c) else np.nan
    )
    out["source"] = "synthetic"
    return out.dropna(subset=["target_continuous"])[UNIFIED_COLUMNS]


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------

def load_all() -> pd.DataFrame:
    """Load all datasets, concatenate, and return df[UNIFIED_COLUMNS]."""
    loaders = [
        load_lifestyle,
        load_lifestyle_100k,
        load_sleep_health,
        load_stress_level,
        load_social_media,
        load_depression,
        load_mental_health2,
        load_sleep_patterns,
        load_stress_survey,
        load_synthetic,
    ]
    frames = [fn() for fn in loaders]
    frames = [f for f in frames if f is not None and len(f) > 0]
    combined = pd.concat(frames, ignore_index=True)
    return combined[UNIFIED_COLUMNS]
