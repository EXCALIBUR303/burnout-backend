"""
Synthetic burnout dataset generator v2 — research-grounded with subjective features.

Built on patterns from:
  - Maslach Burnout Inventory (MBI): exhaustion ↔ sleep deficit, cynicism ↔ overload
  - Salmela-Aro et al. (2009) Student Burnout Inventory: study load × poor recovery
  - American College Health Assessment: 6+ hr/day study + <6 hr sleep is the high-risk archetype
  - Kessler Psychological Distress Scale: physical activity is protective at >150 min/week
  - GAD-7 anxiety scale: anxiety positively correlates with stress + screen overload
  - PHQ-9 / social support literature: low social support amplifies stress effects

v2 generates 11 features (7 objective lifestyle + 4 subjective ratings) so the
trainer can learn how subjective experience interacts with objective behaviour.

Saves to synthetic_burnout_25k.csv with column names matching student_lifestyle_dataset.csv
plus 4 new subjective columns.
"""
import argparse
import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)


def generate(n: int = 25000) -> pd.DataFrame:
    # ── 1. Sample base lifestyle features with realistic distributions ────
    # Study hours: bimodal (cramming students vs. balanced students)
    study = np.where(
        RNG.random(n) < 0.35,
        RNG.normal(8.5, 1.5, n),   # heavy-study cluster
        RNG.normal(4.5, 1.5, n),   # balanced cluster
    ).clip(0, 14)

    # Sleep hours: heavy-study students sleep less (negative correlation)
    sleep_baseline = 8.0 - 0.35 * (study - 5.5)
    sleep = (sleep_baseline + RNG.normal(0, 1.0, n)).clip(3.5, 11)

    # Social hours: anti-correlated with study, log-normal-ish
    social = (RNG.exponential(2.0, n) - 0.15 * (study - 5.5)).clip(0, 8)

    # Physical activity (hrs/day): mostly low, fat tail of athletic students
    physical = np.where(
        RNG.random(n) < 0.18,
        RNG.normal(2.0, 0.8, n),
        RNG.exponential(0.6, n),
    ).clip(0, 5)

    # Screen time: heavy social media users + procrastinators study less
    screen_baseline = 5.0 - 0.25 * study + 0.4 * (8 - sleep)
    screen = (screen_baseline + RNG.normal(0, 1.5, n)).clip(0.5, 14)

    # GPA: positively correlated with moderate study + adequate sleep, U-shaped
    study_optimum = 1 - ((study - 6.5) ** 2) / 50
    sleep_bonus = np.where(sleep >= 6.5, 0.2, -0.3)
    gpa_raw = 0.55 + 0.25 * study_optimum + sleep_bonus + RNG.normal(0, 0.12, n)
    gpa = (gpa_raw * 10).clip(2.0, 10.0)

    # Extracurricular: mostly low, some involved students
    extra = np.where(
        RNG.random(n) < 0.25,
        RNG.normal(2.0, 0.8, n),
        RNG.exponential(0.4, n),
    ).clip(0, 5)

    # ── 2. Compute the latent "burnout score" from objective drivers ──────
    sleep_deficit = np.maximum(0, 7.5 - sleep)
    study_load = np.maximum(0, study - 6.5)
    recovery_balance = (sleep + physical + 0.3 * social) / 12.0
    screen_overload = np.maximum(0, screen - 5.0) / 8.0
    gpa_stress = np.maximum(0, 0.65 - gpa / 10.0)

    burnout_latent = (
        + 1.20 * sleep_deficit / 4.0
        + 0.95 * study_load / 5.0
        + 0.85 * screen_overload
        + 0.60 * gpa_stress
        - 0.70 * recovery_balance
        - 0.25 * np.minimum(extra, 1.5) / 1.5
        + RNG.normal(0, 0.30, n)
    )

    # ── 3. Generate SUBJECTIVE features correlated with latent burnout ────
    # Anxiety: positive correlation with burnout, screen, study load
    # Scale 0-1 (higher = more anxious)
    anxiety_raw = (
        0.4
        + 0.35 * (burnout_latent - burnout_latent.mean()) / (burnout_latent.std() + 1e-6)
        + 0.20 * screen_overload
        + 0.15 * study_load / 5.0
        + RNG.normal(0, 0.15, n)
    )
    anxiety = anxiety_raw.clip(0, 1)

    # Social support deficit: HIGH value = LACKING support (risk signal)
    # Anti-correlated with social hours; people who socialize have support
    social_deficit_raw = (
        0.6
        - 0.25 * social / 4.0           # more social hours → better support
        - 0.10 * (extra > 0.5)          # extracurricular → community
        + 0.20 * (burnout_latent > 0)   # burnt-out students feel isolated
        + RNG.normal(0, 0.15, n)
    )
    social_support_deficit = social_deficit_raw.clip(0, 1)

    # Career concern: linked to GPA stress + burnout
    career_concern_raw = (
        0.4
        + 0.30 * gpa_stress * 1.5
        + 0.20 * (burnout_latent - burnout_latent.mean()) / (burnout_latent.std() + 1e-6)
        + RNG.normal(0, 0.18, n)
    )
    career_concern = career_concern_raw.clip(0, 1)

    # Self-rated mood: subjective stress level (0-1, higher = more stressed)
    # Strongly correlated with latent burnout (this is the "ground truth proxy")
    mood_raw = (
        0.45
        + 0.45 * (burnout_latent - burnout_latent.mean()) / (burnout_latent.std() + 1e-6)
        + 0.10 * sleep_deficit / 4.0
        + RNG.normal(0, 0.12, n)
    )
    mood = mood_raw.clip(0, 1)

    # ── 4. Bin latent score into 3 classes with realistic prevalence ──────
    low_threshold  = np.percentile(burnout_latent, 25)
    high_threshold = np.percentile(burnout_latent, 75)
    stress_label = np.where(
        burnout_latent <  low_threshold, "Low",
        np.where(burnout_latent < high_threshold, "Moderate", "High"),
    )

    df = pd.DataFrame({
        "Student_ID":                       np.arange(1, n + 1),
        "Study_Hours_Per_Day":              np.round(study, 2),
        "Sleep_Hours_Per_Day":              np.round(sleep, 2),
        "Social_Hours_Per_Day":             np.round(social, 2),
        "Physical_Activity_Hours_Per_Day":  np.round(physical, 2),
        "Screen_Time_Hours_Per_Day":        np.round(screen, 2),
        "GPA":                              np.round(gpa, 2),
        "Extracurricular_Hours_Per_Day":    np.round(extra, 2),
        # Subjective features (0-1 normalized)
        "Anxiety_Norm":                     np.round(anxiety, 3),
        "Social_Support_Deficit":           np.round(social_support_deficit, 3),
        "Career_Concern_Norm":              np.round(career_concern, 3),
        "Mood_Norm":                        np.round(mood, 3),
        "Stress_Level":                     stress_label,
    })
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=25000, help="number of synthetic samples")
    parser.add_argument("--out", default="synthetic_burnout_25k.csv")
    args = parser.parse_args()

    df = generate(args.n)
    print(f"Generated {len(df)} synthetic rows")
    print("\nClass distribution:")
    print(df["Stress_Level"].value_counts().to_string())
    print("\nFeature summary:")
    print(df.describe().round(2).to_string())

    # Sanity check: subjective features should correlate with stress class
    print("\nMean of subjective features per stress class:")
    print(df.groupby("Stress_Level")[
        ["Anxiety_Norm","Social_Support_Deficit","Career_Concern_Norm","Mood_Norm"]
    ].mean().round(3).to_string())

    df.to_csv(args.out, index=False)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
