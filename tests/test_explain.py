import os, pytest

def test_explain_returns_contributions():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from explain import explain_prediction
    sample = {
        "study_hours_per_day": 9.0, "sleep_hours_per_day": 5.0,
        "social_hours_per_day": 1.0, "physical_activity_hours_per_day": 0.5,
        "gpa_norm": 0.6, "screen_time_hours": 6.0, "extracurricular_hours": 0.5,
    }
    out = explain_prediction(sample)
    assert "contributions" in out, f"Missing contributions in {out.keys()}"
    assert isinstance(out["contributions"], list)
    assert len(out["contributions"]) >= 3
    for c in out["contributions"]:
        assert "feature" in c and "shap" in c and "direction" in c
        assert c["direction"] in ("toward_high_risk", "toward_low_risk")
    assert "probabilities" in out
    assert "prediction" in out
