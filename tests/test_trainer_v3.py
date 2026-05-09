import os, json, joblib

def test_v3_artifacts_exist_and_load():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert os.path.exists("stress_model_v3.pkl"), "stress_model_v3.pkl not found"
    assert os.path.exists("model_meta_v3.json"), "model_meta_v3.json not found"
    model = joblib.load("stress_model_v3.pkl")
    assert hasattr(model, "predict_proba"), "model must have predict_proba"
    meta = json.load(open("model_meta_v3.json"))
    assert meta["version"] == "3.0"
    assert "honesty_holdout_accuracy" in meta
    assert "features" in meta and len(meta["features"]) == 14
    # honesty holdout should be in realistic range (not 100% leakage, not worse than random)
    acc = meta["honesty_holdout_accuracy"]
    assert 0.30 <= acc <= 0.99, f"Holdout accuracy {acc} outside expected range"
