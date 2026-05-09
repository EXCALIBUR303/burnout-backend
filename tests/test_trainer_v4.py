import os, json, joblib

def test_v4_artifacts():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert os.path.exists("stress_model_v4.pkl")
    assert os.path.exists("model_meta_v4.json")
    model = joblib.load("stress_model_v4.pkl")
    assert hasattr(model, "predict_proba")
    meta = json.load(open("model_meta_v4.json"))
    assert meta["version"] == "4.0"
    assert 0.40 <= meta["honesty_holdout_accuracy"] <= 0.99
