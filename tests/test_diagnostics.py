import json, os, subprocess, sys

def test_diagnostics_runs_and_writes_report(tmp_path):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out  = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, "diagnostics.py", "--out", str(out)],
        cwd=repo, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert "leakage_score" in data
    assert "rule_recovered_accuracy" in data
    assert 0.0 <= data["leakage_score"] <= 1.0
