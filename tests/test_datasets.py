import os, pandas as pd, pytest
from datasets import (
    load_lifestyle, load_sleep_health, load_stress_level,
    load_social_media, load_depression, load_mental_health2,
    load_sleep_patterns, load_stress_survey, load_all, UNIFIED_COLUMNS,
)

@pytest.fixture(autouse=True)
def chdir_backend(monkeypatch):
    monkeypatch.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _check(df):
    if df is None: return
    assert set(UNIFIED_COLUMNS).issubset(df.columns), f"Missing cols: {set(UNIFIED_COLUMNS) - set(df.columns)}"
    assert len(df) > 10
    assert df["target_continuous"].between(0, 1).all(), "target_continuous out of [0,1]"
    assert df["target_class"].isin([0, 1, 2]).all(), "target_class not in 0/1/2"

def test_lifestyle():       _check(load_lifestyle())
def test_sleep_health():    _check(load_sleep_health())
def test_stress_level():    _check(load_stress_level())
def test_social_media():    _check(load_social_media())
def test_depression():      _check(load_depression())
def test_mental_health2():  _check(load_mental_health2())
def test_sleep_patterns():  _check(load_sleep_patterns())
def test_stress_survey():    _check(load_stress_survey())

def test_load_all_schema():
    df = load_all()
    assert "source" in df.columns
    assert df["source"].nunique() >= 2
    assert df["target_continuous"].between(0, 1).all()
    assert len(df) > 500
