import os, sqlite3, pytest
from feedback import init_schema, record_feedback, get_stats

@pytest.fixture
def db(tmp_path):
    p = tmp_path / "test.db"
    conn = sqlite3.connect(str(p))
    init_schema(conn)
    return conn

def test_record_and_stats(db):
    record_feedback(db, prediction_id=None, user_id=1, accurate=1, actual_label=None, notes="spot on")
    record_feedback(db, prediction_id=None, user_id=2, accurate=0, actual_label="High", notes="too low")
    s = get_stats(db)
    assert s["total"] == 2
    assert s["accurate"] == 1
    assert s["accuracy_pct"] == 50.0
