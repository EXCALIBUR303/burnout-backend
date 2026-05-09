# burnout-backend/feedback.py
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER,
    user_id INTEGER,
    accurate INTEGER NOT NULL,
    actual_label TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def init_schema(conn: sqlite3.Connection):
    conn.execute(SCHEMA)
    conn.commit()

def record_feedback(conn, prediction_id, user_id, accurate: int,
                    actual_label=None, notes=None):
    conn.execute(
        "INSERT INTO feedback (prediction_id, user_id, accurate, actual_label, notes) "
        "VALUES (?, ?, ?, ?, ?)",
        (prediction_id, user_id, int(bool(accurate)), actual_label, notes),
    )
    conn.commit()

def get_stats(conn) -> dict:
    cur = conn.execute("SELECT COUNT(*), SUM(accurate) FROM feedback")
    total, accurate = cur.fetchone()
    total = total or 0
    accurate = accurate or 0
    return {
        "total":        int(total),
        "accurate":     int(accurate),
        "accuracy_pct": round(accurate / total * 100, 1) if total else 0.0,
    }
