from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import sqlite3

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MODEL =====
model = joblib.load("stress_model.pkl")
labels = {0: "Low", 1: "Medium", 2: "High"}

# ===== DATABASE (local SQLite, no server required) =====
conn = sqlite3.connect("burnout.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        study    REAL,
        sleep    REAL,
        social   REAL,
        physical REAL,
        result   TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()
print("[OK] SQLite database ready (burnout.db)")


# ===== OFFLINE CHATBOT =====
_RESPONSES = {
    "sleep": (
        "It sounds like sleep is affecting you. Try a wind-down routine: "
        "stop screens 30 minutes before bed, keep your room cool and dark, "
        "and aim for a consistent wake-up time even on weekends."
    ),
    "tired": (
        "Fatigue often signals your body needs recovery time. "
        "Try a 20-minute nap before 3 pm, and check whether you're drinking "
        "enough water — dehydration mimics exhaustion."
    ),
    "focus": (
        "For focus issues, try the 45-15 rule: study for 45 minutes with "
        "full attention, then take a 15-minute break away from your desk. "
        "Remove your phone from the room during the 45-minute block."
    ),
    "concentrate": (
        "Concentration improves with environment. Find a consistent study "
        "spot, use noise-cancelling headphones or brown noise, and write "
        "down distracting thoughts on a notepad to revisit later."
    ),
    "overwhelm": (
        "When everything feels like too much, pick just ONE task for the "
        "next 25 minutes. Break it into the smallest possible first step. "
        "You don't have to solve everything today."
    ),
    "stress": (
        "Stress is a signal, not a verdict. Try box breathing: inhale 4s, "
        "hold 4s, exhale 4s, hold 4s. Repeat 4 times. Then write down "
        "the one thing that's worrying you most and what you can do about it."
    ),
    "anxious": (
        "Anxiety often comes from focusing on things outside your control. "
        "Try the 5-4-3-2-1 grounding technique: name 5 things you can see, "
        "4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste."
    ),
    "exam": (
        "Exam pressure is real. Shorter, frequent study sessions beat "
        "marathon cramming — 3 sessions of 45 minutes beats one 3-hour "
        "block. Review material within 24 hours of first learning it."
    ),
    "assignment": (
        "Break the assignment into a list of small, concrete actions "
        "(not goals, actions). Start with the easiest item to build "
        "momentum, then move to the hardest while your energy is up."
    ),
    "burnout": (
        "Burnout needs real rest, not just a short break. Schedule at least "
        "one full day this week with no academic work. Talk to someone you "
        "trust about how you're feeling — you don't have to carry this alone."
    ),
    "motivation": (
        "Low motivation is often a sign of depleted energy, not laziness. "
        "Make sure you're sleeping enough, eating at regular times, and "
        "getting outside for at least 10 minutes a day."
    ),
}

_DEFAULT = (
    "I hear you. Whatever you're going through, it's okay to feel this way. "
    "Try to take things one small step at a time, and don't hesitate to "
    "reach out to a counselor or trusted person if things feel unmanageable."
)


def offline_chat(message: str) -> str:
    msg = message.lower()
    for keyword, reply in _RESPONSES.items():
        if keyword in msg:
            return reply
    return _DEFAULT


# ===== ROUTES =====

@app.get("/")
def home():
    return {"message": "Burnout AI API running (offline mode)"}


@app.get("/data")
def get_dashboard_data():
    try:
        df = pd.read_csv("student_lifestyle_dataset.csv")
        df.columns = df.columns.str.strip()
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict")
def predict(data: dict):
    try:
        study    = float(data.get("study_hours_per_day", 0))
        sleep    = float(data.get("sleep_hours_per_day", 0))
        social   = float(data.get("social_hours_per_day", 0))
        physical = float(data.get("physical_activity_hours_per_day", 0))

        total = study + sleep + social + physical or 1.0

        df = pd.DataFrame([{
            "study_hours_per_day":             study,
            "sleep_hours_per_day":             sleep,
            "social_hours_per_day":            social,
            "physical_activity_hours_per_day": physical,
            "study_sleep_ratio":               study / (sleep + 0.1),
            "active_hours":                    study + physical,
            "rest_ratio":                      (sleep + physical) / total,
            "productive_vs_leisure":           study / (social + physical + 0.1),
        }])

        prediction   = model.predict(df)[0]
        result_label = labels.get(int(prediction), "Unknown")

        cursor.execute(
            "INSERT INTO predictions (study, sleep, social, physical, result) VALUES (?,?,?,?,?)",
            (study, sleep, social, physical, result_label),
        )
        conn.commit()

        return {"prediction": result_label, "risk": int(prediction)}

    except Exception as e:
        return {"error": str(e)}


@app.post("/plan")
def generate_plan(data: dict):
    risk = int(data.get("risk", 0))

    plans = {
        2: {
            "level": "High Burnout Risk",
            "steps": [
                "Reduce study sessions to 2-hour blocks",
                "Sleep at least 7-8 hours",
                "Take breaks every 25 minutes",
                "Exercise daily",
                "Talk with a mentor or counselor",
            ],
        },
        1: {
            "level": "Moderate Burnout Risk",
            "steps": [
                "Balance study and rest",
                "Sleep 6-8 hours",
                "Take regular breaks",
                "Light physical activity",
            ],
        },
        0: {
            "level": "Low Burnout Risk",
            "steps": [
                "Maintain current study schedule",
                "Keep sleeping 7-8 hours",
                "Take short breaks",
                "Exercise weekly",
            ],
        },
    }

    return {"plan": plans.get(risk, plans[0])}


@app.post("/chat")
def chat(data: dict):
    message = data.get("message", "")
    return {"reply": offline_chat(message)}
