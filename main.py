from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import sqlite3
import bcrypt
import jwt
import os
from datetime import datetime, timedelta, timezone

# ── Gemini AI via REST API (no extra packages — uses stdlib urllib) ────────────
import json
import urllib.request

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-1.5-flash:generateContent?key={key}"
)
GEMINI_SYSTEM = (
    "You are a compassionate AI wellness assistant for Burnout/AI, a student mental-health app at Woxsen University. "
    "Your role: provide empathetic, evidence-based advice about stress, burnout, sleep, study habits, anxiety, and academic pressure. "
    "Keep replies concise (2-4 sentences max), warm, and actionable. "
    "If someone sounds in crisis, gently direct them to the free Woxsen counselling service "
    "(wellness.centre@woxsen.edu.in / 9049980927). "
    "Never diagnose. Remind users you are an AI, not a replacement for professional help. "
    "Speak like a supportive senior student who knows a lot about wellbeing."
)

def gemini_chat(message: str) -> str | None:
    """Call Gemini REST API. Returns reply text or None on failure."""
    if not GEMINI_API_KEY:
        return None
    try:
        payload = json.dumps({
            "system_instruction": {"parts": [{"text": GEMINI_SYSTEM}]},
            "contents": [{"parts": [{"text": message}]}],
            "generationConfig": {"maxOutputTokens": 200, "temperature": 0.7},
        }).encode()
        req = urllib.request.Request(
            GEMINI_URL.format(key=GEMINI_API_KEY),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return None

if GEMINI_API_KEY:
    print("✅ Gemini REST chatbot enabled (no extra packages needed)")

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 7

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

FEATURE_NAMES = [
    "study_hours_per_day", "sleep_hours_per_day", "social_hours_per_day",
    "physical_activity_hours_per_day", "study_sleep_ratio",
    "active_hours", "rest_ratio", "productive_vs_leisure",
]

FEATURE_META = {
    "study_hours_per_day":             {"label": "Study load",         "emoji": "📚", "avg": 6.5,  "high_is_risk": True},
    "sleep_hours_per_day":             {"label": "Sleep hours",        "emoji": "😴", "avg": 7.5,  "high_is_risk": False},
    "social_hours_per_day":            {"label": "Social time",        "emoji": "👥", "avg": 2.0,  "high_is_risk": False},
    "physical_activity_hours_per_day": {"label": "Physical activity",  "emoji": "🏃", "avg": 1.2,  "high_is_risk": False},
    "study_sleep_ratio":               {"label": "Study/sleep ratio",  "emoji": "⚖️", "avg": 0.9,  "high_is_risk": True},
    "active_hours":                    {"label": "Active hours",       "emoji": "⚡", "avg": 7.7,  "high_is_risk": False},
    "rest_ratio":                      {"label": "Rest ratio",         "emoji": "🌙", "avg": 0.7,  "high_is_risk": False},
    "productive_vs_leisure":           {"label": "Productivity ratio", "emoji": "🎯", "avg": 2.0,  "high_is_risk": True},
}

# ===== DATABASE =====
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

cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        email      TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()
print("[OK] Users table ready")

# Migrate predictions table: add user_id column if missing
try:
    cursor.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER REFERENCES users(id)")
    conn.commit()
    print("[OK] Added user_id column to predictions")
except Exception:
    pass  # Column already exists


# ===== CHATBOT =====
_RESPONSES = {
    "sleep":         "Fatigue is one of the earliest burnout signals. Try a fixed wind-down time tonight — even 30 minutes earlier than usual can reset your sleep rhythm. Avoid caffeine after 2 pm and keep your room below 19 C.",
    "tired":         "Fatigue often signals your body needs real recovery, not just a short break. Try a 20-minute nap before 3 pm. Check whether you're drinking enough water — dehydration mimics exhaustion.",
    "insomnia":      "Insomnia during exam periods is very common — cortisol spikes block melatonin. Try 4-7-8 breathing as you lie down: inhale 4s, hold 7s, exhale 8s. Do 4 cycles.",
    "focus":         "Try the 45-15 rule: 45 minutes of deep work, then 15 minutes fully off-screen. Remove your phone from the room. Your brain needs actual downtime — not just a different screen.",
    "concentrate":   "Concentration improves with a consistent environment. Use the same desk, the same time each day. Brown noise or lo-fi music (no lyrics) keeps background distraction low without engaging your language centres.",
    "distract":      "Every notification interrupts focus for 23 minutes on average. Turn your phone face-down in another room for your study blocks. Use website blockers for the first 45 minutes of each session.",
    "overwhelm":     "When everything feels like too much, pick ONE task for the next 25 minutes. Write the others down so your brain stops holding them. You don't have to solve everything today.",
    "stress":        "Stress is a signal, not a verdict. Try box breathing right now: inhale 4s, hold 4s, exhale 4s, hold 4s. Do 4 cycles. Then write down the one stressor you actually control today.",
    "pressure":      "Academic pressure is real. Break whatever is looming into the smallest possible next action — not a goal, a physical action. Start with that. Momentum beats motivation every time.",
    "anxious":       "Anxiety often comes from focusing on things outside your control. Try 5-4-3-2-1 grounding: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. Brings you back to now.",
    "anxiety":       "Anxiety often comes from focusing on things outside your control. Try 5-4-3-2-1 grounding: name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste. Brings you back to now.",
    "panic":         "When panic hits: drop your shoulders, unclench your jaw, slow your exhale to double the inhale (breathe in 4s, out 8s). Your nervous system has a brake pedal — the long exhale activates it.",
    "exam":          "Exam pressure is real. Shorter, frequent sessions beat marathon cramming: 3x45 min beats one 3-hour block. Review material within 24 hours of first learning it — that's when memory consolidation happens.",
    "assignment":    "Break the assignment into a list of small, concrete actions. Start with the easiest item to build momentum, then tackle the hardest while your energy is up. Aim for 'good enough to submit' before 'perfect'.",
    "deadline":      "Deadline stress narrows thinking. Write down what actually needs to be done vs what you're worrying about — often the real list is much smaller. Work backward from the deadline: what needs to be done today?",
    "grade":         "Grades feel final but rarely are. Talk to your lecturer if you're struggling — most are more approachable than students expect. One grade doesn't define your trajectory.",
    "fail":          "Failing a paper or assessment is painful but rarely catastrophic. Take a breath, look at your options (resit, resubmit, extension), and act on one of them. Action reduces the anxiety more than thinking does.",
    "burnout":       "Burnout needs real rest, not just a short break. Schedule at least one full day this week with zero academic work. Recovery is not laziness — you can't pour from an empty cup.",
    "exhausted":     "Exhaustion this deep is your body's emergency signal. The immediate priority is sleep, then food, then one manageable task. Nothing else. Give yourself permission to recover.",
    "motivation":    "Low motivation is usually depleted energy, not laziness. Sleep, regular meals, and 10 minutes outside are the fastest rebuilders. Start with the smallest possible first step — even 2 minutes of work.",
    "procrastinat":  "Procrastination is avoidance, usually of discomfort (not the task itself). The 2-minute rule: commit to just 2 minutes on the task. Most people keep going past 2 minutes once started.",
    "lazy":          "Low motivation is usually a sign of depleted energy or unclear next steps — not laziness. Identify the single smallest first action on your task and just do that one thing.",
    "lonely":        "Isolation amplifies every stressor. You don't need big plans — even a 10-minute call with someone you trust measurably shifts your nervous system state.",
    "alone":         "Social connection is a direct stress buffer. Even a voice note to a friend counts. Isolation feels safe when overwhelmed but actually makes the stress compound.",
    "isolated":      "Withdrawal is one of burnout's traps — it removes the social buffering that prevents stress from compounding. Reach out to one person today, even briefly.",
    "sad":           "Low mood during a stressful period is very common and usually temporary. Try logging 3 small things that went okay today — it gently shifts your brain's attentional bias. If it persists, please talk to a counsellor.",
    "depress":       "It's okay to not be okay. What you're feeling is real. Please consider talking to your university counsellor or a trusted person — you don't have to carry this alone, and support is available.",
    "hopeless":      "When things feel hopeless, that's usually a sign of severe depletion — not the truth about your situation. Please reach out to someone you trust, or contact your university's student support service today.",
    "food":          "Skipping meals is a hidden stressor — blood sugar crashes spike cortisol and impair focus. Even a small snack every 4 hours keeps your brain running steadily.",
    "eating":        "Nutrition directly affects mood and concentration. Irregular eating creates blood sugar swings that amplify anxiety. Even simple, easy food eaten consistently makes a measurable difference.",
    "social media":  "Social media use above 2 hours/day is correlated with increased anxiety in students. Try app timers: 30-minute daily limits. Check it at fixed times rather than whenever you feel like it.",
    "phone":         "Every phone check interrupts focus for up to 23 minutes. Put it face-down or in another room during study blocks. You'll get the same information later — with much better focus now.",
}

_DEFAULT = (
    "I hear you. Whatever you're going through, it's okay to feel this way. "
    "Take one slow breath and tell me more — what's been weighing on you most? "
    "I'm here, and there are no wrong answers."
)


def offline_chat(message: str) -> str:
    msg = message.lower()
    for keyword, reply in _RESPONSES.items():
        if keyword in msg:
            return reply
    return _DEFAULT


# ── ML insights helpers ───────────────────────────────────────────────────────

_feature_importance_cache: list | None = None

def get_cached_feature_importance() -> list:
    global _feature_importance_cache
    if _feature_importance_cache is None:
        importances = model.feature_importances_
        total_imp   = sum(importances) or 1.0
        _feature_importance_cache = sorted(
            [
                {
                    "feature":    name,
                    "label":      FEATURE_META[name]["label"],
                    "emoji":      FEATURE_META[name]["emoji"],
                    "importance": round(float(imp / total_imp * 100), 1),
                }
                for name, imp in zip(FEATURE_NAMES, importances)
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )
    return _feature_importance_cache


def compute_trend(user_id: int) -> dict | None:
    cursor.execute(
        "SELECT result FROM predictions WHERE user_id = ? ORDER BY created_at ASC",
        (user_id,),
    )
    rows = cursor.fetchall()
    if len(rows) < 3:
        return None
    risk_map = {"Low": 0, "Medium": 1, "High": 2}
    values   = [risk_map.get(r[0], 1) for r in rows]
    n        = len(values)
    slope    = (values[-1] - values[0]) / max(n - 1, 1)
    direction = "improving" if slope < -0.2 else "worsening" if slope > 0.2 else "stable"
    return {
        "direction": direction,
        "slope":     round(slope, 3),
        "values":    [{"index": i + 1, "risk": v} for i, v in enumerate(values[-10:])],
        "count":     n,
    }


def compute_streak(user_id: int) -> int:
    """Return the current consecutive-day prediction streak for the user.
    A streak ending on either today or yesterday is considered active
    (grace period so morning users aren't penalised)."""
    cursor.execute(
        "SELECT DATE(created_at) AS day FROM predictions "
        "WHERE user_id = ? GROUP BY DATE(created_at) ORDER BY day DESC",
        (user_id,),
    )
    rows = cursor.fetchall()
    if not rows:
        return 0
    today = datetime.now(timezone.utc).date()
    try:
        latest_day = datetime.strptime(rows[0][0], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 0
    # Streak is broken if latest activity is older than yesterday
    if latest_day < today - timedelta(days=1):
        return 0
    streak = 0
    for i, row in enumerate(rows):
        try:
            day = datetime.strptime(row[0], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            break
        if day == latest_day - timedelta(days=i):
            streak += 1
        else:
            break
    return streak

# ── Auth helpers ──────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_token(user_id: int, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRY_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict | None:
    """Returns payload dict or None if invalid/expired."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


def get_user_from_request(request) -> dict | None:
    """Extract and validate Bearer token from request headers."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    return decode_token(auth[7:])


# ===== HELPERS =====

def _find_col(df, keywords):
    """Find a DataFrame column matching any of the given keyword substrings."""
    for kw in keywords:
        for col in df.columns:
            if kw in col.lower():
                return col
    return None


# ===== ROUTES =====

@app.get("/")
def home():
    return {
        "message": "Burnout AI API running",
        "version": "2.1.0-gemini-rest",
        "gemini_enabled": bool(GEMINI_API_KEY),
    }


@app.post("/register")
def register(data: dict):
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not email or not password:
        return {"error": "Email and password are required"}
    if len(password) < 6:
        return {"error": "Password must be at least 6 characters"}

    try:
        hashed = hash_password(password)
        cursor.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email, hashed),
        )
        conn.commit()
        return {"message": "Account created successfully"}
    except Exception:
        return {"error": "Email already registered"}


@app.post("/login")
def login(data: dict):
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    cursor.execute("SELECT id, password_hash FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()

    if not row or not verify_password(password, row[1]):
        return {"error": "Invalid email or password"}

    token = create_token(user_id=row[0], email=email)
    return {"token": token, "email": email}


@app.get("/me")
def me(request: Request):
    user = get_user_from_request(request)
    if not user:
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"id": user["sub"], "email": user["email"]}


@app.get("/insights")
def insights(request: Request):
    result: dict = {
        "feature_importance": get_cached_feature_importance(),
        "trend":   None,
        "streak":  0,
    }
    user = get_user_from_request(request)
    if user:
        result["trend"]  = compute_trend(user["sub"])
        result["streak"] = compute_streak(user["sub"])
    return result


@app.get("/data")
def get_dashboard_data():
    try:
        df = pd.read_csv("student_lifestyle_dataset.csv")
        df.columns = df.columns.str.strip()
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}


@app.get("/stats")
def get_dataset_stats():
    """Pre-computed statistics from the training dataset (student_lifestyle_dataset.csv)."""
    try:
        df = pd.read_csv("student_lifestyle_dataset.csv")
        df.columns = df.columns.str.strip()

        total = len(df)

        study_col    = _find_col(df, ["study"])
        sleep_col    = _find_col(df, ["sleep"])
        gpa_col      = _find_col(df, ["gpa"])
        physical_col = _find_col(df, ["physical"])
        social_col   = _find_col(df, ["social"])
        extra_col    = _find_col(df, ["extracurricular"])
        stress_col   = _find_col(df, ["stress_level", "stress"])

        # Stress counts
        stress_counts = df[stress_col].value_counts().to_dict() if stress_col else {}
        high     = int(stress_counts.get("High", 0))
        moderate = int(stress_counts.get("Moderate", stress_counts.get("Medium", 0)))
        low      = int(stress_counts.get("Low", 0))

        # Study hours distribution
        study_dist = []
        if study_col:
            buckets = {"0-3h": 0, "3-5h": 0, "5-7h": 0, "7-9h": 0, "9+h": 0}
            for v in df[study_col].dropna():
                if   v < 3: buckets["0-3h"] += 1
                elif v < 5: buckets["3-5h"] += 1
                elif v < 7: buckets["5-7h"] += 1
                elif v < 9: buckets["7-9h"] += 1
                else:       buckets["9+h"]  += 1
            study_dist = [{"range": k, "count": v} for k, v in buckets.items()]

        # GPA by stress level
        gpa_by_stress = []
        if gpa_col and stress_col:
            for level in ["Low", "Moderate", "High"]:
                subset = df[df[stress_col] == level][gpa_col].dropna()
                if len(subset) > 0:
                    gpa_by_stress.append({
                        "level": level,
                        "avg_gpa": round(float(subset.mean()), 2),
                        "count": int(len(subset)),
                    })

        # Sleep distribution buckets
        sleep_dist = []
        if sleep_col:
            buckets = {"<5h": 0, "5-6h": 0, "6-7h": 0, "7-8h": 0, "8-9h": 0, "9+h": 0}
            for v in df[sleep_col].dropna():
                if   v < 5: buckets["<5h"]  += 1
                elif v < 6: buckets["5-6h"] += 1
                elif v < 7: buckets["6-7h"] += 1
                elif v < 8: buckets["7-8h"] += 1
                elif v < 9: buckets["8-9h"] += 1
                else:       buckets["9+h"]  += 1
            sleep_dist = [{"range": k, "count": v} for k, v in buckets.items()]

        # Scatter sample: study vs sleep, coloured by stress (80 points)
        scatter_sample = []
        if study_col and sleep_col and stress_col:
            sample = df.sample(min(80, total), random_state=42)
            for _, row in sample.iterrows():
                scatter_sample.append({
                    "study": round(float(row[study_col]), 1),
                    "sleep": round(float(row[sleep_col]), 1),
                    "stress": str(row[stress_col]),
                })

        return {
            "total":        total,
            "high":         high,
            "moderate":     moderate,
            "low":          low,
            "high_pct":     round(high / total * 100, 1) if total else 0,
            "moderate_pct": round(moderate / total * 100, 1) if total else 0,
            "low_pct":      round(low / total * 100, 1) if total else 0,
            "avg_study":    round(float(df[study_col].mean()), 2)    if study_col    else 0,
            "avg_sleep":    round(float(df[sleep_col].mean()), 2)    if sleep_col    else 0,
            "avg_gpa":      round(float(df[gpa_col].mean()), 2)      if gpa_col      else 0,
            "avg_physical": round(float(df[physical_col].mean()), 2) if physical_col else 0,
            "avg_social":   round(float(df[social_col].mean()), 2)   if social_col   else 0,
            "avg_extra":    round(float(df[extra_col].mean()), 2)    if extra_col    else 0,
            "study_dist":   study_dist,
            "sleep_dist":   sleep_dist,
            "gpa_by_stress": gpa_by_stress,
            "scatter_sample": scatter_sample,
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/history")
def get_prediction_history(request: Request):
    """Returns predictions: personal if token present, all rows otherwise."""
    try:
        user = get_user_from_request(request)
        if user:
            cursor.execute(
                "SELECT id, study, sleep, social, physical, result, created_at "
                "FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 200",
                (user["sub"],),
            )
        else:
            cursor.execute(
                "SELECT id, study, sleep, social, physical, result, created_at "
                "FROM predictions ORDER BY created_at DESC LIMIT 200"
            )
        rows = cursor.fetchall()
        return [
            {
                "id":         r[0],
                "study":      r[1],
                "sleep":      r[2],
                "social":     r[3],
                "physical":   r[4],
                "result":     r[5],
                "created_at": r[6],
            }
            for r in rows
        ]
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict")
def predict(data: dict, request: Request):
    try:
        study    = float(data.get("study_hours_per_day", 0))
        sleep    = float(data.get("sleep_hours_per_day", 0))
        social   = float(data.get("social_hours_per_day", 0))
        physical = float(data.get("physical_activity_hours_per_day", 0))

        total = study + sleep + social + physical or 1.0

        feature_vals = {
            "study_hours_per_day":              study,
            "sleep_hours_per_day":              sleep,
            "social_hours_per_day":             social,
            "physical_activity_hours_per_day":  physical,
            "study_sleep_ratio":                study / (sleep + 0.1),
            "active_hours":                     study + physical,
            "rest_ratio":                       (sleep + physical) / total,
            "productive_vs_leisure":            study / (social + physical + 0.1),
        }

        df = pd.DataFrame([feature_vals])

        prediction   = model.predict(df)[0]
        result_label = labels.get(int(prediction), "Unknown")

        # ── Confidence ────────────────────────────────────────────────
        proba      = model.predict_proba(df)[0]
        confidence = round(float(max(proba)) * 100, 1)
        probabilities = {
            "low":    round(float(proba[0]) * 100, 1),
            "medium": round(float(proba[1]) * 100, 1),
            "high":   round(float(proba[2]) * 100, 1),
        }

        # ── Top drivers ───────────────────────────────────────────────
        importances = model.feature_importances_
        total_imp   = sum(importances) or 1.0
        ranked      = sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda x: x[1], reverse=True
        )
        top_drivers = []
        for fname, imp in ranked[:3]:
            meta       = FEATURE_META[fname]
            val        = feature_vals[fname]
            concerning = (val > meta["avg"]) if meta["high_is_risk"] else (val < meta["avg"])
            top_drivers.append({
                "feature":        fname,
                "label":          meta["label"],
                "emoji":          meta["emoji"],
                "value":          round(val, 2),
                "avg":            meta["avg"],
                "direction":      "risk" if concerning else "ok",
                "importance_pct": round(float(imp / total_imp * 100), 1),
            })

        # ── Persist ───────────────────────────────────────────────────
        user    = get_user_from_request(request)
        user_id = user["sub"] if user else None
        cursor.execute(
            "INSERT INTO predictions (study, sleep, social, physical, result, user_id) VALUES (?,?,?,?,?,?)",
            (study, sleep, social, physical, result_label, user_id),
        )
        conn.commit()

        return {
            "prediction":    result_label,
            "risk":          int(prediction),
            "confidence":    confidence,
            "probabilities": probabilities,
            "top_drivers":   top_drivers,
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/plan")
def generate_plan(data: dict):
    risk = int(data.get("risk", 0))
    plans = {
        2: {
            "level": "High Burnout",
            "steps": [
                "Reduce study sessions to 2-hour blocks",
                "Sleep at least 7-8 hours per night",
                "Take a 15-min break every 45 minutes",
                "Add 20 minutes of daily exercise",
                "Schedule one social touchpoint this week",
                "Talk with a mentor or counsellor",
            ],
        },
        1: {
            "level": "Medium Burnout",
            "steps": [
                "Balance study and rest — no sessions over 3 hours",
                "Aim for 7 hours of sleep",
                "Take regular breaks every 50 minutes",
                "Add light physical activity 3x per week",
                "Stay connected with at least one friend weekly",
            ],
        },
        0: {
            "level": "Low Burnout",
            "steps": [
                "Maintain your current study schedule",
                "Keep sleeping 7-8 hours consistently",
                "Take short breaks between study blocks",
                "Exercise at least twice a week",
                "Track mood to catch early warning signs",
            ],
        },
    }
    return {"plan": plans.get(risk, plans[0])}


@app.post("/chat")
def chat(data: dict):
    message = data.get("message", "").strip()
    if not message:
        return {"reply": "I'm here — what's on your mind?"}

    # Try Gemini first, fall back to keyword responses
    reply = gemini_chat(message)
    if reply:
        return {"reply": reply}
    return {"reply": offline_chat(message)}
