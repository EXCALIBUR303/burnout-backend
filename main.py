from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import sqlite3
import bcrypt
import jwt
import os
from datetime import datetime, timedelta, timezone
from feedback import init_schema as _init_feedback, record_feedback, get_stats as _feedback_stats

# ── Gemini AI via REST API (no extra packages — uses stdlib urllib) ────────────
import json
import urllib.request

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Models in priority order — uses the latest available (2.5 Flash is best free-tier)
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-pro-latest",
]
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
GEMINI_SYSTEM = (
    "You are a compassionate AI wellness assistant for Burnout/AI, a student mental-health app at Woxsen University. "
    "Your role: provide empathetic, evidence-based advice about stress, burnout, sleep, study habits, anxiety, and academic pressure. "
    "Keep replies concise (2-4 sentences max), warm, and actionable. "
    "If someone sounds in crisis, gently direct them to the free Woxsen counselling service "
    "(wellness.centre@woxsen.edu.in / 9049980927). "
    "Never diagnose. Remind users you are an AI, not a replacement for professional help. "
    "Speak like a supportive senior student who knows a lot about wellbeing."
)

_last_gemini_error = ""
_working_model = None  # cached once we find a working model

def _try_model(model: str, payload_bytes: bytes) -> str | None:
    url = GEMINI_BASE.format(model=model, key=GEMINI_API_KEY)
    req = urllib.request.Request(
        url, data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()

def gemini_chat(message: str) -> str | None:
    """Call Gemini REST API. Tries models in order, caches the first that works."""
    global _last_gemini_error, _working_model
    if not GEMINI_API_KEY:
        return None

    full_prompt = f"{GEMINI_SYSTEM}\n\nUser: {message}\nAssistant:"
    payload = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 800,
            "temperature": 0.75,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }).encode()

    # Use cached model if already found
    models_to_try = [_working_model] if _working_model else GEMINI_MODELS

    for model in models_to_try:
        try:
            reply = _try_model(model, payload)
            _working_model = model
            _last_gemini_error = f"OK via {model}"
            print(f"✅ Gemini reply via {model}")
            return reply
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            _last_gemini_error = f"{model} HTTP {e.code}: {body[:200]}"
            print(f"⚠️ {model} failed: HTTP {e.code}")
        except Exception as e:
            _last_gemini_error = f"{model} {type(e).__name__}: {e}"
            print(f"⚠️ {model} failed: {e}")

    return None


def gemini_chat_with_context(full_prompt: str) -> str | None:
    """Call Gemini REST API with a pre-built prompt (supports history + user context)."""
    global _last_gemini_error, _working_model
    if not GEMINI_API_KEY:
        return None

    payload = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 512,
            "temperature": 0.75,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }).encode()

    models_to_try = [_working_model] if _working_model else GEMINI_MODELS

    for model in models_to_try:
        try:
            reply = _try_model(model, payload)
            _working_model = model
            _last_gemini_error = f"OK via {model}"
            print(f"✅ Gemini reply via {model}")
            return reply
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            _last_gemini_error = f"{model} HTTP {e.code}: {body[:200]}"
            print(f"⚠️ {model} failed: HTTP {e.code}")
        except Exception as e:
            _last_gemini_error = f"{model} {type(e).__name__}: {e}"
            print(f"⚠️ {model} failed: {e}")

    return None

if GEMINI_API_KEY:
    print(f"✅ Gemini REST chatbot enabled (key length: {len(GEMINI_API_KEY)})")

JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-in-prod")
if JWT_SECRET == "dev-secret-change-in-prod":
    print("⚠️  [security] JWT_SECRET is using the default dev value — set JWT_SECRET env var in Railway!")
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
import numpy as np

_V5_PKL = "stress_model_v5.pkl"
_V4_PKL = "stress_model_v4.pkl"
_V3_PKL = "stress_model_v3.pkl"
_V2_PKL = "stress_model.pkl"

import traceback as _traceback

model = None
_active_pkl = "none"
_load_errors = {}   # {pkl_name: full_traceback} — exposed via /debug/model
_model_ready = False  # set True once loading finishes (kept for API compat)

# Load priority: v5 → v3 (skip v4 — 29 MB pkl causes cold-start OOM/timeout).
# Run synchronously at module level — must complete before uvicorn accepts
# connections, but xgboost/lightgbm are already imported above so this is
# just joblib.load (disk I/O + numpy deserialisation, typically 5-15 s).
print("[model] loading...")
for _pkl in [_V5_PKL, _V3_PKL, _V2_PKL]:
    if os.path.exists(_pkl):
        try:
            model = joblib.load(_pkl)
            _active_pkl = _pkl
            print(f"[model] loaded {_pkl} OK")
            break
        except Exception as _e:
            _tb = _traceback.format_exc()
            _load_errors[_pkl] = _tb[-2000:]
            print(f"[model] failed to load {_pkl}: {type(_e).__name__}: {_e}")
            print(_tb)
    else:
        _load_errors[_pkl] = "FILE_NOT_PRESENT"
        print(f"[model] {_pkl} not present on disk")

_model_ready = True  # loading finished (success or all-failed)

if model is None:
    print("[model] ⚠️  No model loaded — /predict will return error. Check /debug/model.")
else:
    print(f"[model] ✅ serving {_active_pkl}")

# _IS_V5 is re-evaluated in /predict after model loads.
# It's also exposed as a helper below for endpoints that run after loading finishes.
def _is_v5() -> bool:
    return _active_pkl == _V5_PKL

labels = {0: "Low", 1: "Medium", 2: "High"}

# Feature name lists — selected based on which model is loaded.
# v5: 7 objective + 4 subjective + 7 engineered = 18 features
# v2-v4: 7 objective + 7 engineered = 14 features
_V5_FEATURE_NAMES = [
    # ── Objective lifestyle ──
    "study_hours_per_day", "sleep_hours_per_day", "social_hours_per_day",
    "physical_activity_hours_per_day", "gpa_norm",
    "screen_time_hours", "extracurricular_hours",
    # ── Subjective psychological (NEW v5) ──
    "anxiety_norm", "social_support_deficit",
    "career_concern_norm", "mood_norm",
    # ── Engineered ──
    "study_sleep_ratio", "sleep_deficit", "burnout_index",
    "active_hours", "rest_ratio",
    "productive_vs_leisure", "hourly_load",
]
_V2_FEATURE_NAMES = [
    "study_hours_per_day", "sleep_hours_per_day", "social_hours_per_day",
    "physical_activity_hours_per_day",
    "gpa_norm", "screen_time_hours", "extracurricular_hours",
    "study_sleep_ratio", "sleep_deficit", "burnout_index",
    "active_hours", "rest_ratio", "productive_vs_leisure", "hourly_load",
]


def _get_feature_names() -> list:
    """Return the feature list for the currently loaded model."""
    return _V5_FEATURE_NAMES if _is_v5() else _V2_FEATURE_NAMES


# Legacy alias — kept for code that captured FEATURE_NAMES at import time.
# Always access via _get_feature_names() in functions so it reflects the
# model that was actually loaded.
FEATURE_NAMES = _V5_FEATURE_NAMES  # default; overridden in /predict at runtime

FEATURE_META = {
    "study_hours_per_day":             {"label": "Study load",         "emoji": "📚", "avg": 6.5,  "high_is_risk": True},
    "sleep_hours_per_day":             {"label": "Sleep hours",        "emoji": "😴", "avg": 7.5,  "high_is_risk": False},
    "social_hours_per_day":            {"label": "Social time",        "emoji": "👥", "avg": 2.0,  "high_is_risk": False},
    "physical_activity_hours_per_day": {"label": "Physical activity",  "emoji": "🏃", "avg": 1.2,  "high_is_risk": False},
    "gpa_norm":                        {"label": "GPA",                "emoji": "🎓", "avg": 0.7,  "high_is_risk": False},
    "screen_time_hours":               {"label": "Screen time",        "emoji": "📱", "avg": 2.0,  "high_is_risk": True},
    "extracurricular_hours":           {"label": "Extracurricular",    "emoji": "🎭", "avg": 1.5,  "high_is_risk": False},
    # ── Subjective (v5) ──
    "anxiety_norm":                    {"label": "Anxiety level",      "emoji": "😰", "avg": 0.4,  "high_is_risk": True},
    "social_support_deficit":          {"label": "Support deficit",    "emoji": "🤝", "avg": 0.4,  "high_is_risk": True},
    "career_concern_norm":             {"label": "Career worry",       "emoji": "💼", "avg": 0.4,  "high_is_risk": True},
    "mood_norm":                       {"label": "Self-rated stress",  "emoji": "💭", "avg": 0.4,  "high_is_risk": True},
    # ── Engineered ──
    "study_sleep_ratio":               {"label": "Study/sleep ratio",  "emoji": "⚖️", "avg": 0.9,  "high_is_risk": True},
    "sleep_deficit":                   {"label": "Sleep deficit",      "emoji": "🌙", "avg": 0.5,  "high_is_risk": True},
    "burnout_index":                   {"label": "Burnout index",      "emoji": "🔥", "avg": 0.5,  "high_is_risk": True},
    "active_hours":                    {"label": "Active hours",       "emoji": "⚡", "avg": 7.7,  "high_is_risk": False},
    "rest_ratio":                      {"label": "Rest ratio",         "emoji": "💤", "avg": 0.7,  "high_is_risk": False},
    "productive_vs_leisure":           {"label": "Productivity ratio", "emoji": "🎯", "avg": 2.0,  "high_is_risk": True},
    "hourly_load":                     {"label": "Hourly load",        "emoji": "⏰", "avg": 0.6,  "high_is_risk": True},
}

# ===== DATABASE =====
# DB_PATH lets Railway users point to a persistent volume (e.g. /data/burnout.db)
# Set DB_PATH env var in Railway → Settings → Variables, then add a Volume at /data
_DB_PATH = os.environ.get("DB_PATH", "burnout.db")
conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
print(f"[db] using {_DB_PATH}")
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

# Migrate predictions table: add missing columns if they don't exist yet
for _col, _type in [
    ("user_id",     "INTEGER REFERENCES users(id)"),
    ("screen_time", "REAL"),
    ("gpa_norm",    "REAL"),
    ("extra",       "REAL"),
    ("confidence",  "REAL"),
]:
    try:
        cursor.execute(f"ALTER TABLE predictions ADD COLUMN {_col} {_type}")
        conn.commit()
        print(f"[OK] Added column: {_col}")
    except Exception:
        pass  # column already exists

_init_feedback(conn)
print("[OK] Feedback table ready")


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

def _get_feature_importances(m) -> list:
    """Extract feature importances from XGBoost / CalibratedClassifierCV / StackingClassifier."""
    # Plain XGBoost / LightGBM / RandomForest
    if hasattr(m, "feature_importances_"):
        return list(m.feature_importances_)

    # CalibratedClassifierCV — average across folds
    if hasattr(m, "calibrated_classifiers_"):
        all_folds = []
        for c in m.calibrated_classifiers_:
            est = getattr(c, "estimator", None)
            if est is None:
                continue
            # If the wrapped estimator is a StackingClassifier (v4)
            if hasattr(est, "estimators_"):
                # Average importances across all base learners that have them
                base_imps = []
                for base in est.estimators_:
                    if hasattr(base, "feature_importances_"):
                        base_imps.append(np.asarray(base.feature_importances_, dtype=float))
                if base_imps:
                    all_folds.append(np.mean(base_imps, axis=0))
            elif hasattr(est, "feature_importances_"):
                all_folds.append(np.asarray(est.feature_importances_, dtype=float))
        if all_folds:
            return list(np.mean(all_folds, axis=0))

    # Plain StackingClassifier (no calibration wrapper)
    if hasattr(m, "estimators_"):
        base_imps = []
        for base in m.estimators_:
            if hasattr(base, "feature_importances_"):
                base_imps.append(np.asarray(base.feature_importances_, dtype=float))
        if base_imps:
            return list(np.mean(base_imps, axis=0))

    fn = _get_feature_names()
    return [1.0 / len(fn)] * len(fn)


def get_cached_feature_importance() -> list:
    global _feature_importance_cache
    if _feature_importance_cache is None:
        importances = _get_feature_importances(model)
        total_imp   = sum(importances) or 1.0
        fn = _get_feature_names()
        _feature_importance_cache = sorted(
            [
                {
                    "feature":    name,
                    "label":      FEATURE_META[name]["label"],
                    "emoji":      FEATURE_META[name]["emoji"],
                    "importance": round(float(imp / total_imp * 100), 1),
                }
                for name, imp in zip(fn, importances)
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
        "gemini_enabled": bool(GEMINI_API_KEY),
        "model": _active_pkl,
    }


@app.get("/ping")
def ping():
    """Keep-alive endpoint — frontend pings this every 14 min to prevent Railway cold starts."""
    return {"ok": True}


@app.get("/debug/model")
def debug_model():
    """Diagnostic — shows which model loaded and why others failed."""
    import sys
    try:
        import sklearn, xgboost, lightgbm
        versions = {
            "python":   sys.version.split()[0],
            "sklearn":  sklearn.__version__,
            "xgboost":  xgboost.__version__,
            "lightgbm": lightgbm.__version__,
        }
    except Exception as e:
        versions = {"error": str(e)}

    file_listing = {}
    for pkl in ["stress_model_v5.pkl", "stress_model_v4.pkl", "stress_model_v3.pkl", "stress_model.pkl"]:
        if os.path.exists(pkl):
            file_listing[pkl] = {"present": True, "size_mb": round(os.path.getsize(pkl) / 1e6, 2)}
        else:
            file_listing[pkl] = {"present": False}

    return {
        "active_model":  _active_pkl,
        "load_errors":   _load_errors,
        "files_on_disk": file_listing,
        "versions":      versions,
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
    """Returns the logged-in user's predictions only. Guests get an empty list."""
    try:
        user = get_user_from_request(request)
        if not user:
            return []   # privacy: guests cannot see others' data
        cursor.execute(
            "SELECT id, study, sleep, social, physical, result, created_at, "
            "screen_time, gpa_norm, extra, confidence "
            "FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 200",
            (user["sub"],),
        )
        rows = cursor.fetchall()
        return [
            {
                "id":          r[0],
                "study":       r[1],
                "sleep":       r[2],
                "social":      r[3],
                "physical":    r[4],
                "result":      r[5],
                "created_at":  r[6],
                "screen_time": r[7],
                "gpa_norm":    r[8],
                "extra":       r[9],
                "confidence":  r[10],
            }
            for r in rows
        ]
    except Exception as e:
        return {"error": str(e)}


@app.get("/cohort")
def cohort_stats(request: Request):
    """Anonymous aggregate stats for cohort comparison on the Progress page."""
    try:
        # Overall counts by risk level
        cur2 = conn.execute("SELECT result, COUNT(*) FROM predictions GROUP BY result")
        counts = {r[0]: r[1] for r in cur2.fetchall()}
        total  = sum(counts.values()) or 1

        # User's latest risk for percentile
        user = get_user_from_request(request)
        user_rank_pct = None
        if user:
            cur3 = conn.execute(
                "SELECT result FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
                (user["sub"],),
            )
            row = cur3.fetchone()
            if row:
                user_result = row[0]
                # % of users with WORSE (higher) risk than the current user
                risk_order = {"Low": 0, "Medium": 1, "High": 2}
                user_risk  = risk_order.get(user_result, 1)
                worse = sum(v for k, v in counts.items() if risk_order.get(k, 1) > user_risk)
                user_rank_pct = round(worse / total * 100, 1)

        return {
            "total":       int(total),
            "low_count":   int(counts.get("Low",    0)),
            "medium_count":int(counts.get("Medium", 0)),
            "high_count":  int(counts.get("High",   0)),
            "low_pct":     round(counts.get("Low",    0) / total * 100, 1),
            "medium_pct":  round(counts.get("Medium", 0) / total * 100, 1),
            "high_pct":    round(counts.get("High",   0) / total * 100, 1),
            "user_rank_pct": user_rank_pct,  # % of users with worse risk than you
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict")
def predict(data: dict, request: Request):
    if model is None:
        return {"error": "Model not loaded — check /debug/model for details"}
    try:
        # ── Base features (7) ─────────────────────────────────────────
        study    = float(data.get("study_hours_per_day", 0))
        sleep    = float(data.get("sleep_hours_per_day", 0))
        social   = float(data.get("social_hours_per_day", 0))
        physical = float(data.get("physical_activity_hours_per_day", 0))
        gpa_norm = float(data.get("gpa_norm", 0.70))          # normalised 0-1
        screen   = float(data.get("screen_time_hours", 2.0))  # daily screen hours
        extra    = float(data.get("extracurricular_hours", 1.0))

        # ── Subjective features (4) — defaults to mid-scale when not provided ──
        # These come from the new psychological questions in v5; if older
        # frontends don't send them, we fall back to the population average.
        anxiety_norm           = float(data.get("anxiety_norm",          0.4))
        social_support_deficit = float(data.get("social_support_deficit", 0.4))
        career_concern_norm    = float(data.get("career_concern_norm",   0.4))
        mood_norm              = float(data.get("mood_norm",             0.4))

        # ── Clip to training bounds (mirrors trainer.py CLIP) ─────────
        study    = min(max(study,    0), 18)
        sleep    = min(max(sleep,    2), 14)
        social   = min(max(social,   0), 12)
        physical = min(max(physical, 0),  8)
        gpa_norm = min(max(gpa_norm, 0),  1)
        screen   = min(max(screen,   0), 16)
        extra    = min(max(extra,    0),  8)
        # Subjective features all live on [0, 1]
        anxiety_norm           = min(max(anxiety_norm,           0), 1)
        social_support_deficit = min(max(social_support_deficit, 0), 1)
        career_concern_norm    = min(max(career_concern_norm,    0), 1)
        mood_norm              = min(max(mood_norm,              0), 1)

        total = (study + sleep + social + physical) or 1.0

        # ── Engineered features (7) ───────────────────────────────────
        study_sleep_ratio     = study / (sleep + 0.1)
        sleep_deficit         = max(0.0, 7.5 - sleep)
        burnout_index         = study_sleep_ratio * sleep_deficit
        active_hours          = study + physical
        rest_ratio            = (sleep + physical) / total
        productive_vs_leisure = study / (social + physical + 0.1)
        hourly_load           = (study + social + physical + screen) / 16.0

        feature_vals = {
            "study_hours_per_day":             study,
            "sleep_hours_per_day":             sleep,
            "social_hours_per_day":            social,
            "physical_activity_hours_per_day": physical,
            "gpa_norm":                        gpa_norm,
            "screen_time_hours":               screen,
            "extracurricular_hours":           extra,
            "anxiety_norm":                    anxiety_norm,
            "social_support_deficit":          social_support_deficit,
            "career_concern_norm":             career_concern_norm,
            "mood_norm":                       mood_norm,
            "study_sleep_ratio":               study_sleep_ratio,
            "sleep_deficit":                   sleep_deficit,
            "burnout_index":                   burnout_index,
            "active_hours":                    active_hours,
            "rest_ratio":                      rest_ratio,
            "productive_vs_leisure":           productive_vs_leisure,
            "hourly_load":                     hourly_load,
        }

        # Build DataFrame in exact feature order the model was trained on
        df = pd.DataFrame([feature_vals])[_get_feature_names()]

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
        # Use human-readable features only (objective + subjective, never engineered).
        # Rank by how far each value deviates in the risky direction for THIS user.
        BASE_FEATURES = [
            "study_hours_per_day", "sleep_hours_per_day", "social_hours_per_day",
            "physical_activity_hours_per_day", "gpa_norm", "screen_time_hours",
            "extracurricular_hours",
        ]
        # Include subjective features in driver ranking ONLY if v5 model is loaded
        # AND the frontend supplied real values (not defaults)
        if _is_v5():
            for sub in ["anxiety_norm", "social_support_deficit", "career_concern_norm", "mood_norm"]:
                if sub in data:  # only count if frontend explicitly sent it
                    BASE_FEATURES.append(sub)
        driver_scores = []
        for fname in BASE_FEATURES:
            meta = FEATURE_META[fname]
            val  = feature_vals[fname]
            avg  = meta["avg"]
            # How far is this value from the safe side?
            if meta["high_is_risk"]:
                deviation = val - avg          # positive = risky
            else:
                deviation = avg - val          # positive = risky (low when should be high)
            concerning = deviation > 0
            driver_scores.append((fname, deviation, concerning))

        # Sort by deviation descending — most concerning first
        driver_scores.sort(key=lambda x: x[1], reverse=True)

        # Subjective features (and gpa_norm) are 0-1 scale — display as 0-10 / percent
        SCALED_DISPLAY = {"gpa_norm", "anxiety_norm", "social_support_deficit",
                          "career_concern_norm", "mood_norm"}

        top_drivers = []
        for fname, deviation, concerning in driver_scores[:3]:
            meta = FEATURE_META[fname]
            val  = feature_vals[fname]
            if fname in SCALED_DISPLAY:
                display_val = round(val * 10, 1)
                display_avg = round(meta["avg"] * 10, 1)
            else:
                display_val = round(val, 1)
                display_avg = meta["avg"]
            top_drivers.append({
                "feature":   fname,
                "label":     meta["label"],
                "emoji":     meta["emoji"],
                "value":     display_val,
                "avg":       display_avg,
                "direction": "risk" if concerning else "ok",
            })

        # ── Persist ───────────────────────────────────────────────────
        user    = get_user_from_request(request)
        user_id = user["sub"] if user else None
        cursor.execute(
            "INSERT INTO predictions (study, sleep, social, physical, result, user_id, screen_time, gpa_norm, extra, confidence) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (study, sleep, social, physical, result_label, user_id, screen, gpa_norm, extra, confidence),
        )
        conn.commit()
        pred_id = cursor.lastrowid

        # SHAP per-prediction explanation (best-effort — won't crash /predict if unavailable)
        shap_explanation = None
        try:
            from explain import explain_prediction
            shap_explanation = explain_prediction({
                "study_hours_per_day": study, "sleep_hours_per_day": sleep,
                "social_hours_per_day": social, "physical_activity_hours_per_day": physical,
                "gpa_norm": gpa_norm, "screen_time_hours": screen,
                "extracurricular_hours": extra,
            })
        except Exception as _shap_err:
            print(f"[shap] skipped: {_shap_err}")

        return {
            "prediction":    result_label,
            "risk":          int(prediction),
            "confidence":    confidence,
            "probabilities": probabilities,
            "top_drivers":   top_drivers,
            "prediction_id": pred_id,
            "explanation":   shap_explanation,
            "model_version": _active_pkl.replace("stress_model_", "").replace(".pkl", ""),
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
    history = data.get("history", [])       # list of {role, text} dicts
    user_context = data.get("user_context", "")  # e.g. "High Burnout, studies 9h/day"
    if not message:
        return {"reply": "I'm here — what's on your mind?"}

    # Build conversation prompt with context
    context_prefix = ""
    if user_context:
        context_prefix = f"User context: {user_context}\n\n"

    # Build history string (last 6 turns max)
    history_str = ""
    for turn in history[-6:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        history_str += f"{role}: {turn.get('text', '')}\n"

    if history_str:
        full_prompt = f"{GEMINI_SYSTEM}\n\n{context_prefix}Conversation so far:\n{history_str}\nUser: {message}\nAssistant:"
    else:
        full_prompt = f"{GEMINI_SYSTEM}\n\n{context_prefix}User: {message}\nAssistant:"

    reply = gemini_chat_with_context(full_prompt)
    if reply:
        return {"reply": reply}
    return {"reply": offline_chat(message)}


@app.post("/feedback")
def submit_feedback(data: dict, request: Request):
    user = get_user_from_request(request)
    record_feedback(
        conn,
        prediction_id=data.get("prediction_id"),
        user_id=user["sub"] if user else None,
        accurate=int(data.get("accurate", 1)),
        actual_label=data.get("actual_label"),
        notes=data.get("notes"),
    )
    return {"ok": True}


@app.get("/feedback/stats")
def feedback_stats():
    return _feedback_stats(conn)
