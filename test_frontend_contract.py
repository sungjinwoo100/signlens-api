"""Contract test: replays the exact requests the Flutter client makes
(lib/services/api_service.dart) and asserts the JSON keys the Dart models read.
Run with the API server up:  .venv\\Scripts\\python test_frontend_contract.py
"""
import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000/api/v1"
EMAIL = "contract_test@signlens.ai"
PASSWORD = "secret123"


def call(method, path, body=None, token=None, raw=False):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urllib.request.urlopen(req, data) as r:
            text = r.read().decode()
            return r.status, (text if raw else json.loads(text))
    except urllib.error.HTTPError as e:
        text = e.read().decode()
        try:
            return e.code, json.loads(text)
        except Exception:
            return e.code, text


def need(obj, *keys):
    missing = [k for k in keys if k not in obj]
    assert not missing, f"missing keys {missing} in {obj}"


def ok(msg):
    print(f"[OK] {msg}")


# --- logged-out dictionary: flags must be null -------------------------
st, signs = call("GET", "/signs")
assert st == 200 and len(signs) >= 26
need(signs[0], "id", "label", "category", "description", "hand_formation", "mnemonic_tip", "saved", "mastered")
assert signs[0]["saved"] is None and signs[0]["mastered"] is None
ok(f"logged-out /signs -> {len(signs)} signs, saved/mastered are null")

st, filt = call("GET", "/signs?category=alphabet")
assert st == 200 and all(s["category"] == "alphabet" for s in filt)
st, num = call("GET", "/signs?category=numbers")
assert st == 200
ok(f"category filter ok (alphabet={len(filt)}, numbers={len(num)})")

# --- auth --------------------------------------------------------------
st, _ = call("POST", "/auth/register", {"email": EMAIL, "password": PASSWORD})
assert st in (200, 201, 400)
st, login = call("POST", "/auth/login", {"email": EMAIL, "password": PASSWORD})
assert st == 200
need(login, "access_token", "user")
token = login["access_token"]
st, me = call("GET", "/auth/me", token=token)
need(me, "id", "email")
ok("login + /auth/me")

# --- progress: client sends {"mastered": true} / {"saved": true} -------
sid = signs[0]["id"]
st, p = call("PUT", f"/progress/signs/{sid}", {"mastered": True}, token)
assert st == 200 and p["is_mastered"] is True, p
st, p = call("PUT", f"/progress/signs/{sid}", {"saved": True}, token)
assert st == 200 and p["is_saved"] is True and p["is_mastered"] is True, p
ok("PUT /progress/signs/{id} accepts saved/mastered keys")

st, one = call("GET", f"/signs/{sid}", token=token)
assert one["saved"] is True and one["mastered"] is True
st, only = call("GET", "/signs?saved_only=true", token=token)
assert [s["id"] for s in only] == [sid]
st, only = call("GET", "/signs?mastered_only=true", token=token)
assert [s["id"] for s in only] == [sid]
ok("authenticated flags + saved_only / mastered_only filters")

st, summ = call("GET", "/progress/summary", token=token)
need(summ, "total_signs", "saved_count", "mastered_count", "quiz_high_score", "best_streak")
assert summ["saved_count"] == 1 and summ["mastered_count"] == 1
ok(f"/progress/summary {summ}")

st, anon = call("GET", "/progress/summary")
assert st == 401
ok("summary without token -> 401 (client shows login prompt)")

# --- quiz --------------------------------------------------------------
st, quiz = call("GET", "/quiz/questions?category=alphabet&count=5")
assert st == 200
need(quiz, "category", "total_questions", "questions")
q = quiz["questions"][0]
need(q, "id", "prompt", "target_sign_id", "target_label", "hand_formation", "options", "correct_sign_id")
assert len(q["options"]) == 4 and all({"sign_id", "label"} <= set(o) for o in q["options"])
assert q["correct_sign_id"] in [o["sign_id"] for o in q["options"]]
ok(f"quiz questions: {quiz['total_questions']} x 4 options")

st, bad = call("GET", "/quiz/questions?category=phrases&count=5")
assert st == 400, (st, bad)
ok(f"empty category -> 400: {bad['detail'][:60]}...")

st, att = call(
    "POST",
    "/quiz/attempts",
    {"category": "alphabet", "score": 450, "total_questions": 5, "correct_answers": 4, "streak": 3},
    token,
)
assert st == 201
need(att, "id", "score", "streak", "accuracy_percentage")
st, summ = call("GET", "/progress/summary", token=token)
assert summ["quiz_high_score"] >= 450 and summ["best_streak"] >= 3
ok(f"attempt saved; summary now high={summ['quiz_high_score']} streak={summ['best_streak']}")

# --- detection session -------------------------------------------------
st, sess = call("POST", "/sessions", {"session_name": "Contract test"}, token)
assert st == 201
need(sess, "id", "session_name", "average_confidence", "total_predictions")
lm = [0.5 + (i % 7) * 0.01 for i in range(63)]
st, pred = call("POST", "/predict", {"landmarks": lm, "session_id": sess["id"]})
if st == 503:
    ok("predict -> 503 (model missing) handled cleanly")
else:
    assert st == 200
    need(pred, "predicted_label", "confidence")
    st, pred = call("POST", "/predict", {"landmarks": lm, "session_id": sess["id"]})
    st, stats = call("GET", f"/sessions/{sess['id']}/stats")
    need(stats, "total_predictions", "average_confidence", "spelled_text", "most_frequent_sign", "duration_seconds")
    assert stats["total_predictions"] == 2 and stats["most_frequent_sign"] is not None
    ok(f"stats {stats['total_predictions']} detections, avg={stats['average_confidence']}, top={stats['most_frequent_sign']}")
    st, txt = call("GET", f"/sessions/{sess['id']}/export", raw=True)
    assert st == 200 and "Most Frequent Sign" in txt
    ok("export returns real plain text log")

st, bad = call("POST", "/predict", {"landmarks": [0.1] * 10})
assert st == 422
ok("predict with wrong landmark count -> 422")

# --- reset -------------------------------------------------------------
st, _ = call("DELETE", "/progress/reset", token=token)
assert st == 200
st, summ = call("GET", "/progress/summary", token=token)
assert summ["saved_count"] == 0 and summ["mastered_count"] == 0 and summ["quiz_high_score"] == 0
ok("reset wipes progress; summary re-fetch shows zeros")

print("\nALL FRONTEND CONTRACT CHECKS PASSED")
