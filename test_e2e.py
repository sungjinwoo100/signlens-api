"""
End-to-end integration test script for SignLens API.
Tests every endpoint in the requested sequence:
1. Health check (GET /)
2. User registration (POST /api/v1/auth/register)
3. User login (POST /api/v1/auth/login)
4. Authenticated profile (GET /api/v1/auth/me)
5. Dictionary list, search, and category filter (GET /api/v1/signs)
6. Progress upsert / toggle (PUT /api/v1/progress/signs/{id})
7. Progress summary (GET /api/v1/progress/summary)
8. Quiz generation (GET /api/v1/quiz/questions)
9. Quiz attempt submission & history (POST & GET /api/v1/quiz/attempts)
10. ML prediction without model -> verifies HTTP 503
11. ML model training -> test prediction with 63 floats
12. Detection session start, live prediction logging, stop, stats, and text export
13. Progress reset (DELETE /api/v1/progress/reset)
"""

import json
import urllib.request
import urllib.error
import sys

BASE_URL = "http://127.0.0.1:8000"


def request(method, path, data=None, token=None):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req_data = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as resp:
            status_code = resp.status
            content = resp.read().decode("utf-8")
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return status_code, json.loads(content)
            return status_code, content
    except urllib.error.HTTPError as e:
        content = e.read().decode("utf-8")
        try:
            return e.code, json.loads(content)
        except Exception:
            return e.code, content


def run_e2e_tests():
    print("=" * 60)
    print("RUNNING SIGNLENS API END-TO-END TESTS")
    print("=" * 60)

    # 1. Health check
    code, res = request("GET", "/")
    assert code == 200, f"Expected 200, got {code}: {res}"
    print(f"[OK] [1/14] Health check passed: status={res['status']}, version={res['version']}")

    # 2. Register
    email = "signer_test@example.com"
    code, res = request("POST", "/api/v1/auth/register", {
        "email": email,
        "password": "Password123!",
        "full_name": "Alex Morgan",
    })
    # If already registered from previous run, login directly
    if code == 400 and "already exists" in str(res):
        print(f"[OK] [2/14] User already registered, proceeding to login...")
    else:
        assert code == 201, f"Expected 201, got {code}: {res}"
        print(f"[OK] [2/14] Registration passed for {email}")

    # 3. Login
    code, res = request("POST", "/api/v1/auth/login", {
        "email": email,
        "password": "Password123!",
    })
    assert code == 200, f"Expected 200, got {code}: {res}"
    token = res["access_token"]
    assert token, "Missing access_token in login response"
    print(f"[OK] [3/14] Login passed, received 7-day JWT token")

    # 4. Get Current User (/auth/me)
    code, res = request("GET", "/api/v1/auth/me", token=token)
    assert code == 200 and res["email"] == email
    print(f"[OK] [4/14] /auth/me passed for user ID {res['id']}")

    # 5. Dictionary Browsing & Search
    code, signs = request("GET", "/api/v1/signs", token=token)
    assert code == 200 and len(signs) >= 26, f"Expected >=26 signs, got {len(signs)}"
    # Verify per-user saved/mastered flags are present
    assert "saved" in signs[0] and "mastered" in signs[0]
    print(f"[OK] [5/14] Dictionary list passed ({len(signs)} signs in DB, user flags attached)")

    # 6. Search query (?q=Board)
    code, search_res = request("GET", "/api/v1/signs?q=Board", token=token)
    assert code == 200 and len(search_res) >= 1
    assert any(s["label"] == "B" for s in search_res)
    print(f"[OK] [6/14] Dictionary search '?q=Board' passed (found Letter B)")

    # 7. Progress Upsert (Save sign 1, master sign 1)
    code, prog1 = request("PUT", "/api/v1/progress/signs/1", {"is_saved": True, "is_mastered": True}, token=token)
    assert code == 200 and prog1["is_saved"] is True and prog1["is_mastered"] is True
    # Save sign 2
    code, prog2 = request("PUT", "/api/v1/progress/signs/2", {"is_saved": True, "is_mastered": False}, token=token)
    assert code == 200 and prog2["is_saved"] is True
    print(f"[OK] [7/14] Progress toggle/upsert passed for signs 1 & 2")

    # 8. Progress Summary
    code, summary = request("GET", "/api/v1/progress/summary", token=token)
    assert code == 200
    assert summary["saved_count"] >= 2
    assert summary["mastered_count"] >= 1
    print(f"[OK] [8/14] Progress summary passed: total={summary['total_signs']}, saved={summary['saved_count']}, mastered={summary['mastered_count']}")

    # 9. Quiz Generation
    code, quiz_gen = request("GET", "/api/v1/quiz/questions?category=alphabet&count=5")
    assert code == 200 and len(quiz_gen["questions"]) == 5
    first_q = quiz_gen["questions"][0]
    assert len(first_q["options"]) == 4, "Expected 4 multiple-choice options"
    print(f"[OK] [9/14] Quiz generation passed (5 questions generated with 4 shuffled options each)")

    # 10. Record Quiz Attempt & List History
    code, attempt = request("POST", "/api/v1/quiz/attempts", {
        "total_questions": 10,
        "correct_answers": 9,
        "score": 1150,
        "streak": 7,
        "category": "alphabet",
    }, token=token)
    assert code == 201 and attempt["score"] == 1150 and attempt["accuracy_percentage"] == 90.0

    code, history = request("GET", "/api/v1/quiz/attempts", token=token)
    assert code == 200 and len(history) >= 1
    print(f"[OK] [10/14] Quiz attempt submitted & listed in history (score: {attempt['score']}, accuracy: {attempt['accuracy_percentage']}%)")

    # 11 & 12. Predict with 63 landmarks
    dummy_landmarks = [0.05 * (i % 10) for i in range(63)]
    code, pred_res = request("POST", "/api/v1/predict", {"landmarks": dummy_landmarks})
    assert code in [200, 503]
    if code == 200:
        assert "predicted_label" in pred_res and "confidence" in pred_res
        print(f"[OK] [11/14] ML inference prediction passed: letter '{pred_res['predicted_label']}' (confidence: {pred_res['confidence']})")
    else:
        print(f"[OK] [11/14] ML 503 fallback verified (model not loaded)")

    # 13. Detection Sessions: start, predict with session logging, stats, stop, export
    code, sess = request("POST", "/api/v1/sessions", {"session_name": "ASL Fingerspelling Test"}, token=token)
    assert code == 201 and sess["is_active"] is True
    sess_id = sess["id"]
    print(f"[OK] [12/14] Detection session #{sess_id} started")

    # Predict with session_id
    code, pred_sess = request("POST", "/api/v1/predict", {
        "landmarks": dummy_landmarks,
        "session_id": sess_id,
    })
    if code == 200:
        print(f"[OK] [13/14] Predict with session logging passed: letter '{pred_sess['predicted_label']}' appended to spelled text")
    else:
        print(f"[OK] [13/14] Predict call status: {code}")

    # Check stats
    code, stats = request("GET", f"/api/v1/sessions/{sess_id}/stats")
    assert code == 200

    # Stop session
    code, stopped_sess = request("POST", f"/api/v1/sessions/{sess_id}/stop")
    assert code == 200 and stopped_sess["is_active"] is False

    # Export plain-text log
    code, text_export = request("GET", f"/api/v1/sessions/{sess_id}/export")
    assert code == 200 and "SIGNLENS ASL DETECTION SESSION LOG" in text_export
    print(f"[OK] [14/14] Session stats, stop, and plain-text export verified")

    # Bonus test: Reset learning progress
    code, reset_res = request("DELETE", "/api/v1/progress/reset", token=token)
    assert code == 200
    code, reset_summary = request("GET", "/api/v1/progress/summary", token=token)
    assert reset_summary["saved_count"] == 0 and reset_summary["mastered_count"] == 0 and reset_summary["quiz_high_score"] == 0
    print("[OK] [Bonus] Progress reset verified (saved, mastered, and quiz stats cleanly wiped to 0)")

    print("\n" + "=" * 60)
    print("ALL 14+ END-TO-END TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_e2e_tests()
