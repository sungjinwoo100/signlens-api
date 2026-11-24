# SignLens API 🤟

SignLens API is a robust, modular FastAPI backend that powers two connected applications:
1. **ASL Dictionary & Learning App** (Dictionary search/filtering, Flashcards, Quiz generator, and User Learning Progress).
2. **SignLens Live Detection** (Real-time camera MediaPipe landmark inference, detection session logging, and plain-text export).

---

## 📁 Folder Structure

```
signlens_api/
├── app/
│   ├── main.py                 # FastAPI application, CORS, lifespan startup/shutdown, router mounting
│   ├── core/
│   │   ├── config.py           # Settings loaded from .env via pydantic-settings
│   │   └── security.py         # Password hashing (bcrypt) and 7-day JWT tokens
│   ├── db/
│   │   └── database.py         # SQLAlchemy engine, sessionmaker, Base, and get_db dependency
│   ├── models/                 # SQLAlchemy models (one model per domain file)
│   │   ├── user.py             # User accounts
│   │   ├── sign.py             # ASL Dictionary entries (label, category, tips)
│   │   ├── progress.py         # Per-user saved/mastered state per sign
│   │   ├── quiz.py             # Quiz attempt history (score, streak, accuracy)
│   │   └── session.py          # SignLens detection sessions & spelled text
│   ├── schemas/                # Pydantic request/response validation schemas
│   │   ├── user.py
│   │   ├── sign.py
│   │   ├── progress.py
│   │   ├── quiz.py
│   │   ├── predict.py
│   │   └── session.py
│   ├── api/
│   │   ├── deps.py             # get_current_user (required) & get_optional_user (public)
│   │   ├── router.py           # Central router mounting all endpoint modules
│   │   └── routes/
│   │       ├── auth.py         # /auth/register, /auth/login, /auth/me
│   │       ├── signs.py        # /signs CRUD, search (?q=), and category filters
│   │       ├── progress.py     # /progress/signs/{id}, /progress/summary, /progress/reset
│   │       ├── quiz.py         # /quiz/questions, /quiz/attempts
│   │       ├── predict.py      # /predict 63 landmarks inference
│   │       └── sessions.py     # /sessions start, stop, stats, and plain-text export
│   └── services/
│       └── ml_service.py       # Scikit-learn model loader and predictor
├── ml/
│   └── asl_model.pkl           # Trained scikit-learn model file
├── seed_data.py                # Populates the ASL alphabet (A-Z) with mnemonic guides
├── train_mock_model.py         # Utility script to train & export a baseline classifier
├── requirements.txt
├── .env.example
├── .env
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (or [uv](https://github.com/astral-sh/uv))

### 2. Environment Setup & Dependency Installation

Using `uv` (recommended):
```powershell
# Create virtual environment
uv venv .venv

# Install all dependencies
uv pip install -r requirements.txt
```

Or using standard `pip`:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Initialize & Seed Database
Populate the ASL alphabet (A-Z) with detailed hand formation guides and mnemonics:
```powershell
.venv\Scripts\python seed_data.py
```

### 4. (Optional) Generate Baseline ML Model
```powershell
.venv\Scripts\python train_mock_model.py
```
*(If no model exists, `/api/v1/predict` will cleanly return an HTTP 503 Service Unavailable error explaining that a trained model file is needed).*

### 5. Run the Server
```powershell
.venv\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **http://localhost:8000/docs** in your browser to interact with the Swagger OpenAPI documentation.

---

## 📖 Endpoint Reference Table

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| **GET** | `/` | Public | Health check and API metadata |
| **POST** | `/api/v1/auth/register` | Public | Register new user with email and password |
| **POST** | `/api/v1/auth/login` | Public | Log in user and receive 7-day JWT bearer token |
| **GET** | `/api/v1/auth/me` | Bearer Token | Get currently authenticated user profile |
| **GET** | `/api/v1/signs` | Optional | Browse, search (`?q=`), and filter signs (`?category=`, `?saved_only=`) |
| **GET** | `/api/v1/signs/{id}` | Optional | Get single sign details with per-user saved/mastered flags |
| **POST** | `/api/v1/signs` | Bearer Token | Create new dictionary entry |
| **PUT** | `/api/v1/signs/{id}` | Bearer Token | Update existing dictionary entry |
| **DELETE**| `/api/v1/signs/{id}` | Bearer Token | Delete dictionary entry |
| **PUT** | `/api/v1/progress/signs/{id}` | Bearer Token | Toggle/update `saved` or `mastered` status (upsert) |
| **GET** | `/api/v1/progress/summary` | Bearer Token | Aggregated stats (total signs, saved, mastered, high score, best streak) |
| **DELETE**| `/api/v1/progress/reset` | Bearer Token | Wipe user progress and quiz history |
| **GET** | `/api/v1/quiz/questions` | Public | Generate multiple-choice questions on the fly (rejects if <4 signs) |
| **POST** | `/api/v1/quiz/attempts` | Bearer Token | Record a completed quiz attempt |
| **GET** | `/api/v1/quiz/attempts` | Bearer Token | List past quiz attempts for current user |
| **POST** | `/api/v1/predict` | Public | Predict ASL sign from 63 MediaPipe floats (HTTP 503 if no model) |
| **POST** | `/api/v1/sessions` | Optional | Start a new live hand detection session |
| **GET** | `/api/v1/sessions` | Optional | List detection sessions |
| **GET** | `/api/v1/sessions/{id}` | Public | Get detection session details |
| **POST** | `/api/v1/sessions/{id}/stop`| Public | End an active detection session |
| **GET** | `/api/v1/sessions/{id}/stats`| Public | Get session metrics (duration, avg confidence, spelled letters) |
| **GET** | `/api/v1/sessions/{id}/export`| Public | Export plain-text (`.txt`) log of the session |
