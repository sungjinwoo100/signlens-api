FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# System libraries required by MediaPipe Tasks (CPU/EGL/GLES) and OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight CPU-only PyTorch, headless OpenCV, MediaPipe 0.10.21 (stable on Linux containers), and FastAPI deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir opencv-python-headless "mediapipe==0.10.21" && \
    grep -v -E '^(torch|opencv-python|mediapipe|kagglehub|matplotlib|pandas|tqdm)' requirements.txt > requirements.prod.txt && \
    pip install --no-cache-dir -r requirements.prod.txt

# Copy backend code, seed script, and trained ML + MediaPipe models
COPY app ./app
COPY seed_data.py .
COPY ml/models ./ml/models

ENV ML_MODEL_PATH="ml/models/asl_mlp.pt" \
    MEDIAPIPE_MODEL_PATH="ml/models/hand_landmarker.task" \
    CORS_ORIGINS='["*"]'

# Verify MediaPipe HandLandmarker + PyTorch MLP load cleanly at Docker build time
RUN python -c "from app.services.ml_service import ml_service; ok = ml_service.load_model(); assert ok and ml_service.mediapipe_loaded, f'Load failed: ml={ok} mp={ml_service.mediapipe_loaded} err={ml_service.mediapipe_error}'; ml_service.close(); print('BUILD CHECK OK: ML + MediaPipe loaded!')"

# Seed initial ASL dictionary signs into SQLite if DB is fresh
RUN python seed_data.py || true

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
