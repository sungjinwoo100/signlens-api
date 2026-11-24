FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# System libraries required by OpenCV & MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install lightweight CPU-only PyTorch first, then remaining backend dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    grep -v -E '^(torch|kagglehub|matplotlib|pandas|tqdm)' requirements.txt > requirements.prod.txt && \
    pip install --no-cache-dir -r requirements.prod.txt

# Copy backend code, seed script, and trained ML + MediaPipe models
COPY app ./app
COPY seed_data.py .
COPY ml/models ./ml/models

ENV ML_MODEL_PATH="ml/models/asl_mlp.pt" \
    MEDIAPIPE_MODEL_PATH="ml/models/hand_landmarker.task" \
    CORS_ORIGINS='["*"]'

# Seed initial ASL dictionary signs into SQLite if DB is fresh
RUN python seed_data.py || true

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
