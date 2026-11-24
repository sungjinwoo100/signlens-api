from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router
from app.core.config import settings
from app.db.database import init_db
from app.services.ml_service import ml_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("signlens")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes database tables on startup and loads the ML classifier.
    """
    logger.info("Initializing SignLens API database tables...")
    # NOTE: In production environments, replace Base.metadata.create_all() with Alembic migrations.
    init_db()

    logger.info("Loading ML detection model...")
    ml_service.load_model()

    logger.info("SignLens API is ready to serve requests.")
    yield
    logger.info("Shutting down SignLens API...")
    ml_service.close()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Backend API for SignLens: serves the ASL Dictionary, Flashcards, Quiz, "
        "User Progress tracking, and real-time hand sign detection inference."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all API endpoints under version prefix
app.include_router(api_router, prefix=settings.API_V1_STR)
# Convenience alias at /predict in addition to /api/v1/predict
from app.api.routes.predict import router as predict_router
app.include_router(predict_router)


@app.get("/", tags=["Health"])
def root():
    """
    Root health check endpoint.
    """
    return {
        "status": "healthy",
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "api_v1": settings.API_V1_STR,
        "ml_model_loaded": ml_service.is_loaded,
        "mediapipe_loaded": ml_service.mediapipe_loaded,
    }
