from fastapi import APIRouter
from app.api.routes import auth, predict, progress, quiz, sessions, signs

api_router = APIRouter()

# Mount all route domain modules
api_router.include_router(auth.router)
api_router.include_router(signs.router)
api_router.include_router(progress.router)
api_router.include_router(quiz.router)
api_router.include_router(predict.router)
api_router.include_router(sessions.router)
