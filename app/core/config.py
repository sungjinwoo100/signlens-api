import json
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "SignLens API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Security & Auth (JWT expiry default: 7 days = 10080 minutes)
    SECRET_KEY: str = "signlens-super-secret-key-please-change-in-production-environment"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080

    # Database
    DATABASE_URL: str = "sqlite:///./signlens.db"

    # CORS
    CORS_ORIGINS: Union[List[str], str] = ["*"]

    # ML Model & MediaPipe
    ML_MODEL_PATH: str = "ml/models/asl_mlp.pt"
    MEDIAPIPE_MODEL_PATH: str = "ml/models/hand_landmarker.task"
    MAX_IMAGE_DIM: int = 320

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return ["*"]


settings = Settings()
