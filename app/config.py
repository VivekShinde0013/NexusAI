from functools import lru_cache
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root relative to app/config.py (which is in app/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    APP_NAME: str = "NexusAI"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LLM_PROVIDER: str = "mock"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_API_KEY: str = ""
    LLM_API_URL: str = "https://api.openai.com/v1/chat/completions"
    LLM_TEMPERATURE: float = 0.0
    LLM_RELEVANCE_THRESHOLD: float = 0.4
    DOCUMENT_SCOPED_RELEVANCE_THRESHOLD: float = 0.25
    LLM_HIGH_CONFIDENCE_THRESHOLD: float = 0.6

    # CORS configurations
    ALLOWED_ORIGINS: str = (
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
    )

    # Upload configurations
    MAX_UPLOAD_SIZE: int = 10485760  # 10 MB in bytes
    UPLOAD_DIR: str = "data/raw"

    # Chunking configurations
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Embedding and Vector configurations
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_STORE_DIR: str = "data/vector_store"

    # Tracing / Observability configurations
    LOGS_DIR: str = "logs"
    LOG_QUERY_PREVIEWS: bool = False

    # Evaluation / Benchmarking configurations
    EVALUATOR_PROVIDER: str = "mock"
    EVALUATOR_MODEL: str = "gpt-4o-mini"
    EVAL_QUALITY_PROVIDER: str = "mock"
    EVAL_QUALITY_MODEL: str = "gpt-4o-mini"
    EVAL_QUALITY_API_KEY: str = ""
    EVAL_QUALITY_TEMPERATURE: float = 0.0
    EVAL_QUALITY_RPM_LIMIT: int = 15
    GEMINI_API_KEY: str = ""
    GEMINI_API_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_allowed_origins(self) -> List[str]:
        if not self.ALLOWED_ORIGINS:
            return []
        if isinstance(self.ALLOWED_ORIGINS, list):
            return self.ALLOWED_ORIGINS
        return [
            origin.strip()
            for origin in self.ALLOWED_ORIGINS.split(",")
            if origin.strip()
        ]

    def get_upload_path(self) -> Path:
        path = Path(self.UPLOAD_DIR)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path

    def get_vector_store_path(self) -> Path:
        path = Path(self.VECTOR_STORE_DIR)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path

    def get_logs_path(self) -> Path:
        path = Path(self.LOGS_DIR)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    # Automatically ensure the directories exist
    settings.get_upload_path().mkdir(parents=True, exist_ok=True)
    settings.get_vector_store_path().mkdir(parents=True, exist_ok=True)
    settings.get_logs_path().mkdir(parents=True, exist_ok=True)
    return settings

