"""
config.py — Central configuration via Pydantic Settings.
Reads from environment variables / .env file.
"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Gemini AI
    gemini_api_key: str = ""
    gemini_flash_model: str = "gemini-1.5-flash"
    gemini_pro_model: str = "gemini-1.5-pro"
    gemini_max_context_bytes: int = 2048

    # Database
    database_url: str = "sqlite:///./mlforge.db"

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///./mlruns.db"

    # App
    secret_key: str = "dev_secret_key_change_in_prod"
    upload_max_mb: int = 500
    session_timeout_hours: int = 24
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    experiments_dir: str = "./experiments"
    samples_dir: str = "./data/samples"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def experiments_path(self) -> Path:
        p = Path(self.experiments_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def samples_path(self) -> Path:
        return Path(self.samples_dir)

    @property
    def upload_max_bytes(self) -> int:
        return self.upload_max_mb * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    return Settings()
