from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_ENV_FILE = Path(__file__).resolve().parents[4] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DERMAI_",
        env_file=ROOT_ENV_FILE,
        extra="ignore",
    )

    env: str = "development"
    version: str = "0.1.0"
    commit_sha: str = "dev"
    service_name: str = "dermai-api"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_base_url: str = "https://api.groq.com/openai/v1"


settings = Settings()
