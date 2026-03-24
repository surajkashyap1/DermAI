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
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "dermai_chunks"
    retrieval_dense_model: str = "BAAI/bge-small-en-v1.5"
    retrieval_sparse_model: str = "Qdrant/bm25"
    retrieval_reranker_enabled: bool = True
    retrieval_reranker_model: str = "answerdotai/answerai-colbert-small-v1"
    retrieval_candidate_multiplier: int = 5
    retrieval_candidate_min: int = 12
    vision_api_key: str = ""
    vision_model_id: str = "sreejith782/Dermacare_Skin_Lesion_classification"
    vision_api_base_url: str = "https://router.huggingface.co/hf-inference/models"
    vision_timeout_seconds: float = 60.0


settings = Settings()
