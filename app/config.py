from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0

    chroma_dir: str = "./data/chroma"
    collection_name: str = "documents"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    max_upload_mb: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
