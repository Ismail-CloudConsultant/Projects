from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parent.parent / ".env"  # finagent/.env


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    google_cloud_project: str = Field(default="", alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str = Field(default="us-central1", alias="GOOGLE_CLOUD_LOCATION")
    gcs_bucket_uri: str = Field(default="", alias="GCS_BUCKET_URI")

    vector_index_id: str = Field(default="", alias="VECTOR_INDEX_ID")
    vector_index_endpoint_id: str = Field(default="", alias="VECTOR_INDEX_ENDPOINT_ID")
    deployed_index_id: str = Field(default="rag_tool_deployed_index", alias="DEPLOYED_INDEX_ID")

    embedding_model: str = Field(default="gemini-embedding-001", alias="EMBEDDING_MODEL")
    generation_model: str = Field(default="gemini-2.5-flash", alias="GENERATION_MODEL")
    embedding_dimensions: int = Field(default=768, alias="EMBEDDING_DIMENSIONS")

    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=8, alias="TOP_K")
    rrf_ranking_alpha: float = Field(default=0.5, alias="RRF_RANKING_ALPHA")
    embedding_batch_size: int = Field(default=100, alias="EMBEDDING_BATCH_SIZE")
    api_max_retries: int = Field(default=3, alias="API_MAX_RETRIES")

    documents_dir: Path = Field(default=Path("documents"), alias="DOCUMENTS_DIR")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")

    @property
    def index_payload_path(self) -> Path:
        return self.data_dir / "index" / "items.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()

