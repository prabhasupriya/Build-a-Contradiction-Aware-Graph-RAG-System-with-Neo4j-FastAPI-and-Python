"""Central, pinned configuration. All model IDs are read here and logged by evaluate.py."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_base_url: str
    llm_model_id: str
    embedding_backend: str
    embedding_model_id: str
    top_k: int
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    dataset_path: Path
    auto_ingest: bool

    @property
    def llm_enabled(self) -> bool:
        key = self.openai_api_key
        return bool(key) and not key.startswith("sk-replace_me")


def get_settings() -> Settings:
    ds = Path(os.getenv("DATASET_PATH", "data/dataset.json"))
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
        llm_model_id=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "local"),
        embedding_model_id=os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small"),
        top_k=int(os.getenv("TOP_K", "3")),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "change_me_please"),
        dataset_path=ds if ds.is_absolute() else ROOT / ds,
        auto_ingest=os.getenv("AUTO_INGEST", "true").lower() in ("1", "true", "yes"),
    )
