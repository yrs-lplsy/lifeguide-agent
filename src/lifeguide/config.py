import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    database_url: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    hf_endpoint: str
    embed_cache_dir: Path
    corpus_dir: Path
    data_dir: Path

def settings_from_env() -> Settings:
    repo = Path(__file__).resolve().parents[2]
    return Settings(
        database_url=os.getenv("DATABASE_URL", "postgresql://lifeguide:lifeguide@localhost:5433/lifeguide"),
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", ""),
        hf_endpoint=os.getenv("HF_ENDPOINT", "https://hf-mirror.com"),
        embed_cache_dir=Path(os.getenv("EMBED_CACHE_DIR", Path.home() / ".cache/lifeguide/embed")),
        corpus_dir=Path(os.getenv("CORPUS_DIR", repo / "HowToLiveBetter")),
        data_dir=Path(os.getenv("DATA_DIR", repo / "data")),
    )
