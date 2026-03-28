from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Directory that contains the `app` package (meeting_agent_service root), not process cwd.
_SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DOCUMENT_AGENT_", extra="ignore")

    env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/v1"
    database_url: str = "sqlite:///./document_agent.db"
    redis_url: str = "redis://localhost:6379/0"
    storage_dir: str = "./storage"
    chroma_persist_root: str = ""

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    jwt_secret: str = "dev-secret"
    jwt_algorithm: str = "HS256"
    auth_mode: str = "strict"
    max_upload_mb: int = 50
    max_top_k: int = 8
    default_top_k: int = 4
    chunk_size: int = 1200
    chunk_overlap: int = 150

    @model_validator(mode="after")
    def resolve_paths_under_service_root(self) -> "Settings":
        """Anchor SQLite DB and storage to this service folder so uvicorn cwd does not break writes."""
        sd = Path(self.storage_dir)
        if not sd.is_absolute():
            sd = _SERVICE_ROOT / sd
        try:
            sd.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            print(f"Warning: Could not create storage directory {sd}: {e}. Using /tmp instead.")
            sd = Path("/tmp/doc_agent_storage")
            sd.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "storage_dir", str(sd.resolve()))

        url = self.database_url
        if url.startswith("sqlite:///"):
            path_str = url[len("sqlite:///") :]
            db_path = Path(path_str)
            if not db_path.is_absolute():
                db_path = _SERVICE_ROOT / db_path
            try:
                db_path.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError):
                db_path = Path("/tmp/doc_agent.db")
                db_path.parent.mkdir(parents=True, exist_ok=True)
            resolved = f"sqlite:///{db_path.resolve().as_posix()}"
            object.__setattr__(self, "database_url", resolved)

        cpr = (self.chroma_persist_root or "").strip()
        if cpr:
            cr = Path(cpr)
            if not cr.is_absolute():
                cr = _SERVICE_ROOT / cr
            try:
                cr.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError):
                cr = Path("/tmp/doc_agent_chroma")
                cr.mkdir(parents=True, exist_ok=True)
            object.__setattr__(self, "chroma_persist_root", str(cr.resolve()))
        else:
            object.__setattr__(self, "chroma_persist_root", "")

        return self


settings = Settings()  # type: ignore[call-arg]
