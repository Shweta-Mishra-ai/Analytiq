"""
app/config.py — Analytiq backend configuration.
All values overridable via environment variables (.env supported).
"""
import logging
from pydantic_settings import BaseSettings
from pydantic import Field

logger = logging.getLogger(__name__)


class AppConfig(BaseSettings):
    # ── LLM providers ────────────────────────────────────
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    llm_model: str = "llama-3.3-70b-versatile"
    # Both of these were already-deprecated/shut-down models as of this
    # update — gemini-2.0-flash retired June 1, 2026, text-embedding-004
    # retired January 14, 2026. Any real GEMINI_API_KEY would have been
    # getting hard 404s from Google for every single call. Current GA
    # (generally available, production) replacements as of Aug 2026:
    gemini_model: str = "gemini-3.6-flash"
    gemini_embed_model: str = "gemini-embedding-001"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_timeout_sec: int = 20
    llm_max_retries: int = 3

    # ── Local / open-source model ────────────────────────
    # Any OpenAI-compatible endpoint: Ollama, llama.cpp's server, vLLM,
    # LM Studio. Point LOCAL_LLM_URL at the base (e.g.
    # http://localhost:11434) and name the model you have pulled.
    #
    # Model choice is the client's, but for narrative work over tabular
    # findings an instruction-tuned model in the 7-14B class is enough on
    # a laptop, and a 30B+ one is noticeably better on a workstation with
    # the memory for it. The app never depends on the model: the domain
    # engines write their findings themselves and the LLM only adds prose.
    local_llm_url: str = Field(default="", alias="LOCAL_LLM_URL")
    # Gemma is the default because it is openly licensed, ships in sizes
    # that run on a laptop as well as a workstation, and is instruction-
    # tuned for exactly this kind of work — turning computed figures into
    # a paragraph. Set LOCAL_LLM_MODEL to whatever tag you have actually
    # pulled; this default only saves naming it when it matches.
    local_llm_model: str = Field(default="gemma3:12b", alias="LOCAL_LLM_MODEL")
    local_llm_timeout_sec: int = Field(default=120, alias="LOCAL_LLM_TIMEOUT")
    # When on, no client data may be sent to a third-party API. Cloud
    # calls are refused rather than skipped, so a misconfiguration is
    # visible immediately instead of showing up as slightly worse prose.
    llm_privacy_mode: bool = Field(default=False, alias="LLM_PRIVACY_MODE")

    # ── Upload limits ────────────────────────────────────
    max_file_mb: int = 200
    max_media_mb: int = 100          # images / video / documents for RAG

    # ── Knowledge base limits ────────────────────────────
    # A knowledge base is held in memory and rewritten to disk on every
    # ingest, so it cannot be allowed to grow without bound: one user
    # uploading a library of PDFs would take the process down for
    # everyone. These are per-owner and enforced at ingest with an error
    # that says which limit was hit, rather than by silent truncation.
    rag_max_kbs_per_owner: int = Field(default=25, alias="RAG_MAX_KBS")
    rag_max_files_per_kb: int = Field(default=100, alias="RAG_MAX_FILES")
    rag_max_chunks_per_kb: int = Field(default=8000, alias="RAG_MAX_CHUNKS")
    max_rows_preview: int = 100_000
    max_rows_llm_context: int = 50
    max_cols_llm_context: int = 20

    # ── Storage ──────────────────────────────────────────
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    data_ttl_days: int = Field(default=30, alias="DATA_TTL_DAYS")
    cleanup_interval_hours: int = Field(default=6, alias="CLEANUP_INTERVAL_HOURS")

    # ── App meta ─────────────────────────────────────────
    app_name: str = "Analytiq"
    app_version: str = "2.0.0"
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # ── Auth ─────────────────────────────────────────────
    # APP_ADMIN_KEY gates account management (POST /api/admin/*): creating
    # and removing client accounts. Each client then logs in with their own
    # username/password (POST /api/auth/login) and gets a token scoped to
    # only their own data. Unset APP_ADMIN_KEY *and* no accounts created yet
    # → single-user open dev mode (unchanged from before), so local
    # development and the test suite keep working with zero setup.
    # APP_PASSWORD is accepted as a fallback name for API_ADMIN_KEY so
    # existing single-workspace deployments don't need to change their env
    # immediately — but every client now needs their own account either way.
    app_admin_key: str = Field(default="", alias="APP_ADMIN_KEY")
    app_password: str = Field(default="", alias="APP_PASSWORD")
    app_secret: str = Field(default="", alias="APP_SECRET")
    token_ttl_days: int = Field(default=30, alias="TOKEN_TTL_DAYS")

    @property
    def effective_admin_key(self) -> str:
        return self.app_admin_key or self.app_password

    model_config = {
        "extra": "ignore",
        "populate_by_name": True,
        "env_file": ".env",
    }


config = AppConfig()
