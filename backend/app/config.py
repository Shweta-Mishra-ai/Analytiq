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

    # ── Upload limits ────────────────────────────────────
    max_file_mb: int = 200
    max_media_mb: int = 100          # images / video / documents for RAG
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
