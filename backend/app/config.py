"""
app/config.py — Analytiq backend configuration.
All values overridable via environment variables (.env supported).
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class AppConfig(BaseSettings):
    # ── LLM providers ────────────────────────────────────
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    llm_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-2.0-flash"
    gemini_embed_model: str = "text-embedding-004"
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

    # ── App meta ─────────────────────────────────────────
    app_name: str = "Analytiq"
    app_version: str = "2.0.0"
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # ── Auth: workspace password. Unset → open mode (local dev). ──
    app_password: str = Field(default="", alias="APP_PASSWORD")

    model_config = {
        "extra": "ignore",
        "populate_by_name": True,
        "env_file": ".env",
    }


config = AppConfig()
