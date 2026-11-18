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
    # Anthropic Claude: tried first for chat, narratives and knowledge-base
    # answers when the key is set.
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    claude_model: str = Field(default="claude-opus-5-5", alias="CLAUDE_MODEL")
    claude_timeout_sec: int = Field(default=90, alias="CLAUDE_TIMEOUT")
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
    # 50 MB by default: a 59 MB file was measured at 1 GB of memory while it
    # loads, so the old 200 MB limit needed about 3 GB the server did not
    # have. Raise MAX_FILE_MB only with the memory to match (about 20 MB of
    # RAM per MB of file, on top of 700 MB for the app).
    max_file_mb: int = Field(default=50, alias="MAX_FILE_MB")
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

    # ── Limits (services/limits.py) ──────────────────────
    # Uploads, reports and model training that run at once; the rest queue
    # for up to HEAVY_JOB_WAIT seconds, then are told the server is busy.
    max_heavy_jobs: int = Field(default=2, alias="MAX_HEAVY_JOBS")
    heavy_job_wait_sec: int = Field(default=90, alias="HEAVY_JOB_WAIT")
    # Per client. 0 disables a limit.
    max_storage_mb_per_owner: int = Field(default=1024, alias="MAX_STORAGE_MB")
    max_datasets_per_owner: int = Field(default=50, alias="MAX_DATASETS")
    # Uploads stop while the data disk has less than this free.
    min_free_disk_mb: int = Field(default=300, alias="MIN_FREE_DISK_MB")
    # API requests per caller per minute (0 = unlimited).
    rate_limit_per_minute: int = Field(default=300, alias="RATE_LIMIT_PER_MINUTE")
    # Shown to clients on error screens and the account page.
    support_email: str = Field(default="", alias="SUPPORT_EMAIL")
    # The address clients reach the app at (https://analytics.example.com):
    # used in report links and in the Claude connector address.
    public_url: str = Field(default="", alias="APP_PUBLIC_URL")
    # A snapshot of all data every this many hours, kept on the data disk
    # (0 = off), and how many to keep.
    backup_every_hours: float = Field(default=24, alias="BACKUP_EVERY_HOURS")
    backup_keep: int = Field(default=3, alias="BACKUP_KEEP")
    # Unexpected errors are posted here ({"text": ...}: Slack, Discord
    # "/slack" and Teams-style incoming webhooks all accept it).
    alert_webhook_url: str = Field(default="", alias="ALERT_WEBHOOK_URL")

    # ── App meta ─────────────────────────────────────────
    app_name: str = "Analytiq"
    app_version: str = "2.0.0"
    # Other websites allowed to call the API from a browser. Empty (the
    # default) allows none: the app's own pages are served from the same
    # address and need no entry. "*" allowed every site.
    cors_origins: str = Field(default="", alias="CORS_ORIGINS")

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
