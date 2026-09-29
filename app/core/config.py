"""Application settings with safe local defaults."""
from __future__ import annotations

import secrets
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from typing import Any

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RESUME_AGENT_", env_file=".env", extra="ignore", case_sensitive=False
    )

    # A local SQLite fallback keeps the first-run API usable before Docker is
    # started; .env.example and production deployments use PostgreSQL.
    database_url: str = Field(
        "sqlite+pysqlite:///./data/resume_agent.db",
        validation_alias=AliasChoices("RESUME_AGENT_DATABASE_URL", "DATABASE_URL"),
    )
    data_root: Path = Path("./data")
    database_echo: bool = False
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    internal_token: str = ""
    ollama_base_url: str = "http://127.0.0.1:11434"
    default_chat_model: str = "qwen2.5:7b"
    default_embedding_model: str = "bge-m3"
    task_timeout_seconds: int = Field(default=300, ge=30, le=3600)
    worker_lease_seconds: int = Field(default=60, ge=10, le=3600)
    worker_heartbeat_seconds: int = Field(default=15, ge=5, le=600)
    worker_takeover_grace_seconds: int = Field(default=30, ge=0, le=3600)
    allow_in_memory_store: bool = True
    # Production/default behavior follows the design contract: generation is
    # blocked until a role-correct chat model has passed its probe.  Tests and
    # explicit development deployments may opt into the deterministic fallback.
    strict_model_gate: bool = True
    external_model_consent_required: bool = True
    context_budget_policy_version: str = "context-budget-v1"
    context_safety_margin_ratio: float = Field(default=0.10, ge=0, le=0.5)
    context_safety_margin_min_tokens: int = Field(default=512, ge=0)
    embedding_chunk_policy_version: str = "embedding-chunk-v1"
    embedding_mode_default: str = "embedding"
    log_retention_days: int = Field(default=30, ge=0)
    debug_model_io: bool = False
    edge_path: str | None = None
    # MCP child processes are local-only and can be disabled for setup/testing;
    # when enabled the FastAPI lifespan owns their start/health/close lifecycle.
    mcp_autostart: bool = True

    @field_validator("data_root", mode="before")
    @classmethod
    def _expand_data_root(cls, value: Any) -> Path:
        return Path(value).expanduser()

    @field_validator("ollama_base_url")
    @classmethod
    def _validate_ollama_url(cls, value: str) -> str:
        value = value.rstrip("/")
        if value.startswith("file:"):
            raise ValueError("file:// URLs are not allowed")
        return value

    @field_validator("edge_path", mode="before")
    @classmethod
    def _normalise_edge_path(cls, value: Any) -> str | None:
        path = str(value or "").strip()
        return path or None

    def ensure_runtime(self) -> "Settings":
        for relative in ("tasks", "templates/cache", "skills", "backups", "logs", "edge-profile"):
            (self.data_root / relative).mkdir(parents=True, exist_ok=True)
        token_path = self.data_root / ".internal-token"
        # Local helpers (Streamlit, smoke tests and CLI commands) share this
        # file.  Reuse an existing generated token so a second process cannot
        # invalidate the token held by an already-running API process.
        if not self.internal_token and token_path.is_file():
            try:
                self.internal_token = token_path.read_text(encoding="utf-8").strip()
            except OSError:
                self.internal_token = ""
        if not self.internal_token:
            self.internal_token = secrets.token_urlsafe(32)

        # Explicit RESUME_AGENT_INTERNAL_TOKEN values intentionally replace the
        # published token.  The atomic replace prevents Streamlit from reading
        # a partially-written credential during API startup.
        fd, temporary_name = tempfile.mkstemp(prefix=".internal-token-", suffix=".tmp", dir=self.data_root)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(self.internal_token, encoding="utf-8")
            temporary.replace(token_path)
        finally:
            temporary.unlink(missing_ok=True)
        return self

    def ensure_data_dirs(self) -> Path:
        """Create and return the configured artifact root."""

        self.ensure_runtime()
        return self.data_root.resolve()

    def public_dict(self) -> dict[str, Any]:
        """Return a settings representation safe for API responses."""

        data = self.model_dump(mode="json")
        data.pop("internal_token", None)
        # Do not expose a password embedded in a local PostgreSQL URL either.
        if data.get("database_url"):
            from sqlalchemy.engine import make_url

            data["database_url"] = make_url(data["database_url"]).render_as_string(
                hide_password=True
            )
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings().ensure_runtime()
