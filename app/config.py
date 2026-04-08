"""Environment and configuration loading.

Settings are read from environment variables (optionally populated from a .env
file in the current working directory or its parents). CLI flags take priority
over env values at runtime.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _load_env_file() -> None:
    """Search CWD and up to three parents for a .env file and load it once."""
    cwd = Path.cwd()
    for directory in [cwd, *cwd.parents[:3]]:
        candidate = directory / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


_load_env_file()


DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
}


@dataclass(frozen=True)
class Settings:
    """User-configurable settings sourced from environment variables."""

    provider: str
    model: str | None
    anthropic_api_key: str | None
    timeout_seconds: float
    max_log_chars: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=os.getenv("EXPLAIN_ERROR_PROVIDER", "anthropic").strip().lower(),
            model=(os.getenv("EXPLAIN_ERROR_MODEL") or "").strip() or None,
            anthropic_api_key=(os.getenv("ANTHROPIC_API_KEY") or "").strip() or None,
            timeout_seconds=float(os.getenv("EXPLAIN_ERROR_TIMEOUT_SECONDS", "30")),
            max_log_chars=int(os.getenv("EXPLAIN_ERROR_MAX_LOG_CHARS", "16000")),
        )

    def api_key_for(self, provider: str) -> str | None:
        if provider.lower() == "anthropic":
            return self.anthropic_api_key
        return None


def resolve_model(settings: Settings, provider: str, override: str | None) -> str:
    """Pick the model for a provider, honoring CLI overrides then env defaults."""
    if override:
        return override
    if settings.model:
        return settings.model
    return DEFAULT_MODELS.get(provider, "")
