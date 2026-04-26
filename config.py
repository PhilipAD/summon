"""API key and provider configuration for the summon plugin.

Reads keys from ~/.hermes/.env (and falls back to os.environ).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

_HERMES_HOME = Path.home() / ".hermes"
_ENV_FILE = _HERMES_HOME / ".env"


def _load_env() -> dict[str, str]:
    """Load key=value pairs from ~/.hermes/.env, merging with os.environ."""
    env = dict(os.environ)
    if _ENV_FILE.exists():
        try:
            text = _ENV_FILE.read_text(encoding="utf-8")
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    env[key.strip()] = val.strip()
        except Exception:
            pass
    return env


_ENV_CACHE: Optional[dict[str, str]] = None


def _get_env() -> dict[str, str]:
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = _load_env()
    return _ENV_CACHE


# ---------------------------------------------------------------------------
# Provider API key mapping
# ---------------------------------------------------------------------------

_PROVIDER_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def get_api_key(provider: str) -> Optional[str]:
    """Return the API key for *provider*, or None if not configured."""
    key_name = _PROVIDER_KEYS.get(provider.lower())
    if not key_name:
        return None
    return _get_env().get(key_name)


def get_provider_config(provider: str) -> dict:
    """Return runtime config for *provider* (model, base_url, etc.)."""
    configs = {
        "anthropic": {
            "models": ["claude-sonnet-4-7-25", "claude-3-5-sonnet-20241022"],
            "default_model": "claude-sonnet-4-7-25",
        },
        "xai": {
            "models": ["grok-2", "grok-3"],
            "default_model": "grok-2",
            "base_url": "https://api.x.ai/v1",
        },
        "perplexity": {
            "models": ["sonar", "sonar-pro"],
            "default_model": "sonar",
            "base_url": "https://api.perplexity.ai",
        },
        "openai": {
            "models": ["gpt-4o", "gpt-4o-mini", "o1", "o3"],
            "default_model": "gpt-4o",
        },
        "deepseek": {
            "models": ["deepseek-chat", "deepseek-coder"],
            "default_model": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
        },
        "google": {
            "models": ["gemini-2.0-flash", "gemini-1.5-flash"],
            "default_model": "gemini-2.0-flash",
        },
    }
    return configs.get(provider.lower(), {})
