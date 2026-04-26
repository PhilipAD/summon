"""AI provider implementations for the summon plugin.

Each provider wraps an external API (Anthropic, xAI, Perplexity, OpenAI,
DeepSeek, Google) and exposes a simple call(query, context, model) -> str
interface.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from plugins.summon.config import get_api_key, get_provider_config

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SummonProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(self, message: str, *, is_retryable: bool = False):
        super().__init__(message)
        self.is_retryable = is_retryable


class SummonAuthError(SummonProviderError):
    """API key missing or invalid."""


class SummonRateLimitError(SummonProviderError):
    """Rate limit hit — worth retrying with backoff."""


class SummonTimeoutError(SummonProviderError):
    """Request timed out."""


# ---------------------------------------------------------------------------
# Base protocol
# ---------------------------------------------------------------------------


class AgentProvider(ABC):
    """Abstract base for an AI agent provider."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical provider name (e.g. 'anthropic')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_aliases(self) -> list[str]:
        """Lowercase aliases that map to this provider."""
        raise NotImplementedError

    @abstractmethod
    def call(self, query: str, context: list[dict], model: Optional[str] = None) -> str:
        """Call the provider with *query* and conversation *context*.

        *context* is a list of message dicts with keys: role, content.
        Returns the plain-text response string.

        Raises SummonAuthError / SummonTimeoutError / SummonProviderError as appropriate.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


def _format_context(context: list[dict]) -> str:
    """Render a conversation history as a plain string for providers that
    don't accept multi-message APIs."""
    if not context:
        return ""
    parts = []
    for msg in context:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") for item in content if item.get("type") == "text"
            )
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------


class AnthropicProvider(AgentProvider):
    name = "anthropic"
    supported_aliases = ["claude", "claude-code", "sonnet", "anthropic", "claude-sonnet"]

    def call(self, query: str, context: list[dict], model: Optional[str] = None) -> str:
        key = get_api_key("anthropic")
        if not key:
            raise SummonAuthError(
                "Claude not configured. Add ANTHROPIC_API_KEY to ~/.hermes/.env"
            )

        cfg = get_provider_config("anthropic")
        model = model or cfg.get("default_model", "claude-sonnet-4-7-25")

        import anthropic

        client = anthropic.Anthropic(api_key=key)

        # Build messages list
        messages = []
        for msg in context:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "") for item in content if item.get("type") == "text"
                )
            if role not in ("user", "assistant"):
                role = "user"
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})

        # Retry loop for rate limits
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=messages,
                )
                # resp.text is a list of content blocks
                if hasattr(resp, "text") and resp.text:
                    if isinstance(resp.text[0], str):
                        return resp.text[0]
                    # blocks have .text attr
                    return getattr(resp.text[0], "text", str(resp.text[0]))
                return str(resp)
            except anthropic.RateLimitError:
                last_exc = anthropic.RateLimitError("rate limited")
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise SummonRateLimitError("Claude rate limit exceeded after 3 retries")
            except anthropic.AuthenticationError:
                raise SummonAuthError(f"Anthropic auth failed: {last_exc or 'check your API key'}")
            except anthropic.Timeout:
                raise SummonTimeoutError("Claude request timed out after 60s")
            except Exception as exc:
                raise SummonProviderError(f"Claude API error: {exc}")

        raise SummonProviderError(str(last_exc))


# ---------------------------------------------------------------------------
# xAI (Grok)
# ---------------------------------------------------------------------------


class XAIProvider(AgentProvider):
    name = "xai"
    supported_aliases = ["grok", "grok-2", "grok-3", "xai", "grok2", "grok3"]

    def call(self, query: str, context: list[dict], model: Optional[str] = None) -> str:
        key = get_api_key("xai")
        if not key:
            raise SummonAuthError("Grok not configured. Add XAI_API_KEY to ~/.hermes/.env")

        cfg = get_provider_config("xai")
        model = model or cfg.get("default_model", "grok-2")
        base_url = cfg.get("base_url", "https://api.x.ai/v1")

        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=base_url)

        messages = []
        for msg in context:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "") for item in content if item.get("type") == "text"
                )
            if role not in ("user", "assistant"):
                role = "user"
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=4096,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                if "rate" in err_str or "429" in err_str:
                    if attempt < 2:
                        time.sleep(2 ** (attempt + 1))
                        continue
                    raise SummonRateLimitError("Grok rate limit exceeded after 3 retries")
                raise SummonProviderError(f"Grok API error: {exc}")

        raise SummonProviderError(str(last_exc))


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


class PerplexityProvider(AgentProvider):
    name = "perplexity"
    supported_aliases = ["perplexity", "pplx", "sonar", "pplx-api"]

    def call(self, query: str, context: list[dict], model: Optional[str] = None) -> str:
        key = get_api_key("perplexity")
        if not key:
            raise SummonAuthError(
                "Perplexity not configured. Add PERPLEXITY_API_KEY to ~/.hermes/.env"
            )

        cfg = get_provider_config("perplexity")
        model = model or cfg.get("default_model", "sonar")
        base_url = cfg.get("base_url", "https://api.perplexity.ai")

        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=base_url)

        messages = []
        for msg in context:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "") for item in content if item.get("type") == "text"
                )
            if role not in ("user", "assistant"):
                role = "user"
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=4096,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                if "rate" in err_str or "429" in err_str:
                    if attempt < 2:
                        time.sleep(2 ** (attempt + 1))
                        continue
                    raise SummonRateLimitError(
                        "Perplexity rate limit exceeded after 3 retries"
                    )
                raise SummonProviderError(f"Perplexity API error: {exc}")

        raise SummonProviderError(str(last_exc))


# ---------------------------------------------------------------------------
# OpenAI (GPT)
# ---------------------------------------------------------------------------


class OpenAIProvider(AgentProvider):
    name = "openai"
    supported_aliases = ["gpt", "chatgpt", "o1", "o3", "openai", "gpt4", "gpt-4o", "gpt-4"]

    def call(self, query: str, context: list[dict], model: Optional[str] = None) -> str:
        key = get_api_key("openai")
        if not key:
            raise SummonAuthError("GPT not configured. Add OPENAI_API_KEY to ~/.hermes/.env")

        cfg = get_provider_config("openai")
        model = model or cfg.get("default_model", "gpt-4o")

        from openai import OpenAI

        client = OpenAI(api_key=key)

        messages = []
        for msg in context:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "") for item in content if item.get("type") == "text"
                )
            if role not in ("user", "assistant"):
                role = "user"
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=4096,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                if "rate" in err_str or "429" in err_str:
                    if attempt < 2:
                        time.sleep(2 ** (attempt + 1))
                        continue
                    raise SummonRateLimitError("OpenAI rate limit exceeded after 3 retries")
                raise SummonProviderError(f"OpenAI API error: {exc}")

        raise SummonProviderError(str(last_exc))


# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------


class DeepSeekProvider(AgentProvider):
    name = "deepseek"
    supported_aliases = ["deepseek", "ds", "r1", "v3", "deepseek-chat"]

    def call(self, query: str, context: list[dict], model: Optional[str] = None) -> str:
        key = get_api_key("deepseek")
        if not key:
            raise SummonAuthError(
                "DeepSeek not configured. Add DEEPSEEK_API_KEY to ~/.hermes/.env"
            )

        cfg = get_provider_config("deepseek")
        model = model or cfg.get("default_model", "deepseek-chat")
        base_url = cfg.get("base_url", "https://api.deepseek.com")

        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=base_url)

        messages = []
        for msg in context:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "") for item in content if item.get("type") == "text"
                )
            if role not in ("user", "assistant"):
                role = "user"
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": query})

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=4096,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                if "rate" in err_str or "429" in err_str:
                    if attempt < 2:
                        time.sleep(2 ** (attempt + 1))
                        continue
                    raise SummonRateLimitError(
                        "DeepSeek rate limit exceeded after 3 retries"
                    )
                raise SummonProviderError(f"DeepSeek API error: {exc}")

        raise SummonProviderError(str(last_exc))


# ---------------------------------------------------------------------------
# Google (Gemini)
# ---------------------------------------------------------------------------


class GoogleProvider(AgentProvider):
    name = "google"
    supported_aliases = ["gemini", "google", "gemi", "gemini-api", "bard"]

    def call(self, query: str, context: list[dict], model: Optional[str] = None) -> str:
        key = get_api_key("google")
        if not key:
            raise SummonAuthError(
                "Gemini not configured. Add GOOGLE_API_KEY to ~/.hermes/.env"
            )

        cfg = get_provider_config("google")
        model = model or cfg.get("default_model", "gemini-2.0-flash")

        try:
            from google import genai
        except ImportError:
            raise SummonProviderError(
                "google-genai package not installed. "
                "Install it with: pip install google-genai"
            )

        client = genai.Client(api_key=key)

        # Build contents — Gemini uses a role-based format
        # Format context as a single string since Gemini API takes structured parts
        context_text = _format_context(context)
        if context_text:
            combined_query = f"Conversation so far:\n{context_text}\n\nCurrent query: {query}"
        else:
            combined_query = query

        # Gemini API uses contents list
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=[{"role": "user", "parts": [{"text": combined_query}]}],
                )
                return resp.text or ""
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                if "rate" in err_str or "429" in err_str:
                    if attempt < 2:
                        time.sleep(2 ** (attempt + 1))
                        continue
                    raise SummonRateLimitError("Gemini rate limit exceeded after 3 retries")
                raise SummonProviderError(f"Gemini API error: {exc}")

        raise SummonProviderError(str(last_exc))


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

_ALL_PROVIDERS: list[AgentProvider] = [
    AnthropicProvider(),
    XAIProvider(),
    PerplexityProvider(),
    OpenAIProvider(),
    DeepSeekProvider(),
    GoogleProvider(),
]


def get_provider(alias: str) -> Optional[AgentProvider]:
    """Return the provider matching *alias*, or None if no match found."""
    alias_lower = alias.lower()
    for provider in _ALL_PROVIDERS:
        if alias_lower in provider.supported_aliases:
            return provider
    return None
