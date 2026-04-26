"""summon_agent tool implementation for the summon plugin."""

from __future__ import annotations

import time
from typing import Optional

from plugins.summon.providers import (
    SummonAuthError,
    SummonProviderError,
    SummonRateLimitError,
    SummonTimeoutError,
    get_provider,
)
from tools.registry import tool_error, tool_result

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

SUMMON_AGENT_SCHEMA = {
    "name": "summon_agent",
    "description": (
        "Summon an external AI agent (Claude, Grok, Perplexity, GPT, DeepSeek, Gemini) "
        "to answer a query using the full conversation context. "
        "The response is injected into the session so Hermes can build on it. "
        "Use when a user says 'use X to do Y' instead of using the >agent syntax."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "description": (
                    "Agent alias: claude, grok, perplexity, gpt, deepseek, gemini "
                    "(and their aliases: claude-code, sonnet, grok-2, sonar, chatgpt, r1, etc.)"
                ),
                "examples": ["claude", "grok", "sonar", "gpt", "deepseek", "gemini"],
            },
            "query": {
                "type": "string",
                "description": "The query or task to send to the summoned agent.",
            },
            "model": {
                "type": "string",
                "description": "Optional: specific model to use (e.g. 'claude-sonnet-4-7-25').",
                "nullable": True,
            },
            "context": {
                "type": "array",
                "description": "Optional: additional context dicts with role/content to prepend.",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["role", "content"],
                },
            },
        },
        "required": ["agent", "query"],
    },
}

# ---------------------------------------------------------------------------
# Supported agents listing (for help)
# ---------------------------------------------------------------------------

SUPPORTED_AGENTS = {
    "claude": {
        "aliases": ["claude", "claude-code", "sonnet", "anthropic", "claude-sonnet"],
        "provider": "Anthropic (Claude API)",
        "default_model": "claude-sonnet-4-7-25",
    },
    "grok": {
        "aliases": ["grok", "grok-2", "grok-3", "xai"],
        "provider": "xAI (Grok API)",
        "default_model": "grok-2",
    },
    "perplexity": {
        "aliases": ["perplexity", "pplx", "sonar"],
        "provider": "Perplexity API",
        "default_model": "sonar",
    },
    "gpt": {
        "aliases": ["gpt", "chatgpt", "o1", "o3", "openai"],
        "provider": "OpenAI (GPT API)",
        "default_model": "gpt-4o",
    },
    "deepseek": {
        "aliases": ["deepseek", "ds", "r1", "v3"],
        "provider": "DeepSeek API",
        "default_model": "deepseek-chat",
    },
    "gemini": {
        "aliases": ["gemini", "google", "gemi"],
        "provider": "Google AI (Gemini API)",
        "default_model": "gemini-2.0-flash",
    },
}


def _format_supported_agents() -> str:
    lines = []
    for name, info in SUPPORTED_AGENTS.items():
        aliases = ", ".join(info["aliases"])
        lines.append(f"  {name}: {info['provider']} (aliases: {aliases})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def handle_summon_agent(args: dict, **kwargs) -> str:
    """Tool handler for summon_agent.

    Called by the tool registry when the agent invokes /summon_agent.
    """
    agent_alias = str(args.get("agent") or "").strip().lower()
    query = str(args.get("query") or "").strip()
    model = args.get("model")
    extra_context = args.get("context") or []

    if not agent_alias:
        return tool_error("'agent' is required (e.g. 'claude', 'grok', 'sonar')")
    if not query:
        return tool_error("'query' is required")

    provider = get_provider(agent_alias)
    if provider is None:
        known = ", ".join(
            alias for info in SUPPORTED_AGENTS.values() for alias in info["aliases"]
        )
        return tool_error(
            f"Unknown agent '{agent_alias}'. Supported: {known}\n\n"
            f"{_format_supported_agents()}"
        )

    # Build context list from extra_context and session history
    context = []
    for msg in extra_context:
        if isinstance(msg, dict) and msg.get("content"):
            context.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

    # Call the provider (blocking — runs in thread pool via tool dispatcher)
    try:
        response = provider.call(query, context, model=model)
        return tool_result({
            "agent": agent_alias,
            "provider": provider.name,
            "query": query,
            "response": response,
            "status": "ok",
        })
    except SummonAuthError as exc:
        return tool_error(str(exc))
    except SummonTimeoutError as exc:
        return tool_error(f"Request timed out: {exc}")
    except SummonRateLimitError as exc:
        return tool_error(f"Rate limit: {exc}")
    except SummonProviderError as exc:
        return tool_error(f"Provider error: {exc}")
    except Exception as exc:
        return tool_error(f"summon_agent failed: {type(exc).__name__}: {exc}")
