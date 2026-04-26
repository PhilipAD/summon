"""Hermes Agent "Summon" plugin.

Allows users to summon external AI agents (Claude, Grok, Perplexity, GPT,
DeepSeek, Gemini) mid-conversation via `>agent_name <query>` syntax, or by
calling the `summon_agent` tool explicitly.

Two-layer design:
  Layer 1 — pre_gateway_dispatch hook: intercepts >agent patterns, calls the
    provider API, stores the response, and returns {"action": "skip"}.
  Layer 2 — pre_llm_call hook: reads the stored response and injects it as
    context into the next LLM turn.
  Fallback — summon_agent tool: available when the >pattern doesn't fire.
"""

from __future__ import annotations

from plugins.summon.hooks.gateway_dispatch import gateway_dispatch_hook
from plugins.summon.hooks.llm import pre_llm_call_hook
from plugins.summon.tools.summon_tool import SUMMON_AGENT_SCHEMA, handle_summon_agent

__all__ = ["register"]


def register(ctx) -> None:
    """Register all summon plugin hooks and tools with the PluginContext."""
    # Register the pre_gateway_dispatch hook (async — runs before Hermes dispatches)
    ctx.register_hook("pre_gateway_dispatch", gateway_dispatch_hook)

    # Register the pre_llm_call hook (reads pending response, injects context)
    ctx.register_hook("pre_llm_call", pre_llm_call_hook)

    # Register the summon_agent tool
    ctx.register_tool(
        name="summon_agent",
        toolset="summon",
        schema=SUMMON_AGENT_SCHEMA,
        handler=handle_summon_agent,
        check_fn=None,
        emoji="🔮",
    )
