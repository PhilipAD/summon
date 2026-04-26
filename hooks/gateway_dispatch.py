"""pre_gateway_dispatch hook for the summon plugin.

Intercepts messages matching the >agent_name <query> syntax, calls the
appropriate provider API with full session context, stores the response
in the session store, and returns {"action": "skip"} to suppress Hermes's
normal dispatch.

The response is then picked up by the pre_llm_call hook and injected as
context into the next LLM call.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Any, Optional

from plugins.summon.providers import (
    SummonAuthError,
    SummonProviderError,
    SummonRateLimitError,
    SummonTimeoutError,
    get_provider,
)

logger = logging.getLogger(__name__)

# Module-level store for cross-hook communication (thread-safe)
# Bridges pre_gateway_dispatch → pre_llm_call since session_store is not
# passed to the latter by run_agent.py
_summon_pending: dict[str, dict] = {}
_summon_lock = threading.Lock()

# Pattern: >agent_name <query>
_SUMMON_PATTERN = re.compile(r"^>\s*([a-zA-Z0-9_-]+)\s+(.+)$", re.DOTALL)


async def gateway_dispatch_hook(event, gateway, session_store, **kwargs) -> Optional[dict]:
    """pre_gateway_dispatch hook handler.

    1. Parse >agent_name <query> from event.text
    2. Look up the provider by alias
    3. Build conversation context from session history
    4. Call the provider API asynchronously
    5. Store the response in session_store
    6. Return {"action": "skip"} so Hermes doesn't also process this message

    Returns None when the message doesn't match the summon pattern,
    allowing Hermes to handle it normally.
    """
    text = getattr(event, "text", None) or ""
    text = text.strip()
    if not text:
        return None

    match = _SUMMON_PATTERN.match(text)
    if not match:
        return None

    agent_alias = match.group(1).strip().lower()
    query = match.group(2).strip()

    provider = get_provider(agent_alias)
    if provider is None:
        # Unknown alias — let Hermes handle it normally
        logger.debug("summon: unknown alias '%s', deferring to normal dispatch", agent_alias)
        return None

    # Build context from session history
    try:
        context = _build_context(session_store)
    except Exception as exc:
        logger.warning("summon: failed to build context: %s", exc)
        context = []

    # Run the synchronous provider call in a thread pool so we don't block
    # the gateway's async event loop.
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None, _call_provider_sync, provider, query, context
        )
    except SummonAuthError as exc:
        response = f"[summon:{agent_alias}] Configuration error: {exc}"
        logger.warning("summon: auth error for %s: %s", agent_alias, exc)
    except SummonTimeoutError as exc:
        response = f"[summon:{agent_alias}] Timeout: {exc}"
        logger.warning("summon: timeout for %s: %s", agent_alias, exc)
    except SummonRateLimitError as exc:
        response = f"[summon:{agent_alias}] Rate limit: {exc}"
        logger.warning("summon: rate limit for %s: %s", agent_alias, exc)
    except SummonProviderError as exc:
        response = f"[summon:{agent_alias}] Provider error: {exc}"
        logger.warning("summon: provider error for %s: %s", agent_alias, exc)
    except Exception as exc:
        response = f"[summon:{agent_alias}] Unexpected error: {exc}"
        logger.error("summon: unexpected error for %s: %s", agent_alias, exc)

    # Derive session_id for cross-hook coordination
    session_id = None
    try:
        if gateway and hasattr(gateway, "_session_key_for_source") and event.source:
            session_id = gateway._session_key_for_source(event.source)
    except Exception as exc:
        logger.debug("summon: could not derive session_id: %s", exc)

    # Store the response in _summon_pending for the pre_llm_call hook
    _store_pending_response(session_store, agent_alias, query, response, session_id=session_id)

    logger.info("summon: %s responded: %.80s...", agent_alias, response[:80])
    return {"action": "skip", "reason": f"summon:{agent_alias}"}


def _call_provider_sync(provider, query: str, context: list[dict]) -> str:
    """Synchronous wrapper — runs provider.call() in the thread pool."""
    return provider.call(query, context)


def _build_context(session_store) -> list[dict]:
    """Extract message history from session_store as a list of {role, content} dicts."""
    if session_store is None:
        return []

    # Try to access the session entry's message history
    try:
        # session_store is a SessionStore with _entries dict
        entries = getattr(session_store, "_entries", None)
        if entries is None:
            return []

        # Find the most recent session entry
        # We use the most recently updated entry
        active_entry = None
        latest_mtime = 0
        for key, entry in entries.items():
            mtime = getattr(entry, "updated_at", 0) or 0
            if mtime >= latest_mtime:
                latest_mtime = mtime
                active_entry = entry

        if active_entry is None:
            return []

        # Extract messages from the entry
        messages = getattr(active_entry, "messages", None) or []
        # messages is a list of dicts with role/content
        result = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant", "system"):
                result.append({"role": role, "content": content})
        return result
    except Exception as exc:
        logger.debug("summon: could not build context: %s", exc)
        return []


def _store_pending_response(
    session_store, agent_alias: str, query: str, response: str, session_id: Optional[str] = None
) -> None:
    """Store pending response in _summon_pending for cross-hook communication.

    Uses a module-level dict keyed by session_id to bridge pre_gateway_dispatch
    (which receives session_id from the gateway) to pre_llm_call (which receives
    session_id from run_agent.py). This avoids the need for session_store to be
    passed to pre_llm_call.
    """
    payload = {
        "agent": agent_alias,
        "query": query,
        "response": response,
        "timestamp": time.time(),
    }

    # Use thread-safe module-level store keyed by session_id
    with _summon_lock:
        key = session_id or "_global"
        _summon_pending[key] = payload
