"""pre_gateway_dispatch hook for the summon plugin.

Persistent session mode:
  First message in session starts with >agent → session goes into persistent mode
  (all subsequent messages route to that provider until @hermes mention).

One-shot mode:
  Mid-conversation >agent → provider replies once, back to Hermes.

@hermes mention breaks persistent mode and returns to normal Hermes dispatch.
>different-agent in persistent mode switches provider.
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

# ---------------------------------------------------------------------------
# Cross-hook communication (_summon_pending → pre_llm_call)
# ---------------------------------------------------------------------------
_summon_pending: dict[str, dict] = {}
_summon_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Persistent session state
# ---------------------------------------------------------------------------
_SESSION_STATE: dict[str, dict] = {}
_SESSION_STATE_LOCK = threading.Lock()

_PERSISTENT_TIMEOUT = 1800  # 30 min

# ---------------------------------------------------------------------------
# Bot name detection (cached)
# ---------------------------------------------------------------------------
_BOT_NAMES: set[str] | None = None
_BOT_NAMES_LOCK = threading.Lock()


def _get_bot_names() -> set[str]:
    global _BOT_NAMES
    if _BOT_NAMES is None:
        with _BOT_NAMES_LOCK:
            if _BOT_NAMES is not None:
                return _BOT_NAMES
            try:
                from hermes_cli.config import load_config
                config = load_config()
                agent_name = config.get("display", {}).get("agent_name", "Hermes")
            except Exception:
                agent_name = "Hermes"
            _BOT_NAMES = {agent_name.lower(), "hermes", "assistant", "bot"}
    return _BOT_NAMES


def _is_bot_mention(text: str) -> bool:
    """Return True if text starts with @bot_name for any known bot name."""
    lower = text.strip().lower()
    return any(lower.startswith(f"@{name}") for name in _get_bot_names())


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------
_SUMMON_PATTERN = re.compile(r"^>\s*([a-zA-Z0-9_-]+)\s+(.+)$", re.DOTALL)


# ---------------------------------------------------------------------------
# Session ID resolution
# ---------------------------------------------------------------------------
def _resolve_session_id(gateway, event) -> str | None:
    """Derive a stable session_id from gateway + event."""
    # Try gateway's session key function first
    if gateway and hasattr(gateway, "_session_key_for_source"):
        try:
            source = getattr(event, "source", None)
            if source is not None:
                return gateway._session_key_for_source(source)
        except Exception:
            pass

    # Fallback: platform + chat_id
    try:
        source = getattr(event, "source", None)
        if source is not None:
            plat = getattr(source, "platform", None)
            chat = getattr(source, "chat_id", None)
            if plat and chat:
                return f"{plat}:{chat}"
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# First-message detection
# ---------------------------------------------------------------------------
def _count_user_messages(session_store, session_id: str) -> int:
    """Count non-system user messages in the session store."""
    if session_store is None:
        return 0
    try:
        entries = getattr(session_store, "_entries", None)
        if not entries:
            return 0
        entry = entries.get(session_id)
        if entry is None:
            return 0
        messages = getattr(entry, "messages", None) or []
        count = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            if role == "user" and not msg.get("content", "").startswith(">"):
                count += 1
        return count
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------
def _build_context(session_store) -> list[dict]:
    """Extract message history from session_store."""
    if session_store is None:
        return []
    try:
        entries = getattr(session_store, "_entries", None)
        if not entries:
            return []
        active_entry = None
        latest_mtime = 0
        for key, entry in entries.items():
            mtime = getattr(entry, "updated_at", 0) or 0
            if mtime >= latest_mtime:
                latest_mtime = mtime
                active_entry = entry
        if active_entry is None:
            return []
        messages = getattr(active_entry, "messages", None) or []
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


def _build_extended_context(session_store, state: dict | None) -> list[dict]:
    """Build context for persistent-mode messages: session history + provider history."""
    ctx = _build_context(session_store)
    if state and state.get("history"):
        ctx = ctx + state["history"]
    return ctx


# ---------------------------------------------------------------------------
# Provider call wrapper (runs in thread pool)
# ---------------------------------------------------------------------------
async def _call_provider_async(provider, query: str, context: list[dict]) -> str:
    """Run a synchronous provider call in the async event loop's thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, provider.call, query, context)


# ---------------------------------------------------------------------------
# Pending response store
# ---------------------------------------------------------------------------
def _store_pending_response(session_id: str | None, agent: str, query: str, response: str) -> None:
    """Store response for pre_llm_call hook to pick up."""
    payload = {
        "agent": agent,
        "query": query,
        "response": response,
        "timestamp": time.time(),
    }
    with _summon_lock:
        key = session_id or "_global"
        _summon_pending[key] = payload


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _set_session_state(session_id: str, data: dict | None) -> None:
    with _SESSION_STATE_LOCK:
        if data is None:
            _SESSION_STATE.pop(session_id, None)
        else:
            _SESSION_STATE[session_id] = data


def _get_session_state(session_id: str) -> dict | None:
    """Get state with stale check."""
    key = session_id
    with _SESSION_STATE_LOCK:
        state = _SESSION_STATE.get(key)
        if state is None:
            return None

        # Stale check: 30 min
        started = state.get("started_at", 0)
        if started and (time.time() - started) > _PERSISTENT_TIMEOUT:
            _SESSION_STATE.pop(key, None)
            logger.info("summon: cleared stale persistent session %s", key)
            return None

        return state


# ---------------------------------------------------------------------------
# Main hook handler
# ---------------------------------------------------------------------------

async def gateway_dispatch_hook(event, gateway, session_store, **kwargs) -> Optional[dict]:
    """pre_gateway_dispatch hook handler.

    Three flows:
    1. First message starts with >agent → enter persistent mode (all msgs routed)
    2. Mid-conversation >agent → one-shot summon
    3. In persistent mode, no >pattern → still routed to active provider
    4. @bot_name mention in persistent mode → exit, Hermes takes over
    """
    text = getattr(event, "text", None) or ""
    text = text.strip()
    if not text:
        return None

    session_id = _resolve_session_id(gateway, event)
    if not session_id:
        logger.debug("summon: no session_id, falling through")
        return None

    # Check if in persistent session (non-fresh)
    current_state = _get_session_state(session_id)

    # --- Flow A: @mention breaks persistent mode ---
    if current_state and current_state.get("mode") == "persistent" and _is_bot_mention(text):
        _set_session_state(session_id, None)
        logger.info("summon: @mention broke persistent mode for %s", session_id)
        return None  # Hermes handles normally

    # --- Flow B: >agent pattern ---
    match = _SUMMON_PATTERN.match(text)
    if match:
        agent_alias = match.group(1).strip().lower()
        query = match.group(2).strip()
        provider = get_provider(agent_alias)

        if provider is None:
            return None

        # First user message → persistent mode
        user_msg_count = _count_user_messages(session_store, session_id)
        is_first = user_msg_count <= 1  # current message counted already

        if is_first and current_state is None:
            logger.info("summon: entering persistent mode for %s → %s", session_id, agent_alias)
            _set_session_state(session_id, {
                "mode": "persistent",
                "provider_name": agent_alias,
                "provider": provider,
                "history": [],
                "started_at": time.time(),
            })
        elif current_state and current_state.get("mode") == "persistent":
            # In persistent mode, >agent switches provider
            logger.info("summon: switching persistent provider %s → %s", session_id, agent_alias)
            _set_session_state(session_id, {
                "mode": "persistent",
                "provider_name": agent_alias,
                "provider": provider,
                "history": [],
                "started_at": time.time(),
            })

        # Refresh state
        new_state = _get_session_state(session_id)

        # Build context
        if new_state and new_state.get("mode") == "persistent":
            context = _build_extended_context(session_store, new_state)
        else:
            context = _build_context(session_store)

        # Call provider
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, provider.call, query, context)
        except SummonAuthError as exc:
            response = f"[summon:{agent_alias}] Config error: {exc}"
        except SummonTimeoutError as exc:
            response = f"[summon:{agent_alias}] Timeout: {exc}"
        except SummonRateLimitError as exc:
            response = f"[summon:{agent_alias}] Rate limit: {exc}"
        except SummonProviderError as exc:
            response = f"[summon:{agent_alias}] Error: {exc}"
        except Exception as exc:
            response = f"[summon:{agent_alias}] Unexpected: {exc}"

        # Track history (persistent mode)
        if new_state and new_state.get("mode") == "persistent":
            new_state.setdefault("history", [])
            new_state["history"].append({"role": "user", "content": query})
            new_state["history"].append({"role": "assistant", "content": response})
            # Re-save state
            _set_session_state(session_id, new_state)

        # Store for pre_llm_call
        _store_pending_response(session_id, agent_alias, query, response)

        return {"action": "skip", "reason": f"summon:{agent_alias}"}

    # --- Flow C: In persistent mode (no >pattern) ---
    if current_state and current_state.get("mode") == "persistent":
        provider = current_state.get("provider")
        provider_name = current_state.get("provider_name")
        if not provider:
            return None

        context = _build_extended_context(session_store, current_state)

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, provider.call, text, context)
        except SummonAuthError as exc:
            response = f"[summon:{provider_name}] Config error: {exc}"
        except SummonTimeoutError as exc:
            response = f"[summon:{provider_name}] Timeout: {exc}"
        except SummonRateLimitError as exc:
            response = f"[summon:{provider_name}] Rate limit: {exc}"
        except SummonProviderError as exc:
            response = f"[summon:{provider_name}] Error: {exc}"
        except Exception as exc:
            response = f"[summon:{provider_name}] Unexpected: {exc}"

        # Track history
        current_state.setdefault("history", [])
        current_state["history"].append({"role": "user", "content": text})
        current_state["history"].append({"role": "assistant", "content": response})
        _set_session_state(session_id, current_state)

        _store_pending_response(session_id, provider_name, text, response)

        return {"action": "skip", "reason": f"summon:{provider_name}"}

    # --- Flow D: Normal dispatch ---
    return None