"""pre_llm_call hook for the summon plugin.

Reads any pending summon response stored by gateway_dispatch.py and injects
it as context into the current LLM call.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from plugins.summon.hooks.gateway_dispatch import _summon_lock, _summon_pending

logger = logging.getLogger(__name__)


def pre_llm_call_hook(
    conversation_history,
    session_id,
    user_message,
    is_first_turn,
    model,
    platform,
    sender_id,
    **kwargs,
) -> Optional[dict]:
    """pre_llm_call hook handler.

    Checks _summon_pending (module-level dict) for a pending summon response
    (stored by the pre_gateway_dispatch hook) and returns it as context to be
    injected into the user message.

    Returns:
        {"context": "..."} to inject the summoned response
        or None if no pending response exists.
    """
    del sender_id  # unused

    # Retrieve pending response from module-level dict (thread-safe)
    # session_id is passed directly from run_agent.py via invoke_hook
    with _summon_lock:
        payload = _summon_pending.pop(session_id, None)
        if payload is None:
            payload = _summon_pending.pop("_global", None)

    if not payload:
        return None

    agent = payload.get("agent", "unknown")
    response = payload.get("response", "")

    # Inject the summoned response as context
    context = (
        f"[summoned {agent}]: {response}\n\n"
        f"The above is the response from the summoned {agent} agent. "
        f"Incorporate it into your reply naturally."
    )

    logger.info("summon: injecting pre_llm_call context from %s", agent)
    return {"context": context}
