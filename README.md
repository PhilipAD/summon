![SUMMON](assets/banner.png)

# Hermes Agent "Summon" Plugin

Summon external AI agents (Claude, Grok, Perplexity, GPT, DeepSeek, Gemini) mid-conversation using `>agent <query>` syntax. Their responses are injected naturally into the conversation — Hermes sees them and can build on them.

---

## Usage Modes

The `>` prefix has **two distinct modes** depending on when you use it.

### Persistent Mode — `>` as the first message in a session

Start a fresh conversation with `>agent_name <query>`, and **every message after** goes to that provider automatically. The summoned agent owns the thread until you call Hermes back in.

```
You: >claude write me a python web scraper
     ╰─ Claude responds ─────────────────────────►
You: make it async with aiohttp
     ╰─ Claude responds (still in persistent mode) ►
You: add error handling and logging
     ╰─ Claude responds (still in persistent mode) ►
You: @hermes tell me what you think of this code
     ╰─ Hermes steps back in ───────────────────►
     ╰─ Session returns to normal ──────────────►
You: how does this compare to scraply?
     ╰─ Hermes responds normally ───────────────►
```

**To exit persistent mode:** mention `@hermes` or `@Hermes` — Hermes takes over and the session returns to normal.

**To switch providers mid-persistent:** just use `>grok ...` or `>sonar ...` while in persistent mode. The session switches to the new provider.

### One-Shot Mode — `>` mid-conversation

Use `>agent <query>` during a normal Hermes conversation for a single summon. The agent replies once, then control returns to Hermes.

```
You: what do you think about quantum computing?
     ╰─ Hermes responds ─────────────────────────►
You: >perplexity search the latest quantum breakthroughs
     ╰─ Perplexity replies once ────────────────►
     ╰─ Hermes sees the response and can build on it
You: that's interesting, tell me more
     ╰─ Hermes continues normally ───────────────►
You: >grok explain quantum entanglement like I'm five
     ╰─ Grok replies once ──────────────────────►
     ╰─ Hermes can reference it next turn
```

### Quick Reference

| Situation | What happens |
|---|---|
| `>claude ...` as the **first** message in a channel | **Persistent** — Claude owns the thread |
| `>claude ...` mid-conversation **(after Hermes has replied)** | **One-shot** — Claude replies once, back to Hermes |
| `>grok ...` during a persistent Claude session | **Switches** persistent provider to Grok |
| Normal chat during persistent mode | Routed to current provider automatically |
| `@hermes` during persistent mode | **Exits** persistent, Hermes takes over |
| Normal chat in one-shot mode | Routes to Hermes as usual |
| 30 minutes idle in persistent mode | **Auto-clears**, next message goes to Hermes |

---

## Quick Start

### 1. Add API Keys to `~/.hermes/.env`

```bash
# Add at least one of these:
ANTHROPIC_API_KEY=sk-ant-...       # Claude
XAI_API_KEY=xai-...               # Grok
PERPLEXITY_API_KEY=pplx-...       # Perplexity
OPENAI_API_KEY=sk-...             # GPT
DEEPSEEK_API_KEY=dsk-...          # DeepSeek
GOOGLE_API_KEY=AI...              # Gemini
```

### 2. Enable the Plugin

Add to `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - summon
```

### 3. Install Dependencies

```bash
pip install anthropic openai google-genai
```

### 4. Use It

```
>claude write me a fast quicksort in Python
>grok explain quantum entanglement like I'm five
>sonar what's the weather in Tokyo
>deepseek benchmark this code
>gemini summarize this paper
```

---

## Supported Agents

| Agent | Aliases | Default Model |
|---|---|---|
| **Claude** | `claude`, `claude-code`, `sonnet`, `anthropic`, `claude-sonnet` | `claude-sonnet-4-7-25` |
| **Grok** | `grok`, `grok-2`, `grok-3`, `xai`, `grok2`, `grok3` | `grok-2` |
| **Perplexity** | `perplexity`, `pplx`, `sonar`, `pplx-api` | `sonar` |
| **GPT** | `gpt`, `chatgpt`, `o1`, `o3`, `openai`, `gpt4`, `gpt-4o`, `gpt-4` | `gpt-4o` |
| **DeepSeek** | `deepseek`, `ds`, `r1`, `v3`, `deepseek-chat` | `deepseek-chat` |
| **Gemini** | `gemini`, `google`, `gemi`, `gemini-api`, `bard` | `gemini-2.0-flash` |

> Each provider checks for its API key at call time. Missing keys produce a clear error message.

---

## Architecture

```
plugins/summon/
├── plugin.yaml                  # Manifest (hooks + tool)
├── __init__.py                 # register(ctx) — wires hooks + tool
├── config.py                   # API key reading
├── providers.py                # 6 provider implementations + factory
├── hooks/
│   ├── gateway_dispatch.py    # pre_gateway_dispatch — intercepts >pattern,
│   │                          #   manages persistent session state,
│   │                          #   routes messages to providers
│   └── llm.py                 # pre_llm_call — injects summoned response
│                              #   as context for one-shot mode
└── tools/
    └── summon_tool.py         # summon_agent tool (CLI fallback)
```

### Two-Layer Design

1. **`pre_gateway_dispatch` hook** — Intercepts messages before Hermes processes them. Detects `>agent` patterns, checks persistent session state, calls the external provider API (with full conversation context), stores the response, and returns `{"action": "skip"}` so Hermes doesn't also reply.

2. **`pre_llm_call` hook** — For one-shot mode: reads the pending response and injects it as ephemeral context into the next LLM call so Hermes can build on it naturally.

**State management:** Persistent session state lives in a thread-safe module-level dict (`_SESSION_STATE`) keyed by `session_id`. Cross-hook communication for one-shot mode uses `_summon_pending`. Both are guarded by `threading.Lock()` for concurrent gateway access.

### Hermes Plugin Rules

This plugin **does not modify** any core Hermes files (`run_agent.py`, `cli.py`, `gateway/run.py`, `main.py`). It uses only the public plugin surfaces:

- `ctx.register_hook()` for lifecycle hooks
- `ctx.register_tool()` for tool registration
- Module-level thread-safe dicts for cross-hook state

---

## Troubleshooting

**Plugin not loading?**
```bash
hermes plugins list
```
Look for `summon` in the list. If missing, check `plugins.enabled` in config.yaml contains `summon`.

**"API key not configured" error?**
Make sure keys are in `~/.hermes/.env`, not `~/.bashrc`. Restart Hermes after adding them.

**Provider import errors?**
```bash
pip install anthropic openai google-genai
```

**`>` command not working in CLI mode?**
The `pre_gateway_dispatch` hook only fires on gateway platforms (Discord, Telegram, etc.). In CLI mode, use the `summon_agent` tool explicitly or ask Hermes to summon the agent.

**Persistent mode not ending?**
Mention `@hermes` exactly — spaces before or after are fine, but the `@` prefix is required. If your Hermes agent has a custom name, that works too.

**Session feels stale?**
Persistent sessions auto-clear after 30 minutes of inactivity.