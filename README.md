![SUMMON](assets/banner.png)

# Hermes Agent "Summon" Plugin

Summon external AI agents mid-conversation and incorporate their responses naturally.

**Persistent mode:** start a session with `>claude write me a script` and every message after goes to Claude until you `@hermes` to step back in.

## Quick Start

### 1. Add API Keys to `~/.hermes/.env`

```bash
# At least one of these
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

### 3. Use It

```
You: >claude write me a fast quicksort in Python
You: >grok explain quantum entanglement
You: >sonar what's the weather in Tokyo
You: >deepseek benchmark this code
You: >gemini summarize this paper
```

Agents respond inline — Hermes sees their response and can build on it.

## How It Works

**Two-layer design:**

1. **`pre_gateway_dispatch` hook** — Intercepts `>agent_name <query>` before Hermes processes the message. Calls the external API with full conversation context, stores the response in a thread-safe module store, and returns `{"action": "skip"}` to suppress Hermes's normal reply.

2. **`pre_llm_call` hook** — Reads the pending response on the next LLM call and injects it as ephemeral context. Hermes sees the summoned response in its context window and can incorporate it naturally.

**Fallback: `summon_agent` tool** — Available when the `>` pattern doesn't fire (e.g., "use Claude to write this").

## Supported Agents

| Agent | Aliases | Provider |
|-------|---------|----------|
| Claude | `claude`, `claude-code`, `sonnet`, `anthropic` | Anthropic |
| Grok | `grok`, `grok-2`, `grok-3`, `xai` | xAI |
| Perplexity | `perplexity`, `pplx`, `sonar` | Perplexity |
| GPT | `gpt`, `chatgpt`, `o1`, `o3`, `openai` | OpenAI |
| DeepSeek | `deepseek`, `ds`, `r1`, `v3` | DeepSeek |
| Gemini | `gemini`, `google`, `gemi` | Google AI |

## Required Packages

```bash
pip install anthropic openai google-genai
```

Or let Hermes auto-install them when you first use a provider.

## Architecture

```
plugins/summon/
├── plugin.yaml                  # Manifest (hooks: pre_gateway_dispatch, pre_llm_call)
├── __init__.py                 # register(ctx) — wires hooks + tool
├── config.py                   # API key reading from ~/.hermes/.env
├── providers.py                # Per-provider API clients + factory
│   ├── anthropic.py           # Anthropic (Claude)
│   ├── xai.py                 # xAI (Grok)  
│   ├── perplexity.py          # Perplexity
│   ├── openai.py              # OpenAI (GPT)
│   ├── deepseek.py           # DeepSeek
│   └── google.py             # Google (Gemini)
├── hooks/
│   ├── gateway_dispatch.py    # pre_gateway_dispatch — intercepts >pattern
│   └── llm.py                 # pre_llm_call — injects response as context
└── tools/
    └── summon_tool.py         # summon_agent tool (fallback)
```

## Hermes Plugin Rules

This plugin **does not modify** any core Hermes files. It uses only the public plugin surfaces:

- `ctx.register_hook()` for lifecycle hooks
- `ctx.register_tool()` for tool registration
- Module-level thread-safe dict for cross-hook state

See `hermes_cli/plugins.py` for the full plugin surface.

## Troubleshooting

**Plugin not loading?**
```bash
hermes plugins list
```
Look for `summon` in the list.

**"API key not configured" error?**
Make sure keys are in `~/.hermes/.env`, not `~/.bashrc` or elsewhere. Reload with `export $(grep -v '^#' ~/.hermes/.env | xargs)` or restart Hermes.

**Provider import errors?**
```bash
pip install anthropic openai google-genai
```

**Hook not firing?**
- The `>` prefix is required — no space between `>` and the agent name
- The `pre_gateway_dispatch` hook only fires on gateway platforms (Discord, Telegram, etc.)
- In CLI mode, use the `summon_agent` tool directly
