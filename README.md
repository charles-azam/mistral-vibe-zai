# Mistral Vibe ZAI -- Mistral Vibe fork for GLM-5

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/github/license/charles-azam/mistral-vibe-zai)](https://github.com/charles-azam/mistral-vibe-zai/blob/main/LICENSE)

A fork of [Mistral Vibe](https://github.com/mistralai/mistral-vibe) adapted to run **ZAI's GLM-5** model. Built for benchmarking agentic scaffoldings on [Terminal-Bench 2.0](https://github.com/laude-institute/harbor).

**Benchmark results:** Scored **0.35** on Terminal-Bench, **the highest score among all agents tested** -- beating Claude Code (0.29), Gemini CLI (0.23), and Codex (0.15) using the same model. See the [full writeup](https://github.com/charles-azam/mistral-vibe-zai) for the architecture comparison. <!-- TODO: replace URL with actual article link -->

## Why this fork won

Mistral Vibe has the simplest architecture of the three agents I forked, and it performed the best. A few reasons why:

- **Middleware pipeline** -- features like auto-compaction, price limits, and plan mode are composable middleware rather than being baked into the core loop. They don't interfere with each other.
- **Smart `search_replace` error recovery** -- edits require exact matches, but when one fails, `difflib.SequenceMatcher` finds the closest match and feeds a detailed diff back to the model. The agent loop self-corrects on the next turn rather than giving up -- strict edits, forgiving feedback.
- **Auto-compact context management** -- proactively summarizes the conversation before hitting the context limit, rather than truncating reactively.
- **Error visibility** -- wraps errors in `<vibe_error>` tags and feeds them back to the model, so it can self-correct.
- **Clean adapter pattern** -- adding GLM-5 support took **13 files changed in one commit**. Compare: Codex required deep Rust type changes, Gemini CLI required an 812-line translation layer across 49 files.

## What I changed

Mistral Vibe's Python architecture with clean provider abstractions made this the easiest fork:

- **`ZAIAdapter`** extending `OpenAIAdapter` (via the generic HTTP backend, not the Mistral SDK) -- normalizes tool choice (`"any"` to `"auto"`), injects thinking config and web search into payloads
- **Configuration models** for ZAI's thinking (`enabled`/`disabled`, preserved vs cleared) and web search features
- **`ReasoningEvent`** in the agent loop to surface GLM-5's chain-of-thought output with batched streaming
- **Preserved Thinking** -- reasoning content from previous turns fed back into the next request for cache hits

## Quick install

**Linux and macOS:**

```bash
curl -LsSf https://raw.githubusercontent.com/charles-azam/mistral-vibe-zai/main/scripts/install.sh | bash
```

**Using uv:**

```bash
uv tool install git+https://github.com/charles-azam/mistral-vibe-zai.git
```

**Using pip:**

```bash
pip install git+https://github.com/charles-azam/mistral-vibe-zai.git
```

## Usage

```bash
export ZAI_API_KEY="your_key"

# Interactive (default: GLM-5 with thinking enabled)
vibe

# With a specific agent profile
vibe --agent plan          # read-only exploration
vibe --agent auto-approve  # auto-approve all tools

# Programmatic mode
vibe --prompt "Fix the bug in main.py" --max-turns 10 --output json

# Delegate to a subagent
# Inside vibe, the model can use the `task` tool to spawn explore/plan subagents
```

### ZAI provider config (`~/.vibe/config.toml`)

```toml
[[providers]]
name = "zai-coding"
api_base = "https://api.z.ai/api/coding/paas/v4"
api_key_env_var = "ZAI_API_KEY"
api_style = "zai"
thinking = { type = "enabled", clear_thinking = false }

[[models]]
name = "glm-5"
provider = "zai-coding"
alias = "glm-5"
```


## Features

All features from upstream Mistral Vibe, plus ZAI-specific additions:

**Tools:** `bash` (tree-sitter parsed), `read_file`, `write_file`, `search_replace` (exact match, fuzzy error feedback), `grep` (ripgrep), `ask_user_question`, `todo`, `task` (subagent delegation) + MCP servers

**Agent profiles:** `default` (approval required), `plan` (read-only), `accept-edits` (auto-approve edits), `auto-approve` (auto-approve all), `explore` (subagent for codebase exploration)

**Context management:** Auto-compact middleware triggers LLM summarization at token threshold. Proactive, not reactive.

**Subagents:** `task` tool spawns independent `AgentLoop` instances. Custom agents via TOML in `~/.vibe/agents/`.

**Skills system:** Markdown-based skills with YAML frontmatter in `~/.vibe/skills/` or `.vibe/skills/`. Supports slash commands.

**MCP support:** HTTP, Streamable-HTTP, and STDIO transports. Configure in `config.toml`.

**Sessions:** JSONL session logging with `--continue` / `--resume` for session persistence.

**ACP support:** Works with editors/IDEs that support [Agent Client Protocol](https://agentclientprotocol.com/overview/clients). See [ACP setup docs](docs/acp-setup.md).

## Related

- [Article: I Read the Source Code of Codex, Gemini CLI, and Mistral Vibe](https://github.com/charles-azam/mistral-vibe-zai) -- full benchmark writeup and architecture comparison <!-- TODO: replace URL with actual article link -->
- [codex-zai](https://github.com/charles-azam/codex-zai) -- Codex fork (scored 0.15)
- [gemini-cli-zai](https://github.com/charles-azam/gemini-cli-zai) -- Gemini CLI fork (scored 0.23)
- [Upstream Mistral Vibe](https://github.com/mistralai/mistral-vibe) -- original project

## License

Copyright 2025 Mistral AI. Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
