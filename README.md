# Telegram Group Helper Bot (Rust)

A Rust rewrite of TelegramGroupHelperBot focused on performance and lower resource usage. It keeps feature parity with the Python bot while using async Rust and SQLite by default.

## What it does
- Stores chat history in SQLite for summaries and profiling.
- Provides group-friendly commands for summaries, fact checks, Q and A, and media generation.
- Uses Gemini by default with optional OpenRouter and search integrations.
- Extracts content from Telegraph and Twitter links and can upload images to CWD.PW.
- Writes text logs to `logs/bot.log` and `logs/timing.log`.
- Writes structured JSON logs to `logs/bot.jsonl` and `logs/timing.jsonl`.

## Commands
- `/tldr` - Summarize recent chat history in the thread.
- `/factcheck` - Fact-check a statement (text or reply).
- `/q` - Ask a question (uses model selection when OpenRouter models are configured).
- Mentioning the bot (for example `@YourBot question`) or replying to this bot's message also triggers `/q` behavior automatically.
- `/qq` - Quick Gemini response using the default Gemini model.
- `/agent` - Run the skills-first agent with multi-step tool use.
- `/agent_status` - Show recent agent sessions for your user in this chat.
- `/agent_resume` - Start a new run seeded from a previous session context.
- `/agent_new` - Reset pending/active agent lane for your user in this chat.
- `/img` - Generate or edit an image with Gemini.
- `/image` - Generate an image with selectable resolution and aspect ratio.
- `/vid` - Generate a video from text.
- `/profileme` - Generate a profile based on your chat history.
- `/paintme` - Create an artistic prompt based on your history.
- `/portraitme` - Create a portrait prompt based on your history.
- `/status` - Show a health snapshot (ACL-controlled).
- `/diagnose` - Show extended diagnostics and recent log tails (ACL-controlled).
- `/acl_reload` - Force reload of `acl.json` (owner-only).
- `/support` - Show support message and link.
- `/help` - Show command help.

## Project layout
- `src/main.rs` - Bot entry point and dispatcher wiring.
- `src/config.rs` - Environment loading and defaults.
- `src/handlers/` - Command handlers, access control, and response logic.
- `src/llm/` - Gemini and OpenRouter clients, tool orchestration, media helpers.
- `src/db/` - SQLite access and background writer queue.
- `src/utils/` - Logging, timing, and HTTP helpers.

## Setup (local)
1. Install Rust (1.78+ recommended).
2. Create a `.env` file in the project root (see below).
3. Create `acl.json` for command/tool access control (see ACL section below).
4. Run the bot.

## Run the bot (local)
```bash
cargo build
cargo run
```

Release build:
```bash
cargo build --release
./target/release/telegram_group_helper_bot
```

## Run the bot (Docker)
```bash
docker build -t telegram-group-helper-bot .
docker run --env-file .env -v ./data:/app/data -v ./logs:/app/logs telegram-group-helper-bot
```

The container defaults to `DATABASE_URL=sqlite:///data/bot.db`. Mount `./data` to persist the database.

## Environment variables

### Required
- `BOT_TOKEN` - Telegram bot token from BotFather.
- `GEMINI_API_KEY` - Google AI Studio key for Gemini APIs.

### Database
- `DATABASE_URL` - SQLite connection string.
  - Default: `sqlite+aiosqlite:///bot.db` (normalized to `sqlite:///bot.db`).
  - Examples:
    - `sqlite:///bot.db` (relative to project root)
    - `sqlite:///D:/Bots/telegram/bot.db` (absolute on Windows)

### Telegram runtime
- `RATE_LIMIT_SECONDS` - Per-user cooldown in seconds. Default: `15`.
- `MODEL_SELECTION_TIMEOUT` - Model selection UI timeout seconds. Default: `30`.
- `DEFAULT_Q_MODEL` - Default `/q` model (e.g., `gemini`). Default: `gemini`.
- `TELEGRAM_MAX_LENGTH` - Max message length before truncation or Telegraph. Default: `4000`.
- `USER_HISTORY_MESSAGE_COUNT` - Messages to retain for user history. Default: `200`.
- `LOG_LEVEL` - Logging level (`error`, `warn`, `info`, `debug`, `trace`). Default: `info`.

### Access control (ACL)
- `ACL_FILE_PATH` - Path to ACL file. Default: `acl.json`.
- `ACL_RELOAD_TTL_SECONDS` - Metadata recheck interval for hot reload. Default: `2`.
- `ACL_ENFORCED` - Enable ACL checks for commands and `/agent` tool calls. Default: `true`.

`acl.json` schema:
```json
{
  "version": 1,
  "owner_user_ids": [123456789],
  "full_access_chat_ids": [-1001234567890],
  "global": {
    "allow_commands": ["help", "q", "agent", "status", "diagnose"],
    "allow_tools": ["read_file", "write_file", "edit_file", "exec", "web_search", "memory_store", "memory_recall", "memory_forget"]
  },
  "chats": {
    "-1001234567890": {
      "full_access": false,
      "allow_commands": ["img", "image", "vid"],
      "deny_commands": ["diagnose"],
      "allow_tools": ["web_search"],
      "deny_tools": ["exec"]
    }
  }
}
```

Permission rule:
- Effective allow list = `(global allow ∪ chat allow) - chat deny`.
- `owner_user_ids` bypass ACL checks globally.
- `full_access_chat_ids` or per-chat `full_access=true` bypass ACL checks for that chat.

### Gemini settings
- `GEMINI_MODEL` - Default Gemini model. Default: `gemini-2.0-flash`.
- `GEMINI_PRO_MODEL` - Pro model. Default: `gemini-2.5-pro-exp-03-25`.
- `GEMINI_IMAGE_MODEL` - Image model. Default: `gemini-3-pro-image-preview`.
- `GEMINI_VIDEO_MODEL` - Video model. Default: `veo-3.1-generate-preview`.
- `GEMINI_TEMPERATURE` - Default: `0.7`.
- `GEMINI_TOP_K` - Default: `40`.
- `GEMINI_TOP_P` - Default: `0.95`.
- `GEMINI_MAX_OUTPUT_TOKENS` - Default: `2048`.
- `GEMINI_THINKING_LEVEL` - Default: `high`.
- `GEMINI_SAFETY_SETTINGS` - Safety profile: `standard` or `permissive` (`off`/`none` are treated as `permissive`). Default: `permissive`.
  - `standard` maps to `BLOCK_MEDIUM_AND_ABOVE`; `permissive` maps to `OFF` for all Gemini safety categories.

### OpenRouter (optional)
- `ENABLE_OPENROUTER` - Enable OpenRouter. Default: `true`.
- `OPENROUTER_API_KEY` - OpenRouter API key.
- `OPENROUTER_BASE_URL` - Default: `https://openrouter.ai/api/v1`.
- `OPENROUTER_ALPHA_BASE_URL` - Default: `https://openrouter.ai/api/alpha`.
- `OPENROUTER_TEMPERATURE` - Default: `0.7`.
- `OPENROUTER_TOP_K` - Default: `40`.
- `OPENROUTER_TOP_P` - Default: `0.95`.
- `OPENROUTER_MODELS_CONFIG_PATH` - Path to model config JSON.
  - Defaults to `openrouter_models.json` or `bot/openrouter_models.json` if present.

### Agent runtime
- `AGENT_PROVIDER` - Agent runtime provider: `gemini` or `openrouter`. Default: `gemini`.
- `SKILLS_DIR` - Directory containing Markdown skills with YAML frontmatter. Default: `skills`.
- `AGENT_WORKSPACE_ROOT` - Base directory for agent workspace files (`AGENTS.md`, `MEMORY.md`, tool-created files). Default: `agent_workspace`.
- `AGENT_WORKSPACE_SEPARATE_BY_CHAT` - Create per-chat workspaces under the base root (`chat_<id>` folders). Default: `true`.
- `AGENT_MODEL` - Optional provider-specific model ID for `/agent`.
  - For `gemini`: defaults to `GEMINI_PRO_MODEL`, then `GEMINI_MODEL`.
  - For `openrouter`: defaults to `GPT_MODEL`, then first tools-capable OpenRouter model.
- `AGENT_MAX_TOOL_ITERATIONS` - Max tool loop iterations per run. Default: `4`.
- `AGENT_MAX_ACTIVE_SKILLS` - Max selected skills loaded per run (excluding always-active core skill). Default: `3`.
- `AGENT_SKILL_CANDIDATE_LIMIT` - Candidate skills considered before final selection. Default: `8`.
- `AGENT_MEMORY_ENABLED` - Enable agent memory recall/save loop. Default: `true`.
- `AGENT_MEMORY_RECALL_LIMIT` - Maximum recalled memories prepended per run. Default: `5`.
- `AGENT_MEMORY_MAX_CONTEXT_CHARS` - Character budget for `[Memory context]` block. Default: `2500`.
- `AGENT_MEMORY_MIN_RELEVANCE` - Recall threshold after lexical/recency blending. Default: `0.15`.
- `AGENT_MEMORY_SAVE_SUMMARY_CHARS` - Summary length per saved memory entry. Default: `240`.
- `AGENT_HYGIENE_ENABLED` - Enable periodic retention cleanup for agent memory/session tables. Default: `true`.
- `AGENT_HYGIENE_INTERVAL_SECONDS` - Interval for cleanup loop. Default: `43200` (12 hours).
- `AGENT_MEMORY_RETENTION_DAYS` - Retention window for `agent_memories`. Default: `90`.
- `AGENT_SESSION_RETENTION_DAYS` - Retention window for completed/cancelled agent sessions and related rows. Default: `30`.
- `AGENT_PROMPT_MAX_FILE_CHARS` - Truncation limit for scaffold files (`AGENTS.md`, `MEMORY.md`). Default: `20000`.
- `AGENT_PROMPT_INCLUDE_AGENTS` - Include `AGENTS.md` in `/agent` system prompt. Default: `true`.
- `AGENT_PROMPT_INCLUDE_MEMORY_MD` - Include `MEMORY.md` in `/agent` system prompt. Default: `true`.
- `AGENT_PROMPT_INCLUDE_SKILLS_INDEX` - Include skill catalog in `/agent` system prompt. Default: `true`.
- `/agent` tool authorization now uses `acl.json` (`global.allow_tools`, `chats.<id>.allow_tools`, `chats.<id>.deny_tools`).
- `AGENT_EXEC_ALLOWLIST_REGEX` - Comma-separated regex allowlist for `exec` commands (if set, command must match one regex).
- `AGENT_EXEC_TIMEOUT_SECONDS` - Shell command timeout for `exec` tool. Default: `60`.
- `AGENT_EXEC_MAX_OUTPUT_CHARS` - Output truncation limit for `exec`. Default: `10000`.
- `AGENT_EXEC_RESTRICT_TO_WORKSPACE` - Restrict shell and paths to current workspace. Default: `true`.
- `AGENT_EXEC_DENY_PATTERNS` - Extra comma-separated regex deny patterns for shell guardrails.
- `AGENT_REQUIRE_CONFIRMATION_FOR_WRITE` - Require user confirmation before `write_file`. Default: `true`.
- `AGENT_REQUIRE_CONFIRMATION_FOR_EDIT` - Require user confirmation before `edit_file`. Default: `true`.
- `AGENT_REQUIRE_CONFIRMATION_FOR_EXEC` - Require user confirmation before `exec`. Default: `true`.

Legacy OpenRouter model variables (used if JSON is missing):
- `LLAMA_MODEL`
- `GROK_MODEL`
- `QWEN_MODEL`
- `DEEPSEEK_MODEL`
- `GPT_MODEL`

Example `openrouter_models.json`:
```json
{
  "models": [
    {
      "name": "Llama 4",
      "model": "meta-llama/llama-4",
      "image": true,
      "video": false,
      "audio": false,
      "tools": true
    }
  ]
}
```

### Search and retrieval (optional)
- `ENABLE_BRAVE_SEARCH` - Enable Brave Search. Default: `true`.
- `BRAVE_SEARCH_API_KEY` - Brave Search API key.
- `BRAVE_SEARCH_ENDPOINT` - Default: `https://api.search.brave.com/res/v1/web/search`.
- `ENABLE_EXA_SEARCH` - Enable Exa search. Default: `true`.
- `EXA_API_KEY` - Exa API key.
- `EXA_SEARCH_ENDPOINT` - Default: `https://api.exa.ai/search`.
- `ENABLE_JINA_MCP` - Enable Jina search and reader. Default: `false`.
- `JINA_AI_API_KEY` - Jina AI key.
- `JINA_SEARCH_ENDPOINT` - Default: `https://s.jina.ai/search`.
- `JINA_READER_ENDPOINT` - Default: `https://r.jina.ai/`.
- `WEB_SEARCH_PROVIDERS` - Comma-separated provider order. Default: `brave,exa,jina`.
- `WEB_SEARCH_CACHE_TTL_SECONDS` - Cache TTL for web search results. Default: `900` (15 minutes).

### Hosting and publishing (optional)
- `TELEGRAPH_ACCESS_TOKEN` - Required to publish long responses to Telegraph.
- `TELEGRAPH_AUTHOR_NAME` - Optional author name for Telegraph pages.
- `TELEGRAPH_AUTHOR_URL` - Optional author URL for Telegraph pages.
- `CWD_PW_API_KEY` - API key for CWD.PW image hosting.

### Support message
- `SUPPORT_MESSAGE` - Message shown by `/support`.
- `SUPPORT_LINK` - Link shown by `/support` (must be a valid URL to render a button).

## Development notes
- After big code changes, run `cargo build` and fix any compilation errors.
- On small VMs (1 vCPU/1GB), consider `cargo check`, `CARGO_BUILD_JOBS=1`, and setting `[profile.dev] debug = 0` to speed up builds.
  - Optional setup: install `sccache`, export `RUSTC_WRAPPER=sccache`, and add swap to avoid OOM during linking.
  - Example (Ubuntu):
```bash
sudo apt update && sudo apt install -y sccache
export RUSTC_WRAPPER=sccache
CARGO_BUILD_JOBS=1 cargo check

sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Example `.env`
```bash
BOT_TOKEN=123456789:your-telegram-token
GEMINI_API_KEY=your-gemini-key
DATABASE_URL=sqlite:///bot.db
```

## Notes and limitations
- Webhook mode is not implemented in this port yet; polling only.
- Video generation can take a few minutes and returns an MP4 when the Veo operation completes.
- If `TELEGRAPH_ACCESS_TOKEN` is not set, long messages will be truncated instead of published.
