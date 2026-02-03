# Telegram Group Helper Bot (Rust)

A Rust rewrite of TelegramGroupHelperBot focused on performance and lower resource usage. It keeps feature parity with the Python bot while using async Rust and SQLite by default.

## What it does
- Stores chat history in SQLite for summaries and profiling.
- Provides group-friendly commands for summaries, fact checks, Q and A, and media generation.
- Uses Gemini by default with optional OpenRouter and search integrations.
- Extracts content from Telegraph and Twitter links and can upload images to CWD.PW.
- Writes logs to `logs/bot.log` and `logs/timing.log`.

## Commands
- `/tldr` - Summarize recent chat history in the thread.
- `/factcheck` - Fact-check a statement (text or reply).
- `/q` - Ask a question (uses model selection when OpenRouter models are configured).
- `/qq` - Quick Gemini response using the default Gemini model.
- `/img` - Generate or edit an image with Gemini.
- `/image` - Generate an image with selectable resolution and aspect ratio.
- `/vid` - Generate a video from text.
- `/profileme` - Generate a profile based on your chat history.
- `/paintme` - Create an artistic prompt based on your history.
- `/portraitme` - Create a portrait prompt based on your history.
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
3. Optional: create `allowed_chat.txt` if you want to restrict access.
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

### Access control
- `WHITELIST_FILE_PATH` - Path to whitelist file. Default: `allowed_chat.txt`.
  - File contents: one user ID or chat ID per line. Empty or missing file means no restrictions.
- `ACCESS_CONTROLLED_COMMANDS` - Comma-separated list of commands requiring whitelist access.
  - Example: `/tldr,/factcheck,/profileme`

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
- `ENABLE_EXA_SEARCH` - Enable Exa search. Default: `true`.
- `EXA_API_KEY` - Exa API key.
- `EXA_SEARCH_ENDPOINT` - Default: `https://api.exa.ai/search`.
- `ENABLE_JINA_MCP` - Enable Jina search and reader. Default: `false`.
- `JINA_AI_API_KEY` - Jina AI key.
- `JINA_SEARCH_ENDPOINT` - Default: `https://s.jina.ai/search`.
- `JINA_READER_ENDPOINT` - Default: `https://r.jina.ai/`.

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
