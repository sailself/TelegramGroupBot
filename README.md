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
- `/qq` - Quick Gemini response using the default Gemini model.
- `/ragq` - Ask using retrieved local chat-history context from an external RAG service.
- `/img` - Generate or edit an image with Gemini.
- `/image` - Generate an image with selectable resolution and aspect ratio.
- `/vid` - Generate a video from text.
- `/profileme` - Generate a profile based on your chat history.
- `/paintme` - Create an artistic prompt based on your history.
- `/portraitme` - Create a portrait prompt based on your history.
- `/status` - Show a health snapshot (admin-only via whitelist).
- `/diagnose` - Show extended diagnostics and recent log tails (admin-only via whitelist).
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

One-shot history import before starting the bot:
```bash
cargo run -- import-history --file ./exports/group-history.json --chat-id -1001234567890 --batch-size 128 --resume
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

## Setup OCI Autonomous AI Database (Always Free) for RAG
Use this when you run embeddings/retrieval on a separate service and want Oracle to store vectors.

1. Choose a supported region for Always Free 26ai.
   - As of February 7, 2026, Oracle docs list `PHX`, `IAD`, `LHR`, `CDG`, `SYD`, `BOM`, `SIN`, and `NRT`.
   - If your region is not in that list, 26ai vector features may not be available in Always Free.
2. Create the database instance.
   - OCI Console -> `Oracle Database` -> `Autonomous Database` -> `Create Autonomous Database`.
   - Enable `Always Free`.
   - Select a 26ai-enabled Autonomous type.
   - Set an admin password and database display name.
3. Configure secure network access.
   - Recommended: use a `Private Endpoint` in the same VCN as your RAG VM.
   - Alternative: `Public Endpoint` with access control list restricted to your RAG VM egress IP(s).
4. Open `Database Actions` and create an app schema user.
   - Connect as `ADMIN`.
   - Create a dedicated user for RAG reads/writes.
```sql
CREATE USER rag_app IDENTIFIED BY "ChangeMe_UseStrongPassword";
GRANT CREATE SESSION, CREATE TABLE, CREATE VIEW, CREATE PROCEDURE TO rag_app;
GRANT UNLIMITED TABLESPACE TO rag_app;
```
5. Create message and embedding tables.
   - Connect as `rag_app` in Database Actions SQL Worksheet.
```sql
CREATE TABLE chat_messages (
  chat_id NUMBER NOT NULL,
  message_id NUMBER NOT NULL,
  user_id NUMBER NULL,
  username VARCHAR2(256) NULL,
  msg_time TIMESTAMP NOT NULL,
  reply_to_message_id NUMBER NULL,
  text CLOB,
  CONSTRAINT chat_messages_pk PRIMARY KEY (chat_id, message_id)
);

CREATE TABLE chat_embeddings (
  chat_id NUMBER NOT NULL,
  message_id NUMBER NOT NULL,
  embed_model VARCHAR2(64) NOT NULL,
  embed_dim NUMBER NOT NULL,
  embedding VECTOR(768, FLOAT32) NOT NULL,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
  CONSTRAINT chat_embeddings_pk PRIMARY KEY (chat_id, message_id),
  CONSTRAINT chat_embeddings_fk
    FOREIGN KEY (chat_id, message_id)
    REFERENCES chat_messages (chat_id, message_id)
);
```
6. Add indexing for scale.
   - Add metadata index:
```sql
CREATE INDEX chat_messages_time_idx ON chat_messages (chat_id, msg_time);
```
   - Add a vector index (recommended for large datasets). Use Oracle Vector Index syntax from docs for your target distance metric (`COSINE` is typical for embedding retrieval).
7. Configure client connectivity from your RAG service VM.
   - In the Autonomous DB details page, open `Database connection`.
   - Choose connection mode (mTLS wallet or TLS as appropriate to your network/security setup).
   - Store secrets outside source control and pass them as environment variables to the RAG service.
8. Validate before backfill.
   - Run a small insert/query smoke test from your RAG service.
   - Then run this bot’s historical import:
```bash
cargo run -- import-history --file ./exports/group-history.json --chat-id -1001234567890 --batch-size 128 --resume
```
9. Enable bot-side RAG integration.
```bash
ENABLE_RAG=true
RAG_SERVICE_URL=http://<your-rag-service-host>:<port>
RAG_SERVICE_API_KEY=<optional-token>
RAGQ_TOP_K=8
```

Official references:
- Always Free resource reference: https://docs.oracle.com/iaas/Content/FreeTier/resourceref.htm
- Autonomous Always Free details and limits: https://docs.oracle.com/en/cloud/paas/autonomous-database/serverless/adbsb/autonomous-always-free.html
- Provision Autonomous Database: https://docs.oracle.com/en/cloud/paas/autonomous-database/serverless/adbsb/provision-autonomous-instance.html
- Oracle AI Vector Search docs: https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/

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

### Local RAG integration (optional)
- `ENABLE_RAG` - Enable RAG service integration for live ingest and `/ragq`. Default: `false`.
- `RAG_SERVICE_URL` - Base URL of the external RAG service (example: `http://10.0.0.12:8080`).
- `RAG_SERVICE_API_KEY` - Optional API key sent as both `Authorization: Bearer` and `X-API-Key`.
- `RAG_INGEST_BATCH_SIZE` - Batch size for import backfill ingestion. Default: `128`.
- `RAG_HTTP_TIMEOUT_MS` - HTTP timeout for RAG requests. Default: `3000`.
- `RAGQ_TOP_K` - Number of retrieved snippets for `/ragq`. Default: `8`.
- `RAG_IMPORT_RESUME_DIR` - Directory for import checkpoint files. Default: `data`.

### Access control
- `WHITELIST_FILE_PATH` - Path to whitelist file. Default: `allowed_chat.txt`.
  - File contents: one user ID or chat ID per line. Empty or missing file means no restrictions.
  - `/status` and `/diagnose` require this whitelist file to be present and include your user ID or chat ID.
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
