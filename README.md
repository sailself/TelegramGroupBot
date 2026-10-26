# Telegram Group Helper Bot (Rust)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sailself/TelegramGroupBot)

A Rust rewrite of TelegramGroupHelperBot focused on performance and lower resource usage. It keeps feature parity with the Python bot while using async Rust and SQLite by default.

## What it does
- Stores chat history in SQLite for summaries and profiling.
- Provides group-friendly commands for summaries, fact checks, Q and A, and media generation.
- Uses configurable Gemini or Codex defaults with optional third-party hosted models (OpenRouter, NVIDIA, Ollama Cloud, OpenAI Responses, and ChatGPT-backed OpenAI Codex) plus search integrations.
- Extracts content from Telegraph and Twitter links and can upload images to CWD.PW.
- Writes text logs to `logs/bot.log` and `logs/timing.log`.
- Writes structured JSON logs to `logs/bot.jsonl` and `logs/timing.jsonl`.

## Commands
- `/tldr` - Summarize recent chat history in the thread.
- `/factcheck` - Fact-check a statement (text or reply).
- When Gemini and a ready Codex model can both serve a `/tldr` or `/factcheck` request, the bot first asks the sender to pick one (default model first). Without an answer within `MODEL_SELECTION_TIMEOUT`, it runs on `DEFAULT_TEXT_MODEL`. Requests Codex cannot serve (such as video, audio, or documents it does not support) skip the picker.
- `/q` - Ask a question (uses model selection when third-party models are configured).
- `/qc` - Ask about this chat through independently routed recall, analytics whose results are exact only for the normalized query over eligible stored-text rows, or LLM-assisted topic discovery.
- Mentioning the bot (for example `@YourBot question`) or replying to this bot's message also triggers `/q` behavior automatically. Replies to the bot's generated images or videos (including `/img2` and `/vid2` results) do not, unless the reply also mentions the bot.
- `/qq` - Provider-neutral quick answer: skips model selection, stays brief, and may use one bounded web-search round for genuinely current facts. Use `/q` for deeper verification or research.
- `/burn_baby_burn` - Show how many tokens you have used in the current chat.
- `/token_devourers [n]` - Show the top token consumers in the current group chat.
- `/token_stats [model|user]` - Show bot-wide token usage totals (admin-only).
- `/s` - Search this chat with a tool-capable model and return relevant message links.
- `/img` - Generate or edit an image with the configured default image model, or choose Gemini/Codex when Codex is enabled.
- `/image` - Generate an image with selectable Gemini resolution/aspect ratio or Codex image size; timeout uses the configured default image model.
- `/vid` - Generate a video from text.
- `/mysong` - Generate a theme song from your chat history.
- `/profileme` - Generate a profile based on your chat history.
- `/paintme` - Create an artistic prompt based on your history.
- `/portraitme` - Create a portrait prompt based on your history.
- `/status` - Show a health snapshot (admin-only via whitelist).
- `/diagnose` - Show extended diagnostics and recent log tails (admin-only via whitelist).
- `/codexlogin` - Start ChatGPT Codex device-code login (whitelisted users in private chats only).
- `/codexlogout` - Remove cached ChatGPT Codex credentials (whitelisted users in private chats only).
- `/codexmodel` - Fetch the live Codex model catalog and choose the active Codex model (whitelisted users in private chats only).
- `/codexreasoning` - Choose the active Codex reasoning level supported by the selected Codex model (whitelisted users in private chats only).
- `/codexusage` - Show the current ChatGPT Codex plan and usage windows (whitelisted users in private chats only).
- `/support` - Show support message and link.
- `/help` - Show command help.

## Project layout
- `src/main.rs` - Bot entry point and dispatcher wiring.
- `src/config.rs` - Environment loading and defaults.
- `src/handlers/` - Command handlers, access control, and response logic.
- `src/llm/` - Gemini and third-party clients, resolved model snapshots, tool orchestration, media helpers.
- `src/agents/` - Bounded fact-check, chat-context, analytics, topic, and summary pipelines.
- `src/db/` - SQLite access and background writer queue.
- `src/utils/` - Logging, timing, and HTTP helpers.

## Setup (local)
1. Install Rust 1.88 or newer (the Docker build uses Rust 1.92).
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

## Deploy on a Linux server
If you still deploy from source on the server, use the bundled helper instead of typing the full sequence each time:

```bash
chmod +x scripts/deploy_release.sh
./scripts/deploy_release.sh
```

What it does:
- runs `git pull --ff-only` on the current branch when `.git/` is present
- builds with `CARGO_BUILD_JOBS=1` by default
- sets `RUSTFLAGS="-C debuginfo=0"` unless you already exported `RUSTFLAGS`
- stops the previous bot process using `run/telegram_group_helper_bot.pid`
- starts the new release binary with `nohup` and appends stdout/stderr to `logs/nohup.bot.log`

Useful overrides:

```bash
SKIP_GIT_PULL=1 ./scripts/deploy_release.sh
SKIP_BUILD=1 ./scripts/deploy_release.sh
SKIP_RESTART=1 ./scripts/deploy_release.sh
CARGO_BUILD_JOBS=2 ./scripts/deploy_release.sh
```

The script assumes the bot runs from the repo root so relative paths like `.env`, `logs/`, and SQLite files keep working.

### Recommended: run it with systemd
`nohup` works, but `systemd` is the safer production default because it handles restart policy, boot startup, and process supervision correctly.

Use the template at `deploy/telegram_group_helper_bot.service` and adjust:
- `User`
- `WorkingDirectory`
- `ExecStart`

Example install flow:

```bash
sudo cp deploy/telegram_group_helper_bot.service /etc/systemd/system/telegram_group_helper_bot.service
sudo systemctl daemon-reload
sudo systemctl enable telegram_group_helper_bot
sudo systemctl start telegram_group_helper_bot
sudo systemctl status telegram_group_helper_bot
```

### GitHub Actions release artifacts
The repo now includes a release workflow at `.github/workflows/release.yml`.

How it works:
- `workflow_dispatch` builds downloadable GitHub Actions artifacts
- pushing a tag like `v0.1.0` builds artifacts and publishes them to a GitHub Release
- both Linux and Windows bundles are produced

Artifact contents:
- Linux: `telegram_group_helper_bot`, `README.md`, `.env.example`, `deploy/telegram_group_helper_bot.service`
- Linux deploy bundle: full release layout with `target/release/telegram_group_helper_bot`, `deploy/install_release_bundle.sh`, and the systemd template
- Windows: `telegram_group_helper_bot.exe`, `README.md`, `.env.example`

The generated asset names are:
- `telegram_group_helper_bot-linux-x86_64.tar.gz`
- `telegram_group_helper_bot-linux-x86_64-deploy.tar.gz`
- `telegram_group_helper_bot-windows-x86_64.zip`
- matching `.sha256` checksum files for each archive

Example server install from the deploy bundle:

```bash
tar -xzf telegram_group_helper_bot-linux-x86_64-deploy.tar.gz
cd telegram_group_helper_bot-linux-x86_64-deploy
sudo SERVICE_USER=telegrambot APP_DIR=/opt/telegram_chat_bot ./deploy/install_release_bundle.sh
if [ ! -f /opt/telegram_chat_bot/.env ]; then sudo cp /opt/telegram_chat_bot/.env.example /opt/telegram_chat_bot/.env; fi
sudo editor /opt/telegram_chat_bot/.env
sudo systemctl restart telegram_group_helper_bot
sudo systemctl status telegram_group_helper_bot
```

On upgrades, keep the existing `.env` and just rerun the installer against the new extracted bundle.

Example release flow:

```bash
git tag v0.1.0
git push origin v0.1.0
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
  - Default: `sqlite://bot.db`.
  - Examples:
    - `sqlite:///bot.db` (relative to project root)
    - `sqlite:///D:/Bots/telegram/bot.db` (absolute on Windows)
- `DB_MAX_CONNECTIONS` - SQLite pool size. Default: `5`.
- `DB_QUEUE_CAPACITY` - Buffered async message-write queue size. Default: `2048`.
- `DB_WRITE_BATCH_SIZE` - Max queued message inserts written per DB batch. Default: `32`.
- `DB_WRITE_FLUSH_MS` - Max wait before flushing a partial DB batch. Default: `25`.

### Telegram runtime
- `HEAVY_COMMAND_MAX_CONCURRENCY` - Max number of heavy commands (`/q`, `/qc`, `/tldr`, generation commands, etc.) running at once. Default: `5`.
- `RATE_LIMIT_SECONDS` - Per-user cooldown in seconds. Default: `15`.
- `MODEL_SELECTION_TIMEOUT` - Model selection UI timeout seconds. Image pickers receive a fresh timeout for each step; unfinished choices use defaults while preserving saved choices. Default: `30`.
- `DEFAULT_TEXT_MODEL` - Default text model for model-selection timeouts, `/tldr` and `/factcheck` when no Gemini-or-Codex picker is shown, `/profileme`, and the prompt step for `/paintme`/`/portraitme`; it is also the inherited/fallback model for `/qq`. Use `gemini` or a runtime model such as `openai-codex:selected`/`openai-codex`. Default: `gemini`.
- `DEFAULT_QUICK_TEXT_MODEL` - Optional dedicated `/qq` model. Empty or unset inherits `DEFAULT_TEXT_MODEL`. It accepts configured runtime IDs, `openai-codex:selected`, or an authenticated catalog slug such as `openai-codex:gpt-5.6-terra`; unavailable or incompatible models fall back once to the capable `DEFAULT_TEXT_MODEL` without opening a picker.
- `QUICK_REASONING_EFFORT` - Per-request reasoning override for OpenAI Codex models used by `/qq`. Default: `low`. Public OpenAI and other providers keep their existing request contracts.
- `DEFAULT_IMAGE_MODEL` - Default image model for `/img`, `/image` timeout/default generation, `/tldr` infographics, and `/paintme`/`/portraitme`. Use `gemini` or `codex`. Default: `gemini`.
- `TELEGRAM_MAX_LENGTH` - Max message length before truncation or Telegraph. Default: `4000`.
- `USER_HISTORY_MESSAGE_COUNT` - Messages to retain for user history. Default: `200`.
- `LOG_LEVEL` - Logging level (`error`, `warn`, `info`, `debug`, `trace`). Default: `info`.
- `PUBLISH_BOT_COMMANDS` - When `true`, publish the built-in command list on startup via Telegram `setMyCommands`. Default: `false`.
  - Warning: Telegram treats this as a replacement for the default-scope command list. Leave it `false` if you manage commands in BotFather.
- `ENABLE_BOT_TO_BOT_AUTO_Q` - When `true`, auto-Q responds to another bot that mentions this bot or replies to this bot. This still ignores this bot's own messages. Default: `false`.
- `MEDIA_GROUP_MAX_ITEMS` - Max cached media groups kept in memory at once. Default: `256`.
- `MAX_TOOL_CONTEXT_ITEMS` - Max selected chat-search hits returned in the final `/s` response. Default: `10`.
- `ENABLE_TLDR_INFOGRAPHIC` - When `true`, `/tldr` also runs the configured default image model for an infographic step. Default: `false`.

### Agentic pipelines
`/factcheck` and `/qc` run as multi-phase pipelines with live progress edits on the processing message, while `/tldr` switches to map-reduce chunk summarization above a threshold. `/qc` routes each request independently: recall uses chat-scoped search (plus web research when needed), analytics runs validated read-only queries, and topic discovery uses LLM-assisted map/reduce classification over a bounded chat window. Analytics results are exact only for the normalized query over eligible stored-text rows; they do not represent complete Telegram activity or unqualified semantic truth. Media-only, service, anonymous-admin, channel-post, and other unstored rows are absent, while normalized filters may exclude commands, synthetic records, and AI asks. Topic labels and semantic counts remain LLM-assisted rather than exact database analytics; optional literal-substring results separately count eligible stored-text messages containing the escaped literal string, not FTS matches or occurrences within a message. Cheap orchestration steps use a configurable step model; the final answer keeps using the configured default/user-selected model. Each command still holds a single `HEAVY_COMMAND_MAX_CONCURRENCY` permit for its whole run; while a `/tldr` or `/factcheck` model picker waits for a choice, no permit is held.

- `ENABLE_AGENTIC_FACTCHECK` - When `false`, `/factcheck` reverts to the legacy single-call flow. The legacy flow is also the automatic fallback when claim extraction fails or finds nothing check-worthy. Default: `true`.
- `ENABLE_AGENTIC_QC` - When `false`, `/qc` reverts to the legacy monolithic tool loop. In agentic mode, malformed lane classification takes the recall path; analytics and topic-discovery failures fail closed instead of silently using the legacy loop. Default: `true`.
- `ENABLE_QC_TOPIC_DISCOVERY` - Enables semantic topic discovery for `/qc`. Default: `true`. Topic discovery analyzes at most `TLDR_MAX_MESSAGES` newest eligible stored text messages in the requested UTC range, using `TLDR_CHUNK_SIZE` chunks and up to four concurrent map calls. Results disclose capped and partial coverage and are LLM-assisted classifications, not exact semantic counts.
- `AGENT_STEP_MODEL` - Model for cheap pipeline steps (claim extraction, query planning, reflection, chunk summaries). Accepts `gemini` or a runtime model id; `openai-codex:<slug>` works even for slugs not in the catalog (e.g. `openai-codex:gpt-5.4-mini`). Empty = derive automatically: a Codex final model runs steps on itself at `AGENT_STEP_REASONING`; a public OpenAI final model runs steps on itself at `AGENT_STEP_REASONING`; a Gemini final model uses `GEMINI_LITE_MODEL`. Default: empty.
- `AGENT_STEP_REASONING` - Per-call reasoning effort for Responses-provider cheap pipeline steps and internal `/s` selection or `/qc` analytics gathering. It applies to public OpenAI and Codex, is ignored by other providers, and is validated against available Codex metadata; it does not replace the user's selected Codex reasoning level for final answers. Default: `low`.
- `AGENT_MAX_WALL_CLOCK_SECS` - Absolute time budget for `/qc`, `/factcheck`, and `/tldr` text generation, including model resolution, all pipeline phases, and any legacy fallback. Expiry cancels local work and reports a timeout; fallback never resets the budget. Input preparation, time spent in the `/tldr`/`/factcheck` model picker, final delivery, and the optional TLDR infographic are outside this budget. Default: `480`; native Gemini pipelines that can reach Argon use `GEMINI_ARGON_MAX_WALL_CLOCK_SECS` instead.
- `TLDR_MAP_REDUCE_THRESHOLD` - `/tldr` switches to map-reduce above this many messages; at or below it the original single-call path runs unchanged. Default: `150`.
- `TLDR_CHUNK_SIZE` - Messages per map-reduce chunk (chunks are summarized sequentially to keep memory flat). Default: `100`.
- `TLDR_MAX_MESSAGES` - Hard cap on messages fetched for `/tldr`, including the previously unbounded reply-anchored variant. Default: `2000`.
- `FACTCHECK_MAX_CLAIMS` - Max claims extracted and researched per `/factcheck`. Default: `5` (clamped 1-8).
- `FACTCHECK_SEARCHES_PER_CLAIM` - Max web searches per claim. Default: `2` (clamped 1-3).
- `FACTCHECK_CLAIM_CONCURRENCY` - Claims researched concurrently (network-bound; keep small on 1-CPU hosts). Default: `2` (clamped 1-4).
- `QC_ANALYTICS_MAX_TOTAL_CALLS` - Total tool calls allowed in the `/qc` analytics lane. Default: `12` (clamped 4-24; keep ≥ `QC_ANALYTICS_MAX_QUERY_CALLS` + 2 or the total cap trips first).
- `QC_ANALYTICS_MAX_QUERY_CALLS` - Max `chat_analytics` queries per `/qc` analytics request. Default: `10` (clamped 2-20).
- `QC_ANALYTICS_QUERY_TIMEOUT_SECS` - Per-analytics-query timeout in seconds. Default: `2` (clamped 1-15).

### Access control
- `WHITELIST_FILE_PATH` - Path to whitelist file. Default: `allowed_chat.txt`.
  - File contents: one user ID or chat ID per line. Empty or missing file means no restrictions.
  - `/status` and `/diagnose` require this whitelist file to be present and include your user ID or chat ID.
- `ACCESS_CONTROLLED_COMMANDS` - Comma-separated list of commands requiring whitelist access.
  - Example: `/tldr,/factcheck,/profileme,/mysong`

### Gemini settings
- `ENABLE_GEMINI` - When `false`, Gemini API-backed models and Gemini-only commands (`/vid`, `/mysong`) are disabled and hidden from model/command pickers. `/s` remains available when another ready model with `tools=true` is configured. Default: `true`.
- `GEMINI_MODEL` - Default Gemini model. Default: `gemini-3.8-flash`.
- `GEMINI_LITE_MODEL` - Lite fallback model after `GEMINI_MODEL` failures. Default: `gemini-3.5-flash-lite`.
- `GEMINI_PRO_MODEL` - Pro model. Default: `gemini-4-argon`.
- `GEMINI_IMAGE_MODEL` - Image model. Default: `gemini-3-pro-image-preview`.
- `GEMINI_MUSIC_MODEL` - Music model for `/mysong`. Default: `lyria-3-pro-preview`.
- `GEMINI_VIDEO_MODEL` - Video model. Default: `veo-3.1-generate-preview`.
- `GEMINI_TEMPERATURE` - Default: `0.7`. Argon omits this sampling parameter.
- `GEMINI_TOP_K` - Default: `40`. Argon omits this sampling parameter.
- `GEMINI_TOP_P` - Default: `0.95`. Argon omits this sampling parameter.
- `GEMINI_MAX_OUTPUT_TOKENS` - Positive output budget, including reasoning. Native Argon clamps it to `32768` per hop and honors smaller values. Default: `32768`.
- `GEMINI_THINKING_LEVEL` - Model-aware `thinkingConfig.thinkingLevel` for Gemini 3 and Argon, validated against the target model; omitted for Gemini 2.5. Default: `medium`.
- `GEMINI_SAFETY_SETTINGS` - Safety profile: `standard` or `permissive` (`off`/`none` are treated as `permissive`). Default: `permissive`.
  - `standard` maps to `BLOCK_MEDIUM_AND_ABOVE`; `permissive` maps to `OFF` for all Gemini safety categories.
- `GEMINI_REQUEST_TIMEOUT_SECS` - Per-attempt timeout for non-Argon Gemini `generateContent` requests. Default: `90`.
- `GEMINI_ARGON_REQUEST_TIMEOUT_SECS` - Positive per-attempt timeout for native Argon requests only. Default: `1200`.
- `GEMINI_ARGON_MAX_WALL_CLOCK_SECS` - Positive absolute text-pipeline budget for `/qc`, `/factcheck`, and `/tldr` when native Gemini is selected and the main or Lite role is Argon, or the request uses the pro role and that role is Argon. This covers Argon reached during steps or fallback as well as the final answer. Shared across model resolution, phases, and legacy fallback; does not reset on fallback or extend Codex pipelines. Default: `1800`.
- `GEMINI_IMAGE_REQUEST_TIMEOUT_SECS` - Per-attempt timeout for Gemini image generation. Default: `300`.

### Shared third-party model catalog
- `THIRD_PARTY_MODELS_CONFIG_PATH` - Path to the mixed-provider model config JSON.
  - Defaults to `third_party_models.json`.

### OpenRouter (optional)
- `ENABLE_OPENROUTER` - Enable OpenRouter. Default: `true`.
- `OPENROUTER_API_KEY` - OpenRouter API key.
- `OPENROUTER_BASE_URL` - Default: `https://openrouter.ai/api/v1`.
- `OPENROUTER_TEMPERATURE` - Default: `0.7`.
- `OPENROUTER_TOP_K` - Default: `40`.
- `OPENROUTER_TOP_P` - Default: `0.95`.
- `OPENROUTER_REQUEST_TIMEOUT_SECS` - Per-attempt request timeout. Default: `60`.

### NVIDIA hosted models (optional)
- `ENABLE_NVIDIA` - Enable NVIDIA-hosted chat models. Default: `true`.
- `NVIDIA_API_KEY` - NVIDIA API key for `integrate.api.nvidia.com`.
- `NVIDIA_BASE_URL` - Default: `https://integrate.api.nvidia.com/v1`.
- `NVIDIA_TEMPERATURE` - Default: `0.7`.
- `NVIDIA_TOP_P` - Default: `0.95`.
- `NVIDIA_REQUEST_TIMEOUT_SECS` - Per-attempt request timeout. Default: `60`.
- NVIDIA hosted chat completions are integrated through their OpenAI-compatible endpoint.

### Ollama Cloud (optional)
- `ENABLE_OLLAMA` - Enable Ollama Cloud via the OpenAI-compatible chat endpoint. Default: `true`.
- `OLLAMA_API_KEY` - Ollama API key from ollama.com.
- `OLLAMA_BASE_URL` - Must be HTTPS, except a plain `http://` URL is allowed when the host is loopback (`localhost`, `127.0.0.1`, or `::1`) for a local Ollama server, e.g. `http://localhost:11434/v1`. Default: `https://ollama.com/v1`.
- `OLLAMA_TEMPERATURE` - Default: `0.7`.
- `OLLAMA_TOP_P` - Default: `0.95`.
- `OLLAMA_REQUEST_TIMEOUT_SECS` - Per-attempt request timeout. Default: `60`.
- Ollama models are configured through `third_party_models.json` using `"provider": "ollama"`.

### OpenAI Responses (optional)
- `ENABLE_OPENAI` - Enable the public OpenAI Responses API provider. Default: `false`.
- `OPENAI_API_KEY` - OpenAI API key used for billed fallback models.
- `OPENAI_BASE_URL` - Default: `https://api.openai.com/v1`.
- `OPENAI_REQUEST_TIMEOUT_SECS` - Per-attempt request timeout. Default: `60`.

### Gemini Argon alongside Codex

Enable both providers while retaining your existing Codex default and quick-model settings. For Argon on every explicitly selected Gemini answer, use this rollout configuration (merge it into your configuration; do not replace unrelated settings):

```dotenv
ENABLE_GEMINI=true
ENABLE_OPENAI_CODEX=true
DEFAULT_TEXT_MODEL=openai-codex
# Keep your existing DEFAULT_QUICK_TEXT_MODEL unchanged.
# openai-codex:selected is also accepted as DEFAULT_TEXT_MODEL.
GEMINI_MODEL=gemini-4-argon
GEMINI_PRO_MODEL=gemini-4-argon
GEMINI_LITE_MODEL=gemini-3.5-flash-lite
GEMINI_THINKING_LEVEL=medium
GEMINI_MAX_OUTPUT_TOKENS=32768
GEMINI_ARGON_REQUEST_TIMEOUT_SECS=1200
GEMINI_ARGON_MAX_WALL_CLOCK_SECS=1800
```

The picker retains one native `Gemini` entry alongside ready Codex models. Plain-text `/q` uses `GEMINI_MODEL`; media and YouTube requests use `GEMINI_PRO_MODEL`. The shipped defaults use Flash for the former and Argon for the latter; setting both roles to Argon makes both routes use Argon. `/tldr` uses the pro role. Configured fallbacks can still answer with another model after a retryable failure.

An explicit `AGENT_STEP_MODEL` takes precedence for cheap orchestration steps: keep an existing Codex override if desired. When empty, Gemini final answers derive those steps from `GEMINI_LITE_MODEL`. Image, video, and music models have independent settings; keep `DEFAULT_IMAGE_MODEL` and the existing image/video/music configuration as desired. Argon text selection does not replace those models.

This adapter uses bounded single-hop native `generateContent`, including tool turns. `MAX_TOKENS`, `CONTINUATION`, blocked responses, and empty final responses are reported as incomplete/errors instead of delivered as completed answers. It does not resume continuation tokens or implement long decode; automatic continuation and Interactions/background execution remain separate future work. The output cap includes reasoning. A longer client timeout does not guarantee that the server completes a request; see Google's [native migration guide](https://ai.google.dev/gemini-api/docs/generate-content/latest-model) and [long-decode guide](https://ai.google.dev/gemini-api/docs/generate-content/long-decode).

Argon uses its own per-attempt request timeout. For `/qc`, `/factcheck`, and `/tldr`, native Gemini pipelines that can reach a configured Argon role use one absolute deadline across model resolution, phases, and legacy fallback. Codex pipelines retain `AGENT_MAX_WALL_CLOCK_SECS=480`; non-Argon HTTP calls retain `GEMINI_REQUEST_TIMEOUT_SECS=90`. Input preparation, final delivery, and the optional TLDR infographic remain outside the pipeline budget. `/q`, `/qq`, and `/s` use their existing bounded provider retries/tool loops, not the agent-pipeline deadline; a request with retries or several tool turns can take longer than one HTTP timeout.

After updating configuration, restart the bot and inspect startup validation warnings (invalid positive budgets fall back to defaults). Verify API-project access to Argon before rollout; local configuration and unit tests do not establish account entitlement or successful live generation. Smoke-test selected Gemini text, media, a tool round trip, and structured agent steps, then confirm ordinary Codex `/q` and the existing `/qq` model still work. Check both successful output and incomplete/error reporting. To roll back Gemini availability, set `ENABLE_GEMINI=false` and restart while retaining the Codex default, quick model, and authentication; Gemini-only commands become unavailable. If the image default was Gemini, choose a ready image provider separately.

### OpenAI Codex via ChatGPT (optional)
- `ENABLE_OPENAI_CODEX` - Enable ChatGPT-backed Codex support. Default: `true`.
- `OPENAI_CODEX_BASE_URL` - Default: `https://chatgpt.com/backend-api/codex`.
- `OPENAI_CODEX_ORIGINATOR` - Request originator header. Default: `codex_cli_rs`.
- `OPENAI_CODEX_CLIENT_VERSION` - Codex model-catalog compatibility version sent to `/models`. Default: `0.144.0`.
- `OPENAI_CODEX_WEB_SEARCH_MODE` - Native Codex web search mode: `live`, `cached`, or `disabled`. Default: `live`.
- `OPENAI_CODEX_WEB_SEARCH_CONTEXT_SIZE` - Optional native Codex web search context size (for example `low`, `medium`, `high`).
- `OPENAI_CODEX_WEB_SEARCH_ALLOWED_DOMAINS` - Optional comma-separated domain allowlist for native Codex web search.
- `OPENAI_CODEX_AUTH_STORAGE` - Credential storage mode: `auto` or `file`. Default: `auto`. On Windows, `auto` protects credentials with a current-user DPAPI envelope and migrates legacy JSON; on Unix, `auto` uses the same private `0600` JSON file as `file`. Use `file` explicitly for portable or Docker deployments.
- `OPENAI_CODEX_AUTH_PATH` - Auth cache path used by the selected storage mode. Default: `data/openai_codex_auth.json`.
- `OPENAI_CODEX_MODEL_PATH` - Local selected-model cache path. Default: `data/openai_codex_model.json`.
- `OPENAI_CODEX_REQUEST_TIMEOUT_SECS` - Per-attempt request timeout. Default: `300`.
- `OPENAI_CODEX_IMAGE_RESPONSES_MODEL` - Responses model used to invoke Codex image generation. Default: `gpt-5.5`.
- `OPENAI_CODEX_IMAGE_MODEL` - Codex image-generation tool model offered in `/img` and `/image`, and used when `DEFAULT_IMAGE_MODEL=codex`. Default: `gpt-image-2`.
- Login and model-account administration are restricted to whitelisted users in private chats and managed with `/codexlogin` and `/codexlogout`.
- The active Codex model is selected live with `/codexmodel`, bound to the current ChatGPT account, and exposed in the bot as the runtime alias `openai-codex:selected`; ordinary `/q` continues to use this global selection.
- An explicit Codex slug in `DEFAULT_QUICK_TEXT_MODEL` is resolved from the authenticated catalog only for `/qq`, without changing `/codexmodel`; an unavailable or incompatible slug falls back once to `DEFAULT_TEXT_MODEL`.
- The active Codex reasoning effort is selected with `/codexreasoning` and is only offered when the chosen model advertises supported reasoning levels.
- When the selected Codex model advertises native search support, ordinary `/q` and other existing paths use Codex's built-in `web_search` Responses tool instead of the local external function tool. `/qq` deliberately disables native Codex search and uses the bounded local `web_search` runtime so its one-round limit is enforceable.
- Codex requests also include a condensed response-style addendum tuned for direct answers and shorter Chinese output.

### Hidden Img2 image generation (optional)
- `/img2` is a hidden command: it is callable when enabled, but it is not published with `setMyCommands` and is not shown in `/help`.
- `ENABLE_IMG2` - Enable the hidden `/img2` command. Default: `false`.
- `IMG2_BASE_URL` - API base URL; must be a valid HTTPS URL when set. Default: empty (`/img2` stays unavailable, see `img2_api_available()`, until this is configured).
- `IMG2_API_KEY` - API key sent in the `X-API-Key` header. Keep this in `.env` or deployment secrets only.
- `IMG2_GENERATE_PATH` - Generate endpoint path. Default: `/v1/images/generate`.
- `IMG2_HEALTH_PATH` - Health endpoint path shown in `/status`. Default: `/v1/health`.
- `IMG2_REQUEST_TIMEOUT_SECS` - Per-request timeout. Default: `300`.
- `IMG2_MEDIA_DIR` - Server-side folder for generated PNG files. Default: `data/media/img2`.
- `IMG2_WIDTH`, `IMG2_HEIGHT`, `IMG2_STEPS` - Optional multipart fields sent only when set to positive integers.
- `/img2` sends only `prompt`, optional first replied source image, and the configured optional `width`/`height`/`steps` fields. It does not upload generated files to cwd.pw.
- In Docker, the existing `./data:/app/data` mount persists the default `data/media/img2` folder.

### Hidden Vid2 video generation (optional)
- `/vid2` is a hidden command: it is callable when enabled, but is not published with `setMyCommands` or shown in `/help`. Use `/vid2 <prompt>` for text-to-video, or attach/reply to a photo or a JPEG, PNG, or WebP document for image-to-video. The prompt can be in the attached image's caption; an empty command falls back to replied text/caption. An attached image takes precedence over a replied image, and albums use the particular attached/replied item. An invalid or unavailable selected image fails the request.
- Completion arrives as a new video reply to the original request with normal notifications and an img2-style preformatted prompt caption; like `/img2`, the video and caption are sent blurred behind a spoiler. After Telegram confirms delivery, the bot removes the progress message; if removal is blocked, it attempts a `Video delivered below.` status instead. Generation or delivery failures retain updated error text.
- `ENABLE_VID2` - Enable the hidden `/vid2` command. Default: `false`.
- `VID2_BASE_URL` - HTTPS API base URL, for example `https://video-service.example.com:8443`. Default: empty; `/vid2` stays unavailable until both URL and key are configured, even when enabled. Custom HTTPS ports are accepted; credentials, query strings, and fragments are rejected.
- `VID2_API_KEY` - API key sent in the `X-API-Key` header on submission, status, and download requests. Keep this in `.env` or deployment secrets only. `/vid2` is ready only when enabled with both URL and key; `/status` reports this configuration readiness.
- `VID2_WIDTH` / `VID2_HEIGHT` - Default text-to-video dimensions. Defaults: `1344` × `768`. The service requires both edges to be multiples of 32 and at least 256 pixels, with at most `1344 * 768` (1,032,192) total pixels. Invalid text defaults are rejected locally before submission without preventing bot startup.
- Image-to-video fits within both the configured long-edge cap `max(VID2_WIDTH, VID2_HEIGHT)` and the service pixel budget, then aligns to 32 pixels while approximately preserving the source aspect ratio. Neither requested edge exceeds its source dimension; grid alignment can change the aspect ratio slightly. Native dimensions are retained when already valid; otherwise alignment can reduce even a smaller image. Examples: 1148×768 → 1120×736, 4000×3000 → 1152×864, 3000×4000 → 864×1152, and 1200×1200 → 992×992. The shorter edge may exceed the configured default height if the total area permits. Images too small or too extreme in aspect ratio to produce a valid fit are rejected locally; the bot does not force one side to 256. Original first-frame bytes are uploaded unchanged.
- `VID2_SECONDS` - Configured video duration sent with every generation request. Default: `5` seconds.
- `VID2_REQUEST_TIMEOUT_SECS` - Timeout for each HTTP request. Default: `120`.
- `VID2_GENERATION_TIMEOUT_SECS` - Deadline covering first-frame retrieval, submission, polling, and video download. Default: `1800`. Semaphore waiting, local persistence, and Telegram delivery are separate.
- `VID2_POLL_INTERVAL_SECS` - Delay between status polls. Default: `5`.
- `VID2_MEDIA_DIR` - Local folder for downloaded MP4 files. Default: `data/media/vid2`. Downloads are limited to 50,000,000 bytes and must contain an MP4 container. Completed files are retained, including when Telegram delivery fails; in Docker, the existing `./data:/app/data` mount persists this default folder.
- Dimensions, duration, and timeout/poll values must be positive integers. Dimensions must fit an unsigned 32-bit integer. Zero or unparsable values warn and fall back to their defaults; positive text dimensions that violate the service grid/minimum/area rules are rejected when that command runs.
- The service contract is `POST <base>/v1/videos/generate` with multipart `prompt`, `seconds`, target `width` and `height`, and optional `first_frame` image, returning an `id`. The bot polls `GET <base>/v1/videos/<id>` for `status` (`queued`, `running`, `success`, or `error`), then downloads the MP4 from `GET <base>/v1/videos/<id>/file` after `success`. Redirects are disabled for these authenticated requests.
- Observed server responses include `status: queued` and a `params` object on submission, and `status: running`/`success`, `duration`, and nullable `error` while polling. The bot ignores these response metadata fields; duration stays controlled by `VID2_SECONDS`, dimensions are sent explicitly, and server quality/seed use the service defaults.
- For rejected submissions, inspect the `Vid2 submission rejected` ERROR entry for a bounded, sanitized structured server reason and the source/target dimensions. Telegram keeps the generic HTTP error; prompts, credentials, source URLs, and image payloads are excluded from these diagnostics.
- The bot host or container must be able to reach the service, including its tailnet address when applicable, and trust its HTTPS certificate. Timeout stops the bot's wait without cancelling the remote job. Pending jobs are not recovered after a bot restart.

Example `third_party_models.json`:
```json
{
  "models": [
    {
      "provider": "openrouter",
      "name": "Llama 4",
      "model": "meta-llama/llama-4",
      "image": true,
      "video": false,
      "audio": false,
      "tools": true
    },
    {
      "provider": "nvidia",
      "name": "Gemma 3n",
      "model": "google/gemma-3n-e4b-it",
      "image": true,
      "video": false,
      "audio": true,
      "tools": false
    },
    {
      "provider": "ollama",
      "name": "Qwen 3 32B",
      "model": "qwen3:32b",
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
- `TWITTER_FETCH_PROVIDERS` - Comma-separated X/Twitter extraction provider order. Default: `fxtwitter,vxtwitter,jina`.
- `FXTWITTER_API_BASE` - HTTPS base URL for the FxTwitter adapter. Default: `https://api.fxtwitter.com`.
- `VXTWITTER_API_BASE` - HTTPS base URL for the VxTwitter adapter. Default: `https://api.vxtwitter.com`.
- `TWITTER_FETCH_TOTAL_TIMEOUT_SECS` - Total deadline for one X/Twitter extraction. Default: `20`.
- `TWITTER_PROVIDER_TIMEOUT_SECS` - Per-provider X/Twitter timeout. Default: `8`.
- `TWITTER_RESPONSE_MAX_BYTES` - Maximum provider response body size. Default: `2097152` (2 MiB).
- `EXTERNAL_MEDIA_MAX_BYTES` - Maximum bytes for one downloaded external media file. Default: `20971520` (20 MiB).
- `EXTERNAL_MEDIA_TOTAL_MAX_BYTES` - Maximum aggregate bytes for external media in one request. Default: `52428800` (50 MiB).
- `TWITTER_FETCH_PROVIDERS` controls X/Twitter extraction independently of `ENABLE_JINA_MCP`. Set `TWITTER_FETCH_PROVIDERS=jina` and restart the bot to restore the legacy Jina-only provider order.
- Direct X media is accepted only from `pbs.twimg.com` (images/thumbnails) and `video.twimg.com` (videos), over HTTPS without credentials or non-default ports. Oversized or unavailable videos use an allowed thumbnail and text fallback when available.
- `WEB_SEARCH_PROVIDERS` - Comma-separated provider order. Default: `brave,exa,jina`.
- `WEB_SEARCH_CACHE_TTL_SECONDS` - Cache TTL for web search results. Default: `900` (15 minutes).
- `WEB_SEARCH_CACHE_MAX_ENTRIES` - Max cached web-search queries kept in memory. Default: `256`.
- `EXTERNAL_ENRICH_FANOUT` - Max concurrent Telegraph/Twitter extraction or media-download tasks per request. Default: `4`.
- `GEMINI_UPLOAD_FANOUT` - Max concurrent Gemini media uploads per request. Default: `3`.

### Hosting and publishing (optional)
- `TELEGRAPH_ACCESS_TOKEN` - Required to publish long responses to Telegraph.
- `TELEGRAPH_AUTHOR_NAME` - Optional author name for Telegraph pages.
- `TELEGRAPH_AUTHOR_URL` - Optional author URL for Telegraph pages.
- `CWD_PW_API_KEY` - API key for CWD.PW image hosting, including optional `/tldr` infographic uploads when enabled.

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
ENABLE_IMG2=false
IMG2_API_KEY=
```

## Notes and limitations
- Webhook mode is not implemented in this port yet; polling only.
- Video generation can take a few minutes and returns an MP4 when the Veo operation completes.
- If `TELEGRAPH_ACCESS_TOKEN` is not set, long messages will be truncated instead of published.

## Search normalization and request consistency

Search normalization version 3 retains the first 4,000 semantic characters and up to 8,000 characters of complete tokens, plus a canonical single-token chat identifier. Every MATCH expression includes that identifier as well as an independent SQL chat predicate. SQLite schema version 2 uses an external-content FTS index over `messages`; it no longer duplicates normalized text in an FTS content table. Composite and uniqueness indexes replace the redundant standalone chat/message/date indexes.

Before deploying, stop the old bot and take a consistent backup using SQLite's backup API or a clean shutdown/checkpoint; copying only the main file while WAL writes continue is not a consistent backup. Test the upgrade on a copy first. Schema migration is transactional and preserves raw messages and provenance. One resumable background backfill populates the new derived index; `/s`, `/qc`, and term analytics are unavailable until it finishes, while message recording continues. Missing-index repair retains the same safeguards. Backfill verifies external-content integrity and does not run a second full FTS rebuild or automatic `VACUUM`.

Older binaries reject `PRAGMA user_version = 2`. Rollback requires stopping this binary and restoring the pre-upgrade consistent backup, then reconciling/exporting messages recorded after the backup before restarting an older binary. Do not lower `user_version` manually or replace only derived tables as a rollback. See [SQLite's external-content synchronization requirements](https://www.sqlite.org/fts5.html#external_content_tables).

Model pickers revalidate against the current catalog when answered or timed out. Once execution starts, provider configuration and Codex metadata are fixed for that request; legacy metadata is refreshed before being fixed, and authentication still checks that the active account matches. External media request headers overlap, but body reads and byte-budget admission follow input order so a faster later attachment cannot displace an earlier one.

Answers retain original Markdown until delivery. Telegram receives HTML, Telegraph receives nodes built from the original Markdown, and failed formatting/publishing falls back to readable bounded text with link destinations. Model labels are rendered separately as literal metadata.

Generated-image delivery for `/img`, `/image`, `/img2`, `/paintme`, and `/portraitme` retries transient Telegram failures without repeating generation. Partial delivery reports the number confirmed by Telegram; persona prompts and audit labels remain distinct.

Chat-context tools batch neighboring-message retrieval into one database query. Message text is capped at 4,000 characters per message and 32,000 characters per tool result (including snippets); matched messages take priority over surrounding context. Results include truncation/omission metadata and only returned message IDs are registered for citations. Diagnostic log tails scan backward from the end through at most 256 KiB and mark incomplete tails.

To reproduce the synthetic search benchmark, run `cargo test --offline synthetic_search_measurement -- --ignored --nocapture --test-threads=1` separately from the normal test gate. It creates 32,000 synthetic messages across 200 chats under `target/test-dbs` and writes measurements to ignored `agent_logs/search_measurement.json`. It does not use the configured runtime database. Page savings describe allocated pages in fresh synthetic databases; an upgraded existing file may retain freed pages for reuse because migration does not run `VACUUM`.
