# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains all Rust source files.
  - `src/main.rs` wires the Telegram dispatcher and startup.
  - `src/config.rs` loads environment variables and defaults.
  - `src/state.rs` defines `AppState`, the cloneable shared state (DB handle, bot identity, pending-request maps, media-group cache, heavy-command semaphore).
  - `src/handlers/` holds command routing, access control, and response logic.
  - `src/llm/` integrates Gemini, OpenRouter/NVIDIA/Ollama, OpenAI Responses, and ChatGPT Codex providers, plus the tool-runtime loop and web search helpers.
  - `src/db/` manages SQLite models, the chat-search index, and the async write queue.
  - `src/tools/` extracts Telegraph/Twitter link content and uploads images to CWD.PW.
  - `src/utils/` contains logging, timing, and HTTP helpers.
- Root files: `Cargo.toml`, `Dockerfile`, `README.md`.
- Data/log outputs are runtime artifacts under `./data` and `./logs`.

## Build, Test, and Development Commands
- `cargo build` - Compile the bot in debug mode.
- `cargo run` - Run locally using `.env` in the repo root.
- `cargo build --release` - Produce an optimized binary.
- `cargo test` - Run the unit-test suite (tests live in `#[cfg(test)] mod tests` blocks beside the code).
- `cargo test <name>` - Run a single test or module (e.g. `cargo test build_display_label_map`); add `-- --nocapture` to see captured output.
- `docker build -t telegram-group-helper-bot .` - Build the container image.
- `docker run --env-file .env -v ./data:/app/data -v ./logs:/app/logs telegram-group-helper-bot` - Run with persisted DB/logs.
- `cargo fmt` - Apply Rust formatting (Recommended).
- `cargo clippy` - Lint for common mistakes (Recommended).

## Build Verification
- After big code changes, run `cargo build` and debug any issues before finishing.
- Before delivering Rust code changes, run `cargo clippy --all-targets -- -D warnings` and fix any issues.

## Architecture Overview
Read these together before changing request-handling behavior:
- **Dispatch (`main.rs`).** A teloxide `dptree` routes updates into three message branches (commands, media groups, free text) plus a callback-query branch. The `Command` enum derives `BotCommands` (descriptions are in Chinese). Heavy commands are `tokio::spawn`ed so the dispatcher never blocks; their errors are logged, not propagated. Light commands (`/start`, `/help`, `/support`) run inline.
- **Shared state (`state.rs`).** `AppState` is cloned into every handler. It owns the `Database`, the bot's own id/username (used to detect mentions for auto-`/q`), `Arc<Mutex<HashMap>>` maps of pending interactive requests, an in-memory media-group cache, and a Tokio `Semaphore` enforcing `HEAVY_COMMAND_MAX_CONCURRENCY`. Throttle heavy work with `acquire_heavy_command_permit()`.
- **Interactive flows.** Selection menus (model picker, image options, Codex model/reasoning) reply with inline keyboards, stash a pending request in the matching `AppState` map keyed by callback-data prefix, and resolve in `handle_callback_query`; unanswered menus fall back to defaults after `MODEL_SELECTION_TIMEOUT`.
- **Persistence (`db/`).** Incoming messages are recorded through an async write queue (batched by `DB_WRITE_BATCH_SIZE`/`DB_WRITE_FLUSH_MS`) rather than written inline. `db/search.rs` maintains the normalized chat-search index (jieba tokenization) backing `/s` and `/qc`.
- **LLM layer (`llm/`).** `runtime_models.rs` exposes one model catalog: built-in `gemini` plus runtime models from `third_party_models.json` and the `openai-codex:selected` alias. Tool-capable models run through `*_with_tool_runtime` (`tool_runtime.rs`), which drives web search (`WEB_SEARCH_PROVIDERS` order: brave/exa/jina), Telegraph/Twitter enrichment, and image hosting. `audit.rs` records per-call token usage powering `/burn_baby_burn`, `/token_devourers`, and `/token_stats`.
- **Capability gates.** `ENABLE_GEMINI=false` disables Gemini-only commands (`/vid`, `/mysong`) and hides them from pickers, but `/s` survives if another `tools=true` model is ready. Provider `ENABLE_*` flags plus `DEFAULT_TEXT_MODEL`/`DEFAULT_IMAGE_MODEL` shape what is offered at runtime.

## Coding Style & Naming Conventions
- Rust style follows `rustfmt` defaults; use 4-space indentation.
- Module/file names: `snake_case` (e.g., `handlers/commands.rs`).
- Types: `CamelCase`; functions/vars: `snake_case`; constants: `SCREAMING_SNAKE_CASE`.
- Keep async boundaries clear and prefer early returns for error handling.

## Testing Guidelines
- Unit tests live in `#[cfg(test)] mod tests` blocks next to the code they cover (e.g. `handlers/mod.rs`, `handlers/qa.rs`, `db/database.rs`); run them with `cargo test`.
- Place new unit tests near their module; add cross-module integration tests under `tests/` (none exist yet).
- Aim for coverage on handlers and DB helpers when adding new features.

## Commit & Pull Request Guidelines
- Use short, imperative commit summaries; prefix with a Conventional Commit type (`feat:`, `fix:`) to match existing history.
  - Examples: `feat: add openrouter model loader`, `fix: handle missing whitelist file`.
- PRs should include: a brief description, how you tested (`cargo run`, `cargo test`), and any new env vars.

## Security & Configuration Notes
- Never commit `.env` or secrets. Keep API keys in environment variables only.
- The SQLite database path comes from `DATABASE_URL`; use `sqlite:///...` for absolute paths.
- `allowed_chat.txt` (path via `WHITELIST_FILE_PATH`) gates admin commands (`/status`, `/diagnose`, Codex admin) and any `ACCESS_CONTROLLED_COMMANDS`; keep it out of version control if it contains private IDs.

## Agent Execution Logging
- For each substantial implementation or investigation request, create a log file under `agent_logs/`.
- Name format: `<YYYYMMDD_HHMMSS>_<task-name>.md` — timestamp first so files sort chronologically and the latest logs are easy to find, followed by a descriptive task name.
- Each log file must include:
  - The user prompt/request that triggered the work.
  - The goal of this task
  - The implementation plan used.
  - The actions completed (files touched, code change summary, commands run with results).
  - Validation performed (build/tests/other checks) and outcomes.
  - Follow-ups or TODOs
- Append when relevant (skip a heading if it's empty — don't pad):
  - Design decisions — choices made when the spec/conversation was ambiguous, with the reason.
  - Deviations — intentional departures from spec or original plan, with why.
  - Tradeoffs — alternatives you'd defend to a reviewer, not micro-choices.
  - Open questions — things you want the user to confirm or revise.
  - Update at decision points, not on routine progress. At session end, re-read the originating prompt and check whether anything in the implementation contradicts a literal reading; if yes, log it under Deviations.
- Do not include secrets, API keys, or sensitive runtime credentials in log content.
