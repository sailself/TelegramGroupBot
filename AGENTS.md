# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains all Rust source files.
  - `src/main.rs` wires the Telegram dispatcher and startup.
  - `src/config.rs` loads environment variables and defaults.
  - `src/handlers/` holds command routing and response logic.
  - `src/llm/` integrates Gemini/OpenRouter and web search helpers.
  - `src/db/` manages SQLite models and the async write queue.
  - `src/utils/` contains logging, timing, and HTTP helpers.
- Root files: `Cargo.toml`, `Dockerfile`, `README.md`.
- Data/log outputs are runtime artifacts under `./data` and `./logs`.

## Build, Test, and Development Commands
- `cargo build` - Compile the bot in debug mode.
- `cargo run` - Run locally using `.env` in the repo root.
- `cargo build --release` - Produce an optimized binary.
- `docker build -t telegram-group-helper-bot .` - Build the container image.
- `docker run --env-file .env -v ./data:/app/data -v ./logs:/app/logs telegram-group-helper-bot` - Run with persisted DB/logs.
- `cargo fmt` - Apply Rust formatting (recommended).
- `cargo clippy` - Lint for common mistakes (optional).
## Build Verification
- After big code changes, run `cargo build` and debug any issues before finishing.

## Coding Style & Naming Conventions
- Rust style follows `rustfmt` defaults; use 4-space indentation.
- Module/file names: `snake_case` (e.g., `handlers/commands.rs`).
- Types: `CamelCase`; functions/vars: `snake_case`; constants: `SCREAMING_SNAKE_CASE`.
- Keep async boundaries clear and prefer early returns for error handling.

## Testing Guidelines
- No dedicated test suite yet. Use `cargo test` for any new tests you add.
- When adding tests, place unit tests near the module (`mod tests`) or add integration tests under `tests/`.
- Aim for coverage on handlers and DB helpers when adding new features.

## Commit & Pull Request Guidelines
- This folder does not include Git history. If you create commits, use short, imperative summaries.
  - Examples: `feat: add openrouter model loader`, `fix: handle missing whitelist file`.
- PRs should include: a brief description, how you tested (`cargo run`, `cargo test`), and any new env vars.

## Security & Configuration Notes
- Never commit `.env` or secrets. Keep API keys in environment variables only.
- The SQLite database path comes from `DATABASE_URL`; use `sqlite:///...` for absolute paths.
- `allowed_chat.txt` controls access; keep it out of version control if it contains private IDs.

## Agent Execution Logging
- For each substantial implementation or investigation request, create a log file under `agent_logs/`.
- Name format: `<task-name>_<YYYYMMDD_HHMMSS>.md` using a descriptive task name and timestamp.
- Each log file must include:
  - The user prompt/request that triggered the work.
  - The implementation plan used.
  - The actions completed (files changed, behavior added/updated).
  - Validation performed (build/tests/other checks) and outcomes.
- Do not include secrets, API keys, or sensitive runtime credentials in log content.
