# /qc General Chat-Analytics Capability — Design Spec

**Status:** Approved direction (multi-agent design discussion + owner decisions, 2026-06-21). Supersedes `docs/superpowers/plans/2026-06-21-qc-message-count-stats.md`.

**Owner decisions:**
- Query mechanism → **structured analytics tool** the agent drives in a loop (not raw SQL, not a sandbox).
- v1 scope → **both** the analytics/aggregation capability **and** the LLM map-reduce topic-discovery path.

## Goal
Give the `/qc` agent a genuine, general capability to accurately answer stat/analysis questions about the **active chat** — counts, rankings, per-user/time-bucket trends, keyword-mention frequencies, and "hot topics" — by (1) loosening tool-call budgets and (2) giving it a flexible structured query tool it composes and iterates itself, hard-scoped to the active chat and read-only.

## Non-goals
- Raw/free-form SQL, a SQL parser, or an ephemeral sandbox DB (rejected: the authorizer can't enforce row-level `chat_id` scoping, and the sensitive co-located data — other chats' `messages`, `llm_invocations.message_text`, `llm_requests` — makes raw SQL a bad trade for ~zero gain on realistic questions).
- Access to any table other than `messages`. Token-usage analytics stay in the existing `/token_*` commands.
- Window functions / arbitrary multi-step analytics in v1 (reachable later via iterative tool calls; a sandboxed raw-SQL escalation can be added behind a flag only if real demand appears).
- Changing `/factcheck` — it never consults the chat DB (web-only, `agents/factcheck.rs`); its weakness is a separate effort.

## Architecture overview
Insert a **Phase-0 classifier** into `run_qc_pipeline` (`src/agents/qc.rs:132`, before the keyword planner) that routes each `/qc` into one of three lanes:

1. **`recall`** ("what did X say about Y") → existing planner→search→reflect→compose pipeline, unchanged (keeps the token-cheap path for the common case).
2. **`analytics`** (counts/rankings/trends/mention-frequency) → a **model-driven tool loop** with a new `chat_analytics` tool and raised budgets. The model composes a query spec, sees aggregate numbers, refines (add filter / change grouping / re-rank), then writes prose. The tool-forbidding `QC_EVIDENCE_ADDENDUM` (`qc.rs:59`) does **not** apply on this lane.
3. **`topic_discovery`** ("hot topics") → an LLM map-reduce over the chat-scoped window, then exact counts attached via `chat_analytics`.

The model-driven loop reuses the existing machinery: `call_*_with_tool_runtime` (`src/llm/third_party.rs:767-825`, loop bound `max_total_successful_calls + 2`) and its Gemini twin (used at `qa.rs:1646`). The classifier is one cheap `call_step_text` reusing the `StepModel` already resolved at `qc.rs:124`. Uncertain classification biases to `analytics`.

## Component 1 — `chat_analytics` tool
A new tool exposed via `ToolRuntime` (the existing chat-scope chokepoint; `chat_id` is a private field at `tool_runtime.rs:56`, never in any tool schema). Its argument is a structured, allow-listed **QuerySpec**; Rust compiles it to exactly one parameterized statement.

QuerySpec (serde enum/struct, validated in Rust):
```
metric:       count | distinct_count | min_date | max_date | avg_len | sum_len
distinct_on:  user_id | username            (required iff metric=distinct_count)
group_by:     user | day | hour_of_day | weekday | month | none
filters:
  term:            string|null   -> messages_fts MATCH ?      (word/CJK tokens, reuses /s FTS)
  text_contains:   string|null   -> text LIKE ? ESCAPE '\'    (literal substring; %_\ escaped in Rust)
  date_from:       RFC3339|null  -> AND date >= ?
  date_to:         RFC3339|null  -> AND date <  ?             (exclusive)
  user_id:         i64|null
  username:        string|null
  exclude_commands:  bool        -> AND is_command = 0
  exclude_ai_asks:   bool        -> AND asks_ai = 0
  exclude_synthetic: bool        -> AND is_synthetic_record = 0
order:        value_desc | value_asc | group_asc | group_desc
limit:        1..=50
```
**Compilation rules** (closed translation — every enum arm → a fixed SQL fragment; only *values* bind, never identifiers): `group_by=user` → group on `user_id` with the latest-non-null-username correlated subquery already used by `select_top_chat_token_users` (`database.rs:330-351`); `day/hour_of_day/weekday/month` → `date(date)` / `strftime('%H'|'%w'|'%Y-%m', date)` (the `date` column is RFC3339 TEXT, parsed natively). Always `… FROM messages WHERE chat_id = ? …`, `chat_id` bound from `ToolRuntime.chat_id`. Indices `idx_messages_chat_date` (`database.rs:931`) and `idx_messages_chat_user_date` (`:935`) back the common groupings.

**Output (aggregation-biased):** `{ "rows": [{"group": "...", "value": N}], "row_count": K, "truncated": bool, "scope": {"chat":"active","date_from":...,"date_to":...,"timezone":"UTC"} }` — numbers/groupings only, never message bodies. Row cap = `limit` (≤50, hard ≤100 groups); total payload byte-capped (~16 KiB).

**Default exclusions** (match the message-count semantics already agreed): `is_synthetic_record=0`, optionally `is_command=0`, and exclude GroupAnonymousBot id `1087968824` for per-user groupings; `user_id IS NOT NULL` for user grouping. Disclosed in results.

## Component 2 — Routing & analytics loop
- New `ToolProfile::ChatAnalytics` (`tool_runtime.rs:20-24`) + new `ToolName::ChatAnalytics` (`:676-680`) + a budget counter (`:53-67`) + a gate arm in `begin_tool_call` (`:398-440`) + dispatch in `execute_tool` (`:377-396`) + schema in `build_openai_function_tools`/`build_gemini_tools`.
- **Budgets (config-driven; new `QC_ANALYTICS_*` env vars alongside the `factcheck_*` clamps at `config.rs:679-681`):** total tool calls **12**; `chat_analytics` **10**; `chat_context_query` **3** (so the model can quote one representative message); `web_search` **1–2**. Per-query timeout **2 s** (`tokio::time::timeout` around the sqlx call). Per-request soft budget **60 s** (the global `WallClock` 480 s, `step.rs:283`, stays as the hard ceiling). Output: ≤100 groups, ≤~40 detail messages total, ≤16 KiB/call, ≤64 KiB/turn.
- The analytics lane runs on the existing sqlx pool (no new connection needed — every statement is a bounded, read-only, chat-bound `SELECT`; the timeout + `LIMIT` cover the worst case of a one-chat full scan). `db_max_connections` (default 5, `config.rs:655`) may need a small bump.
- Citation verifier (`qa.rs:1161-1198`) stays a clean no-op on analytics lanes (no message-link citations).

## Component 3 — Topic discovery (hot topics)
Topic discovery is **not** a SQL problem (`GROUP BY text` groups identical strings, not themes). Reuse the **TLDR map-reduce machinery**:
1. Pull the chat-scoped window (date-bounded chat read), capped at `tldr_max_messages` (`config.rs:678`, ~2000).
2. **Map:** chunk (~100 msgs/chunk, `config.rs:677`), extract candidate topics per chunk via a topic-labelling variant of `TLDR_CHUNK_PROMPT` (`config.rs:757`); ≤4 concurrent, ≤20 map calls.
3. **Reduce:** merge/cluster topic labels, emit top-N (≤2 reduce calls, `TLDR_MERGE_PROMPT` variant `config.rs:766`).
4. **Exact counts:** for each named topic, issue a `chat_analytics` call (`metric=count, term=<topic keywords>, date_from/date_to`) to attach a precise count — a clean composition the builder enables.
- Time budget ~90 s. If the window exceeds the cap, **narrow the range or return clearly-labeled sampled estimates** — never present sampled counts as exact. Gate behind a flag (e.g. `QC_ENABLE_TOPIC_DISCOVERY`, default true when a step model exists).

## Safety invariant & test strategy
**Invariant:** *Every DB-derived output is a projection/aggregation exclusively over `messages` rows whose `chat_id` equals the server-bound active chat; no model argument can alter scope or address another table; nothing writes.*
Enforced by: allow-listed QuerySpec (no model-supplied identifiers), `chat_id` bound from `ToolRuntime.chat_id`, aggregation-biased output, per-query timeout + row/byte caps.
**Tests (hard requirement before merge):**
- Spec→SQL compilation unit tests (no DB): each metric/group_by/order emits the expected SQL skeleton + bind order; `text_contains` escapes `% _ \`.
- **Two-chat sentinel property tests:** seed chat A (active) + chat B + `llm_invocations`/`app_meta` rows with unique sentinel strings; fuzz QuerySpecs (incl. a model-supplied `chat_id`, attempts to name other tables, `UNION`, `;`, comments) — assert no result/count is influenced by B or audit rows and no sentinel ever appears. Mirrors `search_chat_messages_stays_within_the_requested_chat` (`database.rs:1704`).
- Per-metric correctness, time-bucket bucketing, filter combos, budget-exhaustion for the new profile (mirror `qc_budget_stops_after_expected_counts`, `tool_runtime.rs:863`).
- Topic-discovery: window cap → sampling flag set; exact-count attach calls `chat_analytics`.
- Gate: `cargo test` + `cargo clippy --all-targets -- -D warnings`.

## Correctness/disclosure caveats (Codex risks)
- State date ranges + timezone (UTC) explicitly in results.
- Group by stable `user_id`; display latest username via `build_display_label_map` (`handlers/mod.rs:25`).
- Text-only coverage: only stored text/caption messages are counted (media-only/stickers/voice/service/edits/bare-commands aren't recorded) — disclose.
- Never present sampled topic counts as exact.

## Scope of impact
Generalizes to `/s` for free (shared `ToolRuntime`). Does **not** apply to `/factcheck` (no chat-DB access). `/q` (no chat retrieval) unaffected.

## Files to touch
- `src/llm/tool_runtime.rs` — `ToolProfile::ChatAnalytics`, `ToolName`, budget field + gate, `execute_tool` arm, `chat_analytics` schema (OpenAI + Gemini), `run_analytics_query`.
- `src/db/database.rs` — `run_chat_analytics(chat_id, spec)` (QuerySpec→parameterized SELECT) + a date-bounded chat-scoped window read for topic discovery (extend `get_messages_from_id`/`get_last_n_text_messages`).
- `src/db/models.rs` — `QuerySpec`, `AnalyticsRow` types (or a new `src/llm/analytics.rs` for the spec + compiler).
- `src/agents/qc.rs` — Phase-0 classifier, lane routing, analytics loop wiring, topic-discovery map-reduce, prompts/addenda.
- `src/handlers/qa.rs` — route `analytics`/`topic_discovery` to the model-driven loop vs the agentic compose.
- `src/config.rs` — `QC_ANALYTICS_*` budgets, `QC_ENABLE_TOPIC_DISCOVERY`.
- Tests in `database.rs`, `tool_runtime.rs`, `qc.rs`.

## Implementation phases (to be expanded into a task-by-task plan)
1. QuerySpec type + `run_chat_analytics` compiler + scope sentinel tests (DB layer).
2. `chat_analytics` tool + `ToolProfile::ChatAnalytics` + budgets + schemas + dispatch (runtime layer).
3. Phase-0 classifier + `recall`/`analytics` routing + model-driven analytics loop wiring.
4. Topic-discovery map-reduce (reuse TLDR) + exact-count attach + sampling disclosure.
5. Config knobs, disclosure wiring, full verification (tests + clippy), manual smoke on all three lanes + regression on plain recall `/qc`.

## Open items to confirm while writing the detailed plan
Exact signatures to read before writing no-placeholder code: `call_*_with_tool_runtime` loop (`third_party.rs:767`) and `call_gemini_with_tool_runtime`; the TLDR chunk/merge functions and prompts; `call_step_text` schema usage for the classifier; `get_messages_from_id`/date-window reads in `database.rs`.
