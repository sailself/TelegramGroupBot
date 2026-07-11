use std::collections::{BTreeMap, BTreeSet};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::config::CONFIG;
use crate::db::database::Database;
use crate::db::models::{ChatSearchHit, MessageRow};
use crate::db::search::SEARCH_INDEX_REBUILDING_ERROR;
use crate::llm::web_search::{self, web_search_tool};
use crate::utils::telegram::build_message_link;

const DEFAULT_QC_SEARCH_LIMIT: usize = 8;
const DEFAULT_S_SEARCH_LIMIT: usize = 20;
const MAX_SEARCH_LIMIT: usize = 20;
const MAX_SEARCH_OFFSET: usize = 250;
const MAX_CONTEXT_WINDOW: usize = 5;
const MAX_WEB_RESULTS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)] // All three variants are "Chat*" by design
pub enum ToolProfile {
    ChatQuestion,
    ChatSearch,
    ChatAnalytics,
}

#[derive(Debug, Clone, Copy)]
pub struct ToolBudgetConfig {
    pub max_total_successful_calls: usize,
    pub max_web_search_calls: usize,
    pub max_chat_context_query_calls: usize,
    pub max_chat_analytics_query_calls: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct ToolBudgetSnapshot {
    total_remaining: usize,
    web_search_remaining: usize,
    chat_context_query_remaining: usize,
    chat_analytics_query_remaining: usize,
}

#[derive(Debug, Clone, Copy)]
enum ToolBudgetErrorKind {
    Total,
    WebSearch,
    ChatContextQuery,
    ChatAnalytics,
    Disabled,
}

#[derive(Debug, Clone, Copy)]
struct ToolBudgetError {
    kind: ToolBudgetErrorKind,
}

#[derive(Clone)]
pub struct ToolRuntime {
    db: Database,
    chat_id: i64,
    profile: ToolProfile,
    budget: ToolBudgetConfig,
    successful_calls: usize,
    web_search_calls: usize,
    chat_context_query_calls: usize,
    chat_analytics_query_calls: usize,
    force_final_answer: bool,
    accumulated_hits: BTreeMap<i64, ChatSearchHit>,
    // Every message id surfaced to the model — search hits plus their context
    // windows plus window-op results — used to verify /qc citations are real.
    returned_message_ids: BTreeSet<i64>,
    // Authoritative analytics results accumulated across tool calls (A3).
    analytics_results: Vec<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum ChatContextQueryArgs {
    Search {
        query: String,
        limit: Option<usize>,
        offset: Option<usize>,
        context_before: Option<usize>,
        context_after: Option<usize>,
    },
    Window {
        message_id: i64,
        context_before: Option<usize>,
        context_after: Option<usize>,
    },
}

#[derive(Debug, Serialize)]
struct ToolMessage {
    message_id: i64,
    username: Option<String>,
    date_utc: String,
    text: String,
    link: Option<String>,
    asks_ai: bool,
    ai_command: Option<String>,
    is_synthetic_record: bool,
}

#[derive(Debug, Serialize)]
struct ToolSearchHit {
    message_id: i64,
    username: Option<String>,
    date_utc: String,
    text: String,
    snippet: String,
    link: Option<String>,
    score: f64,
    match_stage: String,
    asks_ai: bool,
    ai_command: Option<String>,
    is_synthetic_record: bool,
    context_messages: Vec<ToolMessage>,
}

impl ToolRuntime {
    pub fn for_qc(db: Database, chat_id: i64) -> Self {
        Self {
            db,
            chat_id,
            profile: ToolProfile::ChatQuestion,
            budget: ToolBudgetConfig {
                max_total_successful_calls: 8,
                max_web_search_calls: 3,
                max_chat_context_query_calls: 5,
                max_chat_analytics_query_calls: 0,
            },
            successful_calls: 0,
            web_search_calls: 0,
            chat_context_query_calls: 0,
            chat_analytics_query_calls: 0,
            force_final_answer: false,
            accumulated_hits: BTreeMap::new(),
            returned_message_ids: BTreeSet::new(),
            analytics_results: Vec::new(),
        }
    }

    pub fn for_search(db: Database, chat_id: i64) -> Self {
        Self {
            db,
            chat_id,
            profile: ToolProfile::ChatSearch,
            budget: ToolBudgetConfig {
                max_total_successful_calls: 5,
                max_web_search_calls: 0,
                max_chat_context_query_calls: 5,
                max_chat_analytics_query_calls: 0,
            },
            successful_calls: 0,
            web_search_calls: 0,
            chat_context_query_calls: 0,
            chat_analytics_query_calls: 0,
            force_final_answer: false,
            accumulated_hits: BTreeMap::new(),
            returned_message_ids: BTreeSet::new(),
            analytics_results: Vec::new(),
        }
    }

    pub fn for_analytics(db: Database, chat_id: i64) -> Self {
        Self {
            db,
            chat_id,
            profile: ToolProfile::ChatAnalytics,
            budget: ToolBudgetConfig {
                max_total_successful_calls: CONFIG.qc_analytics_max_total_calls,
                max_web_search_calls: 0,
                max_chat_context_query_calls: 1,
                max_chat_analytics_query_calls: CONFIG.qc_analytics_max_query_calls,
            },
            successful_calls: 0,
            web_search_calls: 0,
            chat_context_query_calls: 0,
            chat_analytics_query_calls: 0,
            force_final_answer: false,
            accumulated_hits: BTreeMap::new(),
            returned_message_ids: BTreeSet::new(),
            analytics_results: Vec::new(),
        }
    }

    pub fn analytics_results(&self) -> &[Value] {
        &self.analytics_results
    }

    pub fn force_final_answer(&self) -> bool {
        self.force_final_answer
    }

    pub fn max_total_successful_calls(&self) -> usize {
        self.budget.max_total_successful_calls
    }

    pub fn allows_web_search(&self) -> bool {
        self.profile == ToolProfile::ChatQuestion
    }

    pub fn tool_limit_guidance(&self) -> String {
        match self.profile {
            ToolProfile::ChatQuestion => {
                "Tool budgets for this request: use web_search at most 3 times and chat_context_query at most 5 times. Once a budget is exhausted, answer with the evidence you already have.".to_string()
            }
            ToolProfile::ChatSearch => {
                "Tool budgets for this request: use chat_context_query at most 5 times total. Search is keyword-based FTS, not semantic, so inspect snippets carefully and refine your query if needed.".to_string()
            }
            ToolProfile::ChatAnalytics => {
                format!(
                    "Tool budgets for this request: use chat_analytics for any counting/ranking/trend question (up to {} calls; refine the spec between calls). chat_context_query is limited to 1 small lookup to quote one example message.",
                    self.budget.max_chat_analytics_query_calls
                )
            }
        }
    }

    pub fn build_openai_function_tools(&self) -> Vec<Value> {
        let mut tools = Vec::new();
        if self.allows_web_search() && web_search::is_search_enabled() {
            tools.push(json!({
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web using the configured providers and return a concise Markdown summary.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to look up on the public web."
                            },
                            "max_results": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": MAX_WEB_RESULTS,
                                "description": "Maximum number of results to return."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }));
        }

        if self.profile == ToolProfile::ChatAnalytics {
            tools.push(json!({
                "type": "function",
                "function": {
                    "name": "chat_analytics",
                    "description": "Run a structured analytics query over this chat's message history. Returns counts, rankings, trends, and date metrics. Never accesses other chats.",
                    "parameters": crate::llm::analytics::query_spec_schema()
                }
            }));
        }

        tools.push(json!({
            "type": "function",
            "function": {
                "name": "chat_context_query",
                "description": "Retrieve messages from the current Telegram chat only. This tool never accesses other chats.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["search", "window"]
                        },
                        "query": {
                            "type": "string",
                            "description": "Plain text search intent for keyword/FTS search. Never send SQL or raw FTS syntax."
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_SEARCH_LIMIT,
                            "description": "Maximum number of hits to return."
                        },
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": MAX_SEARCH_OFFSET,
                            "description": "Offset for additional pages of search hits."
                        },
                        "context_before": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": MAX_CONTEXT_WINDOW,
                            "description": "Number of earlier messages to include around each hit."
                        },
                        "context_after": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": MAX_CONTEXT_WINDOW,
                            "description": "Number of later messages to include around each hit."
                        },
                        "message_id": {
                            "type": "integer",
                            "description": "Target message ID for the window operation."
                        }
                    },
                    "required": ["operation"]
                }
            }
        }));

        tools
    }

    pub fn build_gemini_tools(&self) -> Vec<Value> {
        let mut declarations = Vec::new();
        if self.allows_web_search() && web_search::is_search_enabled() {
            declarations.push(json!({
                "name": "web_search",
                "description": "Search the web using the configured providers and return a concise Markdown summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to look up on the public web."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return."
                        }
                    },
                    "required": ["query"]
                }
            }));
        }

        if self.profile == ToolProfile::ChatAnalytics {
            declarations.push(json!({
                "name": "chat_analytics",
                "description": "Run a structured analytics query over this chat's message history. Returns counts, rankings, trends, and date metrics. Never accesses other chats.",
                "parameters": crate::llm::analytics::query_spec_schema()
            }));
        }

        declarations.push(json!({
            "name": "chat_context_query",
            "description": "Retrieve messages from the current Telegram chat only. This tool never accesses other chats.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["search", "window"]
                    },
                    "query": {
                        "type": "string",
                        "description": "Plain text search intent for keyword/FTS search. Never send SQL or raw FTS syntax."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of hits to return."
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Offset for additional pages of search hits."
                    },
                    "context_before": {
                        "type": "integer",
                        "description": "Number of earlier messages to include around each hit."
                    },
                    "context_after": {
                        "type": "integer",
                        "description": "Number of later messages to include around each hit."
                    },
                    "message_id": {
                        "type": "integer",
                        "description": "Target message ID for the window operation."
                    }
                },
                "required": ["operation"]
            }
        }));

        vec![json!({ "functionDeclarations": declarations })]
    }

    /// Message IDs that `chat_context_query` actually returned during this run.
    /// Used to verify that a model's cited message links were genuinely
    /// retrieved rather than fabricated.
    pub fn accumulated_message_ids(&self) -> Vec<i64> {
        self.returned_message_ids.iter().copied().collect()
    }

    pub fn select_hits_by_message_ids(&self, ids: &[i64], max_hits: usize) -> Vec<ChatSearchHit> {
        let mut selected = Vec::new();
        let mut seen = BTreeMap::new();
        for message_id in ids {
            if selected.len() >= max_hits {
                break;
            }
            if seen.insert(*message_id, true).is_some() {
                continue;
            }
            if let Some(hit) = self.accumulated_hits.get(message_id) {
                selected.push(hit.clone());
            }
        }
        selected
    }

    /// Programmatic chat search for the agentic pipelines. Consumes the same
    /// `chat_context_query` budget as a model-driven call and records hits and
    /// returned message ids, so accumulated state and downstream citation
    /// verification behave identically.
    pub async fn run_search_query(
        &mut self,
        query: &str,
        limit: Option<usize>,
        context_before: usize,
        context_after: usize,
    ) -> Result<Value> {
        self.begin_tool_call(ToolName::ChatContextQuery)
            .map_err(|err| anyhow!(tool_budget_error_parts(err).1))?;
        self.run_chat_context_query(ChatContextQueryArgs::Search {
            query: query.to_string(),
            limit,
            offset: None,
            context_before: Some(context_before),
            context_after: Some(context_after),
        })
        .await
    }

    /// Programmatic web search consuming the same `web_search` budget (and
    /// profile gating) as a model-driven call.
    pub async fn run_web_search(&mut self, query: &str, max_results: usize) -> Result<String> {
        self.begin_tool_call(ToolName::WebSearch)
            .map_err(|err| anyhow!(tool_budget_error_parts(err).1))?;
        web_search_tool(query, Some(max_results.clamp(1, MAX_WEB_RESULTS))).await
    }

    pub async fn execute_tool(&mut self, name: &str, arguments: &Value) -> String {
        match name {
            "web_search" => match self.begin_tool_call(ToolName::WebSearch) {
                Ok(()) => self.execute_web_search(arguments).await,
                Err(err) => self.tool_budget_error_payload("web_search", err),
            },
            "chat_context_query" => match self.begin_tool_call(ToolName::ChatContextQuery) {
                Ok(()) => self.execute_chat_context_query(arguments).await,
                Err(err) => self.tool_budget_error_payload("chat_context_query", err),
            },
            "chat_analytics" => match self.begin_tool_call(ToolName::ChatAnalytics) {
                Ok(()) => self.execute_analytics(arguments).await,
                Err(err) => self.tool_budget_error_payload("chat_analytics", err),
            },
            _ => {
                self.force_final_answer = true;
                self.error_payload(
                    name,
                    "unsupported_tool",
                    "Unsupported tool call requested by the model.",
                )
            }
        }
    }

    fn begin_tool_call(&mut self, tool: ToolName) -> std::result::Result<(), ToolBudgetError> {
        if self.force_final_answer {
            return Err(ToolBudgetError {
                kind: ToolBudgetErrorKind::Disabled,
            });
        }
        if self.successful_calls >= self.budget.max_total_successful_calls {
            self.force_final_answer = true;
            return Err(ToolBudgetError {
                kind: ToolBudgetErrorKind::Total,
            });
        }

        match tool {
            ToolName::WebSearch => {
                if self.profile != ToolProfile::ChatQuestion || !web_search::is_search_enabled() {
                    self.force_final_answer = true;
                    return Err(ToolBudgetError {
                        kind: ToolBudgetErrorKind::Disabled,
                    });
                }
                if self.web_search_calls >= self.budget.max_web_search_calls {
                    self.force_final_answer = true;
                    return Err(ToolBudgetError {
                        kind: ToolBudgetErrorKind::WebSearch,
                    });
                }
                self.web_search_calls += 1;
            }
            ToolName::ChatContextQuery => {
                if self.chat_context_query_calls >= self.budget.max_chat_context_query_calls {
                    self.force_final_answer = true;
                    return Err(ToolBudgetError {
                        kind: ToolBudgetErrorKind::ChatContextQuery,
                    });
                }
                self.chat_context_query_calls += 1;
            }
            ToolName::ChatAnalytics => {
                if self.chat_analytics_query_calls >= self.budget.max_chat_analytics_query_calls {
                    self.force_final_answer = true;
                    return Err(ToolBudgetError {
                        kind: ToolBudgetErrorKind::ChatAnalytics,
                    });
                }
                self.chat_analytics_query_calls += 1;
            }
        }

        self.successful_calls += 1;
        Ok(())
    }

    fn remaining_budget_snapshot(&self) -> ToolBudgetSnapshot {
        ToolBudgetSnapshot {
            total_remaining: self
                .budget
                .max_total_successful_calls
                .saturating_sub(self.successful_calls),
            web_search_remaining: self
                .budget
                .max_web_search_calls
                .saturating_sub(self.web_search_calls),
            chat_context_query_remaining: self
                .budget
                .max_chat_context_query_calls
                .saturating_sub(self.chat_context_query_calls),
            chat_analytics_query_remaining: self
                .budget
                .max_chat_analytics_query_calls
                .saturating_sub(self.chat_analytics_query_calls),
        }
    }

    async fn execute_web_search(&self, arguments: &Value) -> String {
        let query = arguments
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        let max_results = arguments
            .get("max_results")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(5)
            .clamp(1, MAX_WEB_RESULTS);

        if query.is_empty() {
            return self.error_payload(
                "web_search",
                "invalid_arguments",
                "The web_search tool requires a non-empty query string.",
            );
        }

        match web_search_tool(query, Some(max_results)).await {
            Ok(result) => self.success_payload(
                "web_search",
                json!({
                    "query": query,
                    "max_results": max_results,
                    "result_markdown": result,
                }),
            ),
            Err(err) => self.error_payload("web_search", "tool_execution_failed", &err.to_string()),
        }
    }

    async fn execute_chat_context_query(&mut self, arguments: &Value) -> String {
        let args: ChatContextQueryArgs = match serde_json::from_value(arguments.clone()) {
            Ok(args) => args,
            Err(err) => {
                return self.error_payload(
                    "chat_context_query",
                    "invalid_arguments",
                    &format!("Invalid chat_context_query arguments: {err}"),
                );
            }
        };

        match self.run_chat_context_query(args).await {
            Ok(payload) => self.success_payload("chat_context_query", payload),
            Err(err) if err.to_string().contains(SEARCH_INDEX_REBUILDING_ERROR) => self
                .error_payload(
                    "chat_context_query",
                    SEARCH_INDEX_REBUILDING_ERROR,
                    "The chat search index is still rebuilding. Stop using chat_context_query and explain that search is temporarily unavailable.",
                ),
            Err(err) => {
                self.error_payload("chat_context_query", "tool_execution_failed", &err.to_string())
            }
        }
    }

    async fn run_chat_context_query(&mut self, args: ChatContextQueryArgs) -> Result<Value> {
        match args {
            ChatContextQueryArgs::Search {
                query,
                limit,
                offset,
                context_before,
                context_after,
            } => {
                let query = query.trim();
                if query.is_empty() {
                    return Err(anyhow!(
                        "chat_context_query search requires a non-empty query"
                    ));
                }

                let default_limit = match self.profile {
                    ToolProfile::ChatQuestion => DEFAULT_QC_SEARCH_LIMIT,
                    ToolProfile::ChatSearch => DEFAULT_S_SEARCH_LIMIT,
                    ToolProfile::ChatAnalytics => 3, // Decision 1: only a representative quote
                };
                let mut limit = limit.unwrap_or(default_limit).clamp(1, MAX_SEARCH_LIMIT);
                let offset = offset.unwrap_or(0).clamp(0, MAX_SEARCH_OFFSET);
                let mut context_before = context_before.unwrap_or(0).clamp(0, MAX_CONTEXT_WINDOW);
                let mut context_after = context_after.unwrap_or(0).clamp(0, MAX_CONTEXT_WINDOW);
                // Hard cap for analytics profile: only a small representative quote.
                if self.profile == ToolProfile::ChatAnalytics {
                    limit = limit.min(3);
                    context_before = 0;
                    context_after = 0;
                }

                let hits = self
                    .db
                    .search_chat_messages(self.chat_id, query, limit as i64, offset as i64)
                    .await?;
                for hit in &hits {
                    self.accumulated_hits.insert(hit.message_id, hit.clone());
                    self.returned_message_ids.insert(hit.message_id);
                }

                let mut results = Vec::new();
                for hit in hits {
                    let context_messages: Vec<ToolMessage> =
                        if context_before > 0 || context_after > 0 {
                            self.db
                                .get_message_window(
                                    self.chat_id,
                                    hit.message_id,
                                    context_before as i64,
                                    context_after as i64,
                                )
                                .await?
                                .unwrap_or_default()
                                .into_iter()
                                .map(message_row_to_tool_message)
                                .collect()
                        } else {
                            Vec::new()
                        };
                    for message in &context_messages {
                        self.returned_message_ids.insert(message.message_id);
                    }
                    results.push(hit_to_tool_search_hit(hit, context_messages));
                }

                Ok(json!({
                    "operation": "search",
                    "query": query,
                    "limit": limit,
                    "offset": offset,
                    "result_count": results.len(),
                    "results": results,
                }))
            }
            ChatContextQueryArgs::Window {
                message_id,
                context_before,
                context_after,
            } => {
                let context_before = context_before.unwrap_or(2).clamp(0, MAX_CONTEXT_WINDOW);
                let context_after = context_after.unwrap_or(2).clamp(0, MAX_CONTEXT_WINDOW);
                let Some(messages) = self
                    .db
                    .get_message_window(
                        self.chat_id,
                        message_id,
                        context_before as i64,
                        context_after as i64,
                    )
                    .await?
                else {
                    return Err(anyhow!(
                        "The requested message_id does not belong to the current chat or is unavailable."
                    ));
                };

                let messages = messages
                    .into_iter()
                    .map(message_row_to_tool_message)
                    .collect::<Vec<_>>();
                for message in &messages {
                    self.returned_message_ids.insert(message.message_id);
                }

                Ok(json!({
                    "operation": "window",
                    "message_id": message_id,
                    "result_count": messages.len(),
                    "messages": messages,
                }))
            }
        }
    }

    async fn execute_analytics(&mut self, arguments: &Value) -> String {
        match self.run_analytics_query(arguments).await {
            Ok(payload) => self.success_payload("chat_analytics", payload),
            Err(err) => {
                let msg = err.to_string();
                if msg.contains(SEARCH_INDEX_REBUILDING_ERROR) {
                    self.error_payload(
                        "chat_analytics",
                        SEARCH_INDEX_REBUILDING_ERROR,
                        "The chat search index is still rebuilding; retry without a term filter or explain that full-text analytics is temporarily unavailable.",
                    )
                } else if msg.starts_with("invalid analytics arguments") {
                    self.error_payload("chat_analytics", "invalid_arguments", &msg)
                } else {
                    self.error_payload("chat_analytics", "tool_execution_failed", &msg)
                }
            }
        }
    }

    pub async fn run_analytics_query(&mut self, arguments: &Value) -> Result<Value> {
        use crate::llm::analytics::QuerySpec;
        let spec: QuerySpec = serde_json::from_value(arguments.clone())
            .map_err(|error| anyhow!("invalid analytics arguments: {error}"))?;
        let (spec, rows) = self
            .db
            .run_chat_analytics(self.chat_id, &spec)
            .await
            .map_err(|error| {
                let message = error.to_string();
                if message.contains("date_")
                    || message.contains("term must")
                    || message.contains("distinct_count is only meaningful")
                {
                    anyhow!("invalid analytics arguments: {message}")
                } else {
                    error
                }
            })?;
        let label_map = crate::handlers::build_display_label_map(rows.iter().filter_map(|r| {
            r.group_user_id
                .map(|uid| (uid, r.group_key.as_deref().unwrap_or("Anonymous")))
        }));
        let out: Vec<Value> = rows
            .iter()
            .map(|r| {
                let group = match (r.group_user_id, &r.group_key) {
                    (Some(uid), _) => label_map
                        .get(&uid)
                        .cloned()
                        .unwrap_or_else(|| "Anonymous".into()),
                    (None, Some(k)) => k.clone(),
                    (None, None) => "all".into(),
                };
                let mut row = json!({ "group": group });
                if let Some(v) = r.value_num {
                    row["value"] = json!(v);
                }
                if let Some(t) = &r.value_text {
                    row["value"] = json!(t);
                }
                row
            })
            .collect();

        let payload = json!({
            "operation": "analytics",
            "query": spec,
            "coverage": {
                "chat": "active",
                "storage": "stored_text_messages",
                "timezone": "UTC",
                "anonymous_admin_and_channel_posts_excluded": true
            },
            "row_count": out.len(),
            "rows": out,
            "note": "Rows are authoritative database results over stored text messages. Media-only, sticker, voice, service, unrecorded edits, anonymous-admin posts, and channel posts are absent or excluded."
        });
        // Accumulate every result for A4's authoritative answer composition (A4 bounds
        // the rendered block length). The per-call budget caps how many accumulate, so
        // memory stays bounded.
        self.analytics_results.push(payload.clone());
        Ok(payload)
    }

    fn success_payload(&self, tool: &str, data: Value) -> String {
        json!({
            "ok": true,
            "tool": tool,
            "remaining": self.remaining_budget_snapshot(),
            "data": data,
        })
        .to_string()
    }

    fn error_payload(&self, tool: &str, error_code: &str, message: &str) -> String {
        json!({
            "ok": false,
            "tool": tool,
            "error_code": error_code,
            "error": message,
            "remaining": self.remaining_budget_snapshot(),
        })
        .to_string()
    }

    fn tool_budget_error_payload(&self, tool: &str, error: ToolBudgetError) -> String {
        let (error_code, message) = tool_budget_error_parts(error);
        self.error_payload(tool, error_code, message)
    }
}

fn tool_budget_error_parts(error: ToolBudgetError) -> (&'static str, &'static str) {
    match error.kind {
        ToolBudgetErrorKind::Total => (
            "total_budget_exhausted",
            "The total tool-call budget for this request is exhausted. Answer using the evidence already gathered.",
        ),
        ToolBudgetErrorKind::WebSearch => (
            "web_search_budget_exhausted",
            "The web_search budget for this request is exhausted. Answer using the evidence already gathered.",
        ),
        ToolBudgetErrorKind::ChatContextQuery => (
            "chat_context_query_budget_exhausted",
            "The chat_context_query budget for this request is exhausted. Answer using the evidence already gathered.",
        ),
        ToolBudgetErrorKind::ChatAnalytics => (
            "chat_analytics_budget_exhausted",
            "The chat_analytics budget for this request is exhausted. Answer using the results already gathered.",
        ),
        ToolBudgetErrorKind::Disabled => (
            "tool_disabled",
            "This tool is unavailable for the current request. Answer using the evidence already gathered.",
        ),
    }
}

#[derive(Debug, Clone, Copy)]
enum ToolName {
    WebSearch,
    ChatContextQuery,
    ChatAnalytics,
}

fn message_row_to_tool_message(row: MessageRow) -> ToolMessage {
    ToolMessage {
        message_id: row.message_id,
        username: row.username,
        date_utc: row.date.to_rfc3339(),
        text: row.text.unwrap_or_default(),
        link: build_message_link(row.chat_id, row.message_id),
        asks_ai: row.asks_ai,
        ai_command: row.ai_command,
        is_synthetic_record: row.is_synthetic_record,
    }
}

fn hit_to_tool_search_hit(hit: ChatSearchHit, context_messages: Vec<ToolMessage>) -> ToolSearchHit {
    ToolSearchHit {
        message_id: hit.message_id,
        username: hit.username,
        date_utc: hit.date.to_rfc3339(),
        text: hit.text,
        snippet: hit.snippet,
        link: hit.link,
        score: hit.score,
        match_stage: hit.match_stage.label().to_string(),
        asks_ai: hit.asks_ai,
        ai_command: hit.ai_command,
        is_synthetic_record: hit.is_synthetic_record,
        context_messages,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::database::Database;
    use chrono::Utc;
    use tokio::runtime::Runtime;

    fn test_db_path(test_name: &str) -> std::path::PathBuf {
        let mut path = std::path::PathBuf::from("target");
        path.push("test-dbs");
        std::fs::create_dir_all(&path).expect("test db directory should exist");
        path.push(format!(
            "telegram-chat-bot-tool-runtime-{}-{}-{}.db",
            test_name,
            std::process::id(),
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let _ = std::fs::File::create(&path).expect("test db file should be creatable");
        path
    }

    fn sqlite_url_for_path(path: &std::path::Path) -> String {
        format!("sqlite://{}", path.to_string_lossy().replace('\\', "/"))
    }

    async fn init_test_db(test_name: &str) -> Database {
        let path = test_db_path(test_name);
        Database::init(&sqlite_url_for_path(&path))
            .await
            .expect("test database should initialize")
    }

    async fn insert_test_message(db: &Database, message_id: i64, chat_id: i64, text: &str) {
        let insert = crate::db::database::build_message_insert(
            Some(123_i64),
            Some("tester".to_string()),
            Some(text.to_string()),
            Some("en".to_string()),
            Utc::now(),
            None,
            Some(chat_id),
            Some(message_id),
            None,
            false,
            None,
            text.trim_start().starts_with('/'),
            false,
        );
        db.queue_message_insert(insert)
            .await
            .expect("message insert should queue");
        // Wait for the async write queue to flush the row before querying it.
        for _ in 0..200 {
            if let Ok(Some(rows)) = db.get_message_window(chat_id, message_id, 0, 0).await {
                if !rows.is_empty() {
                    return;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("message {message_id} was not persisted in time");
    }

    #[test]
    fn window_op_records_returned_message_ids_for_qc_verification() {
        let runtime = Runtime::new().expect("tokio runtime should initialize");
        runtime.block_on(async {
            let db = init_test_db("qc-returned-ids").await;
            let chat_id = -1001374348669_i64;
            for (id, text) in [(10_i64, "first"), (11, "middle"), (12, "third")] {
                insert_test_message(&db, id, chat_id, text).await;
            }

            let mut tool_runtime = ToolRuntime::for_qc(db, chat_id);
            let result = tool_runtime
                .run_chat_context_query(ChatContextQueryArgs::Window {
                    message_id: 11,
                    context_before: Some(2),
                    context_after: Some(2),
                })
                .await
                .expect("window query should succeed");
            assert_eq!(
                result.get("operation").and_then(Value::as_str),
                Some("window")
            );

            // P3 regression: every message surfaced via the window op is recorded,
            // so the /qc citation verifier won't flag them as fabricated.
            let mut ids = tool_runtime.accumulated_message_ids();
            ids.sort_unstable();
            assert_eq!(ids, vec![10, 11, 12]);
        });
    }

    #[test]
    fn programmatic_search_consumes_budget_and_records_ids() {
        let runtime = Runtime::new().expect("tokio runtime should initialize");
        runtime.block_on(async {
            let db = init_test_db("programmatic-search").await;
            let chat_id = -1001374348669_i64;
            for (id, text) in [(21_i64, "rust telegram bot"), (22, "unrelated chatter")] {
                insert_test_message(&db, id, chat_id, text).await;
            }

            let mut tool_runtime = ToolRuntime::for_qc(db, chat_id);
            let result = tool_runtime
                .run_search_query("telegram", None, 0, 0)
                .await
                .expect("programmatic search should succeed");
            assert_eq!(
                result.get("operation").and_then(Value::as_str),
                Some("search")
            );

            assert!(tool_runtime.accumulated_message_ids().contains(&21));
            assert_eq!(tool_runtime.successful_calls, 1);
            assert_eq!(tool_runtime.chat_context_query_calls, 1);
        });
    }

    #[test]
    fn programmatic_web_search_blocked_for_search_profile() {
        let runtime = Runtime::new().expect("tokio runtime should initialize");
        runtime.block_on(async {
            let db = init_test_db("programmatic-web-blocked").await;
            let mut tool_runtime = ToolRuntime::for_search(db, -1001374348669);

            // The search profile has no web budget, so this is rejected by the
            // budget gate before any network access happens.
            let result = tool_runtime.run_web_search("anything", 3).await;
            assert!(result.is_err());
            assert!(tool_runtime.force_final_answer());
            assert_eq!(tool_runtime.web_search_calls, 0);
        });
    }

    #[test]
    fn programmatic_search_errors_once_budget_is_exhausted() {
        let runtime = Runtime::new().expect("tokio runtime should initialize");
        runtime.block_on(async {
            let db = init_test_db("programmatic-budget-exhausted").await;
            let mut tool_runtime = ToolRuntime::for_qc(db, -1001374348669);
            tool_runtime.successful_calls = tool_runtime.budget.max_total_successful_calls;

            let result = tool_runtime.run_search_query("anything", None, 0, 0).await;
            assert!(result.is_err());
            assert!(tool_runtime.force_final_answer());
        });
    }

    #[test]
    fn qc_budget_stops_after_expected_counts() {
        let runtime = Runtime::new().expect("tokio runtime should initialize");
        let db = runtime.block_on(init_test_db("qc-budget"));
        let mut runtime = ToolRuntime::for_qc(db, -1001374348669);

        runtime.successful_calls = runtime.budget.max_total_successful_calls;
        assert!(runtime.begin_tool_call(ToolName::ChatContextQuery).is_err());
        assert!(runtime.force_final_answer());
    }

    #[test]
    fn search_budget_stops_after_five_chat_queries() {
        let runtime = Runtime::new().expect("tokio runtime should initialize");
        let db = runtime.block_on(init_test_db("s-budget"));
        let mut runtime = ToolRuntime::for_search(db, -1001374348669);

        for _ in 0..5 {
            assert!(runtime.begin_tool_call(ToolName::ChatContextQuery).is_ok());
        }
        assert!(runtime.begin_tool_call(ToolName::ChatContextQuery).is_err());
        assert!(runtime.force_final_answer());
    }

    /// Insert a message with an explicit user_id and username so analytics
    /// group_by=user queries return meaningful rows.
    async fn insert_user_message(
        db: &Database,
        message_id: i64,
        chat_id: i64,
        user_id: i64,
        username: &str,
    ) {
        let insert = crate::db::database::build_message_insert(
            Some(user_id),
            Some(username.to_string()),
            Some(format!("message from {username}")),
            Some("en".to_string()),
            Utc::now(),
            None,
            Some(chat_id),
            Some(message_id),
            None,
            false,
            None,
            false,
            false,
        );
        db.queue_message_insert(insert)
            .await
            .expect("message insert should queue");
        // Wait for the async write queue to flush the row.
        for _ in 0..200 {
            if let Ok(Some(rows)) = db.get_message_window(chat_id, message_id, 0, 0).await {
                if !rows.is_empty() {
                    return;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("message {message_id} was not persisted in time");
    }

    #[test]
    fn analytics_tool_ranks_users_and_accumulates_result() {
        let rt = Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let db = init_test_db("analytics-ranks").await;
            let chat_id = -1001374348669_i64;
            // Alice sends 2 messages, Bob sends 1.
            insert_user_message(&db, 1, chat_id, 11, "alice").await;
            insert_user_message(&db, 2, chat_id, 11, "alice").await;
            insert_user_message(&db, 3, chat_id, 12, "bob").await;

            let mut rt = ToolRuntime::for_analytics(db, chat_id);
            let args = serde_json::json!({"metric": "count", "group_by": "user"});
            let payload = rt
                .run_analytics_query(&args)
                .await
                .expect("analytics query should succeed");

            assert_eq!(
                payload.get("operation").and_then(Value::as_str),
                Some("analytics")
            );
            assert_eq!(payload["query"]["metric"], "count");
            assert_eq!(payload["query"]["group_by"], "user");
            assert_eq!(payload["query"]["limit"], 20);
            assert_eq!(payload["query"]["filters"]["exclude_commands"], true);
            assert_eq!(payload["coverage"]["chat"], "active");
            assert_eq!(payload["coverage"]["storage"], "stored_text_messages");
            assert_eq!(
                payload["coverage"]["anonymous_admin_and_channel_posts_excluded"],
                true
            );
            let rows = payload["rows"].as_array().expect("rows array");
            assert!(!rows.is_empty());
            // First row should be alice (2 messages, highest count).
            let first_val = rows[0]["value"].as_f64().unwrap_or(0.0);
            assert!(first_val >= 2.0, "alice should have at least 2 messages");

            // Authoritative result accumulated.
            assert_eq!(rt.analytics_results().len(), 1);
            // No message IDs accumulated (analytics doesn't retrieve chat messages).
            assert!(rt.accumulated_message_ids().is_empty());
        });
    }

    #[test]
    fn analytics_through_tool_is_chat_scoped() {
        let rt = Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let db = init_test_db("analytics-chat-scope").await;
            let chat_a = -1001374348669_i64;
            let chat_b = -1002631835259_i64;

            insert_user_message(&db, 1, chat_a, 11, "alice").await;
            insert_user_message(&db, 2, chat_a, 11, "alice").await;
            // Sentinel in chat B — must never show up in chat A results.
            insert_user_message(&db, 3, chat_b, 99, "sentinel").await;

            let mut runtime = ToolRuntime::for_analytics(db, chat_a);
            let args = serde_json::json!({"metric": "count"});
            let payload = runtime
                .run_analytics_query(&args)
                .await
                .expect("analytics query should succeed");

            let rows = payload["rows"].as_array().expect("rows array");
            // Total count for chat_a should be 2.
            let total = rows[0]["value"].as_f64().unwrap_or(0.0);
            assert_eq!(total, 2.0, "should count only chat A messages");
            // No sentinel group key.
            for row in rows {
                assert_ne!(
                    row["group"].as_str(),
                    Some("sentinel"),
                    "chat B user must not appear in chat A results"
                );
            }
        });
    }

    #[test]
    fn analytics_context_query_is_capped() {
        let rt = Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let db = init_test_db("analytics-ctx-cap").await;
            let chat_id = -1001374348669_i64;
            // Insert enough messages that a limit=20 request would normally return many.
            for i in 1..=15_i64 {
                insert_test_message(&db, i, chat_id, "hello world analytics").await;
            }

            let mut runtime = ToolRuntime::for_analytics(db, chat_id);
            // Simulate a model asking for limit=20 with context_after=5.
            let result = runtime
                .run_chat_context_query(ChatContextQueryArgs::Search {
                    query: "hello".to_string(),
                    limit: Some(20),
                    offset: None,
                    context_before: Some(0),
                    context_after: Some(5),
                })
                .await
                .expect("capped search should succeed");

            let result_count = result["result_count"].as_u64().unwrap_or(99);
            assert!(
                result_count <= 3,
                "analytics profile must cap results to ≤3, got {result_count}"
            );
            // No context messages should appear — context window is forced to 0.
            let results = result["results"].as_array().expect("results array");
            for hit in results {
                let ctx = hit["context_messages"]
                    .as_array()
                    .map(|a| a.len())
                    .unwrap_or(0);
                assert_eq!(ctx, 0, "analytics profile must suppress context window");
            }
        });
    }

    #[test]
    fn analytics_budget_stops() {
        let rt = Runtime::new().expect("tokio runtime");
        let db = rt.block_on(init_test_db("analytics-budget"));
        let mut runtime = ToolRuntime::for_analytics(db, -1001374348669);

        // Exhaust the analytics query budget.
        for _ in 0..runtime.budget.max_chat_analytics_query_calls {
            assert!(
                runtime.begin_tool_call(ToolName::ChatAnalytics).is_ok(),
                "should succeed within budget"
            );
            // begin_tool_call increments successful_calls; make room if total budget is smaller.
        }
        assert!(
            runtime.begin_tool_call(ToolName::ChatAnalytics).is_err(),
            "should fail once analytics budget exhausted"
        );
        assert!(runtime.force_final_answer());
        assert_eq!(
            runtime.chat_analytics_query_calls,
            runtime.budget.max_chat_analytics_query_calls
        );
        assert!(runtime.successful_calls < runtime.budget.max_total_successful_calls);
        assert!(runtime.force_final_answer());
    }

    #[test]
    fn invalid_spec_returns_invalid_arguments() {
        let rt = Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let db = init_test_db("analytics-invalid-spec").await;
            let mut runtime = ToolRuntime::for_analytics(db, -1001374348669);

            // distinct_count + group_by=user is semantically invalid (validate rejects it).
            let args = serde_json::json!({"metric": "distinct_count", "group_by": "user"});
            // execute_analytics wraps run_analytics_query errors as error payloads.
            let result_str = runtime.execute_analytics(&args).await;
            let result: Value = serde_json::from_str(&result_str).expect("valid json");
            assert_eq!(result["ok"].as_bool(), Some(false));
            assert_eq!(
                result["error_code"].as_str(),
                Some("invalid_arguments"),
                "validate rejection should produce invalid_arguments error code"
            );
        });
    }

    #[test]
    fn analytics_invalid_date_returns_invalid_arguments_without_accumulating_result() {
        let rt = Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let db = init_test_db("analytics-invalid-date").await;
            let mut runtime = ToolRuntime::for_analytics(db, -1001374348669);
            let args = serde_json::json!({
                "metric": "count",
                "filters": { "date_from": "last week" }
            });

            let result_str = runtime.execute_analytics(&args).await;
            let result: Value = serde_json::from_str(&result_str).expect("valid json");

            assert_eq!(result["ok"].as_bool(), Some(false));
            assert_eq!(result["error_code"].as_str(), Some("invalid_arguments"));
            assert!(runtime.analytics_results().is_empty());
        });
    }
}
