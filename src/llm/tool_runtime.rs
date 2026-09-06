use std::collections::{BTreeMap, BTreeSet};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::config::CONFIG;
use crate::db::database::Database;
use crate::db::models::{ChatSearchHit, MessageRow};
use crate::db::search::SEARCH_INDEX_REBUILDING_ERROR;
use crate::llm::tool_prompts::{fence_tool_result, TOOL_RESULT_GUIDANCE};
use crate::llm::web_search::{self, web_search_tool};
use crate::utils::telegram::build_message_link;

const DEFAULT_QC_SEARCH_LIMIT: usize = 8;
const DEFAULT_S_SEARCH_LIMIT: usize = 20;
const MAX_SEARCH_LIMIT: usize = 20;
const MAX_SEARCH_OFFSET: usize = 250;
const MAX_CONTEXT_WINDOW: usize = 5;
const MAX_WEB_RESULTS: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolProfile {
    QuickQuestion,
    ChatQuestion,
    ChatSearch,
    ChatAnalytics,
    /// Web search only, for answers that need no chat history (and no chat
    /// database): `/q` in standard mode, `/tldr`, `/factcheck`.
    WebSearch,
}

/// Every tool a model may call, in the order they are offered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ToolKind {
    WebSearch,
    ChatContextQuery,
    ChatAnalytics,
}

impl ToolKind {
    pub const ALL: [ToolKind; 3] = [
        ToolKind::WebSearch,
        ToolKind::ChatContextQuery,
        ToolKind::ChatAnalytics,
    ];

    /// Wire name the model uses to call the tool.
    pub fn name(self) -> &'static str {
        match self {
            ToolKind::WebSearch => "web_search",
            ToolKind::ChatContextQuery => "chat_context_query",
            ToolKind::ChatAnalytics => "chat_analytics",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }

    fn index(self) -> usize {
        self as usize
    }

    /// The single declaration of this tool. Provider adapters render it with
    /// [`ToolSpec::openai_function`], [`ToolSpec::responses_function`] or
    /// [`ToolSpec::gemini_declaration`], so a schema change lands everywhere.
    pub fn spec(self) -> ToolSpec {
        match self {
            ToolKind::WebSearch => ToolSpec {
                name: self.name(),
                description: "Search the web using the configured providers and return a concise Markdown summary.",
                parameters: json!({
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
                }),
            },
            ToolKind::ChatContextQuery => ToolSpec {
                name: self.name(),
                description: "Retrieve messages from the current Telegram chat only. This tool never accesses other chats.",
                parameters: json!({
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
                }),
            },
            ToolKind::ChatAnalytics => ToolSpec {
                name: self.name(),
                description: "Run a structured analytics query over this chat's message history. Returns counts, rankings, trends, and date metrics. Never accesses other chats.",
                parameters: crate::llm::analytics::query_spec_schema(),
            },
        }
    }
}

/// One tool declaration, independent of any provider's wire format.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    /// JSON Schema for the tool's arguments.
    pub parameters: Value,
}

impl ToolSpec {
    /// Chat Completions `tools[]` entry.
    pub fn openai_function(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }

    /// OpenAI Responses `tools[]` entry.
    pub fn responses_function(&self) -> Value {
        json!({
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": false,
        })
    }

    /// Gemini `functionDeclarations[]` entry. Gemini accepts the same JSON
    /// Schema subset (including numeric bounds), so the parameters are shared
    /// verbatim.
    pub fn gemini_declaration(&self) -> Value {
        json!({
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        })
    }
}

/// Per-request tool budget: a cap on successful calls overall plus a cap per
/// tool. A tool with a zero cap is not offered to the model at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolBudget {
    total: usize,
    per_tool: [usize; ToolKind::ALL.len()],
}

impl ToolBudget {
    pub fn new(total: usize) -> Self {
        Self {
            total,
            per_tool: [0; ToolKind::ALL.len()],
        }
    }

    pub fn with(mut self, kind: ToolKind, limit: usize) -> Self {
        self.per_tool[kind.index()] = limit;
        self
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn limit(&self, kind: ToolKind) -> usize {
        self.per_tool[kind.index()]
    }
}

#[derive(Debug, Clone, Copy)]
enum ToolBudgetErrorKind {
    Total,
    Exhausted(ToolKind),
    Disabled,
}

#[derive(Debug, Clone, Copy)]
struct ToolBudgetError {
    kind: ToolBudgetErrorKind,
}

#[derive(Clone)]
pub struct ToolRuntime {
    /// Chat database behind the chat tools; `None` for web-only profiles,
    /// whose zero chat budgets keep those tools from ever being charged.
    db: Option<Database>,
    chat_id: i64,
    profile: ToolProfile,
    budget: ToolBudget,
    // Whether a web-search provider is configured; decided once per request
    // so the offered tools, the guidance and the budget gate cannot disagree.
    web_search_available: bool,
    // The model searches natively (Codex): web_search is offered and described,
    // but no function declaration is emitted and a function call is refused.
    native_web_search: bool,
    successful_calls: usize,
    calls: [usize; ToolKind::ALL.len()],
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
    fn new(db: Option<Database>, chat_id: i64, profile: ToolProfile, budget: ToolBudget) -> Self {
        Self {
            db,
            chat_id,
            profile,
            budget,
            web_search_available: web_search::is_search_enabled(),
            native_web_search: false,
            successful_calls: 0,
            calls: [0; ToolKind::ALL.len()],
            force_final_answer: false,
            accumulated_hits: BTreeMap::new(),
            returned_message_ids: BTreeSet::new(),
            analytics_results: Vec::new(),
        }
    }

    pub fn for_quick(db: Database, chat_id: i64) -> Self {
        Self::new(
            Some(db),
            chat_id,
            ToolProfile::QuickQuestion,
            ToolBudget::new(1).with(ToolKind::WebSearch, 1),
        )
    }

    pub fn for_qc(db: Database, chat_id: i64) -> Self {
        Self::new(
            Some(db),
            chat_id,
            ToolProfile::ChatQuestion,
            ToolBudget::new(8)
                .with(ToolKind::WebSearch, 3)
                .with(ToolKind::ChatContextQuery, 5),
        )
    }

    pub fn for_search(db: Database, chat_id: i64) -> Self {
        Self::new(
            Some(db),
            chat_id,
            ToolProfile::ChatSearch,
            ToolBudget::new(5).with(ToolKind::ChatContextQuery, 5),
        )
    }

    pub fn for_analytics(db: Database, chat_id: i64) -> Self {
        Self::new(
            Some(db),
            chat_id,
            ToolProfile::ChatAnalytics,
            ToolBudget::new(CONFIG.qc_analytics_max_total_calls)
                .with(ToolKind::ChatContextQuery, 1)
                .with(ToolKind::ChatAnalytics, CONFIG.qc_analytics_max_query_calls),
        )
    }

    /// Web search only (up to three calls), no chat tools and no database.
    pub fn for_web_search() -> Self {
        Self::new(
            None,
            0,
            ToolProfile::WebSearch,
            ToolBudget::new(3).with(ToolKind::WebSearch, 3),
        )
    }

    fn chat_db(&self) -> Result<&Database> {
        self.db
            .as_ref()
            .ok_or_else(|| anyhow!("chat tools are unavailable for this request"))
    }

    /// The provider answers web searches itself (Codex's native tool), so
    /// `web_search` counts as offered for the guidance while the function
    /// declaration is left to the provider.
    pub fn use_native_web_search(&mut self) {
        self.web_search_available = true;
        self.native_web_search = true;
    }

    #[cfg(test)]
    pub(crate) fn with_web_search_available(mut self, available: bool) -> Self {
        self.web_search_available = available;
        self
    }

    pub fn analytics_results(&self) -> &[Value] {
        &self.analytics_results
    }

    pub fn force_final_answer(&self) -> bool {
        self.force_final_answer
    }

    pub fn max_total_successful_calls(&self) -> usize {
        self.budget.total()
    }

    /// Whether the model is offered `kind` on this request: it needs a
    /// non-zero budget, and web search additionally needs a configured
    /// provider.
    pub fn offers(&self, kind: ToolKind) -> bool {
        self.budget.limit(kind) > 0 && (kind != ToolKind::WebSearch || self.web_search_available)
    }

    fn offered_kinds(&self) -> impl Iterator<Item = ToolKind> + '_ {
        ToolKind::ALL
            .into_iter()
            .filter(move |kind| self.offers(*kind))
    }

    /// Offered tools that need a function declaration from us.
    fn declared_kinds(&self) -> impl Iterator<Item = ToolKind> + '_ {
        self.offered_kinds()
            .filter(move |kind| !(*kind == ToolKind::WebSearch && self.native_web_search))
    }

    /// Successful calls of `kind` so far.
    pub fn calls(&self, kind: ToolKind) -> usize {
        self.calls[kind.index()]
    }

    /// Whether Codex may answer with its native web search instead of the
    /// budgeted `web_search` function.
    pub fn allows_native_web_search(&self) -> bool {
        matches!(
            self.profile,
            ToolProfile::ChatQuestion | ToolProfile::WebSearch
        )
    }

    pub fn web_search_attempted(&self) -> bool {
        self.calls(ToolKind::WebSearch) > 0
    }

    /// Budget guidance for the system prompt, plus the clause that marks
    /// fenced tool results as untrusted data.
    pub fn tool_limit_guidance(&self) -> String {
        let budget = self.budget_guidance();
        format!("{budget}\n\n{TOOL_RESULT_GUIDANCE}")
    }

    /// Generated from the offered tools and their caps, so the numbers the
    /// model reads can never drift from the ones [`Self::begin_tool_call`]
    /// enforces.
    fn budget_guidance(&self) -> String {
        let offered = self
            .offered_kinds()
            .map(|kind| format!("{} at most {}", kind.name(), times(self.budget.limit(kind))))
            .collect::<Vec<_>>();
        let mut guidance = if offered.is_empty() {
            "No tools are available for this request; answer from the information you already have."
                .to_string()
        } else {
            format!(
                "Tool budgets for this request: use {} ({} in total). Once a budget is exhausted, answer with the evidence you already have.",
                join_naturally(&offered),
                plural(self.budget.total(), "tool call")
            )
        };
        if let Some(advice) = self.profile_advice() {
            guidance.push(' ');
            guidance.push_str(advice);
        }
        guidance
    }

    fn profile_advice(&self) -> Option<&'static str> {
        match self.profile {
            ToolProfile::QuickQuestion => self.offers(ToolKind::WebSearch).then_some(
                "After that one call succeeds or fails, answer immediately without any more tools and recommend /q if deeper verification is needed.",
            ),
            ToolProfile::ChatQuestion | ToolProfile::WebSearch => None,
            ToolProfile::ChatSearch => Some(
                "Search is keyword-based FTS, not semantic, so inspect snippets carefully and refine your query if needed.",
            ),
            ToolProfile::ChatAnalytics => Some(
                "Use chat_analytics for any counting/ranking/trend question and refine the spec between calls; keep chat_context_query for one small lookup to quote an example message.",
            ),
        }
    }

    /// Chat Completions `tools` array for the tools offered on this request.
    pub fn build_openai_function_tools(&self) -> Vec<Value> {
        self.declared_kinds()
            .map(|kind| kind.spec().openai_function())
            .collect()
    }

    /// OpenAI Responses `tools` array for the tools offered on this request.
    pub fn build_responses_tools(&self) -> Vec<Value> {
        self.declared_kinds()
            .map(|kind| kind.spec().responses_function())
            .collect()
    }

    /// Gemini `tools` array (one `functionDeclarations` group) for the tools
    /// offered on this request; empty when nothing is offered.
    pub fn build_gemini_tools(&self) -> Vec<Value> {
        let declarations = self
            .declared_kinds()
            .map(|kind| kind.spec().gemini_declaration())
            .collect::<Vec<_>>();
        if declarations.is_empty() {
            Vec::new()
        } else {
            vec![json!({ "functionDeclarations": declarations })]
        }
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
        self.begin_tool_call(ToolKind::ChatContextQuery)
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
        self.begin_tool_call(ToolKind::WebSearch)
            .map_err(|err| anyhow!(tool_budget_error_parts(err).1))?;
        web_search_tool(query, Some(max_results.clamp(1, MAX_WEB_RESULTS))).await
    }

    /// Run a model-requested tool and return its payload fenced as untrusted
    /// data (see [`fence_tool_result`]).
    pub async fn execute_tool(&mut self, name: &str, arguments: &Value) -> String {
        let payload = self.execute_tool_unfenced(name, arguments).await;
        fence_tool_result(name, &payload)
    }

    async fn execute_tool_unfenced(&mut self, name: &str, arguments: &Value) -> String {
        let Some(kind) = ToolKind::from_name(name) else {
            self.force_final_answer = true;
            return self.error_payload(
                name,
                "unsupported_tool",
                "Unsupported tool call requested by the model.",
            );
        };
        if let Err(err) = self.begin_tool_call(kind) {
            return self.tool_budget_error_payload(name, err);
        }
        match kind {
            ToolKind::WebSearch => self.execute_web_search(arguments).await,
            ToolKind::ChatContextQuery => self.execute_chat_context_query(arguments).await,
            ToolKind::ChatAnalytics => self.execute_analytics(arguments).await,
        }
    }

    /// Charge one call of `kind` against the budget. Any refusal also forces
    /// the final answer so the loop stops offering tools.
    fn begin_tool_call(&mut self, kind: ToolKind) -> std::result::Result<(), ToolBudgetError> {
        if self.force_final_answer {
            return Err(ToolBudgetError {
                kind: ToolBudgetErrorKind::Disabled,
            });
        }
        if self.successful_calls >= self.budget.total() {
            self.force_final_answer = true;
            return Err(ToolBudgetError {
                kind: ToolBudgetErrorKind::Total,
            });
        }
        if !self.offers(kind) || (kind == ToolKind::WebSearch && self.native_web_search) {
            self.force_final_answer = true;
            return Err(ToolBudgetError {
                kind: ToolBudgetErrorKind::Disabled,
            });
        }
        if self.calls(kind) >= self.budget.limit(kind) {
            self.force_final_answer = true;
            return Err(ToolBudgetError {
                kind: ToolBudgetErrorKind::Exhausted(kind),
            });
        }

        self.calls[kind.index()] += 1;
        self.successful_calls += 1;
        Ok(())
    }

    /// `{"total_remaining": n, "<tool>_remaining": n, ...}` attached to every
    /// tool payload so the model can plan its remaining calls.
    fn remaining_budget_snapshot(&self) -> Value {
        let mut snapshot = serde_json::Map::new();
        snapshot.insert(
            "total_remaining".to_string(),
            json!(self.budget.total().saturating_sub(self.successful_calls)),
        );
        for kind in ToolKind::ALL {
            snapshot.insert(
                format!("{}_remaining", kind.name()),
                json!(self.budget.limit(kind).saturating_sub(self.calls(kind))),
            );
        }
        Value::Object(snapshot)
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
                    ToolProfile::QuickQuestion
                    | ToolProfile::ChatQuestion
                    | ToolProfile::WebSearch => DEFAULT_QC_SEARCH_LIMIT,
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
                    .chat_db()?
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
                            self.chat_db()?
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
                let mut context_before = context_before.unwrap_or(2).clamp(0, MAX_CONTEXT_WINDOW);
                let mut context_after = context_after.unwrap_or(2).clamp(0, MAX_CONTEXT_WINDOW);
                if self.profile == ToolProfile::ChatAnalytics {
                    context_before = 0;
                    context_after = 0;
                }
                let Some(messages) = self
                    .chat_db()?
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
            .chat_db()?
            .run_chat_analytics(self.chat_id, &spec)
            .await
            .map_err(|error| {
                let message = error.to_string();
                if message.contains("date_")
                    || message.contains("term must")
                    || message.contains("must be at most")
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
        ToolBudgetErrorKind::Exhausted(ToolKind::WebSearch) => (
            "web_search_budget_exhausted",
            "The web_search budget for this request is exhausted. Answer using the evidence already gathered.",
        ),
        ToolBudgetErrorKind::Exhausted(ToolKind::ChatContextQuery) => (
            "chat_context_query_budget_exhausted",
            "The chat_context_query budget for this request is exhausted. Answer using the evidence already gathered.",
        ),
        ToolBudgetErrorKind::Exhausted(ToolKind::ChatAnalytics) => (
            "chat_analytics_budget_exhausted",
            "The chat_analytics budget for this request is exhausted. Answer using the results already gathered.",
        ),
        ToolBudgetErrorKind::Disabled => (
            "tool_disabled",
            "This tool is unavailable for the current request. Answer using the evidence already gathered.",
        ),
    }
}

fn times(count: usize) -> String {
    match count {
        1 => "once".to_string(),
        2 => "twice".to_string(),
        n => format!("{n} times"),
    }
}

fn plural(count: usize, noun: &str) -> String {
    if count == 1 {
        format!("{count} {noun}")
    } else {
        format!("{count} {noun}s")
    }
}

/// `a`, `a and b`, `a, b and c`.
fn join_naturally(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [only] => only.clone(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
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

/// Test-only helpers shared with the tool-loop tests.
#[cfg(test)]
pub(crate) mod test_support {
    use crate::db::database::Database;
    use chrono::Utc;

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

    /// A fresh on-disk SQLite database for one test.
    pub(crate) async fn init_test_db(test_name: &str) -> Database {
        let path = test_db_path(test_name);
        Database::init(&sqlite_url_for_path(&path))
            .await
            .expect("test database should initialize")
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::init_test_db;
    use super::*;
    use crate::db::database::Database;
    use chrono::Utc;
    use tokio::runtime::Runtime;

    #[tokio::test]
    async fn model_driven_tool_results_are_fenced_as_untrusted_data() {
        let db = init_test_db("fenced-tool-results").await;
        let mut runtime = ToolRuntime::for_quick(db, -100);

        let payload = runtime
            .execute_tool("no_such_tool", &serde_json::json!({}))
            .await;

        assert!(
            payload.starts_with(r#"<tool_result tool="no_such_tool">"#),
            "{payload}"
        );
        assert!(payload.trim_end().ends_with("</tool_result>"), "{payload}");
        assert!(payload.contains("unsupported_tool"), "{payload}");
    }

    #[tokio::test]
    async fn every_profile_guidance_declares_tool_results_untrusted() {
        let db = init_test_db("guidance-untrusted").await;
        for runtime in [
            ToolRuntime::for_quick(db.clone(), -100),
            ToolRuntime::for_qc(db.clone(), -100),
            ToolRuntime::for_search(db.clone(), -100),
            ToolRuntime::for_analytics(db.clone(), -100),
            ToolRuntime::for_web_search(),
        ] {
            let guidance = runtime.tool_limit_guidance();
            assert!(
                guidance.contains(crate::llm::tool_prompts::TOOL_RESULT_GUIDANCE),
                "{guidance}"
            );
        }
    }

    #[test]
    fn every_tool_kind_renders_one_spec_into_all_provider_formats() {
        for kind in ToolKind::ALL {
            let spec = kind.spec();
            assert_eq!(spec.name, kind.name());
            assert_eq!(ToolKind::from_name(spec.name), Some(kind));
            assert!(!spec.description.is_empty());
            assert_eq!(spec.parameters["type"], "object");

            let openai = spec.openai_function();
            assert_eq!(openai["type"], "function");
            assert_eq!(openai["function"]["name"], spec.name);
            assert_eq!(openai["function"]["description"], spec.description);
            assert_eq!(openai["function"]["parameters"], spec.parameters);

            let responses = spec.responses_function();
            assert_eq!(responses["type"], "function");
            assert_eq!(responses["name"], spec.name);
            assert_eq!(responses["description"], spec.description);
            assert_eq!(responses["parameters"], spec.parameters);
            assert_eq!(responses["strict"], false);

            let gemini = spec.gemini_declaration();
            assert_eq!(gemini["name"], spec.name);
            assert_eq!(gemini["description"], spec.description);
            assert_eq!(gemini["parameters"], spec.parameters);
        }
        assert_eq!(ToolKind::from_name("no_such_tool"), None);
    }

    fn openai_tool_names(runtime: &ToolRuntime) -> Vec<String> {
        runtime
            .build_openai_function_tools()
            .iter()
            .filter_map(|tool| tool.pointer("/function/name").and_then(Value::as_str))
            .map(str::to_string)
            .collect()
    }

    fn responses_tool_names(runtime: &ToolRuntime) -> Vec<String> {
        runtime
            .build_responses_tools()
            .iter()
            .filter_map(|tool| tool.get("name").and_then(Value::as_str))
            .map(str::to_string)
            .collect()
    }

    fn gemini_tool_names(runtime: &ToolRuntime) -> Vec<String> {
        runtime
            .build_gemini_tools()
            .iter()
            .flat_map(|tool| {
                tool.get("functionDeclarations")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
            })
            .filter_map(|tool| tool.get("name").and_then(Value::as_str))
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn offered_tools_follow_the_budget_and_web_search_availability() {
        let rt = Runtime::new().expect("tokio runtime should initialize");
        let db = rt.block_on(init_test_db("offered-tools"));

        let qc_without_web = ToolRuntime::for_qc(db.clone(), -100).with_web_search_available(false);
        assert_eq!(openai_tool_names(&qc_without_web), ["chat_context_query"]);
        assert!(!qc_without_web.offers(ToolKind::WebSearch));
        assert!(qc_without_web.offers(ToolKind::ChatContextQuery));

        let qc_with_web = ToolRuntime::for_qc(db.clone(), -100).with_web_search_available(true);
        assert_eq!(
            openai_tool_names(&qc_with_web),
            ["web_search", "chat_context_query"]
        );
        assert_eq!(
            responses_tool_names(&qc_with_web),
            openai_tool_names(&qc_with_web)
        );
        assert_eq!(
            gemini_tool_names(&qc_with_web),
            openai_tool_names(&qc_with_web)
        );

        // A zero budget hides a tool even when nothing else disables it.
        let search = ToolRuntime::for_search(db.clone(), -100).with_web_search_available(true);
        assert_eq!(openai_tool_names(&search), ["chat_context_query"]);
        assert!(!search.offers(ToolKind::ChatAnalytics));

        let analytics = ToolRuntime::for_analytics(db, -100).with_web_search_available(true);
        assert_eq!(
            openai_tool_names(&analytics),
            ["chat_context_query", "chat_analytics"]
        );
        assert!(analytics.build_gemini_tools()[0]["functionDeclarations"].is_array());
    }

    #[tokio::test]
    async fn web_search_profile_needs_no_database_and_offers_only_web_search() {
        let mut runtime = ToolRuntime::for_web_search().with_web_search_available(true);

        assert_eq!(runtime.budget.total(), 3);
        assert_eq!(runtime.budget.limit(ToolKind::WebSearch), 3);
        assert_eq!(openai_tool_names(&runtime), ["web_search"]);
        assert!(
            runtime.allows_native_web_search(),
            "Codex may use its native web search instead of the function tool"
        );
        let guidance = runtime.budget_guidance();
        assert!(
            guidance.contains("web_search at most 3 times"),
            "{guidance}"
        );

        // Chat tools are refused by the budget gate, so the missing database is never touched.
        let err = runtime
            .run_search_query("anything", None, 0, 0)
            .await
            .expect_err("chat tools are unavailable without a chat database");
        assert!(err.to_string().contains("unavailable"), "{err}");
        assert!(runtime.force_final_answer());

        let offline = ToolRuntime::for_web_search().with_web_search_available(false);
        assert!(offline.build_openai_function_tools().is_empty());
        assert!(offline.build_responses_tools().is_empty());
        assert!(offline.build_gemini_tools().is_empty());
    }

    #[tokio::test]
    async fn native_web_search_is_described_in_the_guidance_but_never_declared_as_a_function() {
        let mut runtime = ToolRuntime::for_web_search().with_web_search_available(false);
        assert!(runtime.budget_guidance().contains("No tools are available"));

        runtime.use_native_web_search();

        assert!(runtime.offers(ToolKind::WebSearch));
        let guidance = runtime.budget_guidance();
        assert!(
            guidance.contains("web_search at most 3 times"),
            "{guidance}"
        );
        assert!(
            runtime.build_responses_tools().is_empty(),
            "the provider attaches its own native tool"
        );
        assert!(runtime.build_openai_function_tools().is_empty());
        assert!(runtime.build_gemini_tools().is_empty());

        // A function call for it can only be a hallucination.
        let payload = runtime
            .execute_tool_unfenced("web_search", &json!({"query": "x"}))
            .await;
        let payload: Value = serde_json::from_str(&payload).expect("JSON");
        assert_eq!(payload["error_code"], "tool_disabled");
    }

    #[test]
    fn budget_guidance_is_generated_from_the_budget() {
        let rt = Runtime::new().expect("tokio runtime should initialize");
        let db = rt.block_on(init_test_db("budget-guidance"));

        let qc = ToolRuntime::for_qc(db.clone(), -100)
            .with_web_search_available(true)
            .budget_guidance();
        assert!(qc.contains("web_search at most 3 times"), "{qc}");
        assert!(qc.contains("chat_context_query at most 5 times"), "{qc}");
        assert!(qc.contains("8 tool calls in total"), "{qc}");

        let quick = ToolRuntime::for_quick(db.clone(), -100)
            .with_web_search_available(true)
            .budget_guidance();
        assert!(quick.contains("web_search at most once"), "{quick}");
        assert!(quick.contains("recommend /q"), "{quick}");
        assert!(!quick.contains("chat_context_query"), "{quick}");

        let quick_offline = ToolRuntime::for_quick(db.clone(), -100)
            .with_web_search_available(false)
            .budget_guidance();
        assert!(
            quick_offline.contains("No tools are available"),
            "{quick_offline}"
        );

        let search = ToolRuntime::for_search(db.clone(), -100)
            .with_web_search_available(true)
            .budget_guidance();
        assert!(
            search.contains("chat_context_query at most 5 times"),
            "{search}"
        );
        assert!(!search.contains("web_search"), "{search}");
        assert!(search.contains("keyword-based"), "{search}");

        let analytics = ToolRuntime::for_analytics(db, -100).budget_guidance();
        assert!(
            analytics.contains(&format!(
                "chat_analytics at most {} times",
                CONFIG.qc_analytics_max_query_calls
            )) || analytics.contains("chat_analytics at most once"),
            "{analytics}"
        );
        assert!(
            analytics.contains("chat_context_query at most once"),
            "{analytics}"
        );
    }

    #[test]
    fn budget_snapshot_reports_remaining_calls_per_tool() {
        let rt = Runtime::new().expect("tokio runtime should initialize");
        let db = rt.block_on(init_test_db("budget-snapshot"));
        let mut runtime = ToolRuntime::for_qc(db, -100).with_web_search_available(true);
        assert!(runtime.begin_tool_call(ToolKind::WebSearch).is_ok());

        let snapshot = runtime.remaining_budget_snapshot();
        assert_eq!(snapshot["total_remaining"], 7);
        assert_eq!(snapshot["web_search_remaining"], 2);
        assert_eq!(snapshot["chat_context_query_remaining"], 5);
        assert_eq!(snapshot["chat_analytics_remaining"], 0);
        assert_eq!(runtime.calls(ToolKind::WebSearch), 1);
        assert_eq!(runtime.calls(ToolKind::ChatContextQuery), 0);
    }

    #[test]
    fn exhausting_one_tool_reports_that_tool_and_forces_the_final_answer() {
        let rt = Runtime::new().expect("tokio runtime should initialize");
        let db = rt.block_on(init_test_db("budget-exhausted-kind"));
        let mut runtime = ToolRuntime::for_qc(db, -100).with_web_search_available(true);
        for _ in 0..runtime.budget.limit(ToolKind::WebSearch) {
            assert!(runtime.begin_tool_call(ToolKind::WebSearch).is_ok());
        }

        let payload =
            rt.block_on(runtime.execute_tool_unfenced("web_search", &json!({"query": "one more"})));
        let payload: Value = serde_json::from_str(&payload).expect("budget response is JSON");
        assert_eq!(payload["error_code"], "web_search_budget_exhausted");
        assert!(runtime.force_final_answer());
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
            assert_eq!(tool_runtime.calls(ToolKind::ChatContextQuery), 1);
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
            assert_eq!(tool_runtime.calls(ToolKind::WebSearch), 0);
        });
    }

    #[test]
    fn programmatic_search_errors_once_budget_is_exhausted() {
        let runtime = Runtime::new().expect("tokio runtime should initialize");
        runtime.block_on(async {
            let db = init_test_db("programmatic-budget-exhausted").await;
            let mut tool_runtime = ToolRuntime::for_qc(db, -1001374348669);
            tool_runtime.successful_calls = tool_runtime.budget.total();

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

        runtime.successful_calls = runtime.budget.total();
        assert!(runtime.begin_tool_call(ToolKind::ChatContextQuery).is_err());
        assert!(runtime.force_final_answer());
    }

    #[test]
    fn search_budget_stops_after_five_chat_queries() {
        let runtime = Runtime::new().expect("tokio runtime should initialize");
        let db = runtime.block_on(init_test_db("s-budget"));
        let mut runtime = ToolRuntime::for_search(db, -1001374348669);

        for _ in 0..5 {
            assert!(runtime.begin_tool_call(ToolKind::ChatContextQuery).is_ok());
        }
        assert!(runtime.begin_tool_call(ToolKind::ChatContextQuery).is_err());
        assert!(runtime.force_final_answer());
    }

    #[test]
    fn quick_profile_exposes_only_one_optional_web_search_round() {
        let runtime = Runtime::new().expect("tokio runtime should initialize");
        let db = runtime.block_on(init_test_db("quick-profile"));
        let runtime =
            ToolRuntime::for_quick(db.clone(), -1001374348669).with_web_search_available(true);

        assert_eq!(runtime.profile, ToolProfile::QuickQuestion);
        assert_eq!(runtime.budget.total(), 1);
        assert_eq!(runtime.budget.limit(ToolKind::WebSearch), 1);
        assert_eq!(runtime.budget.limit(ToolKind::ChatContextQuery), 0);
        assert_eq!(runtime.budget.limit(ToolKind::ChatAnalytics), 0);
        assert!(runtime.offers(ToolKind::WebSearch));
        assert!(!runtime.allows_native_web_search());
        assert!(ToolRuntime::for_qc(db, -1001374348669).allows_native_web_search());

        let openai_tools = runtime.build_openai_function_tools();
        let names = openai_tools
            .iter()
            .filter_map(|tool| tool.pointer("/function/name").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert_eq!(names, ["web_search"]);

        let gemini_tools = runtime.build_gemini_tools();
        let gemini_names = gemini_tools
            .iter()
            .flat_map(|tool| {
                tool.get("functionDeclarations")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
            })
            .filter_map(|tool| tool.get("name").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert!(!gemini_names.contains(&"chat_context_query"));
        assert!(!gemini_names.contains(&"chat_analytics"));
    }

    #[test]
    fn quick_failed_or_parallel_extra_searches_exhaust_the_single_round() {
        let tokio_runtime = Runtime::new().expect("tokio runtime should initialize");
        let db = tokio_runtime.block_on(init_test_db("quick-budget"));
        let mut tool_runtime =
            ToolRuntime::for_quick(db, -1001374348669).with_web_search_available(true);

        let failed = tokio_runtime
            .block_on(tool_runtime.execute_tool_unfenced("web_search", &json!({"query": ""})));
        let failed: Value =
            serde_json::from_str(&failed).expect("failed tool response should be JSON");
        assert_eq!(failed["ok"], false);
        assert_eq!(failed["error_code"], "invalid_arguments");
        assert!(tool_runtime.web_search_attempted());

        let extra = tokio_runtime.block_on(
            tool_runtime
                .execute_tool_unfenced("web_search", &json!({"query": "parallel extra search"})),
        );
        let extra: Value = serde_json::from_str(&extra).expect("budget response should be JSON");
        assert_eq!(extra["ok"], false);
        assert_eq!(extra["error_code"], "total_budget_exhausted");
        assert!(tool_runtime.force_final_answer());
    }

    #[test]
    fn quick_profile_rejects_chat_tools_even_when_called_directly() {
        let tokio_runtime = Runtime::new().expect("tokio runtime should initialize");
        let db = tokio_runtime.block_on(init_test_db("quick-chat-tools"));
        let mut tool_runtime = ToolRuntime::for_quick(db, -1001374348669);

        let result = tokio_runtime.block_on(tool_runtime.execute_tool_unfenced(
            "chat_context_query",
            &json!({"operation": "search", "query": "secret history"}),
        ));
        let result: Value = serde_json::from_str(&result).expect("tool response should be JSON");
        assert_eq!(result["ok"], false);
        assert_eq!(result["error_code"], "tool_disabled");
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
    fn analytics_window_query_suppresses_requested_context() {
        let rt = Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let db = init_test_db("analytics-window-context-cap").await;
            let chat_id = -1001374348669_i64;
            for (id, text) in [(10_i64, "before"), (11, "anchor"), (12, "after")] {
                insert_test_message(&db, id, chat_id, text).await;
            }

            let mut runtime = ToolRuntime::for_analytics(db, chat_id);
            let result = runtime
                .run_chat_context_query(ChatContextQueryArgs::Window {
                    message_id: 11,
                    context_before: Some(5),
                    context_after: Some(5),
                })
                .await
                .expect("analytics window query should succeed");

            assert_eq!(result["result_count"], 1);
            assert_eq!(result["messages"][0]["message_id"], 11);
        });
    }

    #[test]
    fn analytics_budget_stops() {
        let rt = Runtime::new().expect("tokio runtime");
        let db = rt.block_on(init_test_db("analytics-budget"));
        let mut runtime = ToolRuntime::for_analytics(db, -1001374348669);

        // Exhaust the analytics query budget.
        for _ in 0..runtime.budget.limit(ToolKind::ChatAnalytics) {
            assert!(
                runtime.begin_tool_call(ToolKind::ChatAnalytics).is_ok(),
                "should succeed within budget"
            );
        }
        assert!(
            runtime.begin_tool_call(ToolKind::ChatAnalytics).is_err(),
            "should fail once analytics budget exhausted"
        );
        assert!(runtime.force_final_answer());
        assert_eq!(
            runtime.calls(ToolKind::ChatAnalytics),
            runtime.budget.limit(ToolKind::ChatAnalytics)
        );
        assert!(runtime.successful_calls < runtime.budget.total());
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

    #[test]
    fn analytics_oversized_filter_returns_invalid_arguments_without_accumulating_result() {
        let rt = Runtime::new().expect("tokio runtime");
        rt.block_on(async {
            let db = init_test_db("analytics-oversized-filter").await;
            let mut runtime = ToolRuntime::for_analytics(db, -1001374348669);
            let args = serde_json::json!({
                "metric": "count",
                "filters": { "text_contains": "x".repeat(257) }
            });

            let result_str = runtime.execute_analytics(&args).await;
            let result: Value = serde_json::from_str(&result_str).expect("valid json");

            assert_eq!(result["ok"].as_bool(), Some(false));
            assert_eq!(result["error_code"].as_str(), Some("invalid_arguments"));
            assert!(runtime.analytics_results().is_empty());
        });
    }
}
