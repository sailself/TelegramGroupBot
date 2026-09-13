//! `/s` and `/qc` chat-search: model-driven message selection and result rendering.

use anyhow::Result;
use serde::Deserialize;
use serde_json::{json, Value};
use teloxide::prelude::*;
use teloxide::types::{MessageId, ParseMode};

use crate::config::CONFIG;
use crate::handlers::enrichment::Enrichment;
use crate::llm::audit::LlmAuditContext;
use crate::llm::tool_runtime::ToolRuntime;
use crate::state::{AppState, PendingQRequest, QaCommandMode};
use crate::utils::telegram::build_message_link;
use crate::utils::text::{escape_html, split_for_telegram, truncate_with_ellipsis};
use crate::utils::timing::{now_unix_seconds, CommandTimer};
use tracing::warn;

use super::model_resolution::{format_llm_error_message, ModelCatalogSnapshot, QaModel};
use super::process::{dispatch, qa_call_label, QaCall};

const CHAT_SEARCH_MESSAGE_LIMIT: usize = 3500;
const CHAT_SEARCH_JSON_OUTPUT_PROMPT: &str = "Final response format: return only valid JSON with this shape: {\"selected_message_ids\":[123],\"note\":\"optional short note\"}. Do not wrap the JSON in Markdown. Do not include message IDs that were not returned by chat_context_query.";

const CHAT_SEARCH_SYSTEM_PROMPT: &str = "You are helping search the current Telegram chat only. The search tool is keyword-based FTS retrieval, not semantic search. You must iteratively use chat_context_query to search this chat, inspect the returned messages, keep only clearly relevant messages, reformulate the query if needed, and continue until you have {result_target} relevant unique message IDs or you exhaust the 5 allowed chat_context_query calls. Never fabricate message IDs. Only choose message IDs that the tool actually returned. If fewer than {result_target} clearly relevant messages exist, return the best verified subset and explain that fewer relevant messages were found.";

pub(super) fn chat_search_rebuilding_message(command_name: &str) -> String {
    format!(
        "The chat search index is rebuilding right now. Please try /{} again in a few minutes.",
        command_name
    )
}

#[derive(Debug, Deserialize)]
pub(super) struct ChatSearchSelection {
    pub(super) selected_message_ids: Vec<i64>,
    pub(super) note: Option<String>,
}

struct ChatSearchModelResponse {
    text: String,
    model_used: String,
}

fn chat_search_response_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "selected_message_ids": {
                "type": "array",
                "items": { "type": "integer" },
                "description": "Up to 15 relevant unique message IDs returned by the search tool, ordered most relevant first."
            },
            "note": {
                "type": "string",
                "description": "Short explanation if fewer than 15 relevant messages were found."
            }
        },
        "required": ["selected_message_ids"],
        "additionalProperties": false
    })
}

pub(super) fn parse_chat_search_selection(text: &str) -> Option<ChatSearchSelection> {
    crate::agents::step::parse_lenient_json::<ChatSearchSelection>(text)
}

/// Collect message IDs referenced via `t.me/c/<chat>/<id>` links for the current
/// chat that were never returned by `chat_context_query` — a sign the model
/// fabricated the link. Pure helper so it can be unit-tested.
pub(super) fn unverified_chat_link_ids(answer: &str, chat_id: i64, valid_ids: &[i64]) -> Vec<i64> {
    // Only supergroup/channel ids (-100<internal>) produce citeable t.me/c/ links.
    let internal = match chat_id.to_string().strip_prefix("-100") {
        Some(rest) => rest.to_string(),
        None => return Vec::new(),
    };
    let needle = format!("t.me/c/{internal}/");
    let valid: std::collections::HashSet<i64> = valid_ids.iter().copied().collect();
    let mut unverified = Vec::new();
    let mut rest = answer;
    while let Some(pos) = rest.find(&needle) {
        let after = &rest[pos + needle.len()..];
        let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
        if let Ok(id) = digits.parse::<i64>() {
            if !valid.contains(&id) && !unverified.contains(&id) {
                unverified.push(id);
            }
        }
        // Advance past the run of ASCII digits — a valid char boundary, and 0 when
        // none follow. `after` already excludes this needle match, so the loop still
        // makes progress even with no digits, without slicing into a multibyte UTF-8
        // boundary or indexing past the end of the string.
        rest = &after[digits.len()..];
    }
    unverified
}

/// Warn (log only) if a /qc answer cites chat message links whose IDs were never
/// returned by `chat_context_query`. Never mutates the user-facing answer.
pub(super) fn warn_on_unverified_chat_links(
    answer: &str,
    chat_id: i64,
    valid_ids: &[i64],
    message_id: i64,
) {
    let unverified = unverified_chat_link_ids(answer, chat_id, valid_ids);
    if !unverified.is_empty() {
        warn!(
            "/qc answer cited unverified chat message links: chat_id={}, message_id={}, ids_not_returned_by_chat_context_query={:?}",
            chat_id, message_id, unverified
        );
    }
}

fn format_chat_search_results_html(
    query: &str,
    hits: &[crate::db::models::ChatSearchHit],
    note: Option<&str>,
    model_name: &str,
) -> String {
    let mut lines = vec![format!("<b>Chat search:</b> {}", escape_html(query.trim()))];

    if hits.is_empty() {
        lines.push("No clearly relevant messages were found.".to_string());
    } else {
        let label_map =
            crate::llm::prompting::build_display_label_map(hits.iter().filter_map(|h| {
                h.user_id
                    .map(|uid| (uid, h.username.as_deref().unwrap_or("Anonymous")))
            }));
        for (index, hit) in hits.iter().enumerate() {
            let raw_label = hit
                .user_id
                .and_then(|uid| label_map.get(&uid))
                .map(String::as_str)
                .unwrap_or_else(|| hit.username.as_deref().unwrap_or("Anonymous"));
            let username = escape_html(raw_label);
            let timestamp = escape_html(&hit.date.format("%Y-%m-%d %H:%M:%S UTC").to_string());
            let snippet = escape_html(&truncate_with_ellipsis(&hit.snippet, 120));
            let provenance_prefix = if hit.asks_ai {
                let command = hit.ai_command.as_deref().unwrap_or("q");
                format!("[AI ask /{}] ", escape_html(command))
            } else {
                String::new()
            };
            let link = hit
                .link
                .clone()
                .or_else(|| build_message_link(hit.chat_id, hit.message_id));
            let link_html = match link {
                Some(url) => format!("<a href=\"{}\">message link</a>", escape_html(&url)),
                None => "link unavailable".to_string(),
            };
            lines.push(format!(
                "{}. {} [{}]: {}{} {}",
                index + 1,
                username,
                timestamp,
                provenance_prefix,
                snippet,
                link_html
            ));
        }
    }

    if let Some(note) = note.filter(|value| !value.trim().is_empty()) {
        lines.push(format!("<i>{}</i>", escape_html(note.trim())));
    }
    lines.push(format!("<i>Model: {}</i>", escape_html(model_name)));
    lines.join("\n")
}

async fn send_chat_search_response(
    bot: &Bot,
    chat_id: ChatId,
    message_id: MessageId,
    response_html: &str,
) -> Result<()> {
    let chunks = split_for_telegram(response_html, CHAT_SEARCH_MESSAGE_LIMIT);
    let mut chunks_iter = chunks.into_iter();
    let first_chunk = chunks_iter.next().unwrap_or_default();

    bot.edit_message_text(chat_id, message_id, first_chunk)
        .parse_mode(ParseMode::Html)
        .await?;

    for chunk in chunks_iter {
        bot.send_message(chat_id, chunk)
            .parse_mode(ParseMode::Html)
            .await?;
    }

    Ok(())
}

async fn run_chat_search_model(
    state: &AppState,
    request: &PendingQRequest,
    query: &str,
    model: &QaModel,
    snapshot: &ModelCatalogSnapshot,
    audit_context: Option<&LlmAuditContext>,
) -> Result<(ChatSearchModelResponse, ToolRuntime)> {
    let mut runtime = ToolRuntime::for_search(state.db.clone(), request.chat_id);
    let chat_search_prompt = CHAT_SEARCH_SYSTEM_PROMPT.replace(
        "{result_target}",
        &CONFIG.limits.max_tool_context_items.to_string(),
    );
    // Gemini takes the tool budget in its prompt and a response schema;
    // third-party models are told the JSON shape in words instead.
    let system_prompt = match model {
        QaModel::Gemini => format!(
            "{}\n\n{}",
            chat_search_prompt,
            runtime.tool_limit_guidance()
        ),
        QaModel::ThirdParty { .. } => {
            format!(
                "{}\n\n{}",
                chat_search_prompt, CHAT_SEARCH_JSON_OUTPUT_PROMPT
            )
        }
    };
    let response_schema = matches!(model, QaModel::Gemini).then(chat_search_response_schema);

    let (text, model_used) = dispatch(
        model,
        QaCall {
            system_prompt,
            user_content: query.to_string(),
            label: qa_call_label(model, QaCommandMode::ChatSearch),
            media_files: None,
            youtube_urls: None,
            tools: Some(&mut runtime),
            reasoning_override: Some(CONFIG.agents.step_reasoning.clone()),
            response_schema,
            prompt_style: crate::llm::CodexPromptStyle::TaskSpecific,
            use_pro: false,
            search_grounding: false,
            audit_context,
        },
    )
    .await?;
    let response = ChatSearchModelResponse {
        text,
        model_used: model_used.unwrap_or_else(|| {
            model.result_display_name(snapshot, QaCommandMode::ChatSearch, None)
        }),
    };

    Ok((response, runtime))
}

pub(super) async fn process_chat_search_request(
    bot: &Bot,
    state: &AppState,
    request: &PendingQRequest,
    query: &str,
    model: &QaModel,
    snapshot: &ModelCatalogSnapshot,
    audit_context: Option<&LlmAuditContext>,
) -> Result<()> {
    let (response, runtime) =
        match run_chat_search_model(state, request, query, model, snapshot, audit_context).await {
            Ok(response) => response,
            Err(err) => {
                let display_model = model.display_name(snapshot, request.mode);
                let message = format_llm_error_message(model, &display_model, &err);
                bot.edit_message_text(
                    ChatId(request.chat_id),
                    MessageId(request.selection_message_id as i32),
                    message,
                )
                .await?;
                return Err(err);
            }
        };

    let max_selected_hits = CONFIG.limits.max_tool_context_items;
    let selection = parse_chat_search_selection(&response.text);
    let mut selected_hits = selection
        .as_ref()
        .map(|selection| {
            runtime.select_hits_by_message_ids(&selection.selected_message_ids, max_selected_hits)
        })
        .unwrap_or_default();
    if selected_hits.len() > max_selected_hits {
        selected_hits.truncate(max_selected_hits);
    }

    let note = selection
        .as_ref()
        .and_then(|value| value.note.as_deref().map(str::to_string))
        .or_else(|| {
            (selected_hits.len() < max_selected_hits).then(|| {
                format!(
                    "Fewer than {} clearly relevant messages were found within the 5 allowed search attempts.",
                    max_selected_hits
                )
            })
        });
    let response_html = format_chat_search_results_html(
        query,
        &selected_hits,
        note.as_deref(),
        &response.model_used,
    );

    send_chat_search_response(
        bot,
        ChatId(request.chat_id),
        MessageId(request.selection_message_id as i32),
        &response_html,
    )
    .await
}

pub(super) fn build_chat_search_pending_request(
    message: &Message,
    user_id: i64,
    query_text: &str,
    selection_message_id: i64,
    audit_context: Option<&LlmAuditContext>,
    command_timer: Option<CommandTimer>,
) -> PendingQRequest {
    PendingQRequest {
        user_id,
        query: query_text.to_string(),
        telegram_language_code: message
            .from
            .as_ref()
            .and_then(|user| user.language_code.as_deref())
            .map(str::to_string),
        enrichment: Enrichment::default(),
        chat_id: message.chat.id.0,
        message_id: message.id.0 as i64,
        selection_message_id,
        original_user_id: user_id,
        llm_invocation_id: audit_context.map(|context| context.invocation_id),
        timestamp: now_unix_seconds(),
        command_timer,
        mode: QaCommandMode::ChatSearch,
    }
}
