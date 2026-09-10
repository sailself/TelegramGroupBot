//! `/factcheck`: fact-check text, images, video, or audio content.

use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::{ChatAction, ParseMode, ReplyParameters};
use tracing::{error, info};

use crate::agents::factcheck::{run_factcheck_pipeline, FactcheckOutcome};
use crate::config::CONFIG;
use crate::handlers::access::{check_access_control, is_rate_limited};
use crate::handlers::enrichment::{
    count_sources, enrich_request, entity_link_urls, render_sources, Enrichment, EnrichmentBudget,
    SourceCounts, SourceKind,
};
use crate::handlers::media::{collect_message_media, MediaCollectionOptions};
use crate::handlers::responses::send_response;
use crate::llm::audit::create_command_audit_context;
use crate::llm::media::{summarize_media_files, MediaSummary};
use crate::llm::text_model::call_configured_text_model;
use crate::prompts::{FACTCHECK_SYSTEM_PROMPT, LANGUAGE_POLICY};
use crate::state::AppState;
use crate::utils::markdown::markdown_to_telegram_html;
use crate::utils::progress::ProgressReporter;
use crate::utils::telegram::{message_entities_for_text, start_chat_action_heartbeat};
use crate::utils::text::escape_html;

fn build_factcheck_system_prompt(telegram_user_language_hint: Option<&str>) -> String {
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
    // Substitute {language_policy} first: it carries the
    // {telegram_user_language_hint} placeholder resolved by the next call.
    FACTCHECK_SYSTEM_PROMPT
        .replace("{language_policy}", LANGUAGE_POLICY)
        .replace("{current_datetime}", &now)
        .replace(
            "{telegram_user_language_hint}",
            telegram_user_language_hint.unwrap_or("unknown"),
        )
}

fn build_factcheck_statement(
    query_text: &str,
    reply_text: &str,
    media_summary: &MediaSummary,
) -> String {
    let query_text = query_text.trim();
    let reply_text = reply_text.trim();

    // Break any injected closing tag in the untrusted content so a crafted
    // message can't escape the trust-boundary fences the factcheck prompt relies on.
    let neutralize = |value: &str| {
        let value = crate::utils::text::neutralize_closing_tag(value, "reply_context");
        crate::utils::text::neutralize_closing_tag(&value, "factcheck_target")
    };
    let reply_text = neutralize(reply_text);
    let query_text = neutralize(query_text);
    let (reply_text, query_text) = (reply_text.as_str(), query_text.as_str());

    if !query_text.is_empty() && !reply_text.is_empty() {
        return format!(
            "<reply_context>\n{}\n</reply_context>\n\n<factcheck_target>\n{}\n</factcheck_target>",
            reply_text, query_text
        );
    }

    if !query_text.is_empty() {
        return format!("<factcheck_target>\n{}\n</factcheck_target>", query_text);
    }

    if !reply_text.is_empty() {
        return format!("<factcheck_target>\n{}\n</factcheck_target>", reply_text);
    }

    if media_summary.videos > 0 {
        return "<auto_factcheck_target source=\"media_only\" kind=\"video\" />".to_string();
    }
    if media_summary.audios > 0 {
        return "<auto_factcheck_target source=\"media_only\" kind=\"audio\" />".to_string();
    }
    if media_summary.images > 0 {
        return "<auto_factcheck_target source=\"media_only\" kind=\"image\" />".to_string();
    }
    if media_summary.documents > 0 {
        return "<auto_factcheck_target source=\"media_only\" kind=\"document\" />".to_string();
    }

    String::new()
}

/// Name the link content in the progress message, the way the media counts are
/// named: either as the whole message or appended to what media already said.
fn append_source_progress(processing_message_text: &mut String, counts: SourceCounts, label: &str) {
    if counts.sources == 0 {
        return;
    }

    let mut media_info = String::new();
    if counts.images > 0 {
        media_info.push_str(&format!(" with {} image(s)", counts.images));
    }
    if counts.videos > 0 {
        if media_info.is_empty() {
            media_info.push_str(&format!(" with {} video(s)", counts.videos));
        } else {
            media_info.push_str(&format!(" and {} video(s)", counts.videos));
        }
    }

    *processing_message_text = if processing_message_text == "Fact-checking message..." {
        format!(
            "Extracting and fact-checking content from {} {}{}...",
            counts.sources, label, media_info
        )
    } else {
        format!(
            "{} and {} {}{}...",
            processing_message_text.trim_end_matches("..."),
            counts.sources,
            label,
            media_info
        )
    };
}

pub async fn factcheck_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    query: Option<String>,
) -> Result<()> {
    if !check_access_control(&bot, &message, "factcheck").await {
        return Ok(());
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        bot.send_message(
            message.chat.id,
            "Rate limit exceeded. Please try again later.",
        )
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
        return Ok(());
    }
    let _heavy_permit = state.acquire_heavy_command_permit().await;

    let reply_message = message.reply_to_message();
    let query_text = query.unwrap_or_default();
    let query_entities = message_entities_for_text(&message);
    let user_language_code = message
        .from
        .as_ref()
        .and_then(|user| user.language_code.as_deref());

    let mut reply_text = String::new();
    let mut reply_entities = None;
    if let Some(reply) = reply_message {
        reply_text = reply
            .text()
            .map(|value| value.to_string())
            .or_else(|| reply.caption().map(|value| value.to_string()))
            .unwrap_or_default();
        reply_entities = message_entities_for_text(reply);
    }

    let mut media_options = MediaCollectionOptions::for_commands();
    media_options.include_reply = true;
    let collected_media = collect_message_media(&bot, &state, &message, media_options).await;

    let budget = EnrichmentBudget::for_factcheck();
    let reply_entity_urls = entity_link_urls(reply_entities.as_deref());
    let query_entity_urls = entity_link_urls(query_entities.as_deref());
    let Enrichment {
        sources,
        media_files,
        ..
    } = enrich_request(
        &[
            &reply_text,
            &reply_entity_urls,
            &query_text,
            &query_entity_urls,
        ],
        collected_media.files,
        &budget,
    )
    .await;

    let media_summary = summarize_media_files(&media_files);
    let statement = build_factcheck_statement(&query_text, &reply_text, &media_summary);

    if statement.trim().is_empty() {
        bot.send_message(message.chat.id, "Please reply to a message to fact-check.")
            .reply_parameters(ReplyParameters::new(message.id))
            .await?;
        return Ok(());
    }
    // Fetched link content is quoted after the statement, fenced and budgeted,
    // so remote text can never be read as part of the claim under evaluation.
    let rendered_sources = render_sources(&sources, &budget);
    let statement = if rendered_sources.is_empty() {
        statement
    } else {
        format!("{statement}\n\n{rendered_sources}")
    };
    let audit_context = create_command_audit_context(&state, &message, "factcheck").await;

    let mut processing_message_text = if media_summary.videos > 0 {
        "Analyzing video and fact-checking content...".to_string()
    } else if media_summary.audios > 0 {
        "Analyzing audio and fact-checking content...".to_string()
    } else if media_summary.images > 0 {
        format!(
            "Analyzing {} image(s) and fact-checking content...",
            media_summary.images
        )
    } else if media_summary.documents > 0 {
        format!(
            "Analyzing {} document(s) and fact-checking content...",
            media_summary.documents
        )
    } else {
        "Fact-checking message...".to_string()
    };

    append_source_progress(
        &mut processing_message_text,
        count_sources(&sources, SourceKind::Telegraph),
        "Telegraph page(s)",
    );
    append_source_progress(
        &mut processing_message_text,
        count_sources(&sources, SourceKind::Twitter),
        "Twitter post(s)",
    );

    let processing_message = bot
        .send_message(message.chat.id, processing_message_text)
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
    let _chat_action =
        start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::Typing);

    if CONFIG.agents.enable_agentic_factcheck {
        let mut progress_reporter =
            ProgressReporter::new(bot.clone(), message.chat.id, processing_message.id);
        match run_factcheck_pipeline(
            &statement,
            &media_files,
            &media_summary,
            user_language_code,
            audit_context.as_ref(),
            &mut progress_reporter,
        )
        .await
        {
            Ok(FactcheckOutcome::Answer(crate::agents::common::ModelAnswer {
                text,
                model_display,
            })) => {
                let response_with_model = format!(
                    "{}\n\nModel: {}",
                    markdown_to_telegram_html(&text),
                    escape_html(&model_display)
                );
                send_response(
                    &bot,
                    processing_message.chat.id,
                    processing_message.id,
                    &response_with_model,
                    "Fact Check",
                    ParseMode::Html,
                )
                .await?;
                return Ok(());
            }
            Ok(FactcheckOutcome::UseLegacy(reason)) => {
                info!("Agentic fact-check fell back to the legacy path: {reason}");
            }
            Err(err) => {
                error!("Agentic fact-check failed: {}", err);
                bot.edit_message_text(
                    processing_message.chat.id,
                    processing_message.id,
                    format!("Failed to fact-check this message.\n\nError: {}", err),
                )
                .await?;
                return Ok(());
            }
        }
    }

    let system_prompt = build_factcheck_system_prompt(user_language_code);
    let response = match call_configured_text_model(
        &system_prompt,
        &statement,
        "Fact Check",
        true,
        media_summary.total > 0,
        Some(media_files),
        Some("FACTCHECK_SYSTEM_PROMPT"),
        audit_context.as_ref(),
    )
    .await
    {
        Ok(response) => response,
        Err(err) => {
            error!("Fact-check generation failed: {}", err);
            bot.edit_message_text(
                processing_message.chat.id,
                processing_message.id,
                format!("Failed to fact-check this message.\n\nError: {}", err),
            )
            .await?;
            return Ok(());
        }
    };

    let (response_text, response_model) = response;
    let response_with_model = format!(
        "{}\n\nModel: {}",
        markdown_to_telegram_html(&response_text),
        escape_html(&response_model)
    );

    send_response(
        &bot,
        processing_message.chat.id,
        processing_message.id,
        &response_with_model,
        "Fact Check",
        ParseMode::Html,
    )
    .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factcheck_prompt_renders_without_placeholders() {
        let rendered = build_factcheck_system_prompt(Some("ja"));
        assert!(
            !rendered.contains('{'),
            "unresolved placeholder in /factcheck prompt: {rendered}"
        );
        // Real output contracts survive the detox.
        assert!(rendered.contains("Partially True"));
        assert!(rendered.contains("Insufficient Evidence"));
        // Trust boundary + shared language policy are present.
        assert!(rendered.contains("untrusted material under evaluation"));
        assert!(rendered.contains("default to Chinese"));
        assert!(rendered.contains("ja"));
    }
}
