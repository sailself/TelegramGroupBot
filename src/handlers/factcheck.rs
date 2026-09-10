//! `/factcheck`: fact-check text, images, video, or audio content.

use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::{ChatAction, ParseMode, ReplyParameters};
use tracing::{error, info};

use crate::agents::factcheck::{run_factcheck_pipeline, FactcheckOutcome};
use crate::config::{CONFIG, FACTCHECK_SYSTEM_PROMPT, LANGUAGE_POLICY};
use crate::handlers::access::{check_access_control, is_rate_limited};
use crate::handlers::content::{
    extract_telegraph_urls_and_content, extract_twitter_urls_and_content,
};
use crate::handlers::media::{collect_message_media, MediaCollectionOptions};
use crate::handlers::responses::send_response;
use crate::llm::audit::create_command_audit_context;
use crate::llm::media::{summarize_media_files, MediaSummary};
use crate::llm::text_model::call_configured_text_model;
use crate::state::AppState;
use crate::tools::external_media::ExternalMediaBudget;
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
    let mut query_text = query.unwrap_or_default();
    let query_entities = message_entities_for_text(&message);
    let user_language_code = message
        .from
        .as_ref()
        .and_then(|user| user.language_code.as_deref());
    let mut telegraph_contents = Vec::new();
    let mut twitter_contents = Vec::new();

    let mut reply_text = String::new();
    if let Some(reply) = reply_message {
        reply_text = reply
            .text()
            .map(|value| value.to_string())
            .or_else(|| reply.caption().map(|value| value.to_string()))
            .unwrap_or_default();
        if !reply_text.trim().is_empty() {
            let reply_entities = message_entities_for_text(reply);
            let (reply_text_processed, reply_telegraph) =
                extract_telegraph_urls_and_content(&reply_text, reply_entities.as_deref(), 5).await;
            let (reply_text_processed, reply_twitter) = extract_twitter_urls_and_content(
                &reply_text_processed,
                reply_entities.as_deref(),
                5,
            )
            .await;
            telegraph_contents.extend(reply_telegraph);
            twitter_contents.extend(reply_twitter);
            reply_text = reply_text_processed;
        }
    }

    if !query_text.trim().is_empty() {
        let (query_text_processed, query_telegraph) =
            extract_telegraph_urls_and_content(&query_text, query_entities.as_deref(), 5).await;
        let (query_text_processed, query_twitter) =
            extract_twitter_urls_and_content(&query_text_processed, query_entities.as_deref(), 5)
                .await;
        telegraph_contents.extend(query_telegraph);
        twitter_contents.extend(query_twitter);
        query_text = query_text_processed;
    }

    let mut media_options = MediaCollectionOptions::for_commands();
    media_options.include_reply = true;
    let max_files = media_options.max_files;
    let collected_media = collect_message_media(&bot, &state, &message, media_options).await;
    let mut media_files = collected_media.files;

    let mut remaining = max_files.saturating_sub(media_files.len());
    let external_media_budget = ExternalMediaBudget::new(CONFIG.external_media_total_max_bytes);
    if remaining > 0 {
        let telegraph_files = crate::handlers::content::download_telegraph_media(
            &telegraph_contents,
            remaining,
            &external_media_budget,
        )
        .await;
        remaining = remaining.saturating_sub(telegraph_files.len());
        media_files.extend(telegraph_files);
    }

    if remaining > 0 {
        let twitter_files = crate::handlers::content::download_twitter_media(
            &twitter_contents,
            remaining,
            &external_media_budget,
        )
        .await;
        media_files.extend(twitter_files);
    }

    let media_summary = summarize_media_files(&media_files);
    let statement = build_factcheck_statement(&query_text, &reply_text, &media_summary);

    if statement.trim().is_empty() {
        bot.send_message(message.chat.id, "Please reply to a message to fact-check.")
            .reply_parameters(ReplyParameters::new(message.id))
            .await?;
        return Ok(());
    }
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

    if !telegraph_contents.is_empty() {
        let image_count: usize = telegraph_contents
            .iter()
            .map(|content| content.image_urls.len())
            .sum();
        let video_count: usize = telegraph_contents
            .iter()
            .map(|content| content.video_urls.len())
            .sum();
        let mut media_info = String::new();
        if image_count > 0 {
            media_info.push_str(&format!(" with {} image(s)", image_count));
        }
        if video_count > 0 {
            if media_info.is_empty() {
                media_info.push_str(&format!(" with {} video(s)", video_count));
            } else {
                media_info.push_str(&format!(" and {} video(s)", video_count));
            }
        }

        if processing_message_text == "Fact-checking message..." {
            processing_message_text = format!(
                "Extracting and fact-checking content from {} Telegraph page(s){}...",
                telegraph_contents.len(),
                media_info
            );
        } else {
            let base = processing_message_text.trim_end_matches("...");
            processing_message_text = format!(
                "{} and {} Telegraph page(s){}...",
                base,
                telegraph_contents.len(),
                media_info
            );
        }
    }

    if !twitter_contents.is_empty() {
        let image_count: usize = twitter_contents
            .iter()
            .map(|content| content.image_urls.len())
            .sum();
        let video_count: usize = twitter_contents
            .iter()
            .map(|content| content.video_urls.len())
            .sum();
        let mut media_info = String::new();
        if image_count > 0 {
            media_info.push_str(&format!(" with {} image(s)", image_count));
        }
        if video_count > 0 {
            if media_info.is_empty() {
                media_info.push_str(&format!(" with {} video(s)", video_count));
            } else {
                media_info.push_str(&format!(" and {} video(s)", video_count));
            }
        }

        if processing_message_text == "Fact-checking message..." {
            processing_message_text = format!(
                "Extracting and fact-checking content from {} Twitter post(s){}...",
                twitter_contents.len(),
                media_info
            );
        } else {
            let base = processing_message_text.trim_end_matches("...");
            processing_message_text = format!(
                "{} and {} Twitter post(s){}...",
                base,
                twitter_contents.len(),
                media_info
            );
        }
    }

    let processing_message = bot
        .send_message(message.chat.id, processing_message_text)
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
    let _chat_action =
        start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::Typing);

    if CONFIG.enable_agentic_factcheck {
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
