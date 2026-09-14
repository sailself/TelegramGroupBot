//! `/tldr`: summarize recent chat messages, with an optional infographic.

use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::{ChatAction, ReplyParameters};
use tracing::{error, info, warn};

use crate::agents::step::WallClock;
use crate::config::CONFIG;
use crate::handlers::access::{check_access_control, is_rate_limited};
use crate::handlers::content::create_telegraph_answer;
use crate::handlers::image::generate_image_with_configured_default;
use crate::handlers::responses::{send_response, ResponseContent};
use crate::llm::audit::create_command_audit_context;
use crate::llm::media::detect_mime_type;
use crate::llm::resolved_model::{ModelCatalogSnapshot, ResolvedTextModel};
use crate::llm::text_model::{call_resolved_text_model, ModelRequestCapabilities};
use crate::llm::{GeminiImageConfig, LlmAuditContext};
use crate::prompts::TLDR_SYSTEM_PROMPT;
use crate::state::AppState;
use crate::tools::cwd_uploader::upload_image_bytes_to_cwd;
use crate::utils::progress::ProgressReporter;
use crate::utils::telegram::start_chat_action_heartbeat;
use crate::utils::timing::{complete_command_timer, start_command_timer};

/// Legacy single-call /tldr: the whole history in one prompt. Used below the
/// map-reduce threshold and as the fallback when the pipeline cannot start.
async fn tldr_single_call(
    model: &ResolvedTextModel,
    messages: &[crate::db::models::MessageRow],
    audit_context: Option<&LlmAuditContext>,
) -> Result<(String, String)> {
    let chat_content = crate::llm::prompting::wrap_chat_history(
        &crate::llm::prompting::format_tldr_chat_content(messages),
    );
    let system_prompt = TLDR_SYSTEM_PROMPT.replace("{bot_name}", &CONFIG.telegraph.author_name);
    call_resolved_text_model(
        model,
        crate::llm::text_model::TextCallRequest {
            system_prompt: &system_prompt,
            user_content: &chat_content,
            response_title: "Message Summary",
            tools_enabled: true,
            use_pro: true,
            media_files: None,
            prompt_name: Some("TLDR_SYSTEM_PROMPT"),
            audit_context,
        },
    )
    .await
}

const TLDR_DEFAULT_MESSAGE_COUNT: i64 = 100;

/// Number of messages `/tldr <n>` should summarize, clamped to a sane range so
/// a negative or huge argument cannot turn into an unbounded fetch.
fn resolve_tldr_count(arg: Option<&str>, max_messages: usize) -> i64 {
    let max = i64::try_from(max_messages).unwrap_or(i64::MAX).max(1);
    arg.and_then(|value| value.trim().parse::<i64>().ok())
        .unwrap_or(TLDR_DEFAULT_MESSAGE_COUNT)
        .clamp(1, max)
}

pub async fn tldr_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    count: Option<String>,
) -> Result<()> {
    if !check_access_control(&bot, &message, "tldr").await {
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

    let mut timer = start_command_timer("tldr", &message);
    let processing_message = bot
        .send_message(message.chat.id, "Summarizing recent messages...")
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
    crate::utils::telegram::run_with_status_message(&bot, processing_message.chat.id, processing_message.id, "Failed to generate a summary. Please try again.", async {
            let _heavy_permit = _heavy_permit;

    let _chat_action =
        start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::Typing);

    let mut messages = if let Some(reply) = message.reply_to_message() {
        // Fetch one past the cap so the truncation notice below still fires
        // without pulling the whole chat into memory.
        let fetch_limit = (CONFIG.agents.tldr_max_messages + 1) as i64;
        state
            .db
            .select_messages_from_id(message.chat.id.0, reply.id.0 as i64, fetch_limit)
            .await?
    } else {
        let n = resolve_tldr_count(count.as_deref(), CONFIG.agents.tldr_max_messages);
        state.db.select_messages(message.chat.id.0, n).await?
    };

    if messages.is_empty() {
        bot.edit_message_text(
            message.chat.id,
            processing_message.id,
            "No messages found to summarize.",
        )
        .await?;
        complete_command_timer(&mut timer, "error", Some("no_messages".to_string()));
        return Ok(());
    }

    // The bounded fetch includes one extra row to detect truncation.
    let truncated_to_cap = messages.len() > CONFIG.agents.tldr_max_messages;
    if truncated_to_cap {
        let skip = messages.len() - CONFIG.agents.tldr_max_messages;
        messages.drain(..skip);
    }
    let audit_context = create_command_audit_context(&state, &message, "tldr").await;

    let clock = WallClock::start();
    let response = clock.run(async {
        let mut snapshot = ModelCatalogSnapshot::load();
        let id = snapshot.resolve_default(ModelRequestCapabilities { require_tools: true, ..Default::default() })?;
        let final_model = ResolvedTextModel::prepare(&id, &mut snapshot, None).await?;
        if messages.len() > CONFIG.agents.tldr_map_reduce_threshold {
        let mut progress_reporter =
            ProgressReporter::new(bot.clone(), message.chat.id, processing_message.id);
        match crate::agents::tldr::summarize_messages_map_reduce(
            &messages,
            audit_context.as_ref(),
            &mut progress_reporter,
            &clock,
            &final_model,
        )
        .await
        {
            Ok(crate::agents::tldr::TldrOutcome::Answer(crate::agents::common::ModelAnswer {
                text,
                model_display,
            })) => Ok((text, model_display)),
            Ok(crate::agents::tldr::TldrOutcome::UseLegacy(reason)) => {
                clock.check()?;
                info!("Map-reduce /tldr fell back to the single-call path: {reason}");
                tldr_single_call(&final_model, &messages, audit_context.as_ref()).await
            }
            Err(err) => Err(err),
        }
    } else {
        tldr_single_call(&final_model, &messages, audit_context.as_ref()).await
    }
    }).await?;

    let (mut summary_text, summary_model) = response;
    if truncated_to_cap {
        summary_text = format!(
            "（注：消息数量超过上限，本次仅总结最近 {} 条消息。）\n\n{}",
            CONFIG.agents.tldr_max_messages, summary_text
        );
    }
    if summary_text.trim().is_empty() {
        bot.edit_message_text(
            processing_message.chat.id,
            processing_message.id,
            "Failed to generate a summary. Please try again later.",
        )
        .await?;
        complete_command_timer(&mut timer, "error", Some("empty_summary".to_string()));
        return Ok(());
    }

    let infographic_enabled = CONFIG.agents.enable_tldr_infographic;

    let _ = bot
        .edit_message_text(
            processing_message.chat.id,
            processing_message.id,
            if infographic_enabled {
                "Summary generated. Generating infographic..."
            } else {
                "Summary generated. Skipping infographic step..."
            },
        )
        .await;

    let infographic_prompt = format!(
        "Create a clear infographic (no walls of text) summarizing the key points below. \
Use a 16:9 layout with readable labels and visual hierarchy suitable for Telegram. \
Use the same language as the summary text for any labels.\
\n\n{}",
        summary_text
    );

    let mut infographic_url = None;
    if infographic_enabled {
        let infographic_config = Some(GeminiImageConfig {
            aspect_ratio: Some("16:9".to_string()),
            image_size: Some("4K".to_string()),
        });
        let (infographic_model, infographic_result) = generate_image_with_configured_default(
            &infographic_prompt,
            &[],
            infographic_config,
            None,
            false,
            audit_context.as_ref(),
        )
        .await;
        match infographic_result {
            Ok(images) => {
                if let Some(image) = images.into_iter().next() {
                    if CONFIG.cwd_pw.api_key.trim().is_empty() {
                        warn!("TLDR infographic generated but CWD_PW_API_KEY is not configured.");
                    } else {
                        let mime_type =
                            detect_mime_type(&image).unwrap_or_else(|| "image/png".to_string());
                        infographic_url = upload_image_bytes_to_cwd(
                            &image,
                            &CONFIG.cwd_pw.api_key,
                            &mime_type,
                            Some(infographic_model.as_str()),
                            Some(&infographic_prompt),
                        )
                        .await;
                        if infographic_url.is_none() {
                            warn!("Failed to upload TLDR infographic to cwd.pw.");
                        }
                    }
                } else {
                    warn!("TLDR infographic generation returned no image.");
                }
            }
            Err(err) => {
                error!("Error generating TLDR infographic: {}", err);
            }
        }
    }

    let mut telegraph_url = None;
    if let Some(url) = &infographic_url {
        let telegraph_content = format!(
            "![Infographic]({})\n\n{}",
            url, summary_text
        );
        telegraph_url =
            create_telegraph_answer("Message Summary with Infographic", &telegraph_content, Some(&summary_model)).await;
    }

    let final_message = if let Some(url) = telegraph_url {
        ResponseContent::new(format!("Chat summary with infographic: [View it here](<{url}>)")).with_model(&summary_model)
    } else if let Some(url) = infographic_url {
        ResponseContent::new(format!("{summary_text}\n\nInfographic: [View it here](<{url}>)")).with_model(&summary_model)
    } else {
        ResponseContent::new(summary_text).with_model(&summary_model)
    };

    let _ = bot
        .edit_message_text(
            processing_message.chat.id,
            processing_message.id,
            if infographic_enabled {
                "Infographic step completed. Finalizing response..."
            } else {
                "Finalizing response..."
            },
        )
        .await;

    send_response(
        &bot,
        processing_message.chat.id,
        processing_message.id,
        &final_message,
        "Message Summary",
    )
    .await?;
    complete_command_timer(&mut timer, "success", None);

    Ok(())
    }).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tldr_count_defaults_when_missing_or_unparsable() {
        assert_eq!(resolve_tldr_count(None, 2_000), TLDR_DEFAULT_MESSAGE_COUNT);
        assert_eq!(
            resolve_tldr_count(Some("lots"), 2_000),
            TLDR_DEFAULT_MESSAGE_COUNT
        );
        assert_eq!(resolve_tldr_count(Some(" 50 "), 2_000), 50);
    }

    #[test]
    fn tldr_count_is_clamped_to_a_bounded_range() {
        assert_eq!(resolve_tldr_count(Some("-1"), 2_000), 1);
        assert_eq!(resolve_tldr_count(Some("0"), 2_000), 1);
        assert_eq!(resolve_tldr_count(Some("9999999"), 2_000), 2_000);
    }
}
