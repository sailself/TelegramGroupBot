//! `/factcheck`: fact-check text, images, video, or audio content.

use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::{ChatAction, MessageId, ReplyParameters};
use tracing::info;

use crate::agents::factcheck::{run_factcheck_pipeline, FactcheckOutcome};
use crate::agents::step::WallClock;
use crate::config::CONFIG;
use crate::handlers::access::{check_access_control, is_rate_limited};
use crate::handlers::agent_model_picker::{
    announce_quota_stand_in, run_with_agent_model_choice, AgentModelRequest, ChosenAgentModel,
};
use crate::handlers::enrichment::{
    count_sources, enrich_request, entity_link_urls, render_sources, Enrichment, EnrichmentBudget,
    SourceCounts, SourceKind,
};
use crate::handlers::media::{collect_message_media, MediaCollectionOptions};
use crate::handlers::responses::{send_response, ResponseContent};
use crate::llm::audit::{create_command_audit_context, LlmAuditContext};
use crate::llm::media::{summarize_media_files, MediaFile, MediaSummary};
use crate::llm::resolved_model::ResolvedTextModel;
use crate::llm::text_model::{call_resolved_text_model, ModelRequestCapabilities};
use crate::prompts::{FACTCHECK_SYSTEM_PROMPT, LANGUAGE_POLICY};
use crate::state::AppState;
use crate::utils::progress::{status_with_note, ProgressReporter};
use crate::utils::telegram::{message_entities_for_text, start_chat_action_heartbeat};

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

fn skipped_attachment_note(skipped: usize) -> Option<String> {
    match skipped {
        0 => None,
        1 => Some("1 rich-message attachment could not be included in the fact-check.".into()),
        count => Some(format!(
            "{count} rich-message attachments could not be included in the fact-check."
        )),
    }
}

pub async fn factcheck_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    query: Option<String>,
    rich: crate::utils::telegram::rich_message::RichUpdate,
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
    let heavy_permit = state.acquire_heavy_command_permit().await;

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
    let collected_media = collect_message_media(&bot, &state, &message, media_options, &rich).await;
    let status_note = skipped_attachment_note(collected_media.rich_attachments_skipped);

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
        let text = if status_note.is_some() {
            "There is no text or available media to fact-check."
        } else {
            "Please reply to a message to fact-check."
        };
        bot.send_message(
            message.chat.id,
            status_with_note(text, status_note.as_deref()).into_owned(),
        )
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

    let capabilities = ModelRequestCapabilities::from_media(
        &media_summary,
        !CONFIG.agents.enable_agentic_factcheck,
    );
    let job = FactcheckJob {
        chat_id: message.chat.id,
        statement,
        media_files,
        media_summary,
        user_language_code: user_language_code.map(str::to_string),
        audit_context,
        status_note: status_note.clone(),
    };
    run_with_agent_model_choice(
        &bot,
        &state,
        &message,
        heavy_permit,
        AgentModelRequest {
            command: "factcheck",
            capabilities,
            processing_text: processing_message_text,
            status_note,
            run: Box::new(move |bot, state, model| Box::pin(run_factcheck(bot, state, job, model))),
        },
    )
    .await
}

/// What `/factcheck` checks, gathered before the model is chosen.
struct FactcheckJob {
    chat_id: ChatId,
    statement: String,
    media_files: Vec<MediaFile>,
    media_summary: MediaSummary,
    user_language_code: Option<String>,
    audit_context: Option<LlmAuditContext>,
    status_note: Option<String>,
}

impl FactcheckJob {
    /// One attempt on `final_model` within `clock`: the agentic pipeline when
    /// it is enabled and takes the request, otherwise the single call.
    async fn check(
        &self,
        bot: &Bot,
        status_message_id: MessageId,
        clock: &WallClock,
        final_model: &ResolvedTextModel,
    ) -> Result<(String, String)> {
        let user_language_code = self.user_language_code.as_deref();
        let audit_context = self.audit_context.as_ref();
        clock
            .run(async {
                if CONFIG.agents.enable_agentic_factcheck {
                    let mut progress =
                        ProgressReporter::new(bot.clone(), self.chat_id, status_message_id)
                            .with_note(self.status_note.as_deref());
                    match run_factcheck_pipeline(
                        &self.statement,
                        &self.media_files,
                        &self.media_summary,
                        user_language_code,
                        audit_context,
                        &mut progress,
                        clock,
                        final_model,
                    )
                    .await?
                    {
                        FactcheckOutcome::Answer(answer) => {
                            return Ok((answer.text, answer.model_display))
                        }
                        FactcheckOutcome::UseLegacy(reason) => {
                            info!("Agentic fact-check fell back: {reason}")
                        }
                    }
                }
                clock.check()?;
                call_resolved_text_model(
                    final_model,
                    crate::llm::text_model::TextCallRequest {
                        system_prompt: &build_factcheck_system_prompt(user_language_code),
                        user_content: &self.statement,
                        response_title: "Fact Check",
                        tools_enabled: true,
                        use_pro: self.media_summary.total > 0,
                        media_files: Some(self.media_files.clone()),
                        prompt_name: Some("FACTCHECK_SYSTEM_PROMPT"),
                        audit_context,
                    },
                )
                .await
            })
            .await
    }
}

async fn run_factcheck(
    bot: Bot,
    state: AppState,
    job: FactcheckJob,
    model: ChosenAgentModel,
) -> Result<()> {
    let chat_id = job.chat_id;
    let uses_pro = job.media_summary.total > 0;
    let failure_text = status_with_note(
        "Failed to fact-check this message. Please try again.",
        job.status_note.as_deref(),
    )
    .into_owned();
    let ChosenAgentModel {
        id,
        mut snapshot,
        capabilities,
        status_message_id,
        heavy_permit,
    } = model;
    let _heavy_permit = state.reuse_or_acquire_heavy_permit(heavy_permit).await;
    crate::utils::telegram::run_with_status_message(
        &bot,
        chat_id,
        status_message_id,
        &failure_text,
        async {
            let _chat_action =
                start_chat_action_heartbeat(bot.clone(), chat_id, ChatAction::Typing);

            let clock = WallClock::start().for_gemini_pipeline(id == "gemini", uses_pro);
            let final_model = clock
                .run(ResolvedTextModel::prepare(&id, &mut snapshot, None))
                .await?;
            let response = match job
                .check(&bot, status_message_id, &clock, &final_model)
                .await
            {
                Ok(response) => response,
                Err(error) => {
                    // Codex refused the request for a spent usage window: check
                    // on the model standing in for Codex rather than failing.
                    let Some(stand_in) = final_model.quota_stand_in(&error, capabilities).await
                    else {
                        return Err(error);
                    };
                    announce_quota_stand_in(
                        &bot,
                        chat_id,
                        status_message_id,
                        "factcheck",
                        &stand_in,
                        job.status_note.as_deref(),
                    )
                    .await;
                    let clock = WallClock::start().for_gemini_pipeline(
                        matches!(stand_in.model, ResolvedTextModel::Gemini),
                        uses_pro,
                    );
                    job.check(&bot, status_message_id, &clock, &stand_in.model)
                        .await?
                }
            };

            let (response_text, response_model) = response;
            let response_with_model = ResponseContent::new(response_text)
                .with_model(response_model)
                .with_notice(job.status_note);

            send_response(
                &bot,
                chat_id,
                status_message_id,
                &response_with_model,
                "Fact Check",
            )
            .await?;

            Ok(())
        },
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{response_with_status, ExpectedRequest, TestServer};
    use crate::utils::telegram::rich_message::RichUpdate;
    use serde_json::json;

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

    #[tokio::test]
    async fn media_only_rich_reply_explains_why_it_cannot_be_checked() {
        let server = TestServer::new(vec![
            ExpectedRequest::new(
                "POST",
                "/bot123:test/GetFile",
                response_with_status(
                    400,
                    br#"{"ok":false,"error_code":400,"description":"Bad Request: file unavailable"}"#
                        .to_vec(),
                ),
            )
            .with_body_fragment(br#""file_id":"unavailable""#),
            ExpectedRequest::new(
                "POST",
                "/bot123:test/SendMessage",
                response_with_status(
                    200,
                    br#"{"ok":true,"result":{"message_id":3,"date":1,"chat":{"id":2,"type":"private"},"text":"notice"}}"#
                        .to_vec(),
                ),
            )
            .with_body_fragment(b"There is no text or available media to fact-check.")
            .with_body_fragment(b"1 rich-message attachment could not be included"),
        ]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let db = crate::db::test_support::init_test_db("factcheck-skipped-rich-reply").await;
        let state = AppState::new(db.clone(), 1, "test_bot".to_string());
        let chat = json!({"id": 2, "type": "private"});
        let mut raw = json!({
            "update_id": 2,
            "message": {
                "message_id": 2, "date": 1, "chat": chat, "text": "/factcheck",
                "from": {"id": 2468, "is_bot": false, "first_name": "fixture"},
                "reply_to_message": {
                    "message_id": 1, "date": 1, "chat": chat,
                    "rich_message": {"blocks": [{
                        "type": "photo", "photo": [{"file_id": "unavailable", "width": 1, "height": 1}]
                    }]}
                }
            }
        });
        let rich = RichUpdate::from_update(&mut raw);
        let message = serde_json::from_value(raw["message"].take()).unwrap();

        factcheck_handler(bot, state, message, None, rich)
            .await
            .unwrap();
        server.join().unwrap();
        db.shutdown().await;
    }
}
