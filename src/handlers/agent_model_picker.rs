//! Gemini-or-Codex picker for `/tldr` and `/factcheck`.
//!
//! When Gemini and a ready Codex model can both serve a request, the command
//! asks its sender which one to use; otherwise it runs on the default text
//! model right away. Everything the command gathered (messages, media, link
//! content) is frozen before the picker, so the choice only decides the model.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::{InlineKeyboardButton, InlineKeyboardMarkup, MessageId};
use tokio::sync::OwnedSemaphorePermit;
use tracing::{error, warn};

use crate::config::{ThirdPartyProvider, CONFIG};
use crate::llm::resolved_model::ModelCatalogSnapshot;
use crate::llm::text_model::{
    available_third_party_models_for_request, resolve_default_text_model_with_models,
    ModelRequestCapabilities, MODEL_GEMINI,
};
use crate::state::{AppState, PendingEntryGuard};
use crate::utils::telegram::{reply_with_retry, retry_telegram, send_message_with_retry};
use crate::utils::timing::now_unix_seconds;

pub const AGENT_MODEL_CALLBACK_PREFIX: &str = "agent_model:";

/// The model a command runs on, the catalog it was validated against, and the
/// message that reports the command's progress.
pub(crate) struct ChosenAgentModel {
    pub id: String,
    pub snapshot: ModelCatalogSnapshot,
    pub status_message_id: MessageId,
    /// The slot held since intake when no picker was shown. A picked request
    /// waits for a fresh one rather than holding a slot while its user decides.
    pub heavy_permit: Option<OwnedSemaphorePermit>,
}

pub(crate) type AgentModelRun = Box<
    dyn FnOnce(Bot, AppState, ChosenAgentModel) -> Pin<Box<dyn Future<Output = Result<()>> + Send>>
        + Send,
>;

/// What a command hands the picker: how to gate and describe it, and how to run it.
pub(crate) struct AgentModelRequest {
    pub command: &'static str,
    pub capabilities: ModelRequestCapabilities,
    /// Status shown while the command runs, e.g. "Summarizing recent messages...".
    pub processing_text: String,
    pub run: AgentModelRun,
}

/// A request waiting on the picker, keyed by `"{chat_id}_{selection_message_id}"`.
pub(crate) struct PendingAgentModelRequest {
    original_user_id: i64,
    chat_id: ChatId,
    selection_message_id: MessageId,
    timestamp: i64,
    /// Model ids in button order; the callback data carries the index.
    choices: Vec<String>,
    request: AgentModelRequest,
}

/// Models to offer for a request, default first; empty when the request runs
/// on the default model without asking. The picker exists to choose between
/// Gemini and Codex, so both must be able to serve the request.
pub(crate) fn agent_model_choices(
    snapshot: &ModelCatalogSnapshot,
    gemini_available: bool,
    default_model: &str,
    request: ModelRequestCapabilities,
) -> Vec<String> {
    let codex_ids = available_third_party_models_for_request(
        &snapshot.models,
        &snapshot.ready_providers,
        request,
    )
    .into_iter()
    .filter(|config| config.provider == ThirdPartyProvider::OpenAICodex)
    .map(|config| config.id.clone())
    .collect::<Vec<_>>();
    if !gemini_available || codex_ids.is_empty() {
        return Vec::new();
    }

    let mut choices = vec![MODEL_GEMINI.to_string()];
    choices.extend(codex_ids);
    // The default leads, so the first button is also what a timeout runs.
    if let Ok(default_id) = resolve_default_text_model_with_models(
        default_model,
        &snapshot.models,
        &snapshot.ready_providers,
        gemini_available,
        request,
    ) {
        choices.retain(|id| *id != default_id);
        choices.insert(0, default_id);
    }
    choices
}

fn agent_model_callback_data(index: usize) -> String {
    format!("{AGENT_MODEL_CALLBACK_PREFIX}{index}")
}

fn parse_agent_model_callback(data: &str) -> Option<usize> {
    data.strip_prefix(AGENT_MODEL_CALLBACK_PREFIX)?.parse().ok()
}

fn agent_model_keyboard(
    snapshot: &ModelCatalogSnapshot,
    choices: &[String],
) -> InlineKeyboardMarkup {
    let buttons = choices
        .iter()
        .enumerate()
        .map(|(index, model_id)| {
            InlineKeyboardButton::callback(
                snapshot.display_name(model_id),
                agent_model_callback_data(index),
            )
        })
        .collect::<Vec<_>>();
    InlineKeyboardMarkup::new(buttons.chunks(2).map(<[_]>::to_vec))
}

/// "Summarizing recent messages..." becomes "Summarizing recent messages with Gemini...".
fn status_with_model(processing_text: &str, model_label: &str) -> String {
    format!(
        "{} with {model_label}...",
        processing_text.trim_end_matches("...")
    )
}

enum AgentModelCallbackAction {
    Missing,
    Ignored,
    InvalidSelection,
    UseDefault(PendingAgentModelRequest),
    UseSelected(PendingAgentModelRequest, String),
}

fn take_pending_agent_model_request(
    pending: &mut PendingEntryGuard<'_, PendingAgentModelRequest>,
    query_user_id: i64,
    now: i64,
    timeout_secs: u64,
    choice: usize,
    can_select: impl FnOnce(&str, ModelRequestCapabilities) -> bool,
) -> AgentModelCallbackAction {
    let Some(request) = pending.get() else {
        return AgentModelCallbackAction::Missing;
    };
    if request.original_user_id != query_user_id {
        return AgentModelCallbackAction::Ignored;
    }

    let timeout_secs = i64::try_from(timeout_secs).unwrap_or(i64::MAX);
    if now.saturating_sub(request.timestamp) > timeout_secs {
        return pending
            .take()
            .map(AgentModelCallbackAction::UseDefault)
            .unwrap_or(AgentModelCallbackAction::Missing);
    }

    let Some(model_id) = request.choices.get(choice).cloned() else {
        return AgentModelCallbackAction::InvalidSelection;
    };
    if !can_select(&model_id, request.request.capabilities) {
        return AgentModelCallbackAction::InvalidSelection;
    }
    pending
        .take()
        .map(|request| AgentModelCallbackAction::UseSelected(request, model_id))
        .unwrap_or(AgentModelCallbackAction::Missing)
}

/// Run `request` on the default model, or first ask which model to use when
/// both Gemini and Codex can serve it. `heavy_permit` is the slot the command
/// held for its intake.
pub(crate) async fn run_with_agent_model_choice(
    bot: &Bot,
    state: &AppState,
    message: &Message,
    heavy_permit: OwnedSemaphorePermit,
    request: AgentModelRequest,
) -> Result<()> {
    let snapshot = ModelCatalogSnapshot::load();
    // A bot (or an anonymous admin, who posts as one) cannot press the buttons.
    let requested_by_bot = message.from.as_ref().is_none_or(|user| user.is_bot);
    let choices = if requested_by_bot {
        Vec::new()
    } else {
        agent_model_choices(
            &snapshot,
            CONFIG.gemini_api_available(),
            &CONFIG.models.default_text_model,
            request.capabilities,
        )
    };

    if choices.is_empty() {
        let model_id = match snapshot.resolve_default(request.capabilities) {
            Ok(model_id) => model_id,
            Err(err) => {
                send_message_with_retry(bot, message.chat.id, &err.to_string(), Some(message.id))
                    .await?;
                return Ok(());
            }
        };
        let status = send_message_with_retry(
            bot,
            message.chat.id,
            &request.processing_text,
            Some(message.id),
        )
        .await?;
        let chosen = ChosenAgentModel {
            id: model_id,
            snapshot,
            status_message_id: status.id,
            heavy_permit: Some(heavy_permit),
        };
        return (request.run)(bot.clone(), state.clone(), chosen).await;
    }

    drop(heavy_permit);
    let selection = reply_with_retry(
        bot,
        message.chat.id,
        &format!(
            "Please select which AI model to use for /{}:",
            request.command
        ),
        Some(message.id),
        None,
        Some(agent_model_keyboard(&snapshot, &choices)),
    )
    .await?;
    let original_user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    let pending = PendingAgentModelRequest {
        original_user_id,
        chat_id: message.chat.id,
        selection_message_id: selection.id,
        timestamp: now_unix_seconds(),
        choices,
        request,
    };
    let timeout_bot = bot.clone();
    let timeout_state = state.clone();
    state.pending_agent_model_requests.insert_with_timeout(
        format!("{}_{}", message.chat.id.0, selection.id.0),
        pending,
        Duration::from_secs(CONFIG.limits.model_selection_timeout),
        move |pending| run_on_default_model(timeout_bot, timeout_state, pending),
    );
    Ok(())
}

pub async fn agent_model_callback(bot: Bot, state: AppState, query: CallbackQuery) -> Result<()> {
    bot.answer_callback_query(query.id.clone()).await?;

    let Some(choice) = query.data.as_deref().and_then(parse_agent_model_callback) else {
        return Ok(());
    };
    let Some(message) = query.message.as_ref() else {
        return Ok(());
    };
    let request_key = format!("{}_{}", message.chat().id.0, message.id().0);
    let query_user_id = i64::try_from(query.from.id.0).unwrap_or_default();
    // Revalidate the choice against the current catalog before freezing it.
    let snapshot = ModelCatalogSnapshot::load();
    let action = {
        let mut pending = state.pending_agent_model_requests.entry(&request_key);
        take_pending_agent_model_request(
            &mut pending,
            query_user_id,
            now_unix_seconds(),
            CONFIG.limits.model_selection_timeout,
            choice,
            |model_id, capabilities| snapshot.can_select(model_id, capabilities),
        )
    };

    match action {
        AgentModelCallbackAction::UseSelected(pending, model_id) => {
            run_on_model(bot, state, pending, model_id, snapshot, false).await
        }
        AgentModelCallbackAction::UseDefault(pending) => {
            run_on_default_model(bot, state, pending).await;
            Ok(())
        }
        AgentModelCallbackAction::Missing
        | AgentModelCallbackAction::Ignored
        | AgentModelCallbackAction::InvalidSelection => Ok(()),
    }
}

/// The picker timed out: run on the default model, or report why it cannot.
async fn run_on_default_model(bot: Bot, state: AppState, pending: PendingAgentModelRequest) {
    let command = pending.request.command;
    let snapshot = ModelCatalogSnapshot::load();
    let result = match snapshot.resolve_default(pending.request.capabilities) {
        Ok(model_id) => run_on_model(bot, state, pending, model_id, snapshot, true).await,
        Err(err) => {
            close_picker(
                &bot,
                pending.chat_id,
                pending.selection_message_id,
                &err.to_string(),
            )
            .await
        }
    };
    if let Err(err) = result {
        error!("/{command} failed after its model picker timed out: {err:#}");
    }
}

async fn run_on_model(
    bot: Bot,
    state: AppState,
    pending: PendingAgentModelRequest,
    model_id: String,
    snapshot: ModelCatalogSnapshot,
    timed_out: bool,
) -> Result<()> {
    let PendingAgentModelRequest {
        chat_id,
        selection_message_id,
        request,
        ..
    } = pending;
    let status = status_with_model(&request.processing_text, &snapshot.display_name(&model_id));
    let status = if timed_out {
        format!("No model selected in time. {status}")
    } else {
        status
    };
    // The picker becomes the command's status message. A failed relabel must
    // not stop the run: the command reports into this message as it goes.
    if let Err(err) = close_picker(&bot, chat_id, selection_message_id, &status).await {
        warn!(
            "Failed to close the /{} model picker: {err:#}",
            request.command
        );
    }
    let chosen = ChosenAgentModel {
        id: model_id,
        snapshot,
        status_message_id: selection_message_id,
        heavy_permit: None,
    };
    (request.run)(bot, state, chosen).await
}

async fn close_picker(bot: &Bot, chat_id: ChatId, message_id: MessageId, text: &str) -> Result<()> {
    retry_telegram("close agent model picker", || {
        bot.edit_message_text(chat_id, message_id, text.to_string())
            .reply_markup(InlineKeyboardMarkup::new(
                Vec::<Vec<InlineKeyboardButton>>::new(),
            ))
    })
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ThirdPartyModelConfig;
    use crate::llm::runtime_models::{CodexSelectedModelRecord, OPENAI_CODEX_SELECTED_MODEL_ID};
    use crate::state::PendingRequests;
    use crate::test_support::{response_with_status, ExpectedRequest, TestServer};
    use teloxide::types::InlineKeyboardButtonKind;

    fn model(provider: ThirdPartyProvider, id: &str, name: &str) -> ThirdPartyModelConfig {
        ThirdPartyModelConfig {
            id: id.to_string(),
            provider,
            name: name.to_string(),
            model: id
                .split_once(':')
                .map_or(id, |(_, model)| model)
                .to_string(),
            image: true,
            video: false,
            audio: false,
            tools: true,
        }
    }

    fn codex_record() -> CodexSelectedModelRecord {
        serde_json::from_value(serde_json::json!({
            "slug": "gpt-5.4",
            "display_name": "GPT-5.4",
            "account_id": "account",
            "selected_reasoning_level": "high",
            "supports_search_tool": true,
            "use_responses_lite": true,
            "fetched_at": "2026-10-01T00:00:00Z"
        }))
        .unwrap()
    }

    /// Gemini plus a ready selected Codex model and a ready OpenRouter model.
    fn snapshot() -> ModelCatalogSnapshot {
        let mut codex = model(
            ThirdPartyProvider::OpenAICodex,
            OPENAI_CODEX_SELECTED_MODEL_ID,
            "Selected Codex",
        );
        codex.model = "gpt-5.4".to_string();
        ModelCatalogSnapshot {
            models: vec![
                codex,
                model(
                    ThirdPartyProvider::OpenRouter,
                    "openrouter:qwen",
                    "OpenRouter Qwen",
                ),
            ],
            ready_providers: vec![
                ThirdPartyProvider::OpenAICodex,
                ThirdPartyProvider::OpenRouter,
            ],
            codex_record: Some(codex_record()),
        }
    }

    fn tools() -> ModelRequestCapabilities {
        ModelRequestCapabilities {
            require_tools: true,
            ..Default::default()
        }
    }

    #[test]
    fn choices_need_both_gemini_and_a_ready_codex_model() {
        let snapshot = snapshot();
        assert_eq!(
            agent_model_choices(&snapshot, true, "gemini", tools()),
            vec!["gemini", OPENAI_CODEX_SELECTED_MODEL_ID]
        );
        assert!(agent_model_choices(&snapshot, false, "gemini", tools()).is_empty());

        let mut codex_not_ready = snapshot.clone();
        codex_not_ready
            .ready_providers
            .retain(|provider| *provider != ThirdPartyProvider::OpenAICodex);
        assert!(agent_model_choices(&codex_not_ready, true, "gemini", tools()).is_empty());

        let mut no_codex = snapshot;
        no_codex
            .models
            .retain(|config| config.provider != ThirdPartyProvider::OpenAICodex);
        assert!(agent_model_choices(&no_codex, true, "gemini", tools()).is_empty());
    }

    #[test]
    fn choices_skip_the_picker_when_codex_cannot_serve_the_request() {
        let snapshot = snapshot();
        for request in [
            ModelRequestCapabilities {
                has_video: true,
                ..tools()
            },
            ModelRequestCapabilities {
                has_documents: true,
                ..tools()
            },
        ] {
            assert!(agent_model_choices(&snapshot, true, "gemini", request).is_empty());
        }
        let images = ModelRequestCapabilities {
            has_images: true,
            ..tools()
        };
        assert_eq!(
            agent_model_choices(&snapshot, true, "gemini", images).len(),
            2
        );
    }

    #[test]
    fn choices_put_the_default_model_first() {
        let snapshot = snapshot();
        for (default_model, expected) in [
            ("gemini", vec!["gemini", OPENAI_CODEX_SELECTED_MODEL_ID]),
            (
                "openai-codex",
                vec![OPENAI_CODEX_SELECTED_MODEL_ID, "gemini"],
            ),
            (
                OPENAI_CODEX_SELECTED_MODEL_ID,
                vec![OPENAI_CODEX_SELECTED_MODEL_ID, "gemini"],
            ),
            // A third default leads too, since it is what a timeout runs.
            (
                "openrouter:qwen",
                vec!["openrouter:qwen", "gemini", OPENAI_CODEX_SELECTED_MODEL_ID],
            ),
            // An unusable default adds nothing.
            (
                "nvidia:missing",
                vec!["gemini", OPENAI_CODEX_SELECTED_MODEL_ID],
            ),
        ] {
            assert_eq!(
                agent_model_choices(&snapshot, true, default_model, tools()),
                expected,
                "default {default_model}"
            );
        }
    }

    #[test]
    fn keyboard_labels_choices_in_order_and_callbacks_carry_their_index() {
        let snapshot = snapshot();
        let choices = vec![
            OPENAI_CODEX_SELECTED_MODEL_ID.to_string(),
            "gemini".to_string(),
        ];
        let buttons = agent_model_keyboard(&snapshot, &choices)
            .inline_keyboard
            .into_iter()
            .flatten()
            .map(|button| match button.kind {
                InlineKeyboardButtonKind::CallbackData(data) => (button.text, data),
                _ => panic!("expected a callback button"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            buttons,
            vec![
                ("gpt-5.4 high".to_string(), "agent_model:0".to_string()),
                ("Gemini".to_string(), "agent_model:1".to_string()),
            ]
        );
        assert_eq!(parse_agent_model_callback("agent_model:1"), Some(1));
        for other in ["agent_model:x", "agent_model:", "model_select:gemini"] {
            assert_eq!(parse_agent_model_callback(other), None, "{other}");
        }
    }

    #[test]
    fn status_names_the_chosen_model() {
        assert_eq!(
            status_with_model("Summarizing recent messages...", "Gemini"),
            "Summarizing recent messages with Gemini..."
        );
        assert_eq!(
            status_with_model(
                "Analyzing 2 image(s) and fact-checking content...",
                "gpt-5.4 high"
            ),
            "Analyzing 2 image(s) and fact-checking content with gpt-5.4 high..."
        );
    }

    type RunReport = (String, MessageId, bool);

    fn pending_request(
        original_user_id: i64,
        timestamp: i64,
    ) -> (
        PendingAgentModelRequest,
        tokio::sync::oneshot::Receiver<RunReport>,
    ) {
        let (report, ran) = tokio::sync::oneshot::channel();
        let run: AgentModelRun = Box::new(move |_bot, _state, chosen: ChosenAgentModel| {
            Box::pin(async move {
                let _ = report.send((
                    chosen.id,
                    chosen.status_message_id,
                    chosen.heavy_permit.is_none(),
                ));
                Ok(())
            })
        });
        let pending = PendingAgentModelRequest {
            original_user_id,
            chat_id: ChatId(2),
            selection_message_id: MessageId(4),
            timestamp,
            choices: vec![
                "gemini".to_string(),
                OPENAI_CODEX_SELECTED_MODEL_ID.to_string(),
            ],
            request: AgentModelRequest {
                command: "tldr",
                capabilities: tools(),
                processing_text: "Summarizing recent messages...".to_string(),
                run,
            },
        };
        (pending, ran)
    }

    fn pending_map(
        original_user_id: i64,
        timestamp: i64,
    ) -> PendingRequests<PendingAgentModelRequest> {
        let pending = PendingRequests::new();
        pending.insert(
            "key".to_string(),
            pending_request(original_user_id, timestamp).0,
        );
        pending
    }

    #[test]
    fn callback_take_ignores_other_users_and_invalid_choices() {
        let pending = pending_map(10, 100);
        let ignored =
            take_pending_agent_model_request(&mut pending.entry("key"), 20, 105, 30, 1, |_, _| {
                true
            });
        assert!(matches!(ignored, AgentModelCallbackAction::Ignored));
        let out_of_range =
            take_pending_agent_model_request(&mut pending.entry("key"), 10, 105, 30, 2, |_, _| {
                true
            });
        assert!(matches!(
            out_of_range,
            AgentModelCallbackAction::InvalidSelection
        ));
        let unavailable = take_pending_agent_model_request(
            &mut pending.entry("key"),
            10,
            105,
            30,
            1,
            |model_id, capabilities| {
                assert_eq!(model_id, OPENAI_CODEX_SELECTED_MODEL_ID);
                assert_eq!(capabilities, tools());
                false
            },
        );
        assert!(matches!(
            unavailable,
            AgentModelCallbackAction::InvalidSelection
        ));
        assert_eq!(pending.count(), 1);
    }

    #[test]
    fn callback_take_consumes_a_valid_choice_and_defaults_after_timeout() {
        let pending = pending_map(10, 100);
        let selected =
            take_pending_agent_model_request(&mut pending.entry("key"), 10, 105, 30, 1, |_, _| {
                true
            });
        let AgentModelCallbackAction::UseSelected(_, model_id) = selected else {
            panic!("expected the selected model");
        };
        assert_eq!(model_id, OPENAI_CODEX_SELECTED_MODEL_ID);
        assert_eq!(pending.count(), 0);

        let pending = pending_map(10, 100);
        let expired =
            take_pending_agent_model_request(&mut pending.entry("key"), 10, 131, 30, 1, |_, _| {
                true
            });
        assert!(matches!(expired, AgentModelCallbackAction::UseDefault(_)));
        assert_eq!(pending.count(), 0);

        let missing = PendingRequests::new();
        assert!(matches!(
            take_pending_agent_model_request(&mut missing.entry("key"), 10, 105, 30, 0, |_, _| {
                true
            }),
            AgentModelCallbackAction::Missing
        ));
    }

    fn edit_ok() -> Vec<u8> {
        response_with_status(
            200,
            br#"{"ok":true,"result":{"message_id":4,"date":1,"chat":{"id":2,"type":"private"},"text":"status"}}"#
                .to_vec(),
        )
    }

    #[tokio::test]
    async fn a_chosen_model_turns_the_picker_into_the_status_message_and_runs() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "POST",
            "/bot123:test/EditMessageText",
            edit_ok(),
        )
        .with_body_fragment(br#""message_id":4"#)
        .with_body_fragment(b"Summarizing recent messages with gpt-5.4 high...")
        .with_body_fragment(br#""reply_markup":{"inline_keyboard":[]}"#)]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let state = AppState::new(
            crate::db::test_support::init_test_db("agent-model-picker-run").await,
            1,
            "test_bot".into(),
        );
        let (pending, ran) = pending_request(10, 100);
        run_on_model(
            bot,
            state,
            pending,
            OPENAI_CODEX_SELECTED_MODEL_ID.to_string(),
            snapshot(),
            false,
        )
        .await
        .unwrap();
        assert_eq!(
            ran.await.unwrap(),
            (
                OPENAI_CODEX_SELECTED_MODEL_ID.to_string(),
                MessageId(4),
                true
            )
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn a_timed_out_picker_says_so_and_still_runs() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "POST",
            "/bot123:test/EditMessageText",
            edit_ok(),
        )
        .with_body_fragment(
            b"No model selected in time. Summarizing recent messages with Gemini...",
        )]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let state = AppState::new(
            crate::db::test_support::init_test_db("agent-model-picker-timeout").await,
            1,
            "test_bot".into(),
        );
        let (pending, ran) = pending_request(10, 100);
        run_on_model(bot, state, pending, "gemini".to_string(), snapshot(), true)
            .await
            .unwrap();
        assert_eq!(ran.await.unwrap().0, "gemini");
        server.join().unwrap();
    }

    fn callback(data: &str, user: i64) -> CallbackQuery {
        serde_json::from_value(serde_json::json!({
            "id": "callback",
            "from": {"id": user, "is_bot": false, "first_name": "fixture"},
            "message": {"message_id": 4, "date": 1, "chat": {"id": 2, "type": "private"}, "text": "picker"},
            "chat_instance": "test",
            "data": data
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn callbacks_from_other_users_or_with_stale_choices_keep_the_picker() {
        let acknowledgement = || {
            ExpectedRequest::new(
                "POST",
                "/bot123:test/AnswerCallbackQuery",
                response_with_status(200, br#"{"ok":true,"result":true}"#.to_vec()),
            )
        };
        let server = TestServer::new(vec![
            acknowledgement(),
            acknowledgement(),
            acknowledgement(),
        ]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let state = AppState::new(
            crate::db::test_support::init_test_db("agent-model-picker-callback").await,
            1,
            "test_bot".into(),
        );
        state
            .pending_agent_model_requests
            .insert("2_4".to_string(), pending_request(10, now_unix_seconds()).0);
        for (data, user) in [
            ("agent_model:0", 20),
            ("agent_model:7", 10),
            ("agent_model:not-a-choice", 10),
        ] {
            agent_model_callback(bot.clone(), state.clone(), callback(data, user))
                .await
                .unwrap();
        }
        assert_eq!(state.pending_agent_model_requests.count(), 1);
        server.join().unwrap();
    }
}
