//! Unit tests for the qa module family (trigger, prompt, chat_search, model_resolution, selection_ui, process, handler).

use super::*;
use super::{
    chat_search::*, handler::*, model_resolution::*, process::*, prompt::*, selection_ui::*,
    trigger::*,
};
use crate::config::{ThirdPartyModelConfig, ThirdPartyProvider};
use crate::handlers::enrichment::Enrichment;
use crate::llm::media::MediaSummary;
use crate::llm::runtime_models::{codex_selected_model_label, ResolvedExplicitCodexModel};
use crate::llm::text_model::{ModelRequestCapabilities, MODEL_GEMINI};
use crate::state::{PendingQRequest, PendingRequests, QaCommandMode};
use serde_json::json;
use teloxide::prelude::*;
use teloxide::types::InlineKeyboardButtonKind;

#[test]
fn q_system_prompt_renders_without_placeholders() {
    let rendered = build_system_prompt(Some("en"));
    assert!(
        !rendered.contains('{'),
        "unresolved placeholder in /q prompt: {rendered}"
    );
    assert!(rendered.contains("untrusted data"));
    assert!(rendered.contains("default to Chinese"));
    assert!(rendered.contains("en"));
    // The citation/verification contract must survive the detox.
    assert!(rendered.contains("Cite the sources"));
    assert!(rendered.to_lowercase().contains("web search"));
    // The shared language policy is composed exactly once.
    assert_eq!(
        rendered
            .matches("Response language — decide it yourself")
            .count(),
        1
    );
}

#[test]
fn quick_system_prompt_renders_bounded_answer_and_search_rules() {
    let rendered = build_quick_system_prompt(Some("en"));

    assert!(
        !rendered.contains('{'),
        "unresolved placeholder: {rendered}"
    );
    assert!(rendered.contains("minimum reasoning"));
    assert!(rendered.contains("1–5 short sentences"));
    assert!(rendered.contains("genuinely current or time-sensitive"));
    assert!(rendered.contains("cite every factual claim supported by the search"));
    assert!(rendered.contains("recommend /q"));
    assert!(rendered.contains("unavailable, inconclusive, conflicting, or insufficient"));
}

#[test]
fn qc_system_prompt_renders_without_placeholders_and_forbids_fabricated_links() {
    let rendered = build_chat_context_system_prompt(None);
    assert!(
        !rendered.contains('{'),
        "unresolved placeholder in /qc prompt: {rendered}"
    );
    assert!(rendered.contains("Never construct, guess, or reformat a message link"));
    assert!(rendered.contains("default to Chinese"));
    // A missing hint renders as the sentinel the policy already handles.
    assert!(rendered.contains("Telegram language hint: unknown"));
}

#[test]
fn unverified_chat_link_ids_flags_only_fabricated_ids() {
    let chat_id = -1001374348669;
    let answer = "See https://t.me/c/1374348669/100 and https://t.me/c/1374348669/999.";
    assert_eq!(unverified_chat_link_ids(answer, chat_id, &[100]), vec![999]);
    // Every cited ID was retrieved -> nothing flagged.
    assert!(unverified_chat_link_ids(answer, chat_id, &[100, 999]).is_empty());
    // Non-supergroup chats have no citeable t.me/c/ links.
    assert!(unverified_chat_link_ids(answer, 12345, &[]).is_empty());
    // A bare link prefix with no id at the very end must not panic.
    assert!(
        unverified_chat_link_ids("see https://t.me/c/1374348669/", -1001374348669, &[]).is_empty()
    );
    // A link immediately followed by a multibyte char must not panic on a
    // UTF-8 boundary — the dominant case for Chinese chats.
    let cjk = "见 https://t.me/c/1374348669/中文消息 和 https://t.me/c/1374348669/42";
    assert_eq!(
        unverified_chat_link_ids(cjk, -1001374348669, &[42]),
        Vec::<i64>::new()
    );
    let cjk_fab = "https://t.me/c/1374348669/中 https://t.me/c/1374348669/777";
    assert_eq!(
        unverified_chat_link_ids(cjk_fab, -1001374348669, &[]),
        vec![777]
    );
}

fn model(provider: ThirdPartyProvider, name: &str, raw_model: &str) -> ThirdPartyModelConfig {
    ThirdPartyModelConfig {
        id: format!("{}:{}", provider.as_str(), raw_model),
        provider,
        name: name.to_string(),
        model: raw_model.to_string(),
        image: false,
        video: false,
        audio: false,
        tools: true,
    }
}

fn pending_q_request(original_user_id: i64, timestamp: i64) -> PendingQRequest {
    PendingQRequest {
        user_id: original_user_id,
        query: "question".to_string(),
        telegram_language_code: None,
        enrichment: Enrichment::default(),
        chat_id: 123,
        message_id: 456,
        selection_message_id: 789,
        original_user_id,
        llm_invocation_id: None,
        timestamp,
        command_timer: None,
        mode: QaCommandMode::Standard,
    }
}

fn pending_with_request(original_user_id: i64, timestamp: i64) -> PendingRequests<PendingQRequest> {
    let pending = PendingRequests::new();
    pending.insert(
        "request".to_string(),
        pending_q_request(original_user_id, timestamp),
    );
    pending
}

#[test]
fn callback_take_keeps_pending_request_for_wrong_user() {
    let pending = pending_with_request(10, 100);

    let action =
        take_pending_q_request_for_callback(&mut pending.entry("request"), 20, 105, 30, |_| true);

    assert!(matches!(action, PendingQRequestCallbackAction::Ignored));
    assert!(pending.entry("request").get().is_some());
}

#[test]
fn callback_take_uses_default_model_when_selection_arrives_after_timeout() {
    let pending = pending_with_request(10, 100);

    let action =
        take_pending_q_request_for_callback(&mut pending.entry("request"), 10, 131, 30, |_| true);

    let PendingQRequestCallbackAction::UseDefault(request) = action else {
        panic!("expected expired callback to use the default model");
    };
    assert_eq!(request.original_user_id, 10);
    assert_eq!(pending.count(), 0);
}

#[test]
fn callback_take_keeps_pending_request_for_invalid_model_selection() {
    let pending = pending_with_request(10, 100);

    let action =
        take_pending_q_request_for_callback(&mut pending.entry("request"), 10, 105, 30, |_| false);

    assert!(matches!(
        action,
        PendingQRequestCallbackAction::InvalidSelection
    ));
    assert!(pending.entry("request").get().is_some());
}

#[test]
fn callback_take_consumes_pending_request_for_valid_selection() {
    let pending = pending_with_request(10, 100);

    let action =
        take_pending_q_request_for_callback(&mut pending.entry("request"), 10, 105, 30, |_| true);

    let PendingQRequestCallbackAction::UseSelected(request) = action else {
        panic!("expected valid callback to use the selected model");
    };
    assert_eq!(request.original_user_id, 10);
    assert_eq!(pending.count(), 0);
}

fn text_message_from(
    sender_id: u64,
    is_bot: bool,
    text: &str,
    entities: Vec<serde_json::Value>,
) -> Message {
    serde_json::from_value(json!({
        "message_id": 42,
        "date": 1,
        "chat": {
            "id": -100123,
            "type": "group",
            "title": "test group"
        },
        "from": {
            "id": sender_id,
            "is_bot": is_bot,
            "first_name": if is_bot { "PeerBot" } else { "Human" },
            "username": if is_bot { "peer_bot" } else { "human_user" }
        },
        "text": text,
        "entities": entities
    }))
    .expect("test message should deserialize")
}

#[test]
fn q_command_insert_records_the_question_as_an_ai_command() {
    let message = text_message_from(10, false, "/q what is rust", vec![]);

    let insert = build_q_command_insert(
        &message,
        10,
        "Human",
        "what is rust",
        "Context from replied message: \"x\"\n\nQuestion: what is rust",
        "q",
    );

    assert_eq!(insert.message_id, 42);
    assert_eq!(insert.chat_id, -100123);
    assert_eq!(insert.user_id, Some(10));
    assert_eq!(insert.username.as_deref(), Some("Human"));
    assert_eq!(insert.text.as_deref(), Some("/q what is rust"));
    assert!(insert
        .search_source_text
        .as_deref()
        .unwrap()
        .ends_with("what is rust"));
    assert!(insert.asks_ai);
    assert!(insert.is_command);
    assert!(insert.is_synthetic_record);
    assert_eq!(insert.ai_command.as_deref(), Some("q"));
}

#[test]
fn auto_q_triggers_for_other_bot_mentioning_this_bot() {
    let message = text_message_from(
        1001,
        true,
        "@HelperBot please review this",
        vec![json!({
            "type": "mention",
            "offset": 0,
            "length": 10
        })],
    );

    assert!(should_auto_q_trigger_with_config(
        &message,
        42,
        "helperbot",
        true
    ));
    assert_eq!(
        build_auto_q_query(&message, 42, "helperbot").as_deref(),
        Some("please review this")
    );
}

#[test]
fn auto_q_ignores_other_bot_mentions_when_bot_to_bot_auto_q_disabled() {
    let message = text_message_from(
        1001,
        true,
        "@HelperBot please review this",
        vec![json!({
            "type": "mention",
            "offset": 0,
            "length": 10
        })],
    );

    assert!(!should_auto_q_trigger_with_config(
        &message,
        42,
        "helperbot",
        false
    ));
}

#[test]
fn auto_q_ignores_messages_from_this_bot() {
    let message = text_message_from(
        42,
        true,
        "@HelperBot please review this",
        vec![json!({
            "type": "mention",
            "offset": 0,
            "length": 10
        })],
    );

    assert!(!should_auto_q_trigger_with_config(
        &message,
        42,
        "helperbot",
        true
    ));
}

fn reply_to_this_bot_message(
    sender_id: u64,
    text: &str,
    entities: Vec<serde_json::Value>,
    bot_user_id: u64,
    reply_has_photo: bool,
) -> Message {
    let mut replied = json!({
        "message_id": 7,
        "date": 1,
        "chat": {
            "id": -100123,
            "type": "group",
            "title": "test group"
        },
        "from": {
            "id": bot_user_id,
            "is_bot": true,
            "first_name": "HelperBot",
            "username": "helperbot"
        }
    });
    if reply_has_photo {
        replied["photo"] = json!([{
            "file_id": "photo-file-id",
            "file_unique_id": "photo-unique-id",
            "file_size": 1024,
            "width": 90,
            "height": 90
        }]);
    } else {
        replied["text"] = json!("here is your answer");
    }

    serde_json::from_value(json!({
        "message_id": 42,
        "date": 1,
        "chat": {
            "id": -100123,
            "type": "group",
            "title": "test group"
        },
        "from": {
            "id": sender_id,
            "is_bot": false,
            "first_name": "Human",
            "username": "human_user"
        },
        "text": text,
        "entities": entities,
        "reply_to_message": replied
    }))
    .expect("test reply message should deserialize")
}

#[test]
fn auto_q_triggers_when_replying_to_bot_text_message() {
    let message = reply_to_this_bot_message(1001, "tell me more", vec![], 42, false);

    assert!(should_auto_q_trigger_with_config(
        &message,
        42,
        "helperbot",
        true
    ));
}

#[test]
fn auto_q_skips_reply_to_bot_image_message() {
    let message = reply_to_this_bot_message(1001, "nice picture", vec![], 42, true);

    assert!(!should_auto_q_trigger_with_config(
        &message,
        42,
        "helperbot",
        true
    ));
}

#[test]
fn auto_q_still_triggers_when_mentioning_bot_despite_reply_image() {
    let message = reply_to_this_bot_message(
        1001,
        "@HelperBot describe this image",
        vec![json!({
            "type": "mention",
            "offset": 0,
            "length": 10
        })],
        42,
        true,
    );

    assert!(should_auto_q_trigger_with_config(
        &message,
        42,
        "helperbot",
        true
    ));
}

#[test]
fn codex_selected_model_label_prefers_selected_reasoning_level() {
    let record = crate::llm::runtime_models::CodexSelectedModelRecord {
        metadata_version: crate::llm::runtime_models::CODEX_SELECTED_MODEL_METADATA_VERSION,
        account_id: None,
        slug: "gpt-5.4".to_string(),
        display_name: "GPT-5.4".to_string(),
        description: None,
        input_modalities: vec!["text".to_string()],
        priority: 1,
        etag: None,
        default_reasoning_level: Some("medium".to_string()),
        supported_reasoning_levels: vec![],
        selected_reasoning_level: Some("high".to_string()),
        web_search_tool_type: crate::llm::openai_codex::CodexWebSearchToolType::Text,
        supports_search_tool: false,
        use_responses_lite: false,
        fetched_at: chrono::Utc::now(),
    };

    assert_eq!(codex_selected_model_label(&record), "gpt-5.4 high");
}

#[test]
fn codex_selected_model_label_falls_back_to_default_reasoning_level() {
    let mut record = crate::llm::runtime_models::CodexSelectedModelRecord {
        metadata_version: crate::llm::runtime_models::CODEX_SELECTED_MODEL_METADATA_VERSION,
        account_id: None,
        slug: "gpt-5.4".to_string(),
        display_name: "GPT-5.4".to_string(),
        description: None,
        input_modalities: vec!["text".to_string()],
        priority: 1,
        etag: None,
        default_reasoning_level: Some("medium".to_string()),
        supported_reasoning_levels: vec![],
        selected_reasoning_level: None,
        web_search_tool_type: crate::llm::openai_codex::CodexWebSearchToolType::Text,
        supports_search_tool: false,
        use_responses_lite: false,
        fetched_at: chrono::Utc::now(),
    };

    assert_eq!(codex_selected_model_label(&record), "gpt-5.4 medium");
    record.selected_reasoning_level = Some(String::new());
    assert_eq!(codex_selected_model_label(&record), "gpt-5.4 medium");
}

#[test]
fn codex_quick_result_label_uses_explicit_model_requested_reasoning() {
    let terra_config = model(
        ThirdPartyProvider::OpenAICodex,
        "GPT-5.6-Terra",
        "gpt-5.6-terra",
    );
    let terra_record = crate::llm::runtime_models::CodexSelectedModelRecord {
        metadata_version: crate::llm::runtime_models::CODEX_SELECTED_MODEL_METADATA_VERSION,
        account_id: Some("acct-1".to_string()),
        slug: "gpt-5.6-terra".to_string(),
        display_name: "GPT-5.6-Terra".to_string(),
        description: None,
        input_modalities: vec!["text".to_string()],
        priority: 1,
        etag: Some("etag-1".to_string()),
        default_reasoning_level: Some("medium".to_string()),
        supported_reasoning_levels: vec![crate::llm::openai_codex::CodexReasoningEffortOption {
            effort: "low".to_string(),
            description: "Low effort".to_string(),
        }],
        selected_reasoning_level: Some("medium".to_string()),
        web_search_tool_type: crate::llm::openai_codex::CodexWebSearchToolType::Text,
        supports_search_tool: true,
        use_responses_lite: false,
        fetched_at: chrono::Utc::now(),
    };

    assert_eq!(
        codex_quick_result_label(&terra_config, Some(&terra_record), Some("low")),
        "gpt-5.6-terra low"
    );
}

#[test]
fn codex_quick_result_label_preserves_a_foreign_model_override_without_catalog_metadata() {
    let foreign_config = model(
        ThirdPartyProvider::OpenAICodex,
        "Configured Codex",
        "configured-q-model",
    );

    assert_eq!(
        codex_quick_result_label(&foreign_config, None, Some("low")),
        "configured-q-model low"
    );
}

#[test]
fn quick_reasoning_catalog_fallback_is_reported_for_unsupported_and_empty_overrides() {
    let terra_config = model(
        ThirdPartyProvider::OpenAICodex,
        "GPT-5.6-Terra",
        "gpt-5.6-terra",
    );
    let terra_record = crate::llm::runtime_models::CodexSelectedModelRecord {
        metadata_version: crate::llm::runtime_models::CODEX_SELECTED_MODEL_METADATA_VERSION,
        account_id: Some("acct-1".to_string()),
        slug: "gpt-5.6-terra".to_string(),
        display_name: "GPT-5.6-Terra".to_string(),
        description: None,
        input_modalities: vec!["text".to_string()],
        priority: 1,
        etag: Some("etag-1".to_string()),
        default_reasoning_level: Some("medium".to_string()),
        supported_reasoning_levels: vec![crate::llm::openai_codex::CodexReasoningEffortOption {
            effort: "medium".to_string(),
            description: "Medium effort".to_string(),
        }],
        selected_reasoning_level: None,
        web_search_tool_type: crate::llm::openai_codex::CodexWebSearchToolType::Text,
        supports_search_tool: true,
        use_responses_lite: true,
        fetched_at: chrono::Utc::now(),
    };

    for requested in [Some("low"), Some("  ")] {
        assert_eq!(
            codex_quick_result_label(&terra_config, Some(&terra_record), requested),
            "gpt-5.6-terra medium"
        );
    }
}

#[test]
fn quick_result_label_uses_the_explicit_record_after_runtime_cache_loss() {
    let config = model(
        ThirdPartyProvider::OpenAICodex,
        "GPT-5.6-Terra",
        "gpt-5.6-terra",
    );
    let mut source_record = Some(crate::llm::runtime_models::CodexSelectedModelRecord {
        metadata_version: crate::llm::runtime_models::CODEX_SELECTED_MODEL_METADATA_VERSION,
        account_id: Some("acct-1".to_string()),
        slug: "gpt-5.6-terra".to_string(),
        display_name: "GPT-5.6-Terra".to_string(),
        description: None,
        input_modalities: vec!["text".to_string()],
        priority: 1,
        etag: Some("etag-1".to_string()),
        default_reasoning_level: Some("medium".to_string()),
        supported_reasoning_levels: vec![crate::llm::openai_codex::CodexReasoningEffortOption {
            effort: "medium".to_string(),
            description: "Medium effort".to_string(),
        }],
        selected_reasoning_level: None,
        web_search_tool_type: crate::llm::openai_codex::CodexWebSearchToolType::Text,
        supports_search_tool: true,
        use_responses_lite: true,
        fetched_at: chrono::Utc::now(),
    });
    let explicit = ResolvedExplicitCodexModel {
        config,
        record: source_record.take().expect("record fixture"),
    };
    assert!(
        source_record.is_none(),
        "simulated runtime reload clears metadata"
    );

    assert_eq!(
        result_model_display_name(
            "openai-codex:gpt-5.6-terra",
            None,
            QaCommandMode::Quick,
            Some(&explicit),
        ),
        "gpt-5.6-terra medium"
    );
}

#[test]
fn media_only_prompt_prefers_image_analysis() {
    let summary = MediaSummary {
        total: 1,
        images: 1,
        videos: 0,
        audios: 0,
        documents: 0,
    };

    assert_eq!(
        build_media_only_qa_prompt(&summary).as_deref(),
        Some("Please analyze the attached image(s).")
    );
}

#[test]
fn media_only_prompt_returns_none_without_media() {
    assert_eq!(build_media_only_qa_prompt(&MediaSummary::default()), None);
}

#[test]
fn quick_model_falls_back_once_to_capable_default_model() {
    let models = vec![
        model(ThirdPartyProvider::OpenRouter, "Quick", "quick/model"),
        model(ThirdPartyProvider::OpenAI, "Default", "gpt-default"),
    ];

    let result = resolve_quick_text_model_with_models(
        "openrouter:quick/model",
        "openai:gpt-default",
        &models,
        &[ThirdPartyProvider::OpenAI],
        false,
        ModelRequestCapabilities::default(),
    );

    assert_eq!(result.as_deref(), Ok("openai:gpt-default"));
}

#[test]
fn explicit_codex_quick_model_does_not_fall_back_to_selected_default() {
    let terra = model(
        ThirdPartyProvider::OpenAICodex,
        "GPT-5.6-Terra",
        "gpt-5.6-terra",
    );
    let luna = model(ThirdPartyProvider::OpenAICodex, "GPT-5.6-Luna", "selected");
    let mut models = vec![luna];
    add_explicit_quick_model(&mut models, Some(terra));
    let result = resolve_quick_text_model_with_models(
        "openai-codex:gpt-5.6-terra",
        "openai-codex:selected",
        &models,
        &[ThirdPartyProvider::OpenAICodex],
        false,
        ModelRequestCapabilities::default(),
    );

    assert_eq!(result.as_deref(), Ok("openai-codex:gpt-5.6-terra"));
}

#[test]
fn explicit_codex_slug_collision_with_selected_terra_keeps_the_explicit_id() {
    let explicit_terra = model(
        ThirdPartyProvider::OpenAICodex,
        "GPT-5.6-Terra explicit",
        "gpt-5.6-terra",
    );
    let mut selected_terra = model(
        ThirdPartyProvider::OpenAICodex,
        "GPT-5.6-Terra selected",
        "selected",
    );
    selected_terra.model = "gpt-5.6-terra".to_string();
    let mut models = vec![selected_terra];

    add_explicit_quick_model(&mut models, Some(explicit_terra));
    let result = resolve_quick_text_model_with_models(
        "openai-codex:gpt-5.6-terra",
        MODEL_GEMINI,
        &models,
        &[ThirdPartyProvider::OpenAICodex],
        true,
        ModelRequestCapabilities::default(),
    );

    assert_eq!(result.as_deref(), Ok("openai-codex:gpt-5.6-terra"));
}

#[test]
fn explicit_codex_slug_collision_replaces_a_static_config_with_catalog_capabilities() {
    let mut static_terra = model(
        ThirdPartyProvider::OpenAICodex,
        "Static Terra",
        "gpt-5.6-terra",
    );
    static_terra.image = false;
    let mut catalog_terra = model(
        ThirdPartyProvider::OpenAICodex,
        "Catalog Terra",
        "gpt-5.6-terra",
    );
    catalog_terra.image = true;
    let mut models = vec![static_terra];

    add_explicit_quick_model(&mut models, Some(catalog_terra));
    let result = resolve_quick_text_model_with_models(
        "openai-codex:gpt-5.6-terra",
        MODEL_GEMINI,
        &models,
        &[ThirdPartyProvider::OpenAICodex],
        true,
        ModelRequestCapabilities {
            has_images: true,
            ..ModelRequestCapabilities::default()
        },
    );

    assert_eq!(result.as_deref(), Ok("openai-codex:gpt-5.6-terra"));
}

#[test]
fn explicit_codex_quick_readiness_does_not_require_a_global_selection() {
    let explicit = crate::llm::runtime_models::ResolvedExplicitCodexModel {
        config: model(
            ThirdPartyProvider::OpenAICodex,
            "GPT-5.6-Terra",
            "gpt-5.6-terra",
        ),
        record: crate::llm::runtime_models::CodexSelectedModelRecord {
            metadata_version: crate::llm::runtime_models::CODEX_SELECTED_MODEL_METADATA_VERSION,
            account_id: Some("acct-1".to_string()),
            slug: "gpt-5.6-terra".to_string(),
            display_name: "GPT-5.6-Terra".to_string(),
            description: None,
            input_modalities: vec!["text".to_string()],
            priority: 1,
            etag: Some("etag-1".to_string()),
            default_reasoning_level: Some("medium".to_string()),
            supported_reasoning_levels: vec![
                crate::llm::openai_codex::CodexReasoningEffortOption {
                    effort: "medium".to_string(),
                    description: "Medium effort".to_string(),
                },
            ],
            selected_reasoning_level: None,
            web_search_tool_type: crate::llm::openai_codex::CodexWebSearchToolType::Text,
            supports_search_tool: true,
            use_responses_lite: true,
            fetched_at: chrono::Utc::now(),
        },
    };

    let prepared = resolve_prepared_quick_text_model_with_models(
        "openai-codex:gpt-5.6-terra",
        MODEL_GEMINI,
        &[],
        &[],
        true,
        ModelRequestCapabilities::default(),
        Some(explicit.clone()),
        ExplicitCodexReadiness {
            enabled: true,
            auth_ready: true,
            current_account_id: Some("acct-1"),
        },
    )
    .expect("the validated explicit model should be ready without /codexmodel state");
    assert_eq!(prepared.model_id, "openai-codex:gpt-5.6-terra");
    assert_eq!(
        prepared
            .explicit_codex
            .as_ref()
            .map(|explicit| explicit.record.slug.as_str()),
        Some("gpt-5.6-terra")
    );

    let mismatched = resolve_prepared_quick_text_model_with_models(
        "openai-codex:gpt-5.6-terra",
        MODEL_GEMINI,
        &[],
        &[],
        true,
        ModelRequestCapabilities::default(),
        Some(explicit),
        ExplicitCodexReadiness {
            enabled: true,
            auth_ready: true,
            current_account_id: Some("acct-2"),
        },
    )
    .expect("an account mismatch should use the configured fallback");
    assert_eq!(mismatched.model_id, MODEL_GEMINI);
    assert!(mismatched.explicit_codex.is_none());
}

#[test]
fn missing_explicit_codex_quick_model_falls_back_once_to_selected_default() {
    let luna = model(ThirdPartyProvider::OpenAICodex, "GPT-5.6-Luna", "selected");
    let mut models = vec![luna];
    add_explicit_quick_model(&mut models, None);
    let result = resolve_quick_text_model_with_models(
        "openai-codex:gpt-5.6-terra",
        "openai-codex:selected",
        &models,
        &[ThirdPartyProvider::OpenAICodex],
        false,
        ModelRequestCapabilities::default(),
    );

    assert_eq!(result.as_deref(), Ok("openai-codex:selected"));
}

#[test]
fn quick_model_media_mismatch_falls_back_to_capable_default_model() {
    let mut quick = model(ThirdPartyProvider::OpenRouter, "Quick", "quick/model");
    quick.video = false;
    let mut default = model(ThirdPartyProvider::OpenAI, "Default", "gpt-video");
    default.video = true;

    let result = resolve_quick_text_model_with_models(
        "openrouter:quick/model",
        "openai:gpt-video",
        &[quick, default],
        &[ThirdPartyProvider::OpenRouter, ThirdPartyProvider::OpenAI],
        false,
        ModelRequestCapabilities {
            has_video: true,
            ..ModelRequestCapabilities::default()
        },
    );

    assert_eq!(result.as_deref(), Ok("openai:gpt-video"));
}

#[test]
fn quick_mode_always_skips_model_selection() {
    for (request, gemini_available, third_party_available, runtime_count) in [
        (ModelRequestCapabilities::default(), false, true, 3),
        (
            ModelRequestCapabilities {
                has_video: true,
                ..ModelRequestCapabilities::default()
            },
            true,
            true,
            4,
        ),
        (
            ModelRequestCapabilities {
                has_video: true,
                ..ModelRequestCapabilities::default()
            },
            false,
            true,
            4,
        ),
    ] {
        assert!(should_use_default_model_without_selection(
            QaCommandMode::Quick,
            request,
            false,
            gemini_available,
            third_party_available,
            runtime_count,
            false,
        ));
    }
}

#[test]
fn quick_reasoning_override_is_codex_only() {
    assert_eq!(
        reasoning_override_for_qa_mode(
            QaCommandMode::Quick,
            ThirdPartyProvider::OpenAICodex,
            "low"
        ),
        Some("low")
    );
    assert_eq!(
        reasoning_override_for_qa_mode(QaCommandMode::Quick, ThirdPartyProvider::OpenAI, "low"),
        None
    );
    assert_eq!(
        reasoning_override_for_qa_mode(
            QaCommandMode::Standard,
            ThirdPartyProvider::OpenAICodex,
            "low"
        ),
        None
    );
}

#[test]
fn quick_tool_runtime_is_used_only_for_tool_capable_models() {
    assert!(uses_quick_tool_runtime(QaCommandMode::Quick, true));
    assert!(!uses_quick_tool_runtime(QaCommandMode::Quick, false));
    assert!(!uses_quick_tool_runtime(QaCommandMode::Standard, true));
}

#[test]
fn quick_search_footer_is_conditional_and_deduplicated() {
    let plain = append_quick_search_footer("Answer".to_string(), QaCommandMode::Quick, false);
    assert_eq!(plain, "Answer");

    let searched = append_quick_search_footer("Answer".to_string(), QaCommandMode::Quick, true);
    assert_eq!(searched.matches(QUICK_SEARCH_FOOTER).count(), 1);

    let repeated = append_quick_search_footer(searched, QaCommandMode::Quick, true);
    assert_eq!(repeated.matches(QUICK_SEARCH_FOOTER).count(), 1);

    let standard = append_quick_search_footer("Answer".to_string(), QaCommandMode::Standard, true);
    assert_eq!(standard, "Answer");
}

#[test]
fn audio_request_with_audio_capable_third_party_model_uses_picker() {
    assert!(!should_use_default_model_without_selection(
        QaCommandMode::Standard,
        ModelRequestCapabilities {
            has_audio: true,
            ..ModelRequestCapabilities::default()
        },
        false,
        true,
        true,
        1,
        false,
    ));
}

#[test]
fn bot_query_message_uses_default_model_without_selection() {
    assert!(should_use_default_model_without_selection(
        QaCommandMode::Standard,
        ModelRequestCapabilities::default(),
        false,
        true,
        true,
        2,
        true,
    ));
}

#[test]
fn audio_selection_includes_only_audio_capable_ready_models() {
    let mut audio_model = model(
        ThirdPartyProvider::Nvidia,
        "NVIDIA Nemotron Omni",
        "nemotron-omni",
    );
    audio_model.audio = true;
    let text_model = model(ThirdPartyProvider::Nvidia, "Text Only", "text-only");
    let mut unavailable_audio = model(
        ThirdPartyProvider::OpenRouter,
        "Unavailable Audio",
        "or-audio",
    );
    unavailable_audio.audio = true;
    let models = vec![audio_model, text_model, unavailable_audio];

    let keyboard = create_model_selection_keyboard_with_models(
        &models,
        &[ThirdPartyProvider::Nvidia],
        false,
        "gemini",
        false,
        false,
        true,
        false,
        false,
    );
    let callbacks = keyboard
        .inline_keyboard
        .iter()
        .flatten()
        .filter_map(|button| match &button.kind {
            InlineKeyboardButtonKind::CallbackData(data) => Some(data.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(callbacks, vec!["model_select:nvidia:nemotron-omni"]);
}

#[test]
fn selectable_models_returns_single_audio_model_when_it_is_the_only_option() {
    let mut audio_model = model(
        ThirdPartyProvider::Nvidia,
        "NVIDIA Nemotron Omni",
        "nemotron-omni",
    );
    audio_model.audio = true;
    let text_model = model(ThirdPartyProvider::Nvidia, "Text Only", "text-only");
    let models = vec![audio_model, text_model];

    let model_ids = selectable_model_ids_for_request_with_models(
        &models,
        &[ThirdPartyProvider::Nvidia],
        false,
        false,
        false,
        true,
        false,
        false,
    );

    assert_eq!(model_ids, vec!["nvidia:nemotron-omni"]);
}

#[test]
fn selectable_models_keeps_picker_when_gemini_and_audio_model_are_available() {
    let mut audio_model = model(
        ThirdPartyProvider::Nvidia,
        "NVIDIA Nemotron Omni",
        "nemotron-omni",
    );
    audio_model.audio = true;
    let models = vec![audio_model];

    let model_ids = selectable_model_ids_for_request_with_models(
        &models,
        &[ThirdPartyProvider::Nvidia],
        true,
        false,
        false,
        true,
        false,
        false,
    );

    assert_eq!(
        model_ids,
        vec!["gemini".to_string(), "nvidia:nemotron-omni".to_string()]
    );
}

#[test]
fn model_selection_keyboard_omits_gemini_when_disabled() {
    let keyboard = create_model_selection_keyboard_with_models(
        &[],
        &[],
        false,
        "gemini",
        false,
        false,
        false,
        false,
        false,
    );

    let callbacks = keyboard
        .inline_keyboard
        .iter()
        .flatten()
        .filter_map(|button| match &button.kind {
            InlineKeyboardButtonKind::CallbackData(data) => Some(data.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert!(!callbacks
        .iter()
        .any(|callback| *callback == format!("{}{}", MODEL_CALLBACK_PREFIX, MODEL_GEMINI)));
}

#[test]
fn video_selection_includes_only_video_capable_ready_models() {
    let mut video_model = model(ThirdPartyProvider::Nvidia, "Video Qwen", "qwen-video");
    video_model.video = true;
    let text_model = model(ThirdPartyProvider::Nvidia, "Text Qwen", "qwen-text");
    let mut unavailable_video = model(
        ThirdPartyProvider::OpenRouter,
        "Unavailable Video",
        "or-video",
    );
    unavailable_video.video = true;
    let models = vec![video_model, text_model, unavailable_video];

    let keyboard = create_model_selection_keyboard_with_models(
        &models,
        &[ThirdPartyProvider::Nvidia],
        false,
        "gemini",
        false,
        true,
        false,
        false,
        false,
    );
    let callbacks = keyboard
        .inline_keyboard
        .iter()
        .flatten()
        .filter_map(|button| match &button.kind {
            InlineKeyboardButtonKind::CallbackData(data) => Some(data.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(callbacks, vec!["model_select:nvidia:qwen-video"]);
}

#[test]
fn model_selection_keyboard_compacts_long_third_party_model_callbacks() {
    let long_model = model(
        ThirdPartyProvider::Nvidia,
        "NVIDIA Nemotron 3 Nano Omni",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    );
    let models = vec![long_model.clone()];

    let keyboard = create_model_selection_keyboard_with_models(
        &models,
        &[ThirdPartyProvider::Nvidia],
        false,
        &long_model.id,
        false,
        false,
        false,
        false,
        false,
    );
    let callbacks = keyboard
        .inline_keyboard
        .iter()
        .flatten()
        .filter_map(|button| match &button.kind {
            InlineKeyboardButtonKind::CallbackData(data) => Some(data.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(callbacks.len(), 1);
    let callback = callbacks[0];
    assert!(callback.len() <= TELEGRAM_CALLBACK_DATA_LIMIT);
    assert!(callback.starts_with("model_select:m:"));
    assert!(!callback.contains(long_model.model.as_str()));

    let token = callback.trim_start_matches(MODEL_CALLBACK_PREFIX);
    assert_eq!(
        resolve_model_callback_token_with_models(token, &models).as_deref(),
        Some(long_model.id.as_str())
    );
}

#[test]
fn model_selection_keyboard_puts_default_third_party_model_first() {
    let openrouter = model(ThirdPartyProvider::OpenRouter, "OpenRouter Qwen", "or-qwen");
    let nvidia = model(ThirdPartyProvider::Nvidia, "NVIDIA Qwen", "nv-qwen");
    let models = vec![openrouter, nvidia];

    let keyboard = create_model_selection_keyboard_with_models(
        &models,
        &[ThirdPartyProvider::OpenRouter, ThirdPartyProvider::Nvidia],
        true,
        "nvidia:nv-qwen",
        false,
        false,
        false,
        false,
        false,
    );
    let callbacks = keyboard
        .inline_keyboard
        .iter()
        .flatten()
        .filter_map(|button| match &button.kind {
            InlineKeyboardButtonKind::CallbackData(data) => Some(data.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(
        callbacks.first().copied(),
        Some("model_select:nvidia:nv-qwen")
    );
    assert_eq!(
        callbacks,
        vec![
            "model_select:nvidia:nv-qwen",
            "model_select:gemini",
            "model_select:openrouter:or-qwen",
        ]
    );
}

#[test]
fn video_request_without_capable_model_uses_error_path_predicate() {
    assert!(!video_request_has_capable_model(false, false));
    assert!(video_request_has_capable_model(true, false));
    assert!(video_request_has_capable_model(false, true));
}

#[test]
fn youtube_query_is_text_when_gemini_is_disabled() {
    let query = "watch this https://www.youtube.com/watch?v=dQw4w9WgXcQ";
    let (text, urls) = extract_youtube_urls_for_available_models(query, false);

    assert_eq!(text, query);
    assert!(urls.is_empty());
}

#[test]
fn quick_third_party_youtube_query_preserves_original_url_when_gemini_is_available() {
    let query = "watch this https://www.youtube.com/watch?v=dQw4w9WgXcQ";
    let (text, urls) = prepare_youtube_inputs_for_qa(
        query,
        QaCommandMode::Quick,
        Some("openrouter:quick-model"),
        true,
    );

    assert_eq!(text, query);
    assert!(urls.is_empty());
}

#[test]
fn quick_gemini_youtube_query_keeps_existing_media_extraction() {
    let query = "watch this https://www.youtube.com/watch?v=dQw4w9WgXcQ";
    let (text, urls) =
        prepare_youtube_inputs_for_qa(query, QaCommandMode::Quick, Some(MODEL_GEMINI), true);

    assert!(!text.contains("https://www.youtube.com/watch?v=dQw4w9WgXcQ"));
    assert_eq!(urls, vec!["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]);
}

#[test]
fn chat_search_mode_requires_custom_tools() {
    assert!(QaCommandMode::ChatSearch.requires_custom_tools());
    assert_eq!(qa_mode_label(QaCommandMode::ChatSearch), "chat_search");
}

#[test]
fn chat_search_selection_accepts_wrapped_json() {
    let selection = parse_chat_search_selection(
        "```json\n{\"selected_message_ids\":[42,43],\"note\":\"two hits\"}\n```",
    )
    .expect("wrapped JSON should parse");

    assert_eq!(selection.selected_message_ids, vec![42, 43]);
    assert_eq!(selection.note.as_deref(), Some("two hits"));
}
