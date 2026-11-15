use anyhow::{anyhow, Result};
use tracing::warn;

use crate::config::{parse_third_party_model_id, ThirdPartyModelConfig, ThirdPartyProvider};
use crate::llm::audit::LlmAuditContext;
use crate::llm::media::{summarize_media_files, MediaFile, MediaSummary};
use crate::llm::resolved_model::{ModelCatalogSnapshot, ResolvedTextModel};
use crate::llm::runtime_models::{is_runtime_provider_ready, OPENAI_CODEX_SELECTED_MODEL_ID};
use crate::llm::tool_runtime::ToolRuntime;
use crate::llm::{call_gemini, call_third_party_with_reasoning_config, GeminiCallRequest};

pub const MODEL_GEMINI: &str = "gemini";

pub(crate) fn resolve_exact_model_identifier_with_models(
    identifier: &str,
    models: &[ThirdPartyModelConfig],
) -> Option<String> {
    let trimmed = identifier.trim();
    if trimmed.is_empty() {
        return None;
    }

    if trimmed.eq_ignore_ascii_case(MODEL_GEMINI) {
        return Some(MODEL_GEMINI.to_string());
    }

    if let Some((provider, model)) = parse_third_party_model_id(trimmed) {
        let qualified = format!("{}:{}", provider.as_str(), model);
        return models
            .iter()
            .any(|config| config.id == qualified)
            .then_some(qualified);
    }

    let exact_matches = models
        .iter()
        .filter(|config| config.model == trimmed)
        .collect::<Vec<_>>();
    if exact_matches.len() == 1 {
        return Some(exact_matches[0].id.clone());
    }

    None
}

fn resolve_alias_to_model_id_with_models(
    identifier: &str,
    models: &[ThirdPartyModelConfig],
    alias_map: &[(&str, &str)],
) -> Option<String> {
    if let Some(exact) = resolve_exact_model_identifier_with_models(identifier, models) {
        return Some(exact);
    }

    let alias = identifier.trim().to_lowercase();
    if alias.is_empty() {
        return None;
    }
    if alias == MODEL_GEMINI {
        return Some(MODEL_GEMINI.to_string());
    }

    for (token, model) in alias_map {
        if alias == *token && !model.trim().is_empty() {
            return Some((*model).to_string());
        }
    }

    let fuzzy_matches = models
        .iter()
        .filter(|config| {
            let haystack = format!(
                "{} {} {}",
                config.provider.as_str(),
                config.name,
                config.model
            )
            .to_lowercase();
            haystack.contains(&alias)
        })
        .collect::<Vec<_>>();
    if fuzzy_matches.len() == 1 {
        return Some(fuzzy_matches[0].id.clone());
    }

    None
}

pub(crate) fn normalize_model_identifier_with_models(
    identifier: &str,
    models: &[ThirdPartyModelConfig],
    alias_map: &[(&str, &str)],
) -> String {
    let stripped = identifier.trim();
    if stripped.is_empty() {
        return MODEL_GEMINI.to_string();
    }
    if stripped.eq_ignore_ascii_case(MODEL_GEMINI) {
        return MODEL_GEMINI.to_string();
    }

    resolve_alias_to_model_id_with_models(stripped, models, alias_map)
        .unwrap_or_else(|| stripped.to_string())
}

pub(crate) fn ready_runtime_providers(models: &[ThirdPartyModelConfig]) -> Vec<ThirdPartyProvider> {
    models
        .iter()
        .filter(|config| is_runtime_provider_ready(config.provider))
        .map(|config| config.provider)
        .collect()
}

fn is_third_party_model_available_with_ready_providers(
    config: &ThirdPartyModelConfig,
    ready_providers: &[ThirdPartyProvider],
) -> bool {
    ready_providers.contains(&config.provider)
}

pub(crate) fn default_text_model_error(model_name: &str, reason: &str) -> String {
    format!(
        "Default text model {} is {}. Update DEFAULT_TEXT_MODEL or complete Codex setup with /codexlogin and /codexmodel.",
        model_name, reason
    )
}

/// What one request needs from a model: which media kinds it carries, and
/// whether the command it came from needs tool calls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ModelRequestCapabilities {
    pub(crate) has_images: bool,
    pub(crate) has_video: bool,
    pub(crate) has_audio: bool,
    pub(crate) has_documents: bool,
    pub(crate) require_tools: bool,
}

impl ModelRequestCapabilities {
    pub(crate) fn from_media(summary: &MediaSummary, require_tools: bool) -> Self {
        Self {
            has_images: summary.images > 0,
            has_video: summary.videos > 0,
            has_audio: summary.audios > 0,
            has_documents: summary.documents > 0,
            require_tools,
        }
    }
}

pub(crate) fn resolve_default_text_model_with_models(
    default_model: &str,
    models: &[ThirdPartyModelConfig],
    ready_providers: &[ThirdPartyProvider],
    gemini_available: bool,
    request: ModelRequestCapabilities,
) -> std::result::Result<String, String> {
    let trimmed = default_model.trim();
    let normalized = if trimmed.is_empty() || trimmed.eq_ignore_ascii_case(MODEL_GEMINI) {
        MODEL_GEMINI.to_string()
    } else if trimmed.eq_ignore_ascii_case("openai-codex") {
        OPENAI_CODEX_SELECTED_MODEL_ID.to_string()
    } else {
        normalize_model_identifier_with_models(trimmed, models, &[])
    };

    if normalized == MODEL_GEMINI {
        if !gemini_available {
            return Err(default_text_model_error(&normalized, "unavailable"));
        }
        return Ok(normalized);
    }

    if request.has_documents {
        return Err(default_text_model_error(
            &normalized,
            "unsupported for document input",
        ));
    }

    let Some(config) = models.iter().find(|config| config.id == normalized) else {
        return Err(default_text_model_error(&normalized, "not configured"));
    };

    if !ready_providers.contains(&config.provider) {
        return Err(default_text_model_error(&normalized, "unavailable"));
    }

    if !third_party_model_matches_request_capabilities(config, request) {
        return Err(default_text_model_error(
            &normalized,
            "unsupported for this request",
        ));
    }

    Ok(normalized)
}

/// The default text model while Codex is on a quota hold. A default that
/// still resolves is kept. A Codex default cannot run, so Gemini stands in
/// for it, or failing that the first ready third-party model that can serve
/// the request. Any other unavailable default keeps its configuration error.
pub(crate) fn resolve_default_text_model_during_quota_hold(
    default_model: &str,
    models: &[ThirdPartyModelConfig],
    ready_providers: &[ThirdPartyProvider],
    gemini_available: bool,
    request: ModelRequestCapabilities,
) -> std::result::Result<String, String> {
    let configured = resolve_default_text_model_with_models(
        default_model,
        models,
        ready_providers,
        gemini_available,
        request,
    );
    let Err(error) = configured else {
        return configured;
    };
    if !default_text_model_is_codex(default_model, models) {
        return Err(error);
    }
    if gemini_available {
        return Ok(MODEL_GEMINI.to_string());
    }
    available_third_party_models_for_request(models, ready_providers, request)
        .first()
        .map(|config| config.id.clone())
        .ok_or(error)
}

/// Whether the configured default names a Codex model, however it is spelled:
/// the bare provider alias, a qualified id, or a slug the catalog resolves.
pub(crate) fn default_text_model_is_codex(
    default_model: &str,
    models: &[ThirdPartyModelConfig],
) -> bool {
    let trimmed = default_model.trim();
    if trimmed.eq_ignore_ascii_case("openai-codex") {
        return true;
    }
    let normalized = normalize_model_identifier_with_models(trimmed, models, &[]);
    match models.iter().find(|config| config.id == normalized) {
        Some(config) => config.provider == ThirdPartyProvider::OpenAICodex,
        None => matches!(
            parse_third_party_model_id(&normalized),
            Some((ThirdPartyProvider::OpenAICodex, _))
        ),
    }
}

pub(crate) fn third_party_model_matches_request_capabilities(
    config: &ThirdPartyModelConfig,
    request: ModelRequestCapabilities,
) -> bool {
    if request.has_documents {
        return false;
    }
    if request.require_tools && !config.tools {
        return false;
    }
    if request.has_images && !config.image {
        return false;
    }
    if request.has_video && !config.video {
        return false;
    }
    if request.has_audio && !config.audio {
        return false;
    }
    true
}

pub(crate) fn available_third_party_models_for_request<'a>(
    models: &'a [ThirdPartyModelConfig],
    ready_providers: &[ThirdPartyProvider],
    request: ModelRequestCapabilities,
) -> Vec<&'a ThirdPartyModelConfig> {
    models
        .iter()
        .filter(|config| {
            is_third_party_model_available_with_ready_providers(config, ready_providers)
        })
        .filter(|config| third_party_model_matches_request_capabilities(config, request))
        .collect()
}

pub(crate) struct TextCallRequest<'a> {
    pub system_prompt: &'a str,
    pub user_content: &'a str,
    pub response_title: &'a str,
    pub tools_enabled: bool,
    pub use_pro: bool,
    pub media_files: Option<Vec<crate::llm::media::MediaFile>>,
    pub prompt_name: Option<&'a str>,
    pub audit_context: Option<&'a LlmAuditContext>,
}

pub(crate) async fn call_configured_text_model(
    request: TextCallRequest<'_>,
) -> Result<(String, String)> {
    let TextCallRequest {
        system_prompt,
        user_content,
        response_title,
        tools_enabled,
        use_pro,
        media_files,
        prompt_name,
        audit_context,
    } = request;
    let media_summary = media_files
        .as_ref()
        .map(|files| summarize_media_files(files))
        .unwrap_or_default();
    let capabilities = ModelRequestCapabilities::from_media(&media_summary, tools_enabled);
    let mut snapshot = ModelCatalogSnapshot::load();
    let id = snapshot.resolve_default(capabilities)?;
    let model = ResolvedTextModel::prepare(&id, &mut snapshot, None).await?;
    let call = |model: &ResolvedTextModel, media_files: Option<Vec<MediaFile>>| {
        let model = model.clone();
        async move {
            call_resolved_text_model(
                &model,
                TextCallRequest {
                    system_prompt,
                    user_content,
                    response_title,
                    tools_enabled,
                    use_pro,
                    media_files,
                    prompt_name,
                    audit_context,
                },
            )
            .await
        }
    };
    // Only a Codex call can end in a rerun, so only it keeps a copy of the media.
    let rerun_media = model.is_codex().then(|| media_files.clone());
    let error = match call(&model, media_files).await {
        Ok(answer) => return Ok(answer),
        Err(error) => error,
    };
    // Codex refused the request for a spent usage window: answer on the model
    // standing in for Codex rather than failing.
    let Some(stand_in) = model.quota_stand_in(&error, capabilities).await else {
        return Err(error);
    };
    warn!(
        "Codex usage limit reached; rerunning the request on {}",
        stand_in.snapshot.display_name(stand_in.model.model_id())
    );
    call(&stand_in.model, rerun_media.flatten()).await
}

pub(crate) async fn call_resolved_text_model(
    model: &ResolvedTextModel,
    request: TextCallRequest<'_>,
) -> Result<(String, String)> {
    let TextCallRequest {
        system_prompt,
        user_content,
        response_title,
        tools_enabled,
        use_pro,
        media_files,
        prompt_name,
        audit_context,
    } = request;
    if let Some(config) = model.config() {
        let summary = media_files
            .as_ref()
            .map(|files| summarize_media_files(files))
            .unwrap_or_default();
        if !third_party_model_matches_request_capabilities(
            config,
            ModelRequestCapabilities::from_media(&summary, tools_enabled),
        ) {
            return Err(anyhow!(
                "The resolved model cannot satisfy this request's capabilities"
            ));
        }
    }
    let model_name = model.model_id();
    if model_name == MODEL_GEMINI {
        let response = call_gemini(GeminiCallRequest {
            system_prompt,
            user_content,
            use_search_grounding: tools_enabled,
            use_pro_model: use_pro,
            media_files: media_files.unwrap_or_default(),
            youtube_urls: Vec::new(),
            system_prompt_label: prompt_name,
            audit_context,
        })
        .await?;
        let model_used = response.model_used;
        return Ok((response.text, model_used));
    }

    let media_files = media_files.unwrap_or_default();
    let mut web_tools = tools_enabled.then(ToolRuntime::for_web_search);
    let response = call_third_party_with_reasoning_config(
        system_prompt,
        user_content,
        model
            .config()
            .expect("non-Gemini resolved model has a config"),
        response_title,
        &media_files,
        web_tools.as_mut(),
        model.options(crate::llm::ThirdPartyCallOptions::new(
            audit_context,
            crate::llm::CodexPromptStyle::TaskSpecific,
        )),
    )
    .await?;
    let model_used = model.result_label(None);

    Ok((response, model_used))
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn request_capabilities_come_from_the_media_summary() {
        let summary = MediaSummary {
            total: 3,
            images: 2,
            videos: 0,
            audios: 1,
            documents: 0,
        };

        assert_eq!(
            ModelRequestCapabilities::from_media(&summary, true),
            ModelRequestCapabilities {
                has_images: true,
                has_video: false,
                has_audio: true,
                has_documents: false,
                require_tools: true,
            }
        );
        assert_eq!(
            ModelRequestCapabilities::from_media(&MediaSummary::default(), false),
            ModelRequestCapabilities::default()
        );
    }

    #[test]
    fn available_third_party_models_can_require_tools() {
        let mut without_tools = model(
            ThirdPartyProvider::OpenRouter,
            "No Tools",
            "openrouter/no-tools",
        );
        without_tools.tools = false;
        let models = [
            model(
                ThirdPartyProvider::OpenRouter,
                "With Tools",
                "openrouter/with-tools",
            ),
            without_tools,
        ];

        let tool_request = ModelRequestCapabilities {
            require_tools: true,
            ..ModelRequestCapabilities::default()
        };

        assert!(third_party_model_matches_request_capabilities(
            &models[0],
            tool_request
        ));
        assert!(!third_party_model_matches_request_capabilities(
            &models[1],
            tool_request
        ));
    }

    #[test]
    fn normalize_model_identifier_prefers_alias_mapping() {
        let models = vec![
            model(
                ThirdPartyProvider::OpenRouter,
                "Qwen 3",
                "qwen/qwen3-next-80b-a3b-instruct:free",
            ),
            model(
                ThirdPartyProvider::Nvidia,
                "Gemma 3n",
                "google/gemma-3n-e4b-it",
            ),
        ];
        let aliases = [
            ("llama", ""),
            ("grok", ""),
            ("qwen", "openrouter:qwen/qwen3-next-80b-a3b-instruct:free"),
            ("deepseek", ""),
            ("gpt", ""),
        ];

        assert_eq!(
            normalize_model_identifier_with_models("qwen", &models, &aliases),
            "openrouter:qwen/qwen3-next-80b-a3b-instruct:free"
        );
        assert_eq!(
            normalize_model_identifier_with_models("google/gemma-3n-e4b-it", &models, &aliases),
            "nvidia:google/gemma-3n-e4b-it"
        );
    }

    #[test]
    fn normalize_model_identifier_keeps_ambiguous_raw_model_ids_unqualified() {
        let models = vec![
            model(ThirdPartyProvider::OpenRouter, "Shared OR", "shared/model"),
            model(ThirdPartyProvider::Nvidia, "Shared NV", "shared/model"),
        ];
        let aliases = [
            ("llama", ""),
            ("grok", ""),
            ("qwen", ""),
            ("deepseek", ""),
            ("gpt", ""),
        ];

        assert_eq!(
            normalize_model_identifier_with_models("shared/model", &models, &aliases),
            "shared/model"
        );
        assert_eq!(
            normalize_model_identifier_with_models("nvidia:shared/model", &models, &aliases),
            "nvidia:shared/model"
        );
        assert_eq!(
            normalize_model_identifier_with_models("openrouter:shared/model", &models, &aliases),
            "openrouter:shared/model"
        );
    }

    #[test]
    fn default_text_model_resolution_errors_when_codex_is_not_ready() {
        let models = vec![model(
            ThirdPartyProvider::OpenAICodex,
            "Codex Selected",
            "selected",
        )];

        let result = resolve_default_text_model_with_models(
            "openai-codex:selected",
            &models,
            &[],
            true,
            ModelRequestCapabilities::default(),
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Default text model openai-codex:selected is unavailable"));
    }

    /// The catalog during a hold: Codex is configured but not among the ready
    /// providers, and two third-party models follow it.
    fn held_catalog() -> Vec<ThirdPartyModelConfig> {
        let mut no_tools = model(ThirdPartyProvider::Nvidia, "No Tools", "nvidia/no-tools");
        no_tools.tools = false;
        vec![
            model(
                ThirdPartyProvider::OpenAICodex,
                "Codex Selected",
                "selected",
            ),
            no_tools,
            model(
                ThirdPartyProvider::OpenRouter,
                "With Tools",
                "openrouter/with-tools",
            ),
        ]
    }

    const READY_WITHOUT_CODEX: [ThirdPartyProvider; 2] =
        [ThirdPartyProvider::Nvidia, ThirdPartyProvider::OpenRouter];

    #[test]
    fn quota_hold_swaps_a_codex_default_for_gemini() {
        for default_model in ["openai-codex", "openai-codex:selected", "selected"] {
            let result = resolve_default_text_model_during_quota_hold(
                default_model,
                &held_catalog(),
                &READY_WITHOUT_CODEX,
                true,
                ModelRequestCapabilities::default(),
            );
            assert_eq!(result.as_deref(), Ok(MODEL_GEMINI), "{default_model}");
        }
    }

    #[test]
    fn quota_hold_without_gemini_uses_the_first_third_party_model_that_fits() {
        let first = resolve_default_text_model_during_quota_hold(
            "openai-codex",
            &held_catalog(),
            &READY_WITHOUT_CODEX,
            false,
            ModelRequestCapabilities::default(),
        );
        assert_eq!(first.as_deref(), Ok("nvidia:nvidia/no-tools"));

        let with_tools = resolve_default_text_model_during_quota_hold(
            "openai-codex",
            &held_catalog(),
            &READY_WITHOUT_CODEX,
            false,
            ModelRequestCapabilities {
                require_tools: true,
                ..Default::default()
            },
        );
        assert_eq!(
            with_tools.as_deref(),
            Ok("openrouter:openrouter/with-tools")
        );

        let only_openrouter_ready = resolve_default_text_model_during_quota_hold(
            "openai-codex",
            &held_catalog(),
            &[ThirdPartyProvider::OpenRouter],
            false,
            ModelRequestCapabilities::default(),
        );
        assert_eq!(
            only_openrouter_ready.as_deref(),
            Ok("openrouter:openrouter/with-tools")
        );
    }

    #[test]
    fn quota_hold_reports_the_configuration_error_when_nothing_can_stand_in() {
        let nothing_ready = resolve_default_text_model_during_quota_hold(
            "openai-codex",
            &held_catalog(),
            &[],
            false,
            ModelRequestCapabilities::default(),
        );
        assert!(nothing_ready.is_err(), "{nothing_ready:?}");

        // Documents are Gemini-only, so no third-party model stands in.
        let documents = resolve_default_text_model_during_quota_hold(
            "openai-codex",
            &held_catalog(),
            &READY_WITHOUT_CODEX,
            false,
            ModelRequestCapabilities {
                has_documents: true,
                ..Default::default()
            },
        );
        assert!(documents.is_err(), "{documents:?}");
    }

    #[test]
    fn quota_hold_leaves_other_defaults_alone() {
        let ready_default = resolve_default_text_model_during_quota_hold(
            "openrouter:openrouter/with-tools",
            &held_catalog(),
            &READY_WITHOUT_CODEX,
            true,
            ModelRequestCapabilities::default(),
        );
        assert_eq!(
            ready_default.as_deref(),
            Ok("openrouter:openrouter/with-tools")
        );

        // A non-Codex default that is unavailable is a configuration problem
        // the hold must not paper over.
        let misconfigured = resolve_default_text_model_during_quota_hold(
            "openrouter:openrouter/with-tools",
            &held_catalog(),
            &[ThirdPartyProvider::Nvidia],
            true,
            ModelRequestCapabilities::default(),
        );
        assert!(misconfigured
            .unwrap_err()
            .contains("Default text model openrouter:openrouter/with-tools is unavailable"));
    }

    #[test]
    fn default_text_model_resolution_accepts_ready_codex() {
        let models = vec![model(
            ThirdPartyProvider::OpenAICodex,
            "Codex Selected",
            "selected",
        )];

        let result = resolve_default_text_model_with_models(
            "openai-codex:selected",
            &models,
            &[ThirdPartyProvider::OpenAICodex],
            true,
            ModelRequestCapabilities::default(),
        );

        assert_eq!(result.as_deref(), Ok("openai-codex:selected"));
    }

    #[test]
    fn default_text_model_resolution_rejects_gemini_when_disabled() {
        let result = resolve_default_text_model_with_models(
            "gemini",
            &[],
            &[],
            false,
            ModelRequestCapabilities::default(),
        );

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Default text model gemini is unavailable"));
    }
}
