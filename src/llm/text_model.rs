use anyhow::{anyhow, Result};

use crate::config::{
    parse_third_party_model_id, ThirdPartyModelConfig, ThirdPartyProvider, CONFIG,
};
use crate::llm::audit::LlmAuditContext;
use crate::llm::media::summarize_media_files;
use crate::llm::runtime_models::{
    codex_selected_model_label, is_runtime_provider_ready, resolve_runtime_model_identifier,
    runtime_model_config, runtime_models, selected_codex_model_record,
    OPENAI_CODEX_SELECTED_MODEL_ID,
};
use crate::llm::tool_runtime::ToolRuntime;
use crate::llm::{call_gemini, call_third_party, GeminiCallRequest};

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

pub(crate) fn resolve_alias_to_model_id_with_models(
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

pub(crate) fn resolve_keyword_alias_with_models(
    identifier: &str,
    models: &[ThirdPartyModelConfig],
) -> Option<String> {
    let alias = identifier.trim().to_lowercase();
    let keywords = match alias.as_str() {
        "llama" => &["llama"][..],
        "grok" => &["grok"][..],
        "qwen" => &["qwen"][..],
        "deepseek" => &["deepseek"][..],
        "gpt" => &["gpt"][..],
        _ => return None,
    };

    let matches = models
        .iter()
        .filter(|config| {
            let name = config.name.to_lowercase();
            keywords.iter().all(|keyword| name.contains(keyword))
        })
        .collect::<Vec<_>>();
    if matches.len() == 1 {
        return Some(matches[0].id.clone());
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

pub(crate) fn normalize_model_identifier(identifier: &str) -> String {
    if let Some(resolved) = resolve_runtime_model_identifier(identifier) {
        return resolved;
    }

    let models = runtime_models();
    resolve_keyword_alias_with_models(identifier, &models)
        .unwrap_or_else(|| normalize_model_identifier_with_models(identifier, &models, &[]))
}

pub(crate) fn is_third_party_model_available(config: &ThirdPartyModelConfig) -> bool {
    is_runtime_provider_ready(config.provider)
}

pub(crate) fn ready_runtime_providers(models: &[ThirdPartyModelConfig]) -> Vec<ThirdPartyProvider> {
    models
        .iter()
        .filter(|config| is_runtime_provider_ready(config.provider))
        .map(|config| config.provider)
        .collect()
}

pub(crate) fn is_third_party_model_available_with_ready_providers(
    config: &ThirdPartyModelConfig,
    ready_providers: &[ThirdPartyProvider],
) -> bool {
    ready_providers.contains(&config.provider)
}

pub(crate) fn has_available_third_party_models_for_request(
    has_images: bool,
    has_video: bool,
    has_audio: bool,
    has_documents: bool,
    require_tools: bool,
) -> bool {
    let models = runtime_models();
    let ready_providers = ready_runtime_providers(&models);
    !available_third_party_models_for_request(
        &models,
        &ready_providers,
        has_images,
        has_video,
        has_audio,
        has_documents,
        require_tools,
    )
    .is_empty()
}

pub(crate) fn model_supports_media_for_request(
    model_name: &str,
    has_images: bool,
    has_video: bool,
    has_audio: bool,
    has_documents: bool,
    require_tools: bool,
) -> bool {
    if model_name == MODEL_GEMINI {
        return CONFIG.gemini_api_available();
    }
    if has_documents {
        return false;
    }

    let Some(config) = runtime_model_config(model_name) else {
        return false;
    };
    if !is_third_party_model_available(&config) {
        return false;
    }
    third_party_model_matches_request_capabilities(
        &config,
        has_images,
        has_video,
        has_audio,
        has_documents,
        require_tools,
    )
}

pub(crate) fn default_text_model_error(model_name: &str, reason: &str) -> String {
    format!(
        "Default text model {} is {}. Update DEFAULT_TEXT_MODEL or complete Codex setup with /codexlogin and /codexmodel.",
        model_name, reason
    )
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ModelRequestCapabilities {
    pub(crate) has_images: bool,
    pub(crate) has_video: bool,
    pub(crate) has_audio: bool,
    pub(crate) has_documents: bool,
    pub(crate) require_tools: bool,
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

    if !third_party_model_matches_request_capabilities(
        config,
        request.has_images,
        request.has_video,
        request.has_audio,
        request.has_documents,
        request.require_tools,
    ) {
        return Err(default_text_model_error(
            &normalized,
            "unsupported for this request",
        ));
    }

    Ok(normalized)
}

pub(crate) fn resolve_default_text_model_for_request(
    request: ModelRequestCapabilities,
) -> Result<String> {
    let models = runtime_models();
    let ready_providers = ready_runtime_providers(&models);

    resolve_default_text_model_with_models(
        &CONFIG.default_text_model,
        &models,
        &ready_providers,
        CONFIG.gemini_api_available(),
        request,
    )
    .map_err(|message| anyhow!(message))
}

pub(crate) fn third_party_model_matches_request_capabilities(
    config: &ThirdPartyModelConfig,
    has_images: bool,
    has_video: bool,
    has_audio: bool,
    has_documents: bool,
    require_tools: bool,
) -> bool {
    if has_documents {
        return false;
    }
    if require_tools && !config.tools {
        return false;
    }
    if has_images && !config.image {
        return false;
    }
    if has_video && !config.video {
        return false;
    }
    if has_audio && !config.audio {
        return false;
    }
    true
}

pub(crate) fn available_third_party_models_for_request<'a>(
    models: &'a [ThirdPartyModelConfig],
    ready_providers: &[ThirdPartyProvider],
    has_images: bool,
    has_video: bool,
    has_audio: bool,
    has_documents: bool,
    require_tools: bool,
) -> Vec<&'a ThirdPartyModelConfig> {
    models
        .iter()
        .filter(|config| {
            is_third_party_model_available_with_ready_providers(config, ready_providers)
        })
        .filter(|config| {
            third_party_model_matches_request_capabilities(
                config,
                has_images,
                has_video,
                has_audio,
                has_documents,
                require_tools,
            )
        })
        .collect()
}

pub(crate) fn default_text_model_display_name(
    model_name: &str,
    gemini_model_used: Option<&str>,
) -> String {
    if model_name == MODEL_GEMINI {
        return gemini_model_used
            .unwrap_or(CONFIG.gemini_model.as_str())
            .to_string();
    }

    if let Some(config) = runtime_model_config(model_name) {
        if config.provider == ThirdPartyProvider::OpenAICodex {
            if let Some(record) = selected_codex_model_record() {
                if record.slug == config.model {
                    return codex_selected_model_label(&record);
                }
            }
        }
        return config.model;
    }

    model_name.to_string()
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn call_configured_text_model(
    system_prompt: &str,
    user_content: &str,
    response_title: &str,
    tools_enabled: bool,
    use_pro: bool,
    media_files: Option<Vec<crate::llm::media::MediaFile>>,
    prompt_name: Option<&str>,
    audit_context: Option<&LlmAuditContext>,
) -> Result<(String, String)> {
    let media_summary = media_files
        .as_ref()
        .map(|files| summarize_media_files(files))
        .unwrap_or_default();
    let model_name = resolve_default_text_model_for_request(ModelRequestCapabilities {
        has_images: media_summary.images > 0,
        has_video: media_summary.videos > 0,
        has_audio: media_summary.audios > 0,
        has_documents: media_summary.documents > 0,
        require_tools: tools_enabled,
    })?;

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
    let response = call_third_party(
        system_prompt,
        user_content,
        &model_name,
        response_title,
        &media_files,
        web_tools.as_mut(),
        crate::llm::ThirdPartyCallOptions::new(
            audit_context,
            crate::llm::CodexPromptStyle::TaskSpecific,
        ),
    )
    .await?;
    let model_used = default_text_model_display_name(&model_name, None);

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

        assert!(third_party_model_matches_request_capabilities(
            &models[0], false, false, false, false, true,
        ));
        assert!(!third_party_model_matches_request_capabilities(
            &models[1], false, false, false, false, true,
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
