use std::collections::HashMap;

use tracing::warn;

use crate::config::{
    qualify_third_party_model_id, ThirdPartyModelConfig, ThirdPartyProvider, CONFIG,
};
use crate::llm::codex_selected_model::{
    self, dynamic_codex_model_config, load_selected_codex_model_record,
    selected_model_matches_account, RuntimeModelsState,
};

pub use crate::llm::codex_selected_model::{
    codex_model_record_for_request, codex_selected_model_label, current_codex_account_id,
    ensure_explicit_codex_model, ensure_selected_codex_model_metadata_current,
    refresh_selected_codex_model_for_etag, refresh_selected_codex_model_metadata,
    save_codex_model_selection, save_selected_codex_reasoning_level, selected_codex_model_record,
    CodexSelectedModelRecord, ResolvedExplicitCodexModel, CODEX_SELECTED_MODEL_METADATA_VERSION,
    OPENAI_CODEX_SELECTED_MODEL_ID,
};

pub(crate) fn build_runtime_models_state() -> RuntimeModelsState {
    let mut models = CONFIG.third_party_models.clone();
    let stored_codex_selected_model = load_selected_codex_model_record();
    let current_account_id = current_codex_account_id();
    let codex_selected_model = stored_codex_selected_model.filter(|record| {
        let matches = selected_model_matches_account(record, current_account_id.as_deref());
        if !matches {
            warn!(
                "Ignoring stored Codex model selection because it is not bound to the current ChatGPT account; reselect it with /codexmodel"
            );
        }
        matches
    });
    if let Some(record) = codex_selected_model.as_ref() {
        models.push(dynamic_codex_model_config(record));
    }
    let models_by_id = models
        .iter()
        .cloned()
        .map(|model| (model.id.clone(), model))
        .collect::<HashMap<_, _>>();

    RuntimeModelsState {
        models,
        models_by_id,
        codex_selected_model,
        explicit_codex_configs_by_id: HashMap::new(),
        explicit_codex_records_by_id: HashMap::new(),
    }
}

pub fn reload_runtime_models() {
    codex_selected_model::replace_state(build_runtime_models_state());
}

pub fn runtime_models() -> Vec<ThirdPartyModelConfig> {
    codex_selected_model::with_state(|state| state.models.clone())
}

pub fn runtime_model_count() -> usize {
    codex_selected_model::with_state(|state| state.models.len())
}

pub fn runtime_model_config(model_id: &str) -> Option<ThirdPartyModelConfig> {
    if model_id.trim().eq_ignore_ascii_case("openai-codex") {
        return codex_selected_model::with_state(|state| {
            state
                .models_by_id
                .get(OPENAI_CODEX_SELECTED_MODEL_ID)
                .cloned()
        });
    }

    codex_selected_model::with_state(|state| {
        if let Some(model) = state.models_by_id.get(model_id).cloned() {
            return Some(model);
        }

        if let Some((provider, slug)) = crate::config::parse_third_party_model_id(model_id) {
            if provider == ThirdPartyProvider::OpenAICodex {
                if let Some(record) = state.codex_selected_model.as_ref() {
                    if record.slug == slug {
                        return state
                            .models_by_id
                            .get(OPENAI_CODEX_SELECTED_MODEL_ID)
                            .cloned();
                    }
                }
            }
        }

        None
    })
}

pub fn resolve_runtime_model_identifier(identifier: &str) -> Option<String> {
    let trimmed = identifier.trim();
    if trimmed.eq_ignore_ascii_case("openai-codex") {
        return runtime_model_config(OPENAI_CODEX_SELECTED_MODEL_ID)
            .map(|_| qualify_third_party_model_id(ThirdPartyProvider::OpenAICodex, "selected"));
    }

    if let Some((provider, slug)) = crate::config::parse_third_party_model_id(trimmed) {
        if provider == ThirdPartyProvider::OpenAICodex {
            if let Some(record) = selected_codex_model_record() {
                if record.slug == slug {
                    return Some(OPENAI_CODEX_SELECTED_MODEL_ID.to_string());
                }
            }
        }
    }

    None
}

pub fn is_runtime_provider_ready(provider: ThirdPartyProvider) -> bool {
    match provider {
        ThirdPartyProvider::OpenRouter => {
            CONFIG.enable_openrouter && !CONFIG.openrouter_api_key.trim().is_empty()
        }
        ThirdPartyProvider::Nvidia => {
            CONFIG.enable_nvidia && !CONFIG.nvidia_api_key.trim().is_empty()
        }
        ThirdPartyProvider::Ollama => {
            CONFIG.enable_ollama && !CONFIG.ollama_api_key.trim().is_empty()
        }
        ThirdPartyProvider::OpenAI => {
            CONFIG.enable_openai && !CONFIG.openai_api_key.trim().is_empty()
        }
        ThirdPartyProvider::OpenAICodex => {
            CONFIG.enable_openai_codex
                && crate::llm::openai_codex::is_auth_ready()
                && selected_codex_model_record().is_some()
        }
    }
}
