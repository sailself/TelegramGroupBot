//! Resolving which model answers a `/q`-family request: default-model
//! fallback, quick-mode explicit-Codex handling, and selectable-model lists.

use anyhow::{anyhow, Result};
use tracing::warn;

use crate::config::{
    parse_third_party_model_id, ThirdPartyModelConfig, ThirdPartyProvider, CONFIG,
};
use crate::llm::responses_provider::effective_reasoning_effort;
use crate::llm::runtime_models::{
    codex_selected_model_label, ensure_explicit_codex_model, CodexSelectedModelRecord,
    ResolvedExplicitCodexModel, CODEX_SELECTED_MODEL_METADATA_VERSION,
    OPENAI_CODEX_SELECTED_MODEL_ID,
};
use crate::llm::text_model::{
    available_third_party_models_for_request, default_text_model_error,
    normalize_model_identifier_with_models, resolve_default_text_model_with_models,
    ModelRequestCapabilities, MODEL_GEMINI,
};
use crate::state::QaCommandMode;

pub(super) use crate::llm::resolved_model::{ModelCatalogSnapshot, ResolvedTextModel as QaModel};

impl QaModel {
    #[cfg(test)]
    pub(super) fn resolve(
        model_name: &str,
        snapshot: &ModelCatalogSnapshot,
        prepared: Option<&PreparedQuickTextModel>,
    ) -> Result<Self> {
        Self::from_snapshot(
            model_name,
            snapshot,
            prepared.and_then(|prepared| prepared.explicit_codex.as_ref()),
        )
    }
    /// Provider whose reasoning conventions this call follows. Gemini has no
    /// third-party provider; the reasoning override it reports is never read.
    pub fn third_party_provider(&self) -> ThirdPartyProvider {
        match self {
            QaModel::Gemini => ThirdPartyProvider::OpenRouter,
            QaModel::ThirdParty { config, .. } => config.provider,
        }
    }

    pub fn provider_label(&self) -> &'static str {
        match self {
            QaModel::Gemini => "Gemini",
            QaModel::ThirdParty { config, .. } => third_party_provider_label(config.provider),
        }
    }

    /// Label shown before the answer: in progress messages and pickers.
    pub fn display_name(&self, snapshot: &ModelCatalogSnapshot, mode: QaCommandMode) -> String {
        match self {
            QaModel::Gemini => "Gemini".to_string(),
            QaModel::ThirdParty {
                config,
                explicit_codex,
            } => match explicit_codex {
                Some(explicit) if mode == QaCommandMode::Quick => codex_quick_result_label(
                    &explicit.config,
                    Some(&explicit.record),
                    Some(&CONFIG.models.quick_reasoning_effort),
                ),
                Some(explicit) => codex_selected_model_label(&explicit.record),
                None => configured_model_display_name(snapshot, &config.id),
            },
        }
    }

    /// Label shown with the answer and in the request logs: the concrete model
    /// that ran, with the reasoning level when quick mode pinned one.
    pub fn result_display_name(
        &self,
        snapshot: &ModelCatalogSnapshot,
        mode: QaCommandMode,
        gemini_model_used: Option<&str>,
    ) -> String {
        let (config, explicit_codex) = match self {
            QaModel::Gemini => {
                return gemini_model_used
                    .unwrap_or(CONFIG.gemini.model.as_str())
                    .to_string()
            }
            QaModel::ThirdParty {
                config,
                explicit_codex,
            } => (config, explicit_codex),
        };

        if let Some(explicit) = explicit_codex {
            if mode == QaCommandMode::Quick {
                return codex_quick_result_label(
                    &explicit.config,
                    Some(&explicit.record),
                    Some(&CONFIG.models.quick_reasoning_effort),
                );
            }
        }

        if config.provider == ThirdPartyProvider::OpenAICodex {
            if mode == QaCommandMode::Quick {
                let record = snapshot
                    .codex_record
                    .as_ref()
                    .filter(|record| record.slug == config.model);
                return codex_quick_result_label(
                    config,
                    record,
                    Some(&CONFIG.models.quick_reasoning_effort),
                );
            }
            if let Some(record) = snapshot
                .codex_record
                .as_ref()
                .filter(|record| record.slug == config.model)
            {
                return codex_selected_model_label(record);
            }
        }

        config.model.clone()
    }
}

pub(super) fn third_party_provider_label(provider: ThirdPartyProvider) -> &'static str {
    match provider {
        ThirdPartyProvider::OpenRouter => "OpenRouter",
        ThirdPartyProvider::Nvidia => "NVIDIA",
        ThirdPartyProvider::Ollama => "Ollama",
        ThirdPartyProvider::OpenAI => "OpenAI",
        ThirdPartyProvider::OpenAICodex => "OpenAI Codex",
    }
}

pub(super) fn configured_model_display_name(
    snapshot: &ModelCatalogSnapshot,
    model_name: &str,
) -> String {
    if model_name == MODEL_GEMINI {
        "Gemini".to_string()
    } else {
        snapshot
            .config(model_name)
            .map(|config| {
                if config.provider == ThirdPartyProvider::OpenAICodex {
                    if let Some(record) = snapshot
                        .codex_record
                        .as_ref()
                        .filter(|record| record.slug == config.model)
                    {
                        return codex_selected_model_label(record);
                    }
                }
                config.name.clone()
            })
            .unwrap_or_else(|| model_name.to_string())
    }
}

pub(super) fn codex_quick_result_label(
    config: &ThirdPartyModelConfig,
    record: Option<&CodexSelectedModelRecord>,
    requested_effort: Option<&str>,
) -> String {
    effective_reasoning_effort(&config.model, record, requested_effort)
        .map(|effort| format!("{} {effort}", config.model))
        .unwrap_or_else(|| config.model.clone())
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ExplicitCodexReadiness<'a> {
    pub(super) enabled: bool,
    pub(super) auth_ready: bool,
    pub(super) current_account_id: Option<&'a str>,
}

#[derive(Debug, Clone)]
pub(super) struct PreparedQuickTextModel {
    pub(super) model_id: String,
    /// Set when the explicit Codex quick model was chosen, so the request
    /// keeps its config and catalog record even if the runtime catalog reloads.
    pub(super) explicit_codex: Option<ResolvedExplicitCodexModel>,
}

fn explicit_codex_model_is_ready(
    resolved: &ResolvedExplicitCodexModel,
    readiness: ExplicitCodexReadiness<'_>,
) -> bool {
    if !readiness.enabled || !readiness.auth_ready {
        return false;
    }
    let Some(current_account_id) = readiness
        .current_account_id
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return false;
    };
    let Some((provider, slug)) = parse_third_party_model_id(&resolved.config.id) else {
        return false;
    };
    provider == ThirdPartyProvider::OpenAICodex
        && slug != "selected"
        && resolved.config.model == slug
        && resolved.record.slug == slug
        && resolved.record.metadata_version >= CODEX_SELECTED_MODEL_METADATA_VERSION
        && resolved.record.account_id.as_deref().map(str::trim) == Some(current_account_id)
}

#[cfg(test)]
pub(super) fn resolve_quick_text_model_with_models(
    quick_model: &str,
    default_model: &str,
    models: &[ThirdPartyModelConfig],
    ready_providers: &[ThirdPartyProvider],
    gemini_available: bool,
    request: ModelRequestCapabilities,
) -> std::result::Result<String, String> {
    resolve_quick_text_model_with_exact_readiness(
        quick_model,
        default_model,
        models,
        ready_providers,
        gemini_available,
        request,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn resolve_quick_text_model_with_exact_readiness(
    quick_model: &str,
    default_model: &str,
    models: &[ThirdPartyModelConfig],
    ready_providers: &[ThirdPartyProvider],
    gemini_available: bool,
    request: ModelRequestCapabilities,
    exact_readiness_model_id: Option<&str>,
    ready_exact_model_id: Option<&str>,
) -> std::result::Result<String, String> {
    let resolve = |model: &str| {
        let normalized = normalize_model_identifier_with_models(model, models, &[]);
        if exact_readiness_model_id == Some(normalized.as_str())
            && ready_exact_model_id != Some(normalized.as_str())
        {
            return Err(default_text_model_error(&normalized, "unavailable"));
        }

        let mut effective_ready_providers = ready_providers.to_vec();
        if ready_exact_model_id == Some(normalized.as_str()) {
            if let Some(config) = models.iter().find(|config| config.id == normalized) {
                effective_ready_providers.push(config.provider);
            }
        }
        resolve_default_text_model_with_models(
            model,
            models,
            &effective_ready_providers,
            gemini_available,
            request,
        )
    };

    match resolve(quick_model) {
        Ok(model) => Ok(model),
        Err(quick_error) => {
            if quick_model
                .trim()
                .eq_ignore_ascii_case(default_model.trim())
            {
                return Err(quick_error);
            }
            resolve(default_model)
            .map_err(|default_error| {
                format!(
                    "Quick text model fallback failed. Quick model: {quick_error} Default fallback: {default_error}"
                )
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_prepared_quick_text_model_with_models(
    quick_model: &str,
    default_model: &str,
    models: &[ThirdPartyModelConfig],
    ready_providers: &[ThirdPartyProvider],
    gemini_available: bool,
    request: ModelRequestCapabilities,
    explicit: Option<ResolvedExplicitCodexModel>,
    readiness: ExplicitCodexReadiness<'_>,
) -> std::result::Result<PreparedQuickTextModel, String> {
    let explicit_model_id = parse_third_party_model_id(quick_model)
        .filter(|(provider, slug)| {
            *provider == ThirdPartyProvider::OpenAICodex && !slug.eq_ignore_ascii_case("selected")
        })
        .map(|_| quick_model.trim());
    let ready_explicit_model_id = explicit
        .as_ref()
        .filter(|resolved| explicit_codex_model_is_ready(resolved, readiness))
        .map(|resolved| resolved.config.id.as_str());
    let mut models = models.to_vec();
    add_explicit_quick_model(
        &mut models,
        explicit.as_ref().map(|resolved| resolved.config.clone()),
    );
    let model_id = resolve_quick_text_model_with_exact_readiness(
        quick_model,
        default_model,
        &models,
        ready_providers,
        gemini_available,
        request,
        explicit_model_id,
        ready_explicit_model_id,
    )?;
    let explicit_codex = explicit.filter(|resolved| resolved.config.id == model_id);
    Ok(PreparedQuickTextModel {
        model_id,
        explicit_codex,
    })
}

pub(super) fn add_explicit_quick_model(
    models: &mut Vec<ThirdPartyModelConfig>,
    explicit: Option<ThirdPartyModelConfig>,
) {
    if let Some(config) = explicit {
        if let Some(existing) = models.iter_mut().find(|model| model.id == config.id) {
            *existing = config;
        } else {
            models.push(config);
        }
    }
}

pub(super) fn should_use_default_model_without_selection(
    mode: QaCommandMode,
    request: ModelRequestCapabilities,
    has_youtube_urls: bool,
    gemini_available: bool,
    third_party_models_available_for_request: bool,
    runtime_model_count: usize,
    query_message_is_from_bot: bool,
) -> bool {
    mode == QaCommandMode::Quick
        || query_message_is_from_bot
        || request.has_documents
        || (has_youtube_urls && gemini_available)
        || (!request.has_video && !third_party_models_available_for_request)
        || (!request.has_video && runtime_model_count == 0)
}

pub(super) async fn resolve_quick_text_model_for_request(
    snapshot: &ModelCatalogSnapshot,
    request: ModelRequestCapabilities,
) -> Result<PreparedQuickTextModel> {
    let configured_model_id = CONFIG.models.default_quick_text_model.trim();
    let explicit = match parse_third_party_model_id(configured_model_id) {
        Some((ThirdPartyProvider::OpenAICodex, slug)) if !slug.eq_ignore_ascii_case("selected") => {
            match ensure_explicit_codex_model(configured_model_id).await {
                Ok(resolved) => Some(resolved),
                Err(err) => {
                    warn!(
                        configured_model_id,
                        error_chain = %format!("{err:#}"),
                        "Failed to resolve explicit Codex quick model; falling back once"
                    );
                    None
                }
            }
        }
        _ => None,
    };
    let current_account_id = crate::llm::runtime_models::current_codex_account_id();
    resolve_prepared_quick_text_model_with_models(
        &CONFIG.models.default_quick_text_model,
        &CONFIG.models.default_text_model,
        &snapshot.models,
        &snapshot.ready_providers,
        CONFIG.gemini_api_available(),
        request,
        explicit,
        ExplicitCodexReadiness {
            enabled: CONFIG.codex.enabled,
            auth_ready: crate::llm::openai_codex::is_auth_ready(),
            current_account_id: current_account_id.as_deref(),
        },
    )
    .map_err(|message| anyhow!(message))
}

pub(super) fn selectable_model_ids_for_request_with_models(
    snapshot: &ModelCatalogSnapshot,
    gemini_available: bool,
    request: ModelRequestCapabilities,
) -> Vec<String> {
    let mut model_ids = Vec::new();
    if gemini_available {
        model_ids.push(MODEL_GEMINI.to_string());
    }

    model_ids.extend(
        available_third_party_models_for_request(
            &snapshot.models,
            &snapshot.ready_providers,
            request,
        )
        .into_iter()
        .filter_map(|config| {
            let model_identifier = config.id.trim();
            (!model_identifier.is_empty()).then(|| model_identifier.to_string())
        }),
    );

    model_ids
}

pub(super) fn selectable_model_ids_for_request(
    snapshot: &ModelCatalogSnapshot,
    request: ModelRequestCapabilities,
) -> Vec<String> {
    selectable_model_ids_for_request_with_models(snapshot, CONFIG.gemini_api_available(), request)
}

pub(super) fn default_model_selection_key(
    default_model: &str,
    models: &[ThirdPartyModelConfig],
) -> String {
    let trimmed = default_model.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case(MODEL_GEMINI) {
        MODEL_GEMINI.to_string()
    } else if trimmed.eq_ignore_ascii_case("openai-codex") {
        OPENAI_CODEX_SELECTED_MODEL_ID.to_string()
    } else {
        normalize_model_identifier_with_models(trimmed, models, &[])
    }
}

pub(super) fn video_request_has_capable_model(
    gemini_available: bool,
    third_party_video_model_available: bool,
) -> bool {
    gemini_available || third_party_video_model_available
}
