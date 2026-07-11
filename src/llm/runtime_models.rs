use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as AsyncMutex;
use tracing::{info, warn};

use crate::config::{
    qualify_third_party_model_id, ThirdPartyModelConfig, ThirdPartyProvider, CONFIG,
};
use crate::llm::openai_codex::{
    self, CodexInputModality, CodexReasoningEffortOption, CodexRemoteModel, CodexWebSearchToolType,
};

pub const OPENAI_CODEX_SELECTED_MODEL_ID: &str = "openai-codex:selected";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodexSelectedModelRecord {
    #[serde(default)]
    pub account_id: Option<String>,
    pub slug: String,
    pub display_name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_modalities: Vec<String>,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub etag: Option<String>,
    #[serde(default)]
    pub default_reasoning_level: Option<String>,
    #[serde(default)]
    pub supported_reasoning_levels: Vec<CodexReasoningEffortOption>,
    #[serde(default)]
    pub selected_reasoning_level: Option<String>,
    #[serde(default)]
    pub web_search_tool_type: CodexWebSearchToolType,
    #[serde(default)]
    pub supports_search_tool: bool,
    pub fetched_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct RuntimeModelsState {
    models: Vec<ThirdPartyModelConfig>,
    models_by_id: HashMap<String, ThirdPartyModelConfig>,
    codex_selected_model: Option<CodexSelectedModelRecord>,
}

static RUNTIME_MODELS: Lazy<RwLock<RuntimeModelsState>> =
    Lazy::new(|| RwLock::new(build_runtime_models_state()));
static CODEX_MODEL_STATE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
static CODEX_MODEL_REFRESH_STATE: Lazy<AsyncMutex<CodexModelRefreshState>> =
    Lazy::new(|| AsyncMutex::new(CodexModelRefreshState::default()));
static CODEX_MODEL_TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
const CODEX_MODEL_REFRESH_FAILURE_BACKOFF: Duration = Duration::from_secs(60);

#[derive(Debug, Default)]
struct CodexModelRefreshState {
    last_failure: Option<CodexModelRefreshFailure>,
}

#[derive(Debug)]
struct CodexModelRefreshFailure {
    account_id: String,
    etag: String,
    attempted_at: Instant,
}

fn normalized_account_id(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn selected_model_matches_account(
    record: &CodexSelectedModelRecord,
    account_id: Option<&str>,
) -> bool {
    normalized_account_id(record.account_id.as_deref()) == normalized_account_id(account_id)
        && normalized_account_id(account_id).is_some()
}

pub fn current_codex_account_id() -> Option<String> {
    openai_codex::auth_summary()
        .account_id
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn effective_codex_reasoning_level(record: &CodexSelectedModelRecord) -> Option<&str> {
    record
        .selected_reasoning_level
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            record
                .default_reasoning_level
                .as_deref()
                .filter(|value| !value.trim().is_empty())
        })
}

pub fn codex_selected_model_label(record: &CodexSelectedModelRecord) -> String {
    let mut label = record.slug.trim().to_string();
    if let Some(level) = effective_codex_reasoning_level(record) {
        label.push(' ');
        label.push_str(level.trim());
    }
    label
}

fn selected_model_path() -> &'static Path {
    Path::new(&CONFIG.openai_codex_model_path)
}

fn load_selected_codex_model_record() -> Option<CodexSelectedModelRecord> {
    let path = selected_model_path();
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) => {
            if err.kind() != std::io::ErrorKind::NotFound {
                warn!(
                    "Failed to read Codex selected model file {}: {}",
                    path.display(),
                    err
                );
            }
            return None;
        }
    };

    match serde_json::from_str::<CodexSelectedModelRecord>(&raw) {
        Ok(record) => Some(record),
        Err(err) => {
            warn!(
                "Failed to parse Codex selected model file {}: {}",
                path.display(),
                err
            );
            None
        }
    }
}

fn dynamic_codex_model_config(record: &CodexSelectedModelRecord) -> ThirdPartyModelConfig {
    let supports_images = record
        .input_modalities
        .iter()
        .any(|value| value.eq_ignore_ascii_case("image"));

    ThirdPartyModelConfig {
        id: OPENAI_CODEX_SELECTED_MODEL_ID.to_string(),
        provider: ThirdPartyProvider::OpenAICodex,
        name: codex_selected_model_label(record),
        model: record.slug.clone(),
        image: supports_images,
        video: false,
        audio: false,
        tools: true,
    }
}

fn build_runtime_models_state() -> RuntimeModelsState {
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
    }
}

pub fn reload_runtime_models() {
    let mut state = RUNTIME_MODELS.write();
    *state = build_runtime_models_state();
}

pub fn runtime_models() -> Vec<ThirdPartyModelConfig> {
    RUNTIME_MODELS.read().models.clone()
}

pub fn runtime_model_count() -> usize {
    RUNTIME_MODELS.read().models.len()
}

pub fn runtime_model_config(model_id: &str) -> Option<ThirdPartyModelConfig> {
    if model_id.trim().eq_ignore_ascii_case("openai-codex") {
        return RUNTIME_MODELS
            .read()
            .models_by_id
            .get(OPENAI_CODEX_SELECTED_MODEL_ID)
            .cloned();
    }

    let state = RUNTIME_MODELS.read();
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

pub fn selected_codex_model_record() -> Option<CodexSelectedModelRecord> {
    let record = RUNTIME_MODELS.read().codex_selected_model.clone()?;
    selected_model_matches_account(&record, current_codex_account_id().as_deref()).then_some(record)
}

fn build_codex_selected_model_record(
    model: &CodexRemoteModel,
    etag: Option<String>,
    account_id: &str,
    previous: Option<&CodexSelectedModelRecord>,
) -> CodexSelectedModelRecord {
    let previous_selection = previous
        .filter(|record| selected_model_matches_account(record, Some(account_id)))
        .and_then(|record| record.selected_reasoning_level.clone())
        .filter(|level| {
            model
                .supported_reasoning_levels
                .iter()
                .any(|option| option.effort == *level)
        });

    CodexSelectedModelRecord {
        account_id: Some(account_id.trim().to_string()),
        slug: model.slug.clone(),
        display_name: model.display_name.clone(),
        description: model.description.clone(),
        input_modalities: model
            .input_modalities
            .iter()
            .map(|modality| match modality {
                CodexInputModality::Text => "text".to_string(),
                CodexInputModality::Image => "image".to_string(),
            })
            .collect(),
        priority: model.priority,
        etag,
        default_reasoning_level: model.default_reasoning_level.clone(),
        supported_reasoning_levels: model.supported_reasoning_levels.clone(),
        selected_reasoning_level: previous_selection,
        web_search_tool_type: model.web_search_tool_type,
        supports_search_tool: model.supports_search_tool,
        fetched_at: Utc::now(),
    }
}

fn write_selected_codex_model_file(record: &CodexSelectedModelRecord) -> Result<()> {
    let path = selected_model_path();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .map(|value| value.to_string_lossy())
        .unwrap_or_else(|| "codex-model".into());
    let sequence = CODEX_MODEL_TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp_path = parent.join(format!(
        ".{file_name}.{}.{}.tmp",
        std::process::id(),
        sequence
    ));
    let contents = serde_json::to_vec_pretty(record)?;

    let result = (|| -> Result<()> {
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp_path)?;
        file.write_all(&contents)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp_path, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

fn save_selected_codex_model_locked(
    record: &CodexSelectedModelRecord,
) -> Result<ThirdPartyModelConfig> {
    let path = selected_model_path();
    write_selected_codex_model_file(record)?;
    info!(
        "Saved selected Codex model {} ({}) to {}",
        record.display_name,
        record.slug,
        path.display()
    );
    reload_runtime_models();
    runtime_model_config(OPENAI_CODEX_SELECTED_MODEL_ID)
        .ok_or_else(|| anyhow::anyhow!("Selected Codex model did not load after save"))
}

fn validate_current_codex_account(expected_account_id: &str) -> Result<()> {
    let expected_account_id = expected_account_id.trim();
    if expected_account_id.is_empty() {
        return Err(anyhow!("Expected Codex account id is empty"));
    }
    let current_account_id = current_codex_account_id()
        .ok_or_else(|| anyhow!("Codex auth token does not include a ChatGPT account id"))?;
    if current_account_id != expected_account_id {
        return Err(anyhow!("The active ChatGPT account changed"));
    }
    Ok(())
}

pub async fn save_codex_model_selection(
    model: &CodexRemoteModel,
    etag: Option<String>,
    expected_account_id: &str,
) -> Result<(CodexSelectedModelRecord, ThirdPartyModelConfig)> {
    openai_codex::with_locked_auth_account(expected_account_id, || {
        let _guard = CODEX_MODEL_STATE_LOCK.lock();
        validate_current_codex_account(expected_account_id)?;
        let previous = selected_codex_model_record();
        let record =
            build_codex_selected_model_record(model, etag, expected_account_id, previous.as_ref());
        let config = save_selected_codex_model_locked(&record)?;
        Ok((record, config))
    })
    .await
}

pub async fn refresh_selected_codex_model_metadata(
    model: &CodexRemoteModel,
    etag: Option<String>,
    expected_account_id: &str,
    expected_model_slug: &str,
) -> Result<CodexSelectedModelRecord> {
    openai_codex::with_locked_auth_account(expected_account_id, || {
        let _guard = CODEX_MODEL_STATE_LOCK.lock();
        validate_current_codex_account(expected_account_id)?;
        let current = selected_codex_model_record()
            .filter(|record| selected_model_matches_account(record, Some(expected_account_id)))
            .filter(|record| record.slug == expected_model_slug)
            .ok_or_else(|| anyhow!("The selected Codex model changed"))?;
        if model.slug != current.slug {
            return Err(anyhow!(
                "The refreshed Codex model does not match the selection"
            ));
        }
        let record =
            build_codex_selected_model_record(model, etag, expected_account_id, Some(&current));
        save_selected_codex_model_locked(&record)?;
        Ok(record)
    })
    .await
}

pub async fn save_selected_codex_reasoning_level(
    level: Option<String>,
    expected_account_id: &str,
    expected_model_slug: &str,
) -> Result<CodexSelectedModelRecord> {
    openai_codex::with_locked_auth_account(expected_account_id, || {
        let _guard = CODEX_MODEL_STATE_LOCK.lock();
        validate_current_codex_account(expected_account_id)?;
        let mut record = selected_codex_model_record()
            .filter(|record| selected_model_matches_account(record, Some(expected_account_id)))
            .filter(|record| record.slug == expected_model_slug)
            .ok_or_else(|| anyhow!("The selected Codex model changed"))?;
        if let Some(level) = level.as_deref() {
            let supported = record
                .supported_reasoning_levels
                .iter()
                .any(|option| option.effort == level);
            if !supported {
                return Err(anyhow!("The selected reasoning level is not supported"));
            }
        }
        record.selected_reasoning_level = level;
        save_selected_codex_model_locked(&record)?;
        Ok(record)
    })
    .await
}

async fn refresh_selected_codex_model_for_etag_once(
    new_etag: &str,
    expected_account_id: &str,
) -> Result<bool> {
    let new_etag = new_etag.trim();
    let Some(record) = selected_codex_model_record()
        .filter(|record| selected_model_matches_account(record, Some(expected_account_id)))
    else {
        return Ok(false);
    };
    if record.etag.as_deref().map(str::trim) == Some(new_etag) {
        return Ok(false);
    }

    let selected_slug = record.slug.clone();
    let list = openai_codex::fetch_models().await?;
    if list.account_id != expected_account_id {
        return Err(anyhow!(
            "The active ChatGPT account changed during model refresh"
        ));
    }
    let model = list
        .models
        .iter()
        .find(|model| model.slug == selected_slug)
        .ok_or_else(|| {
            anyhow!(
                "Selected Codex model '{}' is absent from the refreshed catalog",
                selected_slug
            )
        })?;
    let refreshed_etag = list
        .etag
        .filter(|value| !value.trim().is_empty())
        .or_else(|| Some(new_etag.to_string()));
    refresh_selected_codex_model_metadata(
        model,
        refreshed_etag,
        expected_account_id,
        &selected_slug,
    )
    .await?;
    Ok(true)
}

pub async fn refresh_selected_codex_model_for_etag(
    new_etag: &str,
    expected_account_id: &str,
) -> Result<bool> {
    let new_etag = new_etag.trim();
    let expected_account_id = expected_account_id.trim();
    if new_etag.is_empty() || expected_account_id.is_empty() {
        return Ok(false);
    }

    let mut state = CODEX_MODEL_REFRESH_STATE.lock().await;
    if state.last_failure.as_ref().is_some_and(|failure| {
        failure.account_id == expected_account_id
            && failure.etag == new_etag
            && failure.attempted_at.elapsed() < CODEX_MODEL_REFRESH_FAILURE_BACKOFF
    }) {
        return Ok(false);
    }

    let result = refresh_selected_codex_model_for_etag_once(new_etag, expected_account_id).await;
    match &result {
        Ok(_) => state.last_failure = None,
        Err(_) => {
            state.last_failure = Some(CodexModelRefreshFailure {
                account_id: expected_account_id.to_string(),
                etag: new_etag.to_string(),
                attempted_at: Instant::now(),
            });
        }
    }
    result
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_model_config_maps_image_capability_from_modalities() {
        let record = CodexSelectedModelRecord {
            account_id: Some("acct-1".to_string()),
            slug: "gpt-5.4".to_string(),
            display_name: "GPT-5.4".to_string(),
            description: None,
            input_modalities: vec!["text".to_string(), "image".to_string()],
            priority: 1,
            etag: None,
            default_reasoning_level: Some("medium".to_string()),
            supported_reasoning_levels: vec![CodexReasoningEffortOption {
                effort: "medium".to_string(),
                description: "medium".to_string(),
            }],
            selected_reasoning_level: None,
            web_search_tool_type: CodexWebSearchToolType::Text,
            supports_search_tool: false,
            fetched_at: Utc::now(),
        };

        let config = dynamic_codex_model_config(&record);

        assert_eq!(config.id, OPENAI_CODEX_SELECTED_MODEL_ID);
        assert_eq!(config.provider, ThirdPartyProvider::OpenAICodex);
        assert!(config.image);
        assert!(!config.audio);
        assert!(!config.video);
        assert!(config.tools);
        assert_eq!(config.name, "gpt-5.4 medium");
    }

    #[test]
    fn codex_selected_model_label_prefers_selected_reasoning_level() {
        let record = CodexSelectedModelRecord {
            account_id: Some("acct-1".to_string()),
            slug: "gpt-5.4".to_string(),
            display_name: "GPT-5.4".to_string(),
            description: None,
            input_modalities: vec!["text".to_string()],
            priority: 1,
            etag: None,
            default_reasoning_level: Some("medium".to_string()),
            supported_reasoning_levels: vec![],
            selected_reasoning_level: Some("high".to_string()),
            web_search_tool_type: CodexWebSearchToolType::Text,
            supports_search_tool: false,
            fetched_at: Utc::now(),
        };

        assert_eq!(codex_selected_model_label(&record), "gpt-5.4 high");
    }

    #[test]
    fn selected_model_account_binding_fails_closed() {
        let record = CodexSelectedModelRecord {
            account_id: Some("acct-1".to_string()),
            slug: "gpt-5.4".to_string(),
            display_name: "GPT-5.4".to_string(),
            description: None,
            input_modalities: vec!["text".to_string()],
            priority: 1,
            etag: None,
            default_reasoning_level: None,
            supported_reasoning_levels: vec![],
            selected_reasoning_level: None,
            web_search_tool_type: CodexWebSearchToolType::Text,
            supports_search_tool: false,
            fetched_at: Utc::now(),
        };

        assert!(selected_model_matches_account(&record, Some("acct-1")));
        assert!(!selected_model_matches_account(&record, Some("acct-2")));
        assert!(!selected_model_matches_account(&record, None));
    }

    #[test]
    fn legacy_selected_model_record_deserializes_but_remains_unbound() {
        let raw = r#"{
            "slug":"gpt-5.4",
            "display_name":"GPT-5.4",
            "fetched_at":"2026-07-09T00:00:00Z"
        }"#;

        let record: CodexSelectedModelRecord = serde_json::from_str(raw).unwrap();

        assert_eq!(record.account_id, None);
        assert!(!selected_model_matches_account(&record, Some("acct-1")));
    }
}
