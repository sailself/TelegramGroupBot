//! Catalog snapshots and resolved provider data, shared by handlers and agents.
use crate::config::{
    parse_third_party_model_id, ThirdPartyModelConfig, ThirdPartyProvider, CONFIG,
};
use crate::llm::runtime_models::{
    CodexSelectedModelRecord, ResolvedExplicitCodexModel, OPENAI_CODEX_SELECTED_MODEL_ID,
};
use crate::llm::text_model::{
    available_third_party_models_for_request, default_text_model_error, ready_runtime_providers,
    resolve_default_text_model_with_models, ModelRequestCapabilities, MODEL_GEMINI,
};
use anyhow::{anyhow, Result};

/// The model catalog as one request sees it. Read once per execution. Picker callbacks deliberately revalidate before
/// freezing their execution state.
#[derive(Debug, Clone)]
pub(crate) struct ModelCatalogSnapshot {
    pub models: Vec<ThirdPartyModelConfig>,
    pub ready_providers: Vec<ThirdPartyProvider>,
    pub codex_record: Option<CodexSelectedModelRecord>,
}

impl ModelCatalogSnapshot {
    pub fn load() -> Self {
        let (models, codex_record) = crate::llm::codex_selected_model::with_state(|state| {
            (state.models.clone(), state.codex_selected_model.clone())
        });
        let ready_providers = ready_runtime_providers(&models);
        Self {
            models,
            ready_providers,
            codex_record,
        }
    }

    /// Config for `id`, resolving the same Codex aliases the runtime catalog
    /// does: bare `openai-codex`, and an explicit Codex slug that the selected
    /// record names, both land on the selected-model entry.
    pub fn config(&self, id: &str) -> Option<&ThirdPartyModelConfig> {
        let id = id.trim();
        if id.eq_ignore_ascii_case("openai-codex") {
            return self.selected_codex_config();
        }
        if let Some(config) = self.models.iter().find(|config| config.id == id) {
            return Some(config);
        }
        match parse_third_party_model_id(id) {
            Some((ThirdPartyProvider::OpenAICodex, slug))
                if self
                    .codex_record
                    .as_ref()
                    .is_some_and(|record| record.slug == slug) =>
            {
                self.selected_codex_config()
            }
            _ => None,
        }
    }

    fn selected_codex_config(&self) -> Option<&ThirdPartyModelConfig> {
        self.models
            .iter()
            .find(|config| config.id == OPENAI_CODEX_SELECTED_MODEL_ID)
    }

    pub fn resolve_default(&self, request: ModelRequestCapabilities) -> Result<String> {
        resolve_default_text_model_with_models(
            &CONFIG.models.default_text_model,
            &self.models,
            &self.ready_providers,
            CONFIG.gemini_api_available(),
            request,
        )
        .map_err(|message| anyhow!(message))
    }
    pub fn has_available(&self, request: ModelRequestCapabilities) -> bool {
        !available_third_party_models_for_request(&self.models, &self.ready_providers, request)
            .is_empty()
    }
    pub fn can_select(&self, id: &str, request: ModelRequestCapabilities) -> bool {
        if id == MODEL_GEMINI {
            return CONFIG.gemini_api_available();
        }
        available_third_party_models_for_request(&self.models, &self.ready_providers, request)
            .iter()
            .any(|config| config.id == id)
    }
    pub fn count(&self) -> usize {
        self.models.len()
    }
}

/// The model that answers one `/q`-family request, resolved once against a
/// [`ModelCatalogSnapshot`] so every later step agrees on its capabilities.
#[derive(Debug, Clone)]
pub(crate) enum ResolvedTextModel {
    Gemini,
    ThirdParty {
        config: ThirdPartyModelConfig,
        /// Pinned Codex metadata, for either the selected alias or an explicit slug.
        /// None deliberately means no metadata for this execution.
        explicit_codex: Option<Box<ResolvedExplicitCodexModel>>,
    },
}

impl ResolvedTextModel {
    /// Refresh legacy metadata before freezing execution state. Modern records
    /// require no extra catalog read, and explicit quick models are already prepared.
    pub async fn prepare(
        model_name: &str,
        snapshot: &mut ModelCatalogSnapshot,
        explicit: Option<&ResolvedExplicitCodexModel>,
    ) -> Result<Self> {
        if explicit.is_none()
            && snapshot.codex_record.as_ref().is_some_and(|record| {
                record.metadata_version
                    < crate::llm::runtime_models::CODEX_SELECTED_MODEL_METADATA_VERSION
            })
        {
            if let Some(config) = snapshot
                .config(model_name)
                .filter(|config| config.provider == ThirdPartyProvider::OpenAICodex)
            {
                crate::llm::runtime_models::ensure_selected_codex_model_metadata_current(config)
                    .await?;
                *snapshot = ModelCatalogSnapshot::load();
            }
        }
        Self::from_snapshot(model_name, snapshot, explicit)
    }

    pub fn from_snapshot(
        model_name: &str,
        snapshot: &ModelCatalogSnapshot,
        explicit: Option<&ResolvedExplicitCodexModel>,
    ) -> Result<Self> {
        if model_name == MODEL_GEMINI {
            return Ok(Self::Gemini);
        }

        if let Some(explicit) = explicit {
            if explicit.config.id != model_name {
                return Err(anyhow!("The explicit Codex model changed"));
            }
            return Ok(Self::ThirdParty {
                config: explicit.config.clone(),
                explicit_codex: Some(Box::new(explicit.clone())),
            });
        }

        let config = snapshot
            .config(model_name)
            .ok_or_else(|| anyhow!(default_text_model_error(model_name, "not configured")))?;
        if config.provider == ThirdPartyProvider::OpenAICodex
            && config.id == OPENAI_CODEX_SELECTED_MODEL_ID
            && snapshot
                .codex_record
                .as_ref()
                .is_none_or(|record| record.slug != config.model)
        {
            return Err(anyhow!(
                "The selected Codex model metadata is unavailable; reselect the model"
            ));
        }
        Ok(Self::ThirdParty {
            config: config.clone(),
            explicit_codex: snapshot
                .codex_record
                .as_ref()
                .filter(|record| {
                    config.provider == ThirdPartyProvider::OpenAICodex
                        && record.slug == config.model
                })
                .map(|record| {
                    Box::new(ResolvedExplicitCodexModel {
                        config: config.clone(),
                        record: record.clone(),
                    })
                }),
        })
    }

    pub fn model_id(&self) -> &str {
        match self {
            ResolvedTextModel::Gemini => MODEL_GEMINI,
            ResolvedTextModel::ThirdParty { config, .. } => config.id.as_str(),
        }
    }

    pub fn supports_tools(&self) -> bool {
        match self {
            ResolvedTextModel::Gemini => true,
            ResolvedTextModel::ThirdParty { config, .. } => config.tools,
        }
    }

    pub fn explicit_codex(&self) -> Option<&ResolvedExplicitCodexModel> {
        match self {
            ResolvedTextModel::Gemini => None,
            ResolvedTextModel::ThirdParty { explicit_codex, .. } => explicit_codex.as_deref(),
        }
    }

    pub fn config(&self) -> Option<&ThirdPartyModelConfig> {
        match self {
            Self::Gemini => None,
            Self::ThirdParty { config, .. } => Some(config),
        }
    }
    pub fn options<'a>(
        &'a self,
        options: crate::llm::ThirdPartyCallOptions<'a>,
    ) -> crate::llm::ThirdPartyCallOptions<'a> {
        options
            .with_explicit_codex_model(self.explicit_codex())
            .with_pinned_codex_metadata()
    }
    pub fn result_label(&self, gemini_model: Option<&str>) -> String {
        match self {
            Self::Gemini => gemini_model.unwrap_or(&CONFIG.gemini.model).to_string(),
            Self::ThirdParty {
                config,
                explicit_codex,
            } => explicit_codex
                .as_ref()
                .map(|explicit| {
                    crate::llm::runtime_models::codex_selected_model_label(&explicit.record)
                })
                .unwrap_or_else(|| config.model.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> ThirdPartyModelConfig {
        ThirdPartyModelConfig {
            id: OPENAI_CODEX_SELECTED_MODEL_ID.into(),
            provider: ThirdPartyProvider::OpenAICodex,
            name: "old label".into(),
            model: "old-model".into(),
            image: true,
            video: false,
            audio: false,
            tools: true,
        }
    }
    fn snapshot() -> ModelCatalogSnapshot {
        let record = serde_json::from_value(serde_json::json!({"slug":"old-model", "display_name":"old label", "account_id":"account",
            "selected_reasoning_level":"low", "supports_search_tool":true, "use_responses_lite":true, "fetched_at":"2026-09-12T00:00:00Z"})).unwrap();
        ModelCatalogSnapshot {
            models: vec![config()],
            ready_providers: vec![ThirdPartyProvider::OpenAICodex],
            codex_record: Some(record),
        }
    }
    #[test]
    fn running_model_keeps_configuration_and_metadata_when_catalog_changes() {
        let mut catalog = snapshot();
        let running =
            ResolvedTextModel::from_snapshot(OPENAI_CODEX_SELECTED_MODEL_ID, &catalog, None)
                .unwrap();
        catalog.models[0].model = "new-model".into();
        catalog.models[0].image = false;
        catalog.codex_record.as_mut().unwrap().slug = "new-model".into();
        catalog.codex_record.as_mut().unwrap().use_responses_lite = false;
        let next = ResolvedTextModel::from_snapshot(OPENAI_CODEX_SELECTED_MODEL_ID, &catalog, None)
            .unwrap();
        assert_eq!(running.config().unwrap().model, "old-model");
        assert!(running.config().unwrap().image);
        assert!(running.explicit_codex().unwrap().record.use_responses_lite);
        assert_eq!(next.config().unwrap().model, "new-model");
        assert!(!next.explicit_codex().unwrap().record.use_responses_lite);
        assert!(!catalog.can_select(
            OPENAI_CODEX_SELECTED_MODEL_ID,
            ModelRequestCapabilities {
                has_images: true,
                ..Default::default()
            }
        ));
    }
    #[test]
    fn absent_metadata_remains_absent_in_running_model() {
        let mut catalog = snapshot();
        let record = catalog.codex_record.take();
        catalog.models[0].id = "openai-codex:foreign".into();
        catalog.models[0].model = "foreign".into();
        let running =
            ResolvedTextModel::from_snapshot("openai-codex:foreign", &catalog, None).unwrap();
        catalog.codex_record = record;
        assert!(running.explicit_codex().is_none());
    }
}
