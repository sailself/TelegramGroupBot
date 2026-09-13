//! Who a Codex request is sent as, and the reasoning effort it uses.

use anyhow::{anyhow, Result};
use tracing::warn;

use crate::config::{ThirdPartyModelConfig, ThirdPartyProvider};
use crate::llm::runtime_models::{codex_model_record_for_request, CodexSelectedModelRecord};

/// Who a Codex request is sent as and how it reasons, resolved once per turn
/// from the active login and the catalog. Every request of the turn (tool-loop
/// iterations, retries) reuses it; the only later check is that the login
/// still belongs to `account_id` when the request headers are resolved.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CodexRequestIdentity {
    pub(crate) account_id: String,
    /// Catalog record bound to this account for the requested model: the
    /// selected alias's record, or the explicit record the caller resolved.
    /// `None` for foreign slugs, which carry no metadata.
    pub(crate) record: Option<CodexSelectedModelRecord>,
    pub(crate) reasoning_effort: Option<String>,
    pub(crate) use_responses_lite: bool,
}

impl CodexRequestIdentity {
    /// Resolve against the active login. `explicit_record` is the catalog
    /// record for an explicit Codex slug the caller already resolved (the
    /// Quick path); it must name this model and belong to the active account.
    /// Non-Codex providers have no identity.
    pub(crate) fn resolve(
        model_config: &ThirdPartyModelConfig,
        explicit_record: Option<&CodexSelectedModelRecord>,
        reasoning_override: Option<&str>,
    ) -> Result<Option<Self>> {
        if model_config.provider != ThirdPartyProvider::OpenAICodex {
            return Ok(None);
        }
        let account_id = crate::llm::runtime_models::current_codex_account_id()
            .ok_or_else(|| anyhow!("Codex auth token does not include a ChatGPT account id"))?;
        let selected_record = codex_model_record_for_request(model_config)?;
        Self::resolve_with(
            model_config,
            explicit_record,
            selected_record.as_ref(),
            &account_id,
            reasoning_override,
        )
        .map(Some)
    }

    /// The pure half of [`Self::resolve`]: `selected_record` is the record
    /// behind the `openai-codex:selected` alias when this request uses it.
    pub(super) fn resolve_with(
        model_config: &ThirdPartyModelConfig,
        explicit_record: Option<&CodexSelectedModelRecord>,
        selected_record: Option<&CodexSelectedModelRecord>,
        account_id: &str,
        reasoning_override: Option<&str>,
    ) -> Result<Self> {
        let account_id = account_id.trim();
        if account_id.is_empty() {
            return Err(anyhow!(
                "Codex auth token does not include a ChatGPT account id"
            ));
        }
        let record = match explicit_record.or(selected_record) {
            Some(record) => {
                if record.slug != model_config.model {
                    return Err(anyhow!(
                        "The Codex model metadata does not match the requested model"
                    ));
                }
                if record.account_id.as_deref().map(str::trim) != Some(account_id) {
                    return Err(anyhow!(
                        "The active ChatGPT account changed; retry the request"
                    ));
                }
                Some(record.clone())
            }
            None => None,
        };
        let reasoning_effort =
            effective_reasoning_effort(&model_config.model, record.as_ref(), reasoning_override);
        let use_responses_lite = record
            .as_ref()
            .is_some_and(|record| record.use_responses_lite);
        Ok(Self {
            account_id: account_id.to_string(),
            record,
            reasoning_effort,
            use_responses_lite,
        })
    }
}

/// The `reasoning.effort` to send for `model`: the caller's override when the
/// catalog record supports it (or when no record is known, for foreign slugs),
/// otherwise the operator's selected level, then the catalog default.
pub(crate) fn effective_reasoning_effort(
    model: &str,
    record: Option<&CodexSelectedModelRecord>,
    reasoning_override: Option<&str>,
) -> Option<String> {
    let record = record.filter(|record| record.slug == model);
    let catalog_level = record.and_then(|record| {
        record
            .selected_reasoning_level
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .or_else(|| {
                record
                    .default_reasoning_level
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
            })
            .map(str::to_ascii_lowercase)
    });
    let Some(requested) = reasoning_override
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return catalog_level;
    };

    if let Some(record) = record {
        let supported = record.supported_reasoning_levels.is_empty()
            || record
                .supported_reasoning_levels
                .iter()
                .any(|option| option.effort.eq_ignore_ascii_case(requested));
        if !supported {
            warn!(
                "Reasoning override '{}' is not supported by Codex model '{}'; using the catalog level",
                requested, record.slug
            );
            return catalog_level;
        }
    }

    Some(requested.to_ascii_lowercase())
}
