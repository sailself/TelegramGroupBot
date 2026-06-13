//! Step-model resolution and the single-call primitive used by pipeline
//! phases (claim extraction, query planning, reflection, chunk summaries).

use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use serde::de::DeserializeOwned;
use serde_json::Value;
use tracing::{debug, warn};

use crate::config::{
    parse_third_party_model_id, ThirdPartyModelConfig, ThirdPartyProvider, CONFIG,
};
use crate::llm::call_third_party_with_reasoning_config;
use crate::llm::gemini::call_gemini_model_simple;
use crate::llm::media::MediaFile;
use crate::llm::runtime_models::runtime_model_config;
use crate::llm::LlmAuditContext;

/// Identifier callers use for the built-in Gemini model (mirrors
/// `handlers::qa::MODEL_GEMINI`; kept local so `agents` does not depend on
/// `handlers`).
const GEMINI_MODEL_ID: &str = "gemini";

/// Model used for cheap orchestration steps inside a pipeline.
#[derive(Debug, Clone)]
pub enum StepModel {
    Gemini {
        model: String,
    },
    ThirdParty {
        config: ThirdPartyModelConfig,
        reasoning_override: Option<String>,
    },
}

impl StepModel {
    pub fn display_name(&self) -> String {
        match self {
            StepModel::Gemini { model } => model.clone(),
            StepModel::ThirdParty {
                config,
                reasoning_override,
            } => match reasoning_override {
                Some(level) => format!("{} {}", config.model, level),
                None => config.model.clone(),
            },
        }
    }
}

/// Resolve the step model for a pipeline whose final answer uses
/// `final_model_id`. `AGENT_STEP_MODEL` wins when set; otherwise the step
/// model is derived: a Codex/OpenAI final model runs steps on itself with the
/// `AGENT_STEP_REASONING` per-call override, a Gemini final model uses
/// `GEMINI_LITE_MODEL`, and other providers reuse the final model as-is.
pub fn resolve_step_model(final_model_id: &str) -> Result<StepModel> {
    let step_model = resolve_step_model_value(
        &CONFIG.agent_step_model,
        &CONFIG.agent_step_reasoning,
        final_model_id,
        &CONFIG.gemini_lite_model,
        &CONFIG.gemini_model,
        CONFIG.gemini_api_available(),
        runtime_model_config,
    )?;
    debug!(
        "agent step model resolved: {} (final model {})",
        step_model.display_name(),
        final_model_id
    );
    Ok(step_model)
}

fn resolve_step_model_value(
    agent_step_model: &str,
    agent_step_reasoning: &str,
    final_model_id: &str,
    gemini_lite_model: &str,
    gemini_model: &str,
    gemini_available: bool,
    lookup: impl Fn(&str) -> Option<ThirdPartyModelConfig>,
) -> Result<StepModel> {
    let reasoning = Some(agent_step_reasoning.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string);

    let explicit = agent_step_model.trim();
    if !explicit.is_empty() {
        if explicit.eq_ignore_ascii_case(GEMINI_MODEL_ID) {
            if gemini_available {
                return Ok(StepModel::Gemini {
                    model: pick_gemini_step_model(gemini_lite_model, gemini_model)?,
                });
            }
            warn!("AGENT_STEP_MODEL=gemini but Gemini is unavailable; deriving the step model");
        } else if let Some(config) = lookup(explicit) {
            let reasoning_override = responses_reasoning(&config, reasoning.clone());
            return Ok(StepModel::ThirdParty {
                config,
                reasoning_override,
            });
        } else if let Some((ThirdPartyProvider::OpenAICodex, slug)) =
            parse_third_party_model_id(explicit)
        {
            // Foreign Codex slug (e.g. openai-codex:gpt-5.4-mini): synthesize a
            // config — the backend rejects unknown slugs, and step callers fall
            // back when that happens.
            return Ok(StepModel::ThirdParty {
                config: ThirdPartyModelConfig {
                    id: explicit.to_string(),
                    provider: ThirdPartyProvider::OpenAICodex,
                    name: slug.to_string(),
                    model: slug.to_string(),
                    image: true,
                    video: false,
                    audio: false,
                    tools: false,
                },
                reasoning_override: reasoning,
            });
        } else {
            warn!(
                "AGENT_STEP_MODEL '{}' did not resolve to a runtime model; deriving the step model",
                explicit
            );
        }
    }

    if final_model_id.eq_ignore_ascii_case(GEMINI_MODEL_ID) {
        if gemini_available {
            return Ok(StepModel::Gemini {
                model: pick_gemini_step_model(gemini_lite_model, gemini_model)?,
            });
        }
        return Err(anyhow!(
            "Gemini is the final model but unavailable; no step model"
        ));
    }

    if let Some(config) = lookup(final_model_id) {
        let reasoning_override = responses_reasoning(&config, reasoning);
        return Ok(StepModel::ThirdParty {
            config,
            reasoning_override,
        });
    }

    if gemini_available {
        return Ok(StepModel::Gemini {
            model: pick_gemini_step_model(gemini_lite_model, gemini_model)?,
        });
    }

    Err(anyhow!(
        "No agent step model available for final model '{final_model_id}'"
    ))
}

fn pick_gemini_step_model(lite: &str, standard: &str) -> Result<String> {
    let lite = lite.trim();
    if !lite.is_empty() {
        return Ok(lite.to_string());
    }
    let standard = standard.trim();
    if !standard.is_empty() {
        return Ok(standard.to_string());
    }
    Err(anyhow!("No Gemini model configured for agent steps"))
}

/// Per-call reasoning overrides only mean something to Responses-provider
/// models; other providers get `None` so the request payload is untouched.
fn responses_reasoning(
    config: &ThirdPartyModelConfig,
    reasoning: Option<String>,
) -> Option<String> {
    matches!(
        config.provider,
        ThirdPartyProvider::OpenAI | ThirdPartyProvider::OpenAICodex
    )
    .then_some(reasoning)
    .flatten()
}

fn json_output_instruction(schema: &Value) -> String {
    format!(
        "Respond with ONLY a single valid JSON object matching this JSON Schema — no prose, no Markdown code fences:\n{schema}"
    )
}

/// Run one bounded step call (no tools). With a schema, Gemini gets a native
/// `responseJsonSchema` while Responses/chat providers get a prompt-level JSON
/// instruction; parse the result with [`parse_lenient_json`] either way.
#[allow(clippy::too_many_arguments)]
pub async fn call_step_text(
    step_model: &StepModel,
    system_prompt: &str,
    user_content: &str,
    media_files: &[MediaFile],
    json_schema: Option<&Value>,
    response_title: &str,
    system_prompt_label: Option<&str>,
    audit_context: Option<&LlmAuditContext>,
) -> Result<String> {
    match step_model {
        StepModel::Gemini { model } => {
            let result = call_gemini_model_simple(
                model,
                system_prompt,
                user_content,
                (!media_files.is_empty()).then(|| media_files.to_vec()),
                json_schema,
                system_prompt_label,
                audit_context,
                response_title,
            )
            .await?;
            Ok(result.text)
        }
        StepModel::ThirdParty {
            config,
            reasoning_override,
        } => {
            let system_prompt = match json_schema {
                Some(schema) => {
                    format!("{system_prompt}\n\n{}", json_output_instruction(schema))
                }
                None => system_prompt.to_string(),
            };
            call_third_party_with_reasoning_config(
                &system_prompt,
                user_content,
                config,
                response_title,
                media_files,
                false,
                audit_context,
                reasoning_override.as_deref(),
            )
            .await
        }
    }
}

/// Lenient JSON extraction for model output: direct parse, then with code
/// fences stripped, then the first `{` through the last `}`.
pub fn parse_lenient_json<T: DeserializeOwned>(text: &str) -> Option<T> {
    let trimmed = text.trim();
    if let Ok(value) = serde_json::from_str::<T>(trimmed) {
        return Some(value);
    }

    let unfenced = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|value| value.strip_suffix("```"))
        .map(str::trim);
    if let Some(candidate) = unfenced {
        if let Ok(value) = serde_json::from_str::<T>(candidate) {
            return Some(value);
        }
    }

    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    (start < end)
        .then(|| &trimmed[start..=end])
        .and_then(|candidate| serde_json::from_str::<T>(candidate).ok())
}

/// Wall-clock budget for a whole pipeline run, checked between phases — the
/// pipeline never aborts mid-phase, it just stops starting new work.
pub struct WallClock {
    started: Instant,
    budget: Duration,
}

impl WallClock {
    pub fn start() -> Self {
        Self {
            started: Instant::now(),
            budget: Duration::from_secs(CONFIG.agent_max_wall_clock_secs),
        }
    }

    pub fn exceeded(&self) -> bool {
        self.started.elapsed() >= self.budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Sample {
        answer: String,
    }

    fn config(provider: ThirdPartyProvider, model: &str) -> ThirdPartyModelConfig {
        ThirdPartyModelConfig {
            id: format!("{}:{}", provider.as_str(), model),
            provider,
            name: model.to_string(),
            model: model.to_string(),
            image: false,
            video: false,
            audio: false,
            tools: true,
        }
    }

    #[test]
    fn lenient_json_parses_raw_fenced_and_embedded() {
        let expected = Sample {
            answer: "ok".to_string(),
        };
        assert_eq!(
            parse_lenient_json::<Sample>(r#"{"answer":"ok"}"#).as_ref(),
            Some(&expected)
        );
        assert_eq!(
            parse_lenient_json::<Sample>("```json\n{\"answer\":\"ok\"}\n```").as_ref(),
            Some(&expected)
        );
        assert_eq!(
            parse_lenient_json::<Sample>("Here you go: {\"answer\":\"ok\"} hope it helps").as_ref(),
            Some(&expected)
        );
        assert_eq!(parse_lenient_json::<Sample>("no json here"), None);
        assert_eq!(parse_lenient_json::<Sample>("{broken"), None);
    }

    #[test]
    fn derives_codex_step_model_with_low_reasoning() {
        let codex = config(ThirdPartyProvider::OpenAICodex, "gpt-5.5");
        let resolved = resolve_step_model_value(
            "",
            "low",
            "openai-codex:selected",
            "gemini-flash-lite-latest",
            "gemini-flash-latest",
            true,
            |id| (id == "openai-codex:selected").then(|| codex.clone()),
        )
        .expect("step model should resolve");

        match resolved {
            StepModel::ThirdParty {
                config,
                reasoning_override,
            } => {
                assert_eq!(config.model, "gpt-5.5");
                assert_eq!(reasoning_override.as_deref(), Some("low"));
            }
            other => panic!("unexpected step model: {other:?}"),
        }
    }

    #[test]
    fn derives_gemini_lite_for_gemini_final_model() {
        let resolved = resolve_step_model_value(
            "",
            "low",
            "gemini",
            "gemini-flash-lite-latest",
            "gemini-flash-latest",
            true,
            |_| None,
        )
        .expect("step model should resolve");

        match resolved {
            StepModel::Gemini { model } => assert_eq!(model, "gemini-flash-lite-latest"),
            other => panic!("unexpected step model: {other:?}"),
        }
    }

    #[test]
    fn non_responses_providers_get_no_reasoning_override() {
        let nvidia = config(ThirdPartyProvider::Nvidia, "qwen/qwen3.5-397b-a17b");
        let resolved = resolve_step_model_value(
            "",
            "low",
            "nvidia:qwen/qwen3.5-397b-a17b",
            "",
            "",
            false,
            |id| (id == "nvidia:qwen/qwen3.5-397b-a17b").then(|| nvidia.clone()),
        )
        .expect("step model should resolve");

        match resolved {
            StepModel::ThirdParty {
                reasoning_override, ..
            } => assert_eq!(reasoning_override, None),
            other => panic!("unexpected step model: {other:?}"),
        }
    }

    #[test]
    fn explicit_foreign_codex_slug_synthesizes_config() {
        let resolved = resolve_step_model_value(
            "openai-codex:gpt-5.4-mini",
            "low",
            "openai-codex:selected",
            "",
            "",
            false,
            |_| None,
        )
        .expect("step model should resolve");

        match resolved {
            StepModel::ThirdParty {
                config,
                reasoning_override,
            } => {
                assert_eq!(config.provider, ThirdPartyProvider::OpenAICodex);
                assert_eq!(config.model, "gpt-5.4-mini");
                assert_eq!(reasoning_override.as_deref(), Some("low"));
            }
            other => panic!("unexpected step model: {other:?}"),
        }
    }

    #[test]
    fn unresolvable_setup_errors_out() {
        let result =
            resolve_step_model_value("", "low", "openrouter:not-loaded", "", "", false, |_| None);
        assert!(result.is_err());
    }
}
