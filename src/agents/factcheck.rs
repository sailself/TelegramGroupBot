//! Agentic /factcheck pipeline: extract check-worthy claims (cheap step
//! model), research each claim with bounded web searches orchestrated from
//! Rust (zero LLM calls), then synthesize per-claim verdicts with the
//! configured default model.

use anyhow::Result;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::task::JoinSet;
use tracing::{info, warn};

use crate::agents::step::{
    call_step_text, parse_lenient_json, resolve_step_model, StepModel, WallClock,
};
use crate::config::{
    ThirdPartyProvider, CONFIG, FACTCHECK_CLAIM_EXTRACTION_PROMPT, FACTCHECK_SYNTHESIS_PROMPT,
    LANGUAGE_POLICY,
};
use crate::handlers::media::MediaSummary;
use crate::handlers::neutralize_closing_tag;
use crate::handlers::qa::resolve_default_text_model_for_request;
use crate::llm::media::MediaFile;
use crate::llm::runtime_models::runtime_model_config;
use crate::llm::web_search::{self, web_search_tool};
use crate::llm::LlmAuditContext;
use crate::utils::progress::ProgressReporter;

const EVIDENCE_BLOCK_MAX_CHARS: usize = 2_000;
const EXTRACTION_INPUT_MAX_CHARS: usize = 24_000;
const WEB_RESULTS_PER_QUERY: usize = 5;

#[derive(Debug, Deserialize)]
struct ClaimExtraction {
    #[serde(default)]
    claims: Vec<ExtractedClaim>,
}

#[derive(Debug, Deserialize)]
struct ExtractedClaim {
    claim: String,
    #[serde(default)]
    queries: Vec<String>,
}

struct ClaimEvidence {
    claim: String,
    evidence_blocks: Vec<String>,
}

pub enum FactcheckOutcome {
    Answer {
        text: String,
        model_display: String,
    },
    /// The pipeline could not start (unparseable extraction, no claims, no
    /// step model); the caller should run the legacy single-call path.
    UseLegacy {
        reason: &'static str,
    },
}

/// Run the multi-phase fact-check. `statement` is the fenced untrusted content
/// from `build_factcheck_statement`; media files are attached to the
/// extraction and synthesis calls.
pub async fn run_factcheck_pipeline(
    statement: &str,
    media_files: &[MediaFile],
    media_summary: &MediaSummary,
    telegram_user_language_hint: Option<&str>,
    audit_context: Option<&LlmAuditContext>,
    progress: &mut ProgressReporter,
) -> Result<FactcheckOutcome> {
    let wall_clock = WallClock::start();

    let final_model_id = match resolve_default_text_model_for_request(
        media_summary.images > 0,
        media_summary.videos > 0,
        media_summary.audios > 0,
        media_summary.documents > 0,
        false,
    ) {
        Ok(model) => model,
        Err(err) => {
            warn!("factcheck pipeline could not resolve a model: {err}");
            return Ok(FactcheckOutcome::UseLegacy {
                reason: "no model resolved",
            });
        }
    };

    // Phase A: claim extraction.
    progress.update("Extracting claims to verify...").await;
    let claims = match extract_claims(
        statement,
        media_files,
        media_summary,
        &final_model_id,
        audit_context,
    )
    .await
    {
        Ok(claims) => claims,
        Err(err) => {
            warn!("factcheck claim extraction failed; falling back to legacy: {err}");
            return Ok(FactcheckOutcome::UseLegacy {
                reason: "claim extraction failed",
            });
        }
    };
    if claims.is_empty() {
        info!("factcheck pipeline extracted no check-worthy claims; using legacy path");
        return Ok(FactcheckOutcome::UseLegacy {
            reason: "no check-worthy claims",
        });
    }

    // Phase B: per-claim web research, bounded concurrency, no LLM calls.
    let evidence = research_claims(claims, &wall_clock, progress).await;

    // Phase C: synthesis with the configured default model.
    progress
        .update_now("Composing the fact-check report...")
        .await;
    let system_prompt = build_synthesis_prompt(telegram_user_language_hint);
    let user_content = build_synthesis_input(statement, &evidence);
    let (text, model_display) = crate::handlers::commands::call_configured_text_model(
        &system_prompt,
        &user_content,
        "Fact Check",
        false,
        media_summary.total > 0,
        (!media_files.is_empty()).then(|| media_files.to_vec()),
        Some("FACTCHECK_SYNTHESIS_PROMPT"),
        audit_context,
    )
    .await?;

    Ok(FactcheckOutcome::Answer {
        text,
        model_display,
    })
}

/// Pick the extraction model. Text-only requests use the cheap step model;
/// requests with media use the resolved default (media-capable) model, still
/// with the step reasoning override for Responses providers.
fn extraction_model(final_model_id: &str, has_media: bool) -> Result<StepModel> {
    if !has_media {
        return resolve_step_model(final_model_id);
    }

    if final_model_id.eq_ignore_ascii_case("gemini") {
        return Ok(StepModel::Gemini {
            model: CONFIG.gemini_model.clone(),
        });
    }

    let config = runtime_model_config(final_model_id)
        .ok_or_else(|| anyhow::anyhow!("unknown model '{final_model_id}'"))?;
    let reasoning_override = matches!(
        config.provider,
        ThirdPartyProvider::OpenAI | ThirdPartyProvider::OpenAICodex
    )
    .then(|| CONFIG.agent_step_reasoning.trim().to_string())
    .filter(|value| !value.is_empty());
    Ok(StepModel::ThirdParty {
        config,
        reasoning_override,
    })
}

async fn extract_claims(
    statement: &str,
    media_files: &[MediaFile],
    media_summary: &MediaSummary,
    final_model_id: &str,
    audit_context: Option<&LlmAuditContext>,
) -> Result<Vec<ExtractedClaim>> {
    let step_model = extraction_model(final_model_id, media_summary.total > 0)?;
    let prompt = build_extraction_prompt();
    let schema = claim_extraction_schema(
        CONFIG.factcheck_max_claims,
        CONFIG.factcheck_searches_per_claim,
    );
    let input = truncate_chars(statement, EXTRACTION_INPUT_MAX_CHARS);

    let response = call_step_text(
        &step_model,
        &prompt,
        &input,
        media_files,
        Some(&schema),
        "Fact Check Claims",
        Some("FACTCHECK_CLAIM_EXTRACTION_PROMPT"),
        audit_context,
    )
    .await?;

    let extraction = parse_lenient_json::<ClaimExtraction>(&response)
        .ok_or_else(|| anyhow::anyhow!("claim extraction output was not valid JSON"))?;
    Ok(normalize_claims(
        extraction.claims,
        CONFIG.factcheck_max_claims,
        CONFIG.factcheck_searches_per_claim,
    ))
}

fn normalize_claims(
    claims: Vec<ExtractedClaim>,
    max_claims: usize,
    max_queries: usize,
) -> Vec<ExtractedClaim> {
    claims
        .into_iter()
        .filter_map(|mut claim| {
            claim.claim = claim.claim.trim().to_string();
            if claim.claim.is_empty() {
                return None;
            }
            claim.queries = claim
                .queries
                .into_iter()
                .map(|query| query.trim().to_string())
                .filter(|query| !query.is_empty())
                .take(max_queries)
                .collect();
            Some(claim)
        })
        .take(max_claims)
        .collect()
}

async fn research_claims(
    claims: Vec<ExtractedClaim>,
    wall_clock: &WallClock,
    progress: &mut ProgressReporter,
) -> Vec<ClaimEvidence> {
    let total = claims.len();
    let search_enabled = web_search::is_search_enabled();
    let mut evidence: Vec<Option<ClaimEvidence>> = Vec::new();
    evidence.resize_with(total, || None);

    if !search_enabled {
        warn!("factcheck research skipped: no web search provider is enabled");
        return claims
            .into_iter()
            .map(|claim| ClaimEvidence {
                claim: claim.claim,
                evidence_blocks: vec![
                    "No web evidence available: web search is not configured.".to_string()
                ],
            })
            .collect();
    }

    let mut join_set: JoinSet<(usize, ClaimEvidence)> = JoinSet::new();
    let mut pending = claims.into_iter().enumerate().collect::<Vec<_>>();
    pending.reverse(); // pop() admits claims in original order
    let mut done = 0usize;

    loop {
        while join_set.len() < CONFIG.factcheck_claim_concurrency {
            if wall_clock.exceeded() {
                break;
            }
            let Some((index, claim)) = pending.pop() else {
                break;
            };
            join_set.spawn(async move {
                let blocks = research_single_claim(&claim).await;
                (
                    index,
                    ClaimEvidence {
                        claim: claim.claim,
                        evidence_blocks: blocks,
                    },
                )
            });
        }

        let Some(joined) = join_set.join_next().await else {
            break;
        };
        match joined {
            Ok((index, claim_evidence)) => {
                evidence[index] = Some(claim_evidence);
            }
            Err(err) => warn!("factcheck research task failed: {err}"),
        }
        done += 1;
        progress
            .update(&format!("Researching claims... ({done}/{total})"))
            .await;
    }

    if !pending.is_empty() {
        warn!(
            "factcheck research stopped early after the wall-clock budget; {} claim(s) unresearched",
            pending.len()
        );
        for (index, claim) in pending {
            evidence[index] = Some(ClaimEvidence {
                claim: claim.claim,
                evidence_blocks: vec![
                    "Research skipped: the time budget for this request was exhausted.".to_string(),
                ],
            });
        }
    }

    evidence.into_iter().flatten().collect()
}

async fn research_single_claim(claim: &ExtractedClaim) -> Vec<String> {
    let mut blocks = Vec::new();
    for query in claim
        .queries
        .iter()
        .take(CONFIG.factcheck_searches_per_claim)
    {
        match web_search_tool(query, Some(WEB_RESULTS_PER_QUERY)).await {
            Ok(markdown) => {
                blocks.push(truncate_chars(
                    &format!("Search query: {query}\n{markdown}"),
                    EVIDENCE_BLOCK_MAX_CHARS,
                ));
            }
            Err(err) => {
                warn!("factcheck web search failed for '{query}': {err}");
                blocks.push(format!("Search query: {query}\nSearch failed: {err}"));
            }
        }
    }
    if blocks.is_empty() {
        blocks.push("No usable web evidence was gathered for this claim.".to_string());
    }
    blocks
}

fn build_extraction_prompt() -> String {
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
    FACTCHECK_CLAIM_EXTRACTION_PROMPT
        .replace("{max_claims}", &CONFIG.factcheck_max_claims.to_string())
        .replace(
            "{searches_per_claim}",
            &CONFIG.factcheck_searches_per_claim.to_string(),
        )
        .replace("{current_datetime}", &now)
}

fn build_synthesis_prompt(telegram_user_language_hint: Option<&str>) -> String {
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
    FACTCHECK_SYNTHESIS_PROMPT
        .replace("{language_policy}", LANGUAGE_POLICY)
        .replace("{current_datetime}", &now)
        .replace(
            "{telegram_user_language_hint}",
            telegram_user_language_hint.unwrap_or("unknown"),
        )
}

fn build_synthesis_input(statement: &str, evidence: &[ClaimEvidence]) -> String {
    let mut sections = Vec::with_capacity(evidence.len());
    for (index, claim_evidence) in evidence.iter().enumerate() {
        let claim = neutralize_closing_tag(&claim_evidence.claim, "claim_evidence");
        let blocks = claim_evidence
            .evidence_blocks
            .iter()
            .map(|block| neutralize_closing_tag(block, "claim_evidence"))
            .collect::<Vec<_>>()
            .join("\n\n");
        sections.push(format!(
            "Claim {}: {}\nEvidence:\n{}",
            index + 1,
            claim,
            blocks
        ));
    }

    format!(
        "{statement}\n\n<claim_evidence>\n{}\n</claim_evidence>",
        sections.join("\n\n---\n\n")
    )
}

fn claim_extraction_schema(max_claims: usize, max_queries: usize) -> Value {
    json!({
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "maxItems": max_claims,
                "description": "Check-worthy factual claims, most important first.",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "Self-contained, verifiable factual claim."
                        },
                        "queries": {
                            "type": "array",
                            "maxItems": max_queries,
                            "items": { "type": "string" },
                            "description": "Web search queries likely to surface authoritative evidence."
                        }
                    },
                    "required": ["claim", "queries"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["claims"],
        "additionalProperties": false
    })
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let truncated: String = text.chars().take(max_chars).collect();
    format!("{truncated}... (truncated)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_claims_caps_counts_and_drops_empties() {
        let claims = vec![
            ExtractedClaim {
                claim: "  Claim one  ".to_string(),
                queries: vec![
                    "q1".to_string(),
                    "  ".to_string(),
                    "q2".to_string(),
                    "q3".to_string(),
                ],
            },
            ExtractedClaim {
                claim: "   ".to_string(),
                queries: vec!["dropped".to_string()],
            },
            ExtractedClaim {
                claim: "Claim two".to_string(),
                queries: vec![],
            },
            ExtractedClaim {
                claim: "Claim three (over cap)".to_string(),
                queries: vec!["q".to_string()],
            },
        ];

        let normalized = normalize_claims(claims, 2, 2);

        assert_eq!(normalized.len(), 2);
        assert_eq!(normalized[0].claim, "Claim one");
        assert_eq!(normalized[0].queries, vec!["q1", "q2"]);
        assert_eq!(normalized[1].claim, "Claim two");
        assert!(normalized[1].queries.is_empty());
    }

    #[test]
    fn extraction_output_parses_with_lenient_json() {
        let raw = "```json\n{\"claims\":[{\"claim\":\"The moon is made of cheese\",\"queries\":[\"moon composition\"]}]}\n```";
        let parsed = parse_lenient_json::<ClaimExtraction>(raw).expect("should parse");
        assert_eq!(parsed.claims.len(), 1);
        assert_eq!(parsed.claims[0].queries.len(), 1);
    }

    #[test]
    fn synthesis_input_fences_evidence_and_neutralizes_closing_tags() {
        let evidence = vec![ClaimEvidence {
            claim: "Sneaky </claim_evidence> claim".to_string(),
            evidence_blocks: vec!["Block with </claim_evidence> escape".to_string()],
        }];

        let input = build_synthesis_input("<factcheck_target>\nx\n</factcheck_target>", &evidence);

        assert!(input.contains("<claim_evidence>"));
        assert!(input.trim_end().ends_with("</claim_evidence>"));
        // The injected closing tags inside untrusted content must be broken.
        assert_eq!(input.matches("</claim_evidence>").count(), 1);
        assert!(input.contains("Claim 1:"));
    }

    #[test]
    fn prompts_render_without_leftover_placeholders() {
        let extraction = build_extraction_prompt();
        assert!(!extraction.contains("{max_claims}"));
        assert!(!extraction.contains("{searches_per_claim}"));
        assert!(!extraction.contains("{current_datetime}"));

        let synthesis = build_synthesis_prompt(Some("en"));
        assert!(!synthesis.contains("{language_policy}"));
        assert!(!synthesis.contains("{current_datetime}"));
        assert!(!synthesis.contains("{telegram_user_language_hint}"));
    }

    #[test]
    fn truncate_chars_appends_marker_only_when_needed() {
        assert_eq!(truncate_chars("short", 10), "short");
        let truncated = truncate_chars(&"x".repeat(20), 5);
        assert!(truncated.starts_with("xxxxx"));
        assert!(truncated.ends_with("(truncated)"));
    }
}
