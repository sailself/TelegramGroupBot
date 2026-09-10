//! Admin diagnostics: `/status` and `/diagnose`.

use std::path::Path;

use anyhow::Result;
use chrono::Utc;
use teloxide::prelude::*;
use teloxide::types::ReplyParameters;

use crate::config::{Config, CONFIG};
use crate::handlers::access::check_admin_access;
use crate::llm::openai_codex;
use crate::llm::runtime_models::{runtime_model_count, selected_codex_model_record};
use crate::llm::web_search::is_search_enabled;
use crate::state::AppState;
use crate::utils::logging::read_recent_log_lines;
use crate::utils::text::truncate_with_suffix;

const DIAGNOSE_LOG_TAIL_LINES: usize = 12;
const DIAGNOSE_TEXT_LIMIT: usize = 3900;

fn bool_label(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

/// Every configured credential that must never appear in operator output.
fn config_secret_values(config: &crate::config::Config) -> Vec<String> {
    [
        &config.telegram.bot_token,
        &config.gemini.api_key,
        &config.openrouter.api_key,
        &config.nvidia.api_key,
        &config.ollama.api_key,
        &config.openai.api_key,
        &config.img2.api_key,
        &config.jina.api_key,
        &config.search.brave_api_key,
        &config.search.exa_api_key,
        &config.cwd_pw.api_key,
        &config.telegraph.access_token,
    ]
    .into_iter()
    .map(|value| value.trim().to_string())
    .filter(|value| !value.is_empty())
    .collect()
}

/// Replace every non-empty secret in `secrets` with a placeholder.
fn redact_secrets(text: &str, secrets: &[String]) -> String {
    secrets
        .iter()
        .map(|secret| secret.trim())
        .filter(|secret| !secret.is_empty())
        .fold(text.to_string(), |acc, secret| {
            acc.replace(secret, "[REDACTED]")
        })
}

fn redact_sensitive_text(text: &str) -> String {
    let mut secrets = config_secret_values(&CONFIG);
    secrets.extend(crate::llm::openai_codex::current_auth_secrets());
    redact_secrets(text, &secrets)
}

fn append_log_tail(report: &mut String, base_name: &str, title: &str, max_lines: usize) {
    report.push_str(&format!("\n{title}\n"));
    match read_recent_log_lines(base_name, max_lines) {
        Ok(Some(tail)) => {
            report.push_str(&format!("source: {}\n", tail.path.display()));
            if tail.lines.is_empty() {
                report.push_str("(no lines available)\n");
            } else {
                for line in tail.lines {
                    let line = redact_sensitive_text(&line);
                    report.push_str(&line);
                    report.push('\n');
                }
            }
        }
        Ok(None) => {
            report.push_str("No matching log files found.\n");
        }
        Err(err) => {
            report.push_str(&format!("Failed to read log tail: {err}\n"));
        }
    }
}

fn format_twitter_fetch_diagnostics(config: &Config) -> String {
    format!(
        "twitter_fetch_providers_order: {}\n\
twitter_fetch_total_timeout_secs: {}\n\
twitter_provider_timeout_secs: {}\n\
twitter_response_max_bytes: {}\n\
external_media_max_bytes: {}\n\
external_media_total_max_bytes: {}\n",
        config.twitter.fetch_providers.join(", "),
        config.twitter.fetch_total_timeout_secs,
        config.twitter.provider_timeout_secs,
        config.twitter.response_max_bytes,
        config.external_media.max_bytes,
        config.external_media.total_max_bytes,
    )
}

async fn build_status_report(state: &AppState) -> String {
    let db_result = state.db.health_check().await;
    let db_status = if db_result.is_ok() { "ok" } else { "error" };
    let db_detail = db_result.err().map(|err| err.to_string());

    let queue_max = state.db.queue_max_capacity();
    let queue_pending = state.db.queue_len();
    let queue_available = state.db.queue_available_capacity();
    let heavy_active = state.heavy_command_active();
    let heavy_waiting = state.heavy_command_waiting();
    let media_group_count = state.media_group_count();
    let pending_q_requests = state.pending_q_requests.count();
    let pending_image_requests = state.pending_image_requests.count();
    let pending_codex_model_requests = state.pending_codex_model_requests.count();
    let pending_codex_reasoning_requests = state.pending_codex_reasoning_requests.count();

    let brave_ready = CONFIG.search.enable_brave && !CONFIG.search.brave_api_key.trim().is_empty();
    let exa_ready = CONFIG.search.enable_exa && !CONFIG.search.exa_api_key.trim().is_empty();
    let jina_ready = CONFIG.jina.enable_mcp;
    let openrouter_ready =
        CONFIG.is_third_party_provider_ready(crate::config::ThirdPartyProvider::OpenRouter);
    let nvidia_ready =
        CONFIG.is_third_party_provider_ready(crate::config::ThirdPartyProvider::Nvidia);
    let ollama_ready =
        CONFIG.is_third_party_provider_ready(crate::config::ThirdPartyProvider::Ollama);
    let openai_ready =
        CONFIG.is_third_party_provider_ready(crate::config::ThirdPartyProvider::OpenAI);
    let codex_auth = openai_codex::auth_summary();
    let codex_selected_model = selected_codex_model_record();
    let codex_ready = crate::llm::runtime_models::is_runtime_provider_ready(
        crate::config::ThirdPartyProvider::OpenAICodex,
    );
    let active_codex_login = state.active_codex_login.lock().clone();

    let whitelist_path = Path::new(&CONFIG.access.whitelist_file_path);
    let whitelist_ready = whitelist_path.exists();
    let logs_ready = Path::new("logs").exists();

    let mut report = String::new();
    report.push_str("Status snapshot\n");
    report.push_str(&format!("time_utc: {}\n", Utc::now().to_rfc3339()));
    report.push_str(&format!("db: {db_status}\n"));
    if let Some(detail) = db_detail {
        report.push_str(&format!("db_error: {}\n", detail));
    }
    report.push_str(&format!(
        "db_queue: pending={} available={} max={}\n",
        queue_pending, queue_available, queue_max
    ));
    report.push_str(&format!(
        "db_search_ready: {}\n",
        bool_label(state.db.is_search_ready())
    ));
    report.push_str(&format!(
        "db_max_connections: {}\n",
        CONFIG.db.max_connections
    ));
    report.push_str(&format!(
        "heavy_commands: active={} waiting={} max={}\n",
        heavy_active, heavy_waiting, CONFIG.limits.heavy_command_max_concurrency
    ));
    report.push_str(&format!(
        "pending_requests: q={} image={} codex_model={} codex_reasoning={}\n",
        pending_q_requests,
        pending_image_requests,
        pending_codex_model_requests,
        pending_codex_reasoning_requests
    ));
    report.push_str(&format!("media_groups_cached: {}\n", media_group_count));
    report.push_str(&format!(
        "gemini_configured: {}\n",
        bool_label(!CONFIG.gemini.api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "tldr_infographic_enabled: {}\n",
        bool_label(CONFIG.agents.enable_tldr_infographic)
    ));
    report.push_str(&format!(
        "openrouter_ready: {}\n",
        bool_label(openrouter_ready)
    ));
    report.push_str(&format!("nvidia_ready: {}\n", bool_label(nvidia_ready)));
    report.push_str(&format!("ollama_ready: {}\n", bool_label(ollama_ready)));
    report.push_str(&format!("openai_ready: {}\n", bool_label(openai_ready)));
    report.push_str(&format!(
        "img2_ready: {}\n",
        bool_label(crate::llm::img2_image::img2_available())
    ));
    report.push_str(&format!(
        "img2_health_url: {}\n",
        crate::llm::img2_image::img2_health_url()
    ));
    report.push_str(&format!("img2_media_dir: {}\n", CONFIG.img2.media_dir));
    report.push_str(&format!(
        "openai_codex_ready: {}\n",
        bool_label(codex_ready)
    ));
    report.push_str(&format!(
        "openai_codex_auth_file: {}\n",
        CONFIG.codex.auth_path
    ));
    report.push_str(&format!(
        "openai_codex_auth_present: {}\n",
        bool_label(codex_auth.auth_file_exists)
    ));
    if let Some(auth_mode) = codex_auth.auth_mode {
        report.push_str(&format!("openai_codex_auth_mode: {}\n", auth_mode));
    }
    if let Some(plan_type) = codex_auth.plan_type {
        report.push_str(&format!("openai_codex_plan_type: {}\n", plan_type));
    }
    if let Some(account_id) = codex_auth.account_id {
        report.push_str(&format!("openai_codex_account_id: {}\n", account_id));
    }
    if let Some(email) = codex_auth.email {
        report.push_str(&format!("openai_codex_email: {}\n", email));
    }
    if let Some(last_refresh) = codex_auth.last_refresh {
        report.push_str(&format!(
            "openai_codex_last_refresh: {}\n",
            last_refresh.to_rfc3339()
        ));
    }
    report.push_str(&format!(
        "openai_codex_model_file: {}\n",
        CONFIG.codex.model_path
    ));
    report.push_str(&format!(
        "openai_codex_client_version: {}\n",
        CONFIG.codex.client_version
    ));
    report.push_str(&format!(
        "openai_codex_web_search_mode: {}\n",
        CONFIG.codex.web_search_mode
    ));
    if !CONFIG.codex.web_search_context_size.trim().is_empty() {
        report.push_str(&format!(
            "openai_codex_web_search_context_size: {}\n",
            CONFIG.codex.web_search_context_size
        ));
    }
    if !CONFIG.codex.web_search_allowed_domains.is_empty() {
        report.push_str(&format!(
            "openai_codex_web_search_allowed_domains: {}\n",
            CONFIG.codex.web_search_allowed_domains.join(", ")
        ));
    }
    if let Some(model) = codex_selected_model {
        report.push_str(&format!(
            "openai_codex_selected_model: {} ({})\n",
            model.display_name, model.slug
        ));
        report.push_str(&format!(
            "openai_codex_selected_model_supports_native_search: {}\n",
            bool_label(model.supports_search_tool)
        ));
        if let Some(level) = model.selected_reasoning_level {
            report.push_str(&format!("openai_codex_reasoning_override: {}\n", level));
        } else if let Some(level) = model.default_reasoning_level {
            report.push_str(&format!("openai_codex_reasoning_default: {}\n", level));
        }
    }
    report.push_str(&format!(
        "openai_codex_login_pending: {}\n",
        bool_label(active_codex_login.is_some())
    ));
    if let Some(login) = active_codex_login {
        report.push_str(&format!(
            "openai_codex_login_user_id: {}\n",
            login.admin_user_id
        ));
        report.push_str(&format!("openai_codex_login_chat_id: {}\n", login.chat_id));
        report.push_str(&format!(
            "openai_codex_login_started_at: {}\n",
            login.started_at
        ));
        report.push_str(&format!(
            "openai_codex_login_status_message_id: {}\n",
            login.status_message_id
        ));
    }
    report.push_str(&format!(
        "third_party_models_config_path: {}\n",
        CONFIG.models.third_party_models_config_path.display()
    ));
    report.push_str(&format!(
        "third_party_models_count: {}\n",
        runtime_model_count()
    ));
    report.push_str(&format!(
        "web_search_enabled: {}\n",
        bool_label(is_search_enabled())
    ));
    report.push_str(&format!(
        "web_search_providers_order: {}\n",
        CONFIG.search.providers.join(", ")
    ));
    report.push_str(&format!("brave_ready: {}\n", bool_label(brave_ready)));
    report.push_str(&format!("exa_ready: {}\n", bool_label(exa_ready)));
    report.push_str(&format!("jina_ready: {}\n", bool_label(jina_ready)));
    report.push_str(&format!(
        "whitelist_file: {}\n",
        CONFIG.access.whitelist_file_path
    ));
    report.push_str(&format!(
        "whitelist_present: {}\n",
        bool_label(whitelist_ready)
    ));
    report.push_str(&format!("logs_dir_present: {}\n", bool_label(logs_ready)));
    report
}

async fn build_diagnose_report(state: &AppState) -> String {
    let mut report = String::new();
    report.push_str("Diagnosis report\n");
    report.push_str("Use /status for a compact health view.\n\n");

    let status = build_status_report(state).await;
    report.push_str(&status);

    report.push_str("\n\nConfig checks\n");
    report.push_str(&format!(
        "BOT_TOKEN_present: {}\n",
        bool_label(!CONFIG.telegram.bot_token.trim().is_empty())
    ));
    report.push_str(&format!(
        "GEMINI_API_KEY_present: {}\n",
        bool_label(!CONFIG.gemini.api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "OPENROUTER_API_KEY_present: {}\n",
        bool_label(!CONFIG.openrouter.api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "NVIDIA_API_KEY_present: {}\n",
        bool_label(!CONFIG.nvidia.api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "OLLAMA_API_KEY_present: {}\n",
        bool_label(!CONFIG.ollama.api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "OPENAI_API_KEY_present: {}\n",
        bool_label(!CONFIG.openai.api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "JINA_AI_API_KEY_present: {}\n",
        bool_label(!CONFIG.jina.api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "BRAVE_SEARCH_API_KEY_present: {}\n",
        bool_label(!CONFIG.search.brave_api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "EXA_API_KEY_present: {}\n",
        bool_label(!CONFIG.search.exa_api_key.trim().is_empty())
    ));
    report.push_str(&format!(
        "OPENAI_CODEX_AUTH_FILE_present: {}\n",
        bool_label(Path::new(&CONFIG.codex.auth_path).exists())
    ));
    report.push_str(&format!(
        "OPENAI_CODEX_MODEL_FILE_present: {}\n",
        bool_label(Path::new(&CONFIG.codex.model_path).exists())
    ));
    report.push_str("\nTwitter fetch diagnostics (sanitized)\n");
    report.push_str(&format_twitter_fetch_diagnostics(&CONFIG));

    append_log_tail(
        &mut report,
        "bot.log",
        "Recent bot log lines",
        DIAGNOSE_LOG_TAIL_LINES,
    );
    append_log_tail(
        &mut report,
        "timing.log",
        "Recent timing log lines",
        DIAGNOSE_LOG_TAIL_LINES,
    );

    let report = redact_sensitive_text(&report);
    truncate_with_suffix(
        &report,
        DIAGNOSE_TEXT_LIMIT,
        "\n\n[truncated to fit Telegram message size]",
    )
}

pub async fn status_handler(bot: Bot, state: AppState, message: Message) -> Result<()> {
    if !check_admin_access(&bot, &message, "status").await {
        return Ok(());
    }

    let report = build_status_report(&state).await;
    bot.send_message(message.chat.id, report)
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
    Ok(())
}

pub async fn diagnose_handler(bot: Bot, state: AppState, message: Message) -> Result<()> {
    if !check_admin_access(&bot, &message, "diagnose").await {
        return Ok(());
    }

    let report = build_diagnose_report(&state).await;
    bot.send_message(message.chat.id, report)
        .reply_parameters(ReplyParameters::new(message.id))
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twitter_fetch_diagnostics_are_sanitized_and_complete() {
        let mut config = (*CONFIG).clone();
        config.twitter.fetch_providers = vec![
            "fxtwitter".to_string(),
            "vxtwitter".to_string(),
            "jina".to_string(),
        ];
        config.twitter.fetch_total_timeout_secs = 20;
        config.twitter.provider_timeout_secs = 8;
        config.twitter.response_max_bytes = 2_097_152;
        config.external_media.max_bytes = 20_971_520;
        config.external_media.total_max_bytes = 52_428_800;
        config.jina.api_key = "seeded-jina-secret".to_string();
        config.twitter.fxtwitter_api_base = "https://seeded-fxtwitter.invalid/api".to_string();
        config.twitter.vxtwitter_api_base = "https://seeded-vxtwitter.invalid/api".to_string();
        config.jina.reader_endpoint = "https://seeded-jina.invalid/reader".to_string();
        let report = format_twitter_fetch_diagnostics(&config);

        assert_eq!(
            report,
            "twitter_fetch_providers_order: fxtwitter, vxtwitter, jina\n\
twitter_fetch_total_timeout_secs: 20\n\
twitter_provider_timeout_secs: 8\n\
twitter_response_max_bytes: 2097152\n\
external_media_max_bytes: 20971520\n\
external_media_total_max_bytes: 52428800\n"
        );
        for secret in [
            "JINA_AI_API_KEY",
            "seeded-jina-secret",
            "Bearer seeded-jina-secret",
            "https://seeded-fxtwitter.invalid/api",
            "https://seeded-vxtwitter.invalid/api",
            "https://seeded-jina.invalid/reader",
        ] {
            assert!(!report.contains(secret), "diagnostic leaked {secret}");
        }
    }

    #[test]
    fn redact_secrets_masks_every_non_empty_secret() {
        let secrets = vec!["abc".to_string(), String::new(), "xyz".to_string()];
        assert_eq!(
            redact_secrets("key=abc token=xyz other=abc", &secrets),
            "key=[REDACTED] token=[REDACTED] other=[REDACTED]"
        );
    }

    #[test]
    fn config_secret_values_include_every_provider_credential() {
        let mut config = (*CONFIG).clone();
        config.ollama.api_key = "ollama-secret".to_string();
        config.img2.api_key = "img2-secret".to_string();
        config.telegram.bot_token = "bot-secret".to_string();
        let secrets = config_secret_values(&config);
        for expected in ["ollama-secret", "img2-secret", "bot-secret"] {
            assert!(secrets.iter().any(|s| s == expected), "missing {expected}");
        }
    }
}
