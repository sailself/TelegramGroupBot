use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Deserialize;
use std::sync::LazyLock;
use tracing::{info, warn};

pub use crate::prompts::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
pub enum ThirdPartyProvider {
    #[serde(rename = "openrouter")]
    OpenRouter,
    #[serde(rename = "nvidia")]
    Nvidia,
    #[serde(rename = "ollama")]
    Ollama,
    #[serde(rename = "openai")]
    OpenAI,
    #[serde(rename = "openai-codex")]
    OpenAICodex,
}

impl ThirdPartyProvider {
    pub fn as_str(&self) -> &'static str {
        match self {
            ThirdPartyProvider::OpenRouter => "openrouter",
            ThirdPartyProvider::Nvidia => "nvidia",
            ThirdPartyProvider::Ollama => "ollama",
            ThirdPartyProvider::OpenAI => "openai",
            ThirdPartyProvider::OpenAICodex => "openai-codex",
        }
    }
}

impl std::str::FromStr for ThirdPartyProvider {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_lowercase().as_str() {
            "openrouter" => Ok(ThirdPartyProvider::OpenRouter),
            "nvidia" => Ok(ThirdPartyProvider::Nvidia),
            "ollama" => Ok(ThirdPartyProvider::Ollama),
            "openai" => Ok(ThirdPartyProvider::OpenAI),
            "openai-codex" => Ok(ThirdPartyProvider::OpenAICodex),
            other => Err(anyhow::anyhow!(
                "Unsupported third-party model provider '{}'",
                other
            )),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ThirdPartyModelsFile {
    models: Vec<ThirdPartyModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct ThirdPartyModelEntry {
    provider: ThirdPartyProvider,
    name: String,
    model: String,
    #[serde(default)]
    image: Option<bool>,
    #[serde(default)]
    video: Option<bool>,
    #[serde(default)]
    audio: Option<bool>,
    #[serde(default)]
    tools: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct ThirdPartyModelConfig {
    pub id: String,
    pub provider: ThirdPartyProvider,
    pub name: String,
    pub model: String,
    pub image: bool,
    pub video: bool,
    pub audio: bool,
    pub tools: bool,
}

pub fn qualify_third_party_model_id(provider: ThirdPartyProvider, model: &str) -> String {
    format!("{}:{}", provider.as_str(), model.trim())
}

pub fn parse_third_party_model_id(identifier: &str) -> Option<(ThirdPartyProvider, &str)> {
    let (provider, model) = identifier.trim().split_once(':')?;
    let provider = provider.parse().ok()?;
    let model = model.trim();
    if model.is_empty() {
        None
    } else {
        Some((provider, model))
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub bot_token: String,
    pub log_level: String,
    pub database_url: String,
    pub publish_bot_commands: bool,
    pub enable_bot_to_bot_auto_q: bool,
    pub enable_gemini: bool,
    pub gemini_api_key: String,
    pub gemini_model: String,
    pub gemini_lite_model: String,
    pub gemini_pro_model: String,
    pub gemini_image_model: String,
    pub gemini_music_model: String,
    pub gemini_video_model: String,
    pub gemini_temperature: f32,
    pub gemini_top_k: i32,
    pub gemini_top_p: f32,
    pub gemini_max_output_tokens: i32,
    pub gemini_thinking_level: String,
    pub gemini_safety_settings: String,
    pub gemini_request_timeout_secs: u64,
    pub gemini_image_request_timeout_secs: u64,
    pub enable_openrouter: bool,
    pub openrouter_api_key: String,
    pub openrouter_base_url: String,
    pub openrouter_temperature: f32,
    pub openrouter_top_k: i32,
    pub openrouter_top_p: f32,
    pub openrouter_request_timeout_secs: u64,
    pub enable_nvidia: bool,
    pub nvidia_api_key: String,
    pub nvidia_base_url: String,
    pub nvidia_temperature: f32,
    pub nvidia_top_p: f32,
    pub nvidia_request_timeout_secs: u64,
    pub enable_ollama: bool,
    pub ollama_api_key: String,
    pub ollama_base_url: String,
    pub ollama_temperature: f32,
    pub ollama_top_p: f32,
    pub ollama_request_timeout_secs: u64,
    pub enable_openai: bool,
    pub openai_api_key: String,
    pub openai_base_url: String,
    pub openai_request_timeout_secs: u64,
    pub enable_openai_codex: bool,
    pub openai_codex_base_url: String,
    pub openai_codex_originator: String,
    pub openai_codex_client_version: String,
    pub openai_codex_web_search_mode: String,
    pub openai_codex_web_search_context_size: String,
    pub openai_codex_web_search_allowed_domains: Vec<String>,
    pub openai_codex_auth_path: String,
    pub openai_codex_auth_storage: String,
    pub openai_codex_model_path: String,
    pub openai_codex_request_timeout_secs: u64,
    pub openai_codex_image_responses_model: String,
    pub openai_codex_image_model: String,
    pub enable_img2: bool,
    pub img2_base_url: String,
    pub img2_api_key: String,
    pub img2_generate_path: String,
    pub img2_health_path: String,
    pub img2_request_timeout_secs: u64,
    pub img2_media_dir: String,
    pub img2_width: Option<u32>,
    pub img2_height: Option<u32>,
    pub img2_steps: Option<u32>,
    pub enable_jina_mcp: bool,
    pub jina_ai_api_key: String,
    pub jina_search_endpoint: String,
    pub jina_reader_endpoint: String,
    pub twitter_fetch_providers: Vec<String>,
    pub fxtwitter_api_base: String,
    pub vxtwitter_api_base: String,
    pub twitter_fetch_total_timeout_secs: u64,
    pub twitter_provider_timeout_secs: u64,
    pub twitter_response_max_bytes: usize,
    pub external_media_max_bytes: usize,
    pub external_media_total_max_bytes: usize,
    pub enable_brave_search: bool,
    pub brave_search_api_key: String,
    pub brave_search_endpoint: String,
    pub enable_exa_search: bool,
    pub exa_api_key: String,
    pub exa_search_endpoint: String,
    pub web_search_cache_ttl_seconds: u64,
    pub web_search_cache_max_entries: usize,
    pub web_search_providers: Vec<String>,
    pub heavy_command_max_concurrency: usize,
    pub rate_limit_seconds: u64,
    pub model_selection_timeout: u64,
    pub db_max_connections: u32,
    pub db_queue_capacity: usize,
    pub db_write_batch_size: usize,
    pub db_write_flush_ms: u64,
    pub default_text_model: String,
    pub default_quick_text_model: String,
    pub quick_reasoning_effort: String,
    pub default_image_model: String,
    pub telegram_max_length: usize,
    pub media_group_max_items: usize,
    pub external_enrich_fanout: usize,
    pub gemini_upload_fanout: usize,
    pub max_tool_context_items: usize,
    pub enable_tldr_infographic: bool,
    pub agent_step_model: String,
    pub agent_step_reasoning: String,
    pub enable_agentic_factcheck: bool,
    pub enable_agentic_qc: bool,
    pub enable_qc_topic_discovery: bool,
    pub agent_max_wall_clock_secs: u64,
    pub tldr_map_reduce_threshold: usize,
    pub tldr_chunk_size: usize,
    pub tldr_max_messages: usize,
    pub factcheck_max_claims: usize,
    pub factcheck_searches_per_claim: usize,
    pub factcheck_claim_concurrency: usize,
    pub qc_analytics_max_total_calls: usize,
    pub qc_analytics_max_query_calls: usize,
    pub qc_analytics_query_timeout_secs: u64,
    pub telegraph_access_token: String,
    pub telegraph_author_name: String,
    pub telegraph_author_url: String,
    pub user_history_message_count: i64,
    pub cwd_pw_api_key: String,
    pub support_message: String,
    pub support_link: String,
    pub whitelist_file_path: String,
    pub access_controlled_commands: Vec<String>,
    pub third_party_models_config_path: PathBuf,
    pub third_party_models: Vec<ThirdPartyModelConfig>,
    pub third_party_models_by_id: HashMap<String, ThirdPartyModelConfig>,
}

pub static CONFIG: LazyLock<Config> =
    LazyLock::new(|| Config::load().expect("Failed to load configuration"));

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .map(|value| value.trim().eq_ignore_ascii_case("true"))
        .unwrap_or(default)
}

fn env_string(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_i32(name: &str, default: i32) -> i32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

fn parse_optional_positive_u32(value: &str) -> Option<u32> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse::<u32>().ok().filter(|value| *value > 0)
}

fn env_optional_positive_u32(name: &str) -> Option<u32> {
    env::var(name)
        .ok()
        .and_then(|value| parse_optional_positive_u32(&value))
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_timeout_secs(name: &str, default: u64) -> u64 {
    env_u64(name, default).max(1)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_csv_lowercase(name: &str, default: &str) -> Vec<String> {
    env::var(name)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .map(|value| value.trim().to_lowercase())
        .filter(|value| !value.is_empty())
        .collect()
}

fn normalize_twitter_fetch_providers(providers: Vec<String>) -> Result<Vec<String>> {
    let mut normalized = Vec::with_capacity(providers.len());
    for provider in providers {
        let provider = provider.trim().to_lowercase();
        if !matches!(provider.as_str(), "fxtwitter" | "vxtwitter" | "jina") {
            return Err(anyhow::anyhow!(
                "Unsupported Twitter fetch provider: {provider}"
            ));
        }
        if normalized.iter().any(|known| known == &provider) {
            return Err(anyhow::anyhow!(
                "Duplicate Twitter fetch provider: {provider}"
            ));
        }
        normalized.push(provider);
    }
    if normalized.is_empty() {
        return Err(anyhow::anyhow!(
            "At least one Twitter fetch provider is required"
        ));
    }
    Ok(normalized)
}

fn validate_twitter_fetch_limits(
    total_timeout_secs: u64,
    provider_timeout_secs: u64,
    response_max_bytes: usize,
    external_media_max_bytes: usize,
    external_media_total_max_bytes: usize,
) -> Result<()> {
    if total_timeout_secs == 0 || provider_timeout_secs == 0 {
        return Err(anyhow::anyhow!("Twitter fetch timeouts must be positive"));
    }
    if provider_timeout_secs > total_timeout_secs {
        return Err(anyhow::anyhow!(
            "Twitter provider timeout cannot exceed total timeout"
        ));
    }
    if response_max_bytes == 0
        || external_media_max_bytes == 0
        || external_media_total_max_bytes == 0
    {
        return Err(anyhow::anyhow!("Twitter byte limits must be positive"));
    }
    if external_media_max_bytes > external_media_total_max_bytes {
        return Err(anyhow::anyhow!(
            "Per-file external media limit cannot exceed total limit"
        ));
    }
    Ok(())
}

fn validate_https_base(name: &str, value: String) -> Result<String> {
    crate::utils::http::parse_https_allowlisted(name, &value, None)
        .map(|url| url.as_str().trim_end_matches('/').to_string())
}

fn normalize_database_url(value: String) -> String {
    if value.starts_with("sqlite+aiosqlite://") {
        return value.replacen("sqlite+aiosqlite://", "sqlite://", 1);
    }
    value
}

fn normalize_gemini_safety_settings(value: String) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return "permissive".to_string();
    }

    let lowered = trimmed.to_lowercase();
    match lowered.as_str() {
        "permissive" | "off" | "none" => "permissive".to_string(),
        "standard" => "standard".to_string(),
        _ => {
            warn!(
                "Unknown GEMINI_SAFETY_SETTINGS value '{}'; defaulting to permissive.",
                value
            );
            "permissive".to_string()
        }
    }
}

fn resolve_third_party_models_path() -> PathBuf {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(env_value) = env::var("THIRD_PARTY_MODELS_CONFIG_PATH") {
        let env_path = PathBuf::from(env_value);
        if env_path.is_absolute() {
            candidates.push(env_path);
        } else {
            candidates.push(
                env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join(env_path),
            );
        }
    }
    candidates.push(PathBuf::from("third_party_models.json"));
    candidates.push(PathBuf::from("bot").join("third_party_models.json"));

    for candidate in &candidates {
        if candidate.exists() {
            return candidate.to_path_buf();
        }
    }

    candidates
        .first()
        .cloned()
        .unwrap_or_else(|| PathBuf::from("third_party_models.json"))
}

fn build_third_party_model_config(
    provider: ThirdPartyProvider,
    name: &str,
    model: &str,
    image: bool,
    video: bool,
    audio: bool,
    tools: bool,
) -> ThirdPartyModelConfig {
    ThirdPartyModelConfig {
        id: qualify_third_party_model_id(provider, model),
        provider,
        name: name.to_string(),
        model: model.to_string(),
        image,
        video,
        audio,
        tools,
    }
}

fn parse_third_party_models_from_str(raw: &str) -> Vec<ThirdPartyModelConfig> {
    let parsed: ThirdPartyModelsFile = match serde_json::from_str(raw) {
        Ok(data) => data,
        Err(err) => {
            info!("Failed to parse third-party model config JSON: {}", err);
            return Vec::new();
        }
    };

    let mut models = Vec::new();
    for entry in parsed.models {
        let name = entry.name.trim();
        let model = entry.model.trim();
        if name.is_empty() || model.is_empty() {
            continue;
        }
        models.push(build_third_party_model_config(
            entry.provider,
            name,
            model,
            entry.image.unwrap_or(false),
            entry.video.unwrap_or(false),
            entry.audio.unwrap_or(false),
            entry.tools.unwrap_or(true),
        ));
    }
    models
}

fn load_third_party_models_from_path(path: &Path) -> Vec<ThirdPartyModelConfig> {
    if !path.exists() {
        info!("Third-party model config not found at {}", path.display());
        return Vec::new();
    }

    let raw = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(err) => {
            info!(
                "Failed to read third-party model config at {}: {}",
                path.display(),
                err
            );
            return Vec::new();
        }
    };

    let models = parse_third_party_models_from_str(&raw);
    if models.is_empty() && !raw.trim().is_empty() {
        info!("Parsed zero third-party models from {}", path.display());
    }
    models
}

fn load_third_party_models(path: &Path) -> Vec<ThirdPartyModelConfig> {
    let models = load_third_party_models_from_path(path);
    if !models.is_empty() {
        info!(
            "Loaded {} third-party model(s) from {}",
            models.len(),
            path.display()
        );
    } else {
        info!("No third-party models configured in {}", path.display());
    }
    models
}

#[cfg(test)]
fn resolve_exact_model_identifier(value: &str, models: &[ThirdPartyModelConfig]) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if trimmed.eq_ignore_ascii_case("gemini") {
        return "gemini".to_string();
    }

    if let Some((provider, model)) = parse_third_party_model_id(trimmed) {
        return qualify_third_party_model_id(provider, model);
    }

    let exact_matches = models
        .iter()
        .filter(|config_entry| config_entry.model == trimmed)
        .collect::<Vec<_>>();
    if exact_matches.len() == 1 {
        return exact_matches[0].id.clone();
    }

    trimmed.to_string()
}

fn resolve_default_text_model_value(
    default_text_model: Option<&str>,
    default_q_model: Option<&str>,
) -> String {
    default_text_model
        .and_then(|value| {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        })
        .or_else(|| {
            default_q_model.and_then(|value| {
                let trimmed = value.trim();
                (!trimmed.is_empty()).then(|| trimmed.to_string())
            })
        })
        .unwrap_or_else(|| "gemini".to_string())
}

fn resolve_default_quick_text_model_value(
    default_quick_text_model: Option<&str>,
    default_text_model: &str,
) -> String {
    default_quick_text_model
        .and_then(|value| {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        })
        .unwrap_or_else(|| default_text_model.to_string())
}

fn resolve_quick_reasoning_effort_value(value: Option<&str>) -> String {
    value
        .and_then(|value| {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_lowercase())
        })
        .unwrap_or_else(|| "low".to_string())
}

impl Config {
    pub fn load() -> Result<Self> {
        let bot_token = env::var("BOT_TOKEN").unwrap_or_else(|_| {
            if cfg!(test) {
                "test-bot-token".to_string()
            } else {
                String::new()
            }
        });
        if bot_token.trim().is_empty() {
            return Err(anyhow::anyhow!("BOT_TOKEN is required"));
        }

        let third_party_models_config_path = resolve_third_party_models_path();
        let third_party_models = load_third_party_models(&third_party_models_config_path);
        let third_party_models_by_id = third_party_models
            .iter()
            .cloned()
            .map(|model| (model.id.clone(), model))
            .collect::<HashMap<_, _>>();

        let access_controlled_commands = env::var("ACCESS_CONTROLLED_COMMANDS")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(|entry| entry.trim().to_string())
                    .filter(|entry| !entry.is_empty())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let mut web_search_providers = env_csv_lowercase("WEB_SEARCH_PROVIDERS", "brave,exa,jina");
        if web_search_providers.is_empty() {
            web_search_providers = vec!["brave".to_string(), "exa".to_string(), "jina".to_string()];
        }
        let default_text_model = resolve_default_text_model_value(
            env::var("DEFAULT_TEXT_MODEL").ok().as_deref(),
            env::var("DEFAULT_Q_MODEL").ok().as_deref(),
        );
        let default_quick_text_model = resolve_default_quick_text_model_value(
            env::var("DEFAULT_QUICK_TEXT_MODEL").ok().as_deref(),
            &default_text_model,
        );
        let quick_reasoning_effort = resolve_quick_reasoning_effort_value(
            env::var("QUICK_REASONING_EFFORT").ok().as_deref(),
        );
        let twitter_fetch_providers = normalize_twitter_fetch_providers(env_csv_lowercase(
            "TWITTER_FETCH_PROVIDERS",
            "fxtwitter,vxtwitter,jina",
        ))?;
        let fxtwitter_api_base = validate_https_base(
            "FXTWITTER_API_BASE",
            env_string("FXTWITTER_API_BASE", "https://api.fxtwitter.com"),
        )?;
        let vxtwitter_api_base = validate_https_base(
            "VXTWITTER_API_BASE",
            env_string("VXTWITTER_API_BASE", "https://api.vxtwitter.com"),
        )?;
        let jina_reader_endpoint = validate_https_base(
            "JINA_READER_ENDPOINT",
            env_string("JINA_READER_ENDPOINT", "https://r.jina.ai/"),
        )?;
        let twitter_fetch_total_timeout_secs =
            env_timeout_secs("TWITTER_FETCH_TOTAL_TIMEOUT_SECS", 20);
        let twitter_provider_timeout_secs = env_timeout_secs("TWITTER_PROVIDER_TIMEOUT_SECS", 8);
        let twitter_response_max_bytes = env_usize("TWITTER_RESPONSE_MAX_BYTES", 2_097_152);
        let external_media_max_bytes = env_usize("EXTERNAL_MEDIA_MAX_BYTES", 20_971_520);
        let external_media_total_max_bytes =
            env_usize("EXTERNAL_MEDIA_TOTAL_MAX_BYTES", 52_428_800);
        validate_twitter_fetch_limits(
            twitter_fetch_total_timeout_secs,
            twitter_provider_timeout_secs,
            twitter_response_max_bytes,
            external_media_max_bytes,
            external_media_total_max_bytes,
        )?;

        Ok(Config {
            bot_token,
            log_level: env_string("LOG_LEVEL", "info").to_lowercase(),
            database_url: normalize_database_url(env_string(
                "DATABASE_URL",
                "sqlite+aiosqlite:///bot.db",
            )),
            publish_bot_commands: env_bool("PUBLISH_BOT_COMMANDS", false),
            enable_bot_to_bot_auto_q: env_bool("ENABLE_BOT_TO_BOT_AUTO_Q", false),
            enable_gemini: env_bool("ENABLE_GEMINI", true),
            gemini_api_key: env_string("GEMINI_API_KEY", ""),
            gemini_model: env_string("GEMINI_MODEL", "gemini-flash-latest"),
            gemini_lite_model: env_string("GEMINI_LITE_MODEL", "gemini-flash-lite-latest"),
            gemini_pro_model: env_string("GEMINI_PRO_MODEL", "gemini-2.5-pro"),
            gemini_image_model: env_string("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview"),
            gemini_music_model: env_string("GEMINI_MUSIC_MODEL", "lyria-3-pro-preview"),
            gemini_video_model: env_string("GEMINI_VIDEO_MODEL", "veo-3.1-generate-preview"),
            gemini_temperature: env_f32("GEMINI_TEMPERATURE", 0.7),
            gemini_top_k: env_i32("GEMINI_TOP_K", 40),
            gemini_top_p: env_f32("GEMINI_TOP_P", 0.95),
            gemini_max_output_tokens: env_i32("GEMINI_MAX_OUTPUT_TOKENS", 2048),
            gemini_thinking_level: env_string("GEMINI_THINKING_LEVEL", "high"),
            gemini_safety_settings: normalize_gemini_safety_settings(env_string(
                "GEMINI_SAFETY_SETTINGS",
                "permissive",
            )),
            gemini_request_timeout_secs: env_timeout_secs("GEMINI_REQUEST_TIMEOUT_SECS", 90),
            gemini_image_request_timeout_secs: env_timeout_secs(
                "GEMINI_IMAGE_REQUEST_TIMEOUT_SECS",
                300,
            ),
            enable_openrouter: env_bool("ENABLE_OPENROUTER", true),
            openrouter_api_key: env_string("OPENROUTER_API_KEY", ""),
            openrouter_base_url: env_string("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openrouter_temperature: env_f32("OPENROUTER_TEMPERATURE", 0.7),
            openrouter_top_k: env_i32("OPENROUTER_TOP_K", 40),
            openrouter_top_p: env_f32("OPENROUTER_TOP_P", 0.95),
            openrouter_request_timeout_secs: env_timeout_secs(
                "OPENROUTER_REQUEST_TIMEOUT_SECS",
                60,
            ),
            enable_nvidia: env_bool("ENABLE_NVIDIA", true),
            nvidia_api_key: env_string("NVIDIA_API_KEY", ""),
            nvidia_base_url: env_string("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            nvidia_temperature: env_f32("NVIDIA_TEMPERATURE", 0.7),
            nvidia_top_p: env_f32("NVIDIA_TOP_P", 0.95),
            nvidia_request_timeout_secs: env_timeout_secs("NVIDIA_REQUEST_TIMEOUT_SECS", 60),
            enable_ollama: env_bool("ENABLE_OLLAMA", true),
            ollama_api_key: env_string("OLLAMA_API_KEY", ""),
            ollama_base_url: env_string("OLLAMA_BASE_URL", "https://ollama.com/v1"),
            ollama_temperature: env_f32("OLLAMA_TEMPERATURE", 0.7),
            ollama_top_p: env_f32("OLLAMA_TOP_P", 0.95),
            ollama_request_timeout_secs: env_timeout_secs("OLLAMA_REQUEST_TIMEOUT_SECS", 60),
            enable_openai: env_bool("ENABLE_OPENAI", false),
            openai_api_key: env_string("OPENAI_API_KEY", ""),
            openai_base_url: env_string("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_request_timeout_secs: env_timeout_secs("OPENAI_REQUEST_TIMEOUT_SECS", 60),
            enable_openai_codex: env_bool("ENABLE_OPENAI_CODEX", true),
            openai_codex_base_url: env_string(
                "OPENAI_CODEX_BASE_URL",
                "https://chatgpt.com/backend-api/codex",
            ),
            openai_codex_originator: env_string("OPENAI_CODEX_ORIGINATOR", "codex_cli_rs"),
            openai_codex_client_version: env_string("OPENAI_CODEX_CLIENT_VERSION", "0.144.0"),
            openai_codex_web_search_mode: env_string("OPENAI_CODEX_WEB_SEARCH_MODE", "live")
                .to_lowercase(),
            openai_codex_web_search_context_size: env_string(
                "OPENAI_CODEX_WEB_SEARCH_CONTEXT_SIZE",
                "",
            )
            .to_lowercase(),
            openai_codex_web_search_allowed_domains: env_csv_lowercase(
                "OPENAI_CODEX_WEB_SEARCH_ALLOWED_DOMAINS",
                "",
            ),
            openai_codex_auth_path: env_string(
                "OPENAI_CODEX_AUTH_PATH",
                "data/openai_codex_auth.json",
            ),
            openai_codex_auth_storage: env_string("OPENAI_CODEX_AUTH_STORAGE", "auto")
                .to_lowercase(),
            openai_codex_model_path: env_string(
                "OPENAI_CODEX_MODEL_PATH",
                "data/openai_codex_model.json",
            ),
            openai_codex_request_timeout_secs: env_timeout_secs(
                "OPENAI_CODEX_REQUEST_TIMEOUT_SECS",
                300,
            ),
            openai_codex_image_responses_model: env_string(
                "OPENAI_CODEX_IMAGE_RESPONSES_MODEL",
                "gpt-5.5",
            ),
            openai_codex_image_model: env_string("OPENAI_CODEX_IMAGE_MODEL", "gpt-image-2"),
            enable_img2: env_bool("ENABLE_IMG2", false),
            img2_base_url: env_string("IMG2_BASE_URL", "https://wspark.taild6a660.ts.net:8443"),
            img2_api_key: env_string("IMG2_API_KEY", ""),
            img2_generate_path: env_string("IMG2_GENERATE_PATH", "/v1/images/generate"),
            img2_health_path: env_string("IMG2_HEALTH_PATH", "/v1/health"),
            img2_request_timeout_secs: env_timeout_secs("IMG2_REQUEST_TIMEOUT_SECS", 300),
            img2_media_dir: env_string("IMG2_MEDIA_DIR", "data/media/img2"),
            img2_width: env_optional_positive_u32("IMG2_WIDTH"),
            img2_height: env_optional_positive_u32("IMG2_HEIGHT"),
            img2_steps: env_optional_positive_u32("IMG2_STEPS"),
            enable_jina_mcp: env_bool("ENABLE_JINA_MCP", false),
            jina_ai_api_key: env_string("JINA_AI_API_KEY", ""),
            jina_search_endpoint: env_string("JINA_SEARCH_ENDPOINT", "https://s.jina.ai/search"),
            jina_reader_endpoint,
            twitter_fetch_providers,
            fxtwitter_api_base,
            vxtwitter_api_base,
            twitter_fetch_total_timeout_secs,
            twitter_provider_timeout_secs,
            twitter_response_max_bytes,
            external_media_max_bytes,
            external_media_total_max_bytes,
            enable_brave_search: env_bool("ENABLE_BRAVE_SEARCH", true),
            brave_search_api_key: env_string("BRAVE_SEARCH_API_KEY", ""),
            brave_search_endpoint: env_string(
                "BRAVE_SEARCH_ENDPOINT",
                "https://api.search.brave.com/res/v1/web/search",
            ),
            enable_exa_search: env_bool("ENABLE_EXA_SEARCH", true),
            exa_api_key: env_string("EXA_API_KEY", ""),
            exa_search_endpoint: env_string("EXA_SEARCH_ENDPOINT", "https://api.exa.ai/search"),
            web_search_cache_ttl_seconds: env_u64("WEB_SEARCH_CACHE_TTL_SECONDS", 900),
            web_search_cache_max_entries: env_usize("WEB_SEARCH_CACHE_MAX_ENTRIES", 256),
            web_search_providers,
            heavy_command_max_concurrency: env_usize("HEAVY_COMMAND_MAX_CONCURRENCY", 5).max(1),
            rate_limit_seconds: env_u64("RATE_LIMIT_SECONDS", 15),
            model_selection_timeout: env_u64("MODEL_SELECTION_TIMEOUT", 30),
            db_max_connections: env_u32("DB_MAX_CONNECTIONS", 5).max(1),
            db_queue_capacity: env_usize("DB_QUEUE_CAPACITY", 2048).max(1),
            db_write_batch_size: env_usize("DB_WRITE_BATCH_SIZE", 32).max(1),
            db_write_flush_ms: env_u64("DB_WRITE_FLUSH_MS", 25),
            default_text_model,
            default_quick_text_model,
            quick_reasoning_effort,
            default_image_model: env_string("DEFAULT_IMAGE_MODEL", "gemini"),
            telegram_max_length: env_usize("TELEGRAM_MAX_LENGTH", 4000),
            media_group_max_items: env_usize("MEDIA_GROUP_MAX_ITEMS", 256).max(1),
            external_enrich_fanout: env_usize("EXTERNAL_ENRICH_FANOUT", 4).max(1),
            gemini_upload_fanout: env_usize("GEMINI_UPLOAD_FANOUT", 3).max(1),
            max_tool_context_items: env_usize("MAX_TOOL_CONTEXT_ITEMS", 10).max(1),
            enable_tldr_infographic: env_bool("ENABLE_TLDR_INFOGRAPHIC", false),
            agent_step_model: env_string("AGENT_STEP_MODEL", ""),
            agent_step_reasoning: env_string("AGENT_STEP_REASONING", "low"),
            enable_agentic_factcheck: env_bool("ENABLE_AGENTIC_FACTCHECK", true),
            enable_agentic_qc: env_bool("ENABLE_AGENTIC_QC", true),
            enable_qc_topic_discovery: env_bool("ENABLE_QC_TOPIC_DISCOVERY", true),
            agent_max_wall_clock_secs: env_u64("AGENT_MAX_WALL_CLOCK_SECS", 480).max(30),
            tldr_map_reduce_threshold: env_usize("TLDR_MAP_REDUCE_THRESHOLD", 150).max(1),
            tldr_chunk_size: env_usize("TLDR_CHUNK_SIZE", 100).max(20),
            tldr_max_messages: env_usize("TLDR_MAX_MESSAGES", 2000).max(100),
            factcheck_max_claims: env_usize("FACTCHECK_MAX_CLAIMS", 5).clamp(1, 8),
            factcheck_searches_per_claim: env_usize("FACTCHECK_SEARCHES_PER_CLAIM", 2).clamp(1, 3),
            factcheck_claim_concurrency: env_usize("FACTCHECK_CLAIM_CONCURRENCY", 2).clamp(1, 4),
            qc_analytics_max_total_calls: env_usize("QC_ANALYTICS_MAX_TOTAL_CALLS", 12)
                .clamp(4, 24),
            qc_analytics_max_query_calls: env_usize("QC_ANALYTICS_MAX_QUERY_CALLS", 10)
                .clamp(2, 20),
            qc_analytics_query_timeout_secs: env_u64("QC_ANALYTICS_QUERY_TIMEOUT_SECS", 2)
                .clamp(1, 15),
            telegraph_access_token: env_string("TELEGRAPH_ACCESS_TOKEN", ""),
            telegraph_author_name: env_string("TELEGRAPH_AUTHOR_NAME", ""),
            telegraph_author_url: env_string("TELEGRAPH_AUTHOR_URL", ""),
            user_history_message_count: env_u64("USER_HISTORY_MESSAGE_COUNT", 200) as i64,
            cwd_pw_api_key: env_string("CWD_PW_API_KEY", ""),
            support_message: env_string(
                "SUPPORT_MESSAGE",
                "Thanks for supporting the bot! Tap the button below to open the support page.",
            ),
            support_link: env_string("SUPPORT_LINK", ""),
            whitelist_file_path: env_string("WHITELIST_FILE_PATH", "allowed_chat.txt"),
            access_controlled_commands,
            third_party_models_config_path,
            third_party_models,
            third_party_models_by_id,
        })
    }

    pub fn get_third_party_model_config(&self, model_id: &str) -> Option<&ThirdPartyModelConfig> {
        self.third_party_models_by_id.get(model_id)
    }

    pub fn is_third_party_provider_ready(&self, provider: ThirdPartyProvider) -> bool {
        match provider {
            ThirdPartyProvider::OpenRouter => {
                self.enable_openrouter && !self.openrouter_api_key.trim().is_empty()
            }
            ThirdPartyProvider::Nvidia => {
                self.enable_nvidia && !self.nvidia_api_key.trim().is_empty()
            }
            ThirdPartyProvider::Ollama => {
                self.enable_ollama && !self.ollama_api_key.trim().is_empty()
            }
            ThirdPartyProvider::OpenAI => {
                self.enable_openai && !self.openai_api_key.trim().is_empty()
            }
            ThirdPartyProvider::OpenAICodex => self.enable_openai_codex,
        }
    }

    pub fn gemini_api_available(&self) -> bool {
        gemini_api_available_from(self.enable_gemini, &self.gemini_api_key)
    }

    pub fn img2_api_available(&self) -> bool {
        self.enable_img2 && !self.img2_api_key.trim().is_empty()
    }
}

pub(crate) fn gemini_api_available_from(enable_gemini: bool, api_key: &str) -> bool {
    enable_gemini && !api_key.trim().is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_text_model_prefers_new_env_value_over_legacy_q_value() {
        assert_eq!(
            resolve_default_text_model_value(Some("openai-codex"), Some("gemini")),
            "openai-codex"
        );
    }

    #[test]
    fn default_text_model_uses_legacy_q_value_when_new_value_missing() {
        assert_eq!(
            resolve_default_text_model_value(None, Some("openai-codex:selected")),
            "openai-codex:selected"
        );
    }

    #[test]
    fn default_text_model_defaults_to_gemini_when_both_values_missing() {
        assert_eq!(resolve_default_text_model_value(None, None), "gemini");
    }

    #[test]
    fn default_quick_text_model_inherits_resolved_default_text_model_when_unset_or_blank() {
        assert_eq!(
            resolve_default_quick_text_model_value(None, "openai-codex:selected"),
            "openai-codex:selected"
        );
        assert_eq!(
            resolve_default_quick_text_model_value(Some("   "), "openrouter:fast/model"),
            "openrouter:fast/model"
        );
        assert_eq!(
            resolve_default_quick_text_model_value(Some(" nvidia:quick/model "), "gemini"),
            "nvidia:quick/model"
        );
    }

    #[test]
    fn quick_reasoning_effort_defaults_to_low_when_unset_or_blank() {
        assert_eq!(resolve_quick_reasoning_effort_value(None), "low");
        assert_eq!(resolve_quick_reasoning_effort_value(Some("  ")), "low");
        assert_eq!(
            resolve_quick_reasoning_effort_value(Some(" medium ")),
            "medium"
        );
    }

    #[test]
    fn gemini_api_available_respects_enable_flag() {
        assert!(!gemini_api_available_from(false, "test-key"));
        assert!(!gemini_api_available_from(true, ""));
        assert!(gemini_api_available_from(true, "test-key"));
    }

    #[test]
    fn parse_third_party_models_supports_mixed_providers() {
        let raw = r#"{
            "models": [
                {
                    "provider": "openrouter",
                    "name": "Qwen 3",
                    "model": "qwen/qwen3-next-80b-a3b-instruct:free",
                    "tools": true
                },
                {
                    "provider": "nvidia",
                    "name": "Gemma 3n",
                    "model": "google/gemma-3n-e4b-it",
                    "image": true,
                    "audio": true,
                    "tools": false
                },
                {
                    "provider": "openai",
                    "name": "GPT-5.4 API",
                    "model": "gpt-5.4",
                    "image": true,
                    "tools": true
                },
                {
                    "provider": "ollama",
                    "name": "Qwen 3 32B",
                    "model": "qwen3:32b",
                    "image": true,
                    "tools": true
                },
                {
                    "provider": "openai-codex",
                    "name": "Codex Selected",
                    "model": "selected",
                    "image": true,
                    "tools": true
                }
            ]
        }"#;

        let models = parse_third_party_models_from_str(raw);

        assert_eq!(models.len(), 5);
        assert_eq!(
            models[0].id,
            "openrouter:qwen/qwen3-next-80b-a3b-instruct:free"
        );
        assert_eq!(models[0].provider, ThirdPartyProvider::OpenRouter);
        assert_eq!(models[1].id, "nvidia:google/gemma-3n-e4b-it");
        assert_eq!(models[1].provider, ThirdPartyProvider::Nvidia);
        assert!(models[1].image);
        assert!(models[1].audio);
        assert!(!models[1].tools);
        assert_eq!(models[2].provider, ThirdPartyProvider::OpenAI);
        assert_eq!(models[2].id, "openai:gpt-5.4");
        assert_eq!(models[3].provider, ThirdPartyProvider::Ollama);
        assert_eq!(models[3].id, "ollama:qwen3:32b");
        assert_eq!(models[4].provider, ThirdPartyProvider::OpenAICodex);
        assert_eq!(models[4].id, "openai-codex:selected");
    }

    #[test]
    fn provider_qualified_ids_disambiguate_duplicate_raw_model_ids() {
        let raw = r#"{
            "models": [
                {
                    "provider": "openrouter",
                    "name": "Shared OpenRouter",
                    "model": "shared/model"
                },
                {
                    "provider": "nvidia",
                    "name": "Shared NVIDIA",
                    "model": "shared/model"
                }
            ]
        }"#;

        let models = parse_third_party_models_from_str(raw);
        let model_map = models
            .iter()
            .cloned()
            .map(|model| (model.id.clone(), model))
            .collect::<HashMap<_, _>>();

        assert_eq!(models.len(), 2);
        assert!(model_map.contains_key("openrouter:shared/model"));
        assert!(model_map.contains_key("nvidia:shared/model"));
        assert_eq!(
            resolve_exact_model_identifier("shared/model", &models),
            "shared/model"
        );
    }

    #[test]
    fn resolve_exact_model_identifier_returns_provider_qualified_id_for_unique_raw_match() {
        let models = vec![
            build_third_party_model_config(
                ThirdPartyProvider::OpenRouter,
                "Llama 4",
                "meta-llama/llama-4",
                true,
                false,
                false,
                true,
            ),
            build_third_party_model_config(
                ThirdPartyProvider::Nvidia,
                "Nemotron Super 49B",
                "nvidia/llama-3.3-nemotron-super-49b-v1.5",
                false,
                false,
                false,
                true,
            ),
        ];

        let exact = resolve_exact_model_identifier("meta-llama/llama-4", &models);
        assert_eq!(exact, "openrouter:meta-llama/llama-4");

        let unique_raw =
            resolve_exact_model_identifier("nvidia/llama-3.3-nemotron-super-49b-v1.5", &models);
        assert_eq!(
            unique_raw,
            "nvidia:nvidia/llama-3.3-nemotron-super-49b-v1.5"
        );
    }

    #[test]
    fn img2_optional_u32_values_are_positive_or_omitted() {
        assert_eq!(parse_optional_positive_u32("1024"), Some(1024));
        assert_eq!(parse_optional_positive_u32(" 4 "), Some(4));
        assert_eq!(parse_optional_positive_u32(""), None);
        assert_eq!(parse_optional_positive_u32("0"), None);
        assert_eq!(parse_optional_positive_u32("wide"), None);
    }

    #[test]
    fn twitter_provider_order_requires_known_unique_entries() {
        assert_eq!(
            normalize_twitter_fetch_providers(vec![
                "fxtwitter".into(),
                "vxtwitter".into(),
                "jina".into(),
            ])
            .unwrap(),
            vec!["fxtwitter", "vxtwitter", "jina"]
        );
        assert!(
            normalize_twitter_fetch_providers(vec!["fxtwitter".into(), "fxtwitter".into()])
                .is_err()
        );
        assert!(normalize_twitter_fetch_providers(vec!["unknown".into()]).is_err());
        assert!(normalize_twitter_fetch_providers(Vec::new()).is_err());
    }

    #[test]
    fn twitter_limits_reject_unsafe_combinations() {
        assert!(validate_twitter_fetch_limits(20, 8, 2_097_152, 20_971_520, 52_428_800).is_ok());
        assert!(validate_twitter_fetch_limits(8, 20, 2_097_152, 20_971_520, 52_428_800).is_err());
        assert!(validate_twitter_fetch_limits(20, 8, 0, 20_971_520, 52_428_800).is_err());
        assert!(validate_twitter_fetch_limits(20, 8, 2_097_152, 60_000_000, 52_428_800).is_err());
    }

    #[test]
    fn twitter_provider_bases_require_safe_https() {
        for value in [
            "https://api.fxtwitter.com",
            "https://api.vxtwitter.com",
            "https://r.jina.ai/",
        ] {
            assert!(validate_https_base("TEST_ENDPOINT", value.into()).is_ok());
        }
        for value in [
            "http://api.fxtwitter.com",
            "https://user@api.fxtwitter.com",
            "https://api.fxtwitter.com?debug=1",
            "https://api.fxtwitter.com/#fragment",
            "https://api.fxtwitter.com:8443",
        ] {
            assert!(validate_https_base("TEST_ENDPOINT", value.into()).is_err());
        }
    }
}
