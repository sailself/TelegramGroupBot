use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Deserialize;
use std::sync::LazyLock;
use tracing::{info, warn};

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

/// Telegram-facing bot settings: identity, outbound formatting limits, and
/// the support-command copy.
#[derive(Debug, Clone)]
pub struct TelegramConfig {
    pub bot_token: String,
    pub publish_bot_commands: bool,
    pub enable_bot_to_bot_auto_q: bool,
    pub max_length: usize,
    pub media_group_max_items: usize,
    pub support_message: String,
    pub support_link: String,
}

/// Access control: the admin whitelist file and which commands it gates.
#[derive(Debug, Clone)]
pub struct AccessConfig {
    pub whitelist_file_path: String,
    pub access_controlled_commands: Vec<String>,
    pub rate_limit_seconds: u64,
}

/// SQLite connection and write-queue tuning.
#[derive(Debug, Clone)]
pub struct DbConfig {
    pub url: String,
    pub max_connections: u32,
    pub queue_capacity: usize,
    pub write_batch_size: usize,
    pub write_flush_ms: u64,
}

/// Gemini model selection, sampling parameters, and per-call timeouts.
#[derive(Debug, Clone)]
pub struct GeminiConfig {
    pub enabled: bool,
    pub api_key: String,
    pub model: String,
    pub lite_model: String,
    pub pro_model: String,
    pub image_model: String,
    pub music_model: String,
    pub video_model: String,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub max_output_tokens: i32,
    pub thinking_level: String,
    pub safety_settings: String,
    pub request_timeout_secs: u64,
    pub image_request_timeout_secs: u64,
    pub upload_fanout: usize,
}

/// Shared shape for the OpenAI-compatible chat providers (OpenRouter,
/// NVIDIA, Ollama). `top_k` is only ever populated for OpenRouter; NVIDIA and
/// Ollama have no such knob and always load `None`.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub enabled: bool,
    pub api_key: String,
    pub base_url: String,
    pub temperature: f32,
    pub top_k: Option<i32>,
    pub top_p: f32,
    pub request_timeout_secs: u64,
}

/// OpenAI Responses API settings (distinct from the ChatGPT Codex backend).
#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    pub enabled: bool,
    pub api_key: String,
    pub base_url: String,
    pub request_timeout_secs: u64,
}

/// ChatGPT Codex backend settings: auth/model persistence paths, web search
/// behavior, and the image models it can drive.
#[derive(Debug, Clone)]
pub struct CodexConfig {
    pub enabled: bool,
    pub base_url: String,
    pub originator: String,
    pub client_version: String,
    pub web_search_mode: String,
    pub web_search_context_size: String,
    pub web_search_allowed_domains: Vec<String>,
    pub auth_path: String,
    pub auth_storage: String,
    pub model_path: String,
    pub request_timeout_secs: u64,
    pub image_responses_model: String,
    pub image_model: String,
}

/// The self-hosted image generation backend ("img2"), opt-in and unrelated
/// to Gemini's image model.
#[derive(Debug, Clone)]
pub struct Img2Config {
    pub enabled: bool,
    pub base_url: String,
    pub api_key: String,
    pub generate_path: String,
    pub health_path: String,
    pub request_timeout_secs: u64,
    pub media_dir: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
}

/// Twitter/X link-unfurling: which providers to try, in what order, and
/// their endpoints/timeouts/byte limits.
#[derive(Debug, Clone)]
pub struct TwitterConfig {
    pub fetch_providers: Vec<String>,
    pub fxtwitter_api_base: String,
    pub vxtwitter_api_base: String,
    pub fetch_total_timeout_secs: u64,
    pub provider_timeout_secs: u64,
    pub response_max_bytes: usize,
}

/// Byte limits and fan-out for enriching links (Twitter and otherwise) with
/// externally-fetched media.
#[derive(Debug, Clone)]
pub struct ExternalMediaConfig {
    pub max_bytes: usize,
    pub total_max_bytes: usize,
    pub enrich_fanout: usize,
}

/// Web search provider order, per-provider enablement, and result caching.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub providers: Vec<String>,
    pub enable_brave: bool,
    pub brave_api_key: String,
    pub brave_endpoint: String,
    pub enable_exa: bool,
    pub exa_api_key: String,
    pub exa_endpoint: String,
    pub jina_endpoint: String,
    pub cache_ttl_seconds: u64,
    pub cache_max_entries: usize,
}

/// Jina-specific settings: the MCP toggle (independent of whether Jina is
/// used as a search/Twitter provider) plus its API key and reader endpoint.
#[derive(Debug, Clone)]
pub struct JinaConfig {
    pub enable_mcp: bool,
    pub api_key: String,
    pub reader_endpoint: String,
}

/// Telegraph publishing identity used when the bot posts long-form content.
#[derive(Debug, Clone)]
pub struct TelegraphConfig {
    pub access_token: String,
    pub author_name: String,
    pub author_url: String,
}

/// CWD.PW image-hosting credential.
#[derive(Debug, Clone)]
pub struct CwdPwConfig {
    pub api_key: String,
}

/// Default model selection plus the runtime third-party model catalog
/// loaded from `third_party_models.json`.
#[derive(Debug, Clone)]
pub struct ModelDefaults {
    pub default_text_model: String,
    pub default_quick_text_model: String,
    pub quick_reasoning_effort: String,
    pub default_image_model: String,
    pub third_party_models_config_path: PathBuf,
    pub third_party_models: Vec<ThirdPartyModelConfig>,
    pub third_party_models_by_id: HashMap<String, ThirdPartyModelConfig>,
}

/// Runtime concurrency/timeout knobs that don't belong to any one provider.
#[derive(Debug, Clone)]
pub struct RuntimeLimits {
    pub heavy_command_max_concurrency: usize,
    pub model_selection_timeout: u64,
    pub max_tool_context_items: usize,
    pub user_history_message_count: i64,
}

/// Agentic pipeline tuning: the step model/reasoning used by the agent
/// runtime, and per-pipeline (TL;DR, fact-check, QC analytics) limits.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub step_model: String,
    pub step_reasoning: String,
    pub max_wall_clock_secs: u64,
    pub enable_agentic_factcheck: bool,
    pub enable_agentic_qc: bool,
    pub enable_qc_topic_discovery: bool,
    pub enable_tldr_infographic: bool,
    pub tldr_map_reduce_threshold: usize,
    pub tldr_chunk_size: usize,
    pub tldr_max_messages: usize,
    pub factcheck_max_claims: usize,
    pub factcheck_searches_per_claim: usize,
    pub factcheck_claim_concurrency: usize,
    pub qc_analytics_max_total_calls: usize,
    pub qc_analytics_max_query_calls: usize,
    pub qc_analytics_query_timeout_secs: u64,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub log_level: String,
    pub telegram: TelegramConfig,
    pub access: AccessConfig,
    pub db: DbConfig,
    pub gemini: GeminiConfig,
    pub openrouter: ProviderConfig,
    pub nvidia: ProviderConfig,
    pub ollama: ProviderConfig,
    pub openai: OpenAiConfig,
    pub codex: CodexConfig,
    pub img2: Img2Config,
    pub twitter: TwitterConfig,
    pub external_media: ExternalMediaConfig,
    pub search: SearchConfig,
    pub jina: JinaConfig,
    pub telegraph: TelegraphConfig,
    pub cwd_pw: CwdPwConfig,
    pub models: ModelDefaults,
    pub limits: RuntimeLimits,
    pub agents: AgentConfig,
}

pub static CONFIG: LazyLock<Config> =
    LazyLock::new(|| Config::load().expect("Failed to load configuration"));

/// Parse an environment variable with `T::from_str`, warning once and
/// falling back to `default` when the value is set but does not parse.
fn env_parsed<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(raw) if !raw.trim().is_empty() => raw.trim().parse::<T>().unwrap_or_else(|_| {
            tracing::warn!(env = name, value = %raw, "unparsable value, using the default");
            default
        }),
        _ => default,
    }
}

/// Newtype so `env_parsed` can drive `env_bool`'s looser boolean grammar
/// (`true`/`false`/`1`/`0`/`yes`/`no`) instead of `bool::from_str`'s
/// case-sensitive `true`/`false` only.
struct LooseBool(bool);

impl std::str::FromStr for LooseBool {
    type Err = ();

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" => Ok(LooseBool(true)),
            "false" | "0" | "no" => Ok(LooseBool(false)),
            _ => Err(()),
        }
    }
}

fn env_bool(name: &str, default: bool) -> bool {
    env_parsed(name, LooseBool(default)).0
}

fn env_string(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_f32(name: &str, default: f32) -> f32 {
    env_parsed(name, default)
}

fn env_i32(name: &str, default: i32) -> i32 {
    env_parsed(name, default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    env_parsed(name, default)
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
    env_parsed(name, default)
}

fn env_timeout_secs(name: &str, default: u64) -> u64 {
    env_u64(name, default).max(1)
}

fn env_usize(name: &str, default: usize) -> usize {
    env_parsed(name, default)
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

/// Validate `value` as an absolute HTTPS URL via [`validate_https_base`], or,
/// failing that, as a plain `http` URL whose host is loopback (`localhost`,
/// `127.0.0.1`, or `::1`) — the common local Ollama setup.
fn validate_http_base_allowing_loopback(name: &str, value: String) -> Result<String> {
    let https_err = match validate_https_base(name, value.clone()) {
        Ok(validated) => return Ok(validated),
        Err(err) => err,
    };

    let Ok(parsed) = url::Url::parse(value.trim()) else {
        return Err(https_err);
    };
    if parsed.scheme() != "http" {
        return Err(https_err);
    }
    let is_loopback = match parsed.host() {
        Some(url::Host::Domain(domain)) => domain.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(addr)) => addr.is_loopback(),
        Some(url::Host::Ipv6(addr)) => addr.is_loopback(),
        None => false,
    };
    if !is_loopback {
        return Err(https_err);
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(anyhow::anyhow!("{name} must not contain credentials"));
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        return Err(anyhow::anyhow!(
            "{name} must not contain a query or fragment"
        ));
    }
    Ok(parsed.as_str().trim_end_matches('/').to_string())
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

fn resolve_default_text_model_value(default_text_model: Option<&str>) -> String {
    default_text_model
        .and_then(|value| {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
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

fn load_telegram() -> Result<TelegramConfig> {
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

    Ok(TelegramConfig {
        bot_token,
        publish_bot_commands: env_bool("PUBLISH_BOT_COMMANDS", false),
        enable_bot_to_bot_auto_q: env_bool("ENABLE_BOT_TO_BOT_AUTO_Q", false),
        max_length: env_usize("TELEGRAM_MAX_LENGTH", 4000),
        media_group_max_items: env_usize("MEDIA_GROUP_MAX_ITEMS", 256).max(1),
        support_message: env_string(
            "SUPPORT_MESSAGE",
            "Thanks for supporting the bot! Tap the button below to open the support page.",
        ),
        support_link: env_string("SUPPORT_LINK", ""),
    })
}

fn load_access() -> AccessConfig {
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

    AccessConfig {
        whitelist_file_path: env_string("WHITELIST_FILE_PATH", "allowed_chat.txt"),
        access_controlled_commands,
        rate_limit_seconds: env_u64("RATE_LIMIT_SECONDS", 15),
    }
}

fn load_db() -> DbConfig {
    DbConfig {
        url: env_string("DATABASE_URL", "sqlite://bot.db"),
        max_connections: env_u32("DB_MAX_CONNECTIONS", 5).max(1),
        queue_capacity: env_usize("DB_QUEUE_CAPACITY", 2048).max(1),
        write_batch_size: env_usize("DB_WRITE_BATCH_SIZE", 32).max(1),
        write_flush_ms: env_u64("DB_WRITE_FLUSH_MS", 25),
    }
}

fn load_gemini() -> GeminiConfig {
    GeminiConfig {
        enabled: env_bool("ENABLE_GEMINI", true),
        api_key: env_string("GEMINI_API_KEY", ""),
        model: env_string("GEMINI_MODEL", "gemini-flash-latest"),
        lite_model: env_string("GEMINI_LITE_MODEL", "gemini-flash-lite-latest"),
        pro_model: env_string("GEMINI_PRO_MODEL", "gemini-2.5-pro"),
        image_model: env_string("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview"),
        music_model: env_string("GEMINI_MUSIC_MODEL", "lyria-3-pro-preview"),
        video_model: env_string("GEMINI_VIDEO_MODEL", "veo-3.1-generate-preview"),
        temperature: env_f32("GEMINI_TEMPERATURE", 0.7),
        top_k: env_i32("GEMINI_TOP_K", 40),
        top_p: env_f32("GEMINI_TOP_P", 0.95),
        max_output_tokens: env_i32("GEMINI_MAX_OUTPUT_TOKENS", 2048),
        thinking_level: env_string("GEMINI_THINKING_LEVEL", "high"),
        safety_settings: normalize_gemini_safety_settings(env_string(
            "GEMINI_SAFETY_SETTINGS",
            "permissive",
        )),
        request_timeout_secs: env_timeout_secs("GEMINI_REQUEST_TIMEOUT_SECS", 90),
        image_request_timeout_secs: env_timeout_secs("GEMINI_IMAGE_REQUEST_TIMEOUT_SECS", 300),
        upload_fanout: env_usize("GEMINI_UPLOAD_FANOUT", 3).max(1),
    }
}

fn load_openrouter() -> Result<ProviderConfig> {
    let base_url = validate_https_base(
        "OPENROUTER_BASE_URL",
        env_string("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )?;
    Ok(ProviderConfig {
        enabled: env_bool("ENABLE_OPENROUTER", true),
        api_key: env_string("OPENROUTER_API_KEY", ""),
        base_url,
        temperature: env_f32("OPENROUTER_TEMPERATURE", 0.7),
        top_k: Some(env_i32("OPENROUTER_TOP_K", 40)),
        top_p: env_f32("OPENROUTER_TOP_P", 0.95),
        request_timeout_secs: env_timeout_secs("OPENROUTER_REQUEST_TIMEOUT_SECS", 60),
    })
}

fn load_nvidia() -> Result<ProviderConfig> {
    let base_url = validate_https_base(
        "NVIDIA_BASE_URL",
        env_string("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    )?;
    Ok(ProviderConfig {
        enabled: env_bool("ENABLE_NVIDIA", true),
        api_key: env_string("NVIDIA_API_KEY", ""),
        base_url,
        temperature: env_f32("NVIDIA_TEMPERATURE", 0.7),
        top_k: None,
        top_p: env_f32("NVIDIA_TOP_P", 0.95),
        request_timeout_secs: env_timeout_secs("NVIDIA_REQUEST_TIMEOUT_SECS", 60),
    })
}

fn load_ollama() -> Result<ProviderConfig> {
    let base_url = validate_http_base_allowing_loopback(
        "OLLAMA_BASE_URL",
        env_string("OLLAMA_BASE_URL", "https://ollama.com/v1"),
    )?;
    Ok(ProviderConfig {
        enabled: env_bool("ENABLE_OLLAMA", true),
        api_key: env_string("OLLAMA_API_KEY", ""),
        base_url,
        temperature: env_f32("OLLAMA_TEMPERATURE", 0.7),
        top_k: None,
        top_p: env_f32("OLLAMA_TOP_P", 0.95),
        request_timeout_secs: env_timeout_secs("OLLAMA_REQUEST_TIMEOUT_SECS", 60),
    })
}

fn load_openai() -> Result<OpenAiConfig> {
    let base_url = validate_https_base(
        "OPENAI_BASE_URL",
        env_string("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )?;
    Ok(OpenAiConfig {
        enabled: env_bool("ENABLE_OPENAI", false),
        api_key: env_string("OPENAI_API_KEY", ""),
        base_url,
        request_timeout_secs: env_timeout_secs("OPENAI_REQUEST_TIMEOUT_SECS", 60),
    })
}

fn load_codex() -> Result<CodexConfig> {
    let base_url = validate_https_base(
        "OPENAI_CODEX_BASE_URL",
        env_string(
            "OPENAI_CODEX_BASE_URL",
            "https://chatgpt.com/backend-api/codex",
        ),
    )?;
    Ok(CodexConfig {
        enabled: env_bool("ENABLE_OPENAI_CODEX", true),
        base_url,
        originator: env_string("OPENAI_CODEX_ORIGINATOR", "codex_cli_rs"),
        client_version: env_string(
            "OPENAI_CODEX_CLIENT_VERSION",
            crate::llm::openai_codex::CODEX_CLIENT_VERSION,
        ),
        web_search_mode: env_string("OPENAI_CODEX_WEB_SEARCH_MODE", "live").to_lowercase(),
        web_search_context_size: env_string("OPENAI_CODEX_WEB_SEARCH_CONTEXT_SIZE", "")
            .to_lowercase(),
        web_search_allowed_domains: env_csv_lowercase(
            "OPENAI_CODEX_WEB_SEARCH_ALLOWED_DOMAINS",
            "",
        ),
        auth_path: env_string("OPENAI_CODEX_AUTH_PATH", "data/openai_codex_auth.json"),
        auth_storage: env_string("OPENAI_CODEX_AUTH_STORAGE", "auto").to_lowercase(),
        model_path: env_string("OPENAI_CODEX_MODEL_PATH", "data/openai_codex_model.json"),
        request_timeout_secs: env_timeout_secs("OPENAI_CODEX_REQUEST_TIMEOUT_SECS", 300),
        image_responses_model: env_string("OPENAI_CODEX_IMAGE_RESPONSES_MODEL", "gpt-5.5"),
        image_model: env_string("OPENAI_CODEX_IMAGE_MODEL", "gpt-image-2"),
    })
}

fn load_img2() -> Result<Img2Config> {
    // No personal default: img2 is opt-in, and any non-empty value must
    // pass the same https validation as every other provider endpoint.
    let base_url_raw = env_string("IMG2_BASE_URL", "");
    let base_url = if base_url_raw.trim().is_empty() {
        String::new()
    } else {
        validate_https_base("IMG2_BASE_URL", base_url_raw)?
    };
    Ok(Img2Config {
        enabled: env_bool("ENABLE_IMG2", false),
        base_url,
        api_key: env_string("IMG2_API_KEY", ""),
        generate_path: env_string("IMG2_GENERATE_PATH", "/v1/images/generate"),
        health_path: env_string("IMG2_HEALTH_PATH", "/v1/health"),
        request_timeout_secs: env_timeout_secs("IMG2_REQUEST_TIMEOUT_SECS", 300),
        media_dir: env_string("IMG2_MEDIA_DIR", "data/media/img2"),
        width: env_optional_positive_u32("IMG2_WIDTH"),
        height: env_optional_positive_u32("IMG2_HEIGHT"),
        steps: env_optional_positive_u32("IMG2_STEPS"),
    })
}

fn load_twitter() -> Result<TwitterConfig> {
    let fetch_providers = normalize_twitter_fetch_providers(env_csv_lowercase(
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
    Ok(TwitterConfig {
        fetch_providers,
        fxtwitter_api_base,
        vxtwitter_api_base,
        fetch_total_timeout_secs: env_timeout_secs("TWITTER_FETCH_TOTAL_TIMEOUT_SECS", 20),
        provider_timeout_secs: env_timeout_secs("TWITTER_PROVIDER_TIMEOUT_SECS", 8),
        response_max_bytes: env_usize("TWITTER_RESPONSE_MAX_BYTES", 2_097_152),
    })
}

fn load_external_media() -> ExternalMediaConfig {
    ExternalMediaConfig {
        max_bytes: env_usize("EXTERNAL_MEDIA_MAX_BYTES", 20_971_520),
        total_max_bytes: env_usize("EXTERNAL_MEDIA_TOTAL_MAX_BYTES", 52_428_800),
        enrich_fanout: env_usize("EXTERNAL_ENRICH_FANOUT", 4).max(1),
    }
}

fn load_search() -> Result<SearchConfig> {
    let brave_endpoint = validate_https_base(
        "BRAVE_SEARCH_ENDPOINT",
        env_string(
            "BRAVE_SEARCH_ENDPOINT",
            "https://api.search.brave.com/res/v1/web/search",
        ),
    )?;
    let exa_endpoint = validate_https_base(
        "EXA_SEARCH_ENDPOINT",
        env_string("EXA_SEARCH_ENDPOINT", "https://api.exa.ai/search"),
    )?;
    let jina_endpoint = validate_https_base(
        "JINA_SEARCH_ENDPOINT",
        env_string("JINA_SEARCH_ENDPOINT", "https://s.jina.ai/search"),
    )?;
    let mut providers = env_csv_lowercase("WEB_SEARCH_PROVIDERS", "brave,exa,jina");
    if providers.is_empty() {
        providers = vec!["brave".to_string(), "exa".to_string(), "jina".to_string()];
    }
    Ok(SearchConfig {
        providers,
        enable_brave: env_bool("ENABLE_BRAVE_SEARCH", true),
        brave_api_key: env_string("BRAVE_SEARCH_API_KEY", ""),
        brave_endpoint,
        enable_exa: env_bool("ENABLE_EXA_SEARCH", true),
        exa_api_key: env_string("EXA_API_KEY", ""),
        exa_endpoint,
        jina_endpoint,
        cache_ttl_seconds: env_u64("WEB_SEARCH_CACHE_TTL_SECONDS", 900),
        cache_max_entries: env_usize("WEB_SEARCH_CACHE_MAX_ENTRIES", 256),
    })
}

fn load_jina() -> Result<JinaConfig> {
    let reader_endpoint = validate_https_base(
        "JINA_READER_ENDPOINT",
        env_string("JINA_READER_ENDPOINT", "https://r.jina.ai/"),
    )?;
    Ok(JinaConfig {
        enable_mcp: env_bool("ENABLE_JINA_MCP", false),
        api_key: env_string("JINA_AI_API_KEY", ""),
        reader_endpoint,
    })
}

fn load_telegraph() -> TelegraphConfig {
    TelegraphConfig {
        access_token: env_string("TELEGRAPH_ACCESS_TOKEN", ""),
        author_name: env_string("TELEGRAPH_AUTHOR_NAME", ""),
        author_url: env_string("TELEGRAPH_AUTHOR_URL", ""),
    }
}

fn load_cwd_pw() -> CwdPwConfig {
    CwdPwConfig {
        api_key: env_string("CWD_PW_API_KEY", ""),
    }
}

fn load_models() -> ModelDefaults {
    let third_party_models_config_path = resolve_third_party_models_path();
    let third_party_models = load_third_party_models(&third_party_models_config_path);
    let third_party_models_by_id = third_party_models
        .iter()
        .cloned()
        .map(|model| (model.id.clone(), model))
        .collect::<HashMap<_, _>>();

    let default_text_model =
        resolve_default_text_model_value(env::var("DEFAULT_TEXT_MODEL").ok().as_deref());
    let default_quick_text_model = resolve_default_quick_text_model_value(
        env::var("DEFAULT_QUICK_TEXT_MODEL").ok().as_deref(),
        &default_text_model,
    );
    let quick_reasoning_effort =
        resolve_quick_reasoning_effort_value(env::var("QUICK_REASONING_EFFORT").ok().as_deref());

    ModelDefaults {
        default_text_model,
        default_quick_text_model,
        quick_reasoning_effort,
        default_image_model: env_string("DEFAULT_IMAGE_MODEL", "gemini"),
        third_party_models_config_path,
        third_party_models,
        third_party_models_by_id,
    }
}

fn load_limits() -> RuntimeLimits {
    RuntimeLimits {
        heavy_command_max_concurrency: env_usize("HEAVY_COMMAND_MAX_CONCURRENCY", 5).max(1),
        model_selection_timeout: env_u64("MODEL_SELECTION_TIMEOUT", 30),
        max_tool_context_items: env_usize("MAX_TOOL_CONTEXT_ITEMS", 10).max(1),
        user_history_message_count: env_u64("USER_HISTORY_MESSAGE_COUNT", 200) as i64,
    }
}

fn load_agents() -> AgentConfig {
    AgentConfig {
        step_model: env_string("AGENT_STEP_MODEL", ""),
        step_reasoning: env_string("AGENT_STEP_REASONING", "low"),
        max_wall_clock_secs: env_u64("AGENT_MAX_WALL_CLOCK_SECS", 480).max(30),
        enable_agentic_factcheck: env_bool("ENABLE_AGENTIC_FACTCHECK", true),
        enable_agentic_qc: env_bool("ENABLE_AGENTIC_QC", true),
        enable_qc_topic_discovery: env_bool("ENABLE_QC_TOPIC_DISCOVERY", true),
        enable_tldr_infographic: env_bool("ENABLE_TLDR_INFOGRAPHIC", false),
        tldr_map_reduce_threshold: env_usize("TLDR_MAP_REDUCE_THRESHOLD", 150).max(1),
        tldr_chunk_size: env_usize("TLDR_CHUNK_SIZE", 100).max(20),
        tldr_max_messages: env_usize("TLDR_MAX_MESSAGES", 2000).max(100),
        factcheck_max_claims: env_usize("FACTCHECK_MAX_CLAIMS", 5).clamp(1, 8),
        factcheck_searches_per_claim: env_usize("FACTCHECK_SEARCHES_PER_CLAIM", 2).clamp(1, 3),
        factcheck_claim_concurrency: env_usize("FACTCHECK_CLAIM_CONCURRENCY", 2).clamp(1, 4),
        qc_analytics_max_total_calls: env_usize("QC_ANALYTICS_MAX_TOTAL_CALLS", 12).clamp(4, 24),
        qc_analytics_max_query_calls: env_usize("QC_ANALYTICS_MAX_QUERY_CALLS", 10).clamp(2, 20),
        qc_analytics_query_timeout_secs: env_u64("QC_ANALYTICS_QUERY_TIMEOUT_SECS", 2).clamp(1, 15),
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        let telegram = load_telegram()?;
        let access = load_access();
        let db = load_db();
        let gemini = load_gemini();
        let openrouter = load_openrouter()?;
        let nvidia = load_nvidia()?;
        let ollama = load_ollama()?;
        let openai = load_openai()?;
        let codex = load_codex()?;
        let img2 = load_img2()?;
        let twitter = load_twitter()?;
        let external_media = load_external_media();
        validate_twitter_fetch_limits(
            twitter.fetch_total_timeout_secs,
            twitter.provider_timeout_secs,
            twitter.response_max_bytes,
            external_media.max_bytes,
            external_media.total_max_bytes,
        )?;
        let search = load_search()?;
        let jina = load_jina()?;
        let telegraph = load_telegraph();
        let cwd_pw = load_cwd_pw();
        let models = load_models();
        let limits = load_limits();
        let agents = load_agents();

        Ok(Config {
            log_level: env_string("LOG_LEVEL", "info").to_lowercase(),
            telegram,
            access,
            db,
            gemini,
            openrouter,
            nvidia,
            ollama,
            openai,
            codex,
            img2,
            twitter,
            external_media,
            search,
            jina,
            telegraph,
            cwd_pw,
            models,
            limits,
            agents,
        })
    }

    pub fn get_third_party_model_config(&self, model_id: &str) -> Option<&ThirdPartyModelConfig> {
        self.models.third_party_models_by_id.get(model_id)
    }

    pub fn is_third_party_provider_ready(&self, provider: ThirdPartyProvider) -> bool {
        match provider {
            ThirdPartyProvider::OpenRouter => {
                self.openrouter.enabled && !self.openrouter.api_key.trim().is_empty()
            }
            ThirdPartyProvider::Nvidia => {
                self.nvidia.enabled && !self.nvidia.api_key.trim().is_empty()
            }
            ThirdPartyProvider::Ollama => {
                self.ollama.enabled && !self.ollama.api_key.trim().is_empty()
            }
            ThirdPartyProvider::OpenAI => {
                self.openai.enabled && !self.openai.api_key.trim().is_empty()
            }
            ThirdPartyProvider::OpenAICodex => self.codex.enabled,
        }
    }

    pub fn gemini_api_available(&self) -> bool {
        gemini_api_available_from(self.gemini.enabled, &self.gemini.api_key)
    }

    pub fn img2_api_available(&self) -> bool {
        self.img2.enabled
            && !self.img2.api_key.trim().is_empty()
            && !self.img2.base_url.trim().is_empty()
    }

    /// Every configured credential that must never appear in operator-facing
    /// output (status/diagnose reports, log tails, etc.).
    pub fn secret_values(&self) -> Vec<&str> {
        [
            self.gemini.api_key.as_str(),
            self.openrouter.api_key.as_str(),
            self.nvidia.api_key.as_str(),
            self.ollama.api_key.as_str(),
            self.openai.api_key.as_str(),
            self.img2.api_key.as_str(),
            self.jina.api_key.as_str(),
            self.search.brave_api_key.as_str(),
            self.search.exa_api_key.as_str(),
            self.telegraph.access_token.as_str(),
            self.cwd_pw.api_key.as_str(),
            self.telegram.bot_token.as_str(),
        ]
        .into_iter()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .collect()
    }
}

pub(crate) fn gemini_api_available_from(enable_gemini: bool, api_key: &str) -> bool {
    enable_gemini && !api_key.trim().is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Environment variables are process-wide; hold this lock for the whole
    // span of any test that sets/removes one so tests can't race each other.
    static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Acquire `ENV_TEST_LOCK` for a test that is about to mutate the
    /// environment. `CONFIG` is a process-wide `LazyLock`: if some other
    /// test is the first to dereference it while our temporary env vars are
    /// set, `Config::load()` runs (and can panic) against our dirty
    /// environment, poisoning the static for every later test in the
    /// binary. Force it here, while still holding the lock and before any
    /// `set_var`, so it is always initialised from the clean environment.
    fn lock_env_and_force_config() -> std::sync::MutexGuard<'static, ()> {
        let guard = ENV_TEST_LOCK.lock().unwrap();
        // initialise CONFIG from the clean environment so no other test's
        // first access can observe our temporary values
        std::sync::LazyLock::force(&CONFIG);
        guard
    }

    /// RAII guard that removes an env var when dropped (including while a
    /// panic unwinds), so a test that sets one and then fails an assertion
    /// never leaves it behind for whichever test the runtime schedules
    /// next. Declare the `ENV_TEST_LOCK` guard first so this one, declared
    /// after, drops (and clears the var) before the lock is released.
    struct EnvVarGuard(&'static str);

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            unsafe {
                std::env::remove_var(self.0);
            }
        }
    }

    fn set_env_var_for_test(name: &'static str, value: &str) -> EnvVarGuard {
        unsafe {
            std::env::set_var(name, value);
        }
        EnvVarGuard(name)
    }

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

    #[test]
    fn env_f32_warns_once_and_returns_default_on_unparsable_value() {
        let _lock = lock_env_and_force_config();
        let _env = set_env_var_for_test("GEMINI_TEMPERATURE", "warm");
        let events = crate::utils::log_capture::capture_json_events(|| {
            assert_eq!(env_f32("GEMINI_TEMPERATURE", 0.7), 0.7);
        });

        assert_eq!(events.len(), 1, "{events:?}");
        let fields = &events[0]["fields"];
        assert_eq!(fields["env"], "GEMINI_TEMPERATURE");
        assert_eq!(fields["value"], "warm");
    }

    #[test]
    fn default_text_model_uses_configured_value_when_present() {
        assert_eq!(
            resolve_default_text_model_value(Some("openai-codex")),
            "openai-codex"
        );
    }

    #[test]
    fn default_text_model_defaults_to_gemini_when_unset_or_blank() {
        assert_eq!(resolve_default_text_model_value(None), "gemini");
        assert_eq!(resolve_default_text_model_value(Some("   ")), "gemini");
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

    #[test]
    fn loopback_http_base_accepts_https_and_loopback_http_but_rejects_other_http_hosts() {
        // A valid https URL still goes through the same rules as validate_https_base.
        assert_eq!(
            validate_http_base_allowing_loopback("OLLAMA_BASE_URL", "https://ollama.com/v1".into())
                .unwrap(),
            "https://ollama.com/v1"
        );
        // The common local-Ollama case: plain http on a loopback host/port.
        for value in [
            "http://localhost:11434/v1",
            "http://127.0.0.1:11434/v1",
            "http://[::1]:11434/v1",
        ] {
            assert_eq!(
                validate_http_base_allowing_loopback("OLLAMA_BASE_URL", value.into()).unwrap(),
                value
            );
        }
        // Non-loopback http, and loopback URLs with credentials/query/fragment, are rejected.
        for value in [
            "http://example.com",
            "http://user@localhost:11434/v1",
            "http://localhost:11434/v1?debug=1",
            "http://localhost:11434/v1#fragment",
        ] {
            assert!(validate_http_base_allowing_loopback("OLLAMA_BASE_URL", value.into()).is_err());
        }
    }

    #[test]
    fn config_load_rejects_plain_http_provider_endpoints() {
        let _lock = lock_env_and_force_config();
        for name in [
            "OPENROUTER_BASE_URL",
            "NVIDIA_BASE_URL",
            "OPENAI_BASE_URL",
            "OPENAI_CODEX_BASE_URL",
            "BRAVE_SEARCH_ENDPOINT",
            "EXA_SEARCH_ENDPOINT",
            "JINA_SEARCH_ENDPOINT",
            "IMG2_BASE_URL",
        ] {
            let _env = set_env_var_for_test(name, "http://example.com");
            let result = Config::load();
            let err = result.expect_err(&format!("{name} must reject a plain-http endpoint"));
            assert!(err.to_string().contains(name), "{name}: {err}");
        }
    }

    #[test]
    fn config_load_rejects_non_loopback_http_ollama_endpoint() {
        let _lock = lock_env_and_force_config();
        let _env = set_env_var_for_test("OLLAMA_BASE_URL", "http://example.com");
        let result = Config::load();
        let err = result.expect_err("non-loopback http Ollama endpoint must be rejected");
        assert!(err.to_string().contains("OLLAMA_BASE_URL"), "{err}");
    }

    #[test]
    fn config_load_accepts_ollama_over_loopback_http() {
        let _lock = lock_env_and_force_config();
        let _env = set_env_var_for_test("OLLAMA_BASE_URL", "http://localhost:11434/v1");
        let result = Config::load();
        let config = result.expect("loopback Ollama endpoint should be accepted");
        assert_eq!(config.ollama.base_url, "http://localhost:11434/v1");
    }

    #[test]
    fn secret_values_include_every_provider_credential() {
        let mut config = (*CONFIG).clone();
        config.ollama.api_key = "ollama-secret".to_string();
        config.img2.api_key = "img2-secret".to_string();
        config.telegram.bot_token = "bot-secret".to_string();
        let secrets = config.secret_values();
        for expected in ["ollama-secret", "img2-secret", "bot-secret"] {
            assert!(secrets.contains(&expected), "missing {expected}");
        }
    }
}
