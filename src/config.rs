use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use once_cell::sync::Lazy;
use serde::Deserialize;
use tracing::{info, warn};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    pub images: bool,
    pub video: bool,
    pub audio: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenRouterModelsFile {
    models: Vec<OpenRouterModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenRouterModelEntry {
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
pub struct OpenRouterModelConfig {
    pub name: String,
    pub model: String,
    pub image: bool,
    pub video: bool,
    pub audio: bool,
    pub tools: bool,
}

impl OpenRouterModelConfig {
    #[allow(dead_code)]
    pub fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            images: self.image,
            video: self.video,
            audio: self.audio,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Config {
    pub bot_token: String,
    pub log_level: String,
    pub database_url: String,
    pub gemini_api_key: String,
    pub gemini_model: String,
    pub gemini_pro_model: String,
    pub gemini_image_model: String,
    pub gemini_video_model: String,
    pub gemini_temperature: f32,
    pub gemini_top_k: i32,
    pub gemini_top_p: f32,
    pub gemini_max_output_tokens: i32,
    pub gemini_thinking_level: String,
    pub gemini_safety_settings: String,
    pub vertex_project_id: String,
    pub vertex_location: String,
    pub use_vertex_video: bool,
    pub vertex_video_model: String,
    pub use_vertex_image: bool,
    pub vertex_image_model: String,
    pub enable_openrouter: bool,
    pub openrouter_api_key: String,
    pub openrouter_base_url: String,
    pub openrouter_alpha_base_url: String,
    pub openrouter_temperature: f32,
    pub openrouter_top_k: i32,
    pub openrouter_top_p: f32,
    pub enable_jina_mcp: bool,
    pub jina_ai_api_key: String,
    pub jina_search_endpoint: String,
    pub jina_reader_endpoint: String,
    pub enable_exa_search: bool,
    pub exa_api_key: String,
    pub exa_search_endpoint: String,
    pub llama_model: String,
    pub grok_model: String,
    pub qwen_model: String,
    pub deepseek_model: String,
    pub gpt_model: String,
    pub rate_limit_seconds: u64,
    pub model_selection_timeout: u64,
    pub default_q_model: String,
    pub telegram_max_length: usize,
    pub telegraph_access_token: String,
    pub telegraph_author_name: String,
    pub telegraph_author_url: String,
    pub user_history_message_count: i64,
    pub cwd_pw_api_key: String,
    pub support_message: String,
    pub support_link: String,
    pub whitelist_file_path: String,
    pub access_controlled_commands: Vec<String>,
    pub openrouter_models_config_path: PathBuf,
    pub openrouter_models: Vec<OpenRouterModelConfig>,
    pub openrouter_models_by_model: HashMap<String, OpenRouterModelConfig>,
}

pub static CONFIG: Lazy<Config> = Lazy::new(|| {
    Config::load().expect("Failed to load configuration")
});

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

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
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

fn resolve_openrouter_models_path() -> PathBuf {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(env_value) = env::var("OPENROUTER_MODELS_CONFIG_PATH") {
        let env_path = PathBuf::from(env_value);
        if env_path.is_absolute() {
            candidates.push(env_path);
        } else {
            candidates.push(PathBuf::from(env::current_dir().unwrap_or_else(|_| PathBuf::from("."))).join(env_path));
        }
    }
    candidates.push(PathBuf::from("openrouter_models.json"));
    candidates.push(PathBuf::from("bot").join("openrouter_models.json"));

    for candidate in &candidates {
        if candidate.exists() {
            return candidate.to_path_buf();
        }
    }

    candidates
        .get(0)
        .cloned()
        .unwrap_or_else(|| PathBuf::from("openrouter_models.json"))
}

fn load_openrouter_models_from_path(path: &Path) -> Vec<OpenRouterModelConfig> {
    if !path.exists() {
        info!("OpenRouter model config not found at {}", path.display());
        return Vec::new();
    }

    let raw = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(err) => {
            info!("Failed to read OpenRouter model config at {}: {}", path.display(), err);
            return Vec::new();
        }
    };

    let parsed: OpenRouterModelsFile = match serde_json::from_str(&raw) {
        Ok(data) => data,
        Err(err) => {
            info!("Failed to parse OpenRouter model config at {}: {}", path.display(), err);
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
        models.push(OpenRouterModelConfig {
            name: name.to_string(),
            model: model.to_string(),
            image: entry.image.unwrap_or(false),
            video: entry.video.unwrap_or(false),
            audio: entry.audio.unwrap_or(false),
            tools: entry.tools.unwrap_or(true),
        });
    }
    models
}

fn load_legacy_openrouter_models(config: &LegacyOpenRouterEnv) -> Vec<OpenRouterModelConfig> {
    let legacy_entries: Vec<(&str, &str, bool, bool, bool, bool)> = vec![
        ("Llama 4", &config.llama_model, true, false, false, true),
        ("Grok 4", &config.grok_model, true, false, false, true),
        ("Qwen 3", &config.qwen_model, false, false, false, true),
        ("DeepSeek 3.1", &config.deepseek_model, false, false, false, false),
        ("GPT", &config.gpt_model, true, false, false, true),
    ];

    let mut models = Vec::new();
    for (name, model_id, image, video, audio, tools) in legacy_entries {
        if !model_id.trim().is_empty() {
            models.push(OpenRouterModelConfig {
                name: name.to_string(),
                model: model_id.to_string(),
                image,
                video,
                audio,
                tools,
            });
        }
    }
    models
}

fn build_openrouter_models(
    path: &Path,
    legacy_env: &LegacyOpenRouterEnv,
) -> Vec<OpenRouterModelConfig> {
    let models = load_openrouter_models_from_path(path);
    if !models.is_empty() {
        info!("Loaded {} OpenRouter model(s) from {}", models.len(), path.display());
        return models;
    }
    let legacy_models = load_legacy_openrouter_models(legacy_env);
    if !legacy_models.is_empty() {
        info!("Using legacy OpenRouter model configuration with {} model(s)", legacy_models.len());
    } else {
        info!("No OpenRouter models configured via JSON or environment variables");
    }
    legacy_models
}

fn resolve_model_by_keyword(value: &str, models: &[OpenRouterModelConfig], keywords: &[&str]) -> String {
    if !value.trim().is_empty() {
        return value.to_string();
    }

    let lowered: Vec<String> = keywords.iter().map(|k| k.to_lowercase()).collect();
    for config_entry in models {
        let haystack = format!("{} {}", config_entry.name, config_entry.model).to_lowercase();
        if lowered.iter().all(|keyword| haystack.contains(keyword)) {
            return config_entry.model.clone();
        }
    }

    value.to_string()
}

#[derive(Debug, Clone)]
struct LegacyOpenRouterEnv {
    llama_model: String,
    grok_model: String,
    qwen_model: String,
    deepseek_model: String,
    gpt_model: String,
}

impl Config {
    pub fn load() -> Result<Self> {
        let bot_token = env::var("BOT_TOKEN").unwrap_or_default();
        if bot_token.trim().is_empty() {
            return Err(anyhow::anyhow!("BOT_TOKEN is required"));
        }

        let legacy_env = LegacyOpenRouterEnv {
            llama_model: env_string("LLAMA_MODEL", ""),
            grok_model: env_string("GROK_MODEL", ""),
            qwen_model: env_string("QWEN_MODEL", ""),
            deepseek_model: env_string("DEEPSEEK_MODEL", ""),
            gpt_model: env_string("GPT_MODEL", ""),
        };

        let openrouter_models_config_path = resolve_openrouter_models_path();
        let openrouter_models = build_openrouter_models(&openrouter_models_config_path, &legacy_env);
        let openrouter_models_by_model = openrouter_models
            .iter()
            .cloned()
            .map(|model| (model.model.clone(), model))
            .collect::<HashMap<_, _>>();

        let llama_model = resolve_model_by_keyword(&legacy_env.llama_model, &openrouter_models, &["llama"]);
        let grok_model = resolve_model_by_keyword(&legacy_env.grok_model, &openrouter_models, &["grok"]);
        let qwen_model = resolve_model_by_keyword(&legacy_env.qwen_model, &openrouter_models, &["qwen"]);
        let deepseek_model = resolve_model_by_keyword(&legacy_env.deepseek_model, &openrouter_models, &["deepseek"]);
        let gpt_model = resolve_model_by_keyword(&legacy_env.gpt_model, &openrouter_models, &["gpt"]);

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

        Ok(Config {
            bot_token,
            log_level: env_string("LOG_LEVEL", "info").to_lowercase(),
            database_url: normalize_database_url(env_string("DATABASE_URL", "sqlite+aiosqlite:///bot.db")),
            gemini_api_key: env_string("GEMINI_API_KEY", ""),
            gemini_model: env_string("GEMINI_MODEL", "gemini-2.0-flash"),
            gemini_pro_model: env_string("GEMINI_PRO_MODEL", "gemini-2.5-pro-exp-03-25"),
            gemini_image_model: env_string("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview"),
            gemini_video_model: env_string("GEMINI_VIDEO_MODEL", "veo-2.0-generate-001"),
            gemini_temperature: env_f32("GEMINI_TEMPERATURE", 0.7),
            gemini_top_k: env_i32("GEMINI_TOP_K", 40),
            gemini_top_p: env_f32("GEMINI_TOP_P", 0.95),
            gemini_max_output_tokens: env_i32("GEMINI_MAX_OUTPUT_TOKENS", 2048),
            gemini_thinking_level: env_string("GEMINI_THINKING_LEVEL", "high"),
            gemini_safety_settings: normalize_gemini_safety_settings(env_string("GEMINI_SAFETY_SETTINGS", "permissive")),
            vertex_project_id: env_string("VERTEX_PROJECT_ID", ""),
            vertex_location: env_string("VERTEX_LOCATION", ""),
            use_vertex_video: env_bool("USE_VERTEX_VIDEO", false),
            vertex_video_model: env_string("VERTEX_VIDEO_MODEL", ""),
            use_vertex_image: env_bool("USE_VERTEX_IMAGE", false),
            vertex_image_model: env_string("VERTEX_IMAGE_MODEL", ""),
            enable_openrouter: env_bool("ENABLE_OPENROUTER", true),
            openrouter_api_key: env_string("OPENROUTER_API_KEY", ""),
            openrouter_base_url: env_string("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openrouter_alpha_base_url: env_string("OPENROUTER_ALPHA_BASE_URL", "https://openrouter.ai/api/alpha"),
            openrouter_temperature: env_f32("OPENROUTER_TEMPERATURE", 0.7),
            openrouter_top_k: env_i32("OPENROUTER_TOP_K", 40),
            openrouter_top_p: env_f32("OPENROUTER_TOP_P", 0.95),
            enable_jina_mcp: env_bool("ENABLE_JINA_MCP", false),
            jina_ai_api_key: env_string("JINA_AI_API_KEY", ""),
            jina_search_endpoint: env_string("JINA_SEARCH_ENDPOINT", "https://s.jina.ai/search"),
            jina_reader_endpoint: env_string("JINA_READER_ENDPOINT", "https://r.jina.ai/"),
            enable_exa_search: env_bool("ENABLE_EXA_SEARCH", true),
            exa_api_key: env_string("EXA_API_KEY", ""),
            exa_search_endpoint: env_string("EXA_SEARCH_ENDPOINT", "https://api.exa.ai/search"),
            llama_model,
            grok_model,
            qwen_model,
            deepseek_model,
            gpt_model,
            rate_limit_seconds: env_u64("RATE_LIMIT_SECONDS", 15),
            model_selection_timeout: env_u64("MODEL_SELECTION_TIMEOUT", 30),
            default_q_model: env_string("DEFAULT_Q_MODEL", "gemini").to_lowercase(),
            telegram_max_length: env_usize("TELEGRAM_MAX_LENGTH", 4000),
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
            openrouter_models_config_path,
            openrouter_models,
            openrouter_models_by_model,
        })
    }

    pub fn iter_openrouter_models(&self) -> &[OpenRouterModelConfig] {
        &self.openrouter_models
    }

    pub fn get_openrouter_model_config(&self, model_name: &str) -> Option<&OpenRouterModelConfig> {
        self.openrouter_models_by_model.get(model_name)
    }
}

pub const TLDR_SYSTEM_PROMPT: &str = r#"你是一个AI助手，名叫{bot_name}，请用中文总结以下群聊内容。
请先汇总出群聊主要内容。
再依据发言数量依次列出主要发言用户的名字和观点但不要超过10位用户。
请尽量详细地表述每个人的对各个议题的观点和陈述，字数不限。
非常关键：如果群聊内容中出现投资相关信息，请在总结后再全文最后逐项列出。格式为：投资标的物：投资建议 [由哪位用户提出]。
"#;

pub const FACTCHECK_SYSTEM_PROMPT: &str = "You are an expert fact-checker that is unbiased, honest, and direct. Your job is to evaluate the factual accuracy of the text provided.\n\nFor each significant claim, verify using web search results:\n1. Analyze each claim objectively\n2. Provide a judgment on its accuracy (True, False, Partially True, or Insufficient Evidence)\n3. Briefly explain your reasoning with citations to the sources found through web search\n4. When a claim is not factually accurate, provide corrections\n5. IMPORTANT: The current UTC date and time is {current_datetime}. Verify all temporal claims relative to this date and time.\n6. CRITICAL: List the sources you used to check the facts with links.\n7. CRITICAL: Always respond in the same language as the user's message or the language from the image.\n8. Format your response in an easily readable way using Markdown where appropriate.\n\nAlways cite your sources and only draw definitive conclusions when you have sufficient reliable evidence.\n";

pub const Q_SYSTEM_PROMPT: &str = "You are a helpful assistant in a Telegram group chat. You provide concise, factual, and helpful answers to users' questions.\n\nGuidelines for your responses:\n1. Provide a direct, clear answer to the question.\n2. Be concise but comprehensive.\n3. Fact-check your information using web search and include citations to reliable sources.\n4. When the question asks for technical information, provide accurate and up-to-date information.\n5. IMPORTANT: Use web search to verify all facts and information before answering.\n6. CRITICAL: The current UTC date and time is {current_datetime}. Always verify current political leadership, office holders, and recent events through web search based on this date and time.\n7. If there's uncertainty, acknowledge it and explain the limitations.\n8. Format your response in an easily readable way using Markdown where appropriate.\n9. Keep your response under 400 words unless a detailed explanation is necessary.\n10. If the answer requires multiple parts, use numbered or bulleted lists.\n11. CRITICAL: Respond in {language} language unless you are told otherwise.\n\nRemember to be helpful and accurate in your responses. But do not be too nice and agreeable. If necessary, do not be afraid to be critical.\n";

pub const PROFILEME_SYSTEM_PROMPT: &str = "You are an experienced professional profiler. Based on the following chat history of a user in a group chat, generate a concise and insightful user profile. The profile must highlight their communication style, potential interests, key personality traits, and how they typically interact in the group. Focus on patterns and recurring themes. Address the user directly (e.g., 'You seem to be...').Do not include any specific message content, timestamps or message IDs.The user is asking for their own profile.CRITICAL: Always reply in Chinese";

pub const PAINTME_SYSTEM_PROMPT: &str = "You are a highly creative AI specializing in crafting evocative and detailed image prompts. Based on the following chat history of a user, generate a concise, yet descriptive and imaginative image prompt that metaphorically represents the user's personality, communication style, or recurring themes found in their messages. This prompt will be used by an AI image generation model to create a visual representation of the user. Focus on symbolism, artistic interpretation, and sensory details. Do not refer to the user directly or mention the chat itself in the prompt. The final prompt should be a descriptive phrase or a short paragraph. Strive for prompts that can lead to realistic, artistic, or conceptually rich images.\n\nThe prompt is used for Gemini image model, so the prompt should be in English.";

pub const PORTRAIT_SYSTEM_PROMPT: &str = "You are a highly creative AI specializing in crafting evocative and detailed image prompts for artistic portraits. Based on the following chat history of a user, generate a concise, yet descriptive and imaginative image prompt that creates a portrait of the user. The prompt should translate the user's personality, communication style, or recurring themes into the artistic style, mood, lighting, and background of the portrait. This prompt will be used by an AI image generation model to create a visual representation of the user themselves. Focus on artistic interpretation and sensory details to produce a compelling image. Do not refer to the user by a name or mention the chat itself in the prompt. The final prompt should be a descriptive phrase or a short paragraph.The portrait should be a realistic portrait of the user, not a cartoon or a stylized portrait.The prompt is used for Gemini image model, so the prompt should be in English.";
