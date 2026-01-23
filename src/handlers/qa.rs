use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::{
    InlineKeyboardButton, InlineKeyboardMarkup, MessageEntityRef, MessageId, ParseMode,
    ReplyParameters,
};
use whatlang::detect;

use crate::config::{
    CONFIG, Q_SYSTEM_PROMPT,
};
use crate::db::database::build_message_insert;
use crate::handlers::access::{check_access_control, is_rate_limited};
use crate::handlers::content::{
    download_telegraph_media, download_twitter_media, extract_telegraph_urls_and_content,
    extract_twitter_urls_and_content, extract_youtube_urls,
};
use crate::handlers::media::{collect_message_media, MediaCollectionOptions};
use crate::handlers::responses::send_response;
use crate::llm::{call_gemini, call_openrouter};
use crate::state::{AppState, PendingQRequest};
use crate::utils::timing::{complete_command_timer, start_command_timer};

pub const MODEL_CALLBACK_PREFIX: &str = "model_select:";
pub const MODEL_GEMINI: &str = "gemini";
const MIN_LANGUAGE_CONFIDENCE: f64 = 0.6;

fn now_unix_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn message_entities_for_text(message: &Message) -> Option<Vec<MessageEntityRef<'_>>> {
    if message.text().is_some() {
        message.parse_entities()
    } else {
        message.parse_caption_entities()
    }
}

fn detect_language_name(text: &str) -> Option<String> {
    let info = detect(text.trim())?;
    if !info.is_reliable() || info.confidence() < MIN_LANGUAGE_CONFIDENCE {
        return None;
    }
    Some(info.lang().eng_name().to_string())
}

fn resolve_alias_to_model_id(alias: &str) -> Option<String> {
    let alias = alias.trim().to_lowercase();
    if alias.is_empty() {
        return None;
    }
    if alias == MODEL_GEMINI {
        return Some(MODEL_GEMINI.to_string());
    }

    let mapping = [
        ("llama", &CONFIG.llama_model),
        ("grok", &CONFIG.grok_model),
        ("qwen", &CONFIG.qwen_model),
        ("deepseek", &CONFIG.deepseek_model),
        ("gpt", &CONFIG.gpt_model),
    ];

    for (token, model) in mapping {
        if alias == token && !model.trim().is_empty() {
            return Some(model.clone());
        }
    }

    for config in CONFIG.iter_openrouter_models() {
        let haystack = format!("{} {}", config.name, config.model).to_lowercase();
        if haystack.contains(&alias) {
            return Some(config.model.clone());
        }
    }

    None
}

fn normalize_model_identifier(identifier: &str) -> String {
    let stripped = identifier.trim();
    if stripped.is_empty() {
        return MODEL_GEMINI.to_string();
    }
    if stripped.eq_ignore_ascii_case(MODEL_GEMINI) {
        return MODEL_GEMINI.to_string();
    }

    if let Some(resolved) = resolve_alias_to_model_id(stripped) {
        return resolved;
    }

    if CONFIG.get_openrouter_model_config(stripped).is_some() {
        return stripped.to_string();
    }

    stripped.to_string()
}

fn is_openrouter_available() -> bool {
    CONFIG.enable_openrouter && !CONFIG.openrouter_api_key.trim().is_empty()
}

fn is_model_configured(model_key: &str) -> bool {
    let normalized = normalize_model_identifier(model_key);
    if normalized == MODEL_GEMINI {
        return true;
    }
    CONFIG.get_openrouter_model_config(&normalized).is_some()
}

pub fn create_model_selection_keyboard(
    has_images: bool,
    has_video: bool,
    has_audio: bool,
) -> InlineKeyboardMarkup {
    let mut keyboard: Vec<Vec<InlineKeyboardButton>> = Vec::new();
    let gemini_button = InlineKeyboardButton::callback(
        "Gemini 3",
        format!("{}{}", MODEL_CALLBACK_PREFIX, MODEL_GEMINI),
    );

    let mut first_row = vec![gemini_button];
    let mut openrouter_buttons = Vec::new();

    for config in CONFIG.iter_openrouter_models() {
        if has_images && !config.image {
            continue;
        }
        if has_video && !config.video {
            continue;
        }
        if has_audio && !config.audio {
            continue;
        }

        let model_identifier = config.model.trim();
        if model_identifier.is_empty() {
            continue;
        }
        openrouter_buttons.push(InlineKeyboardButton::callback(
            config.name.clone(),
            format!("{}{}", MODEL_CALLBACK_PREFIX, model_identifier),
        ));
    }

    if !openrouter_buttons.is_empty() {
        first_row.push(openrouter_buttons.remove(0));
    }
    keyboard.push(first_row);

    for chunk in openrouter_buttons.chunks(2) {
        keyboard.push(chunk.to_vec());
    }

    InlineKeyboardMarkup::new(keyboard)
}

fn build_system_prompt(language: &str) -> String {
    let now = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string();
    Q_SYSTEM_PROMPT
        .replace("{current_datetime}", &now)
        .replace("{language}", language)
}

#[allow(deprecated)]
async fn process_request(
    bot: &Bot,
    _state: &AppState,
    request: PendingQRequest,
    model_name: &str,
) -> Result<()> {
    let system_prompt = build_system_prompt(&request.language);

    let mut query = request.query.clone();
    for content in &request.telegraph_contents {
        query.push_str("\n\n");
        query.push_str(content);
    }
    for content in &request.twitter_contents {
        query.push_str("\n\n");
        query.push_str(content);
    }

    let supports_tools = if model_name == MODEL_GEMINI {
        true
    } else {
        CONFIG
            .get_openrouter_model_config(model_name)
            .map(|config| config.tools)
            .unwrap_or(false)
    };

    let response = if model_name == MODEL_GEMINI {
        let use_pro = !request.image_data_list.is_empty()
            || request.video_data.is_some()
            || request.audio_data.is_some()
            || !request.youtube_urls.is_empty();
        call_gemini(
            &system_prompt,
            &query,
            Some(&request.language),
            true,
            false,
            Some(&CONFIG.gemini_thinking_level),
            None,
            use_pro,
            Some(request.image_data_list.clone()),
            request.video_data.clone(),
            request.audio_data.clone(),
            Some(request.youtube_urls.clone()),
        )
        .await?
    } else {
        call_openrouter(
            &system_prompt,
            &query,
            model_name,
            "Answer to Your Question",
            &request.image_data_list,
            supports_tools,
        )
        .await?
    };

    if response.trim().is_empty() {
        bot.edit_message_text(ChatId(request.chat_id), MessageId(request.selection_message_id as i32), "I couldn't find an answer to your question. Please try rephrasing or asking something else.")
            .await?;
        return Ok(());
    }

    let mut response_text = response;
    if !model_name.is_empty() {
        let display_model = if model_name == MODEL_GEMINI {
            if !request.image_data_list.is_empty()
                || request.video_data.is_some()
                || request.audio_data.is_some()
                || !request.youtube_urls.is_empty()
            {
                CONFIG.gemini_pro_model.as_str()
            } else {
                CONFIG.gemini_model.as_str()
            }
        } else {
            model_name
        };
        response_text.push_str(&format!("\n\nModel: {}", display_model));
    }

    send_response(
        bot,
        ChatId(request.chat_id),
        MessageId(request.selection_message_id as i32),
        &response_text,
        "Answer to Your Question",
        ParseMode::Markdown,
    )
    .await?;

    Ok(())
}

#[allow(deprecated)]
pub async fn q_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    query: Option<String>,
    force_gemini: bool,
    command_name: &str,
) -> Result<()> {
    if !check_access_control(&bot, &message, command_name).await {
        return Ok(());
    }

    let user_id = message
        .from.as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        bot.send_message(message.chat.id, "You're sending commands too quickly. Please wait a moment before trying again.")
            .reply_parameters(ReplyParameters::new(message.id))
            .await?;
        return Ok(());
    }

    let query_text_raw = query.unwrap_or_default();
    let query_entities = message_entities_for_text(&message);
    let reply_message = message.reply_to_message();
    let mut reply_text_raw = String::new();
    let mut reply_text = String::new();
    let mut telegraph_contents = Vec::new();
    let mut twitter_contents = Vec::new();

    if let Some(reply) = reply_message {
        reply_text_raw = reply
            .text()
            .map(|value| value.to_string())
            .or_else(|| reply.caption().map(|value| value.to_string()))
            .unwrap_or_default();
        if !reply_text_raw.trim().is_empty() {
            let reply_entities = message_entities_for_text(reply);
            let (reply_text_processed, reply_telegraph) =
                extract_telegraph_urls_and_content(
                    &reply_text_raw,
                    reply_entities.as_deref(),
                    5,
                )
                .await;
            let (reply_text_processed, reply_twitter) =
                extract_twitter_urls_and_content(
                    &reply_text_processed,
                    reply_entities.as_deref(),
                    5,
                )
                .await;
            telegraph_contents.extend(reply_telegraph);
            twitter_contents.extend(reply_twitter);
            reply_text = reply_text_processed;
        }
    }

    let original_query = if query_text_raw.trim().is_empty() {
        reply_text_raw.clone()
    } else {
        query_text_raw.clone()
    };

    if original_query.trim().is_empty() {
        bot.send_message(message.chat.id, "Please provide a question or reply to a message with /q.")
            .reply_parameters(ReplyParameters::new(message.id))
            .await?;
        return Ok(());
    }

    let mut query_text = query_text_raw.clone();
    if !query_text.trim().is_empty() {
        let (query_text_processed, query_telegraph) =
            extract_telegraph_urls_and_content(&query_text, query_entities.as_deref(), 5).await;
        let (query_text_processed, query_twitter) =
            extract_twitter_urls_and_content(&query_text_processed, query_entities.as_deref(), 5)
                .await;
        telegraph_contents.extend(query_telegraph);
        twitter_contents.extend(query_twitter);
        query_text = query_text_processed;
    }

    let query_base = if query_text.trim().is_empty() {
        reply_text.clone()
    } else if reply_text.trim().is_empty() {
        query_text.clone()
    } else {
        format!(
            "Context from replied message: \"{}\"\n\nQuestion: {}",
            reply_text, query_text
        )
    };

    let (query_text, youtube_urls) = extract_youtube_urls(&query_base, 10);

    let language = detect_language_name(&original_query)
        .or_else(|| detect_language_name(&query_text))
        .unwrap_or_else(|| "English".to_string());

    let username = message
        .from.as_ref()
        .map(|user| user.full_name())
        .unwrap_or_else(|| "Anonymous".to_string());

    let db_query_text = if let Some(reply) = message.reply_to_message() {
        let replied_text = reply
            .text()
            .map(|value| value.to_string())
            .or_else(|| reply.caption().map(|value| value.to_string()))
            .unwrap_or_default();
        if replied_text.is_empty() {
            query_text.clone()
        } else {
            format!(
                "Context from replied message: \"{}\"\n\nQuestion: {}",
                replied_text, query_text
            )
        }
    } else {
        query_text.clone()
    };

    let media = collect_message_media(&bot, &state, &message, MediaCollectionOptions::for_qa()).await;
    let mut image_data_list = media.images;
    let mut video_data = media.video;
    let mut video_mime_type = media.video_mime_type;
    let audio_data = media.audio;
    let audio_mime_type = media.audio_mime_type;

    let (telegraph_images, telegraph_video, telegraph_video_mime) = download_telegraph_media(&telegraph_contents, 5, 1).await;
    image_data_list.extend(telegraph_images);
    if video_data.is_none() {
        video_data = telegraph_video;
        video_mime_type = telegraph_video_mime;
    }

    let (twitter_images, twitter_video, twitter_video_mime) = download_twitter_media(&twitter_contents, 5, 1).await;
    image_data_list.extend(twitter_images);
    if video_data.is_none() {
        video_data = twitter_video;
        video_mime_type = twitter_video_mime;
    }

    let has_images = !image_data_list.is_empty();
    let has_video = video_data.is_some();
    let has_audio = audio_data.is_some();

    let must_use_gemini = force_gemini
        || !youtube_urls.is_empty()
        || !is_openrouter_available()
        || CONFIG.iter_openrouter_models().is_empty();
    if must_use_gemini {
        let processing_message_text = if has_video {
            "Analyzing video and processing your question...".to_string()
        } else if has_audio {
            "Analyzing audio and processing your question...".to_string()
        } else if has_images {
            format!(
                "Analyzing {} image(s) and processing your question...",
                image_data_list.len()
            )
        } else if !twitter_contents.is_empty() {
            format!(
                "Analyzing {} Twitter post(s) and processing your question...",
                twitter_contents.len()
            )
        } else if !youtube_urls.is_empty() {
            format!(
                "Analyzing {} YouTube video(s) and processing your question...",
                youtube_urls.len()
            )
        } else {
            "Processing your question...".to_string()
        };
        let processing_message = bot
            .send_message(message.chat.id, processing_message_text)
            .reply_parameters(ReplyParameters::new(message.id))
            .await?;
        let mut timer = start_command_timer(command_name, &message);
        let use_pro = has_images || has_video || has_audio || !youtube_urls.is_empty();
        let response = call_gemini(
            &build_system_prompt(&language),
            &query_text,
            Some(&language),
            true,
            false,
            Some(&CONFIG.gemini_thinking_level),
            None,
            use_pro,
            Some(image_data_list.clone()),
            video_data.clone(),
            audio_data.clone(),
            Some(youtube_urls.clone()),
        )
        .await?;

        send_response(
            &bot,
            processing_message.chat.id,
            processing_message.id,
            &response,
            "Answer to Your Question",
            ParseMode::Markdown,
        )
        .await?;
        complete_command_timer(&mut timer, "success", None);
        return Ok(());
    }

    let has_media = has_images || has_video || has_audio;
    let mut selection_text = "Please select which AI model to use for your question:".to_string();
    if has_media {
        selection_text.push_str("\n\n*Note: Only models that support media are shown.*");
    }

    let keyboard = create_model_selection_keyboard(has_images, has_video, has_audio);
    let selection_message = bot
        .send_message(message.chat.id, selection_text)
        .reply_parameters(ReplyParameters::new(message.id))
        .reply_markup(keyboard)
        .parse_mode(ParseMode::Markdown)
        .await?;

    let request_key = format!("{}_{}", message.chat.id.0, selection_message.id.0);
    let timer = start_command_timer(command_name, &message);

    let pending_request = PendingQRequest {
        user_id,
        username: username.clone(),
        query: query_text.clone(),
        original_query: original_query.clone(),
        db_query_text: db_query_text.clone(),
        language: language.clone(),
        image_data_list,
        video_data,
        video_mime_type,
        audio_data,
        audio_mime_type,
        youtube_urls,
        telegraph_contents: telegraph_contents.iter().map(|c| c.text_content.clone()).collect(),
        twitter_contents: twitter_contents.iter().map(|c| c.text_content.clone()).collect(),
        chat_id: message.chat.id.0,
        message_id: message.id.0 as i64,
        selection_message_id: selection_message.id.0 as i64,
        original_user_id: user_id,
        reply_to_message_id: message.reply_to_message().map(|msg| msg.id.0 as i64),
        timestamp: now_unix_seconds(),
        command_timer: Some(timer),
    };

    state.pending_q_requests.lock().insert(request_key.clone(), pending_request);

    let bot_clone = bot.clone();
    let state_clone = state.clone();
    tokio::spawn(async move {
        handle_model_timeout(bot_clone, state_clone, request_key).await;
    });

    let db_insert = build_message_insert(
        Some(user_id),
        Some(username),
        Some(format!(
            "Ask {}: {}",
            if CONFIG.telegraph_author_name.trim().is_empty() {
                "TelegramGroupHelperBot"
            } else {
                &CONFIG.telegraph_author_name
            },
            db_query_text
        )),
        Some(language),
        message.date,
        message.reply_to_message().map(|msg| msg.id.0 as i64),
        Some(message.chat.id.0),
        Some(message.id.0 as i64),
    );
    let _ = state.db.queue_message_insert(db_insert).await;

    Ok(())
}

pub async fn qq_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    query: Option<String>,
) -> Result<()> {
    q_handler(bot, state, message, query, true, "qq").await
}

pub async fn handle_model_timeout(bot: Bot, state: AppState, request_key: String) {
    tokio::time::sleep(Duration::from_secs(CONFIG.model_selection_timeout)).await;
    let request = state.pending_q_requests.lock().remove(&request_key);
    let Some(mut request) = request else {
        return;
    };

    let default_model = normalize_model_identifier(&CONFIG.default_q_model);
    if default_model != MODEL_GEMINI && !is_model_configured(&default_model) {
        return;
    }

    let _ = bot
        .edit_message_text(ChatId(request.chat_id), MessageId(request.selection_message_id as i32), "No model selected in time. Using default model...")
        .await;

    let command_timer = request.command_timer.take();
    let result = process_request(&bot, &state, request, &default_model).await;
    if let Some(mut timer) = command_timer {
        let status = if result.is_ok() { "success" } else { "error" };
        complete_command_timer(&mut timer, status, Some("timeout_default_model".to_string()));
    }
}

pub async fn model_selection_callback(
    bot: Bot,
    state: AppState,
    query: CallbackQuery,
) -> Result<()> {
    bot.answer_callback_query(query.id.clone()).await?;

    let Some(data) = &query.data else {
        return Ok(());
    };
    if !data.starts_with(MODEL_CALLBACK_PREFIX) {
        return Ok(());
    }

    let selected_token = data.trim_start_matches(MODEL_CALLBACK_PREFIX);
    let selected_model = normalize_model_identifier(selected_token);

    let message = match query.message.clone() {
        Some(msg) => msg,
        None => return Ok(()),
    };

    let request_key = format!("{}_{}", message.chat().id.0, message.id().0);
    let request = {
        let mut pending = state.pending_q_requests.lock();
        pending.remove(&request_key)
    };
    let mut request = match request {
        Some(req) => req,
        None => {
            bot.edit_message_text(message.chat().id, message.id(), "This request has expired.")
                .await?;
            return Ok(());
        }
    };

    if !is_model_configured(selected_token) {
        bot.answer_callback_query(query.id.clone()).await?;
        return Ok(());
    }

    let query_user_id = i64::try_from(query.from.id.0).unwrap_or_default();
    if request.original_user_id != query_user_id {
        bot.answer_callback_query(query.id.clone()).await?;
        return Ok(());
    }

    if now_unix_seconds() - request.timestamp > CONFIG.model_selection_timeout as i64 {
        if let Some(mut timer) = request.command_timer.take() {
            complete_command_timer(&mut timer, "expired", Some("selection_timeout".to_string()));
        }
        bot.edit_message_text(message.chat().id, message.id(), "Selection timed out. Please try again.")
            .await?;
        return Ok(());
    }

    let display_name = if selected_model == MODEL_GEMINI {
        "Gemini".to_string()
    } else {
        selected_model.clone()
    };

    let processing_text = if request.video_data.is_some() {
        format!("Analyzing video and processing your question with {}...", display_name)
    } else if request.audio_data.is_some() {
        format!("Analyzing audio and processing your question with {}...", display_name)
    } else if !request.image_data_list.is_empty() {
        format!("Analyzing {} image(s) and processing your question with {}...", request.image_data_list.len(), display_name)
    } else {
        format!("Processing your question with {}...", display_name)
    };

    bot.edit_message_text(message.chat().id, message.id(), processing_text)
        .await?;

    let command_timer = request.command_timer.take();
    let result = process_request(&bot, &state, request, &selected_model).await;
    if let Some(mut timer) = command_timer {
        let status = if result.is_ok() { "success" } else { "error" };
        complete_command_timer(&mut timer, status, None);
    }

    Ok(())
}
