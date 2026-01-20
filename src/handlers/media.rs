use std::collections::HashSet;

use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::FileId;

use crate::config::CONFIG;
use crate::state::AppState;

#[derive(Debug, Default, Clone)]
pub struct MediaCollection {
    pub images: Vec<Vec<u8>>,
    pub video: Option<Vec<u8>>,
    pub video_mime_type: Option<String>,
    pub audio: Option<Vec<u8>>,
    pub audio_mime_type: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct MediaCollectionOptions {
    pub include_reply: bool,
    pub include_media_group: bool,
    pub max_images: usize,
    pub clear_images_on_video_or_audio: bool,
    pub skip_audio_if_video: bool,
    pub skip_voice_if_audio: bool,
}

impl MediaCollectionOptions {
    pub fn for_commands() -> Self {
        Self {
            include_reply: false,
            include_media_group: true,
            max_images: 5,
            clear_images_on_video_or_audio: true,
            skip_audio_if_video: true,
            skip_voice_if_audio: true,
        }
    }

    pub fn for_qa() -> Self {
        Self {
            include_reply: true,
            include_media_group: true,
            max_images: 5,
            clear_images_on_video_or_audio: false,
            skip_audio_if_video: false,
            skip_voice_if_audio: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum MediaFillMode {
    Always,
    FillMissing,
}

pub async fn get_file_url(bot: &Bot, file_id: &FileId) -> Result<String> {
    let file = bot.get_file(file_id.clone()).await?;
    Ok(format!(
        "https://api.telegram.org/file/bot{}/{}",
        CONFIG.bot_token,
        file.path
    ))
}

fn add_image(collection: &mut MediaCollection, bytes: Vec<u8>, max_images: usize) {
    if collection.images.len() < max_images {
        collection.images.push(bytes);
    }
}

async fn add_image_from_file_id(
    bot: &Bot,
    file_id: &FileId,
    collection: &mut MediaCollection,
    options: MediaCollectionOptions,
    seen_file_ids: &mut HashSet<FileId>,
) {
    if collection.images.len() >= options.max_images {
        return;
    }
    if !seen_file_ids.insert(file_id.clone()) {
        return;
    }
    if let Ok(url) = get_file_url(bot, file_id).await {
        if let Some(bytes) = crate::llm::media::download_media(&url).await {
            add_image(collection, bytes, options.max_images);
        }
    }
}

async fn collect_from_message(
    bot: &Bot,
    message: &Message,
    collection: &mut MediaCollection,
    options: MediaCollectionOptions,
    fill_mode: MediaFillMode,
    seen_file_ids: &mut HashSet<FileId>,
) {
    let allow_images = matches!(fill_mode, MediaFillMode::Always) || collection.images.is_empty();
    if allow_images {
        if let Some(photo_sizes) = message.photo() {
            if let Some(photo) = photo_sizes.last() {
                add_image_from_file_id(bot, &photo.file.id, collection, options, seen_file_ids)
                    .await;
            }
        }
    }

    let allow_video = matches!(fill_mode, MediaFillMode::Always) || collection.video.is_none();
    if allow_video {
        if let Some(video) = message.video() {
            if let Ok(url) = get_file_url(bot, &video.file.id).await {
                if let Some(bytes) = crate::llm::media::download_media(&url).await {
                    collection.video = Some(bytes);
                    collection.video_mime_type = Some("video/mp4".to_string());
                    if options.clear_images_on_video_or_audio {
                        collection.images.clear();
                    }
                }
            }
        }
    }

    let allow_audio = matches!(fill_mode, MediaFillMode::Always) || collection.audio.is_none();
    if allow_audio {
        if options.skip_audio_if_video && collection.video.is_some() {
            return;
        }

        if let Some(audio) = message.audio() {
            if let Ok(url) = get_file_url(bot, &audio.file.id).await {
                if let Some(bytes) = crate::llm::media::download_media(&url).await {
                    collection.audio = Some(bytes);
                    collection.audio_mime_type = Some("audio/mpeg".to_string());
                    if options.clear_images_on_video_or_audio {
                        collection.images.clear();
                    }
                }
            }
        }
    }

    let allow_voice = match fill_mode {
        MediaFillMode::Always => !(options.skip_voice_if_audio && collection.audio.is_some()),
        MediaFillMode::FillMissing => collection.audio.is_none(),
    };
    if allow_voice {
        if options.skip_audio_if_video && collection.video.is_some() {
            return;
        }

        if let Some(voice) = message.voice() {
            if let Ok(url) = get_file_url(bot, &voice.file.id).await {
                if let Some(bytes) = crate::llm::media::download_media(&url).await {
                    collection.audio = Some(bytes);
                    collection.audio_mime_type = Some("audio/ogg".to_string());
                    if options.clear_images_on_video_or_audio {
                        collection.images.clear();
                    }
                }
            }
        }
    }
}

pub async fn collect_message_media(
    bot: &Bot,
    state: &AppState,
    message: &Message,
    options: MediaCollectionOptions,
) -> MediaCollection {
    let mut collection = MediaCollection::default();
    let mut seen_file_ids: HashSet<FileId> = HashSet::new();

    collect_from_message(
        bot,
        message,
        &mut collection,
        options,
        MediaFillMode::Always,
        &mut seen_file_ids,
    )
    .await;

    if options.include_reply {
        if let Some(reply) = message.reply_to_message() {
            collect_from_message(
                bot,
                reply,
                &mut collection,
                options,
                MediaFillMode::FillMissing,
                &mut seen_file_ids,
            )
            .await;
        }
    }

    if options.include_media_group {
        let mut group_ids = Vec::new();
        if let Some(media_group_id) = message.media_group_id() {
            group_ids.push(media_group_id.clone());
        }
        if options.include_reply {
            if let Some(reply) = message.reply_to_message() {
                if let Some(media_group_id) = reply.media_group_id() {
                    if !group_ids.iter().any(|id| id == media_group_id) {
                        group_ids.push(media_group_id.clone());
                    }
                }
            }
        }

        for media_group_id in group_ids {
            let group_items = state
                .media_groups
                .lock()
                .get(&media_group_id)
                .cloned()
                .unwrap_or_default();
            for item in group_items {
                add_image_from_file_id(
                    bot,
                    &item.file_id,
                    &mut collection,
                    options,
                    &mut seen_file_ids,
                )
                .await;
            }
        }
    }

    collection
}
