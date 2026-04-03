use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use anyhow::Result;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use teloxide::prelude::*;
use teloxide::types::FileId;

use crate::config::CONFIG;
use crate::llm::media::{detect_mime_type, download_media, kind_for_mime, MediaFile, MediaKind};
use crate::state::AppState;

const DEFAULT_MAX_FILES: usize = 10;
const FILE_URL_CACHE_TTL: Duration = Duration::from_secs(900);
const FILE_URL_CACHE_MAX_ENTRIES: usize = 512;

#[derive(Debug, Clone)]
struct FileUrlCacheEntry {
    stored_at: Instant,
    url: String,
}

static FILE_URL_CACHE: Lazy<Mutex<HashMap<String, FileUrlCacheEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Debug, Default, Clone)]
pub struct MediaCollection {
    pub files: Vec<MediaFile>,
}

#[derive(Debug, Clone, Copy)]
pub struct MediaCollectionOptions {
    pub include_reply: bool,
    pub include_media_group: bool,
    pub max_files: usize,
}

impl MediaCollectionOptions {
    pub fn for_commands() -> Self {
        Self {
            include_reply: false,
            include_media_group: true,
            max_files: DEFAULT_MAX_FILES,
        }
    }

    pub fn for_qa() -> Self {
        Self {
            include_reply: true,
            include_media_group: true,
            max_files: DEFAULT_MAX_FILES,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MediaSummary {
    pub total: usize,
    pub images: usize,
    pub videos: usize,
    pub audios: usize,
    pub documents: usize,
}

pub fn summarize_media_files(files: &[MediaFile]) -> MediaSummary {
    let mut summary = MediaSummary {
        total: files.len(),
        ..MediaSummary::default()
    };

    for file in files {
        match file.kind {
            MediaKind::Image => summary.images += 1,
            MediaKind::Video => summary.videos += 1,
            MediaKind::Audio => summary.audios += 1,
            MediaKind::Document => summary.documents += 1,
        }
    }

    summary
}

pub async fn get_file_url(bot: &Bot, file_id: &FileId) -> Result<String> {
    let key = file_id.to_string();
    {
        let mut cache = FILE_URL_CACHE.lock();
        prune_file_url_cache(&mut cache);
        if let Some(entry) = cache.get(&key) {
            if entry.stored_at.elapsed() <= FILE_URL_CACHE_TTL {
                return Ok(entry.url.clone());
            }
        }
        cache.remove(&key);
    }

    let file = bot.get_file(file_id.clone()).await?;
    let url = format!(
        "https://api.telegram.org/file/bot{}/{}",
        CONFIG.bot_token, file.path
    );

    let mut cache = FILE_URL_CACHE.lock();
    prune_file_url_cache(&mut cache);
    cache.insert(
        key,
        FileUrlCacheEntry {
            stored_at: Instant::now(),
            url: url.clone(),
        },
    );
    Ok(url)
}

fn prune_file_url_cache(cache: &mut HashMap<String, FileUrlCacheEntry>) {
    cache.retain(|_, entry| entry.stored_at.elapsed() <= FILE_URL_CACHE_TTL);
    if cache.len() <= FILE_URL_CACHE_MAX_ENTRIES {
        return;
    }

    let mut ordered = cache
        .iter()
        .map(|(key, entry)| (key.clone(), entry.stored_at))
        .collect::<Vec<_>>();
    ordered.sort_by_key(|(_, stored_at)| *stored_at);
    let remove_count = cache.len().saturating_sub(FILE_URL_CACHE_MAX_ENTRIES);
    for (key, _) in ordered.into_iter().take(remove_count) {
        cache.remove(&key);
    }
}

fn extension_mime_hint(file_name: &str) -> Option<&'static str> {
    let lower = file_name.to_ascii_lowercase();
    if lower.ends_with(".png") {
        Some("image/png")
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        Some("image/jpeg")
    } else if lower.ends_with(".webp") {
        Some("image/webp")
    } else if lower.ends_with(".gif") {
        Some("image/gif")
    } else if lower.ends_with(".mp4") {
        Some("video/mp4")
    } else if lower.ends_with(".mov") {
        Some("video/quicktime")
    } else if lower.ends_with(".webm") {
        Some("video/webm")
    } else if lower.ends_with(".mp3") {
        Some("audio/mpeg")
    } else if lower.ends_with(".ogg") {
        Some("audio/ogg")
    } else if lower.ends_with(".wav") {
        Some("audio/wav")
    } else if lower.ends_with(".pdf") {
        Some("application/pdf")
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
async fn add_file_from_file_id(
    bot: &Bot,
    file_id: &FileId,
    collection: &mut MediaCollection,
    options: MediaCollectionOptions,
    seen_file_ids: &mut HashSet<FileId>,
    mime_type_hint: Option<&str>,
    display_name: Option<&str>,
    kind_hint: Option<MediaKind>,
) {
    if collection.files.len() >= options.max_files {
        return;
    }
    if !seen_file_ids.insert(file_id.clone()) {
        return;
    }

    let Ok(url) = get_file_url(bot, file_id).await else {
        return;
    };
    let Some(bytes) = download_media(&url).await else {
        return;
    };

    let mut mime_type = mime_type_hint.map(|value| value.to_string());
    if mime_type.is_none() {
        mime_type = detect_mime_type(&bytes);
    }
    if mime_type.is_none() {
        if let Some(name) = display_name {
            mime_type = extension_mime_hint(name).map(|value| value.to_string());
        }
    }
    let mime_type = mime_type.unwrap_or_else(|| "application/octet-stream".to_string());
    let kind = kind_hint.unwrap_or_else(|| kind_for_mime(&mime_type));

    collection.files.push(MediaFile::new(
        bytes,
        mime_type,
        kind,
        display_name.map(|value| value.to_string()),
    ));
}

async fn collect_from_message(
    bot: &Bot,
    message: &Message,
    collection: &mut MediaCollection,
    options: MediaCollectionOptions,
    seen_file_ids: &mut HashSet<FileId>,
) {
    if collection.files.len() >= options.max_files {
        return;
    }

    if let Some(photo_sizes) = message.photo() {
        if let Some(photo) = photo_sizes.last() {
            add_file_from_file_id(
                bot,
                &photo.file.id,
                collection,
                options,
                seen_file_ids,
                None,
                None,
                Some(MediaKind::Image),
            )
            .await;
        }
    }

    if collection.files.len() >= options.max_files {
        return;
    }

    if let Some(document) = message.document() {
        let mime_hint = document.mime_type.as_ref().map(|mime| mime.essence_str());
        let name_hint = document.file_name.as_deref();
        add_file_from_file_id(
            bot,
            &document.file.id,
            collection,
            options,
            seen_file_ids,
            mime_hint,
            name_hint,
            None,
        )
        .await;
    }

    if collection.files.len() >= options.max_files {
        return;
    }

    if let Some(video) = message.video() {
        let mime_hint = video
            .mime_type
            .as_ref()
            .map(|mime| mime.essence_str())
            .or(Some("video/mp4"));
        add_file_from_file_id(
            bot,
            &video.file.id,
            collection,
            options,
            seen_file_ids,
            mime_hint,
            None,
            Some(MediaKind::Video),
        )
        .await;
    }

    if collection.files.len() >= options.max_files {
        return;
    }

    if let Some(audio) = message.audio() {
        let mime_hint = audio
            .mime_type
            .as_ref()
            .map(|mime| mime.essence_str())
            .or(Some("audio/mpeg"));
        add_file_from_file_id(
            bot,
            &audio.file.id,
            collection,
            options,
            seen_file_ids,
            mime_hint,
            audio.file_name.as_deref(),
            Some(MediaKind::Audio),
        )
        .await;
    }

    if collection.files.len() >= options.max_files {
        return;
    }

    if let Some(voice) = message.voice() {
        add_file_from_file_id(
            bot,
            &voice.file.id,
            collection,
            options,
            seen_file_ids,
            Some("audio/ogg"),
            None,
            Some(MediaKind::Audio),
        )
        .await;
    }

    if collection.files.len() >= options.max_files {
        return;
    }

    if let Some(sticker) = message.sticker() {
        if !sticker.flags.is_animated && !sticker.flags.is_video {
            add_file_from_file_id(
                bot,
                &sticker.file.id,
                collection,
                options,
                seen_file_ids,
                Some("image/webp"),
                None,
                Some(MediaKind::Image),
            )
            .await;
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

    collect_from_message(bot, message, &mut collection, options, &mut seen_file_ids).await;

    if options.include_reply {
        if let Some(reply) = message.reply_to_message() {
            collect_from_message(bot, reply, &mut collection, options, &mut seen_file_ids).await;
        }
    }

    if options.include_media_group && collection.files.len() < options.max_files {
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
            if collection.files.len() >= options.max_files {
                break;
            }
            let group_items = state.media_group_items(&media_group_id);
            for item in group_items {
                if collection.files.len() >= options.max_files {
                    break;
                }
                add_file_from_file_id(
                    bot,
                    &item.file_id,
                    &mut collection,
                    options,
                    &mut seen_file_ids,
                    None,
                    None,
                    Some(MediaKind::Image),
                )
                .await;
            }
        }
    }

    collection
}
