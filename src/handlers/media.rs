use std::collections::HashSet;
use std::time::Duration;

use anyhow::Result;
use parking_lot::Mutex;
use std::sync::LazyLock;
use teloxide::prelude::*;
use teloxide::types::FileId;

use crate::config::CONFIG;
use crate::llm::media::{detect_mime_type, download_media, kind_for_mime, MediaFile, MediaKind};
use crate::state::{AppState, MediaGroupItem};
use crate::utils::ttl_cache::TtlCache;

const DEFAULT_MAX_FILES: usize = 10;
const FILE_URL_CACHE_TTL: Duration = Duration::from_secs(900);
const FILE_URL_CACHE_MAX_ENTRIES: usize = 512;

static FILE_URL_CACHE: LazyLock<Mutex<TtlCache<String, String>>> = LazyLock::new(|| {
    Mutex::new(TtlCache::new(
        FILE_URL_CACHE_TTL,
        FILE_URL_CACHE_MAX_ENTRIES,
    ))
});

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

pub async fn get_file_url(bot: &Bot, file_id: &FileId) -> Result<String> {
    let key = file_id.to_string();
    if let Some(url) = FILE_URL_CACHE.lock().get(&key) {
        return Ok(url);
    }

    let file = bot.get_file(file_id.clone()).await?;
    let url = format!(
        "https://api.telegram.org/file/bot{}/{}",
        CONFIG.bot_token, file.path
    );

    FILE_URL_CACHE.lock().insert(key, url.clone());
    Ok(url)
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

fn animation_media_hint(
    mime_type_hint: Option<&str>,
    file_name_hint: Option<&str>,
) -> (String, MediaKind) {
    let mime_type = mime_type_hint
        .filter(|mime| !mime.trim().is_empty())
        .or_else(|| file_name_hint.and_then(extension_mime_hint))
        .unwrap_or("video/mp4");
    (mime_type.to_string(), MediaKind::Video)
}

fn sticker_media_hint(is_animated: bool, is_video: bool) -> Option<(&'static str, MediaKind)> {
    if is_video {
        Some(("video/webm", MediaKind::Video))
    } else if is_animated {
        None
    } else {
        Some(("image/webp", MediaKind::Image))
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

    if let Some(animation) = message.animation() {
        let (mime_hint, kind_hint) = animation_media_hint(
            animation.mime_type.as_ref().map(|mime| mime.essence_str()),
            animation.file_name.as_deref(),
        );
        add_file_from_file_id(
            bot,
            &animation.file.id,
            collection,
            options,
            seen_file_ids,
            Some(&mime_hint),
            animation.file_name.as_deref(),
            Some(kind_hint),
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
        if let Some((mime_hint, kind_hint)) =
            sticker_media_hint(sticker.flags.is_animated, sticker.flags.is_video)
        {
            add_file_from_file_id(
                bot,
                &sticker.file.id,
                collection,
                options,
                seen_file_ids,
                Some(mime_hint),
                None,
                Some(kind_hint),
            )
            .await;
        } else if let Some(thumbnail) = sticker.thumbnail.as_ref() {
            add_file_from_file_id(
                bot,
                &thumbnail.file.id,
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

pub(crate) fn message_has_image(message: &Message) -> bool {
    if message.photo().is_some() {
        return true;
    }

    if let Some(document) = message.document() {
        let mime_is_image = document
            .mime_type
            .as_ref()
            .map(|mime| mime.essence_str().starts_with("image/"))
            .unwrap_or(false);
        let name_is_image = document
            .file_name
            .as_ref()
            .map(|name| {
                let lower = name.to_ascii_lowercase();
                lower.ends_with(".png")
                    || lower.ends_with(".jpg")
                    || lower.ends_with(".jpeg")
                    || lower.ends_with(".webp")
                    || lower.ends_with(".gif")
            })
            .unwrap_or(false);
        if mime_is_image || name_is_image {
            return true;
        }
    }

    if let Some(sticker) = message.sticker() {
        if !sticker.flags.is_animated && !sticker.flags.is_video {
            return true;
        }
    }

    false
}

pub async fn handle_media_group(state: AppState, message: Message) {
    if let Some(media_group_id) = message.media_group_id() {
        if let Some(photo_sizes) = message.photo() {
            if let Some(photo) = photo_sizes.last() {
                state.store_media_group_item(
                    media_group_id,
                    MediaGroupItem {
                        file_id: photo.file.id.clone(),
                    },
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn animation_media_hint_is_video() {
        assert_eq!(
            animation_media_hint(Some("video/mp4"), Some("clip.gif")),
            ("video/mp4".to_string(), MediaKind::Video)
        );
        assert_eq!(
            animation_media_hint(None, Some("clip.gif")),
            ("image/gif".to_string(), MediaKind::Video)
        );
    }

    #[test]
    fn sticker_media_hint_supports_video_and_static_stickers() {
        assert_eq!(
            sticker_media_hint(false, false),
            Some(("image/webp", MediaKind::Image))
        );
        assert_eq!(
            sticker_media_hint(false, true),
            Some(("video/webm", MediaKind::Video))
        );
        assert_eq!(sticker_media_hint(true, false), None);
    }
}
