use std::collections::HashSet;
use std::time::Duration;

use anyhow::Result;
use futures_util::{stream, StreamExt};
use parking_lot::Mutex;
use std::sync::LazyLock;
use teloxide::prelude::*;
use teloxide::types::FileId;
use tokio::sync::mpsc;

use crate::llm::media::{
    detect_mime_type, download_media_limited, kind_for_mime, MediaFile, MediaKind,
    MEDIA_DOWNLOAD_MAX_BYTES,
};
use crate::state::{AppState, MediaGroupItem};
use crate::utils::telegram::rich_message::{RichMedia, RichUpdate};
use crate::utils::ttl_cache::TtlCache;

const DEFAULT_MAX_FILES: usize = 10;
// Rich-message slideshows can exceed the normal album size.
const MAX_RICH_MEDIA_FILES: usize = 32;
const MAX_RICH_MEDIA_BYTES: usize = 32 * 1024 * 1024;
const RICH_FILE_URL_CONCURRENCY: usize = 4;
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
    pub rich_attachments_skipped: usize,
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
    let url = bot
        .api_url()
        .join(&format!("file/bot{}/{}", bot.token(), file.path))?
        .to_string();

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

pub(crate) fn rich_media_kind(attachment: &RichMedia) -> MediaKind {
    attachment.kind.unwrap_or_else(|| {
        let declared = attachment.mime_type.as_deref().map(kind_for_mime);
        declared
            .filter(|kind| *kind != MediaKind::Document)
            .or_else(|| {
                attachment
                    .file_name
                    .as_deref()
                    .and_then(extension_mime_hint)
                    .map(kind_for_mime)
            })
            .unwrap_or(MediaKind::Document)
    })
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

struct OrdinaryMediaBudget {
    files_left: usize,
    seen_file_ids: HashSet<FileId>,
}

impl OrdinaryMediaBudget {
    fn is_full(&self) -> bool {
        self.files_left == 0
    }

    async fn add_file(
        &mut self,
        bot: &Bot,
        file_id: &FileId,
        collection: &mut MediaCollection,
        mime_type_hint: Option<&str>,
        display_name: Option<&str>,
        kind_hint: Option<MediaKind>,
    ) {
        if self.is_full() || !self.seen_file_ids.insert(file_id.clone()) {
            return;
        }
        let Ok(url) = get_file_url(bot, file_id).await else {
            return;
        };
        if let Some(file) = download_telegram_media(
            &url,
            mime_type_hint,
            display_name,
            kind_hint,
            None,
            MEDIA_DOWNLOAD_MAX_BYTES,
        )
        .await
        {
            self.files_left -= 1;
            collection.files.push(file);
        }
    }
}

async fn download_telegram_media(
    url: &str,
    mime_type_hint: Option<&str>,
    display_name: Option<&str>,
    kind_hint: Option<MediaKind>,
    fallback_mime_type: Option<&str>,
    max_bytes: usize,
) -> Option<MediaFile> {
    let bytes = download_media_limited(url, max_bytes).await?;
    let mime_type = mime_type_hint
        .map(str::to_owned)
        .or_else(|| detect_mime_type(&bytes))
        .unwrap_or_else(|| {
            display_name
                .and_then(extension_mime_hint)
                .or(fallback_mime_type)
                .unwrap_or("application/octet-stream")
                .to_owned()
        });
    let kind = kind_hint.unwrap_or_else(|| kind_for_mime(&mime_type));

    Some(MediaFile::new(
        bytes,
        mime_type,
        kind,
        display_name.map(|value| value.to_string()),
    ))
}

struct RichMediaBudget {
    files: usize,
    bytes: usize,
}

impl RichMediaBudget {
    async fn collect(&mut self, bot: &Bot, media: &[RichMedia], collection: &mut MediaCollection) {
        if self.files == 0 || self.bytes == 0 || media.is_empty() {
            collection.rich_attachments_skipped += media.len();
            return;
        }

        let attachments = &media[..media.len().min(self.files)];
        let file_ids = attachments
            .iter()
            .map(|attachment| attachment.file_id.clone())
            .collect::<Vec<_>>();
        let bot = bot.clone();
        let (send_url, mut urls) = mpsc::channel(attachments.len());
        // Keep resolving URLs while an earlier body is being read. Only the
        // consumer starts body requests and updates the remaining byte budget.
        // The bounded channel holds all results so a slow body cannot pause a lookup.
        let prefetch = async move {
            let resolve = async {
                let mut lookups = stream::iter(file_ids)
                    .map(|file_id| {
                        let bot = bot.clone();
                        async move { get_file_url(&bot, &file_id).await }
                    })
                    .buffered(RICH_FILE_URL_CONCURRENCY);
                while let Some(lookup) = lookups.next().await {
                    if send_url.try_send(lookup).is_err() {
                        break;
                    }
                }
            };
            tokio::select! {
                () = send_url.closed() => {},
                () = resolve => {},
            }
        };
        let download = async {
            let mut included = 0;
            for attachment in attachments {
                if self.bytes == 0 {
                    break;
                }
                let Some(url) = urls.recv().await else {
                    break;
                };
                self.files -= 1;
                let Ok(url) = url else { continue };
                if let Some(file) = download_telegram_media(
                    &url,
                    attachment.mime_type.as_deref(),
                    attachment.file_name.as_deref(),
                    attachment.kind,
                    attachment.fallback_mime_type,
                    self.bytes.min(MEDIA_DOWNLOAD_MAX_BYTES),
                )
                .await
                {
                    self.bytes -= file.bytes().len();
                    collection.files.push(file);
                    included += 1;
                }
            }
            drop(urls);
            included
        };
        let ((), included) = tokio::join!(prefetch, download);
        collection.rich_attachments_skipped += media.len() - included;
    }
}

async fn collect_from_message(
    bot: &Bot,
    message: &Message,
    collection: &mut MediaCollection,
    ordinary: &mut OrdinaryMediaBudget,
) {
    if ordinary.is_full() {
        return;
    }

    if let Some(photo_sizes) = message.photo() {
        if let Some(photo) = photo_sizes.last() {
            ordinary
                .add_file(
                    bot,
                    &photo.file.id,
                    collection,
                    None,
                    None,
                    Some(MediaKind::Image),
                )
                .await;
        }
    }

    if let Some(document) = message.document() {
        let mime_hint = document.mime_type.as_ref().map(|mime| mime.essence_str());
        let name_hint = document.file_name.as_deref();
        ordinary
            .add_file(
                bot,
                &document.file.id,
                collection,
                mime_hint,
                name_hint,
                None,
            )
            .await;
    }

    if let Some(video) = message.video() {
        let mime_hint = video
            .mime_type
            .as_ref()
            .map(|mime| mime.essence_str())
            .or(Some("video/mp4"));
        ordinary
            .add_file(
                bot,
                &video.file.id,
                collection,
                mime_hint,
                None,
                Some(MediaKind::Video),
            )
            .await;
    }

    if let Some(animation) = message.animation() {
        let (mime_hint, kind_hint) = animation_media_hint(
            animation.mime_type.as_ref().map(|mime| mime.essence_str()),
            animation.file_name.as_deref(),
        );
        ordinary
            .add_file(
                bot,
                &animation.file.id,
                collection,
                Some(&mime_hint),
                animation.file_name.as_deref(),
                Some(kind_hint),
            )
            .await;
    }

    if let Some(audio) = message.audio() {
        let mime_hint = audio
            .mime_type
            .as_ref()
            .map(|mime| mime.essence_str())
            .or(Some("audio/mpeg"));
        ordinary
            .add_file(
                bot,
                &audio.file.id,
                collection,
                mime_hint,
                audio.file_name.as_deref(),
                Some(MediaKind::Audio),
            )
            .await;
    }

    if let Some(voice) = message.voice() {
        ordinary
            .add_file(
                bot,
                &voice.file.id,
                collection,
                Some("audio/ogg"),
                None,
                Some(MediaKind::Audio),
            )
            .await;
    }

    if let Some(sticker) = message.sticker() {
        if let Some((mime_hint, kind_hint)) =
            sticker_media_hint(sticker.flags.is_animated, sticker.flags.is_video)
        {
            ordinary
                .add_file(
                    bot,
                    &sticker.file.id,
                    collection,
                    Some(mime_hint),
                    None,
                    Some(kind_hint),
                )
                .await;
        } else if let Some(thumbnail) = sticker.thumbnail.as_ref() {
            ordinary
                .add_file(
                    bot,
                    &thumbnail.file.id,
                    collection,
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
    rich: &RichUpdate,
) -> MediaCollection {
    let mut collection = MediaCollection::default();
    let mut ordinary = OrdinaryMediaBudget {
        files_left: options.max_files,
        seen_file_ids: HashSet::new(),
    };
    let mut rich_budget = RichMediaBudget {
        files: MAX_RICH_MEDIA_FILES,
        bytes: MAX_RICH_MEDIA_BYTES,
    };

    if let Some(media) = &rich.message {
        rich_budget.collect(bot, media, &mut collection).await;
    } else {
        collect_from_message(bot, message, &mut collection, &mut ordinary).await;
    }

    if options.include_reply {
        if let Some(reply) = message.reply_to_message() {
            if let Some(media) = &rich.reply {
                if reply.chat.id == message.chat.id {
                    rich_budget.collect(bot, media, &mut collection).await;
                }
            } else {
                collect_from_message(bot, reply, &mut collection, &mut ordinary).await;
            }
        }
    }

    if options.include_media_group && !ordinary.is_full() {
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
            if ordinary.is_full() {
                break;
            }
            let group_items = state.media_group_items(&media_group_id);
            for item in group_items {
                if ordinary.is_full() {
                    break;
                }
                ordinary
                    .add_file(
                        bot,
                        &item.file_id,
                        &mut collection,
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
    use crate::test_support::{response_with_content_length, ExpectedRequest, TestServer};
    use serde_json::json;
    use teloxide::types::MediaGroupId;

    fn rich_attachment(file_id: &str) -> RichMedia {
        RichMedia {
            file_id: FileId(file_id.into()),
            mime_type: None,
            file_name: None,
            kind: Some(MediaKind::Image),
            fallback_mime_type: None,
        }
    }

    fn file_lookup_response(file_id: &str) -> Vec<u8> {
        let body = serde_json::to_vec(&json!({
            "ok": true,
            "result": {"file_id": file_id, "file_unique_id": file_id, "file_path": file_id}
        }))
        .unwrap();
        response_with_content_length(body.len(), body)
    }

    fn file_lookup(file_id: &str) -> ExpectedRequest {
        ExpectedRequest::new(
            "POST",
            "/botmedia-fixture/GetFile",
            file_lookup_response(file_id),
        )
        .with_body_fragment(format!("\"{file_id}\"").as_bytes())
    }

    fn download(file_id: &str, bytes: &[u8]) -> ExpectedRequest {
        ExpectedRequest::new(
            "GET",
            &format!("/file/botmedia-fixture/{file_id}"),
            response_with_content_length(bytes.len(), bytes.to_vec()),
        )
    }

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

    #[tokio::test]
    async fn rich_current_media_preserves_the_ordinary_reply_and_album_budget() {
        let rich = RichUpdate {
            message: Some(vec![
                rich_attachment("mixed-rich-one"),
                rich_attachment("mixed-rich-two"),
            ]),
            ..RichUpdate::default()
        };
        let message = serde_json::from_value(json!({
            "message_id": 42, "date": 123, "chat": {"id": 41, "type": "private"},
            "text": "/q question",
            "reply_to_message": {
                "message_id": 41, "date": 122, "chat": {"id": 41, "type": "private"},
                "photo": [{
                    "file_id": "mixed-ordinary-reply", "file_unique_id": "mixed-reply",
                    "width": 100, "height": 100
                }],
                "media_group_id": "mixed-ordinary-album"
            }
        }))
        .unwrap();
        let db = crate::db::test_support::init_test_db("mixed-media-limits").await;
        let state = AppState::new(db.clone(), 42, "test_bot".into());
        let group_id = MediaGroupId("mixed-ordinary-album".into());
        for id in [
            "mixed-ordinary-reply",
            "mixed-ordinary-album",
            "mixed-extra",
        ] {
            state.store_media_group_item(
                &group_id,
                MediaGroupItem {
                    file_id: FileId(id.into()),
                },
            );
        }
        let server = TestServer::with_unordered_expectations(
            vec![
                download("mixed-rich-one", b"first rich"),
                download("mixed-rich-two", b"second rich"),
                file_lookup("mixed-ordinary-reply"),
                download("mixed-ordinary-reply", b"ordinary reply"),
                file_lookup("mixed-ordinary-album"),
                download("mixed-ordinary-album", b"ordinary album"),
            ],
            vec![file_lookup("mixed-rich-one"), file_lookup("mixed-rich-two")],
        );
        let bot = Bot::new("media-fixture").set_api_url(server.base_url());
        let media = collect_message_media(
            &bot,
            &state,
            &message,
            MediaCollectionOptions {
                max_files: 2,
                ..MediaCollectionOptions::for_qa()
            },
            &rich,
        )
        .await;

        assert_eq!(media.rich_attachments_skipped, 0);
        assert_eq!(
            media.files.iter().map(MediaFile::bytes).collect::<Vec<_>>(),
            [
                b"first rich".as_slice(),
                b"second rich",
                b"ordinary reply",
                b"ordinary album"
            ],
        );
        server.join().unwrap();
        db.shutdown().await;
    }

    #[tokio::test]
    async fn rich_body_admission_uses_the_remaining_budget_in_attachment_order() {
        let mut budget = RichMediaBudget { files: 3, bytes: 5 };
        let mut collection = MediaCollection::default();
        let server = TestServer::with_unordered_expectations(
            vec![
                download("budget-first", b"123"),
                download("budget-oversize", b"456"),
                download("budget-fit", b"78"),
            ],
            vec![
                file_lookup("budget-first"),
                file_lookup("budget-oversize"),
                file_lookup("budget-fit"),
            ],
        );
        let bot = Bot::new("media-fixture").set_api_url(server.base_url());

        budget
            .collect(
                &bot,
                &[
                    rich_attachment("budget-first"),
                    rich_attachment("budget-oversize"),
                ],
                &mut collection,
            )
            .await;
        budget
            .collect(
                &bot,
                &[
                    rich_attachment("budget-fit"),
                    rich_attachment("budget-over-count"),
                ],
                &mut collection,
            )
            .await;

        assert_eq!(budget.files, 0);
        assert_eq!(budget.bytes, 0);
        assert_eq!(collection.rich_attachments_skipped, 2);
        assert_eq!(
            collection
                .files
                .iter()
                .map(MediaFile::bytes)
                .collect::<Vec<_>>(),
            [b"123".as_slice(), b"78"]
        );
        server.join().unwrap();
    }

    #[tokio::test]
    async fn rich_file_urls_are_prefetched_during_ordered_body_downloads() {
        use crate::test_support::read_http_request;
        use std::collections::HashMap;
        use std::io::{ErrorKind, Write};
        use std::net::TcpListener;
        use std::time::Instant;

        const LOOKUP_PATH: &str = "/botmedia-fixture/GetFile";
        const FILE_IDS: [&str; RICH_FILE_URL_CONCURRENCY + 1] = [
            "overlap-one",
            "overlap-two",
            "overlap-three",
            "overlap-four",
            "overlap-five",
        ];
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::task::spawn_blocking(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            let accept = || loop {
                match listener.accept() {
                    Ok((mut socket, _)) => {
                        socket.set_nonblocking(false).unwrap();
                        socket
                            .set_read_timeout(Some(Duration::from_secs(1)))
                            .unwrap();
                        let (method, path, body) = read_http_request(&mut socket);
                        break (socket, method, path, body);
                    }
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        assert!(
                            Instant::now() < deadline,
                            "timed out waiting for overlapping request"
                        );
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("accepting test request: {error}"),
                }
            };
            let file_id = |body: &[u8]| {
                serde_json::from_slice::<serde_json::Value>(body).unwrap()["file_id"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            };
            let mut lookups = HashMap::new();
            for _ in 0..RICH_FILE_URL_CONCURRENCY {
                let (socket, method, path, body) = accept();
                assert_eq!((method.as_str(), path.as_str()), ("POST", LOOKUP_PATH));
                assert!(lookups.insert(file_id(&body), socket).is_none());
            }
            assert!(FILE_IDS[..RICH_FILE_URL_CONCURRENCY]
                .iter()
                .all(|id| lookups.contains_key(*id)));
            lookups
                .remove(FILE_IDS[0])
                .unwrap()
                .write_all(&file_lookup_response(FILE_IDS[0]))
                .unwrap();

            let mut first_body = None;
            for _ in 0..2 {
                let (socket, method, path, body) = accept();
                if method == "POST" {
                    assert_eq!(path, LOOKUP_PATH);
                    let id = file_id(&body);
                    assert_eq!(id, FILE_IDS[RICH_FILE_URL_CONCURRENCY]);
                    assert!(lookups.insert(id, socket).is_none());
                } else {
                    assert_eq!(method, "GET");
                    assert_eq!(path, format!("/file/botmedia-fixture/{}", FILE_IDS[0]));
                    assert!(first_body.replace(socket).is_none());
                }
            }
            // Finish metadata out of order while the first body is still pending.
            for id in FILE_IDS[1..].iter().rev() {
                lookups
                    .remove(*id)
                    .unwrap()
                    .write_all(&file_lookup_response(id))
                    .unwrap();
            }
            first_body
                .unwrap()
                .write_all(&response_with_content_length(
                    FILE_IDS[0].len(),
                    FILE_IDS[0].as_bytes().to_vec(),
                ))
                .unwrap();
            for id in &FILE_IDS[1..] {
                let (mut socket, method, path, _) = accept();
                assert_eq!(method, "GET");
                assert_eq!(path, format!("/file/botmedia-fixture/{id}"));
                socket
                    .write_all(&response_with_content_length(
                        id.len(),
                        id.as_bytes().to_vec(),
                    ))
                    .unwrap();
            }
        });
        let bot =
            Bot::new("media-fixture").set_api_url(format!("http://{address}").parse().unwrap());
        let attachments = FILE_IDS.map(rich_attachment);
        let mut collection = MediaCollection::default();
        let mut budget = RichMediaBudget {
            files: FILE_IDS.len(),
            bytes: 1024,
        };

        tokio::time::timeout(
            Duration::from_secs(10),
            budget.collect(&bot, &attachments, &mut collection),
        )
        .await
        .unwrap();
        server.await.unwrap();
        assert_eq!(collection.rich_attachments_skipped, 0);
        assert_eq!(
            collection
                .files
                .iter()
                .map(MediaFile::bytes)
                .collect::<Vec<_>>(),
            FILE_IDS.map(str::as_bytes),
        );
    }

    #[tokio::test]
    async fn rich_mime_uses_explicit_hint_then_bytes_then_filename_then_kind_fallback() {
        const GIF: &[u8] = b"GIF89a";
        let attachments = [
            RichMedia {
                mime_type: Some("video/quicktime".into()),
                file_name: Some("clip.webm".into()),
                kind: Some(MediaKind::Video),
                fallback_mime_type: Some("video/mp4"),
                ..rich_attachment("mime-explicit")
            },
            RichMedia {
                file_name: Some("clip.webm".into()),
                kind: Some(MediaKind::Video),
                fallback_mime_type: Some("video/mp4"),
                ..rich_attachment("mime-bytes")
            },
            RichMedia {
                file_name: Some("voice.WAV".into()),
                kind: Some(MediaKind::Audio),
                fallback_mime_type: Some("audio/ogg"),
                ..rich_attachment("mime-name")
            },
            RichMedia {
                kind: Some(MediaKind::Audio),
                fallback_mime_type: Some("audio/ogg"),
                ..rich_attachment("mime-fallback")
            },
        ];
        let server = TestServer::with_unordered_expectations(
            vec![
                download("mime-explicit", GIF),
                download("mime-bytes", GIF),
                download("mime-name", b"other"),
                download("mime-fallback", b"other"),
            ],
            attachments
                .iter()
                .map(|attachment| file_lookup(&attachment.file_id.0))
                .collect(),
        );
        let bot = Bot::new("media-fixture").set_api_url(server.base_url());
        let mut budget = RichMediaBudget {
            files: 4,
            bytes: 1024,
        };
        let mut collection = MediaCollection::default();

        budget.collect(&bot, &attachments, &mut collection).await;

        assert_eq!(collection.rich_attachments_skipped, 0);
        assert_eq!(
            collection
                .files
                .iter()
                .map(|file| (file.mime_type.as_str(), file.kind))
                .collect::<Vec<_>>(),
            [
                ("video/quicktime", MediaKind::Video),
                ("image/gif", MediaKind::Video),
                ("audio/wav", MediaKind::Audio),
                ("audio/ogg", MediaKind::Audio),
            ],
        );
        server.join().unwrap();
    }
}
