//! Video commands: `/vid` (Gemini Omni, with optional image and video
//! inputs) and the hidden `/vid2` with its local MP4 delivery.

use std::collections::HashSet;
use std::path::Path;

use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::{
    ChatAction, FileId, InputFile, MediaGroupId, MessageId, ParseMode, ReplyParameters,
};
use tracing::{info, warn};

use crate::config::CONFIG;
use crate::handlers::access::{check_access_control, is_rate_limited};
use crate::handlers::enrichment::generation_telegraph_context;
use crate::handlers::media::{get_file_url, rich_media_kind};
use crate::llm::audit::create_command_audit_context;
use crate::llm::gemini::check_video_input_counts;
use crate::llm::media::{detect_mime_type, download_media, kind_for_mime, MediaFile, MediaKind};
use crate::llm::vid2_video::{
    download_source_image, generate_video_with_vid2, Vid2Error, Vid2GenerationRequest, Vid2Phase,
};
use crate::llm::{
    generate_video_with_omni, VideoGenerationError, VideoGenerationRequest, VideoTask,
};
use crate::state::{AppState, MediaGroupItem};
use crate::utils::telegram::rich_message::{RichMedia, RichUpdate};
use crate::utils::telegram::{
    edit_message_text_with_retry, message_text_or_caption, retry_telegram, run_with_status_message,
    send_message_with_retry, start_chat_action_heartbeat, strip_command_prefix, VideoDeliveryError,
};
use crate::utils::text::escape_html;

const VID_USAGE: &str = "Please provide a prompt for the video.\n\
Usage: /vid [prompt]\n\
Reply to, or attach, images or a video to animate, edit, or extend them.\n\
If the model picks the wrong mode, start the prompt with --t2v, --i2v, --ref, --edit, or --extend.";

fn with_replied_text(message: &Message, prompt: &str, rich: &RichUpdate) -> String {
    let reply = message
        .reply_to_message()
        .and_then(message_text_or_caption)
        .unwrap_or_default()
        .trim();
    if prompt.is_empty() {
        reply.to_string()
    } else if rich.reply.is_some() && !reply.is_empty() {
        format!("{prompt}\n\nReplied-to message:\n{reply}")
    } else {
        prompt.to_string()
    }
}

fn command_prompt(message: &Message, command_prefix: &str, rich: &RichUpdate) -> String {
    let prompt = message_text_or_caption(message)
        .map(|text| strip_command_prefix(text, command_prefix))
        .unwrap_or_default();
    with_replied_text(message, &prompt, rich)
}

/// `--<name>` options that name the generation task outright.
const VID_TASK_OPTIONS: [(&str, VideoTask); 5] = [
    ("t2v", VideoTask::TextToVideo),
    ("i2v", VideoTask::ImageToVideo),
    ("ref", VideoTask::ReferenceToVideo),
    ("edit", VideoTask::Edit),
    ("extend", VideoTask::Extend),
];

/// Split a leading `--<task>` option off the prompt. Phones turn a typed
/// `--` into a dash, so a dash counts as well.
fn split_task_option(prompt: &str) -> Result<(Option<VideoTask>, &str), String> {
    let prompt = prompt.trim();
    let option = ["--", "\u{2014}", "\u{2013}"]
        .iter()
        .find_map(|marker| prompt.strip_prefix(marker));
    let Some(option) = option else {
        return Ok((None, prompt));
    };
    let (name, rest) = option
        .split_once(char::is_whitespace)
        .unwrap_or((option, ""));
    if name.is_empty() {
        // A dash followed by a space opens the prompt; it is not an option.
        return Ok((None, prompt));
    }
    let task = VID_TASK_OPTIONS
        .iter()
        .find(|(option, _)| name.eq_ignore_ascii_case(option))
        .map(|(_, task)| *task)
        .ok_or_else(|| {
            format!("Unknown option --{name}. Use --t2v, --i2v, --ref, --edit, or --extend.")
        })?;
    Ok((Some(task), rest.trim_start()))
}

/// The task option and prompt of a `/vid` request. A command without a
/// prompt of its own takes the replied-to text.
fn vid_request_text(
    message: &Message,
    rich: &RichUpdate,
) -> Result<(Option<VideoTask>, String), String> {
    let own = message_text_or_caption(message)
        .map(|text| strip_command_prefix(text, "/vid"))
        .unwrap_or_default();
    let (task, prompt) = split_task_option(&own)?;
    Ok((task, with_replied_text(message, prompt, rich)))
}

async fn video_model_prompt(message: &Message, prompt: &str) -> String {
    let context = generation_telegraph_context(message).await;
    if context.is_empty() {
        prompt.to_string()
    } else {
        format!("{prompt}\n\n{context}")
    }
}

/// One Telegram file to hand to the `/vid` model.
#[derive(Debug, Clone, PartialEq, Eq)]
struct VidInput {
    file_id: FileId,
    kind: MediaKind,
    /// The type Telegram declares, when it declares one.
    mime_type: Option<String>,
}

fn document_kind(mime_type: Option<&str>, file_name: Option<&str>) -> Option<MediaKind> {
    match mime_type {
        Some(mime) if mime.starts_with("image/") => return Some(MediaKind::Image),
        Some(mime) if mime.starts_with("video/") => return Some(MediaKind::Video),
        _ => {}
    }
    let name = file_name?.to_ascii_lowercase();
    let has_suffix = |suffixes: &[&str]| suffixes.iter().any(|suffix| name.ends_with(suffix));
    if has_suffix(&[".jpg", ".jpeg", ".png", ".webp"]) {
        Some(MediaKind::Image)
    } else if has_suffix(&[".mp4", ".mov", ".webm"]) {
        Some(MediaKind::Video)
    } else {
        None
    }
}

/// The image or video one message carries, if it carries a usable one.
fn message_vid_input(message: &Message) -> Result<Option<VidInput>, &'static str> {
    let input = |file_id: &FileId, kind, mime_type: Option<&str>| {
        Ok(Some(VidInput {
            file_id: file_id.clone(),
            kind,
            mime_type: mime_type.map(str::to_string),
        }))
    };

    if let Some(photo) = message.photo().and_then(|sizes| sizes.last()) {
        return input(&photo.file.id, MediaKind::Image, None);
    }
    if let Some(video) = message.video() {
        let mime_type = video.mime_type.as_ref().map(|mime| mime.essence_str());
        return input(&video.file.id, MediaKind::Video, mime_type);
    }
    if let Some(animation) = message.animation() {
        let mime_type = animation.mime_type.as_ref().map(|mime| mime.essence_str());
        return input(&animation.file.id, MediaKind::Video, mime_type);
    }
    if let Some(note) = message.video_note() {
        return input(&note.file.id, MediaKind::Video, None);
    }
    if let Some(sticker) = message.sticker() {
        return if sticker.flags.is_video {
            input(&sticker.file.id, MediaKind::Video, Some("video/webm"))
        } else if sticker.flags.is_animated {
            Err("animated stickers cannot be used as an input")
        } else {
            input(&sticker.file.id, MediaKind::Image, Some("image/webp"))
        };
    }
    if let Some(document) = message.document() {
        let mime_type = document.mime_type.as_ref().map(|mime| mime.essence_str());
        if let Some(kind) = document_kind(mime_type, document.file_name.as_deref()) {
            // Keep the declared type only when it agrees with the kind; a
            // generic one would hide the real format from detection.
            let mime_type =
                mime_type.filter(|mime| mime.starts_with("image/") || mime.starts_with("video/"));
            return input(&document.file.id, kind, mime_type);
        }
    }
    Ok(None)
}

/// Inputs in the order the model reads them: the replied-to message first
/// (what is being animated, edited or extended), then what the command
/// message attaches. An album contributes all of its cached photos.
fn vid_inputs(
    message: &Message,
    rich: &RichUpdate,
    album_photos: impl Fn(&MediaGroupId) -> Vec<MediaGroupItem>,
) -> Result<Vec<VidInput>, &'static str> {
    let mut inputs: Vec<VidInput> = Vec::new();
    let mut seen = HashSet::new();
    let mut add = |input: VidInput| {
        if seen.insert(input.file_id.clone()) {
            inputs.push(input);
        }
    };

    for (source, rich_media) in message
        .reply_to_message()
        .map(|reply| (reply, rich.reply.as_deref()))
        .into_iter()
        .chain([(message, None)])
    {
        if let Some(attachments) = rich_media {
            for attachment in attachments {
                let kind = rich_media_kind(attachment);
                if matches!(kind, MediaKind::Image | MediaKind::Video) {
                    add(VidInput {
                        file_id: attachment.file_id.clone(),
                        kind,
                        mime_type: attachment
                            .mime_type
                            .as_ref()
                            .filter(|mime| kind_for_mime(mime) == kind)
                            .cloned(),
                    });
                }
            }
            continue;
        }
        let own = message_vid_input(source)?;
        let album = source
            .media_group_id()
            .map(&album_photos)
            .unwrap_or_default();
        // A captioned album item is dispatched as the command and never
        // cached, so it leads; a cached one keeps its place in the album.
        if let Some(own) = own {
            if !album.iter().any(|item| item.file_id == own.file_id) {
                add(own);
            }
        }
        for item in album {
            add(VidInput {
                file_id: item.file_id,
                kind: MediaKind::Image,
                mime_type: None,
            });
        }
    }
    Ok(inputs)
}

async fn download_vid_inputs(bot: &Bot, inputs: &[VidInput]) -> Result<Vec<MediaFile>> {
    let mut files = Vec::with_capacity(inputs.len());
    for input in inputs {
        let bytes = match get_file_url(bot, &input.file_id).await {
            Ok(url) => download_media(&url).await,
            Err(error) => {
                warn!("Could not resolve a /vid input file: {error}");
                None
            }
        }
        .ok_or_else(|| {
            VideoGenerationError(
                "an attached file could not be downloaded from Telegram, which serves bots files of up to 20 MB"
                    .to_string(),
            )
        })?;
        let mime_type = input
            .mime_type
            .clone()
            .or_else(|| detect_mime_type(&bytes))
            .unwrap_or_else(|| "application/octet-stream".to_string());
        files.push(MediaFile::new(bytes, mime_type, input.kind, None));
    }
    Ok(files)
}

async fn send_generated_video(
    bot: &Bot,
    chat_id: ChatId,
    video_bytes: &[u8],
    reply_to: MessageId,
) -> Result<Message> {
    retry_telegram("send_video", || {
        let input = InputFile::memory(video_bytes.to_vec()).file_name("video.mp4");
        bot.send_video(chat_id, input)
            .reply_parameters(ReplyParameters::new(reply_to))
    })
    .await
    .map_err(Into::into)
}

/// `/vid`: generate a clip from the prompt, using any image or video the
/// request replies to or attaches.
pub async fn vid_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    _prompt: Option<String>,
    rich: RichUpdate,
) -> Result<()> {
    if !check_access_control(&bot, &message, "vid").await {
        return Ok(());
    }
    if !CONFIG.gemini_api_available() {
        send_message_with_retry(
            &bot,
            message.chat.id,
            "The /vid command requires Gemini and is disabled.",
            Some(message.id),
        )
        .await?;
        return Ok(());
    }
    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        send_message_with_retry(
            &bot,
            message.chat.id,
            "Rate limit exceeded. Please try again later.",
            Some(message.id),
        )
        .await?;
        return Ok(());
    }
    let (task, prompt) = match vid_request_text(&message, &rich) {
        Ok(request) => request,
        Err(reason) => {
            send_message_with_retry(&bot, message.chat.id, &reason, Some(message.id)).await?;
            return Ok(());
        }
    };
    if prompt.is_empty() {
        send_message_with_retry(&bot, message.chat.id, VID_USAGE, Some(message.id)).await?;
        return Ok(());
    }

    let permit = state.acquire_heavy_command_permit().await;
    let audit = create_command_audit_context(&state, &message, "vid").await;
    let status = send_message_with_retry(
        &bot,
        message.chat.id,
        "Processing video request... This may take a few minutes.",
        Some(message.id),
    )
    .await?;
    run_with_status_message(
        &bot,
        message.chat.id,
        status.id,
        "Failed to generate your video. Please try again.",
        async {
            let _permit = permit;
            let _chat_action =
                start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::Typing);
            // Read the album cache only now: the siblings of a captioned
            // album item are separate updates that land just after it.
            let inputs = vid_inputs(&message, &rich, |album| state.media_group_items(album))
                .map_err(|reason| VideoGenerationError(reason.to_string()))?;
            let count = |kind| inputs.iter().filter(|input| input.kind == kind).count();
            check_video_input_counts(count(MediaKind::Image), count(MediaKind::Video))?;
            let media = download_vid_inputs(&bot, &inputs).await?;
            let prompt = video_model_prompt(&message, &prompt).await;

            let video = generate_video_with_omni(
                VideoGenerationRequest {
                    prompt: &prompt,
                    media: &media,
                    task,
                },
                audit.as_ref(),
            )
            .await?;
            send_generated_video(&bot, message.chat.id, &video, message.id).await?;
            edit_message_text_with_retry(
                &bot,
                message.chat.id,
                status.id,
                "Video generated below.",
            )
            .await?;
            Ok(())
        },
    )
    .await
}

fn attached_image(message: &Message) -> Result<Option<FileId>, Vid2Error> {
    if let Some(photos) = message.photo() {
        return photos
            .iter()
            .max_by_key(|photo| u64::from(photo.width) * u64::from(photo.height))
            .map(|photo| Some(photo.file.id.clone()))
            .ok_or_else(|| Vid2Error::Generation("the attached photo is empty".into()));
    }
    let Some(document) = message.document() else {
        return Ok(None);
    };
    let mime = document.mime_type.as_ref().map(|mime| mime.essence_str());
    let name = document
        .file_name
        .as_deref()
        .unwrap_or("")
        .to_ascii_lowercase();
    let supported_name = [".jpg", ".jpeg", ".png", ".webp"]
        .iter()
        .any(|suffix| name.ends_with(suffix));
    if mime.is_some_and(|mime| matches!(mime, "image/jpeg" | "image/png" | "image/webp"))
        || supported_name
    {
        return Ok(Some(document.file.id.clone()));
    }
    if mime.is_some_and(|mime| mime.starts_with("image/")) || name.ends_with(".gif") {
        return Err(Vid2Error::Generation(
            "first frame must be JPEG, PNG, or WebP".into(),
        ));
    }
    Ok(None)
}

fn selected_image(
    message: &Message,
    rich_reply: Option<&[RichMedia]>,
) -> Result<Option<FileId>, Vid2Error> {
    // Deliberately do not consult the media-group cache: the particular item
    // carrying the command/reply is the first frame chosen by the user.
    if let Some(image) = attached_image(message)? {
        return Ok(Some(image));
    }
    if let Some(attachments) = rich_reply {
        return Ok(attachments
            .iter()
            .find(|attachment| rich_media_kind(attachment) == MediaKind::Image)
            .map(|attachment| attachment.file_id.clone()));
    }
    message
        .reply_to_message()
        .map(attached_image)
        .transpose()
        .map(Option::flatten)
}

fn phase_text(phase: Vid2Phase) -> &'static str {
    match phase {
        Vid2Phase::PreparingImage => "Preparing the first frame for your video...",
        Vid2Phase::Submitting => "Submitting your video request...",
        Vid2Phase::Queued => "Your video is queued...",
        Vid2Phase::Running => "Generating your video...",
        Vid2Phase::Downloading => "Downloading your generated video...",
    }
}

fn video_caption(prompt: &str) -> String {
    const PREFIX: &str = "Generated by vid2 with prompt:\n";
    let budget = 1024usize.saturating_sub(PREFIX.encode_utf16().count());
    let truncated = prompt.encode_utf16().count() > budget;
    let mut remaining = budget.saturating_sub(usize::from(truncated));
    let mut excerpt: String = prompt
        .chars()
        .take_while(|character| {
            let width = character.len_utf16();
            if width > remaining {
                return false;
            }
            remaining -= width;
            true
        })
        .collect();
    if truncated {
        excerpt.push('…');
    }
    // Telegram spoilers cannot contain preformatted text.
    format!("{PREFIX}{}", escape_html(&excerpt))
}

struct PhaseReporter(tokio::task::JoinHandle<()>);

impl Drop for PhaseReporter {
    fn drop(&mut self) {
        self.0.abort();
    }
}

fn start_phase_reporter(
    bot: Bot,
    chat_id: ChatId,
    status_id: MessageId,
) -> (tokio::sync::watch::Sender<Vid2Phase>, PhaseReporter) {
    let (sender, mut receiver) = tokio::sync::watch::channel(Vid2Phase::PreparingImage);
    let task = tokio::spawn(async move {
        while receiver.changed().await.is_ok() {
            let phase = *receiver.borrow_and_update();
            // Phase updates are informational: failure must not interrupt a
            // paid generation, and a blocked edit must not delay the API job.
            if let Err(error) =
                edit_message_text_with_retry(&bot, chat_id, status_id, phase_text(phase)).await
            {
                warn!("Vid2 phase status edit failed: {error}");
            }
        }
    });
    (sender, PhaseReporter(task))
}

async fn deliver_video(
    bot: &Bot,
    chat_id: ChatId,
    status_id: MessageId,
    reply_to: MessageId,
    path: &Path,
    caption: &str,
) -> Result<()> {
    // Blurred like img2: the video and its prompt caption stay hidden until tapped.
    let caption = format!("<tg-spoiler>{caption}</tg-spoiler>");
    retry_telegram("vid2 send_video", || {
        bot.send_video(chat_id, InputFile::file(path.to_path_buf()))
            .caption(caption.clone())
            .parse_mode(ParseMode::Html)
            .disable_notification(false)
            .has_spoiler(true)
            .supports_streaming(true)
            .reply_parameters(ReplyParameters::new(reply_to))
    })
    .await
    .map_err(|source| VideoDeliveryError { source })?;
    // The new video reply is the success notification. Never remove the
    // progress message until Telegram confirms that reply was delivered.
    if let Err(error) = retry_telegram("vid2 delete progress", || {
        bot.delete_message(chat_id, status_id)
    })
    .await
    {
        warn!("Vid2 video was delivered, but progress-message deletion failed: {error}");
        if let Err(error) =
            edit_message_text_with_retry(bot, chat_id, status_id, "Video delivered below.").await
        {
            warn!("Vid2 delivered-status fallback edit failed: {error}");
        }
    }
    Ok(())
}

pub async fn vid2_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    _prompt: Option<String>,
    rich: RichUpdate,
) -> Result<()> {
    if !check_access_control(&bot, &message, "vid2").await {
        return Ok(());
    }
    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    if is_rate_limited(user_id) {
        send_message_with_retry(
            &bot,
            message.chat.id,
            "Rate limit exceeded. Please try again later.",
            Some(message.id),
        )
        .await?;
        return Ok(());
    }
    if !CONFIG.vid2_api_available() {
        send_message_with_retry(&bot, message.chat.id, "Vid2 video generation is disabled or unconfigured. Set ENABLE_VID2=true, VID2_BASE_URL, and VID2_API_KEY to enable it.", Some(message.id)).await?;
        return Ok(());
    }
    let prompt = command_prompt(&message, "/vid2", &rich);
    if prompt.trim().is_empty() {
        send_message_with_retry(
            &bot,
            message.chat.id,
            "Please provide a prompt for /vid2, or reply to a text message.",
            Some(message.id),
        )
        .await?;
        return Ok(());
    }
    let permit = state.acquire_heavy_command_permit().await;
    let audit = create_command_audit_context(&state, &message, "vid2").await;
    let status = send_message_with_retry(
        &bot,
        message.chat.id,
        "Preparing your video request...",
        Some(message.id),
    )
    .await?;
    run_with_status_message(
        &bot,
        message.chat.id,
        status.id,
        "Failed to generate your video. Please try again.",
        async {
            let permit = permit;
            let _heartbeat =
                start_chat_action_heartbeat(bot.clone(), message.chat.id, ChatAction::UploadVideo);
            let model_prompt = video_model_prompt(&message, &prompt).await;
            let (phases, reporter) = start_phase_reporter(bot.clone(), message.chat.id, status.id);
            let source = async {
                let file = selected_image(&message, rich.reply.as_deref())?;
                let Some(file) = file else {
                    return Ok(None);
                };
                phases.send_replace(Vid2Phase::PreparingImage);
                let url = get_file_url(&bot, &file).await.map_err(|_| {
                    Vid2Error::Generation("could not retrieve the selected first frame".into())
                })?;
                download_source_image(&url, CONFIG.vid2.request_timeout_secs)
                    .await
                    .map(Some)
            };
            let generated = generate_video_with_vid2(
                Vid2GenerationRequest {
                    prompt: &model_prompt,
                    chat_id: message.chat.id.0,
                    message_id: i64::from(message.id.0),
                },
                source,
                |phase| {
                    phases.send_replace(phase);
                },
                audit.as_ref(),
            )
            .await;
            // Stop phase updates before terminal/delivery status edits so an old
            // queued/running update cannot replace the final result.
            drop(reporter);
            let video = generated?;
            drop(permit);
            info!(
                "Sending saved Vid2 video: job_id={}, bytes={}, path={}",
                video
                    .job_id
                    .replace(CONFIG.vid2.api_key.trim(), "[REDACTED]"),
                video.byte_len,
                video.path.display()
            );
            if let Err(error) = edit_message_text_with_retry(
                &bot,
                message.chat.id,
                status.id,
                "Uploading your generated video...",
            )
            .await
            {
                warn!("Vid2 upload status edit failed: {error}");
            }
            let caption = video_caption(&prompt);
            deliver_video(
                &bot,
                message.chat.id,
                status.id,
                message.id,
                &video.path,
                &caption,
            )
            .await?;
            Ok(())
        },
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{response_with_status, ExpectedRequest, TestServer};
    use serde_json::{json, Value};

    fn message(mut fields: Value) -> Message {
        fields["message_id"] = json!(42);
        fields["date"] = json!(0);
        fields["chat"] = json!({"id":1,"type":"private"});
        serde_json::from_value(fields).unwrap()
    }

    fn photo(id: &str, size: u32) -> Value {
        json!({"file_id":id,"file_unique_id":id,"width":size,"height":size})
    }

    fn vid2_prompt(message: &Message) -> String {
        command_prompt(message, "/vid2", &RichUpdate::default())
    }

    #[test]
    fn prompt_comes_from_command_text_or_caption_then_replied_text() {
        assert_eq!(
            vid2_prompt(&message(json!({"text":"/vid2@test_bot a fox"}))),
            "a fox"
        );
        assert_eq!(
            vid2_prompt(&message(
                json!({"photo":[photo("x",1)],"caption":"/vid2 wind"})
            )),
            "wind"
        );
        let reply = message(json!({"text":"snowy woods"}));
        assert_eq!(
            vid2_prompt(&message(json!({"text":"/vid2","reply_to_message":reply}))),
            "snowy woods"
        );
        assert!(vid2_prompt(&message(json!({"text":"/vid2"}))).is_empty());
    }

    #[test]
    fn attached_photo_wins_over_replied_photo_and_chooses_largest_size() {
        let reply = message(json!({"photo":[photo("reply",20)],"media_group_id":"reply-album"}));
        let attached = message(
            json!({"photo":[photo("large",30),photo("small",2)],"caption":"/vid2 smile","reply_to_message":reply,"media_group_id":"attached-album"}),
        );
        assert_eq!(
            selected_image(&attached, None)
                .unwrap()
                .unwrap()
                .to_string(),
            "large"
        );
        let replied = message(
            json!({"text":"/vid2 smile","reply_to_message":message(json!({"photo":[photo("specific-reply",10)],"media_group_id":"another-album"}))}),
        );
        assert_eq!(
            selected_image(&replied, None).unwrap().unwrap().to_string(),
            "specific-reply"
        );
    }

    #[test]
    fn supported_documents_select_a_frame_and_unsupported_images_fail() {
        for (name, mime) in [
            ("photo.JPG", "image/jpeg"),
            ("photo.png", "image/png"),
            ("photo.webp", "image/webp"),
            ("photo.png", "application/octet-stream"),
        ] {
            let msg = message(
                json!({"document":{"file_id":"document","file_unique_id":"document","file_name":name,"mime_type":mime},"caption":"/vid2 smile"}),
            );
            assert_eq!(
                selected_image(&msg, None).unwrap().unwrap().to_string(),
                "document"
            );
        }
        let gif = message(
            json!({"document":{"file_id":"gif","file_unique_id":"gif","file_name":"photo.gif","mime_type":"image/gif"},"caption":"/vid2 smile"}),
        );
        assert!(selected_image(&gif, None).is_err());
    }

    #[test]
    fn video_caption_escapes_the_literal_prompt() {
        assert_eq!(
            video_caption("a fox"),
            "Generated by vid2 with prompt:\na fox"
        );
        let caption = video_caption(r#"<tg-spoiler>雪 & "fox's"</tg-spoiler>"#);
        assert_eq!(caption, "Generated by vid2 with prompt:\n&lt;tg-spoiler&gt;雪 &amp; &quot;fox&#39;s&quot;&lt;/tg-spoiler&gt;");
    }

    fn visible_caption(caption: &str) -> String {
        caption
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&amp;", "&")
    }

    #[test]
    fn video_caption_bounds_visible_utf16_and_keeps_complete_html_entities() {
        const PREFIX: &str = "Generated by vid2 with prompt:\n";
        let prompt = "🦊雪❄️ <>&\"' <tg-spoiler>literal</tg-spoiler> ".repeat(400);
        let caption = video_caption(&prompt);
        let visible = visible_caption(&caption);
        assert!(visible.encode_utf16().count() <= 1024);
        assert!(visible.starts_with(PREFIX));
        assert!(visible.ends_with('…'));
        assert!(prompt.starts_with(
            visible
                .strip_prefix(PREFIX)
                .unwrap()
                .strip_suffix('…')
                .unwrap()
        ));
        let escaped_prompt = caption.strip_prefix(PREFIX).unwrap();
        assert!(!escaped_prompt.contains('<') && !escaped_prompt.contains('>'));
        assert!(escaped_prompt.split('&').skip(1).all(|part| [
            "amp;", "lt;", "gt;", "quot;", "#39;"
        ]
        .iter()
        .any(|entity| part.starts_with(entity))));
        let budget = 1024 - PREFIX.encode_utf16().count();
        let boundary = format!("{}&", "x".repeat(budget - 1));
        let exact = video_caption(&boundary);
        assert_eq!(visible_caption(&exact).encode_utf16().count(), 1024);
        assert!(exact.ends_with("&amp;"));
        let over = video_caption(&format!("{boundary}🦊"));
        assert!(visible_caption(&over).encode_utf16().count() <= 1024);
        assert!(over.ends_with('…'));
    }

    fn vid_input(id: &str, kind: MediaKind, mime_type: Option<&str>) -> VidInput {
        VidInput {
            file_id: FileId(id.to_string()),
            kind,
            mime_type: mime_type.map(str::to_string),
        }
    }

    fn file(id: &str) -> Value {
        json!({"file_id":id,"file_unique_id":id})
    }

    fn video(id: &str) -> Value {
        json!({"file_id":id,"file_unique_id":id,"width":640,"height":360,"duration":5,"mime_type":"video/mp4"})
    }

    fn sticker(id: &str, animated: bool, video: bool) -> Message {
        message(
            json!({"sticker":{"file_id":id,"file_unique_id":id,"type":"regular","width":512,"height":512,"is_animated":animated,"is_video":video}}),
        )
    }

    fn no_albums(_: &MediaGroupId) -> Vec<MediaGroupItem> {
        Vec::new()
    }

    fn ids(inputs: &[VidInput]) -> Vec<&str> {
        inputs
            .iter()
            .map(|input| input.file_id.0.as_str())
            .collect()
    }

    #[test]
    fn vid_task_option_is_split_off_the_front_of_the_prompt() {
        let cases = [
            ("--t2v a fox", Some(VideoTask::TextToVideo), "a fox"),
            (
                "--i2v make it move",
                Some(VideoTask::ImageToVideo),
                "make it move",
            ),
            (
                "--ref  the cat plays",
                Some(VideoTask::ReferenceToVideo),
                "the cat plays",
            ),
            (
                "--edit\nmake it snow",
                Some(VideoTask::Edit),
                "make it snow",
            ),
            ("--EXTEND keep going", Some(VideoTask::Extend), "keep going"),
            // Phones turn a typed `--` into a dash.
            (
                "\u{2014}extend keep going",
                Some(VideoTask::Extend),
                "keep going",
            ),
            (
                "\u{2013}edit make it snow",
                Some(VideoTask::Edit),
                "make it snow",
            ),
            ("--extend", Some(VideoTask::Extend), ""),
            ("  a fox --edit  ", None, "a fox --edit"),
            ("\u{2014} a quiet morning", None, "\u{2014} a quiet morning"),
            ("", None, ""),
        ];
        for (prompt, task, rest) in cases {
            assert_eq!(split_task_option(prompt), Ok((task, rest)), "{prompt:?}");
        }

        let unknown = split_task_option("--cinematic a fox").unwrap_err();
        assert!(unknown.contains("--cinematic"), "{unknown}");
        assert!(unknown.contains("--extend"), "{unknown}");
    }

    #[test]
    fn vid_prompt_follows_the_option_or_falls_back_to_the_replied_text() {
        let rich = RichUpdate::default();
        let reply = message(json!({"text":"a storm rolls in"}));
        let request = message(json!({"text":"/vid --t2v","reply_to_message":reply}));
        assert_eq!(
            vid_request_text(&request, &rich),
            Ok((Some(VideoTask::TextToVideo), "a storm rolls in".to_string()))
        );
        assert_eq!(
            vid_request_text(&message(json!({"text":"/vid@test_bot a fox"})), &rich),
            Ok((None, "a fox".to_string()))
        );
        let captioned = message(json!({"photo":[photo("x",1)],"caption":"make it rain"}));
        assert_eq!(
            vid_request_text(
                &message(json!({"text":"/vid","reply_to_message":captioned})),
                &rich,
            ),
            Ok((None, "make it rain".to_string()))
        );
        assert_eq!(
            vid_request_text(&message(json!({"text":"/vid"})), &rich),
            Ok((None, String::new()))
        );
        assert!(vid_request_text(&message(json!({"text":"/vid --nope x"})), &rich).is_err());
    }

    #[test]
    fn video_commands_use_rich_reply_text_and_compatible_attachments() {
        let mut update = json!({"message": {
            "caption": "/vid --ref pan up",
            "photo": [photo("attached", 5)],
            "reply_to_message": {
                "message_id": 2, "date": 0, "chat": {"id": 1, "type":"private"},
                "rich_message": {"blocks": [
                    {"type": "heading", "text": "A frozen lake"},
                    {"type": "video", "video": {"file_id": "clip", "mime_type": "video/mp4"}},
                    {"type": "photo", "photo": [photo("frame", 10), photo("thumb", 1)]},
                    {"type": "audio", "audio": {"file_id": "sound"}},
                    {"type": "document", "document": {"file_id": "second-frame", "file_name": "frame.PNG"}}
                ]}
            }
        }});
        let rich = RichUpdate::from_update(&mut update);
        let vid = message(update["message"].clone());
        assert_eq!(
            vid_request_text(&vid, &rich),
            Ok((
                Some(VideoTask::ReferenceToVideo),
                "pan up\n\nReplied-to message:\nA frozen lake".into()
            ))
        );
        assert_eq!(
            ids(&vid_inputs(&vid, &rich, no_albums).unwrap()),
            ["clip", "frame", "second-frame", "attached"]
        );
        assert_eq!(
            selected_image(&vid, rich.reply.as_deref()).unwrap(),
            Some(FileId("attached".into()))
        );

        let reply = &update["message"]["reply_to_message"];
        let vid2 = message(json!({"text": "/vid2", "reply_to_message": reply}));
        assert_eq!(command_prompt(&vid2, "/vid2", &rich), "A frozen lake");
        assert_eq!(
            selected_image(&vid2, rich.reply.as_deref()).unwrap(),
            Some(FileId("frame".into()))
        );
        let vid2 = message(json!({"text": "/vid2 add snowfall", "reply_to_message": reply}));
        assert_eq!(
            command_prompt(&vid2, "/vid2", &rich),
            "add snowfall\n\nReplied-to message:\nA frozen lake"
        );
    }

    #[test]
    fn vid_reads_every_kind_of_image_and_video_message() {
        let cases = [
            (
                json!({"photo":[photo("small",2),photo("large",30)]}),
                vid_input("large", MediaKind::Image, None),
            ),
            (
                json!({"video":video("clip")}),
                vid_input("clip", MediaKind::Video, Some("video/mp4")),
            ),
            (
                json!({"animation":video("gif"),"document":file("gif")}),
                vid_input("gif", MediaKind::Video, Some("video/mp4")),
            ),
            (
                json!({"video_note":{"file_id":"round","file_unique_id":"round","length":240,"duration":4}}),
                vid_input("round", MediaKind::Video, None),
            ),
            (
                json!({"document":{"file_id":"png","file_unique_id":"png","file_name":"a.bin","mime_type":"image/png"}}),
                vid_input("png", MediaKind::Image, Some("image/png")),
            ),
            (
                json!({"document":{"file_id":"mov","file_unique_id":"mov","file_name":"Clip.MOV"}}),
                vid_input("mov", MediaKind::Video, None),
            ),
            (
                json!({"document":{"file_id":"jpg","file_unique_id":"jpg","file_name":"scan.jpeg","mime_type":"application/octet-stream"}}),
                vid_input("jpg", MediaKind::Image, None),
            ),
        ];
        for (fields, expected) in cases {
            let source = message(fields);
            assert_eq!(
                message_vid_input(&source).unwrap(),
                Some(expected),
                "{source:?}"
            );
        }

        for unrelated in [
            json!({"text":"just words"}),
            json!({"document":{"file_id":"pdf","file_unique_id":"pdf","file_name":"a.pdf","mime_type":"application/pdf"}}),
        ] {
            assert_eq!(message_vid_input(&message(unrelated)).unwrap(), None);
        }
    }

    #[test]
    fn vid_takes_static_and_video_stickers_but_refuses_animated_ones() {
        assert_eq!(
            message_vid_input(&sticker("static", false, false)).unwrap(),
            Some(vid_input("static", MediaKind::Image, Some("image/webp")))
        );
        assert_eq!(
            message_vid_input(&sticker("moving", false, true)).unwrap(),
            Some(vid_input("moving", MediaKind::Video, Some("video/webm")))
        );
        assert!(message_vid_input(&sticker("lottie", true, false)).is_err());
    }

    #[test]
    fn vid_reads_the_replied_message_before_the_attached_one() {
        let reply = message(json!({"video":video("base")}));
        let request = message(
            json!({"photo":[photo("reference",10)],"caption":"/vid add her","reply_to_message":reply}),
        );
        let inputs = vid_inputs(&request, &RichUpdate::default(), no_albums).unwrap();
        assert_eq!(ids(&inputs), ["base", "reference"]);
        assert_eq!(inputs[0].kind, MediaKind::Video);
        assert_eq!(inputs[1].kind, MediaKind::Image);

        assert!(vid_inputs(
            &message(json!({"text":"/vid a fox"})),
            &RichUpdate::default(),
            no_albums,
        )
        .unwrap()
        .is_empty());
    }

    #[test]
    fn vid_uses_a_whole_album_in_order_without_repeating_a_file() {
        let album = |id: &MediaGroupId| {
            assert_eq!(id.0, "album");
            ["second", "third"]
                .into_iter()
                .map(|id| MediaGroupItem {
                    file_id: FileId(id.to_string()),
                })
                .collect()
        };
        // The captioned item is dispatched as the command, so the cache holds
        // only its siblings: it leads, and they follow in album order.
        let captioned = message(
            json!({"photo":[photo("first",10)],"caption":"/vid morph","media_group_id":"album"}),
        );
        assert_eq!(
            ids(&vid_inputs(&captioned, &RichUpdate::default(), album).unwrap()),
            ["first", "second", "third"]
        );

        // A reply to a cached album item keeps the album order and lists the
        // replied item once.
        let reply = message(json!({"photo":[photo("third",10)],"media_group_id":"album"}));
        let request = message(json!({"text":"/vid morph","reply_to_message":reply}));
        assert_eq!(
            ids(&vid_inputs(&request, &RichUpdate::default(), album).unwrap()),
            ["second", "third"]
        );
    }

    #[test]
    fn vid_reports_an_unusable_sticker_wherever_it_is() {
        let request =
            message(json!({"text":"/vid dance","reply_to_message":sticker("lottie", true, false)}));
        assert!(vid_inputs(&request, &RichUpdate::default(), no_albums).is_err());
    }

    fn telegram_response(ok: bool, retry: bool) -> Vec<u8> {
        let body = if ok {
            json!({"ok":true,"result":{"message_id":9,"date":0,"chat":{"id":1,"type":"private"}}})
        } else if retry {
            json!({"ok":false,"error_code":429,"description":"Too Many Requests: retry after 0","parameters":{"retry_after":0}})
        } else {
            json!({"ok":false,"error_code":400,"description":"Bad Request: wrong file identifier/HTTP URL specified"})
        };
        response_with_status(
            if ok {
                200
            } else if retry {
                429
            } else {
                400
            },
            serde_json::to_vec(&body).unwrap(),
        )
    }

    async fn saved_file() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT_DELIVERY_FILE: AtomicU64 = AtomicU64::new(0);
        let directory = Path::new("target/test-vid2-media");
        tokio::fs::create_dir_all(directory).await.unwrap();
        let path = directory.join(format!(
            "delivery_{}_{}_{}.mp4",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap(),
            NEXT_DELIVERY_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        tokio::fs::write(&path, b"SAVED_VIDEO_BYTES").await.unwrap();
        path
    }

    fn delete_progress_response(ok: bool, retry: bool) -> ExpectedRequest {
        ExpectedRequest::new(
            "POST",
            "/bot123:test/DeleteMessage",
            if ok {
                response_with_status(200, br#"{"ok":true,"result":true}"#.to_vec())
            } else {
                telegram_response(false, retry)
            },
        )
        .with_body_fragment(br#""chat_id":1"#)
        .with_body_fragment(br#""message_id":7"#)
    }

    #[tokio::test]
    async fn successful_video_reply_spoils_video_and_full_caption_and_removes_only_progress() {
        let caption = video_caption(r#"<tg-spoiler>A & "B"</tg-spoiler>"#);
        let server = TestServer::new(vec![
            ExpectedRequest::new(
                "POST",
                "/bot123:test/SendVideo",
                telegram_response(true, false),
            )
            .with_body_fragment(b"SAVED_VIDEO_BYTES")
            .with_body_fragment(b"name=\"reply_parameters\"\r\n\r\n{\"message_id\":42}")
            .with_body_fragment(b"name=\"parse_mode\"\r\n\r\nHTML")
            .with_body_fragment(b"name=\"disable_notification\"\r\n\r\nfalse")
            .with_body_fragment(b"name=\"has_spoiler\"\r\n\r\ntrue")
            .with_body_fragment(b"name=\"caption\"\r\n\r\n<tg-spoiler>Generated by vid2 with prompt:\n&lt;tg-spoiler&gt;A &amp; &quot;B&quot;&lt;/tg-spoiler&gt;</tg-spoiler>\r\n"),
            delete_progress_response(true, false),
        ]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let path = saved_file().await;
        deliver_video(
            &bot,
            ChatId(1),
            MessageId(7),
            MessageId(42),
            &path,
            &caption,
        )
        .await
        .unwrap();
        tokio::fs::remove_file(path).await.unwrap();
        server.join().unwrap();
    }

    #[tokio::test]
    async fn video_delivery_retries_the_same_saved_file() {
        let server = TestServer::new(vec![
            ExpectedRequest::new(
                "POST",
                "/bot123:test/SendVideo",
                telegram_response(false, true),
            )
            .with_body_fragment(b"SAVED_VIDEO_BYTES"),
            ExpectedRequest::new(
                "POST",
                "/bot123:test/SendVideo",
                telegram_response(true, false),
            )
            .with_body_fragment(b"SAVED_VIDEO_BYTES"),
            delete_progress_response(true, false),
        ]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let path = saved_file().await;
        deliver_video(
            &bot,
            ChatId(1),
            MessageId(7),
            MessageId(42),
            &path,
            &video_caption("a fox"),
        )
        .await
        .unwrap();
        assert_eq!(tokio::fs::read(&path).await.unwrap(), b"SAVED_VIDEO_BYTES");
        tokio::fs::remove_file(path).await.unwrap();
        server.join().unwrap();
    }

    #[tokio::test]
    async fn delivery_failure_is_distinct_and_keeps_the_completed_file() {
        let server = TestServer::new(vec![
            ExpectedRequest::new(
                "POST",
                "/bot123:test/SendVideo",
                telegram_response(false, false),
            )
            .with_body_fragment(b"SAVED_VIDEO_BYTES"),
            ExpectedRequest::new(
                "POST",
                "/bot123:test/EditMessageText",
                telegram_response(true, false),
            )
            .with_body_fragment(br#""message_id":7"#)
            .with_body_fragment(
                b"Your video was generated and saved locally, but Telegram rejected its delivery.",
            ),
        ]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let path = saved_file().await;
        let caption = video_caption("a fox");
        let error = run_with_status_message(
            &bot,
            ChatId(1),
            MessageId(7),
            "generic failure",
            deliver_video(
                &bot,
                ChatId(1),
                MessageId(7),
                MessageId(42),
                &path,
                &caption,
            ),
        )
        .await
        .unwrap_err();
        assert!(error.is::<VideoDeliveryError>());
        assert_eq!(tokio::fs::read(&path).await.unwrap(), b"SAVED_VIDEO_BYTES");
        tokio::fs::remove_file(path).await.unwrap();
        server.join().unwrap();
    }

    #[tokio::test]
    async fn progress_deletion_retries_without_resending_the_confirmed_video() {
        let server = TestServer::new(vec![
            ExpectedRequest::new(
                "POST",
                "/bot123:test/SendVideo",
                telegram_response(true, false),
            )
            .with_body_fragment(b"SAVED_VIDEO_BYTES"),
            delete_progress_response(false, true),
            delete_progress_response(true, false),
        ]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let path = saved_file().await;
        deliver_video(
            &bot,
            ChatId(1),
            MessageId(7),
            MessageId(42),
            &path,
            &video_caption("a fox"),
        )
        .await
        .unwrap();
        tokio::fs::remove_file(path).await.unwrap();
        server.join().unwrap();
    }

    #[tokio::test]
    async fn deletion_rejection_edits_delivered_status_and_keeps_delivery_successful() {
        let server = TestServer::new(vec![
            ExpectedRequest::new(
                "POST",
                "/bot123:test/SendVideo",
                telegram_response(true, false),
            )
            .with_body_fragment(b"SAVED_VIDEO_BYTES"),
            delete_progress_response(false, false),
            ExpectedRequest::new(
                "POST",
                "/bot123:test/EditMessageText",
                telegram_response(true, false),
            )
            .with_body_fragment(br#""message_id":7"#)
            .with_body_fragment(b"Video delivered below."),
        ]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let path = saved_file().await;
        let caption = video_caption("a fox");
        run_with_status_message(
            &bot,
            ChatId(1),
            MessageId(7),
            "generic failure",
            deliver_video(
                &bot,
                ChatId(1),
                MessageId(7),
                MessageId(42),
                &path,
                &caption,
            ),
        )
        .await
        .unwrap();
        tokio::fs::remove_file(path).await.unwrap();
        server.join().unwrap();
    }

    #[tokio::test]
    async fn exhausted_deletion_retries_and_failed_fallback_edit_do_not_fail_delivery() {
        let server = TestServer::new(vec![
            ExpectedRequest::new(
                "POST",
                "/bot123:test/SendVideo",
                telegram_response(true, false),
            )
            .with_body_fragment(b"SAVED_VIDEO_BYTES"),
            delete_progress_response(false, true),
            delete_progress_response(false, true),
            delete_progress_response(false, true),
            ExpectedRequest::new(
                "POST",
                "/bot123:test/EditMessageText",
                telegram_response(false, false),
            )
            .with_body_fragment(br#""message_id":7"#)
            .with_body_fragment(b"Video delivered below."),
        ]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let path = saved_file().await;
        let caption = video_caption("a fox");
        run_with_status_message(
            &bot,
            ChatId(1),
            MessageId(7),
            "generic failure",
            deliver_video(
                &bot,
                ChatId(1),
                MessageId(7),
                MessageId(42),
                &path,
                &caption,
            ),
        )
        .await
        .unwrap();
        tokio::fs::remove_file(path).await.unwrap();
        server.join().unwrap();
    }
}
