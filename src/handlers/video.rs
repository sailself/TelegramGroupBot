//! Hidden `/vid2` command and local MP4 delivery.

use std::path::Path;

use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::{ChatAction, FileId, InputFile, MessageId, ReplyParameters};
use tracing::{info, warn};

use crate::config::CONFIG;
use crate::handlers::access::{check_access_control, is_rate_limited};
use crate::handlers::media::get_file_url;
use crate::llm::audit::create_command_audit_context;
use crate::llm::vid2_video::{
    download_source_image, generate_video_with_vid2, Vid2Error, Vid2GenerationRequest, Vid2Phase,
};
use crate::state::AppState;
use crate::utils::telegram::{
    edit_message_text_with_retry, message_text_or_caption, retry_telegram, run_with_status_message,
    send_message_with_retry, start_chat_action_heartbeat, strip_command_prefix, VideoDeliveryError,
};

fn prompt_for_message(message: &Message) -> String {
    let prompt = message_text_or_caption(message)
        .map(|text| strip_command_prefix(text, "/vid2"))
        .unwrap_or_default();
    if !prompt.is_empty() {
        return prompt;
    }
    message
        .reply_to_message()
        .and_then(message_text_or_caption)
        .unwrap_or_default()
        .trim()
        .to_string()
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

fn selected_image(message: &Message) -> Result<Option<FileId>, Vid2Error> {
    // Deliberately do not consult the media-group cache: the particular item
    // carrying the command/reply is the first frame chosen by the user.
    if let Some(image) = attached_image(message)? {
        return Ok(Some(image));
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

fn video_caption(prompt: &str, seconds: u64) -> String {
    let prefix = format!("vid2 · {seconds}s\n");
    let mut remaining = 1024usize.saturating_sub(prefix.encode_utf16().count());
    let excerpt: String = prompt
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
    format!("{prefix}{excerpt}")
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
    reply_to: MessageId,
    path: &Path,
    caption: &str,
) -> Result<()> {
    retry_telegram("vid2 send_video", || {
        bot.send_video(chat_id, InputFile::file(path.to_path_buf()))
            .caption(caption.to_string())
            .supports_streaming(true)
            .reply_parameters(ReplyParameters::new(reply_to))
    })
    .await
    .map_err(|source| VideoDeliveryError { source })?;
    Ok(())
}

pub async fn vid2_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    _prompt: Option<String>,
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
    let prompt = prompt_for_message(&message);
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
            let (phases, reporter) = start_phase_reporter(bot.clone(), message.chat.id, status.id);
            let source = async {
                let file = selected_image(&message)?;
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
                    prompt: &prompt,
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
            let caption = video_caption(&prompt, CONFIG.vid2.seconds);
            deliver_video(&bot, message.chat.id, message.id, &video.path, &caption).await?;
            if let Err(error) = edit_message_text_with_retry(
                &bot,
                message.chat.id,
                status.id,
                "Your video is ready.",
            )
            .await
            {
                warn!("Vid2 success status edit failed: {error}");
            }
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

    #[test]
    fn prompt_comes_from_command_text_or_caption_then_replied_text() {
        assert_eq!(
            prompt_for_message(&message(json!({"text":"/vid2@test_bot a fox"}))),
            "a fox"
        );
        assert_eq!(
            prompt_for_message(&message(
                json!({"photo":[photo("x",1)],"caption":"/vid2 wind"})
            )),
            "wind"
        );
        let reply = message(json!({"text":"snowy woods"}));
        assert_eq!(
            prompt_for_message(&message(json!({"text":"/vid2","reply_to_message":reply}))),
            "snowy woods"
        );
        assert!(prompt_for_message(&message(json!({"text":"/vid2"}))).is_empty());
    }

    #[test]
    fn attached_photo_wins_over_replied_photo_and_chooses_largest_size() {
        let reply = message(json!({"photo":[photo("reply",20)],"media_group_id":"reply-album"}));
        let attached = message(
            json!({"photo":[photo("large",30),photo("small",2)],"caption":"/vid2 smile","reply_to_message":reply,"media_group_id":"attached-album"}),
        );
        assert_eq!(
            selected_image(&attached).unwrap().unwrap().to_string(),
            "large"
        );
        let replied = message(
            json!({"text":"/vid2 smile","reply_to_message":message(json!({"photo":[photo("specific-reply",10)],"media_group_id":"another-album"}))}),
        );
        assert_eq!(
            selected_image(&replied).unwrap().unwrap().to_string(),
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
                selected_image(&msg).unwrap().unwrap().to_string(),
                "document"
            );
        }
        let gif = message(
            json!({"document":{"file_id":"gif","file_unique_id":"gif","file_name":"photo.gif","mime_type":"image/gif"},"caption":"/vid2 smile"}),
        );
        assert!(selected_image(&gif).is_err());
    }

    #[test]
    fn video_caption_bounds_mixed_emoji_and_cjk_in_utf16_units() {
        let prompt = "🦊雪❄️".repeat(400);
        let caption = video_caption(&prompt, 5);
        assert!(caption.encode_utf16().count() <= 1024);
        assert!(caption.starts_with("vid2 · 5s\n🦊雪❄️"));
        assert_eq!(video_caption("a fox", 5), "vid2 · 5s\na fox");
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
        let directory = Path::new("target/test-vid2-media");
        tokio::fs::create_dir_all(directory).await.unwrap();
        let path = directory.join(format!(
            "delivery_{}_{}.mp4",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap()
        ));
        tokio::fs::write(&path, b"SAVED_VIDEO_BYTES").await.unwrap();
        path
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
        ]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let path = saved_file().await;
        deliver_video(&bot, ChatId(1), MessageId(42), &path, "caption")
            .await
            .unwrap();
        assert_eq!(tokio::fs::read(&path).await.unwrap(), b"SAVED_VIDEO_BYTES");
        tokio::fs::remove_file(path).await.unwrap();
        server.join().unwrap();
    }

    #[tokio::test]
    async fn delivery_failure_is_distinct_and_keeps_the_completed_file() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "POST",
            "/bot123:test/SendVideo",
            telegram_response(false, false),
        )
        .with_body_fragment(b"SAVED_VIDEO_BYTES")]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let path = saved_file().await;
        let error = deliver_video(&bot, ChatId(1), MessageId(42), &path, "caption")
            .await
            .unwrap_err();
        assert!(error.is::<VideoDeliveryError>());
        assert_eq!(tokio::fs::read(&path).await.unwrap(), b"SAVED_VIDEO_BYTES");
        tokio::fs::remove_file(path).await.unwrap();
        server.join().unwrap();
    }
}
