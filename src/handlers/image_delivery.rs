//! Delivery of already-generated images. Retrying here never repeats generation.
use crate::utils::telegram::{edit_message_text_with_retry, retry_telegram, ImageDeliveryError};
use anyhow::Result;
use teloxide::{
    prelude::*,
    types::{InputFile, InputMedia, InputMediaPhoto, MessageId, ParseMode, ReplyParameters},
};

pub(super) async fn deliver_generated_images(
    bot: &Bot,
    chat: ChatId,
    status: MessageId,
    reply: MessageId,
    images: Vec<InputFile>,
    caption: &str,
    spoiler: bool,
) -> Result<()> {
    if images.is_empty() {
        anyhow::bail!("Image generation returned no images");
    }
    let total = images.len();
    let caption = if spoiler {
        format!("<tg-spoiler>{caption}</tg-spoiler>")
    } else {
        caption.to_string()
    };
    for (delivered, image) in images.into_iter().enumerate() {
        if delivered == 0 {
            let mut photo = InputMediaPhoto::new(image.clone())
                .caption(caption.clone())
                .parse_mode(ParseMode::Html);
            if spoiler {
                photo = photo.spoiler();
            }
            let media = InputMedia::Photo(photo);
            match retry_telegram("edit generated image", || {
                bot.edit_message_media(chat, status, media.clone())
            })
            .await
            {
                Ok(_)
                | Err(teloxide::RequestError::Api(teloxide::ApiError::MessageNotModified)) => {
                    continue
                }
                Err(_) => {}
            }
        }
        retry_telegram("send generated image", || {
            let mut request = bot
                .send_photo(chat, image.clone())
                .reply_parameters(ReplyParameters::new(reply))
                .has_spoiler(spoiler);
            if delivered == 0 {
                request = request.caption(caption.clone()).parse_mode(ParseMode::Html);
            }
            request
        })
        .await
        .map_err(|source| ImageDeliveryError {
            source,
            delivered,
            total,
        })?;
        if delivered == 0 {
            let _ = edit_message_text_with_retry(bot, chat, status, "Generated image below.").await;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{response_with_status, ExpectedRequest, TestServer};

    fn response(ok: bool, retry: bool) -> Vec<u8> {
        let body = if ok {
            serde_json::json!({"ok":true,"result":{"message_id":9,"date":0,"chat":{"id":1,"type":"private"}}})
        } else if retry {
            serde_json::json!({"ok":false,"error_code":429,"description":"Too Many Requests: retry after 0","parameters":{"retry_after":0}})
        } else {
            serde_json::json!({"ok":false,"error_code":400,"description":"Bad Request: wrong file identifier/HTTP URL specified"})
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
    fn photo_request(method: &str, ok: bool, retry: bool) -> ExpectedRequest {
        ExpectedRequest::new(
            "POST",
            &format!("/bot123:test/{method}"),
            response(ok, retry),
        )
        .with_body_fragment(b"GENERATED_IMAGE_BYTES")
    }

    #[tokio::test]
    async fn multiple_images_keep_their_generation_order() {
        let server = TestServer::new(vec![
            ExpectedRequest::new(
                "POST",
                "/bot123:test/EditMessageMedia",
                response(true, false),
            )
            .with_body_fragment(b"FIRST_IMAGE"),
            ExpectedRequest::new("POST", "/bot123:test/SendPhoto", response(true, false))
                .with_body_fragment(b"SECOND_IMAGE"),
            ExpectedRequest::new("POST", "/bot123:test/SendPhoto", response(true, false))
                .with_body_fragment(b"THIRD_IMAGE"),
        ]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let files = [b"FIRST_IMAGE".as_slice(), b"SECOND_IMAGE", b"THIRD_IMAGE"]
            .into_iter()
            .map(|bytes| InputFile::memory(bytes.to_vec()))
            .collect();
        deliver_generated_images(
            &bot,
            ChatId(1),
            MessageId(2),
            MessageId(3),
            files,
            "caption",
            false,
        )
        .await
        .unwrap();
        server.join().unwrap();
    }

    #[tokio::test]
    async fn already_applied_edit_does_not_send_a_duplicate_photo() {
        let unchanged=response_with_status(400,br#"{"ok":false,"error_code":400,"description":"Bad Request: message is not modified: specified new message content and reply markup are exactly the same as a current content and reply markup of the message"}"#.to_vec());
        let server = TestServer::new(vec![
            photo_request("EditMessageMedia", false, true),
            ExpectedRequest::new("POST", "/bot123:test/EditMessageMedia", unchanged)
                .with_body_fragment(b"GENERATED_IMAGE_BYTES"),
        ]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        deliver_generated_images(
            &bot,
            ChatId(1),
            MessageId(2),
            MessageId(3),
            vec![InputFile::memory(b"GENERATED_IMAGE_BYTES".to_vec())],
            "caption",
            false,
        )
        .await
        .unwrap();
        server.join().unwrap();
    }

    #[tokio::test]
    async fn transient_edit_recovers_without_duplicate_send() {
        let server = TestServer::new(vec![
            photo_request("EditMessageMedia", false, true),
            photo_request("EditMessageMedia", true, false),
        ]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        deliver_generated_images(
            &bot,
            ChatId(1),
            MessageId(2),
            MessageId(3),
            vec![InputFile::memory(b"GENERATED_IMAGE_BYTES".to_vec())],
            "caption",
            false,
        )
        .await
        .unwrap();
        server.join().unwrap();
    }

    #[tokio::test]
    async fn img2_file_retries_preserve_bytes_and_spoiler() {
        let path = crate::db::test_support::test_db_path("img2-payload").with_extension("png");
        std::fs::write(&path, b"GENERATED_IMAGE_BYTES").unwrap();
        let server = TestServer::new(vec![
            photo_request("EditMessageMedia", false, false),
            photo_request("SendPhoto", false, true).with_body_fragment(b"has_spoiler"),
            photo_request("SendPhoto", true, false)
                .with_body_fragment(b"has_spoiler")
                .with_body_fragment(b"\r\n\r\ntrue\r\n")
                .with_body_fragment(b"<tg-spoiler>caption</tg-spoiler>"),
            ExpectedRequest::new(
                "POST",
                "/bot123:test/EditMessageText",
                response(true, false),
            ),
        ]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        deliver_generated_images(
            &bot,
            ChatId(1),
            MessageId(2),
            MessageId(3),
            vec![InputFile::file(path.clone())],
            "caption",
            true,
        )
        .await
        .unwrap();
        server.join().unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn all_image_command_delivery_modes_preserve_payload_caption_and_retry() {
        for command in ["img", "image", "img2", "paintme", "portraitme"] {
            let caption = if command == "img2" {
                "<tg-spoiler>caption</tg-spoiler>"
            } else {
                "caption"
            };
            let server = TestServer::new(vec![
                photo_request("EditMessageMedia", false, false),
                photo_request("SendPhoto", false, true)
                    .with_body_fragment(caption.as_bytes())
                    .with_body_fragment(b"reply_parameters"),
                photo_request("SendPhoto", true, false)
                    .with_body_fragment(caption.as_bytes())
                    .with_body_fragment(b"reply_parameters"),
                ExpectedRequest::new(
                    "POST",
                    "/bot123:test/EditMessageText",
                    response(false, false),
                ),
            ]);
            let bot = Bot::new("123:test").set_api_url(server.base_url());
            let generations = std::sync::atomic::AtomicUsize::new(0);
            let generated = async {
                generations.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                vec![InputFile::memory(b"GENERATED_IMAGE_BYTES".to_vec())]
            }
            .await;
            deliver_generated_images(
                &bot,
                ChatId(1),
                MessageId(2),
                MessageId(3),
                generated,
                "caption",
                command == "img2",
            )
            .await
            .unwrap();
            assert_eq!(generations.load(std::sync::atomic::Ordering::SeqCst), 1);
            server.join().unwrap();
        }
    }

    #[tokio::test]
    async fn partial_delivery_records_confirmed_count_and_stops_in_order() {
        let server = TestServer::new(vec![
            photo_request("EditMessageMedia", true, false),
            photo_request("SendPhoto", false, true),
            photo_request("SendPhoto", false, true),
            photo_request("SendPhoto", false, true),
        ]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let error = deliver_generated_images(
            &bot,
            ChatId(1),
            MessageId(2),
            MessageId(3),
            vec![InputFile::memory(b"GENERATED_IMAGE_BYTES".to_vec()); 3],
            "caption",
            false,
        )
        .await
        .unwrap_err();
        let error = error.downcast_ref::<ImageDeliveryError>().unwrap();
        assert_eq!((error.delivered, error.total), (1, 3));
        server.join().unwrap();
    }

    #[tokio::test]
    async fn permanent_rejection_is_not_retried() {
        let server = TestServer::new(vec![
            photo_request("EditMessageMedia", false, false),
            photo_request("SendPhoto", false, false),
        ]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let error = deliver_generated_images(
            &bot,
            ChatId(1),
            MessageId(2),
            MessageId(3),
            vec![InputFile::memory(b"GENERATED_IMAGE_BYTES".to_vec())],
            "caption",
            false,
        )
        .await
        .unwrap_err();
        assert_eq!(
            error
                .downcast_ref::<ImageDeliveryError>()
                .unwrap()
                .delivered,
            0
        );
        server.join().unwrap();
    }
}
