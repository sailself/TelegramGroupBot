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
            if retry_telegram("edit generated image", || {
                bot.edit_message_media(chat, status, media.clone())
            })
            .await
            .is_ok()
            {
                continue;
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
            deliver_generated_images(
                &bot,
                ChatId(1),
                MessageId(2),
                MessageId(3),
                vec![InputFile::memory(b"GENERATED_IMAGE_BYTES".to_vec())],
                "caption",
                command == "img2",
            )
            .await
            .unwrap();
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
