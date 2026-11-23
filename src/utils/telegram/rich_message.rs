//! Project Bot API rich messages into the text and file references used by handlers.

use std::ops::Range;

use serde_json::{json, Value};
use teloxide::types::FileId;

use crate::llm::media::MediaKind;

#[derive(Debug, Clone)]
pub(crate) struct RichMedia {
    pub file_id: FileId,
    /// MIME type supplied by Telegram for this file.
    pub mime_type: Option<String>,
    pub fallback_mime_type: Option<&'static str>,
    pub file_name: Option<String>,
    pub kind: Option<MediaKind>,
}

#[derive(Debug, Clone)]
pub(crate) struct RichMention {
    pub username: String,
    /// Byte range of the displayed mention in the projected text.
    pub range: Range<usize>,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct RichUpdate {
    pub message: Option<Vec<RichMedia>>,
    pub message_mentions: Vec<RichMention>,
    pub reply: Option<Vec<RichMedia>>,
}

impl RichUpdate {
    pub(crate) fn from_update(update: &mut Value) -> Self {
        let Some(message) = update.get_mut("message") else {
            return Self::default();
        };
        let (media, mentions) = project_message(message).map_or_else(
            || (None, Vec::new()),
            |projection| (Some(projection.media), projection.mentions),
        );
        Self {
            message: media,
            message_mentions: mentions,
            reply: message
                .get_mut("reply_to_message")
                .and_then(project_message)
                .map(|projection| projection.media),
        }
    }
}

struct RichMessageProjection {
    media: Vec<RichMedia>,
    mentions: Vec<RichMention>,
}

fn project_message(message: &mut Value) -> Option<RichMessageProjection> {
    let message = message.as_object_mut()?;
    let rich = message.remove("rich_message")?;
    let mut content = RichContent::default();
    content.blocks(&rich["blocks"]);
    // The separator was added outside any entity. A newline supplied by the
    // sender can be part of an entity's range and must stay in the text.
    if content.trailing_separator {
        content.text.pop();
    }
    // A text field (even empty for media-only posts) gives teloxide a common
    // message kind, so it retains the parent, sender and forwarding metadata.
    message.insert("text".into(), Value::String(content.text));
    message.insert("entities".into(), Value::Array(content.entities));
    Some(RichMessageProjection {
        media: content.media,
        mentions: content.mentions,
    })
}

#[derive(Default)]
struct RichContent {
    text: String,
    utf16_len: usize,
    trailing_separator: bool,
    entities: Vec<Value>,
    media: Vec<RichMedia>,
    mentions: Vec<RichMention>,
}

fn items(value: &Value) -> &[Value] {
    value.as_array().map_or(&[], Vec::as_slice)
}

impl RichContent {
    fn blocks(&mut self, blocks: &Value) {
        for block in items(blocks) {
            let kind = block["type"].as_str().unwrap_or_default();
            match kind {
                "list" => {
                    for item in items(&block["items"]) {
                        self.line(&item["label"]);
                        self.blocks(&item["blocks"]);
                    }
                }
                "table" => {
                    self.line(&block["caption"]);
                    for row in items(&block["cells"]) {
                        for (index, cell) in items(row).iter().enumerate() {
                            if index > 0 {
                                self.push_text(" | ");
                            }
                            self.inline(&cell["text"]);
                        }
                        self.line_break();
                    }
                }
                "buttons" => {
                    for button in items(&block["buttons"]) {
                        self.line(&json!({"type": "button", "button": button}));
                    }
                }
                _ => {
                    self.line(&block["text"]);
                    self.line(&block["expression"]);
                    self.line(&block["summary"]);
                    self.collect_media(block, kind);
                    self.blocks(&block["blocks"]);
                    self.line(&block["caption"]["text"]);
                    self.line(&block["caption"]["credit"]);
                    self.line(&block["credit"]);
                }
            }
        }
    }

    fn collect_media(&mut self, block: &Value, kind: &str) {
        let (file, media_kind, default_mime) = match kind {
            "photo" => {
                let Some(photo) = items(&block["photo"])
                    .iter()
                    .filter(|photo| photo["file_id"].as_str().is_some_and(|id| !id.is_empty()))
                    .max_by_key(|photo| {
                        let width = photo["width"].as_u64().unwrap_or_default();
                        let height = photo["height"].as_u64().unwrap_or_default();
                        width.saturating_mul(height)
                    })
                else {
                    return;
                };
                (photo, Some(MediaKind::Image), None)
            }
            "animation" | "video" => (&block[kind], Some(MediaKind::Video), Some("video/mp4")),
            "audio" => (&block[kind], Some(MediaKind::Audio), Some("audio/mpeg")),
            "voice_note" => (&block[kind], Some(MediaKind::Audio), Some("audio/ogg")),
            "document" => (&block[kind], None, None),
            _ => return,
        };
        if let Some(file_id) = file["file_id"].as_str().filter(|id| !id.is_empty()) {
            self.media.push(RichMedia {
                file_id: FileId(file_id.to_owned()),
                mime_type: file["mime_type"]
                    .as_str()
                    .filter(|mime| !mime.trim().is_empty())
                    .map(str::to_owned),
                fallback_mime_type: default_mime,
                file_name: file["file_name"].as_str().map(str::to_owned),
                kind: media_kind,
            });
        }
    }

    fn line(&mut self, value: &Value) {
        self.inline(value);
        self.line_break();
    }

    fn line_break(&mut self) {
        if !self.text.is_empty() && !self.text.ends_with('\n') {
            self.text.push('\n');
            self.utf16_len += 1;
            self.trailing_separator = true;
        }
    }

    fn push_text(&mut self, text: &str) {
        if !text.is_empty() {
            self.text.push_str(text);
            self.utf16_len += text.encode_utf16().count();
            self.trailing_separator = false;
        }
    }

    fn inline(&mut self, value: &Value) {
        match value {
            Value::String(text) => self.push_text(text),
            Value::Array(children) => {
                for child in children {
                    self.inline(child);
                }
            }
            Value::Object(_) => self.inline_object(value),
            _ => {}
        }
    }

    fn inline_object(&mut self, value: &Value) {
        let start = self.utf16_len;
        let kind = value["type"].as_str().unwrap_or_default();
        let entity = match kind {
            "url" => {
                self.inline(&value["text"]);
                link_entity(&value["url"])
            }
            "button" => {
                let button = &value["button"];
                self.inline(&button["text"]);
                link_entity(&button["url"])
            }
            "mention" => {
                // A legacy Mention has only display text. Keep the username
                // separately so it stays tied to the actual mention target.
                let start = self.text.len();
                self.inline(&value["text"]);
                if let Some(username) = value["username"].as_str() {
                    let username = username.trim_start_matches('@');
                    if !username.is_empty() && self.text.len() > start {
                        self.mentions.push(RichMention {
                            username: username.to_owned(),
                            range: start..self.text.len(),
                        });
                    }
                }
                None
            }
            "text_mention" => {
                self.inline(&value["text"]);
                // Only emit entities that the installed decoder can read.
                serde_json::from_value::<teloxide::types::User>(value["user"].clone())
                    .ok()
                    .map(|user| json!({"type": "text_mention", "user": user}))
            }
            "custom_emoji" => {
                self.inline(&value["alternative_text"]);
                None
            }
            "mathematical_expression" => {
                self.inline(&value["expression"]);
                None
            }
            _ => {
                self.inline(&value["text"]);
                None
            }
        };
        if let Some(mut entity) = entity.filter(|_| self.utf16_len > start) {
            entity["offset"] = json!(start);
            entity["length"] = json!(self.utf16_len - start);
            self.entities.push(entity);
        }
    }
}

fn link_entity(value: &Value) -> Option<Value> {
    let url = url::Url::parse(value.as_str()?).ok()?;
    Some(json!({"type": "text_link", "url": url}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use teloxide::types::{MessageEntity, MessageEntityRef};

    #[test]
    fn projection_counts_utf16_offsets_and_keeps_newlines_inside_entities() {
        let mut message = json!({"rich_message": {"blocks": [
            {"type": "paragraph", "text": "🧭"},
            {"type": "table", "cells": [[
                {"text": "a"},
                {"text": {"type": "url", "text": "界🚀\n", "url": "https://example.org/"}}
            ]]}
        ]}});

        project_message(&mut message).unwrap();

        let text = message["text"].as_str().unwrap();
        assert_eq!(text, "🧭\na | 界🚀\n");
        let entities: Vec<MessageEntity> =
            serde_json::from_value(message["entities"].clone()).unwrap();
        assert_eq!(entities[0].offset, 7);
        assert_eq!(entities[0].length, 4);
        assert_eq!(MessageEntityRef::parse(text, &entities)[0].text(), "界🚀\n");
    }

    #[test]
    fn projection_removes_only_the_generated_final_separator() {
        let mut message = json!({"rich_message": {"blocks": [
            {"type": "paragraph", "text": "first"},
            {"type": "paragraph", "text": "second"},
            {"type": "paragraph", "text": ""}
        ]}});

        project_message(&mut message).unwrap();

        assert_eq!(message["text"], "first\nsecond");
    }

    #[test]
    fn projection_keeps_attachment_mime_separate_from_the_block_fallback() {
        let mut message = json!({"rich_message": {"blocks": [
            {"type": "audio", "audio": {"file_id": "explicit", "mime_type": "audio/flac"}},
            {"type": "audio", "audio": {"file_id": "unspecified"}},
            {"type": "voice_note", "voice_note": {"file_id": "voice", "mime_type": ""}}
        ]}});

        let projection = project_message(&mut message).unwrap();

        assert_eq!(projection.media[0].mime_type.as_deref(), Some("audio/flac"));
        assert_eq!(projection.media[0].fallback_mime_type, Some("audio/mpeg"));
        assert_eq!(projection.media[1].mime_type, None);
        assert_eq!(projection.media[1].fallback_mime_type, Some("audio/mpeg"));
        assert_eq!(projection.media[2].mime_type, None);
        assert_eq!(projection.media[2].fallback_mime_type, Some("audio/ogg"));
    }
}
