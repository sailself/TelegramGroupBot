//! Help, support, and start command handlers.

use anyhow::Result;
use teloxide::prelude::*;
use teloxide::types::{InlineKeyboardButton, InlineKeyboardMarkup, ParseMode, ReplyParameters};

use crate::config::CONFIG;
use crate::handlers::access::check_access_control;
use crate::utils::text::escape_html;

const HELP_PARSE_MODE: Option<ParseMode> = None;

fn filter_gemini_help_text(help_text: &str, gemini_available: bool) -> String {
    let mut text = help_text.to_string();
    if gemini_available {
        return text;
    }

    for command in ["vid", "mysong"] {
        let marker = format!("\n/{command} -");
        let Some(start) = text.find(&marker) else {
            continue;
        };
        let after_start = start + marker.len();
        let end = text[after_start..]
            .find("\n\n")
            .map(|offset| after_start + offset + 2)
            .unwrap_or(text.len());
        text.replace_range(start..end, "\n");
    }

    text
}

fn command_help_text() -> &'static str {
    r#"
TelegramGroupHelperBot 指令说明

/tldr - 汇总最近 N 条消息
用法：回复一条消息后发送 `/tldr`，会汇总从那条消息到现在的聊天内容。
也可以直接使用 `/tldr 50` 指定汇总最近 50 条消息。

/factcheck - 对文字、图片、视频、音频消息做事实核查
用法：`/factcheck [要核查的内容]`
或回复一条消息后发送 `/factcheck`

/q - 提问或分析媒体内容
用法：`/q [你的问题]`

/qc - 询问本群历史内容
用法：`/qc [你的问题]`

/qq - Quick Question（快问快答）
用法：`/qq [你的问题]`
跳过模型选择并优先简短回答；只在确有时效性需要时联网搜索一次。需要深入核实时请使用 `/q`。

/burn_baby_burn - 查看你在当前聊天里烧掉了多少 tokens
用法：`/burn_baby_burn`

/token_devourers - 查看本群最能吃 token 的排行榜
用法：`/token_devourers [1-20]`

/s - 搜索本群相关消息并返回直达链接
用法：`/s [搜索关键词]`

/img - 用 Gemini 或 Codex gpt-image-2 生成或编辑图片；Codex 会自动决定尺寸
用法：`/img [描述]` 用于生成新图片
或回复一张图片后发送 `/img [描述]` 来编辑图片

/image - 与 /img 相同；Gemini 可选分辨率和长宽比，Codex 可选图片尺寸
用法：`/image [描述]`，然后选择模型和生成尺寸；Gemini 长宽比可选择 Auto

/vid - 用 Veo 生成视频
用法：`/vid [文本提示词]`

/profileme - 基于你在本群的聊天记录生成个人简介
用法：`/profileme`
或：`/profileme [简介风格说明]`

/mysong - 基于你在本群的聊天记录生成你的主题歌
用法：`/mysong`
或：`/mysong [风格、语言或额外要求]`

/paintme - 基于你在本群的聊天记录生成艺术形象
用法：`/paintme`

/portraitme - 基于你在本群的聊天记录生成肖像
用法：`/portraitme`

/support - 查看投喂信息
用法：`/support`

/help - 查看这份帮助说明

"#
}

#[allow(deprecated)]
pub async fn help_handler(bot: Bot, message: Message) -> Result<()> {
    if !check_access_control(&bot, &message, "help").await {
        return Ok(());
    }

    let help_text = command_help_text();
    let help_text = filter_gemini_help_text(help_text, CONFIG.gemini_api_available());

    let request = bot
        .send_message(message.chat.id, help_text)
        .reply_parameters(ReplyParameters::new(message.id));

    if let Some(parse_mode) = HELP_PARSE_MODE {
        request.parse_mode(parse_mode).await?;
    } else {
        request.await?;
    }

    Ok(())
}

pub async fn support_handler(bot: Bot, message: Message) -> Result<()> {
    if !check_access_control(&bot, &message, "support").await {
        return Ok(());
    }

    let support_message = escape_html(&CONFIG.support_message);

    let support_url = match reqwest::Url::parse(CONFIG.support_link.trim()) {
        Ok(url) => url,
        Err(_) => {
            bot.send_message(message.chat.id, support_message)
                .reply_parameters(ReplyParameters::new(message.id))
                .parse_mode(ParseMode::Html)
                .await?;
            return Ok(());
        }
    };

    let keyboard = InlineKeyboardMarkup::new(vec![vec![InlineKeyboardButton::url(
        "Support the bot",
        support_url,
    )]]);

    bot.send_message(message.chat.id, support_message)
        .reply_parameters(ReplyParameters::new(message.id))
        .reply_markup(keyboard)
        .parse_mode(ParseMode::Html)
        .await?;
    Ok(())
}

pub async fn start_handler(bot: Bot, message: Message) -> Result<()> {
    bot.send_message(
        message.chat.id,
        "Hello! I am TelegramGroupHelperBot. Use /help to see commands.",
    )
    .reply_parameters(ReplyParameters::new(message.id))
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_text_keeps_search_when_gemini_is_disabled() {
        let raw =
            "\n/s - search\nusage\n\n/vid - video\nusage\n\n/mysong - song\nusage\n\n/q - ask\n";
        let filtered = filter_gemini_help_text(raw, false);

        assert!(filtered.contains("/s -"));
        assert!(!filtered.contains("/vid -"));
        assert!(!filtered.contains("/mysong -"));
        assert!(filtered.contains("/q -"));
    }

    #[test]
    fn help_text_is_not_sent_with_markdown_parse_mode() {
        assert!(HELP_PARSE_MODE.is_none());
    }

    #[test]
    fn help_text_keeps_img2_hidden() {
        assert!(!command_help_text().contains("/img2"));
    }
}
