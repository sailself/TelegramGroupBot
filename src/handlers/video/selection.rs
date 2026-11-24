//! One editable settings menu for /vid2.

use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use teloxide::prelude::*;
use teloxide::requests::{JsonRequest, Payload};
use teloxide::types::{
    CallbackQueryId, InlineKeyboardButton, InlineKeyboardMarkup, MessageId, ReplyParameters,
};
use tokio::sync::mpsc;
use tokio::time::{timeout, timeout_at, Instant};
use tracing::debug;

use crate::llm::vid2_video::{Vid2LoraStrength, Vid2Mode, Vid2Options, Vid2Quality};
use crate::state::AppState;
use crate::utils::telegram::retry_telegram;

pub(crate) const VID2_CALLBACK_PREFIX: &str = "vid2:";
const MENU_API_TIMEOUT: Duration = Duration::from_secs(4);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Choice {
    FirstFrame,
    Ref2Va,
    Nsfw,
    NoLora,
    Strength(Vid2LoraStrength),
    High,
    Fast,
    Seconds(u8),
    Generate,
}

#[derive(Clone, Copy)]
enum Row {
    Mode,
    Lora,
    Strength,
    Quality,
    Duration,
    Generate,
}

impl Row {
    const ALL: [Self; 6] = [
        Self::Mode,
        Self::Lora,
        Self::Strength,
        Self::Quality,
        Self::Duration,
        Self::Generate,
    ];

    fn buttons(self) -> &'static [(Choice, &'static str, &'static str)] {
        use Choice::*;
        match self {
            Self::Mode => &[
                (FirstFrame, "frame", "1st Frame"),
                (Ref2Va, "ref", "Ref2Va"),
            ],
            Self::Lora => &[(Nsfw, "nsfw", "NSFW"), (NoLora, "none", "None")],
            Self::Strength => &[
                (Strength(Vid2LoraStrength::Half), "s05", "0.5×"),
                (Strength(Vid2LoraStrength::One), "s10", "1.0×"),
                (Strength(Vid2LoraStrength::OneAndHalf), "s15", "1.5×"),
                (Strength(Vid2LoraStrength::Two), "s20", "2.0×"),
            ],
            Self::Quality => &[(High, "high", "High"), (Fast, "fast", "Fast")],
            Self::Duration => &[
                (Seconds(5), "5", "5 s"),
                (Seconds(10), "10", "10 s"),
                (Seconds(15), "15", "15 s"),
            ],
            Self::Generate => &[(Generate, "go", "Generate")],
        }
    }

    fn visible(self, image_count: usize, options: Vid2Options) -> bool {
        match self {
            Self::Mode => image_count == 1,
            Self::Strength => options.lora.is_some(),
            _ => true,
        }
    }
}

impl Choice {
    fn selected(self, options: Vid2Options) -> bool {
        match self {
            Self::FirstFrame => options.mode == Vid2Mode::FirstFrame,
            Self::Ref2Va => options.mode == Vid2Mode::Ref2Va,
            Self::Nsfw => options.lora.is_some(),
            Self::NoLora => options.lora.is_none(),
            Self::Strength(strength) => options.lora == Some(strength),
            Self::High => options.quality == Vid2Quality::High,
            Self::Fast => options.quality == Vid2Quality::Fast,
            Self::Seconds(seconds) => options.seconds == seconds,
            Self::Generate => false,
        }
    }

    fn apply(self, options: &mut Vid2Options, image_count: usize) {
        match self {
            Self::FirstFrame if image_count == 1 => options.mode = Vid2Mode::FirstFrame,
            Self::Ref2Va if image_count == 1 => options.mode = Vid2Mode::Ref2Va,
            Self::Nsfw => {
                options.lora.get_or_insert_with(Vid2LoraStrength::default);
            }
            Self::NoLora => options.lora = None,
            Self::Strength(strength) if options.lora.is_some() => options.lora = Some(strength),
            Self::High => options.quality = Vid2Quality::High,
            Self::Fast => options.quality = Vid2Quality::Fast,
            Self::Seconds(seconds) => options.seconds = seconds,
            _ => {}
        }
    }
}

fn initial_options(image_count: usize) -> Vid2Options {
    Vid2Options {
        mode: if image_count > 1 {
            Vid2Mode::Ref2Va
        } else {
            Vid2Mode::FirstFrame
        },
        ..Vid2Options::default()
    }
}

fn keyboard(key: &str, image_count: usize, options: Vid2Options) -> InlineKeyboardMarkup {
    InlineKeyboardMarkup::new(
        Row::ALL
            .into_iter()
            .filter(|row| row.visible(image_count, options))
            .map(|row| {
                row.buttons().iter().map(|&(choice, code, label)| {
                    let text = if choice.selected(options) {
                        format!("✓ {label}")
                    } else {
                        label.to_string()
                    };
                    InlineKeyboardButton::callback(
                        text,
                        format!("{VID2_CALLBACK_PREFIX}{key}:{code}"),
                    )
                })
            }),
    )
}

// Teloxide 0.17 predates Telegram's ephemeral message methods and fields.
// These requests still use the configured Bot API client and error handling.
#[derive(Serialize)]
#[serde(transparent)]
struct SendEphemeralMenu(Value);

impl Payload for SendEphemeralMenu {
    type Output = SentEphemeralMenu;
    const NAME: &'static str = "sendMessage";
}

#[derive(Deserialize)]
struct SentEphemeralMenu {
    ephemeral_message_id: Option<i64>,
    message_id: Option<i32>,
}

#[derive(Serialize)]
struct EditEphemeralMenu {
    #[serde(flatten)]
    target: EphemeralTarget,
    reply_markup: InlineKeyboardMarkup,
}

impl Payload for EditEphemeralMenu {
    type Output = bool;
    const NAME: &'static str = "editEphemeralMessageReplyMarkup";
}

#[derive(Clone, Copy, Serialize)]
struct EphemeralTarget {
    chat_id: ChatId,
    receiver_user_id: UserId,
    ephemeral_message_id: i64,
}

impl Payload for EphemeralTarget {
    type Output = bool;
    const NAME: &'static str = "deleteEphemeralMessage";
}

struct MenuSurface {
    chat_id: ChatId,
    owner: UserId,
    ephemeral: Option<EphemeralTarget>,
    regular: Option<MessageId>,
    try_overlay: bool,
}

impl MenuSurface {
    async fn open(
        bot: &Bot,
        command: &Message,
        owner: UserId,
        text: &str,
        markup: InlineKeyboardMarkup,
    ) -> Result<Self> {
        let chat_id = command.chat.id;
        let thread_id = command.thread_id.filter(|_| command.is_topic_message);
        let can_overlay = command.chat.is_group() || command.chat.is_supergroup();
        let mut menu = Self {
            chat_id,
            owner,
            ephemeral: None,
            regular: None,
            try_overlay: can_overlay,
        };
        if can_overlay {
            let mut body = json!({
                "chat_id": chat_id,
                "text": text,
                "reply_parameters": ReplyParameters::new(command.id),
                "reply_markup": markup,
                "ephemeral_message_parameters": {"receiver_user_id": owner}
            });
            if let Some(thread_id) = thread_id {
                body["message_thread_id"] = json!(thread_id);
            }
            // An admin bot can send directly to the requester. A non-admin
            // bot can open the private overlay after the requester's first tap.
            if let Ok(Ok(sent)) = timeout(
                MENU_API_TIMEOUT,
                JsonRequest::new(bot.clone(), SendEphemeralMenu(body)),
            )
            .await
            {
                if let Some(id) = sent.ephemeral_message_id {
                    menu.ephemeral = Some(menu.target(id));
                    menu.try_overlay = false;
                    return Ok(menu);
                }
                // Older servers can ignore the ephemeral field and send a normal message.
                if let Some(id) = sent.message_id.filter(|id| *id > 0) {
                    menu.regular = Some(MessageId(id));
                    return Ok(menu);
                }
            }
        }
        let sent = retry_telegram("vid2 send settings", || {
            let request = bot
                .send_message(chat_id, text)
                .reply_parameters(ReplyParameters::new(command.id))
                .reply_markup(markup.clone());
            match thread_id {
                Some(thread_id) => request.message_thread_id(thread_id),
                None => request,
            }
        })
        .await?;
        menu.regular = Some(sent.id);
        Ok(menu)
    }

    fn target(&self, ephemeral_message_id: i64) -> EphemeralTarget {
        EphemeralTarget {
            chat_id: self.chat_id,
            receiver_user_id: self.owner,
            ephemeral_message_id,
        }
    }

    async fn update(
        &mut self,
        bot: &Bot,
        query_id: CallbackQueryId,
        text: &str,
        markup: InlineKeyboardMarkup,
        changed: bool,
    ) {
        if std::mem::take(&mut self.try_overlay) {
            let body = json!({
                "text": text,
                "reply_markup": markup,
                "ephemeral_message_parameters": {
                    "callback_query_id": query_id,
                    "replace_callback_query_message": true
                }
            });
            if let Ok(Ok(sent)) = timeout(
                MENU_API_TIMEOUT,
                JsonRequest::new(bot.clone(), SendEphemeralMenu(body)),
            )
            .await
            {
                if let Some(id) = sent.ephemeral_message_id {
                    self.ephemeral = Some(self.target(id));
                    return;
                }
            }
        }
        if !changed {
            return;
        }
        if let Some(target) = self.ephemeral {
            let edit = EditEphemeralMenu {
                target,
                reply_markup: markup,
            };
            if !matches!(
                timeout(MENU_API_TIMEOUT, JsonRequest::new(bot.clone(), edit)).await,
                Ok(Ok(true))
            ) {
                debug!("Could not update the ephemeral Vid2 settings menu");
            }
        } else if let Some(id) = self.regular {
            let edit = bot
                .edit_message_reply_markup(self.chat_id, id)
                .reply_markup(markup);
            if !matches!(timeout(MENU_API_TIMEOUT, edit).await, Ok(Ok(_))) {
                debug!("Could not update the Vid2 settings menu");
            }
        }
    }

    async fn close(self, bot: &Bot) {
        let delete_ephemeral = async {
            if let Some(target) = self.ephemeral {
                let _ = timeout(MENU_API_TIMEOUT, JsonRequest::new(bot.clone(), target)).await;
            }
        };
        let delete_regular = async {
            if let Some(id) = self.regular {
                if !matches!(
                    timeout(MENU_API_TIMEOUT, bot.delete_message(self.chat_id, id)).await,
                    Ok(Ok(_))
                ) {
                    let edit = bot
                        .edit_message_text(self.chat_id, id, "Vid2 settings closed.")
                        .reply_markup(InlineKeyboardMarkup::default());
                    let _ = timeout(MENU_API_TIMEOUT, edit).await;
                }
            }
        };
        tokio::join!(delete_ephemeral, delete_regular);
    }
}

struct MenuTap {
    id: CallbackQueryId,
    message_id: MessageId,
    choice: Choice,
}

pub(crate) struct Vid2MenuCallbacks {
    owner: UserId,
    chat_id: ChatId,
    taps: mpsc::Sender<MenuTap>,
}

pub(super) async fn select_options(
    bot: &Bot,
    state: &AppState,
    command: &Message,
    image_count: usize,
    wait: Duration,
) -> Result<Vid2Options> {
    let mut options = initial_options(image_count);
    let Some(owner) = command
        .from
        .as_ref()
        .filter(|user| !user.is_bot)
        .map(|user| user.id)
    else {
        return Ok(options);
    };
    if wait.is_zero() || command.sender_chat.is_some() {
        return Ok(options);
    }
    let mode_note = match image_count {
        0 => " · Text to video",
        1 => "",
        _ => " · Ref2Va",
    };
    let text = format!(
        "Video settings{mode_note}\nTap Generate to start now. Current settings apply after {} s.",
        wait.as_secs()
    );
    let key = format!("{}:{}", command.chat.id.0, command.id.0);
    let markup = keyboard(&key, image_count, options);
    let mut menu = MenuSurface::open(bot, command, owner, &text, markup).await?;
    let (taps, mut events) = mpsc::channel(16);
    let deadline = Instant::now() + wait;
    state.pending_vid2_menus.insert_with_timeout(
        key.clone(),
        Vid2MenuCallbacks {
            owner,
            chat_id: command.chat.id,
            taps,
        },
        wait,
        |_| async {},
    );
    while let Ok(Some(tap)) = timeout_at(deadline, events.recv()).await {
        // An already queued callback must still respect the deadline.
        if Instant::now() >= deadline {
            break;
        }
        if menu.ephemeral.is_none() && menu.regular != Some(tap.message_id) {
            continue;
        }
        if tap.choice == Choice::Generate {
            break;
        }
        let previous = options;
        tap.choice.apply(&mut options, image_count);
        let updated = keyboard(&key, image_count, options);
        let _ = timeout_at(
            deadline,
            menu.update(bot, tap.id, &text, updated, options != previous),
        )
        .await;
    }
    state.pending_vid2_menus.entry(&key).take();
    menu.close(bot).await;
    Ok(options)
}

pub(crate) async fn handle_callback(bot: Bot, state: AppState, query: CallbackQuery) -> Result<()> {
    let answer = {
        let parsed = query
            .data
            .as_deref()
            .and_then(|data| data.strip_prefix(VID2_CALLBACK_PREFIX))
            .and_then(|data| data.rsplit_once(':'))
            .and_then(|(key, code)| {
                Row::ALL
                    .into_iter()
                    .flat_map(Row::buttons)
                    .find(|(_, name, _)| *name == code)
                    .map(|(choice, _, _)| (key, *choice))
            });
        match parsed.zip(query.message.as_ref()) {
            Some(((key, choice), message)) => {
                let entry = state.pending_vid2_menus.entry(key);
                match entry.get() {
                    Some(menu)
                        if menu.owner != query.from.id || menu.chat_id != message.chat().id =>
                    {
                        Some("Only the requester can change these settings.")
                    }
                    Some(menu) => match menu.taps.try_send(MenuTap {
                        id: query.id.clone(),
                        message_id: message.id(),
                        choice,
                    }) {
                        Ok(()) => None,
                        Err(mpsc::error::TrySendError::Full(_)) => Some("Please try again."),
                        Err(mpsc::error::TrySendError::Closed(_)) => {
                            Some("These settings have closed.")
                        }
                    },
                    None => Some("These settings have closed."),
                }
            }
            None => Some("These settings have closed."),
        }
    };
    let request = bot.answer_callback_query(query.id);
    match answer {
        Some(text) => request.text(text).await?,
        None => request.await?,
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::test_support::init_test_db;
    use crate::test_support::{response_with_status, ExpectedRequest, TestServer};

    fn telegram(method: &str, status: u16, body: &[u8]) -> ExpectedRequest {
        ExpectedRequest::new(
            "POST",
            &format!("/bot123:test/{method}"),
            response_with_status(status, body.to_vec()),
        )
    }

    fn command() -> Message {
        serde_json::from_value(json!({
            "message_id": 42, "date": 100, "chat": {"id": -23, "type": "group", "title": "Group"},
            "from": {"id": 10, "is_bot": false, "first_name": "User"}, "text": "/vid2 a fox"
        }))
        .unwrap()
    }

    fn callback(user: u64, message_id: i32, code: &str) -> CallbackQuery {
        serde_json::from_value(json!({
            "id": format!("tap-{user}-{code}"), "chat_instance": "group",
            "from": {"id": user, "is_bot": false, "first_name": "User"},
            "message": {"message_id": message_id, "date": 100, "chat": {"id": -23, "type": "group", "title": "Group"}},
            "data": format!("vid2:-23:42:{code}")
        })).unwrap()
    }

    #[tokio::test]
    async fn direct_ephemeral_menu_times_out_to_multi_image_defaults_and_cleans_up() {
        let server = TestServer::new(vec![
            telegram(
                "sendMessage",
                200,
                br#"{"ok":true,"result":{"message_id":0,"ephemeral_message_id":29}}"#,
            )
            .with_body_fragment(br#""ephemeral_message_parameters":{"receiver_user_id":10}"#)
            .with_body_fragment(br#""callback_data":"vid2:-23:42:go""#),
            telegram(
                "deleteEphemeralMessage",
                200,
                br#"{"ok":true,"result":true}"#,
            )
            .with_body_fragment(br#""ephemeral_message_id":29"#),
        ]);
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let db = init_test_db("vid2-menu-timeout").await;
        let state = AppState::new(db, 1, "test_bot".into());
        let result = select_options(&bot, &state, &command(), 3, Duration::from_millis(40))
            .await
            .unwrap();
        assert_eq!(
            result,
            Vid2Options {
                mode: Vid2Mode::Ref2Va,
                ..Vid2Options::default()
            }
        );
        assert_eq!(state.pending_vid2_menus.count(), 0);
        server.join().unwrap();
    }

    #[tokio::test]
    async fn a_non_admin_bot_can_open_and_edit_the_requesters_private_overlay() {
        let server = TestServer::with_unordered_expectations(
            vec![
                telegram("sendMessage", 400, br#"{"ok":false,"error_code":400,"description":"Bad Request: administrator rights required"}"#)
                    .with_body_fragment(br#""ephemeral_message_parameters""#),
                telegram("SendMessage", 200, br#"{"ok":true,"result":{"message_id":7,"date":100,"chat":{"id":-23,"type":"group","title":"Group"},"text":"settings"}}"#),
                telegram("sendMessage", 200, br#"{"ok":true,"result":{"message_id":0,"ephemeral_message_id":19}}"#)
                    .with_body_fragment(br#""callback_query_id":"tap-10-nsfw""#)
                    .with_body_fragment(br#""replace_callback_query_message":true"#)
                    .with_body_fragment(br#""callback_data":"vid2:-23:42:s10""#),
                telegram("editEphemeralMessageReplyMarkup", 200, br#"{"ok":true,"result":true}"#)
                    .with_body_fragment(br#""ephemeral_message_id":19"#)
                    .with_body_fragment("✓ Ref2Va".as_bytes()),
                telegram("editEphemeralMessageReplyMarkup", 200, br#"{"ok":true,"result":true}"#)
                    .with_body_fragment("✓ 2.0×".as_bytes()),
            ],
            vec![
                telegram("AnswerCallbackQuery", 200, br#"{"ok":true,"result":true}"#).with_body_fragment(b"Only the requester"),
                telegram("AnswerCallbackQuery", 200, br#"{"ok":true,"result":true}"#).with_body_fragment(b"tap-10-nsfw"),
                telegram("AnswerCallbackQuery", 200, br#"{"ok":true,"result":true}"#).with_body_fragment(b"tap-10-ref"),
                telegram("AnswerCallbackQuery", 200, br#"{"ok":true,"result":true}"#).with_body_fragment(b"tap-10-s20"),
                telegram("AnswerCallbackQuery", 200, br#"{"ok":true,"result":true}"#).with_body_fragment(b"tap-10-go"),
                telegram("DeleteMessage", 200, br#"{"ok":true,"result":true}"#).with_body_fragment(br#""message_id":7"#),
                telegram("deleteEphemeralMessage", 200, br#"{"ok":true,"result":true}"#).with_body_fragment(br#""ephemeral_message_id":19"#),
            ],
        );
        let bot = Bot::new("123:test").set_api_url(server.base_url());
        let db = init_test_db("vid2-overlay").await;
        let state = AppState::new(db, 1, "test_bot".into());
        let command = command();
        let (result, ()) = tokio::join!(
            select_options(&bot, &state, &command, 1, Duration::from_secs(3)),
            async {
                timeout(Duration::from_secs(2), async {
                    while state.pending_vid2_menus.count() == 0 {
                        tokio::time::sleep(Duration::from_millis(5)).await;
                    }
                })
                .await
                .unwrap();
                for (user, message_id, code) in [
                    (99, 7, "go"),
                    (10, 7, "nsfw"),
                    (10, 0, "ref"),
                    (10, 0, "s20"),
                    (10, 0, "go"),
                ] {
                    handle_callback(bot.clone(), state.clone(), callback(user, message_id, code))
                        .await
                        .unwrap();
                }
            }
        );
        assert_eq!(
            result.unwrap(),
            Vid2Options {
                mode: Vid2Mode::Ref2Va,
                lora: Some(Vid2LoraStrength::Two),
                ..Vid2Options::default()
            }
        );
        assert_eq!(state.pending_vid2_menus.count(), 0);
        server.join().unwrap();
    }
}
