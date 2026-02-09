use std::time::Duration;

use teloxide::prelude::*;
use teloxide::types::ChatAction;
use tokio::task::JoinHandle;
use tracing::warn;

const CHAT_ACTION_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(4);

pub struct ChatActionHeartbeat {
    task_handle: Option<JoinHandle<()>>,
}

impl Drop for ChatActionHeartbeat {
    fn drop(&mut self) {
        if let Some(handle) = self.task_handle.take() {
            handle.abort();
        }
    }
}

pub fn start_chat_action_heartbeat(
    bot: Bot,
    chat_id: ChatId,
    action: ChatAction,
) -> ChatActionHeartbeat {
    let task_handle = tokio::spawn(async move {
        loop {
            if let Err(err) = bot.send_chat_action(chat_id, action.clone()).await {
                warn!("send_chat_action failed: {err}");
            }
            tokio::time::sleep(CHAT_ACTION_HEARTBEAT_INTERVAL).await;
        }
    });

    ChatActionHeartbeat {
        task_handle: Some(task_handle),
    }
}
