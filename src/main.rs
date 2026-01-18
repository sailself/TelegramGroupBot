use std::error::Error;

use dotenvy::dotenv;
use teloxide::dispatching::UpdateFilterExt;
use teloxide::prelude::*;
use teloxide::utils::command::BotCommands;
use tracing::{error, info};

mod config;
mod db;
mod handlers;
mod llm;
mod state;
mod tools;
mod utils;

use config::CONFIG;
use db::database::Database;
use handlers::qa::MODEL_CALLBACK_PREFIX;
use handlers::{
    commands, qa,
};
use state::AppState;
use utils::logging::init_logging;

#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase")]
enum Command {
    Start,
    Help,
    Tldr(String),
    Factcheck(String),
    Q(String),
    Qq(String),
    Img(String),
    Image(String),
    Vid(String),
    Profileme(String),
    Paintme,
    Portraitme,
    Support,
}

type HandlerResult = Result<(), Box<dyn Error + Send + Sync>>;

#[tokio::main]
async fn main() -> HandlerResult {
    dotenv().ok();
    let _guards = init_logging();

    let bot = Bot::new(CONFIG.bot_token.clone());
    info!("Starting TelegramGroupHelperBot (Rust)");

    let db = Database::init(&CONFIG.database_url).await?;
    let state = AppState::new(db);

    handlers::access::load_whitelist();

    let command_handler = dptree::entry()
        .filter_command::<Command>()
        .endpoint(handle_command);

    let message_handler = Update::filter_message()
        .branch(command_handler)
        .branch(dptree::filter(|msg: Message| msg.media_group_id().is_some())
            .endpoint(handle_media_group))
        .branch(dptree::filter(|msg: Message| msg.text().is_some() || msg.caption().is_some())
            .endpoint(handle_log_message));

    let callback_state = state.clone();
    let callback_handler = Update::filter_callback_query().endpoint(
        move |bot: Bot, query: CallbackQuery| {
            let state = callback_state.clone();
            async move { handle_callback_query(bot, state, query).await }
        },
    );

    let handler = dptree::entry()
        .branch(message_handler)
        .branch(callback_handler);

    Dispatcher::builder(bot, handler)
        .dependencies(dptree::deps![state])
        .build()
        .dispatch()
        .await;

    Ok(())
}

async fn handle_command(
    bot: Bot,
    state: AppState,
    message: Message,
    command: Command,
) -> HandlerResult {
    fn optional_arg(arg: String) -> Option<String> {
        if arg.trim().is_empty() {
            None
        } else {
            Some(arg)
        }
    }

    match command {
        Command::Start => commands::start_handler(bot, message).await?,
        Command::Help => commands::help_handler(bot, message).await?,
        Command::Tldr(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = commands::tldr_handler(bot, state, message, arg).await {
                    error!("tldr handler failed: {err}");
                }
            });
        }
        Command::Factcheck(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = commands::factcheck_handler(bot, state, message, arg).await {
                    error!("factcheck handler failed: {err}");
                }
            });
        }
        Command::Q(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = qa::q_handler(bot, state, message, arg, false, "q").await {
                    error!("q handler failed: {err}");
                }
            });
        }
        Command::Qq(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = qa::qq_handler(bot, state, message, arg).await {
                    error!("qq handler failed: {err}");
                }
            });
        }
        Command::Img(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = commands::img_handler(bot, state, message, arg).await {
                    error!("img handler failed: {err}");
                }
            });
        }
        Command::Image(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = commands::image_handler(bot, state, message, arg).await {
                    error!("image handler failed: {err}");
                }
            });
        }
        Command::Vid(arg) => {
            let bot = bot.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = commands::vid_handler(bot, message, arg).await {
                    error!("vid handler failed: {err}");
                }
            });
        }
        Command::Profileme(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = commands::profileme_handler(bot, state, message, arg).await {
                    error!("profileme handler failed: {err}");
                }
            });
        }
        Command::Paintme => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            tokio::spawn(async move {
                if let Err(err) = commands::paintme_handler(bot, state, message, false).await {
                    error!("paintme handler failed: {err}");
                }
            });
        }
        Command::Portraitme => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            tokio::spawn(async move {
                if let Err(err) = commands::paintme_handler(bot, state, message, true).await {
                    error!("portraitme handler failed: {err}");
                }
            });
        }
        Command::Support => commands::support_handler(bot, message).await?,
    }
    Ok(())
}

async fn handle_callback_query(
    bot: Bot,
    state: AppState,
    query: CallbackQuery,
) -> HandlerResult {
    let Some(data) = query.data.clone() else {
        return Ok(());
    };
    if data.starts_with(MODEL_CALLBACK_PREFIX) {
        let bot = bot.clone();
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(err) = qa::model_selection_callback(bot, state, query).await {
                error!("model selection callback failed: {err}");
            }
        });
        return Ok(());
    }
    if data.starts_with("image_res:") || data.starts_with("image_aspect:") {
        let bot = bot.clone();
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(err) = commands::image_selection_callback(bot, state, query).await {
                error!("image selection callback failed: {err}");
            }
        });
    }
    Ok(())
}

async fn handle_media_group(state: AppState, message: Message) -> HandlerResult {
    commands::handle_media_group(state, message).await;
    Ok(())
}

async fn handle_log_message(state: AppState, message: Message) -> HandlerResult {
    if let Some(text) = message.text().or_else(|| message.caption()) {
        if text.trim_start().starts_with('/') {
            return Ok(());
        }
    }
    handlers::responses::log_message(&state, &message).await;
    Ok(())
}
