use std::error::Error;
use std::path::PathBuf;

use anyhow::anyhow;
use dotenvy::dotenv;
use teloxide::dispatching::UpdateFilterExt;
use teloxide::prelude::*;
use teloxide::utils::command::BotCommands;
use tracing::{error, info};

mod config;
mod db;
mod handlers;
mod importer;
mod llm;
mod rag;
mod state;
mod tools;
mod utils;

use config::CONFIG;
use db::database::Database;
use handlers::qa::MODEL_CALLBACK_PREFIX;
use handlers::{commands, qa};
use importer::history::{parse_filter_datetime, run_import_history, ImportHistoryArgs};
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
    Ragq(String),
    Img(String),
    Image(String),
    Vid(String),
    Profileme(String),
    Paintme,
    Portraitme,
    Status,
    Diagnose,
    Support,
}

type HandlerResult = Result<(), Box<dyn Error + Send + Sync>>;

fn import_history_usage() -> &'static str {
    "Usage: cargo run -- import-history --file <path> --chat-id <id> [--batch-size <n>] [--dry-run] [--resume|--no-resume] [--from-date <YYYY-MM-DD|ISO8601>] [--to-date <YYYY-MM-DD|ISO8601>]"
}

fn parse_import_history_args(args: &[String]) -> anyhow::Result<Option<ImportHistoryArgs>> {
    if args.get(1).map(|value| value.as_str()) != Some("import-history") {
        return Ok(None);
    }

    let mut file_path: Option<PathBuf> = None;
    let mut chat_id: Option<i64> = None;
    let mut batch_size = CONFIG.rag_ingest_batch_size.max(1);
    let mut dry_run = false;
    let mut resume = true;
    let mut from_date = None;
    let mut to_date = None;

    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--file" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("Missing value for --file"))?;
                file_path = Some(PathBuf::from(value));
            }
            "--chat-id" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("Missing value for --chat-id"))?;
                chat_id = Some(
                    value
                        .parse::<i64>()
                        .map_err(|_| anyhow!("Invalid --chat-id value: {value}"))?,
                );
            }
            "--batch-size" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("Missing value for --batch-size"))?;
                batch_size = value
                    .parse::<usize>()
                    .map_err(|_| anyhow!("Invalid --batch-size value: {value}"))?
                    .max(1);
            }
            "--dry-run" => {
                dry_run = true;
            }
            "--resume" => {
                resume = true;
            }
            "--no-resume" => {
                resume = false;
            }
            "--from-date" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("Missing value for --from-date"))?;
                from_date = Some(parse_filter_datetime(value, false)?);
            }
            "--to-date" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow!("Missing value for --to-date"))?;
                to_date = Some(parse_filter_datetime(value, true)?);
            }
            "--help" | "-h" => {
                return Err(anyhow!(import_history_usage()));
            }
            other => {
                return Err(anyhow!(
                    "Unknown import-history argument: {other}\n{}",
                    import_history_usage()
                ));
            }
        }
        index += 1;
    }

    let file_path = file_path.ok_or_else(|| anyhow!("--file is required"))?;
    let chat_id = chat_id.ok_or_else(|| anyhow!("--chat-id is required"))?;

    Ok(Some(ImportHistoryArgs {
        file_path,
        chat_id,
        batch_size,
        dry_run,
        resume,
        from_date,
        to_date,
    }))
}

#[tokio::main]
async fn main() -> HandlerResult {
    dotenv().ok();
    let _guards = init_logging();

    let args: Vec<String> = std::env::args().collect();
    if let Some(import_args) = parse_import_history_args(&args)? {
        let summary = run_import_history(import_args).await?;
        info!(
            "Import summary: total={} upserts={} invalid={} resume_skips={} date_skips={} rag_candidates={} rag_accepted={} rag_skipped={} rag_failed={}",
            summary.total_records,
            summary.db_upserts,
            summary.invalid_records,
            summary.skipped_by_resume,
            summary.skipped_by_date_filter,
            summary.rag_candidates,
            summary.rag_accepted,
            summary.rag_skipped,
            summary.rag_failed
        );
        return Ok(());
    }

    if CONFIG.bot_token.trim().is_empty() {
        return Err("BOT_TOKEN is required unless running import-history".into());
    }

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
        .branch(
            dptree::filter(|msg: Message| msg.media_group_id().is_some())
                .endpoint(handle_media_group),
        )
        .branch(
            dptree::filter(|msg: Message| msg.text().is_some() || msg.caption().is_some())
                .endpoint(handle_log_message),
        )
        .endpoint(ignore_message);

    let callback_state = state.clone();
    let callback_handler =
        Update::filter_callback_query().endpoint(move |bot: Bot, query: CallbackQuery| {
            let state = callback_state.clone();
            async move { handle_callback_query(bot, state, query).await }
        });

    let handler = dptree::entry()
        .branch(message_handler)
        .branch(callback_handler);

    Dispatcher::builder(bot, handler)
        .dependencies(dptree::deps![state])
        .enable_ctrlc_handler()
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
        Command::Ragq(arg) => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            let arg = optional_arg(arg);
            tokio::spawn(async move {
                if let Err(err) = qa::ragq_handler(bot, state, message, arg).await {
                    error!("ragq handler failed: {err}");
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
        Command::Status => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            tokio::spawn(async move {
                if let Err(err) = commands::status_handler(bot, state, message).await {
                    error!("status handler failed: {err}");
                }
            });
        }
        Command::Diagnose => {
            let bot = bot.clone();
            let state = state.clone();
            let message = message.clone();
            tokio::spawn(async move {
                if let Err(err) = commands::diagnose_handler(bot, state, message).await {
                    error!("diagnose handler failed: {err}");
                }
            });
        }
        Command::Support => commands::support_handler(bot, message).await?,
    }
    Ok(())
}

async fn handle_callback_query(bot: Bot, state: AppState, query: CallbackQuery) -> HandlerResult {
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

async fn ignore_message(_message: Message) -> HandlerResult {
    Ok(())
}
