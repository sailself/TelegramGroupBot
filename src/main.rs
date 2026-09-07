use std::error::Error;
use std::fmt::Display;
use std::future::Future;

use dotenvy::dotenv;
use teloxide::dispatching::UpdateFilterExt;
use teloxide::prelude::*;
use teloxide::types::BotCommand;
use teloxide::utils::command::BotCommands;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

mod agents;
mod config;
mod db;
mod handlers;
mod llm;
mod state;
mod tools;
mod utils;

use config::CONFIG;
use db::database::Database;
use handlers::codex_admin::{
    CODEX_MODEL_PAGE_CALLBACK_PREFIX, CODEX_MODEL_SELECT_CALLBACK_PREFIX,
    CODEX_REASONING_SELECT_CALLBACK_PREFIX,
};
use handlers::commands::{
    IMAGE_ASPECT_RATIO_CALLBACK_PREFIX, IMAGE_CODEX_SIZE_CALLBACK_PREFIX,
    IMAGE_MODEL_CALLBACK_PREFIX, IMAGE_RESOLUTION_CALLBACK_PREFIX,
};
use handlers::qa::MODEL_CALLBACK_PREFIX;
use handlers::{commands, media, qa};
use state::AppState;
use utils::logging::init_logging;

#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase")]
enum Command {
    #[command(description = "介绍AI小喵和可用指令")]
    Start,
    #[command(description = "查看帮助与指令说明")]
    Help,
    #[command(description = "汇总最近 N 条消息（默认 100 条，可用 /tldr 50 指定数量）")]
    Tldr(String),
    #[command(
        description = "回复一条文字/图片/视频/音频消息进行事实核查，支持消息内的 Telegraph/Twitter/YouTube 链接"
    )]
    Factcheck(String),
    #[command(
        description = "提问或分析媒体，弹出模型选择（默认 Gemini，自动隐藏不支持当前媒体的模型）"
    )]
    Q(String),
    #[command(description = "询问本群聊里的历史内容，可检索当前聊天记录并在需要时联网搜索")]
    Qc(String),
    #[command(
        description = "Quick Question（快问快答），跳过模型选择并用简短回答，必要时最多联网搜索一次"
    )]
    Qq(String),
    #[command(
        rename = "burn_baby_burn",
        description = "show how many tokens you have used in this chat"
    )]
    BurnBabyBurn,
    #[command(
        rename = "token_devourers",
        description = "rank the top token consumers in this group"
    )]
    TokenDevourers(String),
    #[command(description = "搜索本群聊相关消息，返回命中的消息摘要和直达链接")]
    S(String),
    #[command(description = "用 Gemini 生成/编辑图片，可直接描述或回复图片/贴纸")]
    Img(String),
    #[command(description = "hidden image generation command")]
    Img2(String),
    #[command(description = "与 /img 相同，但附带分辨率与长宽比按钮")]
    Image(String),
    #[command(description = "用 Veo 生成视频")]
    Vid(String),
    #[command(description = "基于你在本群的聊天记录生成你的主题歌")]
    Mysong(String),
    #[command(description = "基于你在本群的聊天记录生成个人简介")]
    Profileme(String),
    #[command(description = "基于你在本群的聊天记录生成艺术形象")]
    Paintme,
    #[command(description = "基于你在本群的聊天记录生成肖像")]
    Portraitme,
    #[command(description = "查看机器人状态（管理员）")]
    Status,
    #[command(description = "查看诊断信息（管理员）")]
    Diagnose,
    #[command(
        rename = "token_stats",
        description = "show bot-wide token statistics (admin)"
    )]
    TokenStats(String),
    #[command(description = "登录 ChatGPT Codex（管理员）")]
    Codexlogin,
    #[command(description = "退出 ChatGPT Codex（管理员）")]
    Codexlogout,
    #[command(description = "选择当前 Codex 模型（管理员）")]
    Codexmodel,
    #[command(description = "set Codex reasoning level (admin)")]
    Codexreasoning,
    #[command(description = "show Codex usage and rate limits (admin)")]
    Codexusage,
    #[command(description = "投喂AI小喵")]
    Support,
}

type HandlerResult = Result<(), Box<dyn Error + Send + Sync>>;

/// Commands that parse and appear in `/help` but are never published to
/// Telegram's default-scope menu: hidden aliases and admin-only tooling.
const UNPUBLISHED_BOT_COMMANDS: [&str; 9] = [
    "img2",
    "status",
    "diagnose",
    "token_stats",
    "codexlogin",
    "codexlogout",
    "codexmodel",
    "codexreasoning",
    "codexusage",
];

/// Commands that only work with a Gemini API key.
const GEMINI_ONLY_BOT_COMMANDS: [&str; 2] = ["vid", "mysong"];

/// The command menu published to Telegram, derived from the `Command` enum so
/// the published descriptions can never drift from `/help`.
fn published_bot_commands(gemini_available: bool) -> Vec<BotCommand> {
    Command::bot_commands()
        .into_iter()
        // teloxide emits `/name`; Telegram's setMyCommands takes bare names.
        .map(|command| {
            BotCommand::new(command.command.trim_start_matches('/'), command.description)
        })
        .filter(|command| !UNPUBLISHED_BOT_COMMANDS.contains(&command.command.as_str()))
        .filter(|command| {
            gemini_available || !GEMINI_ONLY_BOT_COMMANDS.contains(&command.command.as_str())
        })
        .collect()
}

async fn publish_bot_commands(bot: &Bot, commands: Vec<BotCommand>) -> anyhow::Result<()> {
    let expected = commands.len();
    bot.set_my_commands(commands).await?;

    let published = bot.get_my_commands().await?;
    if published.len() != expected {
        warn!(
            "Telegram reported {} published default-scope command(s) after setMyCommands; expected {}",
            published.len(),
            expected
        );
    } else {
        info!(
            "Telegram now reports {} published default-scope command(s)",
            published.len()
        );
    }
    Ok(())
}

/// Run a handler on its own task so the dispatcher never blocks on heavy
/// work. The handler's error is logged under `name` rather than propagated.
fn spawn_logged<F, E>(name: &'static str, handler: F) -> JoinHandle<()>
where
    F: Future<Output = Result<(), E>> + Send + 'static,
    E: Display,
{
    tokio::spawn(async move {
        if let Err(err) = handler.await {
            error!("{name} handler failed: {err}");
        }
    })
}

fn is_image_selection_callback(data: &str) -> bool {
    [
        IMAGE_MODEL_CALLBACK_PREFIX,
        IMAGE_CODEX_SIZE_CALLBACK_PREFIX,
        IMAGE_RESOLUTION_CALLBACK_PREFIX,
        IMAGE_ASPECT_RATIO_CALLBACK_PREFIX,
    ]
    .iter()
    .any(|prefix| data.starts_with(prefix))
}

#[tokio::main]
async fn main() -> HandlerResult {
    dotenv().ok();
    let _guards = init_logging();

    let bot = Bot::new(CONFIG.bot_token.clone());
    let me = bot.get_me().await?;
    let bot_user_id = i64::try_from(me.id.0).unwrap_or_default();
    let bot_username_lower = me
        .username
        .as_ref()
        .map(|username| username.to_lowercase())
        .unwrap_or_default();
    info!("Starting TelegramGroupHelperBot (Rust)");

    let db = Database::init(&CONFIG.database_url).await?;
    let state = AppState::new(db.clone(), bot_user_id, bot_username_lower);

    handlers::access::load_whitelist();
    if CONFIG.publish_bot_commands {
        let commands = published_bot_commands(CONFIG.gemini_api_available());
        info!(
            "Publishing {} bot commands to Telegram because PUBLISH_BOT_COMMANDS=true; \
             this replaces the default-scope command list managed by BotFather",
            commands.len()
        );
        if let Err(err) = publish_bot_commands(&bot, commands).await {
            warn!("Failed to publish bot command descriptions: {err:#}");
        }
    } else {
        info!(
            "Skipping Telegram command publishing; leave PUBLISH_BOT_COMMANDS=false \
             when BotFather manages the command list"
        );
    }

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
                .endpoint(handle_text_message),
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

    // The dispatcher has stopped taking updates; flush what handlers already
    // queued before the process exits.
    info!("Dispatcher stopped; flushing queued database writes");
    db.shutdown().await;

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

    // Light commands answer inline; everything else runs on its own task so
    // the dispatcher keeps draining updates while the LLM works.
    match command {
        Command::Start => commands::start_handler(bot, message).await?,
        Command::Help => commands::help_handler(bot, message).await?,
        Command::Support => commands::support_handler(bot, message).await?,
        Command::Tldr(arg) => {
            spawn_logged(
                "tldr",
                commands::tldr_handler(bot, state, message, optional_arg(arg)),
            );
        }
        Command::Factcheck(arg) => {
            spawn_logged(
                "factcheck",
                commands::factcheck_handler(bot, state, message, optional_arg(arg)),
            );
        }
        Command::Q(arg) => {
            spawn_logged(
                "q",
                qa::q_handler(bot, state, message, optional_arg(arg), "q"),
            );
        }
        Command::Qc(arg) => {
            spawn_logged("qc", qa::qc_handler(bot, state, message, optional_arg(arg)));
        }
        Command::Qq(arg) => {
            spawn_logged("qq", qa::qq_handler(bot, state, message, optional_arg(arg)));
        }
        Command::BurnBabyBurn => {
            spawn_logged(
                "burn_baby_burn",
                commands::burn_baby_burn_handler(bot, state, message),
            );
        }
        Command::TokenDevourers(arg) => {
            spawn_logged(
                "token_devourers",
                commands::token_devourers_handler(bot, state, message, optional_arg(arg)),
            );
        }
        Command::S(arg) => {
            spawn_logged("s", qa::s_handler(bot, state, message, optional_arg(arg)));
        }
        Command::Img(arg) => {
            spawn_logged(
                "img",
                commands::img_handler(bot, state, message, optional_arg(arg)),
            );
        }
        Command::Img2(arg) => {
            spawn_logged(
                "img2",
                commands::img2_handler(bot, state, message, optional_arg(arg)),
            );
        }
        Command::Image(arg) => {
            spawn_logged(
                "image",
                commands::image_handler(bot, state, message, optional_arg(arg)),
            );
        }
        Command::Vid(arg) => {
            spawn_logged(
                "vid",
                commands::vid_handler(bot, state, message, optional_arg(arg)),
            );
        }
        Command::Mysong(arg) => {
            spawn_logged(
                "mysong",
                commands::mysong_handler(bot, state, message, optional_arg(arg)),
            );
        }
        Command::Profileme(arg) => {
            spawn_logged(
                "profileme",
                commands::profileme_handler(bot, state, message, optional_arg(arg)),
            );
        }
        Command::Paintme => {
            spawn_logged(
                "paintme",
                commands::paintme_handler(bot, state, message, false),
            );
        }
        Command::Portraitme => {
            spawn_logged(
                "portraitme",
                commands::paintme_handler(bot, state, message, true),
            );
        }
        Command::Status => {
            spawn_logged("status", commands::status_handler(bot, state, message));
        }
        Command::Diagnose => {
            spawn_logged("diagnose", commands::diagnose_handler(bot, state, message));
        }
        Command::TokenStats(arg) => {
            spawn_logged(
                "token_stats",
                commands::token_stats_handler(bot, state, message, optional_arg(arg)),
            );
        }
        Command::Codexlogin => {
            spawn_logged(
                "codexlogin",
                handlers::codex_admin::codex_login_handler(bot, state, message),
            );
        }
        Command::Codexlogout => {
            spawn_logged(
                "codexlogout",
                handlers::codex_admin::codex_logout_handler(bot, state, message),
            );
        }
        Command::Codexmodel => {
            spawn_logged(
                "codexmodel",
                handlers::codex_admin::codex_model_handler(bot, state, message),
            );
        }
        Command::Codexreasoning => {
            spawn_logged(
                "codexreasoning",
                handlers::codex_admin::codex_reasoning_handler(bot, state, message),
            );
        }
        Command::Codexusage => {
            spawn_logged(
                "codexusage",
                handlers::codex_admin::codex_usage_handler(bot, message),
            );
        }
    }
    Ok(())
}

async fn handle_callback_query(bot: Bot, state: AppState, query: CallbackQuery) -> HandlerResult {
    let Some(data) = query.data.clone() else {
        return Ok(());
    };
    if data.starts_with(MODEL_CALLBACK_PREFIX) {
        spawn_logged(
            "model selection callback",
            qa::model_selection_callback(bot, state, query),
        );
        return Ok(());
    }
    if data.starts_with(CODEX_MODEL_SELECT_CALLBACK_PREFIX)
        || data.starts_with(CODEX_MODEL_PAGE_CALLBACK_PREFIX)
        || data.starts_with(CODEX_REASONING_SELECT_CALLBACK_PREFIX)
    {
        spawn_logged(
            "codex admin callback",
            handlers::codex_admin::codex_admin_callback(bot, state, query),
        );
        return Ok(());
    }
    if is_image_selection_callback(&data) {
        spawn_logged(
            "image selection callback",
            commands::image_selection_callback(bot, state, query),
        );
    }
    Ok(())
}

async fn handle_media_group(state: AppState, message: Message) -> HandlerResult {
    media::handle_media_group(state, message).await;
    Ok(())
}

async fn handle_text_message(bot: Bot, state: AppState, message: Message) -> HandlerResult {
    if let Some(text) = message.text().or_else(|| message.caption()) {
        if text.trim_start().starts_with('/') {
            return Ok(());
        }
    }

    if qa::should_auto_q_trigger(&message, state.bot_user_id, &state.bot_username_lower) {
        let query = qa::build_auto_q_query(&message, state.bot_user_id, &state.bot_username_lower);
        spawn_logged("auto q", qa::q_handler(bot, state, message, query, "q"));
        return Ok(());
    }

    handlers::responses::log_message(&state, &message).await;
    Ok(())
}

async fn ignore_message(_message: Message) -> HandlerResult {
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use super::*;

    #[tokio::test]
    async fn spawn_logged_runs_the_handler_to_completion() {
        let ran = Arc::new(AtomicBool::new(false));
        let flag = ran.clone();
        spawn_logged("test", async move {
            flag.store(true, Ordering::SeqCst);
            Ok::<(), anyhow::Error>(())
        })
        .await
        .expect("task should not panic");
        assert!(ran.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn spawn_logged_swallows_handler_errors_without_panicking() {
        spawn_logged("test", async { Err::<(), _>(anyhow::anyhow!("boom")) })
            .await
            .expect("a failing handler must not abort the task");
    }

    #[test]
    fn published_commands_reuse_the_enum_descriptions_and_hide_admin_commands() {
        let enum_commands = Command::bot_commands();
        let published = published_bot_commands(true);
        assert!(!published.is_empty());

        for command in &published {
            let source = enum_commands
                .iter()
                .find(|candidate| candidate.command.trim_start_matches('/') == command.command)
                .unwrap_or_else(|| panic!("/{} is not a Command variant", command.command));
            assert_eq!(
                command.description, source.description,
                "/{} is published with a description that differs from /help",
                command.command
            );
        }

        let names = published
            .iter()
            .map(|command| command.command.as_str())
            .collect::<Vec<_>>();
        for hidden in [
            "img2",
            "status",
            "diagnose",
            "token_stats",
            "codexlogin",
            "codexlogout",
            "codexmodel",
            "codexreasoning",
            "codexusage",
        ] {
            assert!(!names.contains(&hidden), "/{hidden} must not be published");
        }
        for public in [
            "start",
            "help",
            "q",
            "burn_baby_burn",
            "token_devourers",
            "support",
        ] {
            assert!(names.contains(&public), "/{public} must be published");
        }
    }

    #[test]
    fn image_selection_callbacks_are_recognized_by_prefix() {
        assert!(is_image_selection_callback("image_model:abc|gemini"));
        assert!(is_image_selection_callback(
            "image_codex_size:abc|1024x1024"
        ));
        assert!(is_image_selection_callback("image_res:abc|2K"));
        assert!(is_image_selection_callback("image_aspect:abc|auto"));
        assert!(!is_image_selection_callback("model_select:gemini"));
        assert!(!is_image_selection_callback("codex_model_select:0"));
    }

    #[test]
    fn img2_command_parses_but_is_not_published() {
        assert!(<Command as BotCommands>::parse("/img2 draw a nebula", "test_bot").is_ok());

        let commands = published_bot_commands(true)
            .into_iter()
            .map(|command| command.command)
            .collect::<Vec<_>>();

        assert!(!commands.iter().any(|command| command == "img2"));
    }

    #[test]
    fn command_descriptions_are_readable_and_attached_to_the_right_commands() {
        let text = <Command as BotCommands>::descriptions().to_string();
        let line_for = |command: &str| {
            text.lines()
                .find(|line| line.starts_with(command))
                .unwrap_or_else(|| panic!("no description line for {command}: {text}"))
                .to_string()
        };

        assert!(
            !text.contains('ç') && !text.contains('Ã'),
            "double-encoded UTF-8 in command descriptions: {text}"
        );
        assert!(line_for("/codexlogin").contains("登录 ChatGPT Codex"));
        assert!(line_for("/codexlogout").contains("退出 ChatGPT Codex"));
        assert!(line_for("/codexmodel").contains("选择当前 Codex 模型"));
        assert!(line_for("/support").contains("投喂AI小喵"));
    }

    #[test]
    fn image_command_descriptions_do_not_advertise_the_removed_vertex_backend() {
        let enum_text = <Command as BotCommands>::descriptions().to_string();
        assert!(!enum_text.contains("Vertex"), "{enum_text}");

        let published = published_bot_commands(true);
        let img = published
            .iter()
            .find(|command| command.command == "img")
            .expect("/img is published");
        assert!(!img.description.contains("Vertex"), "{}", img.description);
        assert!(img.description.contains("Gemini"));
    }

    #[test]
    fn published_commands_keep_search_when_gemini_is_disabled() {
        let commands = published_bot_commands(false)
            .into_iter()
            .map(|command| command.command)
            .collect::<Vec<_>>();

        assert!(commands.iter().any(|command| command == "s"));
        assert!(!commands.iter().any(|command| command == "vid"));
        assert!(!commands.iter().any(|command| command == "mysong"));
        assert!(commands.iter().any(|command| command == "q"));
    }
}
