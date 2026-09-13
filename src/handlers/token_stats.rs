//! Token usage reporting: `/burn_baby_burn`, `/token_devourers`, `/token_stats`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use anyhow::Result;
use chrono::Utc;
use teloxide::prelude::*;
use teloxide::types::ParseMode;

use crate::config::CONFIG;
use crate::db::models::{ModelTokenStat, TokenUserStat};
use crate::handlers::access::{check_access_control, check_admin_access};
use crate::handlers::content::create_telegraph_page;
use crate::state::AppState;
use crate::utils::telegram::{send_message_with_retry, send_message_with_retry_parse_mode};
use crate::utils::text::{escape_html, split_for_telegram};

const TOKEN_DEVOURERS_DEFAULT_LIMIT: i64 = 5;
const TOKEN_DEVOURERS_MAX_LIMIT: i64 = 20;
const BURN_BABY_BURN_TEMPLATES: [&str; 3] = [
    "Your token pyre blazes at {tokens} tokens. A worthy offering.",
    "Behold! You have burned {tokens} tokens in this chat. The flame hungers still.",
    "Your personal token bonfire has consumed {tokens} tokens in this chat. Legend behavior.",
];
const TOKEN_DEVOURERS_HEADERS: [&str; 4] = [
    "Behold! The greatest token devourers in this chat:",
    "Bow down to the great LLM conquerors!",
    "The feast is over. These mortals ate the most tokens:",
    "Here stand the reigning lords of consumption:",
];
const TOKEN_FOOTERS: [&str; 5] = [
    "The ledger has spoken.",
    "Glory and bankruptcy.",
    "A noble harvest of tokens.",
    "The accountants are in tears.",
    "This chat alone could frighten a GPU cluster.",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenStatsView {
    Total,
    Model,
    User,
}

fn format_compact_token_count(tokens: i64) -> String {
    if tokens.abs() < 1_000 {
        return tokens.to_string();
    }

    let thresholds = [
        (1_000_000_000_000_f64, "T"),
        (1_000_000_000_f64, "B"),
        (1_000_000_f64, "M"),
        (1_000_f64, "k"),
    ];
    let abs_tokens = tokens.abs() as f64;

    for (divisor, suffix) in thresholds {
        if abs_tokens >= divisor {
            let scaled = tokens as f64 / divisor;
            let formatted = if scaled.abs() >= 10.0 {
                format!("{scaled:.0}")
            } else {
                format!("{scaled:.1}")
            };
            return format!("{}{}", formatted.trim_end_matches(".0"), suffix);
        }
    }

    tokens.to_string()
}

fn pick_copy_variant<'a>(variants: &'a [&'a str], seed: u64) -> &'a str {
    let index = (seed as usize) % variants.len();
    variants[index]
}

fn message_copy_seed(message: &Message, salt: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    salt.hash(&mut hasher);
    message.chat.id.0.hash(&mut hasher);
    message.id.0.hash(&mut hasher);
    Utc::now()
        .timestamp_nanos_opt()
        .unwrap_or_default()
        .hash(&mut hasher);
    hasher.finish()
}

fn pick_message_variant<'a>(variants: &'a [&'a str], message: &Message, salt: &str) -> &'a str {
    pick_copy_variant(variants, message_copy_seed(message, salt))
}

fn apply_token_template_html(template: &str, tokens: i64) -> String {
    template.replace(
        "{tokens}",
        &format!("<b>{}</b>", format_compact_token_count(tokens)),
    )
}

fn random_token_footer(message: &Message, salt: &str) -> &'static str {
    pick_message_variant(&TOKEN_FOOTERS, message, salt)
}

fn build_burn_baby_burn_response_html(message: &Message, total_tokens: i64) -> String {
    let body = apply_token_template_html(
        pick_message_variant(&BURN_BABY_BURN_TEMPLATES, message, "burn_body"),
        total_tokens,
    );
    format!("{body}\n\n{}", random_token_footer(message, "burn_footer"))
}

fn format_token_user_lines_html(rows: &[TokenUserStat]) -> Vec<String> {
    let label_map = crate::llm::prompting::build_display_label_map(
        rows.iter()
            .map(|row| (row.user_id, row.username.as_deref().unwrap_or("Anonymous"))),
    );

    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            let label = label_map.get(&row.user_id).cloned().unwrap_or_else(|| {
                row.username
                    .clone()
                    .unwrap_or_else(|| "Anonymous".to_string())
            });
            format!(
                "{}. <b>{}</b>: <b>{}</b> tokens",
                index + 1,
                escape_html(&label),
                format_compact_token_count(row.total_tokens)
            )
        })
        .collect()
}

fn build_token_devourers_response_html(message: &Message, rows: &[TokenUserStat]) -> String {
    if rows.is_empty() {
        return format!(
            "The feast table is empty. No token devourers have risen in this chat yet.\n\n{}",
            random_token_footer(message, "token_devourers_empty")
        );
    }

    let mut lines =
        vec![
            pick_message_variant(&TOKEN_DEVOURERS_HEADERS, message, "token_devourers_header")
                .to_string(),
        ];
    lines.push(String::new());
    lines.extend(format_token_user_lines_html(rows));
    lines.push(String::new());
    lines.push(random_token_footer(message, "token_devourers_footer").to_string());
    lines.join("\n")
}

fn format_token_user_lines(rows: &[TokenUserStat]) -> Vec<String> {
    let label_map = crate::llm::prompting::build_display_label_map(
        rows.iter()
            .map(|row| (row.user_id, row.username.as_deref().unwrap_or("Anonymous"))),
    );

    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            let label = label_map.get(&row.user_id).cloned().unwrap_or_else(|| {
                row.username
                    .clone()
                    .unwrap_or_else(|| "Anonymous".to_string())
            });
            format!(
                "{}. {}: {} tokens",
                index + 1,
                label,
                format_compact_token_count(row.total_tokens)
            )
        })
        .collect()
}

fn build_token_stats_total_response(total_tokens: i64) -> String {
    format!(
        "Total token usage: {} tokens",
        format_compact_token_count(total_tokens)
    )
}

fn build_token_stats_model_response(rows: &[ModelTokenStat]) -> String {
    let mut lines = vec!["Token usage by model:".to_string()];
    lines.push(String::new());

    if rows.is_empty() {
        lines.push("No model token usage has been recorded yet.".to_string());
    } else {
        lines.extend(rows.iter().enumerate().map(|(index, row)| {
            format!(
                "{}. {}:{}: {} tokens",
                index + 1,
                row.provider,
                row.model,
                format_compact_token_count(row.total_tokens)
            )
        }));
    }

    lines.join("\n")
}

fn build_token_stats_user_response(rows: &[TokenUserStat]) -> String {
    let mut lines = vec!["Token usage by user:".to_string()];
    lines.push(String::new());

    if rows.is_empty() {
        lines.push("No user token usage has been recorded yet.".to_string());
    } else {
        lines.extend(format_token_user_lines(rows));
    }

    lines.join("\n")
}

async fn send_plain_text_report(
    bot: &Bot,
    message: &Message,
    title: &str,
    report: &str,
    telegraph_notice: &str,
) -> Result<()> {
    let too_long = report.lines().count() > 22 || report.len() > CONFIG.telegram_max_length;
    if !too_long {
        send_message_with_retry(bot, message.chat.id, report, Some(message.id)).await?;
        return Ok(());
    }

    if let Some(url) = create_telegraph_page(title, report).await {
        let notice = format!("{telegraph_notice}\n\n{url}");
        send_message_with_retry(bot, message.chat.id, &notice, Some(message.id)).await?;
        return Ok(());
    }

    let chunks = split_for_telegram(
        report,
        CONFIG.telegram_max_length.saturating_sub(100).max(1),
    );
    for (index, chunk) in chunks.into_iter().enumerate() {
        let reply_to = if index == 0 { Some(message.id) } else { None };
        send_message_with_retry(bot, message.chat.id, &chunk, reply_to).await?;
    }

    Ok(())
}

fn parse_token_devourers_limit(limit: Option<&str>) -> Result<i64> {
    let Some(limit) = limit.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(TOKEN_DEVOURERS_DEFAULT_LIMIT);
    };

    let parsed = limit
        .parse::<i64>()
        .map_err(|_| anyhow::anyhow!("Usage: /token_devourers [number from 1 to 20]"))?;
    Ok(parsed.clamp(1, TOKEN_DEVOURERS_MAX_LIMIT))
}

fn parse_token_stats_view(view: Option<&str>) -> Option<TokenStatsView> {
    let Some(normalized) = view.map(str::trim) else {
        return Some(TokenStatsView::Total);
    };
    if normalized.is_empty() {
        return Some(TokenStatsView::Total);
    }
    let normalized = normalized.to_ascii_lowercase();

    match normalized.as_str() {
        "model" => Some(TokenStatsView::Model),
        "user" => Some(TokenStatsView::User),
        _ => None,
    }
}

pub async fn burn_baby_burn_handler(bot: Bot, state: AppState, message: Message) -> Result<()> {
    if !check_access_control(&bot, &message, "burn_baby_burn").await {
        return Ok(());
    }

    let user_id = message
        .from
        .as_ref()
        .and_then(|user| i64::try_from(user.id.0).ok())
        .unwrap_or_default();
    let total_tokens = state
        .db
        .select_chat_token_total_for_user(message.chat.id.0, user_id)
        .await?;

    send_message_with_retry_parse_mode(
        &bot,
        message.chat.id,
        &build_burn_baby_burn_response_html(&message, total_tokens),
        Some(message.id),
        Some(ParseMode::Html),
    )
    .await?;
    Ok(())
}

pub async fn token_devourers_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    limit: Option<String>,
) -> Result<()> {
    if !check_access_control(&bot, &message, "token_devourers").await {
        return Ok(());
    }
    if !message.chat.is_group() && !message.chat.is_supergroup() {
        send_message_with_retry(
            &bot,
            message.chat.id,
            "This command can only be summoned in a group chat. Even legends need an audience.",
            Some(message.id),
        )
        .await?;
        return Ok(());
    }

    let limit = match parse_token_devourers_limit(limit.as_deref()) {
        Ok(limit) => limit,
        Err(err) => {
            send_message_with_retry(&bot, message.chat.id, &err.to_string(), Some(message.id))
                .await?;
            return Ok(());
        }
    };
    let rows = state
        .db
        .select_top_chat_token_users(message.chat.id.0, limit)
        .await?;

    send_message_with_retry_parse_mode(
        &bot,
        message.chat.id,
        &build_token_devourers_response_html(&message, &rows),
        Some(message.id),
        Some(ParseMode::Html),
    )
    .await?;
    Ok(())
}

pub async fn token_stats_handler(
    bot: Bot,
    state: AppState,
    message: Message,
    view: Option<String>,
) -> Result<()> {
    if !check_admin_access(&bot, &message, "token_stats").await {
        return Ok(());
    }

    let view = match parse_token_stats_view(view.as_deref()) {
        Some(view) => view,
        None => {
            send_message_with_retry(
                &bot,
                message.chat.id,
                "Usage: /token_stats, /token_stats model, or /token_stats user",
                Some(message.id),
            )
            .await?;
            return Ok(());
        }
    };

    match view {
        TokenStatsView::Total => {
            let total_tokens = state.db.select_global_token_total().await?;
            send_message_with_retry(
                &bot,
                message.chat.id,
                &build_token_stats_total_response(total_tokens),
                Some(message.id),
            )
            .await?;
        }
        TokenStatsView::Model => {
            let rows = state.db.select_global_token_totals_by_model().await?;
            let report = build_token_stats_model_response(&rows);
            send_plain_text_report(
                &bot,
                &message,
                "Token Stats by Model",
                &report,
                "The scroll grew too vast for Telegram. Read the full imperial ledger here:",
            )
            .await?;
        }
        TokenStatsView::User => {
            let rows = state.db.select_global_token_totals_by_user().await?;
            let report = build_token_stats_user_response(&rows);
            send_plain_text_report(
                &bot,
                &message,
                "Token Stats by User",
                &report,
                "The ledger overflowed its parchment. Read the full champions list here:",
            )
            .await?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_token_count_formats_thresholds() {
        assert_eq!(format_compact_token_count(999), "999");
        assert_eq!(format_compact_token_count(1_000), "1k");
        assert_eq!(format_compact_token_count(1_200_000), "1.2M");
        assert_eq!(format_compact_token_count(1_000_000_000), "1B");
        assert_eq!(format_compact_token_count(2_000_000_000_000), "2T");
    }

    #[test]
    fn token_devourers_limit_defaults_and_clamps() {
        assert_eq!(
            parse_token_devourers_limit(None).expect("default limit should parse"),
            TOKEN_DEVOURERS_DEFAULT_LIMIT
        );
        assert_eq!(
            parse_token_devourers_limit(Some("25")).expect("clamped limit should parse"),
            TOKEN_DEVOURERS_MAX_LIMIT
        );
        assert_eq!(
            parse_token_devourers_limit(Some("0")).expect("low limit should clamp"),
            1
        );
        assert!(parse_token_devourers_limit(Some("abc")).is_err());
    }

    #[test]
    fn token_stats_view_parsing_accepts_known_values() {
        assert_eq!(parse_token_stats_view(None), Some(TokenStatsView::Total));
        assert_eq!(
            parse_token_stats_view(Some("")),
            Some(TokenStatsView::Total)
        );
        assert_eq!(
            parse_token_stats_view(Some("model")),
            Some(TokenStatsView::Model)
        );
        assert_eq!(
            parse_token_stats_view(Some("USER")),
            Some(TokenStatsView::User)
        );
        assert_eq!(parse_token_stats_view(Some("weird")), None);
    }

    #[test]
    fn copy_picker_returns_only_pool_members() {
        let seen = (0..32_u64)
            .map(|seed| pick_copy_variant(&TOKEN_DEVOURERS_HEADERS, seed))
            .collect::<Vec<_>>();
        assert!(seen
            .iter()
            .all(|value| TOKEN_DEVOURERS_HEADERS.contains(value)));
    }

    #[test]
    fn token_template_html_bolds_token_total() {
        let response = apply_token_template_html("Tribute: {tokens} tokens", 12_345);
        assert_eq!(response, "Tribute: <b>12k</b> tokens");
    }

    #[test]
    fn token_user_lines_bold_usernames_and_totals() {
        let rows = vec![TokenUserStat {
            user_id: 42,
            username: Some("Alice".to_string()),
            total_tokens: 9_876,
        }];

        let lines = format_token_user_lines_html(&rows);
        assert_eq!(lines, vec!["1. <b>Alice</b>: <b>9.9k</b> tokens"]);
    }

    #[test]
    fn token_stats_response_is_plain_text() {
        let rows = vec![TokenUserStat {
            user_id: 42,
            username: Some("Alice".to_string()),
            total_tokens: 9_876,
        }];

        assert_eq!(
            build_token_stats_total_response(12_345),
            "Total token usage: 12k tokens"
        );
        assert_eq!(
            build_token_stats_user_response(&rows),
            "Token usage by user:\n\n1. Alice: 9.9k tokens"
        );
    }
}
