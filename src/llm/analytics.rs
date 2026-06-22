//! Structured, chat-scoped analytics query language for the /qc agent.
//! `compile` turns a model-supplied `QuerySpec` into exactly one parameterized
//! read-only SELECT over `messages`. `chat_id` is supplied by the caller and is
//! always the first bind — never sourced from the model.

// This module is consumed by later tasks (A2+); suppress dead_code until they land.
#![allow(dead_code)]

use serde::Deserialize;
use serde_json::{json, Value};

/// Telegram service account for anonymous-admin posts; excluded from per-user
/// stats because all such posts collapse onto this single id.
/// NOTE: verify against this bot's stored rows (see A3 test) — best-effort filter.
const GROUP_ANONYMOUS_BOT_ID: i64 = 1_087_968_824;
pub const MAX_ANALYTICS_LIMIT: i64 = 50; // REVIEW: was 100; spec says 1..=50
const DEFAULT_ANALYTICS_LIMIT: i64 = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    Count,
    DistinctCount,
    MinDate,
    MaxDate,
    AvgLen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupBy {
    User,
    Day,
    HourOfDay,
    Weekday,
    Month,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Order {
    ValueDesc,
    ValueAsc,
    GroupAsc,
    GroupDesc,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Filters {
    #[serde(default)]
    pub term: Option<String>,
    #[serde(default)]
    pub text_contains: Option<String>,
    #[serde(default)]
    pub date_from: Option<String>,
    #[serde(default)]
    pub date_to: Option<String>,
    #[serde(default)]
    pub user_id: Option<i64>,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default = "default_true")]
    pub exclude_commands: bool,
    #[serde(default = "default_true")]
    pub exclude_synthetic: bool,
    #[serde(default)]
    pub exclude_ai_asks: bool,
}
fn default_true() -> bool {
    true
}

// REVIEW (Codex BLOCKER): a derived Default would set the exclude_* bools to
// false, contradicting the serde per-field defaults. Implement Default manually
// so `Filters::default()` (used by Plan B) matches an omitted JSON `filters`.
impl Default for Filters {
    fn default() -> Self {
        Self {
            term: None,
            text_contains: None,
            date_from: None,
            date_to: None,
            user_id: None,
            username: None,
            exclude_commands: true,
            exclude_synthetic: true,
            exclude_ai_asks: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuerySpec {
    pub metric: Metric,
    #[serde(default = "default_group_by")]
    pub group_by: GroupBy,
    #[serde(default)]
    pub filters: Filters,
    #[serde(default = "default_order")]
    pub order: Order,
    #[serde(default)]
    pub limit: Option<i64>,
}
fn default_group_by() -> GroupBy {
    GroupBy::None
}
fn default_order() -> Order {
    Order::ValueDesc
}

#[derive(Debug, Clone, PartialEq)]
pub enum Bind {
    Int(i64),
    Text(String),
}

// REVIEW (SQL M1): reject nonsensical metric×group_by instead of returning
// silently-wrong all-ones rows. Caller surfaces this as `invalid_arguments`.
pub fn validate(spec: &QuerySpec) -> Result<(), String> {
    if spec.metric == Metric::DistinctCount && spec.group_by == GroupBy::User {
        return Err(
            "distinct_count is only meaningful with group_by other than 'user' \
                    (e.g. group_by 'none' for total distinct senders, or 'day')."
                .into(),
        );
    }
    Ok(())
}

/// Compile to `(sql, binds)`. `chat_id` is always the first bind. Call `validate`
/// first. `date_from`/`date_to` are expected already-normalized (see callers).
pub fn compile(spec: &QuerySpec, chat_id: i64) -> (String, Vec<Bind>) {
    let mut binds: Vec<Bind> = vec![Bind::Int(chat_id)];

    let (group_select, group_expr) = match spec.group_by {
        GroupBy::User => (
            "m.user_id AS group_user_id, ( \
             SELECT m2.username FROM messages m2 \
             WHERE m2.chat_id = m.chat_id AND m2.user_id = m.user_id AND m2.username IS NOT NULL \
             ORDER BY m2.date DESC, m2.message_id DESC LIMIT 1 ) AS group_key",
            "m.user_id",
        ),
        GroupBy::Day => (
            "NULL AS group_user_id, date(m.date) AS group_key",
            "date(m.date)",
        ),
        GroupBy::HourOfDay => (
            "NULL AS group_user_id, strftime('%H', m.date) AS group_key",
            "strftime('%H', m.date)",
        ),
        GroupBy::Weekday => (
            "NULL AS group_user_id, strftime('%w', m.date) AS group_key",
            "strftime('%w', m.date)",
        ),
        GroupBy::Month => (
            "NULL AS group_user_id, strftime('%Y-%m', m.date) AS group_key",
            "strftime('%Y-%m', m.date)",
        ),
        GroupBy::None => ("NULL AS group_user_id, NULL AS group_key", ""),
    };
    let (value_num, value_text) = match spec.metric {
        Metric::Count => ("COUNT(*) AS value_num", "NULL AS value_text"),
        Metric::DistinctCount => (
            "COUNT(DISTINCT m.user_id) AS value_num",
            "NULL AS value_text",
        ),
        Metric::AvgLen => ("AVG(LENGTH(m.text)) AS value_num", "NULL AS value_text"),
        Metric::MinDate => ("NULL AS value_num", "MIN(m.date) AS value_text"),
        Metric::MaxDate => ("NULL AS value_num", "MAX(m.date) AS value_text"),
    };

    let mut sql = format!(
        "SELECT {group_select}, {value_num}, {value_text} \
         FROM messages m \
         WHERE m.chat_id = ? AND m.user_id IS NOT NULL AND m.text IS NOT NULL \
           AND m.user_id <> {GROUP_ANONYMOUS_BOT_ID}"
    );
    if spec.filters.exclude_synthetic {
        sql.push_str(" AND m.is_synthetic_record = 0");
    }
    if spec.filters.exclude_commands {
        sql.push_str(" AND m.is_command = 0");
    }
    if spec.filters.exclude_ai_asks {
        sql.push_str(" AND m.asks_ai = 0");
    }

    if let Some(term) = nonempty(&spec.filters.term) {
        // REVIEW (all reviewers): never bind a raw term into MATCH — it throws
        // on `c++`, quotes, `%`, operators. Wrap as a quoted FTS phrase, and use
        // the JOIN form /s uses (perf + scope parity).
        sql.push_str(" AND m.id IN (SELECT mf.rowid FROM messages_fts mf WHERE mf MATCH ?)");
        binds.push(Bind::Text(to_fts_phrase(&term)));
    }
    if let Some(text) = nonempty(&spec.filters.text_contains) {
        sql.push_str(" AND m.text LIKE ? ESCAPE '\\'");
        binds.push(Bind::Text(format!("%{}%", escape_like(&text))));
    }
    // REVIEW (SQL m3): mirror Plan B — drop an inverted/degenerate range.
    let (df, dt) = match (
        nonempty(&spec.filters.date_from),
        nonempty(&spec.filters.date_to),
    ) {
        (Some(f), Some(t)) if f >= t => (None, None),
        (f, t) => (f, t),
    };
    if let Some(f) = df {
        sql.push_str(" AND m.date >= ?");
        binds.push(Bind::Text(f));
    }
    if let Some(t) = dt {
        sql.push_str(" AND m.date < ?");
        binds.push(Bind::Text(t));
    }
    if let Some(uid) = spec.filters.user_id {
        sql.push_str(" AND m.user_id = ?");
        binds.push(Bind::Int(uid));
    }
    if let Some(u) = nonempty(&spec.filters.username) {
        sql.push_str(" AND m.username = ?");
        binds.push(Bind::Text(u));
    }

    if !group_expr.is_empty() {
        sql.push_str(&format!(" GROUP BY {group_expr}"));
    }

    // REVIEW (SQL B2): order by the metric's ACTIVE column (NULL column would
    // misorder). Date metrics use value_text; others value_num.
    let value_col = match spec.metric {
        Metric::MinDate | Metric::MaxDate => "value_text",
        _ => "value_num",
    };
    let order = match spec.order {
        Order::GroupAsc => "group_key ASC".to_string(),
        Order::GroupDesc => "group_key DESC".to_string(),
        Order::ValueAsc => format!("{value_col} ASC, group_key ASC"),
        Order::ValueDesc => format!("{value_col} DESC, group_key ASC"),
    };
    sql.push_str(&format!(" ORDER BY {order}"));

    let limit = spec
        .limit
        .unwrap_or(DEFAULT_ANALYTICS_LIMIT)
        .clamp(1, MAX_ANALYTICS_LIMIT);
    sql.push_str(" LIMIT ?");
    binds.push(Bind::Int(limit));
    (sql, binds)
}

fn nonempty(v: &Option<String>) -> Option<String> {
    v.as_ref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}
/// FTS5-safe phrase: wrap in quotes and neutralize embedded quotes, so arbitrary
/// model text (incl. `c++`, `a:b`, `OR`) is a literal phrase, never an operator.
fn to_fts_phrase(term: &str) -> String {
    format!("\"{}\"", term.replace('"', " "))
}
fn escape_like(input: &str) -> String {
    input
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

pub fn query_spec_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "metric": { "type": "string", "enum": ["count","distinct_count","min_date","max_date","avg_len"],
                "description": "count=messages; distinct_count=distinct senders (use with group_by none/day, not user); min_date/max_date=earliest/latest matching message time; avg_len=avg message length." },
            "group_by": { "type": "string", "enum": ["user","day","hour_of_day","weekday","month","none"] },
            "filters": { "type": "object", "properties": {
                "term": { "type": "string", "description": "Keyword/phrase matched via full-text search. Use for 'how many times X mentioned'." },
                "text_contains": { "type": "string", "description": "Literal substring (LIKE). Prefer term for word matching." },
                "date_from": { "type": "string", "description": "Inclusive UTC lower bound, RFC3339 or YYYY-MM-DD." },
                "date_to": { "type": "string", "description": "Exclusive UTC upper bound." },
                "user_id": { "type": "integer" }, "username": { "type": "string" },
                "exclude_commands": { "type": "boolean", "description": "default true" },
                "exclude_synthetic": { "type": "boolean", "description": "default true" },
                "exclude_ai_asks": { "type": "boolean", "description": "default false" }
            }},
            "order": { "type": "string", "enum": ["value_desc","value_asc","group_asc","group_desc"] },
            "limit": { "type": "integer", "minimum": 1, "maximum": 50 }
        },
        "required": ["metric"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::step::parse_lenient_json;
    fn spec(j: &str) -> QuerySpec {
        parse_lenient_json::<QuerySpec>(j).expect("spec parses")
    }

    #[test]
    fn defaults_exclude_commands_and_synthetic() {
        let s = spec(r#"{"metric":"count"}"#);
        assert!(
            s.filters.exclude_commands && s.filters.exclude_synthetic && !s.filters.exclude_ai_asks
        );
        assert!(Filters::default().exclude_commands); // manual Default matches serde
    }
    #[test]
    fn chat_id_is_first_bind_and_only_messages_named() {
        let (sql, binds) = compile(&spec(r#"{"metric":"count","group_by":"user"}"#), -42);
        assert!(sql.contains("WHERE m.chat_id = ?"));
        assert_eq!(binds.first(), Some(&Bind::Int(-42)));
        assert_eq!(sql.matches("chat_id = ?").count(), 1);
        assert!(!sql.contains("llm_invocations") && !sql.contains("app_meta"));
    }
    #[test]
    fn term_is_quoted_phrase_not_raw_operator() {
        let (_, b) = compile(&spec(r#"{"metric":"count","filters":{"term":"c++"}}"#), 1);
        assert_eq!(b[1], Bind::Text("\"c++\"".to_string())); // safe phrase, no FTS syntax error
        let (_, b2) = compile(
            &spec(r#"{"metric":"count","filters":{"term":"a\"b OR x"}}"#),
            1,
        );
        assert_eq!(b2[1], Bind::Text("\"a b OR x\"".to_string()));
    }
    #[test]
    fn date_metric_orders_by_value_text() {
        let (sql, _) = compile(&spec(r#"{"metric":"max_date","group_by":"user"}"#), 1);
        assert!(sql.contains("ORDER BY value_text DESC"));
        let (sql2, _) = compile(&spec(r#"{"metric":"count","group_by":"user"}"#), 1);
        assert!(sql2.contains("ORDER BY value_num DESC"));
    }
    #[test]
    fn inverted_range_is_dropped() {
        let (sql, b) = compile(
            &spec(
                r#"{"metric":"count","filters":{"date_from":"2026-06-01","date_to":"2026-05-01"}}"#,
            ),
            1,
        );
        assert!(!sql.contains("m.date >="));
        assert_eq!(b.len(), 2); // chat_id + limit only
    }
    #[test]
    fn validate_rejects_distinct_count_by_user() {
        assert!(validate(&spec(r#"{"metric":"distinct_count","group_by":"user"}"#)).is_err());
        assert!(validate(&spec(r#"{"metric":"distinct_count","group_by":"none"}"#)).is_ok());
    }
    #[test]
    fn limit_clamped_to_50() {
        let (_, b) = compile(&spec(r#"{"metric":"count","limit":9999}"#), 1);
        assert_eq!(b.last(), Some(&Bind::Int(50)));
    }
    #[test]
    fn text_contains_escapes_wildcards() {
        let (_, b) = compile(
            &spec(r#"{"metric":"count","filters":{"text_contains":"50%_off"}}"#),
            1,
        );
        assert_eq!(b[1], Bind::Text("%50\\%\\_off%".to_string()));
    }
}
