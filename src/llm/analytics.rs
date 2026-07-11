//! Structured, chat-scoped analytics query language for the /qc agent.
//! `compile` turns a model-supplied `QuerySpec` into exactly one parameterized
//! read-only SELECT over `messages`. `chat_id` is supplied by the caller and is
//! always the first bind — never sourced from the model.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Telegram service account for anonymous-admin posts; excluded from per-user
/// stats because all such posts collapse onto this single id.
/// NOTE: verify against this bot's stored rows (see A3 test) — best-effort filter.
const GROUP_ANONYMOUS_BOT_ID: i64 = 1_087_968_824;
pub const MAX_ANALYTICS_LIMIT: i64 = 50; // REVIEW: was 100; spec says 1..=50
const DEFAULT_ANALYTICS_LIMIT: i64 = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    Count,
    DistinctCount,
    MinDate,
    MaxDate,
    AvgLen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupBy {
    User,
    Day,
    HourOfDay,
    Weekday,
    Month,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Order {
    ValueDesc,
    ValueAsc,
    GroupAsc,
    GroupDesc,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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

#[derive(Debug, Clone, Deserialize, Serialize)]
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

pub fn normalize_and_validate(mut spec: QuerySpec) -> Result<QuerySpec, String> {
    validate(&spec)?;

    fn normalize_bound(field: &str, value: Option<String>) -> Result<Option<String>, String> {
        match value {
            Some(value) => normalize_stats_date(&value)
                .map(Some)
                .ok_or_else(|| format!("{field} must be YYYY-MM-DD or RFC3339")),
            None => Ok(None),
        }
    }

    spec.filters.date_from = normalize_bound("date_from", spec.filters.date_from)?;
    spec.filters.date_to = normalize_bound("date_to", spec.filters.date_to)?;
    if matches!(
        (&spec.filters.date_from, &spec.filters.date_to),
        (Some(from), Some(to)) if from >= to
    ) {
        return Err("date_from must be earlier than date_to".to_string());
    }

    if let Some(term) = spec.filters.term.as_deref() {
        let normalized = crate::db::search::normalize_search_query(term);
        if crate::db::search::build_and_match_expression(&normalized).is_none() {
            return Err("term must contain at least one searchable token".to_string());
        }
    }

    spec.limit = Some(
        spec.limit
            .unwrap_or(DEFAULT_ANALYTICS_LIMIT)
            .clamp(1, MAX_ANALYTICS_LIMIT),
    );
    Ok(spec)
}

/// Compile to `(sql, binds)`. `chat_id` is always the first bind. Call
/// `normalize_and_validate` first; date bounds and the limit must be canonical.
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
        // CAST to REAL so sqlx can decode into Option<f64> regardless of whether
        // SQLite infers INTEGER or REAL affinity for the aggregate result.
        Metric::Count => ("CAST(COUNT(*) AS REAL) AS value_num", "NULL AS value_text"),
        Metric::DistinctCount => (
            "CAST(COUNT(DISTINCT m.user_id) AS REAL) AS value_num",
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
        let normalized = crate::db::search::normalize_search_query(&term);
        let expression = crate::db::search::build_and_match_expression(&normalized)
            .expect("validated analytics term must contain search tokens");
        sql.push_str(
            " AND m.id IN (SELECT messages_fts.rowid FROM messages_fts WHERE messages_fts MATCH ?)",
        );
        binds.push(Bind::Text(expression));
    }
    if let Some(text) = nonempty(&spec.filters.text_contains) {
        sql.push_str(" AND m.text LIKE ? ESCAPE '\\'");
        binds.push(Bind::Text(format!("%{}%", escape_like(&text))));
    }
    if let Some(f) = nonempty(&spec.filters.date_from) {
        sql.push_str(" AND m.date >= ?");
        binds.push(Bind::Text(f));
    }
    if let Some(t) = nonempty(&spec.filters.date_to) {
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
fn escape_like(input: &str) -> String {
    input
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// Normalize a date bound (YYYY-MM-DD or full RFC3339) to a canonical RFC3339
/// UTC string comparable against the stored `date` column. None if unparseable.
pub fn normalize_stats_date(value: &str) -> Option<String> {
    let v = value.trim();
    if v.is_empty() {
        return None;
    }
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(v) {
        return Some(dt.with_timezone(&chrono::Utc).to_rfc3339());
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(v, "%Y-%m-%d") {
        let naive = d.and_hms_opt(0, 0, 0)?;
        return Some(
            chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(naive, chrono::Utc)
                .to_rfc3339(),
        );
    }
    None
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
    fn term_uses_normalized_and_expression() {
        let normalized = normalize_and_validate(spec(
            r#"{"metric":"count","filters":{"term":"Bitcoin rally"}}"#,
        ))
        .unwrap();
        let (_, binds) = compile(&normalized, 1);
        assert_eq!(
            binds[1],
            Bind::Text("search_text : bitcoin AND search_text : rally".to_string())
        );
    }
    #[test]
    fn date_metric_orders_by_value_text() {
        let (sql, _) = compile(&spec(r#"{"metric":"max_date","group_by":"user"}"#), 1);
        assert!(sql.contains("ORDER BY value_text DESC"));
        let (sql2, _) = compile(&spec(r#"{"metric":"count","group_by":"user"}"#), 1);
        assert!(sql2.contains("ORDER BY value_num DESC"));
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

    #[test]
    fn avg_len_uses_length_and_value_num_ordering() {
        let (sql, _) = compile(&spec(r#"{"metric":"avg_len","group_by":"day"}"#), 1);
        assert!(sql.contains("AVG(LENGTH(m.text)) AS value_num"));
        assert!(sql.contains("ORDER BY value_num DESC"));
    }

    #[test]
    fn group_by_none_emits_no_group_by_clause() {
        let (sql, _) = compile(&spec(r#"{"metric":"count"}"#), 1);
        assert!(!sql.contains("GROUP BY"));
        assert!(sql.contains("NULL AS group_key"));
    }

    #[test]
    fn exclude_flags_toggle_clauses() {
        let (on, _) = compile(&spec(r#"{"metric":"count"}"#), 1); // defaults: exclude on
        assert!(on.contains("is_command = 0") && on.contains("is_synthetic_record = 0"));
        let (off, _) = compile(
            &spec(
                r#"{"metric":"count","filters":{"exclude_commands":false,"exclude_synthetic":false,"exclude_ai_asks":true}}"#,
            ),
            1,
        );
        assert!(!off.contains("is_command = 0"));
        assert!(!off.contains("is_synthetic_record = 0"));
        assert!(off.contains("asks_ai = 0"));
    }

    #[test]
    fn normalize_stats_date_accepts_date_and_rfc3339() {
        // YYYY-MM-DD → T00:00:00+00:00
        let result = super::normalize_stats_date("2026-01-15");
        assert_eq!(result, Some("2026-01-15T00:00:00+00:00".to_string()));

        // Full RFC3339 passes through normalized to UTC
        let result2 = super::normalize_stats_date("2026-01-15T12:34:56+05:30");
        assert!(result2.is_some());
        let s = result2.unwrap();
        // Should be normalized to UTC; the offset +05:30 = -5h30m from UTC
        assert!(s.ends_with("+00:00") || s.ends_with('Z'));
        assert!(s.contains("07:04:56") || s.contains("2026-01-15T07:04:56"));

        // Garbage → None
        assert_eq!(super::normalize_stats_date("not-a-date"), None);
        // Empty → None
        assert_eq!(super::normalize_stats_date(""), None);
        assert_eq!(super::normalize_stats_date("   "), None);
    }

    #[test]
    fn normalization_rejects_bad_or_non_increasing_dates() {
        let bad = spec(r#"{"metric":"count","filters":{"date_from":"last week"}}"#);
        assert_eq!(
            normalize_and_validate(bad).unwrap_err(),
            "date_from must be YYYY-MM-DD or RFC3339"
        );

        let equal = spec(
            r#"{"metric":"count","filters":{"date_from":"2026-07-01","date_to":"2026-07-01"}}"#,
        );
        assert_eq!(
            normalize_and_validate(equal).unwrap_err(),
            "date_from must be earlier than date_to"
        );

        let inverted = spec(
            r#"{"metric":"count","filters":{"date_from":"2026-07-02","date_to":"2026-07-01"}}"#,
        );
        assert_eq!(
            normalize_and_validate(inverted).unwrap_err(),
            "date_from must be earlier than date_to"
        );
    }

    #[test]
    fn normalization_canonicalizes_dates_and_limit() {
        let raw = spec(
            r#"{"metric":"count","filters":{"date_from":"2026-07-01","date_to":"2026-07-02T03:00:00+03:00"},"limit":999}"#,
        );
        let normalized = normalize_and_validate(raw).unwrap();
        assert_eq!(
            normalized.filters.date_from.as_deref(),
            Some("2026-07-01T00:00:00+00:00")
        );
        assert_eq!(
            normalized.filters.date_to.as_deref(),
            Some("2026-07-02T00:00:00+00:00")
        );
        assert_eq!(normalized.limit, Some(MAX_ANALYTICS_LIMIT));
    }

    #[test]
    fn normalization_rejects_term_without_search_tokens() {
        let raw = spec(r#"{"metric":"count","filters":{"term":"!!!"}}"#);
        assert_eq!(
            normalize_and_validate(raw).unwrap_err(),
            "term must contain at least one searchable token"
        );
    }
}
