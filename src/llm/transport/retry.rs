//! Retry policy shared by every LLM/HTTP provider call.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::time::Duration;

use reqwest::header::{HeaderMap, RETRY_AFTER};
use reqwest::StatusCode;

/// Upper bound on any single wait between attempts, whatever the server asks.
pub const DEFAULT_MAX_RETRY_DELAY: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backoff {
    /// `base_delay * attempt`
    Linear,
    /// `base_delay * 2^(attempt - 1)`
    Exponential,
}

#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// Total attempts including the first one.
    pub max_attempts: usize,
    pub base_delay: Duration,
    pub backoff: Backoff,
    pub max_delay: Duration,
    /// Add up to 25% of the computed delay so synchronized clients do not
    /// retry in lock-step.
    pub jitter: bool,
    /// Prefer the server's `Retry-After` / rate-limit reset hint over the
    /// computed backoff when one is present.
    pub honor_retry_after: bool,
    /// Treat a 401 as "credentials expired": retry once more after the
    /// request builder has had a chance to refresh them.
    pub refresh_auth_on_unauthorized: bool,
}

impl RetryPolicy {
    pub const fn linear(max_attempts: usize, base_delay: Duration) -> Self {
        Self {
            max_attempts,
            base_delay,
            backoff: Backoff::Linear,
            max_delay: DEFAULT_MAX_RETRY_DELAY,
            jitter: true,
            honor_retry_after: true,
            refresh_auth_on_unauthorized: false,
        }
    }

    pub const fn exponential(max_attempts: usize, base_delay: Duration) -> Self {
        Self {
            backoff: Backoff::Exponential,
            ..Self::linear(max_attempts, base_delay)
        }
    }

    /// Whether another attempt may follow attempt number `attempt` (1-based).
    pub fn allows_retry(&self, attempt: usize) -> bool {
        attempt < self.max_attempts
    }

    /// How long to wait after attempt number `attempt` failed.
    pub fn delay(&self, attempt: usize, retry_after: Option<Duration>) -> Duration {
        if self.honor_retry_after {
            if let Some(wait) = retry_after {
                return wait.min(self.max_delay);
            }
        }

        let attempt = attempt.max(1) as u32;
        let base = match self.backoff {
            Backoff::Linear => self.base_delay.saturating_mul(attempt),
            Backoff::Exponential => self
                .base_delay
                .saturating_mul(1u32.checked_shl(attempt - 1).unwrap_or(u32::MAX)),
        }
        .min(self.max_delay);

        if !self.jitter {
            return base;
        }
        let spread = base / 4;
        if spread.is_zero() {
            return base;
        }
        let jitter = Duration::from_nanos(random_u64() % (spread.as_nanos() as u64 + 1));
        base.saturating_add(jitter).min(self.max_delay)
    }
}

/// Cheap, dependency-free randomness: every `RandomState` is seeded with
/// fresh keys, so hashing nothing yields a well-mixed value.
fn random_u64() -> u64 {
    RandomState::new().build_hasher().finish()
}

/// Statuses that describe a transient condition worth retrying.
pub fn is_retryable_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::REQUEST_TIMEOUT
        || status.is_server_error()
}

/// Send-phase failures that are worth retrying (the request never got a
/// response). Anything else (bad URL, body encoding) will fail identically.
pub fn is_retryable_transport_error(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect()
}

/// The longest wait the server asked for. `Retry-After` (seconds or
/// HTTP-date) is an explicit instruction and counts on any status; the
/// OpenAI-style `x-ratelimit-reset-*` Go durations and the IETF
/// `ratelimit-reset` seconds are window bookkeeping that providers attach to
/// ordinary responses too, so they count only for 429. `None` when no usable
/// hint is present.
pub fn retry_after_from_headers(status: StatusCode, headers: &HeaderMap) -> Option<Duration> {
    let header = |name: &str| {
        headers
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty())
    };

    let mut waits = Vec::new();
    if let Some(value) = header(RETRY_AFTER.as_str()) {
        if let Some(wait) = parse_retry_after(value) {
            waits.push(wait);
        }
    }
    if status == StatusCode::TOO_MANY_REQUESTS {
        for name in ["x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"] {
            if let Some(wait) = header(name).and_then(parse_go_duration) {
                waits.push(wait);
            }
        }
        if let Some(wait) = header("ratelimit-reset")
            .and_then(|value| value.parse::<u64>().ok())
            .map(Duration::from_secs)
        {
            waits.push(wait);
        }
    }
    waits.into_iter().max()
}

fn parse_retry_after(value: &str) -> Option<Duration> {
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    let at = chrono::DateTime::parse_from_rfc2822(value).ok()?;
    let millis = (at.with_timezone(&chrono::Utc) - chrono::Utc::now())
        .num_milliseconds()
        .max(0);
    Some(Duration::from_millis(millis as u64))
}

/// Parse Go's duration syntax (`1.5s`, `250ms`, `6m0s`, `1h`).
pub(crate) fn parse_go_duration(value: &str) -> Option<Duration> {
    let mut rest = value.trim();
    if rest.is_empty() {
        return None;
    }
    let mut total_seconds = 0f64;
    while !rest.is_empty() {
        let number_end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(rest.len());
        if number_end == 0 {
            return None;
        }
        let number: f64 = rest[..number_end].parse().ok()?;
        rest = &rest[number_end..];
        let unit_end = rest
            .find(|c: char| c.is_ascii_digit() || c == '.')
            .unwrap_or(rest.len());
        let unit = &rest[..unit_end];
        rest = &rest[unit_end..];
        let scale = match unit {
            "h" => 3600.0,
            "m" => 60.0,
            "s" => 1.0,
            "ms" => 1e-3,
            "us" | "µs" => 1e-6,
            "ns" => 1e-9,
            _ => return None,
        };
        total_seconds += number * scale;
    }
    if total_seconds.is_finite() && total_seconds >= 0.0 {
        Some(Duration::from_secs_f64(total_seconds))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderMap, HeaderValue};

    trait NoJitter {
        fn no_jitter(self) -> Self;
    }

    impl NoJitter for RetryPolicy {
        fn no_jitter(self) -> Self {
            RetryPolicy {
                jitter: false,
                ..self
            }
        }
    }

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.append(
                reqwest::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_str(value).unwrap(),
            );
        }
        map
    }

    #[test]
    fn linear_delay_grows_with_the_attempt_number() {
        let policy = RetryPolicy::linear(3, Duration::from_millis(500)).no_jitter();
        assert_eq!(policy.delay(1, None), Duration::from_millis(500));
        assert_eq!(policy.delay(2, None), Duration::from_millis(1000));
    }

    #[test]
    fn exponential_delay_doubles_each_attempt() {
        let policy = RetryPolicy::exponential(4, Duration::from_millis(400)).no_jitter();
        assert_eq!(policy.delay(1, None), Duration::from_millis(400));
        assert_eq!(policy.delay(2, None), Duration::from_millis(800));
        assert_eq!(policy.delay(3, None), Duration::from_millis(1600));
    }

    #[test]
    fn delay_never_exceeds_max_delay() {
        let policy = RetryPolicy::exponential(8, Duration::from_secs(10)).no_jitter();
        assert_eq!(policy.delay(6, None), policy.max_delay);
        assert_eq!(
            policy.delay(1, Some(Duration::from_secs(600))),
            policy.max_delay
        );
    }

    #[test]
    fn jitter_adds_at_most_a_quarter_of_the_base_delay() {
        let policy = RetryPolicy::linear(3, Duration::from_secs(1));
        for _ in 0..200 {
            let delay = policy.delay(1, None);
            assert!(delay >= Duration::from_secs(1), "{delay:?}");
            assert!(delay <= Duration::from_millis(1250), "{delay:?}");
        }
    }

    #[test]
    fn retry_after_overrides_backoff_only_when_honoured() {
        let honoured = RetryPolicy::linear(3, Duration::from_millis(500)).no_jitter();
        assert_eq!(
            honoured.delay(1, Some(Duration::from_secs(7))),
            Duration::from_secs(7)
        );

        let ignored = RetryPolicy {
            honor_retry_after: false,
            ..honoured
        };
        assert_eq!(
            ignored.delay(1, Some(Duration::from_secs(7))),
            Duration::from_millis(500)
        );
    }

    #[test]
    fn allows_retry_until_the_attempt_budget_is_spent() {
        let policy = RetryPolicy::linear(3, Duration::from_millis(1));
        assert!(policy.allows_retry(1));
        assert!(policy.allows_retry(2));
        assert!(!policy.allows_retry(3));
    }

    #[test]
    fn retryable_status_classification_matches_transient_failures() {
        for status in [429u16, 408, 500, 502, 503, 504] {
            assert!(
                is_retryable_status(StatusCode::from_u16(status).unwrap()),
                "{status}"
            );
        }
        for status in [400u16, 401, 403, 404, 422] {
            assert!(
                !is_retryable_status(StatusCode::from_u16(status).unwrap()),
                "{status}"
            );
        }
    }

    #[test]
    fn retry_after_header_in_seconds_is_parsed() {
        let map = headers(&[("retry-after", "3")]);
        assert_eq!(
            retry_after_from_headers(StatusCode::TOO_MANY_REQUESTS, &map),
            Some(Duration::from_secs(3))
        );
    }

    #[test]
    fn retry_after_http_date_is_converted_to_a_wait() {
        let future = (chrono::Utc::now() + chrono::Duration::seconds(10)).to_rfc2822();
        let map = headers(&[("retry-after", future.as_str())]);
        let wait = retry_after_from_headers(StatusCode::TOO_MANY_REQUESTS, &map)
            .expect("date should parse");
        assert!(
            wait >= Duration::from_secs(8) && wait <= Duration::from_secs(11),
            "{wait:?}"
        );

        let past = (chrono::Utc::now() - chrono::Duration::seconds(10)).to_rfc2822();
        let map = headers(&[("retry-after", past.as_str())]);
        assert_eq!(
            retry_after_from_headers(StatusCode::TOO_MANY_REQUESTS, &map),
            Some(Duration::ZERO)
        );
    }

    #[test]
    fn rate_limit_reset_headers_take_the_longest_wait() {
        let map = headers(&[
            ("x-ratelimit-reset-requests", "1s"),
            ("x-ratelimit-reset-tokens", "6m0s"),
        ]);
        assert_eq!(
            retry_after_from_headers(StatusCode::TOO_MANY_REQUESTS, &map),
            Some(Duration::from_secs(360))
        );

        let map = headers(&[("ratelimit-reset", "12")]);
        assert_eq!(
            retry_after_from_headers(StatusCode::TOO_MANY_REQUESTS, &map),
            Some(Duration::from_secs(12))
        );
    }

    #[test]
    fn go_style_durations_are_parsed() {
        assert_eq!(parse_go_duration("250ms"), Some(Duration::from_millis(250)));
        assert_eq!(parse_go_duration("1.5s"), Some(Duration::from_millis(1500)));
        assert_eq!(parse_go_duration("2m30s"), Some(Duration::from_secs(150)));
        assert_eq!(parse_go_duration("1h"), Some(Duration::from_secs(3600)));
        assert_eq!(parse_go_duration("soon"), None);
        assert_eq!(parse_go_duration(""), None);
    }

    #[test]
    fn missing_headers_yield_no_retry_after() {
        assert_eq!(
            retry_after_from_headers(StatusCode::TOO_MANY_REQUESTS, &HeaderMap::new()),
            None
        );
        assert_eq!(
            retry_after_from_headers(
                StatusCode::TOO_MANY_REQUESTS,
                &headers(&[("retry-after", "later")])
            ),
            None
        );
    }

    #[test]
    fn rate_limit_window_hints_count_only_on_429() {
        // Providers send the window-reset headers on ordinary responses too;
        // a 5xx carrying `x-ratelimit-reset-tokens: 6m0s` must keep the
        // normal backoff.
        let map = headers(&[
            ("x-ratelimit-reset-tokens", "6m0s"),
            ("ratelimit-reset", "12"),
        ]);
        assert_eq!(
            retry_after_from_headers(StatusCode::SERVICE_UNAVAILABLE, &map),
            None
        );
        assert_eq!(
            retry_after_from_headers(StatusCode::TOO_MANY_REQUESTS, &map),
            Some(Duration::from_secs(360))
        );

        // Retry-After itself is an explicit instruction on any status.
        let map = headers(&[("retry-after", "2"), ("x-ratelimit-reset-tokens", "6m0s")]);
        assert_eq!(
            retry_after_from_headers(StatusCode::SERVICE_UNAVAILABLE, &map),
            Some(Duration::from_secs(2))
        );
    }
}
