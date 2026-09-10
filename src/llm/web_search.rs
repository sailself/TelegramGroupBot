use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use anyhow::{anyhow, Result};
use parking_lot::Mutex;
use std::sync::LazyLock;
use tracing::{info, warn};

use crate::config::CONFIG;
use crate::llm::brave_search::BraveSearch;
use crate::llm::exa_search::ExaSearch;
use crate::llm::jina_search::JinaSearch;
use crate::llm::transport::RetryPolicy;
use crate::utils::text::truncate_with_ellipsis;
use crate::utils::ttl_cache::TtlCache;

const DEFAULT_MAX_RESULTS: usize = 5;
const MAX_RESULTS_LIMIT: usize = 10;
const SNIPPET_LIMIT: usize = 240;
/// Wall-clock budget for one `search_web` call across every provider tried.
pub const WEB_SEARCH_TOTAL_DEADLINE: Duration = Duration::from_secs(45);
/// The most of that budget any single provider may consume, so one hanging
/// or rate-limited backend cannot starve the rest of the chain.
pub(crate) const WEB_SEARCH_PROVIDER_DEADLINE: Duration = Duration::from_secs(15);
/// Timeout for a single provider request.
pub(crate) const WEB_SEARCH_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
/// One attempt per provider: the next provider in the chain is the retry, so
/// a failing or rate-limited backend falls through at once instead of
/// sleeping on `Retry-After` inside its slice.
pub(crate) const WEB_SEARCH_RETRY_POLICY: RetryPolicy = RetryPolicy::linear(1, Duration::ZERO);

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// One web-search backend. Implementations return raw results; the
/// aggregator normalizes, truncates and caches them.
pub(crate) trait SearchProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn is_enabled(&self) -> bool;
    fn search<'a>(
        &'a self,
        query: &'a str,
        max_results: usize,
    ) -> BoxFuture<'a, Result<Vec<SearchResult>>>;
}

static BRAVE: BraveSearch = BraveSearch;
static EXA: ExaSearch = ExaSearch;
static JINA: JinaSearch = JinaSearch;

static SEARCH_CACHE: LazyLock<Mutex<TtlCache<String, Vec<SearchResult>>>> =
    LazyLock::new(|| Mutex::new(TtlCache::new(cache_ttl(), CONFIG.search.cache_max_entries)));

fn provider_by_name(name: &str) -> Option<&'static dyn SearchProvider> {
    match name.trim().to_lowercase().as_str() {
        "brave" => Some(&BRAVE),
        "exa" => Some(&EXA),
        "jina" => Some(&JINA),
        _ => None,
    }
}

/// Providers in `WEB_SEARCH_PROVIDERS` order (unknown names are logged and
/// skipped); Brave, Exa, Jina when the list is empty.
fn configured_providers() -> Vec<&'static dyn SearchProvider> {
    let mut providers = Vec::new();
    for entry in &CONFIG.search.providers {
        match provider_by_name(entry) {
            Some(provider) => providers.push(provider),
            None => warn!(
                "Unknown web search provider '{}' in WEB_SEARCH_PROVIDERS",
                entry
            ),
        }
    }
    if providers.is_empty() {
        providers = vec![&BRAVE, &EXA, &JINA];
    }
    providers
}

pub fn is_search_enabled() -> bool {
    configured_providers()
        .iter()
        .any(|provider| provider.is_enabled())
}

fn normalize_snippet(value: &str) -> String {
    let snippet = value.replace('\n', " ");
    truncate_with_ellipsis(snippet.trim(), SNIPPET_LIMIT)
}

fn normalize_result(mut result: SearchResult) -> Option<SearchResult> {
    if result.url.trim().is_empty() {
        return None;
    }

    if result.title.trim().is_empty() {
        result.title = result.url.clone();
    }

    result.snippet = normalize_snippet(&result.snippet);
    Some(result)
}

fn cache_key(query: &str, max_results: usize) -> String {
    format!(
        "{}::{}",
        query.trim().to_lowercase().replace('\n', " "),
        max_results
    )
}

fn cache_ttl() -> Duration {
    Duration::from_secs(CONFIG.search.cache_ttl_seconds)
}

fn get_cached(query: &str, max_results: usize) -> Option<Vec<SearchResult>> {
    SEARCH_CACHE.lock().get(&cache_key(query, max_results))
}

/// Cache a completed search. Empty result sets are not cached: they are
/// usually a provider hiccup, and caching them would pin "no results" for
/// the whole TTL. A zero WEB_SEARCH_CACHE_TTL_SECONDS disables the cache.
fn cache_search_results(query: &str, max_results: usize, results: &[SearchResult]) {
    if results.is_empty() {
        return;
    }
    SEARCH_CACHE
        .lock()
        .insert(cache_key(query, max_results), results.to_vec());
}

/// Try `providers` in order until one returns results, within one overall
/// `deadline`. Each provider gets at most `provider_deadline` of what is left,
/// so a hanging backend is abandoned while the next one still has a window;
/// once the total is spent nothing further is tried.
async fn search_web_with(
    providers: &[&dyn SearchProvider],
    query: &str,
    max_results: usize,
    deadline: Duration,
    provider_deadline: Duration,
) -> Result<Vec<SearchResult>> {
    let deadline_at = tokio::time::Instant::now() + deadline;
    let mut last_error: Option<String> = None;
    let mut had_success = false;

    for provider in providers.iter().filter(|provider| provider.is_enabled()) {
        let remaining = deadline_at.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            // Keep the earlier failure as the cause; a provider that ate the
            // budget is more informative than the ones it starved.
            last_error.get_or_insert_with(|| {
                format!(
                    "{}: skipped because the {deadline:?} web search deadline is spent",
                    provider.name()
                )
            });
            break;
        }

        let slice = remaining.min(provider_deadline);
        info!("Trying web search provider '{}'", provider.name());
        match tokio::time::timeout(slice, provider.search(query, max_results)).await {
            Ok(Ok(results)) => {
                had_success = true;
                let mut normalized = results
                    .into_iter()
                    .filter_map(normalize_result)
                    .collect::<Vec<_>>();
                normalized.truncate(max_results);
                if !normalized.is_empty() {
                    return Ok(normalized);
                }
            }
            Ok(Err(err)) => {
                last_error = Some(format!("{}: {}", provider.name(), err));
            }
            Err(_) => {
                last_error = Some(format!(
                    "{}: timed out after its {slice:?} share of the {deadline:?} web search deadline",
                    provider.name()
                ));
            }
        }
    }

    if had_success {
        if let Some(message) = last_error {
            warn!("Web search had partial failures: {}", message);
        }
        return Ok(Vec::new());
    }

    if let Some(message) = last_error {
        warn!("Web search failed: {}", message);
        return Err(anyhow!("Web search failed: {}", message));
    }

    Ok(Vec::new())
}

pub async fn search_web(query: &str, max_results: Option<usize>) -> Result<Vec<SearchResult>> {
    if query.trim().is_empty() {
        return Err(anyhow!("query must not be empty"));
    }

    let max_results = max_results
        .unwrap_or(DEFAULT_MAX_RESULTS)
        .clamp(1, MAX_RESULTS_LIMIT);

    if let Some(results) = get_cached(query, max_results) {
        return Ok(results);
    }

    let providers = configured_providers();
    if !providers.iter().any(|provider| provider.is_enabled()) {
        return Err(anyhow!("No web search providers are enabled"));
    }

    let results = search_web_with(
        &providers,
        query,
        max_results,
        WEB_SEARCH_TOTAL_DEADLINE,
        WEB_SEARCH_PROVIDER_DEADLINE,
    )
    .await?;
    cache_search_results(query, max_results, &results);
    Ok(results)
}

pub fn format_results_markdown(query: &str, results: &[SearchResult]) -> String {
    if results.is_empty() {
        return format!("No web results found for query: {}", query);
    }

    let mut lines = vec![format!("Search results for **{}**:", query)];
    for (idx, result) in results.iter().enumerate() {
        lines.push(format!("{}. [{}]({})", idx + 1, result.title, result.url));
        if !result.snippet.is_empty() {
            lines.push(format!("   {}", result.snippet));
        }
    }
    lines.join("\n")
}

pub async fn web_search_tool(query: &str, max_results: Option<usize>) -> Result<String> {
    let results = search_web(query, max_results).await?;
    Ok(format_results_markdown(query, &results))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(url: &str) -> SearchResult {
        SearchResult {
            title: "t".to_string(),
            url: url.to_string(),
            snippet: "s".to_string(),
        }
    }

    /// Scripted provider: `Ok(results)`, `Err(message)`, or hang forever.
    struct FakeProvider {
        name: &'static str,
        enabled: bool,
        outcome: FakeOutcome,
        calls: std::sync::atomic::AtomicUsize,
    }

    enum FakeOutcome {
        Results(Vec<SearchResult>),
        Failure(&'static str),
        Hang,
    }

    impl SearchProvider for FakeProvider {
        fn name(&self) -> &'static str {
            self.name
        }

        fn is_enabled(&self) -> bool {
            self.enabled
        }

        fn search<'a>(
            &'a self,
            _query: &'a str,
            _max_results: usize,
        ) -> BoxFuture<'a, Result<Vec<SearchResult>>> {
            Box::pin(async move {
                self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                match &self.outcome {
                    FakeOutcome::Results(results) => Ok(results.clone()),
                    FakeOutcome::Failure(message) => Err(anyhow!("{message}")),
                    FakeOutcome::Hang => {
                        tokio::time::sleep(Duration::from_secs(3600)).await;
                        Ok(Vec::new())
                    }
                }
            })
        }
    }

    fn provider(name: &'static str, outcome: FakeOutcome) -> FakeProvider {
        FakeProvider {
            name,
            enabled: true,
            outcome,
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    #[tokio::test]
    async fn first_provider_with_results_wins_and_disabled_ones_are_skipped() {
        let disabled = FakeProvider {
            name: "off",
            enabled: false,
            outcome: FakeOutcome::Results(vec![result("https://off.example")]),
            calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let empty = provider("empty", FakeOutcome::Results(Vec::new()));
        let hit = provider(
            "hit",
            FakeOutcome::Results(vec![
                result("https://a.example"),
                result("https://b.example"),
                result("https://c.example"),
            ]),
        );
        let later = provider(
            "later",
            FakeOutcome::Results(vec![result("https://z.example")]),
        );

        let results = search_web_with(
            &[&disabled, &empty, &hit, &later],
            "rust",
            2,
            Duration::from_secs(10),
            Duration::from_secs(10),
        )
        .await
        .expect("a provider answered");

        let urls: Vec<&str> = results.iter().map(|r| r.url.as_str()).collect();
        assert_eq!(urls, vec!["https://a.example", "https://b.example"]);
        assert_eq!(
            disabled.calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "disabled providers are never called"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_hanging_provider_spends_only_its_slice_and_the_next_provider_runs() {
        let slow = provider("slow", FakeOutcome::Hang);
        let fast = provider(
            "fast",
            FakeOutcome::Results(vec![result("https://fast.example")]),
        );
        let started = tokio::time::Instant::now();

        let results = search_web_with(
            &[&slow, &fast],
            "rust",
            5,
            Duration::from_secs(45),
            Duration::from_secs(15),
        )
        .await
        .expect("the next provider answers after the slow one is cut off");

        assert_eq!(results[0].url, "https://fast.example");
        assert_eq!(slow.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(fast.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_secs(15) && elapsed < Duration::from_secs(16),
            "the slow provider was abandoned at its slice: {elapsed:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_total_deadline_still_caps_the_whole_chain() {
        let first = provider("first", FakeOutcome::Hang);
        let second = provider("second", FakeOutcome::Hang);
        let third = provider(
            "third",
            FakeOutcome::Results(vec![result("https://third.example")]),
        );
        let started = tokio::time::Instant::now();

        let err = search_web_with(
            &[&first, &second, &third],
            "rust",
            5,
            Duration::from_secs(20),
            Duration::from_secs(15),
        )
        .await
        .expect_err("the chain ran out of total time");

        assert!(err.to_string().contains("second: timed out"), "{err}");
        assert_eq!(
            third.calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "no window is left for the third provider"
        );
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_secs(20) && elapsed < Duration::from_secs(21),
            "the second provider got only the remaining 5s: {elapsed:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn nothing_runs_once_the_overall_deadline_is_spent() {
        let slow = provider("slow", FakeOutcome::Hang);
        let never_reached = provider(
            "never",
            FakeOutcome::Results(vec![result("https://x.example")]),
        );

        // The first provider consumes the whole budget; the deadline is total,
        // not per provider, so the second one must not get a fresh window.
        let err = search_web_with(
            &[&slow, &never_reached],
            "rust",
            5,
            Duration::from_millis(0),
            Duration::from_secs(15),
        )
        .await
        .expect_err("no provider can run inside a zero deadline");
        assert!(err.to_string().contains("deadline"), "{err}");
    }

    #[tokio::test]
    async fn all_failures_surface_the_last_error_but_a_successful_empty_answer_does_not() {
        let broken = provider("broken", FakeOutcome::Failure("boom"));
        let err = search_web_with(
            &[&broken],
            "rust",
            5,
            Duration::from_secs(5),
            Duration::from_secs(5),
        )
        .await
        .expect_err("every provider failed");
        assert!(err.to_string().contains("broken: boom"), "{err}");

        let empty = provider("empty", FakeOutcome::Results(Vec::new()));
        let results = search_web_with(
            &[&broken, &empty],
            "rust",
            5,
            Duration::from_secs(5),
            Duration::from_secs(5),
        )
        .await
        .expect("one provider answered, even if with nothing");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn results_are_normalized_once_in_the_aggregator() {
        let messy = provider(
            "messy",
            FakeOutcome::Results(vec![
                SearchResult {
                    title: "   ".to_string(),
                    url: "https://untitled.example".to_string(),
                    snippet: format!("line one\nline two {}", "x".repeat(300)),
                },
                SearchResult {
                    title: "no url".to_string(),
                    url: "  ".to_string(),
                    snippet: String::new(),
                },
            ]),
        );

        let results = search_web_with(
            &[&messy],
            "rust",
            5,
            Duration::from_secs(5),
            Duration::from_secs(5),
        )
        .await
        .expect("results");

        assert_eq!(results.len(), 1, "results without a URL are dropped");
        assert_eq!(results[0].title, "https://untitled.example");
        assert!(!results[0].snippet.contains('\n'));
        assert!(results[0].snippet.ends_with("..."));
        assert!(results[0].snippet.chars().count() <= SNIPPET_LIMIT + 3);
    }

    #[test]
    fn empty_search_results_are_not_cached() {
        let query = "phase0 cache policy empty 7f3a";
        cache_search_results(query, 5, &[]);
        assert!(get_cached(query, 5).is_none());
    }

    #[test]
    fn non_empty_search_results_are_cached() {
        let query = "phase0 cache policy nonempty 7f3a";
        cache_search_results(query, 5, &[result("https://example.com")]);
        assert_eq!(get_cached(query, 5).map(|r| r.len()), Some(1));
    }
}
