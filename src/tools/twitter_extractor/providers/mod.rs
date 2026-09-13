use std::time::Duration;

use anyhow::{anyhow, Result};
use reqwest::Response;
use url::Url;

use crate::config::Config;
use crate::tools::twitter_extractor::model::XPost;
use crate::utils::http::NoRedirectClient;

pub(crate) mod fxtwitter;
pub(crate) mod jina;
pub(crate) mod vxtwitter;

const TWITTER_USER_AGENT: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TwitterProvider {
    FxTwitter,
    VxTwitter,
    Jina,
}

impl TwitterProvider {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::FxTwitter => "fxtwitter",
            Self::VxTwitter => "vxtwitter",
            Self::Jina => "jina",
        }
    }
}

impl TwitterProvider {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "fxtwitter" => Ok(Self::FxTwitter),
            "vxtwitter" => Ok(Self::VxTwitter),
            "jina" => Ok(Self::Jina),
            _ => Err(anyhow!("unsupported Twitter provider name")),
        }
    }
}

#[derive(Clone)]
pub(crate) struct TwitterFetchConfig {
    pub(crate) providers: Vec<TwitterProvider>,
    pub(crate) fxtwitter_api_base: Url,
    pub(crate) vxtwitter_api_base: Url,
    pub(crate) jina_reader_endpoint: Url,
    pub(crate) jina_api_key: Option<String>,
    pub(crate) total_timeout: Duration,
    pub(crate) provider_timeout: Duration,
    pub(crate) response_max_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProviderErrorKind {
    Timeout,
    Transport,
    HttpStatus,
    BodyTooLarge,
    Decode,
    Incomplete,
    DeadlineExhausted,
}

#[derive(Debug, Clone)]
pub(crate) struct ProviderError {
    pub(crate) provider: TwitterProvider,
    pub(crate) kind: ProviderErrorKind,
    pub(crate) status: Option<reqwest::StatusCode>,
    pub(crate) detail: String,
}

impl ProviderError {
    pub(crate) fn deadline(provider: TwitterProvider) -> Self {
        Self {
            provider,
            kind: ProviderErrorKind::DeadlineExhausted,
            status: None,
            detail: "total deadline exhausted".to_string(),
        }
    }
}

impl ProviderErrorKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Timeout => "timeout",
            Self::Transport => "transport",
            Self::HttpStatus => "http_status",
            Self::BodyTooLarge => "body_too_large",
            Self::Decode => "decode",
            Self::Incomplete => "incomplete",
            Self::DeadlineExhausted => "deadline_exhausted",
        }
    }
}

pub(crate) fn aggregate_provider_failures(
    identity: &crate::tools::twitter_extractor::url::XStatusIdentity,
    failures: &[ProviderError],
) -> anyhow::Error {
    let summaries = failures
        .iter()
        .map(|failure| {
            let status = failure
                .status
                .map_or_else(|| "-".to_string(), |status| status.as_u16().to_string());
            let detail = failure.detail.trim();
            if detail.is_empty() {
                format!(
                    "{}:{}/{}",
                    failure.provider.as_str(),
                    failure.kind.as_str(),
                    status
                )
            } else {
                format!(
                    "{}:{}/{} ({})",
                    failure.provider.as_str(),
                    failure.kind.as_str(),
                    status,
                    detail
                )
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    anyhow!("failed to fetch X status {}: {}", identity.id, summaries)
}

impl TryFrom<&Config> for TwitterFetchConfig {
    type Error = anyhow::Error;

    fn try_from(config: &Config) -> Result<Self> {
        let providers = config
            .twitter
            .fetch_providers
            .iter()
            .map(|name| TwitterProvider::parse(name))
            .collect::<Result<Vec<_>>>()?;

        fn parse_endpoint(raw: &str) -> Result<Url> {
            Url::parse(raw).map_err(|_| anyhow!("invalid Twitter provider endpoint"))
        }

        Ok(Self {
            providers,
            fxtwitter_api_base: parse_endpoint(&config.twitter.fxtwitter_api_base)?,
            vxtwitter_api_base: parse_endpoint(&config.twitter.vxtwitter_api_base)?,
            jina_reader_endpoint: parse_endpoint(&config.jina.reader_endpoint)?,
            jina_api_key: (!config.jina.api_key.is_empty()).then(|| config.jina.api_key.clone()),
            total_timeout: Duration::from_secs(config.twitter.fetch_total_timeout_secs),
            provider_timeout: Duration::from_secs(config.twitter.provider_timeout_secs),
            response_max_bytes: config.twitter.response_max_bytes,
        })
    }
}

pub(crate) async fn read_limited_body(
    provider: TwitterProvider,
    response: Response,
    max_bytes: usize,
) -> std::result::Result<bytes::Bytes, ProviderError> {
    let status = response.status();
    crate::utils::http::read_body_capped("twitter provider", response, max_bytes)
        .await
        .map_err(|err| match err {
            crate::utils::http::BodyCapError::DeclaredTooLarge { .. }
            | crate::utils::http::BodyCapError::StreamedTooLarge { .. } => ProviderError {
                provider,
                kind: ProviderErrorKind::BodyTooLarge,
                status: Some(status),
                detail: "provider response exceeds configured byte limit".to_string(),
            },
            crate::utils::http::BodyCapError::Read { .. } => ProviderError {
                provider,
                kind: ProviderErrorKind::Transport,
                status: Some(status),
                detail: "provider response body read failed".to_string(),
            },
        })
}

/// Everything a provider's HTTP request needs beyond the endpoint URL: the
/// client to send it with (each provider is invoked with the same shared,
/// no-redirect client the extractor holds), an optional bearer token (jina
/// only), and an optional `User-Agent` override (fx/vx only).
pub(super) struct ProviderFetch<'a> {
    pub(super) url: Url,
    pub(super) bearer: Option<&'a str>,
    pub(super) user_agent: Option<&'static str>,
    pub(super) client: &'a NoRedirectClient,
    pub(super) provider: TwitterProvider,
}

/// One GET with the shared no-redirect client, bounded by `timeout` and
/// `max_bytes`; returns the status and body so the caller's `parse` can
/// back-fill the status on error.
pub(super) async fn fetch_provider_body(
    fetch: ProviderFetch<'_>,
    max_bytes: usize,
    timeout: Duration,
) -> std::result::Result<(reqwest::StatusCode, bytes::Bytes), ProviderError> {
    let provider = fetch.provider;
    let future = async {
        let mut request = fetch.client.get(fetch.url);
        if let Some(agent) = fetch.user_agent {
            request = request.header(reqwest::header::USER_AGENT, agent);
        }
        if let Some(token) = fetch.bearer {
            request = request.bearer_auth(token);
        }
        let response = request.send().await.map_err(|_| ProviderError {
            provider,
            kind: ProviderErrorKind::Transport,
            status: None,
            detail: "provider request failed".to_string(),
        })?;
        let status = response.status();
        if !status.is_success() {
            return Err(ProviderError {
                provider,
                kind: ProviderErrorKind::HttpStatus,
                status: Some(status),
                detail: "provider returned a non-success status".to_string(),
            });
        }
        let body = read_limited_body(provider, response, max_bytes).await?;
        Ok((status, body))
    };
    tokio::time::timeout(timeout, future)
        .await
        .map_err(|_| ProviderError {
            provider,
            kind: ProviderErrorKind::Timeout,
            status: None,
            detail: "provider request timed out".to_string(),
        })?
}

/// Fetches the provider's response then hands the body to `parse`; if
/// `parse` fails without a status, back-fills the HTTP status observed on
/// the wire (every provider gets this uniformly now, closing a gap where
/// jina previously did not).
pub(super) async fn fetch_and_parse<P>(
    fetch: ProviderFetch<'_>,
    max_bytes: usize,
    timeout: Duration,
    parse: P,
) -> std::result::Result<XPost, ProviderError>
where
    P: FnOnce(&[u8]) -> std::result::Result<XPost, ProviderError>,
{
    let (status, body) = fetch_provider_body(fetch, max_bytes, timeout).await?;
    parse(&body).map_err(|mut error| {
        if error.status.is_none() {
            error.status = Some(status);
        }
        error
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CONFIG;
    use crate::tools::twitter_extractor::test_support::{
        chunked_response, response_with_content_length, ExpectedRequest, TestServer,
    };

    #[test]
    fn provider_names_round_trip_without_aliases() {
        assert_eq!(
            TwitterProvider::parse("fxtwitter").unwrap(),
            TwitterProvider::FxTwitter
        );
        assert_eq!(
            TwitterProvider::parse("vxtwitter").unwrap(),
            TwitterProvider::VxTwitter
        );
        assert_eq!(
            TwitterProvider::parse("jina").unwrap(),
            TwitterProvider::Jina
        );
        assert!(TwitterProvider::parse("fx").is_err());
    }

    #[tokio::test]
    async fn limited_body_rejects_declared_and_streamed_overflow() {
        let declared = TestServer::new(vec![
            ExpectedRequest::any(response_with_content_length(2_048, vec![b'x'; 2_048])),
            ExpectedRequest::any(response_with_content_length(2_048, vec![b'x'; 2_048])),
        ]);
        let response = reqwest::Client::new()
            .get(declared.url("/"))
            .send()
            .await
            .unwrap();
        assert_eq!(
            read_limited_body(TwitterProvider::VxTwitter, response, 1_024)
                .await
                .unwrap_err()
                .kind,
            ProviderErrorKind::BodyTooLarge
        );
        let response = reqwest::Client::new()
            .get(declared.url("/"))
            .send()
            .await
            .unwrap();
        let error = read_limited_body(TwitterProvider::Jina, response, 1_024)
            .await
            .unwrap_err();
        assert_eq!(error.provider, TwitterProvider::Jina);
        declared.join().unwrap();

        let chunked = TestServer::single(chunked_response(vec![vec![b'a'; 700], vec![b'b'; 700]]));
        let response = reqwest::Client::new()
            .get(chunked.url("/"))
            .send()
            .await
            .unwrap();
        assert_eq!(
            read_limited_body(TwitterProvider::VxTwitter, response, 1_024)
                .await
                .unwrap_err()
                .kind,
            ProviderErrorKind::BodyTooLarge
        );
        chunked.join().unwrap();
    }

    #[test]
    fn jina_provider_selection_is_independent_of_jina_mcp_flag() {
        let mut disabled = (*CONFIG).clone();
        disabled.twitter.fetch_providers = vec!["jina".into()];
        disabled.jina.enable_mcp = false;
        let mut enabled = disabled.clone();
        enabled.jina.enable_mcp = true;

        assert_eq!(
            TwitterFetchConfig::try_from(&disabled).unwrap().providers,
            vec![TwitterProvider::Jina]
        );
        assert_eq!(
            TwitterFetchConfig::try_from(&enabled).unwrap().providers,
            vec![TwitterProvider::Jina]
        );
    }
}
