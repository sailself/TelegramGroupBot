use std::time::Duration;

use anyhow::{anyhow, Result};
use reqwest::{Client, Response};
use url::Url;

use crate::config::Config;
use crate::utils::http::get_http_client_no_redirect as shared_http_client_no_redirect;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TwitterProvider {
    FxTwitter,
    VxTwitter,
    Jina,
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct ProviderError {
    pub(crate) provider: TwitterProvider,
    pub(crate) kind: ProviderErrorKind,
    pub(crate) status: Option<reqwest::StatusCode>,
    pub(crate) detail: String,
}

impl TryFrom<&Config> for TwitterFetchConfig {
    type Error = anyhow::Error;

    fn try_from(config: &Config) -> Result<Self> {
        let providers = config
            .twitter_fetch_providers
            .iter()
            .map(|name| TwitterProvider::parse(name))
            .collect::<Result<Vec<_>>>()?;

        fn parse_endpoint(raw: &str) -> Result<Url> {
            Url::parse(raw).map_err(|_| anyhow!("invalid Twitter provider endpoint"))
        }

        Ok(Self {
            providers,
            fxtwitter_api_base: parse_endpoint(&config.fxtwitter_api_base)?,
            vxtwitter_api_base: parse_endpoint(&config.vxtwitter_api_base)?,
            jina_reader_endpoint: parse_endpoint(&config.jina_reader_endpoint)?,
            jina_api_key: (!config.jina_ai_api_key.is_empty())
                .then(|| config.jina_ai_api_key.clone()),
            total_timeout: Duration::from_secs(config.twitter_fetch_total_timeout_secs),
            provider_timeout: Duration::from_secs(config.twitter_provider_timeout_secs),
            response_max_bytes: config.twitter_response_max_bytes,
        })
    }
}

#[allow(dead_code)]
pub(crate) async fn read_limited_body(
    provider: TwitterProvider,
    mut response: Response,
    max_bytes: usize,
) -> std::result::Result<Vec<u8>, ProviderError> {
    let status = response.status();
    if response
        .content_length()
        .is_some_and(|length| length > max_bytes as u64)
    {
        return Err(ProviderError {
            provider,
            kind: ProviderErrorKind::BodyTooLarge,
            status: Some(status),
            detail: "provider response exceeds configured byte limit".to_string(),
        });
    }

    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| ProviderError {
        provider,
        kind: ProviderErrorKind::Transport,
        status: Some(status),
        detail: "provider response body read failed".to_string(),
    })? {
        if body.len().saturating_add(chunk.len()) > max_bytes {
            return Err(ProviderError {
                provider,
                kind: ProviderErrorKind::BodyTooLarge,
                status: Some(status),
                detail: "provider response exceeds configured byte limit".to_string(),
            });
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

#[allow(dead_code)]
pub(crate) fn get_http_client_no_redirect() -> &'static Client {
    shared_http_client_no_redirect()
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;
    use crate::config::CONFIG;
    use crate::tools::twitter_extractor::test_support::{
        chunked_response, response_with_content_length, ExpectedRequest, TestServer,
    };

    fn join_with_timeout(server: TestServer) -> Result<Result<(), String>, &'static str> {
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = sender.send(server.join());
        });
        receiver
            .recv_timeout(Duration::from_secs(2))
            .map_err(|_| "TestServer::join timed out")
    }

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
        disabled.twitter_fetch_providers = vec!["jina".into()];
        disabled.enable_jina_mcp = false;
        let mut enabled = disabled.clone();
        enabled.enable_jina_mcp = true;

        assert_eq!(
            TwitterFetchConfig::try_from(&disabled).unwrap().providers,
            vec![TwitterProvider::Jina]
        );
        assert_eq!(
            TwitterFetchConfig::try_from(&enabled).unwrap().providers,
            vec![TwitterProvider::Jina]
        );
    }

    #[test]
    fn test_server_join_surfaces_missing_expectations_without_hanging() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/never-requested",
            response_with_content_length(0, Vec::new()),
        )]);
        let error = join_with_timeout(server)
            .expect("missing-expectation join must terminate")
            .unwrap_err();
        assert!(error.contains("unmet"));
    }

    #[tokio::test]
    async fn test_server_join_surfaces_unexpected_extra_requests() {
        let server = TestServer::single(response_with_content_length(0, Vec::new()));
        let client = reqwest::Client::new();
        client.get(server.url("/first")).send().await.unwrap();
        let extra = client.get(server.url("/extra")).send().await.unwrap();
        assert_eq!(extra.status(), reqwest::StatusCode::INTERNAL_SERVER_ERROR);
        let error = join_with_timeout(server)
            .expect("extra-request join must terminate")
            .unwrap_err();
        assert!(error.contains("unexpected extra request"));
    }
}
