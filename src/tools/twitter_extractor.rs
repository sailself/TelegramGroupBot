use std::time::Duration;

use anyhow::Result;

use crate::utils::http::get_http_client_no_redirect;
use crate::utils::http::NoRedirectClient;

pub(crate) mod model;
pub(crate) mod providers;
#[cfg(test)]
pub(crate) mod test_support;
pub(crate) mod url;
pub(crate) use model::parse_allowed_media_url;
pub(crate) use model::TwitterAttachment;
pub use model::TwitterContent;
pub(crate) use url::{
    canonical_status_key, is_supported_status_url, parse_status_identity, XStatusIdentity,
};

use providers::{aggregate_provider_failures, ProviderError, TwitterFetchConfig, TwitterProvider};

pub(crate) struct TwitterExtractor<'a> {
    client: &'a NoRedirectClient,
    config: TwitterFetchConfig,
}

impl<'a> TwitterExtractor<'a> {
    pub(crate) fn new(client: &'a NoRedirectClient, config: TwitterFetchConfig) -> Self {
        Self { client, config }
    }

    pub(crate) async fn fetch(&self, raw_url: &str) -> Result<TwitterContent> {
        let identity = parse_status_identity(raw_url)?;
        let started = tokio::time::Instant::now();
        let deadline = started + self.config.total_timeout;
        let mut failures = Vec::new();

        for provider in &self.config.providers {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                failures.push(ProviderError::deadline(*provider));
                break;
            }
            let timeout = remaining.min(self.config.provider_timeout);
            let attempt_started = tokio::time::Instant::now();
            match self.fetch_provider(*provider, &identity, timeout).await {
                Ok(post) => {
                    if tokio::time::Instant::now() >= deadline {
                        tracing::debug!(
                            target: "tools.twitter",
                            provider = provider.as_str(),
                            status_id = %identity.id,
                            elapsed_ms = attempt_started.elapsed().as_millis() as u64,
                            result = "deadline_exhausted",
                            "Twitter provider attempt"
                        );
                        failures.push(ProviderError::deadline(*provider));
                        break;
                    }
                    let (media_images, media_videos) = post.media.iter().fold(
                        (0_u64, 0_u64),
                        |(images, videos), media| match media {
                            model::XMedia::Image { .. } => (images + 1, videos),
                            model::XMedia::Video { .. } => (images, videos + 1),
                        },
                    );
                    tracing::debug!(
                        target: "tools.twitter",
                        provider = provider.as_str(),
                        status_id = %identity.id,
                        elapsed_ms = attempt_started.elapsed().as_millis() as u64,
                        result = "success",
                        http_status = 200_u16,
                        media_images,
                        media_videos,
                        "Twitter provider attempt"
                    );
                    return model::build_twitter_content(&identity, post);
                }
                Err(error) => {
                    tracing::debug!(
                        target: "tools.twitter",
                        provider = provider.as_str(),
                        status_id = %identity.id,
                        elapsed_ms = attempt_started.elapsed().as_millis() as u64,
                        result = error.kind.as_str(),
                        status = error.status.map(|status| status.as_u16()),
                        "Twitter provider attempt"
                    );
                    let exhausted = tokio::time::Instant::now() >= deadline;
                    failures.push(error);
                    if exhausted {
                        failures.push(ProviderError::deadline(*provider));
                        break;
                    }
                }
            }
        }

        Err(aggregate_provider_failures(&identity, &failures))
    }

    async fn fetch_provider(
        &self,
        provider: TwitterProvider,
        identity: &XStatusIdentity,
        timeout: Duration,
    ) -> std::result::Result<model::XPost, ProviderError> {
        match provider {
            TwitterProvider::FxTwitter => {
                providers::fxtwitter::fetch(self.client, &self.config, identity, timeout).await
            }
            TwitterProvider::VxTwitter => {
                providers::vxtwitter::fetch(self.client, &self.config, identity, timeout).await
            }
            TwitterProvider::Jina => {
                providers::jina::fetch(self.client, &self.config, identity, timeout).await
            }
        }
    }
}

pub async fn extract_twitter_content(url: &str) -> Result<TwitterContent> {
    let config = TwitterFetchConfig::try_from(&*crate::config::CONFIG)?;
    TwitterExtractor::new(get_http_client_no_redirect(), config)
        .fetch(url)
        .await
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::tools::twitter_extractor::providers::{TwitterFetchConfig, TwitterProvider};
    use crate::tools::twitter_extractor::test_support::{response_with_status, TestServer};

    fn test_chain_config(fx: ::url::Url, vx: ::url::Url, jina: ::url::Url) -> TwitterFetchConfig {
        TwitterFetchConfig {
            providers: vec![
                TwitterProvider::FxTwitter,
                TwitterProvider::VxTwitter,
                TwitterProvider::Jina,
            ],
            fxtwitter_api_base: fx,
            vxtwitter_api_base: vx,
            jina_reader_endpoint: jina,
            jina_api_key: None,
            total_timeout: Duration::from_secs(20),
            provider_timeout: Duration::from_secs(5),
            response_max_bytes: 1024 * 1024,
        }
    }

    #[tokio::test]
    async fn extractor_falls_back_fx_to_vx_and_stops_before_jina() {
        let fx = TestServer::single_status("GET", "/i/status/123", 503);
        let vx = TestServer::single_json(
            "GET",
            "/Twitter/status/123",
            include_bytes!("twitter_extractor/fixtures/vxtwitter_photo_quote.json"),
        );
        let jina = TestServer::expect_no_requests();
        let extractor = TwitterExtractor::new(
            get_http_client_no_redirect(),
            test_chain_config(fx.base_url(), vx.base_url(), jina.base_url()),
        );

        let content = extractor
            .fetch("https://x.com/alice/status/123")
            .await
            .unwrap();
        assert!(content.text_content.contains("root text"));
        fx.join().unwrap();
        vx.join().unwrap();
        jina.join().unwrap();
    }

    #[tokio::test]
    async fn extractor_stops_after_fxtwitter_success() {
        let fx = TestServer::single_json(
            "GET",
            "/i/status/123",
            include_bytes!("twitter_extractor/fixtures/fxtwitter_photo_quote.json"),
        );
        let vx = TestServer::expect_no_requests();
        let jina = TestServer::expect_no_requests();
        let extractor = TwitterExtractor::new(
            get_http_client_no_redirect(),
            test_chain_config(fx.base_url(), vx.base_url(), jina.base_url()),
        );

        extractor
            .fetch("https://x.com/alice/status/123")
            .await
            .unwrap();
        fx.join().unwrap();
        vx.join().unwrap();
        jina.join().unwrap();
    }

    #[tokio::test]
    async fn extractor_falls_back_on_invalid_fxtwitter_json() {
        let fx = TestServer::single(response_with_status(200, b"not json".to_vec()));
        let vx = TestServer::single_json(
            "GET",
            "/Twitter/status/123",
            include_bytes!("twitter_extractor/fixtures/vxtwitter_photo_quote.json"),
        );
        let jina = TestServer::expect_no_requests();
        let extractor = TwitterExtractor::new(
            get_http_client_no_redirect(),
            test_chain_config(fx.base_url(), vx.base_url(), jina.base_url()),
        );

        assert!(extractor
            .fetch("https://x.com/alice/status/123")
            .await
            .unwrap()
            .text_content
            .contains("root text"));
        fx.join().unwrap();
        vx.join().unwrap();
        jina.join().unwrap();
    }

    #[tokio::test]
    async fn extractor_falls_back_on_incomplete_fxtwitter_post() {
        let fx = TestServer::single_json("GET", "/i/status/123", br#"{"status":{}}"#);
        let vx = TestServer::single_json(
            "GET",
            "/Twitter/status/123",
            include_bytes!("twitter_extractor/fixtures/vxtwitter_photo_quote.json"),
        );
        let jina = TestServer::expect_no_requests();
        let extractor = TwitterExtractor::new(
            get_http_client_no_redirect(),
            test_chain_config(fx.base_url(), vx.base_url(), jina.base_url()),
        );

        assert!(extractor
            .fetch("https://x.com/alice/status/123")
            .await
            .unwrap()
            .text_content
            .contains("root text"));
        fx.join().unwrap();
        vx.join().unwrap();
        jina.join().unwrap();
    }

    #[tokio::test]
    async fn extractor_falls_back_fx_and_vx_to_jina() {
        let fx = TestServer::single_status("GET", "/i/status/123", 503);
        let vx = TestServer::single_status("GET", "/Twitter/status/123", 503);
        let body = include_bytes!("twitter_extractor/fixtures/jina_photo.txt");
        let jina = TestServer::new(vec![
            crate::tools::twitter_extractor::test_support::ExpectedRequest::new(
                "GET",
                "/https://x.com/i/status/123",
                response_with_status(200, body.to_vec()),
            ),
        ]);
        let extractor = TwitterExtractor::new(
            get_http_client_no_redirect(),
            test_chain_config(fx.base_url(), vx.base_url(), jina.base_url()),
        );

        assert!(extractor
            .fetch("https://x.com/alice/status/123")
            .await
            .unwrap()
            .text_content
            .contains("fixture body"));
        fx.join().unwrap();
        vx.join().unwrap();
        jina.join().unwrap();
    }

    #[tokio::test]
    async fn extractor_with_jina_only_makes_one_request() {
        let fx = TestServer::expect_no_requests();
        let vx = TestServer::expect_no_requests();
        let body = include_bytes!("twitter_extractor/fixtures/jina_photo.txt");
        let jina = TestServer::new(vec![
            crate::tools::twitter_extractor::test_support::ExpectedRequest::new(
                "GET",
                "/https://x.com/i/status/123",
                response_with_status(200, body.to_vec()),
            ),
        ]);
        let mut config = test_chain_config(fx.base_url(), vx.base_url(), jina.base_url());
        config.providers = vec![TwitterProvider::Jina];
        let extractor = TwitterExtractor::new(get_http_client_no_redirect(), config);

        extractor
            .fetch("https://x.com/alice/status/123")
            .await
            .unwrap();
        fx.join().unwrap();
        vx.join().unwrap();
        jina.join().unwrap();
    }

    #[tokio::test]
    async fn extractor_stops_at_total_deadline() {
        // Servers answer far later than the deadline; the elapsed bound below
        // is generous so scheduling jitter under a loaded test run cannot
        // trip it, yet still well under the server delay.
        let server_delay = Duration::from_millis(500);
        let fx = TestServer::single_delayed("GET", "/i/status/123", server_delay, 200, br#"{}"#);
        let vx =
            TestServer::single_delayed("GET", "/Twitter/status/123", server_delay, 200, br#"{}"#);
        let jina = TestServer::expect_no_requests();
        let mut config = test_chain_config(fx.base_url(), vx.base_url(), jina.base_url());
        config.total_timeout = Duration::from_millis(30);
        config.provider_timeout = Duration::from_millis(15);
        let started = tokio::time::Instant::now();
        let error = TwitterExtractor::new(get_http_client_no_redirect(), config)
            .fetch("https://x.com/a/status/123")
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("deadline"));
        assert!(started.elapsed() < Duration::from_millis(400));
        fx.join_allowing_client_disconnect().unwrap();
        vx.join_allowing_client_disconnect().unwrap();
        jina.join().unwrap();
    }

    #[tokio::test]
    async fn extractor_error_does_not_expose_response_body_or_endpoint() {
        let secret_body = b"response body secret".to_vec();
        let fx = TestServer::new(vec![
            crate::tools::twitter_extractor::test_support::ExpectedRequest::new(
                "GET",
                "/i/status/123",
                response_with_status(503, secret_body.clone()),
            ),
        ]);
        let vx = TestServer::new(vec![
            crate::tools::twitter_extractor::test_support::ExpectedRequest::new(
                "GET",
                "/Twitter/status/123",
                response_with_status(503, secret_body),
            ),
        ]);
        let jina = TestServer::new(vec![
            crate::tools::twitter_extractor::test_support::ExpectedRequest::new(
                "GET",
                "/custom-endpoint-secret/https://x.com/i/status/123",
                response_with_status(503, b"jina body secret".to_vec()),
            ),
        ]);
        let mut config = test_chain_config(fx.base_url(), vx.base_url(), jina.base_url());
        config.jina_reader_endpoint = jina.base_url().join("custom-endpoint-secret").unwrap();
        config.jina_api_key = Some("secret-bearer-value".to_string());
        let error = TwitterExtractor::new(get_http_client_no_redirect(), config)
            .fetch("https://x.com/a/status/123")
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("123"));
        assert!(!error.contains("response body secret"));
        assert!(!error.contains("jina body secret"));
        assert!(!error.contains("custom-endpoint-secret"));
        assert!(!error.contains("secret-bearer-value"));
        fx.join().unwrap();
        vx.join().unwrap();
        jina.join().unwrap();
    }

    #[tokio::test]
    async fn extractor_falls_back_after_fxtwitter_timeout() {
        let fx = TestServer::single_delayed(
            "GET",
            "/i/status/123",
            Duration::from_millis(100),
            200,
            br#"{}"#,
        );
        let vx = TestServer::single_json(
            "GET",
            "/Twitter/status/123",
            include_bytes!("twitter_extractor/fixtures/vxtwitter_photo_quote.json"),
        );
        let jina = TestServer::expect_no_requests();
        let mut config = test_chain_config(fx.base_url(), vx.base_url(), jina.base_url());
        config.provider_timeout = Duration::from_millis(15);
        let content = TwitterExtractor::new(get_http_client_no_redirect(), config)
            .fetch("https://x.com/a/status/123")
            .await
            .unwrap();
        assert!(content.text_content.contains("root text"));
        fx.join_allowing_client_disconnect().unwrap();
        vx.join().unwrap();
        jina.join().unwrap();
    }

    #[tokio::test]
    async fn extractor_falls_back_after_fxtwitter_body_too_large() {
        let fx = TestServer::single_json("GET", "/i/status/123", &vec![b'x'; 2_048]);
        let vx = TestServer::single_json(
            "GET",
            "/Twitter/status/123",
            include_bytes!("twitter_extractor/fixtures/vxtwitter_photo_quote.json"),
        );
        let jina = TestServer::expect_no_requests();
        let mut config = test_chain_config(fx.base_url(), vx.base_url(), jina.base_url());
        config.response_max_bytes = 1_024;
        let content = TwitterExtractor::new(get_http_client_no_redirect(), config)
            .fetch("https://x.com/a/status/123")
            .await
            .unwrap();
        assert!(content.text_content.contains("root text"));
        fx.join().unwrap();
        vx.join().unwrap();
        jina.join().unwrap();
    }

    #[tokio::test]
    async fn extractor_falls_back_after_fxtwitter_transport_failure() {
        let fx = TestServer::new(vec![
            crate::tools::twitter_extractor::test_support::ExpectedRequest::new(
                "GET",
                "/i/status/123",
                Vec::new(),
            ),
        ]);
        let vx = TestServer::single_json(
            "GET",
            "/Twitter/status/123",
            include_bytes!("twitter_extractor/fixtures/vxtwitter_photo_quote.json"),
        );
        let jina = TestServer::expect_no_requests();
        let content = TwitterExtractor::new(
            get_http_client_no_redirect(),
            test_chain_config(fx.base_url(), vx.base_url(), jina.base_url()),
        )
        .fetch("https://x.com/a/status/123")
        .await
        .unwrap();
        assert!(content.text_content.contains("root text"));
        fx.join().unwrap();
        vx.join().unwrap();
        jina.join().unwrap();
    }
}
