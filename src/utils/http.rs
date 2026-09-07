use reqwest::Client;
use std::ops::Deref;
use std::sync::LazyLock;
use std::time::Duration;

// Send TCP keepalive probes so long-lived (especially streaming SSE) connections
// that go idle while a model reasons are kept warm and dead peers are detected,
// reducing intermediary idle-connection drops that surface as body-decode errors.
const TCP_KEEPALIVE: Duration = Duration::from_secs(30);

static HTTP_CLIENT: LazyLock<Client> = LazyLock::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .tcp_keepalive(TCP_KEEPALIVE)
        .build()
        .expect("Failed to build HTTP client")
});

static HTTP_CLIENT_NO_COMPRESSION: LazyLock<Client> = LazyLock::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .tcp_keepalive(TCP_KEEPALIVE)
        .no_gzip()
        .no_brotli()
        .no_deflate()
        .no_zstd()
        .build()
        .expect("Failed to build HTTP client without compression")
});

pub struct NoRedirectClient(Client);

impl NoRedirectClient {
    fn build() -> Self {
        Self(
            Client::builder()
                .timeout(Duration::from_secs(30))
                .tcp_keepalive(TCP_KEEPALIVE)
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .expect("Failed to build no-redirect HTTP client"),
        )
    }
}

impl Deref for NoRedirectClient {
    type Target = Client;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

static NO_REDIRECT_CLIENT: LazyLock<NoRedirectClient> = LazyLock::new(NoRedirectClient::build);

pub fn get_http_client() -> &'static Client {
    &HTTP_CLIENT
}

pub fn get_http_client_no_compression() -> &'static Client {
    &HTTP_CLIENT_NO_COMPRESSION
}

pub fn get_http_client_no_redirect() -> &'static NoRedirectClient {
    &NO_REDIRECT_CLIENT
}

/// Parse `value` as an absolute HTTPS URL with a host, no userinfo, no query,
/// no fragment, default port only. `allowed_hosts` = None accepts any host;
/// Some(list) requires an exact (case-insensitive) host match.
/// `name` is only used in the error message.
pub fn parse_https_allowlisted(
    name: &str,
    value: &str,
    allowed_hosts: Option<&[&str]>,
) -> anyhow::Result<url::Url> {
    parse_https_allowlisted_inner(name, value, allowed_hosts, false)
}

/// Same as [`parse_https_allowlisted`], but permits a query string. Some
/// allowlisted CDNs attach one to every real URL (Twitter's media hosts
/// serve `?format=jpg&name=orig`-style query strings), so a media-URL
/// validator needs this instead of the stricter default.
pub fn parse_https_allowlisted_with_query(
    name: &str,
    value: &str,
    allowed_hosts: Option<&[&str]>,
) -> anyhow::Result<url::Url> {
    parse_https_allowlisted_inner(name, value, allowed_hosts, true)
}

fn parse_https_allowlisted_inner(
    name: &str,
    value: &str,
    allowed_hosts: Option<&[&str]>,
    allow_query: bool,
) -> anyhow::Result<url::Url> {
    let parsed = url::Url::parse(value.trim())
        .map_err(|err| anyhow::anyhow!("{name} must be a valid HTTPS URL: {err}"))?;
    if parsed.scheme() != "https" {
        return Err(anyhow::anyhow!("{name} must use HTTPS"));
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("{name} must contain a host"))?
        .to_string();
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(anyhow::anyhow!("{name} must not contain credentials"));
    }
    if (!allow_query && parsed.query().is_some()) || parsed.fragment().is_some() {
        return Err(anyhow::anyhow!(
            "{name} must not contain a query or fragment"
        ));
    }
    if parsed.port().is_some() && parsed.port_or_known_default() != Some(443) {
        return Err(anyhow::anyhow!("{name} must not use a non-default port"));
    }
    if let Some(hosts) = allowed_hosts {
        if !hosts
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(&host))
        {
            return Err(anyhow::anyhow!("{name} host is not allowlisted"));
        }
    }
    Ok(parsed)
}

#[derive(Debug, thiserror::Error)]
pub enum BodyCapError {
    #[error("{name}: declared body of {declared} bytes exceeds the {max}-byte limit")]
    DeclaredTooLarge {
        name: &'static str,
        declared: u64,
        max: usize,
    },
    #[error("{name}: body exceeded the {max}-byte limit while streaming")]
    StreamedTooLarge { name: &'static str, max: usize },
    #[error("{name}: {source}")]
    Read {
        name: &'static str,
        #[source]
        source: reqwest::Error,
    },
}

/// Read at most `max` bytes: reject up front when Content-Length exceeds `max`,
/// otherwise stream chunks and stop at the first byte over the limit.
pub async fn read_body_capped(
    name: &'static str,
    mut response: reqwest::Response,
    max: usize,
) -> Result<bytes::Bytes, BodyCapError> {
    if let Some(declared) = response.content_length() {
        if declared > max as u64 {
            return Err(BodyCapError::DeclaredTooLarge {
                name,
                declared,
                max,
            });
        }
    }

    let mut body = bytes::BytesMut::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|source| BodyCapError::Read { name, source })?
    {
        if body.len().saturating_add(chunk.len()) > max {
            return Err(BodyCapError::StreamedTooLarge { name, max });
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body.freeze())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::twitter_extractor::test_support::{
        chunked_response, response_with_content_length, TestServer,
    };

    #[test]
    fn https_allowlist_accepts_default_port_and_rejects_everything_else() {
        let ok = parse_https_allowlisted("x", "https://api.fxtwitter.com/", None).unwrap();
        assert_eq!(ok.host_str(), Some("api.fxtwitter.com"));
        for bad in [
            "http://api.fxtwitter.com/",
            "https://user:pw@api.fxtwitter.com/",
            "https://api.fxtwitter.com:8443/",
            "https://api.fxtwitter.com/?q=1",
            "https://api.fxtwitter.com/#frag",
            "not a url",
            "",
        ] {
            assert!(parse_https_allowlisted("x", bad, None).is_err(), "{bad}");
        }
    }

    #[test]
    fn https_allowlist_with_query_accepts_a_query_but_still_rejects_fragment_and_bad_host() {
        let hosts = ["pbs.twimg.com", "video.twimg.com"];
        let ok = parse_https_allowlisted_with_query(
            "m",
            "https://pbs.twimg.com/media/a.jpg?format=jpg&name=orig",
            Some(&hosts),
        )
        .unwrap();
        assert_eq!(ok.query(), Some("format=jpg&name=orig"));
        assert!(parse_https_allowlisted_with_query(
            "m",
            "https://pbs.twimg.com/media/a.jpg#frag",
            Some(&hosts)
        )
        .is_err());
        assert!(parse_https_allowlisted_with_query(
            "m",
            "https://evil.pbs.twimg.com/media/a.jpg?format=jpg",
            Some(&hosts)
        )
        .is_err());
    }

    #[test]
    fn https_allowlist_matches_hosts_case_insensitively_and_exactly() {
        let hosts = ["pbs.twimg.com", "video.twimg.com"];
        assert!(parse_https_allowlisted("m", "https://PBS.twimg.com/a.jpg", Some(&hosts)).is_ok());
        assert!(
            parse_https_allowlisted("m", "https://evil.pbs.twimg.com/a.jpg", Some(&hosts)).is_err()
        );
        assert!(parse_https_allowlisted("m", "https://twimg.com/a.jpg", Some(&hosts)).is_err());
    }

    #[tokio::test]
    async fn read_body_capped_rejects_declared_and_streamed_overflow() {
        let declared = TestServer::single(response_with_content_length(11, vec![b'x'; 11]));
        let response = reqwest::Client::new()
            .get(declared.url("/"))
            .send()
            .await
            .unwrap();
        match read_body_capped("test", response, 10).await.unwrap_err() {
            BodyCapError::DeclaredTooLarge { declared, max, .. } => {
                assert_eq!(declared, 11);
                assert_eq!(max, 10);
            }
            other => panic!("expected DeclaredTooLarge, got {other:?}"),
        }
        declared.join().unwrap();

        let chunked = TestServer::single(chunked_response(vec![vec![b'a'; 10], vec![b'b'; 10]]));
        let response = reqwest::Client::new()
            .get(chunked.url("/"))
            .send()
            .await
            .unwrap();
        match read_body_capped("test", response, 10).await.unwrap_err() {
            BodyCapError::StreamedTooLarge { max, .. } => assert_eq!(max, 10),
            other => panic!("expected StreamedTooLarge, got {other:?}"),
        }
        chunked.join().unwrap();

        let ok_server = TestServer::single(response_with_content_length(10, vec![b'c'; 10]));
        let response = reqwest::Client::new()
            .get(ok_server.url("/"))
            .send()
            .await
            .unwrap();
        let body = read_body_capped("test", response, 10).await.unwrap();
        assert_eq!(body.len(), 10);
        assert_eq!(body.as_ref(), &vec![b'c'; 10][..]);
        ok_server.join().unwrap();
    }
}
