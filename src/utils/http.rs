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
