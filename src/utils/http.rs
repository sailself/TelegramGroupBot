use once_cell::sync::Lazy;
use reqwest::Client;
use std::time::Duration;

// Send TCP keepalive probes so long-lived (especially streaming SSE) connections
// that go idle while a model reasons are kept warm and dead peers are detected,
// reducing intermediary idle-connection drops that surface as body-decode errors.
const TCP_KEEPALIVE: Duration = Duration::from_secs(30);

static HTTP_CLIENT: Lazy<Client> = Lazy::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(30))
        .tcp_keepalive(TCP_KEEPALIVE)
        .build()
        .expect("Failed to build HTTP client")
});

static HTTP_CLIENT_NO_COMPRESSION: Lazy<Client> = Lazy::new(|| {
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

pub fn get_http_client() -> &'static Client {
    &HTTP_CLIENT
}

pub fn get_http_client_no_compression() -> &'static Client {
    &HTTP_CLIENT_NO_COMPRESSION
}
