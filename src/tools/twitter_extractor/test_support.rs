use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread::{self, JoinHandle};

use url::Url;

use super::url::XStatusIdentity;

/// A status identity fixture: `id` as both the numeric ID and the fake
/// canonical URL's path segment. Shared by every provider's test module
/// instead of each redefining its own copy.
pub(crate) fn identity(id: &str) -> XStatusIdentity {
    XStatusIdentity {
        id: id.to_owned(),
        canonical_url: Url::parse(&format!("https://x.com/i/status/{id}")).unwrap(),
    }
}

pub(crate) struct ExpectedRequest {
    method: Option<String>,
    path: Option<String>,
    headers: Vec<(String, String)>,
    absent_headers: Vec<String>,
    response: Vec<u8>,
    delay: Option<std::time::Duration>,
}

impl ExpectedRequest {
    pub(crate) fn any(response: Vec<u8>) -> Self {
        Self {
            method: None,
            path: None,
            headers: Vec::new(),
            absent_headers: Vec::new(),
            response,
            delay: None,
        }
    }

    pub(crate) fn new(method: &str, path: &str, response: Vec<u8>) -> Self {
        Self {
            method: Some(method.to_ascii_uppercase()),
            path: Some(path.to_string()),
            headers: Vec::new(),
            absent_headers: Vec::new(),
            response,
            delay: None,
        }
    }

    pub(crate) fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers
            .push((name.to_ascii_lowercase(), value.to_string()));
        self
    }

    pub(crate) fn without_header(mut self, name: &str) -> Self {
        self.absent_headers.push(name.to_ascii_lowercase());
        self
    }

    pub(crate) fn delayed(mut self, delay: std::time::Duration) -> Self {
        self.delay = Some(delay);
        self
    }
}

pub(crate) struct TestServer {
    address: std::net::SocketAddr,
    shutdown: Arc<AtomicBool>,
    allow_client_disconnect: Arc<AtomicBool>,
    worker: Option<JoinHandle<Result<(), String>>>,
}

impl TestServer {
    pub(crate) fn single_json(method: &str, path: &str, body: &[u8]) -> Self {
        Self::new(vec![ExpectedRequest::new(
            method,
            path,
            response_with_content_length(body.len(), body.to_vec()),
        )])
    }

    pub(crate) fn single_status(method: &str, path: &str, status: u16) -> Self {
        Self::new(vec![ExpectedRequest::new(
            method,
            path,
            response_with_status(status, Vec::new()),
        )])
    }

    pub(crate) fn single_delayed(
        method: &str,
        path: &str,
        delay: std::time::Duration,
        status: u16,
        body: &[u8],
    ) -> Self {
        Self::new(vec![ExpectedRequest::new(
            method,
            path,
            response_with_status(status, body.to_vec()),
        )
        .delayed(delay)])
    }

    pub(crate) fn base_url(&self) -> Url {
        Url::parse(&format!("http://{}", self.address)).expect("test server base URL")
    }

    pub(crate) fn single(response: Vec<u8>) -> Self {
        Self::new(vec![ExpectedRequest::any(response)])
    }

    pub(crate) fn expect_no_requests() -> Self {
        Self::new(Vec::new())
    }

    pub(crate) fn new(expected: Vec<ExpectedRequest>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let address = listener.local_addr().expect("test server address");
        let shutdown = Arc::new(AtomicBool::new(false));
        let worker_shutdown = Arc::clone(&shutdown);
        let allow_client_disconnect = Arc::new(AtomicBool::new(false));
        let worker_allow_client_disconnect = Arc::clone(&allow_client_disconnect);
        let worker = thread::spawn(move || {
            serve(
                listener,
                expected,
                worker_shutdown,
                worker_allow_client_disconnect,
            )
        });
        Self {
            address,
            shutdown,
            allow_client_disconnect,
            worker: Some(worker),
        }
    }

    pub(crate) fn url(&self, path: &str) -> Url {
        Url::parse(&format!("http://{}{}", self.address, path)).expect("test server URL")
    }

    pub(crate) fn join(mut self) -> Result<(), String> {
        self.join_inner(false)
    }

    pub(crate) fn join_allowing_client_disconnect(mut self) -> Result<(), String> {
        self.join_inner(true)
    }

    fn join_inner(&mut self, allow_client_disconnect: bool) -> Result<(), String> {
        self.allow_client_disconnect
            .store(allow_client_disconnect, Ordering::Release);
        self.shutdown.store(true, Ordering::Release);
        if let Ok(mut stream) = TcpStream::connect(self.address) {
            let _ = stream.write_all(
                b"GET /__test_server_shutdown__ HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            );
        }
        let worker = self.worker.take().expect("test server already joined");
        let (sender, receiver) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let _ = sender.send(worker.join());
        });
        match receiver.recv_timeout(std::time::Duration::from_secs(2)) {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err("test server worker panicked".to_string()),
            Err(_) => Err("test server join timed out".to_string()),
        }
    }
}

fn serve(
    listener: TcpListener,
    expected: Vec<ExpectedRequest>,
    shutdown: Arc<AtomicBool>,
    allow_client_disconnect: Arc<AtomicBool>,
) -> Result<(), String> {
    let mut expected = VecDeque::from(expected);
    let mut first_error: Option<ServerError> = None;
    loop {
        let (mut stream, _) = listener.accept().map_err(|error| error.to_string())?;
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(100)));
        let request = read_request(&mut stream);
        if shutdown.load(Ordering::Acquire)
            && request
                .as_ref()
                .is_ok_and(|request| request.path == "/__test_server_shutdown__")
        {
            break;
        }

        let expectation = expected.pop_front();
        match expectation {
            Some(expectation) => {
                match request {
                    Ok(request) => {
                        if let Err(error) = expectation.verify(&request) {
                            record_error(&mut first_error, ServerError::Assertion(error));
                        }
                    }
                    Err(error) => {
                        let error = match error {
                            RequestError::ClientDisconnect(error) => {
                                ServerError::ClientDisconnect(error)
                            }
                            RequestError::Assertion(error) => ServerError::Assertion(error),
                        };
                        record_error(&mut first_error, error);
                    }
                }
                if let Some(delay) = expectation.delay {
                    thread::sleep(delay);
                }
                if let Err(error) = stream
                    .write_all(&expectation.response)
                    .and_then(|_| stream.flush())
                {
                    if expectation.delay.is_some() && is_client_disconnect_kind(error.kind()) {
                        record_error(
                            &mut first_error,
                            ServerError::ClientDisconnect(error.to_string()),
                        );
                    } else {
                        record_error(&mut first_error, ServerError::Assertion(error.to_string()));
                    }
                }
            }
            None => {
                let error = request
                    .map(|request| {
                        format!(
                            "unexpected extra request: {} {}",
                            request.method, request.path
                        )
                    })
                    .unwrap_or_else(|error| format!("unexpected extra request: {error}"));
                record_error(&mut first_error, ServerError::Assertion(error));
                let _ = stream.write_all(
                    b"HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\nContent-Length: 0\r\n\r\n",
                );
            }
        }
    }

    if !expected.is_empty() {
        // When the client is allowed to give up (deadline tests), it may do so
        // before the request ever reaches this server; that is not a failure.
        if allow_client_disconnect.load(Ordering::Acquire) {
            return Ok(());
        }
        return Err(format!("unmet expectations: {}", expected.len()));
    }
    first_error.map_or(Ok(()), |error| {
        if allow_client_disconnect.load(Ordering::Acquire) && error.is_client_disconnect() {
            Ok(())
        } else {
            Err(error.to_string())
        }
    })
}

enum ServerError {
    Assertion(String),
    ClientDisconnect(String),
}

fn record_error(first_error: &mut Option<ServerError>, error: ServerError) {
    if first_error
        .as_ref()
        .is_none_or(|existing| existing.is_client_disconnect() && !error.is_client_disconnect())
    {
        *first_error = Some(error);
    }
}

impl ServerError {
    fn is_client_disconnect(&self) -> bool {
        matches!(self, Self::ClientDisconnect(_))
    }
}

impl std::fmt::Display for ServerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Assertion(error) => formatter.write_str(error),
            Self::ClientDisconnect(error) => write!(formatter, "client disconnected: {error}"),
        }
    }
}

struct Request {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
}

impl ExpectedRequest {
    fn verify(&self, request: &Request) -> Result<(), String> {
        if self
            .method
            .as_deref()
            .is_some_and(|method| method != request.method)
        {
            return Err(format!(
                "expected method {:?}, got {:?}",
                self.method, request.method
            ));
        }
        if self
            .path
            .as_deref()
            .is_some_and(|path| path != request.path)
        {
            return Err(format!(
                "expected path {:?}, got {:?}",
                self.path, request.path
            ));
        }
        for (name, value) in &self.headers {
            if !request
                .headers
                .iter()
                .any(|(actual_name, actual_value)| actual_name == name && actual_value == value)
            {
                return Err(format!("missing expected header {name}"));
            }
        }
        for name in &self.absent_headers {
            if request
                .headers
                .iter()
                .any(|(actual_name, _)| actual_name == name)
            {
                return Err(format!("unexpected header {name}"));
            }
        }
        Ok(())
    }
}

enum RequestError {
    Assertion(String),
    ClientDisconnect(String),
}

pub(crate) fn is_client_disconnect_kind(kind: std::io::ErrorKind) -> bool {
    // `TimedOut` / `WouldBlock` cover the socket read timeout that fires when
    // the client hit its own deadline and never sent (or finished) the request.
    matches!(
        kind,
        std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::BrokenPipe
            | std::io::ErrorKind::TimedOut
            | std::io::ErrorKind::WouldBlock
    )
}

impl std::fmt::Display for RequestError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Assertion(error) | Self::ClientDisconnect(error) => formatter.write_str(error),
        }
    }
}

fn read_request(stream: &mut TcpStream) -> Result<Request, RequestError> {
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 4096];
    while !bytes.windows(4).any(|window| window == b"\r\n\r\n") {
        if bytes.len() >= 64 * 1024 {
            return Err(RequestError::Assertion(
                "request headers exceed test limit".to_string(),
            ));
        }
        let read = stream.read(&mut chunk).map_err(|error| {
            if is_client_disconnect_kind(error.kind()) {
                RequestError::ClientDisconnect(error.to_string())
            } else {
                RequestError::Assertion(error.to_string())
            }
        })?;
        if read == 0 {
            return Err(RequestError::ClientDisconnect(
                "request ended before headers".to_string(),
            ));
        }
        bytes.extend_from_slice(&chunk[..read]);
    }

    let text = std::str::from_utf8(&bytes)
        .map_err(|_| RequestError::Assertion("request headers are not UTF-8".to_string()))?;
    let mut lines = text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| RequestError::Assertion("missing request line".to_string()))?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| RequestError::Assertion("missing request method".to_string()))?
        .to_ascii_uppercase();
    let path = request_parts
        .next()
        .ok_or_else(|| RequestError::Assertion("missing request path".to_string()))?
        .to_string();
    let mut headers = Vec::new();
    for line in lines {
        if line.is_empty() {
            break;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| RequestError::Assertion("malformed request header".to_string()))?;
        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
    }
    Ok(Request {
        method,
        path,
        headers,
    })
}

pub(crate) fn response_with_content_length(declared_length: usize, body: Vec<u8>) -> Vec<u8> {
    let mut response = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {declared_length}\r\nConnection: close\r\n\r\n"
    )
    .into_bytes();
    response.extend_from_slice(&body);
    response
}

/// Raw HTTP/1.1 response with arbitrary extra headers (e.g. `Retry-After`).
pub(crate) fn response_with_headers(
    status: u16,
    headers: &[(&str, &str)],
    body: Vec<u8>,
) -> Vec<u8> {
    let mut response = format!(
        "HTTP/1.1 {status} Test\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (name, value) in headers {
        response.push_str(&format!("{name}: {value}\r\n"));
    }
    response.push_str("\r\n");
    let mut bytes = response.into_bytes();
    bytes.extend_from_slice(&body);
    bytes
}

pub(crate) fn response_with_status(status: u16, body: Vec<u8>) -> Vec<u8> {
    let reason = match status {
        200 => "OK",
        302 => "Found",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Test Status",
    };
    let mut response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    response.extend_from_slice(&body);
    response
}

pub(crate) fn redirect_response(location: &str) -> Vec<u8> {
    format!("HTTP/1.1 302 Found\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").into_bytes()
}

pub(crate) fn chunked_response(chunks: Vec<Vec<u8>>) -> Vec<u8> {
    let mut response =
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n".to_vec();
    for chunk in chunks {
        response.extend_from_slice(format!("{:X}\r\n", chunk.len()).as_bytes());
        response.extend_from_slice(&chunk);
        response.extend_from_slice(b"\r\n");
    }
    response.extend_from_slice(b"0\r\n\r\n");
    response
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;

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

    #[tokio::test]
    async fn delayed_join_still_surfaces_request_mismatch() {
        let server = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/expected",
            response_with_content_length(0, Vec::new()),
        )
        .delayed(Duration::from_millis(1))]);
        reqwest::Client::new()
            .get(server.url("/wrong"))
            .send()
            .await
            .unwrap();
        let error = server.join_allowing_client_disconnect().unwrap_err();
        assert!(error.contains("expected path"));
    }

    #[test]
    fn test_server_client_disconnect_requires_explicit_allowance() {
        let allowing = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/delayed",
            response_with_content_length(0, Vec::new()),
        )
        .delayed(Duration::from_millis(1))]);
        let allowing_address = format!(
            "127.0.0.1:{}",
            allowing.base_url().port().expect("test server port")
        );
        let stream = std::net::TcpStream::connect(allowing_address).unwrap();
        drop(stream);
        allowing.join_allowing_client_disconnect().unwrap();

        let strict = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/delayed",
            response_with_content_length(0, Vec::new()),
        )
        .delayed(Duration::from_millis(1))]);
        let strict_address = format!(
            "127.0.0.1:{}",
            strict.base_url().port().expect("test server port")
        );
        let stream = std::net::TcpStream::connect(strict_address).unwrap();
        drop(stream);
        let error = strict.join().unwrap_err();
        assert!(error.contains("client disconnected"));
    }

    #[test]
    fn test_server_non_delayed_client_disconnect_requires_explicit_allowance() {
        let allowing = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/immediate",
            response_with_content_length(0, Vec::new()),
        )]);
        let allowing_address = format!(
            "127.0.0.1:{}",
            allowing.base_url().port().expect("test server port")
        );
        drop(std::net::TcpStream::connect(allowing_address).unwrap());
        allowing.join_allowing_client_disconnect().unwrap();

        let strict = TestServer::new(vec![ExpectedRequest::new(
            "GET",
            "/immediate",
            response_with_content_length(0, Vec::new()),
        )]);
        let strict_address = format!(
            "127.0.0.1:{}",
            strict.base_url().port().expect("test server port")
        );
        drop(std::net::TcpStream::connect(strict_address).unwrap());
        let error = strict.join().unwrap_err();
        assert!(error.contains("client disconnected"));
    }

    #[test]
    fn client_disconnect_error_kinds_are_typed_and_narrow() {
        assert!(is_client_disconnect_kind(
            std::io::ErrorKind::ConnectionReset
        ));
        assert!(is_client_disconnect_kind(
            std::io::ErrorKind::ConnectionAborted
        ));
        assert!(is_client_disconnect_kind(std::io::ErrorKind::BrokenPipe));
        assert!(!is_client_disconnect_kind(std::io::ErrorKind::Other));
    }
}
