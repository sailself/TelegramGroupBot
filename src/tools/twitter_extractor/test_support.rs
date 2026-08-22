#![allow(dead_code)]

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread::{self, JoinHandle};

use url::Url;

pub(crate) struct ExpectedRequest {
    method: Option<String>,
    path: Option<String>,
    headers: Vec<(String, String)>,
    response: Vec<u8>,
}

impl ExpectedRequest {
    pub(crate) fn any(response: Vec<u8>) -> Self {
        Self {
            method: None,
            path: None,
            headers: Vec::new(),
            response,
        }
    }

    pub(crate) fn new(method: &str, path: &str, response: Vec<u8>) -> Self {
        Self {
            method: Some(method.to_ascii_uppercase()),
            path: Some(path.to_string()),
            headers: Vec::new(),
            response,
        }
    }

    pub(crate) fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers
            .push((name.to_ascii_lowercase(), value.to_string()));
        self
    }
}

pub(crate) struct TestServer {
    address: std::net::SocketAddr,
    shutdown: Arc<AtomicBool>,
    worker: Option<JoinHandle<Result<(), String>>>,
}

impl TestServer {
    pub(crate) fn single(response: Vec<u8>) -> Self {
        Self::new(vec![ExpectedRequest::any(response)])
    }

    pub(crate) fn new(expected: Vec<ExpectedRequest>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let address = listener.local_addr().expect("test server address");
        let shutdown = Arc::new(AtomicBool::new(false));
        let worker_shutdown = Arc::clone(&shutdown);
        let worker = thread::spawn(move || serve(listener, expected, worker_shutdown));
        Self {
            address,
            shutdown,
            worker: Some(worker),
        }
    }

    pub(crate) fn url(&self, path: &str) -> Url {
        Url::parse(&format!("http://{}{}", self.address, path)).expect("test server URL")
    }

    pub(crate) fn join(mut self) -> Result<(), String> {
        self.shutdown.store(true, Ordering::Release);
        let _ = TcpStream::connect(self.address);
        let worker = self.worker.take().expect("test server already joined");
        match worker.join() {
            Ok(result) => result,
            Err(_) => Err("test server worker panicked".to_string()),
        }
    }
}

fn serve(
    listener: TcpListener,
    expected: Vec<ExpectedRequest>,
    shutdown: Arc<AtomicBool>,
) -> Result<(), String> {
    let mut expected = VecDeque::from(expected);
    let mut first_error = None;
    loop {
        let (mut stream, _) = listener.accept().map_err(|error| error.to_string())?;
        if shutdown.load(Ordering::Acquire) {
            break;
        }

        let expectation = expected.pop_front();
        match expectation {
            Some(expectation) => {
                let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(100)));
                match read_request(&mut stream).and_then(|request| expectation.verify(&request)) {
                    Ok(()) => {}
                    Err(error) => {
                        first_error.get_or_insert(error);
                    }
                }
                if let Err(error) = stream
                    .write_all(&expectation.response)
                    .and_then(|_| stream.flush())
                    .map_err(|error| error.to_string())
                {
                    first_error.get_or_insert(error);
                }
            }
            None => {
                let error = read_request(&mut stream)
                    .map(|request| {
                        format!(
                            "unexpected extra request: {} {}",
                            request.method, request.path
                        )
                    })
                    .unwrap_or_else(|error| format!("unexpected extra request: {error}"));
                first_error.get_or_insert(error);
                let _ = stream.write_all(
                    b"HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\nContent-Length: 0\r\n\r\n",
                );
            }
        }
    }

    if !expected.is_empty() {
        return Err(format!("unmet expectations: {}", expected.len()));
    }
    first_error.map_or(Ok(()), Err)
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
        Ok(())
    }
}

fn read_request(stream: &mut TcpStream) -> Result<Request, String> {
    let mut bytes = Vec::new();
    let mut chunk = [0_u8; 4096];
    while !bytes.windows(4).any(|window| window == b"\r\n\r\n") {
        if bytes.len() >= 64 * 1024 {
            return Err("request headers exceed test limit".to_string());
        }
        let read = stream.read(&mut chunk).map_err(|error| error.to_string())?;
        if read == 0 {
            return Err("request ended before headers".to_string());
        }
        bytes.extend_from_slice(&chunk[..read]);
    }

    let text =
        std::str::from_utf8(&bytes).map_err(|_| "request headers are not UTF-8".to_string())?;
    let mut lines = text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| "missing request line".to_string())?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| "missing request method".to_string())?
        .to_ascii_uppercase();
    let path = request_parts
        .next()
        .ok_or_else(|| "missing request path".to_string())?
        .to_string();
    let mut headers = Vec::new();
    for line in lines {
        if line.is_empty() {
            break;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| "malformed request header".to_string())?;
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
