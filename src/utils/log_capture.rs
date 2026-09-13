//! Test-only helper that runs code under a JSON `tracing` layer and returns
//! the emitted events, so tests can assert on structured fields.

use std::io;
use std::sync::{Arc, Mutex};

use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;

#[derive(Clone, Default)]
struct CapturedLog(
    Arc<Mutex<Vec<u8>>>,
    Option<(&'static str, Arc<tokio::sync::Notify>)>,
);

impl io::Write for CapturedLog {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut bytes = self.0.lock().unwrap();
        bytes.extend_from_slice(buf);
        if let Some((needle, notify)) = &self.1 {
            if bytes
                .windows(needle.len())
                .any(|window| window == needle.as_bytes())
            {
                notify.notify_one();
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for CapturedLog {
    type Writer = CapturedLog;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// Run `emit` under a JSON fmt layer and return every event it wrote, in
/// order, parsed as JSON objects.
pub fn capture_json_events(emit: impl FnOnce()) -> Vec<serde_json::Value> {
    let log = CapturedLog::default();
    let subscriber = tracing_subscriber::registry().with(
        tracing_subscriber::fmt::layer()
            .json()
            .with_writer(log.clone()),
    );
    tracing::subscriber::with_default(subscriber, emit);
    let bytes = log.0.lock().unwrap().clone();
    String::from_utf8(bytes)
        .expect("json layer output is utf-8")
        .lines()
        .map(|line| serde_json::from_str(line).expect("each log line is a JSON object"))
        .collect()
}

/// Attach the capture subscriber to every poll of an asynchronous operation.
pub async fn capture_json_events_async<T>(
    work: impl std::future::Future<Output = T>,
) -> (T, Vec<serde_json::Value>) {
    capture_async(work, CapturedLog::default()).await
}

async fn capture_async<T>(
    work: impl std::future::Future<Output = T>,
    log: CapturedLog,
) -> (T, Vec<serde_json::Value>) {
    use tracing::instrument::WithSubscriber;
    let subscriber = tracing_subscriber::registry().with(
        tracing_subscriber::fmt::layer()
            .json()
            .with_writer(log.clone()),
    );
    let result = work.with_subscriber(subscriber).await;
    let bytes = log.0.lock().unwrap().clone();
    let events = String::from_utf8(bytes)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    (result, events)
}

pub async fn capture_json_events_on<T>(
    work: impl std::future::Future<Output = T>,
    needle: &'static str,
    notify: Arc<tokio::sync::Notify>,
) -> (T, Vec<serde_json::Value>) {
    capture_async(
        work,
        CapturedLog(Default::default(), Some((needle, notify))),
    )
    .await
}
