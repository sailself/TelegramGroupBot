//! Test-only helper that runs code under a JSON `tracing` layer and returns
//! the emitted events, so tests can assert on structured fields.

use std::io;
use std::sync::{Arc, Mutex};

use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;

#[derive(Clone, Default)]
struct CapturedLog(Arc<Mutex<Vec<u8>>>);

impl io::Write for CapturedLog {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
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
