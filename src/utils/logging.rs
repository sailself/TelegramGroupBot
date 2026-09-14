use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::filter::{LevelFilter, Targets};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

use crate::config::CONFIG;

pub struct LoggingGuards {
    _file_guard: WorkerGuard,
    _timing_guard: WorkerGuard,
    _json_file_guard: WorkerGuard,
    _json_timing_guard: WorkerGuard,
}

#[derive(Debug, Clone)]
pub struct LogTail {
    pub path: PathBuf,
    pub lines: Vec<String>,
}

fn parse_log_level(value: &str) -> LevelFilter {
    match value.trim().to_lowercase().as_str() {
        "trace" => LevelFilter::TRACE,
        "debug" => LevelFilter::DEBUG,
        "info" => LevelFilter::INFO,
        "warn" | "warning" => LevelFilter::WARN,
        "error" => LevelFilter::ERROR,
        "off" => LevelFilter::OFF,
        _ => LevelFilter::INFO,
    }
}

pub fn init_logging() -> LoggingGuards {
    let logs_dir = Path::new("logs");
    if let Err(err) = fs::create_dir_all(logs_dir) {
        eprintln!("Failed to create logs directory: {err}");
    }

    let file_appender = tracing_appender::rolling::daily(logs_dir, "bot.log");
    let (file_writer, file_guard) = tracing_appender::non_blocking(file_appender);

    let timing_appender = tracing_appender::rolling::daily(logs_dir, "timing.log");
    let (timing_writer, timing_guard) = tracing_appender::non_blocking(timing_appender);

    let json_file_appender = tracing_appender::rolling::daily(logs_dir, "bot.jsonl");
    let (json_file_writer, json_file_guard) = tracing_appender::non_blocking(json_file_appender);

    let json_timing_appender = tracing_appender::rolling::daily(logs_dir, "timing.jsonl");
    let (json_timing_writer, json_timing_guard) =
        tracing_appender::non_blocking(json_timing_appender);

    let general_level = parse_log_level(&CONFIG.log_level);
    let general_filter = Targets::new()
        .with_default(general_level)
        .with_target("bot.timing", LevelFilter::OFF)
        .with_target("hyper", LevelFilter::WARN)
        .with_target("hyper_util", LevelFilter::WARN)
        .with_target("hyper_util::client::legacy::pool", LevelFilter::WARN)
        .with_target("reqwest", LevelFilter::WARN);
    let timing_filter = Targets::new()
        .with_default(LevelFilter::OFF)
        .with_target("bot.timing", LevelFilter::INFO);

    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(file_writer)
        .with_ansi(false)
        .with_filter(general_filter.clone());
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_filter(general_filter.clone());
    let timing_layer = tracing_subscriber::fmt::layer()
        .with_writer(timing_writer)
        .with_ansi(false)
        .with_filter(timing_filter.clone());
    let json_file_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(json_file_writer)
        .with_filter(general_filter);
    let json_timing_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(json_timing_writer)
        .with_filter(timing_filter);

    tracing_subscriber::registry()
        .with(file_layer)
        .with(stdout_layer)
        .with(timing_layer)
        .with(json_file_layer)
        .with(json_timing_layer)
        .init();

    LoggingGuards {
        _file_guard: file_guard,
        _timing_guard: timing_guard,
        _json_file_guard: json_file_guard,
        _json_timing_guard: json_timing_guard,
    }
}

pub fn read_recent_log_lines(base_name: &str, max_lines: usize) -> io::Result<Option<LogTail>> {
    if max_lines == 0 {
        return Ok(None);
    }

    let Some(path) = find_latest_log_file(base_name)? else {
        return Ok(None);
    };

    let lines = tail_file_lines(&path, max_lines)?;
    Ok(Some(LogTail { path, lines }))
}

fn find_latest_log_file(base_name: &str) -> io::Result<Option<PathBuf>> {
    let logs_dir = Path::new("logs");
    if !logs_dir.exists() {
        return Ok(None);
    }

    let mut newest: Option<(SystemTime, PathBuf)> = None;
    for entry in fs::read_dir(logs_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with(base_name) {
            continue;
        }

        let modified = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);

        match &newest {
            Some((current_time, _)) if modified <= *current_time => {}
            _ => newest = Some((modified, path)),
        }
    }

    Ok(newest.map(|(_, path)| path))
}

fn tail_file_lines(path: &Path, max_lines: usize) -> io::Result<Vec<String>> {
    if max_lines == 0 {
        return Ok(Vec::new());
    }
    const MAX_SCAN: usize = 256 * 1024;
    let mut file = File::open(path)?;
    let mut position = file.metadata()?.len();
    let mut blocks = Vec::new();
    let mut scanned = 0;
    let mut newlines = 0;
    while position > 0 && scanned < MAX_SCAN && newlines <= max_lines {
        let count = (position.min(8192) as usize).min(MAX_SCAN - scanned);
        position -= count as u64;
        file.seek(SeekFrom::Start(position))?;
        let mut block = vec![0; count];
        file.read_exact(&mut block)?;
        newlines += block.iter().filter(|b| **b == b'\n').count();
        scanned += count;
        blocks.push(block);
    }
    let bytes: Vec<u8> = blocks.into_iter().rev().flatten().collect();
    let capped = position > 0 && scanned == MAX_SCAN && newlines <= max_lines;
    let start = if position > 0 {
        bytes
            .iter()
            .position(|b| *b == b'\n')
            .map(|i| i + 1)
            .unwrap_or_else(|| {
                bytes
                    .iter()
                    .position(|b| b & 0xc0 != 0x80)
                    .unwrap_or(bytes.len())
            })
    } else {
        0
    };
    let text = String::from_utf8_lossy(&bytes[start..]);
    let mut lines = text
        .lines()
        .rev()
        .take(max_lines)
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    lines.reverse();
    if capped {
        lines.insert(
            0,
            "[log tail truncated: scanned at most 256 KiB]".to_string(),
        );
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_tail_preserves_text_json_and_utf8() {
        let path = crate::db::test_support::test_db_path("log-tail").with_extension("log");
        std::fs::write(
            &path,
            format!(
                "{}\r\nfirst\r\n{{\"message\":\"中文😀\"}}\r\nlast\n",
                "旧".repeat(200_000)
            ),
        )
        .unwrap();
        assert_eq!(
            tail_file_lines(&path, 2).unwrap(),
            vec![r#"{"message":"中文😀"}"#.to_string(), "last".to_string()]
        );
        assert!(tail_file_lines(&path, 0).unwrap().is_empty());
        std::fs::write(&path, "中".repeat(200_000)).unwrap();
        let tail = tail_file_lines(&path, 2).unwrap();
        assert!(tail[0].contains("truncated"));
        assert!(tail[1].len() <= 256 * 1024);
        assert!(!tail[1].contains('�'));
        std::fs::remove_file(path).unwrap();
    }
}
