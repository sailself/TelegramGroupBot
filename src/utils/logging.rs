use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
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
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut ring = VecDeque::with_capacity(max_lines);

    for line in reader.lines() {
        let line = line?;
        if ring.len() == max_lines {
            ring.pop_front();
        }
        ring.push_back(line);
    }

    Ok(ring.into_iter().collect())
}
