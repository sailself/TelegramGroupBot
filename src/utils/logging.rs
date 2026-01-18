use std::fs;
use std::path::Path;

use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::filter::{LevelFilter, Targets};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

use crate::config::CONFIG;

pub struct LoggingGuards {
    _file_guard: WorkerGuard,
    _timing_guard: WorkerGuard,
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

    let general_level = parse_log_level(&CONFIG.log_level);
    let general_filter = Targets::new()
        .with_default(general_level)
        .with_target("bot.timing", LevelFilter::OFF);
    let timing_filter = Targets::new()
        .with_default(LevelFilter::OFF)
        .with_target("bot.timing", LevelFilter::INFO);

    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(file_writer)
        .with_ansi(false)
        .with_filter(general_filter.clone());
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_filter(general_filter);
    let timing_layer = tracing_subscriber::fmt::layer()
        .with_writer(timing_writer)
        .with_ansi(false)
        .with_filter(timing_filter);

    tracing_subscriber::registry()
        .with(file_layer)
        .with(stdout_layer)
        .with(timing_layer)
        .init();

    LoggingGuards {
        _file_guard: file_guard,
        _timing_guard: timing_guard,
    }
}
