use once_cell::sync::Lazy;
use regex::Regex;
use whatlang::{detect, Script};

const MIN_ALPHA_CHARS: usize = 2;
const LATIN_CONFIDENCE_THRESHOLD: f64 = 0.68;
const NON_LATIN_CONFIDENCE_THRESHOLD: f64 = 0.5;

static URL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"https?://\S+|www\.\S+").expect("valid url regex"));
static COMMAND_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)(^|\s)/[a-z0-9_@]+").expect("valid command regex"));
static MENTION_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)(^|\s)@[a-z0-9_]{3,}").expect("valid mention regex"));
static WHITESPACE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\s+").expect("valid whitespace regex"));

fn normalize_text_for_detection(text: &str) -> String {
    let without_urls = URL_RE.replace_all(text, " ");
    let without_commands = COMMAND_RE.replace_all(&without_urls, " ");
    let without_mentions = MENTION_RE.replace_all(&without_commands, " ");
    let without_code_ticks = without_mentions.replace('`', " ");
    WHITESPACE_RE
        .replace_all(&without_code_ticks, " ")
        .trim()
        .to_string()
}

fn alphabetic_char_count(text: &str) -> usize {
    text.chars().filter(|ch| ch.is_alphabetic()).count()
}

pub fn detect_language_name(text: &str) -> Option<String> {
    let normalized = normalize_text_for_detection(text);
    if normalized.is_empty() || alphabetic_char_count(&normalized) < MIN_ALPHA_CHARS {
        return None;
    }

    let info = detect(&normalized)?;
    if info.is_reliable() {
        return Some(info.lang().eng_name().to_string());
    }

    let threshold = match info.script() {
        Script::Latin => LATIN_CONFIDENCE_THRESHOLD,
        _ => NON_LATIN_CONFIDENCE_THRESHOLD,
    };
    if info.confidence() >= threshold {
        return Some(info.lang().eng_name().to_string());
    }

    None
}

fn language_name_from_ietf_tag(language_code: &str) -> Option<&'static str> {
    let primary = language_code.split('-').next()?.trim().to_lowercase();
    match primary.as_str() {
        "en" => Some("English"),
        "zh" => Some("Chinese"),
        "ja" => Some("Japanese"),
        "ko" => Some("Korean"),
        "ru" => Some("Russian"),
        "uk" => Some("Ukrainian"),
        "es" => Some("Spanish"),
        "pt" => Some("Portuguese"),
        "it" => Some("Italian"),
        "fr" => Some("French"),
        "de" => Some("German"),
        "ar" => Some("Arabic"),
        "hi" => Some("Hindi"),
        "tr" => Some("Turkish"),
        "nl" => Some("Dutch"),
        "pl" => Some("Polish"),
        "vi" => Some("Vietnamese"),
        "th" => Some("Thai"),
        "id" => Some("Indonesian"),
        "fa" => Some("Persian"),
        "he" | "iw" => Some("Hebrew"),
        "bn" => Some("Bengali"),
        "ta" => Some("Tamil"),
        _ => None,
    }
}

pub fn detect_language_or_fallback(
    text_candidates: &[&str],
    user_language_code: Option<&str>,
    default_language: &str,
) -> String {
    for text in text_candidates {
        if let Some(language) = detect_language_name(text) {
            return language;
        }
    }

    if let Some(code) = user_language_code {
        if let Some(language) = language_name_from_ietf_tag(code) {
            return language.to_string();
        }
    }

    default_language.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn falls_back_to_user_language_code_for_short_text() {
        let detected = detect_language_or_fallback(&["👍"], Some("pt-BR"), "English");
        assert_eq!(detected, "Portuguese");
    }

    #[test]
    fn prefers_detected_language_over_user_fallback() {
        let detected = detect_language_or_fallback(
            &["¿Puedes resumir esta conversación en español, por favor?"],
            Some("en-US"),
            "English",
        );
        assert_eq!(detected, "Spanish");
    }

    #[test]
    fn keeps_default_when_no_signal_is_available() {
        let detected = detect_language_or_fallback(&["12345"], None, "English");
        assert_eq!(detected, "English");
    }
}
