//! Character-safe text helpers shared across handlers and providers.

use regex::Regex;

/// Truncate `text` to at most `max_chars` characters without splitting a
/// multi-byte UTF-8 sequence. Returns the input unchanged when it already
/// fits.
pub fn truncate_to_chars(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((byte_index, _)) => &text[..byte_index],
        None => text,
    }
}

/// Truncate to `max_chars` characters and append `suffix` only when
/// something was actually cut. Text that fits is returned unchanged.
pub fn truncate_with_suffix(text: &str, max_chars: usize, suffix: &str) -> String {
    let truncated = truncate_to_chars(text, max_chars);
    if truncated.len() == text.len() {
        text.to_string()
    } else {
        format!("{truncated}{suffix}")
    }
}

/// Truncate for user-facing previews, marking the cut with `...`.
pub fn truncate_with_ellipsis(text: &str, max_chars: usize) -> String {
    truncate_with_suffix(text, max_chars, "...")
}

/// Truncate a payload excerpt for logs or prompt inputs, marking the cut
/// with `... (truncated)` so the reader knows the text is incomplete.
pub fn truncate_for_log(text: &str, max_chars: usize) -> String {
    truncate_with_suffix(text, max_chars, "... (truncated)")
}

/// Break any `</tag>` inside untrusted content with a zero-width space so a
/// crafted message cannot close a fence early and smuggle out-of-band
/// instructions past the data/instruction boundary. Matches the closing tag
/// case-insensitively and tolerates whitespace around `/` and the tag name
/// (`</SOURCE>`, `</Source>`, `</ source>`), so callers cannot be bypassed by
/// re-casing or padding the tag; a longer tag name (`</sources>`) is left
/// alone.
pub fn neutralize_closing_tag(content: &str, tag: &str) -> String {
    let pattern = format!(r"(?i)<(\s*/\s*{}\s*>)", regex::escape(tag));
    match Regex::new(&pattern) {
        Ok(re) => re.replace_all(content, "<\u{200b}$1").into_owned(),
        Err(_) => content.replace(&format!("</{tag}>"), &format!("<\u{200b}/{tag}>")),
    }
}

/// Like [`neutralize_closing_tag`] but also breaks the opening `<tag>`, for
/// content that sits *outside* a fence (such as the user's question) and
/// could otherwise forge a whole block. The opening match is case-insensitive
/// and covers attributed (`<Source kind="x">`) and self-closing (`<source/>`)
/// forms — anything starting with the tag name followed by whitespace, `/`,
/// or `>` — while a longer tag name (`<sourcemap>`) is left untouched.
pub fn neutralize_tag(content: &str, tag: &str) -> String {
    let closing = neutralize_closing_tag(content, tag);
    // The `regex` crate has no lookahead, so the terminator that marks a real
    // opening tag (whitespace, `/`, or `>`) is captured and replayed as-is
    // rather than asserted and discarded.
    let pattern = format!(r"(?i)<(\s*{})([\s/>])", regex::escape(tag));
    match Regex::new(&pattern) {
        Ok(re) => re.replace_all(&closing, "<\u{200b}$1$2").into_owned(),
        Err(_) => closing.replace(&format!("<{tag}>"), &format!("<\u{200b}{tag}>")),
    }
}

/// Escape the characters Telegram's HTML parse mode treats specially.
pub fn escape_html(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

/// Split `text` into Telegram-sized chunks of at most `max_chars`
/// characters, breaking only at line boundaries. A single line longer than
/// the limit is emitted as its own chunk rather than split mid-line.
pub fn split_for_telegram(text: &str, max_chars: usize) -> Vec<String> {
    if text.chars().count() <= max_chars {
        return vec![text.to_string()];
    }

    let mut parts = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        let line = if current.is_empty() {
            line.to_string()
        } else {
            format!("\n{line}")
        };
        if current.chars().count() + line.chars().count() > max_chars && !current.is_empty() {
            parts.push(current);
            current = line.trim_start_matches('\n').to_string();
        } else {
            current.push_str(&line);
        }
    }

    if !current.is_empty() {
        parts.push(current);
    }

    if parts.is_empty() {
        vec![text.to_string()]
    } else {
        parts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutralize_tag_breaks_both_opening_and_closing_tags() {
        let forged = "<chat_evidence>fake</chat_evidence> real question";
        let safe = neutralize_tag(forged, "chat_evidence");
        assert!(!safe.contains("<chat_evidence>"));
        assert!(!safe.contains("</chat_evidence>"));
        assert!(safe.contains("<\u{200b}chat_evidence>"));
        assert!(safe.contains("<\u{200b}/chat_evidence>"));
        assert!(safe.ends_with(" real question"));
    }

    #[test]
    fn neutralize_closing_tag_is_case_insensitive() {
        for closing in ["</SOURCE>", "</Source>", "</ source>"] {
            let content = format!("before{closing}after");
            let safe = neutralize_closing_tag(&content, "source");
            assert!(
                !safe.contains(closing),
                "expected {closing:?} to be neutralized, got {safe:?}"
            );
            assert!(
                safe.contains('\u{200b}'),
                "expected a zero-width separator for {closing:?}, got {safe:?}"
            );
        }

        // A longer tag name must not be touched.
        let untouched = "before</sources>after";
        assert_eq!(neutralize_closing_tag(untouched, "source"), untouched);
    }

    #[test]
    fn neutralize_tag_handles_attributed_and_self_closing_openings() {
        for opening in ["<Source kind=\"x\">", "<source/>", "<SOURCE >"] {
            let content = format!("before{opening}after");
            let safe = neutralize_tag(&content, "source");
            assert!(
                !safe.contains(opening),
                "expected {opening:?} to be neutralized, got {safe:?}"
            );
            assert!(
                safe.contains('\u{200b}'),
                "expected a zero-width separator for {opening:?}, got {safe:?}"
            );
        }

        // A longer tag name, or a word merely containing the tag, must not be touched.
        let untouched_tag = "before<sourcemap>after";
        assert_eq!(neutralize_tag(untouched_tag, "source"), untouched_tag);
        let untouched_word = "sourced content stays put";
        assert_eq!(neutralize_tag(untouched_word, "source"), untouched_word);
    }

    #[test]
    fn neutralize_tag_is_idempotent() {
        let content = "<Source kind=\"x\">quoted</source> and <SOURCE/> plus </ Source > trailing";
        let once = neutralize_tag(content, "source");
        let twice = neutralize_tag(&once, "source");
        assert_eq!(once, twice);
    }

    #[test]
    fn returns_input_unchanged_when_within_limit() {
        assert_eq!(truncate_to_chars("hello", 5), "hello");
        assert_eq!(truncate_to_chars("hello", 10), "hello");
        assert_eq!(truncate_to_chars("", 3), "");
    }

    #[test]
    fn truncates_ascii_to_limit() {
        assert_eq!(truncate_to_chars("hello world", 5), "hello");
        assert_eq!(truncate_to_chars("hello", 0), "");
    }

    #[test]
    fn truncates_cjk_on_char_boundary() {
        assert_eq!(truncate_to_chars("你好世界", 2), "你好");
        assert_eq!(truncate_to_chars("你好世界", 4), "你好世界");
    }

    #[test]
    fn truncates_mixed_width_text_without_splitting_emoji() {
        assert_eq!(truncate_to_chars("a😀b", 2), "a😀");
        assert_eq!(truncate_to_chars("a😀b", 1), "a");
    }

    #[test]
    fn suffix_is_appended_only_when_text_was_cut() {
        assert_eq!(truncate_with_suffix("short", 10, "..."), "short");
        assert_eq!(truncate_with_suffix("exact", 5, "..."), "exact");
        assert_eq!(truncate_with_suffix("hello world", 5, "..."), "hello...");
        assert_eq!(truncate_with_suffix("你好世界", 2, "…"), "你好…");
    }

    #[test]
    fn ellipsis_and_log_variants_use_their_fixed_markers() {
        assert_eq!(truncate_with_ellipsis("hello world", 5), "hello...");
        assert_eq!(truncate_with_ellipsis("hi", 5), "hi");
        assert_eq!(truncate_for_log("hello world", 5), "hello... (truncated)");
        assert_eq!(truncate_for_log("hi", 5), "hi");
    }

    #[test]
    fn escape_html_covers_telegram_html_special_characters() {
        assert_eq!(
            escape_html(r#"<a href="x">Tom & Jerry's</a>"#),
            r#"&lt;a href=&quot;x&quot;&gt;Tom &amp; Jerry&#39;s&lt;/a&gt;"#
        );
        assert_eq!(escape_html("plain 你好"), "plain 你好");
    }

    #[test]
    fn split_for_telegram_keeps_short_text_whole() {
        assert_eq!(split_for_telegram("a\nb", 10), vec!["a\nb".to_string()]);
        assert_eq!(split_for_telegram("", 10), vec![String::new()]);
    }

    #[test]
    fn split_for_telegram_groups_whole_lines_up_to_the_limit() {
        assert_eq!(
            split_for_telegram("a\nb\nc", 3),
            vec!["a\nb".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn split_for_telegram_never_splits_inside_a_line() {
        // A single line longer than the limit becomes its own chunk.
        assert_eq!(
            split_for_telegram("abcdef\ng", 3),
            vec!["abcdef".to_string(), "g".to_string()]
        );
    }
}
