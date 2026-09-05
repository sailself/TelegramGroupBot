//! Character-safe text helpers shared across handlers and providers.

/// Truncate `text` to at most `max_chars` characters without splitting a
/// multi-byte UTF-8 sequence. Returns the input unchanged when it already
/// fits.
pub fn truncate_to_chars(text: &str, max_chars: usize) -> &str {
    match text.char_indices().nth(max_chars) {
        Some((byte_index, _)) => &text[..byte_index],
        None => text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
