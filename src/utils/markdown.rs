//! Render a constrained subset of Markdown (LLM output, static prompt text)
//! into the HTML markup Telegram's `ParseMode::Html` accepts. Only
//! `b i u s code pre a blockquote tg-spoiler` are supported tags; this
//! renderer never emits anything else, and every text node/attribute value
//! goes through `crate::utils::text::escape_html` so raw HTML in the input
//! can never reach Telegram unescaped.

use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag};

use crate::utils::text::escape_html;

/// Accumulates the rows of a Markdown table while it is being walked so it
/// can be rendered as a `|`-aligned grid inside a single `<pre>` block.
#[derive(Default)]
struct TableAccumulator {
    rows: Vec<Vec<String>>,
    current_row: Vec<String>,
}

/// Walks the `pulldown_cmark` event stream once and builds the Telegram-safe
/// HTML string. `buffer_stack` holds the in-progress text for whichever link,
/// image or table cell is currently open (nesting is rare but handled), so
/// every text-producing event can write through the same `sink()` regardless
/// of context.
struct Renderer {
    out: String,
    list_stack: Vec<Option<u64>>,
    buffer_stack: Vec<String>,
    link_url_stack: Vec<String>,
    table: Option<TableAccumulator>,
    /// True from `Tag::Table`'s start to its end. Telegram's `<pre>` (which
    /// a table renders inside) cannot itself contain nested entities, so
    /// while this is set every inline construct degrades to plain escaped
    /// text instead of emitting `<b>`/`<i>`/`<s>`/`<code>`/`<a>`.
    in_table: bool,
}

impl Renderer {
    fn new() -> Self {
        Self {
            out: String::new(),
            list_stack: Vec::new(),
            buffer_stack: Vec::new(),
            link_url_stack: Vec::new(),
            table: None,
            in_table: false,
        }
    }

    /// The string currently being written to: the innermost open link/image
    /// alt-text/table-cell buffer, or the top-level output.
    fn sink(&mut self) -> &mut String {
        match self.buffer_stack.last_mut() {
            Some(buffer) => buffer,
            None => &mut self.out,
        }
    }

    /// Break onto a fresh line unless we already are on one, so block-level
    /// markers (list bullets, nested lists) never glue onto prior text.
    fn ensure_newline(&mut self) {
        let sink = self.sink();
        if !sink.is_empty() && !sink.ends_with('\n') {
            sink.push('\n');
        }
    }

    fn push_text(&mut self, text: &str) {
        let escaped = escape_html(text);
        self.sink().push_str(&escaped);
    }

    fn push_raw(&mut self, text: &str) {
        self.sink().push_str(text);
    }

    fn handle(&mut self, event: Event) {
        match event {
            Event::Start(tag) => self.start_tag(tag),
            Event::End(tag) => self.end_tag(tag),
            Event::Text(text) => self.push_text(&text),
            // Raw inline/block HTML is never trusted: escape it like any
            // other text node instead of passing it through.
            Event::Html(text) => self.push_text(&text),
            Event::Code(text) => {
                if self.in_table {
                    self.push_text(&text);
                } else {
                    let wrapped = format!("<code>{}</code>", escape_html(&text));
                    self.sink().push_str(&wrapped);
                }
            }
            Event::SoftBreak | Event::HardBreak => self.push_raw("\n"),
            Event::Rule => {
                self.ensure_newline();
                self.push_raw("———\n\n");
            }
            Event::TaskListMarker(checked) => {
                self.push_raw(if checked { "[x] " } else { "[ ] " });
            }
            Event::FootnoteReference(_) => {}
        }
    }

    fn start_tag(&mut self, tag: Tag) {
        match tag {
            Tag::Paragraph => {}
            Tag::Heading(..) => self.push_raw("<b>"),
            Tag::BlockQuote => self.push_raw("<blockquote>"),
            Tag::CodeBlock(kind) => {
                let tag = code_block_open_tag(&kind);
                self.push_raw(&tag);
            }
            Tag::List(start) => {
                self.ensure_newline();
                self.list_stack.push(start);
            }
            Tag::Item => {
                self.ensure_newline();
                let depth = self.list_stack.len().saturating_sub(1);
                let indent = "  ".repeat(depth);
                let marker = match self.list_stack.last_mut() {
                    Some(Some(number)) => {
                        let rendered = format!("{}. ", number);
                        *number += 1;
                        rendered
                    }
                    _ => "• ".to_string(),
                };
                self.push_raw(&format!("{indent}{marker}"));
            }
            Tag::Emphasis => {
                if !self.in_table {
                    self.push_raw("<i>");
                }
            }
            Tag::Strong => {
                if !self.in_table {
                    self.push_raw("<b>");
                }
            }
            Tag::Strikethrough => {
                if !self.in_table {
                    self.push_raw("<s>");
                }
            }
            Tag::Link(_, dest_url, _) => {
                self.link_url_stack.push(dest_url.to_string());
                self.buffer_stack.push(String::new());
            }
            Tag::Image(..) => self.buffer_stack.push(String::new()),
            Tag::Table(_) => {
                self.table = Some(TableAccumulator::default());
                self.in_table = true;
            }
            Tag::TableHead | Tag::TableRow => {}
            Tag::TableCell => self.buffer_stack.push(String::new()),
            _ => {}
        }
    }

    fn end_tag(&mut self, tag: Tag) {
        match tag {
            Tag::Paragraph => self.push_raw("\n\n"),
            Tag::Heading(..) => self.push_raw("</b>\n"),
            Tag::BlockQuote => self.push_raw("</blockquote>\n\n"),
            Tag::CodeBlock(_) => self.push_raw("</code></pre>\n\n"),
            Tag::List(_) => {
                self.list_stack.pop();
                self.ensure_newline();
                if self.list_stack.is_empty() {
                    self.push_raw("\n");
                }
            }
            Tag::Item => self.ensure_newline(),
            Tag::Emphasis => {
                if !self.in_table {
                    self.push_raw("</i>");
                }
            }
            Tag::Strong => {
                if !self.in_table {
                    self.push_raw("</b>");
                }
            }
            Tag::Strikethrough => {
                if !self.in_table {
                    self.push_raw("</s>");
                }
            }
            Tag::Link(..) => {
                let text = self.buffer_stack.pop().unwrap_or_default();
                let url = self.link_url_stack.pop().unwrap_or_default();
                let escaped_url = escape_html(&url);
                let rendered = if self.in_table {
                    // <pre> cannot contain a nested <a>, so every link
                    // (regardless of scheme) flattens to plain text here.
                    if text == escaped_url {
                        text
                    } else {
                        format!("{} ({})", text, escaped_url)
                    }
                } else if allowed_link_scheme(&url) {
                    format!("<a href=\"{}\">{}</a>", escaped_url, text)
                } else {
                    format!("{} ({})", text, escaped_url)
                };
                self.push_raw(&rendered);
            }
            Tag::Image(..) => {
                let alt = self.buffer_stack.pop().unwrap_or_default();
                self.push_raw(&alt);
            }
            Tag::Table(_) => {
                if let Some(table) = self.table.take() {
                    let grid = render_table(&table.rows);
                    let wrapped = format!("<pre>{grid}</pre>\n\n");
                    self.push_raw(&wrapped);
                }
                self.in_table = false;
            }
            Tag::TableHead | Tag::TableRow => {
                if let Some(table) = self.table.as_mut() {
                    let row = std::mem::take(&mut table.current_row);
                    table.rows.push(row);
                }
            }
            Tag::TableCell => {
                let cell = self.buffer_stack.pop().unwrap_or_default();
                if let Some(table) = self.table.as_mut() {
                    table.current_row.push(cell);
                }
            }
            _ => {}
        }
    }
}

/// The opening `<pre><code>` tag for a code block, with a `language-*` class
/// only when the fence declares one (indented code blocks never do).
fn code_block_open_tag(kind: &CodeBlockKind) -> String {
    let lang = match kind {
        CodeBlockKind::Fenced(info) => info.split_whitespace().next().unwrap_or(""),
        CodeBlockKind::Indented => "",
    };
    if lang.is_empty() {
        "<pre><code>".to_string()
    } else {
        format!("<pre><code class=\"language-{}\">", escape_html(lang))
    }
}

/// Telegram only lets `<a href>` point at a handful of schemes; anything else
/// (e.g. `javascript:`) is flattened to plain text by the caller.
fn allowed_link_scheme(url: &str) -> bool {
    match url.find(':') {
        Some(idx) => matches!(
            url[..idx].to_ascii_lowercase().as_str(),
            "http" | "https" | "tg"
        ),
        None => false,
    }
}

/// Render a Markdown table as a monospace, `|`-aligned grid meant to sit
/// inside a `<pre>` block (Telegram has no native table markup).
fn render_table(rows: &[Vec<String>]) -> String {
    if rows.is_empty() {
        return String::new();
    }
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut widths = vec![0usize; columns];
    for row in rows {
        for (index, cell) in row.iter().enumerate() {
            widths[index] = widths[index].max(cell.chars().count());
        }
    }

    let mut lines = Vec::with_capacity(rows.len() + 1);
    for (row_index, row) in rows.iter().enumerate() {
        let cells: Vec<String> = (0..columns)
            .map(|index| {
                let cell = row.get(index).map(String::as_str).unwrap_or("");
                format!("{:width$}", cell, width = widths[index])
            })
            .collect();
        lines.push(format!("| {} |", cells.join(" | ")));
        if row_index == 0 {
            let separators: Vec<String> = widths.iter().map(|width| "-".repeat(*width)).collect();
            lines.push(format!("|-{}-|", separators.join("-|-")));
        }
    }
    lines.join("\n")
}

/// Render `input` (Markdown, as produced by an LLM or a static prompt) into
/// HTML restricted to Telegram's supported tag set. Unsupported constructs
/// degrade gracefully (unsafe link schemes flatten to plain text, images
/// become their alt text) rather than emitting a tag Telegram would reject.
pub fn markdown_to_telegram_html(input: &str) -> String {
    let options = Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES;
    let parser = Parser::new_ext(input, options);
    let mut renderer = Renderer::new();
    for event in parser {
        renderer.handle(event);
    }
    renderer.out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::markdown_to_telegram_html;

    #[test]
    fn bold_italic_code_and_links_render_as_telegram_tags() {
        let html =
            markdown_to_telegram_html("**bold** and *it* and `x < y` [t](https://a.b/c?d=1&e=2)");
        assert_eq!(
            html,
            "<b>bold</b> and <i>it</i> and <code>x &lt; y</code> <a href=\"https://a.b/c?d=1&amp;e=2\">t</a>"
        );
    }

    #[test]
    fn unmatched_markers_and_raw_html_are_escaped_not_interpreted() {
        let html = markdown_to_telegram_html("5 * 3 = 15, a_b_c, <script>alert(1)</script> & done");
        assert!(html.contains("5 * 3 = 15"));
        assert!(html.contains("a_b_c"));
        assert!(html.contains("&lt;script&gt;alert(1)&lt;/script&gt; &amp; done"));
        assert!(!html.contains("<script"));
    }

    #[test]
    fn fenced_code_keeps_language_and_escapes_body() {
        let html = markdown_to_telegram_html("```rust\nlet a = 1 < 2;\n```");
        assert_eq!(
            html,
            "<pre><code class=\"language-rust\">let a = 1 &lt; 2;\n</code></pre>"
        );
    }

    #[test]
    fn headings_and_lists_become_bold_lines_and_bullets() {
        let html = markdown_to_telegram_html(
            "## Title\n\n- one\n- two\n  - nested\n\n1. first\n2. second",
        );
        assert!(html.starts_with("<b>Title</b>\n"));
        assert!(html.contains("• one\n• two\n  • nested\n"));
        assert!(html.contains("1. first\n2. second"));
    }

    #[test]
    fn unsafe_link_schemes_are_flattened() {
        let html = markdown_to_telegram_html("[x](javascript:alert(1))");
        assert_eq!(html, "x (javascript:alert(1))");
    }

    #[test]
    fn plain_text_round_trips_without_extra_tags() {
        assert_eq!(markdown_to_telegram_html("hello world"), "hello world");
    }

    #[test]
    fn table_cells_render_as_plain_text_inside_pre() {
        let html =
            markdown_to_telegram_html("| **Alice** | [link](https://x.y) |\n|---|---|\n| a | b |");
        let pre_start = html
            .find("<pre>")
            .expect("table should render inside <pre>");
        let pre_end = html.find("</pre>").expect("table's <pre> should close");
        let pre_body = &html[pre_start..pre_end];
        assert!(pre_body.contains("Alice"));
        assert!(pre_body.contains("link (https://x.y)"));
        assert!(!pre_body.contains("<b>"));
        assert!(!pre_body.contains("<a "));
    }
}
