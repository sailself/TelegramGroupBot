//! Bounded conversion of a borderless sprite sheet to a looping GIF.
use image::codecs::gif::{GifEncoder, Repeat};
use image::{imageops::FilterType, GenericImageView, ImageReader};
use std::io::Cursor;

const MAX_SHEET_BYTES: usize = 32 * 1024 * 1024;
const MAX_DIMENSION: u32 = 4096;
const MAX_FRAME_EDGE: u32 = 512;
const MAX_GIF_BYTES: usize = 50_000_000;

pub(super) fn grid(count: u32) -> (u32, u32) {
    // Prefer a filled grid, then near-square proportions, then landscape.
    (1..=4)
        .filter_map(|columns: u32| {
            let rows = count.div_ceil(columns);
            (rows <= 4).then_some((columns, rows))
        })
        .min_by_key(|&(columns, rows)| {
            (
                columns * rows - count,
                columns.abs_diff(rows),
                std::cmp::Reverse(columns),
            )
        })
        .expect("frame count is within 2..=16")
}

fn cells(length: u32, parts: u32) -> Result<Vec<(u32, u32)>, String> {
    if length / parts < 16 {
        return Err("GIF_LAYOUT: sprite sheet cells are too small".into());
    }
    Ok((0..parts)
        .map(|i| (i * length / parts, (i + 1) * length / parts))
        .collect())
}

pub(super) fn assemble(bytes: &[u8], count: u32, duration_ms: u32) -> Result<Vec<u8>, String> {
    if !(2..=16).contains(&count)
        || !(20..=2000).contains(&duration_ms)
        || bytes.len() > MAX_SHEET_BYTES
    {
        return Err("GIF_LIMIT: invalid sprite sheet limits".into());
    }
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| "GIF_LAYOUT: invalid sprite sheet")?;
    if !matches!(
        reader.format(),
        Some(image::ImageFormat::Png | image::ImageFormat::Jpeg | image::ImageFormat::WebP)
    ) {
        return Err("GIF_LAYOUT: sprite sheet must be a still PNG, JPEG, or WebP image".into());
    }
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_DIMENSION);
    limits.max_image_height = Some(MAX_DIMENSION);
    limits.max_alloc = Some(128 * 1024 * 1024);
    reader.limits(limits);
    let sheet = reader
        .decode()
        .map_err(|_| "GIF_LAYOUT: could not decode sprite sheet or dimensions exceed limits")?
        .to_rgba8();
    let (columns, rows) = grid(count);
    tracing::info!(
        sheet_width = sheet.width(),
        sheet_height = sheet.height(),
        columns,
        rows,
        frames = count,
        "Dividing Codex sprite sheet into equal tiles"
    );
    let xs = cells(sheet.width(), columns)?;
    let ys = cells(sheet.height(), rows)?;
    let source_width = xs.iter().map(|(a, b)| b - a).min().unwrap();
    let source_height = ys.iter().map(|(a, b)| b - a).min().unwrap();
    let scale_denominator = source_width.max(source_height).max(MAX_FRAME_EDGE);
    let frame_width = (source_width * MAX_FRAME_EDGE / scale_denominator).max(1);
    let frame_height = (source_height * MAX_FRAME_EDGE / scale_denominator).max(1);
    let mut output = Vec::new();
    let delay = ((duration_ms + 5) / 10) * 10;
    {
        let mut encoder = GifEncoder::new(&mut output);
        encoder
            .set_repeat(Repeat::Infinite)
            .map_err(|_| "GIF_LIMIT: could not initialize animation")?;
        for index in 0..count {
            let (x1, x2) = xs[(index % columns) as usize];
            let (y1, y2) = ys[(index / columns) as usize];
            let frame = sheet.view(x1, y1, x2 - x1, y2 - y1).to_image();
            let frame =
                image::imageops::resize(&frame, frame_width, frame_height, FilterType::Triangle);
            encoder
                .encode_frame(image::Frame::from_parts(
                    frame,
                    0,
                    0,
                    image::Delay::from_numer_denom_ms(delay, 1),
                ))
                .map_err(|_| "GIF_LIMIT: could not encode animation")?;
        }
    }
    if output.len() > MAX_GIF_BYTES {
        return Err("GIF_LIMIT: animation exceeds Telegram's 50 MB limit".into());
    }
    tracing::info!(
        sheet_width = sheet.width(),
        sheet_height = sheet.height(),
        columns,
        rows,
        frames = count,
        frame_width,
        frame_height,
        duration_ms = delay,
        bytes = output.len(),
        "Codex sprite sheet encoded as GIF"
    );
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{AnimationDecoder, ImageFormat, Rgba, RgbaImage};

    fn sheet(count: u32, remainder: u32) -> Vec<u8> {
        let (cols, rows) = grid(count);
        let width = cols * 32 + remainder;
        let height = rows * 32 + remainder;
        let mut image = RgbaImage::new(width, height);
        for n in 0..cols * rows {
            let color = Rgba([(n * 31) as u8, 60, 120, 255]);
            for y in (n / cols) * height / rows..((n / cols) + 1) * height / rows {
                for x in (n % cols) * width / cols..((n % cols) + 1) * width / cols {
                    image.put_pixel(x, y, color);
                }
            }
        }
        let mut output = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut output, ImageFormat::Png)
            .unwrap();
        output.into_inner()
    }
    #[test]
    fn grid_and_remainders_are_deterministic() {
        assert_eq!(grid(6), (3, 2));
        assert_eq!(grid(8), (4, 2));
        assert_eq!(grid(9), (3, 3));
        assert_eq!(grid(13), (4, 4));
        assert_eq!(cells(103, 3).unwrap(), vec![(0, 34), (34, 68), (68, 103)]);
        for count in 2..=16 {
            let (c, r) = grid(count);
            assert!(c <= 4 && r <= 4 && c * r >= count);
        }
    }
    #[test]
    fn frames_are_ordered_looped_and_delays_rounded() {
        for count in [2, 5, 8, 9, 13, 16] {
            let result = assemble(&sheet(count, 1), count, 25).unwrap();
            assert!(result
                .windows(16)
                .any(|b| b == b"NETSCAPE2.0\x03\x01\x00\x00\x00"));
            let frames = image::codecs::gif::GifDecoder::new(Cursor::new(result))
                .unwrap()
                .into_frames()
                .collect_frames()
                .unwrap();
            assert_eq!(frames.len(), count as usize);
            for (i, frame) in frames.iter().enumerate() {
                assert_eq!(frame.delay().numer_denom_ms(), (30, 1));
                let pixel = frame.buffer().get_pixel(16, 16);
                assert!((i16::from(pixel[0]) - i16::from((i * 31) as u8)).abs() <= 3);
                let (columns, rows) = grid(count);
                assert_eq!(
                    frame.buffer().dimensions(),
                    ((columns * 32 + 1) / columns, (rows * 32 + 1) / rows)
                );
            }
        }
    }
    #[test]
    fn borderless_white_and_dark_sheets_are_accepted_and_limits_enforced() {
        for color in [Rgba([255, 255, 255, 255]), Rgba([0, 0, 0, 255])] {
            let image = RgbaImage::from_pixel(128, 64, color);
            let mut bytes = Cursor::new(Vec::new());
            image::DynamicImage::ImageRgba8(image)
                .write_to(&mut bytes, ImageFormat::Png)
                .unwrap();
            assert!(assemble(&bytes.into_inner(), 8, 200).is_ok());
        }
        assert!(cells(31, 2).is_err());
        assert!(assemble(&sheet(5, 0), 1, 200).is_err());
        assert!(assemble(&sheet(5, 0), 5, 19).is_err());
        assert!(assemble(&vec![0; MAX_SHEET_BYTES + 1], 5, 200).is_err());
        assert!(assemble(b"invalid", 5, 200).is_err());
    }
    #[test]
    fn resize_preserves_aspect_ratio() {
        let image = RgbaImage::from_pixel(1200, 300, Rgba([10, 20, 30, 255]));
        let mut bytes = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(image)
            .write_to(&mut bytes, ImageFormat::Png)
            .unwrap();
        let gif = assemble(&bytes.into_inner(), 2, 200).unwrap();
        let frames = image::codecs::gif::GifDecoder::new(Cursor::new(gif))
            .unwrap()
            .into_frames()
            .collect_frames()
            .unwrap();
        assert_eq!(frames[0].buffer().dimensions(), (512, 256));
    }
    #[test]
    #[ignore = "optional offline preview; set GIF_SHEET_PREVIEW_INPUT and GIF_SHEET_PREVIEW_OUTPUT"]
    fn offline_preview() {
        let input = std::env::var("GIF_SHEET_PREVIEW_INPUT").unwrap();
        let output = std::env::var("GIF_SHEET_PREVIEW_OUTPUT").unwrap();
        let gif = assemble(&std::fs::read(input).unwrap(), 8, 125).unwrap();
        std::fs::write(output, gif).unwrap();
    }
}
