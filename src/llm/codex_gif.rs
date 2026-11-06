use crate::config::CONFIG;
use crate::llm::{
    audit::LlmAuditContext,
    codex_image::{self, ImageInput, CODEX_IMAGE_MAX_INPUT_IMAGES},
    gemini::ImageGenerationError,
    gif_spritesheet,
    media::{detect_mime_type, download_media_limited},
};
use serde_json::json;
use std::io::Cursor;

const MAX_REFERENCE_BYTES: usize = 10 * 1024 * 1024;

pub fn model() -> String {
    resolve_model(&CONFIG.codex.gif_model)
}

fn resolve_model(value: &str) -> String {
    let model = value.trim();
    if model.is_empty() {
        "gpt-image-2".to_string()
    } else {
        model.to_string()
    }
}

fn sheet_prompt(prompt: &str, count: u32) -> String {
    let (columns, rows) = gif_spritesheet::grid(count);
    format!("Create ONE still sprite sheet, not an animated image, exactly {columns} columns by {rows} rows of equally sized rectangular tiles edge to edge. Place {count} successive poses in row-major order, left to right then top to bottom; unused tiles may be blank. No gutters, spacing, borders, text, labels, or extra panels. Keep the same subject, scale and placement in each frame, a consistent background and horizon, and the whole subject visible. Produce smooth motion with the final pose transitioning naturally back to the first. Follow this animation request: {prompt}")
}

fn sheet_size(columns: u32, rows: u32) -> &'static str {
    if columns > rows {
        "1536x1024"
    } else if rows > columns {
        "1024x1536"
    } else {
        "1024x1024"
    }
}

pub async fn generate(
    prompt: &str,
    urls: &[String],
    audit: Option<&LlmAuditContext>,
) -> Result<Vec<u8>, ImageGenerationError> {
    if urls.len() > CODEX_IMAGE_MAX_INPUT_IMAGES {
        return Err(ImageGenerationError(
            "Too many reference images (maximum 5)".into(),
        ));
    }
    let mut images = Vec::new();
    for url in urls {
        let bytes = download_media_limited(url, MAX_REFERENCE_BYTES)
            .await
            .ok_or_else(|| ImageGenerationError("Failed to download a reference image".into()))?;
        let mime_type = detect_mime_type(&bytes)
            .filter(|s| matches!(s.as_str(), "image/png" | "image/jpeg" | "image/webp"))
            .ok_or_else(|| {
                ImageGenerationError("Reference must be a static PNG, JPEG, or WebP image".into())
            })?;
        let animated = match mime_type.as_str() {
            "image/png" => image::codecs::png::PngDecoder::new(Cursor::new(&bytes))
                .and_then(|decoder| decoder.is_apng())
                .map_err(|_| ImageGenerationError("Invalid reference image".into()))?,
            "image/webp" => image::codecs::webp::WebPDecoder::new(Cursor::new(&bytes))
                .map_err(|_| ImageGenerationError("Invalid reference image".into()))?
                .has_animation(),
            _ => false,
        };
        if animated {
            return Err(ImageGenerationError(
                "Reference images must be static".into(),
            ));
        }
        images.push(ImageInput { bytes, mime_type });
    }
    let model = model();
    let count = CONFIG.codex.gif_frame_count;
    let duration = CONFIG.codex.gif_frame_duration_ms;
    let (columns, rows) = gif_spritesheet::grid(count);
    let prompt = sheet_prompt(prompt, count);
    let size = sheet_size(columns, rows);
    let payload =
        codex_image::build_codex_image_payload_for_model(&prompt, &images, Some(size), &model);
    let images = codex_image::call_codex_image_api(&payload, &model, "generate_gif_with_codex", Some(json!({"mode":"sprite_sheet", "columns":columns, "rows":rows, "frame_count":count, "frame_duration_ms":duration, "input_images":images.len()})), audit).await?;
    tokio::task::spawn_blocking(move || {
        for sheet in &images {
            if let Err(error) = super::gif_artifacts::save_sheet(sheet) {
                tracing::warn!(%error, "Could not save GIF sprite sheet");
            }
        }
        if images.len() != 1 {
            return Err("GIF_LAYOUT: expected exactly one sprite sheet".to_string());
        }
        gif_spritesheet::assemble(&images[0], count, duration)
    })
    .await
    .map_err(|_| ImageGenerationError("GIF_LIMIT: sprite sheet processing failed".into()))?
    .map_err(|detail| {
        tracing::warn!(%detail, "Could not assemble Codex sprite sheet");
        ImageGenerationError(detail)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_defaults_and_trims_override() {
        assert_eq!(resolve_model("  "), "gpt-image-2");
        assert_eq!(resolve_model(" custom "), "custom");
    }
    #[test]
    fn sprite_sheet_request_uses_explicit_model_and_references() {
        let prompt = sheet_prompt("cat moves", 5);
        let images = [ImageInput {
            bytes: vec![1, 2],
            mime_type: "image/png".into(),
        }];
        let payload =
            codex_image::build_codex_image_payload_for_model(&prompt, &images, None, "gif-model");
        assert_eq!(payload["tools"][0]["model"], "gif-model");
        assert!(payload["tools"][0].get("output_format").is_none());
        assert_eq!(
            payload["input"][0]["content"][1]["image_url"],
            "data:image/png;base64,AQI="
        );
        assert!(prompt.contains("3 columns by 2 rows"));
        assert!(prompt.contains("ONE still sprite sheet"));
        assert!(prompt.contains("row-major"));
        assert!(prompt.contains("No gutters"));
        assert_eq!(sheet_size(4, 2), "1536x1024");
        assert_eq!(sheet_size(3, 3), "1024x1024");
        let landscape = codex_image::build_codex_image_payload_for_model(
            &sheet_prompt("move", 8),
            &[],
            Some(sheet_size(4, 2)),
            "gif-model",
        );
        assert_eq!(landscape["tools"][0]["size"], "1536x1024");
    }
}
