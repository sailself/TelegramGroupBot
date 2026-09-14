use image::codecs::jpeg::JpegEncoder;
use reqwest::multipart::{Form, Part};
use serde::Deserialize;
use tracing::{info, warn};

use crate::utils::http::get_http_client;

#[derive(Debug, Deserialize)]
struct CwdUploadResponse {
    success: bool,
    #[serde(rename = "imageUrl")]
    image_url: Option<String>,
}

pub async fn upload_image_bytes_to_cwd(
    image_bytes: &[u8],
    api_key: &str,
    mime_type: &str,
    model: Option<&str>,
    prompt: Option<&str>,
) -> Option<String> {
    if api_key.trim().is_empty() {
        return None;
    }

    let (upload_bytes, upload_mime_type) = match mime_type {
        "image/jpeg" | "image/jpg" => (image_bytes.to_vec(), "image/jpeg".to_string()),
        _ => match image::load_from_memory(image_bytes) {
            Ok(image) => {
                let mut output = Vec::new();
                if JpegEncoder::new_with_quality(&mut output, 90)
                    .encode_image(&image)
                    .is_ok()
                {
                    (output, "image/jpeg".to_string())
                } else {
                    (image_bytes.to_vec(), mime_type.to_string())
                }
            }
            Err(_) => (image_bytes.to_vec(), mime_type.to_string()),
        },
    };

    let file_ext = upload_mime_type.split('/').nth(1).unwrap_or("png");
    let file_name = format!(
        "upload.{}",
        if file_ext == "jpeg" { "jpg" } else { file_ext }
    );

    let image_part = Part::bytes(upload_bytes)
        .file_name(file_name)
        .mime_str(&upload_mime_type)
        .ok()?;

    let form = Form::new()
        .part("image", image_part)
        .text("api_key", api_key.to_string())
        .text("ai_generated", "true")
        .text("model", model.unwrap_or("").to_string())
        .text("prompt", prompt.unwrap_or("").to_string());

    let client = get_http_client();
    let response = client
        .post("https://cwd.pw/api/upload-image")
        .multipart(form)
        .send()
        .await
        .ok()?;

    let status = response.status();
    let body = crate::utils::http::read_body_capped("CWD upload", response, 1024 * 1024)
        .await
        .ok()?;
    if !status.is_success() {
        let excerpt = crate::utils::text::external_failure_excerpt(
            &String::from_utf8_lossy(&body).replace(api_key, "[REDACTED]"),
        );
        warn!("CWD upload failed with status {}: {}", status, excerpt);
        return None;
    }
    let parsed = serde_json::from_slice::<CwdUploadResponse>(&body).ok()?;
    if parsed.success {
        if let Some(url) = parsed.image_url.clone() {
            info!("Uploaded image to cwd.pw: {}", url);
            return Some(url);
        }
    }

    None
}
