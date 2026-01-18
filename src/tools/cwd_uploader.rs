use base64::{engine::general_purpose, Engine as _};
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

#[allow(dead_code)]
pub async fn upload_base64_image_to_cwd(
    base64_data: &str,
    api_key: &str,
    model: Option<&str>,
    prompt: Option<&str>,
) -> Option<String> {
    if !base64_data.starts_with("data:image/") {
        warn!("Invalid base64 image format - missing data URI prefix");
        return None;
    }

    let mut parts = base64_data.splitn(2, ',');
    let header = parts.next().unwrap_or_default();
    let payload = parts.next().unwrap_or_default();

    let mime_type = header.trim_start_matches("data:").split(';').next().unwrap_or("");
    if !mime_type.starts_with("image/") {
        warn!("Unsupported MIME type: {}", mime_type);
        return None;
    }

    let bytes = match general_purpose::STANDARD.decode(payload) {
        Ok(data) => data,
        Err(err) => {
            warn!("Failed to decode base64 data: {err}");
            return None;
        }
    };

    upload_image_bytes_to_cwd(&bytes, api_key, mime_type, model, prompt).await
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

    let file_ext = mime_type.split('/').nth(1).unwrap_or("png");
    let file_name = format!("upload.{}", if file_ext == "jpeg" { "jpg" } else { file_ext });

    let image_part = Part::bytes(image_bytes.to_vec())
        .file_name(file_name)
        .mime_str(mime_type)
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

    if !response.status().is_success() {
        warn!("CWD upload failed with status {}", response.status());
        return None;
    }

    let parsed = response.json::<CwdUploadResponse>().await.ok()?;
    if parsed.success {
        if let Some(url) = parsed.image_url.clone() {
            info!("Uploaded image to cwd.pw: {}", url);
            return Some(url);
        }
    }

    None
}
