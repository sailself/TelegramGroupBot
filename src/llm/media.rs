use tracing::{error, warn};

use crate::utils::http::get_http_client;

pub fn detect_mime_type(data: &[u8]) -> Option<String> {
    if data.len() > 12 {
        let ftyp = &data[4..12];
        if ftyp.starts_with(b"ftyp") {
            let brand = &ftyp[4..8];
            if brand == b"heic" || brand == b"heif" || brand == b"hevc" {
                return Some("image/heic".to_string());
            }
        }
    }

    infer::get(data).map(|kind| kind.mime_type().to_string())
}

pub async fn download_media(url: &str) -> Option<Vec<u8>> {
    let client = get_http_client();
    let response = match client.get(url).send().await {
        Ok(resp) => resp,
        Err(err) => {
            warn!("Failed to fetch media {url}: {err}");
            return None;
        }
    };

    if !response.status().is_success() {
        warn!("Media download failed for {url} with status {}", response.status());
        return None;
    }

    match response.bytes().await {
        Ok(bytes) => Some(bytes.to_vec()),
        Err(err) => {
            error!("Failed to read media bytes {url}: {err}");
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Video,
    Audio,
    Document,
}

#[derive(Debug, Clone)]
pub struct MediaFile {
    pub bytes: Vec<u8>,
    pub mime_type: String,
    pub kind: MediaKind,
    pub display_name: Option<String>,
}

impl MediaFile {
    pub fn new(
        bytes: Vec<u8>,
        mime_type: String,
        kind: MediaKind,
        display_name: Option<String>,
    ) -> Self {
        Self {
            bytes,
            mime_type,
            kind,
            display_name,
        }
    }
}

pub fn kind_for_mime(mime_type: &str) -> MediaKind {
    if mime_type.starts_with("image/") {
        MediaKind::Image
    } else if mime_type.starts_with("video/") {
        MediaKind::Video
    } else if mime_type.starts_with("audio/") {
        MediaKind::Audio
    } else {
        MediaKind::Document
    }
}
