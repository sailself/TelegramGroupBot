#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_identity_canonicalizes_supported_variants() {
        for raw in [
            "https://x.com/alice/status/123456?s=20",
            "https://mobile.twitter.com/alice/status/123456/photo/1",
            "https://fxtwitter.com/alice/status/123456#fragment",
        ] {
            let identity = parse_status_identity(raw).unwrap();
            assert_eq!(identity.id, "123456");
            assert_eq!(
                identity.canonical_url.as_str(),
                "https://x.com/i/status/123456"
            );
            assert_eq!(canonical_status_key(raw).unwrap(), "123456");
        }
    }

    #[test]
    fn status_identity_rejects_suffix_confusion_and_non_numeric_ids() {
        for raw in [
            "https://evilx.com/alice/status/123456",
            "https://eviltwitter.com/alice/status/123456",
            "https://x.com/alice/status/not-a-number",
            "https://x.com/alice/not-status/123456",
        ] {
            assert!(parse_status_identity(raw).is_err(), "accepted {raw}");
        }
    }

    #[test]
    fn status_identity_accepts_case_insensitive_http_schemes_and_schemeless_urls() {
        for raw in [
            "HTTPS://x.com/alice/status/123456",
            "HtTp://x.com/alice/status/123456",
            "x.com/alice/status/123456",
        ] {
            assert_eq!(
                canonical_status_key(raw).unwrap(),
                "123456",
                "rejected {raw}"
            );
        }
        assert!(parse_status_identity("ftp://x.com/alice/status/123456").is_err());
    }
}
use ::url::Url;
use anyhow::{anyhow, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XStatusIdentity {
    pub id: String,
    pub canonical_url: Url,
}

const SUPPORTED_HOSTS: &[&str] = &[
    "x.com",
    "www.x.com",
    "twitter.com",
    "www.twitter.com",
    "mobile.twitter.com",
    "m.twitter.com",
    "fxtwitter.com",
    "www.fxtwitter.com",
    "vxtwitter.com",
    "www.vxtwitter.com",
    "fixupx.com",
    "www.fixupx.com",
    "fixvx.com",
    "www.fixvx.com",
    "twittpr.com",
    "www.twittpr.com",
    "pxtwitter.com",
    "www.pxtwitter.com",
    "tweetpik.com",
    "www.tweetpik.com",
];

pub(crate) fn parse_status_identity(raw_url: &str) -> Result<XStatusIdentity> {
    if raw_url.trim().is_empty() {
        return Err(anyhow!("Empty URL provided for Twitter extraction"));
    }
    let trimmed = raw_url.trim();
    let parsed = Url::parse(trimmed).or_else(|_| Url::parse(&format!("https://{trimmed}")))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(anyhow!("Twitter/X URL must use HTTP or HTTPS"));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(anyhow!("Twitter/X URL must not contain credentials"));
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| anyhow!("Twitter/X URL has no host"))?
        .to_ascii_lowercase();
    if !SUPPORTED_HOSTS.contains(&host.as_str()) {
        return Err(anyhow!("Unsupported Twitter/X host: {host}"));
    }
    let segments = parsed
        .path_segments()
        .ok_or_else(|| anyhow!("Twitter/X URL has no path"))?
        .collect::<Vec<_>>();
    let status_index = segments
        .iter()
        .position(|segment| *segment == "status")
        .ok_or_else(|| anyhow!("Twitter/X URL does not reference a status update"))?;
    let id = segments
        .get(status_index + 1)
        .filter(|segment| !segment.is_empty() && segment.bytes().all(|byte| byte.is_ascii_digit()))
        .ok_or_else(|| anyhow!("Twitter/X status ID must contain ASCII digits only"))?
        .to_string();
    let canonical_url = Url::parse(&format!("https://x.com/i/status/{id}"))?;
    Ok(XStatusIdentity { id, canonical_url })
}

#[allow(dead_code)]
pub(crate) fn canonical_status_key(raw_url: &str) -> Result<String> {
    Ok(parse_status_identity(raw_url)?.id)
}

#[allow(dead_code)]
pub(crate) fn is_supported_status_url(raw_url: &str) -> bool {
    parse_status_identity(raw_url).is_ok()
}
