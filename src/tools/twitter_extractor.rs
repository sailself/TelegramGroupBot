use anyhow::{anyhow, Result};

use crate::utils::http::get_http_client_no_redirect;

pub(crate) mod model;
pub(crate) mod providers;
#[cfg(test)]
pub(crate) mod test_support;
pub(crate) mod url;
pub use model::TwitterContent;
#[allow(unused_imports)]
pub(crate) use model::VideoThumbnailFallback;
#[allow(unused_imports)]
pub(crate) use url::{
    canonical_status_key, is_supported_status_url, parse_status_identity, XStatusIdentity,
};

pub async fn extract_twitter_content(url: &str) -> Result<TwitterContent> {
    let identity = parse_status_identity(url)?;
    let config = crate::tools::twitter_extractor::providers::TwitterFetchConfig::try_from(
        &*crate::config::CONFIG,
    )?;
    let post = crate::tools::twitter_extractor::providers::jina::fetch(
        get_http_client_no_redirect(),
        &config,
        &identity,
        config.provider_timeout,
    )
    .await
    .map_err(|error| anyhow!(error.detail))?;
    crate::tools::twitter_extractor::model::build_twitter_content(&identity, post)
}
