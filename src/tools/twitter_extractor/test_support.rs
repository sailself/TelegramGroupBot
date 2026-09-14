use super::url::XStatusIdentity;
use url::Url;

/// A status identity fixture: `id` as both the numeric ID and the fake
/// canonical URL's path segment. Shared by every provider's test module
/// instead of each redefining its own copy.
pub(crate) fn identity(id: &str) -> XStatusIdentity {
    XStatusIdentity {
        id: id.to_owned(),
        canonical_url: Url::parse(&format!("https://x.com/i/status/{id}")).unwrap(),
    }
}
