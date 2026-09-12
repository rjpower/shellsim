//! Virtual network.
//!
//! No real egress ever happens. `curl`/`wget` (and, later, simulated servers) resolve
//! requests against a registered route table. Routes can be exact URLs or simple
//! prefix/glob patterns, and may return a static body or be backed by a file in the VFS.
//! Unmatched requests fail deterministically (like a connection refused), which is exactly
//! what you want for a reproducible RL environment.

use std::collections::HashMap;

use serde::Serialize;

/// Maximum retained request records. Further attempts are counted without growing memory.
pub const MAX_REQUEST_LOG: usize = 4_096;
const MAX_REQUEST_FIELD_BYTES: usize = 8 * 1024;

/// One deterministic virtual-network attempt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct NetworkRequest {
    /// Bounded HTTP method supplied by the simulated command.
    pub method: String,
    /// Bounded URL supplied by the simulated command.
    pub url: String,
    /// Whether the request resolved to a configured virtual route.
    pub matched: bool,
}

#[derive(Clone, Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    pub fn ok(body: impl Into<Vec<u8>>) -> Self {
        HttpResponse {
            status: 200,
            headers: vec![],
            body: body.into(),
        }
    }
    pub fn not_found() -> Self {
        HttpResponse {
            status: 404,
            headers: vec![],
            body: b"Not Found".to_vec(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum RouteBody {
    /// literal response bytes
    Static(Vec<u8>),
    /// serve the contents of a VFS path at request time
    VfsFile(String),
}

#[derive(Clone, Debug)]
pub struct Route {
    pub method: Option<String>,
    pub pattern: String,
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: RouteBody,
}

#[derive(Clone, Default)]
pub struct VirtualNet {
    routes: Vec<Route>,
    /// request log for debugging / reward shaping
    pub log: Vec<NetworkRequest>,
    /// Attempts omitted after [`MAX_REQUEST_LOG`] records were retained.
    pub dropped_requests: u64,
    /// arbitrary host:port "services" that tests may probe (registered as up/down)
    pub listening: HashMap<String, bool>,
}

impl VirtualNet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_route(&mut self, route: Route) {
        self.routes.push(route);
    }

    /// Convenience: register a static GET response for an exact url or `*`-glob pattern.
    pub fn route_static(&mut self, pattern: &str, status: u16, body: impl Into<Vec<u8>>) {
        self.routes.push(Route {
            method: None,
            pattern: pattern.to_string(),
            status,
            headers: vec![],
            body: RouteBody::Static(body.into()),
        });
    }

    pub fn route_vfs(&mut self, pattern: &str, vfs_path: &str) {
        self.routes.push(Route {
            method: None,
            pattern: pattern.to_string(),
            status: 200,
            headers: vec![],
            body: RouteBody::VfsFile(vfs_path.to_string()),
        });
    }

    pub fn listen(&mut self, host_port: &str) {
        self.listening.insert(host_port.to_string(), true);
    }

    /// Match a request. Returns the matching route's static parts; VfsFile bodies must be
    /// resolved by the caller against the VFS.
    pub fn resolve(&mut self, method: &str, url: &str) -> Option<Route> {
        let route = self.routes.iter().find(|r| {
            if let Some(m) = &r.method {
                if !m.eq_ignore_ascii_case(method) {
                    return false;
                }
            }
            pattern_matches(&r.pattern, url)
        });
        if self.log.len() < MAX_REQUEST_LOG {
            self.log.push(NetworkRequest {
                method: bounded_field(method),
                url: bounded_field(url),
                matched: route.is_some(),
            });
        } else {
            self.dropped_requests = self.dropped_requests.saturating_add(1);
        }
        route.cloned()
    }
}

fn bounded_field(value: &str) -> String {
    if value.len() <= MAX_REQUEST_FIELD_BYTES {
        return value.to_string();
    }
    value
        .chars()
        .scan(0usize, |bytes, character| {
            *bytes = bytes.saturating_add(character.len_utf8());
            (*bytes <= MAX_REQUEST_FIELD_BYTES).then_some(character)
        })
        .collect()
}

/// Very small glob matcher: `*` matches any run of characters. Anchored at both ends.
fn pattern_matches(pattern: &str, text: &str) -> bool {
    if pattern == text {
        return true;
    }
    if !pattern.contains('*') {
        return false;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    let mut pos = 0usize;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if i == 0 {
            if !text[pos..].starts_with(part) {
                return false;
            }
            pos += part.len();
        } else if i == parts.len() - 1 {
            if !text[pos..].ends_with(part) {
                return false;
            }
        } else if let Some(idx) = text[pos..].find(part) {
            pos += idx + part.len();
        } else {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_matching() {
        assert!(pattern_matches("http://x/*", "http://x/abc"));
        assert!(pattern_matches("*/health", "http://localhost:8080/health"));
        assert!(pattern_matches("https://api/*/v1", "https://api/foo/v1"));
        assert!(!pattern_matches("https://api/*/v1", "https://api/foo/v2"));
        assert!(pattern_matches("exact", "exact"));
    }

    #[test]
    fn route_resolution() {
        let mut net = VirtualNet::new();
        net.route_static("https://example.com/data.json", 200, b"{\"k\":1}".to_vec());
        let r = net.resolve("GET", "https://example.com/data.json").unwrap();
        assert_eq!(r.status, 200);
        assert!(net.resolve("GET", "https://nope.com").is_none());
        assert!(net.log[0].matched);
        assert!(!net.log[1].matched);
    }

    #[test]
    fn request_log_is_bounded_and_counts_overflow() {
        let mut net = VirtualNet::new();
        for index in 0..MAX_REQUEST_LOG + 3 {
            assert!(net
                .resolve("GET", &format!("https://invalid/{index}"))
                .is_none());
        }
        assert_eq!(net.log.len(), MAX_REQUEST_LOG);
        assert_eq!(net.dropped_requests, 3);
    }
}
