//! Deterministic HTTP emulation over explicitly registered routes.
//!
//! Simulated programs never acquire sockets, DNS, or host-network access. HTTP clients submit a
//! typed [`HttpRequest`] to [`VirtualNet`], which returns a configured [`HttpResponse`], a host
//! wait, or a deterministic error. Routes match an optional method and an exact URL or small
//! `*` glob.
//! Static response bodies are bounded at registration; VFS-backed bodies are read only from the
//! simulated filesystem at request time. A host-installed exact URL queues bounded guest HTTP
//! requests for explicit harness completion; it never grants ambient host network access.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::vfs::Vfs;

/// Maximum retained request records. Further attempts are counted without growing memory.
pub const MAX_REQUEST_LOG: usize = 4_096;
/// Maximum static routes in one environment.
pub const MAX_ROUTES: usize = 1_024;
/// Maximum body size accepted for one host-configured static response.
pub const MAX_STATIC_BODY_BYTES: usize = 8 * 1024 * 1024;
/// Maximum combined route patterns, headers, paths, and static bodies.
pub const MAX_ROUTE_BYTES: usize = 64 * 1024 * 1024;
const MAX_REQUEST_FIELD_BYTES: usize = 8 * 1024;
const MAX_ROUTE_HEADERS: usize = 128;
const MAX_ROUTE_HEADER_BYTES: usize = 64 * 1024;
const MAX_LOGGED_HEADERS: usize = 16;
const MAX_LOGGED_HEADER_FIELD_BYTES: usize = 512;
const MAX_HOST_REQUESTS: usize = 16;
const MAX_HOST_MESSAGE_BYTES: usize = 1024 * 1024;

/// Exact virtual URL reserved for host-provided tools. It never resolves through DNS.
pub const HOST_TOOLS_URL: &str = "http://host.shellsim/tools";

/// The single JSON envelope accepted by the host-backed virtual route.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolCall {
    pub tool: String,
    pub arguments: serde_json::Map<String, serde_json::Value>,
}

/// One HTTP request made entirely inside the simulated environment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpRequest {
    pub method: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// One host-bound virtual request, delivered at most once to the harness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostHttpRequest {
    pub id: u64,
    pub request: HttpRequest,
    pub call: ToolCall,
}

/// One guest HTTP operation either has its response or awaits explicit host completion.
pub enum HttpPoll {
    Ready(HttpResponse),
    Blocked(u64),
}

#[derive(Clone)]
struct HostRequestSlot {
    request: HostHttpRequest,
    process_group: u32,
    delivered: bool,
    response: Option<HttpResponse>,
    log_index: Option<usize>,
}

impl HttpRequest {
    /// Construct a request without acquiring any ambient networking capability.
    pub fn new(method: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            url: url.into(),
            headers: Vec::new(),
            body: Vec::new(),
        }
    }
}

/// One deterministic HTTP response supplied by a configured route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    pub fn ok(body: impl Into<Vec<u8>>) -> Self {
        Self {
            status: 200,
            headers: Vec::new(),
            body: body.into(),
        }
    }
}

/// One bounded record of a virtual HTTP request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct NetworkRequest {
    /// Bounded HTTP method supplied by the simulated client.
    pub method: String,
    /// Bounded URL supplied by the simulated client.
    pub url: String,
    /// First bounded request headers, in client order.
    pub headers: Vec<(String, String)>,
    /// Number of request headers omitted from this telemetry record.
    pub dropped_headers: u64,
    /// Exact request body length without retaining another copy of the body.
    pub body_bytes: u64,
    /// Whether the request resolved to a configured virtual route.
    pub matched: bool,
    /// Configured response status when the route matched.
    pub response_status: Option<u16>,
}

/// A route could not be registered without crossing a documented resource or syntax boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteError {
    EmptyPattern,
    PatternTooLong,
    InvalidMethod,
    InvalidStatus(u16),
    TooManyHeaders,
    HeadersTooLarge,
    InvalidHeader,
    BodyTooLarge,
    RouteDataTooLarge,
    TooManyRoutes,
}

impl fmt::Display for RouteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyPattern => "URL pattern cannot be empty",
            Self::PatternTooLong => "URL pattern exceeds 8192 bytes",
            Self::InvalidMethod => "HTTP method must be a non-empty ASCII token",
            Self::InvalidStatus(status) => {
                return write!(formatter, "invalid HTTP status {status}")
            }
            Self::TooManyHeaders => "HTTP response has more than 128 headers",
            Self::HeadersTooLarge => "HTTP response headers exceed 65536 bytes",
            Self::InvalidHeader => "HTTP response header contains an invalid name or newline",
            Self::BodyTooLarge => "static HTTP response body exceeds 8388608 bytes",
            Self::RouteDataTooLarge => "virtual HTTP route data exceeds 67108864 bytes",
            Self::TooManyRoutes => "virtual HTTP route limit of 1024 reached",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for RouteError {}

/// Failure to resolve or materialize one virtual request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequestError {
    NoRoute,
    UnreadableVfsBody { path: String, error: String },
    HostRejected(String),
}

impl fmt::Display for RequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRoute => formatter.write_str("no matching virtual HTTP route"),
            Self::UnreadableVfsBody { path, error } => {
                write!(formatter, "cannot read virtual HTTP body {path}: {error}")
            }
            Self::HostRejected(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for RequestError {}

#[derive(Clone, Debug)]
enum RouteBody {
    Static(Vec<u8>),
    VfsFile(String),
}

#[derive(Clone, Debug)]
struct Route {
    method: Option<String>,
    pattern: String,
    status: u16,
    headers: Vec<(String, String)>,
    body: RouteBody,
}

/// Route table and bounded observability state for deterministic HTTP emulation.
#[derive(Clone, Default)]
pub struct VirtualNet {
    routes: Vec<Route>,
    route_bytes: usize,
    /// Request log for debugging, assertions, and reward shaping.
    pub log: Vec<NetworkRequest>,
    /// Attempts omitted after [`MAX_REQUEST_LOG`] records were retained.
    pub dropped_requests: u64,
    /// Arbitrary host:port services that tests may mark as listening.
    pub listening: HashMap<String, bool>,
    host_tools_enabled: bool,
    next_host_request_id: u64,
    host_requests: BTreeMap<u32, HostRequestSlot>,
}

impl VirtualNet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Install the one host-backed virtual endpoint for this machine.
    pub fn enable_host_tools(&mut self) {
        self.host_tools_enabled = true;
    }

    /// Whether the trusted host installed the exact tool transport URL.
    pub fn host_tools_enabled(&self) -> bool {
        self.host_tools_enabled
    }

    /// Dispatch guest HTTP through the route table or the exact host-backed endpoint.
    ///
    /// A pending host reply is reported as [`HttpPoll::Blocked`], so each guest execution model
    /// can suspend on the same scheduler resource and retry this request.
    pub fn request_for_process(
        &mut self,
        pid: u32,
        process_group: u32,
        request: HttpRequest,
        vfs: &Vfs,
    ) -> Result<HttpPoll, RequestError> {
        if self.host_tools_enabled && request.url == HOST_TOOLS_URL {
            return self
                .host_tool_request(pid, process_group, request)
                .map_err(RequestError::HostRejected);
        }
        self.request(request, vfs).map(HttpPoll::Ready)
    }

    /// Queue or resume a guest POST to the host endpoint without resending it.
    pub fn host_tool_request(
        &mut self,
        pid: u32,
        process_group: u32,
        request: HttpRequest,
    ) -> Result<HttpPoll, String> {
        if !self.host_tools_enabled || request.url != HOST_TOOLS_URL {
            return Err("host tools endpoint is not enabled".into());
        }
        if request.method != "POST" {
            let response = HttpResponse {
                status: 405,
                headers: vec![("Allow".into(), "POST".into())],
                body: Vec::new(),
            };
            let log_index = (self.log.len() < MAX_REQUEST_LOG).then_some(self.log.len());
            self.record_request(&request, None);
            if let Some(index) = log_index {
                self.log[index].matched = true;
                self.log[index].response_status = Some(response.status);
            }
            return Ok(HttpPoll::Ready(response));
        }
        if request.body.len() > MAX_HOST_MESSAGE_BYTES {
            return Err("host request body exceeds 1 MiB".into());
        }
        let call: ToolCall = serde_json::from_slice(&request.body).map_err(|_| {
            "host tool request must contain a tool and object arguments".to_string()
        })?;
        if call.tool.is_empty() || call.tool.len() > MAX_REQUEST_FIELD_BYTES {
            return Err("host tool name is empty or too long".into());
        }
        if request.headers.len() > MAX_ROUTE_HEADERS
            || request
                .headers
                .iter()
                .try_fold(0usize, |bytes, (name, value)| {
                    bytes.checked_add(name.len())?.checked_add(value.len())
                })
                .is_none_or(|bytes| bytes > MAX_ROUTE_HEADER_BYTES)
        {
            return Err("host request headers exceed their limit".into());
        }
        if let Some(slot) = self.host_requests.get(&pid) {
            if slot.request.request != request {
                return Err("a different host request is already pending for this process".into());
            }
            if let Some(response) = &slot.response {
                let response = response.clone();
                self.host_requests.remove(&pid);
                return Ok(HttpPoll::Ready(response));
            }
            return Ok(HttpPoll::Blocked(slot.request.id));
        }
        if self.host_requests.len() >= MAX_HOST_REQUESTS {
            return Err("host request limit reached".into());
        }
        let id = self.next_host_request_id;
        self.next_host_request_id = id
            .checked_add(1)
            .ok_or_else(|| "host request identifier space exhausted".to_string())?;
        let log_index = (self.log.len() < MAX_REQUEST_LOG).then_some(self.log.len());
        self.record_request(&request, None);
        if let Some(index) = log_index {
            self.log[index].matched = true;
        }
        self.host_requests.insert(
            pid,
            HostRequestSlot {
                request: HostHttpRequest { id, request, call },
                process_group,
                delivered: false,
                response: None,
                log_index,
            },
        );
        Ok(HttpPoll::Blocked(id))
    }

    /// Drain requests not yet delivered to the host; repeated reads do not duplicate calls.
    pub fn take_host_requests(&mut self) -> Vec<HostHttpRequest> {
        self.host_requests
            .values_mut()
            .filter(|slot| !slot.delivered)
            .map(|slot| {
                slot.delivered = true;
                slot.request.clone()
            })
            .collect()
    }

    /// Complete exactly one delivered request and return the process to wake.
    pub fn complete_host_request(
        &mut self,
        id: u64,
        response: HttpResponse,
    ) -> Result<u32, String> {
        validate_route(HOST_TOOLS_URL, Some("POST"), &response)
            .map_err(|error| error.to_string())?;
        if response.body.len() > MAX_HOST_MESSAGE_BYTES {
            return Err("host response body exceeds 1 MiB".into());
        }
        let (pid, slot) = self
            .host_requests
            .iter_mut()
            .find(|(_, slot)| slot.request.id == id)
            .ok_or_else(|| format!("host request {id} does not exist"))?;
        if !slot.delivered || slot.response.is_some() {
            return Err(format!("host request {id} cannot be completed"));
        }
        if let Some(index) = slot.log_index {
            self.log[index].response_status = Some(response.status);
        }
        slot.response = Some(response);
        Ok(*pid)
    }

    /// Discard requests whose caller has exited, so cancellation cannot leave stale replies.
    pub fn retain_host_requests(&mut self, mut process_is_live: impl FnMut(u32) -> bool) {
        self.host_requests.retain(|pid, _| process_is_live(*pid));
    }

    /// Invalidate all in-flight calls owned by a cancelled foreground process group.
    pub fn discard_host_requests_for_group(&mut self, process_group: u32) {
        self.host_requests
            .retain(|_, slot| slot.process_group != process_group);
    }

    /// Register a static response for an exact URL or `*` glob.
    ///
    /// `method=None` matches every method. Configuration is rejected atomically when any field or
    /// the route table exceeds its documented bound.
    pub fn route(
        &mut self,
        pattern: &str,
        method: Option<&str>,
        response: HttpResponse,
    ) -> Result<(), RouteError> {
        let route_bytes = validate_route(pattern, method, &response)?;
        if self.routes.len() >= MAX_ROUTES {
            return Err(RouteError::TooManyRoutes);
        }
        let new_route_bytes = self
            .route_bytes
            .checked_add(route_bytes)
            .filter(|bytes| *bytes <= MAX_ROUTE_BYTES)
            .ok_or(RouteError::RouteDataTooLarge)?;
        self.routes.push(Route {
            method: method.map(str::to_ascii_uppercase),
            pattern: pattern.to_string(),
            status: response.status,
            headers: response.headers,
            body: RouteBody::Static(response.body),
        });
        self.route_bytes = new_route_bytes;
        Ok(())
    }

    /// Convenience for a static response without headers or a method restriction.
    pub fn route_static(
        &mut self,
        pattern: &str,
        status: u16,
        body: impl Into<Vec<u8>>,
    ) -> Result<(), RouteError> {
        self.route(
            pattern,
            None,
            HttpResponse {
                status,
                headers: Vec::new(),
                body: body.into(),
            },
        )
    }

    /// Register a response whose body is read from the VFS for every request.
    pub fn route_vfs(&mut self, pattern: &str, vfs_path: &str) -> Result<(), RouteError> {
        let route_bytes = validate_route(pattern, None, &HttpResponse::ok(Vec::new()))?
            .checked_add(vfs_path.len())
            .ok_or(RouteError::RouteDataTooLarge)?;
        if self.routes.len() >= MAX_ROUTES {
            return Err(RouteError::TooManyRoutes);
        }
        let new_route_bytes = self
            .route_bytes
            .checked_add(route_bytes)
            .filter(|bytes| *bytes <= MAX_ROUTE_BYTES)
            .ok_or(RouteError::RouteDataTooLarge)?;
        self.routes.push(Route {
            method: None,
            pattern: pattern.to_string(),
            status: 200,
            headers: Vec::new(),
            body: RouteBody::VfsFile(vfs_path.to_string()),
        });
        self.route_bytes = new_route_bytes;
        Ok(())
    }

    pub fn listen(&mut self, host_port: &str) {
        self.listening.insert(host_port.to_string(), true);
    }

    /// Resolve and materialize one request without consulting the host network.
    pub fn request(
        &mut self,
        request: HttpRequest,
        vfs: &Vfs,
    ) -> Result<HttpResponse, RequestError> {
        let route = self
            .routes
            .iter()
            .find(|route| {
                route
                    .method
                    .as_ref()
                    .is_none_or(|method| method.eq_ignore_ascii_case(&request.method))
                    && pattern_matches(&route.pattern, &request.url)
            })
            .cloned();
        self.record_request(&request, route.as_ref());
        let Some(route) = route else {
            return Err(RequestError::NoRoute);
        };
        let body = match route.body {
            RouteBody::Static(body) => body,
            RouteBody::VfsFile(path) => {
                vfs.read("/", &path)
                    .map_err(|error| RequestError::UnreadableVfsBody {
                        path,
                        error: error.to_string(),
                    })?
            }
        };
        Ok(HttpResponse {
            status: route.status,
            headers: route.headers,
            body,
        })
    }

    fn record_request(&mut self, request: &HttpRequest, route: Option<&Route>) {
        if self.log.len() >= MAX_REQUEST_LOG {
            self.dropped_requests = self.dropped_requests.saturating_add(1);
            return;
        }
        let headers = request
            .headers
            .iter()
            .take(MAX_LOGGED_HEADERS)
            .map(|(name, value)| {
                (
                    bounded_to(name, MAX_LOGGED_HEADER_FIELD_BYTES),
                    bounded_to(value, MAX_LOGGED_HEADER_FIELD_BYTES),
                )
            })
            .collect();
        self.log.push(NetworkRequest {
            method: bounded_to(&request.method, MAX_REQUEST_FIELD_BYTES),
            url: bounded_to(&request.url, MAX_REQUEST_FIELD_BYTES),
            headers,
            dropped_headers: u64::try_from(
                request.headers.len().saturating_sub(MAX_LOGGED_HEADERS),
            )
            .unwrap_or(u64::MAX),
            body_bytes: u64::try_from(request.body.len()).unwrap_or(u64::MAX),
            matched: route.is_some(),
            response_status: route.map(|route| route.status),
        });
    }
}

fn validate_route(
    pattern: &str,
    method: Option<&str>,
    response: &HttpResponse,
) -> Result<usize, RouteError> {
    if pattern.is_empty() {
        return Err(RouteError::EmptyPattern);
    }
    if pattern.len() > MAX_REQUEST_FIELD_BYTES {
        return Err(RouteError::PatternTooLong);
    }
    if method.is_some_and(|method| !is_http_token(method)) {
        return Err(RouteError::InvalidMethod);
    }
    if !(100..=599).contains(&response.status) {
        return Err(RouteError::InvalidStatus(response.status));
    }
    if response.headers.len() > MAX_ROUTE_HEADERS {
        return Err(RouteError::TooManyHeaders);
    }
    let mut header_bytes = 0usize;
    for (name, value) in &response.headers {
        if !is_http_token(name) || value.contains(['\r', '\n']) {
            return Err(RouteError::InvalidHeader);
        }
        header_bytes = header_bytes
            .checked_add(name.len())
            .and_then(|size| size.checked_add(value.len()))
            .ok_or(RouteError::HeadersTooLarge)?;
        if header_bytes > MAX_ROUTE_HEADER_BYTES {
            return Err(RouteError::HeadersTooLarge);
        }
    }
    if response.body.len() > MAX_STATIC_BODY_BYTES {
        return Err(RouteError::BodyTooLarge);
    }
    pattern
        .len()
        .checked_add(method.map_or(0, str::len))
        .and_then(|bytes| bytes.checked_add(header_bytes))
        .and_then(|bytes| bytes.checked_add(response.body.len()))
        .ok_or(RouteError::RouteDataTooLarge)
}

fn is_http_token(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(
                    byte,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

fn bounded_to(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    value
        .chars()
        .scan(0usize, |bytes, character| {
            *bytes = bytes.saturating_add(character.len_utf8());
            (*bytes <= max_bytes).then_some(character)
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
    for (index, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if index == 0 {
            if !text[pos..].starts_with(part) {
                return false;
            }
            pos += part.len();
        } else if index == parts.len() - 1 {
            if !text[pos..].ends_with(part) {
                return false;
            }
        } else if let Some(found) = text[pos..].find(part) {
            pos += found + part.len();
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
    fn host_endpoint_bounds_messages_and_retries_without_redelivery() {
        let mut net = VirtualNet::new();
        let mut request = HttpRequest::new("POST", HOST_TOOLS_URL);
        request.body = br#"{"tool":"workspace.read_file","arguments":{"path":"/work/a"}}"#.to_vec();
        assert!(net.host_tool_request(7, 7, request.clone()).is_err());
        net.enable_host_tools();
        let oversized = HttpRequest {
            body: vec![0; MAX_HOST_MESSAGE_BYTES + 1],
            ..request.clone()
        };
        assert!(net.host_tool_request(7, 7, oversized).is_err());
        let invalid = HttpRequest {
            body: br#"{"tool":"workspace.read_file","arguments":[]}"#.to_vec(),
            ..request.clone()
        };
        assert!(net.host_tool_request(7, 7, invalid).is_err());
        let HttpPoll::Blocked(id) = net.host_tool_request(7, 7, request.clone()).unwrap() else {
            panic!("host request did not block");
        };
        assert!(matches!(
            net.host_tool_request(7, 7, request.clone()),
            Ok(HttpPoll::Blocked(pending)) if pending == id
        ));
        assert_eq!(net.take_host_requests().len(), 1);
        assert!(net.take_host_requests().is_empty());
        assert!(net
            .complete_host_request(id, HttpResponse::ok(vec![0; MAX_HOST_MESSAGE_BYTES + 1]))
            .is_err());
        net.complete_host_request(id, HttpResponse::ok("done"))
            .unwrap();
        assert_eq!(net.log[0].response_status, Some(200));
        assert!(matches!(
            net.host_tool_request(7, 7, request),
            Ok(HttpPoll::Ready(response)) if response.body == b"done"
        ));
        assert!(net.take_host_requests().is_empty());
    }

    #[test]
    fn typed_request_selects_method_and_records_shape() {
        let mut net = VirtualNet::new();
        net.route(
            "https://example.test/items",
            Some("POST"),
            HttpResponse {
                status: 201,
                headers: vec![("Content-Type".into(), "application/json".into())],
                body: br#"{"ok":true}"#.to_vec(),
            },
        )
        .unwrap();
        let mut request = HttpRequest::new("POST", "https://example.test/items");
        request.headers.push(("X-Test".into(), "yes".into()));
        request.body = b"payload".to_vec();

        let response = net.request(request, &Vfs::new()).unwrap();

        assert_eq!(response.status, 201);
        assert_eq!(response.body, br#"{"ok":true}"#);
        assert_eq!(net.log[0].headers, [("X-Test".into(), "yes".into())]);
        assert_eq!(net.log[0].body_bytes, 7);
        assert_eq!(net.log[0].response_status, Some(201));
        assert!(matches!(
            net.request(
                HttpRequest::new("GET", "https://example.test/items"),
                &Vfs::new()
            ),
            Err(RequestError::NoRoute)
        ));
    }

    #[test]
    fn invalid_and_excessive_static_routes_are_rejected() {
        let mut net = VirtualNet::new();
        assert_eq!(
            net.route("", None, HttpResponse::ok(Vec::new())),
            Err(RouteError::EmptyPattern)
        );
        assert_eq!(
            net.route(
                "https://example.test",
                Some("bad method"),
                HttpResponse::ok(Vec::new())
            ),
            Err(RouteError::InvalidMethod)
        );
        assert_eq!(
            net.route(
                "https://example.test",
                None,
                HttpResponse {
                    status: 200,
                    headers: vec![("X-Test".into(), "bad\r\nvalue".into())],
                    body: Vec::new(),
                }
            ),
            Err(RouteError::InvalidHeader)
        );
        assert_eq!(
            net.route_static(
                "https://example.test",
                200,
                vec![0; MAX_STATIC_BODY_BYTES + 1]
            ),
            Err(RouteError::BodyTooLarge)
        );

        for index in 0..MAX_ROUTES {
            net.route_static(&format!("https://example.test/{index}"), 200, Vec::new())
                .unwrap();
        }
        assert_eq!(
            net.route_static("https://example.test/overflow", 200, Vec::new()),
            Err(RouteError::TooManyRoutes)
        );

        let mut net = VirtualNet::new();
        net.route_bytes = MAX_ROUTE_BYTES;
        assert_eq!(
            net.route_static("https://large.test/overflow", 200, b"x"),
            Err(RouteError::RouteDataTooLarge)
        );
    }

    #[test]
    fn request_log_is_bounded_and_counts_overflow() {
        let mut net = VirtualNet::new();
        for index in 0..MAX_REQUEST_LOG + 3 {
            assert!(matches!(
                net.request(
                    HttpRequest::new("GET", format!("https://invalid/{index}")),
                    &Vfs::new()
                ),
                Err(RequestError::NoRoute)
            ));
        }
        assert_eq!(net.log.len(), MAX_REQUEST_LOG);
        assert_eq!(net.dropped_requests, 3);
    }
}
