//! Metered Python access to shellsim's deterministic HTTP broker.
//!
//! This is the only Python capability that reaches [`crate::net::VirtualNet`]. Requests and
//! responses are owned, size-bounded values; missing routes stay distinguishable from malformed
//! requests, and no error path can acquire ambient network access.

use crate::interp::Interp;
use crate::net::{HttpRequest, RequestError};

use super::native::{PyError, PyHttpClient, PyHttpRequest, PyHttpResponse, PyResult};

const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
const MAX_HEADERS: usize = 128;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_URL_BYTES: usize = 8 * 1024;

impl PyHttpClient for Interp {
    fn request(&mut self, request: PyHttpRequest) -> PyResult<Option<PyHttpResponse>> {
        let work = validate_request(&request)?;
        let work = u64::try_from(work).unwrap_or(u64::MAX);
        if !self.resources.charge_cpu(work) {
            return Err(PyError::resource_error("Python CPU limit exceeded"));
        }
        let request = HttpRequest {
            method: request.method,
            url: request.url,
            headers: request.headers,
            body: request.body,
        };
        let response = match self.net.request(request, &self.vfs) {
            Ok(response) => response,
            Err(RequestError::NoRoute) => return Ok(None),
            Err(error) => return Err(PyError::exception("OSError", error.to_string())),
        };
        if response.body.len() > MAX_BODY_BYTES {
            return Err(PyError::resource_error(
                "virtual HTTP response body exceeds the 8 MiB Python limit",
            ));
        }
        let response_bytes = response
            .headers
            .iter()
            .try_fold(response.body.len(), |total, (name, value)| {
                total.checked_add(name.len())?.checked_add(value.len())
            })
            .ok_or_else(|| PyError::resource_error("virtual HTTP response is too large"))?;
        let response_bytes = u64::try_from(response_bytes).unwrap_or(u64::MAX);
        if !self.resources.charge_cpu(response_bytes) {
            return Err(PyError::resource_error("Python CPU limit exceeded"));
        }
        Ok(Some(PyHttpResponse {
            status: response.status,
            headers: response.headers,
            body: response.body,
        }))
    }
}

fn validate_request(request: &PyHttpRequest) -> PyResult<usize> {
    if request.method.is_empty()
        || !request
            .method
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte == b'-')
    {
        return Err(PyError::value_error(
            "HTTP method must be a non-empty uppercase ASCII token",
        ));
    }
    if request.url.len() > MAX_URL_BYTES {
        return Err(PyError::value_error("HTTP URL exceeds 8192 bytes"));
    }
    if !request.url.starts_with("http://") && !request.url.starts_with("https://") {
        return Err(PyError::value_error(
            "only http:// and https:// URLs are supported",
        ));
    }
    if request.headers.len() > MAX_HEADERS {
        return Err(PyError::value_error(
            "HTTP request has more than 128 headers",
        ));
    }
    if request.body.len() > MAX_BODY_BYTES {
        return Err(PyError::resource_error(
            "virtual HTTP request body exceeds the 8 MiB Python limit",
        ));
    }
    let mut bytes = request
        .method
        .len()
        .checked_add(request.url.len())
        .and_then(|size| size.checked_add(request.body.len()))
        .ok_or_else(|| PyError::resource_error("virtual HTTP request is too large"))?;
    let mut header_bytes = 0usize;
    for (name, value) in &request.headers {
        if name.is_empty() || !name.bytes().all(is_header_name_byte) || value.contains(['\r', '\n'])
        {
            return Err(PyError::value_error("invalid HTTP request header"));
        }
        header_bytes = header_bytes
            .checked_add(name.len())
            .and_then(|size| size.checked_add(value.len()))
            .ok_or_else(|| PyError::resource_error("HTTP request headers are too large"))?;
        if header_bytes > MAX_HEADER_BYTES {
            return Err(PyError::value_error(
                "HTTP request headers exceed 65536 bytes",
            ));
        }
    }
    bytes = bytes
        .checked_add(header_bytes)
        .ok_or_else(|| PyError::resource_error("virtual HTTP request is too large"))?;
    Ok(bytes)
}

fn is_header_name_byte(byte: u8) -> bool {
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
}
