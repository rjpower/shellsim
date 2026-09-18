//! Single native request primitive for frozen Python HTTP clients.
//!
//! Public API policy, response objects, and exception classes live in ordinary frozen Python.
//! This module only converts checked values to and from the explicit [`PyHttpClient`] capability.

use super::super::native::{
    CallArgs, FunctionDef, ModuleDef, OwnedPyString, PyBytes, PyResult, PyRuntime, PySequence,
    PyValueCast,
};
use super::super::Value;
use crate::python::native::{PyHttpRequest, PyValue};

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "_shellsim_http",
    functions: &[FunctionDef {
        module: "_shellsim_http",
        name: "request",
        call: request,
    }],
    values: &[],
};

fn request(runtime: &mut dyn PyRuntime, args: CallArgs) -> PyResult {
    args.expect_positional("_shellsim_http.request", 4, 4)?;
    args.reject_keywords("_shellsim_http.request")?;
    let values = args.positional();
    let OwnedPyString(method) = values[0].cast(runtime)?;
    let OwnedPyString(url) = values[1].cast(runtime)?;
    let header_values: PySequence = values[2].cast(runtime)?;
    let mut headers = Vec::new();
    for header in header_values.items(runtime)? {
        let pair: PySequence = header.cast(runtime)?;
        let pair = pair.items(runtime)?;
        if pair.len() != 2 {
            return Err(super::super::native::PyError::type_error(
                "HTTP headers must contain name/value pairs",
            ));
        }
        let OwnedPyString(name) = pair[0].cast(runtime)?;
        let OwnedPyString(value) = pair[1].cast(runtime)?;
        headers.push((name, value));
    }
    let PyBytes(body) = values[3].cast(runtime)?;
    let Some(response) = runtime.http().request(PyHttpRequest {
        method,
        url,
        headers,
        body,
    })?
    else {
        return Ok(Value::None);
    };

    let mut headers = Vec::<PyValue>::with_capacity(response.headers.len());
    for (name, value) in response.headers {
        let name = runtime.new_string(name)?;
        let value = runtime.new_string(value)?;
        headers.push(runtime.new_tuple(vec![name, value])?);
    }
    let headers = runtime.new_list(headers)?;
    let body = runtime.new_bytes(response.body)?;
    runtime.new_tuple(vec![Value::Int(i64::from(response.status)), headers, body])
}
