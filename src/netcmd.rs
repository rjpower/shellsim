//! `curl` and `wget` clients for the typed virtual HTTP broker.
//!
//! Both clients use the shared option scanner and submit typed requests through the active
//! process's virtual-kernel interface. Their supported option tables are closed: unknown flags
//! fail instead of silently changing the request.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use crate::commands::options::{parse_options_or_report, OptionSpec};
use crate::net::{HttpRequest, HttpResponse, RequestError};
use crate::syscalls::System;

type Out<'a> = &'a mut Vec<u8>;

pub(crate) fn curl(system: &mut dyn System, args: &[String], out: Out, err: Out) -> i32 {
    #[derive(Clone, Copy, PartialEq)]
    enum Key {
        Request,
        Url,
        Output,
        RemoteName,
        Silent,
        ShowError,
        Include,
        Fail,
        Head,
        Data,
        Header,
        UserAgent,
        Referer,
        Cookie,
        User,
        Help,
    }
    const OPTIONS: &[OptionSpec<Key>] = &[
        OptionSpec::required(Key::Request, Some('X'), Some("request")),
        OptionSpec::required(Key::Url, None, Some("url")),
        OptionSpec::required(Key::Output, Some('o'), Some("output")),
        OptionSpec::flag(Key::RemoteName, Some('O'), Some("remote-name")),
        OptionSpec::flag(Key::Silent, Some('s'), Some("silent")),
        OptionSpec::flag(Key::ShowError, Some('S'), Some("show-error")),
        OptionSpec::flag(Key::Include, Some('i'), Some("include")),
        OptionSpec::flag(Key::Fail, Some('f'), Some("fail")),
        OptionSpec::flag(Key::Head, Some('I'), Some("head")),
        OptionSpec::required(Key::Data, Some('d'), Some("data")),
        OptionSpec::required(Key::Data, None, Some("data-raw")),
        OptionSpec::required(Key::Data, None, Some("data-binary")),
        OptionSpec::required(Key::Header, Some('H'), Some("header")),
        OptionSpec::required(Key::UserAgent, Some('A'), Some("user-agent")),
        OptionSpec::required(Key::Referer, Some('e'), Some("referer")),
        OptionSpec::required(Key::Cookie, Some('b'), Some("cookie")),
        OptionSpec::required(Key::User, Some('u'), Some("user")),
        OptionSpec::flag(Key::Help, None, Some("help")),
    ];
    let parsed = match parse_options_or_report(
        "curl",
        args,
        OPTIONS,
        (
            Key::Help,
            "usage: curl [OPTIONS] URL\nsupported: -X --url -o -O -sSifI -d -H -A -e -b -u\n",
        ),
        out,
        err,
    ) {
        Ok(parsed) => parsed,
        Err(status) => return status,
    };
    if parsed.operands.len() > 1 {
        ewln(err, "curl: multiple URLs are not supported");
        return 2;
    }

    let mut request = HttpRequest::new("GET", parsed.operands.first().cloned().unwrap_or_default());
    let mut output_file = None;
    let mut remote_name = false;
    let mut silent = false;
    let mut show_error = false;
    let mut include_headers = false;
    let mut fail = false;
    let mut head_only = false;
    for option in parsed.options {
        let value = || option.value.clone().expect("required option value");
        match option.key {
            Key::Request => request.method = value(),
            Key::Url => request.url = value(),
            Key::Output => output_file = Some(value()),
            Key::RemoteName => remote_name = true,
            Key::Silent => silent = true,
            Key::ShowError => show_error = true,
            Key::Include => include_headers = true,
            Key::Fail => fail = true,
            Key::Head => {
                head_only = true;
                request.method = "HEAD".into();
            }
            Key::Data => {
                if !request.body.is_empty() {
                    request.body.push(b'&');
                }
                request.body.extend_from_slice(value().as_bytes());
                if request.method == "GET" {
                    request.method = "POST".into();
                }
            }
            Key::Header => {
                let Some(header) = parse_header(&value()) else {
                    ewln(err, "curl: malformed header; expected 'Name: value'");
                    return 2;
                };
                request.headers.push(header);
            }
            Key::UserAgent => request.headers.push(("User-Agent".into(), value())),
            Key::Referer => request.headers.push(("Referer".into(), value())),
            Key::Cookie => request.headers.push(("Cookie".into(), value())),
            Key::User => request.headers.push((
                "Authorization".into(),
                format!("Basic {}", STANDARD.encode(value())),
            )),
            Key::Help => unreachable!("help is handled by the shared option parser"),
        }
    }
    if request.url.is_empty() {
        ewln(err, "curl: no URL specified");
        return 2;
    }
    let url = request.url.clone();
    let report_errors = !silent || show_error;
    let response = match system.http_request(request) {
        Ok(response) => response,
        Err(RequestError::NoRoute) => {
            if report_errors {
                ewln(
                    err,
                    "curl: (7) Failed to connect: no matching virtual HTTP route",
                );
            }
            return 7;
        }
        Err(error) => {
            if report_errors {
                ewln(err, &format!("curl: (23) {error}"));
            }
            return 23;
        }
    };
    if response.status >= 400 && fail {
        if report_errors {
            ewln(
                err,
                &format!("curl: (22) HTTP response status {}", response.status),
            );
        }
        return 22;
    }

    let payload = response_payload(&response, include_headers || head_only, head_only);
    if output_file.is_some() || remote_name {
        let name = output_file.unwrap_or_else(|| request_filename(&response, &url));
        let cwd = system.cwd().to_string();
        if let Err(error) = system.write_file(&cwd, &name, &payload, 0o644) {
            ewln(err, &format!("curl: (23) {error}"));
            return 23;
        }
    } else {
        out.extend_from_slice(&payload);
    }
    0
}

pub(crate) fn wget(system: &mut dyn System, args: &[String], out: Out, err: Out) -> i32 {
    #[derive(Clone, Copy, PartialEq)]
    enum Key {
        Output,
        Quiet,
        Directory,
        Method,
        Header,
        Body,
        Help,
    }
    const OPTIONS: &[OptionSpec<Key>] = &[
        OptionSpec::required(Key::Output, Some('O'), Some("output-document")),
        OptionSpec::flag(Key::Quiet, Some('q'), Some("quiet")),
        OptionSpec::required(Key::Directory, Some('P'), Some("directory-prefix")),
        OptionSpec::required(Key::Method, None, Some("method")),
        OptionSpec::required(Key::Header, None, Some("header")),
        OptionSpec::required(Key::Body, None, Some("body-data")),
        OptionSpec::required(Key::Body, None, Some("post-data")),
        OptionSpec::flag(Key::Help, None, Some("help")),
    ];
    let parsed = match parse_options_or_report(
        "wget",
        args,
        OPTIONS,
        (
            Key::Help,
            "usage: wget [OPTIONS] URL\nsupported: -O -q -P --method --header --body-data --post-data\n",
        ),
        out,
        err,
    ) {
        Ok(parsed) => parsed,
        Err(status) => return status,
    };
    if parsed.operands.len() != 1 {
        ewln(err, "wget: exactly one URL is required");
        return 2;
    }

    let mut request = HttpRequest::new("GET", &parsed.operands[0]);
    let mut output_file = None;
    let mut directory = None;
    let mut quiet = false;
    for option in parsed.options {
        let value = || option.value.clone().expect("required option value");
        match option.key {
            Key::Output => output_file = Some(value()),
            Key::Quiet => quiet = true,
            Key::Directory => directory = Some(value()),
            Key::Method => request.method = value(),
            Key::Header => {
                let Some(header) = parse_header(&value()) else {
                    ewln(err, "wget: malformed header; expected 'Name: value'");
                    return 2;
                };
                request.headers.push(header);
            }
            Key::Body => {
                request.body = value().into_bytes();
                if request.method == "GET" {
                    request.method = "POST".into();
                }
            }
            Key::Help => unreachable!("help is handled by the shared option parser"),
        }
    }
    let url = request.url.clone();
    let response = match system.http_request(request) {
        Ok(response) => response,
        Err(RequestError::NoRoute) => {
            if !quiet {
                ewln(
                    err,
                    "wget: unable to resolve request: no matching virtual HTTP route",
                );
            }
            return 4;
        }
        Err(error) => {
            if !quiet {
                ewln(err, &format!("wget: {error}"));
            }
            return 4;
        }
    };
    if response.status >= 400 {
        if !quiet {
            ewln(
                err,
                &format!("wget: server returned status {}", response.status),
            );
        }
        return 8;
    }

    let mut name = output_file.unwrap_or_else(|| request_filename(&response, &url));
    if let Some(directory) = directory {
        name = format!("{}/{}", directory.trim_end_matches('/'), name);
    }
    if name == "-" {
        out.extend_from_slice(&response.body);
    } else {
        let cwd = system.cwd().to_string();
        if let Err(error) = system.write_file(&cwd, &name, &response.body, 0o644) {
            ewln(err, &format!("wget: {error}"));
            return 3;
        }
    }
    0
}

fn parse_header(value: &str) -> Option<(String, String)> {
    let (name, value) = value.split_once(':')?;
    let name = name.trim();
    if name.is_empty() || name.contains(['\r', '\n']) || value.contains(['\r', '\n']) {
        return None;
    }
    Some((name.to_string(), value.trim().to_string()))
}

fn response_payload(response: &HttpResponse, include_headers: bool, head_only: bool) -> Vec<u8> {
    let mut bytes = Vec::new();
    if include_headers {
        bytes.extend_from_slice(
            format!(
                "HTTP/1.1 {} {}\r\n",
                response.status,
                reason_phrase(response.status)
            )
            .as_bytes(),
        );
        for (name, value) in &response.headers {
            bytes.extend_from_slice(format!("{name}: {value}\r\n").as_bytes());
        }
        bytes.extend_from_slice(b"\r\n");
    }
    if !head_only {
        bytes.extend_from_slice(&response.body);
    }
    bytes
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        301 => "Moved Permanently",
        302 => "Found",
        304 => "Not Modified",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "",
    }
}

fn request_filename(response: &HttpResponse, url: &str) -> String {
    response
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-disposition"))
        .and_then(|(_, value)| value.split("filename=").nth(1))
        .map(|value| value.trim_matches([' ', '"']).to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| {
            url.split('?')
                .next()
                .and_then(|url| url.rsplit('/').next())
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        })
        .unwrap_or_else(|| "index.html".to_string())
}

fn ewln(err: Out, message: &str) {
    err.extend_from_slice(message.as_bytes());
    err.push(b'\n');
}
