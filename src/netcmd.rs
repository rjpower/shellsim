//! `curl` / `wget` against the virtual network. No real egress.

use crate::interp::Interp;
use crate::net::RouteBody;

type Out<'a> = &'a mut Vec<u8>;

pub fn curl(interp: &mut Interp, args: &[String], out: Out, err: Out) -> i32 {
    let mut method = "GET".to_string();
    let mut url = None;
    let mut output_file: Option<String> = None;
    let mut silent = false;
    let mut include_headers = false;
    let mut fail = false;
    let mut data: Option<String> = None;
    let mut head_only = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-X" | "--request" => method = it.next().cloned().unwrap_or(method),
            "-o" | "--output" => output_file = it.next().cloned(),
            "-O" | "--remote-name" => output_file = Some(String::new()),
            "-s" | "--silent" => silent = true,
            "-i" | "--include" => include_headers = true,
            "-f" | "--fail" => fail = true,
            "-I" | "--head" => {
                head_only = true;
                method = "HEAD".into();
            }
            "-d" | "--data" | "--data-raw" => {
                data = it.next().cloned();
                if method == "GET" {
                    method = "POST".into();
                }
            }
            "-H" | "--header" | "-A" | "-e" | "-b" | "--connect-timeout" | "-m" | "--max-time"
            | "--retry" | "-u" => {
                it.next();
            }
            "-L" | "--location" | "-k" | "--insecure" | "-g" => {}
            s if s.starts_with('-') => {}
            s => url = Some(s.to_string()),
        }
    }
    let _ = (include_headers, head_only, data);
    let Some(url) = url else {
        ewln(err, "curl: no URL specified");
        return 2;
    };
    match interp.net.resolve(&method, &url) {
        Some(route) => {
            let body = match &route.body {
                RouteBody::Static(b) => b.clone(),
                RouteBody::VfsFile(p) => interp.vfs.read("/", p).unwrap_or_default(),
            };
            if route.status >= 400 && fail {
                ewln(err, &format!("curl: ({}) HTTP error", route.status));
                return 22;
            }
            if let Some(of) = output_file {
                let name = if of.is_empty() {
                    url.rsplit('/').next().unwrap_or("index.html").to_string()
                } else {
                    of
                };
                let cwd = interp.cwd.clone();
                let _ = interp.vfs.write(&cwd, &name, &body, 0o644);
            } else {
                out.extend_from_slice(&body);
            }
            0
        }
        None => {
            if !silent {
                ewln(
                    err,
                    &format!("curl: (7) Failed to connect: no virtual route for {url}"),
                );
            }
            7
        }
    }
}

pub fn wget(interp: &mut Interp, args: &[String], out: Out, err: Out) -> i32 {
    let mut url = None;
    let mut output_file: Option<String> = None;
    let mut quiet = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-O" | "--output-document" => output_file = it.next().cloned(),
            "-q" | "--quiet" => quiet = true,
            "-P" => {
                it.next();
            }
            s if s.starts_with('-') => {}
            s => url = Some(s.to_string()),
        }
    }
    let Some(url) = url else {
        ewln(err, "wget: missing URL");
        return 1;
    };
    match interp.net.resolve("GET", &url) {
        Some(route) => {
            let body = match &route.body {
                RouteBody::Static(b) => b.clone(),
                RouteBody::VfsFile(p) => interp.vfs.read("/", p).unwrap_or_default(),
            };
            let name = output_file
                .unwrap_or_else(|| url.rsplit('/').next().unwrap_or("index.html").to_string());
            if name == "-" {
                out.extend_from_slice(&body);
            } else {
                let cwd = interp.cwd.clone();
                let _ = interp.vfs.write(&cwd, &name, &body, 0o644);
            }
            let _ = quiet;
            0
        }
        None => {
            ewln(
                err,
                &format!("wget: unable to resolve host (no virtual route): {url}"),
            );
            4
        }
    }
}

fn ewln(err: Out, s: &str) {
    err.extend_from_slice(s.as_bytes());
    err.push(b'\n');
}
