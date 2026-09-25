//! Local browser demo for a resumable Wasm guest built from external Doomgeneric inputs.
//!
//! This host-only example binds loopback. The guest still sees only shellsim's virtual syscalls;
//! Doomgeneric source and game data are supplied by the user and are never bundled.

#[path = "support/doom.rs"]
mod doom_support;

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use shellsim::{
    commands::{SessionPoll, WasmSession},
    display::{DisplayFrame, KeyEvent},
};

const HTML: &str = r#"<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Doom in shellsim</title>
<style>
body{background:#151515;color:#eee;font:16px system-ui;text-align:center;margin:2rem}
canvas{width:min(95vw,1280px);image-rendering:pixelated;border:1px solid #555}
p{margin:.7rem}button{font:inherit;padding:.4rem .9rem}
</style>
<h1>Doom in shellsim</h1>
<canvas id="screen" width="640" height="400"></canvas>
<p>Arrow keys move, Ctrl fires, Space uses, Shift runs, Esc opens the menu. Click the page to focus.</p>
<p id="status">Waiting for the first frame…</p><button id="quit">Stop guest</button>
<script>
const canvas = document.getElementById('screen');
const ctx = canvas.getContext('2d', {alpha:false});
const status = document.getElementById('status');
const keycodes = {ArrowRight:0xae,ArrowLeft:0xac,ArrowUp:0xad,ArrowDown:0xaf,
  Escape:27,Enter:13,Tab:9,Control:0xa3,' ':0xa2,Shift:0xb6,Alt:0xb8};
function code(event) {
  if (keycodes[event.key] !== undefined) return keycodes[event.key];
  if (event.key.length === 1) return event.key.toLowerCase().charCodeAt(0);
  return null;
}
function sendKey(event, pressed) {
  const key = code(event);
  if (key === null || (pressed && event.repeat)) return;
  event.preventDefault();
  const data = new ArrayBuffer(8);
  const view = new DataView(data);
  view.setUint32(0,key,true); view.setUint32(4,pressed ? 1 : 0,true);
  fetch('/key',{method:'POST',body:data}).catch(() => { status.textContent='Connection lost'; });
}
addEventListener('keydown',event => sendKey(event,true));
addEventListener('keyup',event => sendKey(event,false));
addEventListener('blur',() => { for (const code of [0xad,0xaf,0xac,0xae,0xa3,0xa2,0xb6]) {
  const data=new ArrayBuffer(8); const view=new DataView(data); view.setUint32(0,code,true);
  fetch('/key',{method:'POST',body:data}).catch(() => {});
}});
document.getElementById('quit').onclick=() => fetch('/quit',{method:'POST'});
async function draw() {
  try {
    const response=await fetch('/frame',{cache:'no-store'});
    if (response.ok) {
      const pixels=new Uint8ClampedArray(await response.arrayBuffer());
      if (pixels.length===640*400*4) {
        ctx.putImageData(new ImageData(pixels,640,400),0,0);
        status.textContent='Running';
      }
    } else if (response.status===410) { status.textContent='Guest stopped'; return; }
  } catch (_) { status.textContent='Connection lost'; return; }
  setTimeout(draw,50);
}
draw();
</script></html>"#;

fn respond(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)
}

struct Request {
    method: String,
    path: String,
    origin: Option<String>,
    body: Vec<u8>,
}

fn read_request(stream: &mut TcpStream) -> Result<Request, String> {
    stream
        .set_read_timeout(Some(Duration::from_millis(500)))
        .map_err(|error| error.to_string())?;
    let mut bytes = Vec::new();
    let end = loop {
        if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break end + 4;
        }
        if bytes.len() >= 8192 {
            return Err("request headers too large".into());
        }
        let mut chunk = [0; 1024];
        let count = stream.read(&mut chunk).map_err(|error| error.to_string())?;
        if count == 0 {
            return Err("incomplete request".into());
        }
        bytes.extend_from_slice(&chunk[..count]);
        if bytes.len() > 8192 {
            return Err("request headers too large".into());
        }
    };
    let headers = std::str::from_utf8(&bytes[..end]).map_err(|error| error.to_string())?;
    let mut lines = headers.split("\r\n");
    let mut first = lines.next().unwrap_or("").split_whitespace();
    let method = first.next().ok_or("missing HTTP method")?.to_string();
    let path = first.next().ok_or("missing HTTP path")?.to_string();
    let mut length = 0usize;
    let mut origin = None;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            length = value.parse().map_err(|_| "invalid content length")?;
        } else if name.eq_ignore_ascii_case("origin") {
            origin = Some(value.to_string());
        }
    }
    if length > 8 {
        return Err("request body too large".into());
    }
    while bytes.len() < end + length {
        let mut chunk = [0; 8];
        let count = stream.read(&mut chunk).map_err(|error| error.to_string())?;
        if count == 0 {
            return Err("incomplete request body".into());
        }
        bytes.extend_from_slice(&chunk[..count]);
    }
    Ok(Request {
        method,
        path,
        origin,
        body: bytes[end..end + length].to_vec(),
    })
}

fn handle(
    mut stream: TcpStream,
    origin: &str,
    frame: Option<&DisplayFrame>,
    session: &mut WasmSession,
) -> Result<bool, String> {
    // macOS inherits the listener's nonblocking mode; request I/O uses blocking timeouts.
    stream
        .set_nonblocking(false)
        .map_err(|error| error.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(30)))
        .map_err(|error| error.to_string())?;
    let request = match read_request(&mut stream) {
        Ok(request) => request,
        Err(error) => {
            respond(
                &mut stream,
                "400 Bad Request",
                "text/plain",
                error.as_bytes(),
            )
            .map_err(|error| error.to_string())?;
            return Ok(false);
        }
    };
    if request.method == "POST"
        && request
            .origin
            .as_deref()
            .is_some_and(|value| value != origin)
    {
        respond(&mut stream, "403 Forbidden", "text/plain", b"wrong origin")
            .map_err(|error| error.to_string())?;
        return Ok(false);
    }
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/") => respond(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            HTML.as_bytes(),
        ),
        ("GET", "/frame") => match frame {
            Some(frame) => respond(
                &mut stream,
                "200 OK",
                "application/octet-stream",
                &frame.pixels,
            ),
            None => respond(&mut stream, "204 No Content", "text/plain", b""),
        },
        ("POST", "/key") if request.body.len() == 8 => {
            let code = u32::from_le_bytes(request.body[..4].try_into().expect("length checked"));
            let pressed = u32::from_le_bytes(request.body[4..].try_into().expect("length checked"));
            let status = match session.inject_key(KeyEvent {
                code,
                pressed: pressed != 0,
            }) {
                Ok(()) => "204 No Content",
                Err(_) => "400 Bad Request",
            };
            respond(&mut stream, status, "text/plain", b"")
        }
        ("POST", "/quit") => {
            respond(&mut stream, "204 No Content", "text/plain", b"")
                .map_err(|error| error.to_string())?;
            return Ok(true);
        }
        _ => respond(&mut stream, "404 Not Found", "text/plain", b"not found"),
    }
    .map_err(|error| error.to_string())?;
    Ok(false)
}

fn run() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let source = PathBuf::from(args.next().ok_or(
        "usage: cargo run --example doom_player -- <doomgeneric/doomgeneric> <freedoom1.wad>",
    )?);
    let wad = PathBuf::from(args.next().ok_or("missing WAD path")?);
    if args.next().is_some() {
        return Err("expected exactly two paths".into());
    }
    eprintln!("Building external Doomgeneric inside shellsim...");
    let environment = doom_support::build(&source, &wad)?;
    let mut session = WasmSession::start(
        environment,
        doom_support::PROGRAM,
        &["-iwad".into(), doom_support::WAD.into()],
    )?;
    let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
    listener
        .set_nonblocking(true)
        .map_err(|error| error.to_string())?;
    let origin = format!(
        "http://{}",
        listener.local_addr().map_err(|error| error.to_string())?
    );
    eprintln!("Open {origin} in a browser. The guest has no host network access.");
    let mut frame = None;
    let mut next_frame = Instant::now();
    let mut stopping = false;
    loop {
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    if handle(stream, &origin, frame.as_ref(), &mut session)? && !stopping {
                        stopping = true;
                        session.request_stop();
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error.to_string()),
            }
        }
        if Instant::now() < next_frame && !stopping {
            thread::sleep(Duration::from_millis(2));
            continue;
        }
        match session.poll() {
            SessionPoll::Frame(_) => {
                frame = session.frame();
                next_frame = Instant::now() + Duration::from_millis(28);
            }
            SessionPoll::Running => {}
            SessionPoll::Ready(status) => {
                let result = session.into_result().expect("ready session has result");
                if !result.stderr.is_empty() {
                    eprintln!("{}", String::from_utf8_lossy(&result.stderr));
                }
                eprintln!("Guest stopped with status {status}");
                return if stopping || status == 0 {
                    Ok(())
                } else {
                    Err(format!("guest exited with {status}"))
                };
            }
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("doom_player: {error}");
        std::process::exit(1);
    }
}
