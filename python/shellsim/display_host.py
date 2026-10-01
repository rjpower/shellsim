"""Loopback browser viewer for a host-driven shellsim display action."""

from __future__ import annotations

import hmac
import secrets
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlsplit

from ._api import Action, DisplayFrame

_HTML = b"""<!doctype html><html lang="en"><meta charset="utf-8"><title>shellsim display</title>
<style>body{background:#151515;color:#eee;font:16px system-ui;text-align:center;margin:2rem}canvas{max-width:95vw;image-rendering:pixelated;border:1px solid #555}canvas:focus{outline:2px solid #eee}</style>
<h1>shellsim display</h1><canvas tabindex="0" aria-label="Virtual display"></canvas>
<p id="status">Waiting for a frame</p><button id="stop">Stop guest</button>
<script>
const canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d'),status=document.querySelector('#status');
const token=new URL(location.href).searchParams.get('token');
const special={Escape:27,Enter:13,Tab:9,ArrowUp:61441,ArrowDown:61442,ArrowLeft:61443,ArrowRight:61444,Control:61445,Shift:61446,Alt:61447};
const held=new Map();
function code(event){return special[event.key] ?? (event.key.length===1 ? event.key.toLowerCase().codePointAt(0) : null)}
function key(code,pressed){const body=new ArrayBuffer(8),view=new DataView(body);view.setUint32(0,code,true);view.setUint32(4,pressed?1:0,true);fetch('/key?token='+token,{method:'POST',headers:{'X-Shellsim-Token':token},body})}
canvas.onkeydown=event=>{const value=code(event);if(value===null)return;event.preventDefault();const id=event.code||event.key;if(held.has(id))return;held.set(id,value);key(value,true)};
canvas.onkeyup=event=>{const id=event.code||event.key,value=held.get(id);if(value===undefined)return;event.preventDefault();held.delete(id);key(value,false)};
function release(){for(const value of held.values())key(value,false);held.clear()}
canvas.onblur=release;addEventListener('blur',release);document.addEventListener('visibilitychange',()=>{if(document.hidden)release()});canvas.onpointerdown=()=>canvas.focus();
document.querySelector('#stop').onclick=()=>fetch('/stop?token='+token,{method:'POST',headers:{'X-Shellsim-Token':token}});
async function draw(){try{const response=await fetch('/frame?token='+token,{cache:'no-store'});if(response.status===200){const width=Number(response.headers.get('X-Frame-Width')),height=Number(response.headers.get('X-Frame-Height'));const pixels=new Uint8ClampedArray(await response.arrayBuffer());if(pixels.length===width*height*4){canvas.width=width;canvas.height=height;ctx.putImageData(new ImageData(pixels,width,height),0,0);status.textContent='Running'}}else if(response.status===410){status.textContent='Guest stopped';return}}catch(_){status.textContent='Connection lost';return}setTimeout(draw,50)}
canvas.focus();draw();
</script></html>"""


def _key_event(body: bytes) -> tuple[int, bool]:
    if len(body) != 8:
        raise ValueError("key event must be eight bytes")
    code = int.from_bytes(body[:4], "little")
    pressed = int.from_bytes(body[4:], "little")
    if not 0 < code <= 0xFFFF or pressed not in (0, 1):
        raise ValueError("invalid key event")
    return code, bool(pressed)


class _Server(HTTPServer):
    display_host: DisplayHost


class _Handler(BaseHTTPRequestHandler):
    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(1)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _authorized(self, mutating: bool) -> bool:
        host = self.server.display_host
        parsed = urlsplit(self.path)
        token = parse_qs(parsed.query).get("token", [""])[0]
        if not hmac.compare_digest(token, host.token):
            return False
        if mutating:
            return hmac.compare_digest(self.headers.get("X-Shellsim-Token", ""), host.token) and self.headers.get(
                "Origin", ""
            ) in ("", host.origin)
        return True

    def _send(self, status: int, body: bytes = b"", content_type: str = "text/plain") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if not self._authorized(False):
            self._send(403)
            return
        host = self.server.display_host
        path = urlsplit(self.path).path
        if path == "/":
            self._send(200, _HTML, "text/html; charset=utf-8")
        elif path == "/frame":
            with host._lock:
                frame = host._frame
                stopped = host._stopped
            if stopped or frame is None:
                self._send(410 if stopped else 204)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(frame.pixels)))
            self.send_header("X-Frame-Width", str(frame.width))
            self.send_header("X-Frame-Height", str(frame.height))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(frame.pixels)
        else:
            self._send(404)

    def do_POST(self) -> None:
        if not self._authorized(True):
            self._send(403)
            return
        path = urlsplit(self.path).path
        length = self.headers.get("Content-Length", "")
        if path not in ("/key", "/stop") or length not in ("0", "8", ""):
            self._send(400)
            return
        host = self.server.display_host
        if path == "/stop":
            if length not in ("", "0"):
                self._send(400)
                return
            with host._lock:
                host._stop_requested = True
            self._send(204)
            return
        if length != "8":
            self._send(400)
            return
        try:
            event = _key_event(self.rfile.read(8))
        except ValueError:
            self._send(400)
            return
        with host._lock:
            if len(host._keys) >= 256:
                self._send(429)
                return
            host._keys.append(event)
        self._send(204)


class DisplayHost:
    """Serve frames and collect keys on loopback without exposing guest network access."""

    def __init__(self, action: Action) -> None:
        self.action = action
        self.token = secrets.token_urlsafe(24)
        self._lock = threading.Lock()
        self._frame: Optional[DisplayFrame] = None
        self._keys: deque[tuple[int, bool]] = deque()
        self._stop_requested = False
        self._stopped = False
        self._server: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> DisplayHost:
        server = _Server(("127.0.0.1", 0), _Handler)
        server.display_host = self
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join()

    @property
    def origin(self) -> str:
        """Loopback origin for the active viewer."""

        if self._server is None:
            raise RuntimeError("display host is not running")
        return f"http://127.0.0.1:{self._server.server_port}"

    @property
    def url(self) -> str:
        """Session URL containing its unguessable browser token."""

        return f"{self.origin}/?token={self.token}"

    def publish(self, frame: DisplayFrame) -> None:
        """Replace the visible frame with the latest complete guest frame."""

        with self._lock:
            self._frame = frame

    def drain_keys(self) -> list[tuple[int, bool]]:
        """Take bounded browser key transitions for the owning action."""

        with self._lock:
            events = list(self._keys)
            self._keys.clear()
        return events

    def stop_requested(self) -> bool:
        """Whether the viewer requested cancellation."""

        with self._lock:
            return self._stop_requested

    def mark_stopped(self) -> None:
        """Tell the viewer that no further frames will arrive."""

        with self._lock:
            self._stopped = True
