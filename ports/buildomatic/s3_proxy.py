"""Serve one operator-selected storage prefix over read-only HTTP.

Run this file with marin-rigging installed. Rigging owns S3/GCS permissions,
credentials and routing. Requests resolve beneath one fixed base. Object bodies
stream without local staging or application-level digest verification.
"""

from __future__ import annotations

import argparse
import logging
import mimetypes
import posixpath
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Protocol
from urllib.parse import unquote, urlsplit

if TYPE_CHECKING:
    from rigging.filesystem.storage_path import StoragePath

_LOG = logging.getLogger(__name__)
_BLOCK = 128 * 1024
_READ_AHEAD = 8 * 1024 * 1024


def storage_root(uri: str) -> StoragePath:
    """Parse a storage root; local roots also resolve filesystem symlinks."""
    from rigging.filesystem.storage_path import StoragePath

    parsed = urlsplit(uri)
    if (
        parsed.scheme not in ("", "file", "s3", "gs")
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or any(ord(char) < 32 or ord(char) == 127 for char in uri)
        or "//" in parsed.path
        or (parsed.scheme in ("s3", "gs") and any(part in (".", "..") for part in parsed.path.split("/")))
        or (parsed.scheme in ("s3", "gs") and not parsed.netloc)
        or (parsed.scheme == "file" and parsed.netloc)
    ):
        raise ValueError("expected a local directory or s3:// or gs:// prefix")
    base = StoragePath(uri)
    return StoragePath(str(Path(parsed.path).resolve())) if base.is_local else base


def object_path(base: StoragePath, target: str) -> StoragePath:
    """Resolve a decoded HTTP path, then require structural containment."""
    from rigging.filesystem.storage_path import StoragePath

    if (
        len(target) > 8192
        or not target.startswith("/")
        or target.startswith("//")
        or any(ord(char) < 32 or ord(char) == 127 for char in target)
    ):
        raise ValueError("expected an origin-form request path")
    parsed = urlsplit(target)
    if parsed.netloc or parsed.fragment:
        raise ValueError("invalid request target")
    if re.search(r"%(?![0-9a-fA-F]{2})|%2f|%5c", parsed.path, re.IGNORECASE):
        raise ValueError("invalid encoded object path")
    relative = unquote(parsed.path[1:], errors="strict")
    if any(ord(char) < 32 or ord(char) == 127 or char in "\\%" for char in relative):
        raise ValueError("invalid object path")
    if not relative or relative.endswith("/"):
        relative += "index.html"
    if base.is_local:
        resolved = StoragePath(str((Path(str(base)) / relative).resolve()))
    else:
        normalized = posixpath.normpath("/" + base.key + "/" + relative)
        resolved = StoragePath(f"{base.scheme}://{base.bucket}{normalized}")
    resolved.relative_to(base)
    return resolved


@dataclass
class ObjectResponse:
    """HTTP metadata and a streaming body owned by the caller."""

    status: int
    headers: dict[str, str]
    body: BinaryIO | None = None


class ObjectStore(Protocol):
    def fetch(self, path: StoragePath, *, head: bool, byte_range: str | None, etag: str | None) -> ObjectResponse:
        """Fetch one contained object with conditional and range semantics."""
        ...


def byte_span(value: str | None, size: int) -> tuple[int, int]:
    """Return a half-open single range, rejecting malformed or empty ranges."""
    if value is None:
        return 0, size
    match = re.fullmatch(r"bytes=([0-9]*)-([0-9]*)", value)
    if match is None or not any(match.groups()):
        raise ValueError("invalid range")
    first, last = match.groups()
    start = int(first) if first else max(0, size - int(last))
    stop = min(size, int(last) + 1) if first and last else size
    if start >= stop:
        raise ValueError("unsatisfiable range")
    return start, stop


class RiggingStore:
    """Read through Rigging's bucket routing, retaining representation metadata."""

    def fetch(self, path: StoragePath, *, head: bool, byte_range: str | None, etag: str | None) -> ObjectResponse:
        from rigging.filesystem.buckets import filesystem_for

        body = None
        try:
            fs, key = filesystem_for(str(path))
            if path.scheme == "s3":
                # s3fs.info omits Content-Encoding; use its routed client's HEAD.
                info = fs.call_s3("head_object", Bucket=path.bucket, Key=path.key)
                size = info["ContentLength"]
                object_etag = info.get("ETag")
                options = {"s3_additional_kwargs": {"IfMatch": object_etag}} if object_etag else {}
            else:
                info = fs.info(key)
                if info["type"] != "file":
                    return ObjectResponse(404, {})
                size = info["size"]
                object_etag = info.get("etag")
                options = {}
                if path.scheme == "gs":
                    # gcsfs may decode HTTP gzip automatically. Delivery wheels
                    # are ZIP objects; encoded GCS representations are unsupported.
                    if info.get("contentEncoding") not in (None, "identity"):
                        return ObjectResponse(501, {})
                    # gcsfs carries generations into range reads from the URL,
                    # whereas open(generation=...) only pins its metadata lookup.
                    fs, key = filesystem_for(f"{path}#{info['generation']}")
            headers = {"Accept-Ranges": "bytes"}
            if object_etag:
                object_etag = '"' + object_etag.strip('"') + '"'
                headers["ETag"] = object_etag
            if etag and (etag == "*" or object_etag in [part.strip().removeprefix("W/") for part in etag.split(",")]):
                return ObjectResponse(304, headers)
            content_type = info.get("ContentType") or info.get("contentType")
            headers["Content-Type"] = (
                content_type
                if content_type and content_type != "application/octet-stream"
                else mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            )
            for source, destination in (
                ("ContentEncoding", "Content-Encoding"),
                ("contentEncoding", "Content-Encoding"),
                ("CacheControl", "Cache-Control"),
                ("cacheControl", "Cache-Control"),
            ):
                if source in info:
                    headers[destination] = info[source]
            modified = info.get("LastModified") or info.get("mtime")
            if modified is not None:
                if isinstance(modified, (int, float)):
                    modified = datetime.fromtimestamp(modified, timezone.utc)
                headers["Last-Modified"] = format_datetime(modified.astimezone(timezone.utc), usegmt=True)
            try:
                start, stop = byte_span(byte_range, size)
            except ValueError:
                return ObjectResponse(416, {"Content-Range": f"bytes */{size}"})
            headers["Content-Length"] = str(stop - start)
            if byte_range:
                headers["Content-Range"] = f"bytes {start}-{stop - 1}/{size}"
            if not head:
                # Fetch bounded larger spans instead of issuing a cloud request
                # for every small HTTP write. Short ranges keep small buffers.
                body = fs.open(
                    key,
                    "rb",
                    size=size,
                    cache_type="readahead",
                    block_size=min(_READ_AHEAD, max(1, stop - start)),
                    **options,
                )
                body.seek(start)
            return ObjectResponse(206 if byte_range else 200, headers, body)
        except Exception as error:
            if body is not None:
                body.close()
            # Provider diagnostics may contain private paths; expose only status.
            _LOG.warning("Storage read failed: %s", type(error).__name__)
            status = 404 if isinstance(error, FileNotFoundError) else 403 if isinstance(error, PermissionError) else 502
            return ObjectResponse(status, {})


def handler(base: StoragePath, store: ObjectStore) -> type[BaseHTTPRequestHandler]:
    """Bind the fixed storage base to GET/HEAD handlers."""

    class ProxyHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.serve(head=False)

        def do_HEAD(self):
            self.serve(head=True)

        def serve(self, *, head: bool):
            try:
                # Validate the original target: Python normalizes leading //.
                path = object_path(base, self.requestline.split()[1])
            except (ValueError, UnicodeError):
                self.send_error(400)
                return
            byte_range = None if head or self.headers.get("If-Range") is not None else self.headers.get("Range")
            response = store.fetch(path, head=head, byte_range=byte_range, etag=self.headers.get("If-None-Match"))
            try:
                self.send_response(response.status)
                for name, value in response.headers.items():
                    self.send_header(name, value)
                self.end_headers()
                if not head and response.body is not None:
                    remaining = int(response.headers["Content-Length"])
                    while remaining:
                        chunk = response.body.read(min(_BLOCK, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                if response.body is not None:
                    response.body.close()

        def log_message(self, format, *args):
            _LOG.info("%s", self.command)

    return ProxyHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="S3/GCS prefix or local directory served as HTTP /")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    base = storage_root(args.prefix)
    logging.basicConfig(level=logging.INFO)
    with ThreadingHTTPServer((args.bind, args.port), handler(base, RiggingStore())) as server:
        _LOG.info("Serving %s on %s:%s", base, *server.server_address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
