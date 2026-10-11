"""Serve one operator-selected S3 prefix over read-only HTTP.

Run this file directly with botocore installed. Credentials and endpoint choice
stay on the server; request paths can select only objects beneath its prefix.
Large objects stream without local staging or application-level digest checks.
"""

from __future__ import annotations

import argparse
import logging
import mimetypes
import re
import shutil
from dataclasses import dataclass
from datetime import timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import BinaryIO, Protocol
from urllib.parse import unquote, urlsplit

_LOG = logging.getLogger(__name__)


def path_parts(path: str) -> tuple[str, ...]:
    """Reject ambiguous separators, traversal and control characters."""
    if re.search(r"%(?![0-9a-fA-F]{2})|%2f|%5c", path, re.IGNORECASE):
        raise ValueError("invalid encoded object path")
    path = unquote(path, errors="strict")
    if any(ord(char) < 32 or ord(char) == 127 or char in "\\%" for char in path):
        raise ValueError("invalid object path")
    parts = tuple(path.split("/")) if path else ()
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("object paths must contain ordinary relative components")
    return parts


@dataclass(frozen=True)
class Prefix:
    """Fixed bucket and key prefix selected at server startup."""

    bucket: str
    parts: tuple[str, ...]

    @classmethod
    def parse(cls, uri: str) -> Prefix:
        """Parse the fixed storage root before binding the HTTP listener."""
        if any(ord(char) < 32 or ord(char) == 127 for char in uri):
            raise ValueError("invalid storage prefix")
        parsed = urlsplit(uri)
        if (
            parsed.scheme != "s3"
            or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", parsed.netloc)
            or parsed.query
            or parsed.fragment
            or "//" in parsed.path
        ):
            raise ValueError("prefix must be an s3://bucket/path URI")
        return cls(parsed.netloc, path_parts(parsed.path.removeprefix("/").removesuffix("/")))

    def key(self, target: str) -> str:
        """Map an origin-form URL to an object; directory URLs use index.html."""
        if (
            len(target) > 8192
            or not target.startswith("/")
            or target.startswith("//")
            or any(ord(char) < 32 or ord(char) == 127 for char in target)
        ):
            raise ValueError("expected an origin-form request path")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or parsed.fragment:
            raise ValueError("request cannot select another authority")
        path = parsed.path[1:]
        if not path or path.endswith("/"):
            path += "index.html"
        return "/".join((*self.parts, *path_parts(path)))


@dataclass
class ObjectResponse:
    """S3 response metadata and an optional streaming body owned by the caller."""

    status: int
    headers: dict[str, str]
    body: BinaryIO | None = None


class ObjectStore(Protocol):
    def fetch(self, key: str, *, head: bool, byte_range: str | None, etag: str | None) -> ObjectResponse:
        """Fetch one object, preserving range and conditional-read semantics."""
        ...


class S3Store:
    """Translate S3 reads to HTTP without forwarding provider diagnostics."""

    def __init__(self, client, bucket: str):
        self.client = client
        self.bucket = bucket

    def fetch(self, key: str, *, head: bool, byte_range: str | None, etag: str | None) -> ObjectResponse:
        from botocore.exceptions import BotoCoreError, ClientError

        args = {"Bucket": self.bucket, "Key": key}
        if byte_range and not head:
            args["Range"] = byte_range
        if etag:
            args["IfNoneMatch"] = etag
        try:
            value = self.client.head_object(**args) if head else self.client.get_object(**args)
        except ClientError as error:
            status = error.response["ResponseMetadata"]["HTTPStatusCode"]
            return ObjectResponse(status if status in (304, 403, 404, 416) else 502, {})
        except BotoCoreError:
            return ObjectResponse(502, {})
        headers = {"Content-Length": str(value["ContentLength"]), "Accept-Ranges": "bytes"}
        content_type = value.get("ContentType")
        headers["Content-Type"] = (
            content_type
            if content_type and content_type != "application/octet-stream"
            else mimetypes.guess_type(key)[0] or "application/octet-stream"
        )
        for name, header in (
            ("ETag", "ETag"),
            ("ContentRange", "Content-Range"),
            ("CacheControl", "Cache-Control"),
            ("ContentEncoding", "Content-Encoding"),
        ):
            if name in value:
                headers[header] = value[name]
        if "LastModified" in value:
            headers["Last-Modified"] = format_datetime(value["LastModified"].astimezone(timezone.utc), usegmt=True)
        return ObjectResponse(value["ResponseMetadata"]["HTTPStatusCode"], headers, value.get("Body"))


def handler(prefix: Prefix, store: ObjectStore) -> type[BaseHTTPRequestHandler]:
    """Bind the fixed prefix and store to GET/HEAD handlers."""

    class ProxyHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.serve(head=False)

        def do_HEAD(self):
            self.serve(head=True)

        def serve(self, *, head: bool):
            try:
                # BaseHTTPRequestHandler normalizes leading // before assigning
                # self.path. Validate the original target to avoid that ambiguity.
                key = prefix.key(self.requestline.split()[1])
            except (ValueError, UnicodeError):
                self.send_error(400)
                return
            # A full response is safe for If-Range without comparing validators.
            # HEAD ignores Range, including malformed values.
            byte_range = None if head or self.headers.get("If-Range") is not None else self.headers.get("Range")
            if byte_range and not re.fullmatch(r"bytes=(?:[0-9]+-[0-9]*|-[0-9]+)", byte_range):
                self.send_error(416)
                return
            response = store.fetch(key, head=head, byte_range=byte_range, etag=self.headers.get("If-None-Match"))
            try:
                self.send_response(response.status)
                for name, value in response.headers.items():
                    self.send_header(name, value)
                self.end_headers()
                if not head and response.body is not None:
                    shutil.copyfileobj(response.body, self.wfile, length=128 * 1024)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                if response.body is not None:
                    response.body.close()

        def log_message(self, format, *args):
            # Do not log request URLs, which may carry query credentials.
            _LOG.info("%s", self.command)

    return ProxyHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="S3 bucket and prefix served as HTTP /")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--endpoint-url", help="S3-compatible endpoint; credentials use the standard AWS chain")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--addressing-style", choices=("virtual", "path", "auto"), default="virtual")
    args = parser.parse_args()
    prefix = Prefix.parse(args.prefix)
    import botocore.session
    from botocore.config import Config

    client = botocore.session.get_session().create_client(
        "s3",
        endpoint_url=args.endpoint_url,
        region_name=args.region,
        config=Config(s3={"addressing_style": args.addressing_style}),
    )
    logging.basicConfig(level=logging.INFO)
    with ThreadingHTTPServer((args.bind, args.port), handler(prefix, S3Store(client, prefix.bucket))) as server:
        _LOG.info("Serving %s on %s:%s", args.prefix, *server.server_address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    client.close()


if __name__ == "__main__":
    main()
