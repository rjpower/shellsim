"""Exercise real HTTP parsing in memory and S3 calls with offline SDK stubs."""

import io
from datetime import datetime
from email import message_from_bytes

import pytest

from ports.buildomatic.s3_proxy import ObjectResponse, Prefix, S3Store, handler


class Connection:
    def __init__(self, request):
        self.input = io.BytesIO(request)
        self.output = bytearray()

    def makefile(self, mode, buffering):
        return self.input

    def sendall(self, data):
        self.output.extend(data)


class Body(io.BytesIO):
    def read(self, size=-1):
        assert 0 < size <= 128 * 1024
        return super().read(size)


class Store:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def fetch(self, key, **kwargs):
        self.calls.append((key, kwargs))
        return self.response


def request(store, path, *, method="GET", headers=()):
    wire = f"{method} {path} HTTP/1.1\r\nHost: proxy\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers) + "\r\n"
    connection = Connection(wire.encode())
    handler(Prefix.parse("s3://bucket/public"), store)(connection, ("127.0.0.1", 1), None)
    head, _, body = bytes(connection.output).partition(b"\r\n\r\n")
    status, _, fields = head.partition(b"\r\n")
    return int(status.split()[1]), message_from_bytes(fields), body


@pytest.mark.parametrize(
    "target",
    [
        "/../secret",
        "/%2e%2e/secret",
        "/%252e%252e/secret",
        "/a/./secret",
        "/a//secret",
        "/a%2f../secret",
        "/a%5csecret",
        "/a\\secret",
        "/%00secret",
        "/%GGsecret",
        "/%ffsecret",
        "//other/secret",
        "///secret",
        "http://other/secret",
    ],
)
def test_rejected_paths_never_access_s3(target):
    store = Store(ObjectResponse(200, {}))
    status, _, _ = request(store, target)
    assert status == 400
    assert not store.calls


@pytest.mark.parametrize(
    "path,key",
    [("/", "public/index.html"), ("/simple/", "public/simple/index.html"), ("/a%20b.whl?v=1", "public/a b.whl")],
)
def test_safe_paths_stream_under_fixed_prefix(path, key):
    body = Body(b"wheel" * 100_000)
    store = Store(ObjectResponse(200, {"Content-Length": "500000"}, body))
    status, headers, data = request(store, path)
    assert status == 200
    assert headers["Content-Length"] == "500000"
    assert data == b"wheel" * 100_000
    assert body.closed
    assert store.calls == [(key, {"head": False, "byte_range": None, "etag": None})]


def test_head_and_range_conditionals():
    store = Store(ObjectResponse(206, {"Content-Length": "3", "Content-Range": "bytes 1-3/5"}, Body(b"hee")))
    status, headers, data = request(store, "/test.whl", headers=[("Range", "bytes=1-3"), ("If-None-Match", '"etag"')])
    assert status == 206
    assert headers["Content-Range"] == "bytes 1-3/5"
    assert data == b"hee"
    assert store.calls[0][1] == {"head": False, "byte_range": "bytes=1-3", "etag": '"etag"'}
    store = Store(ObjectResponse(200, {"Content-Length": "5"}))
    status, headers, data = request(store, "/test.whl", method="HEAD")
    assert status == 200 and headers["Content-Length"] == "5" and data == b""
    assert store.calls[0][1]["head"]


@pytest.mark.parametrize(
    "method,headers", [("HEAD", [("Range", "garbage")]), ("GET", [("Range", "bytes=2-"), ("If-Range", '"old"')])]
)
def test_head_and_if_range_ignore_ranges(method, headers):
    body = Body(b"new object")
    store = Store(ObjectResponse(200, {"Content-Length": "10"}, body))
    status, _, data = request(store, "/asset.whl", method=method, headers=headers)
    assert status == 200
    assert data == (b"" if method == "HEAD" else b"new object")
    assert store.calls[0][1]["byte_range"] is None
    assert body.closed


@pytest.mark.parametrize("byte_range", ["bytes=0-1,3-4", "items=1-3", "bytes=-", "bytes=abc"])
def test_invalid_ranges_never_access_s3(byte_range):
    store = Store(ObjectResponse(200, {}))
    assert request(store, "/test.whl", headers=[("Range", byte_range)])[0] == 416
    assert not store.calls


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "CONNECT"])
def test_writes_never_access_s3(method):
    store = Store(ObjectResponse(200, {}))
    assert request(store, "/test.whl", method=method)[0] == 501
    assert not store.calls


@pytest.mark.parametrize(
    "prefix",
    [
        "http://bucket/public",
        "s3://user@bucket/public",
        "s3://bucket/a/../b",
        "s3://bucket/a//b",
        "s3://bucket//",
        "s3://bucket/public?key=secret",
    ],
)
def test_invalid_configured_prefix(prefix):
    with pytest.raises(ValueError):
        Prefix.parse(prefix)


@pytest.fixture
def s3_client():
    session = pytest.importorskip("botocore.session")
    client = session.get_session().create_client(
        "s3", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test"
    )
    yield client
    client.close()


def test_s3_range_and_content_headers(s3_client):
    from botocore.stub import Stubber
    from dateutil.tz import tzutc

    body = Body(b"hee")
    with Stubber(s3_client) as stub:
        stub.add_response(
            "get_object",
            {
                "Body": body,
                "ContentLength": 3,
                "ContentRange": "bytes 1-3/5",
                "ETag": '"etag"',
                "ContentEncoding": "gzip",
                "LastModified": datetime(2026, 10, 11, 12, 0, tzinfo=tzutc()),
                "ResponseMetadata": {"HTTPStatusCode": 206},
            },
            {"Bucket": "bucket", "Key": "public/test.whl", "Range": "bytes=1-3", "IfNoneMatch": '"old"'},
        )
        result = S3Store(s3_client, "bucket").fetch("public/test.whl", head=False, byte_range="bytes=1-3", etag='"old"')
    assert result.status == 206
    assert result.headers["Content-Range"] == "bytes 1-3/5"
    assert result.headers["ETag"] == '"etag"'
    assert result.headers["Content-Encoding"] == "gzip"
    assert result.headers["Last-Modified"] == "Sun, 11 Oct 2026 12:00:00 GMT"
    assert result.body is body
    body.close()


@pytest.mark.parametrize("status", [304, 403, 404, 416, 500])
def test_s3_errors_do_not_expose_provider_details(s3_client, status):
    from botocore.stub import Stubber

    with Stubber(s3_client) as stub:
        stub.add_client_error(
            "get_object",
            service_error_code="ProviderError",
            service_message="private provider diagnostic",
            http_status_code=status,
            expected_params={"Bucket": "bucket", "Key": "public/a"},
        )
        result = S3Store(s3_client, "bucket").fetch("public/a", head=False, byte_range=None, etag=None)
    assert result.status == (502 if status == 500 else status)
    assert result.headers == {} and result.body is None
