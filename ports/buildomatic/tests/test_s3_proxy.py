"""Exercise HTTP parsing in memory and Rigging routing without cloud access."""

import io
from datetime import datetime, timezone
from email import message_from_bytes

import pytest

pytest.importorskip("rigging.filesystem.storage_path")

from ports.buildomatic.s3_proxy import ObjectResponse, RiggingStore, handler, object_path, storage_root


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

    def fetch(self, path, **kwargs):
        self.calls.append((str(path), kwargs))
        return self.response


def request(store, path, *, method="GET", headers=(), base="s3://bucket/public"):
    wire = f"{method} {path} HTTP/1.1\r\nHost: proxy\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers) + "\r\n"
    connection = Connection(wire.encode())
    handler(storage_root(base), store)(connection, ("127.0.0.1", 1), None)
    head, _, body = bytes(connection.output).partition(b"\r\n\r\n")
    status, _, fields = head.partition(b"\r\n")
    return int(status.split()[1]), message_from_bytes(fields), body


@pytest.mark.parametrize(
    "target",
    [
        "/../secret",
        "/%2e%2e/secret",
        "/a/../../secret",
        "/%252e%252e/secret",
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
def test_rejected_paths_never_access_storage(target):
    store = Store(ObjectResponse(200, {}))
    assert request(store, target)[0] == 400
    assert not store.calls


@pytest.mark.parametrize("scheme", ["s3", "gs"])
@pytest.mark.parametrize(
    "path,key",
    [
        ("/", "index.html"),
        ("/simple/", "simple/index.html"),
        ("/a%20b.whl?v=1", "a b.whl"),
        ("/a/../test.whl", "test.whl"),
        ("/a/./test.whl", "a/test.whl"),
        ("/a//test.whl", "a/test.whl"),
    ],
)
def test_paths_resolve_under_base_and_stream(scheme, path, key):
    body = Body(b"wheel" * 100_000)
    store = Store(ObjectResponse(200, {"Content-Length": "500000"}, body))
    status, headers, data = request(store, path, base=f"{scheme}://bucket/public")
    assert status == 200 and headers["Content-Length"] == "500000"
    assert data == b"wheel" * 100_000 and body.closed
    assert store.calls == [(f"{scheme}://bucket/public/{key}", {"head": False, "byte_range": None, "etag": None})]


def test_local_symlink_escape_is_rejected(tmp_path):
    base = tmp_path / "public"
    base.mkdir()
    (base / "outside").symlink_to(tmp_path, target_is_directory=True)
    store = Store(ObjectResponse(200, {}))
    assert request(store, "/outside/private", base=str(base))[0] == 400
    assert not store.calls


@pytest.mark.parametrize(
    "method,headers", [("HEAD", [("Range", "garbage")]), ("GET", [("Range", "bytes=2-"), ("If-Range", '"old"')])]
)
def test_head_and_if_range_ignore_ranges(method, headers):
    body = Body(b"new object")
    store = Store(ObjectResponse(200, {"Content-Length": "10"}, body))
    status, _, data = request(store, "/asset.whl", method=method, headers=headers)
    assert status == 200 and data == (b"" if method == "HEAD" else b"new object")
    assert store.calls[0][1]["byte_range"] is None and body.closed


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH", "CONNECT"])
def test_writes_never_access_storage(method):
    store = Store(ObjectResponse(200, {}))
    assert request(store, "/test.whl", method=method)[0] == 501
    assert not store.calls


@pytest.mark.parametrize(
    "prefix",
    [
        "http://bucket/public",
        "s3://user@bucket/public",
        "s3://bucket/a/../b",
        "s3://bucket//",
        "s3://bucket/public?key=secret",
    ],
)
def test_invalid_configured_prefix(prefix):
    with pytest.raises(ValueError):
        storage_root(prefix)


@pytest.fixture
def routed_filesystem(monkeypatch):
    from rigging.filesystem import buckets

    class Filesystem:
        body = Body(b"wheel")
        opens = []
        routes = []
        metadata = {
            "type": "file",
            "size": 5,
            "ContentLength": 5,
            "ETag": '"etag"',
            "etag": "etag",
            "generation": "7",
            "ContentEncoding": "gzip",
            "LastModified": datetime(2026, 10, 11, 12, tzinfo=timezone.utc),
        }

        def info(self, key):
            return {name: value for name, value in self.metadata.items() if name != "ContentEncoding"}

        def call_s3(self, operation, **kwargs):
            assert operation == "head_object"
            assert kwargs == {"Bucket": "bucket", "Key": "public/test.whl"}
            return self.metadata

        def open(self, key, mode, **kwargs):
            self.opens.append((key, mode, kwargs))
            return self.body

    fs = Filesystem()

    def resolve(url):
        fs.routes.append(url)
        return fs, url.removeprefix("s3://").removeprefix("gs://")

    monkeypatch.setattr(buckets, "filesystem_for", resolve)
    yield fs
    fs.body.close()


@pytest.mark.parametrize("scheme", ["s3", "gs"])
def test_rigging_routing_ranges_and_metadata(routed_filesystem, scheme):
    path = object_path(storage_root(f"{scheme}://bucket/public"), "/test.whl")
    response = RiggingStore().fetch(path, head=False, byte_range="bytes=1-3", etag=None)
    store = Store(response)
    status, headers, data = request(store, "/test.whl")
    assert status == 206 and data == b"hee"
    assert headers["Content-Range"] == "bytes 1-3/5"
    assert headers.get("Content-Encoding") == ("gzip" if scheme == "s3" else None)
    assert headers["ETag"] == '"etag"' and headers["Last-Modified"] == "Sun, 11 Oct 2026 12:00:00 GMT"
    assert routed_filesystem.routes == ([str(path)] if scheme == "s3" else [str(path), str(path) + "#7"])
    options = routed_filesystem.opens[0][2]
    assert options["cache_type"] == "readahead"
    assert options["block_size"] == 3
    assert options.get("s3_additional_kwargs") == ({"IfMatch": '"etag"'} if scheme == "s3" else None)
    if scheme == "gs":
        assert routed_filesystem.opens[0][0].endswith("#7")


def test_encoded_gcs_object_is_explicitly_unsupported(routed_filesystem):
    routed_filesystem.metadata["contentEncoding"] = "gzip"
    result = RiggingStore().fetch(storage_root("gs://bucket/public/test.whl"), head=False, byte_range=None, etag=None)
    assert result.status == 501 and result.body is None
    assert not routed_filesystem.opens


def test_actual_gcs_range_reads_keep_the_selected_generation(monkeypatch):
    import gcsfs
    from rigging.filesystem import buckets
    from rigging.filesystem.factory import url_to_fs

    from ports.buildomatic import s3_proxy

    original = b"old version " * 30_000
    requests = []

    async def info(self, path, **kwargs):
        return {"name": path, "type": "file", "size": len(original), "generation": "7", "etag": "old"}

    async def call(self, method, url, *args, headers=None, **kwargs):
        requests.append(url)
        assert "generation=7" in url
        first, last = headers["Range"].removeprefix("bytes=").split("-")
        return {}, original[int(first) : int(last) + 1]

    monkeypatch.setattr(gcsfs.GCSFileSystem, "_info", info)
    monkeypatch.setattr(gcsfs.GCSFileSystem, "_call", call)
    monkeypatch.setattr(buckets, "filesystem_for", lambda url: url_to_fs(url, token="anon", skip_instance_cache=True))
    monkeypatch.setattr(s3_proxy, "_READ_AHEAD", 64 * 1024)
    response = RiggingStore().fetch(storage_root("gs://bucket/public/test.whl"), head=False, byte_range=None, etag=None)
    status, _, data = request(Store(response), "/test.whl")
    assert status == 200 and data == original
    assert len(requests) >= 3


@pytest.mark.parametrize("etag", ['"etag"', 'W/"etag"', '"other", "etag"', "*"])
def test_conditional_reads_do_not_open_body(routed_filesystem, etag):
    result = RiggingStore().fetch(storage_root("s3://bucket/public/test.whl"), head=False, byte_range=None, etag=etag)
    assert result.status == 304 and result.body is None
    assert not routed_filesystem.opens


@pytest.mark.parametrize("byte_range", ["bytes=0-1,3-4", "items=1-3", "bytes=-", "bytes=5-", "bytes=3-1", "bytes=-0"])
def test_invalid_or_unsatisfiable_ranges(routed_filesystem, byte_range):
    result = RiggingStore().fetch(
        storage_root("s3://bucket/public/test.whl"), head=False, byte_range=byte_range, etag=None
    )
    assert result.status == 416 and result.headers["Content-Range"] == "bytes */5"
    assert not routed_filesystem.opens


@pytest.mark.parametrize(
    "byte_range,expected", [("bytes=3-", "bytes 3-4/5"), ("bytes=-2", "bytes 3-4/5"), ("bytes=0-99", "bytes 0-4/5")]
)
def test_open_suffix_and_clamped_ranges(routed_filesystem, byte_range, expected):
    result = RiggingStore().fetch(
        storage_root("s3://bucket/public/test.whl"), head=False, byte_range=byte_range, etag=None
    )
    assert result.status == 206 and result.headers["Content-Range"] == expected


@pytest.mark.parametrize("error,status", [(FileNotFoundError, 404), (PermissionError, 403), (OSError, 502)])
def test_provider_errors_expose_only_status(monkeypatch, error, status):
    from rigging.filesystem import buckets

    def fail(url):
        raise error("private provider diagnostic")

    monkeypatch.setattr(buckets, "filesystem_for", fail)
    result = RiggingStore().fetch(storage_root("gs://bucket/public/test.whl"), head=False, byte_range=None, etag=None)
    assert result.status == status and result.headers == {} and result.body is None


def test_real_local_storage(tmp_path):
    base = tmp_path / "public"
    base.mkdir()
    (base / "hello.json").write_bytes(b'{"ok":true}')
    result = RiggingStore().fetch(
        object_path(storage_root(str(base)), "/hello.json"), head=False, byte_range=None, etag=None
    )
    status, headers, data = request(Store(result), "/hello.json")
    assert status == 200 and data == b'{"ok":true}'
    assert headers["Content-Type"] == "application/json"
