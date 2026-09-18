"""HTTP client objects over shellsim's deterministic route broker."""

import _shellsim_http


HTTP_PORT = 80
HTTPS_PORT = 443

responses = {
    200: "OK",
    201: "Created",
    202: "Accepted",
    204: "No Content",
    301: "Moved Permanently",
    302: "Found",
    303: "See Other",
    304: "Not Modified",
    307: "Temporary Redirect",
    308: "Permanent Redirect",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}


class HTTPException(Exception):
    def __init__(self, message=""):
        self.message = message

    def __str__(self):
        return str(self.message)


class NotConnected(HTTPException):
    pass


class InvalidURL(HTTPException):
    pass


def _header(items, name, default=None):
    lowered = name.lower()
    for key, value in items:
        if key.lower() == lowered:
            return value
    return default


def _has_header(items, name):
    return _header(items, name) is not None


class HTTPMessage:
    def __init__(self, items=()):
        self._items = list(items)

    def get(self, name, failobj=None):
        return _header(self._items, name, failobj)

    def get_all(self, name, failobj=None):
        lowered = name.lower()
        values = [value for key, value in self._items if key.lower() == lowered]
        return values if values else failobj

    def items(self):
        return list(self._items)

    def keys(self):
        return [name for name, value in self._items]

    def values(self):
        return [value for name, value in self._items]

    def __getitem__(self, name):
        return self.get(name)

    def __contains__(self, name):
        return self.get(name) is not None

    def __iter__(self):
        return iter(self.keys())


class HTTPResponse:
    def __init__(self, url, raw):
        self.url = url
        self.status = raw[0]
        self.code = self.status
        self.reason = responses.get(self.status, "")
        self.headers = HTTPMessage(raw[1])
        self.msg = self.headers
        self.version = 11
        self.length = len(raw[2])
        self.chunked = False
        self.will_close = False
        self.closed = False
        self._body = raw[2]
        self._position = 0

    def read(self, amt=None):
        if self.closed:
            return b""
        if amt is None or amt < 0:
            value = self._body[self._position:]
            self._position = len(self._body)
        else:
            value = self._body[self._position:self._position + amt]
            self._position += len(value)
        return value

    def readinto(self, buffer):
        if not isinstance(buffer, bytearray):
            raise TypeError("readinto() requires a bytearray")
        value = self.read(len(buffer))
        buffer[:len(value)] = value
        return len(value)

    def readline(self, limit=-1):
        if self.closed or self._position >= len(self._body):
            return b""
        end = self._position
        maximum = len(self._body) if limit < 0 else min(len(self._body), self._position + limit)
        while end < maximum and self._body[end] != 10:
            end += 1
        if end < maximum:
            end += 1
        value = self._body[self._position:end]
        self._position = end
        return value

    def readlines(self):
        result = []
        line = self.readline()
        while line:
            result.append(line)
            line = self.readline()
        return result

    def getheader(self, name, default=None):
        values = self.headers.get_all(name)
        if values is None:
            return default
        return ", ".join(values)

    def getheaders(self):
        return self.headers.items()

    def getcode(self):
        return self.status

    def geturl(self):
        return self.url

    def info(self):
        return self.headers

    def isclosed(self):
        return self.closed

    def close(self):
        self.closed = True

    def __iter__(self):
        return self

    def __next__(self):
        value = self.readline()
        if not value:
            raise StopIteration
        return value

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
        return False


def _body_bytes(body):
    if body is None:
        return b""
    if isinstance(body, str):
        return body.encode("utf-8")
    if isinstance(body, bytes):
        return body
    if isinstance(body, bytearray):
        return bytes(body)
    raise TypeError("HTTP request body must be bytes-like or str")


def _request(method, url, body=None, headers=None):
    method = str(method).upper()
    url = str(url)
    body = _body_bytes(body)
    items = []
    if headers is not None:
        items = list(headers.items()) if hasattr(headers, "items") else list(headers)
    if body and not _has_header(items, "Content-Length"):
        items.append(("Content-Length", str(len(body))))
    raw = _shellsim_http.request(method, url, items, body)
    return None if raw is None else HTTPResponse(url, raw)


class HTTPConnection:
    default_port = HTTP_PORT
    _scheme = "http"

    def __init__(self, host, port=None, timeout=None, source_address=None, blocksize=8192):
        host = str(host)
        if "/" in host or host == "":
            raise InvalidURL("invalid host: " + repr(host))
        self.host = host
        self.port = port
        self.timeout = timeout
        self.source_address = source_address
        self.blocksize = blocksize
        self._response = None

    def connect(self):
        return None

    def close(self):
        if self._response is not None:
            self._response.close()
        self._response = None

    def _absolute_url(self, url):
        url = str(url)
        if url.startswith("http://") or url.startswith("https://"):
            return url
        if not url.startswith("/"):
            url = "/" + url
        authority = self.host
        if self.port is not None and self.port != self.default_port:
            authority += ":" + str(self.port)
        return self._scheme + "://" + authority + url

    def request(self, method, url, body=None, headers=None, encode_chunked=False):
        if encode_chunked:
            raise ValueError("chunked request encoding is not supported by shellsim")
        items = {} if headers is None else headers
        self._response = _request(method, self._absolute_url(url), body, items)
        if self._response is None:
            raise OSError("no matching virtual HTTP route")

    def getresponse(self):
        if self._response is None:
            raise NotConnected("request has not been sent")
        return self._response

    def set_tunnel(self, host, port=None, headers=None):
        raise ValueError("HTTP CONNECT tunnels are not supported by shellsim")


class HTTPSConnection(HTTPConnection):
    default_port = HTTPS_PORT
    _scheme = "https"

    def __init__(self, host, port=None, timeout=None, source_address=None,
                 context=None, check_hostname=None, blocksize=8192):
        HTTPConnection.__init__(self, host, port, timeout, source_address, blocksize)
