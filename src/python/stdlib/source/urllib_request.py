"""Common urllib.request behavior over shellsim's deterministic HTTP broker."""

import tempfile
from http.client import _request, responses
from urllib.error import HTTPError, URLError


def _find_header(items, name):
    lowered = name.lower()
    for key, value in items:
        if key.lower() == lowered:
            return value
    return None


class Request:
    def __init__(self, url, data=None, headers=None, origin_req_host=None,
                 unverifiable=False, method=None):
        self.full_url = str(url)
        self.data = data
        self.headers = {}
        self.unredirected_hdrs = {}
        self.origin_req_host = origin_req_host
        self.unverifiable = unverifiable
        self.method = method
        if headers is not None:
            for name, value in headers.items():
                self.add_header(name, value)

    def get_method(self):
        if self.method is not None:
            return str(self.method).upper()
        return "POST" if self.data is not None else "GET"

    def add_header(self, key, value):
        self.headers[str(key)] = str(value)

    def add_unredirected_header(self, key, value):
        self.unredirected_hdrs[str(key)] = str(value)

    def has_header(self, header_name):
        return self.get_header(header_name) is not None

    def get_header(self, header_name, default=None):
        lowered = header_name.lower()
        for source in (self.headers, self.unredirected_hdrs):
            for name, value in source.items():
                if name.lower() == lowered:
                    return value
        return default

    def remove_header(self, header_name):
        lowered = header_name.lower()
        for source in (self.headers, self.unredirected_hdrs):
            for name in list(source.keys()):
                if name.lower() == lowered:
                    del source[name]

    def header_items(self):
        return list(self.unredirected_hdrs.items()) + list(self.headers.items())


def _join_url(base, location):
    if location.startswith("http://") or location.startswith("https://"):
        return location
    scheme_parts = base.split("://", 1)
    if len(scheme_parts) != 2:
        raise URLError("invalid redirect base URL")
    scheme, remainder = scheme_parts
    authority_parts = remainder.split("/", 1)
    authority = scheme + "://" + authority_parts[0]
    if location.startswith("//"):
        return scheme + ":" + location
    if location.startswith("/"):
        return authority + location
    base_path = "/" if len(authority_parts) == 1 else "/" + authority_parts[1]
    path_parts = base_path.split("/")
    directory = "/".join(path_parts[:-1])
    return authority + directory + "/" + location


def _open(request, redirects):
    if not isinstance(request, Request):
        request = Request(request)
    method = request.get_method()
    headers = request.header_items()
    if request.data is not None and _find_header(headers, "Content-Type") is None:
        headers.append(("Content-Type", "application/x-www-form-urlencoded"))
    response = _request(method, request.full_url, request.data, headers)
    if response is None:
        raise URLError("no matching virtual HTTP route for " + request.full_url)
    if response.status in (301, 302, 303, 307, 308):
        location = response.getheader("Location")
        if location is None:
            raise HTTPError(request.full_url, response.status, response.reason,
                            response.headers, response)
        if redirects >= 10:
            raise HTTPError(request.full_url, response.status, "redirect limit exceeded",
                            response.headers, response)
        redirected = Request(_join_url(request.full_url, location), data=request.data,
                             headers=request.headers, method=method)
        if response.status == 303 or (response.status in (301, 302) and method == "POST"):
            redirected.data = None
            redirected.method = "GET"
        return _open(redirected, redirects + 1)
    if response.status >= 400:
        raise HTTPError(request.full_url, response.status,
                        responses.get(response.status, response.reason),
                        response.headers, response)
    return response


class OpenerDirector:
    def open(self, fullurl, data=None, timeout=None):
        request = fullurl if isinstance(fullurl, Request) else Request(fullurl, data=data)
        if isinstance(fullurl, Request) and data is not None:
            request.data = data
        return _open(request, 0)


_opener = OpenerDirector()


def build_opener(*handlers):
    if handlers:
        raise ValueError("custom urllib handlers are not supported by shellsim")
    return OpenerDirector()


def install_opener(opener):
    global _opener
    if not hasattr(opener, "open"):
        raise TypeError("opener must provide open()")
    _opener = opener


def urlopen(url, data=None, timeout=None, cafile=None, capath=None, cadefault=False, context=None):
    if cafile is not None or capath is not None or cadefault:
        raise ValueError("TLS certificate configuration is not modeled by shellsim")
    return _opener.open(url, data, timeout)


def urlretrieve(url, filename=None, reporthook=None, data=None):
    if reporthook is not None:
        raise ValueError("urlretrieve report hooks are not supported by shellsim")
    response = urlopen(url, data=data)
    body = response.read()
    if filename is None:
        stream = tempfile.NamedTemporaryFile(mode="wb", suffix=".download", delete=False)
        filename = stream.name
    else:
        stream = open(filename, "wb")
    with stream:
        stream.write(body)
    return (filename, response.headers)
