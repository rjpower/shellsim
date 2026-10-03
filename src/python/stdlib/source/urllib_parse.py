"""URL parsing, quoting and query-string helpers (``urllib.parse``)."""

__all__ = [
    "urlparse", "urlunparse", "urljoin", "urldefrag", "urlsplit", "urlunsplit", "urlencode",
    "parse_qs", "parse_qsl", "quote", "quote_plus", "quote_from_bytes", "unquote",
    "unquote_plus", "unquote_to_bytes", "DefragResult", "ParseResult", "SplitResult",
    "uses_relative", "uses_netloc", "uses_params", "uses_query", "uses_fragment",
    "scheme_chars", "MAX_CACHE_SIZE", "non_hierarchical",
]

uses_relative = [
    "", "ftp", "http", "gopher", "nntp", "imap", "wais", "file", "https", "shttp", "mms",
    "prospero", "rtsp", "rtsps", "rtspu", "sftp", "svn", "svn+ssh", "ws", "wss",
]
uses_netloc = [
    "", "ftp", "http", "gopher", "nntp", "telnet", "imap", "wais", "file", "mms", "https",
    "shttp", "snews", "prospero", "rtsp", "rtsps", "rtspu", "rsync", "svn", "svn+ssh", "sftp",
    "nfs", "git", "git+ssh", "ws", "wss", "itms-services",
]
uses_params = [
    "", "ftp", "hdl", "prospero", "http", "imap", "https", "shttp", "rtsp", "rtsps", "rtspu",
    "sip", "sips", "mms", "sftp", "tel",
]
non_hierarchical = ["gopher", "hdl", "mailto", "news", "telnet", "wais", "imap", "snews", "sip", "sips"]
uses_query = ["", "http", "wais", "imap", "https", "shttp", "mms", "gopher", "rtsp", "rtsps", "rtspu", "sip", "sips"]
uses_fragment = ["", "ftp", "hdl", "http", "gopher", "news", "nntp", "wais", "https", "shttp", "snews", "file", "prospero"]
scheme_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-."
MAX_CACHE_SIZE = 20
_ALWAYS_SAFE = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-~"
)
_HEX = "0123456789ABCDEF"


class _NetlocMixin:
    def _netloc_parts(self):
        netloc = self.netloc
        userinfo = None
        hostinfo = netloc
        if "@" in netloc:
            userinfo, hostinfo = netloc.rsplit("@", 1)
        return userinfo, hostinfo

    @property
    def username(self):
        userinfo, _ = self._netloc_parts()
        if userinfo is None:
            return None
        return userinfo.split(":", 1)[0]

    @property
    def password(self):
        userinfo, _ = self._netloc_parts()
        if userinfo is None or ":" not in userinfo:
            return None
        return userinfo.split(":", 1)[1]

    @property
    def hostname(self):
        _, hostinfo = self._netloc_parts()
        if hostinfo.startswith("["):
            return hostinfo[1 : hostinfo.find("]")].lower()
        host = hostinfo.split(":", 1)[0]
        return host.lower() if host else None

    @property
    def port(self):
        _, hostinfo = self._netloc_parts()
        if hostinfo.startswith("["):
            hostinfo = hostinfo[hostinfo.find("]") + 1 :]
        elif ":" in hostinfo:
            hostinfo = hostinfo[hostinfo.find(":") :]
        else:
            return None
        if not hostinfo.startswith(":"):
            return None
        port_text = hostinfo[1:]
        if not port_text:
            return None
        if not port_text.isdigit():
            raise ValueError("Port could not be cast to integer value as %r" % port_text)
        port = int(port_text)
        if not 0 <= port <= 65535:
            raise ValueError("Port out of range 0-65535")
        return port

    def geturl(self):
        return self._geturl()


class SplitResult(_NetlocMixin, tuple):
    _fields = ("scheme", "netloc", "path", "query", "fragment")

    def __new__(cls, scheme, netloc, path, query, fragment):
        return tuple.__new__(cls, (scheme, netloc, path, query, fragment))

    scheme = property(lambda self: self[0])
    netloc = property(lambda self: self[1])
    path = property(lambda self: self[2])
    query = property(lambda self: self[3])
    fragment = property(lambda self: self[4])

    def _geturl(self):
        return urlunsplit(self)

    def _replace(self, **changes):
        values = dict(zip(self._fields, self))
        values.update(changes)
        return SplitResult(*[values[field] for field in self._fields])

    def _asdict(self):
        return dict(zip(self._fields, self))

    def __repr__(self):
        return "SplitResult(" + ", ".join(
            field + "=" + repr(value) for field, value in zip(self._fields, self)
        ) + ")"


class ParseResult(_NetlocMixin, tuple):
    _fields = ("scheme", "netloc", "path", "params", "query", "fragment")

    def __new__(cls, scheme, netloc, path, params, query, fragment):
        return tuple.__new__(cls, (scheme, netloc, path, params, query, fragment))

    scheme = property(lambda self: self[0])
    netloc = property(lambda self: self[1])
    path = property(lambda self: self[2])
    params = property(lambda self: self[3])
    query = property(lambda self: self[4])
    fragment = property(lambda self: self[5])

    def _geturl(self):
        return urlunparse(self)

    def _replace(self, **changes):
        values = dict(zip(self._fields, self))
        values.update(changes)
        return ParseResult(*[values[field] for field in self._fields])

    def _asdict(self):
        return dict(zip(self._fields, self))

    def __repr__(self):
        return "ParseResult(" + ", ".join(
            field + "=" + repr(value) for field, value in zip(self._fields, self)
        ) + ")"


class DefragResult(tuple):
    _fields = ("url", "fragment")

    def __new__(cls, url, fragment):
        return tuple.__new__(cls, (url, fragment))

    url = property(lambda self: self[0])
    fragment = property(lambda self: self[1])

    def geturl(self):
        if self.fragment:
            return self.url + "#" + self.fragment
        return self.url

    def __repr__(self):
        return "DefragResult(url=%r, fragment=%r)" % (self[0], self[1])


def _coerce(value):
    if isinstance(value, bytes):
        return value.decode("ascii"), True
    return value, False


def _result(value, was_bytes):
    if was_bytes:
        return value.encode("ascii")
    return value


def urlsplit(url, scheme="", allow_fragments=True):
    url, was_bytes = _coerce(url)
    scheme, _ = _coerce(scheme)
    url = url.lstrip(" \t\r\n\f")
    url = url.replace("\t", "").replace("\r", "").replace("\n", "")
    netloc = query = fragment = ""
    index = url.find(":")
    if index > 0 and url[0].isalpha():
        candidate = url[:index]
        if all(char in scheme_chars for char in candidate):
            scheme, url = candidate.lower(), url[index + 1 :]
    if url[:2] == "//":
        end = len(url)
        for delimiter in "/?#":
            found = url.find(delimiter, 2)
            if found >= 0:
                end = min(end, found)
        netloc, url = url[2:end], url[end:]
        if ("[" in netloc and "]" not in netloc) or ("]" in netloc and "[" not in netloc):
            raise ValueError("Invalid IPv6 URL")
    if allow_fragments and "#" in url:
        url, fragment = url.split("#", 1)
    if "?" in url:
        url, query = url.split("?", 1)
    parts = (scheme, netloc, url, query, fragment)
    if was_bytes:
        parts = tuple(part.encode("ascii") for part in parts)
    return SplitResult(*parts)


def _splitparams(url):
    if "/" in url:
        index = url.find(";", url.rfind("/"))
        if index < 0:
            return url, ""
    else:
        index = url.find(";")
    return url[:index], url[index + 1 :]


def urlparse(url, scheme="", allow_fragments=True):
    split = urlsplit(url, scheme, allow_fragments)
    scheme, netloc, path, query, fragment = split
    text_path, was_bytes = _coerce(path)
    params = ""
    if (_coerce(scheme)[0] in uses_params) and ";" in text_path:
        text_path, params = _splitparams(text_path)
    return ParseResult(
        scheme, netloc, _result(text_path, was_bytes), _result(params, was_bytes), query, fragment
    )


def urlunsplit(components):
    scheme, netloc, url, query, fragment = [_coerce(part)[0] for part in components]
    was_bytes = isinstance(components[0], bytes)
    if netloc or (scheme and scheme in uses_netloc and url[:2] != "//"):
        if url and url[:1] != "/":
            url = "/" + url
        url = "//" + netloc + url
    if scheme:
        url = scheme + ":" + url
    if query:
        url = url + "?" + query
    if fragment:
        url = url + "#" + fragment
    return _result(url, was_bytes)


def urlunparse(components):
    scheme, netloc, url, params, query, fragment = [_coerce(part)[0] for part in components]
    was_bytes = isinstance(components[0], bytes)
    if params:
        url = url + ";" + params
    return _result(urlunsplit((scheme, netloc, url, query, fragment)), was_bytes)


def urljoin(base, url, allow_fragments=True):
    if not base:
        return url
    if not url:
        return base
    base, base_bytes = _coerce(base)
    url, _ = _coerce(url)
    bscheme, bnetloc, bpath, bparams, bquery, bfragment = urlparse(base, "", allow_fragments)
    scheme, netloc, path, params, query, fragment = urlparse(url, bscheme, allow_fragments)
    if scheme != bscheme or scheme not in uses_relative:
        return _result(url, base_bytes)
    if scheme in uses_netloc:
        if netloc:
            return _result(urlunparse((scheme, netloc, path, params, query, fragment)), base_bytes)
        netloc = bnetloc
    if not path and not params:
        path = bpath
        params = bparams
        if not query:
            query = bquery
        return _result(urlunparse((scheme, netloc, path, params, query, fragment)), base_bytes)
    base_parts = bpath.split("/")
    if base_parts[-1] != "":
        del base_parts[-1]
    if path[:1] == "/":
        segments = path.split("/")
    else:
        segments = base_parts + path.split("/")
        segments[1:-1] = [segment for segment in segments[1:-1] if segment]
    resolved = []
    for segment in segments:
        if segment == "..":
            if len(resolved) > 1:
                resolved.pop()
        elif segment == ".":
            continue
        else:
            resolved.append(segment)
    if segments[-1] in (".", ".."):
        resolved.append("")
    joined = "/".join(resolved)
    if not joined.startswith("/") and path[:1] == "/":
        joined = "/" + joined
    return _result(urlunparse((scheme, netloc, joined or "/", params, query, fragment)), base_bytes)


def urldefrag(url):
    url, was_bytes = _coerce(url)
    if "#" in url:
        scheme, netloc, path, params, query, fragment = urlparse(url)
        defrag = urlunparse((scheme, netloc, path, params, query, ""))
    else:
        defrag, fragment = url, ""
    return DefragResult(_result(defrag, was_bytes), _result(fragment, was_bytes))


def unquote_to_bytes(string):
    if isinstance(string, str):
        string = string.encode("utf-8")
    if b"%" not in string:
        return bytes(string)
    pieces = string.split(b"%")
    result = [pieces[0]]
    for piece in pieces[1:]:
        hexpart = piece[:2]
        if len(hexpart) == 2 and all(chr(c) in "0123456789abcdefABCDEF" for c in hexpart):
            result.append(bytes([int(hexpart.decode("ascii"), 16)]) + piece[2:])
        else:
            result.append(b"%" + piece)
    return b"".join(result)


def unquote(string, encoding="utf-8", errors="replace"):
    if isinstance(string, bytes):
        return unquote_to_bytes(string).decode(encoding, errors)
    if "%" not in string:
        return string
    pieces = string.split("%")
    result = [pieces[0]]
    pending = b""
    for piece in pieces[1:]:
        hexpart = piece[:2]
        if len(hexpart) == 2 and all(char in "0123456789abcdefABCDEF" for char in hexpart):
            pending += bytes([int(hexpart, 16)])
            rest = piece[2:]
        else:
            rest = "%" + piece
        if rest or piece is pieces[-1]:
            if pending:
                result.append(pending.decode(encoding, errors))
                pending = b""
            result.append(rest)
    if pending:
        result.append(pending.decode(encoding, errors))
    return "".join(result)


def unquote_plus(string, encoding="utf-8", errors="replace"):
    return unquote(string.replace("+", " "), encoding, errors)


def quote_from_bytes(bs, safe="/"):
    if not isinstance(bs, (bytes, bytearray)):
        raise TypeError("quote_from_bytes() expected bytes")
    if isinstance(safe, str):
        safe = safe.encode("ascii", "ignore")
    safe_set = _ALWAYS_SAFE.union(safe)
    result = []
    for byte in bs:
        if byte in safe_set:
            result.append(chr(byte))
        else:
            result.append("%" + _HEX[byte >> 4] + _HEX[byte & 15])
    return "".join(result)


def quote(string, safe="/", encoding=None, errors=None):
    if isinstance(string, str):
        if not string:
            return string
        if encoding is None:
            encoding = "utf-8"
        if errors is None:
            errors = "strict"
        string = string.encode(encoding, errors)
    elif encoding is not None:
        raise TypeError("quote() doesn't support 'encoding' for bytes")
    elif errors is not None:
        raise TypeError("quote() doesn't support 'errors' for bytes")
    return quote_from_bytes(string, safe)


def quote_plus(string, safe="", encoding=None, errors=None):
    if (isinstance(string, str) and " " not in string) or (
        isinstance(string, bytes) and b" " not in string
    ):
        return quote(string, safe, encoding, errors)
    space = " " if isinstance(safe, str) else b" "
    string = quote(string, safe + space, encoding, errors)
    return string.replace(" ", "+")


def parse_qsl(qs, keep_blank_values=False, strict_parsing=False, encoding="utf-8",
              errors="replace", max_num_fields=None, separator="&"):
    qs, was_bytes = _coerce(qs) if qs is not None else ("", False)
    if not qs:
        return []
    pairs = qs.split(separator)
    if max_num_fields is not None and len(pairs) > max_num_fields:
        raise ValueError("Max number of fields exceeded")
    result = []
    for pair in pairs:
        if not pair:
            continue
        if "=" in pair:
            name, value = pair.split("=", 1)
        elif strict_parsing:
            raise ValueError("bad query field: %r" % pair)
        elif keep_blank_values:
            name, value = pair, ""
        else:
            continue
        if value or keep_blank_values:
            name = unquote_plus(name, encoding, errors)
            value = unquote_plus(value, encoding, errors)
            result.append((_result(name, was_bytes), _result(value, was_bytes)))
    return result


def parse_qs(qs, keep_blank_values=False, strict_parsing=False, encoding="utf-8",
             errors="replace", max_num_fields=None, separator="&"):
    result = {}
    for name, value in parse_qsl(
        qs, keep_blank_values, strict_parsing, encoding, errors, max_num_fields, separator
    ):
        result.setdefault(name, []).append(value)
    return result


def urlencode(query, doseq=False, safe="", encoding=None, errors=None, quote_via=quote_plus):
    if hasattr(query, "items"):
        query = list(query.items())
    else:
        query = list(query)
        if query and not isinstance(query[0], (tuple, list)):
            raise TypeError("not a valid non-string sequence or mapping object")

    def encode(value):
        if isinstance(value, bytes):
            return quote_via(value, safe)
        return quote_via(str(value), safe, encoding, errors)

    pieces = []
    for key, value in query:
        key_text = encode(key)
        if doseq and not isinstance(value, (str, bytes)):
            try:
                values = list(value)
            except TypeError:
                values = [value]
            for element in values:
                pieces.append(key_text + "=" + encode(element))
        else:
            pieces.append(key_text + "=" + encode(value))
    return "&".join(pieces)
