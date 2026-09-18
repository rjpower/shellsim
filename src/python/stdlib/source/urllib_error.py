"""Exceptions shared by shellsim's urllib.request facade."""


class URLError(OSError):
    def __init__(self, reason):
        self.reason = reason

    def __str__(self):
        return "<urlopen error " + str(self.reason) + ">"


class HTTPError(URLError):
    def __init__(self, url, code, msg, hdrs, fp):
        self.url = url
        self.filename = url
        self.code = code
        self.status = code
        self.msg = msg
        self.reason = msg
        self.hdrs = hdrs
        self.headers = hdrs
        self.fp = fp

    def __str__(self):
        return "HTTP Error " + str(self.code) + ": " + str(self.msg)

    def read(self, amt=None):
        return self.fp.read(amt)

    def readline(self, limit=-1):
        return self.fp.readline(limit)

    def geturl(self):
        return self.url

    def info(self):
        return self.hdrs

    def getcode(self):
        return self.code

    def close(self):
        self.fp.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
        return False
