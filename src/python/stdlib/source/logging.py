"""Deterministic no-output logging facade for simulated programs."""

CRITICAL = 50
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0


class Logger:
    def __init__(self):
        self.handlers = []

    def debug(self, message, *args):
        pass

    def info(self, message, *args):
        pass

    def warning(self, message, *args):
        pass

    def error(self, message, *args):
        pass

    def exception(self, message, *args):
        pass

    def critical(self, message, *args):
        pass

    def setLevel(self, level):
        pass

    def addHandler(self, handler):
        self.handlers.append(handler)


class Formatter:
    def __init__(self, format=None, datefmt=None):
        self.format_string = format
        self.datefmt = datefmt


class StreamHandler:
    def __init__(self, stream=None):
        self.stream = stream
        self.formatter = None

    def setFormatter(self, formatter):
        self.formatter = formatter

    def setLevel(self, level):
        pass


class FileHandler(StreamHandler):
    def __init__(self, filename, mode="a", encoding=None):
        StreamHandler.__init__(self, open(filename, mode))


_root = Logger()


def getLogger(name=None):
    return _root


def basicConfig(level=NOTSET, format=None, filename=None):
    pass


def debug(message, *args):
    return _root.debug(message, *args)


def info(message, *args):
    return _root.info(message, *args)


def warning(message, *args):
    return _root.warning(message, *args)


def error(message, *args):
    return _root.error(message, *args)
