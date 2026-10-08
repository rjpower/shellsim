"""HTML text escaping without parsing or host I/O."""


def escape(s, quote=True):
    """Replace markup characters in a string, including quotes by default."""
    s = str(s)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if quote:
        s = s.replace('"', "&quot;").replace("'", "&#x27;")
    return s
