"""Small text wrapping helpers for ordinary prose."""


def _leading_whitespace(line):
    count = 0
    while count < len(line) and line[count] in " \t":
        count += 1
    return count


def dedent(text):
    lines = text.splitlines(True)
    margin = None
    for line in lines:
        stripped = line.lstrip(" \t\r\n")
        if stripped != "":
            width = _leading_whitespace(line)
            if margin is None or width < margin:
                margin = width
    if margin is None or margin == 0:
        return text
    return "".join(line[margin:] if line.lstrip(" \t\r\n") != "" else line for line in lines)


def indent(text, prefix, predicate=None):
    lines = text.splitlines(True)
    result = []
    for line in lines:
        selected = predicate(line) if predicate is not None else line.strip() != ""
        result.append(prefix + line if selected else line)
    return "".join(result)


def wrap(text, width=70, initial_indent="", subsequent_indent=""):
    if width <= 0:
        raise ValueError("invalid width")
    words = text.split()
    if len(words) == 0:
        return []
    lines = []
    line = initial_indent
    indent_value = initial_indent
    for word in words:
        separator = "" if line == indent_value else " "
        if len(line) + len(separator) + len(word) <= width or line == indent_value:
            line += separator + word
        else:
            lines.append(line)
            indent_value = subsequent_indent
            line = indent_value + word
    lines.append(line)
    return lines


def fill(text, width=70, initial_indent="", subsequent_indent=""):
    return "\n".join(wrap(text, width, initial_indent, subsequent_indent))


def shorten(text, width, placeholder=" [...]"):
    collapsed = " ".join(text.split())
    if len(collapsed) <= width:
        return collapsed
    if len(placeholder) > width:
        raise ValueError("placeholder too large for max width")
    available = width - len(placeholder)
    words = collapsed.split()
    result = ""
    for word in words:
        candidate = word if result == "" else result + " " + word
        if len(candidate) > available:
            break
        result = candidate
    return result + placeholder
