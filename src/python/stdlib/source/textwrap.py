"""Paragraph wrapping and indentation helpers.

``TextWrapper`` follows CPython's documented algorithm: the text is split into word and
whitespace chunks, long words are broken when allowed, and lines are filled greedily.
"""

import re

__all__ = ["TextWrapper", "wrap", "fill", "shorten", "dedent", "indent"]

_whitespace = "\t\n\x0b\x0c\r "


class TextWrapper:
    unicode_whitespace_trans = {ord(char): ord(" ") for char in _whitespace}
    sentence_end_re = re.compile(r"[a-z][\.\!\?][\"\']?\Z")

    def __init__(self, width=70, initial_indent="", subsequent_indent="", expand_tabs=True,
                 replace_whitespace=True, fix_sentence_endings=False, break_long_words=True,
                 drop_whitespace=True, break_on_hyphens=True, tabsize=8, *, max_lines=None,
                 placeholder=" [...]"):
        self.width = width
        self.initial_indent = initial_indent
        self.subsequent_indent = subsequent_indent
        self.expand_tabs = expand_tabs
        self.replace_whitespace = replace_whitespace
        self.fix_sentence_endings = fix_sentence_endings
        self.break_long_words = break_long_words
        self.drop_whitespace = drop_whitespace
        self.break_on_hyphens = break_on_hyphens
        self.tabsize = tabsize
        self.max_lines = max_lines
        self.placeholder = placeholder

    def _munge_whitespace(self, text):
        if self.expand_tabs:
            text = text.expandtabs(self.tabsize)
        if self.replace_whitespace:
            text = text.translate(self.unicode_whitespace_trans)
        return text

    def _split(self, text):
        """Split into whitespace and word chunks; with ``break_on_hyphens`` a word also ends
        after a hyphen that joins two lettered parts (``"well-known"`` gives ``"well-"`` and
        ``"known"``) and a run of dashes after a word stands alone."""
        chunks = []
        index = 0
        length = len(text)
        while index < length:
            start = index
            if text[index] in _whitespace:
                while index < length and text[index] in _whitespace:
                    index += 1
                chunks.append(text[start:index])
                continue
            while index < length and text[index] not in _whitespace:
                index += 1
            word = text[start:index]
            if self.break_on_hyphens:
                chunks.extend(_split_hyphens(word))
            else:
                chunks.append(word)
        return chunks

    def _fix_sentence_endings(self, chunks):
        index = 0
        while index < len(chunks) - 1:
            if chunks[index + 1] == " " and self.sentence_end_re.search(chunks[index]):
                chunks[index + 1] = "  "
                index += 2
            else:
                index += 1

    def _handle_long_word(self, reversed_chunks, cur_line, cur_len, width):
        space_left = 1 if width < 1 else width - cur_len
        if self.break_long_words:
            end = space_left
            chunk = reversed_chunks[-1]
            if self.break_on_hyphens and len(chunk) > space_left:
                hyphen = chunk.rfind("-", 0, space_left)
                if hyphen > 0 and any(char != "-" for char in chunk[:hyphen]):
                    end = hyphen + 1
            cur_line.append(chunk[:end])
            reversed_chunks[-1] = chunk[end:]
        elif not cur_line:
            cur_line.append(reversed_chunks.pop())

    def _wrap_chunks(self, chunks):
        lines = []
        if self.width <= 0:
            raise ValueError("invalid width %r (must be > 0)" % self.width)
        if self.max_lines is not None:
            indent = self.subsequent_indent if self.max_lines > 1 else self.initial_indent
            if len(indent) + len(self.placeholder.lstrip()) > self.width:
                raise ValueError("placeholder too large for max width")
        chunks.reverse()
        while chunks:
            cur_line = []
            cur_len = 0
            indent = self.subsequent_indent if lines else self.initial_indent
            width = self.width - len(indent)
            if self.drop_whitespace and chunks[-1].strip() == "" and lines:
                del chunks[-1]
            while chunks:
                length = len(chunks[-1])
                if cur_len + length <= width:
                    cur_line.append(chunks.pop())
                    cur_len += length
                else:
                    break
            if chunks and len(chunks[-1]) > width:
                self._handle_long_word(chunks, cur_line, cur_len, width)
                cur_len = sum(map(len, cur_line))
            if self.drop_whitespace and cur_line and cur_line[-1].strip() == "":
                cur_len -= len(cur_line[-1])
                del cur_line[-1]
            if cur_line:
                if (self.max_lines is None or len(lines) + 1 < self.max_lines
                        or (not chunks or self.drop_whitespace and len(chunks) == 1
                            and not chunks[0].strip()) and cur_len <= width):
                    lines.append(indent + "".join(cur_line))
                else:
                    while cur_line:
                        if cur_line[-1].strip() and cur_len + len(self.placeholder) <= width:
                            lines.append(indent + "".join(cur_line) + self.placeholder)
                            break
                        cur_len -= len(cur_line[-1])
                        del cur_line[-1]
                    else:
                        if lines:
                            prev_line = lines[-1].rstrip()
                            if len(prev_line) + len(self.placeholder) <= self.width:
                                lines[-1] = prev_line + self.placeholder
                                break
                        lines.append(indent + self.placeholder.lstrip())
                    break
        return lines

    def _split_chunks(self, text):
        text = self._munge_whitespace(text)
        return self._split(text)

    def wrap(self, text):
        chunks = self._split_chunks(text)
        if self.fix_sentence_endings:
            self._fix_sentence_endings(chunks)
        return self._wrap_chunks(chunks)

    def fill(self, text):
        return "\n".join(self.wrap(text))


def _split_hyphens(word):
    """Chunks of one whitespace-free word at the hyphens where a line may break."""
    pieces = []
    start = 0
    index = 0
    while index < len(word):
        if word[index] != "-":
            index += 1
            continue
        run_end = index
        while run_end < len(word) and word[run_end] == "-":
            run_end += 1
        before = word[start:index]
        after = word[run_end:]
        if run_end - index >= 2:
            # A dash run after word characters is its own chunk when more word follows.
            if before and after and (before[-1].isalnum() or before[-1] in "!\"'&.,?") and after[0].isalnum():
                pieces.append(before)
                pieces.append(word[index:run_end])
                start = run_end
            index = run_end
            continue
        # A single hyphen breaks after itself when letters surround it: at least two letters
        # (or a letter-hyphen-letter pair) precede it and a letter, optionally hyphenated,
        # follows.
        preceded = len(before) >= 2 and before[-1].isalpha() and (
            before[-2].isalpha() or (before[-2] == "-" and len(before) >= 3 and before[-3].isalpha())
        )
        followed = len(after) >= 1 and after[0].isalpha() and (
            len(after) >= 2 and (after[1].isalpha() or (after[1] == "-" and len(after) >= 3 and after[2].isalpha()))
        )
        if preceded and followed:
            pieces.append(word[start:run_end])
            start = run_end
        index = run_end
    if start < len(word):
        pieces.append(word[start:])
    return pieces


def wrap(text, width=70, **kwargs):
    return TextWrapper(width=width, **kwargs).wrap(text)


def fill(text, width=70, **kwargs):
    return TextWrapper(width=width, **kwargs).fill(text)


def shorten(text, width, **kwargs):
    wrapper = TextWrapper(width=width, max_lines=1, **kwargs)
    return wrapper.fill(" ".join(text.strip().split()))


_whitespace_only_re = re.compile("^[ \t]+$", re.MULTILINE)
_leading_whitespace_re = re.compile("(^[ \t]*)(?:[^ \t\n])", re.MULTILINE)


def dedent(text):
    margin = None
    text = _whitespace_only_re.sub("", text)
    indents = _leading_whitespace_re.findall(text)
    for indent in indents:
        if margin is None:
            margin = indent
        elif indent.startswith(margin):
            pass
        elif margin.startswith(indent):
            margin = indent
        else:
            for index, (left, right) in enumerate(zip(indent, margin)):
                if left != right:
                    margin = margin[:index]
                    break
    if margin:
        text = re.sub(r"(?m)^" + margin, "", text)
    return text


def indent(text, prefix, predicate=None):
    if predicate is None:
        def predicate(line):
            return line.strip()
    prefixed = []
    for line in text.splitlines(True):
        prefixed.append(prefix + line if predicate(line) else line)
    return "".join(prefixed)
