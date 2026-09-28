"""NumPy-compatible text for arrays: print options, ``format_float_*``, and the
``array2string``/``array_repr``/``array_str`` family that ``ndarray.__repr__``/``__str__`` call.

Every digit shellsim prints comes from CPython's own float text: `str(x)` already gives the
shortest decimal that round-trips back to `x` at `x`'s own precision (a NumPy scalar's `str`
honors its dtype, so a `float32` element's shortest digits differ from a `float64` holding the
same value), and `format(x, ".Nf"/".Ne")` gives `x` correctly rounded to `N` digits. This module
never generates digits itself: it decomposes `str(x)`'s text into a leading sign, significant
digits and a decimal exponent (`_decompose`), then reuses that decomposition or asks `format` for
a fresh rounding, and otherwise only does string bookkeeping (trimming, padding, line wrapping).

Array layout works in two passes per axis of formattable (bool/int/float/complex) elements: a
discovery pass formats every element that will actually be shown (see `_visible`) without
padding, to learn column widths and, for floats, whether the whole array should switch to
scientific notation; a second pass reformats each element with the resolved padding so a column
lines up. String and object elements are not column-aligned, matching NumPy.

Dropped relative to NumPy: the ``formatter`` dict, ``legacy`` modes, and the ``nanstr``/``infstr``
overrides (`nan`/`inf` are always fixed strings). ``format_float_positional``/``_scientific``
keep only ``precision``, ``unique``, ``trim``, ``pad_left`` and ``pad_right``.
"""

import contextlib
import math

_DEFAULTS = {
    "precision": 8,
    "threshold": 1000,
    "edgeitems": 3,
    "linewidth": 75,
    "suppress": False,
    "sign": "-",
    "floatmode": "maxprec",
}

_options = dict(_DEFAULTS)

_ELLIPSIS = object()


def get_printoptions():
    """A copy of the current global print options (see `set_printoptions`)."""
    return dict(_options)


def set_printoptions(
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    suppress=None,
    sign=None,
    floatmode=None,
):
    """Update the global print options used by array `repr`/`str` and `array2string`.

    Every option is sticky: an omitted argument keeps its previous value.
    """
    if precision is not None:
        _options["precision"] = precision
    if threshold is not None:
        _options["threshold"] = threshold
    if edgeitems is not None:
        _options["edgeitems"] = edgeitems
    if linewidth is not None:
        _options["linewidth"] = linewidth
    if suppress is not None:
        _options["suppress"] = bool(suppress)
    if sign is not None:
        _options["sign"] = sign
    if floatmode is not None:
        _options["floatmode"] = floatmode


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    """Apply `set_printoptions` for the `with` body and restore the prior options afterwards,
    e.g. ``with np.printoptions(precision=3): ...``."""
    previous = get_printoptions()
    try:
        set_printoptions(*args, **kwargs)
        yield get_printoptions()
    finally:
        _options.clear()
        _options.update(previous)


# ---------------------------------------------------------------------------------------------
# Float digit decomposition and rendering
#
# `_decompose` reads `str(value)`'s shortest round-trip text (CPython's own dtoa, dtype-aware
# for a NumPy scalar) into a sign, digits and decimal exponent. Every other function here either
# reassembles those digits with different padding, or asks `format(value, spec)` for a freshly
# and correctly rounded string when the caller wants fewer digits than the natural count.
# ---------------------------------------------------------------------------------------------


def _decompose(value):
    """`(negative, digits, exponent)` from `str(value)`'s shortest round-trip text: `value ==
    (-1 if negative else 1) * 0.<digits> * 10**(exponent + 1)`, `digits` with no trailing zeros.
    `value` must be finite. Zero returns `digits="0"`, `exponent=0`.
    """
    text = str(value)
    negative = text.startswith("-")
    if negative:
        text = text[1:]
    if value == 0:
        return negative, "0", 0
    mantissa, _, exp_text = text.partition("e")
    exponent = int(exp_text) if exp_text else None
    integer, _, fraction = mantissa.partition(".")
    if exponent is not None:
        return negative, (integer + fraction).rstrip("0") or "0", exponent
    if integer != "0":
        return negative, (integer + fraction).rstrip("0") or "0", len(integer) - 1
    stripped = fraction.lstrip("0")
    return negative, stripped.rstrip("0") or "0", -(len(fraction) - len(stripped)) - 1


def _split_positional(digits, exponent):
    """`digits`/`exponent` (see `_decompose`) as an unpadded `(integer, fraction)` pair: `"123"`
    with `exponent=4` (i.e. `1.23e4`) is `("123", "00")`.
    """
    if exponent >= 0:
        point = exponent + 1
        if len(digits) >= point:
            return digits[:point], digits[point:]
        return digits + "0" * (point - len(digits)), ""
    return "0", "0" * (-exponent - 1) + digits


def _positional_digits(value, digits, exponent, precision, min_digits, unique):
    """`(integer, fraction)` text (no sign) for `value` in positional notation. `unique=True`
    reuses `digits`/`exponent`'s shortest text, capped at `precision` fractional digits and/or
    extended to `min_digits` with zeros; `unique=False` always asks `format` for exactly
    `precision` fractional digits, correctly rounded.
    """
    if not unique:
        integer, _, fraction = format(value, f".{precision}f").lstrip("-").partition(".")
        return integer, fraction
    natural = max(len(digits) - exponent - 1, 0)
    target = natural if precision is None else min(natural, precision)
    if min_digits is not None:
        target = max(target, min_digits)
    if target < natural:
        integer, _, fraction = format(value, f".{target}f").lstrip("-").partition(".")
        return integer, fraction
    integer, fraction = _split_positional(digits, exponent)
    return integer, fraction.ljust(target, "0")


def _scientific_digits(value, digits, exponent, precision, min_digits, unique):
    """`(integer, fraction, exponent)` text (no sign) for `value` in scientific notation; see
    `_positional_digits`. `precision`/`min_digits` count digits after the leading digit.
    """
    if not unique:
        mantissa, exp_text = format(value, f".{precision}e").lstrip("-").split("e")
        integer, _, fraction = mantissa.partition(".")
        return integer, fraction, int(exp_text)
    natural = len(digits) - 1
    target = natural if precision is None else min(natural, precision)
    if min_digits is not None:
        target = max(target, min_digits)
    if target < natural:
        mantissa, exp_text = format(value, f".{target}e").lstrip("-").split("e")
        integer, _, fraction = mantissa.partition(".")
        return integer, fraction, int(exp_text)
    return digits[:1], digits[1:].ljust(target, "0"), exponent


def _trim_fraction(fraction, mode):
    """Trailing-zero cleanup of a fraction digit string. `None` means the point itself should be
    dropped (`mode="-"` with nothing left after trimming)."""
    if mode == "k":
        return fraction
    trimmed = fraction.rstrip("0")
    if mode == ".":
        return trimmed
    if mode == "0":
        return trimmed or "0"
    if mode == "-":
        return trimmed or None
    raise ValueError("trim must be 'k', '.', '0' or '-'")


def _assemble(negative, show_plus, integer, fraction, pad_left, pad_right):
    """Sign, integer part and left padding, then a decimal point/fraction and right padding.
    `fraction=None` means the point is omitted; that column still counts toward `pad_right`, as
    whitespace.
    """
    sign = "-" if negative else ("+" if show_plus else "")
    left = sign + integer
    left_width = max(pad_left if pad_left is not None else 0, len(left))
    content = "" if fraction is None else "." + fraction
    right_width = max((pad_right + 1) if pad_right is not None else 0, len(content))
    return left.rjust(left_width) + content.ljust(right_width)


def _exp_suffix(exponent):
    return f"e{'-' if exponent < 0 else '+'}{abs(exponent):02d}"


def _special_text(value, sign):
    """Display text for a non-finite value, or `None` for a finite one. `sign` is a print-option
    sign mode (`'-'`/`'+'`/`' '`); NaN never carries a sign."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        if value < 0:
            return "-inf"
        return {"+": "+inf", " ": " inf"}.get(sign, "inf")
    return None


def _format_positional(value, precision=None, unique=True, sign="-", trim="k", pad_left=None,
                        pad_right=None, min_digits=None):
    special = _special_text(value, sign)
    if special is not None:
        return special
    negative, digits, exponent = _decompose(value)
    integer, fraction = _positional_digits(value, digits, exponent, precision, min_digits, unique)
    fraction = _trim_fraction(fraction, trim)
    return _assemble(negative, sign == "+", integer, fraction, pad_left, pad_right)


def _format_scientific(value, precision=None, unique=True, sign="-", trim="k", pad_left=None,
                        min_digits=None):
    special = _special_text(value, sign)
    if special is not None:
        return special
    negative, digits, exponent = _decompose(value)
    integer, fraction, exponent = _scientific_digits(
        value, digits, exponent, precision, min_digits, unique
    )
    fraction = _trim_fraction(fraction, trim)
    mantissa = _assemble(negative, sign == "+", integer, fraction, pad_left, None)
    return mantissa + _exp_suffix(exponent)


def format_float_positional(x, precision=None, unique=True, trim="k", pad_left=None, pad_right=None):
    """Decimal text for the real scalar `x`, in positional (non-exponential) notation.

    ``unique=True`` (the default) gives the shortest digits that round-trip back to `x`, capped
    at `precision` digits after the point when given; ``unique=False`` always gives exactly
    `precision` digits, correctly rounded. `trim` controls trailing-zero cleanup: ``'k'`` keeps
    them, ``'.'`` drops them but keeps the point, ``'0'`` drops them but leaves one digit, ``'-'``
    drops the point too if nothing is left after it. `pad_left`/`pad_right` widen the integer and
    fraction parts with spaces, for column alignment.
    """
    if precision is not None and precision < 0:
        raise ValueError("precision must be >= 0")
    if pad_left is not None and pad_left < 0:
        raise ValueError("pad_left must be >= 0")
    if pad_right is not None and pad_right < 0:
        raise ValueError("pad_right must be >= 0")
    if not unique and precision is None:
        raise TypeError("precision is required when unique=False")
    return _format_positional(
        x, precision=precision, unique=unique, trim=trim, pad_left=pad_left, pad_right=pad_right
    )


def format_float_scientific(x, precision=None, unique=True, trim="k", pad_left=None):
    """Decimal text for the real scalar `x`, in scientific notation (e.g. ``1.5e+00``).

    See `format_float_positional` for `precision`/`unique`/`trim`; `precision` here counts digits
    after the single leading mantissa digit.
    """
    if precision is not None and precision < 0:
        raise ValueError("precision must be >= 0")
    if pad_left is not None and pad_left < 0:
        raise ValueError("pad_left must be >= 0")
    if not unique and precision is None:
        raise TypeError("precision is required when unique=False")
    return _format_scientific(x, precision=precision, unique=unique, trim=trim, pad_left=pad_left)


# ---------------------------------------------------------------------------------------------
# Print-option resolution
# ---------------------------------------------------------------------------------------------


def _resolve(precision=None, threshold=None, edgeitems=None, linewidth=None, suppress=None,
             sign=None, floatmode=None):
    resolved = dict(_options)
    if precision is not None:
        resolved["precision"] = precision
    if threshold is not None:
        resolved["threshold"] = threshold
    if edgeitems is not None:
        resolved["edgeitems"] = edgeitems
    if linewidth is not None:
        resolved["linewidth"] = linewidth
    if suppress is not None:
        resolved["suppress"] = suppress
    if sign is not None:
        resolved["sign"] = sign
    if floatmode is not None:
        resolved["floatmode"] = floatmode
    return resolved


# ---------------------------------------------------------------------------------------------
# Summarization and traversal
# ---------------------------------------------------------------------------------------------


def _index_plan(length, edgeitems, summarize):
    """Indices to visit along one axis, with `_ELLIPSIS` marking an elided run."""
    if summarize and length > 2 * edgeitems:
        return list(range(edgeitems)) + [_ELLIPSIS] + list(range(length - edgeitems, length))
    return list(range(length))


def _visible(a, edgeitems, summarize):
    """The elements that will actually be printed, in order, skipping elided runs.

    Indexes `a` directly rather than going through `.tolist()`/`.item()`: a bool/int/float/
    complex element indexed this way stays a NumPy scalar, which is what lets `_decompose` (via
    `str`) find the *value's own* shortest round-trip digits (e.g. 7 for this particular float32)
    instead of `float64`'s (`.item()` always unboxes to a plain, double-precision-for-floats
    Python value). String and object elements unbox to a plain value directly on indexing, with
    no further array API to recurse through, so they are read at the last axis without indexing
    one level deeper.
    """
    if a.ndim == 0:
        return [a[()]]
    n = a.shape[0]
    indices = [i for i in _index_plan(n, edgeitems, summarize) if i is not _ELLIPSIS]
    if a.ndim == 1:
        return [a[i] for i in indices]
    values = []
    for i in indices:
        values.extend(_visible(a[i], edgeitems, summarize))
    return values


# ---------------------------------------------------------------------------------------------
# Per-dtype element formatters
#
# Each builder takes the visible values and resolved options and returns a callable
# `value -> str` giving the final, column-aligned text for one element.
# ---------------------------------------------------------------------------------------------


def _bool_formatter(values, _options, ndim=1):
    # NumPy pads every bool *column* to fit "False" (5 characters) whether or not a False is
    # actually present -- an all-True array still right-aligns to width 5. A 0-d array has no
    # column to align against, so it is the one exception: it shows its bare natural width.
    width = 5 if ndim != 0 else max((len("True" if v else "False") for v in values), default=4)

    def fmt(value):
        return ("True" if value else "False").rjust(width)

    return fmt


def _sign_prefix(text, sign_mode):
    """`text` from an unsigned integer render; add NumPy's `+`/` ` sign for a non-negative
    value."""
    if text.startswith("-") or sign_mode == "-":
        return text
    if sign_mode == "+":
        return "+" + text
    if sign_mode == " ":
        return " " + text
    return text


def _int_formatter(values, options):
    sign_mode = options["sign"]
    width = 0
    for value in values:
        width = max(width, len(_sign_prefix(str(value), sign_mode)))

    def fmt(value):
        return _sign_prefix(str(value), sign_mode).rjust(width)

    return fmt


def _float_digit_plan(floatmode, precision):
    """`(precision, unique, force_equal_digits)` for the float digit engine."""
    if floatmode == "fixed":
        return (8 if precision is None else precision), False, False
    if floatmode == "unique":
        return None, True, False
    if floatmode == "maxprec_equal":
        return precision, True, True
    return precision, True, False


def _float_exponential(values, suppress):
    """Whether the whole column should switch to scientific notation, NumPy's rule."""
    magnitudes = [abs(v) for v in values if math.isfinite(v) and v != 0.0]
    if not magnitudes:
        return False
    largest = max(magnitudes)
    if largest >= 1.0e8:
        return True
    if suppress:
        return False
    smallest = min(magnitudes)
    return smallest < 1.0e-4 or largest / smallest > 1000.0


def _float_column(values, options, pad_fraction=True):
    """A formatter for one float column (plain, or one part of a complex column).

    `pad_fraction=False` skips right-padding the fraction, for a complex value's imaginary part:
    its trailing digits sit right against the `j` suffix, and the whole `<real><imag>j` text is
    padded as a unit by the caller instead.
    """
    precision, unique, force_equal = _float_digit_plan(options["floatmode"], options["precision"])
    exponential = _float_exponential(values, options["suppress"])
    sign_mode = options["sign"]

    finite = [v for v in values if math.isfinite(v)]
    neg_width = pos_width = frac_width = 0
    naturals = []
    for value in finite:
        text = (
            _format_scientific(value, precision=precision, unique=unique, trim=".")
            if exponential
            else _format_positional(value, precision=precision, unique=unique, trim=".")
        )
        left, _, right = text.partition("e")[0].partition(".")
        frac_width = max(frac_width, len(right))
        naturals.append(len(right))
        if left.startswith("-"):
            neg_width = max(neg_width, len(left))
        else:
            pos_width = max(pos_width, len(left))

    # Scientific notation has no space-padding mechanism for a mantissa's fraction digits (unlike
    # positional's `pad_right`), so every floatmode aligns a column's mantissas by forcing real
    # digits up to the widest natural mantissa. Positional columns only do this for
    # `floatmode="maxprec_equal"`; `"maxprec"`/`"unique"` there pad with trailing spaces instead.
    force_min_digits = force_equal or exponential
    min_digits = max(naturals) if force_min_digits and naturals else None
    # `trim="."` unconditionally strips trailing zeros, even ones `min_digits` just added, so it
    # only belongs where nothing forces digits (positional's natural/space-padded case). Every
    # other case -- `unique=False` ("fixed" floatmode's exact `precision` digits) or a forced
    # `min_digits` -- keeps its trailing zeros with `trim="k"` instead.
    trim_mode = "k" if (not unique or force_min_digits) else "."

    reserve = sign_mode in ("+", " ")
    int_width = max(neg_width, pos_width + 1) if reserve else max(neg_width, pos_width, 1)
    natural_total = int_width + 1 + (frac_width if pad_fraction else 0)
    special_texts = [text for value in values if (text := _special_text(value, sign_mode))]
    overall = max([natural_total] + [len(text) for text in special_texts])

    def fmt(value):
        special = _special_text(value, sign_mode)
        if special is not None:
            return special.rjust(overall)
        if exponential:
            text = _format_scientific(
                value,
                precision=precision,
                unique=unique,
                sign=sign_mode,
                trim=trim_mode,
                pad_left=int_width,
                min_digits=min_digits,
            )
        else:
            text = _format_positional(
                value,
                precision=precision,
                unique=unique,
                sign=sign_mode,
                trim=trim_mode,
                pad_left=int_width,
                pad_right=frac_width if pad_fraction else None,
                min_digits=min_digits,
            )
        return text.rjust(overall)

    return fmt


def _float_formatter(values, options):
    return _float_column(values, options)


def _complex_formatter(values, options):
    reals = [v.real for v in values]
    imags = [v.imag for v in values]
    real_fmt = _float_column(reals, options)
    # The imaginary part always shows a sign for non-negative values too.
    imag_options = dict(options, sign="+" if options["sign"] != " " else " ")
    imag_fmt = _float_column(imags, imag_options, pad_fraction=False)

    overall = max(
        (len(real_fmt(v.real)) + len(imag_fmt(v.imag)) + 1 for v in values), default=0
    )

    def fmt(value):
        text = real_fmt(value.real) + imag_fmt(value.imag) + "j"
        return text.ljust(overall)

    return fmt


def _string_formatter(_values, _options):
    return repr


def _object_formatter(_values, _options):
    return repr


_BUILDERS = {
    "b": _bool_formatter,
    "u": _int_formatter,
    "i": _int_formatter,
    "f": _float_formatter,
    "c": _complex_formatter,
    "U": _string_formatter,
    "O": _object_formatter,
}


def _make_formatter(a, values, options):
    kind = a.dtype.kind
    if kind == "b":
        return _bool_formatter(values, options, a.ndim)
    return _BUILDERS[kind](values, options)


# ---------------------------------------------------------------------------------------------
# Line-wrapped, bracket-nested rendering
# ---------------------------------------------------------------------------------------------


def _wrap_row(words, separator, indent, linewidth):
    """`"[" + words joined and wrapped at `linewidth` + "]"`, continuation lines at `indent`.

    A separator like ``", "`` splits across a wrap: its non-space part (the comma) stays at the
    end of the closed line, and the continuation line starts directly with the next word.

    `indent` doubles as the column the very first line's own text starts at: the caller (e.g.
    `array_repr`) prepends a prefix such as ``"array("`` to line one itself, outside this
    function, so the length tracked here starts at `indent` rather than at the literal length of
    `line`, which never contains that prefix text.
    """
    if not words:
        return "[]"
    pad = " " * indent
    marker = separator.rstrip(" ")
    line = "[" + words[0]
    length = indent + len(words[0])
    lines = []
    for word in words[1:]:
        piece = separator + word
        if length + len(piece) >= linewidth:
            lines.append(line + marker)
            line = pad + word
            length = indent + len(word)
        else:
            line += piece
            length += len(piece)
    lines.append(line + "]")
    return "\n".join(lines)


def _render(a, edgeitems, summarize, formatter, separator, indent, linewidth):
    """Renders `a`; see `_visible` for why this indexes the array rather than using
    `.tolist()`."""
    if a.ndim == 0:
        return formatter(a[()])
    plan = _index_plan(a.shape[0], edgeitems, summarize)
    if a.ndim == 1:
        words = ["..." if i is _ELLIPSIS else formatter(a[i]) for i in plan]
        return _wrap_row(words, separator, indent + 1, linewidth)
    blocks = [
        "..."
        if i is _ELLIPSIS
        else _render(a[i], edgeitems, summarize, formatter, separator, indent + 1, linewidth)
        for i in plan
    ]
    blank = "\n\n" if a.ndim - 1 >= 2 else "\n"
    joiner = separator.rstrip(" ") + blank + " " * (indent + 1)
    return "[" + joiner.join(blocks) + "]"


def array2string(
    a,
    max_line_width=None,
    precision=None,
    suppress_small=None,
    separator=" ",
    prefix="",
    threshold=None,
    edgeitems=None,
    sign=None,
    floatmode=None,
):
    """Text for `a`'s data (no ``array(...)`` wrapper, `dtype=` or `shape=` suffix); see
    `array_repr`/`array_str` for those."""
    options = _resolve(
        precision=precision,
        threshold=threshold,
        edgeitems=edgeitems,
        linewidth=max_line_width,
        suppress=suppress_small,
        sign=sign,
        floatmode=floatmode,
    )
    if a.size == 0:
        return "[]"
    summarize = a.size > options["threshold"]
    values = _visible(a, options["edgeitems"], summarize)
    formatter_fn = _make_formatter(a, values, options)
    return _render(a, options["edgeitems"], summarize, formatter_fn, separator, len(prefix),
                    options["linewidth"])


_DEFAULT_DTYPE_NAMES = ("bool", "int64", "float64", "complex128")


def _looks_like_a_name(text):
    """Whether `text` is plain enough (e.g. ``int8``) to show bare in ``dtype=...``, as opposed
    to a type-string descriptor (e.g. ``<U2``) that needs quoting."""
    if not text or text[0].isdigit():
        return False
    return all(character.isalnum() or character == "_" for character in text)


def _dtype_suffix(a):
    empty = a.size == 0
    text = str(a.dtype)
    if not empty and text in _DEFAULT_DTYPE_NAMES:
        return None
    name = text if _looks_like_a_name(text) else repr(text)
    return f"dtype={name}"


def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    """`repr(arr)`: ``array(...)`` around `array2string`'s text, plus `shape=`/`dtype=` when
    they are not obvious from the data alone."""
    options = _resolve(precision=precision, linewidth=max_line_width, suppress=suppress_small)
    prefix = "array("
    if arr.size == 0:
        body = "[]"
        summarize = False
    else:
        summarize = arr.size > options["threshold"]
        values = _visible(arr, options["edgeitems"], summarize)
        formatter_fn = _make_formatter(arr, values, options)
        body = _render(arr, options["edgeitems"], summarize, formatter_fn, ", ", len(prefix),
                        options["linewidth"])
    extras = []
    if (arr.size == 0 and arr.ndim != 1) or summarize:
        extras.append(f"shape={arr.shape}")
    dtype_suffix = _dtype_suffix(arr)
    if dtype_suffix is not None:
        extras.append(dtype_suffix)
    if not extras:
        return f"{prefix}{body})"
    return f"{prefix}{body}, {', '.join(extras)})"


def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    """`str(a)`: just the data, without NumPy's ``array(...)`` wrapper or dtype/shape suffix.

    A 0-d array has no siblings to align a column against, so unlike every other case here, its
    `str()` is exactly its single value's own `str()`: the value's natural, dtype-correct text
    (e.g. a float32's own shortest digits), completely unaffected by `precision`/`floatmode`/etc.
    `repr()` (`array_repr`, below) does not carry this special case and always honors those
    options, so `str(np.array(2/3))` and `repr(np.array(2/3))` can legitimately disagree under
    `np.printoptions(precision=...)`.
    """
    if a.ndim == 0:
        return str(a[()])
    if a.size == 0:
        return "[]"
    options = _resolve(precision=precision, linewidth=max_line_width, suppress=suppress_small)
    summarize = a.size > options["threshold"]
    values = _visible(a, options["edgeitems"], summarize)
    formatter_fn = _make_formatter(a, values, options)
    return _render(a, options["edgeitems"], summarize, formatter_fn, " ", 0,
                    options["linewidth"])


def _array_repr_implementation(arr):
    return array_repr(arr)


def _array_str_implementation(arr):
    return array_str(arr)
