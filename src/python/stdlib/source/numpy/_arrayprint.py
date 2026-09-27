"""NumPy-compatible text for scalars and arrays: print options, ``format_float_*``, and the
``array2string``/``array_repr``/``array_str`` family that ``ndarray.__repr__``/``__str__`` call.

Digit generation for a single float goes through the native ``_numpy_print`` module,
imported below as ``format_positional``/``format_scientific``. This module supplies everything
around that: argument validation for ``format_float_positional``/``format_float_scientific``, the
global print-options state, and the array layout algorithm (per-element formatting, column
alignment, line wrapping, summarization, and the ``dtype=``/``shape=`` suffixes ``repr`` adds).

Array layout works in two passes per axis of formattable (bool/int/float/complex) elements: a
discovery pass formats every element that will actually be shown (see ``_visible``) without
padding, to learn column widths and, for floats, whether the whole array should switch to
scientific notation; a second pass reformats each element with the resolved padding so a column
lines up. String and object elements are not column-aligned, matching NumPy.
"""

import contextlib
import math

from _numpy_print import format_positional, format_scientific

_DEFAULTS = {
    "precision": 8,
    "threshold": 1000,
    "edgeitems": 3,
    "linewidth": 75,
    "suppress": False,
    "nanstr": "nan",
    "infstr": "inf",
    "sign": "-",
    "formatter": None,
    "floatmode": "maxprec",
    "legacy": False,
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
    nanstr=None,
    infstr=None,
    formatter=None,
    sign=None,
    floatmode=None,
    *,
    legacy=None,
):
    """Update the global print options used by array `repr`/`str` and `array2string`.

    Every option is sticky (an omitted argument keeps its previous value) except `formatter`,
    which NumPy always replaces outright, so a call that omits it clears any custom formatter.
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
    if nanstr is not None:
        _options["nanstr"] = nanstr
    if infstr is not None:
        _options["infstr"] = infstr
    if sign is not None:
        _options["sign"] = sign
    if floatmode is not None:
        _options["floatmode"] = floatmode
    if legacy is not None:
        _options["legacy"] = legacy
    _options["formatter"] = formatter


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
# format_float_positional / format_float_scientific
#
# These validate arguments the way NumPy's Python wrappers do (a negative value supplied
# explicitly is an error, not "unset") and then call the native formatter, which uses -1 to
# mean "unset". The checks and their order and messages match NumPy 2.5.3 (observed, since the
# reference implementation is off limits here).
# ---------------------------------------------------------------------------------------------


def _unset(value):
    return -1 if value is None else value


def format_float_positional(
    x,
    precision=None,
    unique=True,
    fractional=True,
    trim="k",
    sign=False,
    pad_left=None,
    pad_right=None,
    min_digits=None,
):
    """Format a real scalar as decimal text in positional (non-exponential) notation.

    See `format_float_scientific` for the shared options. `fractional` selects what `precision`
    and `min_digits` count: digits after the decimal point (including leading zeros) when
    `True`, or total significant digits when `False`.
    """
    if precision is not None and precision < 0:
        raise ValueError("precision must be >= 0")
    if pad_left is not None and pad_left < 0:
        raise ValueError("pad_left must be >= 0")
    if pad_right is not None and pad_right < 0:
        raise ValueError("pad_right must be >= 0")
    if min_digits is not None and min_digits < 0:
        raise ValueError("min_digits must be >= 0")
    if precision and min_digits is not None and min_digits > precision:
        raise ValueError("min_digits must be less than or equal to precision")
    if not fractional and precision == 0:
        raise ValueError("precision must be greater than 0 if fractional=False")
    return format_positional(
        x,
        precision=_unset(precision),
        unique=unique,
        fractional=fractional,
        sign=sign,
        trim=trim,
        pad_left=_unset(pad_left),
        pad_right=_unset(pad_right),
        min_digits=_unset(min_digits),
    )


def format_float_scientific(
    x,
    precision=None,
    unique=True,
    trim="k",
    sign=False,
    pad_left=None,
    exp_digits=None,
    min_digits=None,
):
    """Format a real scalar as decimal text in scientific notation, e.g. ``1.5e+00``.

    `precision`/`min_digits` count digits after the leading (always single) mantissa digit.
    `unique=True` (the default) finds the shortest digits that round-trip, optionally capped by
    `precision` and/or extended by `min_digits`; `unique=False` always uses exactly `precision`
    digits, correctly rounded. `trim` controls trailing-zero cleanup: ``'k'`` keeps them, ``'.'``
    drops them but keeps the point, ``'0'`` drops them but leaves one digit, ``'-'`` drops the
    point too if nothing is left after it.
    """
    if precision is not None and precision < 0:
        raise ValueError("precision must be >= 0")
    if pad_left is not None and pad_left < 0:
        raise ValueError("pad_left must be >= 0")
    if exp_digits is not None and exp_digits < 0:
        raise ValueError("exp_digits must be >= 0")
    if min_digits is not None and min_digits < 0:
        raise ValueError("min_digits must be >= 0")
    if precision and min_digits is not None and min_digits > precision:
        raise ValueError("min_digits must be less than or equal to precision")
    return format_scientific(
        x,
        precision=_unset(precision),
        unique=unique,
        sign=sign,
        trim=trim,
        pad_left=_unset(pad_left),
        exp_digits=_unset(exp_digits),
        min_digits=_unset(min_digits),
    )


# ---------------------------------------------------------------------------------------------
# Print-option resolution
# ---------------------------------------------------------------------------------------------


def _resolve(
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    suppress=None,
    sign=None,
    formatter=None,
    floatmode=None,
    nanstr=None,
    infstr=None,
):
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
    if formatter is not None:
        resolved["formatter"] = formatter
    if floatmode is not None:
        resolved["floatmode"] = floatmode
    if nanstr is not None:
        resolved["nanstr"] = nanstr
    if infstr is not None:
        resolved["infstr"] = infstr
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
    complex element indexed this way stays a NumPy scalar, which is what lets
    `format_float_positional`/`_scientific` (via `_float_column`) find the *value's own* shortest
    round-trip digits (e.g. 7 for this particular float32) instead of `float64`'s (`.item()`
    always unboxes to a plain, double-precision-for-floats Python value). String and object
    elements unbox to a plain value directly on indexing, with no further array API to recurse
    through, so they are read at the last axis without indexing one level deeper.
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
    """`text` from a `sign=False` native call; add NumPy's `+`/` ` sign for a non-negative value."""
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
    """`(precision, unique, force_equal_digits)` for `format_float_positional`/`_scientific`."""
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


def _special_text(value, sign_mode, nanstr, infstr):
    """Display text for a non-finite value, or `None` for a finite one."""
    if math.isnan(value):
        return nanstr
    if math.isinf(value):
        return ("-" if value < 0 else _sign_prefix("", sign_mode)) + infstr
    return None


def _float_column(values, options, pad_fraction=True):
    """A formatter for one float column (plain, or one part of a complex column).

    `pad_fraction=False` skips right-padding the fraction, for a complex value's imaginary part:
    its trailing digits sit right against the `j` suffix, and the whole `<real><imag>j` text is
    padded as a unit by the caller instead.
    """
    precision, unique, force_equal = _float_digit_plan(options["floatmode"], options["precision"])
    exponential = _float_exponential(values, options["suppress"])
    sign_mode = options["sign"]
    nanstr, infstr = options["nanstr"], options["infstr"]

    finite = [v for v in values if math.isfinite(v)]
    neg_width = pos_width = frac_width = exp_width = 0
    naturals = []
    for value in finite:
        if exponential:
            text = format_float_scientific(value, precision=precision, unique=unique, sign=False, trim=".")
            mantissa, _, exponent = text.partition("e")
            exp_width = max(exp_width, len(exponent) - 1)
        else:
            mantissa = format_float_positional(
                value, precision=precision, unique=unique, sign=False, fractional=True, trim="."
            )
        left, _, right = mantissa.partition(".")
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
    show_plus = sign_mode == "+"
    natural_total = int_width + 1 + (frac_width if pad_fraction else 0)
    special_texts = [
        text for value in values if (text := _special_text(value, sign_mode, nanstr, infstr))
    ]
    overall = max([natural_total] + [len(text) for text in special_texts])

    def fmt(value):
        special = _special_text(value, sign_mode, nanstr, infstr)
        if special is not None:
            return special.rjust(overall)
        if exponential:
            text = format_float_scientific(
                value,
                precision=precision,
                unique=unique,
                sign=show_plus,
                trim=trim_mode,
                pad_left=int_width,
                exp_digits=exp_width,
                min_digits=min_digits,
            )
        else:
            text = format_float_positional(
                value,
                precision=precision,
                unique=unique,
                sign=show_plus,
                fractional=True,
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


_FORMATTER_KEYS = {
    "b": (["bool"], "_bool"),
    "u": (["int", "int_kind"], "_int"),
    "i": (["int", "int_kind"], "_int"),
    "f": (["float", "float_kind"], "_float"),
    "c": (["complexfloat", "complex_kind"], "_complex"),
    "U": (["numpystr", "str_kind"], "_str"),
    "O": (["object"], "_object"),
}

_BUILDERS = {
    "b": _bool_formatter,
    "u": _int_formatter,
    "i": _int_formatter,
    "f": _float_formatter,
    "c": _complex_formatter,
    "U": _string_formatter,
    "O": _object_formatter,
}

_PADDED_KINDS = ("b", "u", "i", "f", "c")


def _custom_formatter(formatter_dict, kind):
    if not formatter_dict:
        return None
    keys, _ = _FORMATTER_KEYS[kind]
    for key in (*keys, "all"):
        if key in formatter_dict:
            return formatter_dict[key]
    return None


def _make_formatter(a, values, options):
    kind = a.dtype.kind
    custom = _custom_formatter(options["formatter"], kind)
    if custom is None:
        if kind == "b":
            return _bool_formatter(values, options, a.ndim)
        return _BUILDERS[kind](values, options)
    if kind not in _PADDED_KINDS:
        return custom
    width = max((len(custom(value)) for value in values), default=0)

    def fmt(value):
        return custom(value).rjust(width)

    return fmt


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
    style=None,
    formatter=None,
    threshold=None,
    edgeitems=None,
    sign=None,
    floatmode=None,
    suffix="",
    legacy=None,
):
    """Text for `a`'s data (no ``array(...)`` wrapper, `dtype=` or `shape=` suffix); see
    `array_repr`/`array_str` for those."""
    del style, suffix, legacy
    options = _resolve(
        precision=precision,
        threshold=threshold,
        edgeitems=edgeitems,
        linewidth=max_line_width,
        suppress=suppress_small,
        sign=sign,
        formatter=formatter,
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
