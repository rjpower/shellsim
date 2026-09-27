"""Array printing: ``repr``/``str`` of arrays, ``array2string`` and the print options.

This is a port of ``numpy/_core/arrayprint.py`` and ``printoptions.py`` from NumPy 2.5, so the
layout rules (padding, float modes, line wrapping, summarization and the ``shape=``/``dtype=``
suffixes) are NumPy's own code. The Dragon4 digit generation behind ``format_float_positional``
and ``format_float_scientific`` is native, in ``_numpy_print``.

Differences from NumPy: the options live in a module-level dictionary rather than a
``ContextVar``, ``printoptions`` is a class-based context manager, and the datetime,
timedelta, structured and void formatters are absent because those dtypes are not modeled.
Dispatch on the scalar type uses ``dtype.kind``, which is equivalent for the modeled dtypes.
"""

import operator
import sys
import warnings

import numpy as np
from _numpy_print import dragon4_positional, dragon4_scientific

__all__ = [
    "array2string",
    "array_repr",
    "array_str",
    "format_float_positional",
    "format_float_scientific",
    "get_printoptions",
    "printoptions",
    "set_printoptions",
]

_default_format_options = {
    "edgeitems": 3,
    "threshold": 1000,
    "floatmode": "maxprec",
    "precision": 8,
    "suppress": False,
    "linewidth": 75,
    "nanstr": "nan",
    "infstr": "inf",
    "sign": "-",
    "formatter": None,
    # Stored as an int to simplify comparisons; converted from and to str/False at the API.
    "legacy": sys.maxsize,
    "override_repr": None,
}
_format_options = dict(_default_format_options)

_LEGACY_CODES = {"1.13": 113, "1.21": 121, "1.25": 125, "2.1": 201, "2.2": 202}
_LEGACY_NAMES = {113: "1.13", 121: "1.21", 125: "1.25", 201: "2.1", 202: "2.2", sys.maxsize: False}


def _is_number(value):
    return isinstance(value, (bool, int, float, complex, np.number))


def _make_options_dict(
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    suppress=None,
    nanstr=None,
    infstr=None,
    sign=None,
    formatter=None,
    floatmode=None,
    legacy=None,
    override_repr=None,
):
    """The non-None arguments as an options dictionary, with ``legacy`` converted and checked."""
    given = {
        "precision": precision,
        "threshold": threshold,
        "edgeitems": edgeitems,
        "linewidth": linewidth,
        "suppress": suppress,
        "nanstr": nanstr,
        "infstr": infstr,
        "sign": sign,
        "formatter": formatter,
        "floatmode": floatmode,
        "legacy": legacy,
        "override_repr": override_repr,
    }
    options = {key: value for key, value in given.items() if value is not None}

    if suppress is not None:
        options["suppress"] = bool(suppress)

    modes = ["fixed", "unique", "maxprec", "maxprec_equal"]
    if floatmode not in [*modes, None]:
        raise ValueError("floatmode option must be one of " + ", ".join(f'"{m}"' for m in modes))

    if sign not in [None, "-", "+", " "]:
        raise ValueError("sign option must be one of ' ', '+', or '-'")

    if legacy is False:
        options["legacy"] = sys.maxsize
    elif legacy is None:
        pass
    elif legacy == False:  # noqa: E712 - NumPy deprecates falsy non-False values such as 0.
        warnings.warn(f"Passing `legacy={legacy!r}` is deprecated.", FutureWarning, stacklevel=3)
        options["legacy"] = sys.maxsize
    elif legacy in _LEGACY_CODES:
        options["legacy"] = _LEGACY_CODES[legacy]
    else:
        warnings.warn(
            "legacy printing option can currently only be '1.13', '1.21', "
            "'1.25', '2.1', '2.2' or `False`",
            stacklevel=3,
        )

    if threshold is not None:
        if not _is_number(threshold):
            raise TypeError("threshold must be numeric")
        if np.isnan(threshold):
            raise ValueError(
                "threshold must be non-NAN, try sys.maxsize for untruncated representation"
            )

    if precision is not None:
        try:
            options["precision"] = operator.index(precision)
        except TypeError as error:
            raise TypeError("precision must be an integer") from error

    return options


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
    override_repr=None,
):
    """Set how arrays print. ``formatter`` and ``override_repr`` reset when omitted."""
    _set_printoptions(
        precision,
        threshold,
        edgeitems,
        linewidth,
        suppress,
        nanstr,
        infstr,
        formatter,
        sign,
        floatmode,
        legacy=legacy,
        override_repr=override_repr,
    )


def _set_printoptions(
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
    override_repr=None,
):
    """Replace the options and return the previous ones, so a caller can restore them."""
    global _format_options
    new_options = _make_options_dict(
        precision,
        threshold,
        edgeitems,
        linewidth,
        suppress,
        nanstr,
        infstr,
        sign,
        formatter,
        floatmode,
        legacy,
    )
    new_options["formatter"] = formatter
    new_options["override_repr"] = override_repr

    updated = dict(_format_options)
    updated.update(new_options)
    if updated["legacy"] == 113:
        updated["sign"] = "-"

    previous = _format_options
    _format_options = updated
    return previous


def get_printoptions():
    """Return a copy of the current print options."""
    options = dict(_format_options)
    options["legacy"] = _LEGACY_NAMES[options["legacy"]]
    return options


def _get_legacy_print_mode():
    return _format_options["legacy"]


class printoptions:
    """Context manager that sets print options for a ``with`` block and restores them after.

    The ``as`` target is the options in effect inside the block.
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._previous = None

    def __enter__(self):
        self._previous = _set_printoptions(*self._args, **self._kwargs)
        return get_printoptions()

    def __exit__(self, *exc_info):
        global _format_options
        _format_options = self._previous
        return False


def _leading_trailing(a, edgeitems, index=()):
    """Keep only the N-D corners (leading and trailing edges) of an array."""
    axis = len(index)
    if axis == a.ndim:
        return a[index]

    if a.shape[axis] > 2 * edgeitems:
        return np.concatenate(
            (
                _leading_trailing(a, edgeitems, index + np.index_exp[:edgeitems]),
                _leading_trailing(a, edgeitems, index + np.index_exp[-edgeitems:]),
            ),
            axis=axis,
        )
    return _leading_trailing(a, edgeitems, index + np.index_exp[:])


def _object_format(o):
    """Object arrays containing lists print unambiguously."""
    if type(o) is list:
        return f"list({o!r})"
    return repr(o)


def repr_format(x):
    return repr(x)


def str_format(x):
    return str(x)


def _get_formatdict(data, *, precision, floatmode, suppress, sign, legacy, formatter, **kwargs):
    # Each entry is a thunk so only the formatter for the data's type is built.
    formatdict = {
        "bool": lambda: BoolFormat(data),
        "int": lambda: IntegerFormat(data, sign),
        "float": lambda: FloatingFormat(data, precision, floatmode, suppress, sign, legacy=legacy),
        "longfloat": lambda: FloatingFormat(
            data, precision, floatmode, suppress, sign, legacy=legacy
        ),
        "complexfloat": lambda: ComplexFloatingFormat(
            data, precision, floatmode, suppress, sign, legacy=legacy
        ),
        "longcomplexfloat": lambda: ComplexFloatingFormat(
            data, precision, floatmode, suppress, sign, legacy=legacy
        ),
        "object": lambda: _object_format,
        "void": lambda: str_format,
        "numpystr": lambda: repr_format,
    }

    def indirect(x):
        return lambda: x

    if formatter is not None:
        fkeys = [key for key in formatter.keys() if formatter[key] is not None]
        if "all" in fkeys:
            for key in formatdict.keys():
                formatdict[key] = indirect(formatter["all"])
        if "int_kind" in fkeys:
            formatdict["int"] = indirect(formatter["int_kind"])
        if "float_kind" in fkeys:
            for key in ["float", "longfloat"]:
                formatdict[key] = indirect(formatter["float_kind"])
        if "complex_kind" in fkeys:
            for key in ["complexfloat", "longcomplexfloat"]:
                formatdict[key] = indirect(formatter["complex_kind"])
        if "str_kind" in fkeys:
            formatdict["numpystr"] = indirect(formatter["str_kind"])
        for key in formatdict.keys():
            if key in fkeys:
                formatdict[key] = indirect(formatter[key])

    return formatdict


_KIND_FORMATS = {
    "b": "bool",
    "i": "int",
    "u": "int",
    "f": "float",
    "c": "complexfloat",
    "U": "numpystr",
    "S": "numpystr",
    "O": "object",
    "V": "void",
}


def _get_format_function(data, **options):
    """The formatting function for the dtype of ``data``."""
    formatdict = _get_formatdict(data, **options)
    return formatdict[_KIND_FORMATS.get(data.dtype.kind, "numpystr")]()


_repr_running = set()


def _recursive_guard(function):
    """Return ``'...'`` instead of recursing when ``function`` reaches the same object again,
    as happens for an object array that contains itself."""

    def wrapper(self, *args, **kwargs):
        key = id(self)
        if key in _repr_running:
            return "..."
        _repr_running.add(key)
        try:
            return function(self, *args, **kwargs)
        finally:
            _repr_running.discard(key)

    return wrapper


def _array2string_unguarded(a, options, separator=" ", prefix=""):
    data = np.asarray(a)
    if a.shape == ():
        a = data

    if a.size > options["threshold"]:
        summary_insert = "..."
        data = _leading_trailing(data, options["edgeitems"])
    else:
        summary_insert = ""

    format_function = _get_format_function(data, **options)

    # Skip over "[" and then over the prefix, such as "array(".
    next_line_prefix = " "
    next_line_prefix += " " * len(prefix)

    return _formatArray(
        a,
        format_function,
        options["linewidth"],
        next_line_prefix,
        separator,
        options["edgeitems"],
        summary_insert,
        options["legacy"],
    )


_array2string = _recursive_guard(_array2string_unguarded)


def array2string(
    a,
    max_line_width=None,
    precision=None,
    suppress_small=None,
    separator=" ",
    prefix="",
    *,
    formatter=None,
    threshold=None,
    edgeitems=None,
    sign=None,
    floatmode=None,
    suffix="",
    legacy=None,
):
    """Return a string representation of an array; see ``numpy.array2string``."""
    overrides = _make_options_dict(
        precision,
        threshold,
        edgeitems,
        max_line_width,
        suppress_small,
        None,
        None,
        sign,
        formatter,
        floatmode,
        legacy,
    )
    options = dict(_format_options)
    options.update(overrides)

    if options["legacy"] <= 113:
        if a.shape == () and a.dtype.names is None:
            return repr(a.item())

    if options["legacy"] > 113:
        options["linewidth"] -= len(suffix)

    # An array with any zero-length axis prints as empty.
    if a.size == 0:
        return "[]"

    return _array2string(a, options, separator, prefix)


def _extendLine(s, line, word, line_width, next_line_prefix, legacy):
    needs_wrap = len(line) + len(word) > line_width
    if legacy > 113:
        # Wrapping cannot help a word that is already at the start of a line.
        if len(line) <= len(next_line_prefix):
            needs_wrap = False

    if needs_wrap:
        s += line.rstrip() + "\n"
        line = next_line_prefix
    line += word
    return s, line


def _extendLine_pretty(s, line, word, line_width, next_line_prefix, legacy):
    """Extend ``line`` with a possibly multi-line ``word``."""
    words = word.splitlines()
    if len(words) == 1 or legacy <= 113:
        return _extendLine(s, line, word, line_width, next_line_prefix, legacy)

    max_word_length = max(len(word) for word in words)
    if len(line) + max_word_length > line_width and len(line) > len(next_line_prefix):
        s += line.rstrip() + "\n"
        line = next_line_prefix + words[0]
        indent = next_line_prefix
    else:
        indent = len(line) * " "
        line += words[0]

    for word in words[1::]:
        s += line.rstrip() + "\n"
        line = indent + word

    suffix_length = max_word_length - len(words[-1])
    line += suffix_length * " "

    return s, line


def _formatArray(
    a, format_function, line_width, next_line_prefix, separator, edge_items, summary_insert, legacy
):
    """Format every element, or only the edges with ``summary_insert`` between them."""

    def recurser(index, hanging_indent, curr_width):
        axis = len(index)
        axes_left = a.ndim - axis

        if axes_left == 0:
            return format_function(a[index])

        # When recursing, add a space to align with the "[" added, and reduce the line by one.
        next_hanging_indent = hanging_indent + " "
        if legacy <= 113:
            next_width = curr_width
        else:
            next_width = curr_width - len("]")

        a_len = a.shape[axis]
        show_summary = summary_insert and 2 * edge_items < a_len
        if show_summary:
            leading_items = edge_items
            trailing_items = edge_items
        else:
            leading_items = 0
            trailing_items = a_len

        s = ""

        # Last axis: wrap elements that would not fit on one line.
        if axes_left == 1:
            # The length up to the beginning of the separator or bracket.
            if legacy <= 113:
                elem_width = curr_width - len(separator.rstrip())
            else:
                elem_width = curr_width - max(len(separator.rstrip()), len("]"))

            line = hanging_indent
            for i in range(leading_items):
                word = recurser(index + (i,), next_hanging_indent, next_width)
                s, line = _extendLine_pretty(s, line, word, elem_width, hanging_indent, legacy)
                line += separator

            if show_summary:
                s, line = _extendLine(s, line, summary_insert, elem_width, hanging_indent, legacy)
                if legacy <= 113:
                    line += ", "
                else:
                    line += separator

            for i in range(trailing_items, 1, -1):
                word = recurser(index + (-i,), next_hanging_indent, next_width)
                s, line = _extendLine_pretty(s, line, word, elem_width, hanging_indent, legacy)
                line += separator

            if legacy <= 113:
                # NumPy 1.13 did not count the separator's width.
                elem_width = curr_width
            word = recurser(index + (-1,), next_hanging_indent, next_width)
            s, line = _extendLine_pretty(s, line, word, elem_width, hanging_indent, legacy)

            s += line

        # Other axes: newlines between rows.
        else:
            s = ""
            line_sep = separator.rstrip() + "\n" * (axes_left - 1)

            for i in range(leading_items):
                nested = recurser(index + (i,), next_hanging_indent, next_width)
                s += hanging_indent + nested + line_sep

            if show_summary:
                if legacy <= 113:
                    s += hanging_indent + summary_insert + ", \n"
                else:
                    s += hanging_indent + summary_insert + line_sep

            for i in range(trailing_items, 1, -1):
                nested = recurser(index + (-i,), next_hanging_indent, next_width)
                s += hanging_indent + nested + line_sep

            nested = recurser(index + (-1,), next_hanging_indent, next_width)
            s += hanging_indent + nested

        # Remove the hanging indent and wrap in brackets.
        return "[" + s[len(hanging_indent) :] + "]"

    return recurser(index=(), hanging_indent=next_line_prefix, curr_width=line_width)


def _none_or_positive_arg(x, name):
    if x is None:
        return -1
    if x < 0:
        raise ValueError(f"{name} must be >= 0")
    return x


class FloatingFormat:
    """Formatter for floating-point arrays."""

    def __init__(self, data, precision, floatmode, suppress_small, sign=False, *, legacy=None):
        # For backward compatibility, accept bools.
        if isinstance(sign, bool):
            sign = "+" if sign else "-"

        self._legacy = legacy
        if self._legacy <= 113:
            # When not 0-d, legacy mode does not support '-'.
            if data.shape != () and sign == "-":
                sign = " "

        self.floatmode = floatmode
        if floatmode == "unique":
            self.precision = None
        else:
            self.precision = precision

        self.precision = _none_or_positive_arg(self.precision, "precision")

        self.suppress_small = suppress_small
        self.sign = sign
        self.exp_format = False
        self.large_exponent = False
        self.fillFormat(data)

    def fillFormat(self, data):
        # Only the finite values decide the number of digits.
        finite_vals = data[np.isfinite(data)]

        # Choose exponential mode from the nonzero finite values.
        abs_non_zero = np.absolute(finite_vals[finite_vals != 0])
        if len(abs_non_zero) != 0:
            max_val = np.max(abs_non_zero)
            min_val = np.min(abs_non_zero)
            if self._legacy <= 202:
                exp_cutoff_max = 1.0e8
            else:
                # The cutoff depends on the data type's precision.
                exp_cutoff_max = 10.0 ** min(8, np.finfo(data.dtype).precision)
            with np.errstate(over="ignore"):
                if max_val >= exp_cutoff_max or (
                    not self.suppress_small and (min_val < 0.0001 or max_val / min_val > 1000.0)
                ):
                    self.exp_format = True

        # A first pass over all the numbers determines the field sizes.
        if len(finite_vals) == 0:
            self.pad_left = 0
            self.pad_right = 0
            self.trim = "."
            self.exp_size = -1
            self.unique = True
            self.min_digits = None
        elif self.exp_format:
            trim, unique = ".", True
            if self.floatmode == "fixed" or self._legacy <= 113:
                trim, unique = "k", False
            strs = [
                dragon4_scientific(
                    x, precision=self.precision, unique=unique, trim=trim, sign=self.sign == "+"
                )
                for x in finite_vals
            ]
            frac_strs, _, exp_strs = zip(*(s.partition("e") for s in strs))
            int_part, frac_part = zip(*(s.split(".") for s in frac_strs))
            self.exp_size = max(len(s) for s in exp_strs) - 1

            self.trim = "k"
            self.precision = max(len(s) for s in frac_part)
            self.min_digits = self.precision
            self.unique = unique

            # NumPy 1.13 used two spaces, the sign, and full precision.
            if self._legacy <= 113:
                self.pad_left = 3
            else:
                self.pad_left = max(len(s) for s in int_part)
            # pad_right is only needed for the width of nan and inf.
            self.pad_right = self.exp_size + 2 + self.precision
        else:
            trim, unique = ".", True
            if self.floatmode == "fixed":
                trim, unique = "k", False
            strs = [
                dragon4_positional(
                    x,
                    precision=self.precision,
                    fractional=True,
                    unique=unique,
                    trim=trim,
                    sign=self.sign == "+",
                )
                for x in finite_vals
            ]
            int_part, frac_part = zip(*(s.split(".") for s in strs))
            if self._legacy <= 113:
                self.pad_left = 1 + max(len(s.lstrip("-+")) for s in int_part)
            else:
                self.pad_left = max(len(s) for s in int_part)
            self.pad_right = max(len(s) for s in frac_part)
            self.exp_size = -1
            self.unique = unique

            if self.floatmode in ["fixed", "maxprec_equal"]:
                self.precision = self.min_digits = self.pad_right
                self.trim = "k"
            else:
                self.trim = "."
                self.min_digits = 0

        if self._legacy > 113:
            # Account for sign=' ' by adding one to pad_left.
            if self.sign == " " and not any(np.signbit(finite_vals)):
                self.pad_left += 1

        # Non-finite values may need a wider left part.
        if data.size != finite_vals.size:
            neginf = self.sign != "-" or any(data[np.isinf(data)] < 0)
            offset = self.pad_right + 1  # +1 for the decimal point
            self.pad_left = max(
                self.pad_left,
                len(_format_options["nanstr"]) - offset,
                len(_format_options["infstr"]) + neginf - offset,
            )

    def __call__(self, x):
        if not np.isfinite(x):
            with np.errstate(invalid="ignore"):
                if np.isnan(x):
                    sign = "+" if self.sign == "+" else ""
                    ret = sign + _format_options["nanstr"]
                else:
                    sign = "-" if x < 0 else "+" if self.sign == "+" else ""
                    ret = sign + _format_options["infstr"]
                return " " * (self.pad_left + self.pad_right + 1 - len(ret)) + ret

        if self.exp_format:
            return dragon4_scientific(
                x,
                precision=self.precision,
                min_digits=self.min_digits,
                unique=self.unique,
                trim=self.trim,
                sign=self.sign == "+",
                pad_left=self.pad_left,
                exp_digits=self.exp_size,
            )
        return dragon4_positional(
            x,
            precision=self.precision,
            min_digits=self.min_digits,
            unique=self.unique,
            fractional=True,
            trim=self.trim,
            sign=self.sign == "+",
            pad_left=self.pad_left,
            pad_right=self.pad_right,
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
    """Format a floating-point scalar in scientific notation with Dragon4."""
    precision = _none_or_positive_arg(precision, "precision")
    pad_left = _none_or_positive_arg(pad_left, "pad_left")
    exp_digits = _none_or_positive_arg(exp_digits, "exp_digits")
    min_digits = _none_or_positive_arg(min_digits, "min_digits")
    if min_digits > 0 and precision > 0 and min_digits > precision:
        raise ValueError("min_digits must be less than or equal to precision")
    return dragon4_scientific(
        x,
        precision=precision,
        unique=unique,
        trim=trim,
        sign=sign,
        pad_left=pad_left,
        exp_digits=exp_digits,
        min_digits=min_digits,
    )


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
    """Format a floating-point scalar in positional notation with Dragon4."""
    precision = _none_or_positive_arg(precision, "precision")
    pad_left = _none_or_positive_arg(pad_left, "pad_left")
    pad_right = _none_or_positive_arg(pad_right, "pad_right")
    min_digits = _none_or_positive_arg(min_digits, "min_digits")
    if not fractional and precision == 0:
        raise ValueError("precision must be greater than 0 if fractional=False")
    if min_digits > 0 and precision > 0 and min_digits > precision:
        raise ValueError("min_digits must be less than or equal to precision")
    return dragon4_positional(
        x,
        precision=precision,
        unique=unique,
        fractional=fractional,
        trim=trim,
        sign=sign,
        pad_left=pad_left,
        pad_right=pad_right,
        min_digits=min_digits,
    )


class IntegerFormat:
    def __init__(self, data, sign="-"):
        if data.size > 0:
            data_max = np.max(data)
            data_min = np.min(data)
            data_max_str_len = len(str(data_max))
            if sign == " " and data_min < 0:
                sign = "-"
            if data_max >= 0 and sign in "+ ":
                data_max_str_len += 1
            max_str_len = max(data_max_str_len, len(str(data_min)))
        else:
            max_str_len = 0
        self.format = f"{{:{sign}{max_str_len}d}}"

    def __call__(self, x):
        return self.format.format(x)


class BoolFormat:
    def __init__(self, data, **kwargs):
        # " True" lines up with "False" except in 0-d arrays.
        self.truestr = " True" if data.shape != () else "True"

    def __call__(self, x):
        return self.truestr if x else "False"


class ComplexFloatingFormat:
    """Formatter for complex arrays: a real field and a signed imaginary field."""

    def __init__(self, x, precision, floatmode, suppress_small, sign=False, *, legacy=None):
        # For backward compatibility, accept bools.
        if isinstance(sign, bool):
            sign = "+" if sign else "-"

        floatmode_real = floatmode_imag = floatmode
        if legacy <= 113:
            floatmode_real = "maxprec_equal"
            floatmode_imag = "maxprec"

        self.real_format = FloatingFormat(
            x.real, precision, floatmode_real, suppress_small, sign=sign, legacy=legacy
        )
        self.imag_format = FloatingFormat(
            x.imag, precision, floatmode_imag, suppress_small, sign="+", legacy=legacy
        )

    def __call__(self, x):
        r = self.real_format(x.real)
        i = self.imag_format(x.imag)

        # Put the 'j' before the imaginary part's trailing padding.
        sp = len(i.rstrip())
        i = i[:sp] + "j" + i[sp:]

        return r + i


_TYPELESS_DTYPES = ("int64", "float64", "complex128", "bool")


def dtype_is_implied(dtype):
    """Whether the repr of an array's values implies its dtype, so ``dtype=`` is omitted."""
    dtype = np.dtype(dtype)
    if _format_options["legacy"] <= 113 and dtype.kind == "b":
        return False
    if dtype.names is not None:
        return False
    if not dtype.isnative:
        return False
    return dtype.name in _TYPELESS_DTYPES


def dtype_short_repr(dtype):
    """A short form of ``dtype`` that evaluates to the same dtype."""
    if dtype.names is not None:
        return str(dtype)
    if dtype.kind in "SUV":
        # Flexible dtypes print their string form, such as '<U2'.
        return f"'{str(dtype)}'"

    typename = dtype.name
    if not dtype.isnative:
        return f"'{str(dtype)}'"
    # Quote type names that are not valid Python identifiers.
    if typename and not (typename[0].isalpha() and typename.isalnum()):
        typename = repr(typename)
    return typename


def _array_repr_implementation(arr, max_line_width=None, precision=None, suppress_small=None):
    current_options = _format_options
    override_repr = current_options["override_repr"]
    if override_repr is not None:
        return override_repr(arr)

    if max_line_width is None:
        max_line_width = current_options["linewidth"]

    if type(arr) is not np.ndarray:
        class_name = type(arr).__name__
    else:
        class_name = "array"

    prefix = class_name + "("
    if current_options["legacy"] <= 113 and arr.shape == () and not arr.dtype.names:
        lst = repr(arr.item())
    else:
        lst = array2string(arr, max_line_width, precision, suppress_small, ", ", prefix, suffix=")")

    # Add the shape and dtype when the text does not imply them.
    extras = []
    if (arr.size == 0 and arr.shape != (0,)) or (
        current_options["legacy"] > 210 and arr.size > current_options["threshold"]
    ):
        extras.append(f"shape={arr.shape}")
    if not dtype_is_implied(arr.dtype) or arr.size == 0:
        extras.append(f"dtype={dtype_short_repr(arr.dtype)}")

    if not extras:
        return prefix + lst + ")"

    arr_str = prefix + lst + ","
    extra_str = ", ".join(extras) + ")"
    # Put the extras on a new line if they would extend the last line past the width.
    last_line_len = len(arr_str) - (arr_str.rfind("\n") + 1)
    spacer = " "
    if current_options["legacy"] <= 113:
        if arr.dtype.kind in "SUV":
            spacer = "\n" + " " * len(prefix)
    elif last_line_len + len(extra_str) + 1 > max_line_width:
        spacer = "\n" + " " * len(prefix)

    return arr_str + spacer + extra_str


def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    """Return the string representation of an array."""
    return _array_repr_implementation(arr, max_line_width, precision, suppress_small)


def _repr_or_str(v):
    if isinstance(v, bytes):
        return repr(v)
    return str(v)


_guarded_repr_or_str = _recursive_guard(_repr_or_str)


def _array_str_implementation(a, max_line_width=None, precision=None, suppress_small=None):
    if _format_options["legacy"] <= 113 and a.shape == () and not a.dtype.names:
        return str(a.item())

    # A 0-d array prints like its scalar, so floats keep full precision and strings lose their
    # quotes.
    if a.shape == ():
        return _guarded_repr_or_str(a[()])

    return array2string(a, max_line_width, precision, suppress_small, " ", "")


def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    """Return a string representation of the data in an array."""
    return _array_str_implementation(a, max_line_width, precision, suppress_small)
