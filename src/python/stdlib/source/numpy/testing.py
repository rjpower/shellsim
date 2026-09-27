"""`numpy.testing`: array-aware assertion helpers with NumPy's comparison rules and messages.

Failure messages follow NumPy's `build_err_msg` layout (a header line, an optional inline or
multi-line `err_msg`, then ` ACTUAL: ...` / ` DESIRED: ...` lines) since the portable suite
checks this text exactly. `assert_equal` recurses through lists, tuples, and dicts, prepending
`item=<index>` or `key=<repr>` context to `err_msg` as it goes, so a mismatch deep inside a
nested structure still reports the exact path to it, matching NumPy's own behavior.
"""

import math

import numpy as np


def _isnan(value):
    value = complex(value)
    return math.isnan(value.real) or math.isnan(value.imag)


def _isinf(value):
    value = complex(value)
    return math.isinf(value.real) or math.isinf(value.imag)


def _build_err_msg(actual, desired, err_msg, header="Items are not equal:"):
    lines = ["\n" + header]
    if err_msg:
        if "\n" not in err_msg and len(err_msg) < 79 - len(header):
            lines = [lines[0] + " " + err_msg]
        else:
            lines.append(err_msg)
    lines.append(" ACTUAL: " + repr(actual))
    lines.append(" DESIRED: " + repr(desired))
    return "\n".join(lines)


def _is_negative(value):
    # `math.copysign` does not exist in shellsim; `repr` already tracks the sign bit
    # (`-0.0` prints with its minus sign), so it doubles as a sign test.
    return repr(float(value)).startswith("-")


def _float_equal(a, b):
    if a != a and b != b:
        return True
    if a != b:
        return False
    if a == 0 and b == 0:
        return _is_negative(a) == _is_negative(b)
    return True


def _scalars_equal(actual, desired):
    if isinstance(actual, complex) or isinstance(desired, complex):
        a, d = complex(actual), complex(desired)
        return _float_equal(a.real, d.real) and _float_equal(a.imag, d.imag)
    if isinstance(actual, (float, np.floating)) or isinstance(desired, (float, np.floating)):
        return _float_equal(float(actual), float(desired))
    return actual == desired


def assert_(condition, msg=""):
    if not condition:
        raise AssertionError(msg)


def assert_equal(actual, desired, err_msg="", verbose=True):
    """Recursive equality: arrays compare like `assert_array_equal`, dicts/lists/tuples
    recurse elementwise (reporting the path to a mismatch), and scalars distinguish `nan`
    (treated equal) and signed zero (treated unequal), matching NumPy's `assert_equal`."""
    if isinstance(desired, np.ndarray) or isinstance(actual, np.ndarray):
        assert_array_equal(actual, desired, err_msg=err_msg, verbose=verbose)
        return
    if isinstance(desired, dict) or isinstance(actual, dict):
        if not (isinstance(desired, dict) and isinstance(actual, dict)):
            raise AssertionError(_build_err_msg(actual, desired, err_msg))
        if sorted(actual.keys(), key=repr) != sorted(desired.keys(), key=repr):
            raise AssertionError(_build_err_msg(actual, desired, err_msg))
        for key in desired:
            assert_equal(actual[key], desired[key], err_msg=f"key={key!r}\n" + err_msg, verbose=verbose)
        return
    if isinstance(desired, (list, tuple)) or isinstance(actual, (list, tuple)):
        if not (isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple))):
            raise AssertionError(_build_err_msg(actual, desired, err_msg))
        if len(actual) != len(desired):
            raise AssertionError(_build_err_msg(actual, desired, err_msg))
        for index in range(len(desired)):
            assert_equal(
                actual[index], desired[index], err_msg=f"item={index}\n" + err_msg, verbose=verbose
            )
        return
    if _scalars_equal(actual, desired):
        return
    raise AssertionError(_build_err_msg(actual, desired, err_msg))


def _broadcast_or_fail(actual, desired, err_msg, header):
    a_arr = np.asarray(actual)
    d_arr = np.asarray(desired)
    try:
        return np.broadcast_arrays(a_arr, d_arr)
    except Exception:
        raise AssertionError(_build_err_msg(actual, desired, err_msg, header=header)) from None


def _array_values_equal(a, d):
    ca, cd = complex(a), complex(d)
    if _isnan(ca) or _isnan(cd):
        return _isnan(ca) and _isnan(cd)
    return ca == cd


def assert_array_equal(actual, desired, err_msg="", verbose=True, strict=False):
    header = "Arrays are not equal"
    a_b, d_b = _broadcast_or_fail(actual, desired, err_msg, header)
    a_flat = a_b.reshape(-1).tolist()
    d_flat = d_b.reshape(-1).tolist()
    for a, d in zip(a_flat, d_flat):
        if isinstance(a, str) or isinstance(d, str):
            equal = a == d
        else:
            equal = _array_values_equal(a, d)
        if not equal:
            raise AssertionError(_build_err_msg(actual, desired, err_msg, header=header))


def assert_array_less(actual, desired, err_msg="", verbose=True):
    header = "Arrays are not less-ordered"
    a_b, d_b = _broadcast_or_fail(actual, desired, err_msg, header)
    a_flat = a_b.reshape(-1).tolist()
    d_flat = d_b.reshape(-1).tolist()
    for a, d in zip(a_flat, d_flat):
        if not (a < d):
            raise AssertionError(_build_err_msg(actual, desired, err_msg, header=header))


def _isclose(a, d, rtol, atol, equal_nan):
    ca, cd = complex(a), complex(d)
    if _isnan(ca) or _isnan(cd):
        return equal_nan and _isnan(ca) and _isnan(cd)
    if _isinf(ca) or _isinf(cd):
        return ca == cd
    return abs(ca - cd) <= atol + rtol * abs(cd)


def assert_allclose(actual, desired, rtol=1e-7, atol=0, equal_nan=True, err_msg="", verbose=True):
    header = f"Not equal to tolerance rtol={rtol:g}, atol={atol:g}"
    a_b, d_b = _broadcast_or_fail(actual, desired, err_msg, header)
    a_flat = a_b.reshape(-1).tolist()
    d_flat = d_b.reshape(-1).tolist()
    for a, d in zip(a_flat, d_flat):
        if not _isclose(a, d, rtol, atol, equal_nan):
            raise AssertionError(_build_err_msg(actual, desired, err_msg, header=header))


def _almost_equal(actual, desired, decimal, err_msg, header):
    a_b, d_b = _broadcast_or_fail(actual, desired, err_msg, header)
    threshold = 1.5 * 10.0 ** (-decimal)
    a_flat = a_b.reshape(-1).tolist()
    d_flat = d_b.reshape(-1).tolist()
    for a, d in zip(a_flat, d_flat):
        ca, cd = complex(a), complex(d)
        if _isnan(ca) or _isnan(cd):
            if _isnan(ca) and _isnan(cd):
                continue
            raise AssertionError(_build_err_msg(actual, desired, err_msg, header=header))
        if _isinf(ca) or _isinf(cd):
            if ca == cd:
                continue
            raise AssertionError(_build_err_msg(actual, desired, err_msg, header=header))
        if abs(ca - cd) >= threshold:
            raise AssertionError(_build_err_msg(actual, desired, err_msg, header=header))


def assert_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
    _almost_equal(actual, desired, decimal, err_msg, f"Arrays are not almost equal to {decimal} decimals")


def assert_array_almost_equal(actual, desired, decimal=6, err_msg="", verbose=True):
    _almost_equal(actual, desired, decimal, err_msg, f"Arrays are not almost equal to {decimal} decimals")


class _AssertRaisesContext:
    def __init__(self, expected):
        self.expected = expected
        self.exception = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            raise AssertionError(f"{self.expected.__name__} not raised")
        if not issubclass(exc_type, self.expected):
            return False
        self.exception = exc_value
        return True


def assert_raises(exception_class, *args, **kwargs):
    if not args:
        return _AssertRaisesContext(exception_class)
    func = args[0]
    with _AssertRaisesContext(exception_class):
        func(*args[1:], **kwargs)
    return None


class _AssertWarnsContext:
    def __init__(self, expected):
        self.expected = expected
        self._catcher = None
        self._records = None

    def __enter__(self):
        import warnings

        self._catcher = warnings.catch_warnings(record=True)
        self._records = self._catcher.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._catcher.__exit__(exc_type, exc_value, traceback)
        if exc_type is not None:
            return False
        if not any(issubclass(record.category, self.expected) for record in self._records):
            raise AssertionError(f"{self.expected.__name__} not triggered")
        return False


def assert_warns(warning_class, *args, **kwargs):
    if not args:
        return _AssertWarnsContext(warning_class)
    func = args[0]
    with _AssertWarnsContext(warning_class):
        result = func(*args[1:], **kwargs)
    return result
