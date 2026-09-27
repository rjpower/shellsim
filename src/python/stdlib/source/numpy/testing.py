"""``numpy.testing``: array assertions used by test suites."""

import numpy as np


def _message(header, detail, err_msg):
    lines = [header]
    if err_msg:
        lines.append(err_msg)
    lines.append(detail)
    return "\n".join(lines)


def assert_array_equal(actual, desired, err_msg="", verbose=True, *, strict=False):
    """Raise ``AssertionError`` unless the arrays have one shape and equal elements (NaNs match)."""
    actual = np.asarray(actual)
    desired = np.asarray(desired)
    if strict and (actual.shape != desired.shape or actual.dtype != desired.dtype):
        raise AssertionError(_message("Arrays are not equal", f"(shapes {actual.shape}, {desired.shape} mismatch)", err_msg))
    left = actual.tolist() if actual.ndim else [actual.item()]
    right = desired.tolist() if desired.ndim else [desired.item()]
    if actual.shape != desired.shape and actual.ndim and desired.ndim:
        raise AssertionError(_message("Arrays are not equal", f"(shapes {actual.shape}, {desired.shape} mismatch)", err_msg))
    if not _equal(left, right):
        raise AssertionError(_message("Arrays are not equal", f" ACTUAL: {actual!r}\n DESIRED: {desired!r}", err_msg))


def _flat(values):
    if isinstance(values, list):
        items = []
        for value in values:
            items.extend(_flat(value))
        return items
    return [values]


def _equal(left, right):
    left, right = _flat(left), _flat(right)
    if len(left) != len(right):
        if len(left) == 1:
            left = left * len(right)
        elif len(right) == 1:
            right = right * len(left)
        else:
            return False
    for a, b in zip(left, right):
        if a != a and b != b:
            continue
        if a != b:
            return False
    return True


def assert_allclose(actual, desired, rtol=1e-07, atol=0, equal_nan=True, err_msg="", verbose=True, *, strict=False):
    """Raise ``AssertionError`` unless ``|actual - desired| <= atol + rtol * |desired|`` everywhere."""
    actual_array = np.asarray(actual)
    desired_array = np.asarray(desired)
    left = _flat(actual_array.tolist())
    right = _flat(desired_array.tolist())
    if len(left) != len(right):
        if len(right) == 1:
            right = right * len(left)
        elif len(left) == 1:
            left = left * len(right)
        else:
            raise AssertionError(_message(f"Not equal to tolerance rtol={rtol:g}, atol={atol:g}", f"(shapes {actual_array.shape}, {desired_array.shape} mismatch)", err_msg))
    for a, b in zip(left, right):
        if a != a or b != b:
            if equal_nan and a != a and b != b:
                continue
            raise AssertionError(_message(f"Not equal to tolerance rtol={rtol:g}, atol={atol:g}", f" ACTUAL: {actual_array!r}\n DESIRED: {desired_array!r}", err_msg))
        if a == b:
            continue
        if abs(a - b) > atol + rtol * abs(b):
            raise AssertionError(_message(f"Not equal to tolerance rtol={rtol:g}, atol={atol:g}", f" ACTUAL: {actual_array!r}\n DESIRED: {desired_array!r}", err_msg))


def _build_err_msg(actual, desired, err_msg, header="Items are not equal:"):
    """NumPy's ``build_err_msg`` for an actual and a desired value."""
    lines = ["\n" + header]
    err_msg = str(err_msg)
    if err_msg:
        if "\n" not in err_msg and len(err_msg) < 79 - len(header):
            lines = [lines[0] + " " + err_msg]
        else:
            lines.append(err_msg)
    for name, value in (("ACTUAL", actual), ("DESIRED", desired)):
        text = repr(value)
        if text.count("\n") > 3:
            text = "\n".join(text.splitlines()[:3]) + "..."
        lines.append(f" {name}: {text}")
    return "\n".join(lines)


def assert_equal(actual, desired, err_msg="", verbose=True, *, strict=False):
    """Raise ``AssertionError`` unless the values are equal, as NumPy's ``assert_equal``.

    Containers are compared item by item and arrays elementwise. For scalars, NaN equals NaN,
    zeros of different signs differ, and complex numbers compare their parts separately.
    """
    if isinstance(desired, dict):
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        assert_equal(len(actual), len(desired), err_msg)
        for key in desired:
            if key not in actual:
                raise AssertionError(repr(key))
            assert_equal(actual[key], desired[key], f"key={key!r}\n{err_msg}")
        return
    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        assert_equal(len(actual), len(desired), err_msg)
        for index in range(len(desired)):
            assert_equal(actual[index], desired[index], f"item={index!r}\n{err_msg}")
        return
    if isinstance(actual, np.ndarray) or isinstance(desired, np.ndarray):
        assert_array_equal(actual, desired, err_msg, strict=strict)
        return
    msg = _build_err_msg(actual, desired, err_msg)
    if np.iscomplexobj(actual) or np.iscomplexobj(desired):
        actual_parts = (np.real(actual), np.imag(actual)) if np.iscomplexobj(actual) else (actual, 0)
        desired_parts = (np.real(desired), np.imag(desired)) if np.iscomplexobj(desired) else (desired, 0)
        try:
            assert_equal(actual_parts[0], desired_parts[0])
            assert_equal(actual_parts[1], desired_parts[1])
        except AssertionError:
            raise AssertionError(msg) from None
        return
    if np.isscalar(desired) != np.isscalar(actual):
        raise AssertionError(msg)
    try:
        if np.isnan(desired) and np.isnan(actual):
            return
        if desired == 0 and actual == 0 and np.signbit(desired) != np.signbit(actual):
            raise AssertionError(msg)
    except (TypeError, ValueError, NotImplementedError):
        pass
    if not (desired == actual):
        raise AssertionError(msg)


def assert_array_almost_equal(actual, desired, decimal=6, err_msg="", verbose=True):
    """Raise ``AssertionError`` unless ``|actual - desired| < 1.5 * 10**-decimal`` everywhere."""
    assert_allclose(actual, desired, rtol=0, atol=1.5 * 10.0 ** (-decimal), err_msg=err_msg)


def assert_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
    """Raise ``AssertionError`` unless ``|actual - desired| < 1.5 * 10**-decimal``."""
    assert_allclose(actual, desired, rtol=0, atol=1.5 * 10.0 ** (-decimal), err_msg=err_msg)
