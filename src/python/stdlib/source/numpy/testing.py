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


def assert_equal(actual, desired, err_msg="", verbose=True):
    """Raise ``AssertionError`` unless the values are equal, comparing arrays elementwise."""
    if isinstance(actual, np.ndarray) or isinstance(desired, np.ndarray):
        assert_array_equal(actual, desired, err_msg)
        return
    if isinstance(actual, (list, tuple)) and isinstance(desired, (list, tuple)):
        if len(actual) != len(desired):
            raise AssertionError(_message("Items are not equal:", f"length {len(actual)} != {len(desired)}", err_msg))
        for a, b in zip(actual, desired):
            assert_equal(a, b, err_msg)
        return
    if isinstance(actual, dict) and isinstance(desired, dict):
        if actual.keys() != desired.keys():
            raise AssertionError(_message("Items are not equal:", f" ACTUAL: {actual!r}\n DESIRED: {desired!r}", err_msg))
        for key in actual:
            assert_equal(actual[key], desired[key], err_msg)
        return
    if actual != actual and desired != desired:
        return
    if not actual == desired:
        raise AssertionError(_message("Items are not equal:", f" ACTUAL: {actual!r}\n DESIRED: {desired!r}", err_msg))


def assert_array_almost_equal(actual, desired, decimal=6, err_msg="", verbose=True):
    """Raise ``AssertionError`` unless ``|actual - desired| < 1.5 * 10**-decimal`` everywhere."""
    assert_allclose(actual, desired, rtol=0, atol=1.5 * 10.0 ** (-decimal), err_msg=err_msg)


def assert_almost_equal(actual, desired, decimal=7, err_msg="", verbose=True):
    """Raise ``AssertionError`` unless ``|actual - desired| < 1.5 * 10**-decimal``."""
    assert_allclose(actual, desired, rtol=0, atol=1.5 * 10.0 ** (-decimal), err_msg=err_msg)
