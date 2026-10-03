"""Binary search over sorted sequences."""

__all__ = ["bisect", "bisect_left", "bisect_right", "insort", "insort_left", "insort_right"]


def bisect_right(a, x, lo=0, hi=None, *, key=None):
    if lo < 0:
        raise ValueError("lo must be non-negative")
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        probe = a[mid] if key is None else key(a[mid])
        if x < probe:
            hi = mid
        else:
            lo = mid + 1
    return lo


def bisect_left(a, x, lo=0, hi=None, *, key=None):
    if lo < 0:
        raise ValueError("lo must be non-negative")
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        probe = a[mid] if key is None else key(a[mid])
        if probe < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def insort_right(a, x, lo=0, hi=None, *, key=None):
    position = bisect_right(a, x if key is None else key(x), lo, hi, key=key)
    a.insert(position, x)


def insort_left(a, x, lo=0, hi=None, *, key=None):
    position = bisect_left(a, x if key is None else key(x), lo, hi, key=key)
    a.insert(position, x)


bisect = bisect_right
insort = insort_right
