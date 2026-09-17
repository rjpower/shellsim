"""Capability-free descriptive statistics over ordinary Python iterables."""

import math


def _values(data):
    values = list(data)
    if not values:
        raise ValueError("statistics requires at least one data point")
    return values


def fmean(data, weights=None):
    values = _values(data)
    if weights is None:
        return sum(values) / len(values)
    weights = list(weights)
    if len(values) != len(weights):
        raise ValueError("data and weights must be the same length")
    total = sum(weights)
    if total == 0:
        raise ZeroDivisionError("weights sum to zero")
    return sum(value * weight for value, weight in zip(values, weights)) / total


def mean(data):
    return fmean(data)


def median(data):
    values = sorted(_values(data))
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def median_low(data):
    values = sorted(_values(data))
    return values[(len(values) - 1) // 2]


def median_high(data):
    values = sorted(_values(data))
    return values[len(values) // 2]


def multimode(data):
    values = _values(data)
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    highest = max(counts.values())
    return [value for value in counts if counts[value] == highest]


def mode(data):
    return multimode(data)[0]


def pvariance(data, mu=None):
    values = _values(data)
    center = fmean(values) if mu is None else mu
    return sum((value - center) ** 2 for value in values) / len(values)


def variance(data, xbar=None):
    values = _values(data)
    if len(values) < 2:
        raise ValueError("variance requires at least two data points")
    center = fmean(values) if xbar is None else xbar
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


def pstdev(data, mu=None):
    return math.sqrt(pvariance(data, mu))


def stdev(data, xbar=None):
    return math.sqrt(variance(data, xbar))


def geometric_mean(data):
    values = _values(data)
    product = 1.0
    for value in values:
        if value < 0:
            raise ValueError("geometric mean requires non-negative inputs")
        product *= value
    return product ** (1.0 / len(values))


def harmonic_mean(data, weights=None):
    values = _values(data)
    if weights is None:
        weights = [1] * len(values)
    else:
        weights = list(weights)
    if len(values) != len(weights):
        raise ValueError("data and weights must be the same length")
    if any(value < 0 for value in values):
        raise ValueError("harmonic mean does not support negative values")
    if any(value == 0 for value in values):
        return 0
    return sum(weights) / sum(weight / value for value, weight in zip(values, weights))


def quantiles(data, n=4, method="exclusive"):
    values = sorted(_values(data))
    if n < 1:
        raise ValueError("n must be at least 1")
    if len(values) < 2:
        raise ValueError("quantiles requires at least two data points")
    result = []
    for index in range(1, n):
        if method == "inclusive":
            position = index * (len(values) - 1) / n
        elif method == "exclusive":
            position = index * (len(values) + 1) / n - 1
        else:
            raise ValueError("method must be 'inclusive' or 'exclusive'")
        position = max(0.0, min(float(len(values) - 1), position))
        lower = int(position)
        fraction = position - lower
        upper = min(lower + 1, len(values) - 1)
        result.append(values[lower] + fraction * (values[upper] - values[lower]))
    return result
