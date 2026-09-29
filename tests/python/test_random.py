# Distribution behavior shared by CPython and shellsim; streams need not match.

import math
import random


def raised_type(operation):
    try:
        operation()
    except Exception as error:
        return type(error)
    raise AssertionError("operation did not raise")


def test_variates_return_values_in_their_documented_domains():
    generator = random.Random(11)
    for _ in range(80):
        assert generator.expovariate(2.0) >= 0.0
        assert generator.expovariate(-2.0) <= 0.0
        assert generator.gammavariate(0.5, 2.0) >= 0.0
        assert generator.gammavariate(2.0, 1.0) >= 0.0
        assert 0.0 <= generator.betavariate(2.0, 3.0) <= 1.0
        assert generator.weibullvariate(2.0, 3.0) >= 0.0


def test_variates_have_broad_expected_means():
    generator = random.Random(17)
    count = 300
    exponential = sum(generator.expovariate(2.0) for _ in range(count)) / count
    gamma = sum(generator.gammavariate(2.0, 3.0) for _ in range(count)) / count
    beta = sum(generator.betavariate(2.0, 3.0) for _ in range(count)) / count
    weibull = sum(generator.weibullvariate(2.0, 2.0) for _ in range(count)) / count
    assert abs(exponential - 0.5) < 0.2
    assert abs(gamma - 6.0) < 2.0
    assert abs(beta - 0.4) < 0.15
    assert abs(weibull - math.sqrt(math.pi)) < 0.5


def test_module_functions_delegate_to_seeded_default_generator():
    for method, arguments in (
        ("expovariate", (2.0,)),
        ("gammavariate", (2.0, 1.0)),
        ("betavariate", (2.0, 3.0)),
        ("weibullvariate", (2.0, 2.0)),
    ):
        random.seed(23)
        instance = random.Random(23)
        assert getattr(random, method)(*arguments) == getattr(instance, method)(*arguments)


def test_gamma_sampling_preserves_gauss_cached_sample():
    generator = random.Random(19)
    reference = random.Random(19)
    generator.gauss()
    reference.gauss()
    expected_second = reference.gauss()
    generator.gammavariate(2.0, 1.0)
    assert generator.gauss() == expected_second


def test_invalid_distribution_parameters_raise_by_type():
    assert raised_type(lambda: random.expovariate(0)) is ZeroDivisionError
    assert raised_type(lambda: random.gammavariate(0, 1)) is ValueError
    assert raised_type(lambda: random.gammavariate(1, -1)) is ValueError
    assert raised_type(lambda: random.betavariate(-1, 1)) is ValueError
    assert raised_type(lambda: random.betavariate(1, 0)) is ValueError
    assert raised_type(lambda: random.weibullvariate(1, 0)) is ZeroDivisionError
    assert raised_type(lambda: random.gammavariate(2)) is TypeError
