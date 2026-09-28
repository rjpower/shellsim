# Portable NumPy semantics: shapes, dtypes, reproducibility, and statistics only.
# Scope: numpy.random's public surface (Generator, RandomState, PCG64, default_rng, and the
# legacy module-level functions). Shellsim's random streams do not reproduce NumPy's own -- the
# project's owner decided determinism and correct statistics are enough, not bit-compatible
# streams -- so this suite never asserts a literal seeded value, only properties that hold for
# any statistically sound generator. It is checked against both shellsim and real NumPy 2.5.3.

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# default_rng, Generator, PCG64.
# ---------------------------------------------------------------------------


def test_default_rng_returns_generator():
    assert isinstance(np.random.default_rng(0), np.random.Generator)
    assert isinstance(np.random.default_rng(), np.random.Generator)
    assert isinstance(np.random.default_rng([1, 2, 3]), np.random.Generator)


def test_default_rng_accepts_generator_and_pcg64():
    rng = np.random.default_rng(1)
    assert np.random.default_rng(rng) is rng
    bit_generator = np.random.PCG64(1)
    wrapped = np.random.default_rng(bit_generator)
    assert isinstance(wrapped, np.random.Generator)
    assert wrapped.bit_generator is bit_generator


def test_generator_over_explicit_pcg64():
    rng = np.random.Generator(np.random.PCG64(3))
    values = rng.random(5)
    assert values.shape == (5,)
    assert values.dtype == np.float64


def test_reproducibility_same_seed_same_values():
    first = np.random.default_rng(123).standard_normal(20).tolist()
    second = np.random.default_rng(123).standard_normal(20).tolist()
    assert first == second


def test_reproducibility_different_seeds_differ():
    first = np.random.default_rng(1).random(10).tolist()
    second = np.random.default_rng(2).random(10).tolist()
    assert first != second


def test_generator_stream_advances_and_reseeding_replays_it():
    rng = np.random.default_rng(5)
    first = rng.random(5).tolist()
    second = rng.random(5).tolist()
    assert first != second
    replay = np.random.default_rng(5)
    assert replay.random(5).tolist() == first
    assert replay.random(5).tolist() == second


# ---------------------------------------------------------------------------
# random / integers / uniform: shapes, dtypes, and ranges.
# ---------------------------------------------------------------------------


def test_random_shape_dtype_and_range():
    rng = np.random.default_rng(0)
    scalar = rng.random()
    assert type(scalar) is float
    assert 0.0 <= scalar < 1.0
    values = rng.random((2, 3))
    assert values.shape == (2, 3)
    assert values.dtype == np.float64
    assert np.all(values >= 0.0) and np.all(values < 1.0)


def test_random_float32():
    values = np.random.default_rng(0).random(100, dtype=np.float32)
    assert values.dtype == np.float32
    assert np.all(values >= 0.0) and np.all(values < 1.0)


def test_integers_range_and_dtype():
    values = np.random.default_rng(0).integers(5, 15, size=1000)
    assert values.dtype == np.int64
    assert np.all(values >= 5) and np.all(values < 15)


def test_integers_endpoint_is_inclusive():
    values = np.random.default_rng(0).integers(1, 6, size=1000, endpoint=True)
    assert np.all(values >= 1) and np.all(values <= 6)
    assert values.max() == 6


def test_integers_high_only_and_negative_range():
    values = np.random.default_rng(0).integers(5, size=1000)
    assert np.all(values >= 0) and np.all(values < 5)
    values = np.random.default_rng(0).integers(-50, 50, size=1000)
    assert np.all(values >= -50) and np.all(values < 50)


def test_integers_scalar_is_numpy_int64():
    value = np.random.default_rng(0).integers(100)
    assert isinstance(value, np.int64)
    assert 0 <= value < 100


@pytest.mark.parametrize(
    "dtype, low, high",
    [
        ("bool", 0, 2),
        ("int8", -100, 100),
        ("uint8", 0, 200),
        ("int16", -1000, 1000),
        ("uint16", 0, 50000),
        ("int32", -100000, 100000),
        ("uint32", 0, 100000),
        ("int64", -100000, 100000),
    ],
)
def test_integers_every_dtype_stays_in_bounds(dtype, low, high):
    values = np.random.default_rng(3).integers(low, high, size=300, dtype=dtype)
    assert values.dtype == np.dtype(dtype)
    assert np.all(values >= low) and np.all(values < high)


def test_integers_empty_range_raises():
    with pytest.raises(ValueError):
        np.random.default_rng(0).integers(5, 5)


def test_integers_bounds_errors():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        rng.integers(-1, 5, dtype=np.uint8)
    with pytest.raises(ValueError):
        rng.integers(0, 257, dtype=np.uint8)
    with pytest.raises(ValueError):
        rng.integers(0)
    with pytest.raises(ValueError):
        rng.integers(5, 4, endpoint=True)
    with pytest.raises(TypeError):
        rng.integers(5, dtype=np.float64)


def test_integers_bounds_broadcast():
    values = np.random.default_rng(3).integers([0, 10, 100], [5, 20, 1000])
    assert values.shape == (3,)
    assert 0 <= values[0] < 5
    assert 10 <= values[1] < 20
    assert 100 <= values[2] < 1000


def test_uniform_range_and_shape():
    values = np.random.default_rng(1).uniform(-2, 2, size=1000)
    assert values.shape == (1000,)
    assert np.all(values >= -2) and np.all(values < 2)
    scalar = np.random.default_rng(1).uniform()
    assert type(scalar) is float
    assert 0.0 <= scalar < 1.0


# ---------------------------------------------------------------------------
# choice, shuffle, permutation.
# ---------------------------------------------------------------------------


def test_choice_with_replacement_stays_in_population():
    values = np.random.default_rng(3).choice(10, size=200)
    assert values.shape == (200,)
    assert np.all(values >= 0) and np.all(values < 10)


def test_choice_without_replacement_is_unique():
    values = np.random.default_rng(3).choice(1000, size=200, replace=False)
    assert len(set(values.tolist())) == 200


def test_choice_without_replacement_from_huge_population():
    values = np.random.default_rng(1).choice(10**9, size=5, replace=False)
    assert len(set(values.tolist())) == 5
    assert np.all(values >= 0) and np.all(values < 10**9)


def test_choice_without_replacement_cannot_exceed_population():
    with pytest.raises(ValueError):
        np.random.default_rng(0).choice(3, size=4, replace=False)


def test_choice_from_array_returns_elements():
    values = np.random.default_rng(3).choice(["a", "b", "c", "d"], size=6)
    assert values.shape == (6,)
    assert set(values.tolist()) <= {"a", "b", "c", "d"}


def test_choice_scalar_is_numpy_scalar():
    value = np.random.default_rng(3).choice([10, 20, 30])
    assert isinstance(value, np.int64)
    assert value in (10, 20, 30)


def test_choice_errors():
    with pytest.raises(ValueError):
        np.random.default_rng(0).choice(3, p=[0.5, 0.6, -0.1])
    with pytest.raises(ValueError):
        np.random.default_rng(0).choice([])


def test_choice_with_probabilities_favors_higher_weight():
    values = np.random.default_rng(3).choice(2, size=20000, p=[0.1, 0.9])
    assert abs(np.mean(values) - 0.9) < 0.02


def test_shuffle_preserves_multiset():
    values = np.arange(20)
    result = np.random.default_rng(5).shuffle(values)
    assert result is None
    assert sorted(values.tolist()) == list(range(20))


def test_shuffle_python_list():
    values = [1, 2, 3, 4, 5]
    np.random.default_rng(2).shuffle(values)
    assert sorted(values) == [1, 2, 3, 4, 5]


def test_shuffle_rejects_read_only_array():
    rng = np.random.default_rng(1)
    values = np.arange(3)
    values.setflags(write=False)
    with pytest.raises(ValueError):
        rng.shuffle(values)


def test_permutation_copies_array():
    source = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    result = np.random.default_rng(5).permutation(source)
    assert sorted(result.tolist()) == sorted(source.tolist())
    assert source.tolist() == [1.5, 2.5, 3.5, 4.5, 5.5]


def test_permutation_of_integer_gives_arange_permutation():
    result = np.random.default_rng(5).permutation(10)
    assert sorted(result.tolist()) == list(range(10))


# ---------------------------------------------------------------------------
# Continuous distributions: shape, dtype, and statistics.
# ---------------------------------------------------------------------------


def test_normal_statistics():
    draws = np.random.default_rng(0).normal(size=20000)
    assert abs(np.mean(draws)) < 0.05
    assert abs(np.std(draws) - 1.0) < 0.05
    shifted = np.random.default_rng(1).normal(5.0, 2.0, size=20000)
    assert abs(np.mean(shifted) - 5.0) < 0.1
    assert abs(np.std(shifted) - 2.0) < 0.1


def test_standard_normal_is_deterministic_and_shaped():
    first = np.random.default_rng(9).standard_normal((3, 4))
    second = np.random.default_rng(9).standard_normal((3, 4))
    assert first.shape == (3, 4)
    assert first.dtype == np.float64
    assert first.tolist() == second.tolist()
    assert type(np.random.default_rng(9).standard_normal()) is float


def test_standard_normal_float32():
    values = np.random.default_rng(9).standard_normal(2000, dtype=np.float32)
    assert values.dtype == np.float32
    assert abs(np.mean(values)) < 0.1


def test_standard_exponential_statistics():
    draws = np.random.default_rng(0).standard_exponential(20000)
    assert np.all(draws >= 0)
    assert draws.dtype == np.float64
    assert abs(np.mean(draws) - 1.0) < 0.05


def test_exponential_scales_mean():
    draws = np.random.default_rng(0).exponential(3.0, size=20000)
    assert np.all(draws >= 0)
    assert abs(np.mean(draws) - 3.0) < 0.2


def test_standard_gamma_statistics():
    for shape in (0.3, 1.0, 2.5, 10.0):
        draws = np.random.default_rng(1).standard_gamma(shape, size=20000)
        assert np.all(draws >= 0)
        assert abs(np.mean(draws) - shape) < 0.5 + 0.1 * shape
        assert abs(np.var(draws) - shape) < 1.0 + 0.2 * shape


def test_gamma_scales_mean_and_variance():
    shape, scale = 3.0, 2.0
    draws = np.random.default_rng(1).gamma(shape, scale, size=20000)
    assert abs(np.mean(draws) - shape * scale) < 0.5
    assert abs(np.var(draws) - shape * scale**2) < 3.0


def test_beta_statistics():
    a, b = 2.0, 5.0
    draws = np.random.default_rng(1).beta(a, b, size=20000)
    assert np.all(draws > 0) and np.all(draws < 1)
    assert abs(np.mean(draws) - a / (a + b)) < 0.02


def test_chisquare_mean_matches_df():
    df = 6.0
    draws = np.random.default_rng(2).chisquare(df, size=20000)
    assert np.all(draws >= 0)
    assert abs(np.mean(draws) - df) < 0.3


def test_f_distribution_is_positive_with_expected_mean():
    draws = np.random.default_rng(2).f(5.0, 20.0, size=20000)
    assert np.all(draws >= 0)
    assert abs(np.mean(draws) - 20.0 / (20.0 - 2.0)) < 0.3


def test_standard_t_is_centered():
    draws = np.random.default_rng(2).standard_t(10.0, size=20000)
    assert abs(np.mean(draws)) < 0.1


def test_lognormal_is_positive_with_expected_mean():
    mean, sigma = 0.0, 0.5
    draws = np.random.default_rng(2).lognormal(mean, sigma, size=20000)
    assert np.all(draws > 0)
    assert abs(np.mean(draws) - np.exp(mean + sigma**2 / 2)) < 0.1


def test_distribution_result_types():
    rng = np.random.default_rng(0)
    assert type(rng.gamma(2.0)) is float
    assert type(rng.beta(2.0, 3.0)) is float
    assert type(rng.lognormal()) is float
    assert type(rng.poisson(2.0)) is int
    assert type(rng.binomial(3, 0.5)) is int
    assert type(rng.standard_gamma(2.0, dtype=np.float32)) is float
    assert rng.binomial(5, 0.5, size=2).dtype == np.int64
    assert np.random.RandomState(0).poisson(2.0, 2).dtype == np.int64
    out = np.empty((2, 2))
    assert rng.standard_gamma([1.5, 2.0], out=out) is out


# ---------------------------------------------------------------------------
# Discrete distributions.
# ---------------------------------------------------------------------------


def test_binomial_statistics_small_and_large_np():
    n, p = 10, 0.3  # inversion branch: n * min(p, 1 - p) <= 30
    small = np.random.default_rng(3).binomial(n, p, size=20000)
    assert np.all(small >= 0) and np.all(small <= n)
    assert abs(np.mean(small) - n * p) < 0.2
    n, p = 1000, 0.4  # BTPE branch
    large = np.random.default_rng(3).binomial(n, p, size=20000)
    assert abs(np.mean(large) - n * p) < 3.0
    assert abs(np.var(large) - n * p * (1 - p)) < 20.0


def test_binomial_degenerate_edges():
    rng = np.random.default_rng(3)
    assert rng.binomial(0, 0.5) == 0
    assert rng.binomial(5, 0.0) == 0
    assert rng.binomial(5, 1.0) == 5


def test_poisson_statistics_small_and_large_lambda():
    small = np.random.default_rng(4).poisson(0.5, size=20000)  # multiplication branch
    assert abs(np.mean(small) - 0.5) < 0.05
    large = np.random.default_rng(4).poisson(50.0, size=20000)  # PTRS branch
    assert abs(np.mean(large) - 50.0) < 1.0
    assert abs(np.var(large) - 50.0) < 5.0


def test_distribution_parameter_errors():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        rng.normal(0, -1)
    with pytest.raises(ValueError):
        rng.uniform(2, 1)
    with pytest.raises(OverflowError):
        rng.uniform(0, np.inf)
    with pytest.raises(ValueError):
        rng.exponential(-1.0)
    with pytest.raises(ValueError):
        rng.gamma(-1)
    with pytest.raises(ValueError):
        rng.beta(-1, 2)
    with pytest.raises(ValueError):
        rng.chisquare(0)
    with pytest.raises(ValueError):
        rng.f([1, 2], [0, 1])
    with pytest.raises(ValueError):
        rng.standard_t(-1)
    with pytest.raises(ValueError):
        rng.binomial(-1, 0.5)
    with pytest.raises(TypeError):
        rng.binomial(np.array([10.5]), 0.5)
    with pytest.raises(ValueError):
        rng.binomial(10, 1.5)
    with pytest.raises(ValueError):
        rng.poisson(-1)
    with pytest.raises(ValueError):
        rng.poisson(1e19)
    with pytest.raises(ValueError):
        rng.poisson([np.nan])
    with pytest.raises(TypeError):
        rng.standard_gamma(2, dtype=np.int32)
    with pytest.raises(ValueError):
        rng.exponential([1, 2], size=3)


# ---------------------------------------------------------------------------
# Legacy RandomState and module-level functions.
# ---------------------------------------------------------------------------


def test_random_state_same_seed_same_values():
    a = np.random.RandomState(0)
    b = np.random.RandomState(0)
    assert a.rand(5).tolist() == b.rand(5).tolist()
    c = np.random.RandomState(1)
    assert a.rand(5).tolist() != c.rand(5).tolist()


def test_random_state_reseed_replays_stream():
    rs = np.random.RandomState(7)
    first = rs.rand(4).tolist()
    rs.seed(7)
    assert rs.rand(4).tolist() == first


def test_random_state_methods_shapes_and_ranges():
    rs = np.random.RandomState(123)
    assert rs.rand(3).shape == (3,)
    values = rs.randint(0, 100, size=4)
    assert values.dtype == np.int64
    assert np.all(values >= 0) and np.all(values < 100)
    assert rs.randn(3).shape == (3,)
    chosen = rs.choice(5, 3, replace=False)
    assert len(set(chosen.tolist())) == 3
    assert rs.uniform(2, 3, size=2).shape == (2,)
    assert rs.random_sample(2).shape == (2,)
    assert sorted(rs.permutation(6).tolist()) == list(range(6))
    assert rs.normal(0, 1, 2).shape == (2,)


def test_random_state_scipy_surface_shapes():
    # SciPy's `check_random_state` calls these directly on a `RandomState`.
    rs = np.random.RandomState(1)
    assert rs.standard_normal(3).shape == (3,)
    assert rs.standard_t(5.0, 3).shape == (3,)
    assert rs.chisquare(3.0, 3).shape == (3,)
    assert rs.f(3.0, 7.0, 3).shape == (3,)
    assert rs.standard_exponential(3).shape == (3,)
    assert rs.binomial(10, 0.3, 3).shape == (3,)
    assert rs.poisson(2.5, 3).shape == (3,)


def test_random_state_is_independent_of_global_state():
    np.random.seed(0)
    rs = np.random.RandomState(0)
    first = rs.random_sample(2).tolist()
    np.random.rand(10)
    rs.seed(0)
    assert rs.random_sample(2).tolist() == first


def test_random_state_shuffle_preserves_multiset():
    rs = np.random.RandomState(123)
    values = np.arange(8)
    rs.shuffle(values)
    assert sorted(values.tolist()) == list(range(8))


def test_random_state_array_seed_is_reproducible():
    first = np.random.RandomState([1, 2, 3]).random_sample(4).tolist()
    second = np.random.RandomState([1, 2, 3]).random_sample(4).tolist()
    assert first == second


def test_legacy_scalar_draws_are_python_floats():
    np.random.seed(0)
    assert type(np.random.rand()) is float
    assert type(np.random.random()) is float
    assert type(np.random.randn()) is float


def test_legacy_shape_arguments():
    np.random.seed(0)
    values = np.random.rand(2, 3)
    assert values.shape == (2, 3)


def test_legacy_reseeding_reproduces_stream():
    np.random.seed(2024)
    first = np.random.rand(4).tolist()
    np.random.seed(2024)
    assert np.random.rand(4).tolist() == first


def test_legacy_randint_scalar_is_python_int():
    np.random.seed(1)
    value = np.random.randint(10)
    assert type(value) is int
    assert 0 <= value < 10


def test_legacy_randint_empty_range_raises():
    with pytest.raises(ValueError):
        np.random.randint(5, 5)


def test_legacy_choice_errors():
    with pytest.raises(ValueError):
        np.random.choice(3, size=5, replace=False)
    with pytest.raises(ValueError):
        np.random.choice(3, p=[0.5, 0.2, 0.2])
    with pytest.raises(ValueError):
        np.random.choice([])


def test_legacy_choice_without_replacement_is_unique():
    np.random.seed(3)
    values = np.random.choice(10, size=5, replace=False)
    assert len(set(values.tolist())) == 5


def test_mtrand_shared_global_reflects_seed():
    assert isinstance(np.random.mtrand._rand, np.random.RandomState)
    np.random.seed(11)
    expected = np.random.rand(3).tolist()
    np.random.seed(11)
    assert np.random.mtrand._rand.rand(3).tolist() == expected


@pytest.mark.parametrize(
    "size, shape", [(3, (3,)), ((2, 3), (2, 3)), ((1, 2, 2), (1, 2, 2)), ((0,), (0,))]
)
def test_size_argument_sets_shape(size, shape):
    np.random.seed(0)
    assert np.random.random_sample(size).shape == shape
    assert np.random.randint(0, 5, size=size).shape == shape
    assert np.random.normal(size=size).shape == shape
    rng = np.random.default_rng(0)
    assert rng.random(size).shape == shape
    assert rng.integers(0, 5, size=size).shape == shape
    assert rng.uniform(size=size).shape == shape
    assert rng.normal(size=size).shape == shape
