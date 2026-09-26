# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: exact legacy MT19937 and default_rng PCG64 streams, result types, and seeding rules.

import numpy as np
import pytest


def test_legacy_rand_stream():
    np.random.seed(0)
    values = np.random.rand(5)
    assert values.dtype == np.float64
    assert values.tolist() == [
        0.5488135039273248,
        0.7151893663724195,
        0.6027633760716439,
        0.5448831829968969,
        0.4236547993389047,
    ]


def test_legacy_rand_shape_arguments():
    np.random.seed(0)
    values = np.random.rand(2, 3)
    assert values.shape == (2, 3)
    assert values.tolist() == [
        [0.5488135039273248, 0.7151893663724195, 0.6027633760716439],
        [0.5448831829968969, 0.4236547993389047, 0.6458941130666561],
    ]


def test_legacy_stream_continues_across_calls():
    np.random.seed(0)
    first = np.random.rand(3).tolist()
    second = np.random.rand(3).tolist()
    assert first == [0.5488135039273248, 0.7151893663724195, 0.6027633760716439]
    assert second == [0.5448831829968969, 0.4236547993389047, 0.6458941130666561]


def test_legacy_reseeding_reproduces_stream():
    np.random.seed(2024)
    first = np.random.rand(4).tolist()
    np.random.seed(2024)
    assert np.random.rand(4).tolist() == first


def test_legacy_scalar_draws_are_python_floats():
    np.random.seed(0)
    value = np.random.rand()
    assert type(value) is float
    assert value == 0.5488135039273248
    np.random.seed(0)
    value = np.random.random()
    assert type(value) is float
    assert value == 0.5488135039273248
    np.random.seed(0)
    value = np.random.randn()
    assert type(value) is float
    assert value == 1.764052345967664


def test_legacy_random_and_random_sample():
    np.random.seed(42)
    assert np.random.random(3).tolist() == [0.3745401188473625, 0.9507143064099162, 0.7319939418114051]
    np.random.seed(42)
    values = np.random.random_sample((2, 2))
    assert values.shape == (2, 2)
    assert values.tolist() == [
        [0.3745401188473625, 0.9507143064099162],
        [0.7319939418114051, 0.5986584841970366],
    ]


def test_legacy_randint_stream():
    np.random.seed(1)
    values = np.random.randint(0, 10, size=8)
    assert values.dtype == np.int64
    assert values.tolist() == [5, 8, 9, 5, 0, 0, 1, 7]


def test_legacy_randint_high_only_and_negative_range():
    np.random.seed(1)
    assert np.random.randint(5, size=6).tolist() == [3, 4, 0, 1, 3, 0]
    np.random.seed(1)
    values = np.random.randint(-100, 100, size=(2, 3))
    assert values.shape == (2, 3)
    assert values.tolist() == [[-63, 40, -28], [37, 33, -21]]


def test_legacy_randint_wide_range():
    np.random.seed(0)
    assert np.random.randint(0, 2**40, size=3).tolist() == [741280623151, 506137267392, 291447657211]


def test_legacy_randint_scalar_is_python_int():
    np.random.seed(1)
    value = np.random.randint(10)
    assert type(value) is int
    assert value == 5


def test_legacy_randint_empty_range_raises():
    with pytest.raises(ValueError):
        np.random.randint(5, 5)


def test_legacy_choice_with_replacement():
    np.random.seed(3)
    assert np.random.choice(10, size=5).tolist() == [8, 9, 3, 8, 8]
    np.random.seed(3)
    assert np.random.choice(["a", "b", "c", "d"], size=6).tolist() == ["c", "a", "b", "d", "a", "a"]
    np.random.seed(0)
    values = np.random.choice([0.5, 1.5, 2.5], size=4)
    assert values.dtype == np.float64
    assert values.tolist() == [0.5, 1.5, 0.5, 1.5]


def test_legacy_choice_without_replacement():
    np.random.seed(3)
    values = np.random.choice(10, size=5, replace=False)
    assert values.tolist() == [5, 4, 1, 2, 9]
    assert len(set(values.tolist())) == 5


def test_legacy_choice_with_probabilities():
    np.random.seed(3)
    assert np.random.choice(4, size=8, p=[0.1, 0.2, 0.3, 0.4]).tolist() == [2, 3, 1, 2, 3, 3, 1, 1]
    np.random.seed(3)
    assert np.random.choice(5, size=3, replace=False, p=[0.1, 0.2, 0.3, 0.2, 0.2]).tolist() == [2, 3, 1]


def test_legacy_choice_scalar_is_numpy_scalar():
    np.random.seed(3)
    value = np.random.choice([10, 20, 30])
    assert isinstance(value, np.int64)
    assert value == 30


def test_legacy_choice_errors():
    with pytest.raises(ValueError):
        np.random.choice(3, size=5, replace=False)
    with pytest.raises(ValueError):
        np.random.choice(3, p=[0.5, 0.2, 0.2])
    with pytest.raises(ValueError):
        np.random.choice([])


def test_legacy_shuffle_and_permutation_share_stream():
    np.random.seed(5)
    values = np.arange(10)
    assert np.random.shuffle(values) is None
    assert values.tolist() == [9, 5, 2, 4, 7, 1, 0, 8, 6, 3]
    np.random.seed(5)
    assert np.random.permutation(10).tolist() == [9, 5, 2, 4, 7, 1, 0, 8, 6, 3]


def test_legacy_permutation_copies_array():
    source = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    np.random.seed(5)
    result = np.random.permutation(source)
    assert result.tolist() == [5.5, 1.5, 2.5, 3.5, 4.5]
    assert source.tolist() == [1.5, 2.5, 3.5, 4.5, 5.5]


def test_legacy_shuffle_moves_rows_of_2d_array():
    np.random.seed(2)
    values = np.arange(6).reshape(3, 2)
    np.random.shuffle(values)
    assert values.tolist() == [[4, 5], [2, 3], [0, 1]]


def test_legacy_shuffle_python_list():
    np.random.seed(2)
    values = [1, 2, 3, 4, 5]
    np.random.shuffle(values)
    assert values == [3, 5, 2, 4, 1]


def test_legacy_uniform():
    np.random.seed(7)
    assert np.random.uniform(-1, 1, size=4).tolist() == [
        -0.8473834212520857,
        0.5598375844802292,
        -0.123181537118213,
        0.44693035566188244,
    ]
    np.random.seed(7)
    assert np.random.uniform(size=3).tolist() == [0.07630828937395717, 0.7799187922401146, 0.4384092314408935]


def test_legacy_randn_is_bit_exact():
    np.random.seed(0)
    assert np.random.randn(2, 3).tolist() == [
        [1.764052345967664, 0.4001572083672233, 0.9787379841057392],
        [2.240893199201458, 1.8675579901499675, -0.977277879876411],
    ]


def test_legacy_normal_scales_and_shifts():
    np.random.seed(0)
    assert np.random.normal(10, 2, size=4).tolist() == [
        13.528104691935328,
        10.800314416734446,
        11.957475968211478,
        14.481786398402916,
    ]
    np.random.seed(0)
    value = np.random.normal()
    assert type(value) is float
    assert value == 1.764052345967664


def test_random_state_methods():
    rs = np.random.RandomState(123)
    assert rs.rand(3).tolist() == [0.6964691855978616, 0.28613933495037946, 0.2268514535642031]
    assert rs.randint(0, 100, size=4).tolist() == [17, 83, 57, 86]
    assert rs.randn(3).tolist() == [1.5953011188100306, -1.7830943139593969, -0.2864514709599152]
    assert rs.choice(5, 3, replace=False).tolist() == [4, 2, 1]
    assert rs.uniform(2, 3, size=2).tolist() == [2.24475927695392, 2.694755177185269]
    assert rs.random_sample(2).tolist() == [0.5939023996740237, 0.6317920176870504]
    assert rs.permutation(6).tolist() == [1, 5, 4, 0, 2, 3]
    assert rs.normal(0, 1, 2).tolist() == [-1.041968961197559, -1.7319730537003741]


def test_random_state_matches_global_seed():
    np.random.seed(123)
    global_values = np.random.rand(3).tolist()
    assert np.random.RandomState(123).rand(3).tolist() == global_values


def test_random_state_is_independent_of_global_state():
    np.random.seed(0)
    rs = np.random.RandomState(0)
    first = rs.random_sample(2).tolist()
    np.random.rand(10)
    rs.seed(0)
    assert rs.random_sample(2).tolist() == first
    assert first == [0.5488135039273248, 0.7151893663724195]


def test_random_state_shuffle():
    rs = np.random.RandomState(123)
    values = np.arange(8)
    rs.shuffle(values)
    assert values.tolist() == [0, 1, 3, 7, 4, 2, 5, 6]


def test_default_rng_returns_generator():
    rng = np.random.default_rng(0)
    assert isinstance(rng, np.random.Generator)


def test_generator_random_stream():
    rng = np.random.default_rng(42)
    values = rng.random(4)
    assert values.dtype == np.float64
    assert values.tolist() == [0.7739560485559633, 0.4388784397520523, 0.8585979199113825, 0.6973680290593639]


def test_generator_random_shape_and_scalar():
    values = np.random.default_rng(42).random((2, 2))
    assert values.shape == (2, 2)
    assert values.tolist() == [
        [0.7739560485559633, 0.4388784397520523],
        [0.8585979199113825, 0.6973680290593639],
    ]
    value = np.random.default_rng(42).random()
    assert type(value) is float
    assert value == 0.7739560485559633


def test_generator_stream_continues_across_methods():
    rng = np.random.default_rng(12345)
    assert rng.random(2).tolist() == [0.22733602246716966, 0.31675833970975287]
    assert rng.integers(0, 1000, 3).tolist() == [204, 797, 642]
    assert rng.random(2).tolist() == [0.391109550601909, 0.33281392786638453]


def test_generator_integers():
    rng = np.random.default_rng(0)
    values = rng.integers(0, 10, size=8)
    assert values.dtype == np.int64
    assert values.tolist() == [8, 6, 5, 2, 3, 0, 0, 0]


def test_generator_integers_endpoint():
    rng = np.random.default_rng(0)
    assert rng.integers(1, 6, size=8, endpoint=True).tolist() == [6, 4, 4, 2, 2, 1, 1, 1]


def test_generator_integers_high_only_negative_and_wide():
    assert np.random.default_rng(0).integers(5, size=6).tolist() == [4, 3, 2, 1, 1, 0]
    assert np.random.default_rng(0).integers(-50, 50, size=(2, 3)).tolist() == [[35, 13, 1], [-24, -20, -46]]
    assert np.random.default_rng(0).integers(0, 2**40, size=3).tolist() == [700346781657, 296633628802, 45050865998]


def test_generator_integers_scalar_is_numpy_int64():
    value = np.random.default_rng(0).integers(100)
    assert isinstance(value, np.int64)
    assert value == 85


def test_generator_integers_empty_range_raises():
    with pytest.raises(ValueError):
        np.random.default_rng(0).integers(5, 5)


def test_generator_uniform():
    rng = np.random.default_rng(1)
    assert rng.uniform(-2, 2, size=4).tolist() == [
        0.047286498801026866,
        1.8018547853037412,
        -1.423361549121465,
        1.7945977885489754,
    ]
    value = np.random.default_rng(1).uniform()
    assert type(value) is float
    assert value == 0.5118216247002567


def test_generator_choice_with_replacement():
    assert np.random.default_rng(3).choice(10, size=5).tolist() == [8, 0, 1, 2, 1]
    assert np.random.default_rng(3).choice(["a", "b", "c", "d"], size=6).tolist() == ["d", "a", "a", "a", "a", "d"]
    assert np.random.default_rng(0).choice([0.5, 1.5, 2.5], size=4).tolist() == [2.5, 1.5, 1.5, 0.5]


def test_generator_choice_without_replacement():
    assert np.random.default_rng(3).choice(10, size=5, replace=False).tolist() == [1, 4, 0, 2, 9]
    assert np.random.default_rng(3).choice(5, size=5, replace=False).tolist() == [4, 1, 2, 3, 0]


def test_generator_choice_with_probabilities():
    rng = np.random.default_rng(3)
    assert rng.choice(4, size=8, p=[0.1, 0.2, 0.3, 0.4]).tolist() == [0, 1, 3, 2, 0, 2, 2, 1]


def test_generator_choice_scalar_and_errors():
    value = np.random.default_rng(3).choice([10, 20, 30])
    assert isinstance(value, np.int64)
    assert value == 30
    with pytest.raises(ValueError):
        np.random.default_rng(0).choice(3, size=4, replace=False)
    with pytest.raises(ValueError):
        np.random.default_rng(0).choice(3, p=[0.5, 0.6, -0.1])


def test_generator_permutation_and_shuffle():
    assert np.random.default_rng(5).permutation(10).tolist() == [7, 6, 1, 3, 2, 4, 0, 9, 5, 8]
    values = np.arange(10)
    assert np.random.default_rng(5).shuffle(values) is None
    assert values.tolist() == [7, 6, 1, 3, 2, 4, 0, 9, 5, 8]


def test_generator_permutation_copies_array():
    source = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    assert np.random.default_rng(5).permutation(source).tolist() == [5.5, 4.5, 2.5, 3.5, 1.5]
    assert source.tolist() == [1.5, 2.5, 3.5, 4.5, 5.5]


def test_generator_shuffle_rows_and_lists():
    values = np.arange(6).reshape(3, 2)
    np.random.default_rng(5).shuffle(values)
    assert values.tolist() == [[2, 3], [4, 5], [0, 1]]
    items = [1, 2, 3, 4, 5]
    np.random.default_rng(5).shuffle(items)
    assert items == [5, 4, 2, 3, 1]


def test_generator_normal_is_deterministic():
    first = np.random.default_rng(9).normal(size=(3, 4))
    second = np.random.default_rng(9).normal(size=(3, 4))
    assert first.shape == (3, 4)
    assert first.dtype == np.float64
    assert first.tolist() == second.tolist()
    assert type(np.random.default_rng(9).normal()) is float


def test_generator_normal_statistics():
    draws = np.random.default_rng(0).normal(size=10000)
    assert abs(np.mean(draws)) < 0.05
    assert abs(np.std(draws) - 1.0) < 0.05
    shifted = np.random.default_rng(1).normal(5.0, 0.1, size=1000)
    assert abs(np.mean(shifted) - 5.0) < 0.05


def test_generator_seeds_are_independent_streams():
    assert np.random.default_rng(1).random(3).tolist() != np.random.default_rng(2).random(3).tolist()


@pytest.mark.parametrize("size, shape", [(3, (3,)), ((2, 3), (2, 3)), ((1, 2, 2), (1, 2, 2)), ((0,), (0,))])
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
