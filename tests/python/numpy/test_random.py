# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: exact legacy MT19937 and default_rng PCG64 streams, result types, and seeding rules.
# Generator's ziggurat tables are computed from Marsaglia and Tsang's recursion, so its normal
# and exponential draws, and the gamma-family draws built on them, match NumPy to a tolerance.

import numpy as np
import pytest
from numpy.testing import assert_allclose

ZIGGURAT_RTOL = 1e-13


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


def test_seed_sequence_generates_numpy_words():
    seq = np.random.SeedSequence(12345)
    assert seq.generate_state(3).tolist() == [2688385916, 3048105090, 4196366895]
    assert seq.generate_state(2, np.uint64).tolist() == [13091511679009522556, 13538552136045918767]
    child = seq.spawn(2)[1]
    assert child.spawn_key == (1,)
    assert child.generate_state(2).tolist() == [1457248422, 358904087]
    with pytest.raises(ValueError, match="only support uint32 or uint64"):
        seq.generate_state(2, np.int32)
    with pytest.raises(TypeError, match="SeedSequence expects int or sequence of ints"):
        np.random.SeedSequence(1.5)


def test_pcg64_state_advance_and_jumped():
    bit_generator = np.random.PCG64(7)
    assert bit_generator.state == {
        "bit_generator": "PCG64",
        "state": {
            "state": 208745520555909116978795849195383758904,
            "inc": 261136684632268670825940853076396136793,
        },
        "has_uint32": 0,
        "uinteger": 0,
    }
    assert np.random.PCG64(1).advance(12345).random_raw(2).tolist() == [
        6725986476597619031,
        15801925886187179136,
    ]
    assert np.random.PCG64(1).jumped(3).random_raw(2).tolist() == [
        5863065313089933784,
        11785604140192313526,
    ]


def test_pcg64_jumped_returns_a_new_generator_for_any_stream():
    bit_generator = np.random.PCG64(2)
    jumped = bit_generator.jumped()
    assert jumped is not bit_generator and type(jumped) is np.random.PCG64
    assert jumped.random_raw(2).tolist() == [17613949133350213019, 14352335435533080182]
    assert bit_generator.state == np.random.PCG64(2).state
    assert np.random.PCG64(1).advance(-1).advance(1).random_raw() == np.random.PCG64(1).random_raw()


def test_pcg64_advance_and_jumped_discard_the_buffered_half_word():
    bit_generator = np.random.PCG64(1)
    np.random.Generator(bit_generator).integers(0, 10, dtype=np.int32)
    assert bit_generator.state["has_uint32"] == 1
    assert bit_generator.jumped().state["has_uint32"] == 0
    assert bit_generator.advance(5).state["has_uint32"] == 0


def test_generator_state_round_trip_includes_buffered_half():
    rng = np.random.default_rng(7)
    rng.integers(0, 10, dtype=np.int32)
    state = rng.bit_generator.state
    assert state["has_uint32"] == 1
    first = rng.random(3).tolist()
    rng.bit_generator.state = state
    assert rng.random(3).tolist() == first


def test_mt19937_seeded_from_seed_sequence():
    assert np.random.MT19937(42).random_raw(3).tolist() == [2327846034, 3904886566, 2661450408]
    assert np.random.MT19937(42).state["state"]["pos"] == 623
    rng = np.random.Generator(np.random.MT19937(42))
    assert rng.random(2).tolist() == [0.5419938930062744, 0.6196672126927824]


def test_generator_single_precision_streams():
    values = np.random.default_rng(2024).random(3, dtype=np.float32)
    assert values.dtype == np.float32
    assert values.tolist() == [0.24152815341949463, 0.6758313179016113, 0.09234333038330078]
    normals = np.random.default_rng(2024).standard_normal(4, dtype=np.float32)
    assert normals.tolist() == [
        0.6110844016075134,
        0.846587061882019,
        0.21217942237854004,
        0.38399016857147217,
    ]


def test_generator_normal_follows_numpy_ziggurat():
    expected = [1.0288568739519013, 1.6419200406711503, 1.1467195295966137, -0.9731795154745656]
    assert_allclose(np.random.default_rng(2024).standard_normal(4), expected, rtol=ZIGGURAT_RTOL)
    # Enough draws to reach the ziggurat's base-layer tail and wedge rejections.
    draws = np.random.default_rng(5).standard_normal(200000)
    assert_allclose([draws.min(), draws.max()], [-4.820175842608121, 4.371795668265591])
    assert_allclose(draws.sum(), 365.95781498290273, rtol=1e-11)
    expected = [0.691168384129572, 11.643236287002317, 100.66087415236677]
    assert_allclose(np.random.default_rng(1).normal([0, 10, 100], 2), expected, rtol=ZIGGURAT_RTOL)



def _force_next_word(rng, word):
    # PCG64 advances its 128-bit LCG state and then outputs it; a state whose high half is zero
    # outputs its low half unchanged, so stepping back from `word` makes it the next raw word.
    multiplier = 0x2360ED051FC65DA44385DF649FCCF645
    state = rng.bit_generator.state
    increment = state["state"]["inc"]
    state["state"]["state"] = (word - increment) * pow(multiplier, -1, 1 << 128) % (1 << 128)
    rng.bit_generator.state = state


def test_generator_normal_rejects_candidates_above_the_top_layer_density():
    rng = np.random.default_rng(1)
    _force_next_word(rng, 12345)
    assert rng.bit_generator.random_raw() == 12345
    # Layer 1 with the largest mantissa lies above the density, so the draw restarts.
    _force_next_word(rng, 1 | ((1 << 52) - 1) << 9)
    assert_allclose(rng.standard_normal(), 0.7171486007530798, rtol=ZIGGURAT_RTOL)

@pytest.mark.parametrize(
    "legacy, seed, low, high, dtype, expected",
    [
        (False, 3, -100, 100, "int8", [93, 51, 48, 61, 14, -80]),
        (False, 9, 0, 100, "uint16", [52, 42, 65, 87, 1, 96]),
        (False, 9, 0, 2, "bool", [True, False, False, False, False, True]),
        (True, 9, 0, 100, "uint8", [97, 39, 2, 92, 81, 77]),
    ],
)
def test_integer_dtypes_use_numpy_buffered_draws(legacy, seed, low, high, dtype, expected):
    if legacy:
        values = np.random.RandomState(seed).randint(low, high, size=6, dtype=dtype)
    else:
        values = np.random.default_rng(seed).integers(low, high, size=6, dtype=dtype)
    assert values.dtype == np.dtype(dtype)
    assert values.tolist() == expected


def test_integer_bounds_broadcast():
    assert np.random.default_rng(3).integers([0, 10, 100], [5, 20, 1000]).tolist() == [4, 10, 261]


def test_integer_bounds_errors():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="low is out of bounds for uint8"):
        rng.integers(-1, 5, dtype=np.uint8)
    with pytest.raises(ValueError, match="high is out of bounds for uint8"):
        rng.integers(0, 257, dtype=np.uint8)
    with pytest.raises(ValueError, match="high <= 0"):
        rng.integers(0)
    with pytest.raises(ValueError, match="low > high"):
        rng.integers(5, 4, endpoint=True)
    with pytest.raises(TypeError, match="Unsupported dtype"):
        rng.integers(5, dtype=np.float64)


def test_generator_choice_without_replacement_uses_floyd_and_tail_shuffle():
    assert np.random.default_rng(11).choice(1000, 5, replace=False).tolist() == [128, 590, 133, 795, 498]
    assert np.random.default_rng(11).choice(20000, 1000, replace=False)[:4].tolist() == [15148, 11237, 12942, 13198]


def test_legacy_array_seed_and_gauss_state():
    rs = np.random.RandomState([1, 2, 3])
    assert rs.random_sample(2).tolist() == [0.6098612722867289, 0.8866970434146851]
    rs.randn()
    state = rs.get_state()
    assert state[3] == 1
    first = rs.randn(3).tolist()
    rs.set_state(state)
    assert rs.randn(3).tolist() == first
    with pytest.raises(ValueError, match="Seed must be between 0 and 2\\*\\*32 - 1"):
        np.random.RandomState(2**32)


def test_distribution_parameter_errors():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="scale < 0"):
        rng.normal(0, -1)
    with pytest.raises(ValueError, match="high - low < 0"):
        rng.uniform(2, 1)
    with pytest.raises(OverflowError):
        rng.uniform(0, np.inf)
    with pytest.raises(ValueError, match="array is read-only"):
        values = np.arange(3)
        values.setflags(write=False)
        rng.shuffle(values)


def test_generator_exponential_and_gamma_streams():
    rng = np.random.default_rng(12345)
    expected = [0.18413256735377503, 0.6450270693873458, 4.690218692461341, 0.4185586661538189]
    assert_allclose(rng.standard_exponential(4), expected, rtol=ZIGGURAT_RTOL)
    expected = [[0.25552372206434737, 2.645608513542529], [0.7271540468558736, 0.39866300558055573]]
    assert_allclose(rng.exponential([0.5, 2.0], size=(2, 2)), expected, rtol=ZIGGURAT_RTOL)
    # Shapes below 1 use rejection from the exponential; above 1, Marsaglia and Tsang.
    expected = [0.2668038100283782, 0.009614776660842888, 0.06569752471329031]
    assert_allclose(rng.standard_gamma(0.3, 3), expected, rtol=ZIGGURAT_RTOL)
    expected = [3.5475905768350997, 105.5268819117015]
    assert_allclose(rng.standard_gamma([2.5, 100.0]), expected, rtol=ZIGGURAT_RTOL)
    expected = [10.922463334055482, 8.859155392065684]
    assert_allclose(rng.gamma(3.0, 2.0, 2), expected, rtol=ZIGGURAT_RTOL)


def test_generator_chisquare_f_and_t_streams():
    rng = np.random.default_rng(7)
    expected = [0.8293904490318487, 46.66008573980642]
    assert_allclose(rng.chisquare([1.0, 50.0]), expected, rtol=ZIGGURAT_RTOL)
    expected = [0.5281338185443447, 0.42095518115438546]
    assert_allclose(rng.f(3.0, [7.0, 20.0]), expected, rtol=ZIGGURAT_RTOL)
    expected = [0.16747494601466173, 0.8474464134087695]
    assert_allclose(rng.standard_t([1.0, 30.0]), expected, rtol=ZIGGURAT_RTOL)


def test_generator_binomial_and_poisson_streams():
    rng = np.random.default_rng(3)
    # Inversion for a mean of at most 30, BTPE above, on either side of p = 0.5.
    assert rng.binomial(10, 0.3, 5).tolist() == [1, 2, 4, 3, 1]
    assert rng.binomial([1000, 5000], [0.4, 0.55]).tolist() == [405, 2789]
    rng = np.random.default_rng(4)
    # Multiplication of uniforms below lam = 10, transformed rejection from 10.
    assert rng.poisson([0.5, 9.9, 10.0, 1e6]).tolist() == [1, 16, 11, 1002093]


def test_degenerate_binomial_draws_only_in_the_legacy_stream():
    rng = np.random.default_rng(3)
    assert rng.binomial([0, 5], [0.5, 0.0]).tolist() == [0, 0]
    assert rng.random() == np.random.default_rng(3).random()
    legacy = np.random.RandomState(1)
    assert legacy.binomial(5, 0.0) == 0
    assert legacy.random_sample() == np.random.RandomState(1).random_sample(2)[1]


def test_generator_single_precision_and_inverse_exponentials():
    rng = np.random.default_rng(5)
    values = rng.standard_exponential(3, dtype=np.float32)
    assert values.dtype == np.float32
    assert values.tolist() == [2.142340898513794, 4.171111583709717, 0.1283617615699768]
    assert rng.standard_exponential(2, method="inv").tolist() == [
        0.7242778733211445,
        0.33659417617893683,
    ]
    values = rng.standard_gamma([0.4, 3.3], dtype=np.float32)
    assert values.dtype == np.float32
    assert values.tolist() == [0.7545587420463562, 2.917715072631836]


def test_legacy_distribution_streams():
    rs = np.random.RandomState(99)
    assert rs.standard_exponential(3).tolist() == [
        1.1155912955453395,
        0.669583789183997,
        1.7458028817673252,
    ]
    assert rs.standard_gamma([0.3, 2.5]).tolist() == [9.815276649129864e-06, 0.47209584409374594]
    assert rs.chisquare(3.0, 2).tolist() == [3.7946030732640663, 1.1419588440027002]
    assert rs.f(3.0, 7.0, 2).tolist() == [2.077965567296253, 1.1061800372715582]
    assert rs.standard_t(4.0, 2).tolist() == [0.4749392466856027, -0.5068920511897095]
    assert rs.binomial([10, 1000], [0.3, 0.4]).tolist() == [3, 400]
    assert rs.poisson([3.0, 250.0]).tolist() == [2, 237]
    np.random.seed(5)
    assert np.random.exponential(2.0, 2).tolist() == [0.5020399546453052, 4.091739757699749]
    assert np.random.poisson(3.0) == 4
    assert np.random.binomial(20, 0.5) == 10


def test_distribution_result_types():
    rng = np.random.default_rng(0)
    assert type(rng.gamma(2.0)) is float
    assert type(rng.poisson(2.0)) is int
    assert type(rng.binomial(3, 0.5)) is int
    assert type(rng.standard_gamma(2.0, dtype=np.float32)) is float
    assert rng.binomial(5, 0.5, size=2).dtype == np.int64
    assert np.random.RandomState(0).poisson(2.0, 2).dtype == np.int64
    out = np.empty((2, 2))
    assert rng.standard_gamma([1.5, 2.0], out=out) is out


@pytest.mark.parametrize(
    ("method", "args", "message"),
    [
        ("gamma", (-1,), "shape < 0"),
        ("gamma", ([1], [-1]), "scale < 0"),
        ("chisquare", (0,), "df <= 0"),
        ("f", ([1, 2], [0]), "dfden <= 0"),
        ("binomial", (10, 1.5), "p < 0, p > 1 or p is NaN"),
        ("binomial", ([10], [1.5]), "p < 0, p > 1 or p contains NaNs"),
        ("binomial", (-1, 0.5), "n < 0"),
        ("poisson", (-1,), "lam < 0 or lam is NaN"),
        ("poisson", (1e19,), "lam value too large"),
    ],
)
def test_distribution_parameter_constraints(method, args, message):
    with pytest.raises(ValueError, match=message):
        getattr(np.random.default_rng(1), method)(*args)


def test_distribution_argument_errors():
    rng = np.random.default_rng(1)
    # The upper bound is checked first, and NaN fails it.
    with pytest.raises(ValueError, match="lam value too large"):
        rng.poisson([np.nan])
    with pytest.raises(TypeError, match="Cannot cast array data"):
        rng.binomial(np.array([10.5]), 0.5)
    with pytest.raises(TypeError, match="Unsupported dtype"):
        rng.standard_gamma(2, dtype=np.int32)
    with pytest.raises(ValueError, match="shape mismatch"):
        rng.exponential([1, 2], size=3)
