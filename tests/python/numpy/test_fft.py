# Portable NumPy semantics. Expectations checked against NumPy 2.5.3 on CPython 3.14.4.
# Scope: numpy.fft transforms, inverses, real transforms, and sample-frequency helpers.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal


@pytest.mark.parametrize("n", [1, 2, 4, 6, 7, 8, 12, 15, 16])
def test_fft_ifft_round_trip(n):
    signal = np.cos(np.arange(n) * 0.7) + 1j * np.sin(np.arange(n) * 0.3)
    assert_allclose(np.fft.ifft(np.fft.fft(signal)), signal, atol=1e-12)


@pytest.mark.parametrize("n", [4, 6, 7, 12])
def test_fft_matches_direct_dft(n):
    signal = np.arange(n) ** 2 - 3.0 * np.arange(n)
    k = np.arange(n)
    kernel = np.exp(-2j * np.pi * np.outer(k, k) / n)
    assert_allclose(np.fft.fft(signal), kernel @ signal, atol=1e-10)


def test_fft_of_impulse_is_flat():
    impulse = np.zeros(8)
    impulse[0] = 1.0
    assert_allclose(np.fft.fft(impulse), np.ones(8), atol=1e-12)


def test_fft_of_shifted_impulse_is_phase_ramp():
    impulse = np.zeros(6)
    impulse[1] = 1.0
    expected = np.exp(-2j * np.pi * np.arange(6) / 6)
    assert_allclose(np.fft.fft(impulse), expected, atol=1e-12)


def test_fft_of_constant_is_dc_only():
    result = np.fft.fft(np.full(7, 2.0))
    expected = np.zeros(7, dtype=complex)
    expected[0] = 14.0
    assert_allclose(result, expected, atol=1e-12)


def test_fft_of_cosine_has_symmetric_peaks():
    n = 16
    signal = np.cos(2 * np.pi * 3 * np.arange(n) / n)
    spectrum = np.fft.fft(signal)
    expected = np.zeros(n, dtype=complex)
    expected[3] = n / 2
    expected[n - 3] = n / 2
    assert_allclose(spectrum, expected, atol=1e-12)


def test_fft_of_sine_on_non_power_of_two_length():
    n = 12
    signal = np.sin(2 * np.pi * 2 * np.arange(n) / n)
    spectrum = np.fft.fft(signal)
    expected = np.zeros(n, dtype=complex)
    expected[2] = -6j
    expected[n - 2] = 6j
    assert_allclose(spectrum, expected, atol=1e-12)


def test_fft_small_known_values():
    assert_allclose(np.fft.fft([1, 2, 3, 4]), [10, -2 + 2j, -2, -2 - 2j], atol=1e-12)
    assert_allclose(
        np.fft.fft([1.0, 2.0, 3.0]),
        [6.0, -1.5 + 0.8660254037844386j, -1.5 - 0.8660254037844386j],
        atol=1e-12,
    )


def test_fft_returns_complex128_for_real_and_int_input():
    assert np.fft.fft(np.arange(4)).dtype == np.complex128
    assert np.fft.fft(np.arange(4.0)).dtype == np.complex128
    assert np.fft.ifft(np.arange(5.0)).dtype == np.complex128


def test_fft_accepts_python_lists():
    result = np.fft.fft([0.0, 1.0, 0.0, -1.0])
    assert isinstance(result, np.ndarray)
    assert_allclose(result, [0, -2j, 0, 2j], atol=1e-12)


def test_ifft_normalizes_by_length():
    assert_allclose(np.fft.ifft(np.ones(5)), [1, 0, 0, 0, 0], atol=1e-12)


def test_fft_n_pads_and_truncates():
    padded = np.fft.fft([1.0, 2.0, 3.0], n=5)
    assert padded.shape == (5,)
    assert_allclose(padded, np.fft.fft([1.0, 2.0, 3.0, 0.0, 0.0]), atol=1e-12)
    truncated = np.fft.fft([1.0, 2.0, 3.0, 4.0], n=2)
    assert_allclose(truncated, [3.0, -1.0], atol=1e-12)


def test_fft_operates_on_last_axis_by_default():
    data = np.arange(6.0).reshape(2, 3)
    result = np.fft.fft(data)
    assert result.shape == (2, 3)
    assert_allclose(result[0], np.fft.fft([0.0, 1.0, 2.0]), atol=1e-12)
    assert_allclose(result[1], np.fft.fft([3.0, 4.0, 5.0]), atol=1e-12)


def test_fft_axis_zero():
    data = np.arange(6.0).reshape(2, 3)
    result = np.fft.fft(data, axis=0)
    assert_allclose(result, [[3, 5, 7], [-3, -3, -3]], atol=1e-12)


def test_parseval_identity():
    signal = np.sin(np.arange(10) * 1.3) + 0.5
    spectrum = np.fft.fft(signal)
    assert_allclose(np.sum(np.abs(spectrum) ** 2) / 10, np.sum(signal**2), rtol=1e-12)


def test_rfft_length_and_values():
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = np.fft.rfft(signal)
    assert result.shape == (4,)
    assert result.dtype == np.complex128
    assert_allclose(result, [21, -3 + 5.196152422706632j, -3 + 1.7320508075688772j, -3], atol=1e-12)
    assert_allclose(result, np.fft.fft(signal)[:4], atol=1e-12)


@pytest.mark.parametrize("n, bins", [(1, 1), (2, 2), (5, 3), (6, 4), (7, 4), (8, 5)])
def test_rfft_output_length(n, bins):
    assert np.fft.rfft(np.ones(n)).shape == (bins,)


@pytest.mark.parametrize("n", [4, 6, 7, 8, 12])
def test_irfft_round_trip_with_n(n):
    signal = np.cos(np.arange(n) * 0.9) - 0.25 * np.arange(n)
    restored = np.fft.irfft(np.fft.rfft(signal), n=n)
    assert restored.shape == (n,)
    assert restored.dtype == np.float64
    assert_allclose(restored, signal, atol=1e-12)


def test_irfft_default_length_is_even():
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    restored = np.fft.irfft(np.fft.rfft(signal))
    assert restored.shape == (4,)


def test_irfft_of_dc_bin():
    assert_allclose(np.fft.irfft([4.0, 0.0, 0.0]), [1.0, 1.0, 1.0, 1.0], atol=1e-12)


def test_fftfreq_even_and_odd():
    assert_array_equal(np.fft.fftfreq(8, d=0.25), [0.0, 0.5, 1.0, 1.5, -2.0, -1.5, -1.0, -0.5])
    assert_array_equal(np.fft.fftfreq(5), [0.0, 0.2, 0.4, -0.4, -0.2])
    assert np.fft.fftfreq(4).tolist() == [0.0, 0.25, -0.5, -0.25]


def test_fftfreq_scaling_matches_reciprocal_product():
    assert np.fft.fftfreq(6).tolist() == [
        0.0,
        0.16666666666666666,
        0.3333333333333333,
        -0.5,
        -0.3333333333333333,
        -0.16666666666666666,
    ]
    assert np.fft.fftfreq(7, d=0.5).tolist() == [
        0.0,
        0.2857142857142857,
        0.5714285714285714,
        0.8571428571428571,
        -0.8571428571428571,
        -0.5714285714285714,
        -0.2857142857142857,
    ]


def test_rfftfreq_values():
    assert_array_equal(np.fft.rfftfreq(8), [0.0, 0.125, 0.25, 0.375, 0.5])
    assert np.fft.rfftfreq(7, d=0.1).tolist() == [
        0.0,
        1.4285714285714284,
        2.8571428571428568,
        4.285714285714285,
    ]
    assert np.fft.rfftfreq(8).dtype == np.float64


def test_fftfreq_locates_cosine_peak():
    n = 12
    rate = 24.0
    signal = np.cos(2 * np.pi * 4.0 * np.arange(n) / rate)
    freqs = np.fft.rfftfreq(n, d=1.0 / rate)
    peak = np.argmax(np.abs(np.fft.rfft(signal)))
    assert freqs[peak] == 4.0


def test_fftfreq_and_rfftfreq_reject_zero_length():
    with pytest.raises(ZeroDivisionError):
        np.fft.fftfreq(0)
    with pytest.raises(ZeroDivisionError):
        np.fft.rfftfreq(0)


# 13 and 17 use pocketfft's generic prime pass; 101 and 202 use Bluestein's algorithm.
@pytest.mark.parametrize("n", [13, 17, 101, 202])
def test_generic_radix_and_bluestein_lengths_match_direct_dft(n):
    signal = np.cos(np.arange(n) * 0.37) + 1j * np.sin(np.arange(n) * 0.11)
    k = np.arange(n)
    kernel = np.exp(-2j * np.pi * np.outer(k, k) / n)
    assert_allclose(np.fft.fft(signal), kernel @ signal, atol=1e-9)
    assert_allclose(np.fft.irfft(np.fft.rfft(signal.real), n), signal.real, atol=1e-12)


@pytest.mark.parametrize("norm", [None, "backward", "ortho", "forward"])
def test_norm_modes_round_trip(norm):
    signal = np.arange(6.0) - 1.5j
    assert_allclose(np.fft.ifft(np.fft.fft(signal, norm=norm), norm=norm), signal, atol=1e-12)


def test_norm_modes_scale_forward_and_inverse():
    ones = np.ones(4)
    assert_allclose(np.fft.fft(ones, norm="ortho"), [2, 0, 0, 0], atol=1e-12)
    assert_allclose(np.fft.fft(ones, norm="forward"), [1, 0, 0, 0], atol=1e-12)
    assert_allclose(np.fft.ifft(ones, norm="forward"), [4, 0, 0, 0], atol=1e-12)
    assert_allclose(np.fft.rfft(ones, norm="ortho"), [2, 0, 0], atol=1e-12)


def test_single_precision_input_gives_single_precision_output():
    x32 = np.arange(5, dtype=np.float32)
    assert np.fft.fft(x32).dtype == np.complex64
    assert np.fft.rfft(x32).dtype == np.complex64
    assert np.fft.irfft(np.fft.rfft(x32)).dtype == np.float32
    assert np.fft.ifft(x32.astype(np.complex64)).dtype == np.complex64
    assert np.fft.fft(np.arange(4, dtype=np.float16)).dtype == np.complex64


def test_single_precision_input_stays_within_float32_precision():
    # NumPy dispatches this to internal float32-precision loops (a normalized transform passes
    # a float32 scale, which selects the float loop even for the unnormalized forward
    # transform), so the exact last bit is an implementation detail; compare within float32
    # precision (~1.2e-7 relative) rather than bit for bit.
    x = (np.arange(7, dtype=np.float32) * 0.3) ** 2
    assert_allclose(
        np.fft.ifft(x),
        [
            (1.1700000762939453 + 0j),
            (-0.07596264779567719 - 0.6541042923927307j),
            (-0.24138164520263672 - 0.25120416283607483j),
            (-0.26765576004981995 - 0.07189671695232391j),
            (-0.26765576004981995 + 0.07189671695232391j),
            (-0.24138164520263672 + 0.25120416283607483j),
            (-0.07596264779567719 + 0.6541042923927307j),
        ],
        rtol=1e-6,
        atol=1e-6,
    )
    assert_allclose(
        np.fft.fft(x),
        [
            (8.190000534057617 + 0j),
            (-0.5317385196685791 + 4.578729629516602j),
            (-1.6896713972091675 + 1.7584290504455566j),
            (-1.873590350151062 + 0.5032769441604614j),
            (-1.873590350151062 - 0.5032769441604614j),
            (-1.6896713972091675 - 1.7584290504455566j),
            (-0.5317385196685791 - 4.578729629516602j),
        ],
        rtol=1e-6,
        atol=1e-6,
    )


def test_multidimensional_transforms():
    a = np.arange(24.0).reshape(2, 3, 4) ** 1.5
    nested = np.fft.fft(np.fft.fft(np.fft.fft(a, axis=2), axis=1), axis=0)
    assert_allclose(np.fft.fftn(a), nested, atol=1e-9)
    assert_allclose(np.fft.ifftn(np.fft.fftn(a)), a, atol=1e-12)
    assert_allclose(np.fft.ifft2(np.fft.fft2(a)), a, atol=1e-12)
    assert np.fft.fft2(a, s=(5, 3)).shape == (2, 5, 3)
    with pytest.raises(ValueError):
        np.fft.fftn(a, s=(2, 2), axes=(0,))


def test_fftshift_and_ifftshift():
    assert np.fft.fftshift(np.arange(5)).tolist() == [3, 4, 0, 1, 2]
    assert np.fft.ifftshift(np.fft.fftshift(np.arange(5))).tolist() == [0, 1, 2, 3, 4]
    grid = np.arange(6).reshape(2, 3)
    assert np.fft.fftshift(grid).tolist() == [[5, 3, 4], [2, 0, 1]]
    assert np.fft.fftshift(grid, axes=1).tolist() == [[2, 0, 1], [5, 3, 4]]
    assert np.fft.fftshift(np.fft.fftfreq(4)).tolist() == [-0.5, -0.25, 0.0, 0.25]


def test_out_argument_receives_the_result():
    out = np.zeros(4, dtype=np.complex64)
    result = np.fft.fft(np.arange(4.0), out=out)
    assert result is out
    assert out.tolist() == [6 + 0j, -2 + 2j, -2 + 0j, -2 - 2j]
    with pytest.raises(ValueError):
        np.fft.fft(np.arange(4.0), out=np.zeros(3, dtype=complex))
    with pytest.raises(TypeError):
        np.fft.fft(np.arange(4.0), out=np.zeros(4))


def test_invalid_arguments_raise_numpy_errors():
    with pytest.raises(ValueError):
        np.fft.fft([1.0, 2.0], n=0)
    with pytest.raises(ValueError):
        np.fft.fft([1.0], norm="bad")
    with pytest.raises(TypeError):
        np.fft.rfft([1j, 2.0])
    with pytest.raises(ValueError):
        np.fft.fftfreq(2.5)


def test_non_finite_values_propagate():
    result = np.fft.fft([np.nan, 1.0, 2.0])
    assert np.isnan(result).all()
    assert np.isinf(np.fft.fft([np.inf, 0.0])).real.all()


def test_axis_out_of_range_raises_index_error():
    with pytest.raises(IndexError):
        np.fft.fft(np.zeros((2, 3)), axis=5)
    with pytest.raises(IndexError):
        np.fft.ifft(np.zeros((2, 3)), axis=-5)


def test_rfft_and_irfft_out_argument_errors():
    with pytest.raises(ValueError):
        np.fft.rfft(np.ones(4), out=np.zeros(2, dtype=complex))
    with pytest.raises(TypeError):
        np.fft.rfft(np.ones(4), out=np.zeros(3))
    with pytest.raises(ValueError):
        np.fft.irfft(np.ones(4, dtype=complex), out=np.zeros(5))
    with pytest.raises(TypeError):
        np.fft.irfft(np.ones(4, dtype=complex), out=np.zeros(6, dtype=np.int32))


def test_shape_and_axes_reject_non_sequences():
    with pytest.raises(TypeError):
        np.fft.fftn(np.zeros((2, 3)), s=4)


def test_multidimensional_dtype_propagation():
    single = np.ones((2, 3), dtype=np.float32)
    assert np.fft.fft2(single).dtype == np.complex64
    assert np.fft.ifft2(np.fft.fft2(single)).dtype == np.complex64
