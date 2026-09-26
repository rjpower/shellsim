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
