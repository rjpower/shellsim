import numpy as np

assert np.fft._pocketfft_umath.__spec__.origin == "built-in"
values = np.array([1.0, 2.0, 3.0, 4.0])
assert np.allclose(np.fft.fft(values), [10, -2 + 2j, -2, -2 - 2j])
assert np.allclose(np.fft.ifft(np.fft.fft(values)).real, values)
assert np.allclose(np.fft.irfft(np.fft.rfft(values)), values)
try:
    np.fft.fft(values, n=0)
except ValueError:
    pass
else:
    raise AssertionError("zero FFT length accepted")
print("FFT round trip and invalid length passed")
