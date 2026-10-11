# OpenCV guest profile

The port generates upstream OpenCV 5.0.0 Python bindings over the static core,
imgproc and required Geometry/Flann libraries. NumPy 2.3.5 provides target array headers
and the guest array runtime. The native producer exports OpenCV headers and its
bundled DLPack header; no host OpenCV installation supplies target code.

The first acceptance scope is array resize, flip, threshold, Gaussian blur,
mean and invalid-image error translation. GUI, video capture, image codecs,
accelerators and parallel OpenCV execution are outside this profile. Native
OpenCV compiles with the declared WASI SDK; the binding generator runs with
ordinary worker Python. Neither generation nor compilation runs inside the
simulator. Acceptance installs the published wheel in the virtual filesystem
and executes its real extension in shellsim.

The sdist uses an old PKG-INFO that omits setup.py's NumPy requirements. The build
extracts that literal requirements list from the admitted setup.py into wheel
metadata so the public guest installer resolves NumPy normally.
