# Portable SciPy semantics. Expectations checked against SciPy 1.18.1 and NumPy 2.5.3 on
# CPython 3.14.4.
# Scope: scipy.interpolate's one-dimensional interpolators: interp1d, PPoly, CubicHermiteSpline,
# CubicSpline and PCHIP. interp1d's linear and step kinds are compared exactly; spline
# constructions solve banded systems whose last bits may differ between implementations, so
# they are compared with a tight tolerance.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import interpolate

X = np.array([0.0, 1.0, 2.5, 4.0, 5.0])
Y = np.array([1.0, 3.0, 2.0, 0.5, 4.0])
# Samples at the data points, between them, and at both ends.
XN = np.array([0.0, 0.3, 1.25, 1.75, 2.5, 3.3, 4.9, 5.0])
# The same curve with its last value repeated from the first, for periodic splines.
YP = np.array([1.0, 3.0, 2.0, 0.5, 1.0])


def close(actual, expected):
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)


def left_limits(spline, nu):
    """The `nu`-th derivative at each interior knot from the piece that ends there."""
    return [
        interpolate.PPoly(spline.c[:, i : i + 1], spline.x[i : i + 2])(spline.x[i + 1], nu)
        for i in range(len(spline.x) - 2)
    ]


def test_interp1d_linear_rounds_like_scipy():
    result = interpolate.interp1d(X, Y)(XN)
    assert result.dtype == np.float64
    expected = [1.0, 1.6, 2.8333333333333335, 2.5, 2.0, 1.2000000000000002, 3.6500000000000012, 4.0]
    assert result.tolist() == expected
    # Extrapolating, or interpolating more than one column, weights the two neighbors by
    # distance instead, which rounds 0.3 * 3 + 0.7 * 1 differently.
    extrapolating = interpolate.interp1d(X, Y, fill_value="extrapolate")
    assert extrapolating(0.3).item() == 1.5999999999999999
    assert extrapolating([-1.0, 6.0]).tolist() == [-1.0, 7.5]
    columns = interpolate.interp1d(X, np.column_stack([Y, 2 * Y]), axis=0)(0.3)
    assert columns.tolist() == [1.5999999999999999, 3.1999999999999997]


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("nearest", [1.0, 1.0, 3.0, 3.0, 2.0, 0.5, 4.0, 4.0]),
        ("nearest-up", [1.0, 1.0, 3.0, 2.0, 2.0, 0.5, 4.0, 4.0]),
        ("previous", [1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 0.5, 4.0]),
        ("next", [1.0, 3.0, 2.0, 2.0, 2.0, 0.5, 4.0, 4.0]),
        ("zero", [1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 0.5, 4.0]),
        (0, [1.0, 1.0, 3.0, 3.0, 2.0, 2.0, 0.5, 4.0]),
    ],
)
def test_interp1d_step_kinds(kind, expected):
    # 1.75 is the midpoint of 1.0 and 2.5: "nearest" rounds the tie down, "nearest-up" up.
    assert interpolate.interp1d(X, Y, kind=kind)(XN).tolist() == expected


@pytest.mark.parametrize(
    "kind, expected",
    [
        (
            "slinear",
            [1.0, 1.5999999999999999, 2.8333333333333335, 2.5, 2.0, 1.2000000000000002,
             3.6500000000000012, 4.0],
        ),
        (
            "quadratic",
            [1.0, 1.85203431372549, 3.1249489379084965, 2.9247855392156863, 2.0,
             0.6351388888888891, 3.4544852941176494, 4.0],
        ),
        (
            "cubic",
            [1.0, 1.9012566666666666, 3.1013454861111103, 2.935390625, 2.0,
             0.7150711111111117, 3.3693200000000023, 4.0],
        ),
        (
            4,
            [1.0, 1.8526953333333331, 3.1246744791666656, 2.9706640624999987, 2.0,
             0.7514586666666674, 3.3445520000000024, 4.0],
        ),
    ],
)
def test_interp1d_spline_kinds(kind, expected):
    close(interpolate.interp1d(X, Y, kind=kind)(XN), expected)


def test_interp1d_cubic_is_the_not_a_knot_spline():
    points = np.linspace(-1.0, 6.0, 15)
    cubic = interpolate.interp1d(X, Y, kind="cubic", fill_value="extrapolate")
    close(cubic(points), interpolate.CubicSpline(X, Y)(points))


def test_interp1d_out_of_range_points():
    linear = interpolate.interp1d(X, Y)
    assert linear.bounds_error is True
    with pytest.raises(ValueError, match=r"A value \(6\.0\) in x_new is above the interpolation "
                       r"range's maximum value \(5\.0\)\."):
        linear([1.0, 6.0])
    with pytest.raises(ValueError, match=r"A value \(-1\.0\) in x_new is below the interpolation "
                       r"range's minimum value \(0\.0\)\."):
        linear(-1.0)
    filled = interpolate.interp1d(X, Y, bounds_error=False)
    assert np.isnan(filled.fill_value) and filled.fill_value.shape == ()
    assert_array_equal(filled([-1.0, 2.5, 6.0]), [np.nan, 2.0, np.nan])
    pair = interpolate.interp1d(X, Y, bounds_error=False, fill_value=(-5, 7))
    assert pair.fill_value == (-5, 7)
    assert pair([-1.0, 6.0]).tolist() == [-5.0, 7.0]
    # A fill value pair alone does not turn off the bounds check.
    with pytest.raises(ValueError, match="above the interpolation range"):
        interpolate.interp1d(X, Y, fill_value=(-5, 7))(6.0)
    with pytest.raises(ValueError, match="Cannot extrapolate and raise at the same time."):
        interpolate.interp1d(X, Y, bounds_error=True, fill_value="extrapolate")


def test_interp1d_extrapolates_step_kinds_one_sided():
    outside = [-1.0, 6.0]
    extrapolate = {"fill_value": "extrapolate"}
    assert_array_equal(interpolate.interp1d(X, Y, kind="previous", **extrapolate)(outside), [np.nan, 4.0])
    assert_array_equal(interpolate.interp1d(X, Y, kind="next", **extrapolate)(outside), [1.0, np.nan])
    assert interpolate.interp1d(X, Y, kind="nearest", **extrapolate)(outside).tolist() == [1.0, 4.0]
    assert interpolate.interp1d(X, Y, kind="zero", **extrapolate)(outside).tolist() == [1.0, 4.0]
    close(interpolate.interp1d(X, Y, kind="cubic", **extrapolate)(outside),
          [-4.303888888888886, 14.946111111111113])


def test_interp1d_fill_values_broadcast_over_columns():
    columns = np.array([[1.0, 2.0], [3.0, 4.0]])
    f = interpolate.interp1d([0.0, 1.0], columns, axis=0, bounds_error=False, fill_value=([1.0, 2.0], 7))
    assert f([-1.0, 0.5, 2.0]).tolist() == [[1.0, 2.0], [2.0, 3.0], [7.0, 7.0]]
    with pytest.raises(ValueError, match=r"fill_value \(below\) argument must be able to broadcast "
                       r"up to shape \(2,\) but had shape \(3,\)"):
        interpolate.interp1d([0.0, 1.0], columns, axis=0, bounds_error=False,
                             fill_value=([1.0, 2.0, 3.0], 7))
    with pytest.raises(ValueError, match=r"fill_value argument must be able to broadcast up to "
                       r"shape \(1,\) but had shape \(2,\)"):
        interpolate.interp1d([0.0, 1.0], [1.0, 2.0], bounds_error=False, fill_value=[1.0, 2.0])


def test_interp1d_shapes_and_dtypes():
    f = interpolate.interp1d(X, Y)
    scalar = f(2.5)
    assert isinstance(scalar, np.ndarray) and scalar.shape == () and scalar == 2.0
    assert f([]).shape == (0,)
    # The points' shape replaces the interpolation axis of y.
    stacked = np.vstack([Y, 2 * Y, 3 * Y])
    along_last = interpolate.interp1d(X, stacked)
    assert along_last.axis == 1 and along_last.y.shape == (3, 5)
    assert along_last([[0.5, 1.0]]).shape == (3, 1, 2)
    assert interpolate.interp1d(X, stacked.T, axis=0)([[0.5, 1.0]]).shape == (1, 2, 3)
    # Integer and boolean samples interpolate as floats; complex samples stay complex.
    assert interpolate.interp1d(X, [1, 2, 3, 4, 5]).y.dtype == np.float64
    assert interpolate.interp1d(X, [True, False, True, True, False])(1.3) == pytest.approx(0.2)
    assert interpolate.interp1d(X, Y + 1j * Y)(1.3) == pytest.approx(2.8 + 2.8j)
    assert np.isnan(interpolate.interp1d(X, Y, kind="cubic")(np.nan))


def test_interp1d_sorts_unless_told_the_points_are_sorted():
    f = interpolate.interp1d(X[::-1], Y[::-1])
    assert f.x.tolist() == X.tolist() and f.y.tolist() == Y.tolist()
    assert f(1.25).item() == 2.8333333333333335
    trusted = interpolate.interp1d([3.0, 1.0, 2.0, 0.0], [1.0, 2.0, 3.0, 4.0], assume_sorted=True)
    with pytest.raises(ValueError, match=r"below the interpolation range's minimum value \(3\.0\)"):
        trusted(1.5)


def test_interp1d_repeated_abscissae():
    x, y = [0.0, 1.0, 1.0, 2.0], [0.0, 1.0, 5.0, 3.0]
    # Linear reads the last sample at a repeated point, nearest and previous/next pick by
    # position, and the spline kinds refuse.
    assert interpolate.interp1d(x, y)([0.5, 1.0, 1.5]).tolist() == [0.5, 5.0, 4.0]
    assert interpolate.interp1d(x, y, kind="nearest")([0.9, 1.0, 1.1]).tolist() == [1.0, 1.0, 5.0]
    assert interpolate.interp1d(x, y, kind="previous")([0.9, 1.0, 1.1]).tolist() == [0.0, 5.0, 5.0]
    assert interpolate.interp1d(x, y, kind="next")([0.9, 1.0, 1.1]).tolist() == [1.0, 1.0, 3.0]
    for kind in ("zero", "slinear", "cubic"):
        with pytest.raises(ValueError, match="Expect x to not have duplicates"):
            interpolate.interp1d(x, y, kind=kind)


def test_interp1d_rejects_invalid_arguments():
    with pytest.raises(NotImplementedError, match="bogus is unsupported: Use fitpack routines for other types."):
        interpolate.interp1d(X, Y, kind="bogus")
    with pytest.raises(ValueError, match="Expect non-negative k."):
        interpolate.interp1d(X, Y, kind=-1)
    with pytest.raises(ValueError, match="x and y arrays must be equal in length along interpolation axis."):
        interpolate.interp1d(X, Y[:4])
    with pytest.raises(ValueError, match="x and y arrays must have at least 2 entries"):
        interpolate.interp1d([0.0], [1.0], kind="slinear")
    with pytest.raises(ValueError, match="The number of derivatives at boundaries does not match: "
                       "expected 1, got 0\\+0"):
        interpolate.interp1d([0.0, 1.0, 2.0], [0.0, 1.0, 3.0], kind="cubic")
    with pytest.raises(ValueError, match="expected 3, got 0\\+0"):
        interpolate.interp1d([0.0, 1.0, 2.0], [0.0, 1.0, 3.0], kind=5)
    # One sample is enough for the kinds that only look up neighbors.
    assert interpolate.interp1d([1.0], [2.0])(1.0) == 2.0
    assert interpolate.interp1d([1.0], [2.0], kind="nearest")(1.0) == 2.0


@pytest.mark.parametrize(
    "bc_type, expected",
    [
        (
            "not-a-knot",
            [[0.16722222222222172, 0.16722222222222236, 0.5494444444444445, 0.5494444444444442],
             [-1.6519444444444429, -1.1502777777777782, -0.39777777777777795, 2.0747222222222224],
             [3.484722222222221, 0.6825000000000002, -1.6395833333333332, 0.8758333333333332],
             [1.0, 3.0, 2.0, 0.5]],
        ),
        (
            "natural",
            [[-0.4813725490196079, 0.20544662309368178, 0.7501089324618736, -0.9519607843137248],
             [0.0, -1.444117647058823, -0.5196078431372548, 2.8558823529411757],
             [2.481372549019608, 1.0372549019607842, -1.9083333333333334, 1.5960784313725491],
             [1.0, 3.0, 2.0, 0.5]],
        ),
        (
            "clamped",
            [[-2.102777777777778, 0.3308641975308641, 1.1382716049382717, -3.9527777777777775],
             [4.102777777777778, -2.2055555555555553, -0.7166666666666667, 4.405555555555555],
             [2.9605947323337506e-16, 1.8972222222222221, -2.486111111111111, 3.0472222222222225],
             [1.0, 3.0, 2.0, 0.5]],
        ),
        (
            ((1, 2.0), (2, -1.0)),
            [[-0.8052805280528057, 0.24642464246424645, 0.7671433810047673, -1.1501650165016502],
             [0.8052805280528064, -1.6105610561056105, -0.5016501650165021, 2.9504950495049505],
             [1.9999999999999993, 1.1947194719471947, -1.9735973597359735, 1.6996699669967],
             [1.0, 3.0, 2.0, 0.5]],
        ),
    ],
)
def test_cubic_spline_boundary_conditions(bc_type, expected):
    spline = interpolate.CubicSpline(X, Y, bc_type=bc_type)
    assert spline.c.shape == (4, 4) and spline.x.tolist() == X.tolist()
    assert spline.axis == 0 and spline.extrapolate is True
    close(spline.c, expected)
    # Every condition interpolates, and the pieces join with matching first two derivatives.
    close(spline(X), Y)
    for nu in (0, 1, 2):
        assert_allclose(left_limits(spline, nu), spline(X[1:-1], nu), rtol=1e-12, atol=1e-12)


def test_cubic_spline_evaluation_derivatives_and_integrals():
    spline = interpolate.CubicSpline(X, Y)
    close(spline([-1.0, 0.5, 3.3, 6.0]), [-4.303888888888886, 2.3502777777777775, 0.7150711111111114, 14.94611111111111])
    close(spline([0.5, 3.3], 1), [1.9581944444444446, -1.2210944444444447])
    close(spline([0.5, 3.3], 2), [-2.8022222222222206, 1.8417777777777773])
    close(spline([0.5, 3.3], 3), [1.0033333333333303, 3.296666666666667])
    assert spline([0.5, 3.3], 4).tolist() == [0.0, 0.0]
    assert_array_equal(spline([-1.0, 2.5, 6.0], extrapolate=False), [np.nan, 2.0, np.nan])
    close(spline.integrate(0.5, 4.5), 7.407407407407406)
    close(spline.integrate(4.5, 0.5), -7.407407407407406)
    close(spline.integrate(-1, 6), 16.96949074074074)
    assert np.isnan(spline.integrate(-1, 6, extrapolate=False))
    scalar = spline(1.3)
    assert isinstance(scalar, np.ndarray) and scalar.shape == ()
    with pytest.raises(ValueError, match="Order of derivative cannot be negative"):
        spline(1.0, -1)


def test_cubic_spline_derivative_and_antiderivative_objects():
    spline = interpolate.CubicSpline(X, Y)
    derivative = spline.derivative()
    antiderivative = spline.antiderivative()
    assert type(derivative) is interpolate.CubicSpline and derivative.c.shape == (3, 4)
    assert type(antiderivative) is interpolate.CubicSpline and antiderivative.c.shape == (5, 4)
    points = np.linspace(-0.5, 5.5, 13)
    close(derivative(points), spline(points, 1))
    close(antiderivative(points, 1), spline(points))
    close(antiderivative(4.5) - antiderivative(0.5), spline.integrate(0.5, 4.5))
    assert antiderivative(0.0) == 0.0
    assert spline.derivative(4).c.tolist() == [[0.0, 0.0, 0.0, 0.0]]
    close(spline.antiderivative(2).derivative(2).c, spline.c)


def test_periodic_cubic_spline():
    spline = interpolate.CubicSpline(X, YP, bc_type="periodic")
    assert spline.extrapolate == "periodic"
    close(spline.c, [[-1.1416666666666666, 0.4740740740740739, 0.08148148148148139, 0.30833333333333335],
                     [1.4749999999999999, -1.9499999999999997, 0.18333333333333357, 0.55],
                     [1.6666666666666667, 1.1916666666666667, -1.4583333333333335, -0.35833333333333334],
                     [1.0, 3.0, 2.0, 0.5]])
    # Both ends agree in value, slope and curvature, and evaluation wraps around.
    for nu in (0, 1, 2):
        assert_allclose(spline(5.0, nu, extrapolate=True), spline(0.0, nu), rtol=1e-12, atol=1e-12)
    close(spline([-7.3, 12.1]), [1.7163185185185184, 2.5823259259259266])
    close(spline([-7.3, 12.1]), spline([2.7, 2.1]))
    close(spline.integrate(-3, 11), 22.545023148148147)
    # Integrating turns off the wrap-around, since the antiderivative is not periodic.
    assert spline.antiderivative().extrapolate is False
    assert spline.derivative().extrapolate == "periodic"
    # Equal to within machine precision is equal enough.
    interpolate.CubicSpline(X, YP + np.r_[0, 0, 0, 0, 1e-15], bc_type="periodic")
    with pytest.raises(ValueError, match="The first and last `y` point along axis 0 must be identical"):
        interpolate.CubicSpline(X, Y, bc_type="periodic")


def test_cubic_spline_with_few_points():
    # Two points give the straight line; three give the parabola through them.
    close(interpolate.CubicSpline([0.0, 1.0], [1.0, 3.0]).c, [[0.0], [0.0], [2.0], [1.0]])
    parabola = interpolate.CubicSpline([0.0, 1.0, 2.5], [1.0, 3.0, 2.0])
    points = np.linspace(-1.0, 3.0, 9)
    close(parabola(points), np.polyval(np.polyfit([0.0, 1.0, 2.5], [1.0, 3.0, 2.0], 2), points))
    close(interpolate.CubicSpline([0.0, 1.0], [1.0, 3.0], bc_type="clamped").c, [[-4.0], [6.0], [0.0], [1.0]])
    close(interpolate.CubicSpline([0.0, 1.0], [1.0, 3.0], bc_type=("not-a-knot", (1, 3.0))).c,
          [[1.0], [-1.0], [2.0], [1.0]])
    close(interpolate.CubicSpline([0.0, 1.0, 2.5], [1.0, 3.0, 2.0], bc_type=("not-a-knot", "natural")).c,
          [[0.2666666666666657, 0.2666666666666667], [-1.9999999999999982, -1.2000000000000002],
           [3.7333333333333325, 0.5333333333333337], [1.0, 3.0]])
    close(interpolate.CubicSpline([0.0, 1.0, 2.5], [1.0, 3.0, 1.0], bc_type="periodic").c,
          [[-2.6666666666666665, 1.7777777777777777], [4.0, -4.0],
           [0.6666666666666667, 0.6666666666666667], [1.0, 3.0]])


def test_cubic_spline_axes_and_dtypes():
    columns = np.column_stack([Y, 2 * Y, Y**2])
    spline = interpolate.CubicSpline(X, columns)
    assert spline.c.shape == (4, 4, 3)
    assert spline([[0.5, 1.0]]).shape == (1, 2, 3)
    rows = interpolate.CubicSpline(X, columns.T, axis=1)
    assert rows.axis == 1 and rows.c.shape == (4, 4, 3)
    assert rows([[0.5, 1.0]]).shape == (3, 1, 2)
    close(rows(1.3), spline(1.3))
    close(spline(1.3)[1], 2 * spline(1.3)[0])
    assert interpolate.CubicSpline(X, [1, 2, 3, 4, 5]).c.dtype == np.float64
    complex_spline = interpolate.CubicSpline(X, Y + 1j * Y[::-1])
    close(complex_spline(1.3), interpolate.CubicSpline(X, Y)(1.3) + 1j * interpolate.CubicSpline(X, Y[::-1])(1.3))
    # A per-column derivative condition has the shape of one sample.
    clamped = interpolate.CubicSpline(X, columns, bc_type=((1, [2.0, 3.0, 4.0]), "natural"))
    close(clamped(0.0, 1), [2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    "x, y, kwargs, message",
    [
        (X[::-1], Y, {}, "`x` must be strictly increasing sequence."),
        ([0.0, 1.0, 1.0, 2.0], [0, 1, 2, 3], {}, "`x` must be strictly increasing sequence."),
        (X, Y[:4], {}, "The length of `y` along `axis`=0 doesn't match the length of `x`"),
        ([0.0], [1.0], {}, "`x` must contain at least 2 elements."),
        ([[0.0, 1.0]], [[0.0, 1.0]], {}, "`x` must be 1-dimensional."),
        ([0.0, 1.0, np.inf], [1.0, 2.0, 3.0], {}, "`x` must contain only finite values."),
        (X, [1.0, 2.0, np.nan, 4.0, 5.0], {}, "`y` must contain only finite values."),
        (X, Y, {"bc_type": "bogus"}, "bc_type=bogus is not allowed."),
        (X, Y, {"bc_type": ("natural",)}, "`bc_type` must contain 2 elements to specify start and end conditions."),
        (X, Y, {"bc_type": ("periodic", "natural")}, "'periodic' `bc_type` is defined for both curve ends"),
        (X, Y, {"bc_type": ((3, 1.0), (1, 0.0))}, "The specified derivative order must be 1 or 2."),
        (X, Y, {"bc_type": ((1, 2.0, 3), "natural")}, r"A specified derivative value must be given in the form \(order, value\)."),
        (X, Y, {"bc_type": ((1, [2.0, 3.0]), "natural")}, r"`deriv_value` shape \(2,\) is not the expected one \(\)."),
    ],
)
def test_cubic_spline_rejects_invalid_input(x, y, kwargs, message):
    with pytest.raises(ValueError, match=message):
        interpolate.CubicSpline(x, y, **kwargs)


def test_cubic_hermite_spline():
    hermite = interpolate.CubicHermiteSpline(X, Y, 2 * Y)
    assert isinstance(hermite, interpolate.PPoly)
    close(hermite.c, [[4.0, 5.037037037037037, 3.111111111111111, 2.0],
                      [-4.0, -12.0, -8.0, 0.5], [2.0, 6.0, 4.0, 1.0], [1.0, 3.0, 2.0, 0.5]])
    close(hermite(X), Y)
    close(hermite(X, 1), 2 * Y)
    with pytest.raises(ValueError, match="The shapes of `y` and `dydx` must be identical."):
        interpolate.CubicHermiteSpline(X, Y, Y[:4])


def test_pchip_preserves_shape():
    pchip = interpolate.PchipInterpolator(X, Y)
    assert type(pchip).__mro__[1:3] == (interpolate.CubicHermiteSpline, interpolate.PPoly)
    assert interpolate.pchip is interpolate.PchipInterpolator
    close(pchip.c, [[-0.9333333333333331, 0.23703703703703696, 0.5333333333333333, -1.7000000000000002],
                    [-0.13333333333333375, -0.7999999999999998, -0.9333333333333332, 5.2],
                    [3.066666666666667, 0.0, -0.8, 0.0], [1.0, 3.0, 2.0, 0.5]])
    # Slopes vanish where the data turn, and monotone data give a monotone interpolant.
    rising = np.array([0.0, 1.0, 5.0, 5.5, 6.0])
    close(interpolate.PchipInterpolator(X, rising).c[2],
          [0.33333333333333337, 1.4117647058823528, 0.5925925925925926, 0.4054054054054054])
    fine = np.linspace(0.0, 5.0, 101)
    assert np.all(np.diff(interpolate.PchipInterpolator(X, rising)(fine)) >= 0)
    close(interpolate.PchipInterpolator(X, [1.0, 2.0, 2.0, 3.0, 5.0]).c[2], [1.4, 0.0, 0.0, 1.0344827586206897])
    # The end slope is zeroed when the three-point estimate points the wrong way.
    close(interpolate.PchipInterpolator(X, [5.0, 1.0, 0.9, 5.5, 6.0]).c[2],
          [-5.573333333333333, -0.14018691588785043, 0.0, 0.8204518430439951])
    assert_array_equal(interpolate.PchipInterpolator(X, Y, extrapolate=False)([-1.0, 6.0]), [np.nan, np.nan])
    close(interpolate.PchipInterpolator([0.0, 1.0], [1.0, 3.0]).c, [[0.0], [0.0], [2.0], [1.0]])
    with pytest.raises(ValueError, match="`PchipInterpolator` only works with real values for `y`."):
        interpolate.PchipInterpolator(X, Y + 1j)


def test_pchip_interpolate():
    close(interpolate.pchip_interpolate(X, Y, [0.5, 3.3]), [2.3833333333333333, 1.0357333333333336])
    values, slopes, curvatures = interpolate.pchip_interpolate(X, Y, [1.3], der=[0, 1, 2])
    close(values, [2.9344])
    close(slopes, [-0.416])
    close(curvatures, [-1.173333333333333])
    scalar = interpolate.pchip_interpolate(X, Y, 1.3)
    assert isinstance(scalar, np.ndarray) and scalar.shape == ()


def test_ppoly_evaluation():
    poly = interpolate.PPoly([[1.0, 2.0], [3.0, 4.0]], [0, 1, 3])
    assert poly.c.dtype == np.float64 and poly.x.dtype == np.float64
    # Each piece is a polynomial in the offset from its own left breakpoint.
    assert poly([0.0, 0.5, 1.0, 2.0, 3.0, 4.0]).tolist() == [3.0, 3.5, 4.0, 6.0, 8.0, 10.0]
    assert np.isnan(poly(np.nan))
    assert_array_equal(poly([-1.0, 3.0, 4.0], extrapolate=False), [np.nan, 8.0, np.nan])
    periodic = interpolate.PPoly([[1.0, 2.0], [3.0, 4.0]], [0, 1, 3], extrapolate="periodic")
    assert periodic([-1.0, 4.0, 3.0, 6.0]).tolist() == [6.0, 4.0, 3.0, 3.0]
    assert periodic.integrate(-1, 4) == 26.0
    assert poly.integrate(0, 3) == 15.5
    assert np.isnan(interpolate.PPoly([[1.0, 2.0], [3.0, 4.0]], [0, 1, 3], extrapolate=False).integrate(-1, 4))
    descending = interpolate.PPoly(np.ones((2, 3)), [3, 2, 1, 0])
    assert descending([2.5, 0.5, -1.0, 4.0]).tolist() == [0.5, 0.5, -1.0, 2.0]
    assert descending.integrate(0.5, 2.5) == 1.0


def test_ppoly_calculus():
    poly = interpolate.PPoly([[1.0, 2.0], [3.0, 4.0]], [0, 1, 3])
    assert poly.antiderivative().c.tolist() == [[0.5, 1.0], [3.0, 4.0], [0.0, 3.5]]
    assert poly.derivative(-1).c.tolist() == poly.antiderivative().c.tolist()
    close(poly.antiderivative(2).c, [[1 / 6, 1 / 3], [1.5, 2.0], [0.0, 3.5], [0.0, 5 / 3]])
    assert poly.derivative().c.tolist() == [[1.0, 2.0]]
    assert poly.derivative(0).c.tolist() == poly.c.tolist()
    assert poly.antiderivative(-1).c.tolist() == [[1.0, 2.0]]


def test_ppoly_axes():
    assert interpolate.PPoly(np.ones((2, 3)), [0, 1, 2, 3])([[0.5, 1.5]]).shape == (1, 2)
    assert interpolate.PPoly(np.ones((2, 3, 4)), [0, 1, 2, 3])([0.5, 1.5]).shape == (2, 4)
    # With axis=1 the degree and interval axes are c's second and third, and the points land
    # at axis 1 of the values.
    moved = interpolate.PPoly(np.arange(24.0).reshape(2, 4, 3), [0, 1, 2, 3], axis=1)
    assert moved.c.shape == (4, 3, 2) and moved.axis == 1
    assert moved(0.5).tolist() == [12.75, 35.25]
    assert interpolate.PPoly(np.ones((3, 2, 4, 2)), [0, 1, 2, 3, 4], axis=1)([0.5, 1.5]).shape == (3, 2, 2)


@pytest.mark.parametrize(
    "c, x, kwargs, message",
    [
        (np.ones((2, 3)), [0, 1, 2], {}, "number of coefficients != len\\(x\\)-1"),
        (np.ones((2, 3)), [0, 2, 1, 3], {}, "`x` must be strictly increasing or decreasing."),
        (np.ones((2, 3)), [0, 1, 2, np.nan], {}, "`x` must be strictly increasing or decreasing."),
        (np.ones(2), [0, 1], {}, "Coefficients array must be at least 2-dimensional."),
        (np.ones((0, 3)), [0, 1, 2, 3], {}, "polynomial must be at least of order 0"),
        (np.ones((2, 2)), [0, 1, 3], {"axis": 1}, "axis=1 must be between 0 and 1"),
    ],
)
def test_ppoly_rejects_invalid_input(c, x, kwargs, message):
    with pytest.raises(ValueError, match=message):
        interpolate.PPoly(c, x, **kwargs)
