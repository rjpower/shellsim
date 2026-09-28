"""shellsim's ``scipy.optimize``: local minimization, scalar root finding and curve fitting.

Implemented: ``minimize`` (Nelder-Mead simplex search, and BFGS with a finite-difference
gradient by default), ``minimize_scalar`` (Brent's method, or golden-section-style bounded
search when ``bounds`` is given), the scalar root finders ``root_scalar``/``brentq``/``bisect``/
``newton``, and ``curve_fit`` (Levenberg-Marquardt with a finite-difference Jacobian). Other
methods, bounds and constraints on ``minimize``, and ``least_squares``/``fsolve``/``root``, are
not provided; results agree with SciPy within the requested tolerance, not iteration for
iteration.
"""

import math

import numpy as np

__all__ = [
    "OptimizeResult",
    "bisect",
    "brentq",
    "curve_fit",
    "minimize",
    "minimize_scalar",
    "newton",
    "root_scalar",
]


class OptimizeResult(dict):
    """A ``dict`` whose keys are also readable and writable as attributes, as SciPy's is."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if not self:
            return f"{type(self).__name__}()"
        width = max(len(key) for key in self)
        return "\n".join(f"{key.rjust(width)}: {value!r}" for key, value in sorted(self.items()))

    def __dir__(self):
        return list(self.keys())


def _signed(magnitude, sign):
    """`magnitude` with the sign of `sign` (``math.copysign``, which shellsim's `math` omits)."""
    return -magnitude if sign < 0 else magnitude


def _status_message(converged):
    if converged:
        return 0, "Optimization terminated successfully."
    return 1, "Maximum number of iterations has been exceeded."


# -------------------------------------------------------------------------------------------
# minimize: Nelder-Mead and BFGS
# -------------------------------------------------------------------------------------------


def _finite_diff_gradient(f, x, f0):
    """The forward-difference gradient of the scalar function `f` at `x`, one extra call of
    `f` per coordinate of `x`."""
    steps = np.sqrt(np.finfo(float).eps) * np.maximum(np.abs(x), 1.0)
    grad = np.empty_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = steps[i]
        grad[i] = (f(x + step) - f0) / steps[i]
    return grad


def _nelder_mead(f, x0, maxiter, xatol, fatol):
    """The Nelder-Mead (1965) downhill simplex search for a local minimum of `f`."""
    n = x0.size
    simplex = np.tile(x0, (n + 1, 1))
    for i in range(n):
        simplex[i + 1, i] += 0.05 * x0[i] if x0[i] != 0 else 0.00025
    values = np.array([f(point) for point in simplex])
    nfev, nit, converged = n + 1, 0, False
    while nit < maxiter:
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]
        if (
            np.max(np.abs(simplex[1:] - simplex[0])) <= xatol
            and np.max(np.abs(values[1:] - values[0])) <= fatol
        ):
            converged = True
            break
        nit += 1
        centroid = simplex[:-1].mean(axis=0)
        reflected = centroid + (centroid - simplex[-1])
        f_reflected = f(reflected)
        nfev += 1
        if values[0] <= f_reflected < values[-2]:
            simplex[-1], values[-1] = reflected, f_reflected
            continue
        if f_reflected < values[0]:
            expanded = centroid + 2.0 * (reflected - centroid)
            f_expanded = f(expanded)
            nfev += 1
            simplex[-1], values[-1] = (
                (expanded, f_expanded) if f_expanded < f_reflected else (reflected, f_reflected)
            )
            continue
        shrink_target = reflected if f_reflected < values[-1] else simplex[-1]
        contracted = centroid + 0.5 * (shrink_target - centroid)
        f_contracted = f(contracted)
        nfev += 1
        if f_contracted < min(f_reflected, values[-1]):
            simplex[-1], values[-1] = contracted, f_contracted
            continue
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
            values[i] = f(simplex[i])
        nfev += n
    order = np.argsort(values)
    simplex, values = simplex[order], values[order]
    return simplex[0], values[0], converged, nit, nfev


def _line_search(f, x, direction, f0, d0, max_backtracks=60):
    """The largest ``step = 2**-k`` giving an Armijo sufficient decrease from `f0` along the
    descent direction whose directional derivative is `d0 = grad(x) @ direction < 0`.

    Returns ``(step, f(x + step * direction), calls)``.
    """
    step = 1.0
    for attempt in range(1, max_backtracks + 1):
        f_step = f(x + step * direction)
        if f_step <= f0 + 1e-4 * step * d0 or attempt == max_backtracks:
            return step, f_step, attempt
        step *= 0.5


def _bfgs(f, grad, x0, gtol, maxiter):
    """Quasi-Newton minimization of `f` with the Broyden-Fletcher-Goldfarb-Shanno (1970)
    update to an approximate inverse Hessian `H`, backtracking (Armijo) for the step length.

    The curvature update is skipped, keeping the previous (positive-definite) `H`, whenever a
    step's secant pair fails the curvature condition ``s @ y > 0``; this trades the fast
    convergence a Wolfe line search would give for a much simpler, still globally safe, update.
    """
    x = np.array(x0, dtype=float)
    n = x.size
    g = grad(x)
    fx = f(x)
    nfev, njev, nit = 1, 1, 0
    H = np.eye(n)
    while nit < maxiter and np.max(np.abs(g)) > gtol:
        direction = -H @ g
        d0 = g @ direction
        if d0 >= 0:
            H = np.eye(n)
            direction = -g
            d0 = g @ direction
        step, f_new, calls = _line_search(f, x, direction, fx, d0)
        nfev += calls
        x_new = x + step * direction
        g_new = grad(x_new)
        njev += 1
        s, y = x_new - x, g_new - g
        sy = s @ y
        if sy > 1e-12 * (np.linalg.norm(s) * np.linalg.norm(y) + 1e-300):
            rho = 1.0 / sy
            identity = np.eye(n)
            H = (identity - rho * np.outer(s, y)) @ H @ (identity - rho * np.outer(y, s))
            H = H + rho * np.outer(s, s)
        x, fx, g = x_new, f_new, g_new
        nit += 1
    return x, fx, g, np.max(np.abs(g)) <= gtol, nit, nfev, njev


def minimize(fun, x0, args=(), method=None, jac=None, tol=None, options=None, bounds=None, constraints=()):
    """Minimize the scalar function `fun(x, *args)` from the starting point `x0`.

    `method` is `"Nelder-Mead"` (derivative-free simplex search) or `"BFGS"` (the default;
    quasi-Newton with a finite-difference gradient unless `jac` is given). `options` may set
    `maxiter`, and `xatol`/`fatol` (Nelder-Mead) or `gtol` (BFGS, overriding `tol`). `bounds`,
    `constraints` and other methods are not supported.

    >>> minimize(lambda x: (x[0] - 1) ** 2 + (x[1] + 2) ** 2, [0.0, 0.0]).x
    array([ 1., -2.])
    """
    if bounds is not None:
        raise NotImplementedError("minimize(bounds=...) is not supported by shellsim's SciPy")
    if constraints:
        raise NotImplementedError("minimize(constraints=...) is not supported by shellsim's SciPy")
    if not isinstance(args, tuple):
        args = (args,)
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    options = dict(options or {})
    method_given = method or "BFGS"
    method = method_given.lower()

    def f(x):
        return float(fun(x, *args))

    if method == "nelder-mead":
        maxiter = options.get("maxiter", 200 * x0.size)
        xatol, fatol = options.get("xatol", 1e-4), options.get("fatol", 1e-4)
        x, fx, converged, nit, nfev = _nelder_mead(f, x0, maxiter, xatol, fatol)
        status, message = _status_message(converged)
        return OptimizeResult(
            x=x, fun=fx, success=converged, status=status, message=message, nit=nit, nfev=nfev
        )
    if method == "bfgs":
        maxiter = options.get("maxiter", 200 * x0.size)
        gtol = options.get("gtol", tol if tol is not None else 1e-5)
        if jac is None:

            def grad(x):
                return _finite_diff_gradient(f, x, f(x))

        else:

            def grad(x):
                return np.asarray(jac(x, *args), dtype=float)

        x, fx, gx, converged, nit, nfev, njev = _bfgs(f, grad, x0, gtol, maxiter)
        status, message = _status_message(converged)
        return OptimizeResult(
            x=x, fun=fx, jac=gx, success=converged, status=status, message=message,
            nit=nit, nfev=nfev, njev=njev,
        )
    raise NotImplementedError(
        f"minimize(method={method_given!r}) is not supported by shellsim's SciPy"
    )


# -------------------------------------------------------------------------------------------
# minimize_scalar: Brent's method, bounded or bracketed
# -------------------------------------------------------------------------------------------


def _expand_bracket(f, xa, xb, grow=1.618034, max_iter=100):
    """An interval containing a local minimum of `f`, by geometric expansion from `(xa, xb)`
    in the downhill direction until `f` rises again."""
    fa, fb = f(xa), f(xb)
    if fb > fa:
        xa, xb, fa, fb = xb, xa, fb, fa
    for _ in range(max_iter):
        xc = xb + grow * (xb - xa)
        fc = f(xc)
        if fc >= fb:
            return (xa, xc) if xa < xc else (xc, xa)
        xa, xb, fa, fb = xb, xc, fb, fc
    return (min(xa, xb), max(xa, xb))


def _brent_bounded(f, lo, hi, xatol=1e-5, maxiter=500):
    """Brent's (1973) golden-section/parabolic-interpolation hybrid for the minimum of `f` on
    `[lo, hi]`."""
    golden = (3.0 - math.sqrt(5.0)) / 2.0
    a, b = lo, hi
    x = w = v = a + golden * (b - a)
    fx = fw = fv = f(x)
    d = e = 0.0
    nfev, nit = 1, 0
    for nit in range(1, maxiter + 1):
        mid = 0.5 * (a + b)
        tol1 = xatol * abs(x) + 1e-11
        tol2 = 2.0 * tol1
        if abs(x - mid) <= tol2 - 0.5 * (b - a):
            break
        use_golden = True
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            p = -p if q > 0 else p
            q = abs(q)
            previous_e, e = e, d
            if abs(p) < abs(0.5 * q * previous_e) and a - x < p / q < b - x:
                d = p / q
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = tol1 if mid >= x else -tol1
                use_golden = False
        if use_golden:
            e = (b - x) if x < mid else (a - x)
            d = golden * e
        u = x + d if abs(d) >= tol1 else x + (_signed(tol1, d) if d else tol1)
        fu = f(u)
        nfev += 1
        if fu <= fx:
            a, b = (a, x) if u < x else (x, b)
            v, fv, w, fw, x, fx = w, fw, x, fx, u, fu
        else:
            a, b = (u, b) if u < x else (a, u)
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x, fx, nit, nfev


def minimize_scalar(fun, bracket=None, bounds=None, args=(), method=None):
    """Minimize the scalar function `fun(x, *args)` of one variable.

    With `bounds=(lo, hi)` (`method="bounded"`), the search stays inside the interval.
    Otherwise Brent's method searches from `bracket` (two points to expand outward from, or
    three already bracketing a minimum; the default starts from `(0, 1)`).

    >>> round(minimize_scalar(lambda x: (x - 2) ** 2).x, 6)
    2.0
    """
    if not isinstance(args, tuple):
        args = (args,)

    def f(x):
        return float(fun(x, *args))

    if bounds is not None:
        if method not in (None, "bounded"):
            raise NotImplementedError(
                f"minimize_scalar(bounds=..., method={method!r}) is not supported by "
                "shellsim's SciPy"
            )
        lo, hi = sorted((float(bounds[0]), float(bounds[1])))
    else:
        if method not in (None, "brent", "Brent"):
            raise NotImplementedError(
                f"minimize_scalar(method={method!r}) is not supported by shellsim's SciPy"
            )
        if bracket is None:
            lo, hi = _expand_bracket(f, 0.0, 1.0)
        elif len(bracket) == 2:
            lo, hi = _expand_bracket(f, float(bracket[0]), float(bracket[1]))
        else:
            lo, hi = min(bracket), max(bracket)
    x, fx, nit, nfev = _brent_bounded(f, lo, hi)
    return OptimizeResult(x=x, fun=fx, success=True, status=0, nit=nit, nfev=nfev)


# -------------------------------------------------------------------------------------------
# Scalar root finding
# -------------------------------------------------------------------------------------------


def _require_bracket(fa, fb):
    if fa == 0 or fb == 0:
        return
    if (fa > 0) == (fb > 0):
        raise ValueError("f(a) and f(b) must have different signs")


def _brentq_core(f, a, b, xtol, rtol, maxiter):
    """Brent's (1973) bracketed root finder (inverse quadratic interpolation, falling back to
    the secant step and then bisection), also known from Forsythe, Malcolm and Moler's
    `zeroin`."""
    fa, fb = f(a), f(b)
    calls = 2
    _require_bracket(fa, fb)
    if fa == 0:
        return a, 0, calls, True
    if fb == 0:
        return b, 0, calls, True
    c, fc = a, fa
    d = e = b - a
    for iterations in range(1, maxiter + 1):
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol = 2 * rtol * abs(b) + xtol / 2
        m = 0.5 * (c - b)
        if abs(m) <= tol or fb == 0:
            return b, iterations, calls, True
        if abs(e) < tol or abs(fa) <= abs(fb):
            d = e = m
        else:
            s = fb / fa
            if a == c:
                p, q = 2 * m * s, 1 - s
            else:
                q, r = fa / fc, fb / fc
                p = s * (2 * m * q * (q - r) - (b - a) * (r - 1))
                q = (q - 1) * (r - 1) * (s - 1)
            p, q = (-p, q) if p > 0 else (p, -q)
            if 2 * p < min(3 * m * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = m
        a, fa = b, fb
        b += d if abs(d) > tol else _signed(tol, m)
        fb = f(b)
        calls += 1
        if (fb > 0) == (fc > 0):
            c, fc = a, fa
            d = e = b - a
    return b, maxiter, calls, abs(0.5 * (c - b)) <= tol


def _bisect_core(f, a, b, xtol, rtol, maxiter):
    """Plain bisection of the bracket `[a, b]`."""
    fa, fb = f(a), f(b)
    calls = 2
    _require_bracket(fa, fb)
    if fa == 0:
        return a, 0, calls, True
    if fb == 0:
        return b, 0, calls, True
    for iterations in range(1, maxiter + 1):
        mid = 0.5 * (a + b)
        fmid = f(mid)
        calls += 1
        if (fa > 0) == (fmid > 0):
            a, fa = mid, fmid
        else:
            b, fb = mid, fmid
        if abs(b - a) <= xtol + rtol * abs(mid) or fmid == 0:
            return mid, iterations, calls, True
    return 0.5 * (a + b), maxiter, calls, False


def _newton_core(f, x0, fprime, x1, tol, maxiter):
    """Newton's method when `fprime` is given, otherwise the secant method."""
    calls = 0
    if fprime is not None:
        x, fx = x0, f(x0)
        calls += 1
        for iterations in range(1, maxiter + 1):
            dfx = fprime(x)
            calls += 1
            if dfx == 0:
                return x, iterations, calls, False
            x_new = x - fx / dfx
            fx_new = f(x_new)
            calls += 1
            if abs(x_new - x) <= tol:
                return x_new, iterations, calls, True
            x, fx = x_new, fx_new
        return x, maxiter, calls, False
    x_prev = x0
    x = x1 if x1 is not None else x0 + (1e-4 if x0 == 0 else 1e-4 * abs(x0))
    f_prev, f_cur = f(x_prev), f(x)
    calls += 2
    for iterations in range(1, maxiter + 1):
        if f_cur == f_prev:
            return x, iterations, calls, False
        x_new = x - f_cur * (x - x_prev) / (f_cur - f_prev)
        f_new = f(x_new)
        calls += 1
        if abs(x_new - x) <= tol:
            return x_new, iterations, calls, True
        x_prev, f_prev, x, f_cur = x, f_cur, x_new, f_new
    return x, maxiter, calls, False


class RootResult:
    """The result of `root_scalar`: `root`, `converged`, `iterations` and `function_calls`."""

    def __init__(self, root, converged, iterations, function_calls, method):
        self.root = root
        self.converged = converged
        self.iterations = iterations
        self.function_calls = function_calls
        self.method = method


def root_scalar(f, args=(), method=None, bracket=None, x0=None, x1=None, fprime=None,
                 xtol=None, rtol=None, maxiter=None):
    """Find a root of `f(x, *args)` near `x0`, or inside `bracket`.

    `bracket=(a, b)` (values of opposite sign at the ends) uses `method="brentq"` (the
    default) or `"bisect"`. Otherwise `x0` starts Newton's method when `fprime` is given, or
    the secant method (optionally seeded with a second point `x1`).

    >>> round(root_scalar(lambda x: x**3 - 2, bracket=(1, 2)).root, 6)
    1.259921
    """
    if not isinstance(args, tuple):
        args = (args,)

    def wrapped(x):
        return f(x, *args)

    if bracket is not None:
        if method not in (None, "brentq", "bisect"):
            raise NotImplementedError(
                f"root_scalar(bracket=..., method={method!r}) is not supported by "
                "shellsim's SciPy"
            )
        core = _bisect_core if method == "bisect" else _brentq_core
        root, iterations, calls, converged = core(
            wrapped, float(bracket[0]), float(bracket[1]),
            xtol or 2e-12, rtol or 8.881784197001252e-16, maxiter or 100,
        )
        return RootResult(root, converged, iterations, calls, method or "brentq")
    if x0 is None:
        raise ValueError("either `bracket` or `x0` must be given")
    if method not in (None, "newton", "secant"):
        raise NotImplementedError(
            f"root_scalar(x0=..., method={method!r}) is not supported by shellsim's SciPy"
        )
    wrapped_prime = None if fprime is None else (lambda x: fprime(x, *args))
    root, iterations, calls, converged = _newton_core(
        wrapped, float(x0), wrapped_prime, x1, xtol or 1.48e-8, maxiter or 50
    )
    return RootResult(root, converged, iterations, calls, method or ("newton" if fprime else "secant"))


def brentq(f, a, b, args=(), xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100):
    """Brent's bracketed root finder for `f(x, *args)` between `a` and `b`, which must give
    values of opposite sign."""
    if not isinstance(args, tuple):
        args = (args,)
    root, _, _, _ = _brentq_core(lambda x: f(x, *args), float(a), float(b), xtol, rtol, maxiter)
    return root


def bisect(f, a, b, args=(), xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100):
    """Bisection root finder for `f(x, *args)` between `a` and `b`, which must give values of
    opposite sign."""
    if not isinstance(args, tuple):
        args = (args,)
    root, _, _, _ = _bisect_core(lambda x: f(x, *args), float(a), float(b), xtol, rtol, maxiter)
    return root


def newton(func, x0, fprime=None, args=(), tol=1.48e-8, maxiter=50, x1=None):
    """Newton's method for a root of `func(x, *args)` near `x0` when `fprime` is given,
    otherwise the secant method (optionally seeded with a second point `x1`)."""
    if not isinstance(args, tuple):
        args = (args,)
    wrapped_prime = None if fprime is None else (lambda x: fprime(x, *args))
    root, _, _, _ = _newton_core(
        lambda x: func(x, *args), float(x0), wrapped_prime, x1, tol, maxiter
    )
    return root


# -------------------------------------------------------------------------------------------
# curve_fit
# -------------------------------------------------------------------------------------------


def _default_p0(f):
    """The number of fit parameters `f` takes after its first (independent-variable) argument,
    the way SciPy infers `p0=None` from `f`'s signature."""
    import inspect

    try:
        count = len(inspect.signature(f).parameters) - 1
    except (TypeError, ValueError):
        count = 0
    if count <= 0:
        raise ValueError("`p0` must be given: the number of parameters could not be inferred")
    return np.ones(count)


def _finite_diff_jacobian(residuals, p, r0):
    """The forward-difference Jacobian of `residuals` at `p`, one extra vectorized call per
    parameter."""
    eps = np.sqrt(np.finfo(float).eps)
    jacobian = np.empty((r0.size, p.size))
    for i in range(p.size):
        step = eps * max(abs(p[i]), 1.0)
        perturbed = p.copy()
        perturbed[i] += step
        jacobian[:, i] = (residuals(perturbed) - r0) / step
    return jacobian


def curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, maxfev=None,
              bounds=(-np.inf, np.inf)):
    """Fit `ydata ~ f(xdata, *params)` by nonlinear least squares (Levenberg-Marquardt with a
    finite-difference Jacobian). Returns `(popt, pcov)`. `bounds` other than the default
    (unbounded) are not supported.

    >>> f = lambda x, a, b: a * np.exp(-b * x)
    >>> x = np.linspace(0, 4, 10)
    >>> popt, _ = curve_fit(f, x, f(x, 2.5, 1.3), p0=[1.0, 1.0])
    >>> np.round(popt, 4)
    array([2.5, 1.3])
    """
    lower, upper = bounds
    if not (np.all(np.asarray(lower) == -np.inf) and np.all(np.asarray(upper) == np.inf)):
        raise NotImplementedError("curve_fit(bounds=...) is not supported by shellsim's SciPy")
    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)
    p = np.array(_default_p0(f) if p0 is None else p0, dtype=float)
    n_data, n_params = ydata.size, p.size
    weights = 1.0 if sigma is None else 1.0 / np.asarray(sigma, dtype=float)
    maxfev = maxfev or 200 * (n_params + 1)

    def residuals(params):
        return weights * (f(xdata, *params) - ydata)

    lam = 1e-3
    r = residuals(p)
    cost = r @ r
    for _ in range(maxfev):
        jacobian = _finite_diff_jacobian(residuals, p, r)
        gram = jacobian.T @ jacobian
        diag = np.diag(gram)
        diag = diag if np.any(diag > 0) else np.ones(n_params)
        step = np.linalg.solve(gram + lam * np.diag(diag), -(jacobian.T @ r))
        p_new = p + step
        r_new = residuals(p_new)
        cost_new = r_new @ r_new
        if cost_new < cost:
            improved = cost - cost_new
            p, r, cost = p_new, r_new, cost_new
            lam *= 0.5
            if improved <= 1e-14 * (cost + 1e-300):
                break
        else:
            lam *= 2.0
    jacobian = _finite_diff_jacobian(residuals, p, r)
    gram = jacobian.T @ jacobian
    dof = max(n_data - n_params, 1)
    try:
        cov = np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        cov = np.full((n_params, n_params), np.inf)
    if not absolute_sigma:
        cov = cov * (cost / dof)
    return p, cov
