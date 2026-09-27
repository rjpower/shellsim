"""Small helpers shared by ``scipy.linalg`` and ``scipy.stats``.

These are private, observable-behavior reimplementations of the handful of ``scipy._lib._util``
names other ``scipy`` modules import: ``_apply_over_batch`` (batching decorator),
``_asarray_validated`` and ``_deprecate_dtypes`` (argument checks), and ``check_random_state``
(turn a seed into a NumPy random generator).
"""

import numpy as np

# The dtypes SciPy's compiled LAPACK/BLAS wrappers accept without a deprecation warning.
_LAPACK_DTYPE_CHARS = "fdFD"


def _apply_over_batch(*argdefs):
    """Decorator factory that runs a function over the leading (batch) axes of array arguments.

    ``argdefs`` is a sequence of ``(name, core_ndim)`` pairs: the keyword name of an array
    argument and how many of its trailing axes are its "core" shape (for example 2 for a
    matrix argument, or 1 for a vector). Axes before those are batch axes; they broadcast
    together across every named argument, and the wrapped function runs once per batch index,
    the way ``scipy.linalg`` applies a matrix factorization to each matrix in a stack. The
    wrapped function may return a single array or a tuple of arrays.

    ``core_ndim`` may also be a ``"n|m"`` string for an argument whose rank is ambiguous on its
    own, such as a right-hand side that accepts either a single vector or a matrix of several
    columns (``solve_triangular``'s ``b``, for instance). Each such argument is resolved once the
    batch rank is known from the arguments with a plain integer ``core_ndim``: whichever
    alternative leaves that many leading axes wins, or, if every named argument is ambiguous,
    the largest alternative that still fits inside the array's rank.

    Per SciPy's own documented assumption, the named array arguments are always consecutive
    keyword-or-positional parameters; this maps them to the first ``len(argdefs)`` positional
    slots, in the order given, matching what SciPy's own implementation does (confirmed by
    probing it directly: it mishandles a leading non-array parameter the same way).
    """
    coredim_specs = dict(argdefs)
    positions = {name: index for index, (name, _dims) in enumerate(argdefs)}

    def _resolve_coredim(spec, array, batch_rank):
        if isinstance(spec, int):
            return spec
        options = [int(option) for option in spec.split("|")]
        if batch_rank is not None:
            for option in options:
                if array.ndim - option == batch_rank:
                    return option
        return max(option for option in options if option <= array.ndim)

    def decorator(func):
        def wrapper(*args, **kwargs):
            def fetch(name):
                if name in kwargs:
                    return kwargs[name]
                index = positions.get(name)
                return args[index] if index is not None and index < len(args) else None

            provided = {name: fetch(name) for name in coredim_specs}
            arrays = {name: np.asarray(value) for name, value in provided.items() if value is not None}
            if not arrays:
                return func(*args, **kwargs)

            # Fixed-rank arguments pin down the batch rank so ambiguous ("n|m") ones can be
            # resolved against it; if every named argument is ambiguous there is nothing to pin
            # against, and `_resolve_coredim` falls back to the largest fitting alternative.
            fixed_ranks = [
                array.ndim - coredim_specs[name]
                for name, array in arrays.items()
                if isinstance(coredim_specs[name], int)
            ]
            batch_rank = max(fixed_ranks) if fixed_ranks else None
            coredims = {
                name: _resolve_coredim(coredim_specs[name], array, batch_rank) for name, array in arrays.items()
            }

            batch_shapes = [array.shape[: array.ndim - coredims[name]] for name, array in arrays.items()]
            if all(shape == () for shape in batch_shapes):
                return func(*args, **kwargs)

            batch_shape = np.broadcast_shapes(*batch_shapes)
            broadcasted = {
                name: np.broadcast_to(array, batch_shape + array.shape[array.ndim - coredims[name] :])
                for name, array in arrays.items()
            }
            slots = None
            multiple = False
            for index in np.ndindex(batch_shape):
                call_args = list(args)
                call_kwargs = dict(kwargs)
                for name, array in broadcasted.items():
                    value = array[index]
                    if name in kwargs:
                        call_kwargs[name] = value
                    else:
                        call_args[positions[name]] = value
                result = func(*call_args, **call_kwargs)
                multiple = isinstance(result, tuple)
                pieces = result if multiple else (result,)
                if slots is None:
                    slots = [[] for _ in pieces]
                for slot, piece in zip(slots, pieces):
                    slot.append(np.asarray(piece))
            stacked = tuple(np.stack(slot).reshape(batch_shape + slot[0].shape) for slot in slots)
            return stacked if multiple else stacked[0]

        # Not `functools.wraps`: shellsim functions cannot have `__name__`/`__doc__` reassigned.
        return wrapper

    return decorator


def _asarray_validated(a, check_finite=True, sparse_ok=False, objects_ok=False, mask_ok=False, as_inexact=False):
    """Validate array-like input the way SciPy's linear algebra wrappers do.

    Rejects sparse arrays (unless ``sparse_ok``), masked arrays (unless ``mask_ok``) and object
    arrays (unless ``objects_ok``); optionally rejects non-finite values and coerces to an
    inexact dtype.
    """
    if not sparse_ok and hasattr(a, "toarray"):
        raise ValueError(
            "Sparse arrays/matrices are not supported by this function. "
            "Perhaps one of the `scipy.sparse.linalg` functions would work instead."
        )
    # shellsim's NumPy has no `numpy.ma`, so a masked array can never actually reach here; this
    # duck-types the same check SciPy makes instead of assuming `np.ma` exists.
    if not mask_ok and hasattr(a, "mask") and hasattr(a, "filled"):
        raise ValueError("masked arrays are not supported")
    array = np.asarray(a)
    if not objects_ok and array.dtype == object:
        raise ValueError("object arrays are not supported")
    if as_inexact and not np.issubdtype(array.dtype, np.inexact):
        array = array.astype(np.float64)
    if check_finite and not np.isfinite(array).all():
        raise ValueError("array must not contain infs or NaNs")
    return array


def _deprecate_dtypes(func_name, *arrays):
    """Warn once, as SciPy does, about the first array whose dtype LAPACK/BLAS does not accept.

    Integer and boolean arrays are exempt because callers cast them to a floating dtype before
    reaching LAPACK; every other dtype outside ``float32``, ``float64``, ``complex64`` and
    ``complex128`` is deprecated.
    """
    for array in arrays:
        array = np.asarray(array)
        if array.dtype.kind in "iu" or array.dtype.char in _LAPACK_DTYPE_CHARS:
            continue
        import warnings

        warnings.warn(
            f"Calling {func_name} with arguments of dtype={array.dtype.name} "
            f"(a.dtype.char = {array.dtype.char!r}) is deprecated in SciPy 1.18.0 and will be "
            "removed in SciPy 1.20.0. Please cast array inputs to one of np.float{32,64} or "
            "np.complex{64,128} manually.",
            DeprecationWarning,
            stacklevel=3,
        )
        return


def check_random_state(seed):
    """Turn ``seed`` into a NumPy random generator, as SciPy's distributions do for ``rvs``.

    ``None`` gives NumPy's shared global state, an integer seeds a fresh legacy
    :class:`numpy.random.RandomState`, and an existing ``RandomState`` or ``Generator`` passes
    through unchanged.
    """
    if seed is None:
        return np.random.mtrand._rand
    if isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    raise ValueError(f"{seed!r} cannot be used to seed a numpy.random.RandomState instance")
