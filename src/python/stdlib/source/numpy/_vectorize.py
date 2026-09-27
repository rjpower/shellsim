"""``numpy.vectorize``, following ``numpy/lib/_function_base_impl.py`` (NumPy 2.5).

NumPy wraps the Python function in a ``frompyfunc`` object ufunc and calls it on object arrays.
shellsim has no ``frompyfunc``, so the call broadcasts the object arrays and loops over them in
C order itself. What the function sees is the same: when ``otypes`` is not given, a first call on
the first elements, as NumPy scalars, infers the output types, and every element is then passed
as the Python object an object array holds. The ``signature=`` form loops over the broadcast
loop dimensions with ``ndindex``, as NumPy's does.
"""

import re

from _numpy import asanyarray, asarray, dtype, empty, empty_like
from _numpy_shape import broadcast_arrays, broadcast_shapes, broadcast_to
from numpy._function_base import iterable
from numpy._index_tricks import ndindex

_NoValue = object()

typecodes = {
    "Character": "c",
    "Integer": "bhilqnp",
    "UnsignedInteger": "BHILQNP",
    "Float": "efdg",
    "Complex": "FDG",
    "AllInteger": "bBhHiIlLqQnNpP",
    "AllFloat": "efdgFDG",
    "Datetime": "Mm",
    "All": "?bhilqnpBHILQNPefdgFDGSUVOMm",
}

_DIMENSION_NAME = r"\w+"
_CORE_DIMENSION_LIST = f"(?:{_DIMENSION_NAME}(?:,{_DIMENSION_NAME})*)?"
_ARGUMENT = rf"\({_CORE_DIMENSION_LIST}\)"
_ARGUMENT_LIST = f"{_ARGUMENT}(?:,{_ARGUMENT})*"
_SIGNATURE = f"^{_ARGUMENT_LIST}->{_ARGUMENT_LIST}$"


def _parse_gufunc_signature(signature):
    signature = re.sub(r"\s+", "", signature)
    if not re.match(_SIGNATURE, signature):
        raise ValueError(f"not a valid gufunc signature: {signature}")
    return tuple(
        [tuple(re.findall(_DIMENSION_NAME, arg)) for arg in re.findall(_ARGUMENT, arg_list)]
        for arg_list in signature.split("->")
    )


def _update_dim_sizes(dim_sizes, arg, core_dims):
    if not core_dims:
        return
    num_core_dims = len(core_dims)
    if arg.ndim < num_core_dims:
        raise ValueError(
            f"{arg.ndim}-dimensional argument does not have enough dimensions for all core "
            f"dimensions {core_dims!r}"
        )
    core_shape = arg.shape[-num_core_dims:]
    for dim, size in zip(core_dims, core_shape):
        if dim in dim_sizes:
            if size != dim_sizes[dim]:
                raise ValueError(
                    f"inconsistent size for core dimension {dim!r}: {size!r} vs "
                    f"{dim_sizes[dim]!r}"
                )
        else:
            dim_sizes[dim] = size


def _parse_input_dimensions(args, input_core_dims):
    dim_sizes = {}
    loop_shapes = []
    for arg, core_dims in zip(args, input_core_dims):
        _update_dim_sizes(dim_sizes, arg, core_dims)
        ndim = arg.ndim - len(core_dims)
        loop_shapes.append(arg.shape[:ndim])
    return broadcast_shapes(*loop_shapes), dim_sizes


def _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims):
    return [
        broadcast_shape + tuple(dim_sizes[dim] for dim in core_dims)
        for core_dims in list_of_core_dims
    ]


def _create_arrays(broadcast_shape, dim_sizes, list_of_core_dims, dtypes, results=None):
    shapes = _calculate_shapes(broadcast_shape, dim_sizes, list_of_core_dims)
    if dtypes is None:
        dtypes = [None] * len(shapes)
    if results is None:
        return tuple(empty(shape=shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes))
    return tuple(
        empty_like(result, shape=shape, dtype=dtype)
        for result, shape, dtype in zip(results, shapes, dtypes)
    )


def _get_vectorize_dtype(dtype):
    if dtype.char in "SU":
        return dtype.char
    return dtype


class vectorize:
    """Evaluate a Python function element by element over broadcast array arguments."""

    def __init__(
        self, pyfunc=_NoValue, otypes=None, doc=None, excluded=None, cache=False, signature=None
    ):
        if pyfunc is not _NoValue and not callable(pyfunc):
            raise TypeError("When used as a decorator, only accepts keyword arguments.")
        self.pyfunc = pyfunc
        self.cache = cache
        self.signature = signature
        if pyfunc is not _NoValue and hasattr(pyfunc, "__name__"):
            self.__name__ = pyfunc.__name__
        self._doc = None
        self.__doc__ = doc
        if doc is None and hasattr(pyfunc, "__doc__"):
            self.__doc__ = pyfunc.__doc__
        else:
            self._doc = doc
        if isinstance(otypes, str):
            for char in otypes:
                if char not in typecodes["All"]:
                    raise ValueError(f"Invalid otype specified: {char}")
        elif iterable(otypes):
            otypes = [_get_vectorize_dtype(dtype(x)) for x in otypes]
        elif otypes is not None:
            raise ValueError("Invalid otype specification")
        self.otypes = otypes
        if excluded is None:
            excluded = set()
        self.excluded = set(excluded)
        if signature is not None:
            self._in_and_out_core_dims = _parse_gufunc_signature(signature)
        else:
            self._in_and_out_core_dims = None

    def _init_stage_2(self, pyfunc, *args, **kwargs):
        self.__name__ = pyfunc.__name__
        self.pyfunc = pyfunc
        if self._doc is None:
            # shellsim functions keep no docstring, so ``__doc__`` may be absent.
            self.__doc__ = getattr(pyfunc, "__doc__", None)
        else:
            self.__doc__ = self._doc

    def _call_as_normal(self, *args, **kwargs):
        excluded = self.excluded
        if not kwargs and not excluded:
            func = self.pyfunc
            vargs = args
        else:
            # Excluded arguments pass through unchanged; the rest are vectorized over.
            nargs = len(args)
            names = [_n for _n in kwargs if _n not in excluded]
            inds = [_i for _i in range(nargs) if _i not in excluded]
            the_args = list(args)

            def func(*vargs):
                for _n, _i in enumerate(inds):
                    the_args[_i] = vargs[_n]
                kwargs.update(zip(names, vargs[len(inds) :]))
                return self.pyfunc(*the_args, **kwargs)

            vargs = [args[_i] for _i in inds]
            vargs.extend([kwargs[_n] for _n in names])
        return self._vectorize_call(func=func, args=vargs)

    def __call__(self, *args, **kwargs):
        if self.pyfunc is _NoValue:
            self._init_stage_2(*args, **kwargs)
            return self
        return self._call_as_normal(*args, **kwargs)

    def _get_otypes(self, func, args):
        """The output types, and the function to loop with, which may replay a cached result."""
        if self.otypes is not None:
            return self.otypes, func
        args = [asarray(a) for a in args]
        if any(arg.size == 0 for arg in args):
            raise ValueError("cannot call `vectorize` on size 0 inputs unless `otypes` is set")
        inputs = [arg.flat[0] for arg in args]
        outputs = func(*inputs)
        if self.cache:
            _cache = [outputs]

            def _func(*vargs):
                if _cache:
                    return _cache.pop()
                return func(*vargs)

        else:
            _func = func
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        otypes = "".join([asarray(output).dtype.char for output in outputs])
        return otypes, _func

    def _vectorize_call(self, func, args):
        if self.signature is not None:
            return self._vectorize_call_with_signature(func, args)
        if not args:
            return func()
        otypes, func = self._get_otypes(func, args)
        nout = len(otypes)
        args = broadcast_arrays(*[asanyarray(a, dtype=object) for a in args])
        shape = args[0].shape
        outputs = [empty(shape, dtype=object) for _ in range(nout)]
        for index in ndindex(shape):
            results = func(*[arg[index] for arg in args])
            if nout == 1:
                outputs[0][index] = results
                continue
            if not isinstance(results, tuple) or len(results) != nout:
                raise ValueError(
                    f"vectorize: the function returned {results!r}, not a tuple of {nout} "
                    "outputs"
                )
            for output, result in zip(outputs, results):
                output[index] = result
        if nout == 1:
            return asanyarray(outputs[0], dtype=otypes[0])
        return tuple(asanyarray(x, dtype=t) for x, t in zip(outputs, otypes))

    def _vectorize_call_with_signature(self, func, args):
        input_core_dims, output_core_dims = self._in_and_out_core_dims
        if len(args) != len(input_core_dims):
            raise TypeError(
                "wrong number of positional arguments: "
                f"expected {len(input_core_dims)!r}, got {len(args)!r}"
            )
        args = tuple(asanyarray(arg) for arg in args)
        broadcast_shape, dim_sizes = _parse_input_dimensions(args, input_core_dims)
        input_shapes = _calculate_shapes(broadcast_shape, dim_sizes, input_core_dims)
        args = [broadcast_to(arg, shape) for arg, shape in zip(args, input_shapes)]
        outputs = None
        otypes = self.otypes
        nout = len(output_core_dims)
        for index in ndindex(*broadcast_shape):
            results = func(*(arg[index] for arg in args))
            n_results = len(results) if isinstance(results, tuple) else 1
            if nout != n_results:
                raise ValueError(
                    f"wrong number of outputs from pyfunc: expected {nout!r}, got {n_results!r}"
                )
            if nout == 1:
                results = (results,)
            if outputs is None:
                for result, core_dims in zip(results, output_core_dims):
                    _update_dim_sizes(dim_sizes, asarray(result), core_dims)
                outputs = _create_arrays(
                    broadcast_shape,
                    dim_sizes,
                    output_core_dims,
                    otypes,
                    [asarray(result) for result in results],
                )
            for output, result in zip(outputs, results):
                output[index] = result
        if outputs is None:
            if otypes is None:
                raise ValueError(
                    "cannot call `vectorize` on size 0 inputs unless `otypes` is set"
                )
            if any(dim not in dim_sizes for dims in output_core_dims for dim in dims):
                raise ValueError(
                    "cannot call `vectorize` with a signature including new output "
                    "dimensions on size 0 inputs"
                )
            outputs = _create_arrays(broadcast_shape, dim_sizes, output_core_dims, otypes)
        return outputs[0] if nout == 1 else outputs
