"""``np.vectorize``: apply a plain Python function element by element.

Without a ``signature=``, every non-excluded argument is broadcast together and the function
runs once per broadcast position on plain Python scalars (``ndarray.tolist()``'s elements, not
NumPy scalars, matching NumPy's own object-array loop). When ``otypes`` is not given, one extra
"trial" call on NumPy-scalar inputs at position zero discovers the output dtype(s) before the
real loop runs, exactly duplicating NumPy's own (mildly wasteful) behavior.

With a ``signature=``, a gufunc-style ``"(n),(m)->(k)"`` string, the leading, non-core
dimensions broadcast instead, and the function runs once per broadcast position on the core-shaped
slices, matching :func:`numpy.core.sum`-style reductions applied along the last axes.
"""

import re

import numpy as np

__all__ = ["vectorize"]

_DIMENSION = r"\w+"
_DIMENSION_LIST = rf"(?:{_DIMENSION}(?:,{_DIMENSION})*)?"
_ARGUMENT = rf"\({_DIMENSION_LIST}\)"
_ARGUMENT_LIST = rf"{_ARGUMENT}(?:,{_ARGUMENT})*"
_SIGNATURE = re.compile(rf"^{_ARGUMENT_LIST}->{_ARGUMENT_LIST}$")
_ARGUMENT_DIMS = re.compile(r"\(([^)]*)\)")


def _parse_signature(signature):
    """A ``(input_dims, output_dims)`` pair, each a list of dimension-name tuples per argument."""
    text = signature.replace(" ", "")
    if not _SIGNATURE.match(text):
        raise ValueError(f"not a valid gufunc signature: {signature}")
    inputs_text, outputs_text = text.split("->")
    parse = lambda part: [
        tuple(name for name in dims.split(",") if name) for dims in _ARGUMENT_DIMS.findall(part)
    ]
    return parse(inputs_text), parse(outputs_text)


def _parse_otypes(otypes):
    """One dtype per output, from a type-code string or a list of dtype-likes."""
    if isinstance(otypes, str):
        resolved = []
        for code in otypes:
            try:
                resolved.append(np.dtype(code))
            except TypeError:
                raise ValueError(f"Invalid otype specified: {code}") from None
        return resolved
    try:
        return [np.dtype(one) for one in otypes]
    except TypeError:
        raise ValueError("Invalid otype specification") from None


class vectorize:
    """A callable that applies `pyfunc` element by element over its array arguments.

    Used directly (``np.vectorize(f)``) or as a decorator, including with only keyword
    arguments (``@np.vectorize(otypes=[float])``), in which case the decorated function is
    filled in as `pyfunc` on the next call.
    """

    def __init__(self, pyfunc=None, otypes=None, doc=None, excluded=None, cache=False, signature=None):
        self.pyfunc = pyfunc
        self.otypes = _parse_otypes(otypes) if otypes is not None else None
        self.excluded = set(excluded) if excluded is not None else set()
        self.signature = signature
        self._core_dims = _parse_signature(signature) if signature is not None else None
        self.__doc__ = doc if doc is not None else getattr(pyfunc, "__doc__", None)
        if pyfunc is not None:
            self.__name__ = getattr(pyfunc, "__name__", "vectorize")

    def __call__(self, *args, **kwargs):
        if self.pyfunc is None:
            (func,) = args
            return vectorize(
                func,
                otypes=self.otypes,
                doc=self.__doc__,
                excluded=self.excluded or None,
                signature=self.signature,
            )
        if self._core_dims is not None:
            return self._call_with_signature(args, kwargs)
        return self._call_elementwise(args, kwargs)

    def _call_elementwise(self, args, kwargs):
        included = [(index, np.asanyarray(value)) for index, value in enumerate(args) if index not in self.excluded]
        if not included:
            raise TypeError("vectorize needs at least one non-excluded argument")
        shape = np.broadcast_shapes(*(value.shape for _, value in included))
        size = 1
        for dim in shape:
            size *= dim

        if self.otypes is None:
            if size == 0:
                raise ValueError("cannot call `vectorize` on size 0 inputs")
            trial_args = list(args)
            for index, value in included:
                trial_args[index] = np.broadcast_to(value, shape).reshape(-1)[0]
            trial_result = self.pyfunc(*trial_args, **kwargs)
            multiple = isinstance(trial_result, tuple)
            outputs = trial_result if multiple else (trial_result,)
            otypes = [np.asanyarray(one).dtype for one in outputs]
        else:
            otypes = self.otypes
            multiple = len(otypes) > 1

        if size == 0:
            empty = [np.empty(shape, dtype=dtype) for dtype in otypes]
            return tuple(empty) if multiple else empty[0]

        flat_args = list(args)
        for index, value in included:
            flat_args[index] = np.broadcast_to(value, shape).reshape(-1).tolist()

        collected = [[] for _ in otypes]
        for position in range(size):
            call_args = list(flat_args)
            for index, _ in included:
                call_args[index] = flat_args[index][position]
            result = self.pyfunc(*call_args, **kwargs)
            values = result if multiple else (result,)
            for slot, value in zip(collected, values):
                slot.append(value)

        outputs = [np.array(values, dtype=dtype).reshape(shape) for values, dtype in zip(collected, otypes)]
        return tuple(outputs) if multiple else outputs[0]

    def _call_with_signature(self, args, kwargs):
        input_dims, output_dims = self._core_dims
        if len(args) != len(input_dims):
            raise TypeError(f"wrong number of positional arguments: expected {len(input_dims)}, got {len(args)}")
        arrays = [np.asanyarray(value) for value in args]
        outer_shapes = [
            array.shape[: array.ndim - len(dims)] if dims else array.shape
            for array, dims in zip(arrays, input_dims)
        ]
        outer_shape = np.broadcast_shapes(*outer_shapes)
        size = 1
        for dim in outer_shape:
            size *= dim

        flat_arrays = []
        for array, dims in zip(arrays, input_dims):
            core_shape = array.shape[array.ndim - len(dims) :] if dims else ()
            broadcast = np.broadcast_to(array, outer_shape + core_shape)
            flat_arrays.append(broadcast.reshape((size,) + core_shape))

        output_count = len(output_dims)
        collected = [[] for _ in range(output_count)]
        for position in range(size):
            call_args = [flat[position] for flat in flat_arrays]
            result = self.pyfunc(*call_args, **kwargs)
            values = result if output_count > 1 else (result,)
            for slot, value in zip(collected, values):
                slot.append(np.asanyarray(value))

        outputs = [
            np.stack(values, axis=0).reshape(outer_shape + values[0].shape) for values in collected
        ]
        return tuple(outputs) if output_count > 1 else outputs[0]
