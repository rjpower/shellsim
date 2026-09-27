"""``axis``, ``nan_policy`` and ``keepdims`` handling for ``scipy.stats`` functions, following
SciPy 1.18's ``scipy/stats/_axis_nan_policy.py``.

SciPy's ``_axis_nan_policy_factory`` returns a decorator that reads the wrapped function's
signature with ``inspect`` to find its samples and adds the three parameters to it. shellsim
has neither ``inspect`` nor writable function names, so the factory here returns an ``apply``
function instead: each public function declares SciPy's full signature itself and passes its
samples and remaining keyword arguments to ``apply``. The steps after that are SciPy's:

- samples are promoted to a common dtype, and ``axis`` is moved to the end and raveled;
- for 1-D samples, NaNs propagate, are omitted, or raise, and samples that are too small give
  NaN with a ``SmallSampleWarning``;
- n-D samples without NaNs go to the function in one vectorized call along the last axis;
  otherwise the function runs on each axis-slice through ``np.apply_along_axis``;
- ``keepdims`` restores the reduced axes.

Masked arrays are not supported, so there is no sentinel-value handling.
"""

import math
import warnings

import numpy as np
from numpy.exceptions import AxisError
from scipy._lib._util import _contains_nan, _get_nan, _promote

too_small_1d_not_omit = (
    "One or more sample arguments is too small; all "
    "returned values will be NaN. "
    "See documentation for sample size requirements."
)

too_small_1d_omit = (
    "After omitting NaNs, one or more sample arguments "
    "is too small; all returned values will be NaN. "
    "See documentation for sample size requirements."
)

too_small_nd_not_omit = (
    "All axis-slices of one or more sample arguments are "
    "too small; all elements of returned arrays will be NaN. "
    "See documentation for sample size requirements."
)

too_small_nd_omit = (
    "After omitting NaNs, one or more axis-slices of one "
    "or more sample arguments is too small; corresponding "
    "elements of returned arrays will be NaN. "
    "See documentation for sample size requirements."
)


class SmallSampleWarning(RuntimeWarning):
    pass


def _broadcast_arrays(arrays, axis=None):
    """Broadcast shapes of arrays, ignoring incompatibility of specified axes."""
    arrays = tuple(arrays)
    if not arrays:
        return arrays
    arrays = [np.asarray(arr) for arr in arrays]
    shapes = [arr.shape for arr in arrays]
    new_shapes = _broadcast_shapes(shapes, axis)
    if axis is None:
        new_shapes = [new_shapes] * len(arrays)
    return [np.broadcast_to(array, new_shape) for array, new_shape in zip(arrays, new_shapes)]


def _broadcast_shapes(shapes, axis=None):
    """Broadcast shapes, ignoring incompatibility of specified axes.

    With ``axis``, returns one shape per input, each keeping its own lengths along ``axis``;
    without, returns the single broadcast shape.
    """
    if not shapes:
        return shapes

    # input validation
    if axis is not None:
        axis = np.atleast_1d(axis)
        message = "`axis` must be an integer, a tuple of integers, or `None`."
        try:
            with np.errstate(invalid="ignore"):
                axis_int = axis.astype(int)
        except (ValueError, TypeError) as e:
            raise AxisError(message) from e
        if not np.array_equal(axis_int, axis):
            raise AxisError(message)
        axis = axis_int

    # First, ensure all shapes have same number of dimensions by prepending 1s.
    n_dims = max([len(shape) for shape in shapes])
    new_shapes = np.ones((len(shapes), n_dims), dtype=int)
    for i, shape in enumerate(shapes):
        if shape:
            new_shapes[i, n_dims - len(shape) :] = shape

    # Remove the shape elements of the axes to be ignored, but remember them.
    if axis is not None:
        axis[axis < 0] = n_dims + axis[axis < 0]
        axis = np.sort(axis)
        if axis[-1] >= n_dims or axis[0] < 0:
            message = f"`axis` is out of bounds for array of dimension {n_dims}"
            raise AxisError(message)

        if len(np.unique(axis)) != len(axis):
            raise AxisError("`axis` must contain only distinct elements")

        removed_shapes = new_shapes[:, axis]
        removed_axes = set(axis.tolist())
        new_shapes = new_shapes[:, [i for i in range(n_dims) if i not in removed_axes]]

    # If arrays are broadcastable, shape elements that are 1 may be replaced
    # with a corresponding non-1 shape element. Assuming arrays are
    # broadcastable, that final shape element can be found with:
    new_shape = np.max(new_shapes, axis=0)
    # except in case of an empty array:
    new_shape *= new_shapes.all(axis=0)

    # Among all arrays, there can only be one unique non-1 shape element.
    # Therefore, if any non-1 shape element does not match what we found
    # above, the arrays must not be broadcastable after all.
    if np.any(~((new_shapes == 1) | (new_shapes == new_shape))):
        raise ValueError("Array shapes are incompatible for broadcasting.")

    if axis is not None:
        # Add back the shape elements that were ignored
        result = []
        for removed_shape in removed_shapes:
            removed = dict(zip(axis.tolist(), removed_shape.tolist()))
            kept = iter(new_shape.tolist())
            result.append(
                tuple(removed[i] if i in removed else next(kept) for i in range(n_dims))
            )
        return result
    else:
        return tuple(new_shape.tolist())


def _broadcast_array_shapes_remove_axis(arrays, axis=None):
    """The broadcast shape of ``arrays`` after dropping ``axis``.

    For example, arrays of shapes ``(5, 2, 1)`` and ``(9, 3)`` with ``axis=1`` give ``(5, 3)``:
    the output shape of a hypothesis test vectorized along ``axis``.
    """
    shapes = [arr.shape for arr in arrays]
    return _broadcast_shapes_remove_axis(shapes, axis)


def _broadcast_shapes_remove_axis(shapes, axis=None):
    """Same as ``_broadcast_array_shapes_remove_axis``, given the shapes."""
    shapes = _broadcast_shapes(shapes, axis)
    shape = shapes[0]
    if axis is not None:
        removed = {int(i) % len(shape) for i in np.atleast_1d(axis)}
        shape = tuple(length for i, length in enumerate(shape) if i not in removed)
    return tuple(shape)


def _broadcast_concatenate(arrays, axis, paired=False):
    """Concatenate arrays along an axis with broadcasting."""
    arrays = _broadcast_arrays(arrays, axis if not paired else None)
    return np.concatenate(arrays, axis=axis)


def _remove_nans(samples, paired):
    "Remove nans from paired or unpaired 1D samples"
    if not paired:
        return [sample[~np.isnan(sample)] for sample in samples]

    # for paired samples, we need to remove the whole pair when any part
    # has a nan
    nans = np.isnan(samples[0])
    for sample in samples[1:]:
        nans = nans | np.isnan(sample)
    not_nans = ~nans
    return [sample[not_nans] for sample in samples]


def _check_empty_inputs(samples, axis):
    """NaNs of the output shape if a sample is empty; ``None`` if none is."""
    # if none of the samples are empty, we need to perform the test
    if not any(sample.size == 0 for sample in samples):
        return None
    # otherwise, the statistic and p-value will be either empty arrays or
    # arrays with NaNs. Produce the appropriate array and return it.
    output_shape = _broadcast_array_shapes_remove_axis(samples, axis)
    NaN = _get_nan(*samples)
    return np.full(output_shape, np.nan, dtype=NaN.dtype)


def _add_reduced_axes(res, reduced_axes, keepdims):
    """Add reduced axes back to all the arrays in the result object if keepdims = True."""
    return (
        [
            np.expand_dims(output, axis=reduced_axes) if not isinstance(output, int) else output
            for output in res
        ]
        if keepdims
        else res
    )


def _axis_nan_policy_factory(
    tuple_to_result,
    paired=False,
    result_to_tuple=None,
    too_small=0,
    n_outputs=2,
    override=None,
):
    """Return ``apply(function, samples, kwds, axis, nan_policy, keepdims)``.

    ``apply`` evaluates ``function(*samples, **kwds)`` with SciPy's ``axis``, ``nan_policy``
    and ``keepdims`` semantics and returns ``tuple_to_result(*outputs)``.

    - ``tuple_to_result`` builds the result from its components; ``result_to_tuple(res, n)``
      is its inverse and defaults to returning ``res``.
    - ``paired`` samples are broadcast along ``axis`` too, and lose whole pairs to NaNs.
    - ``too_small`` is the largest unacceptable sample size, or a function of the samples and
      keyword arguments that says whether they are too small.
    - ``n_outputs`` is the number of outputs, or a function of the keyword arguments.
    - ``override``: ``{'nan_propagation': False}`` leaves NaNs to the function when
      ``nan_policy`` is ``'propagate'``.
    """
    # Specify which existing behaviors the decorator must override
    temp = override or {}
    override = {"vectorization": False, "nan_propagation": True}
    override.update(temp)

    if result_to_tuple is None:

        def result_to_tuple(res, _):
            return res

    if not callable(too_small):

        def is_too_small(samples, *ts_args, axis=-1, **ts_kwargs):
            for sample in samples:
                if sample.shape[axis] <= too_small:
                    return True
            return False

    else:
        is_too_small = too_small

    def axis_nan_policy_apply(hypotest_fun_out, samples, kwds, axis, nan_policy, keepdims):
        n_samp = len(samples)
        n_out = n_outputs(kwds) if callable(n_outputs) else n_outputs

        samples = _promote(*samples)
        samples = (samples,) if not isinstance(samples, tuple) else samples
        samples = [np.atleast_1d(sample) for sample in samples]

        # standardize to always work along last axis
        reduced_axes = axis
        if axis is None:
            if samples:
                # when axis=None, take the maximum of all dimensions since
                # all the dimensions are reduced.
                n_dims = max([sample.ndim for sample in samples])
                reduced_axes = tuple(range(n_dims))
            samples = [sample.reshape(-1) for sample in samples]
        else:
            # don't ignore any axes when broadcasting if paired
            samples = _broadcast_arrays(samples, axis=axis if not paired else None)
            axis = (axis,) if np.isscalar(axis) else tuple(axis)
            n_axes = len(axis)
            # move all axes in `axis` to the end to be raveled
            samples = [np.moveaxis(sample, axis, tuple(range(-len(axis), 0))) for sample in samples]
            shapes = [sample.shape for sample in samples]
            # New shape is unchanged for all axes _not_ in `axis`
            # At the end, we append the product of the shapes of the axes
            # in `axis`. Appending -1 doesn't work for zero-size arrays!
            new_shapes = [shape[:-n_axes] + (math.prod(shape[-n_axes:]),) for shape in shapes]
            samples = [
                np.reshape(sample, new_shape) for sample, new_shape in zip(samples, new_shapes)
            ]
        axis = -1  # work over the last axis

        NaN = _get_nan(*samples) if samples else np.nan

        # if axis is not needed, just handle nan_policy and return
        if all(sample.ndim <= 1 for sample in samples):
            # Addresses nan_policy == "raise"
            if nan_policy != "propagate" or override["nan_propagation"]:
                contains_nan = [_contains_nan(sample, nan_policy) for sample in samples]
            else:
                # Behave as though there are no NaNs (even if there are)
                contains_nan = [False] * len(samples)

            any_contains_nan = any(contains_nan)
            # Addresses nan_policy == "propagate"
            if any_contains_nan and (nan_policy == "propagate" and override["nan_propagation"]):
                res = np.full(n_out, np.nan, dtype=NaN.dtype)
                res = _add_reduced_axes(res, reduced_axes, keepdims)
                return tuple_to_result(*res)

            # Addresses nan_policy == "omit"
            too_small_msg = too_small_1d_not_omit
            if any_contains_nan and nan_policy == "omit":
                samples = _remove_nans(samples, paired)
                too_small_msg = too_small_1d_omit

            if is_too_small(samples, kwds):
                warnings.warn(too_small_msg, SmallSampleWarning, stacklevel=3)
                res = np.full(n_out, np.nan, dtype=NaN.dtype)
                res = _add_reduced_axes(res, reduced_axes, keepdims)
                return tuple_to_result(*res)

            res = hypotest_fun_out(*samples, **kwds)
            res = result_to_tuple(res, n_out)
            res = _add_reduced_axes(res, reduced_axes, keepdims)
            return tuple_to_result(*res)

        # check for empty input
        empty_output = _check_empty_inputs(samples, axis)
        # only return empty output if zero sized input is too small.
        if empty_output is not None and (is_too_small(samples, kwds) or empty_output.size == 0):
            if is_too_small(samples, kwds) and empty_output.size != 0:
                warnings.warn(too_small_nd_not_omit, SmallSampleWarning, stacklevel=3)
            res = [empty_output.copy() for i in range(n_out)]
            res = _add_reduced_axes(res, reduced_axes, keepdims)
            return tuple_to_result(*res)

        # otherwise, concatenate all samples along axis, remembering where
        # each separate sample begins
        lengths = np.array([sample.shape[axis] for sample in samples])
        split_indices = np.cumsum(lengths)
        x = _broadcast_concatenate(samples, axis, paired=paired)

        # Addresses nan_policy == "raise"
        if nan_policy != "propagate" or override["nan_propagation"]:
            contains_nan = _contains_nan(x, nan_policy)
        else:
            contains_nan = False  # behave like there are no NaNs

        if not contains_nan:
            res = hypotest_fun_out(*samples, axis=axis, **kwds)
            res = result_to_tuple(res, n_out)
            res = _add_reduced_axes(res, reduced_axes, keepdims)
            return tuple_to_result(*res)

        # Addresses nan_policy == "omit"
        if nan_policy == "omit":

            def hypotest_fun(x):
                samples = np.split(x, split_indices)[:n_samp]
                samples = _remove_nans(samples, paired)
                if is_too_small(samples, kwds):
                    warnings.warn(too_small_nd_omit, SmallSampleWarning, stacklevel=5)
                    return np.full(n_out, NaN)
                return result_to_tuple(hypotest_fun_out(*samples, **kwds), n_out)

        # Addresses nan_policy == "propagate"
        elif nan_policy == "propagate" and override["nan_propagation"]:

            def hypotest_fun(x):
                if np.isnan(x).any():
                    return np.full(n_out, NaN)

                samples = np.split(x, split_indices)[:n_samp]
                if is_too_small(samples, kwds):
                    return np.full(n_out, NaN)
                return result_to_tuple(hypotest_fun_out(*samples, **kwds), n_out)

        else:

            def hypotest_fun(x):
                samples = np.split(x, split_indices)[:n_samp]
                if is_too_small(samples, kwds):
                    return np.full(n_out, NaN)
                return result_to_tuple(hypotest_fun_out(*samples, **kwds), n_out)

        x = np.moveaxis(x, axis, 0)
        res = np.apply_along_axis(hypotest_fun, axis=0, arr=x)
        res = _add_reduced_axes(res, reduced_axes, keepdims)
        return tuple_to_result(*res)

    return axis_nan_policy_apply
