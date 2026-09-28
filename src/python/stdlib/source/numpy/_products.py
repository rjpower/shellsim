"""Products built on the native ``dot``/``matmul`` primitives: ``vdot``, ``inner``, ``outer``,
``tensordot``, and ``einsum``.

Every contraction here reduces to reshape, transpose, and one ``dot``/``matmul`` call, so it
never allocates the full outer-product space of its operands' indices. ``tensordot`` moves its
contracted axes to the end of the left operand and the front of the right one, flattens each
operand's free axes and contracted axes into two dimensions, and calls ``dot`` on the resulting
matrices. ``einsum`` contracts its operands pairwise, left to right, the same way: for each pair
it groups indices into batch (shared with a later operand or the output), contracted (shared, but
needed nowhere else), and free indices, then transposes, reshapes to 3-D, and calls ``matmul``
(whose leading axis already broadcasts the batch group). Indices repeated within one operand
(``'ii'``) are reduced first through ``diagonal``, which is also how ``trace``/``diag`` read a
diagonal elsewhere in this port.
"""

import numpy as np

__all__ = ["einsum", "inner", "outer", "tensordot", "vdot"]


def vdot(a, b):
    """The dot product of `a` and `b` flattened, conjugating `a`; always a scalar."""
    a = np.asanyarray(a).reshape(-1)
    b = np.asanyarray(b).reshape(a.size)
    if a.dtype.kind == "c":
        a = np.conjugate(a)
    return np.dot(a, b)


def inner(a, b):
    """The sum product over the last axes of `a` and `b`; a 0-d operand multiplies elementwise."""
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    if a.ndim == 0 or b.ndim == 0:
        return a * b
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(
            f"shapes {a.shape} and {b.shape} not aligned: {a.shape[-1]} (dim {a.ndim - 1}) != "
            f"{b.shape[-1]} (dim {b.ndim - 1})"
        )
    result = tensordot(a, b, axes=(-1, -1))
    return result[()] if result.ndim == 0 else result


def outer(a, b, out=None):
    """`multiply(a.ravel()[:, None], b.ravel()[None, :], out=out)`."""
    a = np.asanyarray(a).reshape(-1)
    b = np.asanyarray(b).reshape(-1)
    return np.multiply(a.reshape(-1, 1), b.reshape(1, -1), out=out)


def _product(values):
    total = 1
    for value in values:
        total *= value
    return total


def _python_index(index, length):
    index = int(index)
    resolved = index + length if index < 0 else index
    if not 0 <= resolved < length:
        raise IndexError("tuple index out of range")
    return resolved


def tensordot(a, b, axes=2):
    """Contract the last `axes` axes of `a` with the first `axes` axes of `b`, or the explicit
    axis lists `(axes_a, axes_b)`. Built as one ``dot`` of the operands reshaped to 2-D: the free
    axes of each operand collapse to one dimension and the contracted axes to the other."""
    a = np.asanyarray(a)
    b = np.asanyarray(b)
    if isinstance(axes, (int, np.integer)):
        count = int(axes)
        a_axes, b_axes = ((), ()) if count < 0 else (tuple(range(-count, 0)), tuple(range(count)))
    else:
        first, second = axes
        a_axes = (int(first),) if isinstance(first, (int, np.integer)) else tuple(int(x) for x in first)
        b_axes = (int(second),) if isinstance(second, (int, np.integer)) else tuple(int(x) for x in second)
    for raw_axes in (a_axes, b_axes):
        if len(set(raw_axes)) != len(raw_axes):
            raise ValueError("duplicate axes are not allowed in tensordot")
    if len(a_axes) != len(b_axes):
        raise ValueError("shape-mismatch for sum")
    a_axes = tuple(_python_index(ax, a.ndim) for ax in a_axes)
    b_axes = tuple(_python_index(ax, b.ndim) for ax in b_axes)
    for a_axis, b_axis in zip(a_axes, b_axes):
        if a.shape[a_axis] != b.shape[b_axis]:
            raise ValueError("shape-mismatch for sum")

    a_free = [ax for ax in range(a.ndim) if ax not in a_axes]
    b_free = [ax for ax in range(b.ndim) if ax not in b_axes]
    a_free_shape = [a.shape[ax] for ax in a_free]
    b_free_shape = [b.shape[ax] for ax in b_free]
    contract_count = _product(a.shape[ax] for ax in a_axes)

    a2 = a.transpose(a_free + list(a_axes)).reshape(_product(a_free_shape), contract_count)
    b2 = b.transpose(list(b_axes) + b_free).reshape(contract_count, _product(b_free_shape))
    return np.dot(a2, b2).reshape(a_free_shape + b_free_shape)


def _dedup_labels(a, labels):
    """`a` with every axis repeated in `labels` folded onto its diagonal, so each label names at
    most one axis of the result; the diagonal always becomes the new last axis, as
    :func:`numpy.diagonal` places it."""
    labels = list(labels)
    while True:
        seen = {}
        pair = None
        for position, label in enumerate(labels):
            if label in seen:
                pair = (seen[label], position)
                break
            seen[label] = position
        if pair is None:
            return a, labels
        axis1, axis2 = pair
        a = np.diagonal(a, axis1=axis1, axis2=axis2)
        labels = [label for i, label in enumerate(labels) if i not in pair] + [labels[axis1]]


def _pairwise_contract(a, la, b, lb, keep):
    """Contract `a` (labels `la`) with `b` (labels `lb`), summing shared labels not in `keep` and
    any label unique to one operand that is also absent from `keep`. Shared labels that survive
    in `keep` become a batch dimension carried through :func:`numpy.matmul` instead of summed."""
    a_only = [label for label in la if label not in lb and label not in keep]
    if a_only:
        a = np.sum(a, axis=tuple(la.index(label) for label in a_only))
        la = [label for label in la if label not in a_only]
    b_only = [label for label in lb if label not in la and label not in keep]
    if b_only:
        b = np.sum(b, axis=tuple(lb.index(label) for label in b_only))
        lb = [label for label in lb if label not in b_only]

    shared = [label for label in la if label in lb]
    batch = [label for label in shared if label in keep]
    contract = [label for label in shared if label not in keep]
    a_free = [label for label in la if label not in shared]
    b_free = [label for label in lb if label not in shared]

    a_perm = [la.index(label) for label in batch + a_free + contract]
    b_perm = [lb.index(label) for label in batch + contract + b_free]
    a2 = a.transpose(a_perm)
    b2 = b.transpose(b_perm)

    batch_shape = [a.shape[la.index(label)] for label in batch]
    a_free_shape = [a.shape[la.index(label)] for label in a_free]
    b_free_shape = [b.shape[lb.index(label)] for label in b_free]
    contract_count = _product(a.shape[la.index(label)] for label in contract)
    batch_count = _product(batch_shape)

    a3 = a2.reshape((batch_count, _product(a_free_shape), contract_count))
    b3 = b2.reshape((batch_count, contract_count, _product(b_free_shape)))
    result = np.matmul(a3, b3).reshape(tuple(batch_shape) + tuple(a_free_shape) + tuple(b_free_shape))
    return result, batch + a_free + b_free


def _validate_label(ch):
    if not (len(ch) == 1 and ch.isalpha() and ch.isascii()):
        raise ValueError(f"invalid subscript {ch!r} in einsum")


def _operand_labels(part, ndim):
    """`(explicit_labels, ellipsis_rank)`: `ellipsis_rank` is `None` without an ellipsis, else
    the number of dimensions `...` stands for in this operand."""
    if "..." in part:
        if part.count("...") > 1:
            raise NotImplementedError("einsum: at most one ellipsis per operand is supported")
        before, after = part.split("...")
        for ch in before + after:
            _validate_label(ch)
        fixed = len(before) + len(after)
        if fixed > ndim:
            raise ValueError("einsum subscript has too many indices for the operand")
        return list(before), list(after), ndim - fixed
    for ch in part:
        _validate_label(ch)
    if len(part) != ndim:
        raise ValueError(
            f"einsum subscript has {len(part)} indices but the operand has {ndim} dimensions"
        )
    return list(part), [], None


def _split_equation(subscripts):
    text = subscripts.replace(" ", "")
    if text.count("->") > 1:
        raise ValueError("einsum subscripts string contains too many '->'")
    if "->" in text:
        inputs_text, output_text = text.split("->")
        return inputs_text.split(","), output_text, True
    return text.split(","), None, False


def einsum(subscripts, *operands, optimize=False, **kwargs):
    """Evaluate the Einstein-summation `subscripts` over `operands`.

    Supports explicit (``'ij,jk->ik'``) and implicit (``'ij,jk'``, output = indices appearing
    exactly once, sorted) subscripts, repeated indices within one operand (``'ii->i'`` for a
    diagonal, ``'ii'`` for a trace), sums over indices absent from the output, outer products,
    transposes, batched contractions (``'bij,bjk->bik'``), and any number of operands. `optimize`
    is accepted and ignored: every contraction already avoids the full product space. A leading
    ``'...'`` is supported when every operand that has one gives it the same number of axes.
    """
    if not isinstance(subscripts, str):
        raise NotImplementedError("einsum() only supports a subscripts string")
    if kwargs:
        raise NotImplementedError(f"einsum() with {sorted(kwargs)} is not supported")
    arrays = [np.asanyarray(operand) for operand in operands]
    input_parts, output_text, explicit = _split_equation(subscripts)
    if len(input_parts) != len(arrays):
        raise ValueError(
            f"einsum(): {len(input_parts)} operand string(s) but {len(arrays)} array(s) given"
        )

    parsed = [_operand_labels(part, arr.ndim) for part, arr in zip(input_parts, arrays)]
    ranks = {rank for _, _, rank in parsed if rank is not None}
    if len(ranks) > 1:
        raise NotImplementedError("einsum: ellipsis must cover the same rank in every operand")
    ellipsis_labels = [f"\0{i}" for i in range(ranks.pop() if ranks else 0)]
    uses_ellipsis = any(rank is not None for _, _, rank in parsed)

    full_labels = [
        before + (ellipsis_labels if rank is not None else []) + after
        for before, after, rank in parsed
    ]

    if explicit:
        if "..." in output_text:
            if output_text.count("...") > 1:
                raise NotImplementedError("einsum: at most one ellipsis in the output")
            before_out, after_out = output_text.split("...")
            for ch in before_out + after_out:
                _validate_label(ch)
            output_labels = list(before_out) + ellipsis_labels + list(after_out)
        else:
            if uses_ellipsis and ellipsis_labels:
                raise NotImplementedError("einsum: explicit output must include '...' when an input does")
            for ch in output_text:
                _validate_label(ch)
            output_labels = list(output_text)
        if len(set(output_labels)) != len(output_labels):
            raise ValueError("einsum output subscripts must not contain repeated indices")
        available = {label for labels in full_labels for label in labels}
        if any(label not in available for label in output_labels):
            raise ValueError("einsum output subscript not found among the input subscripts")
    else:
        counts = {}
        for labels in full_labels:
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
        named = sorted(label for label, count in counts.items() if count == 1 and not label.startswith("\0"))
        output_labels = ellipsis_labels + named

    deduped = [_dedup_labels(arr, labels) for arr, labels in zip(arrays, full_labels)]

    running, running_labels = deduped[0]
    for index in range(1, len(deduped)):
        current, current_labels = deduped[index]
        keep = set(output_labels)
        for _, later_labels in deduped[index + 1 :]:
            keep.update(later_labels)
        running, running_labels = _pairwise_contract(running, running_labels, current, current_labels, keep)

    extra = tuple(i for i, label in enumerate(running_labels) if label not in output_labels)
    if extra:
        running = np.sum(running, axis=extra)
        running_labels = [label for i, label in enumerate(running_labels) if i not in extra]

    if sorted(running_labels) != sorted(output_labels):
        raise ValueError("einsum: could not match operand subscripts to the requested output")
    perm = [running_labels.index(label) for label in output_labels]
    result = running.transpose(perm) if perm else running
    return result[()] if result.ndim == 0 else result
