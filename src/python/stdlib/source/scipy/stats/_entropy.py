"""Shannon and relative entropy, following SciPy 1.18's ``scipy/stats/_entropy.py``.

``differential_entropy`` is not implemented.
"""

import math

import numpy as np
from scipy import special
from scipy._lib._util import _promote
from scipy.stats._axis_nan_policy import _axis_nan_policy_factory

__all__ = ["entropy"]


_entropy_policy = _axis_nan_policy_factory(
    lambda x: x,
    n_outputs=1,
    result_to_tuple=lambda x, _: (x,),
    paired=True,
    too_small=-1,  # entropy doesn't have too small inputs
)


def entropy(pk, qk=None, base=None, axis=0, *, nan_policy="propagate", keepdims=False):
    """Calculate the Shannon entropy or relative entropy of the given distribution(s).

    ``pk`` and ``qk`` are normalized to sum to 1 along ``axis``. The result is in nats unless
    ``base`` is given.
    """
    samples = [pk] if qk is None else [pk, qk]
    return _entropy_policy(_entropy, samples, {"base": base}, axis, nan_policy, keepdims)


def _entropy(pk, qk=None, base=None, axis=0):
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")

    pk, qk = _promote(pk, qk, broadcast=True)
    with np.errstate(invalid="ignore"):
        if qk is not None:
            qk = qk / np.sum(qk, axis=axis, keepdims=True)
        pk = pk / np.sum(pk, axis=axis, keepdims=True)

    if qk is None:
        vec = special.entr(pk)
    else:
        vec = special.rel_entr(pk, qk)

    S = np.sum(vec, axis=axis)
    if base is not None:
        S /= math.log(base)
    return S
