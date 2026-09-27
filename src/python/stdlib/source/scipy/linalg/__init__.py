"""shellsim's ``scipy.linalg``.

The implemented surface follows SciPy 1.18 for real ``float32`` and ``float64`` input (other
real dtypes are cast, as in SciPy):

- linear systems: ``solve``, ``solve_triangular``, ``solve_banded``, ``solve_circulant``,
  ``inv``, ``det``, ``lstsq``, ``pinv`` and ``pinvh``;
- decompositions: ``lu``, ``lu_factor``, ``lu_solve``, ``cholesky``, ``cho_factor``,
  ``cho_solve``, ``qr``, ``svd``, ``svdvals``, ``eigh``, ``eigvalsh`` and ``polar``, with
  ``diagsvd``, ``orth``, ``null_space``, ``subspace_angles`` and ``orthogonal_procrustes``;
- matrix functions: ``expm``, ``coshm``, ``sinhm``, ``tanhm`` and ``khatri_rao``;
- ``norm``, ``bandwidth``, ``issymmetric``, ``ishermitian``, the special matrix constructors,
  ``LinAlgError`` and ``LinAlgWarning``;
- ``get_lapack_funcs``, ``get_blas_funcs`` and ``find_best_blas_type``, with the ``lapack`` and
  ``blas`` modules (see their documentation for the routines they provide).

Complex input raises ``NotImplementedError``, as does accessing any other name in SciPy's
``scipy.linalg.__all__``. See ``docs/scipy.md`` for how results can differ from SciPy's.
"""

from scipy.linalg import blas, lapack
from scipy.linalg._basic import (
    det,
    inv,
    lstsq,
    pinv,
    pinvh,
    solve,
    solve_banded,
    solve_circulant,
    solve_triangular,
)
from scipy.linalg._decomp import eigh, eigvalsh
from scipy.linalg._decomp_cholesky import cho_factor, cho_solve, cholesky
from scipy.linalg._decomp_lu import lu, lu_factor, lu_solve
from scipy.linalg._decomp_polar import polar
from scipy.linalg._decomp_qr import qr
from scipy.linalg._decomp_svd import diagsvd, null_space, orth, subspace_angles, svd, svdvals
from scipy.linalg._matfuncs import coshm, expm, khatri_rao, sinhm, tanhm
from scipy.linalg._misc import (
    LinAlgError,
    LinAlgWarning,
    bandwidth,
    ishermitian,
    issymmetric,
    norm,
)
from scipy.linalg._procrustes import orthogonal_procrustes
from scipy.linalg._special_matrices import (
    block_diag,
    circulant,
    companion,
    convolution_matrix,
    dft,
    fiedler,
    fiedler_companion,
    hadamard,
    hankel,
    helmert,
    hilbert,
    invhilbert,
    invpascal,
    leslie,
    pascal,
    toeplitz,
)
from scipy.linalg.blas import find_best_blas_type, get_blas_funcs
from scipy.linalg.lapack import get_lapack_funcs

__all__ = [
    "LinAlgError",
    "LinAlgWarning",
    "bandwidth",
    "blas",
    "block_diag",
    "cho_factor",
    "cho_solve",
    "cholesky",
    "circulant",
    "companion",
    "convolution_matrix",
    "coshm",
    "det",
    "dft",
    "diagsvd",
    "eigh",
    "eigvalsh",
    "expm",
    "fiedler",
    "fiedler_companion",
    "find_best_blas_type",
    "get_blas_funcs",
    "get_lapack_funcs",
    "hadamard",
    "hankel",
    "helmert",
    "hilbert",
    "inv",
    "invhilbert",
    "invpascal",
    "ishermitian",
    "issymmetric",
    "khatri_rao",
    "lapack",
    "leslie",
    "lstsq",
    "lu",
    "lu_factor",
    "lu_solve",
    "norm",
    "null_space",
    "orth",
    "orthogonal_procrustes",
    "pascal",
    "pinv",
    "pinvh",
    "polar",
    "qr",
    "sinhm",
    "solve",
    "solve_banded",
    "solve_circulant",
    "solve_triangular",
    "subspace_angles",
    "svd",
    "svdvals",
    "tanhm",
    "toeplitz",
]

# The rest of SciPy 1.18's `scipy.linalg.__all__`.
_UNSUPPORTED = frozenset(
    (
        "basic", "cdf2rdf", "cho_solve_banded", "cholesky_banded", "clarkson_woodruff_transform",
        "cosm", "cossin", "cython_blas", "cython_lapack", "decomp", "decomp_cholesky",
        "decomp_lu", "decomp_qr", "decomp_schur", "decomp_svd", "eig", "eig_banded",
        "eigh_tridiagonal", "eigvals", "eigvals_banded", "eigvalsh_tridiagonal", "expm_cond",
        "expm_frechet", "fractional_matrix_power", "funm", "hessenberg", "ldl", "logm",
        "matfuncs", "matmul_toeplitz", "matrix_balance", "misc", "ordqz", "qr_delete",
        "qr_insert", "qr_multiply", "qr_update", "qz", "rq", "rsf2csf", "schur", "signm",
        "sinm", "solve_continuous_are", "solve_continuous_lyapunov", "solve_discrete_are",
        "solve_discrete_lyapunov", "solve_lyapunov", "solve_sylvester", "solve_toeplitz",
        "solveh_banded", "special_matrices", "sqrtm", "tanm",
    )
)


def __getattr__(name):
    if name in _UNSUPPORTED:
        raise NotImplementedError(f"scipy.linalg.{name} is not supported by shellsim's SciPy")
    raise AttributeError(f"module 'scipy.linalg' has no attribute '{name}'")
