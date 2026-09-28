//! Observable NumPy compatibility tests for the deliberately generic ndarray implementation.
//!
//! The cases emphasize the shape/stride contract, view aliasing, broadcasting, reductions, and
//! fixed-width scalar behavior, and explicit failure boundaries rather than performance.

use shellsim::{python, Environment, Limits};

fn run(source: &str) -> (i32, Vec<u8>, Vec<u8>) {
    run_with_environment(Environment::new(), source)
}

fn run_with_environment(mut environment: Environment, source: &str) -> (i32, Vec<u8>, Vec<u8>) {
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status = python::run_python(
        &mut environment,
        &["python3.14".into(), "-c".into(), source.into()],
        Vec::new(),
        &mut stdout,
        &mut stderr,
    );
    (status, stdout, stderr)
}

fn assert_fails_with(source: &str, expected: &str) {
    let (status, _, stderr) = run(source);
    assert_ne!(status, 0, "operation unexpectedly succeeded: {source}");
    let stderr = String::from_utf8_lossy(&stderr);
    assert!(
        stderr.contains(expected),
        "expected {expected:?} in failure for {source:?}, got {stderr:?}"
    );
}

/// Asserts that `stderr` holds one `RuntimeWarning` per listed script line, in order.
fn assert_warning_lines(stderr: &[u8], lines: &[usize]) {
    let stderr = String::from_utf8_lossy(stderr);
    let reported: Vec<&str> = stderr.lines().collect();
    assert_eq!(reported.len(), lines.len(), "{stderr:?}");
    for (report, line) in reported.iter().zip(lines) {
        let prefix = format!("<string>:{line}: RuntimeWarning: ");
        assert!(report.starts_with(&prefix), "{stderr:?}");
    }
}

#[test]
fn construction_indexing_and_views_share_storage() {
    let source = r#"import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape, a.ndim, a.size, a.dtype)
print(a, isinstance(a, np.ndarray), bool(np.array([1])))
print(a.tolist(), a[1, 2], a[0].tolist())
t = a.T
t[1, 0] = 20
print(t.shape, t.tolist(), a.tolist())
r = a.reshape(3, 2)
r[2, 1] = 60
print(r.tolist(), a.tolist())
print(a.flatten().tolist(), a.ravel().tolist())
print(a.reshape(-1).tolist(), [row.tolist() for row in a])
print(a.astype(float).dtype, a.astype(float).tolist())
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"(2, 3) 2 6 int64\n[[1 2 3]\n [4 5 6]] True True\n[[1, 2, 3], [4, 5, 6]] 6 [1, 2, 3]\n(3, 2) [[1, 4], [20, 5], [3, 6]] [[1, 20, 3], [4, 5, 6]]\n[[1, 20], [3, 4], [5, 60]] [[1, 20, 3], [4, 5, 60]]\n[1, 20, 3, 4, 5, 60] [1, 20, 3, 4, 5, 60]\n[1, 20, 3, 4, 5, 60] [[1, 20, 3], [4, 5, 60]]\nfloat64 [[1.0, 20.0, 3.0], [4.0, 5.0, 60.0]]\n".to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn broadcasting_reductions_and_naive_matrix_multiplication_work() {
    let source = r#"import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])
print((a + b).tolist())
print((a + [10, 20, 30]).tolist())
print((2 * a - 1).tolist())
print((a / 2).tolist())
print(a.sum(), a.sum(axis=0).tolist(), np.sum(a, axis=1).tolist())
print(a.mean(), a.min(), a.max())
left = np.array([[1, 2], [3, 4]])
right = np.array([[5, 6], [7, 8]])
print(np.matmul(left, right).tolist())
print((left @ right).tolist())
print(np.dot(np.array([1, 2, 3]), np.array([4, 5, 6])))
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"[[11, 22, 33], [14, 25, 36]]\n[[11, 22, 33], [14, 25, 36]]\n[[1, 3, 5], [7, 9, 11]]\n[[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]\n21 [5, 7, 9] [6, 15]\n3.5 1 6\n[[19, 22], [43, 50]]\n[[19, 22], [43, 50]]\n32\n".to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn constructors_and_dtype_conversion_are_coherent() {
    let source = r#"import numpy as np
print(np.zeros((2, 2)).tolist())
print(np.ones(3, dtype=int).tolist())
print(np.full((2, 2), 7).tolist())
print(np.arange(1, 6, 2).tolist())
print(np.arange(0, 1, 0.25, dtype=float).tolist())
print(np.asarray([0, 1, 2], dtype=bool).tolist())
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"[[0.0, 0.0], [0.0, 0.0]]\n[1, 1, 1]\n[[7, 7], [7, 7]]\n[1, 3, 5]\n[0.0, 0.25, 0.5, 0.75]\n[False, True, True]\n".to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn deterministic_random_padding_and_npy_persistence_use_the_array_contract() {
    let source = r#"import numpy as np
from pathlib import Path

first = np.random.default_rng(123).standard_normal((2, 3), dtype=np.float32)
second = np.random.default_rng(123).standard_normal((2, 3), dtype=np.float32)
print(first.shape, first.dtype, np.array_equal(first, second))
values = np.random.default_rng(9).integers(2, 7, size=5, dtype=np.int16)
print(values.dtype, len(values), bool(np.all(values >= 2)), bool(np.all(values < 7)))

base = np.array([[1, 2], [3, 4]], dtype=np.int16)
print(np.pad(base, ((1, 0), (2, 1)), constant_values=-1).tolist())
print(np.pad(base, 1, mode='edge').tolist())

np.save(Path('/work/model'), first)
loaded = np.load('/work/model.npy')
print(loaded.shape, loaded.dtype, np.array_equal(first, loaded))
"#;
    assert_eq!(
        run(source),
        (
            0,
            concat!(
                "(2, 3) float32 True\n",
                "int16 5 True True\n",
                "[[-1, -1, -1, -1, -1], [-1, -1, 1, 2, -1], [-1, -1, 3, 4, -1]]\n",
                "[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]\n",
                "(2, 3) float32 True\n",
            )
            .as_bytes()
            .to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn registered_scalars_keep_fixed_width_numeric_semantics() {
    let source = r#"import numpy as np
x = np.int64(9223372036854775807)
print(type(x), isinstance(x, np.int64), x + np.int64(1))
print(type(np.float64(1.5)), np.float64(1.5) + np.int64(2))
print(np.array([np.int64(2), 3]).dtype, np.array([np.int64(2), 3]).tolist())
print(np.array([1], dtype=np.int64).dtype)
"#;
    let (status, stdout, stderr) = run(source);
    assert_eq!(
        (status, stdout),
        (
            0,
            b"<class 'numpy.int64'> True -9223372036854775808\n<class 'numpy.float64'> 3.5\nint64 [2, 3]\nint64\n".to_vec(),
        )
    );
    assert_warning_lines(&stderr, &[3]);

    assert_fails_with(
        "import numpy as np\nnp.int64(9223372036854775808)",
        "OverflowError",
    );
}

#[test]
fn ordinary_fixed_width_dtypes_share_one_registered_scalar_contract() {
    let source = r#"import numpy as np
for dtype in (np.bool_, np.int8, np.int16, np.int32, np.int64,
              np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64):
    a = np.array([0, 1], dtype=dtype)
    print(a.dtype, type(a[0]) is dtype)
print(np.byte is np.int8, np.short is np.int16, np.intc is np.int32)
print(np.ubyte is np.uint8, np.ushort is np.uint16, np.uintc is np.uint32)
print(np.single is np.float32, np.double is np.float64)
print(np.bool is np.bool_, np.int_ is np.int64, np.intp is np.int64, np.longlong is np.int64)
print(np.uint is np.uint64, np.uintp is np.uint64, np.ulonglong is np.uint64)
for spelling in ('i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8', 'f4', 'f8'):
    print(np.array([1], dtype=spelling).dtype)
print(np.array([]).dtype)
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"bool True\nint8 True\nint16 True\nint32 True\nint64 True\nuint8 True\nuint16 True\nuint32 True\nuint64 True\nfloat32 True\nfloat64 True\nTrue True True\nTrue True True\nTrue True\nTrue True True True\nTrue True True\nint8\nint16\nint32\nint64\nuint8\nuint16\nuint32\nuint64\nfloat32\nfloat64\nfloat64\n".to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn fixed_width_arithmetic_uses_the_dtype_table() {
    let source = r#"import numpy as np
print(np.int8(127) + np.int8(1), type(np.int8(1) + np.uint8(1)))
print(np.uint8(255) + np.uint8(1), type(np.int16(1) + np.uint16(1)))
print(type(np.int32(1) + np.uint32(1)), type(np.int64(1) + np.uint64(1)))
print(np.float32(16777216) + np.float32(1))
print(type(np.float32(1) + np.int16(1)), type(np.float32(1) + np.int32(1)))
print(type(np.float32(1) + np.float64(1)), type(np.float64(1) + np.float32(1)))
print(np.array([np.float32(1), 2.5]).dtype, np.array([2.5, np.float32(1)]).dtype)
print(np.float32(0.1))
a = np.array([127, -128], dtype=np.int8) + np.array([1, 1], dtype=np.int8)
print(a.dtype, a.tolist())
b = np.array([255], dtype=np.uint8) + np.array([1], dtype=np.uint8)
print(b.dtype, b.tolist(), (~b).tolist())
for value in (np.array([100], dtype=np.int8) * 2,
              np.array([1], dtype=np.uint8) + 1,
              np.array([1], dtype=np.uint64) + 1,
              np.array([1.5], dtype=np.float32) * 2.0):
    print(value.dtype, value.tolist())
"#;
    // The two scalar overflows warn, reported against the script's lines.
    let (status, stdout, stderr) = run(source);
    assert_eq!(
        (status, stdout),
        (
            0,
            b"-128 <class 'numpy.int16'>\n0 <class 'numpy.int32'>\n<class 'numpy.int64'> <class 'numpy.float64'>\n1.6777216e+07\n<class 'numpy.float32'> <class 'numpy.float64'>\n<class 'numpy.float64'> <class 'numpy.float64'>\nfloat64 float64\n0.1\nint8 [-128, -127]\nuint8 [0] [255]\nint8 [-56]\nuint8 [2]\nuint64 [2]\nfloat32 [3.0]\n".to_vec(),
        )
    );
    assert_warning_lines(&stderr, &[2, 3]);
}

#[test]
fn reductions_and_indices_respect_fixed_width_classes() {
    let source = r#"import numpy as np
signed = np.array([120, 10], dtype=np.int8)
unsigned = np.array([250, 10], dtype=np.uint8)
print(signed.sum(), type(signed.sum()), signed.cumsum().dtype, signed.cumsum().tolist())
print(unsigned.sum(), type(unsigned.sum()), unsigned.cumsum().dtype, unsigned.cumsum().tolist())
floating = np.array([1, 2], dtype=np.float32)
print(type(floating.mean()), floating.mean(), type(floating.var()))
matrix = np.array([[100, 100]], dtype=np.int8)
product = np.matmul(matrix, np.array([[2], [2]], dtype=np.int8))
print(product.dtype, product.tolist())
rows = np.array([[1], [2], [3]])
print(rows[np.array([2, 0], dtype=np.uint8)].tolist())
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"130 <class 'numpy.int64'> int64 [120, 130]\n260 <class 'numpy.uint64'> uint64 [250, 260]\n<class 'numpy.float32'> 1.5 <class 'numpy.float32'>\nint8 [[-112]]\n[[3], [1]]\n".to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn fixed_width_conversions_reject_values_outside_the_declared_dtype() {
    for source in [
        "import numpy as np\nnp.int8(128)",
        "import numpy as np\nnp.int16(-32769)",
        "import numpy as np\nnp.uint8(-1)",
        "import numpy as np\nnp.uint32(4294967296)",
        "import numpy as np\nnp.array([1], dtype=np.uint8) + 256",
    ] {
        assert_fails_with(source, "OverflowError");
    }

    assert_fails_with(
        "import numpy as np\nnp.array([1], dtype='numpy.int8')",
        "TypeError",
    );
}

#[test]
fn boolean_and_integer_array_indexing_gather_and_assign() {
    let source = r#"import numpy as np
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a[[2, 0]].tolist())
print(a[np.array([True, False, True])].tolist())
print(a[np.array([[True, False], [False, True], [True, False]])].tolist())
a[[0, 2]] = [[10, 20], [50, 60]]
a[np.array([[False, True], [True, False], [False, False]])] = 9
print(a.tolist())
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"[[5, 6], [1, 2]]\n[[1, 2], [5, 6]]\n[1, 4, 5]\n[[10, 9], [9, 4], [50, 60]]\n"
                .to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn common_statistics_share_the_generic_reduction_model() {
    let source = r#"import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.prod(), np.prod(a, axis=0).tolist())
print(a.var(), a.std(), np.median(a))
print(np.var(a, axis=0).tolist(), np.std(a, axis=1).tolist())
print(a.all(), np.any(np.array([0, 0, 1])))
print(np.all(a, axis=0).tolist(), np.any(a, axis=1).tolist())
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"720 [4, 10, 18]\n2.9166666666666665 1.707825127659933 3.5\n[2.25, 2.25, 2.25] [0.816496580927726, 0.816496580927726]\nTrue True\n[True, True, True] [True, True]\n".to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn comparisons_unary_maps_and_boolean_expressions_compose() {
    let source = r#"import numpy as np
a = np.array([-2, -1, 0, 1, 2])
print((a > 0).tolist(), (a <= 0).tolist(), (a != 1).tolist())
print(a[a > 0].tolist())
print((-a).tolist(), abs(a).tolist(), (~(a > 0)).tolist())
print(np.sqrt([1, 4, 9]).tolist(), np.floor([1.2, -1.2]).tolist())
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"[False, False, False, True, True] [True, True, True, False, False] [True, True, True, False, True]\n[1, 2]\n[2, 1, 0, -1, -2] [2, 1, 0, 1, 2] [True, True, True, False, False]\n[1.0, 2.0, 3.0] [1.0, -2.0]\n".to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn first_axis_slices_are_signed_stride_views() {
    let source = r#"import numpy as np
a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
print(a[1:4:2].tolist(), a[::-1].tolist(), a[-3:-1].tolist())
view = a[::-1]
view[0, 1] = 80
print(a.tolist(), view.tolist())
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"[[3, 4], [7, 8]] [[7, 8], [5, 6], [3, 4], [1, 2]] [[3, 4], [5, 6]]\n[[1, 2], [3, 4], [5, 6], [7, 80]] [[7, 80], [5, 6], [3, 4], [1, 2]]\n".to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn multidimensional_basic_slices_are_views_for_reads_and_writes() {
    let source = r#"import numpy as np
values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(values[:, 1].tolist())
print(values[1:, ::-1].tolist())
values[:2, 1:] = [[20, 30], [50, 60]]
print(values.tolist())
values[:, 0] = 0
print(values.tolist())
values[2] = [70, 80, 90]
print(values.tolist())
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"[2, 5, 8]\n[[6, 5, 4], [9, 8, 7]]\n[[1, 20, 30], [4, 50, 60], [7, 8, 9]]\n[[0, 20, 30], [0, 50, 60], [0, 8, 9]]\n[[0, 20, 30], [0, 50, 60], [70, 80, 90]]\n".to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn ellipsis_indices_keep_every_unindexed_axis_for_reads_and_writes() {
    let source = r#"import numpy as np
a = np.arange(24).reshape(2, 3, 4)
print(a[..., 0].tolist(), a[0, ...].shape, a[...].shape)
print(a[..., 1, 2].tolist(), a[1, ..., 3].tolist(), a[..., 1:3].shape)
b = np.zeros((2, 2, 2), dtype=np.int64)
b[..., 0] = 1
print(b.tolist())
z = np.array(5)
view = z[...]
view[...] = 7
print(view.shape, z.tolist(), a[1, 2, 3, ...].shape)
try:
    a[..., 0, ...]
except IndexError as error:
    print(error)
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"[[0, 4, 8], [12, 16, 20]] (3, 4) (2, 3, 4)\n[6, 18] [15, 19, 23] (2, 3, 2)\n[[[1, 0], [1, 0]], [[1, 0], [1, 0]]]\n() 7 ()\nan index can only have a single ellipsis ('...')\n".to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn shape_views_constructors_and_joining_use_shared_kernels() {
    let source = r#"import numpy as np
a = np.array([[[1], [2]]])
print(a.squeeze().shape, np.expand_dims(a.squeeze(), 1).shape)
print(np.swapaxes(np.array([[1, 2], [3, 4]]), 0, 1).tolist())
print(np.broadcast_to(np.array([1, 2]), (3, 2)).tolist())
print(np.zeros_like(np.array([1, 2])).tolist(), np.full_like(np.array([1.0]), 3).tolist())
print(np.linspace(0, 1, 5).tolist())
print(np.eye(2, 3, 1).tolist(), np.identity(2, dtype=int).tolist())
print(np.concatenate(([1, 2], [3, 4])).tolist())
print(np.stack(([1, 2], [3, 4]), axis=1).tolist())
print(np.vstack(([1, 2], [3, 4])).tolist(), np.hstack(([1, 2], [3, 4])).tolist())
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"(2,) (2, 1)\n[[1, 3], [2, 4]]\n[[1, 2], [1, 2], [1, 2]]\n[0, 0] [3.0]\n[0.0, 0.25, 0.5, 0.75, 1.0]\n[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] [[1, 0], [0, 1]]\n[1, 2, 3, 4]\n[[1, 3], [2, 4]]\n[[1, 2], [3, 4]] [1, 2, 3, 4]\n".to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn selection_cumulative_and_product_helpers_cover_common_array_code() {
    let source = r#"import numpy as np
a = np.array([[1, 4, 2], [3, 0, 5]])
print(np.where(a > 2, a, -1).tolist())
print(np.minimum(a, 2).tolist(), np.maximum(a, 2).tolist())
print(np.clip(a, 1, 3).tolist())
print(a.argmin(), a.argmax(), np.argmin(a, axis=0).tolist(), np.argmax(a, axis=1).tolist())
print(a.cumsum().tolist(), np.cumprod(a, axis=1).tolist())
print(np.inner([1, 2], [3, 4]), np.outer([1, 2], [3, 4]).tolist())
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"[[-1, 4, -1], [3, -1, 5]]\n[[1, 2, 2], [2, 0, 2]] [[2, 4, 2], [3, 2, 5]]\n[[1, 3, 2], [3, 1, 3]]\n4 5 [0, 1, 0] [1, 2]\n[1, 5, 7, 10, 10, 15] [[1, 4, 8], [3, 0, 0]]\n11 [[3, 4], [6, 8]]\n".to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn workload_driven_numeric_helpers_use_the_generic_array_contract() {
    let source = r#"import numpy as np
print(np.pi > 3.14, np.inf > 1e300, np.__version__)
print(np.log2([1, 8]).tolist(), np.rint([1.5, 2.5, -1.5]).tolist())
print(np.sign([-3, 0, 4]).tolist())
print(np.isnan([0.0, np.nan]).tolist(), np.isinf([np.inf, 1.0]).tolist())
print(np.diag([1, 2, 3]).tolist())
print(np.diag([1, 2, 3], k=1).tolist())
print(np.diag([[1, 2, 3], [4, 5, 6]], k=1).tolist())
print(np.argsort([3, 1, 2], kind="stable").tolist())
print(np.argsort([[3, 1], [2, 0]]).tolist())
print(np.allclose([[1.0], [2.0]], [1.0, 2.0]))
print(np.percentile([0, 10, 20, 30], 25))
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"True True 2.5.3\n[0.0, 3.0] [2.0, 2.0, -2.0]\n[-1, 0, 1]\n[False, True] [True, False]\n[[1, 0, 0], [0, 2, 0], [0, 0, 3]]\n[[0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3], [0, 0, 0, 0]]\n[2, 6]\n[1, 2, 0]\n[[1, 0], [1, 0]]\nFalse\n7.5\n".to_vec(),
            Vec::new(),
        )
    );
}

#[test]
fn unsupported_or_invalid_array_operations_fail_explicitly() {
    for (source, expected) in [
        (
            "import numpy as np\nnp.array([[1], [2, 3]])",
            "ValueError",
        ),
        (
            "import numpy as np\nnp.array([1, 2]) + np.array([1, 2, 3])",
            "ValueError",
        ),
        (
            "import numpy as np\nnp.zeros((-1, 2))",
            "ValueError",
        ),
        (
            "import numpy as np\nnp.array([1], dtype='S3')",
            "NumPy dtype 'S3' is not supported",
        ),
        (
            "import numpy as np\nbool(np.array([1, 2]))",
            "ValueError",
        ),
        (
            "import numpy as np\na = np.zeros((2, 2)); a[[0, 1]] = [1, 2, 3]",
            "ValueError",
        ),
        (
            "import numpy as np\na = np.zeros((2, 2)); a[...] = np.ones((2, 2, 2))",
            "ValueError",
        ),
        (
            "import numpy as np\nnp.zeros([1] * 65)",
            "ValueError",
        ),
        (
            "import numpy as np\nvalue = 1\nfor _ in range(65):\n    value = [value]\nnp.array(value)",
            "ValueError",
        ),
        (
            "import numpy as np\nnp.linspace(0, 1j, 3, dtype=float)",
            "linspace() with complex bounds and a real dtype is not supported",
        ),
    ] {
        assert_fails_with(source, expected);
    }
}

/// Storage is little-endian whatever the dtype's byte order, so operations that would expose
/// the difference are rejected rather than answered with wrong bytes.
#[test]
fn byte_order_frontier_fails_explicitly() {
    for (source, expected) in [
        (
            "import numpy as np\nnp.arange(2.0).view('>f8')",
            "ndarray.view between byte orders is not supported",
        ),
        (
            "import numpy as np\nnp.array([1j], dtype='>c8').view('>f8')",
            "ndarray.view between byte orders is not supported",
        ),
        (
            "import numpy as np\nnp.dtype('>U3')",
            "big-endian dtype '>U3' is not supported",
        ),
        (
            "import numpy as np\nnp.array([1, 'a'], dtype=object).tobytes()",
            "ndarray.tobytes of an object array is not supported",
        ),
    ] {
        assert_fails_with(source, expected);
    }
}

/// `frombuffer` copies because arrays cannot share a `bytearray`'s storage; the copy is
/// read-only so a write NumPy would pass through fails instead of silently diverging.
#[test]
fn frombuffer_over_a_bytearray_is_a_read_only_copy() {
    let source = r#"import numpy as np
data = bytearray(b"\x01\x00\x02\x00")
values = np.frombuffer(data, dtype="<i2")
data[0] = 9
print(values.tolist(), values.flags.writeable)
try:
    values[0] = 5
except ValueError as error:
    print(error)
"#;
    let (status, stdout, stderr) = run(source);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(
        String::from_utf8_lossy(&stdout),
        "[1, 2] False\nassignment destination is read-only\n"
    );
}

/// Object arrays load through a restricted unpickler. A pickle that names any global other
/// than NumPy's reconstructors and a few builtins fails before anything is imported or called.
#[test]
fn npy_object_pickles_resolve_only_numpy_globals() {
    let source = r#"import io
import numpy as np
from numpy.lib import format

def load(payload):
    buffer = io.BytesIO()
    format.write_array_header_1_0(buffer, {"descr": "|O", "fortran_order": False, "shape": (1,)})
    buffer.write(payload)
    buffer.seek(0)
    try:
        np.load(buffer, allow_pickle=True)
    except Exception as error:
        print(type(error).__name__, error)

load(b"\x80\x04cos\nsystem\n\x8c\x07echo hi\x85R.")
load(b"\x80\x04\x8c\x08builtins\x8c\x04eval\x93\x8c\x011\x85R.")
load(b"\x80\x04cnumpy\nndarray\n)R.")
load(b"\x80\x04]\x94(K\x01")
buffer = io.BytesIO()
np.save(buffer, np.array([1, "a"], dtype=object))
buffer.seek(0)
try:
    np.load(buffer)
except ValueError as error:
    print(error)
"#;
    let (status, stdout, stderr) = run(source);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(
        String::from_utf8_lossy(&stdout),
        concat!(
            "UnpicklingError global 'os.system' is forbidden\n",
            "UnpicklingError global 'builtins.eval' is forbidden\n",
            "UnpicklingError only NumPy array and scalar reconstructors may be called\n",
            "UnpicklingError pickle data was truncated\n",
            "Object arrays cannot be loaded when allow_pickle=False\n",
        )
    );
}

/// shellsim memoizes equal strings by value because short strings have no identity. CPython
/// memoizes by identity, so NumPy writes each computed duplicate again; both files load to
/// equal arrays.
#[test]
fn npy_object_pickles_write_equal_strings_once() {
    let source = r#"import io
import numpy as np
values = np.empty(3, dtype=object)
values[:] = [str(10), str(10), "10"]
buffer = io.BytesIO()
np.save(buffer, values)
print(buffer.getvalue().count(b"\x8c\x0210"))
buffer.seek(0)
print(np.load(buffer, allow_pickle=True).tolist())
"#;
    let (status, stdout, stderr) = run(source);
    assert_eq!(status, 0, "{}", String::from_utf8_lossy(&stderr));
    assert_eq!(String::from_utf8_lossy(&stdout), "1\n['10', '10', '10']\n");
}

#[test]
fn numpy_file_io_frontier_fails_explicitly() {
    for (source, expected) in [
        (
            "np.genfromtxt(io.StringIO('a b\\n1 2'), names=True)",
            "genfromtxt with names= builds structured arrays",
        ),
        (
            "np.genfromtxt(io.StringIO('1 2'), usemask=True)",
            "masked arrays are not supported by shellsim's NumPy",
        ),
        (
            "np.genfromtxt(io.StringIO('1 x'), dtype=None, encoding='utf-8')",
            "genfromtxt columns of different types need a structured array",
        ),
        (
            "np.loadtxt(io.StringIO('1 2'), dtype=[('a', int), ('b', int)])",
            "NumPy dtype [('a', <class 'int'>), ('b', <class 'int'>)] is not supported",
        ),
        (
            "np.loadtxt(io.StringIO('1 2'), dtype='i4,i4')",
            "NumPy dtype 'i4,i4' is not supported",
        ),
        (
            "b = io.BytesIO(); np.save(b, np.arange(2)); b.seek(0); np.load(b, mmap_mode='r')",
            "memory-mapped arrays are not supported by shellsim's NumPy",
        ),
        (
            "from numpy.lib import format\nb = io.BytesIO()\nformat.write_array_header_1_0(b, {'descr': [('a', '<i4')], 'fortran_order': False, 'shape': (1,)})\nb.write(bytes(4)); b.seek(0); np.load(b)",
            "structured dtypes are not supported by shellsim's NumPy",
        ),
        (
            "class Point: pass\na = np.empty(1, dtype=object); a[0] = Point(); np.save(io.BytesIO(), a)",
            "np.save cannot pickle 'Point' array elements in shellsim",
        ),
    ] {
        assert_fails_with(&format!("import io\nimport numpy as np\n{source}"), expected);
    }
}

#[test]
fn array_allocation_obeys_the_modeled_memory_limit() {
    // Importing NumPy fits in 8 MiB; the array needs 32 MB.
    let environment = Environment::with_limits(Limits {
        cpu: 100_000_000,
        memory: 8 * 1024 * 1024,
        disk: 1024 * 1024,
        output: 1024,
    });
    let source = "import numpy as np\nprint('ready')\nnp.zeros((2000, 2000))";
    let (status, stdout, stderr) = run_with_environment(environment, source);
    assert_eq!((status, stdout.as_slice()), (137, b"ready\n".as_slice()));
    assert!(stderr.is_empty());
}

#[test]
fn kron_and_block_results_obey_the_memory_limit() {
    // Importing NumPy fits in 8 MiB; each result holds 4 million float64 values (32 MB).
    for operation in ["np.kron(a, a)", "np.block([a] * 2000)"] {
        let environment = Environment::with_limits(Limits {
            cpu: 100_000_000,
            memory: 8 * 1024 * 1024,
            disk: 1024 * 1024,
            output: 1024,
        });
        let source = format!("import numpy as np\na = np.ones(2000)\nprint('ready')\n{operation}");
        let (status, stdout, stderr) = run_with_environment(environment, &source);
        assert_eq!(
            (status, stdout.as_slice()),
            (137, b"ready\n".as_slice()),
            "{operation}"
        );
        assert!(stderr.is_empty());
    }
}

#[test]
fn array_and_scalar_attributes_are_native_getters() {
    // Expected output was recorded from NumPy 2.5 on CPython 3.14.
    let source = r#"import numpy as np
a = np.array([[1, 2], [3, 4]], dtype=np.int16)
print(a.shape, a.ndim, a.size, a.dtype, a.T.tolist())
print(np.ndarray.shape)
try:
    a.T = a
except AttributeError as error:
    print(error)
x = np.float64(2.5)
print(x.real, x.imag, x.conjugate(), type(x.imag) is np.float64)
print(np.int8(3).real, np.int8(3).imag, type(np.int8(3).imag) is np.int8)
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"(2, 2) 2 4 int16 [[1, 3], [2, 4]]\n<attribute 'shape' of 'numpy.ndarray' objects>\nattribute 'T' of 'numpy.ndarray' objects is not writable\n2.5 0.0 2.5 True\n3 0 True\n".to_vec(),
            Vec::new()
        )
    );
}

#[test]
fn complex_arrays_support_arithmetic_components_and_reductions() {
    // Every assertion also holds under NumPy 2.5 on CPython 3.14.
    let source = r#"import numpy as np
x = np.array([1+2j, 3-4j], dtype=complex)
assert x.dtype == 'complex128'
assert np.array([1, 2j]).dtype == 'complex128'
assert np.array([1, 2], dtype='cdouble').dtype == 'complex128'
assert np.ones(2, dtype='c16')[0] == 1+0j
assert np.zeros(2, dtype=np.complex128)[0] == 0j
assert isinstance(np.complex128(1j), np.complex128)
assert isinstance(x[0], complex) and x[1] == 3-4j
assert x.tolist() == [1+2j, 3-4j]
assert x.real.tolist() == [1.0, 3.0] and x.real.dtype == 'float64'
assert x.imag.tolist() == [2.0, -4.0] and x.imag.dtype == 'float64'
assert np.real(x).tolist() == [1.0, 3.0]
assert np.imag(x).tolist() == [2.0, -4.0]
assert np.array([1, 2]).imag.tolist() == [0, 0]
assert np.array([1.5]).real.tolist() == [1.5]
y = x.conjugate()
assert y.tolist() == [1-2j, 3+4j]
assert np.array_equal(x.conj(), y) and np.array_equal(np.conj(x), y)
assert np.array_equal(np.conjugate(x), y)
assert np.array([1, 2]).conj().tolist() == [1, 2]
assert (x + x).tolist() == [2+4j, 6-8j]
assert (x - 1).tolist() == [2j, 2-4j]
assert (x * y).tolist() == [5+0j, 25+0j]
assert (x / 2).tolist() == [0.5+1j, 1.5-2j]
assert (x / x).tolist() == [1+0j, 1+0j]
assert (-x).tolist() == [-1-2j, -3+4j]
assert (np.array([1, 2]) + 1j).dtype == 'complex128'
assert np.abs(x).dtype == 'float64'
assert np.abs(x).tolist() == [5 ** 0.5, 5.0] and abs(x).tolist() == [5 ** 0.5, 5.0]
assert np.sum(x) == 4-2j and x.sum() == 4-2j
assert np.prod(x) == 11+2j
assert np.mean(x) == 2-1j and x.mean() == 2-1j
m = np.array([[1+1j, 2], [3, 4j]])
assert np.sum(m, axis=1).tolist() == [3+1j, 3+4j]
assert np.mean(m, axis=0).tolist() == [2+0.5j, 1+2j]
assert np.cumsum(x).tolist() == [1+2j, 4-2j]
assert np.dot(x, y) == 30+0j
assert (m @ np.array([[1], [1j]])).tolist() == [[1+3j], [-1+0j]]
assert (x == np.array([1+2j, 3+4j])).tolist() == [True, False]
assert (x != (1+2j)).tolist() == [False, True]
assert np.float64(2).real == 2 and np.float64(2).imag == 0
assert np.int8(3).conjugate() == 3
assert np.float64(2) * 1j == 2j and 1j * np.float64(2) == 2j
assert np.array(1j) == 1j
print('ok')
"#;
    assert_eq!(run(source), (0, b"ok\n".to_vec(), Vec::new()));
}

#[test]
fn complex_arrays_round_trip_through_npy_files() {
    // `reference` is the file NumPy 2.5 writes for np.array([1+2j, -0.5j]). NumPy pads the header
    // to 64 bytes and shellsim to 16; both are valid version 1.0 files, so only the payload is
    // compared byte for byte.
    let source = r#"import numpy as np
payload = (b"\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@"
    + b"\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x00\x00\x00\xe0\xbf")
reference = (b"\x93NUMPY\x01\x00v\x00{'descr': '<c16', 'fortran_order': False, 'shape': (2,), }"
    + b" " * 59 + b"\n" + payload)
np.save('complex.npy', np.array([1+2j, -0.5j]))
with open('complex.npy', 'rb') as handle:
    saved = handle.read()
assert "'descr': '<c16'" in saved[:64].decode('latin-1')
assert saved[-32:] == payload
with open('reference.npy', 'wb') as handle:
    handle.write(reference)
loaded = np.load('reference.npy')
assert loaded.dtype == 'complex128' and loaded.tolist() == [1+2j, -0.5j]
print('ok')
"#;
    assert_eq!(run(source), (0, b"ok\n".to_vec(), Vec::new()));
}

#[test]
fn complex_operations_without_a_real_value_domain_fail_explicitly() {
    let prelude = "import numpy as np\nx = np.array([1+2j, 3-4j])\n";
    for operation in [
        "np.floor(x)",
        "~x",
        "np.percentile(x, 50)",
        "np.array([1j], dtype=np.int32)",
    ] {
        assert_fails_with(&format!("{prelude}{operation}"), "TypeError");
    }
}
