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
            b"(2, 3) 2 6 int64\narray([[1, 2, 3], [4, 5, 6]]) True True\n[[1, 2, 3], [4, 5, 6]] 6 [1, 2, 3]\n(3, 2) [[1, 4], [20, 5], [3, 6]] [[1, 20, 3], [4, 5, 6]]\n[[1, 20], [3, 4], [5, 60]] [[1, 20, 3], [4, 5, 60]]\n[1, 20, 3, 4, 5, 60] [1, 20, 3, 4, 5, 60]\n[1, 20, 3, 4, 5, 60] [[1, 20, 3], [4, 5, 60]]\nfloat64 [[1.0, 20.0, 3.0], [4.0, 5.0, 60.0]]\n".to_vec(),
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
fn registered_scalars_keep_fixed_width_numeric_semantics() {
    let source = r#"import numpy as np
x = np.int64(9223372036854775807)
print(type(x), isinstance(x, np.int64), x + np.int64(1))
print(type(np.float64(1.5)), np.float64(1.5) + np.int64(2))
print(np.array([np.int64(2), 3]).dtype, np.array([np.int64(2), 3]).tolist())
print(np.array([1], dtype=np.int64).dtype)
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"<class 'numpy.int64'> True -9223372036854775808\n<class 'numpy.float64'> 3.5\nint64 [2, 3]\nint64\n".to_vec(),
            Vec::new(),
        )
    );

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
    assert_eq!(
        run(source),
        (
            0,
            b"-128 <class 'numpy.int16'>\n0 <class 'numpy.int32'>\n<class 'numpy.int64'> <class 'numpy.float64'>\n16777216.0\n<class 'numpy.float32'> <class 'numpy.float64'>\n<class 'numpy.float64'> <class 'numpy.float64'>\nfloat64 float64\n0.1\nint8 [-128, -127]\nuint8 [0] [255]\nint8 [-56]\nuint8 [2]\nuint64 [2]\nfloat32 [3.0]\n".to_vec(),
            Vec::new(),
        )
    );
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
        "import numpy as np\nnp.array([1], dtype='float16')",
        "unsupported numpy dtype",
    );
    assert_fails_with(
        "import numpy as np\nnp.array([1], dtype='numpy.int8')",
        "unsupported numpy dtype",
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
fn unsupported_or_invalid_array_operations_fail_explicitly() {
    for (source, expected) in [
        (
            "import numpy as np\nnp.array([[1], [2, 3]])",
            "setting an array element with a sequence",
        ),
        (
            "import numpy as np\nnp.array([1, 2]) + np.array([1, 2, 3])",
            "operands could not be broadcast together",
        ),
        (
            "import numpy as np\nnp.matmul(np.array([1, 2]), np.array([3, 4]))",
            "matmul requires aligned two-dimensional arrays",
        ),
        (
            "import numpy as np\nnp.zeros((-1, 2))",
            "negative dimensions are not allowed",
        ),
        (
            "import numpy as np\nnp.array([1], dtype='complex128')",
            "unsupported numpy dtype",
        ),
        (
            "import numpy as np\nbool(np.array([1, 2]))",
            "truth value of an array",
        ),
        (
            "import numpy as np\na = np.zeros((2, 2)); a[[0, 1]] = [1, 2, 3]",
            "operands could not be broadcast together",
        ),
        (
            "import numpy as np\na = np.zeros((2, 2)); a[[0, 1]] = [[[1, 2], [3, 4]]]",
            "assignment value cannot be broadcast to the indexed shape",
        ),
        (
            "import numpy as np\nnp.zeros([1] * 65)",
            "arrays support at most 64 dimensions",
        ),
        (
            "import numpy as np\nvalue = 1\nfor _ in range(65):\n    value = [value]\nnp.array(value)",
            "arrays support at most 64 dimensions",
        ),
        (
            "import numpy as np\nnp.array([1.0]) / 0",
            "ZeroDivisionError",
        ),
    ] {
        assert_fails_with(source, expected);
    }
}

#[test]
fn array_allocation_obeys_the_modeled_memory_limit() {
    let environment = Environment::with_limits(Limits {
        cpu: 1_000_000,
        memory: 16 * 1024,
        disk: 1024 * 1024,
        output: 1024,
    });
    let (status, _, stderr) =
        run_with_environment(environment, "import numpy as np\nnp.zeros((100, 100))");
    assert_eq!(status, 137);
    assert!(stderr.is_empty());
}
