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
print(np.dot(np.array([1, 2, 3]), np.array([4, 5, 6])))
"#;
    assert_eq!(
        run(source),
        (
            0,
            b"[[11, 22, 33], [14, 25, 36]]\n[[11, 22, 33], [14, 25, 36]]\n[[1, 3, 5], [7, 9, 11]]\n[[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]\n21 [5, 7, 9] [6, 15]\n3.5 1 6\n[[19, 22], [43, 50]]\n32\n".to_vec(),
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

    let (status, _, stderr) = run("import numpy as np\nnp.int64(9223372036854775808)");
    assert_ne!(status, 0);
    assert!(!stderr.is_empty());
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
    for source in [
        "import numpy as np\nnp.array([[1], [2, 3]])",
        "import numpy as np\nnp.array([1, 2]) + np.array([1, 2, 3])",
        "import numpy as np\nnp.matmul(np.array([1, 2]), np.array([3, 4]))",
        "import numpy as np\nnp.zeros((-1, 2))",
        "import numpy as np\nnp.array([1], dtype='complex128')",
        "import numpy as np\nbool(np.array([1, 2]))",
        "import numpy as np\nnp.array([[1, 2]])[:, 0]",
    ] {
        let (status, _, stderr) = run(source);
        assert_ne!(
            status, 0,
            "unsupported operation unexpectedly succeeded: {source}"
        );
        assert!(
            !stderr.is_empty(),
            "unsupported operation failed silently: {source}"
        );
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
