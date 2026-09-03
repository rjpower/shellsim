use shellsim::Environment;

fn run_python(source: &str) -> (i32, String, String) {
    let mut env = Environment::new();
    let argv = vec!["python3.14".into(), "-c".into(), source.into()];
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let status =
        shellsim::python::run_python(&mut env, &argv, Vec::new(), &mut stdout, &mut stderr);
    (
        status,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn catches_typed_exception_and_runs_else_only_on_success() {
    let source = "try:\n    raise ValueError('bad')\nexcept ValueError as error:\n    print(error)\nelse:\n    print('wrong')\nprint('after')";
    let (status, out, err) = run_python(source);
    assert_eq!(status, 0, "{err}");
    assert_eq!(out, "bad\nafter\n");
    assert!(err.is_empty());
}

#[test]
fn finally_runs_for_propagating_exception_and_function_frames() {
    let source = "def fail():\n    try:\n        raise RuntimeError('boom')\n    finally:\n        print('cleanup')\ntry:\n    fail()\nexcept Exception:\n    print('caught')";
    let (status, out, err) = run_python(source);
    assert_eq!(status, 0, "{err}");
    assert_eq!(out, "cleanup\ncaught\n");
    assert!(err.is_empty());
}

#[test]
fn with_calls_enter_and_exit_and_can_suppress() {
    let source = "class Context:\n    def __enter__(self):\n        print('enter')\n        return self\n    def __exit__(self, kind, value, traceback):\n        print('exit', kind)\n        return True\nwith Context() as item:\n    print('body')\n    raise ValueError('ignored')\nprint('after')";
    let (status, out, err) = run_python(source);
    assert_eq!(status, 0, "{err}");
    assert_eq!(out, "enter\nbody\nexit ValueError\nafter\n");
    assert!(err.is_empty());
}
