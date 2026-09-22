//! Control-flow unwinding across functions, loops, generators, and context managers.

use std::process::Command;

use super::support::run_python_text as run;

fn cpython(source: &str) -> Option<(i32, String, String)> {
    let output = Command::new("python3.14")
        .arg("-c")
        .arg(source)
        .output()
        .ok()?;
    Some((
        output.status.code().unwrap_or(1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

#[test]
fn return_runs_finally_before_returning() {
    let source = "def f():\n    try:\n        return 'value'\n    finally:\n        print('cleanup')\nprint(f())";
    assert_eq!(run(source), (0, "cleanup\nvalue\n".into(), String::new()));
    if let Some(reference) = cpython(source) {
        assert_eq!(reference, (0, "cleanup\nvalue\n".into(), String::new()));
    }
}

#[test]
fn break_and_continue_run_finally() {
    let source = "for i in [0, 1, 2]:\n    try:\n        if i == 1:\n            continue\n        if i == 2:\n            break\n        print('body', i)\n    finally:\n        print('cleanup', i)\nprint('done')";
    assert_eq!(
        run(source),
        (
            0,
            "body 0\ncleanup 0\ncleanup 1\ncleanup 2\ndone\n".into(),
            String::new()
        )
    );
    if let Some(reference) = cpython(source) {
        assert_eq!(
            reference,
            (
                0,
                "body 0\ncleanup 0\ncleanup 1\ncleanup 2\ndone\n".into(),
                String::new()
            )
        );
    }
}

#[test]
fn exception_in_handler_runs_finally_and_propagates_new_exception() {
    let source = "try:\n    raise ValueError('first')\nexcept ValueError:\n    print('handler')\n    raise TypeError('second')\nfinally:\n    print('cleanup')";
    let (status, out, err) = run(source);
    assert_eq!(status, 2, "out={out:?} err={err:?}");
    assert_eq!(out, "handler\ncleanup\n");
    assert!(err.contains("TypeError"), "{err}");
    if let Some((reference_status, reference_out, reference_err)) = cpython(source) {
        assert_eq!(reference_status, 1);
        assert_eq!(reference_out, out);
        assert!(
            reference_err.contains("TypeError: second"),
            "{reference_err}"
        );
    }
}

#[test]
fn abrupt_control_flow_exits_a_context_manager_once() {
    let source = "class C:\n    def __enter__(self):\n        print('enter')\n        return self\n    def __exit__(self, kind, value, traceback):\n        print('exit')\ndef f():\n    with C():\n        return 7\nprint(f())";
    let (status, out, err) = run(source);
    assert_eq!(
        (status, out.clone()),
        (0, "enter\nexit\n7\n".into()),
        "{err}"
    );
    if let Some(reference) = cpython(source) {
        assert_eq!(reference, (0, out, String::new()));
    }
}

#[test]
fn loop_control_does_not_exit_an_enclosing_context_manager() {
    let source = "class C:\n    def __enter__(self):\n        print('enter')\n    def __exit__(self, kind, value, traceback):\n        print('exit')\nwith C():\n    for value in [1, 2, 3]:\n        if value == 1:\n            continue\n        if value == 2:\n            break\nprint('done')";
    let (status, out, err) = run(source);
    assert_eq!(status, 0, "{err}");
    assert_eq!(out, "enter\nexit\ndone\n");
    assert!(err.is_empty());
}

#[test]
fn generator_suspends_and_resumes_inside_finally_region() {
    let source = "def values():\n    try:\n        yield 1\n        yield 2\n    finally:\n        print('cleanup')\ngenerator = values()\nprint(next(generator))\nprint(next(generator))\nprint(next(generator, 'done'))";
    let (status, out, err) = run(source);
    assert_eq!(status, 0, "out={out:?} err={err:?}");
    assert_eq!(out, "1\n2\ncleanup\ndone\n");
    assert!(err.is_empty());
}

#[test]
fn generator_suspends_and_resumes_inside_with_region() {
    let source = "class C:\n    def __enter__(self):\n        print('enter')\n        return self\n    def __exit__(self, kind, value, traceback):\n        print('exit')\ndef values():\n    with C():\n        yield 1\n        yield 2\ngenerator = values()\nprint(next(generator))\nprint(next(generator))\nprint(next(generator, 'done'))";
    let (status, out, err) = run(source);
    assert_eq!(status, 0, "out={out:?} err={err:?}");
    assert_eq!(out, "enter\n1\n2\nexit\ndone\n");
    assert!(err.is_empty());
}
