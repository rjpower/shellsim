//! Compatibility tests for xargs batching, replacement, and explicit invalid forms.

use shellsim::{
    interp::{Environment, Interp},
    vfs::{NativeProgram, NodeKind},
};

fn run(environment: &mut Interp, source: &str) -> (i32, Vec<u8>, String) {
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        stdout,
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn batching_replacement_and_nul_delimiters_compose() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf 'a b' | xargs -n1 printf '<%s>'; printf 'one\\ntwo\\n' | xargs -I{} printf '[{}]'; printf 'a\\0b c\\0' | xargs -0 -n1 printf '{%s}'",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(stdout, b"<a><b>[one][two]{a}{b c}");
}

#[test]
fn empty_input_and_invalid_batch_sizes_are_explicit() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(
        &mut environment,
        "printf '' | xargs -r echo skipped; printf done; printf x | xargs -n0 echo",
    );
    assert_eq!(status, 1);
    assert_eq!(stdout, b"done");
    assert!(stderr.contains("requires a positive number"), "{stderr}");
}

#[test]
fn native_xargs_waits_for_pipe_eof_and_runs_children_in_the_current_cwd() {
    let mut environment = Environment::new();
    let image = environment
        .vfs
        .metadata("/", "/usr/bin/xargs", true)
        .unwrap();
    assert!(matches!(
        image.kind,
        NodeKind::NativeExecutable(NativeProgram::Xargs)
    ));
    environment
        .vfs
        .write("/", "/work/input", &vec![b'a'; 128 * 1024], 0o644)
        .unwrap();
    let (status, stdout, stderr) = run(
        &mut environment,
        "cat /work/input | xargs printf '%s' | wc -c; cd /work; printf input | xargs wc -c",
    );
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(
        stdout,
        b"131072\n131072 input\n",
        "{stderr}; {:?}",
        environment.invocations.events()
    );
}
