//! Compatibility tests for deterministic line and column formatting.

use shellsim::interp::{Environment, Interp};

fn run(environment: &mut Interp, source: &str) -> (i32, Vec<u8>, String) {
    let (outcome, stdout, stderr) = environment.run_script_capture(source);
    (
        outcome.exit_status,
        stdout,
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

#[test]
fn formatting_commands_transform_columns_and_tabs() {
    let mut environment = Environment::new();
    let (status, stdout, stderr) = run(&mut environment, "printf 'a\\tb\\n' | expand -t 4; printf 'a   b\\n' | unexpand -a -t 4; printf 'a:long\\nb:x\\n' | column -t -s :; printf 'one two three four\\n' | fmt -w 9; printf 'abcdef\\n' | fold -w 3");
    assert_eq!(status, 0, "{stderr}");
    assert_eq!(
        stdout,
        b"a   b\na\tb\na  long\nb  x\none two\nthree\nfour\nabc\ndef\n"
    );
}
