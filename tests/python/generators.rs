//! Resource-boundary tests for generator execution.

use super::support::run_python;

#[test]
fn generator_resource_usage_is_metered() {
    let source = r#"def endless():
    value = 0
    while True:
        yield value
        value += 1

for value in endless():
    print(value)
"#;
    let (status, _, _) = run_python(source);
    assert_eq!(status, 137);
}
