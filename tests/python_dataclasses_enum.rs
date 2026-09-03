use shellsim::Environment;

fn run(source: &str) -> (i32, Vec<u8>, Vec<u8>) {
    let mut environment = Environment::new();
    environment
        .vfs
        .put_file("/probe.py", source.as_bytes().to_vec(), 0o644)
        .expect("install probe");
    let (outcome, stdout, stderr) = environment.run_script_capture("python3.14 /probe.py");
    (outcome.exit_status, stdout, stderr)
}

#[test]
fn dataclass_initializes_required_and_default_fields() {
    let (status, stdout, stderr) = run(r#"import dataclasses
@dataclasses.dataclass
class Point:
    x: int
    y: int = 4
print(Point(2).x, Point(2).y)
print(Point(y=9, x=3).x, Point(y=9, x=3).y)
"#);
    assert_eq!(status, 0, "{stderr:?}");
    assert_eq!(stdout, b"2 4\n3 9\n");
    assert!(stderr.is_empty());
}

#[test]
fn enum_members_have_identity_attributes_and_ordered_iteration() {
    let (status, stdout, stderr) = run(r#"import enum
class Color(enum.Enum):
    RED = 1
    GREEN = 2
print(Color.RED.name, Color.RED.value, Color.RED is Color.RED)
print([member.name for member in Color])
"#);
    assert_eq!(status, 0, "{stderr:?}");
    assert_eq!(stdout, b"RED 1 True\n['RED', 'GREEN']\n");
    assert!(stderr.is_empty());
}
