// Opt-in probes use the real source-built FreeType archive and a redistributable font fixture.
use shellsim::{Environment, Limits, StopReason};
use std::path::PathBuf;

fn install(environment: &mut Environment) {
    let directory = std::env::var_os("SHELLSIM_FREETYPE_ARTIFACTS")
        .expect("set SHELLSIM_FREETYPE_ARTIFACTS to the verified FreeType probe directory");
    let bytes = std::fs::read(PathBuf::from(directory).join("probe.wasm")).unwrap();
    environment.vfs.write("/", "/probe", &bytes, 0o755).unwrap();
    environment
        .vfs
        .write(
            "/",
            "/font.ttf",
            include_bytes!("fixtures/fonts/DejaVuSans.ttf"),
            0o644,
        )
        .unwrap();
}

#[test]
#[ignore = "requires verified FreeType artifacts"]
fn freetype_scalable_glyph_and_malformed_font() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 100_000_000,
        memory: 64 * 1024 * 1024,
        disk: 8 * 1024 * 1024,
        ..Limits::default()
    });
    install(&mut environment);
    let (outcome, stdout, stderr) = environment.run_script_capture("/probe");
    assert_eq!(outcome.exit_status, 0, "{stderr:?}");
    assert_eq!(
        stdout,
        b"FreeType: scalable glyph and malformed font passed\n"
    );
    assert!(stderr.is_empty());
}

#[test]
#[ignore = "requires verified FreeType artifacts"]
fn freetype_rasterization_obeys_cpu_budget() {
    let mut environment = Environment::with_limits(Limits {
        cpu: 20_000_000,
        memory: 64 * 1024 * 1024,
        disk: 8 * 1024 * 1024,
        ..Limits::default()
    });
    install(&mut environment);
    let (outcome, _, _) = environment.run_script_capture("/probe loop");
    assert_eq!(outcome.stop_reason, Some(StopReason::CpuExhausted));
}
