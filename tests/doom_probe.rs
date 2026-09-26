//! Opt-in external Doomgeneric build probe; GPL engine source and WAD are never bundled.
//!
//! Set SHELLSIM_DOOM_SOURCE to the external doomgeneric/doomgeneric source directory. This
//! ignored test imports that trusted tree only into the test VFS, then invokes virtual cc.

use std::path::PathBuf;

#[path = "../examples/support/doom.rs"]
mod doom_support;

use shellsim::{
    commands::{SessionPoll, WasmSession},
    display::KeyEvent,
    realtime::ClockMode,
};

#[test]
#[ignore = "requires separately obtained Doomgeneric source and Freedoom data"]
fn external_doomgeneric_builds_and_reacts_to_input_in_virtual_environment() {
    let source =
        PathBuf::from(std::env::var_os("SHELLSIM_DOOM_SOURCE").expect("set SHELLSIM_DOOM_SOURCE"));
    let wad = PathBuf::from(std::env::var_os("SHELLSIM_DOOM_WAD").expect("set SHELLSIM_DOOM_WAD"));
    let environment = doom_support::build(&source, &wad, ClockMode::Virtual).unwrap();
    let mut session = WasmSession::start(
        environment,
        doom_support::PROGRAM,
        &["-iwad".into(), doom_support::WAD.into()],
    )
    .unwrap();
    let mut frames = 0;
    for _ in 0..20_000 {
        match session.poll() {
            SessionPoll::Frame(_) => {
                frames += 1;
                if frames >= 16 {
                    break;
                }
            }
            SessionPoll::Ready(status) => {
                panic!("Doom exited before first interactive frame: {status}")
            }
            SessionPoll::Running | SessionPoll::Sleeping(_) => {}
        }
    }
    assert!(
        frames >= 16,
        "Doom yielded only {frames} frames in 20,000 polls"
    );
    let baseline = session.frame().unwrap();
    assert_eq!((baseline.width, baseline.height), (640, 400));
    assert!(baseline.pixels.iter().any(|byte| *byte != 0));
    session
        .inject_key(KeyEvent {
            code: 27, // Doom's KEY_ESCAPE opens the menu.
            pressed: true,
        })
        .unwrap();
    let mut changed = 0;
    for _ in 0..20_000 {
        match session.poll() {
            SessionPoll::Frame(_) => {
                changed = session
                    .frame()
                    .unwrap()
                    .pixels
                    .iter()
                    .zip(&baseline.pixels)
                    .filter(|(left, right)| left != right)
                    .count();
                if changed > 0 {
                    break;
                }
            }
            SessionPoll::Ready(status) => panic!("Doom exited before handling input: {status}"),
            SessionPoll::Running | SessionPoll::Sleeping(_) => {}
        }
    }
    assert!(changed > 0, "injected key did not change the final frame");
    session.request_stop();
    assert_eq!(session.poll(), SessionPoll::Ready(130));
    let result = session.into_result().unwrap();
    assert!(
        result.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}
