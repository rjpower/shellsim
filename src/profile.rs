//! Versioned environment profiles for reproducible compatibility workloads.
//!
//! Profiles configure only modeled state. They never mount host paths, inherit host variables,
//! or grant network access. Corpus fixtures and virtual routes remain explicit runner inputs.

use serde::{Deserialize, Serialize};

use crate::{Environment, Limits};

/// A stable, named baseline for compatibility corpora.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EnvironmentProfile {
    StockAgentV1,
}

impl EnvironmentProfile {
    /// Build a fresh environment containing this profile's virtual filesystem and variables.
    pub fn create(self, limits: Limits) -> Result<Environment, String> {
        match self {
            Self::StockAgentV1 => stock_agent_v1(limits),
        }
    }
}

fn stock_agent_v1(limits: Limits) -> Result<Environment, String> {
    let mut environment = Environment::with_limits(limits);
    environment.vfs.seed_dirs([
        "/bin",
        "/etc",
        "/home",
        "/home/shellsim",
        "/usr",
        "/usr/bin",
    ]);
    environment
        .vfs
        .put_file(
            "/etc/passwd",
            b"root:x:0:0:root:/root:/bin/bash\nshellsim:x:1000:1000:Shellsim:/home/shellsim:/bin/bash\n"
                .to_vec(),
            0o644,
        )
        .map_err(|error| format!("cannot install stock /etc/passwd: {error:?}"))?;
    environment
        .vfs
        .put_file(
            "/etc/group",
            b"root:x:0:\nshellsim:x:1000:\n".to_vec(),
            0o644,
        )
        .map_err(|error| format!("cannot install stock /etc/group: {error:?}"))?;

    for (name, value) in [
        ("HOME", "/home/shellsim"),
        ("USER", "shellsim"),
        ("LOGNAME", "shellsim"),
        ("SHELL", "/bin/bash"),
        ("PATH", "/usr/local/bin:/usr/bin:/bin"),
        ("TMPDIR", "/tmp"),
        ("LANG", "C.UTF-8"),
        ("LC_ALL", "C.UTF-8"),
        ("TZ", "UTC"),
        ("HOSTNAME", "shellsim"),
        ("PWD", "/work"),
    ] {
        environment.set_var(name, value);
        environment.exported.insert(name.to_string());
    }
    environment.cwd = "/work".to_string();
    environment.uid = 1_000;
    environment.processes.set_uid(environment.pid, 1_000);
    environment.umask = 0o022;
    Ok(environment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stock_agent_profile_is_conventional_and_deterministic() {
        let environment = EnvironmentProfile::StockAgentV1
            .create(Limits::default())
            .unwrap();

        assert_eq!(environment.cwd, "/work");
        assert_eq!(
            environment.get_var("HOME").as_deref(),
            Some("/home/shellsim")
        );
        assert_eq!(environment.get_var("USER").as_deref(), Some("shellsim"));
        assert_eq!(environment.get_var("TZ").as_deref(), Some("UTC"));
        assert_eq!(environment.uid, 1_000);
        assert_eq!(environment.umask, 0o022);
        assert_eq!(
            environment.processes.get(environment.pid).unwrap().uid,
            1_000
        );
        assert!(environment.vfs.exists("/", "/etc/passwd"));
        assert!(environment.vfs.exists("/", "/home/shellsim"));
    }
}
