//! Bounded ownership and routing for independent harness sessions.
//!
//! A manager assigns host-side session identities and routes existing typed operations into one
//! [`HarnessSession`]. Forking uses the session's complete-state
//! clone, so active actions, processes, descriptors, timers, Python heaps, and telemetry branch
//! together. The manager never grants simulated code a capability or shares mutable machine state
//! between sessions.

use std::collections::BTreeMap;

use crate::harness::{
    HarnessOperation, HarnessRequest, HarnessResponse, HarnessResult, HarnessSession,
};

/// Maximum independent machines retained by one protocol connection.
pub const MAX_HARNESS_SESSIONS: usize = 8;

/// Connection-local collection of independently owned deterministic machines.
pub struct HarnessManager {
    sessions: BTreeMap<u64, HarnessSession>,
    next_session_id: u64,
}

impl HarnessManager {
    /// Create a manager whose compatibility session has identity zero.
    pub fn new(initial: HarnessSession) -> Self {
        Self {
            sessions: BTreeMap::from([(0, initial)]),
            next_session_id: 1,
        }
    }

    /// Route one request or apply a manager-owned fork/drop operation.
    pub fn handle(&mut self, request: HarnessRequest) -> HarnessResponse {
        let HarnessRequest {
            id,
            session_id,
            operation,
        } = request;
        match operation {
            HarnessOperation::ForkSession { source } => self.fork_response(id, source),
            HarnessOperation::DropSession { target } => self.drop_response(id, target),
            operation => {
                let session_id = session_id.unwrap_or(0);
                let Some(session) = self.sessions.get_mut(&session_id) else {
                    return error_response(id, format!("session {session_id} does not exist"));
                };
                session.handle(HarnessRequest {
                    id,
                    session_id: None,
                    operation,
                })
            }
        }
    }

    fn fork_response(&mut self, id: Option<serde_json::Value>, source: u64) -> HarnessResponse {
        if self.sessions.len() >= MAX_HARNESS_SESSIONS {
            return error_response(
                id,
                format!("harness session limit exceeded ({MAX_HARNESS_SESSIONS})"),
            );
        }
        let Some(source_session) = self.sessions.get(&source) else {
            return error_response(id, format!("session {source} does not exist"));
        };
        let fork = match source_session.fork() {
            Ok(fork) => fork,
            Err(error) => return error_response(id, error),
        };
        let session_id = self.next_session_id;
        let Some(next) = self.next_session_id.checked_add(1) else {
            return error_response(id, "session identifier space exhausted".to_string());
        };
        self.next_session_id = next;
        self.sessions.insert(session_id, fork);
        HarnessResponse {
            id,
            ok: true,
            result: Some(HarnessResult::Session { session_id }),
            error: None,
        }
    }

    fn drop_response(&mut self, id: Option<serde_json::Value>, target: u64) -> HarnessResponse {
        if self.sessions.remove(&target).is_none() {
            return error_response(id, format!("session {target} does not exist"));
        }
        HarnessResponse {
            id,
            ok: true,
            result: Some(HarnessResult::Acknowledged),
            error: None,
        }
    }
}

fn error_response(id: Option<serde_json::Value>, error: String) -> HarnessResponse {
    HarnessResponse {
        id,
        ok: false,
        result: None,
        error: Some(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Limits;
    use base64::Engine;

    fn request(session_id: Option<u64>, operation: HarnessOperation) -> HarnessRequest {
        HarnessRequest {
            id: None,
            session_id,
            operation,
        }
    }

    #[test]
    fn forks_isolate_complete_session_state_and_drop_removes_routing() {
        let mut manager = HarnessManager::new(HarnessSession::new(Limits::default()));
        let response = manager.handle(request(
            None,
            HarnessOperation::Execute {
                source: "X=parent; printf base > value".into(),
                stdin_base64: String::new(),
            },
        ));
        assert!(response.ok);
        let fork = manager.handle(request(None, HarnessOperation::ForkSession { source: 0 }));
        assert!(matches!(
            fork.result,
            Some(HarnessResult::Session { session_id: 1 })
        ));
        let response = manager.handle(request(
            Some(1),
            HarnessOperation::Execute {
                source: "X=child; printf branch > value".into(),
                stdin_base64: String::new(),
            },
        ));
        assert!(response.ok);
        for (session_id, expected) in [
            (None, b"parent:base".as_slice()),
            (Some(1), b"child:branch".as_slice()),
        ] {
            let response = manager.handle(request(
                session_id,
                HarnessOperation::Execute {
                    source: "printf '%s:' \"$X\"; cat value".into(),
                    stdin_base64: String::new(),
                },
            ));
            let Some(HarnessResult::Execute(result)) = response.result else {
                panic!("session execute did not return output");
            };
            assert_eq!(
                base64::engine::general_purpose::STANDARD
                    .decode(result.stdout_base64)
                    .unwrap(),
                expected
            );
        }
        assert!(
            manager
                .handle(request(None, HarnessOperation::DropSession { target: 1 }))
                .ok
        );
        let missing = manager.handle(request(Some(1), HarnessOperation::Inspect));
        assert!(!missing.ok);
        assert!(missing.error.unwrap().contains("does not exist"));
    }

    #[test]
    fn session_count_is_bounded_before_cloning() {
        let mut manager = HarnessManager::new(HarnessSession::new(Limits::default()));
        for expected in 1..MAX_HARNESS_SESSIONS as u64 {
            let response =
                manager.handle(request(None, HarnessOperation::ForkSession { source: 0 }));
            assert!(matches!(
                response.result,
                Some(HarnessResult::Session { session_id }) if session_id == expected
            ));
        }
        let rejected = manager.handle(request(None, HarnessOperation::ForkSession { source: 0 }));
        assert!(!rejected.ok);
        assert!(rejected.error.unwrap().contains("session limit"));
    }
}
