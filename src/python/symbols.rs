//! Interned Python identifiers.
//!
//! Names that the VM resolves repeatedly (globals, attributes, locals) are interned once per
//! interpreter so lookups compare small integers instead of strings. The table only grows; its
//! storage is charged to the guest when a name is first seen and released when the interpreter
//! is discarded.

use std::collections::HashMap;
use std::sync::Arc;

use crate::python::error::{PyError, PyResult};
use crate::resources::Resources;

const SYMBOL_NAME_BYTES: u64 = 24;

/// Runtime-local identity for an interned Python identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SymbolId(u32);

impl SymbolId {
    /// A symbol that names nothing. Code compiled after a failed charge carries it, and such
    /// code is discarded without running.
    pub(super) const UNBOUND: Self = Self(u32::MAX);

    pub(super) const fn index(self) -> usize {
        self.0 as usize
    }

    pub(super) fn from_index(index: usize) -> Option<Self> {
        u32::try_from(index).ok().map(Self)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Symbols {
    ids: HashMap<Arc<str>, SymbolId>,
    names: Vec<Arc<str>>,
    modeled_bytes: u64,
}

impl Symbols {
    /// The identity of an identifier already known to this interpreter.
    pub fn id(&self, name: &str) -> Option<SymbolId> {
        self.ids.get(name).copied()
    }

    /// Resolve a symbol identity back to its interpreter-owned name.
    pub fn name(&self, symbol: SymbolId) -> Option<&str> {
        self.names.get(symbol.index()).map(AsRef::as_ref)
    }

    /// The name of a symbol this table issued. Compiled code and symbol-keyed tables hold
    /// only symbols their own interpreter interned, so the name is always present.
    pub fn issued(&self, symbol: SymbolId) -> &str {
        &self.names[symbol.index()]
    }

    /// A shared handle to an issued symbol's name, for callers that need the name while they
    /// mutate the interpreter.
    pub fn shared(&self, symbol: SymbolId) -> Arc<str> {
        self.names[symbol.index()].clone()
    }

    /// Intern one identifier, charging its process-lifetime storage before mutation.
    pub fn intern(&mut self, name: &str, resources: &mut Resources) -> PyResult<SymbolId> {
        if let Some(symbol) = self.id(name) {
            return Ok(symbol);
        }
        let symbol =
            SymbolId(u32::try_from(self.names.len()).map_err(|_| "too many Python identifiers")?);
        let bytes = SYMBOL_NAME_BYTES
            .checked_add(u64::try_from(name.len()).unwrap_or(u64::MAX))
            .ok_or("identifier storage size overflow")?;
        let modeled_bytes = self
            .modeled_bytes
            .checked_add(bytes)
            .ok_or("modeled symbol table size overflow")?;
        if !resources.reserve_memory(bytes) {
            return Err(PyError::resource_error("memory limit exceeded"));
        }
        self.modeled_bytes = modeled_bytes;
        let name: Arc<str> = name.into();
        self.ids.insert(name.clone(), symbol);
        self.names.push(name);
        Ok(symbol)
    }

    /// Transfer the table's accounting to the caller when an interpreter is discarded.
    pub fn take_modeled_bytes(&mut self) -> u64 {
        std::mem::take(&mut self.modeled_bytes)
    }
}
