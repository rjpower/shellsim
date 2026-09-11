//! Explicitly empty subprocess frontier; no host process capability is exposed.

use super::super::native::ModuleDef;

pub(super) static MODULE: ModuleDef = ModuleDef {
    name: "subprocess",
    functions: &[],
    values: &[],
};
