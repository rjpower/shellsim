//! Shell variable storage with running memory estimates.
//!
//! Guest scripts grow variables, arrays, and positional parameters without bound unless the
//! machine charges that growth. Each container here keeps an exact running estimate of its
//! modeled bytes, updated by every mutation, so the environment can reserve growth against the
//! machine memory budget at command boundaries without rescanning the state.
//!
//! Reads go through `Deref` to the underlying collection. Mutations go through the methods
//! below, which keep the estimate in step with the contents.
//!
//! Indexed arrays are sparse, as in Bash: `a[9999999999]=x` stores one element.

use std::collections::{BTreeMap, HashMap};
use std::ops::Deref;

/// Modeled bookkeeping cost of one stored string or map entry beyond its bytes.
const ENTRY_OVERHEAD: u64 = 48;

/// Modeled bytes of one stored string.
pub(crate) fn string_bytes(value: &str) -> u64 {
    (value.len() as u64).saturating_add(ENTRY_OVERHEAD)
}

fn entry_bytes(name: &str, value: &str) -> u64 {
    string_bytes(name).saturating_add(string_bytes(value))
}

/// A Bash array value. Indexed arrays are sparse maps from index to value. Associative arrays
/// preserve sorted key order (Bash uses an unspecified hash order; sorted is deterministic).
#[derive(Clone, Debug)]
pub enum ArrayVal {
    Indexed(BTreeMap<usize, String>),
    Assoc(BTreeMap<String, String>),
}

impl ArrayVal {
    /// Modeled bytes of the array's elements.
    pub(crate) fn bytes(&self) -> u64 {
        match self {
            Self::Indexed(values) => values
                .values()
                .fold(0, |total, value| total.saturating_add(string_bytes(value))),
            Self::Assoc(values) => values.iter().fold(0, |total, (key, value)| {
                total.saturating_add(entry_bytes(key, value))
            }),
        }
    }

    /// One past the highest set index of an indexed array, as Bash uses for negative
    /// subscripts and appends; zero for an empty or associative array.
    pub fn end_index(&self) -> usize {
        match self {
            Self::Indexed(values) => values
                .last_key_value()
                .map_or(0, |(index, _)| index.saturating_add(1)),
            Self::Assoc(_) => 0,
        }
    }
}

/// Scalar shell variables.
#[derive(Clone, Debug, Default)]
pub struct Variables {
    values: HashMap<String, String>,
    bytes: u64,
}

impl Deref for Variables {
    type Target = HashMap<String, String>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl FromIterator<(String, String)> for Variables {
    fn from_iter<I: IntoIterator<Item = (String, String)>>(iter: I) -> Self {
        let mut variables = Self::default();
        for (name, value) in iter {
            variables.insert(name, value);
        }
        variables
    }
}

impl Variables {
    pub fn insert(&mut self, name: String, value: String) -> Option<String> {
        self.bytes = self.bytes.saturating_add(entry_bytes(&name, &value));
        let previous = self.values.insert(name.clone(), value);
        if let Some(previous) = &previous {
            self.bytes = self.bytes.saturating_sub(entry_bytes(&name, previous));
        }
        previous
    }

    pub fn remove(&mut self, name: &str) -> Option<String> {
        let previous = self.values.remove(name);
        if let Some(previous) = &previous {
            self.bytes = self.bytes.saturating_sub(entry_bytes(name, previous));
        }
        previous
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&String, &String) -> bool) {
        let mut released = 0_u64;
        self.values.retain(|name, value| {
            let kept = keep(name, value);
            if !kept {
                released = released.saturating_add(entry_bytes(name, value));
            }
            kept
        });
        self.bytes = self.bytes.saturating_sub(released);
    }

    pub fn clear(&mut self) {
        self.values.clear();
        self.bytes = 0;
    }

    /// Modeled bytes of every variable.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// Shell arrays keyed by variable name.
#[derive(Clone, Debug, Default)]
pub struct Arrays {
    values: HashMap<String, ArrayVal>,
    bytes: u64,
}

impl Deref for Arrays {
    type Target = HashMap<String, ArrayVal>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl Arrays {
    pub fn insert(&mut self, name: String, value: ArrayVal) -> Option<ArrayVal> {
        self.bytes = self
            .bytes
            .saturating_add(string_bytes(&name))
            .saturating_add(value.bytes());
        let previous = self.values.insert(name.clone(), value);
        if let Some(previous) = &previous {
            self.bytes = self
                .bytes
                .saturating_sub(string_bytes(&name))
                .saturating_sub(previous.bytes());
        }
        previous
    }

    pub fn remove(&mut self, name: &str) -> Option<ArrayVal> {
        let previous = self.values.remove(name);
        if let Some(previous) = &previous {
            self.bytes = self
                .bytes
                .saturating_sub(string_bytes(name))
                .saturating_sub(previous.bytes());
        }
        previous
    }

    pub fn clear(&mut self) {
        self.values.clear();
        self.bytes = 0;
    }

    /// Store `value` at `key` of the existing array `name`; `None` if `name` is not an array.
    ///
    /// For an indexed array `key` must already be a parsed index.
    pub fn set_element(&mut self, name: &str, key: ArrayKey, value: String) -> Option<()> {
        let added = string_bytes(&value);
        let (removed, extra) = match (self.values.get_mut(name)?, key) {
            (ArrayVal::Indexed(values), ArrayKey::Index(index)) => {
                (values.insert(index, value).map(|old| string_bytes(&old)), 0)
            }
            (ArrayVal::Assoc(values), ArrayKey::Name(key)) => {
                let extra = string_bytes(&key);
                (
                    values
                        .insert(key, value)
                        .map(|old| string_bytes(&old).saturating_add(extra)),
                    extra,
                )
            }
            _ => return None,
        };
        self.bytes = self
            .bytes
            .saturating_add(added)
            .saturating_add(extra)
            .saturating_sub(removed.unwrap_or(0));
        Some(())
    }

    /// Remove one element; `None` if `name` is not an array or the key kind does not match.
    pub fn unset_element(&mut self, name: &str, key: ArrayKey) -> Option<()> {
        let removed = match (self.values.get_mut(name)?, key) {
            (ArrayVal::Indexed(values), ArrayKey::Index(index)) => {
                values.remove(&index).map(|old| string_bytes(&old))
            }
            (ArrayVal::Assoc(values), ArrayKey::Name(key)) => {
                values.remove(&key).map(|old| entry_bytes(&key, &old))
            }
            _ => return None,
        };
        self.bytes = self.bytes.saturating_sub(removed.unwrap_or(0));
        Some(())
    }

    /// Remove every element of `name`, keeping it an array of the same kind.
    pub fn clear_elements(&mut self, name: &str) {
        let Some(array) = self.values.get_mut(name) else {
            return;
        };
        let released = array.bytes();
        match array {
            ArrayVal::Indexed(values) => values.clear(),
            ArrayVal::Assoc(values) => values.clear(),
        }
        self.bytes = self.bytes.saturating_sub(released);
    }

    /// Modeled bytes of every array.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// A parsed array subscript.
pub enum ArrayKey {
    Index(usize),
    Name(String),
}

/// Positional parameters `$1 $2 ...`.
#[derive(Clone, Debug, Default)]
pub struct Positional {
    values: Vec<String>,
    bytes: u64,
}

impl Deref for Positional {
    type Target = Vec<String>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl From<Vec<String>> for Positional {
    fn from(values: Vec<String>) -> Self {
        let bytes = values.iter().fold(0, |total: u64, value| {
            total.saturating_add(string_bytes(value))
        });
        Self { values, bytes }
    }
}

impl Positional {
    /// Replace every parameter and return the previous ones.
    pub fn replace(&mut self, values: Vec<String>) -> Positional {
        std::mem::replace(self, values.into())
    }

    /// Drop the first `count` parameters, as `shift` does.
    pub fn shift(&mut self, count: usize) {
        let count = count.min(self.values.len());
        let released = self.values[..count].iter().fold(0, |total: u64, value| {
            total.saturating_add(string_bytes(value))
        });
        self.values.drain(..count);
        self.bytes = self.bytes.saturating_sub(released);
    }

    /// Modeled bytes of every parameter.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn into_vec(self) -> Vec<String> {
        self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recount(arrays: &Arrays) -> u64 {
        arrays.values.iter().fold(0, |total, (name, array)| {
            total
                .saturating_add(string_bytes(name))
                .saturating_add(array.bytes())
        })
    }

    #[test]
    fn variable_bytes_follow_inserts_replacements_and_removals() {
        let mut variables = Variables::default();
        variables.insert("x".into(), "abc".into());
        variables.insert("x".into(), "abcdef".into());
        variables.insert("y".into(), String::new());
        assert_eq!(
            variables.bytes(),
            entry_bytes("x", "abcdef") + entry_bytes("y", "")
        );
        variables.retain(|name, _| name == "y");
        assert_eq!(variables.bytes(), entry_bytes("y", ""));
        variables.remove("y");
        assert_eq!(variables.bytes(), 0);
    }

    #[test]
    fn array_bytes_follow_element_updates() {
        let mut arrays = Arrays::default();
        arrays.insert("a".into(), ArrayVal::Indexed(BTreeMap::new()));
        arrays.set_element("a", ArrayKey::Index(9_999_999_999), "x".into());
        arrays.set_element("a", ArrayKey::Index(9_999_999_999), "xyz".into());
        arrays.set_element("a", ArrayKey::Index(1), "q".into());
        arrays.insert("m".into(), ArrayVal::Assoc(BTreeMap::new()));
        arrays.set_element("m", ArrayKey::Name("k".into()), "v".into());
        arrays.set_element("m", ArrayKey::Name("k".into()), "vv".into());
        assert_eq!(arrays.bytes(), recount(&arrays));
        arrays.unset_element("a", ArrayKey::Index(1));
        arrays.unset_element("m", ArrayKey::Name("k".into()));
        assert_eq!(arrays.bytes(), recount(&arrays));
        arrays.clear_elements("a");
        assert_eq!(arrays.bytes(), recount(&arrays));
        assert_eq!(arrays["a"].end_index(), 0);
        arrays.remove("a");
        arrays.remove("m");
        assert_eq!(arrays.bytes(), 0);
    }

    #[test]
    fn positional_bytes_follow_shift_and_replace() {
        let mut positional = Positional::from(vec!["a".into(), "bb".into(), "ccc".into()]);
        positional.shift(2);
        assert_eq!(positional.bytes(), string_bytes("ccc"));
        let previous = positional.replace(Vec::new());
        assert_eq!(previous.bytes(), string_bytes("ccc"));
        assert_eq!(positional.bytes(), 0);
    }
}
