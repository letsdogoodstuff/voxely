//! Building: symmetry, more primitives, stamps.

use anyhow::Result;
use serde_json::Value;

use super::Ctx;

#[derive(Default)]
pub struct State {}

pub const PART_TOOLS: &[&str] = &[];

pub fn tools() -> Vec<Value> {
    Vec::new()
}

pub fn dispatch(_ctx: &mut Ctx, _name: &str, _args: &Value) -> Option<Result<Vec<Value>>> {
    None
}

/// Called on every edit before it's committed (e.g. to mirror it).
pub fn before_commit(_ctx: &mut Ctx, new: crate::scene::Volume) -> crate::scene::Volume {
    new
}
