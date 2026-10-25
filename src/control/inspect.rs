//! Inspecting and safeguarding: slice/projection maps, checkpoints, autosave.

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

/// Called after every committed edit (e.g. to count edits for autosave).
pub fn after_commit(_ctx: &mut Ctx) {}
