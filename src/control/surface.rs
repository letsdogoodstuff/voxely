//! Surface work: recolor noise, outlines, paint by normal, palette presets, reference images.

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
