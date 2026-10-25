//! The Agents panel: whether MCP agents may drive this window, what they
//! have done recently, and one-click setup for Claude Code and Codex.

use std::collections::VecDeque;
use std::time::Instant;

use egui::{Color32, RichText, Ui};

/// Recent actions kept for the panel.
const LOG_LEN: usize = 50;
/// An agent counts as active this long after its last action.
const ACTIVE_SECS: u64 = 15;

pub struct AgentEvent {
    pub at: Instant,
    pub tool: String,
    /// First line of the result.
    pub summary: String,
    pub ok: bool,
}

pub struct AgentState {
    /// Agents may control this window (otherwise their calls are refused).
    pub allow: bool,
    /// The Agents panel is showing.
    pub open: bool,
    pub log: VecDeque<AgentEvent>,
    /// Result of the last setup button.
    pub setup_note: Option<(bool, String)>,
}

impl Default for AgentState {
    fn default() -> Self {
        AgentState { allow: true, open: false, log: VecDeque::new(), setup_note: None }
    }
}

/// Where a status is shown: badge text and color.
pub enum Status {
    Paused,
    Active,
    Idle,
}

impl AgentState {
    pub fn record(&mut self, tool: &str, result: &serde_json::Value) {
        let ok = !result["isError"].as_bool().unwrap_or(false);
        let summary = result["content"]
            .as_array()
            .and_then(|c| c.iter().find_map(|i| i["text"].as_str()))
            .map(|t| t.lines().next().unwrap_or("").chars().take(90).collect())
            .unwrap_or_else(|| if ok { "image".to_string() } else { String::new() });
        self.log.push_front(AgentEvent { at: Instant::now(), tool: tool.to_string(), summary, ok });
        self.log.truncate(LOG_LEN);
    }

    pub fn status(&self) -> Status {
        if !self.allow {
            Status::Paused
        } else if self.log.front().is_some_and(|e| e.at.elapsed().as_secs() < ACTIVE_SECS) {
            Status::Active
        } else {
            Status::Idle
        }
    }

    /// The reply sent to an agent while control is paused.
    pub fn paused_reply() -> serde_json::Value {
        serde_json::json!({
            "content": [{ "type": "text", "text": "The voxely user has paused agent control of this window. Ask them to turn on \"Allow agents\" in the Agents panel (bottom bar)." }],
            "isError": true,
        })
    }
}

fn ago(at: Instant) -> String {
    let s = at.elapsed().as_secs();
    match s {
        0..=59 => format!("{s}s"),
        60..=3599 => format!("{}m", s / 60),
        _ => format!("{}h", s / 3600),
    }
}

const GREEN: Color32 = Color32::from_rgb(90, 200, 120);
const AMBER: Color32 = Color32::from_rgb(230, 170, 60);
const RED: Color32 = Color32::from_rgb(235, 110, 110);

/// The bottom-bar badge; clicking it opens the panel.
pub fn badge(ui: &mut Ui, a: &mut AgentState) {
    let (dot, label) = match a.status() {
        Status::Paused => (AMBER, "Agents paused"),
        Status::Active => (GREEN, "Agent active"),
        Status::Idle => (Color32::GRAY, "Agents"),
    };
    let text = RichText::new(label).color(dot);
    if ui.selectable_label(a.open, text).on_hover_text("MCP agents (Claude Code, Codex, ...)").clicked() {
        a.open ^= true;
    }
}

fn exe() -> String {
    std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_else(|_| "voxely".into())
}

fn claude_command() -> String {
    format!("claude mcp add --scope user voxely -- \"{}\" mcp", exe())
}

fn codex_config() -> String {
    format!("[mcp_servers.voxely]\ncommand = \"{}\"\nargs = [\"mcp\"]\n", exe())
}

/// Register voxely with Claude Code for every project.
fn add_to_claude() -> (bool, String) {
    let out = std::process::Command::new("claude").args(["mcp", "add", "--scope", "user", "voxely", "--", &exe(), "mcp"]).output();
    match out {
        Ok(o) => {
            let text = String::from_utf8_lossy(if o.stdout.is_empty() { &o.stderr } else { &o.stdout }).trim().to_string();
            if o.status.success() {
                (true, "Added. New Claude Code sessions can use voxely.".into())
            } else if text.contains("already exists") {
                (true, "Already set up in Claude Code.".into())
            } else {
                (false, if text.is_empty() { "claude mcp add failed".into() } else { text })
            }
        }
        Err(_) => (false, "Couldn't run `claude`. Install Claude Code, or copy the command below into a terminal.".into()),
    }
}

/// The Agents panel, docked along the bottom of the 3D view (like
/// MagicaVoxel's console bar) so it never covers the model: controls and
/// setup on the left, recent actions on the right.
pub fn panel(ui: &mut Ui, a: &mut AgentState) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("Agents").strong());
        let (color, line) = match a.status() {
            Status::Paused => (AMBER, "Paused: agent requests are refused".to_string()),
            Status::Active => (GREEN, format!("Active, last action {} ago", ago(a.log[0].at))),
            Status::Idle => match a.log.front() {
                Some(e) => (Color32::GRAY, format!("Idle, last action {} ago", ago(e.at))),
                None => (Color32::GRAY, "Idle, no agent actions yet".to_string()),
            },
        };
        ui.label(RichText::new(line).color(color));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("Close").clicked() {
                a.open = false;
            }
        });
    });
    ui.separator();
    ui.columns(2, |cols| {
        let ui = &mut cols[0];
        ui.checkbox(&mut a.allow, "Allow agents to control this window")
            .on_hover_text("When off, agents get a reply saying control is paused");
        ui.weak("Agent edits are undoable like yours (Cmd+Z).");
        ui.add_space(4.0);
        ui.label(RichText::new("Connect an agent").strong());
        ui.horizontal_wrapped(|ui| {
            if ui.button("Add to Claude Code").on_hover_text(claude_command()).clicked() {
                a.setup_note = Some(add_to_claude());
            }
            if ui.button("Copy command").on_hover_text("The Claude Code command, for a terminal").clicked() {
                ui.ctx().copy_text(claude_command());
                a.setup_note = Some((true, "Copied the Claude Code command.".into()));
            }
            if ui.button("Copy Codex config").on_hover_text("Paste into ~/.codex/config.toml").clicked() {
                ui.ctx().copy_text(codex_config());
                a.setup_note = Some((true, "Copied. Paste it into ~/.codex/config.toml.".into()));
            }
        });
        match &a.setup_note {
            Some((ok, note)) => {
                ui.add(egui::Label::new(RichText::new(note).small().color(if *ok { GREEN } else { RED })).wrap());
            }
            None => {
                ui.add(
                    egui::Label::new(
                        RichText::new("Agents start voxely's MCP server themselves; there's nothing to run. With this window closed they work headless.")
                            .weak()
                            .small(),
                    )
                    .wrap(),
                );
            }
        }

        let ui = &mut cols[1];
        ui.label(RichText::new("Recent actions").strong());
        if a.log.is_empty() {
            ui.weak("Nothing yet. Connect an agent, then ask it to build something.");
        }
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            egui::Grid::new("agent_log").num_columns(3).spacing([8.0, 2.0]).striped(true).show(ui, |ui| {
                for e in &a.log {
                    ui.weak(ago(e.at));
                    ui.label(RichText::new(&e.tool).monospace().color(if e.ok { Color32::LIGHT_GRAY } else { RED }));
                    ui.add(egui::Label::new(RichText::new(&e.summary).small()).truncate()).on_hover_text(&e.summary);
                    ui.end_row();
                }
            });
        });
    });
}
