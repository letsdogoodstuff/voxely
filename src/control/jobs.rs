//! Renders that run off the window's thread, so the window stays live
//! (Model/Render switching, the camera, the panels) while they trace.
//! `voxely mcp` waits on them and reports progress; if its wait runs out the
//! agent gets the job id and collects the image with `render {job}`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::{Result, bail};
use serde_json::{Value, json};

use super::text;

/// Finished jobs kept for collecting; older ones are dropped.
const KEEP_DONE: usize = 8;

struct Job {
    /// Samples traced so far, of `total`.
    done: Arc<AtomicU32>,
    total: u32,
    started: Instant,
    handle: Option<JoinHandle<Result<Vec<Value>>>>,
    result: Option<Result<Vec<Value>, String>>,
}

static JOBS: Mutex<BTreeMap<u64, Job>> = Mutex::new(BTreeMap::new());
static NEXT: AtomicU32 = AtomicU32::new(1);

/// Run `work` on its own thread. It counts traced samples into the counter
/// it's given, `total` in all. Returns the job id.
pub fn start(total: u32, work: impl FnOnce(&AtomicU32) -> Result<Vec<Value>> + Send + 'static) -> String {
    let id = NEXT.fetch_add(1, Ordering::Relaxed) as u64;
    let done = Arc::new(AtomicU32::new(0));
    let counter = done.clone();
    let handle = std::thread::spawn(move || work(&counter));
    let mut jobs = JOBS.lock().unwrap();
    jobs.insert(id, Job { done, total: total.max(1), started: Instant::now(), handle: Some(handle), result: None });
    // Forget the oldest finished jobs.
    let finished: Vec<u64> = jobs.iter().filter(|(_, j)| j.handle.as_ref().is_none_or(|h| h.is_finished())).map(|(&k, _)| k).collect();
    for k in finished.iter().take(finished.len().saturating_sub(KEEP_DONE)) {
        jobs.remove(k);
    }
    format!("r{id}")
}

/// The reply to `render {job}`: the finished render's content, or a
/// "still rendering" note with a `_job` marker `voxely mcp` waits on.
pub fn collect(id: &str) -> Result<Vec<Value>> {
    let Some(n) = id.trim().strip_prefix('r').and_then(|n| n.parse::<u64>().ok()) else {
        bail!("render jobs are named like \"r3\" (the id an earlier render returned), not {id:?}");
    };
    let mut jobs = JOBS.lock().unwrap();
    let Some(job) = jobs.get_mut(&n) else {
        let known: Vec<String> = jobs.keys().map(|k| format!("r{k}")).collect();
        bail!("no render job {id}; jobs kept: {known:?} (finished renders are dropped after {KEEP_DONE} newer ones)");
    };
    if job.handle.as_ref().is_some_and(|h| h.is_finished()) {
        let out = job.handle.take().unwrap().join();
        job.result = Some(match out {
            Ok(Ok(content)) => Ok(content),
            Ok(Err(e)) => Err(format!("{e:#}")),
            Err(_) => Err("the render thread crashed".to_string()),
        });
    }
    match &job.result {
        Some(Ok(content)) => Ok(content.clone()),
        Some(Err(e)) => bail!("render {id} failed: {e}"),
        None => Ok(running(id, job)),
    }
}

fn running(id: &str, job: &Job) -> Vec<Value> {
    let done = job.done.load(Ordering::Relaxed).min(job.total);
    let secs = job.started.elapsed().as_secs_f32();
    vec![
        text(format!(
            "Still rendering {id}: {done} of {} samples after {secs:.0}s. The window stays usable meanwhile. Call render with job \"{id}\" to collect the image.",
            job.total
        )),
        json!({ "type": "_job", "id": id, "done": done, "total": job.total }),
    ]
}

/// Take a `_job` marker off the end of a tool's content (see `collect`).
pub fn take_marker(content: &mut Vec<Value>) -> Option<Value> {
    if content.last().is_some_and(|c| c["type"] == "_job") { content.pop() } else { None }
}
