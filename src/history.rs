//! Persistent log of past coaching sessions.
//!
//! Stored as JSON-lines under the platform data dir
//! (`%APPDATA%\speakflow\history.jsonl` on Windows,
//! `~/.local/share/speakflow/history.jsonl` on Linux,
//! `~/Library/Application Support/speakflow/history.jsonl` on macOS).
//!
//! Append-only at write time, full read at start-up. Sessions are tiny
//! (a few KB each), so this scales for hundreds of exercises before we'd
//! need an index.

use crate::analysis::DeliveryMetrics;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unix epoch seconds. Doubles as a stable id (ascending).
    pub timestamp: i64,
    pub theme: String,
    pub prompt: String,
    pub transcript: String,
    #[serde(default)]
    pub followups: Vec<String>,
    #[serde(default)]
    pub followup_answers: Vec<String>,
    pub metrics: DeliveryMetrics,
    pub feedback: String,
    /// Name of the persona used for this session. `serde(default)` so rows
    /// written before personas existed still load (they'll show as "").
    #[serde(default)]
    pub persona_name: String,
}

impl Session {
    #[allow(clippy::too_many_arguments)]
    pub fn now(
        theme: String,
        prompt: String,
        transcript: String,
        followups: Vec<String>,
        followup_answers: Vec<String>,
        metrics: DeliveryMetrics,
        feedback: String,
        persona_name: String,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Self { timestamp, theme, prompt, transcript, followups, followup_answers, metrics, feedback, persona_name }
    }

    pub fn formatted_timestamp(&self) -> String {
        format_utc(self.timestamp)
    }
}

fn history_path() -> Option<PathBuf> {
    let mut p = dirs::data_dir()?;
    p.push("speakflow");
    if let Err(e) = fs::create_dir_all(&p) {
        log::warn!("could not create history dir: {e}");
        return None;
    }
    p.push("history.jsonl");
    Some(p)
}

/// Append a session to disk. Failures are logged but never bubble up — losing a
/// history entry is far less important than a clean UI flow on the active session.
pub fn append(session: &Session) {
    let Some(path) = history_path() else { return };
    let line = match serde_json::to_string(session) {
        Ok(s) => s,
        Err(e) => { log::warn!("history serialize failed: {e}"); return; }
    };
    let res = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| writeln!(f, "{line}"));
    if let Err(e) = res {
        log::warn!("history write to {} failed: {e}", path.display());
    }
}

/// Load all sessions, oldest first. Bad lines are skipped, not fatal.
pub fn load_all() -> Vec<Session> {
    let Some(path) = history_path() else { return Vec::new() };
    let Ok(file) = fs::File::open(&path) else { return Vec::new() };
    let mut out = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        match serde_json::from_str::<Session>(trimmed) {
            Ok(s) => out.push(s),
            Err(e) => log::warn!("skipping bad history line: {e}"),
        }
    }
    out.sort_by_key(|s| s.timestamp);
    out
}

pub fn history_path_display() -> String {
    history_path().map(|p| p.display().to_string()).unwrap_or_else(|| "<unavailable>".into())
}

/// Remove all sessions belonging to a specific persona.
pub fn delete_for_persona(name: &str) {
    let Some(path) = history_path() else { return };
    let sessions = load_all();

    // Sessions with an empty persona_name are treated as belonging to the
    // default engineering leader persona (matching the migration logic in app.rs).
    let legacy_name = crate::personas::default_engineering_leader().name;

    let filtered: Vec<Session> = sessions.into_iter()
        .filter(|s| {
            let actual_name = if s.persona_name.is_empty() { &legacy_name } else { &s.persona_name };
            actual_name != name
        })
        .collect();

    // Overwrite the file with the filtered set.
    let mut file = match fs::File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            log::warn!("failed to truncate history file for clear: {e}");
            return;
        }
    };

    for s in filtered {
        if let Ok(line) = serde_json::to_string(&s) {
            if let Err(e) = writeln!(file, "{line}") {
                log::warn!("failed to write history line during clear: {e}");
            }
        }
    }
}

/// UTC formatter — avoids a chrono dependency. Howard Hinnant's
/// "civil_from_days" algorithm; correct for any reasonable date range.
fn format_utc(unix_secs: i64) -> String {
    let days = unix_secs.div_euclid(86_400);
    let sod = unix_secs.rem_euclid(86_400);
    let h = sod / 3600;
    let m = (sod / 60) % 60;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as i64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y0 = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y0 + 1 } else { y0 };

    format!("{:04}-{:02}-{:02} {:02}:{:02} UTC", y, mo, d, h, m)
}
