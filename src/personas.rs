//! User-configurable coaching personas.
//!
//! A persona bundles three things the coach prompts care about:
//!   - a *description* of the coaching style and intended audience, woven
//!     verbatim into the system prompt;
//!   - a *topic list* the LLM picks from when proposing today's exercise;
//!   - optional *background* about the user (role, hobbies, etc.) added as
//!     extra context so feedback can be grounded.
//!
//! Persisted as `personas.json` in the same data dir as `history.jsonl` so
//! edits survive restart. Built-in defaults are restored if the file is
//! missing or unparseable.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Persona {
    pub name: String,
    /// Slotted into the system prompt after "You are ". Should read naturally
    /// when prefixed that way (e.g. "an executive communications coach…").
    pub description: String,
    pub topics: Vec<String>,
    #[serde(default)]
    pub background: String,
    /// Kokoro voice ID (0–10) used when reading prompts aloud via sherpa-onnx.
    /// Defaults to 0 so existing personas.json files continue to work.
    #[serde(default)]
    pub voice_id: u32,
}

/// Human-readable labels for the kokoro-en-v0_19 voice IDs (index = SID).
pub const KOKORO_VOICES: &[&str] = &[
    "af",           // 0
    "af_bella",     // 1
    "af_nicole",    // 2
    "af_sarah",     // 3
    "af_sky",       // 4
    "am_adam",      // 5
    "am_michael",   // 6
    "bf_emma",      // 7
    "bf_isabella",  // 8
    "bm_george",    // 9
    "bm_lewis",     // 10
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaStore {
    pub active: usize,
    pub personas: Vec<Persona>,
}

impl Default for PersonaStore {
    fn default() -> Self {
        Self {
            active: 0,
            personas: vec![default_engineering_leader(), default_social_banter()],
        }
    }
}

impl PersonaStore {
    pub fn active(&self) -> &Persona {
        &self.personas[self.active.min(self.personas.len().saturating_sub(1))]
    }
}

pub fn default_engineering_leader() -> Persona {
    Persona {
        name: "Engineering Leader".into(),
        description: "an executive communications coach. Your client is a senior engineering \
            leader (Head of Engineering, VP Eng, or CTO) who wants to sharpen how they speak \
            in high-stakes settings: exec staff meetings, board updates, all-hands, investor \
            conversations, and difficult 1:1s".into(),
        topics: vec![
            "Engineering leadership".into(),
            "People management & 1:1s".into(),
            "Technical strategy".into(),
            "System architecture trade-offs".into(),
            "Platform vs product engineering".into(),
            "Build vs buy decisions".into(),
            "Hiring & team composition".into(),
            "Cross-functional alignment with Product / Design".into(),
            "Communicating to the executive team or board".into(),
            "Incident response & blameless culture".into(),
            "Org design & team topologies".into(),
            "Migrations and tech debt".into(),
            "AI / ML adoption strategy".into(),
            "Developer productivity & DX".into(),
            "Performance management & feedback".into(),
            "Roadmapping under uncertainty".into(),
        ],
        background: String::new(),
        voice_id: 0,
    }
}

pub fn default_social_banter() -> Persona {
    Persona {
        name: "Social & Banter".into(),
        description: "a communications coach focused on everyday social conversations. Your \
            client wants to be a more engaging, witty, and present conversationalist in casual \
            settings: pub chat, dinner parties, dates, group hangouts, and conversations with \
            strangers".into(),
        topics: vec![
            "Telling a story from your week".into(),
            "Small talk openers that aren't the weather".into(),
            "Sharing an unpopular opinion, playfully".into(),
            "Riffing on a current cultural moment".into(),
            "Banter in a group setting".into(),
            "Recommending a book / film / show with conviction".into(),
            "Asking a great question".into(),
            "Recovering from an awkward silence".into(),
            "Making fun of yourself well".into(),
            "Disagreeing without picking a fight".into(),
            "Holding the floor at a dinner party".into(),
            "First-date conversation".into(),
            "Bumping into someone you half-know".into(),
        ],
        background: String::new(),
        voice_id: 0,
    }
}

fn store_path() -> Option<PathBuf> {
    let mut p = dirs::data_dir()?;
    p.push("comms-coach");
    if let Err(e) = fs::create_dir_all(&p) {
        log::warn!("could not create personas dir: {e}");
        return None;
    }
    p.push("personas.json");
    Some(p)
}

pub fn load() -> PersonaStore {
    let Some(path) = store_path() else { return PersonaStore::default() };
    let Ok(bytes) = fs::read(&path) else { return PersonaStore::default() };
    match serde_json::from_slice::<PersonaStore>(&bytes) {
        Ok(mut s) => {
            // Recover gracefully from a hand-edited file rather than panicking
            // later when we index `personas[active]`.
            if s.personas.is_empty() {
                return PersonaStore::default();
            }
            if s.active >= s.personas.len() {
                s.active = 0;
            }
            s
        }
        Err(e) => {
            log::warn!("personas.json unreadable, using defaults: {e}");
            PersonaStore::default()
        }
    }
}

pub fn save(store: &PersonaStore) {
    let Some(path) = store_path() else { return };
    let bytes = match serde_json::to_vec_pretty(store) {
        Ok(b) => b,
        Err(e) => { log::warn!("personas serialize: {e}"); return; }
    };
    if let Err(e) = fs::write(&path, bytes) {
        log::warn!("personas write to {} failed: {e}", path.display());
    }
}

/// Pick a random topic from the persona's list. Falls back to a generic prompt
/// if the user has emptied the list — the app should still function.
pub fn pick_random_topic(persona: &Persona) -> String {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    persona
        .topics
        .choose(&mut rng)
        .cloned()
        .unwrap_or_else(|| "An open-ended speaking prompt".into())
}
