//! Coaching orchestration: themes, follow-ups, and final feedback prompts.
//!
//! The coaching style and topic list are driven by the active `Persona`. The
//! built-in defaults cover an executive engineering coach and a casual social
//! coach, but the user can edit either or add their own.

use crate::analysis::DeliveryMetrics;
use crate::llm::ChatMessage;
use crate::personas::{self, Persona};

/// Pick a topic from the persona's configured list.
pub fn pick_topic(persona: &Persona) -> String {
    personas::pick_random_topic(persona)
}

/// Build the system prompt by weaving the persona's description and any
/// background context into a fixed coaching shell. Doing this in code rather
/// than storing the whole prompt on disk keeps the "tone" rules consistent
/// across personas — a user editing a persona only changes audience and
/// topics, not the bar for feedback quality.
fn coach_system(persona: &Persona) -> String {
    let mut s = format!(
        "You are {}. \
         Your bar is high. Avoid platitudes, generic encouragement, and filler praise. \
         Be specific, candid, and concise. Speak to them as a peer-level coach.",
        persona.description.trim()
    );
    let bg = persona.background.trim();
    if !bg.is_empty() {
        s.push_str("\n\nBackground on your client (use this to ground your feedback):\n");
        s.push_str(bg);
    }
    s
}

/// Ask the model to propose a single tight speaking prompt for the picked topic.
pub fn theme_messages(persona: &Persona, bucket: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage::system(coach_system(persona)),
        ChatMessage::user(format!(
            "Propose ONE speaking exercise for today's session in the area: \"{bucket}\".\n\n\
             Format your reply EXACTLY as:\n\
             THEME: <a 4–8 word title>\n\
             PROMPT: <one to two sentences asking the client to speak for 1–3 minutes \
             on something concrete — a position to take, a story to tell, or a specific \
             scenario to navigate. Not a definition or textbook recap.>\n\n\
             No preamble, no closing remarks."
        )),
    ]
}

/// Generate 1–3 follow-up questions based on what they actually said.
pub fn followup_messages(
    persona: &Persona,
    theme: &str,
    prompt: &str,
    transcript: &str,
) -> Vec<ChatMessage> {
    vec![
        ChatMessage::system(coach_system(persona)),
        ChatMessage::user(format!(
            "The speaker just answered the following exercise.\n\n\
             THEME: {theme}\n\
             ORIGINAL PROMPT: {prompt}\n\n\
             WHAT THEY SAID (auto-transcribed, may have minor STT errors):\n\
             ---\n{transcript}\n---\n\n\
             Write 1 to 3 sharp follow-up questions a thoughtful listener would ask \
             after hearing that answer. Each question should probe a weak point, an \
             unstated assumption, or something they glossed over. Number them. \
             No preamble. Maximum 3 questions."
        )),
    ]
}

/// Final feedback. We feed the LLM the *measured* delivery metrics so it can
/// reason about pace / pauses / fillers from data, not guesswork.
pub fn feedback_messages(
    persona: &Persona,
    theme: &str,
    prompt: &str,
    transcript: &str,
    metrics: &DeliveryMetrics,
) -> Vec<ChatMessage> {
    let metrics_block = metrics.for_prompt();
    vec![
        ChatMessage::system(coach_system(persona)),
        ChatMessage::user(format!(
            "Provide structured spoken-communication feedback on the answer below.\n\n\
             THEME: {theme}\n\
             ORIGINAL PROMPT: {prompt}\n\n\
             AUTO-TRANSCRIPT (verbatim, may include STT errors — do not nitpick spelling):\n\
             ---\n{transcript}\n---\n\n\
             MEASURED DELIVERY METRICS (computed from the audio):\n\
             ---\n{metrics_block}\n---\n\n\
             Reference numbers for context: \
             a comfortable speaking pace is 130–160 wpm; \
             frequent mid-utterance pauses can either signal weight or hesitation; \
             energy_coefficient_of_variation below ~0.4 tends to read as monotone; \
             more than ~3 fillers per minute starts to undercut presence.\n\n\
             Produce feedback in EXACTLY this format using markdown headings:\n\n\
             ## Overall impression\n\
             Two sentences. State the strongest thing and the single biggest issue.\n\n\
             ## Pace & rhythm\n\
             Reference the measured wpm and pause behaviour. Say what to adjust.\n\n\
             ## Articulation & clarity\n\
             Comment on whether ideas landed crisply. Quote (briefly) any sentence \
             that was structurally muddled and rewrite it more cleanly.\n\n\
             ## Intonation & energy\n\
             Use the energy variation metric. If monotone, suggest where to vary. \
             If varied well, say where it worked.\n\n\
             ## Filler words & verbal tics\n\
             Reference the measured filler counts. Be specific.\n\n\
             ## Substance\n\
             Was the position or story clear? Did key ideas land? Would this work \
             for the intended audience for this persona? Be candid.\n\n\
             ## One thing to practise next time\n\
             A single, concrete drill they can run in 5 minutes.\n\n\
             Be tight — the entire response should fit in roughly 350–500 words. \
             No closing pep-talk."
        )),
    ]
}

/// Parse the THEME / PROMPT block out of the model's first reply.
/// Falls back gracefully if the model didn't follow format.
pub fn parse_theme_response(raw: &str) -> (String, String) {
    let mut theme = String::new();
    let mut prompt = String::new();
    for line in raw.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("THEME:").or_else(|| line.strip_prefix("Theme:")) {
            theme = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("PROMPT:").or_else(|| line.strip_prefix("Prompt:")) {
            prompt = rest.trim().to_string();
        } else if !prompt.is_empty() && !line.is_empty() {
            // Continuation of multi-line prompt.
            prompt.push(' ');
            prompt.push_str(line);
        }
    }
    if theme.is_empty() { theme = "Speaking exercise".into(); }
    if prompt.is_empty() { prompt = raw.trim().to_string(); }
    (theme, prompt)
}
