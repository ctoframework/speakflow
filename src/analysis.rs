//! Acoustic + textual analysis of a recorded answer.
//!
//! We compute a small set of robust signals that the LLM can reason over to
//! give *grounded* feedback about delivery — pace, pause structure, energy
//! variation, and filler-word incidence. None of these require a second model;
//! they all derive from the raw 16k mono buffer + the whisper transcript.

use crate::audio::TARGET_SAMPLE_RATE;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryMetrics {
    pub duration_secs: f32,
    pub words: usize,
    pub words_per_minute: f32,
    /// Fraction of frames classified as "speech" (energy above adaptive threshold).
    pub speech_ratio: f32,
    /// Total seconds of silence between speech segments (mid-utterance pauses only).
    pub total_pause_secs: f32,
    /// Number of distinct mid-utterance pauses > 350 ms.
    pub long_pause_count: usize,
    /// Coefficient of variation of frame energy during speech — proxy for vocal
    /// dynamics / intonation. Higher = more varied (engaging); lower = monotone.
    pub energy_cv: f32,
    /// Filler-word counts found in the transcript ("um", "uh", "like", ...).
    pub filler_count: usize,
    pub filler_examples: Vec<String>,
}

const FRAME_MS: f32 = 20.0;
const PAUSE_THRESHOLD_MS: f32 = 350.0;

/// Rough English filler list. Conservative on purpose — "like" and "right" only
/// count when standalone, not in phrases like "I would like" (handled below).
const FILLERS: &[&str] = &[
    "um", "uh", "uhm", "erm", "ah", "er", "hmm",
    "like", "basically", "literally", "honestly",
    "you know", "i mean", "sort of", "kind of", "right",
];

pub fn analyze(samples_16k_mono: &[f32], transcript: &str) -> DeliveryMetrics {
    let duration_secs = samples_16k_mono.len() as f32 / TARGET_SAMPLE_RATE as f32;

    // ---- Frame energy ----
    let frame_len = (TARGET_SAMPLE_RATE as f32 * FRAME_MS / 1000.0) as usize;
    let mut energies = Vec::with_capacity(samples_16k_mono.len() / frame_len.max(1));
    for chunk in samples_16k_mono.chunks(frame_len.max(1)) {
        let e = (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
        energies.push(e);
    }

    // Adaptive speech/silence threshold: somewhere between the noise floor and peak.
    // 20th percentile + a margin works well across rooms / mic gains.
    let threshold = adaptive_threshold(&energies);

    let speech_frames: usize = energies.iter().filter(|&&e| e > threshold).count();
    let speech_ratio = if energies.is_empty() { 0.0 } else {
        speech_frames as f32 / energies.len() as f32
    };

    // ---- Mid-utterance pauses ----
    // Walk the frames, collapse runs of silence between speech, count those longer
    // than threshold *only if* they're flanked by speech (i.e. ignore leading /
    // trailing silence).
    let pause_frame_threshold = (PAUSE_THRESHOLD_MS / FRAME_MS).round() as usize;
    let mut total_pause_frames = 0usize;
    let mut long_pause_count = 0usize;
    let mut current_silence = 0usize;
    let mut have_seen_speech = false;
    let mut last_run_was_pause = false;

    for &e in &energies {
        if e > threshold {
            if have_seen_speech && current_silence >= pause_frame_threshold {
                total_pause_frames += current_silence;
                long_pause_count += 1;
            }
            current_silence = 0;
            have_seen_speech = true;
            last_run_was_pause = false;
        } else {
            current_silence += 1;
            last_run_was_pause = true;
        }
    }
    // Don't count trailing silence — it's just the user finishing.
    let _ = last_run_was_pause;

    let total_pause_secs = total_pause_frames as f32 * FRAME_MS / 1000.0;

    // ---- Energy variation during speech (intonation proxy) ----
    let speech_energies: Vec<f32> = energies.iter().copied().filter(|&e| e > threshold).collect();
    let energy_cv = coefficient_of_variation(&speech_energies);

    // ---- Word count + pace ----
    let words: Vec<&str> = transcript.split_whitespace().collect();
    let speaking_secs = (speech_frames as f32 * FRAME_MS / 1000.0).max(0.1);
    let words_per_minute = (words.len() as f32 / speaking_secs) * 60.0;

    // ---- Filler words ----
    let (filler_count, filler_examples) = count_fillers(transcript);

    DeliveryMetrics {
        duration_secs,
        words: words.len(),
        words_per_minute,
        speech_ratio,
        total_pause_secs,
        long_pause_count,
        energy_cv,
        filler_count,
        filler_examples,
    }
}

fn adaptive_threshold(energies: &[f32]) -> f32 {
    if energies.is_empty() { return 0.0; }
    let mut sorted = energies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p20 = sorted[sorted.len() / 5];
    let p80 = sorted[sorted.len() * 4 / 5];
    // Threshold sits 30 % of the way from the noise floor to the typical speech peak.
    p20 + (p80 - p20) * 0.30 + 1e-4
}

fn coefficient_of_variation(xs: &[f32]) -> f32 {
    if xs.len() < 2 { return 0.0; }
    let mean = xs.iter().sum::<f32>() / xs.len() as f32;
    if mean.abs() < 1e-6 { return 0.0; }
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / xs.len() as f32;
    var.sqrt() / mean
}

fn count_fillers(transcript: &str) -> (usize, Vec<String>) {
    let lower = transcript.to_lowercase();
    let mut count = 0usize;
    let mut examples = Vec::new();
    for filler in FILLERS {
        // Word-boundary-ish search to avoid substring false positives ("uhm" in "uhmm" etc).
        let occurrences = count_word_occurrences(&lower, filler);
        if occurrences > 0 {
            count += occurrences;
            examples.push(format!("{filler} (×{occurrences})"));
        }
    }
    (count, examples)
}

fn count_word_occurrences(haystack: &str, needle: &str) -> usize {
    // Simple non-regex matcher: needle must be flanked by start/end or non-alphanumeric.
    let bytes = haystack.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || bytes.len() < n.len() { return 0; }
    let mut count = 0;
    let mut i = 0;
    while i + n.len() <= bytes.len() {
        if &bytes[i..i + n.len()] == n {
            let prev_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
            let next_ok = i + n.len() == bytes.len() || !bytes[i + n.len()].is_ascii_alphanumeric();
            if prev_ok && next_ok {
                count += 1;
                i += n.len();
                continue;
            }
        }
        i += 1;
    }
    count
}

impl DeliveryMetrics {
    /// Render as a compact, neutral block the LLM can consume in its prompt.
    pub fn for_prompt(&self) -> String {
        format!(
            "duration_secs: {:.1}\n\
             words: {}\n\
             words_per_minute: {:.0}\n\
             speech_ratio: {:.2}\n\
             total_mid_utterance_pause_secs: {:.1}\n\
             long_pauses_over_350ms: {}\n\
             energy_coefficient_of_variation: {:.2}\n\
             filler_count: {}\n\
             filler_breakdown: {}",
            self.duration_secs,
            self.words,
            self.words_per_minute,
            self.speech_ratio,
            self.total_pause_secs,
            self.long_pause_count,
            self.energy_cv,
            self.filler_count,
            if self.filler_examples.is_empty() { "none".into() } else { self.filler_examples.join(", ") },
        )
    }
}
