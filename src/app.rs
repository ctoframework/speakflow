//! Egui application: a small finite state machine driving the session.
//!
//! States: Welcome -> ProposingTheme -> ConfirmReady -> Recording -> Transcribing
//!         -> AskingFollowups -> RecordingFollowup* -> GeneratingFeedback -> Feedback
//!
//! Long-running work (LLM calls, whisper inference) runs on the tokio runtime
//! handed in from main(). The UI thread receives partial updates via mpsc and
//! requests a repaint when new bytes arrive.

use crate::analysis::{analyze, DeliveryMetrics};
use crate::audio::Recorder;
use crate::coach;
use crate::history::{self, Session};
use crate::llm::{ChatMessage, LlmUpdate, OllamaClient};
use crate::personas::{self, Persona, PersonaStore, KOKORO_VOICES};
use crate::stt::Transcriber;
use crate::tts::{Player, Tts};

use eframe::CreationContext;
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::runtime::Handle;
use tokio::sync::mpsc;

const DEFAULT_MODEL: &str = "llama3.1:8b";
const MIN_RECORD_SECS: f32 = 20.0;
const MAX_RECORD_SECS: f32 = 180.0;
const FOLLOWUP_MAX_SECS: f32 = 90.0;

#[derive(Default)]
enum Stage {
    #[default]
    Welcome,
    ProposingTheme,
    ConfirmReady {
        theme: String,
        prompt: String,
    },
    Recording {
        theme: String,
        prompt: String,
        recorder: Recorder,
    },
    Transcribing {
        theme: String,
        prompt: String,
        samples: Vec<f32>,
    },
    AskingFollowups {
        theme: String,
        prompt: String,
        transcript: String,
        samples: Vec<f32>,
    },
    RecordingFollowup {
        theme: String,
        prompt: String,
        primary_transcript: String,
        primary_samples: Vec<f32>,
        followups: Vec<String>,
        current: usize,
        followup_answers: Vec<String>,
        recorder: Recorder,
    },
    TranscribingFollowup {
        theme: String,
        prompt: String,
        primary_transcript: String,
        primary_samples: Vec<f32>,
        followups: Vec<String>,
        current: usize,
        followup_answers: Vec<String>,
        samples: Vec<f32>,
    },
    GeneratingFeedback {
        theme: String,
        prompt: String,
        transcript: String,
        followups: Vec<String>,
        followup_answers: Vec<String>,
        metrics: DeliveryMetrics,
    },
    Feedback {
        theme: String,
        prompt: String,
        transcript: String,
        followups: Vec<String>,
        followup_answers: Vec<String>,
        metrics: DeliveryMetrics,
        feedback: String,
        saved: bool,
    },
    History {
        selected: Option<usize>,
    },
    Trends,
    Personas,
    EditPersona {
        draft: PersonaDraft,
    },
    Debug {
        test_recorder: Option<Recorder>,
        test_result: Option<Result<crate::stt::TranscriptionResult, String>>,
    },
    Error(String),
}

/// In-progress edit of a persona. `topics_text` is a single string the user
/// edits as multi-line text; we split it on newlines on save. Storing it that
/// way keeps the text-editing experience natural — moving lines around, pasting
/// a list, etc. — without forcing a per-row UI.
#[derive(Clone)]
pub struct PersonaDraft {
    /// `None` when creating a new persona; `Some(i)` when editing existing.
    edit_idx: Option<usize>,
    name: String,
    description: String,
    topics_text: String,
    background: String,
    voice_id: u32,
}

impl PersonaDraft {
    fn new() -> Self {
        Self {
            edit_idx: None,
            name: String::new(),
            description: String::new(),
            topics_text: String::new(),
            background: String::new(),
            voice_id: 0,
        }
    }

    fn from_persona(idx: Option<usize>, p: &Persona) -> Self {
        Self {
            edit_idx: idx,
            name: p.name.clone(),
            description: p.description.clone(),
            topics_text: p.topics.join("\n"),
            background: p.background.clone(),
            voice_id: p.voice_id,
        }
    }

    fn into_persona(self) -> Persona {
        let topics: Vec<String> = self
            .topics_text
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        Persona {
            name: self.name.trim().to_string(),
            description: self.description.trim().to_string(),
            topics,
            background: self.background.trim().to_string(),
            voice_id: self.voice_id,
        }
    }
}

impl Stage {
    /// Stages where it's safe to switch the active persona or navigate to the
    /// personas screen. We block this during anything in-flight (recording,
    /// transcribing, streaming, or an unsaved edit) to avoid mid-session
    /// inconsistency or losing in-progress text.
    fn is_stable(&self) -> bool {
        matches!(
            self,
            Stage::Welcome
                | Stage::Feedback { .. }
                | Stage::History { .. }
                | Stage::Trends
                | Stage::Personas
                | Stage::Debug { .. }
                | Stage::Error(_)
        )
    }
}

/// Streaming text shared between the worker tokio task and the UI.
#[derive(Default, Clone)]
struct StreamBuffer(Arc<Mutex<String>>);
impl StreamBuffer {
    fn snapshot(&self) -> String {
        self.0.lock().clone()
    }
    fn clear(&self) {
        self.0.lock().clear()
    }
    fn append(&self, s: &str) {
        self.0.lock().push_str(s);
    }
}

pub struct CoachApp {
    rt: Handle,
    ollama: OllamaClient,
    transcriber: Option<Arc<Transcriber>>, // None if model file is missing
    transcriber_error: Option<String>,

    /// Optional local TTS engine (kokoros). `None` when the binary isn't
    /// installed — the app stays fully usable, just text-only.
    tts: Option<Arc<Tts>>,
    tts_error: Option<String>,
    /// Background audio player thread. Always present; play/stop calls are
    /// no-ops if rodio's output stream couldn't open.
    player: Player,

    stage: Stage,

    /// Current streaming text (theme proposal, follow-ups, feedback) shared with worker.
    stream: StreamBuffer,
    /// One-shot completion signal from the worker for the current stream.
    stream_done_rx: Option<mpsc::UnboundedReceiver<StreamDone>>,

    /// Receiver for STT completions.
    stt_rx: Option<mpsc::UnboundedReceiver<SttResult>>,

    model_name: String,

    /// All persisted sessions, oldest first. Loaded once at startup; appended
    /// to in memory whenever a new session reaches the Feedback stage so the
    /// user sees their latest run immediately in History / Trends.
    sessions: Vec<Session>,

    /// Available personas + currently active one. Persisted to disk; saved
    /// whenever the user changes the active selection or edits the list.
    persona_store: PersonaStore,

    /// UI state for the personas list: which persona is awaiting history clear.
    confirm_clear_idx: Option<usize>,
}

enum StreamDone {
    Ok(String),
    Err(String),
}

enum SttResult {
    Ok(crate::stt::TranscriptionResult),
    Err(String),
}

impl CoachApp {
    pub fn new(_cc: &CreationContext<'_>, rt: Handle) -> Self {
        let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let ollama = OllamaClient::new(model.clone());

        // Try to load whisper eagerly — it's the most likely thing to fail (missing model).
        let (transcriber, transcriber_error) = match Transcriber::load() {
            Ok(t) => (Some(Arc::new(t)), None),
            Err(e) => (None, Some(format!("{e:#}"))),
        };

        // TTS is purely additive: if kokoros isn't installed we just stay
        // text-only. Don't gate the app on it.
        let (tts, tts_error) = match Tts::load() {
            Ok(t) => (Some(Arc::new(t)), None),
            Err(e) => (None, Some(format!("{e:#}"))),
        };
        let player = Player::spawn();

        let persona_store = personas::load();

        let mut app = Self {
            rt,
            ollama,
            transcriber,
            transcriber_error,
            tts,
            tts_error,
            player,
            stage: Stage::Welcome,
            stream: StreamBuffer::default(),
            stream_done_rx: None,
            stt_rx: None,
            model_name: model,
            sessions: Vec::new(),
            persona_store,
            confirm_clear_idx: None,
        };
        app.reload_sessions();
        app
    }

    // -------------------------------------------------------------------- TTS

    /// Read `text` aloud through kokoros, replacing any currently-playing
    /// clip. No-op when TTS isn't configured. Synthesis runs on the tokio
    /// blocking pool so the UI stays responsive while kokoros works.
    fn speak(&self, text: String) {
        let voice_id = self.persona_store.active().voice_id;
        self.speak_with_voice_id(text, voice_id);
    }

    fn speak_with_voice_id(&self, text: String, voice_id: u32) {
        let Some(tts) = self.tts.clone() else {
            return;
        };
        if text.trim().is_empty() {
            return;
        }
        let player = self.player.clone();
        // Stop any prior clip immediately — synthesis may take a beat, and
        // we don't want the previous question still playing under the new one.
        // Mark busy before spawning so the UI disables the replay button at once.
        player.stop();
        player.mark_busy();
        self.rt
            .spawn_blocking(move || match tts.synthesize_to_temp_wav(&text, voice_id) {
                Ok(path) => {
                    let bytes = std::fs::read(&path);
                    let _ = std::fs::remove_file(&path);
                    match bytes {
                        Ok(b) => player.play_wav_bytes(b),
                        Err(e) => {
                            log::warn!("TTS read-back failed: {e:#}");
                            player.stop();
                        }
                    }
                }
                Err(e) => {
                    log::warn!("TTS synth failed: {e:#}");
                    player.stop();
                }
            });
    }

    fn stop_speaking(&self) {
        self.player.stop();
    }

    fn tts_enabled(&self) -> bool {
        self.tts.is_some()
    }

    // -------------------------------------------------------------------- workers

    fn launch_stream(
        &mut self,
        messages: Vec<ChatMessage>,
        temperature: f32,
        max_tokens: i32,
        ctx: egui::Context,
    ) {
        self.stream.clear();
        let (done_tx, done_rx) = mpsc::unbounded_channel();
        self.stream_done_rx = Some(done_rx);

        let buf = self.stream.clone();
        let ollama = self.ollama.clone();
        let ctx_for_repaint = ctx.clone();

        self.rt.spawn(async move {
            let (tok_tx, mut tok_rx) = mpsc::unbounded_channel::<LlmUpdate>();
            let ollama2 = ollama.clone();
            tokio::spawn(async move {
                ollama2
                    .chat_stream(messages, temperature, max_tokens, tok_tx)
                    .await;
            });

            while let Some(update) = tok_rx.recv().await {
                match update {
                    LlmUpdate::Token(t) => {
                        buf.append(&t);
                        ctx_for_repaint.request_repaint();
                    }
                    LlmUpdate::Done(full) => {
                        let _ = done_tx.send(StreamDone::Ok(full));
                        ctx_for_repaint.request_repaint();
                        break;
                    }
                    LlmUpdate::Error(e) => {
                        let _ = done_tx.send(StreamDone::Err(e));
                        ctx_for_repaint.request_repaint();
                        break;
                    }
                }
            }
        });
    }

    /// Spawn whisper transcription on a blocking task — it's CPU-bound.
    fn launch_transcription(&mut self, samples: Vec<f32>, ctx: egui::Context) {
        let (tx, rx) = mpsc::unbounded_channel();
        self.stt_rx = Some(rx);
        let transcriber = match self.transcriber.as_ref() {
            Some(t) => t.clone(),
            None => {
                let _ = tx.send(SttResult::Err(
                    self.transcriber_error
                        .clone()
                        .unwrap_or_else(|| "whisper not loaded".into()),
                ));
                return;
            }
        };
        let ctx2 = ctx.clone();
        self.rt.spawn_blocking(move || {
            let result = match transcriber.transcribe(&samples) {
                Ok(res) => SttResult::Ok(res),
                Err(e) => SttResult::Err(format!("{e:#}")),
            };
            let _ = tx.send(result);
            ctx2.request_repaint();
        });
    }

    fn reload_sessions(&mut self) {
        let legacy_persona = personas::default_engineering_leader().name;
        self.sessions = history::load_all()
            .into_iter()
            .map(|mut s| {
                if s.persona_name.is_empty() {
                    s.persona_name = legacy_persona.clone();
                }
                s
            })
            .collect();
    }

    // -------------------------------------------------------------------- session filtering

    /// Absolute indices into `self.sessions` for rows belonging to the active
    /// persona, in original (oldest-first) order. We return indices rather
    /// than references so callers can use them as stable identifiers — the
    /// History stage stores `selected` as an absolute index.
    fn active_persona_indices(&self) -> Vec<usize> {
        let name = &self.persona_store.active().name;
        self.sessions
            .iter()
            .enumerate()
            .filter(|(_, s)| &s.persona_name == name)
            .map(|(i, _)| i)
            .collect()
    }

    // -------------------------------------------------------------------- update tick

    /// Drain any pending worker messages and advance the state machine.
    fn poll_workers(&mut self, ctx: &egui::Context) {
        // Stream completion — applies to whichever stage spawned it.
        if let Some(rx) = self.stream_done_rx.as_mut() {
            if let Ok(done) = rx.try_recv() {
                self.stream_done_rx = None;
                self.handle_stream_done(done, ctx);
            }
        }
        // STT completion.
        if let Some(rx) = self.stt_rx.as_mut() {
            if let Ok(result) = rx.try_recv() {
                self.stt_rx = None;
                match result {
                    SttResult::Ok(res) => self.handle_stt_done(res, ctx),
                    SttResult::Err(err) => self.handle_stt_error(err),
                }
            }
        }
    }

    fn handle_stream_done(&mut self, done: StreamDone, ctx: &egui::Context) {
        let text = match done {
            StreamDone::Ok(t) => t,
            StreamDone::Err(e) => {
                self.stage = Stage::Error(format!(
                    "Ollama error: {e}\n\nIs `ollama serve` running and is the model `{}` pulled?\n\
                     Try: `ollama pull {}`", self.model_name, self.model_name
                ));
                return;
            }
        };

        // Take ownership of the current stage so we can match-and-replace.
        let cur = std::mem::replace(&mut self.stage, Stage::Welcome);
        match cur {
            Stage::ProposingTheme => {
                let (theme, prompt) = coach::parse_theme_response(&text);
                // Read the prompt aloud so the user hears today's exercise as
                // well as seeing it. The theme is a label, not a sentence —
                // skipping it sounds more natural.
                self.speak(prompt.clone());
                self.stage = Stage::ConfirmReady { theme, prompt };
            }
            Stage::AskingFollowups {
                theme,
                prompt,
                transcript,
                samples,
                ..
            } => {
                let followups = parse_numbered_questions(&text);
                if followups.is_empty() {
                    // No parseable follow-ups: skip straight to feedback.
                    let metrics = analyze(&samples, &transcript);
                    let persona = self.persona_store.active().clone();
                    let msgs =
                        coach::feedback_messages(&persona, &theme, &prompt, &transcript, &metrics);
                    self.stage = Stage::GeneratingFeedback {
                        theme,
                        prompt,
                        transcript,
                        followups: vec![],
                        followup_answers: vec![],
                        metrics,
                    };
                    self.launch_stream(msgs, 0.4, 700, ctx.clone());
                } else {
                    // Speak the first follow-up as we open the recording stage.
                    self.speak(followups[0].clone());
                    self.stage = Stage::RecordingFollowup {
                        theme,
                        prompt,
                        primary_transcript: transcript,
                        primary_samples: samples,
                        followups,
                        current: 0,
                        followup_answers: vec![],
                        recorder: match Recorder::start() {
                            Ok(r) => r,
                            Err(e) => {
                                self.stage = Stage::Error(format!("Mic error: {e:#}"));
                                return;
                            }
                        },
                    };
                }
            }
            Stage::GeneratingFeedback {
                theme,
                prompt,
                transcript,
                followups,
                followup_answers,
                metrics,
            } => {
                self.stage = Stage::Feedback {
                    theme,
                    prompt,
                    transcript,
                    followups,
                    followup_answers,
                    metrics,
                    feedback: text,
                    saved: false,
                };
            }
            other => {
                // Stream finished for a stage that didn't expect it — just restore.
                self.stage = other;
            }
        }
    }

    fn handle_stt_done(&mut self, result: crate::stt::TranscriptionResult, ctx: &egui::Context) {
        let transcript = result.text.clone();
        let cur = std::mem::replace(&mut self.stage, Stage::Welcome);
        match cur {
            Stage::Transcribing {
                theme,
                prompt,
                samples,
            } => {
                // Generate follow-ups using the transcript.
                let persona = self.persona_store.active().clone();
                let msgs = coach::followup_messages(&persona, &theme, &prompt, &transcript);
                self.stage = Stage::AskingFollowups {
                    theme,
                    prompt,
                    transcript,
                    samples,
                };
                self.launch_stream(msgs, 0.5, 300, ctx.clone());
            }
            Stage::TranscribingFollowup {
                theme,
                prompt,
                primary_transcript,
                primary_samples,
                followups,
                current,
                mut followup_answers,
                samples: _,
            } => {
                followup_answers.push(transcript);
                let next = current + 1;
                if next < followups.len() {
                    // Speak the next follow-up as we re-open the recorder.
                    self.speak(followups[next].clone());
                    self.stage = Stage::RecordingFollowup {
                        theme,
                        prompt,
                        primary_transcript,
                        primary_samples,
                        followups,
                        current: next,
                        followup_answers,
                        recorder: match Recorder::start() {
                            Ok(r) => r,
                            Err(e) => {
                                self.stage = Stage::Error(format!("Mic error: {e:#}"));
                                return;
                            }
                        },
                    };
                } else {
                    // All follow-ups answered — generate final feedback using the primary answer's metrics.
                    let metrics = analyze(&primary_samples, &primary_transcript);
                    let persona = self.persona_store.active().clone();
                    let msgs = coach::feedback_messages(
                        &persona,
                        &theme,
                        &prompt,
                        &primary_transcript,
                        &metrics,
                    );
                    self.stage = Stage::GeneratingFeedback {
                        theme,
                        prompt,
                        transcript: primary_transcript,
                        followups,
                        followup_answers,
                        metrics,
                    };
                    self.launch_stream(msgs, 0.4, 700, ctx.clone());
                }
            }
            Stage::Debug { test_recorder, .. } => {
                self.stage = Stage::Debug {
                    test_recorder,
                    test_result: Some(Ok(result)),
                };
            }
            other => self.stage = other,
        }
    }

    fn handle_stt_error(&mut self, err: String) {
        let cur = std::mem::replace(&mut self.stage, Stage::Welcome);
        match cur {
            Stage::Debug { test_recorder, .. } => {
                self.stage = Stage::Debug {
                    test_recorder,
                    test_result: Some(Err(err)),
                };
            }
            _ => {
                self.stage = Stage::Error(format!("Transcription error: {err}"));
            }
        }
    }
}

fn parse_numbered_questions(raw: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Accept "1.", "1)", "1 -" etc.
        let bytes = line.as_bytes();
        let mut i = 0;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i > 0 && i < bytes.len() && matches!(bytes[i], b'.' | b')' | b'-' | b':') {
            let q = line[i + 1..].trim().to_string();
            if !q.is_empty() {
                out.push(q);
            }
        }
    }
    // Cap at 3 to honour the spec.
    out.truncate(3);
    out
}

// ============================================================== eframe::App

impl eframe::App for CoachApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_workers(ctx);

        // Recording stages need continuous repaint for the timer / VU meter.
        // Also repaint while audio is busy so the replay button re-enables promptly.
        if matches!(
            self.stage,
            Stage::Recording { .. } | Stage::RecordingFollowup { .. }
        ) || self.player.is_busy()
        {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }

        // Persona switching happens in the top bar. We capture the user's
        // intent into locals here, then act on them after the panel returns —
        // mutating self.stage inside the closure would fight the borrow
        // checker against the central panel's mem::replace below.
        let stable = self.stage.is_stable();
        let mut new_active: Option<usize> = None;
        let mut go_personas_top = false;
        let mut go_debug_top = false;
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("🎙  Communications Coach");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_enabled_ui(stable, |ui| {
                        if ui.link("View config").clicked() {
                            go_debug_top = true;
                        }
                    });
                    ui.add_space(12.0);
                    ui.add_enabled_ui(stable, |ui| {
                        let active_idx = self.persona_store.active;
                        let active_name = self.persona_store.active().name.clone();
                        egui::ComboBox::from_id_salt("persona_picker")
                            .selected_text(format!("👤  {}", active_name))
                            .show_ui(ui, |ui| {
                                for (i, p) in self.persona_store.personas.iter().enumerate() {
                                    if ui.selectable_label(i == active_idx, &p.name).clicked()
                                        && i != active_idx
                                    {
                                        new_active = Some(i);
                                    }
                                }
                                ui.separator();
                                if ui.selectable_label(false, "Manage personas…").clicked() {
                                    go_personas_top = true;
                                }
                            });
                    });
                    if !stable {
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new("(locked during session)")
                                .weak()
                                .small(),
                        );
                    }
                });
            });
            ui.add_space(2.0);
            ui.separator();
        });

        if let Some(i) = new_active {
            self.persona_store.active = i;
            personas::save(&self.persona_store);
            // The History stage's `selected` is an absolute index into
            // self.sessions; once the filter changes it could point at a
            // session of a different persona. Reset it so the detail pane
            // shows something meaningful.
            if let Stage::History { selected } = &mut self.stage {
                *selected = None;
            }
        }
        if go_personas_top && stable {
            self.stage = Stage::Personas;
        }
        if go_debug_top && stable {
            self.stage = Stage::Debug {
                test_recorder: None,
                test_result: None,
            };
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(8.0);
            // Take ownership of the stage briefly so we can mutate self while rendering.
            // We split rendering into a method that returns the (possibly new) stage.
            let stage = std::mem::replace(&mut self.stage, Stage::Welcome);
            let new_stage = self.render_stage(ui, ctx, stage);
            self.stage = new_stage;
        });
    }
}

// ============================================================== rendering

impl CoachApp {
    fn render_stage(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, stage: Stage) -> Stage {
        match stage {
            Stage::Welcome => self.render_welcome(ui, ctx),
            Stage::ProposingTheme => self.render_proposing(ui),
            Stage::ConfirmReady { theme, prompt } => self.render_confirm(ui, ctx, theme, prompt),
            Stage::Recording {
                theme,
                prompt,
                recorder,
            } => self.render_recording(ui, ctx, theme, prompt, recorder),
            Stage::Transcribing {
                theme,
                prompt,
                samples,
            } => {
                self.spinner_with_label(ui, "Transcribing your answer locally with Whisper…");
                Stage::Transcribing {
                    theme,
                    prompt,
                    samples,
                }
            }
            Stage::AskingFollowups {
                theme,
                prompt,
                transcript,
                samples,
                ..
            } => {
                self.render_followups_streaming(ui, &theme, &transcript);

                Stage::AskingFollowups {
                    theme,
                    prompt,
                    transcript,
                    samples,
                }
            }
            Stage::RecordingFollowup {
                theme,
                prompt,
                primary_transcript,
                primary_samples,
                followups,
                current,
                followup_answers,
                recorder,
            } => self.render_recording_followup(
                ui,
                ctx,
                theme,
                prompt,
                primary_transcript,
                primary_samples,
                followups,
                current,
                followup_answers,
                recorder,
            ),
            Stage::TranscribingFollowup {
                theme,
                prompt,
                primary_transcript,
                primary_samples,
                followups,
                current,
                followup_answers,
                samples,
            } => {
                self.spinner_with_label(
                    ui,
                    &format!(
                        "Transcribing follow-up {}/{}…",
                        current + 1,
                        followups.len()
                    ),
                );
                Stage::TranscribingFollowup {
                    theme,
                    prompt,
                    primary_transcript,
                    primary_samples,
                    followups,
                    current,
                    followup_answers,
                    samples,
                }
            }
            Stage::GeneratingFeedback {
                theme,
                prompt,
                transcript,
                followups,
                followup_answers,
                metrics,
            } => {
                self.render_feedback_streaming(ui, &metrics);
                Stage::GeneratingFeedback {
                    theme,
                    prompt,
                    transcript,
                    followups,
                    followup_answers,
                    metrics,
                }
            }
            Stage::Feedback {
                theme,
                prompt,
                transcript,
                followups,
                followup_answers,
                metrics,
                feedback,
                saved,
            } => {
                // Persist on first render of this stage. We do it here (not in
                // handle_stream_done) so the saved row already includes the
                // fully-streamed feedback text.
                let saved = if !saved {
                    let session = Session::now(
                        theme.clone(),
                        prompt.clone(),
                        transcript.clone(),
                        followups.clone(),
                        followup_answers.clone(),
                        metrics.clone(),
                        feedback.clone(),
                        self.persona_store.active().name.clone(),
                    );
                    history::append(&session);
                    self.sessions.push(session);
                    true
                } else {
                    saved
                };
                self.render_final_feedback(
                    ui,
                    &theme,
                    &prompt,
                    &transcript,
                    &followups,
                    &followup_answers,
                    &metrics,
                    &feedback,
                    saved,
                )
            }
            Stage::History { selected } => self.render_history(ui, selected),
            Stage::Trends => self.render_trends(ui),
            Stage::Personas => self.render_personas_list(ui),
            Stage::EditPersona { draft } => self.render_edit_persona(ui, draft),
            Stage::Debug {
                test_recorder,
                test_result,
            } => self.render_debug(ui, ctx, test_recorder, test_result),
            Stage::Error(msg) => self.render_error(ui, msg),
        }
    }

    fn render_welcome(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) -> Stage {
        ui.vertical_centered(|ui| {
            ui.add_space(40.0);
            ui.label(
                egui::RichText::new("Sharpen how you speak in high-stakes moments.").size(18.0),
            );
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(
                    "Each session: I propose a topic. You speak for 1–3 minutes. I ask 1–3 \
                 follow-ups. Then I give you grounded feedback on pace, articulation, \
                 intonation, fillers and substance.",
                )
                .weak(),
            );
            ui.add_space(24.0);
        });

        if let Some(err) = &self.transcriber_error {
            ui.colored_label(
                egui::Color32::from_rgb(220, 120, 120),
                format!("Whisper not ready: {err}"),
            );
            ui.add_space(8.0);
        }
        // TTS is optional. Surface its status quietly — informational, not red.
        if let Some(err) = &self.tts_error {
            ui.label(
                egui::RichText::new(format!(
                    "Voice playback off (sherpa-onnx not configured): {err}"
                ))
                .weak()
                .small(),
            );
            ui.add_space(4.0);
        }

        let mut clicked = false;
        let mut go_history = false;
        let mut go_trends = false;
        let active_name = self.persona_store.active().name.clone();
        let topic_count = self.persona_store.active().topics.len();
        let session_count = self.active_persona_indices().len();
        ui.vertical_centered(|ui| {
            ui.label(
                egui::RichText::new(format!(
                "Persona: {active_name} · {topic_count} topic{} · {session_count} past session{}",
                if topic_count == 1 { "" } else { "s" },
                if session_count == 1 { "" } else { "s" },
            ))
                .weak(),
            );
            ui.add_space(8.0);
            let btn =
                egui::Button::new(egui::RichText::new("  Start a coaching exercise  ").size(16.0))
                    .min_size(egui::vec2(260.0, 44.0));
            if ui.add(btn).clicked() {
                clicked = true;
            }
            ui.add_space(12.0);
            ui.horizontal(|ui| {
                ui.add_space(ui.available_width() / 2.0 - 170.0);
                let history_label = if session_count == 0 {
                    "📜  Past exercises".to_string()
                } else {
                    format!("📜  Past exercises ({session_count})")
                };
                if ui
                    .add_enabled(
                        session_count > 0,
                        egui::Button::new(history_label).min_size(egui::vec2(170.0, 32.0)),
                    )
                    .clicked()
                {
                    go_history = true;
                }
                ui.add_space(8.0);
                if ui
                    .add_enabled(
                        session_count > 0,
                        egui::Button::new("📈  Metrics over time")
                            .min_size(egui::vec2(170.0, 32.0)),
                    )
                    .clicked()
                {
                    go_trends = true;
                }
            });
        });

        if clicked {
            let persona = self.persona_store.active().clone();
            let bucket = coach::pick_topic(&persona);
            let msgs = coach::theme_messages(&persona, &bucket);
            self.launch_stream(msgs, 0.8, 200, ctx.clone());
            return Stage::ProposingTheme;
        }
        if go_history {
            return Stage::History { selected: None };
        }
        if go_trends {
            return Stage::Trends;
        }
        Stage::Welcome
    }

    fn render_proposing(&mut self, ui: &mut egui::Ui) -> Stage {
        ui.add_space(20.0);
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Picking a theme for today…");
        });
        ui.add_space(12.0);
        let live = self.stream.snapshot();
        if !live.is_empty() {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label(egui::RichText::new(live).monospace().weak());
            });
        }
        Stage::ProposingTheme
    }

    fn render_confirm(
        &mut self,
        ui: &mut egui::Ui,
        _ctx: &egui::Context,
        theme: String,
        prompt: String,
    ) -> Stage {
        ui.add_space(8.0);
        ui.label(egui::RichText::new("Today's theme").weak());
        ui.heading(&theme);
        ui.add_space(12.0);
        egui::Frame::group(ui.style())
            .fill(ui.style().visuals.faint_bg_color)
            .inner_margin(12.0)
            .show(ui, |ui| {
                ui.label(egui::RichText::new(&prompt).size(15.0));
            });
        let mut replay = false;
        if self.tts_enabled() {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let busy = self.player.is_busy();
                if ui
                    .add_enabled(!busy, egui::Button::new("🔊  Replay").small())
                    .clicked()
                {
                    replay = true;
                }
                ui.label(
                    egui::RichText::new("Reading aloud via Kokoro")
                        .weak()
                        .small(),
                );
            });
        }
        ui.add_space(16.0);
        ui.label(
            egui::RichText::new(format!(
                "When you click Start, I'll record for up to {:.0} seconds. \
                     You can stop earlier (after at least {:.0}s).",
                MAX_RECORD_SECS, MIN_RECORD_SECS
            ))
            .weak(),
        );
        ui.add_space(16.0);

        let mut start = false;
        let mut cancel = false;
        ui.horizontal(|ui| {
            if ui
                .add(
                    egui::Button::new(
                        egui::RichText::new("🔴  I'm ready — start recording").size(15.0),
                    )
                    .min_size(egui::vec2(280.0, 40.0)),
                )
                .clicked()
            {
                start = true;
            }
            ui.add_space(8.0);
            if ui.button("Cancel").clicked() {
                cancel = true;
            }
        });

        if replay {
            self.speak(prompt.clone());
        }
        if start {
            // Cut the prompt audio so it doesn't play under the user's voice.
            self.stop_speaking();
            return match Recorder::start() {
                Ok(recorder) => Stage::Recording {
                    theme,
                    prompt,
                    recorder,
                },
                Err(e) => Stage::Error(format!("Mic error: {e:#}")),
            };
        }
        if cancel {
            self.stop_speaking();
            return Stage::Welcome;
        }
        Stage::ConfirmReady { theme, prompt }
    }

    fn render_recording(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        theme: String,
        prompt: String,
        recorder: Recorder,
    ) -> Stage {
        let elapsed = recorder.elapsed_secs();
        let level = recorder.vu_level();

        ui.label(egui::RichText::new(&theme).weak());
        ui.add_space(4.0);
        ui.label(egui::RichText::new(&prompt).size(14.0));
        ui.separator();
        ui.add_space(20.0);

        let mut stop_now = elapsed >= MAX_RECORD_SECS;
        ui.vertical_centered(|ui| {
            ui.heading(format!(
                "● Recording  {:>5.1}s / {:.0}s",
                elapsed, MAX_RECORD_SECS
            ));
            ui.add_space(12.0);
            let bar = (level * 6.0).min(1.0);
            let pb = egui::ProgressBar::new(bar)
                .desired_width(360.0)
                .fill(egui::Color32::from_rgb(80, 200, 120));
            ui.add(pb);
            ui.add_space(20.0);

            let can_stop = elapsed >= MIN_RECORD_SECS;
            let stop_label = if can_stop {
                "■  Stop recording".to_string()
            } else {
                format!(
                    "Speak for at least {:.0}s ({:.0} more)…",
                    MIN_RECORD_SECS,
                    MIN_RECORD_SECS - elapsed
                )
            };
            let stop_btn = egui::Button::new(egui::RichText::new(stop_label).size(15.0))
                .min_size(egui::vec2(260.0, 40.0));
            if ui.add_enabled(can_stop, stop_btn).clicked() {
                stop_now = true;
            }
        });

        if stop_now {
            let samples = recorder.finish();
            self.launch_transcription(samples.clone(), ctx.clone());
            return Stage::Transcribing {
                theme,
                prompt,
                samples,
            };
        }
        Stage::Recording {
            theme,
            prompt,
            recorder,
        }
    }

    fn render_followups_streaming(&self, ui: &mut egui::Ui, theme: &str, transcript: &str) {
        ui.label(egui::RichText::new(theme).weak());
        ui.add_space(8.0);
        ui.collapsing("Show what I heard you say", |ui| {
            ui.label(egui::RichText::new(transcript).italics());
        });
        ui.add_space(12.0);
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Drafting follow-up questions…");
        });
        ui.add_space(8.0);
        let live = self.stream.snapshot();
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.label(egui::RichText::new(live).size(15.0));
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn render_recording_followup(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        theme: String,
        prompt: String,
        primary_transcript: String,
        primary_samples: Vec<f32>,
        followups: Vec<String>,
        current: usize,
        followup_answers: Vec<String>,
        recorder: Recorder,
    ) -> Stage {
        ui.label(
            egui::RichText::new(format!("Follow-up {} of {}", current + 1, followups.len())).weak(),
        );
        ui.add_space(6.0);
        egui::Frame::group(ui.style())
            .fill(ui.style().visuals.faint_bg_color)
            .inner_margin(12.0)
            .show(ui, |ui| {
                ui.label(egui::RichText::new(&followups[current]).size(15.0));
            });

        let mut replay = false;
        if self.tts_enabled() {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let busy = self.player.is_busy();
                if ui
                    .add_enabled(!busy, egui::Button::new("🔊  Replay").small())
                    .clicked()
                {
                    replay = true;
                }
                ui.label(
                    egui::RichText::new("Reading aloud via Kokoro")
                        .weak()
                        .small(),
                );
            });
        }

        let elapsed = recorder.elapsed_secs();
        let level = recorder.vu_level();
        ui.add_space(20.0);

        // We compute "should we stop?" here, render the controls, and only at the end
        // decide which Stage to return — avoiding cloning the large state into the closure.
        let mut stop_now = elapsed >= FOLLOWUP_MAX_SECS;
        ui.vertical_centered(|ui| {
            ui.heading(format!(
                "● Recording  {:>5.1}s / {:.0}s",
                elapsed, FOLLOWUP_MAX_SECS
            ));
            ui.add_space(8.0);
            let bar = (level * 6.0).min(1.0);
            ui.add(
                egui::ProgressBar::new(bar)
                    .desired_width(320.0)
                    .fill(egui::Color32::from_rgb(80, 200, 120)),
            );
            ui.add_space(16.0);
            let stop_btn =
                egui::Button::new("■  Done with this answer").min_size(egui::vec2(220.0, 36.0));
            if ui.add(stop_btn).clicked() {
                stop_now = true;
            }
        });

        if replay {
            self.speak(followups[current].clone());
        }
        if stop_now {
            // Cut audio in case the question is still reading aloud.
            self.stop_speaking();
            let samples = recorder.finish();
            self.launch_transcription(samples.clone(), ctx.clone());
            return Stage::TranscribingFollowup {
                theme,
                prompt,
                primary_transcript,
                primary_samples,
                followups,
                current,
                followup_answers,
                samples,
            };
        }
        Stage::RecordingFollowup {
            theme,
            prompt,
            primary_transcript,
            primary_samples,
            followups,
            current,
            followup_answers,
            recorder,
        }
    }

    fn render_feedback_streaming(&self, ui: &mut egui::Ui, metrics: &DeliveryMetrics) {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Composing your feedback…");
        });
        ui.add_space(12.0);
        self.render_metrics_strip(ui, metrics);
        ui.add_space(12.0);
        ui.separator();
        let live = self.stream.snapshot();
        egui::ScrollArea::vertical()
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                ui.label(egui::RichText::new(live).size(14.0));
            });
    }

    fn render_metrics_strip(&self, ui: &mut egui::Ui, m: &DeliveryMetrics) {
        egui::Frame::group(ui.style())
            .fill(ui.style().visuals.faint_bg_color)
            .inner_margin(10.0)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    metric_chip(ui, "Duration", &format!("{:.1}s", m.duration_secs));
                    metric_chip(ui, "Words", &format!("{}", m.words));
                    metric_chip(ui, "Pace", &format!("{:.0} wpm", m.words_per_minute));
                    metric_chip(ui, "Long pauses", &format!("{}", m.long_pause_count));
                    metric_chip(ui, "Pause time", &format!("{:.1}s", m.total_pause_secs));
                    metric_chip(ui, "Energy CV", &format!("{:.2}", m.energy_cv));
                    metric_chip(ui, "Fillers", &format!("{}", m.filler_count));
                });
            });
    }

    #[allow(clippy::too_many_arguments)]
    fn render_final_feedback(
        &mut self,
        ui: &mut egui::Ui,
        theme: &str,
        prompt: &str,
        transcript: &str,
        followups: &[String],
        followup_answers: &[String],
        metrics: &DeliveryMetrics,
        feedback: &str,
        saved: bool,
    ) -> Stage {
        ui.label(egui::RichText::new(theme).weak());
        ui.add_space(2.0);
        ui.label(egui::RichText::new(prompt).italics());
        ui.add_space(10.0);
        self.render_metrics_strip(ui, metrics);
        ui.add_space(10.0);

        egui::ScrollArea::vertical()
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                ui.label(egui::RichText::new(feedback).size(14.0));
                ui.add_space(16.0);
                ui.collapsing("Transcript of your main answer", |ui| {
                    ui.label(egui::RichText::new(transcript).italics());
                });
                if !followups.is_empty() {
                    ui.collapsing("Follow-ups & your answers", |ui| {
                        for (i, q) in followups.iter().enumerate() {
                            ui.strong(format!("Q{}: {}", i + 1, q));
                            if let Some(a) = followup_answers.get(i) {
                                ui.label(egui::RichText::new(a).italics());
                            }
                            ui.add_space(6.0);
                        }
                    });
                }
                ui.add_space(20.0);
            });

        ui.separator();
        let mut restart = false;
        let mut copy = false;
        let mut go_history = false;
        let mut go_trends = false;
        ui.horizontal(|ui| {
            if ui.button("⟲  New exercise").clicked() {
                restart = true;
            }
            ui.add_space(8.0);
            if ui.button("Copy feedback").clicked() {
                copy = true;
            }
            ui.add_space(8.0);
            if ui.button("📜  Past exercises").clicked() {
                go_history = true;
            }
            ui.add_space(8.0);
            if ui.button("📈  Metrics over time").clicked() {
                go_trends = true;
            }
        });
        if copy {
            ui.ctx()
                .output_mut(|o| o.copied_text = feedback.to_string());
        }
        if restart {
            return Stage::Welcome;
        }
        if go_history {
            return Stage::History { selected: None };
        }
        if go_trends {
            return Stage::Trends;
        }
        Stage::Feedback {
            theme: theme.to_string(),
            prompt: prompt.to_string(),
            transcript: transcript.to_string(),
            followups: followups.to_vec(),
            followup_answers: followup_answers.to_vec(),
            metrics: metrics.clone(),
            feedback: feedback.to_string(),
            saved,
        }
    }

    fn render_history(&mut self, ui: &mut egui::Ui, selected: Option<usize>) -> Stage {
        let mut next: Stage = Stage::History { selected };
        let active_name = self.persona_store.active().name.clone();
        let filtered = self.active_persona_indices();

        ui.horizontal(|ui| {
            if ui.button("← Back").clicked() {
                next = Stage::Welcome;
            }
            ui.add_space(8.0);
            if ui.button("📈  Metrics over time").clicked() {
                next = Stage::Trends;
            }
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(format!(
                    "{} · {} session{}",
                    active_name,
                    filtered.len(),
                    if filtered.len() == 1 { "" } else { "s" }
                ))
                .weak(),
            );
        });
        ui.add_space(8.0);

        if filtered.is_empty() {
            ui.add_space(40.0);
            ui.vertical_centered(|ui| {
                ui.label(format!("No past exercises for {active_name} yet."));
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(
                        "Switch persona at the top, or run a session to start filling this in.",
                    )
                    .weak(),
                );
            });
            return next;
        }

        // Two-pane layout: list on the left, detail of the selected one on the right.
        // `selected` stores an absolute index into self.sessions so it stays
        // stable when new sessions are appended; we just check it's still in
        // the persona-filtered set when rendering.
        let mut new_selected = selected.filter(|i| filtered.contains(i));

        egui::SidePanel::left("history_list")
            .resizable(true)
            .default_width(280.0)
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    // Newest first.
                    for &i in filtered.iter().rev() {
                        let s = &self.sessions[i];
                        let is_sel = selected == Some(i);
                        let title = if s.theme.is_empty() {
                            "(untitled)".to_string()
                        } else {
                            s.theme.clone()
                        };
                        let label =
                            egui::RichText::new(format!("{}\n{}", title, s.formatted_timestamp()));
                        let resp = ui.selectable_label(is_sel, label);
                        if resp.clicked() {
                            new_selected = Some(i);
                        }
                        ui.add_space(2.0);
                    }
                });
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let idx = new_selected.or(filtered.last().copied());
            if let Some(i) = idx {
                if let Some(s) = self.sessions.get(i).cloned() {
                    ui.label(egui::RichText::new(&s.formatted_timestamp()).weak());
                    ui.heading(&s.theme);
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new(&s.prompt).italics());
                    ui.add_space(10.0);
                    self.render_metrics_strip(ui, &s.metrics);
                    ui.add_space(10.0);
                    egui::ScrollArea::vertical()
                        .auto_shrink([false; 2])
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new(&s.feedback).size(14.0));
                            ui.add_space(16.0);
                            ui.collapsing("Transcript of your main answer", |ui| {
                                ui.label(egui::RichText::new(&s.transcript).italics());
                            });
                            if !s.followups.is_empty() {
                                ui.collapsing("Follow-ups & your answers", |ui| {
                                    for (qi, q) in s.followups.iter().enumerate() {
                                        ui.strong(format!("Q{}: {}", qi + 1, q));
                                        if let Some(a) = s.followup_answers.get(qi) {
                                            ui.label(egui::RichText::new(a).italics());
                                        }
                                        ui.add_space(6.0);
                                    }
                                });
                            }
                            ui.add_space(20.0);
                        });
                }
            } else {
                ui.label("Pick a session on the left to see its feedback.");
            }
        });

        // Apply selection change after rendering.
        if let Stage::History { .. } = next {
            next = Stage::History {
                selected: new_selected,
            };
        }
        next
    }

    fn render_trends(&mut self, ui: &mut egui::Ui) -> Stage {
        let mut next: Stage = Stage::Trends;
        let active_name = self.persona_store.active().name.clone();
        let filtered: Vec<&Session> = self
            .active_persona_indices()
            .into_iter()
            .map(|i| &self.sessions[i])
            .collect();

        ui.horizontal(|ui| {
            if ui.button("← Back").clicked() {
                next = Stage::Welcome;
            }
            ui.add_space(8.0);
            if ui.button("📜  Past exercises").clicked() {
                next = Stage::History { selected: None };
            }
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new(format!(
                    "{} · {} session{} plotted",
                    active_name,
                    filtered.len(),
                    if filtered.len() == 1 { "" } else { "s" }
                ))
                .weak(),
            );
        });
        ui.add_space(8.0);

        if filtered.is_empty() {
            ui.add_space(40.0);
            ui.vertical_centered(|ui| {
                ui.label(format!("Nothing to chart for {active_name} yet."));
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("Complete an exercise to start tracking trends.").weak(),
                );
            });
            return next;
        }

        // X axis: per-persona session ordinal (1-indexed). Easier to read than
        // epoch seconds and it keeps spacing uniform whether you do one session
        // a day or ten.
        let series: Vec<(f64, &DeliveryMetrics)> = filtered
            .iter()
            .enumerate()
            .map(|(i, s)| ((i + 1) as f64, &s.metrics))
            .collect();

        // Latest run summary line, so the user can see "where am I now" without
        // scanning every chart.
        if let Some(last) = filtered.last() {
            ui.label(
                egui::RichText::new(format!(
                    "Latest ({}): {:.0} wpm · {} fillers · {} long pauses · CV {:.2}",
                    last.formatted_timestamp(),
                    last.metrics.words_per_minute,
                    last.metrics.filler_count,
                    last.metrics.long_pause_count,
                    last.metrics.energy_cv,
                ))
                .weak(),
            );
            ui.add_space(8.0);
        }

        egui::ScrollArea::vertical()
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                // Six small charts in a 2-column grid (one column on narrow windows).
                // Each chart has its own y-scale so an outlier in one metric doesn't
                // squash the others.
                let charts: [(&str, fn(&DeliveryMetrics) -> f64); 6] = [
                    ("Pace (wpm)", |m| m.words_per_minute as f64),
                    ("Fillers", |m| m.filler_count as f64),
                    ("Long pauses", |m| m.long_pause_count as f64),
                    ("Pause time (s)", |m| m.total_pause_secs as f64),
                    ("Energy CV", |m| m.energy_cv as f64),
                    ("Words spoken", |m| m.words as f64),
                ];

                let cols: usize = if ui.available_width() > 720.0 { 2 } else { 1 };
                let plot_w = ui.available_width() / cols as f32 - 16.0;
                let plot_h = 180.0;

                for chunk in charts.chunks(cols) {
                    ui.horizontal(|ui| {
                        for (title, accessor) in chunk {
                            ui.vertical(|ui| {
                                ui.label(egui::RichText::new(*title).strong());
                                trend_plot(ui, title, plot_w, plot_h, &series, *accessor);
                            });
                        }
                    });
                    ui.add_space(6.0);
                }

                ui.add_space(12.0);
                ui.collapsing("Where is this stored?", |ui| {
                    ui.label(
                        egui::RichText::new(history::history_path_display())
                            .monospace()
                            .weak(),
                    );
                    ui.label(
                        egui::RichText::new(
                            "JSON-lines, one row per session. Safe to delete to reset trends.",
                        )
                        .weak(),
                    );
                });
                ui.add_space(8.0);
            });

        next
    }

    fn render_personas_list(&mut self, ui: &mut egui::Ui) -> Stage {
        let mut next = Stage::Personas;

        ui.horizontal(|ui| {
            if ui.button("← Back").clicked() {
                next = Stage::Welcome;
            }
            ui.add_space(8.0);
            ui.heading("Personas");
        });
        ui.add_space(4.0);
        ui.label(
            egui::RichText::new(
                "Personas configure the coaching style, the topic list, and any background \
             info the coach should know about you.",
            )
            .weak(),
        );
        ui.add_space(10.0);

        let n = self.persona_store.personas.len();
        let active = self.persona_store.active;
        let mut to_set_active: Option<usize> = None;
        let mut to_edit: Option<usize> = None;
        let mut to_delete: Option<usize> = None;
        let mut to_clear: Option<String> = None;
        let mut to_create_new = false;

        let legacy_name = personas::default_engineering_leader().name;

        egui::ScrollArea::vertical()
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                for i in 0..n {
                    let p = &self.persona_store.personas[i];
                    let is_active = i == active;
                    let name = p.name.clone();
                    let topic_count = p.topics.len();
                    let has_bg = !p.background.trim().is_empty();

                    let exercise_count = self
                        .sessions
                        .iter()
                        .filter(|s| {
                            let s_name = if s.persona_name.is_empty() {
                                &legacy_name
                            } else {
                                &s.persona_name
                            };
                            s_name == &name
                        })
                        .count();

                    egui::Frame::group(ui.style())
                        .fill(if is_active {
                            ui.style().visuals.faint_bg_color
                        } else {
                            ui.style().visuals.window_fill
                        })
                        .inner_margin(10.0)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                let marker = if is_active { "●  " } else { "○  " };
                                ui.label(
                                    egui::RichText::new(format!("{marker}{}", name))
                                        .strong()
                                        .size(15.0),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if n > 1 {
                                            if ui.small_button("Delete").clicked() {
                                                to_delete = Some(i);
                                            }
                                            ui.add_space(4.0);
                                        }
                                        if !is_active {
                                            if ui.small_button("Set active").clicked() {
                                                to_set_active = Some(i);
                                            }
                                            ui.add_space(4.0);
                                        }
                                        if ui.small_button("Edit").clicked() {
                                            to_edit = Some(i);
                                        }
                                    },
                                );
                            });
                            ui.add_space(2.0);
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{} topic{} · {} · {} exercise{}",
                                        topic_count,
                                        if topic_count == 1 { "" } else { "s" },
                                        if has_bg {
                                            "background provided"
                                        } else {
                                            "no background"
                                        },
                                        exercise_count,
                                        if exercise_count == 1 { "" } else { "s" },
                                    ))
                                    .weak(),
                                );

                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if exercise_count > 0 {
                                            if self.confirm_clear_idx == Some(i) {
                                                if ui
                                                    .small_button(
                                                        egui::RichText::new("Confirm Clear?")
                                                            .color(egui::Color32::LIGHT_RED),
                                                    )
                                                    .clicked()
                                                {
                                                    to_clear = Some(name.clone());
                                                    self.confirm_clear_idx = None;
                                                }
                                                if ui.small_button("Cancel").clicked() {
                                                    self.confirm_clear_idx = None;
                                                }
                                            } else {
                                                if ui.small_button("Clear history").clicked() {
                                                    self.confirm_clear_idx = Some(i);
                                                }
                                            }
                                        }
                                    },
                                );
                            });
                        });
                    ui.add_space(6.0);
                }

                ui.add_space(4.0);
                if ui.button("+ New persona").clicked() {
                    to_create_new = true;
                }
            });

        if let Some(name) = to_clear {
            crate::history::delete_for_persona(&name);
            self.reload_sessions();
        }

        if let Some(i) = to_set_active {
            self.persona_store.active = i;
            personas::save(&self.persona_store);
        }
        if let Some(i) = to_edit {
            let draft = PersonaDraft::from_persona(Some(i), &self.persona_store.personas[i]);
            return Stage::EditPersona { draft };
        }
        if let Some(i) = to_delete {
            self.persona_store.personas.remove(i);
            // Keep `active` valid: shift it down if it was past the removed
            // index, clamp it if the removed entry was the active one and was
            // also the last in the list.
            if self.persona_store.active > i
                || self.persona_store.active >= self.persona_store.personas.len()
            {
                self.persona_store.active = self
                    .persona_store
                    .active
                    .saturating_sub(1)
                    .min(self.persona_store.personas.len().saturating_sub(1));
            }
            personas::save(&self.persona_store);
        }
        if to_create_new {
            return Stage::EditPersona {
                draft: PersonaDraft::new(),
            };
        }

        next
    }

    fn render_edit_persona(&mut self, ui: &mut egui::Ui, mut draft: PersonaDraft) -> Stage {
        let mut save = false;
        let mut cancel = false;
        let mut preview_voice = false;

        ui.horizontal(|ui| {
            if ui.button("← Cancel").clicked() {
                cancel = true;
            }
            ui.add_space(8.0);
            ui.heading(if draft.edit_idx.is_some() {
                "Edit persona"
            } else {
                "New persona"
            });
        });
        ui.add_space(8.0);

        egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            ui.label(egui::RichText::new("Name").strong());
            ui.add(egui::TextEdit::singleline(&mut draft.name)
                .desired_width(f32::INFINITY)
                .hint_text("e.g. Engineering Leader"));
            ui.add_space(10.0);

            ui.label(egui::RichText::new("Coaching style").strong());
            ui.label(egui::RichText::new(
                "Slotted into the system prompt after \"You are \". Describe the coach's \
                 role and the intended audience.",
            ).weak().small());
            ui.add(egui::TextEdit::multiline(&mut draft.description)
                .desired_rows(4)
                .desired_width(f32::INFINITY)
                .hint_text("an executive communications coach. Your client is a senior engineering leader…"));
            ui.add_space(10.0);

            ui.label(egui::RichText::new("Topics").strong());
            ui.label(egui::RichText::new(
                "One per line. The app picks one at random to prompt today's exercise.",
            ).weak().small());
            ui.add(egui::TextEdit::multiline(&mut draft.topics_text)
                .desired_rows(10)
                .desired_width(f32::INFINITY)
                .hint_text("Engineering leadership\nPeople management & 1:1s\nTechnical strategy"));
            ui.add_space(10.0);

            ui.label(egui::RichText::new("Background (optional)").strong());
            ui.label(egui::RichText::new(
                "Anything you want the coach to know about you — role, hobbies, the kind \
                 of audience you actually face. Used as extra context for feedback.",
            ).weak().small());
            ui.add(egui::TextEdit::multiline(&mut draft.background)
                .desired_rows(5)
                .desired_width(f32::INFINITY)
                .hint_text("I'm a Head of Engineering at a 50-person fintech. I present to the board monthly…"));
            ui.add_space(10.0);

            ui.label(egui::RichText::new("Voice (Kokoro TTS)").strong());
            ui.label(egui::RichText::new(
                "Voice used when the coach reads prompts aloud. Only applies when \
                 sherpa-onnx is configured."
            ).weak().small());
            let voice_label = KOKORO_VOICES
                .get(draft.voice_id as usize)
                .copied()
                .unwrap_or("unknown");
            egui::ComboBox::from_id_salt("voice_picker")
                .selected_text(format!("{} — {}", draft.voice_id, voice_label))
                .show_ui(ui, |ui| {
                    for (id, name) in KOKORO_VOICES.iter().enumerate() {
                        let id = id as u32;
                        ui.selectable_value(&mut draft.voice_id, id,
                            format!("{id} — {name}"));
                    }
                });
            if self.tts_enabled() {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    let busy = self.player.is_busy();
                    if ui.add_enabled(!busy, egui::Button::new("▶  Preview voice").small()).clicked() {
                        preview_voice = true;
                    }
                    if busy {
                        ui.label(egui::RichText::new("Synthesising…").weak().small());
                    }
                });
            }
            ui.add_space(16.0);

            ui.horizontal(|ui| {
                let can_save = !draft.name.trim().is_empty();
                if ui.add_enabled(can_save,
                    egui::Button::new(egui::RichText::new("Save").size(14.0))
                        .min_size(egui::vec2(120.0, 32.0))
                ).clicked() { save = true; }
                ui.add_space(8.0);
                if ui.button("Cancel").clicked() { cancel = true; }

                if !can_save {
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new("Name is required").weak());
                }
            });
            ui.add_space(8.0);
        });

        if preview_voice {
            self.speak_with_voice_id(
                "Hello! This is a preview of the selected voice.".to_string(),
                draft.voice_id,
            );
        }
        if save {
            let edit_idx = draft.edit_idx;
            let persona = draft.into_persona();
            match edit_idx {
                Some(i) => {
                    self.persona_store.personas[i] = persona;
                }
                None => {
                    self.persona_store.personas.push(persona);
                    // A freshly-created persona becomes active so the user
                    // immediately sees the topics they configured take effect.
                    self.persona_store.active = self.persona_store.personas.len() - 1;
                }
            }
            personas::save(&self.persona_store);
            return Stage::Personas;
        }
        if cancel {
            return Stage::Personas;
        }
        Stage::EditPersona { draft }
    }

    fn render_debug(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        mut test_recorder: Option<Recorder>,
        test_result: Option<Result<crate::stt::TranscriptionResult, String>>,
    ) -> Stage {
        let mut go_back = false;
        ui.horizontal(|ui| {
            if ui.button("← Back").clicked() {
                go_back = true;
            }
            ui.heading("🛠 Debug / Configuration");
        });
        if go_back {
            if let Some(r) = test_recorder.take() {
                r.finish();
            }
            return Stage::Welcome;
        }
        ui.add_space(4.0);
        ui.label(egui::RichText::new(format!("Ollama model: {}", self.model_name)).weak());
        ui.add_space(10.0);

        egui::ScrollArea::vertical().show(ui, |ui| {
            // --- Audio Info ---
            ui.collapsing("🎤 Audio Hardware", |ui| {
                ui.label(format!(
                    "Default Device: {}",
                    crate::audio::default_device_name()
                ));
                ui.add_space(4.0);
                ui.label("Available Input Devices:");
                for name in crate::audio::list_devices() {
                    ui.label(format!(" • {name}"));
                }
            });
            ui.add_space(8.0);

            // --- Mic Test ---
            ui.collapsing("🎙 Mic Test / Transcription Test", |ui| {
                if let Some(recorder) = &test_recorder {
                    let elapsed = recorder.elapsed_secs();
                    let level = recorder.vu_level();
                    let device = recorder.device_name();
                    ui.horizontal(|ui| {
                        ui.label(format!("Recording from {device}... {:.1}s", elapsed));
                        let bar = (level * 6.0).min(1.0);
                        ui.add(
                            egui::ProgressBar::new(bar)
                                .desired_width(200.0)
                                .fill(egui::Color32::from_rgb(80, 200, 120)),
                        );
                    });
                    if ui.button("Stop & Transcribe (Test)").clicked() {
                        let samples = test_recorder.take().unwrap().finish();
                        self.launch_transcription(samples, ctx.clone());
                    }
                } else {
                    if ui.button("Start Mic Test (5s or manual stop)").clicked() {
                        match Recorder::start() {
                            Ok(r) => {
                                test_recorder = Some(r);
                            }
                            Err(e) => {
                                ui.colored_label(egui::Color32::RED, format!("Mic error: {e:#}"));
                            }
                        }
                    }
                }

                if let Some(res) = &test_result {
                    ui.add_space(8.0);
                    ui.label("Test Result:");
                    match res {
                        Ok(res) => {
                            egui::Frame::group(ui.style())
                                .fill(ui.style().visuals.faint_bg_color)
                                .show(ui, |ui| {
                                    if res.text.is_empty() {
                                        ui.label(
                                            egui::RichText::new("(Empty transcript text)")
                                                .italics()
                                                .weak(),
                                        );
                                    } else {
                                        ui.label(format!("Cleaned: {}", res.text));
                                    }
                                });
                            ui.collapsing("Raw Output Details", |ui| {
                                ui.label("STDOUT:");
                                ui.label(egui::RichText::new(&res.stdout).monospace().small());
                                ui.separator();
                                ui.label("STDERR:");
                                ui.label(egui::RichText::new(&res.stderr).monospace().small());
                            });
                        }
                        Err(e) => {
                            ui.colored_label(
                                egui::Color32::RED,
                                format!("Transcription failed: {e}"),
                            );
                        }
                    }
                }
            });
            ui.add_space(8.0);

            // --- STT Info ---
            ui.collapsing("🗣 Speech-to-Text (Whisper)", |ui| {
                if let Some(t) = &self.transcriber {
                    ui.label(egui::RichText::new(t.bin_info()).monospace().small());
                } else {
                    ui.colored_label(egui::Color32::RED, "Transcriber not loaded.");
                    if let Some(err) = &self.transcriber_error {
                        ui.label(err);
                    }
                }
            });
            ui.add_space(8.0);

            // --- TTS Info ---
            ui.collapsing("🔊 Text-to-Speech (Kokoro)", |ui| {
                if let Some(tts) = &self.tts {
                    ui.label(format!("Binary: {}", tts.bin_path().display()));
                    ui.label(format!("Model Dir: {}", tts.model_dir().display()));
                } else {
                    ui.label("TTS not loaded.");
                    if let Some(err) = &self.tts_error {
                        ui.label(err);
                    }
                }
            });
        });

        // Auto-stop test recorder after 10s to be safe
        if let Some(r) = &test_recorder {
            if r.elapsed_secs() > 10.0 {
                let samples = test_recorder.take().unwrap().finish();
                self.launch_transcription(samples, ctx.clone());
            } else {
                ctx.request_repaint();
            }
        }

        Stage::Debug {
            test_recorder,
            test_result,
        }
    }

    fn render_error(&mut self, ui: &mut egui::Ui, msg: String) -> Stage {
        ui.add_space(20.0);
        ui.colored_label(
            egui::Color32::from_rgb(220, 120, 120),
            "Something went wrong",
        );
        ui.add_space(8.0);
        egui::ScrollArea::vertical()
            .max_height(280.0)
            .show(ui, |ui| {
                ui.label(egui::RichText::new(&msg).monospace());
            });
        ui.add_space(16.0);
        if ui.button("Back to start").clicked() {
            return Stage::Welcome;
        }
        Stage::Error(msg)
    }

    fn spinner_with_label(&self, ui: &mut egui::Ui, label: &str) {
        ui.add_space(40.0);
        ui.vertical_centered(|ui| {
            ui.spinner();
            ui.add_space(8.0);
            ui.label(label);
        });
    }
}

fn metric_chip(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.group(|ui| {
        ui.vertical(|ui| {
            ui.label(egui::RichText::new(label).small().weak());
            ui.label(egui::RichText::new(value).strong());
        });
    });
}

fn trend_plot(
    ui: &mut egui::Ui,
    id_source: &str,
    width: f32,
    height: f32,
    series: &[(f64, &DeliveryMetrics)],
    accessor: fn(&DeliveryMetrics) -> f64,
) {
    use egui_plot::{Line, Plot, PlotPoints, Points};

    let pts: Vec<[f64; 2]> = series.iter().map(|(x, m)| [*x, accessor(m)]).collect();

    Plot::new(id_source)
        .width(width.max(120.0))
        .height(height)
        .show_axes([false, true])
        .show_grid([false, true])
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .show(ui, |plot_ui| {
            // Line + dot markers — a single point still shows up that way.
            let line = Line::new(PlotPoints::from(pts.clone()))
                .color(egui::Color32::from_rgb(80, 160, 220))
                .width(2.0);
            let points = Points::new(PlotPoints::from(pts))
                .color(egui::Color32::from_rgb(80, 160, 220))
                .radius(3.0);
            plot_ui.line(line);
            plot_ui.points(points);
        });
}
