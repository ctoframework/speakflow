//! Local text-to-speech by shelling out to `sherpa-onnx-offline-tts` from the
//! k2-fsa Sherpa-ONNX project, running the Kokoro model. Mirrors the
//! architecture of `stt.rs` (subprocess + temp WAV) so we don't pull a heavy
//! ML dependency into our build.
//!
//! Setup (Windows):
//!   1. Download the latest `sherpa-onnx-vX.Y.Z-win-x64.tar.bz2` from
//!      https://github.com/k2-fsa/sherpa-onnx/releases. Extract it to e.g.
//!      `C:\tools\sherpa-onnx\`. The folder contains `sherpa-onnx-offline-tts.exe`.
//!   2. Either add that folder to PATH, or set:
//!         SHERPA_BIN=C:\tools\sherpa-onnx\sherpa-onnx-offline-tts.exe
//!   3. Download a Kokoro model bundle (English-only, ~325 MB):
//!         https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
//!      Extract it. Resulting layout:
//!         kokoro-en-v0_19/
//!           model.onnx
//!           voices.bin
//!           tokens.txt
//!           espeak-ng-data/
//!      Either drop the folder into `%APPDATA%\comms-coach\models\kokoro-en-v0_19`
//!      or set KOKORO_MODEL_DIR to its full path.
//!   4. Optional: pick a different voice with KOKORO_VOICE_ID (0..10,
//!      default 0). For kokoro-en-v0_19, the IDs map to af, af_bella,
//!      af_nicole, af_sarah, af_sky, am_adam, am_michael, bf_emma,
//!      bf_isabella, bm_george, bm_lewis — see the sherpa-onnx docs.
//!
//! Playback runs on a dedicated thread that owns rodio's OutputStream
//! (which is `!Send`). The UI submits play/stop commands through an mpsc
//! channel; the player thread replaces any currently-playing clip when a new
//! one arrives. Synthesis itself runs on the tokio blocking pool.

use anyhow::{anyhow, Context, Result};
use std::io::Cursor;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

pub struct Tts {
    bin: PathBuf,
    model_dir: PathBuf,
    threads: u32,
}

impl Tts {
    pub fn bin_path(&self) -> &std::path::Path { &self.bin }
    pub fn model_dir(&self) -> &std::path::Path { &self.model_dir }

    pub fn load() -> Result<Self> {
        let bin = resolve_binary()?;
        let model_dir = resolve_model_dir()?;
        // Sanity-check the files sherpa-onnx will demand. Failing here gives
        // a clear "missing X" error instead of a cryptic subprocess crash later.
        for f in ["model.onnx", "voices.bin", "tokens.txt"] {
            let p = model_dir.join(f);
            if !p.exists() {
                return Err(anyhow!(
                    "Kokoro model file missing: {}\nDownload kokoro-en-v0_19 from \
                     https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models",
                    p.display()
                ));
            }
        }
        if !model_dir.join("espeak-ng-data").is_dir() {
            return Err(anyhow!(
                "espeak-ng-data directory missing under {}",
                model_dir.display()
            ));
        }

        let threads: u32 = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(2)
            .clamp(1, 4);

        log::info!("Using sherpa-onnx binary:   {}", bin.display());
        log::info!("Using Kokoro model dir:     {}", model_dir.display());
        Ok(Self { bin, model_dir, threads })
    }

    /// Synthesize `text` into a temp WAV file using the given Kokoro voice ID.
    /// Returns the path; the caller is expected to read + delete it.
    /// Synchronous — call from a blocking task off the UI thread.
    pub fn synthesize_to_temp_wav(&self, text: &str, voice_id: u32) -> Result<PathBuf> {
        let tmp_dir = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let out_path = tmp_dir.join(format!("comms-coach-tts-{stamp}.wav"));

        // sherpa-onnx-offline-tts wants the model paths as `--key=value` and
        // the text as a positional argument.
        let model = self.model_dir.join("model.onnx");
        let voices = self.model_dir.join("voices.bin");
        let tokens = self.model_dir.join("tokens.txt");
        let data_dir = self.model_dir.join("espeak-ng-data");

        let mut cmd = Command::new(&self.bin);
        cmd.arg(format!("--kokoro-model={}", model.display()))
            .arg(format!("--kokoro-voices={}", voices.display()))
            .arg(format!("--kokoro-tokens={}", tokens.display()))
            .arg(format!("--kokoro-data-dir={}", data_dir.display()))
            .arg(format!("--num-threads={}", self.threads))
            .arg(format!("--sid={voice_id}"))
            .arg(format!("--output-filename={}", out_path.display()))
            .arg(text);

        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let output = cmd.output()
            .with_context(|| format!("failed to spawn `{}`", self.bin.display()))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!(
                "sherpa-onnx-offline-tts exited with {}: {}",
                output.status, stderr.trim()
            ));
        }
        if !out_path.exists() {
            return Err(anyhow!(
                "sherpa-onnx-offline-tts did not produce {}",
                out_path.display()
            ));
        }
        Ok(out_path)
    }
}

fn resolve_binary() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("SHERPA_BIN") {
        let p = PathBuf::from(p);
        if p.exists() { return Ok(p); }
        return Err(anyhow!("SHERPA_BIN points to {} which doesn't exist", p.display()));
    }
    let candidates = if cfg!(windows) {
        &["sherpa-onnx-offline-tts.exe"][..]
    } else {
        &["sherpa-onnx-offline-tts"][..]
    };
    for name in candidates {
        if let Some(p) = which_on_path(name) {
            return Ok(p);
        }
    }
    Err(anyhow!(
        "sherpa-onnx-offline-tts not found. Either add it to PATH or set SHERPA_BIN. \
         Download a prebuilt from https://github.com/k2-fsa/sherpa-onnx/releases."
    ))
}

fn which_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() { return Some(candidate); }
    }
    None
}

fn resolve_model_dir() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("KOKORO_MODEL_DIR") {
        return Ok(PathBuf::from(p));
    }
    // Default: drop the extracted Kokoro folder next to the whisper models
    // under the platform config dir, so the user has one place for everything.
    let mut p = dirs::config_dir().context("no config dir")?;
    p.push("comms-coach/models/kokoro-en-v0_19");
    Ok(p)
}

// ---------------------------------------------------------------- playback

enum PlayCmd {
    Play(Vec<u8>),
    Stop,
}

/// Cheap, cloneable handle to the background player thread.
#[derive(Clone)]
pub struct Player {
    tx: mpsc::Sender<PlayCmd>,
    /// True while synthesis is in-flight or audio is playing.
    busy: Arc<AtomicBool>,
}

impl Player {
    pub fn spawn() -> Self {
        let (tx, rx) = mpsc::channel();
        let busy = Arc::new(AtomicBool::new(false));
        let busy_clone = busy.clone();
        thread::Builder::new()
            .name("comms-coach-tts-player".into())
            .spawn(move || run_player(rx, busy_clone))
            .expect("spawn audio player thread");
        Self { tx, busy }
    }

    /// Queue a WAV byte buffer for playback, replacing anything currently playing.
    pub fn play_wav_bytes(&self, bytes: Vec<u8>) {
        let _ = self.tx.send(PlayCmd::Play(bytes));
    }

    /// Stop the currently playing clip (if any) and clear the busy flag.
    pub fn stop(&self) {
        self.busy.store(false, Ordering::Relaxed);
        let _ = self.tx.send(PlayCmd::Stop);
    }

    /// Mark as busy before synthesis begins, so the UI disables replay immediately.
    pub fn mark_busy(&self) {
        self.busy.store(true, Ordering::Relaxed);
    }

    /// True while synthesis is in-flight or a clip is playing.
    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::Relaxed)
    }
}

fn run_player(rx: mpsc::Receiver<PlayCmd>, busy: Arc<AtomicBool>) {
    let (_stream, handle) = match rodio::OutputStream::try_default() {
        Ok(s) => s,
        Err(e) => {
            log::warn!("rodio output stream unavailable, TTS audio disabled: {e}");
            // Drain so callers don't pile up on a dead channel.
            while rx.recv().is_ok() {}
            return;
        }
    };

    // We recreate the Sink for each clip — once `stop()` is called on a Sink
    // it stops accepting new sources, so reuse isn't possible.
    let mut sink: Option<rodio::Sink> = None;

    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(PlayCmd::Play(bytes)) => {
                if let Some(s) = sink.take() { s.stop(); }
                let new_sink = match rodio::Sink::try_new(&handle) {
                    Ok(s) => s,
                    Err(e) => { log::warn!("rodio sink error: {e}"); busy.store(false, Ordering::Relaxed); continue; }
                };
                let decoder = match rodio::Decoder::new(Cursor::new(bytes)) {
                    Ok(d) => d,
                    Err(e) => { log::warn!("rodio decode error: {e}"); busy.store(false, Ordering::Relaxed); continue; }
                };
                new_sink.append(decoder);
                sink = Some(new_sink);
            }
            Ok(PlayCmd::Stop) => {
                if let Some(s) = sink.take() { s.stop(); }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Clear busy when a clip finishes playing naturally.
                if let Some(s) = &sink {
                    if s.empty() {
                        busy.store(false, Ordering::Relaxed);
                        sink = None;
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
}
