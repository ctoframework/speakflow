//! Local speech-to-text by shelling out to a prebuilt `whisper-cli` binary
//! from the whisper.cpp project. Avoids any FFI / bindgen / C++ build pipeline.
//!
//! Setup (Windows):
//!   1. Download the latest whisper-bin-x64.zip from
//!      https://github.com/ggerganov/whisper.cpp/releases
//!   2. Extract it. Newer releases name the binary `whisper-cli.exe`;
//!      older ones call it `main.exe` — we accept either.
//!   3. Put the folder anywhere, e.g. C:\tools\whisper\
//!   4. Either add that folder to PATH, or set:
//!         WHISPER_BIN=C:\tools\whisper\whisper-cli.exe
//!   5. Download a model (e.g. ggml-base.en.bin) into:
//!         %APPDATA%\speakflow\models\ggml-base.en.bin
//!      or set WHISPER_MODEL to a custom path.
//!
//! macOS / Linux:
//!   `brew install whisper-cpp`  (provides `whisper-cli`)
//!   or build from source: https://github.com/ggerganov/whisper.cpp

use anyhow::{anyhow, Context, Result};
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhisperDialect {
    /// whisper.cpp (ggerganov)
    Cpp,
    /// OpenAI's Python whisper
    OpenAi,
}

pub struct Transcriber {
    bin: PathBuf,
    model: PathBuf,
    dialect: WhisperDialect,
    info: String,
}

pub struct TranscriptionResult {
    pub text: String,
    pub stdout: String,
    pub stderr: String,
}

impl Transcriber {
    pub fn load() -> Result<Self> {
        let bin = resolve_binary()?;
        let model = resolve_model()?;
        
        let help = Command::new(&bin).arg("--help").output()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string() + &String::from_utf8_lossy(&o.stderr))
            .unwrap_or_else(|e| format!("Error running --help: {e}"));

        let dialect = if help.contains("ggerganov") || help.contains("whisper.cpp") {
            WhisperDialect::Cpp
        } else {
            WhisperDialect::OpenAi
        };

        if dialect == WhisperDialect::Cpp && !model.exists() {
            return Err(anyhow!(
                "Whisper model not found at {}. Download e.g. ggml-base.en.bin from \
                 https://huggingface.co/ggerganov/whisper.cpp and place it there, \
                 or set WHISPER_MODEL.",
                model.display()
            ));
        }

        log::info!("Using whisper binary: {} ({:?})", bin.display(), dialect);
        log::info!("Using whisper model:  {}", model.display());

        let info = format!(
            "Binary: {}\nDialect: {:?}\nModel: {}\n\n--help output:\n{}",
            bin.display(), dialect, model.display(), help
        );

        Ok(Self { bin, model, dialect, info })
    }

    pub fn bin_info(&self) -> &str {
        &self.info
    }

    /// Transcribe 16 kHz mono f32 samples. Writes them to a temp WAV, invokes
    /// whisper CLI, parses plain-text output. Synchronous; call from a blocking
    /// task off the UI thread.
    pub fn transcribe(&self, samples_16k_mono: &[f32]) -> Result<TranscriptionResult> {
        log::info!("Transcribing {} samples ({:.2}s)", 
            samples_16k_mono.len(), 
            samples_16k_mono.len() as f32 / crate::audio::TARGET_SAMPLE_RATE as f32);

        // 1) Write input WAV to a temp path.
        let tmp_dir = std::env::temp_dir();
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let wav_path = tmp_dir.join(format!("speakflow-{stamp}.wav"));
        crate::audio::write_wav_16k(&wav_path, samples_16k_mono)
            .context("writing temp WAV for whisper")?;

        // 2) Invoke whisper CLI.
        let threads = std::thread::available_parallelism()
            .map(|n| n.get()).unwrap_or(2).min(8).to_string();

        let mut cmd = Command::new(&self.bin);

        match self.dialect {
            WhisperDialect::Cpp => {
                cmd.arg("--model").arg(&self.model);
                cmd.arg("--language").arg("en");
                cmd.arg("--threads").arg(&threads);
                cmd.arg("-nt");
                cmd.arg("--no-prints");
            }
            WhisperDialect::OpenAi => {
                // OpenAI whisper expects a model name (e.g. 'base') or a .pt file.
                // If we have a .bin file (ggml), it won't work directly.
                // We'll try to pass the filename without extension if it looks like a standard name.
                let model_str = self.model.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.strip_prefix("ggml-").unwrap_or(s))
                    .unwrap_or("base");
                
                cmd.arg("--model").arg(model_str);
                cmd.arg("--language").arg("en");
                cmd.arg("--threads").arg(&threads);
                cmd.arg("--device").arg("cpu"); // Avoid CUDA OOM errors
                cmd.arg("--verbose").arg("True");
                cmd.arg("--output_format").arg("txt");
            }
        }

        // Positional argument for the audio file works for both
        cmd.arg(&wav_path);

        // On Windows, hide the console window that would otherwise flash up.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let output = cmd.output()
            .with_context(|| format!("failed to spawn `{}`", self.bin.display()))?;

        // Best-effort cleanup; ignore errors.
        let _ = std::fs::remove_file(&wav_path);

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !output.status.success() {
            return Err(anyhow!(
                "whisper-cli exited with {}:\nSTDOUT: {}\nSTDERR: {}",
                output.status, stdout.trim(), stderr.trim()
            ));
        }

        let cleaned = clean_transcript(&stdout);
        if cleaned.is_empty() {
            log::warn!("Whisper returned empty transcript!");
            log::debug!("Raw STDOUT: {}", stdout);
            log::debug!("Raw STDERR: {}", stderr);
        } else {
            log::debug!("Whisper cleaned text: {}", cleaned);
        }

        Ok(TranscriptionResult {
            text: cleaned,
            stdout,
            stderr,
        })
    }
}

/// whisper-cli with `-nt` outputs lines like `   Hello world.` plus possibly
/// banner / info lines. OpenAI's version may include timestamps like
/// `[00:00.000 --> 00:05.000]  Hello`. We trim, drop logs, and strip timestamps.
fn clean_transcript(raw: &str) -> String {
    let mut parts = Vec::new();
    for line in raw.lines() {
        let mut line = line.trim();
        if line.is_empty() { continue; }
        if line.starts_with("whisper_") || line.starts_with("system_info") || line.starts_with("main:") {
            continue;
        }
        // Strip OpenAI-style timestamps: [00:00.000 --> 00:05.000]
        if line.starts_with('[') && line.contains(" --> ") {
            if let Some(pos) = line.find(']') {
                line = line[pos + 1..].trim();
            }
        }
        if line.is_empty() { continue; }
        parts.push(line);
    }
    parts.join(" ")
}

fn resolve_binary() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("WHISPER_BIN") {
        let p = PathBuf::from(p);
        if p.exists() { return Ok(p); }
        return Err(anyhow!("WHISPER_BIN points to {} which doesn't exist", p.display()));
    }
    let candidates = if cfg!(windows) {
        &["whisper-cli.exe", "whisper.exe", "main.exe"][..]
    } else {
        &["whisper-cli", "whisper", "main"][..]
    };
    for name in candidates {
        if let Some(p) = which_on_path(name) {
            return Ok(p);
        }
    }
    Err(anyhow!(
        "whisper-cli not found. Either add it to PATH or set WHISPER_BIN. \
         Download from https://github.com/ggerganov/whisper.cpp/releases \
         (Windows: whisper-bin-x64.zip)."
    ))
}

/// Tiny PATH search to avoid pulling in the `which` crate.
fn which_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() { return Some(candidate); }
    }
    None
}

fn resolve_model() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("WHISPER_MODEL") {
        return Ok(PathBuf::from(p));
    }
    let mut p = dirs::config_dir().context("no config dir")?;
    p.push("speakflow/models/ggml-base.en.bin");
    Ok(p)
}
