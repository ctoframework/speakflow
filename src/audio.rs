//! Microphone capture for the coaching exercise.
//!
//! Whisper expects 16 kHz mono f32 PCM. The system mic typically delivers
//! 44.1/48 kHz interleaved stereo, so we downmix and linearly resample on the
//! capture thread. Linear resampling is good enough for speech recognition and
//! keeps the dependency footprint small (no rubato/dasp).

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Instant;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Shared recording state — written by the audio callback, read by UI.
#[derive(Default)]
struct Shared {
    samples_16k_mono: Vec<f32>, // accumulated mono 16k f32
    /// Rolling RMS of the last callback, for a live VU meter (0.0..~1.0).
    last_rms: f32,
    /// Source-rate accumulator for the linear resampler (fractional position).
    resample_pos: f64,
}

pub struct Recorder {
    _stream: Stream,            // dropped on stop
    shared: Arc<Mutex<Shared>>,
    started_at: Instant,
    device_name: String,
    source_sample_rate: u32,
    source_channels: u16,
}

impl Recorder {
    pub fn start() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("no default input device — check OS mic permissions")?;
        let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());
        let config = device
            .default_input_config()
            .context("could not query input config")?;

        let source_sample_rate = config.sample_rate().0;
        let source_channels = config.channels();
        let sample_format = config.sample_format();
        let stream_config: StreamConfig = config.clone().into();

        let shared = Arc::new(Mutex::new(Shared::default()));
        let shared_cb = shared.clone();

        let err_cb = |e| log::error!("audio stream error: {e}");

        // We need one branch per sample format because cpal types the buffer.
        // The actual processing logic is identical and lives in `process_block`.
        let stream = match sample_format {
            SampleFormat::F32 => device.build_input_stream(
                &stream_config,
                move |data: &[f32], _| process_block(
                    data, source_channels, source_sample_rate, &shared_cb,
                    |x| x,
                ),
                err_cb, None,
            )?,
            SampleFormat::I16 => device.build_input_stream(
                &stream_config,
                move |data: &[i16], _| process_block(
                    data, source_channels, source_sample_rate, &shared_cb,
                    |x| x as f32 / i16::MAX as f32,
                ),
                err_cb, None,
            )?,
            SampleFormat::U16 => device.build_input_stream(
                &stream_config,
                move |data: &[u16], _| process_block(
                    data, source_channels, source_sample_rate, &shared_cb,
                    |x| (x as f32 - u16::MAX as f32 / 2.0) / (u16::MAX as f32 / 2.0),
                ),
                err_cb, None,
            )?,
            other => return Err(anyhow!("unsupported sample format: {other:?}")),
        };

        stream.play().context("failed to start input stream")?;

        Ok(Self {
            _stream: stream,
            shared,
            started_at: Instant::now(),
            device_name,
            source_sample_rate,
            source_channels,
        })
    }

    /// Seconds since recording started.
    pub fn elapsed_secs(&self) -> f32 { self.started_at.elapsed().as_secs_f32() }

    /// Latest RMS for VU meter. Roughly 0.0 (silence) to ~0.3 (loud speech).
    pub fn vu_level(&self) -> f32 { self.shared.lock().last_rms }

    pub fn device_name(&self) -> &str { &self.device_name }

    #[allow(dead_code)]
    pub fn source_info(&self) -> (u32, u16) { (self.source_sample_rate, self.source_channels) }

    /// Stop the stream and return the accumulated 16 kHz mono samples.
    pub fn finish(self) -> Vec<f32> {
        // Destructure to drop the stream explicitly before locking.
        let Self { _stream, shared, .. } = self;
        drop(_stream);
        let mut guard = shared.lock();
        std::mem::take(&mut guard.samples_16k_mono)
    }
}

pub fn list_devices() -> Vec<String> {
    let host = cpal::default_host();
    match host.input_devices() {
        Ok(devices) => devices
            .filter_map(|d| d.name().ok())
            .collect(),
        Err(_) => vec![],
    }
}

pub fn default_device_name() -> String {
    cpal::default_host()
        .default_input_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_else(|| "none".to_string())
}

/// Convert a block of interleaved samples into mono f32 at TARGET_SAMPLE_RATE
/// and append to the shared buffer. `to_f32` normalises the input sample type.
fn process_block<S: Copy>(
    data: &[S],
    channels: u16,
    src_rate: u32,
    shared: &Arc<Mutex<Shared>>,
    to_f32: impl Fn(S) -> f32,
) {
    if data.is_empty() { return; }
    let chans = channels.max(1) as usize;

    // 1) Downmix interleaved -> mono by averaging channels.
    let frame_count = data.len() / chans;
    let mut mono = Vec::with_capacity(frame_count);
    for f in 0..frame_count {
        let mut acc = 0.0f32;
        for c in 0..chans {
            acc += to_f32(data[f * chans + c]);
        }
        mono.push(acc / chans as f32);
    }

    // 2) RMS for VU meter.
    let rms = if mono.is_empty() {
        0.0
    } else {
        (mono.iter().map(|x| x * x).sum::<f32>() / mono.len() as f32).sqrt()
    };

    // 3) Linear resample mono -> TARGET_SAMPLE_RATE.
    //    Maintain a fractional read position across callbacks to avoid clicks
    //    at block boundaries.
    let ratio = src_rate as f64 / TARGET_SAMPLE_RATE as f64;
    let mut out = Vec::with_capacity((frame_count as f64 / ratio) as usize + 4);

    let mut s = shared.lock();
    let mut pos = s.resample_pos;
    while pos < mono.len() as f64 - 1.0 {
        let i = pos.floor() as usize;
        let frac = (pos - i as f64) as f32;
        let sample = mono[i] * (1.0 - frac) + mono[i + 1] * frac;
        out.push(sample);
        pos += ratio;
    }
    // Carry remainder into next callback (subtract block length).
    s.resample_pos = pos - mono.len() as f64;
    s.samples_16k_mono.extend_from_slice(&out);
    s.last_rms = rms;
}

/// Persist mono 16k f32 samples to a WAV file.
pub fn write_wav_16k(path: &std::path::Path, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}
