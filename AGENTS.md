# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Cursor, etc.) working in
this repo. Humans should read [README.md](README.md) first; this file covers
conventions and gotchas you'd otherwise rediscover by trial and error.

## What this project is

A small, local-first desktop app written in Rust. It records the user
speaking, transcribes locally with `whisper-cli`, computes acoustic + textual
delivery metrics, and asks a local Ollama model for structured coaching
feedback. It also persists each completed session and shows the metrics over
time as charts.

Everything runs on the user's machine. Nothing leaves `localhost`.

## Build, check, run

```bash
cargo check                # fast type-check; preferred during iteration
cargo build --release      # full optimised build
cargo run --release        # build + launch the GUI
RUST_LOG=debug cargo run   # verbose logs
```

There are no automated tests yet. The UI loop and external dependencies
(microphone, whisper, Ollama) make end-to-end tests awkward. Pure-functional
modules (`analysis.rs`, `coach::parse_*`, `app::parse_numbered_questions`,
`history` round-trip) are good targets if you add tests.

## Module map

```
main.rs            tokio runtime + eframe entry point
src/app.rs         egui app + the Stage state machine. The big file.
src/audio.rs       cpal capture → 16 kHz mono f32
src/stt.rs         shells out to whisper-cli; never links libwhisper
src/llm.rs         Ollama /api/chat streaming client (reqwest + serde)
src/coach.rs       prompt templates and theme buckets
src/analysis.rs    pace, pause structure, energy CV, filler counts
src/history.rs     JSONL persistence + UTC date formatter
```

## Architectural rules

- **UI thread is sync.** All long-running work (LLM streaming, whisper
  inference) runs on the tokio runtime built in `main.rs`. The UI talks to
  workers via `tokio::sync::mpsc`. The UI thread polls those channels each
  frame in `CoachApp::poll_workers`.
- **The state machine lives in `Stage`.** Every screen is a variant. New
  screens get a new variant, a new `render_*` method, and a new arm in
  `render_stage`. Don't sprinkle UI state across other structs.
- **Ownership trick.** `render_stage` does
  `let stage = std::mem::replace(&mut self.stage, Stage::Welcome);` so it
  can match-and-move the current variant, mutate `self`, then assign back.
  Preserve that pattern when adding stages — it's why we don't fight the
  borrow checker on large state.
- **Streaming text** lives in `StreamBuffer` (Arc<Mutex<String>>) shared
  between the worker and the UI. Workers `append`; the UI calls
  `snapshot()` and asks egui to repaint.
- **Whisper is a subprocess, not a library.** This is intentional — it
  removes the C/C++ build toolchain requirement on Windows. Don't replace
  with `whisper-rs` without a clear win.
- **No new heavyweight dependencies.** This crate aims to stay small.
  Justify additions; prefer hand-rolling small utilities (e.g. the UTC
  formatter in `history.rs`) over pulling in chrono/time for one call site.

## Conventions

- **Comments explain WHY, not WHAT.** Treat the existing comment style as the
  bar — they describe rationale, trade-offs, and gotchas. Don't add comments
  that just narrate what the next line of code does. If a comment would only
  restate identifiers, delete it.
- **Use `log::` macros, not `println!`.** `env_logger` is initialised in
  `main`. `RUST_LOG=info` is the default.
- **Error UX over panics.** User-visible failures route through
  `Stage::Error(msg)` so the user gets a recoverable screen. Reserve `panic!`
  / `expect` for genuinely impossible states (e.g. runtime build failure at
  startup).
- **History writes never block the UI.** `history::append` logs and swallows
  errors — losing a row is far less bad than freezing the UI mid-feedback.
- **One blocking task per CPU-bound job.** Whisper transcription uses
  `rt.spawn_blocking` so it doesn't starve the tokio worker pool.

## When adding a feature

1. Decide whether it's a new `Stage` (full screen) or extra UI inside an
   existing one. Most user-facing features are new stages.
2. Add the variant to `Stage`, render method to `impl CoachApp`, and an arm
   to `render_stage`.
3. If it produces persistent data, extend `Session` in `history.rs`. Keep
   new fields `#[serde(default)]` so older `history.jsonl` files still load.
4. If you spawn async work, plumb a fresh mpsc receiver through `CoachApp`
   and drain it from `poll_workers`. Don't block the UI on `.await`.
5. Run `cargo check` before claiming done. The CI on this repo is "did the
   author run cargo check"; nothing else gates a merge.

## Things that are easy to break

- **The recording timer.** `render_recording` requests a 50 ms repaint to
  animate the VU meter. Don't move that elsewhere or the timer will stutter.
- **Stream completion handling.** `handle_stream_done` switches behaviour
  based on the *current* `Stage`. If you add a stage that launches a stream,
  you must also add an arm here, otherwise the response is silently dropped.
- **History schema compatibility.** `Session` is serialised verbatim into
  `history.jsonl`. Renaming a field without a `serde(rename / alias)` shim
  will silently drop data on load. Add fields, don't rename.
- **Egui IDs in loops.** When generating widgets in a loop (e.g. trend
  plots), give each one a unique `Id` / `id_source`. Reusing IDs causes
  state collisions (collapsing headers stuck open, plots mis-rendering).

## Privacy guarantees the user expects

- No audio, transcript, feedback, or history leaves the machine.
- The only network call is `http://localhost:11434` (Ollama).
- Don't add telemetry, crash reporting, or external HTTP without an
  explicit, opt-in setting and a banner the user can see.

## Known limits / good first improvements

- **Tests.** `analysis.rs` has zero coverage and is pure-functional — start
  there.
- **Resampling.** Linear resample in `audio.rs` is fine for STT; swap to
  `rubato` if you need pitch-quality audio.
- **Filler list.** English-only and static (`analysis::FILLERS`). Multilingual
  support needs both this list and a different whisper model.
- **Intonation proxy.** Energy CV is a coarse stand-in for pitch variation;
  a real pitch tracker would be a more direct signal.
- **Date display.** `history.rs::format_utc` is UTC. A small local-time
  conversion (or a one-line chrono dep) would be friendlier.
