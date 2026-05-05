//! Communications Coach — local-first speaking practice for engineering leaders.
//!
//! Architecture:
//!   - eframe/egui drives the UI on the main thread (immediate mode).
//!   - A dedicated tokio runtime runs network + heavy CPU work off the UI thread.
//!   - The UI and worker communicate via mpsc channels carrying typed messages.

mod analysis;
mod app;
mod audio;
mod coach;
mod history;
mod llm;
mod personas;
mod stt;
mod tts;

use app::CoachApp;

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Build a multi-threaded tokio runtime that the UI can dispatch work onto.
    // We leak it intentionally — it lives for the entire process.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime");
    let rt_handle = rt.handle().clone();
    std::mem::forget(rt);

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 720.0])
            .with_min_inner_size([640.0, 520.0])
            .with_title("Communications Coach"),
        ..Default::default()
    };

    eframe::run_native(
        "Communications Coach",
        native_options,
        Box::new(move |cc| Ok(Box::new(CoachApp::new(cc, rt_handle)))),
    )
}
