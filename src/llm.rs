//! Minimal Ollama client. Talks to the local daemon at OLLAMA_HOST (default
//! http://localhost:11434) using the /api/chat endpoint with streaming enabled.

use anyhow::{Context, Result};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String, // "system" | "user" | "assistant"
    pub content: String,
}

impl ChatMessage {
    pub fn system(s: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: s.into(),
        }
    }
    pub fn user(s: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: s.into(),
        }
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
    options: ChatOptions,
}

#[derive(Serialize)]
struct ChatOptions {
    temperature: f32,
    // Keep responses bounded so feedback stays focused.
    num_predict: i32,
}

#[derive(Deserialize)]
struct ChatChunk {
    message: Option<ChunkMessage>,
    done: bool,
}

#[derive(Deserialize)]
struct ChunkMessage {
    content: String,
}

/// One streaming token-ish update from the model.
#[derive(Debug)]
pub enum LlmUpdate {
    /// Incremental text fragment.
    Token(String),
    /// Stream finished cleanly with the full accumulated text.
    Done(String),
    /// Stream failed — string is human-readable.
    Error(String),
}

#[derive(Clone)]
pub struct OllamaClient {
    http: Client,
    base_url: String,
    pub model: String,
}

impl OllamaClient {
    pub fn new(model: impl Into<String>) -> Self {
        let base_url =
            std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());
        Self {
            http: Client::builder()
                // Long timeout — model load + first token can take a while on cold start.
                .timeout(std::time::Duration::from_secs(300))
                .build()
                .expect("reqwest client"),
            base_url,
            model: model.into(),
        }
    }

    /// Stream a chat completion. Sends `LlmUpdate`s on the channel until done/error.
    /// The receiver side decides what to do with the partial text (typically: forward
    /// to the UI thread via egui::Context::request_repaint after pushing into shared state).
    pub async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        temperature: f32,
        max_tokens: i32,
        tx: mpsc::UnboundedSender<LlmUpdate>,
    ) {
        if let Err(e) = self
            .chat_stream_inner(messages, temperature, max_tokens, &tx)
            .await
        {
            let _ = tx.send(LlmUpdate::Error(format!("{e:#}")));
        }
    }

    async fn chat_stream_inner(
        &self,
        messages: Vec<ChatMessage>,
        temperature: f32,
        max_tokens: i32,
        tx: &mpsc::UnboundedSender<LlmUpdate>,
    ) -> Result<()> {
        let req = ChatRequest {
            model: &self.model,
            messages: &messages,
            stream: true,
            options: ChatOptions {
                temperature,
                num_predict: max_tokens,
            },
        };

        let resp = self
            .http
            .post(format!("{}/api/chat", self.base_url))
            .json(&req)
            .send()
            .await
            .context("POST /api/chat — is `ollama serve` running?")?
            .error_for_status()
            .context("Ollama returned an error status")?;

        // Ollama streams newline-delimited JSON. We accumulate bytes in a buffer
        // and split on '\n' since chunks may arrive mid-line.
        let mut buf = Vec::<u8>::new();
        let mut full = String::new();
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("network read")?;
            buf.extend_from_slice(&chunk);

            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=pos).collect();
                let line = &line[..line.len() - 1]; // strip \n
                if line.is_empty() {
                    continue;
                }

                let parsed: ChatChunk = match serde_json::from_slice(line) {
                    Ok(c) => c,
                    Err(_) => continue, // tolerate the occasional partial / metadata line
                };

                if let Some(msg) = parsed.message {
                    if !msg.content.is_empty() {
                        full.push_str(&msg.content);
                        let _ = tx.send(LlmUpdate::Token(msg.content));
                    }
                }
                if parsed.done {
                    let _ = tx.send(LlmUpdate::Done(full.clone()));
                    return Ok(());
                }
            }
        }

        // Stream ended without a `done: true` — still return what we have.
        let _ = tx.send(LlmUpdate::Done(full));
        Ok(())
    }
}
