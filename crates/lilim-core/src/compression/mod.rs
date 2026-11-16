//! Token Compression - conversation/payload compression for context-window management
//!
//! Stage 1 (shell output via `rtk` binary) is handled by the shell tool invoking the
//! `rtk` CLI as a subprocess (rtk is bundled in packaging, not a library dependency).
//! Stage 2 (outbound payloads) uses ai_tokenopt. This module provides in-process
//! conversation compression using a simple char/4 token approximation so it compiles
//! without the rtk library crate.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::config::AgentConfig;

/// Compression errors
#[derive(Debug, thiserror::Error)]
pub enum CompressionError {
    #[error("Token counting error: {0}")]
    TokenCount(String),
    #[error("Compression error: {0}")]
    Compression(String),
    #[error("Model not available")]
    ModelNotAvailable,
    #[error("Input too large")]
    InputTooLarge,
}

/// Approximate tokens as characters / 4 (standard rough estimate).
fn approx_tokens(text: &str) -> usize {
    text.len().saturating_div(4).max(1)
}

/// Token compressor (in-process, no external library dependency)
pub struct TokenCompressor {
    config: AgentConfig,
}

impl TokenCompressor {
    /// Create a new token compressor
    pub async fn new() -> Result<Self> {
        Ok(Self {
            config: AgentConfig::default(),
        })
    }

    /// Compress a conversation to fit within token budget
    #[tracing::instrument(skip(self))]
    pub async fn compress_conversation(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: usize,
        preserve_recent: usize,
    ) -> Result<Vec<ChatMessage>, CompressionError> {
        let current_tokens = self.count_messages(&messages)?;

        if current_tokens <= max_tokens {
            return Ok(messages);
        }

        info!("Compressing conversation: {} -> {} tokens", current_tokens, max_tokens);

        if messages.len() <= preserve_recent + 2 {
            return Ok(messages);
        }

        let system_msg = messages.first().cloned();
        let recent = messages
            .iter()
            .rev()
            .take(preserve_recent)
            .cloned()
            .collect::<Vec<_>>();
        let middle = messages[1..messages.len() - preserve_recent].to_vec();

        let summary = self.summarize_messages(&middle).await?;

        let mut compressed = Vec::new();
        if let Some(sys) = system_msg {
            compressed.push(sys);
        }

        compressed.push(ChatMessage {
            role: "system".to_string(),
            content: format!("[Conversation summary]: {}", summary),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });

        compressed.extend(recent.into_iter().rev());

        let new_tokens = self.count_messages(&compressed)?;
        debug!("Compressed to {} tokens", new_tokens);

        if new_tokens > max_tokens {
            return self.truncate_to_fit(compressed, max_tokens).await;
        }

        Ok(compressed)
    }

    async fn summarize_messages(&self, messages: &[ChatMessage]) -> Result<String, CompressionError> {
        if messages.is_empty() {
            return Ok("No prior conversation".to_string());
        }

        let key_points: Vec<String> = messages
            .iter()
            .filter(|m| m.role == "user" || m.role == "assistant")
            .map(|m| {
                let content = m.content.chars().take(200).collect::<String>();
                format!("{}: {}", m.role, content)
            })
            .collect();

        let summary = format!("Key exchange points:\n{}", key_points.join("\n"));

        let summary_tokens = approx_tokens(&summary);
        if summary_tokens > 500 {
            let chars_per_token = summary.len() as f32 / summary_tokens as f32;
            let target_chars = (500.0 * chars_per_token) as usize;
            Ok(summary.chars().take(target_chars).collect::<String>() + "...")
        } else {
            Ok(summary)
        }
    }

    async fn truncate_to_fit(
        &self,
        mut messages: Vec<ChatMessage>,
        max_tokens: usize,
    ) -> Result<Vec<ChatMessage>, CompressionError> {
        while self.count_messages(&messages)? > max_tokens && messages.len() > 2 {
            let idx = messages
                .iter()
                .position(|m| m.role != "system")
                .unwrap_or(1);
            messages.remove(idx);
        }
        Ok(messages)
    }

    pub fn count_message(&self, message: &ChatMessage) -> Result<usize, CompressionError> {
        let text = format!("{}: {}", message.role, message.content);
        Ok(approx_tokens(&text))
    }

    pub fn count_messages(&self, messages: &[ChatMessage]) -> Result<usize, CompressionError> {
        let total: usize = messages.iter().map(|m| self.count_message(m).unwrap_or(0)).sum();
        Ok(total)
    }

    pub async fn compress_message(
        &self,
        message: &str,
        max_tokens: usize,
    ) -> Result<String, CompressionError> {
        let tokens = approx_tokens(message);
        if tokens <= max_tokens {
            return Ok(message.to_string());
        }
        let chars_per_token = message.len() as f32 / tokens as f32;
        let target_chars = (max_tokens as f32 * chars_per_token * 0.9) as usize;
        let compressed = message.chars().take(target_chars).collect::<String>() + "... [truncated]";
        Ok(compressed)
    }

    pub async fn smart_compress(
        &self,
        messages: &[ChatMessage],
        target_tokens: usize,
        _preserve_patterns: &[String],
    ) -> Result<Vec<ChatMessage>, CompressionError> {
        let current = self.count_messages(messages)?;
        if current <= target_tokens {
            return Ok(messages.to_vec());
        }
        let mut compressed: Vec<ChatMessage> = messages
            .iter()
            .map(|m| {
                let mut m = m.clone();
                m.content = normalize_whitespace(&m.content);
                m
            })
            .collect();
        if self.count_messages(&compressed)? <= target_tokens {
            return Ok(compressed);
        }
        self.compress_conversation(compressed, target_tokens, 6).await
    }
}

fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<ToolCallRequest>>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}
