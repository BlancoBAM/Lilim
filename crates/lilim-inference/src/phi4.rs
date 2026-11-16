// lilim-inference: Phi-4-mini Engine
//
// Loads and runs Microsoft Phi-4-mini-instruct for token-by-token streaming.
// Uses mistral.rs for high-performance local inference with automatic ISQ quantization.
//
// Model: microsoft/Phi-4-mini-instruct (GGUF Q4_K_M via mistral.rs auto-ISQ)
// Context: 128K tokens (capped to 4096 in practice for local CPU use)
// Params: 3.8B
// Speed: ~10-30 tokens/sec on CPU (depends on hardware, uses paged attention)
//
// Chat template (Phi-4 instruct format):
//   <|system|>\n{system}\n<|end|>\n<|user|>\n{user}\n<|end|>\n<|assistant|>\n
//
// Why Phi-4-mini over Phi-3.5:
//   • Better reasoning, math, coding, and function-calling capabilities
//   • Same 3.8B param count but significantly improved quality
//   • Native tool calling support in the model
//   • mistral.rs provides faster inference with paged attention and ISQ quantization

use anyhow::{Context, Result};
use mistralrs::{
    IsqType, ModelBuilder, PagedAttentionMetaBuilder, TextMessageRole, TextMessages,
    MemoryGpuConfig,
};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::config::InferenceConfig;
use crate::downloader::get_model_dir;
use crate::TokenStream;

/// The Phi-4-mini inference engine using mistral.rs.
/// Thread-safe: wrapped in Arc<Mutex<>> so it can be shared across concurrent handlers.
pub struct Phi3Engine {
    inner: Arc<Mutex<Phi3Inner>>,
    config: InferenceConfig,
}

struct Phi3Inner {
    model: mistralrs::Model,
    config: InferenceConfig,
}

impl Phi3Engine {
    /// Load the model weights and tokenizer into memory using mistral.rs.
    pub async fn load(config: &InferenceConfig) -> Result<Self> {
        let config = config.clone();
        let _model_dir = get_model_dir(&config);
        let weights_path = config.weights_path();

        // Determine ISQ quantization type based on device
        let isq_type = if config.use_cuda {
            IsqType::Q4K
        } else if config.use_metal {
            IsqType::AFQ4
        } else {
            IsqType::Q4K
        };

        info!(
            "Loading Phi-4-mini on {} from {}",
            config.device_label(),
            weights_path.display()
        );

        // Build the model using mistral.rs ModelBuilder
        // Use local GGUF path if available, otherwise fall back to HF model ID
        let model = if weights_path.exists() {
            info!("Using local GGUF weights at {}", weights_path.display());
            ModelBuilder::new(weights_path.to_string_lossy().to_string())
                .with_isq(isq_type)
                .with_logging()
                .with_paged_attn(
                    PagedAttentionMetaBuilder::default()
                        .with_gpu_memory(MemoryGpuConfig::ContextSize(config.context_size))
                        .build()?,
                )
                .build()
                .await
                .context("Failed to build Phi-4-mini model from local GGUF")?
        } else {
            info!("Local GGUF not found, using HuggingFace model ID");
            ModelBuilder::new("microsoft/Phi-4-mini-instruct")
                .with_auto_isq(mistralrs::IsqBits::Four)
                .with_logging()
                .with_paged_attn(
                    PagedAttentionMetaBuilder::default()
                        .with_gpu_memory(MemoryGpuConfig::ContextSize(config.context_size))
                        .build()?,
                )
                .build()
                .await
                .context("Failed to build Phi-4-mini model from HuggingFace")?
        };

        info!("Phi-4-mini loaded ✓");
        let inner_config = config.clone();
        Ok(Self {
            inner: Arc::new(Mutex::new(Phi3Inner { model, config: inner_config })),
            config,
        })
    }

    /// Generate tokens as an async stream using mistral.rs streaming API.
    pub async fn generate_stream(
        &self,
        user_message: &str,
        max_tokens: usize,
    ) -> Result<TokenStream> {
        let prompt = format_phi4_prompt(user_message, &self.config.system_prompt);
        let inner = self.inner.clone();
        let config = self.config.clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String>>(64);

        tokio::task::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            generate_blocking(&mut guard, &prompt, max_tokens, &config, tx);
        });

        let token_stream = futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });

        Ok(Box::pin(token_stream))
    }
}

/// Format the Phi-4 instruct prompt.
///
/// Phi-4 instruct chat template:
///   <|system|>\n{system}\n<|end|>\n<|user|>\n{user}\n<|end|>\n<|assistant|>\n
///
/// The system prompt is loaded from config (defaults to Lilim's agent persona).
fn format_phi4_prompt(user_message: &str, system_prompt: &str) -> String {
    format!(
        "<|system|>\n{system}\n<|end|>\n<|user|>\n{user}\n<|end|>\n<|assistant|>\n",
        system = system_prompt,
        user = user_message,
    )
}

/// Synchronous token generation — runs in a blocking thread using mistral.rs streaming.
fn generate_blocking(
    inner: &mut Phi3Inner,
    prompt: &str,
    max_tokens: usize,
    _config: &InferenceConfig,
    tx: tokio::sync::mpsc::Sender<Result<String>>,
) {
    let start = Instant::now();

    // Create messages for mistral.rs
    let messages = TextMessages::new()
        .add_message(TextMessageRole::System, inner.config.system_prompt.clone())
        .add_message(TextMessageRole::User, prompt);

    // Use mistral.rs streaming chat request
    let rt = tokio::runtime::Handle::current();
    let stream_result = rt.block_on(async {
        inner.model.stream_chat_request(messages).await
    });

    let mut stream = match stream_result {
        Err(e) => {
            let _ = tx.blocking_send(Err(anyhow::anyhow!("Stream creation failed: {e}")));
            return;
        }
        Ok(s) => s,
    };

    let mut token_count = 0usize;
    let mut first_token = true;
    let mut first_token_time = None;

    while let Some(response) = rt.block_on(stream.next()) {
        match response {
            mistralrs::Response::Chunk(chunk) => {
                if let Some(choice) = chunk.choices.first() {
                    if let Some(content) = &choice.delta.content {
                        if !content.is_empty() {
                            if first_token {
                                first_token_time = Some(Instant::now());
                                first_token = false;
                            }

                            token_count += 1;

                            // Stop on Phi-4 end-of-turn markers
                            if content.contains("<|end|>") || content.contains("<|endoftext|>") {
                                break;
                            }

                            if tx.blocking_send(Ok(content.clone())).is_err() {
                                // Receiver dropped — client disconnected
                                break;
                            }
                        }
                    }
                }

                // Check for finish reason
                if let Some(choice) = chunk.choices.first() {
                    if choice.finish_reason.is_some() {
                        break;
                    }
                }
            }
            mistralrs::Response::Done(reason) => {
                debug!("Generation finished: {:?}", reason);
                break;
            }
            _ => {
                // Handle other response variants (errors, completion, etc.)
                break;
            }
        }

        if token_count >= max_tokens {
            break;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    if let Some(ft_time) = first_token_time {
        let ttft = ft_time.duration_since(start).as_secs_f64();
        let gen_elapsed = elapsed - ttft;
        if token_count > 0 && gen_elapsed > 0.0 {
            let tps = token_count as f64 / gen_elapsed;
            info!(
                "Generated {token_count} tokens in {gen_elapsed:.1}s ({tps:.1} tok/s) | TTFT: {ttft:.1}s"
            );
            if tps < _config.min_tokens_per_sec {
                warn!("Speed {tps:.2} tok/s below threshold — online routing recommended");
            }
        }
    }
}