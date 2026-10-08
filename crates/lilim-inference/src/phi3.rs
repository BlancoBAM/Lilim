// lilim-inference: Phi-3.5-mini Engine
//
// Loads and runs Microsoft Phi-3.5-mini-instruct for token-by-token streaming.
// Uses HuggingFace Candle's built-in quantized_phi3 model — no new dependencies.
//
// Model: microsoft/Phi-3.5-mini-instruct (GGUF Q4_K_M)
// Context: 128K tokens (capped to 4096 in practice for local CPU use)
// Params: 3.8B
// Speed: ~2-8 tokens/sec on CPU (depends on hardware)
//
// Chat template (Phi-3 instruct format):
//   <|system|>\n{system}\n<|end|>\n<|user|>\n{user}\n<|end|>\n<|assistant|>\n
//
// Why Phi-3.5 over Phi-2:
//   • Proper instruct fine-tune → far better at following agentic tool instructions
//   • 3.8B vs 2.7B: meaningfully better reasoning with only ~40% more RAM
//   • RMSNorm + RoPE → more stable outputs across long conversations
//   • Same Candle module (quantized_phi3): zero new Rust dependencies

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3Weights;
use futures_util::stream;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::config::InferenceConfig;
use crate::downloader::get_model_dir;
use crate::TokenStream;

// ── Token IDs for Phi-3 special tokens ────────────────────────────────────────
// These are the standard IDs from the microsoft/Phi-3.5-mini-instruct tokenizer.
// We look them up dynamically at load time; these are the fallbacks if lookup fails.
const PHI3_EOS_TOKEN: &str = "<|endoftext|>";
const PHI3_EOS_FALLBACK: u32 = 32000;

/// The Phi-3.5-mini inference engine.
/// Thread-safe: wrapped in Arc<Mutex<>> so it can be shared across concurrent handlers.
pub struct Phi3Engine {
    inner: Arc<Mutex<Phi3Inner>>,
    config: InferenceConfig,
}

struct Phi3Inner {
    model: Phi3Weights,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
}

impl Phi3Engine {
    /// Load the model weights and tokenizer into memory.
    pub async fn load(config: &InferenceConfig) -> Result<Self> {
        let config = config.clone();
        let model_dir = get_model_dir(&config);

        let config_for_task = config.clone();
        let inner = tokio::task::spawn_blocking(move || -> Result<Phi3Inner> {
            let device = select_device(&config_for_task)?;
            info!(
                "Loading Phi-3.5-mini on {} from {}",
                config_for_task.device_label(),
                model_dir.display()
            );

            // ── Load GGUF weights ──────────────────────────────────────────────
            let weights_path = config_for_task.weights_path();
            let mut gguf_file = std::fs::File::open(&weights_path)
                .with_context(|| format!("Cannot open weights at {}", weights_path.display()))?;

            let content = candle_core::quantized::gguf_file::Content::read(&mut gguf_file)
                .context("Failed to parse GGUF file")?;

            let model = Phi3Weights::from_gguf(false, content, &mut gguf_file, &device)
                .context("Failed to load Phi-3.5 weights from GGUF")?;

            // ── Load tokenizer ─────────────────────────────────────────────────
            let tokenizer_path = config_for_task.tokenizer_path();
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

            // Resolve EOS token ID
            let eos_token_id = tokenizer
                .token_to_id(PHI3_EOS_TOKEN)
                .unwrap_or(PHI3_EOS_FALLBACK);

            info!("Phi-3.5-mini loaded ✓ (EOS token id: {})", eos_token_id);
            Ok(Phi3Inner { model, tokenizer, device, eos_token_id })
        })
        .await
        .context("Blocking task panicked")?
        .context("Failed to load Phi-3.5-mini")?;

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
            config,
        })
    }

    /// Generate tokens as an async stream.
    pub async fn generate_stream(
        &self,
        user_message: &str,
        max_tokens: usize,
    ) -> Result<TokenStream> {
        let prompt = format_phi3_prompt(user_message, &self.config.system_prompt);
        let inner = self.inner.clone();
        let config = self.config.clone();

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String>>(64);

        tokio::task::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            generate_blocking(&mut guard, &prompt, max_tokens, &config, tx);
        });

        let token_stream = stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });

        Ok(Box::pin(token_stream))
    }
}

/// Format the Phi-3 instruct prompt.
///
/// Phi-3 instruct chat template:
///   <|system|>\n{system}\n<|end|>\n<|user|>\n{user}\n<|end|>\n<|assistant|>\n
///
/// The system prompt is loaded from config (defaults to Lilim's agent persona).
fn format_phi3_prompt(user_message: &str, system_prompt: &str) -> String {
    format!(
        "<|system|>\n{system}\n<|end|>\n<|user|>\n{user}\n<|end|>\n<|assistant|>\n",
        system = system_prompt,
        user = user_message,
    )
}

/// Extract the last-position logits from whatever shape the model returns.
///
/// Phi-3 quantized can return [vocab], [1, vocab], or [1, seq, vocab].
/// Normalise all to 1D [vocab] for sampling.
fn extract_last_logits(logits: &Tensor) -> candle_core::Result<Tensor> {
    match logits.dims() {
        [_vocab] => Ok(logits.clone()),
        [1, _vocab] => logits.squeeze(0),
        [1, seq_len, _vocab] => {
            let last = logits.narrow(1, seq_len - 1, 1)?;
            last.squeeze(1)?.squeeze(0)
        }
        dims => {
            let total: usize = dims.iter().product();
            logits.reshape((total,))
        }
    }
}

/// Synchronous token generation — runs in a blocking thread.
///
/// Two-phase inference:
///   Phase 1: Process prompt tokens sequentially to fill KV cache
///   Phase 2: Autoregressive generation, one token per step
fn generate_blocking(
    inner: &mut Phi3Inner,
    prompt: &str,
    max_tokens: usize,
    config: &InferenceConfig,
    tx: tokio::sync::mpsc::Sender<Result<String>>,
) {
    let start = Instant::now();
    let mut token_count = 0usize;


    // ── Tokenize ────────────────────────────────────────────────────────────
    let prompt_tokens = match inner.tokenizer.encode(prompt, true) {
        Ok(enc) => enc.get_ids().to_vec(),
        Err(e) => {
            let _ = tx.blocking_send(Err(anyhow::anyhow!("Tokenization failed: {e}")));
            return;
        }
    };

    // Respect context window — keep the tail of the prompt if it overflows
    let max_prompt = config.context_size.saturating_sub(max_tokens + 64);
    let prompt_tokens = if prompt_tokens.len() > max_prompt {
        prompt_tokens[prompt_tokens.len() - max_prompt..].to_vec()
    } else {
        prompt_tokens
    };

    let prompt_len = prompt_tokens.len();
    info!("Prompt tokenized: {} tokens", prompt_len);

    // ── Sampling config ──────────────────────────────────────────────────────
    let mut logits_processor = LogitsProcessor::new(42, Some(config.temperature), Some(config.top_p));

    // ── Phase 1: Prefill prompt into KV cache ───────────────────────────────
    let mut pos = 0usize;
    let mut last_raw_logits: Option<Tensor> = None;

    for &token in prompt_tokens.iter() {
        let token_tensor = match Tensor::new(&[token], &inner.device).and_then(|t| t.unsqueeze(0)) {
            Ok(t) => t,
            Err(e) => {
                let _ = tx.blocking_send(Err(anyhow::anyhow!("Tensor error during prefill: {e}")));
                return;
            }
        };

        let logits = match inner.model.forward(&token_tensor, pos) {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.blocking_send(Err(anyhow::anyhow!("Forward pass error (prefill): {e}")));
                return;
            }
        };

        last_raw_logits = Some(logits);
        pos += 1;
    }

    let prompt_elapsed = start.elapsed().as_secs_f64();
    info!("Prompt prefilled in {prompt_elapsed:.1}s");


    // Sample first generated token
    let last_logits = match extract_last_logits(last_raw_logits.as_ref().unwrap()) {
        Ok(l) => l,
        Err(e) => {
            let _ = tx.blocking_send(Err(anyhow::anyhow!("Logits extraction error: {e}")));
            return;
        }
    };

    let mut next_token = match logits_processor.sample(&last_logits) {
        Ok(t) => t,
        Err(e) => {
            let _ = tx.blocking_send(Err(anyhow::anyhow!("Sampling error: {e}")));
            return;
        }
    };

    // ── Phase 2: Autoregressive generation ──────────────────────────────────
    loop {
        if token_count >= max_tokens {
            break;
        }

        // Phi-3 EOS and special tokens to stop on
        if next_token == inner.eos_token_id {
            debug!("EOS token — stopping");
            break;
        }

        token_count += 1;

        // Decode token
        let token_text = inner.tokenizer.decode(&[next_token], false).unwrap_or_default();

        // Stop on Phi-3 end-of-turn markers that may appear as text
        if token_text.contains("<|end|>") || token_text.contains("<|endoftext|>") {
            break;
        }

        if !token_text.is_empty() {
            if tx.blocking_send(Ok(token_text)).is_err() {
                // Receiver dropped — client disconnected
                break;
            }
        }

        // Forward pass for next token
        let token_tensor = match Tensor::new(&[next_token], &inner.device).and_then(|t| t.unsqueeze(0)) {
            Ok(t) => t,
            Err(e) => {
                let _ = tx.blocking_send(Err(anyhow::anyhow!("Tensor error: {e}")));
                break;
            }
        };

        let raw_logits = match inner.model.forward(&token_tensor, pos) {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.blocking_send(Err(anyhow::anyhow!("Forward pass error: {e}")));
                break;
            }
        };

        pos += 1;

        let logits = match extract_last_logits(&raw_logits) {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.blocking_send(Err(anyhow::anyhow!("Logits extraction error: {e}")));
                break;
            }
        };

        next_token = match logits_processor.sample(&logits) {
            Ok(t) => t,
            Err(e) => {
                let _ = tx.blocking_send(Err(anyhow::anyhow!("Sampling error: {e}")));
                break;
            }
        };
    }

    let elapsed = start.elapsed().as_secs_f64();
    let gen_elapsed = elapsed - prompt_elapsed;
    if token_count > 0 && gen_elapsed > 0.0 {
        let tps = token_count as f64 / gen_elapsed;
        info!(
            "Generated {token_count} tokens in {gen_elapsed:.1}s ({tps:.1} tok/s) | prefill: {prompt_elapsed:.1}s"
        );
        if tps < config.min_tokens_per_sec {
            warn!("Speed {tps:.2} tok/s below threshold — online routing recommended");
        }
    }
}

/// Select compute device based on config.
fn select_device(_config: &InferenceConfig) -> Result<Device> {
    #[cfg(feature = "cuda")]
    if _config.use_cuda {
        match Device::new_cuda(0) {
            Ok(d) => {
                info!("Using CUDA device 0");
                return Ok(d);
            }
            Err(e) => warn!("CUDA unavailable: {e} — falling back to CPU"),
        }
    }

    #[cfg(feature = "metal")]
    if config.use_metal {
        match Device::new_metal(0) {
            Ok(d) => {
                info!("Using Metal device");
                return Ok(d);
            }
            Err(e) => warn!("Metal unavailable: {e} — falling back to CPU"),
        }
    }

    info!("Using CPU for inference");
    Ok(Device::Cpu)
}
