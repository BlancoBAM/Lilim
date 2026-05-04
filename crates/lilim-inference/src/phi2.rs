// lilim-inference: Phi-2 Engine
//
// Loads and runs Microsoft Phi-2 for token-by-token streaming inference.
// Uses HuggingFace Candle for pure-Rust ML — no Python, no Ollama.
//
// Model: microsoft/phi-2 (GGUF Q4_K_M quantization)
// Context: 2048 tokens
// Params: 2.7B
// Speed: ~3-15 tokens/sec on CPU (depends on hardware)
//
// The chat template wraps user messages in Phi-2's instruct format:
//   "Instruct: {user_message}\nOutput:"

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_phi::ModelWeights as Phi2Weights;
use futures_util::stream;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::config::InferenceConfig;
use crate::downloader::get_model_dir;
use crate::TokenStream;

/// The Phi-2 inference engine.
/// Thread-safe: wrapped in Arc<Mutex<>> internally so it can be shared
/// across concurrent Axum request handlers.
pub struct Phi2Engine {
    inner: Arc<Mutex<Phi2Inner>>,
    config: InferenceConfig,
}

struct Phi2Inner {
    model: Phi2Weights,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
}

impl Phi2Engine {
    /// Load the model weights and tokenizer into memory.
    pub async fn load(config: &InferenceConfig) -> Result<Self> {
        let config = config.clone();
        let model_dir = get_model_dir(&config);

        // Run heavy IO/compute in blocking thread pool
        let config_for_task = config.clone();
        let inner = tokio::task::spawn_blocking(move || -> Result<Phi2Inner> {
            let device = select_device(&config_for_task)?;
            info!("Loading Phi-2 on {} from {}", config_for_task.device_label(), model_dir.display());

            // Load GGUF weights
            let weights_path = config_for_task.weights_path();
            let mut gguf_file = std::fs::File::open(&weights_path)
                .with_context(|| format!("Cannot open weights at {}", weights_path.display()))?;

            let content = candle_core::quantized::gguf_file::Content::read(&mut gguf_file)
                .context("Failed to parse GGUF file")?;

            let model = Phi2Weights::from_gguf(content, &mut gguf_file, &device)
                .context("Failed to load Phi-2 weights from GGUF")?;

            // Load tokenizer
            let tokenizer_path = config_for_task.tokenizer_path();
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

            // Find EOS token
            let eos_token_id = tokenizer
                .token_to_id("<|endoftext|>")
                .unwrap_or(50256); // GPT-2 / Phi-2 default EOS

            info!("Phi-2 loaded ✓ (EOS token: {})", eos_token_id);

            Ok(Phi2Inner { model, tokenizer, device, eos_token_id })
        })
        .await
        .context("Blocking task panicked")?
        .context("Failed to load Phi-2")?;

        Ok(Self {
            inner: Arc::new(Mutex::new(inner)),
            config,
        })
    }

    /// Generate tokens as an async stream.
    /// Each item in the stream is a Result<String> containing one or more chars.
    pub async fn generate_stream(
        &self,
        user_message: &str,
        max_tokens: usize,
    ) -> Result<TokenStream> {
        let prompt = format_phi2_prompt(user_message);
        let inner = self.inner.clone();
        let config = self.config.clone();

        // Generate in a blocking thread (Candle is not async internally)
        // We use a channel to bridge sync generation → async stream
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String>>(64);

        tokio::task::spawn_blocking(move || {
            let mut guard = inner.blocking_lock();
            generate_blocking(&mut guard, &prompt, max_tokens, &config, tx);
        });

        // Convert the receiver into a Stream
        let token_stream = stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|item| (item, rx))
        });

        Ok(Box::pin(token_stream))
    }
}

/// Format the Phi-2 instruct prompt.
/// Phi-2 uses: "Instruct: {message}\nOutput:" for instruction following.
fn format_phi2_prompt(user_message: &str) -> String {
    format!("Instruct: {user_message}\nOutput:")
}

/// Synchronous token generation — runs in blocking thread pool.
///
/// Uses two-phase inference for performance:
///   Phase 1: Process the full prompt in one forward pass (builds KV cache)
///   Phase 2: Generate tokens one at a time, feeding only the new token each step
///
/// This is O(n) instead of O(n²) — critical for acceptable speed on CPU.
fn generate_blocking(
    inner: &mut Phi2Inner,
    prompt: &str,
    max_tokens: usize,
    config: &InferenceConfig,
    tx: tokio::sync::mpsc::Sender<Result<String>>,
) {
    let start = Instant::now();
    let mut token_count = 0usize;

    // Tokenize prompt
    let prompt_tokens = match inner.tokenizer.encode(prompt, true) {
        Ok(enc) => enc.get_ids().to_vec(),
        Err(e) => {
            let _ = tx.blocking_send(Err(anyhow::anyhow!("Tokenization failed: {e}")));
            return;
        }
    };

    // Truncate prompt if it exceeds context window
    let context_size = config.context_size;
    let prompt_tokens = if prompt_tokens.len() > context_size - 64 {
        // Leave room for generation
        prompt_tokens[prompt_tokens.len() - (context_size - 64)..].to_vec()
    } else {
        prompt_tokens
    };

    let prompt_len = prompt_tokens.len();
    debug!("Prompt tokenized: {} tokens", prompt_len);

    // Set up logits processor for sampling
    let seed = 42u64;
    let mut logits_processor = LogitsProcessor::new(
        seed,
        Some(config.temperature),
        Some(config.top_p),
    );

    // ── Phase 1: Process the full prompt in one forward pass ──
    // This builds the internal KV-cache inside the GGUF model.
    let prompt_tensor = match Tensor::new(prompt_tokens.as_slice(), &inner.device)
        .and_then(|t| t.unsqueeze(0))
    {
        Ok(t) => t,
        Err(e) => {
            let _ = tx.blocking_send(Err(anyhow::anyhow!("Tensor error during prompt: {e}")));
            return;
        }
    };

    let prompt_logits = match inner.model.forward(&prompt_tensor, 0) {
        Ok(l) => l,
        Err(e) => {
            let _ = tx.blocking_send(Err(anyhow::anyhow!("Forward pass error (prompt): {e}")));
            return;
        }
    };

    // Extract logits for the last prompt token
    let last_logits = match prompt_logits.squeeze(0).and_then(|l| {
        let seq_len = l.dim(0)?;
        l.get(seq_len - 1)
    }) {
        Ok(l) => l,
        Err(e) => {
            let _ = tx.blocking_send(Err(anyhow::anyhow!("Logits extraction error: {e}")));
            return;
        }
    };

    // Sample first generated token from the prompt's output
    let mut next_token = match logits_processor.sample(&last_logits) {
        Ok(t) => t,
        Err(e) => {
            let _ = tx.blocking_send(Err(anyhow::anyhow!("Sampling error: {e}")));
            return;
        }
    };

    let prompt_elapsed = start.elapsed().as_secs_f64();
    debug!("Prompt processed in {prompt_elapsed:.1}s");

    // Track position for incremental KV-cache indexing
    let mut pos = prompt_len;

    // ── Phase 2: Generate tokens one at a time (incremental) ──
    // Each step feeds ONLY the single new token. The GGUF model's internal
    // KV-cache retains context from all previous tokens.
    loop {
        if token_count >= max_tokens {
            break;
        }

        // Check for EOS
        if next_token == inner.eos_token_id {
            debug!("EOS token generated — stopping");
            break;
        }

        token_count += 1;

        // Decode and send the token to the stream
        let token_text = match inner.tokenizer.decode(&[next_token], false) {
            Ok(t) => t,
            Err(_) => String::new(),
        };

        if !token_text.is_empty() {
            if tx.blocking_send(Ok(token_text)).is_err() {
                // Receiver dropped — UI probably closed
                break;
            }
        }

        // Feed ONLY the new token for the next step
        let token_tensor = match Tensor::new(&[next_token], &inner.device)
            .and_then(|t| t.unsqueeze(0))
        {
            Ok(t) => t,
            Err(e) => {
                let _ = tx.blocking_send(Err(anyhow::anyhow!("Tensor error: {e}")));
                break;
            }
        };

        let logits = match inner.model.forward(&token_tensor, pos) {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.blocking_send(Err(anyhow::anyhow!("Forward pass error: {e}")));
                break;
            }
        };

        pos += 1;

        // For single-token input, squeeze and get the only position
        let logits = match logits.squeeze(0).and_then(|l| l.get(0)) {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.blocking_send(Err(anyhow::anyhow!("Logits extraction error: {e}")));
                break;
            }
        };

        // Sample next token
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
            "Generated {token_count} tokens in {gen_elapsed:.1}s ({tps:.1} tok/s) | prompt: {prompt_elapsed:.1}s",
        );
        if tps < 0.5 {
            warn!("Inference speed ({tps:.2} tok/s) is below threshold — online routing recommended");
        }
    }
}

/// Select the compute device based on config and available hardware.
fn select_device(_config: &InferenceConfig) -> Result<Device> {
    #[cfg(feature = "cuda")]
    if config.use_cuda {
        match Device::new_cuda(0) {
            Ok(device) => {
                info!("Using CUDA device 0");
                return Ok(device);
            }
            Err(e) => {
                warn!("CUDA requested but unavailable: {e} — falling back to CPU");
            }
        }
    }

    #[cfg(feature = "metal")]
    if config.use_metal {
        match Device::new_metal(0) {
            Ok(device) => {
                info!("Using Metal device");
                return Ok(device);
            }
            Err(e) => {
                warn!("Metal requested but unavailable: {e} — falling back to CPU");
            }
        }
    }

    info!("Using CPU for inference");
    Ok(Device::Cpu)
}
