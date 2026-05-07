// lilim-inference: Public API
//
// Provides the InferenceEngine — a self-contained Phi-2 inference engine
// built on HuggingFace Candle. No Ollama, no Python, no external tools.
//
// Key features:
//   • Loads Phi-2 (GGUF Q4_K_M) from the bundled model path
//   • Streams tokens via async Stream
//   • Reports inference speed; caller can decide to fall back to online model
//   • Works on CPU (default) and CUDA/Metal (optional features)
//
// Usage:
//   let engine = InferenceEngine::new(InferenceConfig::default()).await?;
//   let stream = engine.generate_stream("Explain ATP synthesis", 512).await;
//   pin_mut!(stream);
//   while let Some(token) = stream.next().await {
//       print!("{}", token?);
//   }

pub mod downloader;
pub mod phi2;
pub mod config;

pub use config::InferenceConfig;
pub use phi2::Phi2Engine;

use anyhow::Result;
use futures_util::Stream;
use std::pin::Pin;
use tracing::{info, warn};

/// Token stream type returned by generate_stream.
pub type TokenStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;

/// The main inference engine handle.
/// Create one at startup and reuse across requests (model stays loaded in RAM).
pub struct InferenceEngine {
    inner: Option<Phi2Engine>,
    config: InferenceConfig,
}

impl InferenceEngine {
    /// Initialize the engine. Downloads/verifies the model if needed.
    /// Returns Ok even if model is unavailable — is_available() will return false.
    pub async fn new(config: InferenceConfig) -> Self {
        info!("Initializing Lilim local inference engine (Phi-2 via Candle)…");

        // Ensure model files are present
        match downloader::ensure_model_ready(&config).await {
            Ok(()) => {
                info!("Model files verified ✓");
            }
            Err(e) => {
                warn!("Model not available: {e}");
                return Self { inner: None, config };
            }
        }

        // Load model into memory
        match Phi2Engine::load(&config).await {
            Ok(engine) => {
                info!("Phi-2 engine loaded ✓ ({} device)", config.device_label());
                let mut this = Self { inner: Some(engine), config };

                // ── Model warmup ──────────────────────────────────────────
                // Run a tiny dummy forward pass immediately after loading.
                // This pre-warms CPU caches, memory mappings, and any JIT
                // paths inside Candle/BLAS so the first *real* user request
                // doesn't pay an extra cold-start penalty on top of its own
                // prompt-processing time.
                if this.config.warmup_on_startup {
                    info!("Running model warmup pass…");
                    match this.generate("hi", 1).await {
                        Ok(_) => info!("Model warmup complete ✓"),
                        Err(e) => warn!("Model warmup failed (non-fatal): {e}"),
                    }
                }

                this
            }
            Err(e) => {
                warn!("Failed to load Phi-2 engine: {:?}", e);
                Self { inner: None, config }
            }
        }
    }

    /// Returns true if the engine is loaded and ready for inference.
    pub fn is_available(&self) -> bool {
        self.inner.is_some()
    }

    /// Generate a streaming response for the given prompt.
    ///
    /// Yields individual tokens as they are generated.
    /// The stream ends when EOS is reached or max_tokens is hit.
    ///
    /// Returns Err immediately if the engine is not available.
    pub async fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<TokenStream> {
        match &self.inner {
            Some(engine) => engine.generate_stream(prompt, max_tokens).await,
            None => anyhow::bail!(
                "Local inference engine not available. \
                 Check model files at {}",
                self.config.model_dir().display()
            ),
        }
    }

    /// Synchronous generate — collects all tokens and returns full string.
    /// Useful for testing and routing decisions.
    pub async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        use futures_util::StreamExt;

        let stream = self.generate_stream(prompt, max_tokens).await?;
        futures_util::pin_mut!(stream);

        let mut result = String::new();
        while let Some(token) = stream.next().await {
            result.push_str(&token?);
        }
        Ok(result)
    }

    /// Returns the device being used for inference.
    pub fn device_label(&self) -> &str {
        self.config.device_label()
    }

    /// Returns the model path being used.
    pub fn model_path(&self) -> std::path::PathBuf {
        self.config.model_dir()
    }
}
