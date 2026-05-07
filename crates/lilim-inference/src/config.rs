// lilim-inference: Configuration
//
// InferenceConfig controls where the model lives and what hardware to use.

use std::path::PathBuf;

/// Configuration for the local inference engine.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Directory containing the model weights and tokenizer.
    /// Default: ~/.local/share/lilim/models/phi-2-q4
    pub model_dir_override: Option<PathBuf>,

    /// Maximum context window size in tokens.
    pub context_size: usize,

    /// Default max tokens to generate for local inference.
    /// Kept shorter than remote to ensure responsive UX on CPU.
    pub max_gen_tokens: usize,

    /// Temperature for sampling (0.0 = greedy, 1.0 = random).
    pub temperature: f64,

    /// Top-p nucleus sampling cutoff.
    pub top_p: f64,

    /// Whether to use CUDA if available.
    pub use_cuda: bool,

    /// Whether to use Metal (Apple Silicon) if available.
    pub use_metal: bool,

    /// Minimum tokens/second before we recommend falling back to online.
    /// Set to 0.0 to disable speed-based fallback.
    pub min_tokens_per_sec: f64,

    /// Run a single-token warmup pass after model load to pre-warm CPU
    /// caches. Adds ~10s to startup but removes cold-start latency from
    /// the first real user request.
    pub warmup_on_startup: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_dir_override: None,
            context_size: 2048,
            max_gen_tokens: 256,
            temperature: 0.5,
            top_p: 0.9,
            use_cuda: cfg!(feature = "cuda"),
            use_metal: cfg!(feature = "metal"),
            min_tokens_per_sec: 0.5,
            warmup_on_startup: true,
        }
    }
}

impl InferenceConfig {
    /// The canonical model directory.
    /// Priority: env override > config override > default XDG path
    pub fn model_dir(&self) -> PathBuf {
        // 1. Environment variable
        if let Ok(path) = std::env::var("LILIM_MODEL_DIR") {
            return PathBuf::from(path);
        }
        // 2. Config override
        if let Some(ref p) = self.model_dir_override {
            return p.clone();
        }
        // 3. Default: ~/.local/share/lilim/models/phi-2-q4
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("lilim")
            .join("models")
            .join("phi-2-q4")
    }

    /// Returns a human-readable label for the device being used.
    pub fn device_label(&self) -> &str {
        if self.use_cuda {
            "CUDA"
        } else if self.use_metal {
            "Metal"
        } else {
            "CPU"
        }
    }

    /// Path to the GGUF model weights file.
    pub fn weights_path(&self) -> PathBuf {
        self.model_dir().join("phi-2.Q4_K_M.gguf")
    }

    /// Path to the tokenizer file.
    pub fn tokenizer_path(&self) -> PathBuf {
        self.model_dir().join("tokenizer.json")
    }

    /// Path to the tokenizer config file.
    pub fn tokenizer_config_path(&self) -> PathBuf {
        self.model_dir().join("tokenizer_config.json")
    }
}
