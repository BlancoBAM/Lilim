// lilim-inference: Configuration
//
// InferenceConfig controls where the model lives and what hardware to use.
// Default model: microsoft/Phi-3.5-mini-instruct (GGUF Q4_K_M, ~2.4 GB on disk)

use std::path::PathBuf;

/// The default Lilim system prompt for the Phi-3.5-mini agent.
///
/// Kept concise (~80 tokens) to minimise TTFT on CPU while still establishing
/// the assistant persona and tool-calling behaviour.
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are Lilim, an intelligent AI assistant running locally on the user's Linux system. \
You can execute shell commands, read and write files, and take autonomous action to \
complete tasks. When asked to perform an action, do it — don't just describe how. \
For commands requiring elevated privileges, ask the user to confirm, then execute. \
Be concise, helpful, and prefer direct action over lengthy explanation.";

/// Configuration for the local inference engine.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Directory containing the model weights and tokenizer.
    /// Default: ~/.local/share/lilim/models/phi-3.5-mini-q4
    pub model_dir_override: Option<PathBuf>,

    /// Maximum context window size in tokens.
    /// Phi-3.5-mini supports 128K, but we cap at 4096 for CPU practicality.
    pub context_size: usize,

    /// Default max tokens to generate for local inference.
    pub max_gen_tokens: usize,

    /// Temperature for sampling (0.0 = greedy, 1.0 = random).
    pub temperature: f64,

    /// Top-p nucleus sampling cutoff.
    pub top_p: f64,

    /// Whether to use CUDA if available.
    pub use_cuda: bool,

    /// Whether to use Metal (Apple Silicon) if available.
    pub use_metal: bool,

    /// Minimum tokens/second before we log a warning about speed.
    /// Set to 0.0 to disable speed-based fallback warnings.
    pub min_tokens_per_sec: f64,

    /// Run a single-token warmup pass after model load to pre-warm CPU caches.
    pub warmup_on_startup: bool,

    /// System prompt injected at the start of every conversation.
    pub system_prompt: String,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_dir_override: None,
            context_size: 4096,
            max_gen_tokens: 512,
            temperature: 0.4,
            top_p: 0.9,
            use_cuda: cfg!(feature = "cuda"),
            use_metal: cfg!(feature = "metal"),
            min_tokens_per_sec: 0.5,
            warmup_on_startup: true,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
        }
    }
}

impl InferenceConfig {
    /// The canonical model directory.
    /// Priority: env var > config override > XDG default
    pub fn model_dir(&self) -> PathBuf {
        if let Ok(path) = std::env::var("LILIM_MODEL_DIR") {
            return PathBuf::from(path);
        }
        if let Some(ref p) = self.model_dir_override {
            return p.clone();
        }
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("lilim")
            .join("models")
            .join("phi-3.5-mini-q4")
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
        self.model_dir().join("Phi-3.5-mini-instruct-Q4_K_M.gguf")
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
