use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LilimConfig {
    pub models: ModelsConfig,
    pub providers: ProvidersConfig,
    pub routing: RoutingConfig,
    pub rag: RagConfig,
    pub memory: MemoryConfig,
    pub agent: AgentConfig,
    pub server: ServerConfig,
    pub personality: PersonalityConfig,
    pub logging: LoggingConfig,
    pub performance: PerformanceConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelsConfig {
    pub local: LocalModelsConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LocalModelsConfig {
    pub router: LocalModelConfig,
    pub main: LocalModelConfig,
    pub tts: TTSConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LocalModelConfig {
    pub enabled: bool,
    pub model_path: PathBuf,
    pub model_type: String, // "gguf", "safetensors", "onnx"
    pub timeout_ms: u64,
    pub max_tokens: usize,
    pub temperature: f32,
    #[serde(default)]
    pub ctx_size: Option<usize>,
    #[serde(default)]
    pub gpu_layers: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TTSConfig {
    pub enabled: bool,
    pub engine: String, // "piper", "coqui", "espeak"
    pub model_path: PathBuf,
    pub voice: String,
    pub speed: f32,
    pub pitch: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProvidersConfig {
    pub online: Vec<ProviderConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProviderConfig {
    pub name: String,
    pub enabled: bool,
    pub api_url: String,
    pub auth_type: String, // "bearer", "api-key", "custom", "none"
    #[serde(default)]
    pub api_key_env: Option<String>,
    pub model: String,
    pub priority: u8,
    pub timeout_s: u64,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub custom_headers: HashMap<String, String>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RoutingConfig {
    pub complexity_threshold: ComplexityThresholds,
    pub max_local_latency_ms: u64,
    pub prefer_local: bool,
    pub task_routes: HashMap<String, String>, // task_type -> "local"/"online"/"tts"
    pub fallback_triggers: Vec<HashMap<String, bool>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ComplexityThresholds {
    pub trivial: f32,
    pub simple: f32,
    pub moderate: f32,
    pub complex: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RagConfig {
    pub vector_index: PathBuf,
    pub keyword_index: PathBuf,
    pub max_results: usize,
    pub min_similarity: f32,
    pub high_relevance_threshold: f32,
    pub use_rag_for_routing: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MemoryConfig {
    pub db_path: PathBuf,
    pub max_history_messages: usize,
    pub session_timeout: u64,
    pub context_strategy: String, // "sliding", "summarize", "truncate"
    pub max_context_tokens: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AgentConfig {
    pub enable_web_search: bool,
    pub enable_file_search: bool,
    pub enable_terminal: bool,
    pub yolo_mode: bool,
    pub allowed_commands: Vec<String>,
    pub max_web_results: usize,
    pub max_search_depth: usize,
    pub tool_timeout_s: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub cors_origins: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PersonalityConfig {
    pub responses_yaml_path: PathBuf,
    pub theme_intensity: f32,
    pub sarcasm_level: f32,
    pub always_apply: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggingConfig {
    pub level: String,
    pub file: PathBuf,
    pub log_queries: bool,
    pub log_responses: bool,
    pub log_routing_decisions: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PerformanceConfig {
    pub query_timeout: u64,
    pub rag_timeout: u64,
    pub max_concurrent_queries: usize,
    pub enable_cache: bool,
    pub cache_ttl: u64,
    pub warmup_models: bool,
    pub warmup_timeout_s: u64,
}

impl LilimConfig {
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: LilimConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Get enabled online providers sorted by priority
    pub fn get_enabled_providers(&self) -> Vec<&ProviderConfig> {
        let mut providers: Vec<&ProviderConfig> = self
            .providers
            .online
            .iter()
            .filter(|p| p.enabled)
            .collect();
        
        providers.sort_by_key(|p| p.priority);
        providers
    }

    /// Get API key for a provider from environment
    pub fn get_provider_api_key(&self, provider: &ProviderConfig) -> Option<String> {
        provider
            .api_key_env
            .as_ref()
            .and_then(|env_var| std::env::var(env_var).ok())
    }
}
