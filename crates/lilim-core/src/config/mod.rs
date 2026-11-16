//! Configuration management for Lilim Core

use anyhow::Result;
use directories::ProjectDirs;
use figment::{Figment, providers::{Env, Format, Toml, Yaml}};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration structure
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub server: ServerConfig,
    pub inference: InferenceConfig,
    pub router: RouterConfig,
    pub memory: MemoryConfig,
    pub mcp: McpConfig,
    pub scheduler: SchedulerConfig,
    pub agent: AgentConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
    pub cors_origins: Vec<String>,
    pub request_timeout_secs: u64,
    pub max_request_size_mb: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InferenceConfig {
    pub model_path: PathBuf,
    pub mistral_rs: MistralRsConfig,
    pub context_window: usize,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MistralRsConfig {
    pub isq: String,
    pub paged_attention: bool,
    pub max_seq_len: usize,
    pub num_gpu_layers: i32,
    pub use_flash_attn: bool,
    pub kv_cache_quant: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RouterConfig {
    pub strategy: RoutingStrategy,
    pub providers: Vec<ProviderConfig>,
    pub zero_config_enabled: bool,
    pub fallback_chain: Vec<String>,
    pub budget_limits: BudgetLimits,
    pub complexity_thresholds: ComplexityThresholds,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Copy)]
#[serde(rename_all = "snake_case")]
pub enum RoutingStrategy {
    LocalFirst,
    CloudFirst,
    CostOptimized,
    QualityOptimized,
    Balanced,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProviderConfig {
    pub name: String,
    pub provider_type: ProviderType,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub models: Vec<ModelConfig>,
    pub enabled: bool,
    pub priority: u8,
    pub rate_limit_rpm: Option<u32>,
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderType {
    OpenAI,
    Anthropic,
    Google,
    Cohere,
    Ollama,
    Mistral,
    Groq,
    Together,
    OpenRouter,
    Custom,
}

impl std::str::FromStr for ProviderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(ProviderType::OpenAI),
            "anthropic" => Ok(ProviderType::Anthropic),
            "google" => Ok(ProviderType::Google),
            "cohere" => Ok(ProviderType::Cohere),
            "ollama" => Ok(ProviderType::Ollama),
            "mistral" => Ok(ProviderType::Mistral),
            "groq" => Ok(ProviderType::Groq),
            "together" => Ok(ProviderType::Together),
            "openrouter" => Ok(ProviderType::OpenRouter),
            "custom" => Ok(ProviderType::Custom),
            other => Err(format!("Unknown provider type: {other}")),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    pub name: String,
    pub display_name: String,
    pub context_window: usize,
    pub max_output_tokens: usize,
    pub cost_per_million_input: f64,
    pub cost_per_million_output: f64,
    pub capabilities: Vec<ModelCapability>,
    pub tier: ModelTier,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ModelCapability {
    Chat,
    Completion,
    Embedding,
    Vision,
    FunctionCalling,
    JsonMode,
    Streaming,
}

impl std::str::FromStr for ModelCapability {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "chat" => Ok(ModelCapability::Chat),
            "completion" => Ok(ModelCapability::Completion),
            "embedding" => Ok(ModelCapability::Embedding),
            "vision" => Ok(ModelCapability::Vision),
            "function_calling" => Ok(ModelCapability::FunctionCalling),
            "json_mode" => Ok(ModelCapability::JsonMode),
            "streaming" => Ok(ModelCapability::Streaming),
            other => Err(format!("Unknown model capability: {other}")),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ModelTier {
    Free,
    Budget,
    Standard,
    Premium,
    Local,
}

impl std::str::FromStr for ModelTier {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "free" => Ok(ModelTier::Free),
            "budget" => Ok(ModelTier::Budget),
            "standard" => Ok(ModelTier::Standard),
            "premium" => Ok(ModelTier::Premium),
            "local" => Ok(ModelTier::Local),
            other => Err(format!("Unknown model tier: {other}")),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BudgetLimits {
    pub daily_usd: f64,
    pub monthly_usd: f64,
    pub per_request_usd: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ComplexityThresholds {
    pub simple_max_tokens: usize,
    pub medium_max_tokens: usize,
    pub complex_max_tokens: usize,
    pub embedding_model: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MemoryConfig {
    pub sqlite_path: PathBuf,
    pub vault_path: PathBuf,
    pub qdrant: QdrantConfig,
    pub embedding_model: String,
    pub embedding_dim: usize,
    pub max_context_tokens: usize,
    pub compression_enabled: bool,
    pub compression_ratio: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QdrantConfig {
    pub host: String,
    pub port: u16,
    pub collection_name: String,
    pub vector_size: usize,
    pub distance: String,
    pub use_grpc: bool,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct McpConfig {
    pub config_path: PathBuf,
    pub max_servers: usize,
    pub max_tool_output: usize,
    pub connection_timeout_secs: u64,
    pub call_timeout_secs: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SchedulerConfig {
    pub db_path: PathBuf,
    pub use_systemd: bool,
    pub notification_command: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct AgentConfig {
    pub system_prompt: String,
    pub max_iterations: usize,
    pub tool_approval_required: Vec<String>,
    pub enable_compression: bool,
    pub compression_threshold_tokens: usize,
}

impl Config {
    /// Load configuration from standard locations
    pub fn load() -> Result<Self> {
        let project_dirs = ProjectDirs::from("org", "lilim", "lilim")
            .ok_or_else(|| anyhow::anyhow!("Could not determine config directory"))?;

        let config_dir = project_dirs.config_dir();
        std::fs::create_dir_all(config_dir)?;

        let config_file = config_dir.join("config.yaml");
        let routing_file = config_dir.join("routing.toml");

        // Start with defaults
        let mut figment = Figment::new()
            .merge(Yaml::file(&config_file))
            .merge(Toml::file(&routing_file))
            .merge(Env::prefixed("LILIM_"));

        // If config files don't exist, create them with defaults
        if !config_file.exists() {
            let default_config = Self::default();
            let yaml = serde_yaml::to_string(&default_config)?;
            std::fs::write(&config_file, yaml)?;
        }

        if !routing_file.exists() {
            let default_routing = RouterConfig::default_routing();
            let toml = toml::to_string_pretty(&default_routing)?;
            std::fs::write(&routing_file, toml)?;
        }

        figment = figment
            .merge(Yaml::file(&config_file))
            .merge(Toml::file(&routing_file));

        Ok(figment.extract()?)
    }

    /// Get the config directory path
    pub fn config_dir() -> PathBuf {
        ProjectDirs::from("org", "lilim", "lilim")
            .map(|d| d.config_dir().to_path_buf())
            .unwrap_or_else(|| PathBuf::from("~/.config/lilim"))
    }

    /// Get the data directory path
    pub fn data_dir() -> PathBuf {
        ProjectDirs::from("org", "lilim", "lilim")
            .map(|d| d.data_dir().to_path_buf())
            .unwrap_or_else(|| PathBuf::from("~/.local/share/lilim"))
    }
}

impl Default for Config {
    fn default() -> Self {
        let data_dir = Self::data_dir();
        let config_dir = Self::config_dir();

        Self {
            server: ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 8080,
                workers: 4,
                cors_origins: vec!["http://localhost:1420".to_string(), "tauri://localhost".to_string()],
                request_timeout_secs: 120,
                max_request_size_mb: 50,
            },
            inference: InferenceConfig {
                model_path: data_dir.join("models/phi-4-mini-q4_k_m.gguf"),
                mistral_rs: MistralRsConfig {
                    isq: "Q4_K_M".to_string(),
                    paged_attention: true,
                    max_seq_len: 8192,
                    num_gpu_layers: -1,
                    use_flash_attn: true,
                    kv_cache_quant: Some("Q4_0".to_string()),
                },
                context_window: 8192,
                max_tokens: 4096,
                temperature: 0.7,
                top_p: 0.9,
                repeat_penalty: 1.1,
            },
            router: RouterConfig::default(),
            memory: MemoryConfig {
                sqlite_path: data_dir.join("memory.db"),
                vault_path: data_dir.join("vault"),
                qdrant: QdrantConfig {
                    host: "127.0.0.1".to_string(),
                    port: 6334,
                    collection_name: "lilim_memory".to_string(),
                    vector_size: 384,
                    distance: "Cosine".to_string(),
                    use_grpc: true,
                    api_key: None,
                },
                embedding_model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                embedding_dim: 384,
                max_context_tokens: 4096,
                compression_enabled: true,
                compression_ratio: 0.5,
            },
            mcp: McpConfig {
                config_path: config_dir.join("mcp-servers.json"),
                max_servers: 8,
                max_tool_output: 12000,
                connection_timeout_secs: 10,
                call_timeout_secs: 90,
            },
            scheduler: SchedulerConfig {
                db_path: data_dir.join("schedules.db"),
                use_systemd: true,
                notification_command: None,
            },
            agent: AgentConfig {
                system_prompt: "You are Lilim, a helpful local-first AI assistant. You have access to tools for file operations, shell execution, memory, and scheduling. Use them to help the user.".to_string(),
                max_iterations: 10,
                tool_approval_required: vec![
                    "file_write".to_string(),
                    "file_delete".to_string(),
                    "shell_exec".to_string(),
                    "web_search".to_string(),
                    "code_exec".to_string(),
                ],
                enable_compression: true,
                compression_threshold_tokens: 2000,
            },
        }
    }
}

impl RouterConfig {
    fn default() -> Self {
        Self {
            strategy: RoutingStrategy::Balanced,
            providers: vec![],
            zero_config_enabled: true,
            fallback_chain: vec!["local".to_string(), "pollinations".to_string(), "opencode".to_string()],
            budget_limits: BudgetLimits {
                daily_usd: 5.0,
                monthly_usd: 50.0,
                per_request_usd: 0.50,
            },
            complexity_thresholds: ComplexityThresholds {
                simple_max_tokens: 512,
                medium_max_tokens: 2048,
                complex_max_tokens: 8192,
                embedding_model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            },
        }
    }

    fn default_routing() -> serde_json::Value {
        serde_json::json!({
            "strategy": "balanced",
            "tiers": {
                "free": ["pollinations", "opencode", "fx", "aitopia", "powerbrain"],
                "budget": ["groq", "together"],
                "standard": ["openai", "anthropic", "google", "cohere", "mistral"],
                "premium": ["gpt-4", "claude-3-opus", "gemini-ultra"],
                "local": ["phi-4-mini", "mistral-7b", "llama-3.1-8b"]
            },
            "categories": {
                "general": ["free", "budget", "standard"],
                "coding": ["standard", "premium", "local"],
                "reasoning": ["premium", "standard"],
                "creative": ["premium", "standard"],
                "analysis": ["standard", "premium"],
                "quick": ["free", "budget", "local"]
            },
            "complexity_thresholds": {
                "simple": 512,
                "medium": 2048,
                "complex": 8192
            }
        })
    }
}