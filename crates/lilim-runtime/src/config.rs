// lilim-runtime: configuration loader
//
// Reads lilim.yaml from (in order):
//   1. /etc/lilith/lilim.yaml
//   2. ~/.config/lilim/lilim.yaml
//   3. ./config/lilim.yaml  (dev mode)

use anyhow::Result;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tracing::info;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct LilimConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub brain: BrainConfig,
    #[serde(default)]
    #[allow(dead_code)] // Reserved: used for security enforcement in future tool-execution hardening
    pub security: SecurityConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    /// Host for the Rust gateway (UI talks to this)
    pub host: String,
    /// Port for the Rust gateway
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8080,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct BrainConfig {
    /// Host where the Python FastAPI brain runs
    pub host: String,
    /// Port the Python brain listens on
    pub port: u16,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8081,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SecurityConfig {
    /// Whether shell commands require explicit UI confirmation
    #[allow(dead_code)] // Read by tools.rs in future — intentionally kept
    pub require_shell_confirmation: bool,
    /// Max response size in bytes from brain (safety cap)
    #[allow(dead_code)] // Read by proxy.rs in future — intentionally kept
    pub max_response_bytes: usize,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            require_shell_confirmation: true,
            max_response_bytes: 1_048_576, // 1 MB
        }
    }
}

/// Candidate config file paths in priority order.
fn config_candidates() -> Vec<PathBuf> {
    let mut paths = vec![
        PathBuf::from("/etc/lilith/lilim.yaml"),
    ];

    if let Some(config_dir) = dirs::config_dir() {
        paths.push(config_dir.join("lilim").join("lilim.yaml"));
    }

    // Dev-mode: look in repo root config/
    paths.push(PathBuf::from("config/lilim.yaml"));

    paths
}

pub fn load() -> Result<LilimConfig> {
    for path in config_candidates() {
        if path.exists() {
            return load_from(&path);
        }
    }

    info!("No config file found; using built-in defaults");
    Ok(LilimConfig::default())
}

fn load_from(path: &Path) -> Result<LilimConfig> {
    let text = std::fs::read_to_string(path)?;
    let cfg: LilimConfig = serde_yaml::from_str(&text)
        .unwrap_or_else(|e| {
            tracing::warn!("Could not parse {}: {}; using defaults", path.display(), e);
            LilimConfig::default()
        });
    info!("Loaded config from {}", path.display());
    Ok(cfg)
}
