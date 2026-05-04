// lilim-inference: Model Downloader
//
// Verifies model files are present and downloads them from HuggingFace
// if not. For packaging into the Lilith Linux .deb, the model is bundled
// at /usr/lib/lilim/models/phi-2-q4/ and this code uses that path first.

use anyhow::{Context, Result};
use std::path::Path;
use tracing::{info, warn};

use crate::InferenceConfig;

/// Bundled model path (installed by the .deb package).
/// When this path exists, we never download anything.
const BUNDLED_MODEL_DIR: &str = "/usr/lib/lilim/models/phi-2-q4";

/// HuggingFace model repository for Phi-2 GGUF weights.
/// We use TheBloke's Q4_K_M quantization — best quality/size tradeoff.
const HF_GGUF_REPO_ID: &str = "TheBloke/phi-2-GGUF";
const HF_GGUF_FILENAME: &str = "phi-2.Q4_K_M.gguf";

const HF_TOKENIZER_REPO_ID: &str = "microsoft/phi-2";
const HF_TOKENIZER_FILES: &[&str] = &["tokenizer.json", "tokenizer_config.json"];

/// Ensure all required model files are present.
/// Priority:
///   1. /usr/lib/lilim/models/phi-2-q4/ (bundled in .deb — preferred)
///   2. LILIM_MODEL_DIR env var
///   3. ~/.local/share/lilim/models/phi-2-q4/ (downloaded on first run)
pub async fn ensure_model_ready(config: &InferenceConfig) -> Result<()> {
    // Check bundled path first (fastest, offline, no download needed)
    let bundled = Path::new(BUNDLED_MODEL_DIR);
    if bundled.exists() && has_required_files(bundled) {
        info!("Using bundled model at {}", BUNDLED_MODEL_DIR);
        return Ok(());
    }

    // Check configured model dir
    let model_dir = config.model_dir();
    if model_dir.exists() && has_required_files(&model_dir) {
        info!("Using model at {}", model_dir.display());
        return Ok(());
    }

    // Need to download
    info!("Model not found locally. Downloading Phi-2 from HuggingFace…");
    info!("Target: {}", model_dir.display());
    info!("This is a one-time ~1.7GB download.");

    download_model(&model_dir).await
        .context("Failed to download Phi-2 model from HuggingFace")?;

    info!("Model downloaded successfully ✓");
    Ok(())
}

/// Check if the model directory has all required files.
pub fn has_required_files(dir: &Path) -> bool {
    dir.join(HF_GGUF_FILENAME).exists()
        && dir.join("tokenizer.json").exists()
}

/// Get the actual model directory to use (bundled takes priority).
pub fn get_model_dir(config: &InferenceConfig) -> std::path::PathBuf {
    let bundled = Path::new(BUNDLED_MODEL_DIR);
    if bundled.exists() && has_required_files(bundled) {
        return bundled.to_path_buf();
    }
    config.model_dir()
}

/// Download model files from HuggingFace Hub.
async fn download_model(target_dir: &Path) -> Result<()> {
    use hf_hub::api::tokio::ApiBuilder;

    tokio::fs::create_dir_all(target_dir).await
        .context("Failed to create model directory")?;

    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .context("Failed to initialize HuggingFace API client")?;

    let gguf_repo = api.model(HF_GGUF_REPO_ID.to_string());

    // Download GGUF weights
    info!("Downloading {} from {}…", HF_GGUF_FILENAME, HF_GGUF_REPO_ID);
    let gguf_path = gguf_repo.get(HF_GGUF_FILENAME).await
        .context(format!("Failed to download {HF_GGUF_FILENAME}"))?;

    // Copy to target directory
    let target_gguf = target_dir.join(HF_GGUF_FILENAME);
    if gguf_path != target_gguf {
        tokio::fs::copy(&gguf_path, &target_gguf).await
            .context("Failed to copy GGUF weights to model dir")?;
    }

    // Download tokenizer files using reqwest (hf-hub fails on microsoft/phi-2 relative redirects)
    let client = reqwest::Client::new();
    for filename in HF_TOKENIZER_FILES {
        info!("Downloading {} from {}…", filename, HF_TOKENIZER_REPO_ID);
        let url = format!("https://huggingface.co/{HF_TOKENIZER_REPO_ID}/resolve/main/{filename}");
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                let target = target_dir.join(filename);
                match resp.bytes().await {
                    Ok(bytes) => {
                        if let Err(e) = tokio::fs::write(&target, &bytes).await {
                            warn!("Could not write {filename}: {e}");
                        }
                    }
                    Err(e) => warn!("Could not download {filename} content: {e}"),
                }
            }
            Ok(resp) => {
                warn!("Could not download {filename}: HTTP {}", resp.status());
            }
            Err(e) => {
                warn!("Could not download {filename}: {e} (non-fatal)");
            }
        }
    }

    Ok(())
}

/// Returns download status for the settings panel / health check.
pub fn model_status(config: &InferenceConfig) -> ModelStatus {
    let bundled = Path::new(BUNDLED_MODEL_DIR);
    if bundled.exists() && has_required_files(bundled) {
        return ModelStatus {
            available: true,
            location: BUNDLED_MODEL_DIR.to_string(),
            source: "bundled".to_string(),
            size_mb: estimate_dir_size_mb(bundled),
        };
    }

    let dir = config.model_dir();
    let available = dir.exists() && has_required_files(&dir);
    ModelStatus {
        available,
        location: dir.display().to_string(),
        source: if available { "downloaded".to_string() } else { "missing".to_string() },
        size_mb: if available { estimate_dir_size_mb(&dir) } else { 0 },
    }
}

#[derive(Debug, serde::Serialize)]
pub struct ModelStatus {
    pub available: bool,
    pub location: String,
    pub source: String,
    pub size_mb: u64,
}

fn estimate_dir_size_mb(dir: &Path) -> u64 {
    let mut total: u64 = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total / (1024 * 1024)
}
