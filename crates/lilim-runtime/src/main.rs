// lilim-runtime: main Axum server
//
// The Rust gateway for Lilim AI assistant.
//
// Responsibilities:
//   1. Spawn and supervise the Python brain (lilim_core/server.py)
//   2. Serve the Tauri desktop UI via HTTP on :8080
//   3. Proxy /chat requests to the Python brain (SSE streaming)
//   4. Execute system tools with safety enforcement
//   5. Handle task scheduling
//   6. CORS for Tauri WebView origin
//
// All AI/LLM logic lives in Python. Rust handles: process management,
// security enforcement, HTTP serving, tool execution, and scheduling.

mod brain;
mod config;
mod inference;
mod proxy;
mod scheduler;
mod tools;


use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{Query, State},
    http::{Method, StatusCode},
    response::{Json, Response},
    routing::{delete, get, post},
    Router,
};
use reqwest::Client;
use serde_json::{json, Value};
use tokio::signal;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{error, info};

// ── Shared application state ──────────────────────────────────

pub struct AppState {
    pub brain_base_url: String,
    pub http_client: reqwest::Client,
    /// Local Candle Phi-2 inference engine (None if model not available)
    pub inference_engine: Option<lilim_inference::InferenceEngine>,
}

// ── Entry point ───────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Structured logging
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "lilim_runtime=info,tower_http=warn".into()),
        )
        .compact()
        .init();

    info!("══════════════════════════════════════");
    info!("  Lilim Runtime v{}", env!("CARGO_PKG_VERSION"));
    info!("══════════════════════════════════════");

    // Load config
    let mut cfg = config::load()?;

    // Override port from CLI if present: --port <N>
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--port" && i + 1 < args.len() {
            if let Ok(p) = args[i + 1].parse::<u16>() {
                cfg.server.port = p;
            }
        }
    }

    let bind_addr = format!("{}:{}", cfg.server.host, cfg.server.port);
    let brain_port = cfg.brain.port;
    let brain_base_url = format!("http://{}:{}", cfg.brain.host, brain_port);

    // Spawn the Python brain
    info!("Starting Python brain on port {brain_port}…");
    let mut brain = brain::BrainProcess::new(brain_port);
    if let Err(e) = brain.start() {
        error!("Failed to start Python brain: {e}");
        error!("Continuing without brain — only system tools will work.");
        error!("Install FastAPI: pip install fastapi uvicorn litellm");
    }

    // Initialize local inference engine (Phi-2 via Candle)
    info!("Initializing local Phi-2 inference engine…");
    let inference_config = lilim_inference::InferenceConfig::default();
    let inference_engine = lilim_inference::InferenceEngine::new(inference_config).await;
    let engine_available = inference_engine.is_available();
    if engine_available {
        info!("Local Phi-2 engine ready ✓");
    } else {
        info!("Local engine unavailable — all requests will route to online providers");
    }

    // Build shared HTTP client (for proxying to brain)
    let http_client = Client::builder()
        .timeout(Duration::from_secs(120)) // LLM calls can be slow
        .build()?;

    let state = Arc::new(AppState {
        brain_base_url: brain_base_url.clone(),
        http_client,
        inference_engine: if engine_available { Some(inference_engine) } else { None },
    });

    // CORS — allow Tauri WebView and local dev origins
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
        .allow_headers(Any);

    // Router
    let app = Router::new()
        // ── Health ──────────────────────────────────────────────────
        .route("/health", get(handle_health))

        // ── Chat (smart local/remote routing) ─────────────────────
        .route("/chat", post(handle_chat))
        .route("/chat/sync", post(handle_chat_sync))
        .route("/internal/generate", post(inference::handle_internal_generate))

        // ── Model status (for Settings panel) ───────────────────
        .route("/model/status", get(handle_model_status))
        .route("/providers/status", get(handle_providers_status))

        // ── Memory (proxied to Python brain) ──────────────────
        .route("/memory/search", post(handle_memory_search))
        .route("/memory/stats", get(handle_memory_stats))
        .route("/memory/context", get(handle_memory_context))

        // ── System tools (Rust-native) ────────────────────────
        .route("/tools/shell", post(tools::handle_shell))
        .route("/tools/file", get(tools::handle_file_read))
        .route("/system/info", get(tools::handle_system_info))

        // ── Scheduling (proxied to Python brain) ──────────────
        .route("/schedule/once", post(scheduler::handle_schedule_once))
        .route("/schedule/recurring", post(scheduler::handle_schedule_recurring))
        .route("/schedule/list", get(scheduler::handle_schedule_list))
        .route("/schedule/:id", delete(scheduler::handle_schedule_cancel))

        // ── Settings (write config, proxied to brain) ─────────
        .route("/settings/model-config", post(handle_settings_model_config))

        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    info!("Gateway listening on http://{bind_addr}");
    info!("Brain proxied at {brain_base_url}");
    info!("Press Ctrl+C to stop.");

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Lilim Runtime stopped.");
    drop(brain); // Triggers BrainProcess::drop → stops Python process
    Ok(())
}

// ── Shutdown signal ───────────────────────────────────────────

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c().await.expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("Received Ctrl+C"),
        _ = terminate => info!("Received SIGTERM"),
    }
}

// ── Health endpoint ───────────────────────────────────────────

async fn handle_health(State(state): State<Arc<AppState>>) -> Json<Value> {
    let brain_ok = state
        .http_client
        .get(format!("{}/health", state.brain_base_url))
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "brain": if brain_ok { "healthy" } else { "unreachable" },
        "ts": chrono::Utc::now().to_rfc3339(),
    }))
}

// ── Chat endpoints ───────────────────────────────────────

async fn handle_chat(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Response {
    // Smart routing: local Phi-2 or remote via Python brain + FreeRouter
    inference::handle_chat_routed(State(state), body).await
}

async fn handle_chat_sync(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let url = format!("{}/chat/sync", state.brain_base_url);
    match proxy::proxy_json_post(&url, &state.http_client, body).await {
        Ok(v) => (StatusCode::OK, Json(v)),
        Err((code, msg)) => (code, Json(json!({"error": msg}))),
    }
}

// ── Model / Provider status endpoints ─────────────────────────

async fn handle_model_status(State(state): State<Arc<AppState>>) -> Json<Value> {
    let cfg = lilim_inference::InferenceConfig::default();
    let status = lilim_inference::downloader::model_status(&cfg);
    Json(json!({
        "local_engine": {
            "available": state.inference_engine.is_some(),
            "model": "phi-2",
            "device": state.inference_engine
                .as_ref()
                .map(|e| e.device_label())
                .unwrap_or("none"),
            "model_status": status,
        }
    }))
}

async fn handle_providers_status(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<Value>) {
    let url = format!("{}/providers/status", state.brain_base_url);
    match proxy::proxy_json_get(&url, &state.http_client, &Default::default()).await {
        Ok(v) => (StatusCode::OK, Json(v)),
        Err((code, msg)) => (code, Json(json!({"error": msg}))),
    }
}

// ── Memory endpoints ──────────────────────────────────────────

async fn handle_memory_search(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let url = format!("{}/memory/search", state.brain_base_url);
    match proxy::proxy_json_post(&url, &state.http_client, body).await {
        Ok(v) => (StatusCode::OK, Json(v)),
        Err((code, msg)) => (code, Json(json!({"error": msg}))),
    }
}

async fn handle_memory_stats(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<Value>) {
    let url = format!("{}/memory/stats", state.brain_base_url);
    match proxy::proxy_json_get(&url, &state.http_client, &Default::default()).await {
        Ok(v) => (StatusCode::OK, Json(v)),
        Err((code, msg)) => (code, Json(json!({"error": msg}))),
    }
}

async fn handle_memory_context(
    State(state): State<Arc<AppState>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> (StatusCode, Json<Value>) {
    let url = format!("{}/memory/context", state.brain_base_url);
    match proxy::proxy_json_get(&url, &state.http_client, &params).await {
        Ok(v) => (StatusCode::OK, Json(v)),
        Err((code, msg)) => (code, Json(json!({"error": msg}))),
    }
}

// ── Settings endpoints ────────────────────────────────────────

async fn handle_settings_model_config(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    // Write to the user config directory
    let config_path = dirs::config_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("lilim")
        .join("model-config.json");

    if let Some(parent) = config_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    match serde_json::to_string_pretty(&body) {
        Ok(json_str) => {
            if let Err(e) = std::fs::write(&config_path, &json_str) {
                error!("Failed to write model config: {e}");
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()})));
            }
            info!("Model config updated at {}", config_path.display());
        }
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({"error": e.to_string()}))),
    }

    // Also forward to brain so it reloads
    let url = format!("{}/settings/model-config", state.brain_base_url);
    match proxy::proxy_json_post(&url, &state.http_client, body).await {
        Ok(v) => (StatusCode::OK, Json(v)),
        Err(_) => (StatusCode::OK, Json(json!({"status": "saved", "brain_reload": "pending"}))),
    }
}
