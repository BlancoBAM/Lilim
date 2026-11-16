// lilim-runtime: Main Axum server using lilim-core
//
// The Rust gateway for Lilim AI assistant - now fully powered by lilim-core
// (no Python brain needed).

use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{Path as AxumPath, Query, State},
    http::{Method, StatusCode},
    response::{Json, Response},
    routing::{delete, get, post},
    Router,
};
use lilim_core::{config::Config, LilimCore};
use serde_json::{json, Value};
use tokio::signal;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{error, info};

// ── Shared application state ──────────────────────────────────

pub struct AppState {
    pub core: Arc<LilimCore>,
    pub config: Config,
}

// ── Entry point ───────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Structured logging
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "lilim_runtime=info,lilim_core=info,tower_http=warn".into()),
        )
        .compact()
        .init();

    info!("══════════════════════════════════════");
    info!("  Lilim Runtime v{} (pure Rust)", env!("CARGO_PKG_VERSION"));
    info!("══════════════════════════════════════");

    // Load config
    let mut cfg = Config::load()?;

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

    // Initialize lilim-core (replaces Python brain)
    info!("Initializing Lilim Core…");
    let core = Arc::new(LilimCore::new(cfg.clone()).await?);
    core.start().await?;

    info!("Lilim Core initialized ✓");

    let state = Arc::new(AppState {
        core: core.clone(),
        config: cfg.clone(),
    });

    // CORS — allow Tauri WebView and local dev origins
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
        .allow_headers(Any);

    // Router - using lilim-core's gateway (state already baked in)
    let app = core
        .gateway
        .router()
        .layer(cors)
        .layer(TraceLayer::new_for_http());
    let _state = state;
    info!("Gateway listening on http://{bind_addr}");
    info!("Press Ctrl+C to stop.");

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(core.clone()))
        .await?;

    info!("Lilim Runtime stopped.");
    Ok(())
}

// ── Shutdown signal ───────────────────────────────────────────

async fn shutdown_signal(core: Arc<LilimCore>) {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
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

    // Graceful shutdown
    if let Err(e) = core.shutdown().await {
        error!("Error during shutdown: {}", e);
    }
}