// lilim-runtime: Inference routing handler
//
// This module bridges the Rust inference engine (Candle Phi-2) with the
// Python brain's routing decisions.
//
// Flow for POST /chat:
//   1. Ask Python brain's /route endpoint: "local or remote?"
//   2a. If local + engine available → stream via Candle Phi-2 directly
//   2b. If remote OR local engine unavailable → proxy to Python brain /chat (litellm)
//
// SSE format (same in both cases):
//   data: {"type":"meta","model":"phi-2","source":"local"}
//   data: {"type":"token","text":"Hello"}
//   ...
//   data: {"type":"done"}

use axum::{body::Body, extract::State, response::Response};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::{info, warn};

use crate::AppState;

/// Handle POST /chat by always proxying to the Python brain so the Agent Loop runs.
pub async fn handle_chat_routed(
    State(state): State<Arc<AppState>>,
    body: Value,
) -> Response {
    // ALWAYS route to Python brain so the ReAct loop runs for all requests.
    let url = format!("{}/chat", state.brain_base_url);
    crate::proxy::proxy_sse_stream(&url, &state.http_client, body).await
}

/// Handle POST /internal/generate from the Python brain for local model requests.
pub async fn handle_internal_generate(
    State(state): State<Arc<AppState>>,
    axum::extract::Json(body): axum::extract::Json<Value>,
) -> Response {
    let prompt = body.get("prompt").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let max_tokens = body.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(256) as usize;

    if let Some(ref engine) = state.inference_engine {
        if engine.is_available() {
            info!("Local Phi-2 engine generating {} tokens", max_tokens);
            return stream_local_inference(engine, &prompt, max_tokens).await;
        }
    }
    
    // Return error SSE if engine not available
    let events = format!(
        "data: {}\n\ndata: {}\n\n",
        serde_json::json!({"type": "token", "text": "*Local engine unavailable*"}),
        serde_json::json!({"type": "done", "provider": "LOCAL"})
    );
    sse_response(Body::from(events))
}

/// Ask the Python brain's /route endpoint for a routing decision.
/// Returns None if brain is unreachable (graceful degradation to remote).
#[allow(dead_code)]
async fn get_route_decision(state: &AppState, message: &str, session_id: &str) -> Option<Value> {
    let url = format!("{}/route", state.brain_base_url);
    let body = json!({"message": message, "session_id": session_id});

    match state.http_client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            resp.json::<Value>().await.ok()
        }
        Ok(resp) => {
            warn!("Route endpoint returned {}", resp.status());
            None
        }
        Err(e) => {
            warn!("Could not reach brain /route endpoint: {e}");
            None
        }
    }
}

/// Build a compact prompt for local Phi-2 inference.
///
/// Memory context and enhanced messages from the Python brain are designed
/// for large online models (GPT-4, Claude) that handle long contexts quickly.
/// For local Phi-2 Q4_K_M on CPU, each extra token costs ~0.23s of TTFT:
///   282-token prompt → 64s before first token
///   20-token prompt  → ~5s before first token
///
/// We therefore pass only the raw user message. The Phi-2 instruct format
/// (applied in phi2.rs) already provides enough framing for good answers.
#[allow(dead_code)]
fn build_local_prompt(route: &Option<Value>, original_message: &str) -> String {
    // If the brain provided an enhanced message (with memory context), use it.
    // Otherwise fall back to the original message.
    route.as_ref()
        .and_then(|r| r.get("enhanced_message"))
        .and_then(|v| v.as_str())
        .unwrap_or(original_message)
        .to_string()
}

/// Stream inference from the local Candle Phi-2 engine as SSE.
async fn stream_local_inference(
    engine: &lilim_inference::InferenceEngine,
    prompt: &str,
    max_tokens: usize,
) -> Response {
    // Meta event
    let meta_event = format!(
        "data: {}\n\n",
        serde_json::json!({"type": "meta", "model": "phi-2", "source": "local"})
    );

    let prompt = prompt.to_string();
    let engine_result = engine.generate_stream(&prompt, max_tokens).await;

    match engine_result {
        Err(e) => {
            // Engine failed — return error SSE
            let err_msg = format!(
                "*The local engine encountered an issue: {}. Try adding an online provider in Settings.*",
                e
            );
            let events = format!(
                "{meta_event}data: {}\n\ndata: {}\n\n",
                serde_json::json!({"type": "token", "text": err_msg}),
                serde_json::json!({"type": "done"})
            );
            return sse_response(Body::from(events));
        }
        Ok(token_stream) => {
            // Prepend meta event then stream tokens
            let meta_bytes = meta_event.into_bytes();

            let token_sse = token_stream.map(|result| {
                let text = match result {
                    Ok(t) => t,
                    Err(e) => format!("*inference error: {e}*"),
                };
                let event = format!(
                    "data: {}\n\n",
                    serde_json::json!({"type": "token", "text": text})
                );
                Ok::<_, std::io::Error>(bytes::Bytes::from(event))
            });

            let done_stream = futures_util::stream::once(async {
                let done = format!("data: {}\n\n", serde_json::json!({"type": "done"}));
                Ok::<_, std::io::Error>(bytes::Bytes::from(done))
            });

            let meta_stream = futures_util::stream::once(async move {
                Ok::<_, std::io::Error>(bytes::Bytes::from(meta_bytes))
            });

            let combined = meta_stream.chain(token_sse).chain(done_stream);
            return sse_response(Body::from_stream(combined));
        }
    }
}

fn sse_response(body: Body) -> Response {
    Response::builder()
        .status(200)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("X-Accel-Buffering", "no")
        .body(body)
        .unwrap()
}
