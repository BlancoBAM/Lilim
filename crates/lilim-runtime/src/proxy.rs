// lilim-runtime: HTTP proxy to the Python brain
//
// All AI/chat requests arriving at the Rust gateway are forwarded
// to the Python FastAPI brain server. Responses are streamed back
// as SSE so the desktop UI gets real-time token delivery.
//
// Routes proxied:
//   POST /chat          → brain POST /chat        (SSE stream)
//   POST /chat/sync     → brain POST /chat/sync   (JSON)
//   POST /memory/search → brain POST /memory/search
//   GET  /memory/stats  → brain GET  /memory/stats
//   GET  /memory/context → brain GET /memory/context

use axum::{
    body::Body,
    http::StatusCode,
    response::Response,
};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use tracing::{error, warn};

/// Forward a JSON POST request to the brain and return the JSON response.
pub async fn proxy_json_post(
    brain_url: &str,
    client: &Client,
    body: Value,
) -> Result<Value, (StatusCode, String)> {
    match client.post(brain_url).json(&body).send().await {
        Ok(resp) => {
            let status = resp.status();
            match resp.json::<Value>().await {
                Ok(json) => {
                    if status.is_success() {
                        Ok(json)
                    } else {
                        Err((
                            StatusCode::from_u16(status.as_u16())
                                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                            json.to_string(),
                        ))
                    }
                }
                Err(e) => Err((StatusCode::BAD_GATEWAY, format!("Brain response parse error: {e}"))),
            }
        }
        Err(e) => Err((StatusCode::BAD_GATEWAY, format!("Brain unreachable: {e}"))),
    }
}

/// Forward a GET request with query params to the brain.
pub async fn proxy_json_get(
    brain_url: &str,
    client: &Client,
    params: &HashMap<String, String>,
) -> Result<Value, (StatusCode, String)> {
    match client.get(brain_url).query(params).send().await {
        Ok(resp) => {
            if resp.status().is_success() {
                resp.json::<Value>()
                    .await
                    .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))
            } else {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                Err((
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                    text,
                ))
            }
        }
        Err(e) => Err((StatusCode::BAD_GATEWAY, format!("Brain unreachable: {e}"))),
    }
}

/// Stream a POST response from the brain as SSE.
/// Used for /chat — the brain returns `text/event-stream`.
pub async fn proxy_sse_stream(
    brain_url: &str,
    client: &Client,
    body: Value,
) -> Response {
    let result = client
        .post(brain_url)
        .json(&body)
        .send()
        .await;

    match result {
        Err(e) => {
            error!("Brain SSE request failed: {e}");
            let err_event = format!(
                "data: {{\"type\":\"error\",\"text\":\"Brain connection failed: {}\"}}\n\n",
                e.to_string().replace('"', "'")
            );
            Response::builder()
                .status(200)
                .header("Content-Type", "text/event-stream")
                .header("Cache-Control", "no-cache")
                .header("X-Accel-Buffering", "no")
                .body(Body::from(err_event))
                .unwrap()
        }
        Ok(brain_resp) => {
            let status = brain_resp.status();
            if !status.is_success() {
                let text = brain_resp.text().await.unwrap_or_default();
                warn!("Brain returned error {}: {}", status, text);
                let err_event = format!(
                    "data: {{\"type\":\"error\",\"text\":\"Brain error {}: {}\"}}\n\n",
                    status.as_u16(),
                    text.replace('"', "'")
                );
                return Response::builder()
                    .status(200)
                    .header("Content-Type", "text/event-stream")
                    .body(Body::from(err_event))
                    .unwrap();
            }

            // Pipe the brain's SSE stream directly to the client
            let byte_stream = brain_resp
                .bytes_stream()
                .map(|chunk| {
                    chunk.map_err(|e| {
                        error!("Brain stream error: {e}");
                        std::io::Error::new(std::io::ErrorKind::BrokenPipe, e)
                    })
                });

            Response::builder()
                .status(200)
                .header("Content-Type", "text/event-stream")
                .header("Cache-Control", "no-cache")
                .header("X-Accel-Buffering", "no")
                .body(Body::from_stream(byte_stream))
                .unwrap()
        }
    }
}
