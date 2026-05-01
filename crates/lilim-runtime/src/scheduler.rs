// lilim-runtime: scheduling endpoints
//
// Delegates scheduling to the Python brain's scheduler module via HTTP.
// The Rust layer validates the request shape and forwards cleanly.
//
// Endpoints:
//   POST /schedule/once       — one-time reminder
//   POST /schedule/recurring  — recurring reminder
//   GET  /schedule/list       — list all active schedules
//   DELETE /schedule/:id      — cancel a schedule

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::AppState;
use crate::proxy::proxy_json_post;

#[derive(Debug, Deserialize)]
pub struct ScheduleOnceRequest {
    pub message: String,
    pub when: String,   // e.g. "in 30 minutes"
}

#[derive(Debug, Deserialize)]
pub struct ScheduleRecurringRequest {
    pub message: String,
    pub when: String,   // e.g. "every day at 9am"
}

pub async fn handle_schedule_once(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ScheduleOnceRequest>,
) -> (StatusCode, Json<Value>) {
    if req.message.trim().is_empty() || req.when.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "message and when are required"})),
        );
    }

    let url = format!("{}/schedule/once", state.brain_base_url);
    let body = json!({ "message": req.message, "natural_time": req.when });

    match proxy_json_post(&url, &state.http_client, body).await {
        Ok(v) => (StatusCode::OK, Json(v)),
        Err((code, msg)) => (code, Json(json!({"error": msg}))),
    }
}

pub async fn handle_schedule_recurring(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ScheduleRecurringRequest>,
) -> (StatusCode, Json<Value>) {
    if req.message.trim().is_empty() || req.when.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "message and when are required"})),
        );
    }

    let url = format!("{}/schedule/recurring", state.brain_base_url);
    let body = json!({ "message": req.message, "natural_time": req.when });

    match proxy_json_post(&url, &state.http_client, body).await {
        Ok(v) => (StatusCode::OK, Json(v)),
        Err((code, msg)) => (code, Json(json!({"error": msg}))),
    }
}

pub async fn handle_schedule_list(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<Value>) {
    let url = format!("{}/schedule/list", state.brain_base_url);
    match crate::proxy::proxy_json_get(&url, &state.http_client, &Default::default()).await {
        Ok(v) => (StatusCode::OK, Json(v)),
        Err((code, msg)) => (code, Json(json!({"error": msg}))),
    }
}

pub async fn handle_schedule_cancel(
    State(state): State<Arc<AppState>>,
    Path(id): Path<u64>,
) -> (StatusCode, Json<Value>) {
    // Forward as a POST to the brain's cancel endpoint
    let url = format!("{}/schedule/{}/cancel", state.brain_base_url, id);
    match state.http_client.post(&url).send().await {
        Ok(resp) => {
            if resp.status().is_success() {
                (StatusCode::OK, Json(json!({"id": id, "cancelled": true})))
            } else {
                (StatusCode::BAD_GATEWAY, Json(json!({"error": "cancel failed"})))
            }
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": format!("Brain unreachable: {e}")})),
        ),
    }
}
