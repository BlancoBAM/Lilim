use axum::{routing::{get, post}, Json, Router};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::sync::Mutex;
use once_cell::sync::Lazy;

static MEMORY: Lazy<Mutex<HashMap<String, String>>> = Lazy::new(|| {
    let mut m: HashMap<String, String> = HashMap::new();
    let path = "/var/lib/lilim/memory.json";
    if let Ok(data) = fs::read_to_string(path) {
        if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&data) {
            m = map;
        }
    }
    Mutex::new(m)
});

#[derive(Deserialize)]
struct Query {
    prompt: String,
    session: Option<String>,
}

#[derive(Serialize)]
struct Answer {
    response: String,
    memory: HashMap<String, String>,
}

async fn health() -> &'static str {
    "OK"
}

async fn query(Json(payload): Json<Query>) -> Json<Answer> {
    // Minimal placeholder logic: echo with a tiny enrichment for demonstration
    let enriched = format!("{}", payload.prompt);

    // Simple in-process memory update for session context
    let mut mem = MEMORY.lock().unwrap();
    if let Some(sess) = &payload.session {
        mem.insert(sess.clone(), payload.prompt.clone());
    }
    // Persist memory to disk
    if let Ok(json) = serde_json::to_string_pretty(&*mem) {
        if let Err(e) = fs::create_dir_all("/var/lib/lilim") {
            eprintln!("Mem dir create error: {e}");
        }
        if let Ok(mut f) = File::create("/var/lib/lilim/memory.json") {
            let _ = f.write_all(json.as_bytes());
        }
    }
    Json(Answer { response: format!("{}", enriched), memory: mem.clone().into_iter().collect() })
}

#[tokio::main]
async fn main() {
    // Initialize simple logger
    std::env::set_var("RUST_LOG", "info");
    let _ = env_logger::builder().is_test(false).try_init();

    // Build our app with routes
    let app = Router::new()
        .route("/health", get(health))
        .route("/query", post(query));

    // Bind address and run
    let addr = "127.0.0.1:8000".parse().unwrap();
    println!("Lilim runtime listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
