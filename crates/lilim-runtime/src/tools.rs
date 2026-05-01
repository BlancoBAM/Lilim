// lilim-runtime: system tool execution endpoints
//
// Exposes safe system tool operations to the desktop UI.
// All shell commands MUST have confirmed=true (set by UI after user clicks "Run it").
//
// Endpoints:
//   POST /tools/shell     — run a shell command (requires confirmed flag)
//   GET  /tools/file      — read a file's contents
//   GET  /tools/list      — list directory contents
//   GET  /system/info     — OS, disk, memory snapshot
//   GET  /system/service  — systemctl status for a service

use axum::{
    extract::Query,
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::process::Command;
use tokio::process::Command as TokioCommand;
use tracing::{info, warn};

// ── Request / Response types ──────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ShellRequest {
    pub command: String,
    /// Must be true — set by UI after user explicitly confirms
    pub confirmed: bool,
}

#[derive(Debug, Serialize)]
pub struct ShellResponse {
    pub command: String,
    pub stdout: String,
    pub stderr: String,
    pub returncode: i32,
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct FileQuery {
    pub path: String,
    pub max_chars: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct FileResponse {
    pub path: String,
    pub content: String,
    pub truncated: bool,
    pub size_bytes: Option<u64>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SystemInfo {
    pub os: String,
    pub disk: String,
    pub memory: String,
    pub cpu: String,
    pub uptime: String,
    pub load: String,
}

// ── Absolute forbidden patterns (never run) ──────────────────

const FORBIDDEN: &[&str] = &[
    "rm -rf /",
    "mkfs",
    ":(){:|:&};:",
    "dd if=/dev/zero",
    "> /dev/sd",
    "chmod -R 777 /",
    "shred /dev",
    "wipefs",
];

const FORBIDDEN_READ: &[&str] = &[
    "/etc/shadow",
    "/etc/gshadow",
    "/root/",
    "/proc/kcore",
];

// ── Handlers ─────────────────────────────────────────────────

pub async fn handle_shell(
    Json(req): Json<ShellRequest>,
) -> (StatusCode, Json<ShellResponse>) {
    if !req.confirmed {
        return (
            StatusCode::BAD_REQUEST,
            Json(ShellResponse {
                command: req.command,
                stdout: String::new(),
                stderr: String::new(),
                returncode: -1,
                error: Some("Command not confirmed by user.".into()),
            }),
        );
    }

    // Safety check
    let cmd_lower = req.command.to_lowercase();
    for pattern in FORBIDDEN {
        if cmd_lower.contains(pattern) {
            warn!("Rejected forbidden command: {}", req.command);
            return (
                StatusCode::FORBIDDEN,
                Json(ShellResponse {
                    command: req.command,
                    stdout: String::new(),
                    stderr: String::new(),
                    returncode: -1,
                    error: Some(format!("Forbidden command pattern: '{pattern}'")),
                }),
            );
        }
    }

    info!("Executing confirmed command: {}", req.command);

    // Run with timeout
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        TokioCommand::new("bash")
            .arg("-c")
            .arg(&req.command)
            .output(),
    )
    .await;

    match result {
        Err(_) => (
            StatusCode::GATEWAY_TIMEOUT,
            Json(ShellResponse {
                command: req.command,
                stdout: String::new(),
                stderr: String::new(),
                returncode: -1,
                error: Some("Command timed out after 30 seconds".into()),
            }),
        ),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ShellResponse {
                command: req.command,
                stdout: String::new(),
                stderr: String::new(),
                returncode: -1,
                error: Some(e.to_string()),
            }),
        ),
        Ok(Ok(output)) => {
            let code = output.status.code().unwrap_or(-1);
            audit_log(&req.command, code);
            (
                StatusCode::OK,
                Json(ShellResponse {
                    command: req.command,
                    stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                    stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                    returncode: code,
                    error: None,
                }),
            )
        }
    }
}

pub async fn handle_file_read(
    Query(params): Query<FileQuery>,
) -> (StatusCode, Json<FileResponse>) {
    let path = &params.path;
    let max_chars = params.max_chars.unwrap_or(10_000);

    // Forbidden path check
    for forbidden in FORBIDDEN_READ {
        if path.starts_with(forbidden) {
            return (
                StatusCode::FORBIDDEN,
                Json(FileResponse {
                    path: path.clone(),
                    content: String::new(),
                    truncated: false,
                    size_bytes: None,
                    error: Some(format!("Reading '{forbidden}' is not permitted.")),
                }),
            );
        }
    }

    match tokio::fs::read_to_string(path).await {
        Ok(content) => {
            let size = tokio::fs::metadata(path)
                .await
                .ok()
                .map(|m| m.len());
            let truncated = content.len() > max_chars;
            (
                StatusCode::OK,
                Json(FileResponse {
                    path: path.clone(),
                    content: content.chars().take(max_chars).collect(),
                    truncated,
                    size_bytes: size,
                    error: None,
                }),
            )
        }
        Err(e) => (
            StatusCode::from_u16(match e.kind() {
                std::io::ErrorKind::NotFound => 404,
                std::io::ErrorKind::PermissionDenied => 403,
                _ => 500,
            }).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(FileResponse {
                path: path.clone(),
                content: String::new(),
                truncated: false,
                size_bytes: None,
                error: Some(e.to_string()),
            }),
        ),
    }
}

pub async fn handle_system_info() -> Json<SystemInfo> {
    let os = run_cmd("uname", &["-a"]);
    let disk = run_cmd_line2("df", &["-h", "/"]);
    let memory = run_cmd_line2("free", &["-h"]);
    let cpu = run_cmd("grep", &["-m1", "model name", "/proc/cpuinfo"])
        .split(':')
        .nth(1)
        .unwrap_or("N/A")
        .trim()
        .to_string();
    let uptime = run_cmd("uptime", &["-p"]);
    let load = run_cmd("cat", &["/proc/loadavg"]);

    Json(SystemInfo {
        os,
        disk,
        memory,
        cpu,
        uptime,
        load,
    })
}

// ── Helpers ───────────────────────────────────────────────────

fn run_cmd(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "N/A".into())
}

fn run_cmd_line2(program: &str, args: &[&str]) -> String {
    let out = run_cmd(program, args);
    out.lines()
        .nth(1)
        .unwrap_or("N/A")
        .to_string()
}

fn audit_log(command: &str, returncode: i32) {
    use std::io::Write;
    let log_dir = std::path::Path::new("/var/log/lilim");
    let _ = std::fs::create_dir_all(log_dir);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_dir.join("commands.log"))
    {
        let now = chrono::Utc::now().to_rfc3339();
        let _ = writeln!(f, "{now} rc={returncode} cmd={command:?}");
    }
}
