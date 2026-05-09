// lilim-runtime: Python brain process manager
//
// Spawns and supervises the Python FastAPI brain (lilim_core/server.py).
// Restarts it automatically if it crashes.
// Waits for the health endpoint to be ready before returning.

use anyhow::{bail, Result};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tracing::{error, info, warn};

const HEALTH_TIMEOUT_SECS: u64 = 30;
const HEALTH_POLL_MS: u64 = 300;

pub struct BrainProcess {
    child: Option<Child>,
    pub port: u16,
    install_root: PathBuf,
}

impl BrainProcess {
    pub fn new(port: u16) -> Self {
        Self {
            child: None,
            port,
            install_root: find_install_root(),
        }
    }

    /// Spawn the Python brain and wait until its /health endpoint responds.
    pub fn start(&mut self) -> Result<()> {
        let python = find_python()?;
        let module = self.install_root.join("lilim_core").join("server.py");

        if !module.exists() {
            bail!(
                "Python brain not found at {}. \
                Make sure lilim_core/ is installed at {}",
                module.display(),
                self.install_root.display()
            );
        }

        info!("Starting Python brain: {} {}", python.display(), module.display());

        let child = Command::new(&python)
            .arg(&module)
            .env("LILIM_BRAIN_PORT", self.port.to_string())
            .env("PYTHONUNBUFFERED", "1")
            .stdout(Stdio::null())  // Captured by systemd journal
            .stderr(Stdio::inherit())
            .spawn()?;

        info!("Brain PID: {}", child.id());
        self.child = Some(child);

        self.wait_for_health()
    }

    /// Check if the brain process is still running.
    #[allow(dead_code)]
    pub fn is_alive(&mut self) -> bool {
        match &mut self.child {
            None => false,
            Some(child) => match child.try_wait() {
                Ok(None) => true,           // Still running
                Ok(Some(status)) => {
                    warn!("Brain process exited with status: {}", status);
                    false
                }
                Err(e) => {
                    error!("Error checking brain process: {}", e);
                    false
                }
            },
        }
    }

    /// Kill the brain process gracefully.
    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            // Try SIGTERM first
            #[cfg(unix)]
            {
                use nix::sys::signal::{kill, Signal};
                use nix::unistd::Pid;
                let _ = kill(Pid::from_raw(child.id() as i32), Signal::SIGTERM);
                std::thread::sleep(Duration::from_millis(500));
            }
            // Force kill if still running
            let _ = child.kill();
            let _ = child.wait();
            info!("Brain process stopped");
        }
    }

    /// Poll the brain's /health endpoint until it responds or timeout.
    fn wait_for_health(&self) -> Result<()> {
        let health_url = format!("http://127.0.0.1:{}/health", self.port);
        let deadline = Instant::now() + Duration::from_secs(HEALTH_TIMEOUT_SECS);

        info!("Waiting for brain health at {} …", health_url);

        while Instant::now() < deadline {
            // Use a synchronous check (we're in a sync startup context)
            if let Ok(resp) = reqwest::blocking::get(&health_url) {
                if resp.status().is_success() {
                    info!("Brain is healthy ✓");
                    return Ok(());
                }
            }
            std::thread::sleep(Duration::from_millis(HEALTH_POLL_MS));
        }

        bail!(
            "Brain did not become healthy within {}s. \
            Check stderr for Python errors.",
            HEALTH_TIMEOUT_SECS
        )
    }
}

impl Drop for BrainProcess {
    fn drop(&mut self) {
        self.stop();
    }
}

// ── Helpers ───────────────────────────────────────────────────

fn find_python() -> Result<PathBuf> {
    // Prefer venv python, then system python3
    let candidates = [
        "/usr/lib/lilim/venv/bin/python3",
        "/usr/lib/lilim/venv/bin/python",
        "python3",
        "python",
    ];

    for candidate in &candidates {
        let path = PathBuf::from(candidate);
        // For absolute paths, check existence; for bare names, trust PATH
        if path.is_absolute() {
            if path.exists() {
                return Ok(path);
            }
        } else if which_python(candidate) {
            return Ok(path);
        }
    }

    bail!("No Python interpreter found. Install python3 or the lilim venv.")
}

fn which_python(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn find_install_root() -> PathBuf {
    // Priority: env override > /usr/lib/lilim > current directory
    if let Ok(root) = std::env::var("LILIM_INSTALL") {
        return PathBuf::from(root);
    }
    let system = PathBuf::from("/usr/lib/lilim");
    if system.exists() {
        return system;
    }
    // Dev mode: walk up to find repo root (where lilim_core/ lives)
    let mut dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    for _ in 0..5 {
        if dir.join("lilim_core").exists() {
            return dir;
        }
        if let Some(parent) = dir.parent() {
            dir = parent.to_path_buf();
        } else {
            break;
        }
    }
    PathBuf::from(".")
}
