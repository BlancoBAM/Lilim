# Lilim — Master Implementation Plan
> **Last updated:** 2026-05-01  
> **Status:** Phase 1 in progress  
> **Purpose:** Canonical living document. Any AI model or contributor should read this first, then pick up the next `[ ]` task. Mark tasks `[x]` when done and update status notes. **Never skip ahead — complete phases in order.**

---

## 0. Plain-English Vision (for all readers)

Lilim is the built-in AI assistant for **Lilith Linux**, a single-user Linux distribution. Think of it like Apple's Siri or Microsoft's Copilot, but living entirely on the user's desktop and deeply integrated with the operating system.

The target user is **a first-year medical assistant student** who:
- Has little Linux or computer experience
- Needs tutoring help (anatomy, physiology, medical terminology, clinical procedures)
- Needs a patient Linux helper that can actually *fix* things, not just describe how
- Benefits from an assistant that remembers past conversations and gets smarter over time

**What Lilim must do:**
1. **Chat** — Answer questions naturally, in plain English, with a sarcastic-but-caring personality
2. **Tutor** — Specialise in medical assistant curriculum (1st year) and college academic skills
3. **Fix Linux** — Read real system files, run real commands, apply real fixes with user confirmation
4. **Remember** — Store facts between sessions, personalise over time (persistent memory)
5. **Route smartly** — Send easy questions to a fast local model; hard questions to a better remote model; stay within a daily cost budget
6. **Enhance prompts** — Auto-improve vague user messages before sending to the LLM
7. **Schedule tasks** — "Remind me in 30 minutes" via cron-style scheduling
8. **Desktop app** — Activated by `Ctrl+L` or clicking the taskbar icon; custom flame UI already designed
9. **iPhone access** — Optional: control the desktop assistant from iPhone via a secure gateway API (lowest priority, can be skipped if too complex)
10. **Ship as a .deb** — Installable on Lilith Linux as a single Debian package

---

## 1. Current State Audit

### What Already Exists (Good!)

| File/Directory | What it does | Quality |
|---|---|---|
| `lilim_core/prompt_enhancer.py` | Classifies user messages, enriches them with system context | ✅ Solid — production-usable |
| `lilim_core/model_router.py` | Routes to local vs remote model by complexity + budget | ✅ Solid — production-usable |
| `lilim_core/memory_manager.py` | Wraps `cortex-mem` CLI for persistent memory | ⚠️ Stub — depends on external binary not yet built |
| `lilim_core/src/api_client.rs` | Rust HTTP client for OpenAI/Anthropic/Ollama APIs | ✅ Good foundation |
| `lilim_core/src/config.rs` | YAML config loader in Rust | ✅ Good foundation |
| `lilim_core/src/memory.rs` | Rust memory module | ⚠️ Needs review |
| `lilim_core/src/rag.rs` | Rust RAG (retrieval-augmented generation) | ⚠️ Needs review/testing |
| `crates/lilim-runtime/src/main.rs` | Rust Axum HTTP server — `/health` and `/query` | ⚠️ Minimal stub — echo only, no real AI |
| `config/lilim.yaml` | Full multi-model config schema | ✅ Well-designed |
| `config/lilim-identity.json` | Personality spec (ENTP, sarcastic, caring) | ✅ Complete |
| `config/routing.toml` | Model routing config | ✅ Good |
| `config/zeroclaw.toml` | ZeroClaw runtime config | ✅ Present |
| `scripts/lilim-serve` | Bash script to start OI + ZeroClaw | ✅ Good |
| `systemd/` | systemd service unit | ✅ Present |
| `packaging/` | .deb packaging skeleton | ⚠️ Skeleton only — needs to include built binary |
| `fix.sh` | Generates host apply script | ✅ Works |
| `lilim_desktop/` | Tauri desktop app | ❌ Empty — only package-lock.json |

### What Is Missing / Broken

1. **`lilim_desktop/`** — The entire Tauri UI is absent. This is the user-facing app.
2. **`lilim-runtime` main.rs** — The Rust server only echoes. It needs to: load personality, call LLMs, apply routing + enhancement, manage memory.
3. **`cortex-mem` binary** — `memory_manager.py` calls `cortex-mem` CLI which doesn't exist locally. Need a fallback or replacement.
4. **No system tool execution** — Lilim cannot yet run shell commands, read files, or apply fixes.
5. **No tests** — Zero unit or integration tests.
6. **Desktop app** — `lilim_desktop/` has no source code.
7. **`.deb` packaging** — Build script exists but references binaries not yet built.
8. **ZeroClaw dependency** — Referenced extensively but not included. Gateway, scheduler, browser control all depend on it.
9. **iPhone gateway** — Referenced in README and config but not implemented.
10. **No `open-interpreter`** — Referenced in serve script and README but no Python OI integration code exists.

### Key Architecture Decision

> **Do we need ZeroClaw and Open Interpreter as external dependencies, or do we build Lilim's own implementation?**

**Recommendation: Build Lilim-native implementations.** Here's why:
- ZeroClaw is a full external Rust project with its own build complexity
- Open Interpreter is a Python package we *can* install, but its server interface may change
- For a distro component that ships as a `.deb`, we want minimal external runtime dependencies
- The *features* from these projects are well-understood — we can implement them directly in Lilim's existing Rust runtime and Python core

**What we take from each project (as inspiration/implementation, not as a submodule):**
- **Open Interpreter** → system tool execution + code runner (implement in `lilim-runtime`)
- **ZeroClaw** → scheduling (cron), gateway API for iPhone, browser automation (implement natively)  
- **Cortex-Mem** → replace with a simpler file-based + SQLite memory that doesn't need an external binary
- **Plano** → already implemented in `model_router.py` ✅
- **Promptomatix** → already implemented in `prompt_enhancer.py` ✅

---

## 2. Target Architecture (What We're Building)

```
User presses Ctrl+L  ──►  lilim_desktop (Tauri UI)
                               │  HTTP/SSE
                               ▼
                     lilim-runtime (Rust Axum server :8080)
                          │
              ┌───────────┼────────────────┐
              ▼           ▼                ▼
     prompt_enhancer   model_router    memory_manager
     (Python module)  (Python module)  (Python module)
              │           │                │
              └───────────┼────────────────┘
                          ▼
                    LLM Provider (via litellm)
                    ┌──────────┬──────────┐
                    │          │          │
                 Ollama    OpenAI    Anthropic
                (local)  (remote)   (remote)
                          │
                    tool_executor (Rust)
                    ┌─────────┬─────────┐
                    │         │         │
                  shell    file_io   cron
                (confirmed) (read)  (schedule)
```

**Communication:** Tauri UI ↔ Rust runtime via HTTP/SSE on localhost:8080. Python intelligence layer called as subprocess or via Python FFI from Rust.

---

## 3. Phased Implementation Plan

### PHASE 1 — Core Intelligence Layer (Python) — IN PROGRESS
*Goal: Reliable Python brain that accepts a message and returns a smart response.*

- [x] **1.1** `prompt_enhancer.py` — classify + enrich prompts *(already done)*
- [x] **1.2** `model_router.py` — route by complexity + budget *(already done)*
- [x] **1.3** `memory_manager.py` — persistent memory interface *(stub done)*
- [x] **1.4** Replace cortex-mem dependency with self-contained SQLite memory *(2026-05-01)*
  - Created `lilim_core/memory_sqlite.py` — SQLite store with full turn/fact/preference API
  - Keyword search, context injection, category classification, auto-extraction of facts
  - `MemoryManager` compat wrapper so existing code continues to work
  - **STATUS:** ✅ Complete — 16 unit tests pass
- [x] **1.5** Create `lilim_core/server.py` — Python FastAPI brain server *(2026-05-01)*
  - `POST /chat` + `POST /chat/sync` → enhance → route → LLM → memory → SSE stream
  - `GET /health`, `POST /memory/search`, `GET /memory/context`, `GET /memory/stats`
  - `POST /tools/shell` (confirmed flag required), `GET /system/info`
  - Loads personality from `lilim-identity.json`, builds Lilim system prompt
  - litellm unified LLM calls, cost logging via ModelRouter
  - **STATUS:** ✅ Complete
- [x] **1.6** Create `lilim_core/tool_executor.py` — safe system tool runner *(2026-05-01)*
  - `shell_command` (confirmed flag), `file_read`, `file_list`, `system_info`, `service_status`, `package_search`
  - Absolute forbidden pattern list, forbidden read paths, 30s timeout, audit log
  - **STATUS:** ✅ Complete
- [x] **1.7** Create `lilim_core/scheduler.py` — cron-style task scheduling *(2026-05-01)*
  - `schedule_once` / `schedule_recurring` / `list_schedules` / `cancel`
  - Natural language parsing: "in 30 minutes", "every day at 9am", "every morning"
  - systemd-run backend (survives process restarts) + threading.Timer fallback
  - Desktop notifications via notify-send
  - SQLite persistence at `~/.local/share/lilim/schedules.db`
  - **STATUS:** ✅ Complete
- [x] **1.8** Write unit tests for Phase 1 Python modules *(2026-05-01)*
  - `tests/test_prompt_enhancer.py` — 13 tests (classification, enhance, should_enhance)
  - `tests/test_model_router.py` — 14 tests (routing, strategy, complexity, budget)
  - `tests/test_memory.py` — 19 tests (CRUD, search, context, session, compat)
  - **All 47 tests pass** (`python3 -m unittest discover -s tests -v`)
  - **STATUS:** ✅ Complete

### PHASE 2 — Rust Runtime (Backend Server) 
*Goal: A production-quality Rust Axum server that orchestrates the Python brain.*

- [ ] **2.1** Refactor `crates/lilim-runtime/src/main.rs`
  - Proper Axum 0.7 routing
  - Config loading from `/etc/lilith/lilim.yaml` or `~/.config/lilim/lilim.yaml`
  - Spawn Python brain server as subprocess on startup
  - Proxy `/chat` requests to Python server
  - SSE streaming support
  - Graceful shutdown
  - **STATUS:** Stub only — minimal echo server
- [ ] **2.2** Add tool execution endpoints to Rust runtime
  - `POST /tools/shell` — confirm + run shell command
  - `GET /tools/file` — read file contents
  - `GET /system/info` — OS, disk, memory snapshot
  - **STATUS:** Not started
- [ ] **2.3** Add scheduling endpoints
  - `POST /schedule/once` — one-time reminder
  - `POST /schedule/recurring` — recurring reminder
  - `GET /schedule/list`
  - `DELETE /schedule/{id}`
  - **STATUS:** Not started
- [ ] **2.4** iPhone gateway (OPTIONAL — implement last)
  - `POST /gateway/query` — authenticated external access
  - HMAC pairing token generation
  - Rate limiting
  - Cloudflare tunnel support (config only)
  - **STATUS:** Not started — lowest priority
- [ ] **2.5** Add Rust unit tests
  - Test config loading
  - Test API proxy behaviour
  - Test tool execution safety checks
  - **STATUS:** Not started

### PHASE 3 — Desktop UI (Tauri + React)
*Goal: The user-facing chat window that matches Lilith Linux aesthetics.*

- [ ] **3.1** Scaffold Tauri 2.x + React + TypeScript app in `lilim_desktop/`
  - `npm create tauri-app@latest` or manual scaffold
  - Configure `tauri.conf.json` for Lilith Linux (frameless window, always-on-top option)
  - **STATUS:** Directory exists but is empty
- [ ] **3.2** Implement global hotkey (`Ctrl+L`) in Tauri
  - Register global shortcut
  - Toggle window visibility
  - **STATUS:** Not started
- [ ] **3.3** Build chat interface (React)
  - Message list with user/assistant bubbles
  - Input box with Enter-to-send
  - Streaming SSE response rendering (typewriter effect)
  - Code block rendering with syntax highlight
  - "Tool use" indicator (when Lilim runs a command, show it)
  - Confirmation dialog for shell commands before execution
  - **STATUS:** Not started
- [ ] **3.4** System tray icon
  - Show/hide window
  - Status indicator (thinking / idle / error)
  - **STATUS:** Not started
- [ ] **3.5** Apply Lilith Linux flame aesthetic
  - Dark theme (deep charcoal / ember colors)
  - Smooth animations (message appear, thinking pulse)
  - Lilim avatar/icon in header
  - **STATUS:** Not started
- [ ] **3.6** Settings panel (basic)
  - Current model info
  - Daily spend display
  - Memory on/off toggle
  - **STATUS:** Not started

### PHASE 4 — Packaging & Deployment
*Goal: A single .deb file that installs Lilim completely on Lilith Linux.*

- [ ] **4.1** Fix `packaging/build_deb.sh` 
  - Build Rust runtime binary
  - Bundle Python venv
  - Bundle Tauri UI binary
  - Install to correct paths
  - **STATUS:** Skeleton script exists but references missing files
- [ ] **4.2** Write proper `packaging/deb_root/DEBIAN/control`
  - Package name: `lilim`
  - Dependencies: `python3 (>=3.10)`, `libwebkit2gtk-4.1-0`, `libgtk-3-0`, etc.
  - Maintainer, description, etc.
  - **STATUS:** DEBIAN dir exists, needs correct control file
- [ ] **4.3** Write `packaging/deb_root/DEBIAN/postinst`
  - Enable systemd service
  - Create `/var/lib/lilim` directory
  - Set permissions
  - **STATUS:** Not started
- [ ] **4.4** Write `packaging/deb_root/DEBIAN/prerm`
  - Stop and disable service before removal
  - **STATUS:** Not started
- [ ] **4.5** CI/CD — GitHub Actions workflow
  - Build on push to `main`
  - Run tests
  - Produce .deb artifact
  - **STATUS:** Not started

### PHASE 5 — Polish, Testing & Documentation
- [ ] **5.1** Integration tests — test full chat flow end-to-end
- [ ] **5.2** Test all tool execution paths (shell, file read, scheduling)
- [ ] **5.3** Write user-facing documentation (`docs/`)
- [ ] **5.4** Write `docs/iphone-setup.md` (if iPhone gateway implemented)
- [ ] **5.5** Verify .deb installs cleanly on fresh Ubuntu 22.04
- [ ] **5.6** Performance test — response time with local model
- [ ] **5.7** Final README update

---

## 4. File Map — What Lives Where

```
Lilim/
├── MASTER_PLAN.md              ← THIS FILE IN REPO — update as work progresses
├── README.md                   ← User-facing docs
├── config/
│   ├── lilim.yaml              ← Main config (models, routing, server)
│   ├── lilim-identity.json     ← Personality spec
│   ├── routing.toml            ← Model routing overrides
│   └── zeroclaw.toml           ← ZeroClaw/gateway config (kept for compat)
├── crates/
│   └── lilim-runtime/          ← Rust Axum backend
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs         ← Server entrypoint + routing
│           ├── proxy.rs        ← [TO CREATE] HTTP proxy to Python brain
│           ├── tools.rs        ← [TO CREATE] Shell/file tool execution
│           ├── scheduler.rs    ← [TO CREATE] Cron/scheduling
│           └── gateway.rs      ← [TO CREATE] iPhone gateway (optional)
├── lilim_core/                 ← Python intelligence layer
│   ├── server.py               ← [TO CREATE] FastAPI brain server
│   ├── prompt_enhancer.py      ← Done
│   ├── model_router.py         ← Done
│   ├── memory_manager.py       ← Stub (cortex-mem dep)
│   ├── memory_sqlite.py        ← [TO CREATE] Self-contained memory
│   ├── tool_executor.py        ← [TO CREATE] Safe system tool runner
│   ├── scheduler.py            ← [TO CREATE] Task scheduling
│   ├── __init__.py             ← Exists
│   ├── Cargo.toml              ← (Rust crate within Python dir — review)
│   └── src/                   ← Rust sub-crate (api_client, config, etc.)
│       ├── api_client.rs       ← Good HTTP client
│       ├── config.rs           ← Config loader
│       ├── memory.rs           ← Review
│       ├── rag.rs              ← Review
│       ├── vlm.rs              ← Review
│       └── lib.rs
├── lilim_desktop/              ← Tauri desktop UI
│   └── [ENTIRE DIRECTORY TO BE BUILT]
├── packaging/
│   ├── build_deb.sh            ← Fix needed
│   └── deb_root/
│       ├── DEBIAN/
│       │   ├── control         ← Needs correct content
│       │   ├── postinst        ← [TO CREATE]
│       │   └── prerm           ← [TO CREATE]
│       ├── etc/lilith/         ← Config files
│       └── usr/                ← Binaries, scripts, services
├── scripts/
│   ├── lilim-serve             ← Start script (needs Python brain path fix)
│   └── release_to_github.sh
├── systemd/
│   └── system/lilith-ai.service
├── tests/                      ← [TO CREATE] Python tests
├── fix.sh                      ← Generates host apply script
└── dist/                       ← Built .deb goes here
```

---

## 5. Key Design Decisions (Reference)

### Memory: SQLite over Cortex-Mem (v1)
`memory_manager.py` currently calls `cortex-mem` — an external Rust binary that may not be installed. For v1, we replace this with `memory_sqlite.py` which uses Python's built-in `sqlite3` module. No external binary needed. Cortex-mem can be added back in v2 as an optional upgrade path.

**Schema:**
```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    role TEXT,           -- 'user' | 'assistant' | 'fact'
    content TEXT,
    category TEXT,       -- 'anatomy' | 'linux' | 'general' | 'preference'
    created_at TIMESTAMP,
    importance REAL      -- 0-1, higher = keep longer
);
CREATE INDEX idx_session ON memories(session_id);
CREATE INDEX idx_category ON memories(category);
```

### LLM Integration: litellm (Python)
The Python brain uses `litellm` to call any model with a unified API. The Rust runtime proxies to the Python FastAPI server. This means:
- Rust handles: HTTP serving, tool execution (shell commands), security, scheduling
- Python handles: LLM calls, prompt enhancement, routing, memory

### Tool Execution Safety Model
Before running any shell command, Lilim MUST:
1. Show the command to the user in the UI
2. Wait for explicit "Run it" confirmation
3. Log the command to `/var/log/lilim/commands.log`
4. Never run commands as root unless specifically whitelisted
5. Timeout all commands at 30 seconds

### iPhone Access (Phase 2.4)
This is **optional** and lowest priority. If implemented:
- Rust gateway server listens on `:42617`
- Cloudflare Tunnel makes it accessible remotely
- Pairing via 6-digit code shown in desktop app
- iPhone Shortcuts app POSTs to the gateway
- If this is too complex, it is explicitly **DEFERRED to v2**

---

## 6. Current Working State (for pickup)

**What compiles/runs today:**
- `lilim_core/*.py` — All Python modules compile; 47 unit tests pass
- `lilim_core/memory_sqlite.py` — SQLite memory (no external binary needed)
- `lilim_core/server.py` — FastAPI brain server (needs `pip install fastapi uvicorn litellm`)
- `lilim_core/tool_executor.py` — Safe system tool runner
- `lilim_core/scheduler.py` — Task scheduler with systemd-run + fallback
- `crates/lilim-runtime` — Axum echo server (Phase 2 will flesh this out)

**What does NOT work yet:**
- End-to-end chat (Python brain server not wired to Rust runtime yet — Phase 2)
- Desktop UI (Tauri app — Phase 3)
- .deb packaging (Phase 4)
- iPhone gateway (Phase 2.4 — optional/deferred)

**Next task to do:** `2.1` — Refactor `crates/lilim-runtime/src/main.rs` to proxy to Python brain

---

## 7. Handoff Protocol

When one AI session ends and another begins:
1. Read this file first (it lives in the repo at `Lilim/MASTER_PLAN.md`)
2. Find the **last completed** `[x]` task
3. Find the **next** `[ ]` task
4. Do that task
5. Mark it `[x]`, add a completion note below the checkbox
6. Update "Current Working State" section (section 6)
7. Commit changes with a clear message

**Never skip tasks without documenting why.** If a task is blocked or deferred, mark it `[~]` and add a note explaining why.

---

## 8. Progress Log

| Date | Task | Notes |
|------|------|-------|
| 2026-05-01 | Plan created | Full audit of existing codebase. Identified all gaps. Established build order. |
| 2026-05-01 | Phase 1 complete | Tasks 1.1–1.8 done. 47 tests pass. Python brain layer complete. |
