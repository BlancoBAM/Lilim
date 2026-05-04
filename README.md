<p align="center">
  <img src="assets/lilith-icon.png" alt="Lilim" width="120" />
</p>

<h1 align="center">Lilim</h1>

<p align="center">
  <strong>Infernal AI Assistant for Lilith Linux</strong><br/>
  Sarcastic. Capable. Yours.
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#intelligence-layer">Intelligence</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#iphone-access">iPhone Access</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## Features

🔥 **Intelligent Chatbot** — Sarcastic but caring AI assistant with infernal personality, powered by FreeRouter and LiteLLM
🧠 **Local Phi-2 Inference** — Runs Microsoft Phi-2 locally on CPU/GPU via HuggingFace Candle (no Ollama required)

🧠 **Native SQLite Memory** — Fast, hierarchical vector-based memory backend embedded directly into the Python brain (no external binaries)

✨ **Automatic Prompt Enhancement** — Transparently enriches vague prompts with context, task structure, and system info for better LLM responses

🎯 **Smart Model Routing** — Simple requests route to fast local models; complex ones auto-escalate to the best remote model within your daily budget

⌨️ **Global Hotkey** — `Ctrl+L` summons Lilim from anywhere on your desktop

💻 **Code Execution** — Safely runs Python, JavaScript, and shell commands with explicit UI confirmation

📅 **Task Scheduling** — Schedule one-time and recurring tasks via natural language, backed by `systemd-run`

📱 **iPhone Access** — (Planned) Control your desktop from your iPhone via secure Gateway API with pairing authentication

🛡️ **Security First** — Rust-native API gateway, sandboxed execution, strict command blocklists, and audit logging

## Architecture

```
 Ctrl+L
     │
 ┌───▼──────────┐    HTTP/SSE     ┌──────────────────────────┐
 │ Tauri UI      │ ◄────────────► │   Rust Proxy Gateway     │
 │ (React flame) │                │   (lilim-runtime :8080)  │
 └──────────────┘                │   ┌──────────────────┐   │
                                  │   │ Process Manager  │   │
                                  │   │ Security Filter  │   │
                                  │   └───────┬──────────┘   │
                                  │           ▼              │
                                  │   ┌──────────────────┐   │
                                  │   │ Python Brain API │   │
                                  │   │ (FastAPI :8081)  │   │
                                  │   └───────┬──────────┘   │
 ┌──────────────┐                │           ▼              │
 │ SQLite Mem   │ ◄────────────► │   local ◄─┤─► remote    │
 │ (Embedded)   │                │   Phi-2   │  groq       │
 └──────────────┘                │  (Candle) │  openrouter │
                                 └──────────────────────────┘
```

| Component | Tech | Purpose |
|-----------|------|---------|
| **Brain** | Python / FastAPI / LiteLLM | LLM routing, task parsing, memory ops, personality |
| **Memory** | **Python / SQLite** | High-performance semantic memory embedded in the Brain |
| **Enhancer** | Python / DSPy-inspired | Automatic prompt classification and enrichment |
| **Router** | Python / FreeRouter | Smart model selection with auto-fallback to free APIs |
| **Runtime Gateway** | Rust / Axum | Security, Candle Phi-2 inference, system tool sandbox |
| **Desktop UI** | TypeScript / React / Tauri | Flame-themed chat interface with streaming |

## Intelligence Layer

Three modules in `lilim_core/` run transparently to make Lilim smarter:

### Native SQLite Persistent Memory

Lilim uses an embedded SQLite-based memory system (`lilim_core/memory_sqlite.py`) to provide a powerful, hierarchical long-term memory system! Unlike fragile text logs, it uses token-overlap based search and basic vector approximations to retrieve facts instantly and handles profile updating automatically.

### Prompt Enhancement

Short or vague prompts are automatically enriched before hitting the LLM:

| You type | Lilim sees (enhanced) |
|----------|----------------------|
| "fix my wifi" | `[Task: system_admin. Provide exact commands...] fix my wifi [System: Ubuntu 22.04, wlan0 down...]` |
| "quiz me on bones" | `[Task: tutoring. Use ELI10 approach...] quiz me on bones [Memory: studying for anatomy exam]` |
| "hey" | `hey` *(casual messages pass through unchanged)* |

### Smart Routing (Plano + LiteLLM)

Requests are routed to the optimal model based on complexity:

| Request | Model | Why |
|---------|-------|-----|
| "What time is it?" | `phi-2` (local) | Simple, fast, free, no API key needed |
| "Help me study anatomy" | `phi-2` (local) | Tutoring, standard knowledge |
| "Write a REST API server" | `groq/llama3-70b` (remote) | Code generation needs precision (fallback to free tier) |
| "Debug this Python traceback" | `openrouter/...` (remote) | Deep code reasoning |
| "Debug this Python traceback" | `claude-sonnet-4-20250514` (remote) | Deep code reasoning |

Configure your API keys dynamically in the Settings UI (Ctrl+L -> Gear Icon). Lilim will auto-detect the provider and route to it automatically when local inference is insufficient or unavailable.

## Installation

### Prerequisites

- Linux (Ubuntu 22.04+ / Lilith Linux)
- Python 3.10+
- Rust 1.75+ (for ZeroClaw build)
- Node.js 18+ (for Tauri UI)
- `espeak-ng`, `cmake` (for TTS)

### Quick Install

```bash
# Clone
git clone https://github.com/BlancoBAM/Lilim.git
cd Lilim

# Install system dependencies
sudo apt install python3-pip python3-venv nodejs npm pkg-config libssl-dev libwebkit2gtk-4.1-dev librsvg2-dev

# 1. Setup Python Brain
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn litellm apscheduler pydantic

# 2. Build Rust Runtime Gateway
cd crates/lilim-runtime
cargo build --release
cd ../..

# 3. Build Tauri Desktop UI
cd lilim_desktop
npm install
npm run tauri build
cd ..

# 4. Deploy configs and test
./fix.sh
~/lilim_host_apply_and_test.sh
```

### Production Readiness

This distribution component includes a production-readiness workflow implemented by `fix.sh`, which generates a robust host orchestration script `lilim_host_apply_and_test.sh`. The generator is designed to be idempotent and safe to re-run, with explicit logging and guarded state transitions. It covers: repo synchronization, dependency installation, building the Rust proxy and Tauri UI, packaging, configuration deployment, systemd integration, environment overrides, service startup, and health verification.

The packaging step natively bundles the Rust-based Lilim Runtime binary (`lilim-runtime`), the Python brain (`lilim_core`), and the integrated desktop UI (`lilim`) into a single `.deb` file for your system.

- How to use:
- 1) Run fix.sh to regenerate the host script.
- 2) Run the generated script: ~/lilim_host_apply_and_test.sh
- 3) Inspect logs under /var/log/lilim (e.g., lilim_host_apply_and_test.log) for traceability.
- 4) Verify the service is running and healthy via curl http://127.0.0.1:8000/health and systemctl status lilith-ai.service.
- 5) If needed, push changes to the main branch using the RUN_PUSH option on the host script or manually via git.

### Desktop App

```bash
cd lilim_desktop
npm install
npm run tauri dev
```

## Usage

### Chat via Desktop

Press **`Ctrl+Shift+L`** to toggle the Lilim window. Type your message and press Enter.

### Read Aloud (TTS)

Press **`Ctrl+Shift+T`** to read highlighted text (or clipboard contents) aloud via Lilith-TTS.

### Example Conversations

```
You: What's my disk usage?
Lilim: Let me check that for you...
       > df -h
       Your root partition is at 67% — plenty of room. No fires to put out... yet.

You: Remind me to take a break in 30 minutes
Lilim: Done. I'll bug you in 30 minutes. Don't blame me when you're startled.
       > systemd-run --on-active="30m" notify-send "Lilim" "Time for a break!"

You: Help me study anatomy terms
Lilim: *Cracks knuckles like a judgmental tutor*
       Alright, let's quiz you. What's the difference between the axial
       and appendicular skeleton?
```

## Configuration

### Model Routing

Edit `config/routing.toml`:

```toml
[routing]
strategy = "auto"           # "auto", "local-only", "remote-only"
local_model = "ollama/qwen3:4b"
complexity_threshold = 0.6  # 0-1, above routes to remote
budget_limit_daily = 5.00   # USD spending cap

[routing.remote_models]
fast = "gpt-4o-mini"
balanced = "gpt-4o"
reasoning = "claude-sonnet-4-20250514"
```

### Security Layer

Edit `config/routing.toml` to manage safety thresholds.
All system commands executed by Lilim are subject to strict blocklists in the Rust Proxy Gateway and require explicit UI confirmation.

### Memory

Inspect and edit your memory vault directly:

```bash
ls ~/.local/share/lilim/memory/
# Edit with any Markdown editor or Obsidian
```

## iPhone Access

See [docs/iphone-setup.md](docs/iphone-setup.md) for full instructions.

**Quick version:**
1. Enable tunnel in `zeroclaw.toml` (`tunnel.provider = "cloudflare"`)
2. Get pairing code from Lilim desktop app
3. Create iOS Shortcut that POSTs to the gateway

## Project Structure

Lilim/
├── config/                    # Runtime configuration
│   ├── lilim-identity.json    # Persona specification
│   └── routing.toml           # Model routing + budget config
├── crates/
│   └── lilim-runtime/         # Rust Proxy Gateway Server
├── lilim_core/                # Intelligence layer (Python)
│   ├── server.py              # FastAPI Backend
│   ├── memory_sqlite.py       # Persistent knowledge graph
│   ├── prompt_enhancer.py     # Automatic prompt optimization
│   ├── model_router.py        # Smart model routing
│   ├── tool_executor.py       # Safe system tool executor
│   └── scheduler.py           # Task scheduler
├── lilim_desktop/             # Tauri desktop app (React + Rust)
│   ├── src/                   # React frontend
│   └── src-tauri/             # Tauri backend (Rust)
├── packaging/                 # Debian .deb build scripts
├── systemd/                   # systemd service files
└── fix.sh                     # Host orchestrator generator
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Inspired By / Credits

Lilim's architecture and capabilities were heavily inspired by several phenomenal open-source projects. Please check them out:

- **[Open Interpreter](https://github.com/OpenInterpreter/open-interpreter)** — Inspiration for the code execution workflow.
- **[ZeroClaw](https://github.com/zeroclaw-labs/zeroclaw)** — Inspiration for the secure gateway, scheduling, and system abstraction layers.
- **[Rowboat](https://github.com/rowboatlabs/rowboat)** — Inspiration for persistent memory systems.
- **[Promptomatix](https://github.com/SalesforceAIResearch/promptomatix)** — Inspiration for the automatic, transparent prompt enhancement and classification layer.
- **[Plano](https://github.com/katanemo/plano)** — Inspiration for the intelligent, complexity-based model routing layer.
- **[Cortex Mem](https://github.com/sopaco/cortex-mem)** - Inspiration for the hierarchical memory architecture.

## License

This project is licensed under **AGPL-3.0**.

See [LICENSE](LICENSE) for details.
