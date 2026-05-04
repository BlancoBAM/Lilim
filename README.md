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
  <a href="#contributing">Contributing</a>
</p>

---

## Features

🔥 **Intelligent Chatbot** — Sarcastic but caring AI assistant with infernal personality

🧠 **Local Phi-2 Inference** — Runs Microsoft Phi-2 (2.7B) locally on CPU via HuggingFace Candle — no Ollama, no Python inference, no API key required

🧠 **Native SQLite Memory** — Fast, hierarchical vector-based memory embedded directly into the Python brain

✨ **Automatic Prompt Enhancement** — Transparently enriches vague prompts with context, task structure, and system info

🎯 **Smart Model Routing** — Simple requests route to the local Phi-2 model; complex ones auto-escalate to remote providers when API keys are configured

⌨️ **Global Hotkey** — `Ctrl+L` summons Lilim from anywhere on your desktop

💻 **Code Execution** — Safely runs shell commands with explicit UI confirmation

📅 **Task Scheduling** — Schedule one-time and recurring tasks via natural language, backed by `systemd-run`

🛡️ **Security First** — Rust-native API gateway, sandboxed execution, strict command blocklists, and audit logging

## Architecture

```
 Ctrl+L
     │
 ┌───▼──────────┐    HTTP/SSE     ┌──────────────────────────┐
 │ Tauri UI      │ ◄────────────► │   Rust Runtime Gateway   │
 │ (React flame) │                │   (lilim-runtime :8080)  │
 └──────────────┘                │   ┌──────────────────┐   │
                                  │   │ Candle Phi-2     │   │
                                  │   │ (local, CPU)     │   │
                                  │   └───────┬──────────┘   │
                                  │           │              │
                                  │   ┌───────▼──────────┐   │
                                  │   │ Python Brain API │   │
                                  │   │ (FastAPI :8081)  │   │
 ┌──────────────┐                │   └───────┬──────────┘   │
 │ SQLite Mem   │ ◄────────────► │           │              │
 │ (Embedded)   │                │   remote fallback:       │
 └──────────────┘                │   groq · openrouter ·    │
                                  │   gemini · cerebras …   │
                                  └──────────────────────────┘
```

| Component | Tech | Purpose |
|-----------|------|---------|
| **Inference** | Rust / Candle / GGUF | Local Phi-2 model, incremental token generation on CPU |
| **Brain** | Python / FastAPI / LiteLLM | LLM routing, task parsing, memory ops, personality |
| **Memory** | Python / SQLite | Semantic memory embedded in the Brain |
| **Enhancer** | Python / DSPy-inspired | Automatic prompt classification and enrichment |
| **Router** | Python / FreeRouter | Smart model selection with auto-fallback to free APIs |
| **Runtime Gateway** | Rust / Axum | HTTP server, security, process management, tool sandbox |
| **Desktop UI** | TypeScript / React / Tauri | Flame-themed chat interface with SSE streaming |

## Intelligence Layer

Three modules in `lilim_core/` run transparently to make Lilim smarter:

### Native SQLite Persistent Memory

Lilim uses an embedded SQLite-based memory system (`lilim_core/memory_sqlite.py`) to provide a powerful, hierarchical long-term memory system. Unlike fragile text logs, it uses token-overlap based search and basic vector approximations to retrieve facts instantly and handles profile updating automatically.

### Prompt Enhancement

Short or vague prompts are automatically enriched before hitting the LLM:

| You type | Lilim sees (enhanced) |
|----------|----------------------|
| "fix my wifi" | `[Task: system_admin. Provide exact commands...] fix my wifi [System: Ubuntu 22.04, wlan0 down...]` |
| "quiz me on bones" | `[Task: tutoring. Use ELI10 approach...] quiz me on bones [Memory: studying for anatomy exam]` |
| "hey" | `hey` *(casual messages pass through unchanged)* |

### Smart Routing

Requests are routed to the optimal model based on complexity:

| Request | Model | Why |
|---------|-------|-----|
| "What time is it?" | `phi-2` (local) | Simple, fast, free, no API key needed |
| "Help me study anatomy" | `phi-2` (local) | Tutoring, standard knowledge |
| "Write a REST API server" | `groq/llama3-70b` (remote) | Code generation needs precision (free tier) |
| "Debug this Python traceback" | `openrouter/...` (remote) | Deep code reasoning (requires API key) |

When no API keys are configured, **all requests are handled locally by Phi-2**. Configure API keys dynamically in the Settings UI (Ctrl+L → Gear Icon) to unlock remote providers for complex tasks.

## Installation

### Prerequisites

- Linux (Ubuntu 22.04+ / Lilith Linux)
- Python 3.10+
- Rust 1.75+
- Node.js 18+ (for Tauri UI build)

### Quick Install (Development)

```bash
# Clone
git clone https://github.com/BlancoBAM/Lilim.git
cd Lilim

# Install system dependencies
sudo apt install python3-pip python3-venv nodejs npm \
  pkg-config libssl-dev libwebkit2gtk-4.1-dev librsvg2-dev

# Build and install locally
./local_install.sh
```

The install script will:
1. Build the Rust runtime with Candle Phi-2 inference
2. Build the Tauri desktop app
3. Create a `.deb` package
4. Install and restart the `lilith-ai` service

On first launch, Lilim will download the Phi-2 GGUF model (~1.7GB, one-time).

### Desktop App (Dev Mode)

```bash
cd lilim_desktop
npm install
npm run tauri dev
```

## Usage

### Chat via Desktop

Press **`Ctrl+Shift+L`** to toggle the Lilim window. Type your message and press Enter.

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
local_model = "phi-2"       # Built-in Candle inference
complexity_threshold = 0.6  # 0-1, above routes to remote
budget_limit_daily = 5.00   # USD spending cap

[routing.remote_models]
fast = "gpt-4o-mini"
balanced = "gpt-4o"
reasoning = "claude-sonnet-4-20250514"
```

### Security Layer

All system commands executed by Lilim are subject to strict blocklists in the Rust Runtime Gateway and require explicit UI confirmation. Edit `config/routing.toml` to manage safety thresholds.

### Memory

Lilim's memory is stored in SQLite at `~/.local/share/lilim/memory/`:

```bash
ls ~/.local/share/lilim/memory/
```

## Project Structure

```
Lilim/
├── config/                    # Runtime configuration
│   ├── lilim-identity.json    # Persona specification
│   ├── lilim-responses.yaml   # Personality response library
│   ├── lilim.yaml             # Full model/server config
│   └── routing.toml           # Model routing + budget config
├── crates/
│   ├── lilim-inference/       # Candle Phi-2 inference engine (Rust)
│   │   └── src/
│   │       ├── lib.rs         # Public API (InferenceEngine)
│   │       ├── phi2.rs        # Token generation (incremental KV-cache)
│   │       ├── downloader.rs  # HuggingFace model downloader
│   │       └── config.rs      # Inference configuration
│   └── lilim-runtime/         # Rust Runtime Gateway Server
│       └── src/
│           ├── main.rs        # Server entrypoint + Axum routing
│           ├── inference.rs   # Local/remote routing handler
│           ├── proxy.rs       # HTTP proxy to Python brain
│           ├── tools.rs       # Shell/file tool execution
│           └── scheduler.rs   # Task scheduling
├── lilim_core/                # Intelligence layer (Python)
│   ├── server.py              # FastAPI Brain Server
│   ├── memory_sqlite.py       # Persistent knowledge store
│   ├── prompt_enhancer.py     # Automatic prompt optimization
│   ├── model_router.py        # Smart model routing
│   ├── free_router.py         # Provider-agnostic free-tier router
│   ├── tool_executor.py       # Safe system tool executor
│   └── scheduler.py           # Task scheduler
├── lilim_desktop/             # Tauri desktop app (React + Rust)
│   ├── src/                   # React frontend
│   └── src-tauri/             # Tauri backend (Rust)
├── packaging/                 # Debian .deb build scripts
├── systemd/                   # systemd service files
├── tests/                     # Python unit tests (47 tests)
└── local_install.sh           # Build + install script
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Inspired By / Credits

Lilim's architecture and capabilities were inspired by several open-source projects:

- **[Open Interpreter](https://github.com/OpenInterpreter/open-interpreter)** — Code execution workflow
- **[Rowboat](https://github.com/rowboatlabs/rowboat)** — Persistent memory systems
- **[Promptomatix](https://github.com/SalesforceAIResearch/promptomatix)** — Automatic prompt enhancement
- **[Plano](https://github.com/katanemo/plano)** — Complexity-based model routing
- **[Cortex Mem](https://github.com/sopaco/cortex-mem)** — Hierarchical memory architecture
- **[HuggingFace Candle](https://github.com/huggingface/candle)** — Rust ML framework for local inference

## License

This project is licensed under **AGPL-3.0**.

See [LICENSE](LICENSE) for details.
