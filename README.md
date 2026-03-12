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

🔥 **Intelligent Chatbot** — Sarcastic but caring AI assistant with infernal personality, powered by LiteLLM (GPT-4o, Claude, Ollama, and more)

🧠 **Cortex-Mem Persistent Memory** — Powerful vector-based memory backend using [Cortex-Mem](https://github.com/sopaco/cortex-mem) (Abstract/Overview/Detail hierarchy)

✨ **Automatic Prompt Enhancement** — Transparently enriches vague prompts with context, task structure, and system info for better LLM responses

🎯 **Smart Model Routing** — Simple requests route to fast local models; complex ones auto-escalate to the best remote model within your daily budget

⌨️ **Global Hotkey** — `Ctrl+Shift+L` summons Lilim from anywhere on your desktop

💻 **Code Execution** — Safely runs Python, JavaScript, and shell commands with user confirmation

📅 **Task Scheduling** — Schedule one-time and recurring tasks via natural language

🌐 **Browser Control** — Sandboxed web browsing for research and troubleshooting

📱 **iPhone Access** — Control your desktop from your iPhone via secure Gateway API with pairing authentication

(*Note: Voice Synthesis has been separated to the standalone [Lilith-TTS](https://github.com/BlancoBAM/Lilith-TTS) repository*)

🛡️ **Security First** — Supervised autonomy, sandboxed execution, forbidden path enforcement, pairing-based auth

## Architecture

```
 Ctrl+Shift+L
     │
 ┌───▼──────────┐    HTTP/SSE     ┌──────────────────────────┐
 │ Tauri UI      │ ◄────────────► │   Open Interpreter :8000 │
 │ (React flame) │                │   ┌──────────────────┐   │
 └──────────────┘                │   │ Prompt Enhancer  │   │
                                  │   │ (classify+enrich)│   │
 ┌──────────────┐  HTTPS/WSS     │   └───────┬──────────┘   │
 │ iPhone       │ ◄──────────►  │           ▼              │
 │ (Shortcuts)  │  :42617        │   ┌──────────────────┐   │
 └──────────────┘                │   │  Model Router    │   │
                                  │   │  (Plano+LiteLLM) │   │
 ┌──────────────┐                │   └───────┬──────────┘   │
 │ Cortex-Mem   │ ◄────────────► │           ▼              │
 │ (Rust API)   │  cortex://     │   local ◄─┤─► remote    │
 └──────────────┘                │   ollama  │  gpt-4o     │
                                 │           │  claude     │
                                 ├──────────────────────────┤
                                  │   ZeroClaw Runtime       │
                                  │   • Gateway + pairing    │
                                  │   • Cron scheduler       │
                                  │   • Browser control      │
                                  └──────────────────────────┘
```

| Component | Tech | Purpose |
|-----------|------|---------|
| **Brain** | Python / Open Interpreter / LiteLLM | LLM routing, code execution, personality |
| **Memory** | **Rust / Cortex-Mem** | High-performance vector database, multi-tier abstraction |
| **Enhancer** | Python / DSPy-inspired | Automatic prompt classification and enrichment |
| **Router** | Python / Plano-inspired | Smart model selection with budget tracking |
| **Runtime** | Rust / ZeroClaw | Security, scheduling, gateway, sandboxing |
| **Desktop UI** | TypeScript / React / Tauri | Flame-themed chat interface |

## Intelligence Layer

Three modules in `lilim_core/` run transparently to make Lilim smarter:

### Cortex-Mem Persistent Memory

Lilim integrates [Cortex-Mem](https://github.com/sopaco/cortex-mem) to provide a powerful, hierarchical long-term memory system! Unlike fragile text logs, Cortex-Mem uses Qdrant vector-search to retrieve facts instantly and handles profile updating automatically via `cortex://` URIs!


### Prompt Enhancement (Promptomatix-inspired)

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
| "What time is it?" | `ollama/qwen3` (local) | Simple, fast, free |
| "Help me study anatomy" | `ollama/qwen3` (local) | Tutoring, standard knowledge |
| "Write a REST API server" | `gpt-4o-mini` (remote) | Code generation needs precision |
| "Debug this Python traceback" | `claude-sonnet-4-20250514` (remote) | Deep code reasoning |

Configure in `config/routing.toml` — set daily budget caps, override category routing, or force local-only mode.

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
sudo apt install cmake espeak-ng python3-pip nodejs npm xsel xclip

# Install Open Interpreter
cd ../Lilim-v2 && pip install -e . && cd ../Lilim

# Build ZeroClaw runtime
git clone https://github.com/zeroclaw-labs/zeroclaw.git zeroclaw
cd zeroclaw && cargo build --release && cd ..
sudo cp zeroclaw/target/release/zeroclaw /usr/bin/

# Deploy configs
sudo mkdir -p /etc/lilith
sudo cp config/zeroclaw.toml /etc/lilith/
sudo cp config/lilim-identity.json /etc/lilith/
sudo cp config/routing.toml /etc/lilith/

# Install service
sudo cp scripts/lilim-serve /usr/bin/
sudo chmod +x /usr/bin/lilim-serve
sudo cp systemd/system/lilith-ai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now lilith-ai
```

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
       > zeroclaw cron add-at "..." "Time for a break!"

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

### Autonomy Level

Edit `config/zeroclaw.toml`:

```toml
[autonomy]
level = "supervised"  # "readonly", "supervised", "full"
```

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

```
Lilim/
├── assets/                    # Icons and images
├── config/                    # Runtime configuration
│   ├── zeroclaw.toml          # ZeroClaw runtime settings
│   ├── lilim-identity.json    # AIEOS persona specification
│   └── routing.toml           # Model routing + budget config
├── desktop/                   # .desktop launcher entries
├── docs/                      # Documentation
├── lilim_core/                # Intelligence layer (Python)
│   ├── memory_manager.py      # Persistent knowledge graph
│   ├── prompt_enhancer.py     # Automatic prompt optimization
│   └── model_router.py        # Smart model routing
├── lilim_desktop/             # Tauri desktop app (React + Rust)
│   ├── src/                   # React frontend
│   └── src-tauri/             # Tauri backend (Rust)
├── scripts/                   # Server launch + panel scripts
├── systemd/                   # systemd service files
└── lilith-tts/                # TTS module (separate repo)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Inspired By / Credits

Lilim's architecture and capabilities were heavily inspired by and built upon several phenomenal open-source projects. Please check them out:

- **[Open Interpreter](https://github.com/OpenInterpreter/open-interpreter)** — The core LLM execution engine and computer API that powers Lilim's brain.
- **[ZeroClaw](https://github.com/zeroclaw-labs/zeroclaw)** — The secure Rust runtime, gateway API, and system abstraction layer.
- **[Rowboat](https://github.com/rowboatlabs/rowboat)** — Inspiration for the local Markdown-vault persistent memory system.
- **[Promptomatix](https://github.com/SalesforceAIResearch/promptomatix)** — Inspiration for the automatic, transparent prompt enhancement and classification layer.
- **[Plano](https://github.com/katanemo/plano)** — Inspiration for the intelligent, complexity-based model routing layer.
- **[Cortex Mem](https://github.com/sopaco/cortex-mem)** - Updated Rust based memory layer for Lilim.
## License

This project is licensed under **AGPL-3.0** (inherited from Open Interpreter and ZeroClaw).

See [LICENSE](LICENSE) for details.
