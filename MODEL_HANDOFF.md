# Lilim Project: Comprehensive Model Handoff Specification

**Document Version:** 1.0.0  
**Target Audience:** Succeeding AI Models & Core Maintainers  
**Repository:** [Lilim (BlancoBAM/Lilim)](file:///home/s8n/Lilim)  
**Date:** October 2026  
**System Target:** Lilith Linux (Debian/Ubuntu-derived OS)

---

## 1. Executive Summary & Objective

Lilim is designed as the native, local-first AI system agent for Lilith Linux. It combines a high-performance **Rust runtime and ML engine** with a modular **Python cognitive brain** and a **Tauri v2 desktop shell**.

The user requested an overhaul of Lilim to transform it from a passive conversational chat assistant into an **autonomous system agent**. Specifically:
1. **True Autonomy:** Lilim must actually execute tasks in the Linux environment rather than merely printing bash commands to chat for the user to copy-paste.
2. **Elevated Privileges (Sudo Flow):** When an operation requires elevated root privileges, Lilim must recognize it, request user confirmation, securely handle authentication (via Polkit `pkexec` or in-memory session tokens with zero plain-text disk storage), and execute the command.
3. **Model Modernization:** Upgrade from the obsolete Microsoft Phi-2 (2.7B) to a modern, capable quantized model (Phi-3.5-mini-instruct / Phi-4 class) while strictly preserving the **pure Rust ML engine using HuggingFace Candle** (no Ollama dependency, no Python PyTorch/vLLM daemon).
4. **Fix Systemd Daemon Installation:** The installer failed with `Loaded: bad-setting (Reason: Unit lilith-ai.service has a bad unit file setting.)`. The underlying template/service bug must be permanently resolved.
5. **Modernize Subsystems:** Overhaul the Prompt Enhancer, Memory Store, Tool Execution Loop, Zero-Config fallback providers, MCP (Model Context Protocol), and BrowserOS integration.

---

## 2. Requirements & Technical Decisions

### 2.1 The Inference Engine: Candle & Phi-3.5 vs. Phi-4
* **User Constraint:** The user insisted on keeping Candle and maximizing Rust-native code: *"I dont want to replace candle. I want as much code as possible to be rust based. If candle and phi-4 is not stable/production ready, keep phi-2 or replace with phi-3 etc. that WILL work without re-writing everything."*
* **Investigation Findings:**
  - `candle` (0.8.x) contains native GGUF support for quantized Phi-3 via `candle_transformers::models::quantized_phi3::ModelWeights`.
  - Phi-4 GGUF uses RoPE and architecture variations that are currently unmerged or require community forks in Candle.
  - **Selected Engine:** `microsoft/Phi-3.5-mini-instruct` quantized to `Q4_K_M` (~2.4 GB GGUF, distributed by `bartowski/Phi-3.5-mini-instruct-GGUF`).
  - **Capabilities:** 3.8B parameters, 128k context support, native ChatML instruct tags (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`), high multi-step reasoning capabilities.
  - **Implementation:** Implemented in pure Rust in `crates/lilim-inference/src/phi3.rs`. Compiles cleanly in the workspace.

### 2.2 Autonomous Task Execution: Why Lilim Was Only Printing Commands
Investigation into `lilim_core/server.py` and `lilim_core/prompt_enhancer.py` uncovered two major design flaws that prevented autonomous execution:
1. **Prose Window Truncation in `server.py`:**
   Lines 1041–1051 only inspected the first 400 characters of the LLM response for code blocks:
   ```python
   scan_window = sanitised_reply[:400]
   bash_matches = list(re.finditer(r"```(?:bash|sh|shell)?\s*\n([\s\S]*?)```", scan_window))
   ```
   If the model generated an introductory thought or polite response longer than 400 characters, any subsequent `bash` block was ignored and simply printed to the chat UI.
2. **Hardcoded Local Loop Disablement in `server.py`:**
   Lines 1135–1146 contained an explicit abort:
   ```python
   # LOCAL models: one execution only — stop here
   if active_provider == "LOCAL":
       break # No ReAct loop for local models
   ```
   The ReAct iterative loop was explicitly disabled for local inference because the previous Phi-2 model was prone to infinite loops.
3. **Passive System Prompts in `prompt_enhancer.py`:**
   The `system_admin` task enrichment explicitly instructed:
   ```python
   "Provide exact shell commands the user can copy and run."
   ```
   This prompted the model to tell the user to run commands manually.

### 2.3 Elevated Privileges Architecture
* **Safety Principle:** Commands requiring `sudo` must never run silently.
* **Classification:** `ToolExecutor.classify_command()` checks commands against `sudo_patterns`. If matched, it returns `"sudo_confirm"`.
* **Execution Strategy:**
  1. **Polkit (`pkexec`):** First preference on desktop environments. Invokes standard GUI authentication dialog.
  2. **Session Token (`sudo -S`):** When user confirms with password, stores password in memory (`_SUDO_TOKENS[session_id]`) with a 5-minute TTL. The token is never written to disk, sqlite, or logs.
  3. **Elevation Event:** Server pauses SSE stream, emits a `sudo_pending` or `tool_pending` event to the UI, and waits for explicit user confirmation.

### 2.4 Systemd Service Installation Bug
* **Root Cause:**
  - `systemd/system/lilith-ai.service` originally contained `User=%i`, `Group=%i`, and `WorkingDirectory=/home/%i`.
  - `%i` is only valid in systemd **template units** (e.g. `lilith-ai@username.service`).
  - When installed as `lilith-ai.service`, systemd marked the unit as:
    `Loaded: bad-setting (Reason: Unit lilith-ai.service has a bad unit file setting.)`
  - In `packaging/build_deb.sh`, `sed` was searching for `User=aegon`, which never matched `User=%i`.
* **Fix Applied:**
  - Replaced `%i` with `LILIM_USER_PLACEHOLDER` in `systemd/system/lilith-ai.service`.
  - Updated memory thresholds to `MemoryMax=6G` / `MemoryHigh=5G` to provide headroom for Phi-3.5 / Phi-4 inference.
  - Updated `local_install.sh` and `packaging/build_deb.sh` to safely patch the placeholder with `$CURRENT_USER` / `$TARGET_USER`.

---

## 3. Architecture & Subsystems

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Tauri Desktop Shell                             │
│                  (lilim_desktop - Vue / TypeScript)                    │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ HTTP / SSE (:8420)
┌────────────────────────────────────▼───────────────────────────────────┐
│                        lilim-runtime (Rust)                            │
│  - Gateway / Reverse Proxy (Axum / Tokio)                              │
│  - Process Supervisor for Python Brain (:8421)                         │
│  - Direct link to lilim-inference engine                               │
└──────────────────┬─────────────────────────────────┬───────────────────┘
                   │                                 │
                   │ In-process API                  │ Internal HTTP / SSE (:8421)
┌──────────────────▼─────────────┐ ┌─────────────────▼───────────────────┐
│     lilim-inference (Rust)     │ │        lilim_core (Python Brain)    │
│  - HuggingFace Candle 0.8      │ │  - FastAPI / Uvicorn (:8421)        │
│  - Phi3Engine (Phi-3.5-mini)   │ │  - Server & ReAct Loop (server.py)  │
│  - GGUF Q4_K_M Reader          │ │  - Tool Executor (tool_executor.py) │
│  - Token-by-token streaming    │ │  - Prompt Enhancer & ChatML         │
│  - Downloader (Hugging Face)   │ │  - SQLite Memory & Session Manager  │
│  - CPU / CUDA Acceleration     │ │  - Zero-Config Router (fallback)    │
└────────────────────────────────┘ └─────────────────────────────────────┘
```

### Component Directory Mapping
- [crates/lilim-inference](file:///home/s8n/Lilim/crates/lilim-inference): Pure Rust Candle inference library.
- [crates/lilim-runtime](file:///home/s8n/Lilim/crates/lilim-runtime): Axum daemon and proxy.
- [lilim_core](file:///home/s8n/Lilim/lilim_core): Python FastAPI daemon, ReAct loop, tool executor.
- [systemd/system](file:///home/s8n/Lilim/systemd/system): Service unit files.
- [local_install.sh](file:///home/s8n/Lilim/local_install.sh): Local developer and user installation script.
- [packaging](file:///home/s8n/Lilim/packaging): Debian `.deb` package generation scripts.

---

## 4. Current Implementation Status

### Completed Changes
| File | Status | Description |
|---|---|---|
| [crates/lilim-inference/src/phi3.rs](file:///home/s8n/Lilim/crates/lilim-inference/src/phi3.rs) | **Done** | Implemented `Phi3Engine` with `candle_transformers::models::quantized_phi3`, streaming generation, prompt prefill, and tokenizer decoding. |
| [crates/lilim-inference/src/config.rs](file:///home/s8n/Lilim/crates/lilim-inference/src/config.rs) | **Done** | Configured for `Phi-3.5-mini-instruct`, 4096 context window, 512 max output tokens, 0.4 temperature, ChatML system prompt. |
| [crates/lilim-inference/src/lib.rs](file:///home/s8n/Lilim/crates/lilim-inference/src/lib.rs) | **Done** | Exported `Phi3Engine` as `InferenceEngine`, preserving 100% API compatibility with `lilim-runtime`. |
| [crates/lilim-inference/src/downloader.rs](file:///home/s8n/Lilim/crates/lilim-inference/src/downloader.rs) | **Done** | Configured automated Hugging Face downloads for `bartowski/Phi-3.5-mini-instruct-GGUF` and Microsoft tokenizer. |
| [crates/lilim-inference/src/phi2.rs](file:///home/s8n/Lilim/crates/lilim-inference/src/phi2.rs) | **Done** | Marked superseded; retained as reference. |
| [systemd/system/lilith-ai.service](file:///home/s8n/Lilim/systemd/system/lilith-ai.service) | **Done** | Replaced `%i` template bug with `LILIM_USER_PLACEHOLDER`; raised memory limit to 6GB. |
| [lilim_core/tool_executor.py](file:///home/s8n/Lilim/lilim_core/tool_executor.py) | **Done** | Implemented `sudo_confirm` classification, `shell_command_sudo()`, `pkexec` discovery, and TTL-cached session tokens. |
| [lilim_core/zero_config_router.py](file:///home/s8n/Lilim/lilim_core/zero_config_router.py) | **Done** | Created direct HTTP zero-config provider router (Pollinations, KoboldAI, etc.). |
| **Workspace Compilation** | **Done** | Ran `cargo check --workspace`; all 106 crates compiled cleanly with zero errors. |

---

## 5. Remaining Work & Implementation Blueprint

To fully realize the autonomous agent goals, the following components must be completed in order:

### Phase 1: ReAct Agent Loop & Execution in `lilim_core/server.py`
1. **Uncap the Bash Code Block Scanner:**
   - Remove `scan_window = sanitised_reply[:400]`.
   - Scan the entire reply for ` ```bash ... ``` ` blocks.
   - Extract commands regardless of preceding introductory text.
2. **Enable Multi-Turn ReAct for Local Engine:**
   - Remove lines 1136–1142 (`if active_provider == "LOCAL": break`).
   - Allow local Phi-3.5-mini to receive observation results (`Observation: <output>`) and generate follow-up actions up to `MAX_TURNS = 10`.
3. **Handle `sudo_confirm` Stream Events:**
   - When `classification == "sudo_confirm"`:
     - Check if a valid session token exists via `executor.has_valid_sudo_token(session_id)`.
     - If token exists, execute via `executor.shell_command_sudo(command, session_id=session_id)`.
     - If no token exists and `pkexec` is not available, emit `data: {"type": "sudo_pending", "command": command, "short": short}\n\n` and yield `done` with `pending_tool: True`.
4. **Update Stop Tokens for Phi-3.5:**
   - Replace legacy checks for `"User:"` and `"Assistant:"` with `<|user|>`, `<|assistant|>`, `<|end|>`, and `<|endoftext|>`.

### Phase 2: Autonomous Persona in `lilim_core/prompt_enhancer.py`
1. **Rewrite Action Enriched Prompts:**
   - Modify `system_admin`:
     ```python
     "enrich": (
         "You are an autonomous Linux agent. Execute all necessary actions directly "
         "using bash blocks: ```bash\n<command>\n```. Lilim will run the command, "
         "inspect the output, and complete the goal. Do not instruct the user to run "
         "the command manually unless elevated confirmation is explicitly requested."
     )
     ```
   - Update `code_generation`, `devops`, and `file_management` with similar autonomous instructions.
2. **Standardize on ChatML Tagging:**
   - Ensure prompts sent to the local inference engine format user and assistant turns with `<|user|>\n...\n<|end|>\n<|assistant|>\n`.

### Phase 3: Memory Store & Context Management
1. **Expand Context Window Limits in `lilim_core/memory_manager.py`:**
   - Increase context token budget from 2048 to 4096 (Phi-3.5-mini natively supports up to 128k).
2. **Pruning for ReAct History:**
   - In long agentic tasks (5–10 turns), truncate middle tool observations to prevent context exhaustion while retaining the initial goal and the latest command outputs.

### Phase 4: Installer & Packaging Alignment
1. **Fix `local_install.sh`:**
   - Ensure line 121 substitutes `LILIM_USER_PLACEHOLDER`:
     ```bash
     sudo sed -i "s|LILIM_USER_PLACEHOLDER|${CURRENT_USER}|g" /lib/systemd/system/lilith-ai.service
     ```
2. **Fix `packaging/build_deb.sh`:**
   - In the `postinst` script (lines 222–225), update `sed` to target `LILIM_USER_PLACEHOLDER` instead of `aegon`.
3. **Verify Service Start:**
   - Execute `systemctl daemon-reload && systemctl status lilith-ai.service` to verify active/loaded state without `bad-setting`.

### Phase 5: Terminal CLI Mode (`lilim_cli.py`)
1. **Create `lilim_core/lilim_cli.py`:**
   - Command-line interface allowing direct execution: `lilim "check disk usage and clean /tmp"`.
   - Gathers current terminal context: `$PWD`, active user, `$SHELL`.
   - Streams SSE tokens directly to stdout with ANSI color-coding for tool execution blocks.
   - Prompts for sudo password directly in terminal if `sudo_confirm` is triggered.

---

## 6. Verification & Validation Checklist

### Continuation Update (October 7, 2026)

Completed in the continuation after this handoff was written:
- `lilim_core/server.py` scans the full response for fenced shell blocks, permits up to 10 ReAct turns for local and remote providers, strips Phi-3 ChatML stop markers, and prunes older agent observations while retaining the original task.
- `lilim_core/prompt_enhancer.py` now asks code-generation, system-admin, file-management, and DevOps requests to perform safe actions through Lilim's fenced-command flow.
- `lilim_core/memory_manager.py` allows up to 4096 characters of persistent memory context.
- Installer scripts replace `LILIM_USER_PLACEHOLDER` in the systemd service.
- Desktop and CLI sudo commands require explicit approval, submit credentials through the loopback Rust gateway, and keep the validated password only in the Python process's five-minute session token cache. Session reset clears the token.
- Added `lilim-cli` with streaming output, command approval, and terminal sudo password prompts. It is installed by both local and Debian package installers.
- Added a standalone `install.sh` that downloads and installs the latest published amd64 Debian package without cloning the source repository; README now provides a one-paste curl command.
- Published the standalone installer and README quick-install instructions to `main` in GitHub commits `aff0659` and `c04b179`.
- Replaced the host's stale `%i` service unit, reloaded systemd, and verified the installed unit with `systemd-analyze verify`.
- Rebuilt and installed `lilim-runtime` for this host after the prebuilt binary crashed with `SIGILL`; `lilith-ai.service` now reports `active (running)` and the Python brain reports healthy.

### MCP, BrowserOS, and Desktop Build Update (October 8, 2026)

- Added `lilim_core/mcp_manager.py` using the official MCP Python SDK with stdio and Streamable HTTP transports, private per-user configuration, HTTPS requirements for non-loopback HTTP servers, server/tool discovery, and bounded client timeouts.
- Added MCP settings and Rust gateway routes for configuring/removing servers and invoking tools. The chat loop advertises discovered tools; tools marked read-only may run directly, while other calls pause for explicit UI approval.
- Added BrowserOS support through its MCP endpoint, plus a validated BrowserOS launcher with an `xdg-open` fallback for user-supplied HTTP(S) URLs. BrowserOS endpoints are copied from its own MCP settings because the port varies by installation.
- Added the Settings UI for HTTP MCP endpoints and local stdio servers, including optional bearer auth stored in the owner-only MCP config file.
- Pinned desktop development to Node 22 (`.nvmrc`, package engine constraint, and CI node-version-file). Node 22.22.1 meets the existing Vite requirement; no major version upgrade was needed.
- Rebuilt the incomplete local `node_modules` tree with `npm ci`; `npm run build` now passes.
- Validation: 56 Python unit tests passed; Rust runtime `cargo check` passed; `cargo test -p lilim-inference` completed with no tests; Python compile, shell syntax, ToolExecutor classification, `git diff --check`, and systemd unit verification passed.
- MCP SDK 1.30.0 imports were verified in a temporary Python 3.11 environment. No BrowserOS server was running for an end-to-end browser session test.

Still outstanding:
- The MCP integration has not been tested against a live external MCP server or BrowserOS instance; server compatibility depends on the configured server's implementation and endpoint.
- RTK was not integrated: its upstream README describes a local CLI proxy that filters shell command output, not a transport for Lilim's remote HTTP model requests. Installing it would add a binary dependency without improving remote requests. [RTK README](https://github.com/rtk-ai/rtk).

Previously used verification commands:

```bash
# 1. Build Verification
cd /home/s8n/Lilim
PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig cargo check --workspace
PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig cargo test -p lilim-inference

# 2. Local Python Core Syntax & Import Check
/usr/bin/python3 -m py_compile lilim_core/server.py
/usr/bin/python3 -m py_compile lilim_core/tool_executor.py
/usr/bin/python3 -m py_compile lilim_core/prompt_enhancer.py
/usr/bin/python3 -m py_compile lilim_core/zero_config_router.py

# 3. Systemd Configuration Test
sudo cp systemd/system/lilith-ai.service /tmp/lilith-ai.test.service
sudo sed -i "s|LILIM_USER_PLACEHOLDER|$USER|g" /tmp/lilith-ai.test.service
systemd-analyze verify /tmp/lilith-ai.test.service

# 4. Tool Executor Sudo Unit Test
python3 -c "
from lilim_core.tool_executor import ToolExecutor
te = ToolExecutor()
assert te.classify_command('ls -la') == 'auto'
assert te.classify_command('sudo systemctl restart nginx') == 'sudo_confirm'
print('ToolExecutor classification tests passed!')
"
```

---

## 7. Key Environmental Notes for Developer / Model

1. **Host Architecture:** Linux x86_64 (`uname -m` = x86_64).
2. **Credentials:** Never store sudo passwords in project documentation or files. Request a password interactively only when an approved operation requires it.
3. **OpenSSL Path:** When building Rust crates that link against `openssl-sys`, provide:
   `PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig`
4. **Active Branch:** `main`.
5. **Preserved Artifact:** [lilim_upgrade_plan.md](file:///home/s8n/.gemini/antigravity-ide/brain/bc6cd3f7-54cd-4835-bee9-9ee5b086a290/lilim_upgrade_plan.md) contains historical high-level notes. This document (`MODEL_HANDOFF.md`) is the authoritative source of truth.
