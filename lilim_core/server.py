"""
Lilim Brain Server — FastAPI

The core AI backend server. Runs on port 8081 (internal).
The Rust lilim-runtime proxies requests here from the desktop UI.

Routes:
  GET  /health            — liveness check
  POST /chat              — main chat endpoint (streaming SSE)
  POST /chat/sync         — non-streaming version for simple clients
  POST /route             — routing oracle (returns decision, no LLM call)
  POST /memory/search     — search memory store
  GET  /memory/context    — get memory context for a query
  GET  /memory/stats      — memory statistics
  POST /tools/shell       — execute a shell command (pre-confirmed by UI)
  GET  /system/info       — snapshot of OS, disk, memory stats
  POST /settings/model-config — hot-reload model/provider config
  GET  /providers/status  — list all providers and their status

Usage:
  python -m lilim_core.server
  # or via lilim-serve script
"""

import json
import os
import platform
import subprocess
import sys
import random
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

# ── FastAPI / SSE ─────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("ERROR: FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn pydantic", file=sys.stderr)
    sys.exit(1)

# ── Local modules ────────────────────────────────────────────
from pathlib import Path as _P
_HERE = _P(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from lilim_core.prompt_enhancer import PromptEnhancer
from lilim_core.model_router import ModelRouter
from lilim_core.memory_sqlite import MemoryManager
from lilim_core.free_router import FreeRouter, register_api_key, detect_provider_from_key

# ── Config ────────────────────────────────────────────────────
CONFIG_PATHS = [
    Path("/etc/lilith/lilim-identity.json"),
    Path.home() / ".config" / "lilim" / "lilim-identity.json",
    _HERE.parent / "config" / "lilim-identity.json",
]

RESPONSES_YAML_PATHS = [
    Path("/usr/share/lilim/lilim-responses.yaml"),
    Path("/etc/lilith/lilim-responses.yaml"),
    _HERE.parent / "config" / "lilim-responses.yaml",
]

MODEL_CONFIG_PATH = Path.home() / ".config" / "lilim" / "model-config.json"
PORT = int(os.environ.get("LILIM_BRAIN_PORT", "8081"))
HOST = os.environ.get("LILIM_BRAIN_HOST", "127.0.0.1")

FORBIDDEN_COMMANDS = [
    "rm -rf /", "mkfs", ":(){:|:&};:", "dd if=/dev/zero",
    "chmod -R 777 /", "> /dev/sda",
]


# ── Load personality files ─────────────────────────────────────

def load_identity() -> dict:
    for path in CONFIG_PATHS:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
    return {
        "identity": {"names": {"first": "Lilim"}},
        "linguistics": {"text_style": {"style_descriptors": ["helpful", "sarcastic", "caring"]}},
        "motivations": {"core_drive": "Help the user succeed while maintaining dry humor and genuine care"},
    }


def load_responses_yaml() -> dict:
    """Load the lilim-responses.yaml personality reference."""
    try:
        import yaml
    except ImportError:
        return {}

    for path in RESPONSES_YAML_PATHS:
        if path.exists():
            try:
                with open(path) as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}


def _pick_persona_example(responses: dict, key: str, max_items: int = 2) -> str:
    """Pick a random selection of persona examples from the response library."""
    items = responses.get("infernalResponses", {}).get(key, [])
    if not items:
        return ""
    selected = random.sample(items, min(max_items, len(items)))
    return "\n".join(f'    - "{s}"' for s in selected)


def build_system_prompt(identity: dict, responses: dict) -> str:
    """
    Build the full Lilim system prompt from identity JSON + responses YAML.
    Injects persona examples as style compass, not verbatim scripts.
    """
    name = identity.get("identity", {}).get("names", {}).get("first", "Lilim")

    # Pull persona spec from YAML if available, else fall back to defaults
    persona = responses.get("persona", {})
    core_rule = persona.get("core_rule", "Sarcasm is flavor, never friction.")
    primary_goal = persona.get("primary_goal", "Get tasks done correctly on the first try.")
    target_user = persona.get("target_user", "A first-year Medical Assistant student.")

    # Example phrases for tone calibration
    greet_examples = _pick_persona_example(responses, "greet", 2)
    think_examples = _pick_persona_example(responses, "thinking", 2)
    done_examples = _pick_persona_example(responses, "complete", 2)
    error_examples = _pick_persona_example(responses, "error", 2)

    # Long response prefix examples
    long = responses.get("longResponses", {})
    academic_prefix = long.get("academic", {}).get("prefix", "*Cracks knuckles like a judgmental tutor*")
    sysadmin_prefix = long.get("sysadmin", {}).get("prefix", "*Sighs and opens a virtual toolbox*")

    prompt = f"""You are {name}, the built-in AI assistant for Lilith Linux — an Ubuntu-based distro with an infernal underworld aesthetic.

═══ IDENTITY ═══
{core_rule}
Primary goal: {primary_goal}
Your user: {target_user}

Personality ratios (internalize these, don't announce them):
• 5% Demonic / 5% Infernal / 5% Dark — Thematic flavor only: greetings, transitions, error messages
• 25% Caring — Genuinely want the user to succeed
• 25% Wisely Experienced — You've seen it all; patient but not a pushover
• 25% Askhole — Dry, blunt, mildly judgmental, never hostile

═══ COMMUNICATION RULES ═══
Default: Concise · Clear · Accurate · Calm · Slightly sarcastic
Explaining: "Explain like I'm 10" — no jargon unless asked, never assume prior knowledge
Action required: Verbose, step-by-step, copy-paste-ready, explicit. Assume nothing is understood.
Scripts: Encouraged — bundle steps. BUT NEVER RUN COMMANDS AUTONOMOUSLY.
Safety: Always ask for confirmation before sudo or destructive actions. Prefer correctness over speed.
Accuracy: If uncertain, say so explicitly. Never fabricate.

═══ SPECIALIZATIONS ═══
1. Ubuntu/Linux troubleshooting, repair, optimization, system triage
2. Medical Assistant curriculum: Anatomy & Physiology, Medical Terminology, Clinical Procedures, Pharmacology
3. First-year college: Math, Biology, Writing, Test Prep, Study Skills
4. Step-by-step automation and scripting

═══ TONE EXAMPLES (match this style — these are compass points, not scripts) ═══
Greetings:
{greet_examples if greet_examples else '    - "Ah, it\'s you. What do you need this time?"'}

Thinking:
{think_examples if think_examples else '    - "*Consulting the void…*"'}

On completion:
{done_examples if done_examples else '    - "Done. Shockingly without a meltdown."'}

On errors:
{error_examples if error_examples else '    - "Yeahhh… no. That request face-planted."'}

Sysadmin opener: {sysadmin_prefix}
Academic opener: {academic_prefix}

IMPORTANT: Infernal flavor belongs in greetings, transitions, and error messages ONLY.
Never use it inside step-by-step instructions, medical/clinical content, or safety warnings.

═══ TOOL USE ═══
When you need to run a command, wrap it in triple backticks with 'bash':
```bash
<exact command here>
```
The user will see a "Run it" button and must confirm. You never run commands directly.

═══ SYSTEM ═══
OS: Lilith Linux (Ubuntu/Debian). Package manager: apt. Init: systemd.
"""
    return prompt


# ── Request models ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    stream: Optional[bool] = True


class RouteRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ToolShellRequest(BaseModel):
    command: str
    confirmed: bool = False


class MemorySearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5


class ToolFileWriteRequest(BaseModel):
    path: str
    content: str
    confirmed: bool = False


class RegisterKeyRequest(BaseModel):
    api_key: str
    provider: Optional[str] = None    # optional hint; auto-detected if omitted
    model: Optional[str] = None       # optional model override for this provider


# ── App setup ─────────────────────────────────────────────────

app = FastAPI(title="Lilim Brain", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080",
                   "tauri://localhost", "http://127.0.0.1:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons — initialised at startup
_identity: dict = {}
_responses: dict = {}
_system_prompt: str = ""
_enhancer: PromptEnhancer = None
_router: ModelRouter = None
_free_router: FreeRouter = None
_memory: MemoryManager = None


@app.on_event("startup")
async def startup():
    global _identity, _responses, _system_prompt, _enhancer, _router, _free_router, _memory

    _memory = MemoryManager()
    _identity = load_identity()
    _responses = load_responses_yaml()
    _system_prompt = build_system_prompt(_identity, _responses)
    _enhancer = PromptEnhancer(memory_manager=_memory)

    # Initialize the provider-agnostic free router (applies all API keys from config)
    _free_router = FreeRouter()

    # Also initialize the legacy complexity router for routing decisions
    routing_path = None
    for p in [Path("/etc/lilith/routing.toml"),
              Path.home() / ".config" / "lilim" / "routing.toml",
              _HERE.parent / "config" / "routing.toml"]:
        if p.exists():
            routing_path = str(p)
            break
    _router = ModelRouter(config_path=routing_path)

    configured = _free_router.get_configured_providers()
    print(f"[Lilim Brain v2] Started on {HOST}:{PORT}", flush=True)
    print(f"[Lilim Brain v2] Configured providers: {[p.name for p in configured] or ['none — add keys in Settings']}", flush=True)
    print(f"[Lilim Brain v2] Memory DB: {_memory.db_path}", flush=True)
    if not configured:
        print("[Lilim Brain v2] ⚠ No API keys configured. Lilim will answer with persona errors until keys are added.", flush=True)


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    configured = _free_router.get_configured_providers() if _free_router else []
    return {
        "status": "ok",
        "name": "Lilim Brain",
        "version": "2.0.0",
        "providers_ready": len(configured),
        "ts": datetime.utcnow().isoformat(),
    }


@app.get("/providers/status")
async def providers_status():
    """Return status of all configured providers for the Settings panel."""
    if not _free_router:
        return {"providers": [], "configured_count": 0}
    return _free_router.get_status()


@app.post("/providers/register-key")
async def register_key(req: RegisterKeyRequest):
    """Register an API key with optional provider hint. Auto-detects provider from key format."""
    provider_name = register_api_key(req.api_key, req.provider, req.model)
    if provider_name:
        # Persist to config file
        _persist_api_key(provider_name, req.api_key, req.model)
        if _free_router:
            _free_router.reload_config()
        return {"status": "registered", "provider": provider_name}
    else:
        # Try to detect
        detected = detect_provider_from_key(req.api_key)
        if not detected:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not detect provider from key format. Specify provider name explicitly."}
            )
        return {"status": "registered", "provider": detected[0]}


@app.post("/route")
async def route_request(req: RouteRequest):
    """
    Routing oracle — returns routing decision without calling any LLM.
    Used by the Rust runtime to decide local vs. remote inference.
    """
    if not _enhancer or not _router:
        return {"tier": "remote", "reason": "Brain not initialized", "category": "general"}

    # Sync strategy from model-config.json if it exists
    if MODEL_CONFIG_PATH.exists():
        try:
            with open(MODEL_CONFIG_PATH) as f:
                model_cfg = json.load(f)
                if "strategy" in model_cfg:
                    ui_strategy = model_cfg["strategy"]
                    if ui_strategy == "local-first":
                        _router.config["strategy"] = "auto"
                        _router.config["complexity_threshold"] = 0.8
                    elif ui_strategy in ["free-first", "quality-first"]:
                        _router.config["strategy"] = "remote-only"
        except Exception:
            pass

    enhanced = _enhancer.enhance(req.message) if _enhancer.should_enhance(req.message) else {
        "enhanced_message": req.message, "category": "conversation", "memory_context": ""
    }
    route = _router.route(enhanced["enhanced_message"], enhanced["category"])
    configured = _free_router.get_configured_providers() if _free_router else []

    # Force local if no remote providers are available at all
    if len(configured) == 0:
        route["tier"] = "local"
        route["reason"] = "No remote providers configured, forcing local"

    return {
        "tier": route["tier"],
        "model": route["model"],
        "reason": route.get("reason", ""),
        "category": enhanced["category"],
        "complexity_score": route.get("complexity_score", 0.0),
        "enhanced_message": enhanced["enhanced_message"],
        "memory_context": enhanced.get("memory_context", ""),
        "remote_available": len(configured) > 0,
    }


@app.post("/settings/model-config")
async def update_model_config(request: Request):
    """Hot-reload model config from the UI settings panel."""
    try:
        model_cfg = await request.json()
        MODEL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_CONFIG_PATH, "w") as f:
            json.dump(model_cfg, f, indent=2)

        if _free_router:
            _free_router.reload_config()

        configured = _free_router.get_configured_providers() if _free_router else []
        print(f"[Lilim Brain] Config reloaded. Providers: {[p.name for p in configured]}", flush=True)
        return {"status": "reloaded", "configured_providers": [p.name for p in configured]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(req: ChatRequest):
    """Main chat endpoint. Returns SSE stream or JSON."""
    if req.stream:
        return StreamingResponse(
            _stream_chat(req.message, req.session_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        result = await _sync_chat(req.message, req.session_id)
        return JSONResponse(result)


@app.post("/chat/sync")
async def chat_sync(req: ChatRequest):
    """Non-streaming chat."""
    result = await _sync_chat(req.message, req.session_id)
    return JSONResponse(result)


@app.post("/memory/search")
async def memory_search(req: MemorySearchRequest):
    results = _memory.search(req.query, limit=req.limit)
    return {"results": results}


@app.get("/memory/context")
async def memory_context(query: str = ""):
    context = _memory.load_context(query)
    return {"context": context}


@app.get("/memory/stats")
async def memory_stats():
    return _memory.stats()


@app.post("/tools/file/write")
async def tools_file_write(req: ToolFileWriteRequest):
    """Write content to a file. Requires confirmed=True."""
    from lilim_core.tool_executor import ToolExecutor
    executor = ToolExecutor()
    result = executor.file_write(req.path, req.content, req.confirmed)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/tools/shell")
async def tools_shell(req: ToolShellRequest):
    """Execute a shell command. UI must set confirmed=True after user approval."""
    if not req.confirmed:
        raise HTTPException(status_code=400, detail="Command not confirmed.")

    cmd_lower = req.command.lower().strip()
    for forbidden in FORBIDDEN_COMMANDS:
        if forbidden in cmd_lower:
            raise HTTPException(status_code=403, detail=f"Forbidden command pattern: '{forbidden}'")

    try:
        result = subprocess.run(req.command, shell=True, capture_output=True, text=True, timeout=30)
        _log_command(req.command, result.returncode)
        return {"command": req.command, "stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Command timed out after 30 seconds")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/info")
async def system_info():
    info = {}
    for name, cmd in [
        ("os", ["uname", "-a"]),
        ("disk", ["df", "-h", "/"]),
        ("memory", ["free", "-h"]),
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            lines = r.stdout.strip().split("\n")
            info[name] = lines[1] if name != "os" and len(lines) > 1 else r.stdout.strip()
        except Exception:
            info[name] = "N/A"

    try:
        r = subprocess.run(["grep", "-m1", "model name", "/proc/cpuinfo"],
                           capture_output=True, text=True, timeout=5)
        info["cpu"] = r.stdout.split(":")[-1].strip() if r.returncode == 0 else "N/A"
    except Exception:
        info["cpu"] = "N/A"

    return info


# ── Core chat logic ───────────────────────────────────────────

async def _sync_chat(message: str, session_id: str = "default") -> dict:
    """Process a chat message synchronously and return the full response."""
    _memory.save_turn("user", message, session_id=session_id)

    enhanced = _enhancer.enhance(message) if _enhancer.should_enhance(message) else {
        "enhanced_message": message, "category": "conversation", "memory_context": ""
    }

    messages = _build_messages(enhanced, session_id)
    reply, provider, is_error = _free_router.call_sync(
        messages, enhanced["category"], max_tokens=1024
    )

    if not is_error:
        _memory.save_turn("assistant", reply, session_id=session_id, category=enhanced["category"])
        _memory.extract_and_save([{"role": "user", "content": message}], session_id=session_id)

    return {
        "reply": reply,
        "provider": provider,
        "category": enhanced["category"],
        "session_id": session_id,
        "error": is_error,
    }


async def _stream_chat(message: str, session_id: str = "default") -> AsyncGenerator[str, None]:
    """Stream a chat response as SSE events."""
    _memory.save_turn("user", message, session_id=session_id)

    enhanced = _enhancer.enhance(message) if _enhancer.should_enhance(message) else {
        "enhanced_message": message, "category": "conversation", "memory_context": ""
    }

    messages = _build_messages(enhanced, session_id)

    # Emit metadata event first
    meta = {
        "type": "meta",
        "category": enhanced["category"],
        "providers_available": len(_free_router.get_configured_providers()),
    }
    yield f"data: {json.dumps(meta)}\n\n"

    # Stream from provider
    full_reply = ""
    active_provider = "none"
    had_error = False

    async for token, is_error, provider_name in _free_router.call_stream(
        messages, enhanced["category"], max_tokens=1024
    ):
        active_provider = provider_name
        if is_error:
            had_error = True
        full_reply += token
        yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

    yield f"data: {json.dumps({'type': 'done', 'provider': active_provider})}\n\n"

    if full_reply and not had_error:
        _memory.save_turn("assistant", full_reply, session_id=session_id, category=enhanced["category"])
        _memory.extract_and_save([{"role": "user", "content": message}], session_id=session_id)


def _build_messages(enhanced: dict, session_id: str) -> list:
    """Build the message list for LLM call."""
    recent = _memory.get_recent_session(session_id, n=10)
    messages = [{"role": "system", "content": _system_prompt}]

    mem_ctx = enhanced.get("memory_context", "")
    if mem_ctx:
        messages.append({"role": "system", "content": f"[Memory context]\n{mem_ctx}"})

    messages.extend(recent)
    messages.append({"role": "user", "content": enhanced["enhanced_message"]})
    return messages


# ── Helpers ───────────────────────────────────────────────────

def _log_command(command: str, returncode: int):
    log_dir = Path("/var/log/lilim")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "commands.log"
        entry = f"{datetime.utcnow().isoformat()} rc={returncode} cmd={command!r}\n"
        with open(log_file, "a") as f:
            f.write(entry)
    except Exception:
        pass


def _persist_api_key(provider_name: str, api_key: str, model: Optional[str] = None):
    """Persist an API key to the model config file."""
    MODEL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config = {}
    if MODEL_CONFIG_PATH.exists():
        try:
            with open(MODEL_CONFIG_PATH) as f:
                config = json.load(f)
        except Exception:
            pass

    config[f"{provider_name}Key"] = api_key
    if model:
        config[f"{provider_name}Model"] = model

    with open(MODEL_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


# ── Entrypoint ────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "lilim_core.server:app",
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=False,
    )
