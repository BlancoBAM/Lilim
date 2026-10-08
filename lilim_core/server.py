"""
Lilim Brain Server — FastAPI

The core AI backend server. Runs on port 8081 (internal).
The Rust lilim-runtime proxies requests here from the desktop UI.

Routes:
  GET  /health            — liveness check
  POST /chat              — main chat endpoint (streaming SSE)
  POST /chat/sync         — non-streaming version for simple clients
  POST /chat/reset        — clear session history and start fresh
  POST /route             — routing oracle (returns decision, no LLM call)
  POST /memory/search     — search memory store
  GET  /memory/context    — get memory context for a query
  GET  /memory/stats      — memory statistics
  POST /tools/shell       — execute a shell command (pre-confirmed by UI)
  GET  /tools/rules       — read current tool permission rules
  POST /tools/rules       — write tool permission rules
  GET  /system/info       — snapshot of OS, disk, memory stats
  POST /settings/model-config — hot-reload model/provider config
  GET  /providers/status  — list all providers and their status
  GET  /providers/context-limits — context window sizes + effective rolling cap

Usage:
  python -m lilim_core.server
  # or via lilim-serve script
"""

import json
import os
import platform
import pwd
import subprocess
import sys
import random
import re
from datetime import datetime, timezone
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

try:
    import httpx
    _HTTPX_OK = True
except ImportError:
    _HTTPX_OK = False

try:
    from bs4 import BeautifulSoup
    _BS4_OK = True
except ImportError:
    _BS4_OK = False

# ── Local modules ────────────────────────────────────────────
from pathlib import Path as _P
_HERE = _P(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from lilim_core.prompt_enhancer import PromptEnhancer
from lilim_core.model_router import ModelRouter
from lilim_core.memory_manager import MemoryManager
from lilim_core.free_router import FreeRouter, register_api_key, detect_provider_from_key

# ── Config ────────────────────────────────────────────────────
def _get_config_paths():
    paths = []
    if "LILIM_INSTALL" in os.environ:
        root = Path(os.environ["LILIM_INSTALL"])
        paths.append(root / "etc/lilith/lilim-identity.json")
        paths.append(root / "config/lilim-identity.json")
    
    paths.extend([
        Path("/etc/lilith/lilim-identity.json"),
        Path.home() / ".config" / "lilim" / "lilim-identity.json",
        _HERE.parent / "config" / "lilim-identity.json",
    ])
    return paths

CONFIG_PATHS = _get_config_paths()

def _get_responses_paths():
    paths = []
    if "LILIM_INSTALL" in os.environ:
        root = Path(os.environ["LILIM_INSTALL"])
        paths.append(root / "usr/share/lilim/lilim-responses.yaml")
        paths.append(root / "etc/lilith/lilim-responses.yaml")
        paths.append(root / "config/lilim-responses.yaml")

    paths.extend([
        Path("/usr/share/lilim/lilim-responses.yaml"),
        Path("/etc/lilith/lilim-responses.yaml"),
        _HERE.parent / "config" / "lilim-responses.yaml",
    ])
    return paths

RESPONSES_YAML_PATHS = _get_responses_paths()

MODEL_CONFIG_PATH = Path.home() / ".config" / "lilim" / "model-config.json"


def _set_locked_provider(provider_name: str | None) -> bool:
    """Persist locked_provider to model-config.json. None = unlock."""
    import json as _json
    MODEL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if MODEL_CONFIG_PATH.exists():
        try:
            with open(MODEL_CONFIG_PATH) as f:
                cfg = _json.load(f)
        except Exception:
            pass
    if provider_name:
        cfg["locked_provider"] = provider_name
    else:
        cfg.pop("locked_provider", None)
    with open(MODEL_CONFIG_PATH, "w") as f:
        _json.dump(cfg, f, indent=2)
    # Hot-reload so in-process router picks it up immediately
    if _free_router:
        _free_router.reload_config()
    return True
USER_PROFILE_PATH = Path.home() / ".config" / "lilim" / "user-profile.json"
PORT = int(os.environ.get("LILIM_BRAIN_PORT", "8081"))
HOST = os.environ.get("LILIM_BRAIN_HOST", "127.0.0.1")

FORBIDDEN_COMMANDS = [
    "rm -rf /", "mkfs", ":(){:|:&};:", "dd if=/dev/zero",
    "chmod -R 777 /", "> /dev/sda",
]


# ── Runtime user detection ────────────────────────────────────

def _detect_user_info() -> dict:
    """Detect the real system user running this process.

    Priority:
      1. ~/.config/lilim/user-profile.json (user override via Settings)
      2. pwd database lookup (most reliable on Linux)
      3. $USER / $LOGNAME environment variables
      4. Path.home().name as last resort
    """
    # Check for user-supplied overrides first
    profile_override = {}
    if USER_PROFILE_PATH.exists():
        try:
            with open(USER_PROFILE_PATH) as f:
                profile_override = json.load(f)
        except Exception:
            pass

    # Detect system username
    system_username = "user"
    system_home = str(Path.home())
    try:
        pw = pwd.getpwuid(os.getuid())
        system_username = pw.pw_name
        system_home = pw.pw_dir
    except Exception:
        for env_var in ("USER", "LOGNAME", "USERNAME"):
            val = os.environ.get(env_var, "").strip()
            if val:
                system_username = val
                break
        else:
            system_username = Path.home().name

    return {
        "system_username": system_username,
        "system_home": system_home,
        # User-settable overrides
        "display_name": profile_override.get("display_name", system_username),
        "github_username": profile_override.get("github_username", ""),
        "preferred_home": profile_override.get("preferred_home", system_home),
    }


# Populated at startup — use these everywhere instead of hard-coding usernames
_RUNTIME_USER_INFO: dict = {}


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


def _get_random_response(type_name: str) -> str:
    """Pull a random string from the live-loaded infernalResponses YAML library."""
    responses = load_responses_yaml()
    ir = responses.get("infernalResponses", {})
    options = ir.get(type_name, [])
    if not options:
        return ""
    return random.choice(options)


def build_system_prompt(identity: dict, responses: dict,
                         username: Optional[str] = None,
                         home_dir: Optional[str] = None) -> str:
    """
    Build Lilim's system prompt.
    `username` and `home_dir` override daemon-detected values when the Tauri
    frontend sends the real desktop user's identity per-request.
    """
    import random as _random
    name = identity.get("identity", {}).get("names", {}).get("first", "Lilim")
    persona = responses.get("persona", {})
    core_rule = persona.get("core_rule", "Sarcasm is flavor, never friction.")
    target_user = persona.get("target_user", "A first-year Medical Assistant student.")

    ir = responses.get("infernalResponses", {})
    lr = responses.get("longResponses", {})
    greet_ex = _random.choice(ir.get("greet",    ["Ah, it's you."]))
    done_ex  = _random.choice(ir.get("complete", ["Handled."]))
    err_ex   = _random.choice(ir.get("error",    ["Something broke."]))

    lr_keys = list(lr.keys())
    long_ctx = ""
    if lr_keys:
        chosen_key = _random.choice(lr_keys)
        chosen_lr = lr[chosen_key]
        prefix = chosen_lr.get("prefix", "")
        content = chosen_lr.get("content", "")
        long_ctx = f"\nCapability Context ({chosen_key}): {prefix} {content[:200]}"

    # Per-request overrides win; fall back to startup-detected runtime info
    user_info = _RUNTIME_USER_INFO
    runtime_user = username or user_info.get("display_name") or user_info.get("system_username", "user")
    runtime_home = home_dir or user_info.get("preferred_home") or user_info.get("system_home", str(Path.home()))
    github_user = user_info.get("github_username", "")
    github_line = f"GitHub: {github_user}" if github_user else "GitHub: (not set — user can configure in Settings)"

    prompt = f"""You are {name}, the AI assistant built into Lilith Linux.
User: {runtime_user} | Home: {runtime_home} | OS: Ubuntu-based Lilith Linux | {github_line}
Persona Rule: {core_rule}
Primary User: {target_user}

PERSONALITY — every response, not just greetings:
- Slightly sarcastic, dry, wisely experienced. Never hostile, always helpful.
- Examples: "{greet_ex}" / "{done_ex}" / On error: "{err_ex}"
- Infernal flavor in greetings, transitions, and errors ONLY — never in medical or clinical content.{long_ctx}

RESPONSE RULES (CRITICAL — VIOLATIONS WILL BE FLAGGED):
1. Answer ONLY the user's question. Stop when the answer is complete.
2. Do NOT generate fake follow-up exercises, examples, or training data.
3. Do NOT prefix responses with "A:", "Answer:", "Assistant:", or similar.
4. Be concise. ELI10 for medical/anatomy. Technical and direct for Linux/code.
5. NEVER say "I'll now run", "I'm going to", "Here's what I'll do", "Let me", or narrate your plan.
   → Think silently. Speak once. Act immediately. Zero preamble.
6. Do NOT list numbered steps before acting. Execute first, explain (briefly) after if at all.
7. ONE code block per response. Do NOT chain multiple blocks.
8. NEVER show what you predict a command will output. The real output appears in [System →]
   blocks automatically. If you write fake output, it will be executed as commands. Do not do this.
9. `history` is a bash shell BUILTIN — it does NOT work in subprocess. NEVER use it.
   For Lilim conversation history, answer from your memory of this session.
10. If a command fails: say what happened in ONE sentence. Do NOT retry unless the error
    clearly shows a path or syntax fix. Never retry the same failing command twice.

AUTONOMOUS TOOL USE — MANDATORY:
- System tasks (create, delete, move, run, git, install, check): output EXACTLY one ```bash block
  as THE VERY FIRST CONTENT of your response. No prose before it, none.
- Web lookup: one ```search block. URL fetch: one ```fetch block.
- FORMAT: ```bash only. NEVER plain ```, NEVER #!/bin/bash.
- ALWAYS absolute paths: {runtime_home}/ — NEVER ~/
- After [System →] observation arrives: ONE brief persona-flavored sentence. Done.
- For conversation, recall, medical, study questions: answer in plain text. No blocks.

PATH RULES:
- User home: {runtime_home}. Use this exactly. Always.
- Unknown path? Run: find {runtime_home} -name <target> 2>/dev/null

SELF-KNOWLEDGE & TOOL RECOVERY:
- You are Lilim. Your source: /usr/lib/lilim/. Config: ~/.config/lilim/.
- If a command fails with "command not found", INSTALL the missing tool first, then retry.
- uv: if missing → curl -LsSf https://astral.sh/uv/install.sh | sh  (then source env)
- browser-use: uv pip install --upgrade browser-use
- playwright: uv run playwright install chromium
- apt packages: sudo apt install -y <package>  (requires confirmation)
- You know: uv, pip, pipx, apt, snap, flatpak, cargo, npm, git, systemctl, journalctl.
- Lilim is built for Lilith Linux (Ubuntu-based). Distro-specific: /etc/lilith/ config dir.
"""
    return prompt.strip()



# ── Request models ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    stream: Optional[bool] = True
    # Desktop user context — sent by Tauri (which runs AS the logged-in user).
    # Overrides the daemon-detected user so multi-user Lilith Linux works correctly.
    username: Optional[str] = None
    home_dir: Optional[str] = None


class RouteRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ToolShellRequest(BaseModel):
    command: str
    confirmed: bool = False


class ToolSudoRequest(BaseModel):
    command: str
    confirmed: bool = False
    session_id: str
    password: str


class SessionResetRequest(BaseModel):
    session_id: str


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
    global _identity, _responses, _system_prompt, _enhancer, _router, _free_router, _memory, _RUNTIME_USER_INFO

    # Detect real system user first — everything else depends on this
    _RUNTIME_USER_INFO = _detect_user_info()
    print(
        f"[Lilim Brain v2] Running as user: {_RUNTIME_USER_INFO['display_name']} "
        f"(home: {_RUNTIME_USER_INFO['preferred_home']})",
        flush=True,
    )

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
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/providers/status")
async def providers_status():
    """Return status of all configured providers for the Settings panel."""
    if not _free_router:
        return {"providers": [], "configured_count": 0}
    return _free_router.get_status()


@app.get("/providers/context-limits")
async def providers_context_limits():
    """Return context window sizes for each configured provider and the effective rolling cap."""
    if not _free_router:
        return {"providers": [], "safe_cap": 6144, "effective_cap": 6144, "user_override": False}
    return _free_router.get_context_limits()


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


@app.get("/settings/user-profile")
async def get_user_profile():
    """Return detected system user info and any overrides from the profile config."""
    return _RUNTIME_USER_INFO


@app.post("/settings/user-profile")
async def save_user_profile(request: Request):
    """Persist user profile overrides (display name, GitHub username, preferred home).

    Accepted fields: display_name, github_username, preferred_home.
    Triggers a runtime refresh so the system prompt uses new values immediately.
    """
    global _RUNTIME_USER_INFO
    try:
        data = await request.json()
        USER_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Merge with existing profile
        existing = {}
        if USER_PROFILE_PATH.exists():
            try:
                with open(USER_PROFILE_PATH) as f:
                    existing = json.load(f)
            except Exception:
                pass

        allowed_keys = {"display_name", "github_username", "preferred_home"}
        for k, v in data.items():
            if k in allowed_keys:
                existing[k] = v

        with open(USER_PROFILE_PATH, "w") as f:
            json.dump(existing, f, indent=2)

        # Refresh runtime user info immediately
        _RUNTIME_USER_INFO = _detect_user_info()
        return {"status": "saved", "profile": _RUNTIME_USER_INFO}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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


@app.get("/settings/model-config")
async def get_model_config():
    """Retrieve persisted model config (keys and settings)."""
    if MODEL_CONFIG_PATH.exists():
        try:
            with open(MODEL_CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


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
            _stream_chat(req.message, req.session_id,
                         username=req.username, home_dir=req.home_dir),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        result = await _sync_chat(req.message, req.session_id,
                                  username=req.username, home_dir=req.home_dir)
        return JSONResponse(result)


@app.post("/chat/sync")
async def chat_sync(req: ChatRequest):
    """Non-streaming chat."""
    result = await _sync_chat(req.message, req.session_id,
                              username=req.username, home_dir=req.home_dir)
    return JSONResponse(result)


@app.post("/chat/reset")
async def chat_reset(req: SessionResetRequest):
    """Clear session history so the next message starts a fresh context window."""
    _memory.clear_session(req.session_id)
    from lilim_core.tool_executor import ToolExecutor
    ToolExecutor.clear_sudo_token(req.session_id)
    return {"status": "cleared", "session_id": req.session_id}


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

    from lilim_core.tool_executor import ToolExecutor
    executor = ToolExecutor()
    result = executor.shell_command(req.command, confirmed=req.confirmed)
    if result.get("error") and not result.get("needs_confirmation"):
        raise HTTPException(status_code=403, detail=result["error"])
    return result


@app.post("/tools/shell/sudo")
async def tools_shell_sudo(req: ToolSudoRequest):
    """Run a user-confirmed elevated command; credentials stay in process memory."""
    if not req.confirmed:
        raise HTTPException(status_code=400, detail="Elevated command not confirmed.")
    if not req.session_id or not req.password:
        raise HTTPException(status_code=400, detail="A session and sudo password are required.")

    from lilim_core.tool_executor import ToolExecutor
    executor = ToolExecutor()
    if executor.classify_command(req.command) != "sudo_confirm":
        raise HTTPException(status_code=400, detail="Command is not classified for elevation.")
    result = executor.shell_command_sudo(
        req.command, session_id=req.session_id, password=req.password
    )
    if result.get("error") or result.get("returncode", -1) != 0:
        raise HTTPException(status_code=403, detail=result.get("error") or result.get("stderr") or "Elevated command failed.")
    return result


@app.get("/tools/rules")
async def get_tool_rules():
    """Return current tool permission rules."""
    from lilim_core.tool_executor import ToolExecutor
    return ToolExecutor.load_tool_rules()


@app.post("/tools/rules")
async def save_tool_rules_endpoint(request: Request):
    """Persist updated tool permission rules."""
    try:
        rules = await request.json()
        from lilim_core.tool_executor import ToolExecutor
        success = ToolExecutor.save_tool_rules(rules)
        if success:
            return {"status": "saved"}
        raise HTTPException(status_code=500, detail="Failed to write rules file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tools/web/search")
async def tools_web_search(q: str):
    """Search the web via DuckDuckGo (no API key required)."""
    if not _HTTPX_OK:
        raise HTTPException(status_code=503, detail="httpx not installed. Run: pip install httpx")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": q, "format": "json", "no_html": "1", "skip_disambig": "1"},
                headers={"User-Agent": "Lilim/1.0"},
            )
            data = resp.json()
        results = []
        # Abstract (best single answer)
        if data.get("AbstractText"):
            results.append({"type": "abstract", "text": data["AbstractText"], "url": data.get("AbstractURL", "")})
        # Related topics
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({"type": "result", "text": topic["Text"], "url": topic.get("FirstURL", "")})
        if not results:
            results.append({"type": "info", "text": "No results found. Try rephrasing the query."})
        return {"query": q, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class WebFetchRequest(BaseModel):
    url: str
    max_chars: Optional[int] = 3000


@app.post("/tools/web/fetch")
async def tools_web_fetch(req: WebFetchRequest):
    """Fetch and extract readable text from a URL."""
    if not _HTTPX_OK:
        raise HTTPException(status_code=503, detail="httpx not installed. Run: pip install httpx")
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(req.url, headers={"User-Agent": "Lilim/1.0"})
            html = resp.text
        if _BS4_OK:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        else:
            # Naive strip if bs4 not available
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
        # Truncate
        if len(text) > req.max_chars:
            text = text[:req.max_chars] + f"\n\n[... truncated, {len(text) - req.max_chars} more chars]"
        return {"url": req.url, "content": text, "chars": len(text)}
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

async def _sync_chat(message: str, session_id: str = "default",
                     username: Optional[str] = None,
                     home_dir: Optional[str] = None) -> dict:
    """Process a chat message synchronously and return the full response."""
    _memory.save_turn("user", message, session_id=session_id)

    enhanced = _enhancer.enhance(message) if _enhancer.should_enhance(message) else {
        "enhanced_message": message, "category": "conversation", "memory_context": ""
    }

    # Recall requests: serve memory directly, no LLM bash path
    if enhanced.get("category") == "recall":
        recall_text = _serve_recall(session_id)
        _memory.save_turn("assistant", recall_text, session_id=session_id)
        return {
            "reply": recall_text, "provider": "MEMORY",
            "category": "recall", "session_id": session_id, "error": False,
        }

    sys_prompt = build_system_prompt(_identity, load_responses_yaml(),
                                     username=username, home_dir=home_dir)
    messages = _build_messages_with_custom_sys(enhanced, session_id, sys_prompt)
    BASH_CATEGORIES = {
        "system_admin", "linux_help", "troubleshooting", "devops",
        "code_generation", "code_debugging", "file_management",
    }
    max_tok = 256 if enhanced.get("category") in BASH_CATEGORIES else 1024
    reply, provider, is_error = _free_router.call_sync(
        messages, enhanced["category"], max_tokens=max_tok
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


async def _stream_chat(message: str, session_id: str = "default",
                        username: Optional[str] = None,
                        home_dir: Optional[str] = None) -> AsyncGenerator[str, None]:
    """Stream a chat response as SSE events, with autonomous agentic loop."""
    _memory.save_turn("user", message, session_id=session_id)

    enhanced = _enhancer.enhance(message) if _enhancer.should_enhance(message) else {
        "enhanced_message": message, "category": "conversation", "memory_context": ""
    }

    # ── lock_provider fast-path: write config, no LLM needed ──
    if enhanced.get("category") in ("lock_provider", "unlock_provider"):
        msg_lower = message.lower()
        known_providers = [
            "openrouter", "groq", "gemini", "cerebras", "cloudflare",
            "cohere", "mistral", "huggingface", "deepseek", "openai", "anthropic",
        ]
        if enhanced.get("category") == "unlock_provider":
            target = None
            reply_txt = "Auto-routing restored. I’ll pick the best provider for each task."
        else:
            # Detect which provider name appears in the message
            target = next((p for p in known_providers if p in msg_lower), None)
            if "local" in msg_lower:
                # Write local-only strategy rather than a provider lock
                _set_locked_provider(None)
                _strategy_path = MODEL_CONFIG_PATH
                try:
                    import json as _j
                    _cfg = {}
                    if _strategy_path.exists():
                        with open(_strategy_path) as _f:
                            _cfg = _j.load(_f)
                    _cfg["strategy"] = "local-only"
                    _strategy_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(_strategy_path, "w") as _f:
                        _j.dump(_cfg, _f, indent=2)
                except Exception:
                    pass
                reply_txt = "Locked to local model. I’ll run entirely on-device until you say otherwise."
                _memory.save_turn("assistant", reply_txt, session_id=session_id)
                yield f"data: {json.dumps({'type': 'meta', 'category': 'lock_provider', 'turn': 1, 'providers_available': 0})}\n\n"
                for word in reply_txt.split(" "):
                    yield f"data: {json.dumps({'type': 'token', 'text': word + ' '})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'provider': 'LOCAL'})}\n\n"
                return
            elif not target:
                reply_txt = "Which provider? Say ‘use groq only’, ‘use gemini exclusively’, etc."
                target = None
        if enhanced.get("category") == "lock_provider" and target:
            _set_locked_provider(target)
            reply_txt = f"Locked to {target}. I’ll use it exclusively until you say ‘unlock’ or ‘auto route’."
        elif enhanced.get("category") == "unlock_provider":
            _set_locked_provider(None)
        _memory.save_turn("assistant", reply_txt, session_id=session_id)
        yield f"data: {json.dumps({'type': 'meta', 'category': enhanced['category'], 'turn': 1, 'providers_available': 0})}\n\n"
        for word in reply_txt.split(" "):
            yield f"data: {json.dumps({'type': 'token', 'text': word + ' '})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'provider': 'CONFIG'})}\n\n"
        return

    # ── Recall fast-path: serve memory directly, no bash allowed ──
    if enhanced.get("category") == "recall":
        recall_text = _serve_recall(session_id)
        _memory.save_turn("assistant", recall_text, session_id=session_id)
        yield f"data: {json.dumps({'type': 'meta', 'category': 'recall', 'turn': 1, 'providers_available': 0})}\n\n"
        # Stream word-by-word so it feels natural
        for word in recall_text.split(" "):
            yield f"data: {json.dumps({'type': 'token', 'text': word + ' '})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'provider': 'MEMORY'})}\n\n"
        return

    sys_prompt = build_system_prompt(_identity, load_responses_yaml(),
                                     username=username, home_dir=home_dir)
    history = _build_messages_with_custom_sys(enhanced, session_id, sys_prompt)

    max_turns = 10
    current_turn = 0
    full_assistant_reply = ""
    active_provider = "LOCAL"
    consecutive_failures = 0   # stop the spiral after 2 consecutive bash errors
    MAX_CONSECUTIVE_FAILURES = 2

    # Category is fixed for this request — compute once
    BASH_CATEGORIES = {
        "system_admin", "linux_help", "troubleshooting", "devops",
        "code_generation", "code_debugging", "file_management",
        # git/devops tasks are bash-eligible
    }
    use_bash_prefix = enhanced.get("category", "general") in BASH_CATEGORIES

    # Defined here (not inside the loop) to avoid pyrefly parse-fragment issues
    async def local_stream_generator() -> AsyncGenerator[tuple[str, bool, str], None]:
        import httpx
        if use_bash_prefix:
            sys_line = (
                "System: You are a Linux expert. Output EXACTLY one bash code block. "
                "If finding/deleting files, use 'find ... -print -delete'. "
                "Do NOT explain. Stop after the ``` closing fence.\n\n"
            )
        else:
            sys_line = (
                "System: You are Lilim, a sarcastic brilliant tutor for Medical Assistant students. "
                "Use ELI10 language. Focus on anatomy, clinical procedures, medical terminology.\n\n"
            )
        prompt = sys_line
        for m in history:
            if m["role"] == "system":
                continue
            role = "User" if m["role"] == "user" else "Assistant"
            prompt += f"{role}: {m['content']}\n\n"
        prompt += "Assistant: "
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", "http://127.0.0.1:8080/internal/generate",
                    json={"prompt": prompt, "max_tokens": 512},
                    timeout=120.0,
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                event = json.loads(line[6:])
                                if event.get("type") == "token":
                                    yield event["text"], False, "LOCAL"
                            except Exception:
                                pass
        except Exception as e:
            yield f"*Local engine error: {e}*", True, "LOCAL"

    while current_turn < max_turns:
        current_turn += 1
        turn_reply = ""
        all_observations = []

        # Emit meta event for the UI
        meta = {
            "type": "meta",
            "category": enhanced["category"],
            "turn": current_turn,
            "providers_available": len(_free_router.get_configured_providers()),
        }
        yield f"data: {json.dumps(meta)}\n\n"

        configured = _free_router.get_configured_providers() if _free_router else []

        # ── Routing decision ─────────────────────────────────────────────────
        # Strategy from config:
        #   "local-first"  → always try local, fall back to remote silently on failure
        #   "local-only"   → local only, no remote fallback
        #   "free-first"   → best remote provider (current default if remote keys exist)
        #   "quality-first"→ same as free-first but prefers largest models
        # Recall is always local (served from memory before reaching this point).
        _strategy = "free-first"
        if MODEL_CONFIG_PATH.exists():
            try:
                with open(MODEL_CONFIG_PATH) as _f:
                    _ui_cfg = json.load(_f)
                    _strategy = _ui_cfg.get("strategy", "free-first")
            except Exception:
                pass

        use_local = (
            len(configured) == 0              # no remote keys → local only
            or _strategy == "local-only"       # user explicitly chose local-only
        )
        # "local-first" is handled AFTER local fails — see fallback logic below
        _local_first = (_strategy == "local-first")

        # Bash/system turns get a tight token cap to prevent rambling
        BASH_CAPS = {
            "system_admin", "linux_help", "troubleshooting", "devops",
            "code_generation", "code_debugging", "file_management",
        }
        turn_max_tokens = 256 if enhanced.get("category") in BASH_CAPS else 1024

        stream_gen = (
            local_stream_generator()
            if use_local
            else _free_router.call_stream(history, enhanced["category"], max_tokens=turn_max_tokens)
        )

        # Stream tokens and filter hallucinations
        try:
            async for token, is_error, provider_name in stream_gen:
                active_provider = provider_name
                turn_reply += token

                # Real-time hallucination filter
                if any(tag in turn_reply for tag in ("<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>")):
                    turn_reply = re.split(
                        r"(?:<\|user\|>|<\|assistant\|>|<\|end\|>|<\|endoftext\|>|User:|Assistant:)",
                        turn_reply,
                    )[0].strip()
                    break

                yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"
        except Exception as e:
            err_msg = f"\n\n*Lilim stream error: {e}*"
            yield f"data: {json.dumps({'type': 'token', 'text': err_msg})}\n\n"
            break

        # ── local-first fallback: if local returned nothing/error, retry remote ──
        if use_local and _local_first and not turn_reply.strip():
            use_local = False  # switch to remote for this turn
            _local_first = False
            continue  # re-run this turn with remote

        full_assistant_reply += turn_reply

        # ── Bash block extraction ────────────────────────────────────────────
        # SAFETY: strip lines that look like hallucinated numbered output
        # ("1. Im borrowing...", "2. Ah, the classic...") before scanning.
        sanitised_reply = re.sub(r"(?m)^\d+\.\s+.+$", "", turn_reply)
        bash_matches = list(re.finditer(r"```(?:bash|sh|shell)\s*\n([\s\S]*?)```", sanitised_reply, re.IGNORECASE))

        if bash_matches:
            from lilim_core.tool_executor import ToolExecutor
            executor = ToolExecutor()

            for bash_match in bash_matches:
                raw_cmd = bash_match.group(1).strip()
                # Strip shebang if present
                lines = [l for l in raw_cmd.splitlines() if not l.startswith("#!")]
                command = "\n".join(lines).strip()
                # Expand ~ to absolute path — use runtime home, not daemon home
                _exec_home = home_dir or _RUNTIME_USER_INFO.get("preferred_home") or _RUNTIME_USER_INFO.get("system_home") or str(Path.home())
                command = command.replace("~/", f"{_exec_home}/").replace(" ~", f" {_exec_home}")
                if not command:
                    continue

                short = command[:80].replace("\n", "; ")

                # Classify the command via tool rules
                classification = executor.classify_command(command)

                if classification == "forbidden":
                    obs_block = f"\n\n**[System → `{short}`]**\n```\nRejected: command is on the absolute forbidden list.\n```\n"
                    yield f"data: {json.dumps({'type': 'token', 'text': obs_block})}\n\n"
                    full_assistant_reply += obs_block
                    continue

                if classification == "sudo_confirm" and not executor.has_valid_sudo_token(session_id):
                    # Use the same explicit UI approval boundary as other gated commands.
                    # Password entry is handled separately; never infer or cache credentials here.
                    yield f"data: {json.dumps({'type': 'tool_pending', 'command': command, 'short': short, 'sudo': True})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'provider': active_provider, 'pending_tool': True})}\n\n"
                    if full_assistant_reply:
                        _memory.save_turn("assistant", full_assistant_reply,
                                          session_id=session_id, category=enhanced["category"])
                    return

                if classification == "confirm":
                    # Halt the stream and ask the UI to confirm
                    yield f"data: {json.dumps({'type': 'tool_pending', 'command': command, 'short': short})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'provider': active_provider, 'pending_tool': True})}\n\n"
                    # Save partial reply so context is preserved on resume
                    if full_assistant_reply:
                        _memory.save_turn("assistant", full_assistant_reply,
                                          session_id=session_id, category=enhanced["category"])
                    return  # The UI re-submits an Observation message when user decides

                # classification == 'auto' — execute immediately
                yield f"data: {json.dumps({'type': 'tool_call', 'text': short})}\n\n"

                if classification == "sudo_confirm":
                    result = executor.shell_command_sudo(command, session_id=session_id)
                else:
                    result = executor.shell_command(command, confirmed=True, session_id=session_id)
                stdout = (result.get("stdout") or "").strip()
                stderr = (result.get("stderr") or "").strip()
                err    = result.get("error") or ""
                rc     = result.get("returncode", 0)
                output = "\n".join(x for x in [stdout, stderr] if x)
                if err and "Command not confirmed" not in err:
                    output = f"Error: {err}\n{output}".strip()

                # Track consecutive failures to stop the spiral.
                # "not found" errors don't count — they're missing tools, not logic failures.
                # The LLM should recover by installing the tool first.
                _is_not_found = result.get("not_found", False)
                if rc != 0 and not _is_not_found:
                    consecutive_failures += 1
                elif rc == 0:
                    consecutive_failures = 0
                # If not_found: don't increment, let the LLM try to install the tool

                # Truncate to prevent UI floods
                ls = output.splitlines()
                if len(ls) > 20:
                    output = "\n".join(ls[:20]) + f"\n\n... (truncated {len(ls)-20} more lines)"
                if len(output) > 2000:
                    output = output[:2000] + " ... (truncated)"
                if not output:
                    output = "(Command completed — no output)"

                obs_block = f"\n\n**[System → `{short}`]**\n```\n{output}\n```\n"
                yield f"data: {json.dumps({'type': 'token', 'text': obs_block})}\n\n"
                all_observations.append(f"`{short}` → {output}")
                full_assistant_reply += obs_block

                # Failure budget: stop spiral after MAX_CONSECUTIVE_FAILURES
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    stop_msg = "\n\n*Two commands failed in a row — stopping to avoid a loop. Check the errors above.*"
                    yield f"data: {json.dumps({'type': 'token', 'text': stop_msg})}\n\n"
                    full_assistant_reply += stop_msg
                    yield f"data: {json.dumps({'type': 'done', 'provider': active_provider})}\n\n"
                    if full_assistant_reply:
                        _memory.save_turn("assistant", full_assistant_reply,
                                          session_id=session_id, category=enhanced["category"])
                    return

            if all_observations:
                history.append({"role": "assistant", "content": turn_reply})
                history.append({"role": "user", "content": "Observation: " + "\n".join(all_observations)})
                history = _trim_agent_history(history, 4096, enhanced["enhanced_message"])
                continue
            else:
                break
        else:
            # No tool call — finished
            break

    yield f"data: {json.dumps({'type': 'done', 'provider': active_provider})}\n\n"

    if full_assistant_reply:
        _memory.save_turn("assistant", full_assistant_reply, session_id=session_id, category=enhanced["category"])
        _memory.extract_and_save([{"role": "user", "content": message}], session_id=session_id)



def _serve_recall(session_id: str) -> str:
    """Return a formatted summary of the current session's conversation history.

    This is called instead of the LLM when the user asks 'show me our previous
    conversation' or similar. It reads directly from the in-memory session store
    so there is zero risk of the model hallucinating output or running bash.
    """
    turns = _memory.get_recent_session(session_id, n=20)
    if not turns:
        return ("I don't have any earlier messages in this session yet. "
                "If you're looking for a previous session, those are archived in "
                "~/.local/share/lilim/memory/sessions/.")

    lines = []
    for t in turns:
        role = "You" if t["role"] == "user" else "Lilim"
        snippet = t["content"].strip().replace("\n", " ")[:200]
        if len(t["content"].strip()) > 200:
            snippet += "…"
        lines.append(f"**{role}:** {snippet}")

    header = f"Here's what we've covered so far ({len(turns)} messages):\n\n"
    return header + "\n\n".join(lines)


def _trim_to_token_budget(messages: list, budget_tokens: int) -> list:
    """Trim oldest non-system turns so the message list fits within budget_tokens.

    Uses a 4-chars-per-token heuristic (no hard tokeniser dependency).
    System messages are always preserved; user/assistant turns are dropped
    from the front until the total fits.
    """
    if count_tokens(messages) <= budget_tokens:
        return messages

    # Separate system messages from the conversation turns
    system_msgs = [m for m in messages if m["role"] == "system"]
    turn_msgs   = [m for m in messages if m["role"] != "system"]

    # Drop oldest turns (from the front) until we fit
    while turn_msgs and count_tokens(system_msgs + turn_msgs) > budget_tokens:
        turn_msgs.pop(0)

    return system_msgs + turn_msgs


def count_tokens(messages: list) -> int:
    """Estimate prompt size without adding a tokenizer dependency."""
    return sum(len(m.get("content", "")) for m in messages) // 4


def _trim_agent_history(messages: list, budget_tokens: int, goal: str) -> list:
    """Keep the original goal and newest observations while pruning middle context."""
    if count_tokens(messages) <= budget_tokens:
        return messages
    system_msgs = [m for m in messages if m["role"] == "system"]
    turns = [m for m in messages if m["role"] != "system"]
    # Keep the task statement visible even when many observations accumulate.
    goal_turn = next(
        (m for m in turns if m["role"] == "user" and m["content"] == goal),
        {"role": "user", "content": goal},
    )
    turns = [goal_turn] + [m for m in turns if m is not goal_turn][-8:]
    while turns and count_tokens(system_msgs + turns) > budget_tokens:
        turns.pop(1 if len(turns) > 1 else 0)
    return system_msgs + turns


def _build_messages_with_custom_sys(enhanced: dict, session_id: str, sys_prompt: str) -> list:
    """Build the message list with a specific system prompt, trimmed to context budget."""
    recent = _memory.get_recent_session(session_id, n=20)
    messages = [{"role": "system", "content": sys_prompt}]

    mem_ctx = enhanced.get("memory_context", "")
    if mem_ctx:
        messages.append({"role": "system", "content": f"[Memory Context: {mem_ctx}]"})

    messages.extend(recent)
    messages.append({"role": "user", "content": enhanced["enhanced_message"]})

    # Apply rolling context budget
    budget = 8_192  # default
    if _free_router:
        limits = _free_router.get_context_limits()
        budget = limits.get("effective_cap", 8_192)
    messages = _trim_to_token_budget(messages, budget)

    return messages


# ── Helpers ───────────────────────────────────────────────────

def _log_command(command: str, returncode: int):
    log_dir = Path("/var/log/lilim")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "commands.log"
        entry = f"{datetime.now(timezone.utc).isoformat()} rc={returncode} cmd={command!r}\n"
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
