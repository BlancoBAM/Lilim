"""
Lilim Brain Server — FastAPI

The core AI backend server. Runs on port 8081 (internal).
The Rust lilim-runtime proxies requests here from the desktop UI.

Routes:
  GET  /health            — liveness check
  POST /chat              — main chat endpoint (streaming SSE)
  POST /chat/sync         — non-streaming version for simple clients
  POST /memory/search     — search memory store
  GET  /memory/context    — get memory context for a query
  GET  /memory/stats      — memory statistics
  POST /tools/shell       — execute a shell command (pre-confirmed by UI)
  GET  /system/info       — snapshot of OS, disk, memory stats

Usage:
  python -m lilim_core.server
  # or via lilim-serve script
"""

import json
import os
import platform
import subprocess
import sys
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

# ── LLM client ───────────────────────────────────────────────
try:
    import litellm
    from litellm import completion, acompletion
    litellm.suppress_debug_info = True
except ImportError:
    print("ERROR: litellm not installed. Run: pip install litellm", file=sys.stderr)
    sys.exit(1)

# ── Local modules ────────────────────────────────────────────
from pathlib import Path as _P
_HERE = _P(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from lilim_core.prompt_enhancer import PromptEnhancer
from lilim_core.model_router import ModelRouter
from lilim_core.memory_sqlite import MemoryManager

# ── Config ────────────────────────────────────────────────────
CONFIG_PATHS = [
    Path("/etc/lilith/lilim-identity.json"),
    Path.home() / ".config" / "lilim" / "lilim-identity.json",
    _HERE.parent / "config" / "lilim-identity.json",
]

ROUTING_PATHS = [
    Path("/etc/lilith/routing.toml"),
    Path.home() / ".config" / "lilim" / "routing.toml",
    _HERE.parent / "config" / "routing.toml",
]

PORT = int(os.environ.get("LILIM_BRAIN_PORT", "8081"))
HOST = os.environ.get("LILIM_BRAIN_HOST", "127.0.0.1")

# Commands we always refuse to run, no matter what
FORBIDDEN_COMMANDS = [
    "rm -rf /",
    "mkfs",
    ":(){:|:&};:",
    "dd if=/dev/zero",
    "chmod -R 777 /",
    "> /dev/sda",
]


# ── Load identity ─────────────────────────────────────────────

def load_identity() -> dict:
    for path in CONFIG_PATHS:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
    # Fallback minimal identity
    return {
        "identity": {"names": {"first": "Lilim"}},
        "linguistics": {
            "text_style": {"style_descriptors": ["helpful", "sarcastic", "caring"]}
        },
        "motivations": {
            "core_drive": "Help the user succeed while maintaining dry humor and genuine care"
        },
    }


def build_system_prompt(identity: dict) -> str:
    name = identity.get("identity", {}).get("names", {}).get("first", "Lilim")
    style = identity.get("linguistics", {}).get("text_style", {}).get(
        "style_descriptors", ["helpful"]
    )
    drive = identity.get("motivations", {}).get("core_drive", "Help the user.")

    style_str = ", ".join(style)

    return f"""You are {name}, the built-in AI assistant for Lilith Linux.

Personality: {style_str}. {drive}

Your user is a first-year medical assistant student who:
- Is new to Linux and computers
- Needs help with medical assistant curriculum (anatomy, physiology, medical terminology, clinical procedures, pharmacology basics)
- May need help with general college academic skills
- Sometimes needs Linux/technical help — fix things for them, don't just describe how

Guidelines:
- Be genuinely helpful above all else
- Use plain English — avoid jargon unless explaining it
- For medical topics: be accurate, use correct terminology, include memory aids and analogies
- For Linux tasks: show exact commands, explain what they do before running, confirm destructive actions
- For scheduling: confirm the exact time before setting any reminder
- Keep responses concise but complete
- If you need to run a command or read a file to help, say so and the UI will let the user confirm

When showing shell commands, wrap them in triple backticks with 'bash' label:
```bash
command here
```

Remember: you live on Lilith Linux (based on Ubuntu/Debian). Use apt, systemctl, etc.
"""


# ── Request / Response models ─────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    stream: Optional[bool] = True


class ToolShellRequest(BaseModel):
    command: str
    confirmed: bool = False    # Must be True — UI sends this after user confirms


class MemorySearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5


# ── App setup ─────────────────────────────────────────────────

app = FastAPI(title="Lilim Brain", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080",
                   "tauri://localhost", "http://127.0.0.1:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons (initialised at startup)
_identity: dict = {}
_system_prompt: str = ""
_enhancer: PromptEnhancer = None
_router: ModelRouter = None
_memory: MemoryManager = None


@app.on_event("startup")
async def startup():
    global _identity, _system_prompt, _enhancer, _router, _memory

    _memory = MemoryManager()
    _identity = load_identity()
    _system_prompt = build_system_prompt(_identity)
    _enhancer = PromptEnhancer(memory_manager=_memory)

    # Load routing config
    routing_path = None
    for p in ROUTING_PATHS:
        if p.exists():
            routing_path = str(p)
            break
    _router = ModelRouter(config_path=routing_path)

    print(f"[Lilim Brain] Started on {HOST}:{PORT}", flush=True)
    print(f"[Lilim Brain] Identity loaded: {_identity.get('identity', {}).get('names', {}).get('first', '?')}", flush=True)
    print(f"[Lilim Brain] Memory DB: {_memory.db_path}", flush=True)


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "name": "Lilim Brain", "ts": datetime.utcnow().isoformat()}


@app.post("/chat")
async def chat(req: ChatRequest):
    """Main chat endpoint. Returns SSE stream or JSON depending on req.stream."""
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
    """Non-streaming chat. Simpler for testing."""
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


@app.post("/tools/shell")
async def tools_shell(req: ToolShellRequest):
    """Execute a shell command. The UI must set confirmed=True after the user approves."""
    if not req.confirmed:
        raise HTTPException(
            status_code=400,
            detail="Command not confirmed. UI must set confirmed=True after user approval."
        )

    # Safety check
    cmd_lower = req.command.lower().strip()
    for forbidden in FORBIDDEN_COMMANDS:
        if forbidden in cmd_lower:
            raise HTTPException(
                status_code=403,
                detail=f"Forbidden command pattern detected: '{forbidden}'"
            )

    try:
        result = subprocess.run(
            req.command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Log the command
        _log_command(req.command, result.returncode)

        return {
            "command": req.command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Command timed out after 30 seconds")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/info")
async def system_info():
    """Return a snapshot of system information."""
    info = {}

    # OS info
    try:
        r = subprocess.run(["uname", "-a"], capture_output=True, text=True, timeout=5)
        info["os"] = r.stdout.strip()
    except Exception:
        info["os"] = platform.uname()._asdict()

    # Disk
    try:
        r = subprocess.run(["df", "-h", "/"], capture_output=True, text=True, timeout=5)
        lines = r.stdout.strip().split("\n")
        info["disk"] = lines[1] if len(lines) > 1 else "N/A"
    except Exception:
        info["disk"] = "N/A"

    # Memory
    try:
        r = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=5)
        lines = r.stdout.strip().split("\n")
        info["memory"] = lines[1] if len(lines) > 1 else "N/A"
    except Exception:
        info["memory"] = "N/A"

    # CPU
    try:
        r = subprocess.run(
            ["grep", "-m1", "model name", "/proc/cpuinfo"],
            capture_output=True, text=True, timeout=5
        )
        info["cpu"] = r.stdout.split(":")[-1].strip() if r.returncode == 0 else "N/A"
    except Exception:
        info["cpu"] = "N/A"

    return info


# ── Core chat logic ───────────────────────────────────────────

async def _sync_chat(message: str, session_id: str = "default") -> dict:
    """Process a chat message synchronously and return the full response."""
    # 1. Save user turn to memory
    _memory.save_turn("user", message, session_id=session_id)

    # 2. Enhance prompt
    enhanced = _enhancer.enhance(message) if _enhancer.should_enhance(message) else {
        "enhanced_message": message, "category": "conversation", "memory_context": ""
    }

    # 3. Route to model
    route = _router.route(enhanced["enhanced_message"], enhanced["category"])
    model = route["model"]

    # 4. Build message list with system prompt + memory context + history
    recent = _memory.get_recent_session(session_id, n=10)
    messages = [{"role": "system", "content": _system_prompt}]

    # Inject memory context as a system note
    mem_ctx = enhanced.get("memory_context", "")
    if mem_ctx:
        messages.append({"role": "system", "content": mem_ctx})

    messages.extend(recent)
    messages.append({"role": "user", "content": enhanced["enhanced_message"]})

    # 5. Call LLM
    try:
        response = completion(model=model, messages=messages, max_tokens=1024, stream=False)
        reply = response.choices[0].message.content or ""

        # Log cost
        usage = getattr(response, "usage", None)
        if usage:
            _router.log_cost(model, usage.prompt_tokens or 0, usage.completion_tokens or 0)

    except Exception as e:
        reply = f"*Something went wrong calling the model ({type(e).__name__}): {e}*"

    # 6. Save assistant turn
    _memory.save_turn("assistant", reply, session_id=session_id,
                      category=enhanced["category"])
    # Auto-extract facts
    _memory.extract_and_save([{"role": "user", "content": message}], session_id=session_id)

    return {
        "reply": reply,
        "model": model,
        "category": enhanced["category"],
        "route_reason": route.get("reason", ""),
        "session_id": session_id,
    }


async def _stream_chat(message: str, session_id: str = "default") -> AsyncGenerator[str, None]:
    """Stream a chat response as SSE events."""
    # 1. Save user turn
    _memory.save_turn("user", message, session_id=session_id)

    # 2. Enhance
    enhanced = _enhancer.enhance(message) if _enhancer.should_enhance(message) else {
        "enhanced_message": message, "category": "conversation", "memory_context": ""
    }

    # 3. Route
    route = _router.route(enhanced["enhanced_message"], enhanced["category"])
    model = route["model"]

    # 4. Build messages
    recent = _memory.get_recent_session(session_id, n=10)
    messages = [{"role": "system", "content": _system_prompt}]
    mem_ctx = enhanced.get("memory_context", "")
    if mem_ctx:
        messages.append({"role": "system", "content": mem_ctx})
    messages.extend(recent)
    messages.append({"role": "user", "content": enhanced["enhanced_message"]})

    # Emit metadata event first
    meta = {
        "type": "meta",
        "model": model,
        "category": enhanced["category"],
        "reason": route.get("reason", ""),
    }
    yield f"data: {json.dumps(meta)}\n\n"

    # 5. Stream LLM response
    full_reply = ""
    try:
        stream = completion(model=model, messages=messages, max_tokens=1024, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_reply += delta
                yield f"data: {json.dumps({'type': 'token', 'text': delta})}\n\n"
    except Exception as e:
        err_msg = f"*Error: {type(e).__name__}: {e}*"
        yield f"data: {json.dumps({'type': 'token', 'text': err_msg})}\n\n"
        full_reply = err_msg

    # End event
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

    # 6. Save assistant reply to memory
    if full_reply:
        _memory.save_turn("assistant", full_reply, session_id=session_id,
                          category=enhanced["category"])
        _memory.extract_and_save([{"role": "user", "content": message}],
                                  session_id=session_id)


# ── Helpers ───────────────────────────────────────────────────

def _log_command(command: str, returncode: int):
    """Log shell command execution for audit trail."""
    log_dir = Path("/var/log/lilim")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "commands.log"
        entry = f"{datetime.utcnow().isoformat()} rc={returncode} cmd={command!r}\n"
        with open(log_file, "a") as f:
            f.write(entry)
    except Exception:
        pass  # Best-effort logging; don't break the response


# ── Entrypoint ────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "lilim_core.server:app",
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=False,  # Reduce noise; Rust runtime logs at its level
    )
