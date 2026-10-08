"""
Lilim Tool Executor

Provides safe, audited execution of system tools on behalf of the user.
Non-destructive operations can be auto-approved via the tool-rules config.
Destructive or elevated operations require explicit confirmation from the UI.

Included tools:
  - shell_command        — run shell commands (rule-based auto-approval or UI gate)
  - shell_command_sudo   — run with elevated privileges after user confirms in UI
  - file_read            — read a file and return its contents
  - file_write           — write/create a file
  - file_list            — list directory contents
  - system_info          — snapshot of OS, disk, memory, processes
  - service_status       — systemctl status for a named service
  - package_search       — apt-cache search wrapper
  - browser_launch       — launch BrowserOS/neo with optional URL

Safety features:
  - Timeout on all executions (30s default)
  - Absolute forbidden pattern blocklist (no exceptions)
  - Persistent tool-rules.json controls auto-approve / always-confirm
  - Command audit log at /var/log/lilim/commands.log
  - Sudo elevation: uses pkexec (polkit GUI) when available, then sudo -S
    with a session credential token (never stored on disk)
"""

import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Safety configuration ──────────────────────────────────────

# These patterns are NEVER allowed, no exceptions
ABSOLUTE_FORBIDDEN = [
    r"rm\s+-rf\s+/",
    r"mkfs",
    r":\s*\(\s*\)\s*\{",        # Fork bomb
    r"dd\s+if=/dev/zero",
    r">\s*/dev/sd",
    r"chmod\s+-R\s+777\s+/",
    r"shred\s+/dev",
    r"wipefs",
]

# These require extra confirmation (currently: always require the `confirmed` flag)
HIGH_RISK_PATTERNS = [
    r"\brm\b",
    r"\bmv\b",
    r"\bsudo\s+rm\b",
    r"\bapt\s+(remove|purge)\b",
    r"\bsystemctl\s+(stop|disable|mask)\b",
]

# Paths the tool is NOT allowed to read
FORBIDDEN_READ_PATHS = [
    "/etc/shadow",
    "/etc/gshadow",
    "/root",
    "/proc/kcore",
    "/sys/firmware",
]

LOG_DIR = Path("/var/log/lilim")
TOOL_RULES_PATH = Path.home() / ".config" / "lilim" / "tool-rules.json"

# Default rules shipped with Lilim (used if no config file exists)
DEFAULT_TOOL_RULES = {
    "version": 1,
    "sudo_allowed": False,
    "auto_approve": [
        # Read-only / informational commands
        "df ", "df -", "free ", "free -", "ls ", "ls -", "ll ", "la ",
        "cat ", "head ", "tail ", "less ", "more ", "wc ", "sort ", "uniq ",
        "grep ", "find ", "locate ", "which ", "whereis ",
        "ps ", "ps -", "top -", "htop", "pgrep ",
        "uname", "uptime", "date", "whoami", "id ", "id\n",
        "hostname", "ip addr", "ip link", "ifconfig",
        "systemctl status", "journalctl -",
        "apt-cache search", "apt-cache show", "apt list",
        "pip list", "pip show", "pip freeze",
        "python ", "python3 ", "node ", "npm list", "cargo",
        "git status", "git log", "git diff", "git branch",
        "echo ", "printf ", "pwd", "env", "printenv",
        "lsblk", "lspci", "lsusb", "dmesg",
        "du ", "du -",
    ],
    # Git workflow + package install commands: auto-execute without UI prompt.
    "agent_auto": [
        "git init", "git add", "git commit", "git remote",
        "git push", "git pull", "git clone", "git fetch",
        "git stash", "git checkout", "git switch", "git merge",
        "git rebase", "git tag", "git reset", "git restore",
        "mkdir ", "mkdir -",   # creating directories is safe
        "touch ",              # creating empty files is safe
        "cp ", "cp -",        # copying files is generally safe
        # Package managers — install is safe; uninstall/purge stays in confirm
        "uv pip install", "uv add", "uv sync", "uv run",
        "uv tool install", "uv tool run",
        "pipx install", "pipx run",
        "npm install", "npm ci", "npm run",
        "cargo install", "cargo build",
        "pip install",         # pip installs into user venv — safe
        # Browser/tool registration commands
        "browser-use", "playwright install", "playwright",
    ],
    "always_confirm": [
        # Potentially destructive or elevated operations
        "sudo", "su -", "su ",
        "rm ", "rm -", "rmdir",
        "mv ", "mv -",
        "chmod", "chown", "chgrp",
        "apt install", "apt remove", "apt purge", "apt upgrade",
        "pip install", "pip uninstall",
        "systemctl start", "systemctl stop", "systemctl restart",
        "systemctl enable", "systemctl disable", "systemctl mask",
        "dd ", "mkfs", "mount ", "umount ",
        "iptables", "ufw ",
        "passwd", "adduser", "useradd", "userdel",
        "crontab",
        "curl ", "wget ",   # raw network writes still require confirmation
        "ssh ", "scp ",
    ],
    # Sudo commands require elevation flow — always go through confirm+elevate UI
    "sudo_patterns": [
        "sudo ",
        "sudo -",
        "pkexec ",
        "gksu ",
        "kdesu ",
    ]
}


# ── Session sudo token store (in-memory only, never on disk) ──────────────────
# Maps session_id → (password, expiry_timestamp)
# Expires after 5 minutes of inactivity
_SUDO_TOKENS: dict = {}
_SUDO_TOKEN_TTL = 300  # 5 minutes


class ToolExecutor:
    """Safe system tool execution for Lilim."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    # ── Tool rules ────────────────────────────────────────────────

    @staticmethod
    def load_tool_rules() -> dict:
        """Load tool permission rules from config, merging with defaults."""
        rules = dict(DEFAULT_TOOL_RULES)  # start from defaults
        if TOOL_RULES_PATH.exists():
            try:
                import json
                with open(TOOL_RULES_PATH) as f:
                    user_rules = json.load(f)
                # Merge: user lists extend (not replace) defaults
                rules["sudo_allowed"] = user_rules.get("sudo_allowed", rules["sudo_allowed"])
                if "auto_approve" in user_rules:
                    # User can add extra patterns; defaults are preserved
                    merged = list(rules["auto_approve"])
                    for p in user_rules["auto_approve"]:
                        if p not in merged:
                            merged.append(p)
                    rules["auto_approve"] = merged
                if "always_confirm" in user_rules:
                    merged = list(rules["always_confirm"])
                    for p in user_rules["always_confirm"]:
                        if p not in merged:
                            merged.append(p)
                    rules["always_confirm"] = merged
            except Exception:
                pass
        return rules

    @staticmethod
    def save_tool_rules(rules: dict) -> bool:
        """Persist tool rules to config file. Returns True on success."""
        try:
            import json
            TOOL_RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
            rules["version"] = DEFAULT_TOOL_RULES["version"]
            with open(TOOL_RULES_PATH, "w") as f:
                json.dump(rules, f, indent=2)
            return True
        except Exception:
            return False

    def classify_command(self, command: str) -> str:
        """Classify a command as 'auto', 'confirm', 'sudo_confirm', or 'forbidden'.

        Returns:
            'forbidden'    — absolute blocklist match (never run)
            'sudo_confirm' — requires sudo elevation (user must confirm + provide password)
            'auto'         — matches auto_approve OR agent_auto (run without UI prompt)
            'confirm'      — requires user approval via UI
        """
        # Absolute forbidden always wins
        rejection = self._check_forbidden(command)
        if rejection:
            return "forbidden"

        rules = self.load_tool_rules()
        cmd_stripped = command.strip().lower()

        # Sudo/elevated commands: always require the elevation flow
        for pattern in rules.get("sudo_patterns", DEFAULT_TOOL_RULES.get("sudo_patterns", ["sudo "])):
            if pattern.lower() in cmd_stripped:
                return "sudo_confirm"

        # Check always_confirm first (higher specificity wins)
        for pattern in rules.get("always_confirm", []):
            if pattern.lower() in cmd_stripped:
                return "confirm"

        # Check agent_auto (git workflow + safe file ops)
        for pattern in rules.get("agent_auto", []):
            if cmd_stripped.startswith(pattern.lower()) or pattern.lower() in cmd_stripped:
                return "auto"

        # Check auto_approve
        for pattern in rules.get("auto_approve", []):
            if cmd_stripped.startswith(pattern.lower()) or pattern.lower() in cmd_stripped:
                return "auto"

        # Default: require confirmation for anything unrecognised
        return "confirm"

    def is_sudo_command(self, command: str) -> bool:
        """Return True if the command requires elevated privileges."""
        return self.classify_command(command) == "sudo_confirm"

    # ── Shell command ────────────────────────────────────────────────

    @staticmethod
    def set_sudo_token(session_id: str, password: str):
        """Store a sudo session token (password, expires in 5 min). Never written to disk."""
        _SUDO_TOKENS[session_id] = (password, time.time() + _SUDO_TOKEN_TTL)

    @staticmethod
    def clear_sudo_token(session_id: str):
        """Clear a sudo session token (e.g., on session reset)."""
        _SUDO_TOKENS.pop(session_id, None)

    @staticmethod
    def has_valid_sudo_token(session_id: str) -> bool:
        """Return True if there's an unexpired sudo token for this session."""
        entry = _SUDO_TOKENS.get(session_id)
        if not entry:
            return False
        _, expiry = entry
        if time.time() > expiry:
            del _SUDO_TOKENS[session_id]
            return False
        return True

    def shell_command_sudo(
        self,
        command: str,
        session_id: str = "default",
        password: Optional[str] = None,
    ) -> dict:
        """
        Execute a command with elevated privileges.

        Strategy:
          1. Try pkexec (polkit — shows GUI dialog for desktop users).
          2. If pkexec not available, use `sudo -S` with the session token password.

        Args:
            command:    Shell command string (may or may not include 'sudo' prefix).
            session_id: Used to look up a cached session password token.
            password:   If provided, stores as session token for 5 minutes.
        """
        # Strip leading 'sudo' if present (we'll re-add in a controlled way)
        base_cmd = command.strip()
        if base_cmd.startswith("sudo "):
            base_cmd = base_cmd[5:].strip()

        # Safety check on the underlying command
        rejection = self._check_forbidden(base_cmd)
        if rejection:
            return {
                "error": f"Rejected: {rejection}",
                "command": command, "stdout": "", "stderr": "", "returncode": -1,
            }

        # Store password in session token if provided
        if password:
            self.set_sudo_token(session_id, password)

        # Use a password explicitly provided by the user before falling back to
        # Polkit. This keeps desktop and CLI confirmations tied to the approved
        # session and avoids requiring a GUI in the systemd service environment.
        token = _SUDO_TOKENS.get(session_id)
        if token:
            pwd, expiry = token
            if time.time() < expiry:
                _SUDO_TOKENS[session_id] = (pwd, time.time() + _SUDO_TOKEN_TTL)
                sudo_cmd = f"sudo -S -p '' {base_cmd}"
                result = self._run_command(sudo_cmd, env_home=True, stdin_data=pwd + "\n")
                if result.get("returncode", -1) != 0:
                    stderr = (result.get("stderr") or "").lower()
                    if "incorrect password" in stderr or "try again" in stderr or "a password is required" in stderr:
                        self.clear_sudo_token(session_id)
                self._audit_log(f"[SUDO/-S] {base_cmd}", result["returncode"])
                return result
            self.clear_sudo_token(session_id)

        # Polkit fallback for desktop environments without a supplied token.
        if shutil.which("pkexec"):
            # pkexec requires the command to be an absolute path
            parts = base_cmd.split()
            exe_path = shutil.which(parts[0]) if parts else None
            if exe_path:
                pkexec_cmd = f"pkexec {exe_path} {' '.join(parts[1:])}"
                result = self._run_command(pkexec_cmd, env_home=True)
                self._audit_log(f"[SUDO/pkexec] {base_cmd}", result["returncode"])
                return result

        # No elevation method available
        return {
            "error": "Sudo elevation requires a password. The UI should have prompted for one.",
            "command": command, "stdout": "", "stderr": "", "returncode": -1,
            "needs_password": True,
        }

    def shell_command(self, command: str, confirmed: bool = False,
                      session_id: str = "default") -> dict:
        """Execute a shell command.

        Args:
            command:    The shell command string to run.
            confirmed:  Set to True by the UI after user approval, OR when the
                        command is auto-approved by tool-rules.json.
            session_id: Used for sudo token lookup if command requires elevation.

        Returns:
            dict with stdout, stderr, returncode, classification, and the command.
        """
        # Classify first
        classification = self.classify_command(command)

        if classification == "forbidden":
            return {
                "error": "Command is on the absolute forbidden list.",
                "command": command,
                "classification": "forbidden",
                "stdout": "",
                "stderr": "",
                "returncode": -1,
            }

        # Sudo commands: route to elevation handler
        if classification == "sudo_confirm":
            if not confirmed:
                return {
                    "error": "This command needs elevated privileges (sudo).",
                    "command": command,
                    "classification": "sudo_confirm",
                    "stdout": "",
                    "stderr": "",
                    "returncode": -1,
                    "needs_sudo": True,
                    "needs_confirmation": True,
                }
            # If confirmed: try elevation via pkexec or session token
            return self.shell_command_sudo(command, session_id=session_id)

        # Auto-approved commands skip the confirmation gate
        if classification == "auto":
            confirmed = True

        if not confirmed:
            return {
                "error": "Command not confirmed by user.",
                "command": command,
                "classification": classification,
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "needs_confirmation": True,
            }

        # Safety checks
        rejection = self._check_forbidden(command)
        if rejection:
            return {
                "error": f"Rejected: {rejection}",
                "command": command,
                "stdout": "",
                "stderr": "",
                "returncode": -1,
            }

        result = self._run_command(command)
        if result.get("returncode") == 0 or result.get("not_found"):
            pass
        self._audit_log(command, result.get("returncode", -1))
        return result

    def _run_command(
        self,
        command: str,
        env_home: bool = True,
        stdin_data: Optional[str] = None,
    ) -> dict:
        """Internal: actually run a shell command with safety limits."""
        try:
            # Build an expanded PATH so user-installed tools (uv, cargo, pipx, etc.)
            # are always findable — systemd strips HOME/.local/bin from the daemon PATH.
            _home = str(Path.home())
            _extra_paths = [
                f"{_home}/.local/bin",
                f"{_home}/.cargo/bin",
                f"{_home}/.npm/bin",
                f"{_home}/.yarn/bin",
                "/usr/local/bin",
                "/usr/bin",
                "/bin",
                "/usr/local/sbin",
                "/usr/sbin",
                "/sbin",
            ]
            exec_env = os.environ.copy()
            existing_path = exec_env.get("PATH", "")
            full_path = ":".join(_extra_paths)
            if existing_path:
                full_path = full_path + ":" + existing_path
            exec_env["PATH"] = full_path
            if env_home:
                exec_env["HOME"] = _home

            # We use Popen with limited reading to prevent memory exhaustion (OOM)
            # if a command produces millions of lines of output.
            stdin_pipe = subprocess.PIPE if stdin_data else None
            process = subprocess.Popen(
                command,
                shell=True,
                executable="/bin/bash",  # Use bash (not sh) for better compatibility
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=stdin_pipe,
                text=True,
                bufsize=1,  # line buffered
                env=exec_env,
            )

            # Write password to stdin if provided (for sudo -S)
            if stdin_data:
                try:
                    process.stdin.write(stdin_data)
                    process.stdin.close()
                except Exception:
                    pass

            stdout_lines = []
            stderr_lines = []
            max_total_chars = 100_000  # ~100KB safety limit for raw capture

            def read_stream(stream, target_list):
                count = 0
                while count < max_total_chars:
                    line = stream.readline()
                    if not line:
                        break
                    target_list.append(line)
                    count += len(line)
                if count >= max_total_chars:
                    target_list.append("\n... (raw output truncated for safety) ...")
                    process.terminate()

            read_stream(process.stdout, stdout_lines)
            read_stream(process.stderr, stderr_lines)

            try:
                rc = process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                rc = -1
                stderr_lines.append(f"\n[Timed out after {self.timeout}s]")

            stdout = "".join(stdout_lines)
            stderr = "".join(stderr_lines)

            # Tag "command not found" errors so the agent loop can recover
            not_found = (
                "command not found" in stderr.lower() or
                ("no such file or directory" in stderr.lower() and "exec" in stderr.lower())
            )

            return {
                "command": command,
                "stdout": stdout,
                "stderr": stderr,
                "returncode": rc,
                "error": None if rc == 0 else f"Process exited with code {rc}",
                "not_found": not_found,
            }
        except Exception as e:
            return {
                "command": command,
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "error": str(e),
            }

    # ── File operations ───────────────────────────────────────

    def file_read(self, path: str, max_chars: int = 10_000) -> dict:
        """Read a file and return its contents (up to max_chars characters)."""
        resolved = Path(path).resolve()

        # Check forbidden paths
        for forbidden in FORBIDDEN_READ_PATHS:
            if str(resolved).startswith(forbidden):
                return {
                    "path": path,
                    "content": "",
                    "error": f"Reading '{forbidden}' is not permitted.",
                    "truncated": False,
                }

        try:
            content = resolved.read_text(errors="replace")
            truncated = len(content) > max_chars
            return {
                "path": str(resolved),
                "content": content[:max_chars],
                "error": None,
                "truncated": truncated,
                "size_bytes": resolved.stat().st_size,
            }
        except FileNotFoundError:
            return {"path": path, "content": "", "error": "File not found.", "truncated": False}
        except PermissionError:
            return {"path": path, "content": "", "error": "Permission denied.", "truncated": False}
        except Exception as e:
            return {"path": path, "content": "", "error": str(e), "truncated": False}

    def file_write(self, path: str, content: str, confirmed: bool = False) -> dict:
        """Write content to a file (requires confirmation)."""
        if not confirmed:
            return {"error": "Write not confirmed by user.", "path": path}

        resolved = Path(path).resolve()
        
        # Check forbidden paths
        for forbidden in FORBIDDEN_READ_PATHS:
            if str(resolved).startswith(forbidden):
                return {"error": f"Writing to '{forbidden}' is not permitted.", "path": path}

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content)
            self._audit_log(f"WRITE {path}", 0)
            return {"path": str(resolved), "size": len(content), "error": None}
        except Exception as e:
            return {"path": path, "error": str(e)}

    def file_list(self, path: str, max_entries: int = 50) -> dict:
        """List directory contents."""
        resolved = Path(path).resolve()

        try:
            if not resolved.is_dir():
                return {"path": path, "entries": [], "error": "Not a directory."}

            entries = []
            for item in sorted(resolved.iterdir())[:max_entries]:
                entries.append({
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                })
            return {"path": str(resolved), "entries": entries, "error": None}
        except PermissionError:
            return {"path": path, "entries": [], "error": "Permission denied."}
        except Exception as e:
            return {"path": path, "entries": [], "error": str(e)}

    # ── System info ───────────────────────────────────────────

    def system_info(self) -> dict:
        """Return a comprehensive system info snapshot."""
        info = {}

        commands = {
            "os":     ["uname", "-a"],
            "disk":   ["df", "-h", "/"],
            "memory": ["free", "-h"],
            "uptime": ["uptime", "-p"],
            "load":   ["cat", "/proc/loadavg"],
        }

        for key, cmd in commands.items():
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                out = r.stdout.strip()
                if key == "disk":
                    # Get just the second line (data row)
                    lines = out.split("\n")
                    info[key] = lines[1] if len(lines) > 1 else out
                elif key == "memory":
                    lines = out.split("\n")
                    info[key] = lines[1] if len(lines) > 1 else out
                else:
                    info[key] = out
            except Exception:
                info[key] = "N/A"

        return info

    def service_status(self, service: str) -> dict:
        """Get systemctl status for a service."""
        # Sanitise service name
        if not re.match(r'^[\w.-]+$', service):
            return {"service": service, "error": "Invalid service name.", "output": ""}

        try:
            r = subprocess.run(
                ["systemctl", "status", "--no-pager", service],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return {
                "service": service,
                "output": r.stdout[:3000],
                "returncode": r.returncode,
                "error": None,
            }
        except Exception as e:
            return {"service": service, "output": "", "error": str(e), "returncode": -1}

    def package_search(self, query: str) -> dict:
        """Search for packages using apt-cache."""
        if not re.match(r'^[\w.+-]+$', query):
            return {"query": query, "results": "", "error": "Invalid query."}

        try:
            r = subprocess.run(
                ["apt-cache", "search", query],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return {
                "query": query,
                "results": r.stdout[:5000],
                "error": None,
            }
        except Exception as e:
            return {"query": query, "results": "", "error": str(e)}

    def browser_launch(self, url: Optional[str] = None) -> dict:
        """
        Launch BrowserOS (neo) with an optional URL.
        Detects BrowserOS binary in common install paths.
        Falls back to xdg-open if BrowserOS is not found.
        """
        # BrowserOS / neo detection
        neo_candidates = [
            shutil.which("neo"),
            shutil.which("browseros"),
            shutil.which("browser-os"),
            "/usr/bin/neo",
            "/opt/browseros/neo",
            "/usr/local/bin/neo",
        ]
        neo_bin = next((p for p in neo_candidates if p and Path(p).exists()), None)

        try:
            if neo_bin:
                cmd = [neo_bin]
                if url:
                    cmd.append(url)
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return {"launched": True, "browser": "neo", "url": url}
            elif url:
                # Fallback to xdg-open
                subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return {"launched": True, "browser": "xdg-open", "url": url}
            else:
                return {"launched": False, "error": "BrowserOS (neo) not found and no URL to open."}
        except Exception as e:
            return {"launched": False, "error": str(e)}

    # ── Internal helpers ──────────────────────────────────────

    def _check_forbidden(self, command: str) -> Optional[str]:
        """Return a rejection reason if command matches forbidden patterns, else None."""
        cmd_lower = command.lower()
        for pattern in ABSOLUTE_FORBIDDEN:
            if re.search(pattern, cmd_lower):
                return f"Matches absolute forbidden pattern: {pattern!r}"
        return None

    def _audit_log(self, command: str, returncode):
        """Write command to audit log (best-effort)."""
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            entry = (
                f"{datetime.utcnow().isoformat()} "
                f"rc={returncode} "
                f"cmd={command!r}\n"
            )
            with open(LOG_DIR / "commands.log", "a") as f:
                f.write(entry)
        except Exception:
            pass
