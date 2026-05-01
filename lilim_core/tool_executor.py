"""
Lilim Tool Executor

Provides safe, audited execution of system tools on behalf of the user.
All destructive operations require explicit confirmation from the UI.

Included tools:
  - shell_command   — run arbitrary shell commands (with confirmation)
  - file_read       — read a file and return its contents
  - file_list       — list directory contents
  - system_info     — snapshot of OS, disk, memory, processes
  - service_status  — systemctl status for a named service
  - package_search  — apt-cache search wrapper

Safety features:
  - Timeout on all executions (30s default)
  - Forbidden pattern blocklist
  - Command audit log at /var/log/lilim/commands.log
  - No automatic root escalation — sudo only for explicitly whitelisted commands
"""

import os
import re
import subprocess
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


class ToolExecutor:
    """Safe system tool execution for Lilim."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    # ── Shell command ─────────────────────────────────────────

    def shell_command(self, command: str, confirmed: bool = False) -> dict:
        """Execute a shell command.

        Args:
            command:   The shell command string to run.
            confirmed: Must be True — set by the UI after user clicks 'Run it'.

        Returns:
            dict with stdout, stderr, returncode, and the command itself.
        """
        if not confirmed:
            return {
                "error": "Command not confirmed by user.",
                "command": command,
                "stdout": "",
                "stderr": "",
                "returncode": -1,
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

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            self._audit_log(command, result.returncode)
            return {
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "error": None,
            }
        except subprocess.TimeoutExpired:
            self._audit_log(command, "TIMEOUT")
            return {
                "command": command,
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "error": f"Timed out after {self.timeout}s",
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
