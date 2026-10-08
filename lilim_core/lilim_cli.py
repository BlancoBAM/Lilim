"""Terminal client for Lilim's SSE agent loop."""

from __future__ import annotations

import argparse
import http.client
import getpass
import json
import os
import socket
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit


API_BASE = os.environ.get("LILIM_API_URL", "http://127.0.0.1:8080").rstrip("/")
MAX_TOOL_CONFIRMATIONS = 20


def _connection(timeout: int = 600):
    parsed = urlsplit(API_BASE)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("LILIM_API_URL must be an HTTP or HTTPS URL")
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    base_path = parsed.path.rstrip("/")
    return connection, base_path


def _post_json(endpoint: str, payload: dict) -> dict:
    connection, base_path = _connection(timeout=30)
    body = json.dumps(payload).encode("utf-8")
    connection.request(
        "POST", f"{base_path}{endpoint}", body=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    response = connection.getresponse()
    try:
        raw = response.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"error": raw}
        if not 200 <= response.status < 300:
            raise RuntimeError(data.get("error") or data.get("detail") or raw)
    finally:
        connection.close()
    return data


def _stream_message(message: str, session_id: str) -> dict | None:
    payload = {
        "message": message,
        "session_id": session_id,
        "stream": True,
        "username": getpass.getuser(),
        "home_dir": str(Path.home()),
    }
    connection, base_path = _connection()
    try:
        connection.request(
            "POST", f"{base_path}/chat", body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        response = connection.getresponse()
        if not 200 <= response.status < 300:
            raise RuntimeError(response.read().decode("utf-8", errors="replace"))
        while True:
            line_bytes = response.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "token" and event.get("text"):
                print(event["text"], end="", flush=True)
            elif kind == "tool_call":
                print(f"\n\033[36m⚡ Running: {event.get('text', 'command')}\033[0m", flush=True)
            elif kind == "tool_pending":
                print()
                return event
            elif kind == "error":
                print(f"\nError: {event.get('text', 'request failed')}", file=sys.stderr)
            elif kind == "done":
                print()
                return None
    finally:
        connection.close()
    print()
    return None


def _run_approved_tool(command: str, session_id: str, sudo: bool) -> str:
    print(f"\nCommand requires your approval:\n  {command}")
    try:
        approved = input("Run this command? [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        approved = False
    if not approved:
        return "[User declined to run the command. Do not retry it.]"

    payload = {"command": command, "confirmed": True}
    endpoint = "/tools/shell"
    if sudo:
        password = getpass.getpass("Sudo password: ")
        if not password:
            return "[No sudo password supplied. The command was not run.]"
        payload.update({"session_id": session_id, "password": password})
        endpoint = "/tools/shell/sudo"
        password = ""

    try:
        result = _post_json(endpoint, payload)
    finally:
        if sudo:
            payload["password"] = ""
    stdout = (result.get("stdout") or "").strip()
    stderr = (result.get("stderr") or "").strip()
    output = "\n".join(part for part in (stdout, stderr) if part)
    return output[:8000] or "(Command completed — no output)"


def main() -> int:
    global API_BASE
    parser = argparse.ArgumentParser(prog="lilim-cli", description="Ask Lilim to complete a task from the terminal.")
    parser.add_argument("message", nargs="+", help="Task or question for Lilim")
    parser.add_argument("--api-url", default=API_BASE, help="Lilim gateway URL (default: %(default)s)")
    args = parser.parse_args()

    API_BASE = args.api_url.rstrip("/")
    message = " ".join(args.message)
    session_id = f"cli_{uuid.uuid4().hex}"

    try:
        pending = _stream_message(message, session_id)
        confirmations = 0
        while pending:
            confirmations += 1
            if confirmations > MAX_TOOL_CONFIRMATIONS:
                print("Stopped after too many confirmation prompts.", file=sys.stderr)
                return 2
            observation = _run_approved_tool(
                pending.get("command", ""),
                session_id,
                pending.get("sudo") is True,
            )
            pending = _stream_message(f"Observation: {observation}", session_id)
        return 0
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    except (OSError, http.client.HTTPException, RuntimeError, ValueError, socket.timeout) as exc:
        print(f"Lilim CLI: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
