"""
Lilim Memory Manager — Rowboat-Inspired Persistent Knowledge Graph

Maintains a Markdown vault of facts, decisions, preferences, and conversation
summaries. Loaded before each conversation for persistent context. Updated
after each conversation with extracted knowledge.
"""

import datetime
import json
import os
import re
from pathlib import Path
from typing import Optional


# ── Default vault path ────────────────────────────────────
DEFAULT_VAULT = os.path.expanduser("~/.local/share/lilim/memory")


class MemoryManager:
    """Persistent knowledge graph stored as Markdown files."""

    def __init__(self, vault_path: Optional[str] = None):
        self.vault = Path(vault_path or DEFAULT_VAULT)
        self._ensure_vault()
        self._active_sessions = {} # session_id -> list of turns

    @property
    def db_path(self) -> str:
        return str(self.vault)

    # ── Vault initialization ──────────────────────────────

    def _ensure_vault(self):
        """Create vault directories if they don't exist."""
        for subdir in ["people", "facts", "decisions", "sessions"]:
            (self.vault / subdir).mkdir(parents=True, exist_ok=True)

        # Create user profile if missing
        user_profile = self.vault / "people" / "user.md"
        if not user_profile.exists():
            user_profile.write_text(
                "# User Profile\n\n"
                "## Preferences\n\n"
                "- *(Lilim will learn your preferences over time)*\n\n"
                "## Known Facts\n\n"
                "- Running Lilith Linux\n"
            )

    # ── Load context for a conversation ───────────────────

    def load_context(self, query: str = "", max_notes: int = 5, max_chars: int = 2000) -> str:
        """Load relevant memory context for the current conversation."""
        relevant = []

        # Always include user profile
        user_profile = self.vault / "people" / "user.md"
        if user_profile.exists():
            content = user_profile.read_text()
            relevant.append(("User Profile", content, 999))

        # Search facts and decisions by keyword relevance
        if query:
            keywords = self._extract_keywords(query)
            for subdir in ["facts", "decisions"]:
                dir_path = self.vault / subdir
                if dir_path.exists():
                    for note in sorted(dir_path.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
                        content = note.read_text()
                        score = self._relevance_score(content, keywords)
                        if score > 0:
                            relevant.append((note.stem, content, score))

        # Sort by relevance
        relevant.sort(key=lambda x: x[2], reverse=True)

        if not relevant:
            return ""

        context_parts = []
        total_chars = 0

        for name, content, _ in relevant[:max_notes]:
            if total_chars + len(content) > max_chars:
                break
            context_parts.append(f"### {name}\n{content.strip()}")
            total_chars += len(content)

        return (
            "\n## Your Memory (persistent context from past conversations)\n"
            + "\n\n".join(context_parts)
            + "\n"
        )

    # ── Session turns (In-memory for performance, flushed to Markdown at end) ──

    def save_turn(self, role: str, content: str, session_id: str = "default", category: str = "general"):
        """Save a single conversation turn to the active session."""
        if session_id not in self._active_sessions:
            self._active_sessions[session_id] = []
        
        # Don't save empty content
        if not content.strip():
            return

        self._active_sessions[session_id].append({"role": role, "content": content})
        
        # Keep only last 20 turns per session in memory
        if len(self._active_sessions[session_id]) > 20:
            self._active_sessions[session_id] = self._active_sessions[session_id][-20:]

    def get_recent_session(self, session_id: str, n: int = 10) -> list:
        """Return the last N turns of a session."""
        history = self._active_sessions.get(session_id, [])
        return history[-n:]

    def clear_session(self, session_id: str):
        """Clear active session history."""
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]

    # ── Finalize session to Markdown ──────────────────────

    def extract_and_save(self, messages: list[dict], session_id: str = "default"):
        """Save a summary of the session to the Markdown vault."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        session_file = self.vault / "sessions" / f"{timestamp}.md"
        
        # Combine in-memory turns with passed messages if any
        all_msgs = self.get_recent_session(session_id, n=20)
        if not all_msgs:
            all_msgs = messages

        if not all_msgs:
            return

        summary = "Topic: " + ", ".join([m["content"][:40].replace("\n", " ") for m in all_msgs if m["role"] == "user"][:3])
        
        content = f"# Session {timestamp}\n\n{summary}\n\n## Messages\n"
        for m in all_msgs:
            content += f"- **{m['role']}**: {m['content'].strip()}\n"
        
        try:
            session_file.write_text(content)
        except Exception:
            pass

    def update_user_profile(self, key: str, value: str):
        """Add a fact to the user profile."""
        profile = self.vault / "people" / "user.md"
        try:
            content = profile.read_text() if profile.exists() else "# User Profile\n\n"
            if value not in content:
                content = content.rstrip() + f"\n- {key}: {value}\n"
                profile.write_text(content)
        except Exception:
            pass

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search across all memories by keyword relevance."""
        keywords = self._extract_keywords(query)
        results = []
        for subdir in ["facts", "decisions", "sessions"]:
            dir_path = self.vault / subdir
            if dir_path.exists():
                for note in dir_path.glob("*.md"):
                    content = note.read_text()
                    score = self._relevance_score(content, keywords)
                    if score > 0:
                        results.append({
                            "id": note.stem,
                            "role": "fact" if subdir != "sessions" else "session",
                            "content": content[:500],
                            "category": subdir,
                            "importance": score / 10.0,
                            "created_at": datetime.datetime.fromtimestamp(note.stat().st_mtime).isoformat()
                        })
        results.sort(key=lambda x: x["importance"], reverse=True)
        return results[:limit]

    def stats(self) -> dict:
        """Return memory statistics."""
        stats = {"total_rows": 0, "facts": 0, "sessions": 0, "db_path": str(self.vault)}
        for subdir in ["facts", "decisions", "sessions"]:
            dir_path = self.vault / subdir
            if dir_path.exists():
                count = len(list(dir_path.glob("*.md")))
                stats["total_rows"] += count
                if subdir == "sessions":
                    stats["sessions"] = count
                else:
                    stats["facts"] += count
        return stats

    # ── Helpers ───────────────────────────────────────────

    def _extract_keywords(self, text: str) -> list[str]:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        stop = {"this", "that", "with", "from", "your", "what", "then", "when"}
        return [w for w in words if w not in stop]

    def _relevance_score(self, content: str, keywords: list[str]) -> int:
        c = content.lower()
        return sum(1 for kw in keywords if kw in c)
