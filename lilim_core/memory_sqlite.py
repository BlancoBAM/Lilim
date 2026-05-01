"""
Lilim Memory — SQLite Backend (v1)

Replaces the cortex-mem external binary with a fully self-contained
SQLite store that needs zero extra installs beyond Python's stdlib.

Features:
  - Store conversation turns (user / assistant messages)
  - Store extracted facts (name, subject area, importance)
  - Retrieve relevant context by keyword search + recency
  - Auto-trim old/unimportant memories when DB grows large
  - Export / import for backups

DB lives at: ~/.local/share/lilim/memory.db
"""

import json
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# ── Constants ────────────────────────────────────────────────
DB_PATH = Path.home() / ".local" / "share" / "lilim" / "memory.db"
MAX_TURNS_STORED = 500          # Rolling window of stored turns
MAX_CONTEXT_CHARS = 2500        # Max chars injected into prompts
FACT_IMPORTANCE_DECAY = 0.95    # Multiply importance by this each day unused

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,
    role        TEXT NOT NULL,       -- 'user' | 'assistant' | 'fact' | 'preference'
    content     TEXT NOT NULL,
    category    TEXT DEFAULT 'general',
    keywords    TEXT DEFAULT '',     -- space-separated keyword list for search
    importance  REAL DEFAULT 0.5,    -- 0.0-1.0; higher = kept longer
    created_at  TEXT NOT NULL,
    last_used   TEXT
);
CREATE INDEX IF NOT EXISTS idx_session   ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_category  ON memories(category);
CREATE INDEX IF NOT EXISTS idx_role      ON memories(role);
CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
"""

# Categories we recognise
KNOWN_CATEGORIES = {
    "anatomy", "physiology", "medical", "clinical", "pharmacology",
    "linux", "system", "network",
    "academic", "study", "exam",
    "preference", "general",
}

# Patterns that help extract facts from conversations
FACT_PATTERNS = [
    (r"my name is (\w+)", "preference"),
    (r"i(?:'m| am) (?:a |an )?(.+?)(?:\.|,|$)", "preference"),
    (r"i(?:'m| am) studying (.+?)(?:\.|,|$)", "academic"),
    (r"i(?:'m| am) learning (.+?)(?:\.|,|$)", "academic"),
    (r"remind me (?:about )?(.+?)(?:\.|,|$)", "general"),
    (r"i(?:'ve| have) been having (.+?) (?:issues?|problems?|trouble)", "general"),
]


class MemorySQLite:
    """Persistent memory store using SQLite. Zero external dependencies."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Public API ────────────────────────────────────────────

    def save_turn(self, role: str, content: str, session_id: str = "default",
                  category: str = "general") -> int:
        """Save a single conversation turn (user message or assistant reply).

        Returns the row id.
        """
        keywords = self._extract_keywords(content)
        now = datetime.utcnow().isoformat()

        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO memories
                   (session_id, role, content, category, keywords, importance, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, role, content, category, keywords, 0.5, now),
            )
            row_id = cur.lastrowid

        self._trim_old_turns()
        return row_id

    def save_fact(self, content: str, category: str = "general",
                  importance: float = 0.8, session_id: str = "default") -> int:
        """Save an extracted fact with higher importance than a turn."""
        keywords = self._extract_keywords(content)
        now = datetime.utcnow().isoformat()

        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO memories
                   (session_id, role, content, category, keywords, importance, created_at)
                   VALUES (?, 'fact', ?, ?, ?, ?, ?)""",
                (session_id, content, category, keywords, importance, now),
            )
            return cur.lastrowid

    def save_preference(self, key: str, value: str) -> int:
        """Save a user preference (name, likes, goals, etc.)."""
        content = f"{key}: {value}"
        return self.save_fact(content, category="preference", importance=0.95)

    def extract_and_save(self, messages: list[dict], session_id: str = "default"):
        """Process a list of chat messages, save turns and auto-extract facts."""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue
            cat = self._classify_category(content)
            self.save_turn(role, content, session_id=session_id, category=cat)

        # Auto-extract facts from user messages
        user_text = " ".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )
        self._auto_extract_facts(user_text, session_id=session_id)

    def load_context(self, query: str = "", max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """Load relevant memory context to inject into a prompt.

        Returns a formatted string block, or empty string if nothing useful.
        """
        parts: list[str] = []
        total = 0

        # 1. High-importance facts + preferences (always include)
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT content, category FROM memories
                   WHERE role IN ('fact', 'preference') AND importance >= 0.7
                   ORDER BY importance DESC, created_at DESC
                   LIMIT 10""",
            ).fetchall()

        for content, cat in rows:
            snippet = f"[{cat}] {content}"
            if total + len(snippet) > max_chars * 0.5:
                break
            parts.append(snippet)
            total += len(snippet)
            self._touch(content)  # Update last_used

        # 2. Recent session turns relevant to this query
        if query:
            keyword_clause, kw_params = self._keyword_clause(query)
            with self._conn() as conn:
                rows = conn.execute(
                    f"""SELECT content, role FROM memories
                        WHERE role IN ('user', 'assistant')
                        AND {keyword_clause}
                        ORDER BY created_at DESC
                        LIMIT 6""",
                    kw_params,
                ).fetchall()

            for content, role in rows:
                snippet = f"[prev {role}] {content[:200]}"
                if total + len(snippet) > max_chars:
                    break
                parts.append(snippet)
                total += len(snippet)

        # 3. Most recent turns (for continuity), regardless of query
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT content, role FROM memories
                   WHERE role IN ('user', 'assistant')
                   ORDER BY created_at DESC
                   LIMIT 4""",
            ).fetchall()

        for content, role in reversed(rows):
            snippet = f"[recent {role}] {content[:200]}"
            if total + len(snippet) > max_chars:
                break
            if snippet not in "".join(parts):  # avoid duplicates
                parts.append(snippet)
                total += len(snippet)

        if not parts:
            return ""

        return (
            "\n\n## Memory (from past conversations)\n"
            + "\n".join(parts)
            + "\n"
        )

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Full keyword search across all memory rows."""
        keyword_clause, kw_params = self._keyword_clause(query)
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT id, role, content, category, importance, created_at
                    FROM memories
                    WHERE {keyword_clause}
                    ORDER BY importance DESC, created_at DESC
                    LIMIT ?""",
                [*kw_params, limit],
            ).fetchall()

        return [
            {
                "id": r[0], "role": r[1], "content": r[2],
                "category": r[3], "importance": r[4], "created_at": r[5],
            }
            for r in rows
        ]

    def get_recent_session(self, session_id: str = "default", n: int = 20) -> list[dict]:
        """Return the last N turns of a session in chronological order."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT role, content FROM memories
                   WHERE session_id = ? AND role IN ('user', 'assistant')
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (session_id, n),
            ).fetchall()

        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def clear_session(self, session_id: str):
        """Remove all turns for a specific session (facts are preserved)."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM memories WHERE session_id = ? AND role IN ('user', 'assistant')",
                (session_id,),
            )

    def stats(self) -> dict:
        """Return memory statistics."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            facts = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE role IN ('fact', 'preference')"
            ).fetchone()[0]
            sessions = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM memories"
            ).fetchone()[0]

        return {"total_rows": total, "facts": facts, "sessions": sessions, "db_path": str(self.db_path)}

    # ── Internal helpers ──────────────────────────────────────

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _extract_keywords(self, text: str) -> str:
        """Extract a set of lowercase keywords from text for search indexing."""
        # Remove punctuation, split, deduplicate, skip stopwords
        stopwords = {
            "i", "a", "an", "the", "and", "or", "but", "in", "on", "at",
            "to", "for", "of", "with", "is", "was", "are", "be", "been",
            "it", "this", "that", "my", "me", "you", "we", "they",
        }
        words = re.findall(r"[a-z]+", text.lower())
        keywords = {w for w in words if len(w) > 2 and w not in stopwords}
        return " ".join(sorted(keywords))

    def _classify_category(self, text: str) -> str:
        """Guess a memory category for a piece of text."""
        text_lower = text.lower()
        category_keywords = {
            "anatomy": ["bone", "muscle", "organ", "tissue", "skeleton", "anatomy",
                        "axial", "appendicular", "femur", "tibia", "cranium"],
            "physiology": ["heart", "blood", "pressure", "pulse", "respiration",
                           "cardiac", "vascular", "homeostasis", "physiology"],
            "medical": ["diagnosis", "symptom", "treatment", "medication", "dose",
                        "clinical", "patient", "medical", "nursing", "assistant",
                        "pharmacology", "drug", "injection", "vital"],
            "linux": ["systemctl", "apt", "package", "install", "config", "service",
                      "terminal", "command", "file", "directory", "linux", "ubuntu"],
            "academic": ["study", "exam", "quiz", "test", "class", "lecture",
                         "assignment", "grade", "semester", "course"],
        }
        scores = {}
        for cat, words in category_keywords.items():
            scores[cat] = sum(1 for w in words if w in text_lower)

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    def _auto_extract_facts(self, text: str, session_id: str = "default"):
        """Try to extract memorable facts from user text using regex patterns."""
        for pattern, category in FACT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fact = match.group(0).strip().rstrip(".,;")
                if len(fact) > 5:
                    self.save_fact(fact, category=category, importance=0.85,
                                   session_id=session_id)

    def _keyword_clause(self, query: str) -> tuple[str, list]:
        """Build a SQLite LIKE clause for keyword matching."""
        keywords = re.findall(r"[a-z]+", query.lower())
        # Filter to meaningful words (>3 chars)
        keywords = [k for k in keywords if len(k) > 3][:5]
        if not keywords:
            return "1=1", []

        clauses = " OR ".join("keywords LIKE ?" for _ in keywords)
        params = [f"%{k}%" for k in keywords]
        return f"({clauses})", params

    def _touch(self, content: str):
        """Update last_used timestamp for a memory row (used in context loading)."""
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE memories SET last_used = ? WHERE content = ?",
                (now, content),
            )

    def _trim_old_turns(self):
        """Keep only the most recent MAX_TURNS_STORED conversation turns."""
        with self._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE role IN ('user', 'assistant')"
            ).fetchone()[0]

            if count > MAX_TURNS_STORED:
                excess = count - MAX_TURNS_STORED
                conn.execute(
                    """DELETE FROM memories WHERE id IN (
                        SELECT id FROM memories
                        WHERE role IN ('user', 'assistant')
                        ORDER BY importance ASC, created_at ASC
                        LIMIT ?
                    )""",
                    (excess,),
                )


# ── Convenience: drop-in replacement for MemoryManager ───────

class MemoryManager(MemorySQLite):
    """Backwards-compatible wrapper. Replaces the cortex-mem-based MemoryManager."""

    def __init__(self, tenant: Optional[str] = None):
        # 'tenant' param kept for API compatibility, ignored in SQLite impl
        super().__init__()

    def update_user_profile(self, key: str, value: str):
        self.save_preference(key, value)
