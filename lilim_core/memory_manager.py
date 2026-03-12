"""
Lilim Memory Manager — Cortex-Mem Integration
Replaces the legacy Rowboat markdown vault with the high-performance Cortex-Mem framework.
Leverages vector semantic search and automatic LLM extraction.
"""

import json
import subprocess
from typing import Optional

# ── Cortex-Mem Integration ────────────────────────────────
DEFAULT_TENANT = "lilim"

class MemoryManager:
    """Persistent knowledge graph interface via cortex-mem CLI."""

    def __init__(self, tenant: Optional[str] = None):
        self.tenant = tenant or DEFAULT_TENANT

    # ── Load context for a conversation ───────────────────

    def load_context(self, query: str = "", max_notes: int = 5, max_chars: int = 2000) -> str:
        """Load relevant memory context using Vector Semantic Search via Cortex-Mem."""
        relevant = []

        # Always include user profile (cortex://user/preferences/user_profile)
        profile_content = self._run_cortex("get", "cortex://user/preferences/user_profile", [])
        if profile_content and not "Error" in profile_content:
            relevant.append(("User Profile", profile_content))

        # Search facts and decisions using Qdrant vector search
        if query:
            search_out = self._run_cortex("search", query, ["--limit", str(max_notes)])
            if search_out:
                relevant.append(("Semantic Memory Search", search_out))

        # Build context block
        if not relevant:
            return ""

        context_parts = []
        total_chars = 0

        for name, content in relevant:
            if total_chars + len(content) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    content = content[:remaining] + "..."
                else:
                    break

            context_parts.append(f"### {name}\n{content.strip()}")
            total_chars += len(content)

        return (
            "\n## Your Memory (persistent context from past conversations via Cortex-Mem)\n"
            + "\n\n".join(context_parts)
            + "\n"
        )

    # ── Save knowledge after a conversation ───────────────

    def extract_and_save(self, messages: list[dict], llm_fn=None):
        """Pass full session content to Cortex-Mem auto-indexer for extraction.
        Cortex-Mem handles LLM JSON extraction natively."""
        
        # Combine messages into a text block
        conversation_text = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in messages[-10:]
        )
        
        # Add to default session thread for cortex-mem event watcher to process
        self._run_cortex("add", conversation_text, ["--thread", "lilim_session", "--role", "assistant"])

    def update_user_profile(self, key: str, value: str):
        """Update a fact in the user profile using Cortex-Mem direct API"""
        # Store as standard memory block, cortex-mem CLI routes dimensions
        self._run_cortex("add", f"Profile update: {key} = {value}", ["--thread", "user_profile_data"])

    # ── Internal helpers ──────────────────────────────────

    def _run_cortex(self, command: str, target: str, args: list) -> str:
        """Execute cortex-mem-cli commands."""
        base_cmd = ["cortex-mem", "--tenant", self.tenant, command]
        if command in ["search", "add", "get"]:
            base_cmd.append(target)
            
        base_cmd.extend(args)
        
        try:
            result = subprocess.run(
                base_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return ""
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""
