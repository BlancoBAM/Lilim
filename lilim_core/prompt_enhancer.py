"""
Lilim Prompt Enhancer — Promptomatix-Inspired Automatic Prompt Optimization

Runs transparently in the background before each LLM call:
1. Classifies the request type (Q&A, code, tutoring, scheduling, etc.)
2. Enriches the prompt with memory context, system info, and structure
3. Returns the enhanced prompt for the LLM — user never sees or manages this

For complex or ambiguous prompts, the enhancer expands them with clarified
intent, relevant background, and structured formatting.
"""

import os
import re
import subprocess
from typing import Optional


# ── Task categories and their enhancement strategies ──────
TASK_CATEGORIES = {
    "code_generation": {
        "keywords": ["write", "create", "build", "implement", "make", "code",
                     "script", "function", "class", "program", "api", "app"],
        "enrich": (
            "Provide working, complete code. Include comments explaining key decisions. "
            "If the language is not specified, use Python. Show the full file contents."
        ),
    },
    "code_debugging": {
        "keywords": ["error", "bug", "fix", "broken", "doesn't work", "traceback",
                     "exception", "crash", "fail", "debug", "issue", "problem"],
        "enrich": (
            "Diagnose the root cause step by step. Show the fix with a clear diff. "
            "Explain WHY the error occurred so the user learns."
        ),
    },
    "system_admin": {
        "keywords": ["systemctl", "service", "install", "apt", "package", "config",
                     "permission", "firewall", "disk", "network", "mount", "cron",
                     "systemd", "linux", "ubuntu", "daemon"],
        "enrich": (
            "Provide exact shell commands the user can copy and run. "
            "Explain what each command does before execution. "
            "Check for common pitfalls on Ubuntu/Lilith Linux."
        ),
    },
    "tutoring": {
        "keywords": ["explain", "teach", "study", "quiz", "test", "learn",
                     "anatomy", "physiology", "medical", "biology", "term",
                     "definition", "concept", "example", "practice"],
        "enrich": (
            "Use an ELI10 approach. Break complex concepts into digestible parts. "
            "Include memory aids and real-world analogies. "
            "If quizzing, provide the answer after the user responds."
        ),
    },
    "scheduling": {
        "keywords": ["remind", "schedule", "alarm", "timer", "appointment",
                     "calendar", "later", "tomorrow", "minutes", "hours",
                     "every day", "recurring", "weekly", "daily"],
        "enrich": (
            "Use the zeroclaw cron system to schedule the task. "
            "Confirm the exact time with the user. Show the cron command."
        ),
    },
    "research": {
        "keywords": ["find", "search", "look up", "what is", "who is",
                     "compare", "difference", "versus", "vs", "best",
                     "recommend", "review", "analysis"],
        "enrich": (
            "Provide a structured, well-sourced answer. "
            "If comparing options, use a table. "
            "Cite sources when possible."
        ),
    },
    "file_management": {
        "keywords": ["file", "folder", "directory", "move", "copy", "delete",
                     "rename", "find", "backup", "restore", "zip", "extract"],
        "enrich": (
            "Show exact file paths. Confirm before any destructive operations. "
            "Use safe commands (mv -i, rm -i) for non-technical users."
        ),
    },
    "conversation": {
        "keywords": [],  # Default fallback
        "enrich": "Be concise and conversational. Match the user's energy.",
    },
}


class PromptEnhancer:
    """Automatically enhances user prompts before LLM processing."""

    def __init__(self, memory_manager=None):
        """Initialize the enhancer.

        Args:
            memory_manager: Optional MemoryManager instance for context injection
        """
        self.memory = memory_manager

    def enhance(self, user_message: str, conversation_history: list[dict] = None) -> dict:
        """Enhance a user prompt with context, classification, and structure.

        Args:
            user_message: The raw user message
            conversation_history: Recent conversation messages

        Returns:
            Dict with:
                - enhanced_message: The enriched prompt for the LLM
                - category: Detected task category
                - memory_context: Injected memory context (if any)
                - metadata: Additional enhancement metadata
        """
        # Step 1: Classify the task
        category = self._classify_task(user_message)

        # Step 2: Get memory context
        memory_context = ""
        if self.memory:
            memory_context = self.memory.load_context(user_message)

        # Step 3: Get system context for relevant categories
        system_context = ""
        if category in ("system_admin", "code_debugging", "file_management"):
            system_context = self._get_system_context()

        # Step 4: Build enhanced prompt
        enhanced = self._build_enhanced_prompt(
            user_message=user_message,
            category=category,
            memory_context=memory_context,
            system_context=system_context,
            conversation_history=conversation_history,
        )

        return {
            "enhanced_message": enhanced,
            "category": category,
            "memory_context": memory_context,
            "metadata": {
                "original_length": len(user_message),
                "enhanced_length": len(enhanced),
                "has_memory": bool(memory_context),
                "has_system_context": bool(system_context),
            },
        }

    def _classify_task(self, message: str) -> str:
        """Classify the user's message into a task category."""
        message_lower = message.lower()
        scores = {}

        for category, config in TASK_CATEGORIES.items():
            if not config["keywords"]:
                continue
            score = sum(1 for kw in config["keywords"] if kw in message_lower)
            if score > 0:
                scores[category] = score

        if not scores:
            return "conversation"

        return max(scores, key=scores.get)

    def _get_system_context(self) -> str:
        """Gather relevant system context for sysadmin/debug tasks."""
        context_parts = []

        try:
            # OS info
            result = subprocess.run(
                ["uname", "-a"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                context_parts.append(f"OS: {result.stdout.strip()}")
        except Exception:
            pass

        try:
            # Disk usage
            result = subprocess.run(
                ["df", "-h", "/"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    context_parts.append(f"Disk: {lines[1]}")
        except Exception:
            pass

        try:
            # Memory
            result = subprocess.run(
                ["free", "-h"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    context_parts.append(f"Memory: {lines[1]}")
        except Exception:
            pass

        if not context_parts:
            return ""

        return "\n[System Info: " + " | ".join(context_parts) + "]"

    def _build_enhanced_prompt(
        self,
        user_message: str,
        category: str,
        memory_context: str,
        system_context: str,
        conversation_history: list[dict] = None,
    ) -> str:
        """Build the enhanced prompt from all context sources."""
        parts = []

        # For short/ambiguous messages, expand the prompt
        if len(user_message.split()) < 5 and category != "conversation":
            task_config = TASK_CATEGORIES.get(category, {})
            enrich_hint = task_config.get("enrich", "")
            if enrich_hint:
                parts.append(f"[Task type: {category}. {enrich_hint}]")

        # Add system context
        if system_context:
            parts.append(system_context)

        # Add the original message
        parts.append(user_message)

        # Add memory context as a suffix hint (not visible to user)
        if memory_context:
            parts.append(f"\n[Relevant context from memory:{memory_context}]")

        return "\n".join(parts)

    def should_enhance(self, message: str) -> bool:
        """Determine if a message benefits from enhancement.

        Very simple messages (greetings, single words) skip enhancement.
        """
        # Skip very short casual messages
        if len(message.split()) <= 2:
            casual = {"hi", "hello", "hey", "thanks", "bye", "ok", "yes", "no",
                      "sure", "cool", "nice", "lol", "haha", "yep", "nope"}
            if message.lower().strip("!?.") in casual:
                return False
        return True
