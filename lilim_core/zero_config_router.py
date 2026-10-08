"""
Lilim Zero-Config Router — Free Providers With No API Keys

These providers work out of the box, no sign-up required.
They are tried BEFORE asking for API keys, giving users an
immediate working experience on first launch.

Sources: tgpt free providers (https://github.com/aandrew-me/tgpt)
"""

import asyncio
import json
import logging
import re
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)

# ── Provider definitions ─────────────────────────────────────────────────────

ZERO_CONFIG_PROVIDERS = [
    {
        "name": "pollinations",
        "label": "Pollinations",
        "description": "Free, no API key — many open models",
        "url": "https://text.pollinations.ai/openai",
        "model": "openai",   # default; supports many models
        "type": "openai_compat",
        "priority": 1,
    },
    {
        "name": "opencode",
        "label": "OpenCode",
        "description": "Free — opencode.ai/zen, deepseek-v4-flash-free",
        "url": "https://opencode.ai/api/chat/completions",
        "api_key": "public",
        "model": "deepseek-v4-flash-free",
        "type": "openai_compat",
        "priority": 2,
    },
    {
        "name": "fx",
        "label": "Fx",
        "description": "Free — fx.sh gateway",
        "url": "https://fx.sh/api/v1/chat/completions",
        "model": "zai/glm-5.2",
        "type": "openai_compat",
        "priority": 3,
    },
    {
        "name": "aitopia",
        "label": "Aitopia",
        "description": "Free — uses gpt-4o-mini by default",
        "url": "https://extensions.aitopia.ai/api/chat",
        "model": "gpt-4o-mini",
        "type": "openai_compat",
        "priority": 4,
    },
    {
        "name": "powerbrain",
        "label": "PowerBrain",
        "description": "Free — no API key",
        "url": "https://chatbot.theb.ai/api/chat/completions",
        "model": "gpt-3.5-turbo",
        "type": "openai_compat",
        "priority": 5,
    },
]

# Isou uses a different API style (searxng-enhanced deepseek)
ISOU_URL = "https://isou.chat/api/chat"

# KoboldAI uses the KoboldCPP API
KOBOLD_URL = "https://koboldai-koboldcpp-tiefighter.hf.space/api/v1/generate"


class ZeroConfigRouter:
    """
    Routes requests through zero-config free providers.
    Falls back through the list on failure.
    Integrates with FreeRouter as a pre-step before requiring API keys.
    """

    def __init__(self):
        self._failures: dict = {}
        self._enabled = True  # Can be disabled in settings

    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    def get_providers(self) -> list:
        """Return all zero-config providers, sorted by priority."""
        return sorted(ZERO_CONFIG_PROVIDERS, key=lambda p: p["priority"])

    async def call_stream(
        self,
        messages: list,
        category: str = "general",
        max_tokens: int = 1024,
        preferred_provider: Optional[str] = None,
    ) -> AsyncGenerator:
        """
        Stream a response from the best available zero-config provider.
        Yields (token_text, is_error, provider_name) tuples.
        """
        try:
            import httpx
        except ImportError:
            yield ("*httpx not installed — pip install httpx*", True, "none")
            return

        providers = self.get_providers()
        if preferred_provider:
            providers = sorted(
                providers,
                key=lambda p: 0 if p["name"] == preferred_provider else 1
            )

        for provider in providers:
            name = provider["name"]
            failures = self._failures.get(name, 0)
            if failures >= 3:
                continue

            try:
                async for token, is_err, pname in self._call_provider_stream(
                    provider, messages, max_tokens
                ):
                    if is_err:
                        self._failures[name] = failures + 1
                        break
                    yield token, is_err, pname
                    return
            except Exception as e:
                logger.warning(f"Zero-config provider {name} failed: {e}")
                self._failures[name] = failures + 1
                continue

        yield (
            "*No zero-config providers available right now. "
            "Add a free API key in Settings to unlock more options.*",
            True,
            "none"
        )

    async def _call_provider_stream(
        self,
        provider: dict,
        messages: list,
        max_tokens: int,
    ) -> AsyncGenerator:
        """Stream from an OpenAI-compatible zero-config endpoint."""
        import httpx

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Lilim/2.0 (Lilith Linux)",
        }
        api_key = provider.get("api_key", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = {
            "model": provider["model"],
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }

        url = provider["url"]
        name = provider["name"]

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code >= 400:
                    yield (f"*{name} returned HTTP {resp.status_code}*", True, name)
                    return

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if raw == "[DONE]":
                        return
                    try:
                        chunk = json.loads(raw)
                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        ) or ""
                        if delta:
                            yield (delta, False, name)
                    except Exception:
                        continue

    def call_sync(
        self,
        messages: list,
        category: str = "general",
        max_tokens: int = 512,
    ) -> tuple:
        """
        Synchronous call (used for testing and non-streaming endpoints).
        Returns (response_text, provider_name, error_bool).
        """
        try:
            import httpx

            for provider in self.get_providers():
                name = provider["name"]
                if self._failures.get(name, 0) >= 3:
                    continue

                headers = {"Content-Type": "application/json", "User-Agent": "Lilim/2.0"}
                api_key = provider.get("api_key", "")
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                body = {
                    "model": provider["model"],
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "stream": False,
                }

                try:
                    resp = httpx.post(
                        provider["url"],
                        json=body,
                        headers=headers,
                        timeout=20.0,
                        follow_redirects=True,
                    )
                    if resp.status_code < 400:
                        data = resp.json()
                        text = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        if text:
                            return text, name, False
                except Exception as e:
                    logger.warning(f"Zero-config {name} sync failed: {e}")
                    self._failures[name] = self._failures.get(name, 0) + 1

        except ImportError:
            pass

        return "*All zero-config providers unavailable. Add API keys in Settings.*", "none", True

    def get_status(self) -> list:
        """Return status of all zero-config providers."""
        return [
            {
                "name": p["name"],
                "label": p["label"],
                "description": p["description"],
                "failures": self._failures.get(p["name"], 0),
                "available": self._failures.get(p["name"], 0) < 3,
            }
            for p in self.get_providers()
        ]
