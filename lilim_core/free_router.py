"""
Lilim Free-Tier Provider Router — Provider-Agnostic Auto-Detection

Manages a priority-ordered list of free/generous LLM API providers.
Detects which providers are configured (by API key presence), builds a
litellm-compatible model string, and cycles through providers on failure.

Priority (most generous free tier first):
  1. OpenRouter    — single key, 30+ free models with :free suffix
  2. Groq          — fastest inference, 14k req/day free
  3. Google Gemini — 500 req/day Gemini Flash Lite free
  4. Cerebras      — 14.4k req/day, very fast
  5. Cloudflare    — 10k neurons/day, OpenAI-compatible
  6. Cohere        — 1k req/month, good for reasoning
  7. Mistral       — generous token limits, phone required
  8. HuggingFace   — $0.10/month credits, many models
  9. DeepSeek      — cheap, strong reasoning
  10. Any paid provider as final fallback (OpenAI, Anthropic, etc.)

Provider-agnostic: user enters any API key and we detect the provider
from key prefix/format, OR they set PROVIDER_NAME env var.

Usage:
    router = FreeRouter()
    result = await router.call_with_fallback(messages, category="tutoring")
"""

import os
import re
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Provider definitions ──────────────────────────────────────────────────────

@dataclass
class Provider:
    """A configured LLM provider with free tier metadata."""
    name: str
    litellm_prefix: str          # e.g. "openrouter", "groq", "gemini"
    env_key: str                  # Environment variable name for API key
    free_models: list             # Ordered list of free model IDs to try
    daily_limit: int              # Approximate free daily request limit
    tokens_per_min: int           # Approximate free tokens/minute
    base_url: Optional[str] = None  # Optional custom base URL
    extra_env: dict = field(default_factory=dict)  # Extra env vars to set

    def is_configured(self) -> bool:
        return bool(os.environ.get(self.env_key, "").strip())

    def get_model_string(self, model_index: int = 0) -> str:
        """Build litellm model string."""
        model = self.free_models[model_index % len(self.free_models)]
        return f"{self.litellm_prefix}/{model}"

    def setup_env(self):
        """Set any required environment variables."""
        for k, v in self.extra_env.items():
            if not os.environ.get(k):
                os.environ[k] = v
        if self.base_url:
            # litellm uses OPENROUTER_API_BASE etc.
            env_base_key = f"{self.litellm_prefix.upper()}_API_BASE"
            if not os.environ.get(env_base_key):
                os.environ[env_base_key] = self.base_url


# ── Provider registry — priority ordered ─────────────────────────────────────

FREE_PROVIDERS = [
    Provider(
        name="openrouter",
        litellm_prefix="openrouter",
        env_key="OPENROUTER_API_KEY",
        daily_limit=50,  # 50/day free, 1000/day with $10 topup
        tokens_per_min=200_000,
        free_models=[
            # Best free models on OpenRouter — ordered by capability
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemma-3-27b-it:free",
            "google/gemma-3-12b-it:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "google/gemma-3-4b-it:free",
            "liquid/lfm-2.5-1.2b-instruct:free",
        ],
        base_url="https://openrouter.ai/api/v1",
        extra_env={"OR_SITE_URL": "https://lilithlinux.local", "OR_APP_NAME": "Lilim"},
    ),
    Provider(
        name="groq",
        litellm_prefix="groq",
        env_key="GROQ_API_KEY",
        daily_limit=14_400,
        tokens_per_min=30_000,
        free_models=[
            "llama3-70b-8192",      # Best quality on Groq free tier
            "llama-3.3-70b-versatile",
            "llama3-8b-8192",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ],
    ),
    Provider(
        name="gemini",
        litellm_prefix="gemini",
        env_key="GEMINI_API_KEY",
        daily_limit=500,
        tokens_per_min=250_000,
        free_models=[
            "gemini-2.0-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-8b-latest",
        ],
    ),
    Provider(
        name="cerebras",
        litellm_prefix="cerebras",
        env_key="CEREBRAS_API_KEY",
        daily_limit=14_400,
        tokens_per_min=60_000,
        free_models=[
            "llama-3.3-70b",
            "llama-3.1-8b",
        ],
        base_url="https://api.cerebras.ai/v1",
    ),
    Provider(
        name="cloudflare",
        litellm_prefix="cloudflare",
        env_key="CLOUDFLARE_API_TOKEN",
        daily_limit=500,   # 10k neurons/day (rough estimate)
        tokens_per_min=10_000,
        free_models=[
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "@cf/google/gemma-3-12b-it",
            "@cf/meta/llama-3.2-3b-instruct",
            "@cf/qwen/qwen3-30b-a3b-fp8",
        ],
        extra_env={},  # Also needs CLOUDFLARE_ACCOUNT_ID from env
    ),
    Provider(
        name="cohere",
        litellm_prefix="cohere",
        env_key="COHERE_API_KEY",
        daily_limit=33,   # 1000/month ≈ 33/day
        tokens_per_min=20_000,
        free_models=[
            "command-a-03-2025",
            "command-r-plus-08-2024",
            "command-r-08-2024",
            "command-r7b-12-2024",
        ],
    ),
    Provider(
        name="mistral",
        litellm_prefix="mistral",
        env_key="MISTRAL_API_KEY",
        daily_limit=1_000,
        tokens_per_min=500_000,
        free_models=[
            "mistral-small-latest",
            "mistral-medium-latest",
            "open-mistral-7b",
        ],
    ),
    Provider(
        name="huggingface",
        litellm_prefix="huggingface",
        env_key="HUGGINGFACE_API_KEY",
        daily_limit=100,
        tokens_per_min=30_000,
        free_models=[
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
            "microsoft/Phi-3.5-mini-instruct",
        ],
        base_url="https://api-inference.huggingface.co/models",
    ),
    Provider(
        name="deepseek",
        litellm_prefix="deepseek",
        env_key="DEEPSEEK_API_KEY",
        daily_limit=500,
        tokens_per_min=60_000,
        free_models=[
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        base_url="https://api.deepseek.com",
    ),
    # ── Paid fallbacks (only used if no free provider works) ─────────────────
    Provider(
        name="openai",
        litellm_prefix="openai",
        env_key="OPENAI_API_KEY",
        daily_limit=999_999,
        tokens_per_min=90_000,
        free_models=["gpt-4o-mini", "gpt-4o"],
    ),
    Provider(
        name="anthropic",
        litellm_prefix="anthropic",
        env_key="ANTHROPIC_API_KEY",
        daily_limit=999_999,
        tokens_per_min=40_000,
        free_models=["claude-haiku-3-5", "claude-sonnet-4-5"],
    ),
]

# ── Key auto-detection patterns ───────────────────────────────────────────────

KEY_PATTERNS = [
    # (regex_pattern, provider_name, env_var_name)
    (r"^gsk_[A-Za-z0-9]{30,}$", "groq", "GROQ_API_KEY"),
    (r"^sk-or-v1-[A-Za-z0-9]{60,}$", "openrouter", "OPENROUTER_API_KEY"),
    (r"^AIza[A-Za-z0-9_\-]{35,}$", "gemini", "GEMINI_API_KEY"),
    (r"^sk-ant-[A-Za-z0-9\-_]{80,}$", "anthropic", "ANTHROPIC_API_KEY"),
    (r"^sk-[A-Za-z0-9]{40,}$", "openai", "OPENAI_API_KEY"),
    (r"^[A-Za-z0-9]{40}$", "cohere", "COHERE_API_KEY"),
    (r"^csk-[A-Za-z0-9]{40,}$", "cerebras", "CEREBRAS_API_KEY"),
    (r"^hf_[A-Za-z0-9]{30,}$", "huggingface", "HUGGINGFACE_API_KEY"),
    (r"^[a-f0-9]{32}:[A-Za-z0-9\-_]{40,}$", "cloudflare", "CLOUDFLARE_API_TOKEN"),
    (r"^dsk-[A-Za-z0-9]{40,}$", "deepseek", "DEEPSEEK_API_KEY"),
    (r"^[A-Za-z0-9]{32,}$", "mistral", "MISTRAL_API_KEY"),  # broad fallback
]


def detect_provider_from_key(api_key: str) -> Optional[tuple]:
    """
    Auto-detect provider from API key format.
    Returns (provider_name, env_var_name) or None.
    """
    key = api_key.strip()
    for pattern, provider, env_var in KEY_PATTERNS:
        if re.match(pattern, key):
            return provider, env_var
    return None


def register_api_key(api_key: str, provider_hint: Optional[str] = None,
                     model_hint: Optional[str] = None) -> Optional[str]:
    """
    Register an API key with auto-detection or explicit provider hint.
    Sets the appropriate environment variable.
    Returns the detected/registered provider name, or None if undetected.
    """
    api_key = api_key.strip()
    if not api_key:
        return None

    # Try explicit hint first
    if provider_hint:
        for p in FREE_PROVIDERS:
            if p.name == provider_hint.lower():
                os.environ[p.env_key] = api_key
                if model_hint:
                    # Store model preference
                    os.environ[f"LILIM_{p.name.upper()}_MODEL"] = model_hint
                logger.info(f"Registered API key for {p.name}")
                return p.name

    # Auto-detect from key format
    detected = detect_provider_from_key(api_key)
    if detected:
        provider_name, env_var = detected
        os.environ[env_var] = api_key
        logger.info(f"Auto-detected API key as {provider_name}")
        return provider_name

    logger.warning(f"Could not detect provider from API key (length={len(api_key)}). Set LILIM_PROVIDER env var.")
    return None


# ── Model config file ─────────────────────────────────────────────────────────

MODEL_CONFIG_PATH = Path.home() / ".config" / "lilim" / "model-config.json"


def load_and_apply_model_config() -> dict:
    """
    Load model config from file and apply all API keys to environment.
    Returns the loaded config dict.
    """
    config = {}
    if MODEL_CONFIG_PATH.exists():
        try:
            with open(MODEL_CONFIG_PATH) as f:
                config = json.load(f)
        except Exception:
            pass

    # Apply all configured keys
    for provider in FREE_PROVIDERS:
        key_field = f"{provider.name}Key"
        model_field = f"{provider.name}Model"

        key = config.get(key_field, "").strip()
        if key:
            os.environ[provider.env_key] = key
            # Set any extra env
            provider.setup_env()

        # Model override
        model_override = config.get(model_field, "").strip()
        if model_override:
            os.environ[f"LILIM_{provider.name.upper()}_MODEL"] = model_override

    return config


# ── Free Router ────────────────────────────────────────────────────────────────

class FreeRouter:
    """
    Provider-agnostic router that tries free-tier providers in priority order.
    Falls back gracefully through the list on rate limits, auth errors, or timeouts.
    """

    def __init__(self):
        self._load_config()
        self._failure_counts: dict = {}   # provider → consecutive failures
        self._last_success: dict = {}     # provider → timestamp of last success
        self._last_provider_idx = 0       # For round-robin

    def _load_config(self):
        """Load config and apply API keys from config file."""
        self.config = load_and_apply_model_config()

    def reload_config(self):
        """Hot-reload config (called after settings save)."""
        self._load_config()

    def get_configured_providers(self) -> list:
        """Return list of providers that have API keys set."""
        return [p for p in FREE_PROVIDERS if p.is_configured()]

    def get_best_provider(self, category: str = "general") -> Optional[Provider]:
        """
        Return the best available provider for this category.
        Skips providers with recent consecutive failures.
        """
        configured = self.get_configured_providers()
        if not configured:
            return None

        for provider in configured:
            failures = self._failure_counts.get(provider.name, 0)
            if failures >= 3:
                # Back off: skip if failed 3+ times in a row, unless it's been >5min
                last_fail_time = self._last_success.get(f"fail_{provider.name}", 0)
                if time.time() - last_fail_time < 300:
                    continue
                else:
                    # Reset failure count after backoff
                    self._failure_counts[provider.name] = 0

            return provider

        # All failed — return first configured anyway as last resort
        return configured[0] if configured else None

    def get_model_for_provider(self, provider: Provider, category: str = "general") -> str:
        """
        Get the best model string for this provider and category.
        Respects user model overrides from config.
        """
        # Check for user override
        override = os.environ.get(f"LILIM_{provider.name.upper()}_MODEL", "").strip()
        if override:
            return f"{provider.litellm_prefix}/{override}"

        # Category-to-model mapping for quality selection
        # Maps task categories to preferred model index in the provider's list
        category_preference = {
            "code_generation": 0,     # Use best model
            "code_debugging": 0,
            "research": 0,
            "medical": 0,
            "tutoring": 1,            # Second-best is fine for tutoring
            "conversation": -1,       # Use fastest (last) model
            "simple_qa": -1,
            "scheduling": -1,
            "system_admin": 1,
        }

        idx = category_preference.get(category, 0)
        if idx < 0:
            idx = max(0, len(provider.free_models) + idx)

        return provider.get_model_string(idx)

    def record_success(self, provider_name: str):
        """Record successful call to reset failure counter."""
        self._failure_counts[provider_name] = 0
        self._last_success[provider_name] = time.time()

    def record_failure(self, provider_name: str):
        """Record failed call to trigger backoff."""
        self._failure_counts[provider_name] = self._failure_counts.get(provider_name, 0) + 1
        self._last_success[f"fail_{provider_name}"] = time.time()
        logger.warning(f"Provider {provider_name} failure #{self._failure_counts[provider_name]}")

    def build_litellm_call_kwargs(self, provider: Provider, model_str: str,
                                   messages: list, stream: bool = True,
                                   max_tokens: int = 1024) -> dict:
        """Build kwargs for a litellm.completion() call."""
        kwargs = {
            "model": model_str,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            "timeout": 45,          # 45s timeout before we try next provider
            "num_retries": 0,       # We handle retries ourselves
        }

        # Provider-specific extras
        if provider.name == "openrouter":
            kwargs["extra_headers"] = {
                "HTTP-Referer": os.environ.get("OR_SITE_URL", "https://lilithlinux.local"),
                "X-Title": os.environ.get("OR_APP_NAME", "Lilim"),
            }
        elif provider.name == "cloudflare":
            account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
            if account_id:
                kwargs["api_base"] = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"

        if provider.base_url and provider.name not in ("cloudflare",):
            kwargs["api_base"] = provider.base_url

        return kwargs

    async def call_stream(self, messages: list, category: str = "general",
                          max_tokens: int = 1024) -> AsyncGenerator:
        """
        Call the best available free provider with SSE streaming.
        Falls back through providers on failure.
        Yields (token_text, is_error, provider_name) tuples.
        """
        try:
            import litellm
            from litellm import acompletion
            litellm.suppress_debug_info = True
        except ImportError:
            yield ("*litellm not installed — pip install litellm*", True, "none")
            return

        balancing = self.config.get("balancing_strategy", "failover")
        configured = self.get_configured_providers()
        if not configured:
            yield (self._no_provider_message(), True, "none")
            return

        # Determine start index for search
        start_idx = 0
        if balancing == "round-robin":
            start_idx = (self._last_provider_idx + 1) % len(configured)
        
        # Try providers starting from start_idx
        tried_count = 0
        current_idx = start_idx
        
        while tried_count < len(configured):
            provider = configured[current_idx]
            tried_count += 1
            
            # Check for backoff
            failures = self._failure_counts.get(provider.name, 0)
            if failures >= 3:
                last_fail = self._last_success.get(f"fail_{provider.name}", 0)
                if time.time() - last_fail < 300:
                    current_idx = (current_idx + 1) % len(configured)
                    continue
                self._failure_counts[provider.name] = 0

            model_str = self.get_model_for_provider(provider, category)
            provider.setup_env()
            kwargs = self.build_litellm_call_kwargs(
                provider, model_str, messages, stream=True, max_tokens=max_tokens
            )

            logger.info(f"Trying provider: {provider.name} ({balancing}) / {model_str}")
            try:
                stream = await acompletion(**kwargs)
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        yield (delta, False, provider.name)
                
                self.record_success(provider.name)
                self._last_provider_idx = current_idx
                return  # Done — don't try next provider

            except Exception as e:
                err_str = str(e).lower()
                self.record_failure(provider.name)
                
                # If rate limited or auth error, try next provider in loop
                if any(sig in err_str for sig in [
                    "rate limit", "429", "quota", "exceeded",
                    "insufficient_quota", "too many requests",
                    "auth", "401", "403", "invalid api key", "unauthorized",
                    "timeout", "timed out", "connection"
                ]):
                    logger.warning(f"Provider {provider.name} failed — trying next in cycle")
                    current_idx = (current_idx + 1) % len(configured)
                    continue
                else:
                    # For other errors, we might want to fail the request or try next
                    current_idx = (current_idx + 1) % len(configured)
                    continue

        # All providers exhausted
        yield (self._all_failed_message(), True, "exhausted")

    def call_sync(self, messages: list, category: str = "general",
                  max_tokens: int = 1024) -> tuple:
        """
        Synchronous version for non-streaming calls.
        Returns (response_text, provider_name, error_bool).
        """
        try:
            import litellm
            from litellm import completion
            litellm.suppress_debug_info = True
        except ImportError:
            return self._no_provider_message(), "none", True

        configured = self.get_configured_providers()
        if not configured:
            return self._no_provider_message(), "none", True

        for provider in configured:
            failures = self._failure_counts.get(provider.name, 0)
            if failures >= 3:
                last_fail = self._last_success.get(f"fail_{provider.name}", 0)
                if time.time() - last_fail < 300:
                    continue
                self._failure_counts[provider.name] = 0

            model_str = self.get_model_for_provider(provider, category)
            provider.setup_env()
            kwargs = self.build_litellm_call_kwargs(
                provider, model_str, messages, stream=False, max_tokens=max_tokens
            )

            try:
                response = completion(**kwargs)
                text = response.choices[0].message.content or ""
                self.record_success(provider.name)
                return text, provider.name, False
            except Exception as e:
                self.record_failure(provider.name)
                logger.warning(f"{provider.name} failed: {e}")
                continue

        return self._all_failed_message(), "exhausted", True

    def get_status(self) -> dict:
        """Return status of all providers for the settings panel."""
        result = []
        for p in FREE_PROVIDERS:
            result.append({
                "name": p.name,
                "configured": p.is_configured(),
                "daily_limit": p.daily_limit,
                "tokens_per_min": p.tokens_per_min,
                "failures": self._failure_counts.get(p.name, 0),
                "free_models": p.free_models[:3],  # Show first 3
            })
        return {"providers": result, "configured_count": len(self.get_configured_providers())}

    @staticmethod
    def _no_provider_message() -> str:
        return (
            "*Sighs in infernal* No API keys configured. "
            "Open Settings (⚙) and add at least one provider key. "
            "Groq is free — sign up at groq.com."
        )

    @staticmethod
    def _all_failed_message() -> str:
        return (
            "*The flames flicker* All configured providers are either rate-limited or unreachable. "
            "Try again in a few minutes, or add another provider in Settings."
        )
