"""
Lilim Model Router — Plano + LiteLLM Intelligent Routing

Routes each request to the optimal model based on:
- Request complexity (token count, task type, reasoning depth)
- Local model capabilities vs. remote model strengths
- Cost constraints (daily budget caps)
- Latency requirements

Simple requests → local model (fast, free)
Complex requests → best remote model (accurate, cost-efficient)
"""

import os
import re
import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional


# ── Default routing config ────────────────────────────────
DEFAULT_CONFIG = {
    "strategy": "auto",  # "auto", "local-only", "remote-only"
    "local_model": "ollama/qwen3:4b",
    "complexity_threshold": 0.6,
    "remote_models": {
        "fast": "gpt-4o-mini",
        "balanced": "gpt-4o",
        "reasoning": "claude-sonnet-4-20250514",
    },
    "budget_limit_daily": 5.00,  # USD
    "category_routes": {
        "conversation": "local",
        "simple_qa": "local",
        "tutoring": "local",
        "system_admin": "local",
        "scheduling": "local",
        "code_generation": "remote.fast",
        "code_debugging": "remote.reasoning",
        "research": "remote.balanced",
        "file_management": "local",
    },
}

# ── Complexity signals ────────────────────────────────────
COMPLEXITY_SIGNALS = {
    # High complexity indicators
    "high": [
        r"refactor", r"architect", r"design pattern", r"optimize",
        r"debug.*traceback", r"stack trace", r"segfault",
        r"entire project", r"all files", r"codebase",
        r"review.*code", r"security audit", r"vulnerability",
        r"machine learning", r"neural network", r"training",
        r"explain.*in detail", r"comprehensive", r"thorough",
        r"multi.?step", r"complex", r"advanced",
    ],
    # Low complexity indicators
    "low": [
        r"^(hi|hello|hey|thanks|bye|ok)",
        r"what time", r"what day", r"what date",
        r"how are you", r"who are you",
        r"^(yes|no|ok|sure|cool)$",
        r"remind me", r"set a timer",
        r"what is my", r"show me",
    ],
}


class ModelRouter:
    """Routes requests to the optimal model based on complexity and cost."""

    def __init__(self, config: Optional[dict] = None, config_path: Optional[str] = None):
        """Initialize the router.

        Args:
            config: Routing configuration dict
            config_path: Path to routing.toml config file
        """
        self.config = config or dict(DEFAULT_CONFIG)

        if config_path:
            self._load_config(config_path)

        # Cost tracking
        self.cost_log_path = Path(
            os.path.expanduser("~/.local/share/lilim/routing_costs.json")
        )
        self.cost_log_path.parent.mkdir(parents=True, exist_ok=True)

    def route(self, message: str, category: str = "conversation") -> dict:
        """Determine which model to use for this request.

        Args:
            message: The user's message
            category: Task category from PromptEnhancer

        Returns:
            Dict with:
                - model: The model identifier for LiteLLM
                - tier: "local", "remote.fast", "remote.balanced", "remote.reasoning"
                - reason: Why this model was chosen
                - complexity_score: 0-1 estimated complexity
        """
        strategy = self.config.get("strategy", "auto")

        # Fixed strategy overrides
        if strategy == "local-only":
            return self._local_route("Strategy set to local-only")
        if strategy == "remote-only":
            return self._remote_route("fast", "Strategy set to remote-only")

        # Auto routing
        complexity = self._estimate_complexity(message, category)

        # Check budget
        if not self._within_budget():
            return self._local_route(
                f"Daily budget (${ self.config['budget_limit_daily']:.2f}) exceeded, using local model",
                complexity,
            )

        # Category-based routing
        category_routes = self.config.get("category_routes", {})
        preferred_tier = category_routes.get(category, "local")

        # Override with complexity if it exceeds threshold
        threshold = self.config.get("complexity_threshold", 0.6)
        if complexity > threshold and preferred_tier == "local":
            # Upgrade to remote for complex requests
            if complexity > 0.8:
                preferred_tier = "remote.reasoning"
            elif complexity > 0.6:
                preferred_tier = "remote.balanced"
            else:
                preferred_tier = "remote.fast"

        # Route based on tier
        if preferred_tier.startswith("remote"):
            tier_name = preferred_tier.split(".")[-1] if "." in preferred_tier else "fast"
            return self._remote_route(
                tier_name,
                f"Category '{category}' complexity {complexity:.2f} → {preferred_tier}",
                complexity,
            )
        else:
            return self._local_route(
                f"Category '{category}' complexity {complexity:.2f} → local",
                complexity,
            )

    def log_cost(self, model: str, tokens_in: int, tokens_out: int):
        """Log the cost of a model call for budget tracking.

        Args:
            model: Model identifier
            tokens_in: Input token count
            tokens_out: Output token count
        """
        # Approximate costs per 1M tokens
        cost_per_1m = {
            "gpt-4o-mini": {"in": 0.15, "out": 0.60},
            "gpt-4o": {"in": 2.50, "out": 10.00},
            "claude-sonnet-4-20250514": {"in": 3.00, "out": 15.00},
        }

        model_base = model.split("/")[-1] if "/" in model else model
        rates = cost_per_1m.get(model_base, {"in": 0.50, "out": 2.00})

        cost = (tokens_in * rates["in"] / 1_000_000) + (tokens_out * rates["out"] / 1_000_000)

        # Load existing log
        log = self._load_cost_log()
        today = date.today().isoformat()

        if today not in log:
            log[today] = {"total": 0.0, "calls": []}

        log[today]["total"] += cost
        log[today]["calls"].append({
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost": round(cost, 6),
            "time": datetime.now().isoformat(),
        })

        self._save_cost_log(log)

    def get_daily_spend(self) -> float:
        """Get today's total spend in USD."""
        log = self._load_cost_log()
        today = date.today().isoformat()
        return log.get(today, {}).get("total", 0.0)

    # ── Internal routing helpers ──────────────────────────

    def _local_route(self, reason: str, complexity: float = 0.0) -> dict:
        return {
            "model": self.config.get("local_model", "ollama/qwen3:4b"),
            "tier": "local",
            "reason": reason,
            "complexity_score": complexity,
        }

    def _remote_route(self, tier_name: str, reason: str, complexity: float = 0.0) -> dict:
        remote_models = self.config.get("remote_models", DEFAULT_CONFIG["remote_models"])
        model = remote_models.get(tier_name, remote_models.get("fast", "gpt-4o-mini"))
        return {
            "model": model,
            "tier": f"remote.{tier_name}",
            "reason": reason,
            "complexity_score": complexity,
        }

    def _estimate_complexity(self, message: str, category: str) -> float:
        """Estimate request complexity on a 0-1 scale.

        Factors:
        - Message length (longer = more complex)
        - High/low complexity signal keywords
        - Task category default complexity
        - Token count heuristic
        """
        score = 0.3  # Base score

        message_lower = message.lower()

        # Length factor (normalized, caps at ~500 chars)
        length_factor = min(len(message) / 500, 0.3)
        score += length_factor

        # High complexity signals
        for pattern in COMPLEXITY_SIGNALS["high"]:
            if re.search(pattern, message_lower):
                score += 0.15

        # Low complexity signals
        for pattern in COMPLEXITY_SIGNALS["low"]:
            if re.search(pattern, message_lower):
                score -= 0.2

        # Category base complexity
        category_complexity = {
            "conversation": -0.2,
            "simple_qa": -0.1,
            "scheduling": -0.1,
            "file_management": -0.05,
            "system_admin": 0.0,
            "tutoring": 0.0,
            "code_generation": 0.1,
            "research": 0.1,
            "code_debugging": 0.2,
        }
        score += category_complexity.get(category, 0)

        # Code block detection (user pasting code = likely complex)
        if "```" in message or "traceback" in message_lower:
            score += 0.2

        # Multi-step detection
        if any(marker in message_lower for marker in ["first,", "then,", "step 1", "1.", "2.", "3."]):
            score += 0.1

        return max(0.0, min(1.0, score))

    def _within_budget(self) -> bool:
        """Check if we're within the daily budget."""
        limit = self.config.get("budget_limit_daily", 5.00)
        spent = self.get_daily_spend()
        return spent < limit

    def _load_cost_log(self) -> dict:
        try:
            with open(self.cost_log_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_cost_log(self, log: dict):
        # Keep only last 30 days
        cutoff = (datetime.now() - __import__("datetime").timedelta(days=30)).date().isoformat()
        log = {k: v for k, v in log.items() if k >= cutoff}

        with open(self.cost_log_path, "w") as f:
            json.dump(log, f, indent=2)

    def _load_config(self, config_path: str):
        """Load config from a TOML file."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return

        try:
            with open(config_path, "rb") as f:
                toml_config = tomllib.load(f)

            routing = toml_config.get("routing", {})
            self.config.update({
                "strategy": routing.get("strategy", self.config["strategy"]),
                "local_model": routing.get("local_model", self.config["local_model"]),
                "complexity_threshold": routing.get("complexity_threshold", self.config["complexity_threshold"]),
            })

            if "remote_models" in routing:
                self.config["remote_models"].update(routing["remote_models"])

            if "budget_limit_daily" in routing:
                self.config["budget_limit_daily"] = routing["budget_limit_daily"]

            if "categories" in routing:
                self.config["category_routes"].update(routing["categories"])
        except Exception:
            pass
