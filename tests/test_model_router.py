"""
Unit tests for lilim_core.model_router
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from lilim_core.model_router import ModelRouter, DEFAULT_CONFIG


class TestModelRouter(unittest.TestCase):

    def setUp(self):
        self.router = ModelRouter(config=dict(DEFAULT_CONFIG))

    def test_returns_dict_with_required_keys(self):
        result = self.router.route("hello")
        self.assertIn("model", result)
        self.assertIn("tier", result)
        self.assertIn("reason", result)
        self.assertIn("complexity_score", result)

    def test_greeting_routes_local(self):
        result = self.router.route("hi", "conversation")
        self.assertEqual(result["tier"], "local")

    def test_simple_qa_routes_local(self):
        result = self.router.route("what time is it", "simple_qa")
        self.assertEqual(result["tier"], "local")

    def test_tutoring_routes_local(self):
        result = self.router.route("explain the anatomy of the heart", "tutoring")
        self.assertEqual(result["tier"], "local")

    def test_local_only_strategy(self):
        router = ModelRouter(config={**DEFAULT_CONFIG, "strategy": "local-only"})
        result = router.route("write a complex REST API server", "code_generation")
        self.assertEqual(result["tier"], "local")

    def test_remote_only_strategy(self):
        router = ModelRouter(config={**DEFAULT_CONFIG, "strategy": "remote-only"})
        result = router.route("hi", "conversation")
        self.assertTrue(result["tier"].startswith("remote"))

    def test_complexity_score_range(self):
        result = self.router.route("hello")
        score = result["complexity_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_high_complexity_escalates(self):
        # A very complex message should escalate to remote
        complex_msg = (
            "Please do a comprehensive security audit of my entire codebase, "
            "refactor the architecture, and write a detailed technical report with "
            "recommendations for all vulnerabilities found. This is advanced work."
        )
        result = self.router.route(complex_msg, "code_debugging")
        self.assertTrue(
            result["tier"].startswith("remote"),
            f"Expected remote tier for complex message, got: {result['tier']}"
        )

    def test_code_generation_category_routes_remote(self):
        result = self.router.route("write a function", "code_generation")
        # code_generation default is remote.fast
        self.assertTrue(result["tier"].startswith("remote"))

    def test_daily_spend_starts_at_zero(self):
        spend = self.router.get_daily_spend()
        self.assertIsInstance(spend, float)
        self.assertGreaterEqual(spend, 0.0)

    def test_budget_exceeded_falls_back_to_local(self):
        # Set a zero budget
        router = ModelRouter(config={**DEFAULT_CONFIG, "budget_limit_daily": 0.0})
        result = router.route("write me a complex system", "code_generation")
        self.assertEqual(result["tier"], "local")


class TestComplexityEstimate(unittest.TestCase):

    def setUp(self):
        self.router = ModelRouter()

    def test_greeting_low_complexity(self):
        score = self.router._estimate_complexity("hi", "conversation")
        self.assertLess(score, 0.4)

    def test_simple_question_medium_complexity(self):
        score = self.router._estimate_complexity("what is diabetes", "tutoring")
        # Longer than a greeting but not super complex
        self.assertLess(score, 0.7)

    def test_code_with_traceback_high_complexity(self):
        score = self.router._estimate_complexity(
            "```python\nTraceback (most recent call last):\n  File ...\nException: error\n```",
            "code_debugging"
        )
        self.assertGreater(score, 0.5)


if __name__ == "__main__":
    unittest.main()
