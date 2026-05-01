"""
Unit tests for lilim_core.prompt_enhancer
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from lilim_core.prompt_enhancer import PromptEnhancer, TASK_CATEGORIES


class TestPromptClassification(unittest.TestCase):

    def setUp(self):
        self.enhancer = PromptEnhancer()

    def test_code_generation_classified(self):
        cat = self.enhancer._classify_task("write a python script to list files")
        self.assertEqual(cat, "code_generation")

    def test_tutoring_classified(self):
        cat = self.enhancer._classify_task("explain the anatomy of the heart")
        self.assertEqual(cat, "tutoring")

    def test_system_admin_classified(self):
        cat = self.enhancer._classify_task("systemctl status nginx")
        self.assertEqual(cat, "system_admin")

    def test_code_debug_classified(self):
        cat = self.enhancer._classify_task("there's a bug in my code, traceback shows error")
        self.assertEqual(cat, "code_debugging")

    def test_scheduling_classified(self):
        cat = self.enhancer._classify_task("remind me to take my meds in 30 minutes")
        self.assertEqual(cat, "scheduling")

    def test_file_management_classified(self):
        cat = self.enhancer._classify_task("how do I move a file to another folder")
        self.assertEqual(cat, "file_management")

    def test_conversation_default(self):
        cat = self.enhancer._classify_task("hey what's up")
        self.assertEqual(cat, "conversation")

    def test_research_classified(self):
        cat = self.enhancer._classify_task("what is the difference between veins and arteries")
        self.assertEqual(cat, "research")


class TestShouldEnhance(unittest.TestCase):

    def setUp(self):
        self.enhancer = PromptEnhancer()

    def test_greeting_skipped(self):
        self.assertFalse(self.enhancer.should_enhance("hello"))

    def test_hi_skipped(self):
        self.assertFalse(self.enhancer.should_enhance("hi"))

    def test_thanks_skipped(self):
        self.assertFalse(self.enhancer.should_enhance("thanks"))

    def test_long_message_enhanced(self):
        self.assertTrue(self.enhancer.should_enhance("fix my wifi connection, it keeps dropping"))

    def test_technical_question_enhanced(self):
        self.assertTrue(self.enhancer.should_enhance("explain the axial skeleton"))


class TestEnhancedOutput(unittest.TestCase):

    def setUp(self):
        self.enhancer = PromptEnhancer()

    def test_enhance_returns_dict(self):
        result = self.enhancer.enhance("fix my wifi")
        self.assertIn("enhanced_message", result)
        self.assertIn("category", result)

    def test_enhanced_message_is_string(self):
        result = self.enhancer.enhance("help me study anatomy")
        self.assertIsInstance(result["enhanced_message"], str)

    def test_category_is_valid(self):
        result = self.enhancer.enhance("write a bash script")
        self.assertIn(result["category"], TASK_CATEGORIES)

    def test_short_technical_gets_hint(self):
        result = self.enhancer.enhance("fix wifi")
        # Short technical messages should get an enrich hint injected
        self.assertGreater(len(result["enhanced_message"]), len("fix wifi"))


if __name__ == "__main__":
    unittest.main()
