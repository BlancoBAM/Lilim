"""
Unit tests for lilim_core.memory_sqlite
"""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from lilim_core.memory_sqlite import MemorySQLite, MemoryManager


class TestMemorySQLite(unittest.TestCase):

    def setUp(self):
        # Use a temp DB for each test
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.mem = MemorySQLite(db_path=Path(self.tmp.name))

    def tearDown(self):
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_save_and_retrieve_turn(self):
        self.mem.save_turn("user", "hello there", session_id="test")
        recent = self.mem.get_recent_session("test", n=5)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["content"], "hello there")
        self.assertEqual(recent[0]["role"], "user")

    def test_save_multiple_turns_ordered(self):
        self.mem.save_turn("user", "first message", session_id="sess1")
        self.mem.save_turn("assistant", "first reply", session_id="sess1")
        self.mem.save_turn("user", "second message", session_id="sess1")

        recent = self.mem.get_recent_session("sess1", n=10)
        self.assertEqual(len(recent), 3)
        # Should be in chronological order
        self.assertEqual(recent[0]["content"], "first message")
        self.assertEqual(recent[2]["content"], "second message")

    def test_save_fact(self):
        row_id = self.mem.save_fact("The user is studying anatomy", category="academic")
        self.assertIsInstance(row_id, int)
        self.assertGreater(row_id, 0)

    def test_save_preference(self):
        row_id = self.mem.save_preference("name", "Alice")
        self.assertIsInstance(row_id, int)

    def test_keyword_search(self):
        self.mem.save_fact("User is studying anatomy", category="medical")
        self.mem.save_fact("User likes Python programming", category="general")

        results = self.mem.search("anatomy", limit=5)
        self.assertTrue(any("anatomy" in r["content"].lower() for r in results))

    def test_load_context_empty(self):
        ctx = self.mem.load_context("some query")
        # Empty DB returns empty string
        self.assertEqual(ctx, "")

    def test_load_context_with_facts(self):
        self.mem.save_fact("User name is Alice", category="preference", importance=0.9)
        ctx = self.mem.load_context("who am I")
        self.assertIn("Alice", ctx)

    def test_stats_returns_dict(self):
        stats = self.mem.stats()
        self.assertIn("total_rows", stats)
        self.assertIn("facts", stats)
        self.assertIn("sessions", stats)

    def test_clear_session_removes_turns(self):
        self.mem.save_turn("user", "message", session_id="to_clear")
        self.mem.save_fact("a fact", session_id="to_clear")

        self.mem.clear_session("to_clear")
        recent = self.mem.get_recent_session("to_clear")
        self.assertEqual(len(recent), 0)
        # Facts should be preserved
        stats = self.mem.stats()
        self.assertGreater(stats["facts"], 0)

    def test_extract_and_save(self):
        messages = [
            {"role": "user", "content": "My name is Bob and I am studying anatomy"},
            {"role": "assistant", "content": "Nice to meet you, Bob!"},
        ]
        self.mem.extract_and_save(messages, session_id="extract_test")

        recent = self.mem.get_recent_session("extract_test")
        self.assertEqual(len(recent), 2)

    def test_category_classification(self):
        cat = self.mem._classify_category("I need help with anatomy and bone structure")
        self.assertEqual(cat, "anatomy")

    def test_category_linux(self):
        cat = self.mem._classify_category("how do I use systemctl to restart a service")
        self.assertEqual(cat, "linux")

    def test_category_general_fallback(self):
        cat = self.mem._classify_category("the weather is nice today")
        self.assertEqual(cat, "general")


class TestMemoryManagerCompat(unittest.TestCase):
    """Test that MemoryManager (compat wrapper) works the same as MemorySQLite."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        # Patch the db_path
        self.mem = MemoryManager.__new__(MemoryManager)
        MemorySQLite.__init__(self.mem, db_path=Path(self.tmp.name))

    def tearDown(self):
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_update_user_profile(self):
        self.mem.update_user_profile("study_subject", "medical assisting")
        stats = self.mem.stats()
        self.assertGreater(stats["facts"], 0)

    def test_load_context_callable(self):
        ctx = self.mem.load_context()
        self.assertIsInstance(ctx, str)

    def test_extract_and_save_callable(self):
        self.mem.extract_and_save([{"role": "user", "content": "hello"}])


if __name__ == "__main__":
    unittest.main()
