import os
import asyncio
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lilim_core import mcp_manager
from lilim_core.tool_executor import ToolExecutor


class MCPConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "mcp-servers.json"
        self.config_patch = patch.object(mcp_manager, "CONFIG_PATH", self.config_path)
        self.config_patch.start()

    def tearDown(self):
        self.config_patch.stop()
        self.temp_dir.cleanup()

    def test_loopback_http_server_is_allowed_and_saved_private(self):
        server = {"name": "browseros", "url": "http://127.0.0.1:9100/mcp"}
        mcp_manager.save_config([server])
        self.assertEqual(mcp_manager.load_config()[0]["transport"], "http")
        self.assertEqual(os.stat(self.config_path).st_mode & 0o777, 0o600)

    def test_remote_http_requires_tls_and_url_cannot_embed_credentials(self):
        with self.assertRaises(mcp_manager.MCPConfigError):
            mcp_manager.validate_server({"name": "unsafe", "url": "http://example.com/mcp"})
        with self.assertRaises(mcp_manager.MCPConfigError):
            mcp_manager.validate_server({"name": "unsafe", "url": "https://user:pass@example.com/mcp"})

    def test_http_headers_are_accepted_but_header_injection_is_rejected(self):
        server = mcp_manager.validate_server({
            "name": "remote", "url": "https://mcp.example.com/mcp",
            "headers": {"Authorization": "Bearer example"},
        })
        self.assertEqual(server["headers"]["Authorization"], "Bearer example")
        with self.assertRaises(mcp_manager.MCPConfigError):
            mcp_manager.validate_server({
                "name": "remote", "url": "https://mcp.example.com/mcp",
                "headers": {"Authorization": "Bearer token\r\nInjected: yes"},
            })

    def test_stdio_config_uses_argument_array_and_valid_env(self):
        result = mcp_manager.validate_server({
            "name": "filesystem", "command": "uvx", "args": ["server-filesystem", "/tmp"],
            "env": {"MCP_TOKEN": "secret"},
        })
        self.assertEqual(result["transport"], "stdio")
        with self.assertRaises(mcp_manager.MCPConfigError):
            mcp_manager.validate_server({"name": "bad", "command": "uvx", "args": "unsafe"})

    def test_server_names_are_unique(self):
        entry = {"name": "local", "url": "http://localhost:8000/mcp"}
        with self.assertRaises(mcp_manager.MCPConfigError):
            mcp_manager.save_config([entry, entry])

    def test_no_servers_needs_no_sdk_and_returns_no_tools(self):
        tools, errors = asyncio.run(mcp_manager.MCPManager().list_tools())
        self.assertEqual(tools, [])
        self.assertEqual(errors, [])

    def test_lists_mcp_tool_metadata_and_calls_configured_tool(self):
        mcp_manager.save_config([{"name": "test", "command": "test-server"}])

        class FakeSession:
            async def list_tools(self):
                return SimpleNamespace(tools=[SimpleNamespace(
                    name="echo", description="Echo a value", inputSchema={"type": "object"},
                    annotations=SimpleNamespace(readOnlyHint=True),
                )])

            async def call_tool(self, tool_name, arguments):
                return SimpleNamespace(
                    content=[SimpleNamespace(text=arguments["value"], type="text")], isError=False,
                )

        async def exercise():
            manager = mcp_manager.MCPManager()
            manager._session = lambda _server: asyncio.sleep(0, result=FakeSession())
            tools, errors = await manager.list_tools()
            self.assertEqual(errors, [])
            self.assertEqual(tools[0]["name"], "echo")
            self.assertTrue(tools[0]["read_only"])
            result = await manager.call_tool("test", "echo", {"value": "mcp-connected"})
            self.assertEqual(result, {"output": "mcp-connected", "is_error": False})

        asyncio.run(exercise())


class BrowserLaunchValidationTests(unittest.TestCase):
    def test_rejects_non_http_urls_without_launching_processes(self):
        executor = ToolExecutor()
        with patch("lilim_core.tool_executor.subprocess.Popen") as launch:
            result = executor.browser_launch("file:///etc/passwd")
        self.assertFalse(result["launched"])
        launch.assert_not_called()

    def test_opens_http_url_with_desktop_opener_when_browseros_is_missing(self):
        executor = ToolExecutor()
        with patch("lilim_core.tool_executor.shutil.which", side_effect=lambda name: "/usr/bin/xdg-open" if name == "xdg-open" else None):
            with patch("lilim_core.tool_executor.Path.exists", return_value=False):
                with patch("lilim_core.tool_executor.subprocess.Popen") as launch:
                    result = executor.browser_launch("https://example.com")
        self.assertTrue(result["launched"])
        self.assertEqual(result["browser"], "xdg-open")
        self.assertEqual(launch.call_args.args[0], ["/usr/bin/xdg-open", "https://example.com"])


if __name__ == "__main__":
    unittest.main()
