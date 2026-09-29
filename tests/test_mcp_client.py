from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.mcp.client import (
    MCPClientManager,
    MCPProcessExitedError,
    MCPServerSpec,
    MCPTimeoutError,
    MCPUnknownServerError,
    _OfficialClientSDK,
    default_server_specs,
    official_stdio_connector,
)
from app.mcp.stdio_server import MCPServerDependencyError, StdioMCPServer


class FakeSession:
    def __init__(self, name: str) -> None:
        self.name = name
        self.initialized = False
        self.closed = False
        self.pings = 0
        self.argument_references: list[dict[str, object]] = []

    async def initialize(self) -> dict[str, str]:
        self.initialized = True
        return {"server": self.name}

    async def list_tools(self) -> SimpleNamespace:
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="echo",
                    description="Echo a correlation value",
                    inputSchema={"type": "object"},
                ),
                SimpleNamespace(
                    name="exit",
                    description="Simulate process exit",
                    inputSchema={"type": "object"},
                ),
            ]
        )

    async def call_tool(self, name: str, arguments: dict[str, object]) -> SimpleNamespace:
        self.argument_references.append(arguments)
        if name == "exit":
            raise EOFError("simulated child stdout closed")
        await asyncio.sleep(float(arguments.get("delay", 0)))
        value = arguments.get("correlation")
        return SimpleNamespace(
            structuredContent={"ok": True, "data": {"correlation": value}},
            content=[],
            isError=False,
        )

    async def send_ping(self) -> None:
        self.pings += 1


class FakeConnector:
    def __init__(self) -> None:
        self.sessions: dict[str, FakeSession] = {}

    def __call__(self, spec: MCPServerSpec):
        @asynccontextmanager
        async def context():
            session = FakeSession(spec.name)
            self.sessions[spec.name] = session
            try:
                yield session
            finally:
                session.closed = True

        return context()


def spec(name: str = "fake", *, call_timeout: float = 0.2) -> MCPServerSpec:
    return MCPServerSpec(
        name=name,
        command=sys.executable,
        args=("-c", "pass"),
        cwd=Path.cwd(),
        startup_timeout_seconds=0.2,
        call_timeout_seconds=call_timeout,
        close_timeout_seconds=0.2,
    )


class MCPClientManagerTests(unittest.IsolatedAsyncioTestCase):
    def test_default_boss_spec_passes_detected_edge_path_to_minimal_environment(self) -> None:
        detected = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
        with patch("app.mcp.edge_adapter.detect_edge_path", return_value=str(detected)):
            specs = default_server_specs(project_root=Path.cwd())

        self.assertEqual(
            specs["boss"].env["RESUME_AGENT_EDGE_PATH"],
            str(detected.resolve()),
        )

    async def test_sync_tool_runs_outside_mcp_event_loop(self) -> None:
        from app.mcp.stdio_server import StdioMCPServer

        observed: list[bool] = []

        def handler(_: dict[str, object]) -> dict[str, object]:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                observed.append(False)
            else:
                observed.append(True)
            return {"ok": True}

        server = StdioMCPServer(
            "test",
            "1",
            {"probe": ({"description": "", "inputSchema": {"type": "object"}}, handler)},
        )
        try:
            result = await server._invoke_tool(handler, {})
            self.assertEqual(result, {"ok": True})
            self.assertEqual(observed, [False])
        finally:
            server.close()

    async def test_lifecycle_initialize_list_call_health_and_close(self) -> None:
        connector = FakeConnector()
        manager = MCPClientManager({"fake": spec()}, connector=connector)
        await manager.start()

        session = connector.sessions["fake"]
        self.assertTrue(session.initialized)
        self.assertEqual([item["name"] for item in await manager.list_tools("fake")], ["echo", "exit"])
        result = await manager.call_tool("fake", "echo", {"correlation": "request-1"})
        self.assertEqual(result["data"]["correlation"], "request-1")
        # The manager does not retain a tool argument object after the request.
        self.assertEqual(session.argument_references[-1], {})

        health = await manager.health(probe=True)
        self.assertEqual(health["fake"]["status"], "ready")
        self.assertEqual(health["fake"]["tool_count"], 2)
        self.assertEqual(session.pings, 1)

        await manager.close()
        self.assertTrue(session.closed)
        self.assertEqual((await manager.health())["fake"]["status"], "stopped")

    async def test_concurrent_calls_keep_responses_correlated(self) -> None:
        connector = FakeConnector()
        manager = MCPClientManager({"fake": spec()}, connector=connector)
        await manager.start()
        slow, fast = await asyncio.gather(
            manager.call_tool("fake", "echo", {"correlation": "slow", "delay": 0.03}),
            manager.call_tool("fake", "echo", {"correlation": "fast", "delay": 0}),
        )
        self.assertEqual(slow["data"]["correlation"], "slow")
        self.assertEqual(fast["data"]["correlation"], "fast")
        await manager.close()

    async def test_timeout_does_not_expose_or_retain_credentials(self) -> None:
        connector = FakeConnector()
        manager = MCPClientManager({"fake": spec(call_timeout=0.01)}, connector=connector)
        await manager.start()
        secret = "github_pat_must_not_leak"
        with self.assertRaises(MCPTimeoutError) as caught:
            await manager.call_tool(
                "fake",
                "echo",
                {"correlation": "late", "delay": 0.1, "github_token": secret},
            )
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(secret, json.dumps(await manager.health(), ensure_ascii=False))
        self.assertEqual(connector.sessions["fake"].argument_references[-1], {})
        await manager.close()

    async def test_process_exit_has_distinct_error_and_failed_health(self) -> None:
        connector = FakeConnector()
        manager = MCPClientManager({"fake": spec()}, connector=connector)
        await manager.start()
        with self.assertRaises(MCPProcessExitedError):
            await manager.call_tool("fake", "exit", {})
        health = await manager.health()
        self.assertEqual(health["fake"]["status"], "failed")
        self.assertEqual(health["fake"]["last_error_code"], "mcp_process_exited")
        await manager.start_server("fake")
        self.assertEqual((await manager.health())["fake"]["status"], "ready")
        await manager.close()

    async def test_unknown_server_is_explicit(self) -> None:
        manager = MCPClientManager({"fake": spec()}, connector=FakeConnector())
        with self.assertRaises(MCPUnknownServerError):
            await manager.start_server("missing")

    @unittest.skipUnless(importlib.util.find_spec("mcp"), "official MCP SDK not installed")
    async def test_real_child_exit_is_reported_without_stderr(self) -> None:
        dead = MCPServerSpec(
            name="dead",
            command=sys.executable,
            args=("-c", "import sys; sys.exit(0)"),
            cwd=Path.cwd(),
            startup_timeout_seconds=2.0,
            call_timeout_seconds=0.5,
            close_timeout_seconds=0.5,
        )
        manager = MCPClientManager({"dead": dead})
        with self.assertRaises(MCPProcessExitedError) as caught:
            await manager.start()
        self.assertEqual(caught.exception.error_code, "mcp_process_exited")
        await manager.close()

    def test_secret_environment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "凭据不得通过"):
            MCPServerSpec(name="unsafe", command=sys.executable, env={"GITHUB_TOKEN": "secret"})


class OfficialTransportBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_official_connector_routes_stderr_to_null_device(self) -> None:
        captured: dict[str, object] = {}

        class Parameters:
            def __init__(self, **values: object) -> None:
                captured["parameters"] = values

        @asynccontextmanager
        async def stdio_client(parameters: object, *, errlog: object):
            captured["errlog"] = errlog
            yield "read", "write"

        class ClientSession:
            def __init__(self, read: object, write: object) -> None:
                self.read = read
                self.write = write

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback) -> None:
                return None

        sdk = _OfficialClientSDK(
            ClientSession=ClientSession,
            StdioServerParameters=Parameters,
            stdio_client=stdio_client,
        )
        with patch("app.mcp.client._load_official_client_sdk", return_value=sdk):
            async with official_stdio_connector(spec()) as session:
                self.assertIsInstance(session, ClientSession)
                sink = captured["errlog"]
                self.assertIsNot(sink, sys.stderr)
                self.assertEqual(os.path.normcase(str(sink.name)), os.path.normcase(os.devnull))

    def test_server_dependency_failure_has_no_protocol_fallback(self) -> None:
        server = StdioMCPServer("test", "1", {})
        with patch(
            "app.mcp.stdio_server._load_official_sdk",
            side_effect=MCPServerDependencyError("missing"),
        ):
            with self.assertRaises(MCPServerDependencyError):
                server.run()

    @unittest.skipUnless(importlib.util.find_spec("mcp"), "official MCP SDK not installed")
    async def test_real_official_stdio_initialize_list_call_and_close(self) -> None:
        manager = MCPClientManager(default_server_specs())
        try:
            await manager.start(["github"])
            names = [item["name"] for item in await manager.list_tools("github")]
            self.assertIn("github_preview_template", names)
            result = await manager.call_tool("github", "github_preview_template", {})
            self.assertFalse(result["ok"])
            self.assertEqual(result["error_code"], "missing_template_reference")
            self.assertEqual((await manager.health(probe=True))["github"]["status"], "ready")
        finally:
            await manager.close()

    @unittest.skipUnless(importlib.util.find_spec("mcp"), "official MCP SDK not installed")
    async def test_real_boss_stdio_validate_tool(self) -> None:
        manager = MCPClientManager(default_server_specs())
        try:
            await manager.start(["boss"])
            names = [item["name"] for item in await manager.list_tools("boss")]
            self.assertIn("boss_validate_url", names)
            result = await manager.call_tool(
                "boss", "boss_validate_url", {"url": "https://www.zhipin.com/job_detail/x"}
            )
            self.assertTrue(result["ok"])
            self.assertTrue(result["data"]["allowed"])
        finally:
            await manager.close()


if __name__ == "__main__":
    unittest.main()
