"""FastAPI-friendly lifecycle manager for official MCP stdio clients."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncContextManager, Callable, Mapping, Protocol


_SECRET_ENV_MARKERS = ("API_KEY", "AUTH", "BEARER", "GITHUB_TOKEN", "PASSWORD", "SECRET", "TOKEN")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MCPClientError(RuntimeError):
    error_code = "mcp_client_error"

    def __init__(self, server_name: str, message: str) -> None:
        super().__init__(message)
        self.server_name = server_name


class MCPDependencyError(MCPClientError):
    error_code = "mcp_sdk_missing"


class MCPTimeoutError(MCPClientError):
    error_code = "mcp_timeout"


class MCPProcessExitedError(MCPClientError):
    error_code = "mcp_process_exited"


class MCPProtocolError(MCPClientError):
    error_code = "mcp_protocol_error"


class MCPNotReadyError(MCPClientError):
    error_code = "mcp_not_ready"


class MCPUnknownServerError(MCPClientError):
    error_code = "mcp_unknown_server"


class MCPUnknownToolError(MCPClientError):
    error_code = "mcp_unknown_tool"


@dataclass(frozen=True)
class MCPServerSpec:
    """One local stdio server process definition with non-secret settings."""

    name: str
    command: str
    args: tuple[str, ...] = ()
    cwd: Path | None = None
    env: Mapping[str, str] | None = None
    startup_timeout_seconds: float = 15.0
    call_timeout_seconds: float = 30.0
    close_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.command.strip():
            raise ValueError("MCP server name and command are required")
        for value in (
            self.startup_timeout_seconds,
            self.call_timeout_seconds,
            self.close_timeout_seconds,
        ):
            if value <= 0:
                raise ValueError("MCP timeouts must be positive")
        for key in (self.env or {}):
            upper = str(key).upper()
            if any(marker in upper for marker in _SECRET_ENV_MARKERS):
                raise ValueError("凭据不得通过 MCP 子进程环境变量传递，请使用单次工具参数")

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "args": list(self.args),
            "cwd": str(self.cwd) if self.cwd else None,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "call_timeout_seconds": self.call_timeout_seconds,
        }


class MCPSessionPort(Protocol):
    async def initialize(self) -> Any: ...

    async def list_tools(self) -> Any: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...

    async def send_ping(self) -> Any: ...


SessionConnector = Callable[[MCPServerSpec], AsyncContextManager[MCPSessionPort]]


@dataclass(frozen=True)
class _OfficialClientSDK:
    ClientSession: Any
    StdioServerParameters: Any
    stdio_client: Any


def _load_official_client_sdk() -> _OfficialClientSDK:
    try:
        from mcp import ClientSession, StdioServerParameters  # type: ignore
        from mcp.client.stdio import stdio_client  # type: ignore
    except (ImportError, AttributeError) as exc:
        raise MCPDependencyError(
            "mcp",
            "缺少兼容的官方 MCP Python SDK；请安装项目锁定的 mcp 依赖后重试",
        ) from exc
    return _OfficialClientSDK(
        ClientSession=ClientSession,
        StdioServerParameters=StdioServerParameters,
        stdio_client=stdio_client,
    )


@asynccontextmanager
async def official_stdio_connector(spec: MCPServerSpec):
    """Open one official SDK stdio session with child stderr discarded.

    Tool arguments may contain a session-only GitHub token.  Child stderr is
    therefore routed to the OS null device instead of application logs, while
    stdout remains reserved exclusively for MCP framing.
    """

    sdk = _load_official_client_sdk()
    parameters = sdk.StdioServerParameters(
        command=spec.command,
        args=list(spec.args),
        env=dict(spec.env) if spec.env else None,
        cwd=str(spec.cwd) if spec.cwd else None,
    )
    with open(os.devnull, "w", encoding="utf-8") as error_sink:
        async with sdk.stdio_client(parameters, errlog=error_sink) as streams:
            read_stream, write_stream = streams
            async with sdk.ClientSession(read_stream, write_stream) as session:
                yield session


def _object_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json", by_alias=True)
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    result: dict[str, Any] = {}
    for name in ("name", "description", "inputSchema", "input_schema"):
        if hasattr(value, name):
            key = "inputSchema" if name == "input_schema" else name
            result[key] = copy.deepcopy(getattr(value, name))
    return result


def _tool_list(result: Any) -> list[dict[str, Any]]:
    values = result.get("tools", []) if isinstance(result, dict) else getattr(result, "tools", [])
    tools: list[dict[str, Any]] = []
    for value in values or []:
        item = _object_dict(value)
        if item.get("name"):
            tools.append(item)
    return tools


def _call_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict) and "ok" in result:
        return copy.deepcopy(result)
    structured = None
    if isinstance(result, dict):
        structured = result.get("structuredContent") or result.get("structured_content")
        content = result.get("content") or []
        is_error = bool(result.get("isError") or result.get("is_error"))
    else:
        structured = getattr(result, "structuredContent", None)
        if structured is None:
            structured = getattr(result, "structured_content", None)
        content = getattr(result, "content", None) or []
        is_error = bool(getattr(result, "isError", False) or getattr(result, "is_error", False))
    if isinstance(structured, dict):
        return copy.deepcopy(structured)
    text_blocks: list[str] = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text") if block.get("type") == "text" else None
        else:
            text = getattr(block, "text", None) if getattr(block, "type", None) == "text" else None
        if isinstance(text, str):
            text_blocks.append(text)
            try:
                value = json.loads(text)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                return value
    return {
        "ok": not is_error,
        "data": {"text": "\n".join(text_blocks)} if text_blocks else None,
        "error_code": "mcp_tool_error" if is_error else None,
        "retryable": False,
        "requires_user": False,
    }


def _exception_chain(exc: BaseException) -> list[BaseException]:
    nested = getattr(exc, "exceptions", None)
    if isinstance(nested, tuple):
        result = [exc]
        for item in nested:
            result.extend(_exception_chain(item))
        return result
    return [exc]


def _looks_like_process_exit(exc: BaseException) -> bool:
    direct_types = (EOFError, BrokenPipeError, ConnectionError)
    for item in _exception_chain(exc):
        if isinstance(item, direct_types):
            return True
        if type(item).__name__ in {
            "BrokenResourceError",
            "ClosedResourceError",
            "EndOfStream",
            "IncompleteReadError",
        }:
            return True
        # The official SDK converts EOF from a dead stdio child into McpError
        # before it reaches ClientSession callers.  Treat only its explicit
        # closed-connection sentinel as process exit; other McpError values are
        # normal protocol/tool failures.
        if type(item).__name__ == "McpError" and "connection closed" in str(item).lower():
            return True
    return False


@dataclass
class _ClientState:
    status: str = "stopped"
    started_at: str | None = None
    last_ok_at: str | None = None
    last_error_code: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _ClientCommand:
    operation: str
    arguments: dict[str, Any]
    future: asyncio.Future[Any]


class MCPStdioClient:
    """Own one official SDK session and its stdio child process."""

    def __init__(
        self,
        spec: MCPServerSpec,
        *,
        connector: SessionConnector = official_stdio_connector,
    ) -> None:
        self.spec = spec
        self._connector = connector
        self._session: MCPSessionPort | None = None
        self._state = _ClientState()
        self._lifecycle_lock = asyncio.Lock()
        self._owner_task: asyncio.Task[None] | None = None
        self._commands: asyncio.Queue[_ClientCommand] | None = None

    async def _wait(self, awaitable: Any, timeout: float, operation: str) -> Any:
        try:
            # ``asyncio.timeout`` keeps execution in the current task.  The
            # official SDK's AnyIO cancel scopes must be entered and exited by
            # that same task; ``wait_for`` would wrap the awaitable in another
            # task and break stdio cleanup on shutdown.
            async with asyncio.timeout(timeout):
                return await awaitable
        except TimeoutError as exc:
            raise MCPTimeoutError(
                self.spec.name,
                f"MCP {self.spec.name} {operation} timed out",
            ) from exc

    def _translate(self, exc: Exception, operation: str) -> MCPClientError:
        if isinstance(exc, MCPClientError):
            return exc
        if _looks_like_process_exit(exc):
            return MCPProcessExitedError(
                self.spec.name,
                f"MCP {self.spec.name} process exited during {operation}",
            )
        return MCPProtocolError(
            self.spec.name,
            f"MCP {self.spec.name} protocol failed during {operation}",
        )

    @staticmethod
    def _finish_future(
        future: asyncio.Future[Any],
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    async def _owner_main(self, started: asyncio.Future[None]) -> None:
        """Own the AnyIO-backed SDK context for its complete lifetime.

        AnyIO cancel scopes must be entered and exited by the same asyncio
        task.  API requests therefore submit commands to this owner instead of
        touching the SDK session directly.
        """

        queue = self._commands
        if queue is None:
            self._finish_future(
                started,
                error=MCPProtocolError(self.spec.name, "MCP command queue is unavailable"),
            )
            return
        try:
            async with AsyncExitStack() as stack:
                try:
                    session = await self._wait(
                        stack.enter_async_context(self._connector(self.spec)),
                        self.spec.startup_timeout_seconds,
                        "process start",
                    )
                    await self._wait(
                        session.initialize(),
                        self.spec.startup_timeout_seconds,
                        "initialize",
                    )
                    listed = await self._wait(
                        session.list_tools(),
                        self.spec.startup_timeout_seconds,
                        "tools/list",
                    )
                except Exception as exc:
                    translated = self._translate(exc, "startup")
                    self._state.status = "failed"
                    self._state.last_error_code = translated.error_code
                    self._finish_future(started, error=translated)
                    return

                self._session = session
                self._state.status = "ready"
                self._state.started_at = _utc_iso()
                self._state.last_ok_at = self._state.started_at
                self._state.tools = _tool_list(listed)
                self._finish_future(started)

                while True:
                    command = await queue.get()
                    if command.operation == "close":
                        self._finish_future(command.future)
                        break
                    try:
                        if command.operation == "list_tools":
                            raw = await self._wait(
                                session.list_tools(),
                                self.spec.call_timeout_seconds,
                                "tools/list",
                            )
                            self._state.tools = _tool_list(raw)
                            value: Any = copy.deepcopy(self._state.tools)
                        elif command.operation == "call_tool":
                            tool_name = str(command.arguments.pop("tool_name"))
                            timeout = float(command.arguments.pop("timeout"))
                            value = _call_result(
                                await self._wait(
                                    session.call_tool(tool_name, command.arguments),
                                    timeout,
                                    f"tools/call:{tool_name}",
                                )
                            )
                        elif command.operation == "ping":
                            await self._wait(
                                session.send_ping(),
                                min(self.spec.call_timeout_seconds, 5.0),
                                "ping",
                            )
                            value = None
                        else:
                            raise MCPProtocolError(
                                self.spec.name,
                                f"Unsupported MCP client operation: {command.operation}",
                            )
                    except Exception as exc:
                        translated = self._translate(exc, command.operation)
                        if isinstance(translated, (MCPProcessExitedError, MCPProtocolError)):
                            self._state.status = "failed"
                            self._state.last_error_code = translated.error_code
                        self._finish_future(command.future, error=translated)
                        if isinstance(translated, (MCPProcessExitedError, MCPProtocolError)):
                            break
                    else:
                        self._state.last_ok_at = _utc_iso()
                        self._finish_future(command.future, result=value)
                    finally:
                        # Tool arguments may include a write-only credential.
                        command.arguments.clear()
        except asyncio.CancelledError:
            self._finish_future(started, error=MCPNotReadyError(self.spec.name, "MCP startup cancelled"))
            raise
        finally:
            self._session = None
            while not queue.empty():
                pending = queue.get_nowait()
                pending.arguments.clear()
                self._finish_future(
                    pending.future,
                    error=MCPNotReadyError(self.spec.name, f"MCP {self.spec.name} is not ready"),
                )

    async def _submit(self, operation: str, arguments: Mapping[str, Any] | None = None) -> Any:
        if self._state.status != "ready" or self._commands is None or self._owner_task is None:
            raise MCPNotReadyError(self.spec.name, f"MCP {self.spec.name} is not ready")
        future = asyncio.get_running_loop().create_future()
        await self._commands.put(_ClientCommand(operation, dict(arguments or {}), future))
        return await asyncio.shield(future)

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._state.status == "ready":
                return
            if self._owner_task is not None and not self._owner_task.done():
                self._owner_task.cancel()
                try:
                    await self._owner_task
                except (Exception, asyncio.CancelledError):
                    pass
            self._state.status = "starting"
            self._state.last_error_code = None
            self._commands = asyncio.Queue()
            started = asyncio.get_running_loop().create_future()
            self._owner_task = asyncio.create_task(
                self._owner_main(started),
                name=f"mcp-owner-{self.spec.name}",
            )
            try:
                await asyncio.shield(started)
            except asyncio.CancelledError:
                # The owner remains responsible for closing its own AnyIO
                # scopes; application shutdown will cancel it explicitly.
                raise

    async def close(self) -> None:
        async with self._lifecycle_lock:
            owner = self._owner_task
            queue = self._commands
            self._state.status = "closing" if owner and not owner.done() else "stopped"
            if owner is not None and not owner.done():
                future = asyncio.get_running_loop().create_future()
                if queue is not None:
                    await queue.put(_ClientCommand("close", {}, future))
                try:
                    async with asyncio.timeout(self.spec.close_timeout_seconds):
                        if queue is not None:
                            await asyncio.shield(future)
                        await asyncio.shield(owner)
                except (Exception, asyncio.CancelledError, TimeoutError):
                    owner.cancel()
                    try:
                        await owner
                    except (Exception, asyncio.CancelledError):
                        pass
            self._owner_task = None
            self._commands = None
            self._session = None
            self._state.status = "stopped"
            self._state.tools = []

    async def list_tools(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        if refresh:
            return await self._submit("list_tools")
        if self._state.status != "ready":
            raise MCPNotReadyError(self.spec.name, f"MCP {self.spec.name} is not ready")
        return copy.deepcopy(self._state.tools)

    async def call_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        known = {str(item.get("name")) for item in self._state.tools}
        if tool_name not in known:
            raise MCPUnknownToolError(
                self.spec.name,
                f"MCP {self.spec.name} does not expose tool {tool_name}",
            )
        timeout = timeout_seconds or self.spec.call_timeout_seconds
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")
        # This local value is deliberately never attached to state, health,
        # exceptions or logs.  It may contain a write-only credential.
        call_arguments = dict(arguments or {})
        call_arguments["tool_name"] = tool_name
        call_arguments["timeout"] = timeout
        try:
            return await self._submit("call_tool", call_arguments)
        finally:
            call_arguments.clear()

    async def health(self, *, probe: bool = False) -> dict[str, Any]:
        if probe and self._state.status == "ready":
            try:
                await self._submit("ping")
            except MCPClientError:
                pass
        return {
            "name": self.spec.name,
            "status": self._state.status,
            "started_at": self._state.started_at,
            "last_ok_at": self._state.last_ok_at,
            "last_error_code": self._state.last_error_code,
            "tool_count": len(self._state.tools),
            "tool_names": [str(item.get("name")) for item in self._state.tools],
        }


class MCPClientManager:
    """Own BOSS/GitHub MCP clients for an application lifespan."""

    def __init__(
        self,
        specs: Mapping[str, MCPServerSpec],
        *,
        connector: SessionConnector = official_stdio_connector,
    ) -> None:
        self._clients = {
            name: MCPStdioClient(spec, connector=connector) for name, spec in specs.items()
        }

    def _client(self, server_name: str) -> MCPStdioClient:
        try:
            return self._clients[server_name]
        except KeyError as exc:
            raise MCPUnknownServerError(
                server_name,
                f"Unknown MCP server: {server_name}",
            ) from exc

    async def start(self, server_names: list[str] | tuple[str, ...] | None = None) -> None:
        names = list(server_names) if server_names is not None else list(self._clients)
        started: list[MCPStdioClient] = []
        try:
            for name in names:
                client = self._client(name)
                await client.start()
                started.append(client)
        except Exception:
            for client in reversed(started):
                try:
                    await client.close()
                except Exception:
                    pass
            raise

    async def start_server(self, server_name: str) -> None:
        await self._client(server_name).start()

    async def close(self) -> None:
        # Context exit remains in the application lifespan task that called
        # ``start``; AnyIO-backed stdio transports require this ownership.
        for client in reversed(list(self._clients.values())):
            try:
                await client.close()
            except Exception:
                pass

    async def list_tools(self, server_name: str, *, refresh: bool = False) -> list[dict[str, Any]]:
        return await self._client(server_name).list_tools(refresh=refresh)

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return await self._client(server_name).call_tool(
            tool_name,
            arguments,
            timeout_seconds=timeout_seconds,
        )

    async def health(self, *, probe: bool = False) -> dict[str, dict[str, Any]]:
        values = await asyncio.gather(
            *(client.health(probe=probe) for client in self._clients.values())
        )
        return {str(value["name"]): value for value in values}

    async def __aenter__(self) -> "MCPClientManager":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()


def default_server_specs(
    *,
    project_root: Path | None = None,
    python_executable: str | None = None,
    data_root: Path | None = None,
    edge_path: str | None = None,
    edge_profile_dir: Path | None = None,
) -> dict[str, MCPServerSpec]:
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    executable = python_executable or sys.executable
    # The SDK adds its own conservative inherited environment.  Keep this map
    # deliberately minimal so credentials from the API process cannot leak to
    # stdio children.
    child_env = {
        "RESUME_AGENT_DATA_ROOT": str((data_root or (root / "data")).resolve())
    }
    if edge_profile_dir is None:
        local_app_data = os.getenv("LOCALAPPDATA")
        if os.name == "nt" and local_app_data:
            # Edge sandboxed GPU/renderer processes may be denied access to a
            # profile beneath Desktop.  LocalAppData is Edge's native writable
            # location and prevents the browser from exiting during CDP use.
            edge_profile_dir = Path(local_app_data) / "ResumeAgent" / "edge-profile"
        else:
            edge_profile_dir = Path(child_env["RESUME_AGENT_DATA_ROOT"]) / "edge-profile"
    child_env["RESUME_AGENT_EDGE_PROFILE_DIR"] = str(edge_profile_dir.expanduser().resolve())
    resolved_edge_path = edge_path
    if not resolved_edge_path:
        # The official stdio transport intentionally gives children a minimal
        # environment, which may omit Windows' ProgramFiles variables.  Resolve
        # Edge in the parent and pass only the non-secret executable path.
        try:
            from app.mcp.edge_adapter import detect_edge_path

            resolved_edge_path = detect_edge_path()
        except (ImportError, OSError):
            resolved_edge_path = None
    if resolved_edge_path:
        child_env["RESUME_AGENT_EDGE_PATH"] = str(
            Path(resolved_edge_path).expanduser().resolve()
        )
    return {
        "boss": MCPServerSpec(
            name="boss",
            command=executable,
            args=("-m", "app.mcp.boss_server"),
            cwd=root,
            env=child_env,
            call_timeout_seconds=45.0,
        ),
        "github": MCPServerSpec(
            name="github",
            command=executable,
            args=("-m", "app.mcp.github_server"),
            cwd=root,
            env=child_env,
            call_timeout_seconds=30.0,
        ),
    }


__all__ = [
    "MCPClientError",
    "MCPClientManager",
    "MCPDependencyError",
    "MCPNotReadyError",
    "MCPProcessExitedError",
    "MCPProtocolError",
    "MCPServerSpec",
    "MCPStdioClient",
    "MCPTimeoutError",
    "MCPUnknownServerError",
    "MCPUnknownToolError",
    "default_server_specs",
    "official_stdio_connector",
]
