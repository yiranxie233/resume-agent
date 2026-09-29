"""BOSS MCP stdio server backed by a user-visible Edge session."""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from app.mcp.edge_adapter import EdgeActionRequired, EdgeBossAdapter, default_edge_adapter
from app.mcp.stdio_server import StdioMCPServer
from app.services.job_parser import validate_boss_url


_adapter: EdgeBossAdapter | None = None
_adapter_lock = threading.Lock()


def _get_adapter() -> EdgeBossAdapter:
    global _adapter
    if _adapter is None:
        with _adapter_lock:
            if _adapter is None:
                _adapter = default_edge_adapter()
    return _adapter


def _envelope(
    *,
    data: Any = None,
    error_code: str | None = None,
    retryable: bool = False,
    requires_user: bool = False,
) -> dict[str, Any]:
    return {
        "ok": error_code is None,
        "data": data,
        "error_code": error_code,
        "retryable": retryable,
        "requires_user": requires_user,
    }


def _adapter_call(operation: Callable[[], Any]) -> dict[str, Any]:
    """Map adapter outcomes to the stable MCP result envelope."""

    try:
        return _envelope(data=operation())
    except EdgeActionRequired as exc:
        retryable = exc.code in {
            "boss_navigation_failed",
            "boss_network_unavailable",
            "boss_page_unavailable",
            "edge_session_closed",
            "boss_search_session_expired",
        }
        requires_user = exc.code in {
            "boss_city_code_required",
            "boss_login_required",
            "boss_verification_required",
            "edge_not_found",
            "edge_launch_failed",
            "playwright_not_installed",
            "boss_network_unavailable",
            "edge_session_closed",
            "boss_search_session_expired",
        }
        return _envelope(
            data={"message": str(exc)},
            error_code=exc.code,
            retryable=retryable,
            requires_user=requires_user,
        )
    except (TypeError, ValueError) as exc:
        return _envelope(
            data={"message": str(exc)},
            error_code="invalid_tool_arguments",
            requires_user=True,
        )
    except Exception:
        # Unexpected browser internals and local paths stay out of MCP output.
        return _envelope(error_code="boss_adapter_error", retryable=False)


def validate_url(args: dict[str, Any]) -> dict[str, Any]:
    try:
        return _envelope(data={"url": validate_boss_url(str(args.get("url", ""))), "allowed": True})
    except ValueError as exc:
        return _envelope(error_code="invalid_boss_url", data={"message": str(exc)})


def environment(_: dict[str, Any]) -> dict[str, Any]:
    """Inspect local browser prerequisites without opening a BOSS page."""

    return _adapter_call(lambda: _get_adapter().environment())


def check_login(_: dict[str, Any]) -> dict[str, Any]:
    return _adapter_call(lambda: _get_adapter().check_login())


def search_jobs(args: dict[str, Any]) -> dict[str, Any]:
    if not args.get("title") or not args.get("city"):
        return _envelope(error_code="missing_search_fields", data={"required": ["title", "city"]})
    return _adapter_call(
        lambda: _get_adapter().search(
            str(args["title"]),
            str(args["city"]),
            city_code=str(args.get("city_code") or "") or None,
            cursor=str(args.get("cursor") or "") or None,
            limit=int(args.get("limit", 20)),
        )
    )


def get_detail(args: dict[str, Any]) -> dict[str, Any]:
    if not args.get("url"):
        return _envelope(error_code="missing_job_url")
    return _adapter_call(lambda: _get_adapter().detail(str(args["url"])))


def open_job_page(args: dict[str, Any]) -> dict[str, Any]:
    if not args.get("url"):
        return _envelope(error_code="missing_job_url")
    return _adapter_call(lambda: _get_adapter().open_job_page(str(args["url"])))


def extract_post_time(args: dict[str, Any]) -> dict[str, Any]:
    if not args.get("url") and not args.get("text"):
        return _envelope(error_code="missing_detail_source")
    return _adapter_call(
        lambda: _get_adapter().extract_post_time(
            url=str(args.get("url") or "") or None,
            text=str(args.get("text") or "") or None,
        )
    )


def snapshot(args: dict[str, Any]) -> dict[str, Any]:
    if not args.get("url"):
        return _envelope(error_code="missing_job_url")
    return _adapter_call(lambda: _get_adapter().snapshot(str(args["url"])))


def reparse(args: dict[str, Any]) -> dict[str, Any]:
    # A standalone MCP has no database and cannot resolve an internal job_id.
    if not args.get("url"):
        return _envelope(
            error_code="job_url_required",
            data={"message": "重新解析需要职位 source URL"},
            requires_user=True,
        )

    def operation() -> dict[str, Any]:
        job = _get_adapter().detail(str(args["url"]))
        return {"status": "reparsed", "job": job}

    return _adapter_call(operation)


def logout(_: dict[str, Any]) -> dict[str, Any]:
    return _adapter_call(lambda: _get_adapter().logout())


def close_browser(_: dict[str, Any]) -> dict[str, Any]:
    return _adapter_call(lambda: _get_adapter().close_browser())


def _tool_definitions() -> dict[str, tuple[dict[str, Any], Callable[[dict[str, Any]], dict[str, Any]]]]:
    return {
        "boss_environment": (
            {
                "description": "Inspect Playwright and Microsoft Edge prerequisites without navigation",
                "inputSchema": {"type": "object"},
            },
            environment,
        ),
        "boss_validate_url": (
            {
                "description": "Validate an official BOSS URL",
                "inputSchema": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
            validate_url,
        ),
        "boss_check_login": (
            {"description": "Check visible Edge login state", "inputSchema": {"type": "object"}},
            check_login,
        ),
        "boss_search_jobs": (
            {
                "description": "Search BOSS jobs by title and city",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "city": {"type": "string"},
                        "city_code": {
                            "type": "string",
                            "pattern": "^[0-9]{9}$",
                            "description": "Optional explicit BOSS city code",
                        },
                        "cursor": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["title", "city"],
                },
            },
            search_jobs,
        ),
        "boss_get_job_detail": (
            {
                "description": "Get selected job detail",
                "inputSchema": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
            get_detail,
        ),
        "boss_open_job_page": (
            {
                "description": "Open a selected job in the retained collection Edge window",
                "inputSchema": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
            open_job_page,
        ),
        "boss_extract_post_time": (
            {
                "description": "Extract publication time from independent DOM metadata rules",
                "inputSchema": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}, "text": {"type": "string"}},
                },
            },
            extract_post_time,
        ),
        "boss_snapshot_job": (
            {
                "description": "Create a sanitized immutable job snapshot",
                "inputSchema": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
            snapshot,
        ),
        "boss_reparse_job": (
            {
                "description": "Explicitly reparse a selected job URL",
                "inputSchema": {
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}, "url": {"type": "string"}},
                    "required": ["url"],
                },
            },
            reparse,
        ),
        "boss_logout": (
            {"description": "Clear the visible Edge BOSS session", "inputSchema": {"type": "object"}},
            logout,
        ),
        "boss_close_browser": (
            {
                "description": "Close the visible BOSS search window without clearing login state",
                "inputSchema": {"type": "object"},
            },
            close_browser,
        ),
    }


def main() -> None:
    server = StdioMCPServer("resume-agent-boss", "0.1.0", _tool_definitions())
    try:
        server.run()
    finally:
        if _adapter is not None:
            _adapter.close()


if __name__ == "__main__":
    main()
