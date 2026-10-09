"""BOSS MCP stdio server backed by a user-visible Edge session."""
import threading
from collections.abc import Callable
from typing import Any

from app.mcp.edge_adapter import EdgeActionRequired, EdgeBossAdapter, default_edge_adapter
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
    except PermissionError:
        return _envelope(
            data={"message": "Edge 登录目录没有写入权限，请更换为当前用户可写的 profile 目录"},
            error_code="edge_profile_permission_denied",
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


def create_server() -> Any:
    """Build the official SDK FastMCP stdio server.

    Handler functions above intentionally keep their dictionary contract so
    unit tests and the FastAPI layer can invoke the exact same implementation.
    FastMCP owns JSON-schema generation, protocol framing and tool dispatch.
    """

    from mcp.server.fastmcp import FastMCP

    server = FastMCP(
        "resume-agent-boss",
        instructions="通过用户授权的 Microsoft Edge 会话采集 BOSS 职位信息。",
        log_level="ERROR",
    )

    @server.tool(name="boss_environment", description="检查 Playwright 和 Edge 环境，不打开职位页")
    def tool_environment() -> dict[str, Any]:
        return environment({})

    @server.tool(name="boss_validate_url", description="校验 BOSS 官方职位网址")
    def tool_validate_url(url: str) -> dict[str, Any]:
        return validate_url({"url": url})

    @server.tool(name="boss_check_login", description="检查采集 Edge 的 BOSS 登录状态")
    def tool_check_login() -> dict[str, Any]:
        return check_login({})

    @server.tool(name="boss_search_jobs", description="按岗位名称和城市搜索 BOSS 职位")
    def tool_search_jobs(
        title: str,
        city: str,
        city_code: str | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        return search_jobs(
            {
                "title": title,
                "city": city,
                "city_code": city_code,
                "cursor": cursor,
                "limit": limit,
            }
        )

    @server.tool(name="boss_get_job_detail", description="解析用户选中的职位详情")
    def tool_get_detail(url: str) -> dict[str, Any]:
        return get_detail({"url": url})

    @server.tool(name="boss_open_job_page", description="在保留的采集 Edge 中打开职位页")
    def tool_open_job_page(url: str) -> dict[str, Any]:
        return open_job_page({"url": url})

    @server.tool(name="boss_extract_post_time", description="从独立 DOM 元数据规则提取发布时间")
    def tool_extract_post_time(
        url: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        return extract_post_time({"url": url, "text": text})

    @server.tool(name="boss_snapshot_job", description="生成经过清理的职位快照")
    def tool_snapshot(url: str) -> dict[str, Any]:
        return snapshot({"url": url})

    @server.tool(name="boss_reparse_job", description="显式重新解析职位网址")
    def tool_reparse(url: str, job_id: str | None = None) -> dict[str, Any]:
        return reparse({"url": url, "job_id": job_id})

    @server.tool(name="boss_logout", description="清除采集 Edge 中的 BOSS 登录会话")
    def tool_logout() -> dict[str, Any]:
        return logout({})

    @server.tool(name="boss_close_browser", description="关闭采集窗口但保留登录数据")
    def tool_close_browser() -> dict[str, Any]:
        return close_browser({})

    return server


def main() -> None:
    server = create_server()
    try:
        server.run(transport="stdio")
    finally:
        if _adapter is not None:
            _adapter.close()


if __name__ == "__main__":
    main()
