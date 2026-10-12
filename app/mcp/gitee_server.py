"""FastMCP stdio server for the fixed Gitee resume-template collection."""

import os
import urllib.error
from pathlib import Path
from typing import Any, Callable

from app.services.gitee_template_service import (
    GiteeAPIError,
    cache_gitee_template,
    check_gitee_template_update,
    list_gitee_templates,
    preview_cached_gitee_template,
    preview_gitee_template,
    probe_gitee_access,
)


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


def _token(args: dict[str, Any]) -> str | None:
    # The token lives only for this call.  It is never sourced from an MCP
    # environment variable and is never included in an envelope.
    value = str(args.get("gitee_access_token") or "").strip()
    return value or None


def _cache_root() -> Path:
    data_root = Path(os.getenv("RESUME_AGENT_DATA_ROOT", "./data")).expanduser().resolve()
    return data_root / "templates" / "cache"


def _error(exc: Exception) -> dict[str, Any]:
    status = exc.status_code if isinstance(exc, GiteeAPIError) else None
    if status == 401:
        return _envelope(
            error_code="gitee_auth_failed",
            data={"message": "Gitee 授权码无效或已过期"},
            requires_user=True,
        )
    if status in {403, 429}:
        return _envelope(
            error_code="gitee_rate_limited",
            data={"message": "Gitee API 请求受限，请检查授权码或稍后重试"},
            retryable=True,
            requires_user=True,
        )
    if status == 404:
        return _envelope(error_code="gitee_resource_not_found", requires_user=True)
    if status is not None:
        return _envelope(
            error_code="gitee_http_error",
            data={"status_code": status},
            retryable=status >= 500,
        )
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return _envelope(error_code="gitee_unavailable", retryable=True)
    if isinstance(exc, FileNotFoundError):
        return _envelope(error_code="template_not_found", data={"message": str(exc)})
    if isinstance(exc, (ValueError, OSError)):
        return _envelope(error_code="invalid_template_request", data={"message": str(exc)})
    return _envelope(error_code="gitee_tool_failed", retryable=True)


def _guard(
    handler: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def wrapped(args: dict[str, Any]) -> dict[str, Any]:
        try:
            return handler(args)
        except Exception as exc:
            return _error(exc)

    return wrapped


@_guard
def probe(args: dict[str, Any]) -> dict[str, Any]:
    return _envelope(data=probe_gitee_access(token=_token(args)))


@_guard
def list_templates(args: dict[str, Any]) -> dict[str, Any]:
    result = list_gitee_templates(
        token=_token(args),
        page=max(1, int(args.get("page") or 1)),
        per_page=min(50, max(1, int(args.get("per_page") or 5))),
    )
    return _envelope(data=result)


@_guard
def preview(args: dict[str, Any]) -> dict[str, Any]:
    template_id = str(args.get("template_id") or "").strip()
    file_path = str(args.get("file_path") or "").strip()
    if template_id:
        try:
            return _envelope(
                data=preview_cached_gitee_template(
                    cache_root=_cache_root(), template_id=template_id
                )
            )
        except FileNotFoundError:
            if not file_path:
                raise
    if not file_path:
        return _envelope(error_code="missing_template_reference")
    return _envelope(data=preview_gitee_template(file_path=file_path, token=_token(args)))


@_guard
def download(args: dict[str, Any]) -> dict[str, Any]:
    file_path = str(args.get("file_path") or "").strip()
    if not file_path:
        return _envelope(error_code="missing_template_path")
    result = cache_gitee_template(
        file_path=file_path,
        cache_root=_cache_root(),
        token=_token(args),
        expected_remote_version=str(args.get("remote_version") or "").strip() or None,
        allow_overwrite=bool(args.get("confirm_update", False)),
    )
    return _envelope(data=result)


@_guard
def check_update(args: dict[str, Any]) -> dict[str, Any]:
    template_id = str(args.get("template_id") or "").strip()
    if not template_id:
        return _envelope(error_code="missing_template_id")
    result = check_gitee_template_update(
        cache_root=_cache_root(),
        template_id=template_id,
        token=_token(args),
    )
    return _envelope(data=result, requires_user=result["status"] == "update_available")


def create_server() -> Any:
    """Build the standard FastMCP server; transport is selected only in main."""

    from mcp.server.fastmcp import FastMCP

    server = FastMCP(
        "resume-agent-gitee",
        instructions=(
            "列取、预览、下载并检查固定 Gitee 仓库中 templates/02_简约简历/单页 "
            "目录的 DOCX 简历模板。"
        ),
        log_level="ERROR",
    )

    @server.tool(name="gitee_probe_access", description="检测固定 Gitee 模板目录和授权码")
    def tool_probe(gitee_access_token: str | None = None) -> dict[str, Any]:
        return probe(locals())

    @server.tool(name="gitee_list_resume_templates", description="列取固定目录中的 DOCX 简历模板")
    def tool_list(
        page: int = 1,
        per_page: int = 5,
        gitee_access_token: str | None = None,
    ) -> dict[str, Any]:
        return list_templates(locals())

    @server.tool(name="gitee_preview_template", description="读取远程或本机缓存模板的安全预览元数据")
    def tool_preview(
        template_id: str | None = None,
        file_path: str | None = None,
        gitee_access_token: str | None = None,
    ) -> dict[str, Any]:
        return preview(locals())

    @server.tool(name="gitee_download_template", description="下载并缓存固定目录中的一个 DOCX 模板")
    def tool_download(
        file_path: str,
        remote_version: str | None = None,
        confirm_update: bool = False,
        gitee_access_token: str | None = None,
    ) -> dict[str, Any]:
        return download(locals())

    @server.tool(name="gitee_check_template_update", description="检查缓存模板的 Gitee 版本但不自动覆盖")
    def tool_check_update(
        template_id: str,
        gitee_access_token: str | None = None,
    ) -> dict[str, Any]:
        return check_update(locals())

    return server


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
