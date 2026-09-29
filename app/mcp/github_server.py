"""GitHub Chinese resume-template MCP server."""
from __future__ import annotations

import os
import re
import urllib.error
from pathlib import Path
from typing import Any, Callable

from app.mcp.stdio_server import StdioMCPServer
from app.services.template_service import (
    cache_github_template,
    check_github_template_update,
    get_github_repository_license,
    preview_cached_template,
    preview_github_template,
    search_github_templates,
)


_TEMPLATE_ID_RE = re.compile(r"^github-[0-9a-f]{24}$")


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
    # The token exists only in this tool-call object. It is never read from a
    # persisted config/environment fallback and is never returned or logged.
    value = str(args.get("github_token") or "").strip()
    return value or None


def _cache_root() -> Path:
    data_root = Path(os.getenv("RESUME_AGENT_DATA_ROOT", "./data")).expanduser().resolve()
    return data_root / "templates" / "cache"


def _error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 401:
            return _envelope(
                error_code="github_auth_failed",
                data={"message": "GitHub Token 无效或已过期"},
                requires_user=True,
            )
        if exc.code in {403, 429}:
            return _envelope(
                error_code="github_rate_limited",
                data={"message": "GitHub API 请求受限，请稍后重试或为本次会话提供 Token"},
                retryable=True,
                requires_user=True,
            )
        if exc.code == 404:
            return _envelope(error_code="github_resource_not_found")
        return _envelope(error_code="github_http_error", data={"status_code": exc.code})
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return _envelope(error_code="github_unavailable", retryable=True)
    if isinstance(exc, FileNotFoundError):
        return _envelope(error_code="template_not_found", data={"message": str(exc)})
    if isinstance(exc, (ValueError, OSError)):
        return _envelope(error_code="invalid_template_request", data={"message": str(exc)})
    return _envelope(error_code="github_tool_failed")


def _guard(
    handler: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def wrapped(args: dict[str, Any]) -> dict[str, Any]:
        try:
            return handler(args)
        except Exception as exc:  # MCP errors must stay in the versioned envelope.
            return _error(exc)

    return wrapped


@_guard
def search(args: dict[str, Any]) -> dict[str, Any]:
    result = search_github_templates(
        query=str(args.get("query") or "中文 简历 模板"),
        language=str(args.get("language") or "zh"),
        file_type=str(args.get("file_type") or "") or None,
        page=max(1, int(args.get("page") or 1)),
        per_page=min(5, max(1, int(args.get("per_page") or 5))),
        token=_token(args),
    )
    return _envelope(data=result)


@_guard
def license_info(args: dict[str, Any]) -> dict[str, Any]:
    repository = str(args.get("repository") or "").strip()
    if not repository:
        return _envelope(error_code="missing_repository")
    result = get_github_repository_license(repository, token=_token(args))
    if result["license_allowed"]:
        return _envelope(data=result)
    error_code = (
        "template_license_unknown"
        if result["status"] == "unknown"
        else "template_license_not_allowed"
    )
    return _envelope(error_code=error_code, data=result)


def _cached_preview(template_id: str) -> dict[str, Any]:
    if not _TEMPLATE_ID_RE.fullmatch(template_id):
        raise ValueError("模板 ID 不合法")
    template_root = (_cache_root() / template_id).resolve()
    cache_root = _cache_root().resolve()
    if template_root.parent != cache_root or not template_root.is_dir():
        raise FileNotFoundError("模板缓存不存在")
    candidates = sorted(
        item
        for item in template_root.iterdir()
        if item.is_file() and item.suffix.lower() in {".md", ".markdown", ".docx"}
    )
    if not candidates:
        raise FileNotFoundError("模板缓存文件不存在")
    return preview_cached_template(candidates[0], cache_root=cache_root)


@_guard
def preview(args: dict[str, Any]) -> dict[str, Any]:
    source_url = str(args.get("source_url") or args.get("url") or "").strip()
    template_id = str(args.get("template_id") or "").strip()
    legacy_path = str(args.get("path") or "").strip()
    token = _token(args)
    if source_url:
        result = preview_github_template(source_url, token=token)
        license_result = get_github_repository_license(result["repository"], token=token)
        if not license_result["license_allowed"]:
            code = (
                "template_license_unknown"
                if license_result["status"] == "unknown"
                else "template_license_not_allowed"
            )
            return _envelope(error_code=code, data={"license": license_result})
        return _envelope(data={"status": "ready", **result, "license": license_result})
    if template_id:
        return _envelope(
            data={"status": "ready", "template_id": template_id, **_cached_preview(template_id)}
        )
    if legacy_path:
        result = preview_cached_template(Path(legacy_path), cache_root=_cache_root())
        return _envelope(data={"status": "ready", **result})
    return _envelope(error_code="missing_template_reference")


@_guard
def download(args: dict[str, Any]) -> dict[str, Any]:
    source_url = str(args.get("source_url") or args.get("url") or "").strip()
    if not source_url:
        return _envelope(error_code="missing_template_url")
    result = cache_github_template(
        source_url=source_url,
        cache_root=_cache_root(),
        token=_token(args),
        repository=str(args.get("repository") or "").strip() or None,
        file_path=str(args.get("file_path") or "").strip() or None,
        remote_version=str(args.get("remote_version") or "").strip() or None,
        filename=str(args.get("filename") or "").strip() or None,
        allow_overwrite=bool(args.get("confirm_update", False)),
    )
    return _envelope(
        data={
            "status": "cached",
            **result,
            "auto_overwrite": False,
        }
    )


@_guard
def check_update(args: dict[str, Any]) -> dict[str, Any]:
    template_id = str(args.get("template_id") or "").strip()
    if not template_id:
        return _envelope(error_code="missing_template_id")
    result = check_github_template_update(
        cache_root=_cache_root(),
        template_id=template_id,
        token=_token(args),
    )
    if result["status"] == "license_blocked":
        return _envelope(error_code="template_license_not_allowed", data=result)
    if result["status"] == "remote_missing":
        return _envelope(error_code="template_remote_missing", data=result, requires_user=True)
    return _envelope(data=result, requires_user=result["status"] == "update_available")


def main() -> None:
    credential_property = {
        "github_token": {
            "type": "string",
            "description": "Optional write-only token for this call; never persisted",
        }
    }
    server = StdioMCPServer(
        "resume-agent-github",
        "0.2.0",
        {
            "github_search_resume_templates": (
                {
                    "description": "Search verified Chinese DOCX/Markdown resume template files",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "language": {
                                "type": "string",
                                "enum": ["zh", "zh-CN", "Chinese", "中文"],
                            },
                            "file_type": {"type": "string", "enum": ["docx", "md", "markdown"]},
                            "page": {"type": "integer", "minimum": 1},
                            "per_page": {"type": "integer", "minimum": 1, "maximum": 5},
                            **credential_property,
                        },
                    },
                },
                search,
            ),
            "github_get_template_license": (
                {
                    "description": "Fetch and evaluate a repository license",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"repository": {"type": "string"}, **credential_property},
                        "required": ["repository"],
                    },
                },
                license_info,
            ),
            "github_preview_template": (
                {
                    "description": (
                        "Return bounded preview metadata for a remote or cached template"
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "source_url": {"type": "string"},
                            "template_id": {"type": "string"},
                            **credential_property,
                        },
                    },
                },
                preview,
            ),
            "github_download_template": (
                {
                    "description": "Validate and cache one licensed Chinese resume template",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "source_url": {"type": "string"},
                            "repository": {"type": "string"},
                            "file_path": {"type": "string"},
                            "filename": {"type": "string"},
                            "remote_version": {"type": "string"},
                            "confirm_update": {"type": "boolean"},
                            **credential_property,
                        },
                        "required": ["source_url"],
                    },
                },
                download,
            ),
            "github_check_template_update": (
                {
                    "description": "Compare a cached template with GitHub without overwriting it",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"template_id": {"type": "string"}, **credential_property},
                        "required": ["template_id"],
                    },
                },
                check_update,
            ),
        },
    )
    server.run()


if __name__ == "__main__":
    main()
