"""Fixed-source Gitee DOCX resume-template discovery and cache helpers.

The repository, branch and directory are deliberately constants rather than
tool arguments.  A caller can therefore list and download only the approved
single-page template collection.  The access token is accepted at the call
boundary and is never written to template metadata or returned to clients.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from app.services.template_service import (
    MAX_API_RESPONSE_BYTES,
    MAX_DOWNLOAD_BYTES,
    _preview_metadata,
    preview_cached_template,
)


GITEE_REPOSITORY = "hicv-cn/hicv-word-resume-templates-jianyue-single"
GITEE_BRANCH = "main"
GITEE_TEMPLATE_PATH = "templates/02_简约简历/单页"
GITEE_REPOSITORY_URL = f"https://gitee.com/{GITEE_REPOSITORY}"
GITEE_TEMPLATE_URL = (
    f"{GITEE_REPOSITORY_URL}/tree/{GITEE_BRANCH}/"
    + urllib.parse.quote(GITEE_TEMPLATE_PATH, safe="/")
)
GITEE_CREDENTIAL_ID = "gitee-template-access-token"
_API_HOST = "gitee.com"
_TEMPLATE_ID_PREFIX = "gitee-"


class GiteeAPIError(RuntimeError):
    """Sanitised Gitee API error that never includes a credential-bearing URL."""

    def __init__(self, status_code: int, message: str = "Gitee API 请求失败") -> None:
        super().__init__(message)
        self.status_code = int(status_code)


def _normalise_token(token: str | None) -> str | None:
    value = str(token or "").strip()
    if not value:
        return None
    if "\r" in value or "\n" in value:
        raise ValueError("Gitee 授权码格式不合法")
    return value


def _api_url(path: str, *, token: str | None = None, **query: str) -> str:
    # Gitee API v5 documents access_token as a query parameter.  This URL is
    # kept request-local and is never returned, logged, or stored in metadata.
    values = {key: value for key, value in query.items() if str(value).strip()}
    normalised = _normalise_token(token)
    if normalised:
        values["access_token"] = normalised
    suffix = "?" + urllib.parse.urlencode(values) if values else ""
    return f"https://{_API_HOST}/api/v5{path}{suffix}"


def _request_json(url: str, *, timeout: float = 45.0) -> Any:
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != _API_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
    ):
        raise ValueError("Gitee 请求地址不在允许范围内")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "resume-agent/0.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_API_RESPONSE_BYTES:
                raise ValueError("Gitee API 响应超过大小限制")
            data = response.read(MAX_API_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        # Never re-raise HTTPError: its URL may contain the access_token query.
        raise GiteeAPIError(exc.code) from None
    if len(data) > MAX_API_RESPONSE_BYTES:
        raise ValueError("Gitee API 响应超过大小限制")
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Gitee 返回了无效 JSON") from exc


def _contents_url(path: str, *, token: str | None = None) -> str:
    encoded = urllib.parse.quote(path, safe="/")
    return _api_url(
        f"/repos/{GITEE_REPOSITORY}/contents/{encoded}",
        token=token,
        ref=GITEE_BRANCH,
    )


def _validate_file_path(file_path: str) -> str:
    value = str(file_path or "").strip("/")
    path = PurePosixPath(value)
    expected_parent = PurePosixPath(GITEE_TEMPLATE_PATH)
    if (
        not value
        or path.parent != expected_parent
        or path.suffix.casefold() != ".docx"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("只允许读取固定 Gitee 目录中的 DOCX 文件")
    return path.as_posix()


def template_id_for_gitee(file_path: str) -> str:
    path = _validate_file_path(file_path)
    digest = hashlib.sha256(
        f"{GITEE_REPOSITORY}|{GITEE_BRANCH}|{path}".encode("utf-8")
    ).hexdigest()[:24]
    return _TEMPLATE_ID_PREFIX + digest


def _source_url(file_path: str) -> str:
    return (
        f"{GITEE_REPOSITORY_URL}/blob/{GITEE_BRANCH}/"
        + urllib.parse.quote(_validate_file_path(file_path), safe="/")
    )


def _normalise_listing_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict) or str(item.get("type") or "") != "file":
        return None
    try:
        file_path = _validate_file_path(str(item.get("path") or ""))
    except ValueError:
        return None
    size = item.get("size")
    if isinstance(size, int) and size > MAX_DOWNLOAD_BYTES:
        return None
    filename = PurePosixPath(file_path).name
    sha = str(item.get("sha") or "").strip()
    return {
        "template_id": template_id_for_gitee(file_path),
        "name": PurePosixPath(filename).stem,
        "filename": filename,
        "repository": GITEE_REPOSITORY,
        "repository_url": GITEE_REPOSITORY_URL,
        "ref": GITEE_BRANCH,
        "file_path": file_path,
        "source_url": str(item.get("html_url") or _source_url(file_path)),
        "remote_version": f"git:{sha}" if sha else None,
        "size_bytes": size,
        "file_type": "docx",
        "language": "zh",
        "license": "upstream-repository",
        "license_allowed": True,
        "preview_available": True,
    }


def list_gitee_templates(
    *,
    token: str | None = None,
    page: int = 1,
    per_page: int = 5,
) -> dict[str, Any]:
    """List direct DOCX children of the one approved repository directory."""

    page = max(1, int(page))
    per_page = min(50, max(1, int(per_page)))
    payload = _request_json(_contents_url(GITEE_TEMPLATE_PATH, token=token))
    if not isinstance(payload, list):
        raise ValueError("Gitee 模板目录响应格式不正确")
    items = [normalised for item in payload if (normalised := _normalise_listing_item(item))]
    items.sort(key=lambda item: str(item["filename"]).casefold())
    start = (page - 1) * per_page
    selected = items[start : start + per_page]
    return {
        "repository": GITEE_REPOSITORY,
        "branch": GITEE_BRANCH,
        "template_path": GITEE_TEMPLATE_PATH,
        "directory_url": GITEE_TEMPLATE_URL,
        "page": page,
        "per_page": per_page,
        "total": len(items),
        "has_more": start + per_page < len(items),
        "next_page": page + 1 if start + per_page < len(items) else None,
        "items": selected,
    }


def probe_gitee_access(*, token: str | None = None) -> dict[str, Any]:
    result = list_gitee_templates(token=token, page=1, per_page=1)
    return {
        "status": "ready",
        "repository": GITEE_REPOSITORY,
        "branch": GITEE_BRANCH,
        "template_path": GITEE_TEMPLATE_PATH,
        "template_count": result["total"],
        "authenticated": bool(_normalise_token(token)),
    }


def _file_payload(file_path: str, *, token: str | None = None) -> tuple[bytes, dict[str, Any]]:
    path = _validate_file_path(file_path)
    payload = _request_json(_contents_url(path, token=token), timeout=60.0)
    if not isinstance(payload, dict) or str(payload.get("type") or "file") != "file":
        raise ValueError("Gitee 模板文件响应格式不正确")
    raw_content = str(payload.get("content") or "").replace("\n", "")
    try:
        data = base64.b64decode(raw_content, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Gitee 模板文件内容不是有效 Base64") from exc
    if not data or len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Gitee 模板为空或超过 5MB 限制")
    metadata = _preview_metadata(
        data,
        PurePosixPath(path).name,
        trusted_resume_directory=True,
    )
    return data, {**payload, "preview_metadata": metadata}


def preview_gitee_template(
    *,
    file_path: str,
    token: str | None = None,
) -> dict[str, Any]:
    path = _validate_file_path(file_path)
    _, payload = _file_payload(path, token=token)
    sha = str(payload.get("sha") or "").strip()
    return {
        "status": "ready",
        "template_id": template_id_for_gitee(path),
        "repository": GITEE_REPOSITORY,
        "ref": GITEE_BRANCH,
        "file_path": path,
        "source_url": _source_url(path),
        "remote_version": f"git:{sha}" if sha else None,
        "license": "upstream-repository",
        "license_allowed": True,
        **payload["preview_metadata"],
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cache_gitee_template(
    *,
    file_path: str,
    cache_root: Path,
    token: str | None = None,
    expected_remote_version: str | None = None,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    path = _validate_file_path(file_path)
    template_id = template_id_for_gitee(path)
    data, payload = _file_payload(path, token=token)
    sha = str(payload.get("sha") or "").strip()
    remote_version = f"git:{sha}" if sha else f"sha256:{hashlib.sha256(data).hexdigest()}"
    if expected_remote_version and str(expected_remote_version) != remote_version:
        raise ValueError("Gitee 模板远程版本校验失败，请刷新模板列表后重试")
    cache_root = Path(cache_root).resolve()
    target_root = (cache_root / template_id).resolve()
    if target_root.parent != cache_root:
        raise ValueError("模板缓存路径不合法")
    target_root.mkdir(parents=True, exist_ok=True)
    filename = PurePosixPath(path).name
    target = (target_root / filename).resolve()
    if target.parent != target_root:
        raise ValueError("模板文件名不合法")
    content_sha = hashlib.sha256(data).hexdigest()
    already_cached = False
    if target.is_file():
        existing_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        if existing_sha == content_sha:
            already_cached = True
        elif not allow_overwrite:
            raise ValueError("缓存中已有不同版本，请明确确认更新后再覆盖")
    if not already_cached:
        fd, temporary_name = tempfile.mkstemp(prefix=".template-", suffix=".tmp", dir=target_root)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            temporary.write_bytes(data)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    preview = dict(payload["preview_metadata"])
    metadata = {
        "schema_version": 1,
        "source": "gitee",
        "template_id": template_id,
        "repository": GITEE_REPOSITORY,
        "repository_url": GITEE_REPOSITORY_URL,
        "source_url": _source_url(path),
        "file_path": path,
        "ref": GITEE_BRANCH,
        "filename": filename,
        "file_type": "docx",
        "language": "zh",
        "license": "upstream-repository",
        "license_allowed": True,
        "remote_version": remote_version,
        "local_version": f"sha256:{content_sha}",
        "cached_path": str(target),
        "cached_at": datetime.now(UTC).isoformat(),
        "preview_metadata": preview,
    }
    _write_json_atomic(target_root / "metadata.json", metadata)
    return {
        "status": "cached",
        "template_id": template_id,
        "path": str(target),
        "size_bytes": len(data),
        "version": f"sha256:{content_sha}",
        "local_version": f"sha256:{content_sha}",
        "remote_version": remote_version,
        "source_url": _source_url(path),
        "already_cached": already_cached,
        "preview_metadata": preview,
        "metadata": metadata,
        "auto_overwrite": False,
    }


def _load_cached_metadata(cache_root: Path, template_id: str) -> dict[str, Any]:
    if not str(template_id or "").startswith(_TEMPLATE_ID_PREFIX):
        raise ValueError("Gitee 模板 ID 不合法")
    root = Path(cache_root).resolve()
    metadata_path = (root / template_id / "metadata.json").resolve()
    if root not in metadata_path.parents or not metadata_path.is_file():
        raise FileNotFoundError("Gitee 模板缓存元数据不存在")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Gitee 模板缓存元数据损坏") from exc
    if not isinstance(metadata, dict) or metadata.get("template_id") != template_id:
        raise ValueError("Gitee 模板缓存元数据与模板 ID 不匹配")
    if any("token" in str(key).casefold() or "credential" in str(key).casefold() for key in metadata):
        raise ValueError("模板缓存元数据包含不允许的凭据字段")
    _validate_file_path(str(metadata.get("file_path") or ""))
    return metadata


def preview_cached_gitee_template(*, cache_root: Path, template_id: str) -> dict[str, Any]:
    metadata = _load_cached_metadata(cache_root, template_id)
    path = Path(str(metadata.get("cached_path") or ""))
    preview = preview_cached_template(path, cache_root=Path(cache_root))
    return {"status": "ready", "template_id": template_id, **metadata, **preview}


def check_gitee_template_update(
    *,
    cache_root: Path,
    template_id: str,
    token: str | None = None,
) -> dict[str, Any]:
    metadata = _load_cached_metadata(cache_root, template_id)
    path = _validate_file_path(str(metadata.get("file_path") or ""))
    payload = _request_json(_contents_url(path, token=token))
    if not isinstance(payload, dict):
        raise ValueError("Gitee 模板版本响应格式不正确")
    sha = str(payload.get("sha") or "").strip()
    remote_version = f"git:{sha}" if sha else None
    current = str(metadata.get("remote_version") or "")
    changed = bool(remote_version and remote_version != current)
    return {
        "template_id": template_id,
        "status": "update_available" if changed else "up_to_date",
        "update_available": changed,
        "auto_overwrite": False,
        "local_remote_version": current or None,
        "remote_version": remote_version,
        "repository": GITEE_REPOSITORY,
        "file_path": path,
        "source_url": _source_url(path),
        "checked_at": datetime.now(UTC).isoformat(),
    }


__all__ = [
    "GITEE_BRANCH",
    "GITEE_CREDENTIAL_ID",
    "GITEE_REPOSITORY",
    "GITEE_TEMPLATE_PATH",
    "GITEE_TEMPLATE_URL",
    "GiteeAPIError",
    "cache_gitee_template",
    "check_gitee_template_update",
    "list_gitee_templates",
    "preview_cached_gitee_template",
    "preview_gitee_template",
    "probe_gitee_access",
    "template_id_for_gitee",
]
