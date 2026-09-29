"""GitHub-backed Chinese resume template discovery and cache helpers.

The module keeps credentials at the call boundary: callers pass a token for one
request and no function writes it to disk or returns it. Remote files are
treated as untrusted input and accepted only after host, size, format, language,
resume-content, and repository-license checks.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# These licenses explicitly permit modification and local use without requiring
# an attribution notice in the rendered resume itself. CC-BY is intentionally
# absent because its attribution condition is unsuitable for generated resumes.
ALLOWED_LICENSES = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC0-1.0",
        "ISC",
        "MIT",
        "Unlicense",
        "Zlib",
    }
)
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
MAX_API_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_DOCX_XML_BYTES = 2 * 1024 * 1024
MAX_DOCX_UNCOMPRESSED_BYTES = 20 * 1024 * 1024
ALLOWED_GITHUB_HOSTS = frozenset({"github.com", "raw.githubusercontent.com"})
_ALLOWED_API_HOSTS = frozenset({"api.github.com"})
_ALLOWED_SUFFIXES = frozenset({".docx", ".md", ".markdown"})
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$")
_TEMPLATE_ID_RE = re.compile(r"^github-[0-9a-f]{24}$")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_RESUME_MARKERS = (
    "个人信息",
    "个人简介",
    "教育经历",
    "工作经历",
    "项目经历",
    "专业技能",
    "技能清单",
    "求职目标",
    "求职意向",
    "联系方式",
    "简历",
)
_PATH_MARKERS = ("中文", "简历", "resume", "cv", "template", "模板")
_IGNORED_PATH_PARTS = frozenset(
    {".git", ".github", "node_modules", "vendor", "dist", "build"}
)


def template_id_for(repository: str | None, source_url: str | None = None) -> str:
    """Return a stable opaque ID without putting repository text in paths."""

    value = f"{repository or ''}|{source_url or ''}".encode("utf-8")
    return "github-" + hashlib.sha256(value).hexdigest()[:24]


def _normalise_token(token: str | None) -> str | None:
    value = str(token or "").strip()
    if not value:
        return None
    if "\r" in value or "\n" in value:
        raise ValueError("GitHub Token 格式不合法")
    return value


def _normalise_repository(repository: str) -> str:
    value = str(repository or "").strip().removesuffix(".git")
    if value.startswith("https://github.com/"):
        value = value[len("https://github.com/") :].strip("/")
    if not _REPOSITORY_RE.fullmatch(value):
        raise ValueError("GitHub 仓库必须使用 owner/repository 格式")
    return value


def _validate_https_url(url: str, allowed_hosts: frozenset[str]) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    host = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("GitHub 请求端口不合法") from exc
    if (
        parsed.scheme.lower() != "https"
        or host not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        raise ValueError("GitHub 请求必须使用允许的 HTTPS 地址")
    return parsed


def _headers(*, token: str | None, accept: str) -> dict[str, str]:
    result = {
        "Accept": accept,
        "User-Agent": "resume-agent/0.1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    normalised = _normalise_token(token)
    if normalised:
        result["Authorization"] = f"Bearer {normalised}"
    return result


def _request_bytes(
    url: str,
    *,
    token: str | None = None,
    timeout: float = 15.0,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    allowed_hosts: frozenset[str] = ALLOWED_GITHUB_HOSTS,
    accept: str = "application/octet-stream",
) -> tuple[bytes, str, str | None]:
    _validate_https_url(url, allowed_hosts)
    request = urllib.request.Request(url, headers=_headers(token=token, accept=accept))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        final_url = str(response.geturl() or url)
        _validate_https_url(final_url, allowed_hosts)
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                too_large = int(content_length) > max_bytes
            except (TypeError, ValueError):
                too_large = False
            if too_large:
                raise ValueError("GitHub 响应超过大小限制")
        data = response.read(max_bytes + 1)
        content_type = response.headers.get("Content-Type")
    if len(data) > max_bytes:
        raise ValueError("GitHub 响应超过大小限制")
    return data, final_url, content_type


def _request_json(url: str, *, token: str | None = None, timeout: float = 10.0) -> Any:
    data, _, _ = _request_bytes(
        url,
        token=token,
        timeout=timeout,
        max_bytes=MAX_API_RESPONSE_BYTES,
        allowed_hosts=_ALLOWED_API_HOSTS,
        accept="application/vnd.github+json",
    )
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("GitHub 返回了无效的 JSON") from exc


def _license_policy(spdx_id: str | None) -> dict[str, Any]:
    value = str(spdx_id or "").strip()
    allowed = value in ALLOWED_LICENSES
    return {
        "license": value or None,
        "license_allowed": allowed,
        "allows_local_use": allowed,
        "allows_modification": allowed,
        "requires_output_attribution": False if allowed else None,
        "policy_version": "github-template-license-v1",
    }


def get_github_repository_license(repository: str, *, token: str | None = None) -> dict[str, Any]:
    """Fetch and evaluate a repository license against the local allowlist."""

    repository = _normalise_repository(repository)
    url = f"https://api.github.com/repos/{repository}/license"
    try:
        payload = _request_json(url, token=token)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {
                "repository": repository,
                **_license_policy(None),
                "license_name": None,
                "status": "unknown",
                "license_url": None,
            }
        raise
    license_data = payload.get("license") if isinstance(payload, dict) else None
    license_data = license_data if isinstance(license_data, dict) else {}
    spdx_id = str(license_data.get("spdx_id") or "").strip()
    if spdx_id.upper() in {"NOASSERTION", "OTHER"}:
        spdx_id = ""
    policy = _license_policy(spdx_id)
    return {
        "repository": repository,
        **policy,
        "license_name": license_data.get("name"),
        "status": "allowed" if policy["license_allowed"] else ("blocked" if spdx_id else "unknown"),
        "license_url": payload.get("html_url") if isinstance(payload, dict) else None,
    }


def _source_descriptor(url: str) -> tuple[str, str, str]:
    """Return ``(repository, ref, path)`` for a supported GitHub file URL."""

    parsed = _validate_https_url(url, ALLOWED_GITHUB_HOSTS)
    if parsed.query or parsed.fragment:
        raise ValueError("GitHub 模板文件地址不能包含查询参数或片段")
    parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    if parsed.hostname == "raw.githubusercontent.com":
        if len(parts) < 4:
            raise ValueError("GitHub 模板地址缺少仓库、版本或文件路径")
        owner, repo, ref, *path_parts = parts
    else:
        if len(parts) < 5 or parts[2] not in {"blob", "raw"}:
            raise ValueError("GitHub 模板地址必须指向具体文件")
        owner, repo, _, ref, *path_parts = parts
    repository = _normalise_repository(f"{owner}/{repo}")
    path = "/".join(path_parts)
    if not path or any(part in {".", ".."} for part in path_parts):
        raise ValueError("GitHub 模板文件路径不合法")
    if Path(path).suffix.lower() not in _ALLOWED_SUFFIXES:
        raise ValueError("只允许 Markdown 或 DOCX 模板")
    return repository, ref, path


def _raw_url(repository: str, ref: str, path: str) -> str:
    owner, repo = _normalise_repository(repository).split("/", 1)
    return "https://raw.githubusercontent.com/{}/{}/{}/{}".format(
        urllib.parse.quote(owner, safe=""),
        urllib.parse.quote(repo, safe=""),
        urllib.parse.quote(ref, safe=""),
        urllib.parse.quote(path, safe="/"),
    )


def _downloadable_url(url: str, repository: str, ref: str, path: str) -> str:
    parsed = _validate_https_url(url, ALLOWED_GITHUB_HOSTS)
    if parsed.hostname == "github.com":
        return _raw_url(repository, ref, path)
    return url


def _git_blob_sha(data: bytes) -> str:
    prefix = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(prefix + data).hexdigest()  # nosec B324 - Git object identity


def _extract_template_text(data: bytes, suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in {".md", ".markdown"}:
        if b"\x00" in data:
            raise ValueError("Markdown 模板包含无效二进制内容")
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Markdown 模板必须使用 UTF-8 编码") from exc
    if suffix != ".docx" or not data.startswith(b"PK"):
        raise ValueError("DOCX 模板文件结构无效")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise ValueError("DOCX 模板缺少必要文档结构")
            total_size = sum(max(0, entry.file_size) for entry in archive.infolist())
            if total_size > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise ValueError("DOCX 模板解压后超过大小限制")
            info = archive.getinfo("word/document.xml")
            if info.file_size > MAX_DOCX_XML_BYTES:
                raise ValueError("DOCX 正文超过预览大小限制")
            xml = archive.read(info)
    except zipfile.BadZipFile as exc:
        raise ValueError("DOCX 模板文件结构无效") from exc
    text = re.sub(r"<w:tab\b[^>]*/>", "\t", xml.decode("utf-8", errors="replace"))
    text = re.sub(r"</w:(?:p|tr)>", "\n", text)
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def _preview_metadata(data: bytes, filename: str) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise ValueError("只允许 Markdown 或 DOCX 模板")
    text = _extract_template_text(data, suffix)
    compact_text = re.sub(r"[ \t]+", " ", text).strip()
    cjk_count = len(_CJK_RE.findall(compact_text))
    markers = [marker for marker in _RESUME_MARKERS if marker in compact_text]
    if cjk_count < 4 or len(markers) < 2:
        raise ValueError("文件未通过中文简历模板内容校验")
    if suffix in {".md", ".markdown"}:
        headings = [
            match.group(1).strip()
            for match in re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", text)
        ][:20]
        render_mode = "markdown"
    else:
        headings = markers[:20]
        render_mode = "docx_text_extract"
    return {
        "filename": Path(filename).name,
        "file_type": "docx" if suffix == ".docx" else "markdown",
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "text_characters": len(compact_text),
        "chinese_characters": cjk_count,
        "detected_sections": markers,
        "headings": headings,
        "excerpt": compact_text[:1200],
        "render_mode": render_mode,
        "requires_office_render": suffix == ".docx",
        "language": "zh",
    }


def _path_score(path: str) -> tuple[int, int, str]:
    lower = path.lower()
    marker_score = sum(1 for marker in _PATH_MARKERS if marker in lower)
    readme_penalty = 2 if Path(path).name.lower().startswith("readme") else 0
    depth_penalty = path.count("/")
    return (marker_score * 5 - readme_penalty - depth_penalty, -len(path), path)


def _tree_candidates(payload: Any, file_type: str | None) -> list[dict[str, Any]]:
    wanted_suffixes = _ALLOWED_SUFFIXES
    if file_type == "docx":
        wanted_suffixes = frozenset({".docx"})
    elif file_type in {"md", "markdown"}:
        wanted_suffixes = frozenset({".md", ".markdown"})
    candidates: list[dict[str, Any]] = []
    tree = payload.get("tree", []) if isinstance(payload, dict) else []
    for item in tree:
        if not isinstance(item, dict) or item.get("type") != "blob":
            continue
        path = str(item.get("path") or "")
        parts = {part.lower() for part in Path(path).parts}
        if parts & _IGNORED_PATH_PARTS or Path(path).suffix.lower() not in wanted_suffixes:
            continue
        size = item.get("size")
        if isinstance(size, int) and size > MAX_DOWNLOAD_BYTES:
            continue
        candidates.append({"path": path, "sha": item.get("sha"), "size": size})
    return sorted(candidates, key=lambda item: _path_score(str(item["path"])), reverse=True)


def search_github_templates(
    *,
    query: str = "中文 简历 模板",
    language: str = "zh",
    file_type: str | None = None,
    page: int = 1,
    per_page: int = 5,
    token: str | None = None,
) -> dict[str, Any]:
    """Discover verified Chinese resume template files, ordered by repository stars."""

    language_value = str(language or "zh").strip().lower()
    if language_value not in {"zh", "zh-cn", "chinese", "中文"}:
        raise ValueError("第一版只支持搜索中文模板")
    type_value = str(file_type or "").strip().lower() or None
    if type_value not in {None, "docx", "md", "markdown"}:
        raise ValueError("模板类型只支持 docx 或 markdown")
    page = max(1, int(page))
    per_page = min(5, max(1, int(per_page)))
    query_value = re.sub(r"\s+", " ", str(query or "").strip())[:160]
    if not query_value:
        query_value = "中文 简历 模板"
    search_query = f"{query_value} 中文 简历 in:name,description,readme"
    repository_page_size = min(25, max(10, per_page * 4))
    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
        {
            "q": search_query,
            "sort": "stars",
            "order": "desc",
            "page": page,
            "per_page": repository_page_size,
        }
    )
    payload = _request_json(url, token=token)
    repositories = payload.get("items", []) if isinstance(payload, dict) else []
    total_count = int(payload.get("total_count") or 0) if isinstance(payload, dict) else 0
    items: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    scanned = 0
    for repository_item in repositories:
        if len(items) >= per_page or scanned >= 10:
            break
        if not isinstance(repository_item, dict):
            continue
        repository = str(repository_item.get("full_name") or "")
        try:
            repository = _normalise_repository(repository)
        except ValueError:
            continue
        search_license = repository_item.get("license") or {}
        search_spdx = search_license.get("spdx_id") if isinstance(search_license, dict) else None
        if not _license_policy(str(search_spdx or ""))["license_allowed"]:
            continue
        scanned += 1
        try:
            license_result = get_github_repository_license(repository, token=token)
            if not license_result["license_allowed"]:
                continue
            default_branch = str(repository_item.get("default_branch") or "main")
            tree_url = "https://api.github.com/repos/{}/git/trees/{}?recursive=1".format(
                repository,
                urllib.parse.quote(default_branch, safe=""),
            )
            tree_payload = _request_json(tree_url, token=token)
            candidates = _tree_candidates(tree_payload, type_value)
            for candidate in candidates[:8]:
                source_url = _raw_url(repository, default_branch, str(candidate["path"]))
                try:
                    data, final_url, _ = _request_bytes(source_url, token=token)
                    preview = _preview_metadata(data, str(candidate["path"]))
                except (ValueError, OSError, urllib.error.URLError):
                    continue
                remote_sha = str(candidate.get("sha") or "")
                if remote_sha and _git_blob_sha(data) != remote_sha:
                    continue
                html_url = "https://github.com/{}/blob/{}/{}".format(
                    repository,
                    urllib.parse.quote(default_branch, safe=""),
                    urllib.parse.quote(str(candidate["path"]), safe="/"),
                )
                items.append(
                    {
                        "template_id": template_id_for(repository, final_url),
                        "name": Path(str(candidate["path"])).stem,
                        "repository": repository,
                        "repository_url": str(
                            repository_item.get("html_url")
                            or f"https://github.com/{repository}"
                        ),
                        "url": html_url,
                        "source_url": final_url,
                        "file_path": str(candidate["path"]),
                        "filename": Path(str(candidate["path"])).name,
                        "file_type": preview["file_type"],
                        "language": "zh",
                        "stars": int(repository_item.get("stargazers_count") or 0),
                        "updated_at": repository_item.get("updated_at"),
                        "default_branch": default_branch,
                        "remote_version": f"git:{remote_sha}" if remote_sha else None,
                        **license_result,
                        "preview_available": True,
                        "preview_metadata": preview,
                    }
                )
                break
        except (ValueError, OSError, urllib.error.URLError) as exc:
            warnings.append({"repository": repository, "reason": type(exc).__name__})
            continue
    items.sort(key=lambda item: (-int(item.get("stars", 0)), str(item.get("repository") or "")))
    has_more = page * repository_page_size < total_count
    return {
        "query": query_value,
        "language": "zh",
        "file_type": type_value,
        "page": page,
        "items": items[:per_page],
        "has_more": has_more,
        "next_page": page + 1 if has_more else None,
        "repositories_scanned": scanned,
        "warnings": warnings,
    }


def preview_github_template(url: str, *, token: str | None = None) -> dict[str, Any]:
    """Fetch an allowed GitHub file and return bounded preview metadata."""

    repository, ref, path = _source_descriptor(url)
    data, final_url, _ = _request_bytes(
        _downloadable_url(url, repository, ref, path), token=token
    )
    preview = _preview_metadata(data, path)
    return {
        "repository": repository,
        "ref": ref,
        "file_path": path,
        "source_url": final_url,
        "remote_version": f"git:{_git_blob_sha(data)}",
        "preview_available": True,
        **preview,
    }


def preview_cached_template(path: Path, *, cache_root: Path) -> dict[str, Any]:
    """Preview a cache file while preventing reads outside the template cache."""

    root = Path(cache_root).resolve()
    candidate = Path(path).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError("模板路径不在缓存目录中")
    if not candidate.is_file() or candidate.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise ValueError("模板缓存文件不存在或类型不受支持")
    data = candidate.read_bytes()
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError("模板文件超过大小限制")
    return {
        "path": str(candidate),
        "preview_available": True,
        **_preview_metadata(data, candidate.name),
    }


def download_github_file(
    url: str,
    *,
    root: Path,
    token: str | None = None,
    filename: str | None = None,
    expected_sha: str | None = None,
    repository: str | None = None,
    verify_license: bool = True,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Download one verified Chinese template into a caller-owned cache root."""

    inferred_repository, ref, source_path = _source_descriptor(url)
    repository = _normalise_repository(repository or inferred_repository)
    if repository != inferred_repository:
        raise ValueError("模板地址与仓库不匹配")
    license_result: dict[str, Any] | None = None
    if verify_license:
        license_result = get_github_repository_license(repository, token=token)
        if not license_result["license_allowed"]:
            raise ValueError("模板仓库许可证未知或不在允许列表中")
    source_suffix = Path(source_path).suffix.lower()
    safe_name = Path(filename or Path(source_path).name).name
    if not safe_name or Path(safe_name).suffix.lower() not in _ALLOWED_SUFFIXES:
        raise ValueError("只允许 Markdown 或 DOCX 模板")
    if Path(safe_name).suffix.lower() != source_suffix:
        raise ValueError("模板文件名与来源文件类型不匹配")
    data, final_url, _ = _request_bytes(
        _downloadable_url(url, inferred_repository, ref, source_path), token=token
    )
    preview = _preview_metadata(data, safe_name)
    git_sha = _git_blob_sha(data)
    normalised_expected = str(expected_sha or "").removeprefix("git:")
    if normalised_expected and normalised_expected != git_sha:
        raise ValueError("模板远程版本校验失败")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = (root / safe_name).resolve()
    if root != target.parent:
        raise ValueError("模板文件名不合法")
    content_sha = preview["sha256"]
    if target.exists():
        existing_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        if existing_sha == content_sha:
            return {
                "path": str(target),
                "size_bytes": len(data),
                "version": f"sha256:{content_sha}",
                "local_version": f"sha256:{content_sha}",
                "remote_version": f"git:{git_sha}",
                "source_url": final_url,
                "already_cached": True,
                "preview_metadata": preview,
                "license": license_result,
            }
        if not allow_overwrite:
            raise ValueError("缓存中已有不同版本，请先检查更新并明确确认覆盖")
    fd, tmp_name = tempfile.mkstemp(prefix=".template-", suffix=".tmp", dir=root)
    os.close(fd)
    temporary = Path(tmp_name)
    try:
        temporary.write_bytes(data)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(target),
        "size_bytes": len(data),
        "version": f"sha256:{content_sha}",
        "local_version": f"sha256:{content_sha}",
        "remote_version": f"git:{git_sha}",
        "source_url": final_url,
        "already_cached": False,
        "preview_metadata": preview,
        "license": license_result,
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(tmp_name)
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cache_github_template(
    *,
    source_url: str,
    cache_root: Path,
    token: str | None = None,
    repository: str | None = None,
    file_path: str | None = None,
    remote_version: str | None = None,
    filename: str | None = None,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """License-check, download, and register a template without persisting credentials."""

    inferred_repository, ref, inferred_path = _source_descriptor(source_url)
    repository = _normalise_repository(repository or inferred_repository)
    if repository != inferred_repository:
        raise ValueError("模板地址与仓库不匹配")
    if file_path and str(file_path).strip("/") != inferred_path:
        raise ValueError("模板地址与文件路径不匹配")
    license_result = get_github_repository_license(repository, token=token)
    if not license_result["license_allowed"]:
        raise ValueError("模板仓库许可证未知或不在允许列表中")
    canonical_source_url = _downloadable_url(source_url, repository, ref, inferred_path)
    template_id = template_id_for(repository, canonical_source_url)
    cache_root_resolved = Path(cache_root).resolve()
    target_root = (cache_root_resolved / template_id).resolve()
    if cache_root_resolved != target_root.parent:
        raise ValueError("模板缓存路径不合法")
    result = download_github_file(
        source_url,
        root=target_root,
        token=token,
        filename=filename,
        expected_sha=remote_version,
        repository=repository,
        verify_license=False,
        allow_overwrite=allow_overwrite,
    )
    metadata = {
        "schema_version": 1,
        "template_id": template_id,
        "repository": repository,
        "repository_url": f"https://github.com/{repository}",
        "source_url": result["source_url"],
        "file_path": inferred_path,
        "ref": ref,
        "filename": Path(result["path"]).name,
        "file_type": result["preview_metadata"]["file_type"],
        "language": "zh",
        "license": license_result["license"],
        "license_name": license_result["license_name"],
        "license_allowed": True,
        "remote_version": result["remote_version"],
        "local_version": result["local_version"],
        "cached_path": result["path"],
        "cached_at": datetime.now(UTC).isoformat(),
        "preview_metadata": result["preview_metadata"],
    }
    target_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(target_root / "metadata.json", metadata)
    return {**result, "template_id": template_id, "metadata": metadata}


def _load_cached_metadata(cache_root: Path, template_id: str) -> tuple[Path, dict[str, Any]]:
    if not _TEMPLATE_ID_RE.fullmatch(str(template_id or "")):
        raise ValueError("模板 ID 不合法")
    root = Path(cache_root).resolve()
    metadata_path = (root / template_id / "metadata.json").resolve()
    if root not in metadata_path.parents or not metadata_path.is_file():
        raise FileNotFoundError("模板缓存元数据不存在")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("模板缓存元数据损坏") from exc
    if not isinstance(metadata, dict) or metadata.get("template_id") != template_id:
        raise ValueError("模板缓存元数据与模板 ID 不匹配")
    for key in metadata:
        if "token" in key.lower() or "credential" in key.lower():
            raise ValueError("模板缓存元数据包含不允许的凭据字段")
    return metadata_path, metadata


def get_github_file_version(
    repository: str,
    file_path: str,
    *,
    ref: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    repository = _normalise_repository(repository)
    path = str(file_path or "").strip("/")
    if not path or any(part in {".", ".."} for part in path.split("/")):
        raise ValueError("GitHub 模板文件路径不合法")
    if Path(path).suffix.lower() not in _ALLOWED_SUFFIXES:
        raise ValueError("只允许 Markdown 或 DOCX 模板")
    url = f"https://api.github.com/repos/{repository}/contents/{urllib.parse.quote(path, safe='/')}"
    if ref:
        url += "?" + urllib.parse.urlencode({"ref": str(ref)})
    payload = _request_json(url, token=token)
    if not isinstance(payload, dict) or payload.get("type") not in {None, "file"}:
        raise ValueError("GitHub 模板文件元数据无效")
    sha = str(payload.get("sha") or "")
    if not sha:
        raise ValueError("GitHub 模板文件缺少版本标识")
    return {
        "repository": repository,
        "file_path": path,
        "ref": ref,
        "remote_version": f"git:{sha}",
        "remote_size_bytes": payload.get("size"),
        "source_url": payload.get("download_url"),
        "url": payload.get("html_url"),
    }


def check_github_template_update(
    *,
    cache_root: Path,
    template_id: str,
    token: str | None = None,
) -> dict[str, Any]:
    """Compare a cached record with GitHub; never download or overwrite it."""

    _, metadata = _load_cached_metadata(cache_root, template_id)
    license_result = get_github_repository_license(
        str(metadata.get("repository") or ""), token=token
    )
    if not license_result["license_allowed"]:
        return {
            "template_id": template_id,
            "status": "license_blocked",
            "auto_overwrite": False,
            "license": license_result,
        }
    try:
        remote = get_github_file_version(
            str(metadata.get("repository") or ""),
            str(metadata.get("file_path") or ""),
            ref=str(metadata.get("ref") or "") or None,
            token=token,
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {
                "template_id": template_id,
                "status": "remote_missing",
                "auto_overwrite": False,
                "local_remote_version": metadata.get("remote_version"),
            }
        raise
    current = str(metadata.get("remote_version") or "")
    changed = current != remote["remote_version"]
    return {
        "template_id": template_id,
        "status": "update_available" if changed else "up_to_date",
        "update_available": changed,
        "auto_overwrite": False,
        "local_remote_version": current or None,
        **remote,
        "checked_at": datetime.now(UTC).isoformat(),
    }


__all__ = [
    "ALLOWED_LICENSES",
    "cache_github_template",
    "check_github_template_update",
    "download_github_file",
    "get_github_file_version",
    "get_github_repository_license",
    "preview_cached_template",
    "preview_github_template",
    "search_github_templates",
    "template_id_for",
]
