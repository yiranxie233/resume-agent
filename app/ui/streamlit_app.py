"""Minimal Streamlit client.  It deliberately delegates all business work to FastAPI."""
from __future__ import annotations

import base64
import binascii
import io
import os
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


_STATUS_LABELS = {
    "auth_failed": "认证失败",
    "capability_mismatch": "模型能力不匹配",
    "credential_missing": "缺少访问凭据",
    "connection_refused": "服务商拒绝连接",
    "connection_timeout": "连接超时",
    "dns_resolution_failed": "域名解析失败",
    "edge_environment_failed": "浏览器环境读取失败",
    "edge_not_found": "未找到 Microsoft Edge",
    "edge_profile_permission_denied": "Edge 登录目录没有写入权限",
    "executable_missing": "未找到 Ollama 程序",
    "mcp_process_exited": "检测服务已退出",
    "model_not_installed": "模型未安装",
    "network_permission_denied": "后端没有外网访问权限",
    "needs_setup": "环境尚未配置完成",
    "in_memory": "本地存储正常",
    "degraded": "部分服务不可用",
    "playwright_not_installed": "未安装 Playwright",
    "probe_failed": "能力检测失败",
    "proxy_connection_failed": "代理连接失败",
    "ready": "正常",
    "service_unreachable": "服务无法连接",
    "tls_certificate_error": "HTTPS 证书校验失败",
    "endpoint_not_found": "接口路径不存在",
    "provider_http_error": "服务商返回请求错误",
    "preview_pending": "文件已生成，分页预览待检测",
    "preview_overflow": "预览超过一页",
    "needs_compression_consent": "等待确认压缩",
    "conversion_failed": "格式转换失败",
    "dependency_missing": "缺少本地转换组件",
    "blocked": "暂不可用",
    "unavailable": "服务不可用",
    "unprobed": "未检测",
}

_MASKED_API_KEY = "••••••••••••"


class ApiRequestError(RuntimeError):
    """A compact, user-facing representation of a FastAPI error envelope."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


def _request(method: str, path: str, **kwargs: Any) -> Any:
    import httpx

    base = os.getenv("RESUME_AGENT_API_URL", "http://127.0.0.1:8000")
    base_headers = dict(kwargs.pop("headers", {}))
    token_path = Path(os.getenv("RESUME_AGENT_DATA_ROOT", "./data")) / ".internal-token"
    candidates = [os.getenv("RESUME_AGENT_INTERNAL_TOKEN", "").strip()]
    try:
        candidates.append(token_path.read_text(encoding="utf-8").strip())
    except OSError:
        pass
    tokens = list(dict.fromkeys(token for token in candidates if token)) or [""]

    response = None
    timeout = kwargs.pop("timeout", 30)
    for index, token in enumerate(tokens):
        headers = dict(base_headers)
        if token:
            headers["X-Resume-Agent-Token"] = token
        try:
            # FastAPI is a loopback service.  A VPN may install HTTP(S)_PROXY
            # without a matching NO_PROXY entry; never send this private local
            # API traffic through that proxy.  The backend's outbound GitHub
            # and model calls still honour the user's system/VPN settings.
            response = httpx.request(
                method,
                f"{base}{path}",
                headers=headers,
                timeout=timeout,
                trust_env=False,
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise ApiRequestError(
                504,
                "connection_timeout",
                "本地后端处理超时；模型或网络可能仍在响应，请稍后重试当前步骤",
            ) from exc
        except httpx.RequestError as exc:
            raise ApiRequestError(
                503,
                "service_unreachable",
                "无法连接本地后端，请确认 FastAPI 服务仍在运行",
            ) from exc
        if response.status_code != 401 or index == len(tokens) - 1:
            break
    assert response is not None
    if response.is_error:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        error = payload.get("error") if isinstance(payload, dict) else {}
        if not isinstance(error, dict):
            error = {}
        code = str(
            payload.get("error_code")
            or error.get("code")
            or response.reason_phrase
            or "request_error"
        )
        message = str(payload.get("message") or error.get("message") or response.text)
        raise ApiRequestError(response.status_code, code, message)
    return response.json()


def _request_bytes(method: str, path: str, **kwargs: Any) -> bytes:
    """Read a private backend artifact without trusting a filesystem path."""

    import httpx

    if not re.fullmatch(
        r"/api/tasks/[^/]+/artifacts/(?:previews|exports)/[^/]+", str(path or "")
    ):
        raise ApiRequestError(422, "artifact_path_invalid", "导出文件地址不合法")
    base = os.getenv("RESUME_AGENT_API_URL", "http://127.0.0.1:8000")
    base_headers = dict(kwargs.pop("headers", {}))
    token_path = Path(os.getenv("RESUME_AGENT_DATA_ROOT", "./data")) / ".internal-token"
    candidates = [os.getenv("RESUME_AGENT_INTERNAL_TOKEN", "").strip()]
    try:
        candidates.append(token_path.read_text(encoding="utf-8").strip())
    except OSError:
        pass
    tokens = list(dict.fromkeys(token for token in candidates if token)) or [""]
    response = None
    timeout = kwargs.pop("timeout", 30)
    for index, token in enumerate(tokens):
        headers = dict(base_headers)
        if token:
            headers["X-Resume-Agent-Token"] = token
        try:
            response = httpx.request(
                method,
                f"{base}{path}",
                headers=headers,
                timeout=timeout,
                trust_env=False,
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise ApiRequestError(504, "connection_timeout", "读取导出文件超时，请重试") from exc
        except httpx.RequestError as exc:
            raise ApiRequestError(503, "service_unreachable", "无法读取本地导出文件") from exc
        if response.status_code != 401 or index == len(tokens) - 1:
            break
    assert response is not None
    if response.is_error:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        error = payload.get("error") if isinstance(payload, dict) else {}
        error = error if isinstance(error, dict) else {}
        raise ApiRequestError(
            response.status_code,
            str(payload.get("error_code") or error.get("code") or "artifact_read_failed"),
            str(payload.get("message") or error.get("message") or "读取导出文件失败"),
        )
    return bytes(response.content)


def _new_api_key(value: str) -> str | None:
    """Return only a newly typed secret; the visible mask is never persisted."""

    cleaned = str(value or "").strip()
    if not cleaned or cleaned == _MASKED_API_KEY:
        return None
    return cleaned


def _field_hash(value: Any) -> str:
    if isinstance(value, str):
        serialized = value
    else:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _friendly_status(value: Any) -> str:
    status = str(value or "unknown")
    return _STATUS_LABELS.get(status, status)


def _job_model_parse_failed(detail: dict[str, Any] | None) -> bool:
    """Return whether a saved job detail is showing a model fallback result."""

    if (detail or {}).get("model_fallback") is True:
        return True
    parser = (detail or {}).get("parser")
    if not isinstance(parser, dict) or not parser:
        return False
    return parser.get("parser_mode") not in {"model_enhanced", "human_corrected"}


def _edge_detection_result(payload: dict[str, Any]) -> tuple[bool, str, str | None]:
    boss = dict((payload.get("servers") or {}).get("boss") or {})
    browser = dict(boss.get("browser") or {})
    if boss.get("status") == "ready" and browser.get("status") == "ready":
        path = str(browser.get("edge_path") or "Microsoft Edge")
        return True, "Edge 浏览器检测成功", f"Playwright 已就绪；浏览器路径：{path}"
    code = browser.get("error_code") or browser.get("status") or boss.get("last_error_code") or boss.get("status")
    return (
        False,
        f"Edge 浏览器检测失败：{_friendly_status(code)}",
        str(browser.get("message") or "") or None,
    )


def _ollama_detection_result(payload: dict[str, Any]) -> tuple[bool, str, str | None]:
    if payload.get("status") == "ready":
        names = list(dict.fromkeys(str(item) for item in payload.get("installed_names") or []))
        visible_names = names[:3]
        suffix = " 等" if len(names) > len(visible_names) else ""
        detail = (
            "已下载模型：" + "、".join(visible_names) + suffix
            if visible_names
            else "当前没有已下载模型"
        )
        return True, f"Ollama 检测成功，共发现 {len(names)} 个模型", detail
    code = payload.get("error_code") or payload.get("status")
    return False, f"Ollama 检测失败：{_friendly_status(code)}", None


def _model_detection_result(payload: dict[str, Any]) -> tuple[bool, str]:
    model_name = str(payload.get("model_name") or "模型")
    if payload.get("status") == "ready":
        return True, f"{model_name} 检测成功"
    code = payload.get("error_code") or payload.get("status")
    message = f"{model_name} 检测失败：{_friendly_status(code)}"
    detail = str(payload.get("error_message") or "").strip()
    if detail:
        # Provider messages are useful for distinguishing a bad model ID from
        # an authentication/header mismatch, but should stay compact and never
        # render provider-supplied HTML in the page.
        detail = re.sub(r"<[^>]+>", " ", detail)
        detail = re.sub(r"[\x00-\x1f\x7f]+", " ", detail)
        detail = re.sub(r"\s+", " ", detail).strip()[:240]
        if detail:
            message += f"（服务商信息：{detail}）"
    return False, message


def _canonical_openai_base_url(value: str) -> str:
    """Mirror the backend's safe handling of pasted concrete endpoints."""

    base = str(value or "").strip().rstrip("/")
    lowered = base.casefold()
    for suffix in ("/chat/completions", "/embeddings", "/models", "/responses"):
        if lowered.endswith(suffix):
            base = base[: -len(suffix)].rstrip("/")
            lowered = base.casefold()
            break
    if base and not lowered.endswith("/v1"):
        base += "/v1"
    return base


def _display_model_profiles(
    models: list[dict[str, Any]],
    *,
    limit: int = 3,
    role: str | None = None,
    ready_only: bool = False,
) -> list[dict[str, Any]]:
    """Return a small, stable model list without mutating saved profiles."""

    candidates = [
        item
        for item in models
        if (role is None or item.get("role") == role)
        and (not ready_only or item.get("status") == "ready")
    ]
    candidates.sort(
        key=lambda item: (
            not bool(item.get("default_for_role")),
            item.get("status") != "ready",
            str(item.get("role") or ""),
            str(item.get("model_name") or "").casefold(),
            str(item.get("profile_id") or ""),
        )
    )
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in candidates:
        identity = (
            str(item.get("role") or ""),
            str(item.get("provider") or ""),
            str(item.get("base_url") or "").rstrip("/").casefold(),
            str(item.get("model_name") or "").casefold(),
        )
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(item)
        if len(selected) >= max(1, limit):
            break
    return selected


def _ready_model_profiles(
    models: list[dict[str, Any]], role: str = "chat"
) -> list[dict[str, Any]]:
    """Return every genuinely probed model for a runtime model picker."""

    return _display_model_profiles(
        models,
        limit=max(1, len(models)),
        role=role,
        ready_only=True,
    )


def _model_profile_label(item: dict[str, Any]) -> str:
    location = "云端" if item.get("provider") == "openai_compatible" else "本地"
    default = " · 默认" if item.get("default_for_role") else ""
    return f"{item.get('model_name') or item.get('profile_id')}（{location}{default}）"


def _ollama_installed_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the live Ollama /api/tags result for dynamic UI selection."""

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in payload.get("models") or []:
        if not isinstance(value, dict):
            continue
        name = str(value.get("name") or value.get("model") or "").strip()
        if not name or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        details = value.get("details") if isinstance(value.get("details"), dict) else {}
        capabilities = [
            str(item).strip().casefold()
            for item in value.get("capabilities") or []
            if str(item).strip()
        ]
        selected.append(
            {
                "name": name,
                "capabilities": capabilities,
                "parameter_size": str(details.get("parameter_size") or "未知"),
                "quantization_level": str(details.get("quantization_level") or "未知"),
                "context_length": details.get("context_length"),
                "embedding_length": details.get("embedding_length"),
                "size": value.get("size"),
            }
        )
    return selected


def _ollama_role_model_names(payload: dict[str, Any], role: str) -> list[str]:
    """Filter scanned models by capabilities without using fixed model names."""

    names: list[str] = []
    for model in _ollama_installed_models(payload):
        capabilities = set(model["capabilities"])
        lowered_name = str(model["name"]).casefold()
        looks_like_embedding = any(
            marker in lowered_name
            for marker in ("embedding", "embed", "bge-", "e5-", "gte-")
        )
        if role == "embedding":
            supported = "embedding" in capabilities or (
                not capabilities and looks_like_embedding
            )
        else:
            supported = bool(
                capabilities.intersection({"completion", "chat", "generate"})
            ) or (not capabilities and not looks_like_embedding)
        if supported:
            names.append(str(model["name"]))
    return names


def _job_search_feedback(payload: dict[str, Any]) -> tuple[str, str] | None:
    """Return a concise UI message for non-result search outcomes."""

    jobs = list(payload.get("jobs") or [])
    if payload.get("load_pending"):
        return "info", "本次下拉暂未加载出新岗位，可再次点击“下一页”继续尝试"
    status = str(payload.get("status") or "")
    code = str(payload.get("error_code") or "")
    if status == "waiting_user" or code:
        messages = {
            "boss_login_required": "请在已打开的岗位采集 Edge 中登录 BOSS，完成后再次点击搜索",
            "boss_verification_required": "请在已打开的 Edge 窗口中完成人工验证，完成后再次点击搜索",
            "boss_network_unavailable": "Edge 无法访问 BOSS，请检查系统网络或代理是否允许访问 zhipin.com",
            "boss_navigation_failed": "BOSS 页面打开失败，浏览器会话已重置，请重新搜索",
            "boss_page_unavailable": "暂时无法读取 BOSS 页面，请在 Edge 中确认页面可正常打开后重试",
            "edge_session_closed": "Edge 窗口已关闭，请重新点击搜索",
            "edge_login_attach_failed": "登录窗口正在准备中，请保持 Edge 打开并重新点击搜索",
            "boss_search_session_expired": "岗位搜索窗口已关闭或已离开当前页面，请重新搜索",
            "edge_launch_failed": "无法启动 Edge，请返回“环境与模型”重新检测浏览器",
            "mcp_unavailable": "BOSS 检测服务暂不可用，请返回“环境与模型”重新检测",
        }
        if code in messages:
            return "warning", messages[code]
        message = str(payload.get("message") or "职位搜索暂不可用，请稍后重试").splitlines()[0].strip()
        if len(message) > 160:
            message = f"{message[:157]}..."
        return "warning", message
    if status == "ready" and not jobs:
        return "info", "未找到符合条件的职位，请调整岗位名称或城市后重试"
    return None


def _task_action(task: dict[str, Any], action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "thread_id": task["thread_id"],
        "checkpoint_version": task["checkpoint_version"],
        "action": action,
        "payload": payload or {},
    }


def _selected_candidate(task: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve the explicit API selection without guessing another draft."""

    selected_id = task.get("selected_candidate_id")
    return next(
        (
            item
            for item in task.get("candidates", [])
            if item.get("status") != "duplicate" and item.get("candidate_id") == selected_id
        ),
        None,
    )


def _candidate_plain_text(candidate: dict[str, Any]) -> str:
    """Render one confirmed project in the user's compact resume format."""

    title = str(candidate.get("title") or "项目经历").strip()
    period = str(candidate.get("period") or "").strip()
    lines = [f"- {title}{('  ' + period) if period else ''}"]
    introduction = str(candidate.get("introduction") or "").strip()
    if introduction:
        lines.append(f"项目简介：{introduction}")
    stack = [str(value).strip() for value in candidate.get("tech_stack") or [] if str(value).strip()]
    if stack:
        lines.append("技术栈：" + "、".join(stack))
    solutions = [str(value).strip() for value in candidate.get("solutions") or [] if str(value).strip()]
    if solutions:
        lines.append("解决方案：")
        lines.extend(f"{index}. {value}" for index, value in enumerate(solutions, 1))
    results = [str(value).strip() for value in candidate.get("results") or [] if str(value).strip()]
    if results:
        lines.append("结果：" + "；".join(results))
    return "\n".join(lines).strip()


def _candidate_editor_list(value: str, *, tech_stack: bool = False) -> list[str]:
    """Convert one candidate-editor text box back to stable list fields."""

    separator = r"[\n、,，]+" if tech_stack else r"\r?\n+"
    values: list[str] = []
    for item in re.split(separator, str(value or "")):
        cleaned = re.sub(r"^\s*(?:[-•·]|\d+[.、)])\s*", "", item).strip()
        if cleaned:
            values.append(cleaned)
    return values


_CANDIDATE_CONFIRMATION_FIELDS = (
    "title",
    "period",
    "introduction",
    "tech_stack",
    "solutions",
    "results",
)


def _set_all_candidate_confirmations(
    session_state: Any,
    *,
    master_key: str,
    field_key_prefix: str,
) -> None:
    """Synchronize the six field widgets from one user-controlled master box."""

    checked = bool(session_state.get(master_key, False))
    for field in _CANDIDATE_CONFIRMATION_FIELDS:
        session_state[f"{field_key_prefix}{field}"] = checked


def _template_download_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in ("source_url", "repository", "file_path", "remote_version", "filename")
        if item.get(key)
    }


def _template_error_detail(
    error: BaseException | str,
    *,
    stage: str,
    item_name: str | None = None,
) -> dict[str, Any]:
    """Classify a template failure into an actionable, non-secret UI message."""

    code = error.code if isinstance(error, ApiRequestError) else ""
    if not code and isinstance(error, str) and re.fullmatch(r"[a-z][a-z0-9_]+", error):
        code = error
    raw = error.message if isinstance(error, ApiRequestError) else str(error)
    lowered = f"{code} {raw}".casefold()
    if code in {"github_auth_failed", "credential_missing"} or "401" in lowered:
        category = "配置问题"
        message = "GitHub Token 无效、已过期或未提供"
        suggestion = "返回环境设置重新填写本次会话的 GitHub Token；公开仓库也可稍后匿名重试。"
    elif code in {"github_rate_limited"} or "rate limit" in lowered or "429" in lowered:
        category = "GitHub 限流"
        message = "GitHub API 请求次数已受限"
        suggestion = "稍后重试，或在设置中临时填写 GitHub Token 后再试。"
    elif code in {"mcp_unavailable", "mcp_process_exited", "mcp_startup_failed"}:
        category = "MCP 服务问题"
        message = "GitHub 模板 MCP 没有正常启动或已退出"
        suggestion = "确认项目 MCP 依赖已安装，并点击重新尝试；系统会保留内置模板。"
    elif code in {"template_license_unknown", "template_license_not_allowed"}:
        category = "许可证问题"
        message = "模板许可证缺失或不在允许范围内"
        suggestion = "该模板不会下载，请改选具有 MIT、Apache-2.0 等明确许可证的模板。"
    elif code in {"template_download_failed", "template_preview_unavailable", "template_not_found"}:
        category = "模板文件问题"
        message = raw or "模板文件下载、校验或预览失败"
        suggestion = "可重试或改选其他模板；系统内置模板仍可继续使用。"
    elif any(
        marker in lowered
        for marker in (
            "dns",
            "name or service not known",
            "getaddrinfo",
            "connecterror",
            "connection",
            "timed out",
            "timeout",
            "github_unavailable",
        )
    ):
        category = "网络问题"
        message = "无法连接 GitHub 或请求超时"
        suggestion = "检查网络、DNS、系统代理和防火墙是否允许访问 github.com 与 raw.githubusercontent.com。"
    else:
        category = "未知问题"
        message = raw.splitlines()[0][:240] if raw else "模板处理失败"
        suggestion = "点击重新尝试；若持续失败，可先使用系统内置模板。"
    return {
        "stage": stage,
        "item": item_name,
        "code": code or None,
        "category": category,
        "message": message,
        "suggestion": suggestion,
    }


def _render_wizard_template_picker(st: Any) -> str | None:
    """Fetch, cache, preview and select a verified Chinese GitHub template."""

    if "wizard_template_catalog" not in st.session_state:
        catalog: dict[str, Any] = {"items": [], "errors": [], "stage": "github_search"}
        progress = st.progress(0, text="正在 GitHub 查找中文简历模板…")
        stage_text = st.empty()
        try:
            candidates_by_id: dict[str, dict[str, Any]] = {}
            searched: dict[str, Any] = {}
            # One GitHub result page may contain fewer than five licensed,
            # content-verified files. Continue to the next repository page
            # before falling back, so the picker normally offers five real
            # high-star/high-relevance templates rather than one text snippet.
            for search_page in (1, 2, 3):
                searched = _request(
                    "POST",
                    "/api/templates/search",
                    timeout=240,
                    json={
                        "query": "中文 简历 模板",
                        "source": "github",
                        "page": search_page,
                        "per_page": 5,
                        "credential_handle_id": st.session_state.get(
                            "github_credential_handle"
                        ),
                    },
                )
                catalog["source"] = searched.get("source")
                if searched.get("error_code"):
                    catalog["errors"].append(
                        _template_error_detail(
                            str(searched.get("error_code")),
                            stage="在 GitHub 查找",
                        )
                    )
                for item in searched.get("items") or []:
                    if not isinstance(item, dict):
                        continue
                    template_id = str(item.get("template_id") or "")
                    if template_id.startswith("github-"):
                        candidates_by_id.setdefault(template_id, item)
                if len(candidates_by_id) >= 5 or searched.get("source") == "builtin_fallback":
                    break
                stage_text.info(
                    f"当前找到 {len(candidates_by_id)}/5 个合格模板，继续查找下一批仓库…"
                )
            candidates = sorted(
                candidates_by_id.values(),
                key=lambda item: (-int(item.get("stars") or 0), str(item.get("repository") or "")),
            )[:5]
            if 0 < len(candidates) < 5:
                catalog["errors"].append(
                    {
                        "stage": "在 GitHub 查找",
                        "item": None,
                        "code": "template_catalog_incomplete",
                        "category": "模板数量不足",
                        "message": f"仅找到 {len(candidates)} 个通过中文简历结构和许可证校验的模板",
                        "suggestion": "可暂时从现有模板中选择，或填写 GitHub Token 后重新查找以提高 API 限额。",
                    }
                )
            total = len(candidates)
            progress.progress(15, text=f"GitHub 查找完成，准备下载 {total} 个模板")
            for index, item in enumerate(candidates, 1):
                template_id = str(item["template_id"])
                name = str(item.get("name") or template_id)
                stage_text.info(f"简历下载（{index}/{total}）：{name}")
                progress.progress(
                    15 + int((index - 1) / max(total, 1) * 75),
                    text=f"简历下载（{index}/{total}）",
                )
                try:
                    cached = _request(
                        "POST",
                        f"/api/templates/{template_id}/download",
                        timeout=240,
                        json={
                            **_template_download_payload(item),
                            "credential_handle_id": st.session_state.get(
                                "github_credential_handle"
                            ),
                        },
                    )
                    resolved_id = str(cached.get("template_id") or template_id)
                    preview = _request(
                        "POST",
                        f"/api/templates/{resolved_id}/preview",
                        timeout=120,
                        json={},
                    )
                    catalog["items"].append(
                        {**item, **cached, "template_id": resolved_id, "preview": preview}
                    )
                except Exception as exc:
                    catalog["errors"].append(
                        _template_error_detail(exc, stage="简历下载/预览", item_name=name)
                    )
            progress.progress(90, text="正在完成模板校验与预览…")
        except Exception as exc:
            catalog["errors"].append(_template_error_detail(exc, stage="在 GitHub 查找"))

        if not catalog["items"]:
            try:
                stage_text.info("GitHub 模板暂不可用，正在准备系统内置模板…")
                builtin = {
                    "template_id": "builtin-cn-single-page",
                    "name": "内置中文单页模板",
                }
                cached = _request(
                    "POST", "/api/templates/builtin-cn-single-page/download", json={}
                )
                preview = _request(
                    "POST", "/api/templates/builtin-cn-single-page/preview", json={}
                )
                catalog["items"].append({**builtin, **cached, "preview": preview})
            except Exception as exc:
                catalog["errors"].append(
                    _template_error_detail(exc, stage="准备内置模板", item_name="内置中文单页模板")
                )
        catalog["stage"] = "completed" if catalog["items"] else "failed"
        progress.progress(
            100,
            text="模板准备完成" if catalog["items"] else "模板准备失败",
        )
        stage_text.empty()
        st.session_state["wizard_template_catalog"] = catalog

    catalog = st.session_state.get("wizard_template_catalog") or {}
    items = list(catalog.get("items") or [])
    errors = [item for item in catalog.get("errors") or [] if isinstance(item, dict)]
    if errors:
        with st.expander("模板获取问题与解决建议", expanded=not items):
            for error in errors:
                target = f" · {error['item']}" if error.get("item") else ""
                st.warning(
                    f"{error.get('stage')}：{error.get('category')}{target}\n\n"
                    f"{error.get('message')}\n\n建议：{error.get('suggestion')}"
                )
        if st.button("重新尝试查找和下载模板", key="wizard_retry_template_catalog"):
            st.session_state.pop("wizard_template_catalog", None)
            st.session_state.pop("wizard_template_visual_previews", None)
            st.rerun()
    if not items:
        st.error("没有可用模板；请根据上方诊断修复配置或网络后重新尝试。")
        return None
    labels = {
        str(item["template_id"]): (
            f"{item.get('name') or item.get('filename') or item['template_id']} · "
            f"{item.get('file_type') or 'markdown'} · "
            f"{item.get('stars') if item.get('stars') is not None else '内置'}"
        )
        for item in items
    }
    selected_id = st.selectbox(
        "选择已下载的中文简历模板",
        list(labels),
        format_func=lambda value: labels[value],
        key="wizard_template_id",
    )
    selected = next(item for item in items if str(item["template_id"]) == selected_id)
    preview = selected.get("preview") if isinstance(selected.get("preview"), dict) else {}
    visual_cache = st.session_state.setdefault("wizard_template_visual_previews", {})
    if selected_id not in visual_cache:
        with st.spinner("正在使用本机 Office 生成所选模板的排版预览…"):
            try:
                visual_cache[selected_id] = _request(
                    "POST",
                    f"/api/templates/{selected_id}/preview",
                    timeout=240,
                    json={"render_visual": True},
                )
            except Exception as exc:
                visual_cache[selected_id] = {
                    "visual_preview": {
                        "status": "unavailable",
                        "message": _template_error_detail(
                            exc,
                            stage="生成排版预览",
                            item_name=str(selected.get("name") or selected_id),
                        )["message"],
                    }
                }
    rendered = visual_cache.get(selected_id) or preview
    visual = (
        rendered.get("visual_preview")
        if isinstance(rendered, dict)
        and isinstance(rendered.get("visual_preview"), dict)
        else {}
    )
    with st.expander("模板预览", expanded=True):
        st.caption(
            f"来源：{selected.get('repository') or selected.get('source_url') or '系统内置'} ｜ "
            f"许可证：{selected.get('license') or 'internal'}"
        )
        if visual.get("status") == "ready" and visual.get("mime_type") == "application/pdf":
            try:
                pdf_data = base64.b64decode(
                    str(visual.get("content_base64") or ""), validate=True
                )
            except (ValueError, binascii.Error):
                st.error("模板排版预览数据损坏，请重新尝试")
            else:
                _render_pdf_document(
                    st,
                    pdf_data,
                    str(visual.get("filename") or f"{selected_id}.pdf"),
                    heading=None,
                    caption_suffix="由本机 Office 渲染，仅用于选择模板",
                )
        elif visual.get("status") == "ready" and visual.get("mime_type") == "text/markdown":
            st.markdown(str(visual.get("content") or ""))
        elif visual.get("status") == "ready" and visual.get("mime_type") == "text/html":
            if visual.get("approximate_layout"):
                st.info(
                    str(
                        visual.get("message")
                        or "当前显示浏览器排版预览；安装 Word 或 LibreOffice 后可查看精确分页。"
                    )
                )
            from streamlit.components.v1 import html as render_html

            render_html(str(visual.get("content") or ""), height=760, scrolling=True)
        else:
            st.warning(
                str(visual.get("message") or "无法生成模板排版预览；请确认本机安装了 Microsoft Word 或 LibreOffice。")
            )
            if st.button("重新生成所选模板预览", key=f"retry-template-visual-{selected_id}"):
                visual_cache.pop(selected_id, None)
                st.rerun()

    st.markdown("#### 使用聊天模型解析模板结构")
    st.caption(
        "模板预览只用于查看排版。还需要由聊天模型识别栏目映射，并由你逐块校正和确认；"
        "确认前不会进入岗位与项目生成步骤。"
    )
    try:
        template_models = _ready_model_profiles(_request("GET", "/api/models"), "chat")
    except Exception:
        template_models = []
    template_chat_profile_id = (
        st.selectbox(
            "模板结构解析模型",
            [item["profile_id"] for item in template_models],
            index=next(
                (
                    index
                    for index, item in enumerate(template_models)
                    if item.get("default_for_role")
                ),
                0,
            ),
            format_func=lambda value: next(
                (
                    _model_profile_label(item)
                    for item in template_models
                    if item["profile_id"] == value
                ),
                value,
            ),
            key=f"wizard_template_chat_profile_{selected_id}",
            help="可选择检测成功的云端 OpenAI 兼容模型或本地 Ollama 聊天模型。",
        )
        if template_models
        else None
    )
    structures = st.session_state.setdefault("wizard_template_structures", {})
    if selected_id not in structures:
        try:
            structures[selected_id] = _request(
                "GET", f"/api/templates/{selected_id}/structure"
            )
        except ApiRequestError as exc:
            if exc.code != "template_structure_not_found":
                st.warning(f"读取模板解析记录失败：{exc.message}")
        except Exception:
            pass
    parse_label = (
        "重新使用所选模型解析"
        if isinstance(structures.get(selected_id), dict)
        else "使用所选模型解析模板"
    )
    if st.button(
        parse_label,
        key=f"wizard_parse_template_{selected_id}",
        disabled=not bool(template_chat_profile_id),
        use_container_width=True,
    ):
        try:
            parsed = _request(
                "POST",
                f"/api/templates/{selected_id}/model-parse",
                timeout=240,
                json={
                    "chat_profile_id": template_chat_profile_id,
                    "credential_handle_id": st.session_state.get(
                        f"credential-{template_chat_profile_id}"
                    ),
                },
            )
            structures[selected_id] = parsed
            if parsed.get("status") == "ready":
                st.rerun()
            parser = parsed.get("parser") if isinstance(parsed, dict) else {}
            warning = next(iter((parser or {}).get("warnings") or []), "模型解析模板失败")
            st.warning(str(warning))
        except ApiRequestError as exc:
            st.error(f"模板结构解析失败：{exc.message}")

    structure = structures.get(selected_id)
    if not isinstance(structure, dict) or not structure.get("sections"):
        if not template_models:
            st.warning("没有检测成功的聊天模型，请先在“环境与模型”中配置并检测云端或本地模型。")
        else:
            st.info("请选择模型并解析模板；解析完成后会显示可编辑的栏目区块。")
        return None
    if _render_template_structure_confirmation(st, selected_id, structure):
        return selected_id
    return None


_RESUME_MODULE_LABELS = {
    "basic": "基本信息",
    "summary": "个人简介",
    "education": "教育经历",
    "work": "工作/实习经历",
    "projects": "项目经历",
    "skills": "技能清单",
    "evaluation": "个人评价",
    "certificates": "证书",
    "objective": "求职目标",
    "other": "其他栏目",
}


def _render_template_structure_confirmation(
    st: Any,
    template_id: str,
    structure: dict[str, Any],
) -> bool:
    """Render editable template blocks and enforce one explicit confirmation."""

    sections = [item for item in structure.get("sections") or [] if isinstance(item, dict)]
    if not sections:
        st.error("模型没有返回可编辑的模板区块，请更换模型后重试。")
        return False
    confirmed = bool(structure.get("confirmed") and structure.get("status") == "confirmed")
    parser = structure.get("parser") if isinstance(structure.get("parser"), dict) else {}
    profile = parser.get("model_name") or parser.get("model_profile_id") or "所选聊天模型"
    if confirmed:
        st.success(f"模板结构已经确认（解析模型：{profile}），可以继续下一步。")
    else:
        st.info(f"已由 {profile} 识别 {len(sections)} 个区块。请逐块校正后明确确认。")
    for warning in parser.get("warnings") or []:
        st.warning(str(warning))

    edited: list[dict[str, Any]] = []
    version_key = str(structure.get("structure_id") or "unknown")[-16:]
    module_values = list(_RESUME_MODULE_LABELS)
    for index, section in enumerate(sections, 1):
        section_id = str(section.get("section_id") or f"section-{index}")
        current_module = str(section.get("module") or "other")
        with st.expander(
            f"{index}. {_RESUME_MODULE_LABELS.get(current_module, current_module)} · "
            f"{section.get('title') or '未命名区块'}",
            expanded=not confirmed and (index <= 2 or current_module == "projects"),
        ):
            module = st.selectbox(
                "区块类型",
                module_values,
                index=(
                    module_values.index(current_module)
                    if current_module in module_values
                    else module_values.index("other")
                ),
                format_func=lambda value: _RESUME_MODULE_LABELS[value],
                key=f"template_module_{template_id}_{version_key}_{section_id}",
                disabled=confirmed,
            )
            title = st.text_input(
                "模板栏目标题",
                value=str(section.get("title") or ""),
                key=f"template_title_{template_id}_{version_key}_{section_id}",
                disabled=confirmed,
            )
            content = st.text_area(
                "该区块中的模板文字或占位内容",
                value=str(section.get("content") or ""),
                height=140,
                key=f"template_content_{template_id}_{version_key}_{section_id}",
                disabled=confirmed,
                help="这里校正的是栏目映射与占位内容，不会直接改写原模板文件。",
            )
            edited.append(
                {
                    "section_id": section_id,
                    "old_value_hash": section.get("value_hash"),
                    "module": module,
                    "title": title or None,
                    "content": content,
                }
            )
    if confirmed:
        return True
    if st.button(
        "保存修改并确认模板结构",
        key=f"confirm_template_structure_{template_id}_{version_key}",
        type="primary",
        use_container_width=True,
    ):
        try:
            result = _request(
                "POST",
                f"/api/templates/{template_id}/confirm-structure",
                json={
                    "structure_id": structure.get("structure_id"),
                    "source_sha256": structure.get("source_sha256"),
                    "sections": edited,
                },
            )
            st.session_state.setdefault("wizard_template_structures", {})[
                template_id
            ] = result
            st.rerun()
        except ApiRequestError as exc:
            st.error(f"模板结构确认失败：{exc.message}")
    return False


def _render_pdf_document(
    st: Any,
    data: bytes,
    filename: str,
    *,
    heading: str | None,
    caption_suffix: str,
) -> None:
    """Render private PDF bytes in-session without exposing a public URL."""

    if not data:
        return
    if heading:
        st.markdown(f"#### {heading}")
    st.caption(f"{filename}（{caption_suffix}）")
    # Render the actual pages to images first. The browser PDF plug-in used by
    # ``st.pdf``/data-URL iframes is not consistently available in embedded
    # Streamlit sessions on Windows, which previously left an empty preview.
    try:
        import fitz  # type: ignore

        with fitz.open(stream=data, filetype="pdf") as document:
            for page_number, page in enumerate(document, 1):
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                st.image(
                    pixmap.tobytes("png"),
                    caption=f"{filename} · 第 {page_number} / {document.page_count} 页",
                    use_container_width=True,
                )
        return
    except (ImportError, OSError, RuntimeError, ValueError):
        pass
    if hasattr(st, "pdf"):
        try:
            st.pdf(data, height=720)
            return
        except Exception:
            pass
    from streamlit.components.v1 import html

    encoded = base64.b64encode(data).decode("ascii")
    html(
        '<iframe title="PDF 预览" '
        f'src="data:application/pdf;base64,{encoded}" '
        'style="width:100%;height:700px;border:1px solid #ddd;border-radius:6px"></iframe>',
        height=720,
        scrolling=True,
    )


def _render_export_result(
    st: Any,
    payload: dict[str, Any],
    *,
    key_prefix: str,
    content_label: str = "生成内容",
) -> None:
    """Render status, visual preview and the exact requested file download."""

    if not isinstance(payload, dict):
        return
    format_name = str(payload.get("format") or "").casefold()
    format_label = {"md": "Markdown", "markdown": "Markdown", "docx": "DOCX", "pdf": "PDF"}.get(
        format_name, format_name.upper() or "文件"
    )
    status = str(payload.get("status") or "unknown")
    page_count = payload.get("page_count")
    page_text = f" · {page_count} 页" if page_count is not None else ""
    st.write(f"{format_label} 状态：{_friendly_status(status)}{page_text}")

    content = str(payload.get("content") or "")
    if content:
        st.text_area(
            content_label,
            value=content,
            height=320,
            key=f"{key_prefix}-content",
        )

    warnings: list[str] = []
    for source in (
        payload,
        payload.get("docx") or {},
        payload.get("pdf") or {},
        payload.get("visual_preview") or {},
    ):
        if not isinstance(source, dict):
            continue
        warnings.extend(str(item) for item in source.get("warnings") or [] if str(item).strip())
    for warning in dict.fromkeys(warnings):
        st.warning(warning)

    artifact = payload.get("artifact")
    artifact = artifact if isinstance(artifact, dict) else None
    visual = payload.get("visual_preview")
    visual = visual if isinstance(visual, dict) else {}
    visual_artifact = visual.get("artifact")
    visual_artifact = visual_artifact if isinstance(visual_artifact, dict) else None
    byte_cache: dict[str, bytes] = {}

    def read_artifact(value: dict[str, Any] | None) -> bytes | None:
        if not value or value.get("status") != "ready":
            return None
        url = str(value.get("download_url") or "")
        if not url:
            return None
        if url not in byte_cache:
            byte_cache[url] = _request_bytes("GET", url)
        return byte_cache[url]

    if visual_artifact:
        try:
            visual_bytes = read_artifact(visual_artifact)
            if visual_bytes:
                _render_pdf_document(
                    st,
                    visual_bytes,
                    str(visual_artifact.get("file_name") or "resume-preview.pdf"),
                    heading="视觉预览",
                    caption_suffix=(
                        "由 DOCX 转换，仅用于预览" if format_name == "docx" else "最终 PDF 页面"
                    ),
                )
        except ApiRequestError as exc:
            st.warning(f"视觉预览暂不可用：{exc.message}")
    elif format_name == "docx" and artifact:
        st.info("DOCX 文件已生成；本机未检测到 Word/LibreOffice，因此暂时只能预览文字内容。")

    if artifact:
        try:
            artifact_bytes = read_artifact(artifact)
            if artifact_bytes is not None:
                st.download_button(
                    f"下载 {format_label}",
                    data=artifact_bytes,
                    file_name=str(artifact.get("file_name") or f"resume.{format_name}"),
                    mime=str(artifact.get("mime_type") or "application/octet-stream"),
                    key=f"{key_prefix}-download-{format_name}",
                )
        except ApiRequestError as exc:
            st.error(f"文件下载失败：{exc.message}")
    elif status not in {"needs_compression_consent", "blocked"}:
        st.warning("当前格式尚未生成可下载文件，请根据上方状态修复环境后重试。")


def _render_original_pdf(st: Any, data: bytes, filename: str) -> None:
    """Render the uploaded source PDF next to its structured fields."""

    _render_pdf_document(
        st,
        data,
        filename,
        heading="原始 PDF 对照",
        caption_suffix="仅在当前本地页面会话中显示",
    )


def _render_resume_parse_confirmation(st: Any, resume: dict[str, Any]) -> dict[str, Any] | None:
    """Show the machine parse as editable sections and require confirmation."""

    sections = [item for item in resume.get("sections") or [] if isinstance(item, dict)]
    if not sections:
        st.error("没有从简历中识别出可编辑区块，请检查文件内容。")
        return None
    st.markdown("#### 校验机器解析结果")
    st.caption("请检查每个区块的类型、标题和正文。确认后才会进入岗位匹配与生成流程。")
    edited: list[dict[str, Any]] = []
    module_values = list(_RESUME_MODULE_LABELS)
    for index, section in enumerate(sections, 1):
        section_id = str(section.get("section_id") or f"section-{index}")
        current_module = str(section.get("module") or "basic")
        with st.expander(
            f"{index}. {_RESUME_MODULE_LABELS.get(current_module, current_module)} · "
            f"{section.get('title') or '未命名区块'}",
            expanded=index <= 2 or current_module == "projects",
        ):
            module = st.selectbox(
                "区块类型",
                module_values,
                index=module_values.index(current_module) if current_module in module_values else 0,
                format_func=lambda value: _RESUME_MODULE_LABELS[value],
                key=f"wizard_resume_module_{section_id}",
            )
            title = st.text_input(
                "区块标题",
                value=str(section.get("title") or ""),
                key=f"wizard_resume_title_{section_id}",
            )
            content = st.text_area(
                "解析正文",
                value=str(section.get("content") or ""),
                height=180,
                key=f"wizard_resume_content_{section_id}",
            )
            edited.append(
                {
                    "section_id": section_id,
                    "old_value_hash": section.get("value_hash")
                    or _field_hash(str(section.get("content") or "")),
                    "module": module,
                    "title": title.strip() or None,
                    "content": content,
                    "confirmed": True,
                }
            )
    confirmed = bool(sections) and all(bool(item.get("confirmed")) for item in sections)
    if not confirmed:
        if st.button("保存校正并确认简历解析", key="wizard_confirm_resume_parse"):
            if any(not str(item.get("content") or "").strip() for item in edited):
                st.warning("简历区块正文不能为空")
            elif not any(item.get("module") == "projects" for item in edited):
                st.warning("请至少把一个区块标记为“项目经历”")
            else:
                try:
                    updated = _request(
                        "PATCH",
                        f"/api/resumes/{resume['resume_id']}/sections",
                        json={"sections": edited},
                    )
                    st.session_state["wizard_resume"] = updated["resume"]
                    st.rerun()
                except ApiRequestError as exc:
                    st.error(exc.message)
        return None
    st.success("简历解析结果已确认")
    return resume


def _job_page(
    jobs: list[dict[str, Any]], page_index: int, page_size: int = 10
) -> tuple[list[dict[str, Any]], int, int]:
    """Return one stable UI page and its clamped index/page count."""

    page_size = max(1, int(page_size))
    page_count = max(1, (len(jobs) + page_size - 1) // page_size)
    page_index = min(max(0, int(page_index)), page_count - 1)
    start = page_index * page_size
    return jobs[start : start + page_size], page_index, page_count


def _preview_matches_task(preview: dict[str, Any] | None, task: dict[str, Any], export_format: str) -> bool:
    if not preview:
        return False
    expected_format = "md" if export_format == "markdown" else export_format
    return bool(
        preview.get("task_id") == task.get("task_id")
        and preview.get("snapshot_id") == task.get("resume_snapshot_id")
        and preview.get("format") in {export_format, expected_format}
    )


def _legacy_main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - depends on optional UI extra
        raise SystemExit("请安装 UI 依赖：pip install -e .[ui]") from exc

    st.set_page_config(page_title="Resume Agent", layout="wide")
    st.markdown(
        """
        <style>
        div[data-testid="stTabs"] [role="tablist"] {
            display: flex !important;
            flex-wrap: nowrap !important;
            white-space: nowrap !important;
            min-height: 2.5rem;
            overflow-x: auto;
        }
        div[data-testid="stTabs"] [role="tab"] {
            flex: 0 0 auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Resume Agent")
    st.caption("本地中文简历制作与岗位匹配智能体")
    tabs = st.tabs(["环境与模型", "职位与生成", "简历", "模板"])

    with tabs[0]:
        if st.button("刷新环境"):
            try:
                st.session_state["environment"] = _request("GET", "/api/environment")
            except Exception as exc:
                st.error(f"环境检测失败：{exc}")
        env = st.session_state.get("environment")
        if env:
            api_status = str((env.get("api") or {}).get("status") or "unknown")
            if api_status == "ready":
                st.success("本地环境读取成功")
            else:
                st.error(f"本地环境读取失败：{_friendly_status(api_status)}")
            st.caption(
                "数据库："
                f"{_friendly_status((env.get('database') or {}).get('status'))}；"
                "MCP："
                f"{_friendly_status((env.get('mcp') or {}).get('status'))}"
            )
        with st.expander("Microsoft Edge 配置"):
            try:
                edge_settings = _request("GET", "/api/settings")
            except Exception:
                edge_settings = {"settings_version": 0}
            edge_path = st.text_input(
                "msedge.exe 路径（留空则自动检测）",
                value=str(edge_settings.get("edge_path") or ""),
                key="edge-path",
            )
            edge_cols = st.columns(2)
            if edge_cols[0].button("保存 Edge 路径", key="save-edge-path"):
                try:
                    version = int(edge_settings.get("settings_version", 0))
                    values = {
                        key: value
                        for key, value in edge_settings.items()
                        if key not in {"settings_version", "configured_in_session"}
                    }
                    values["edge_path"] = edge_path or None
                    _request(
                        "PUT",
                        "/api/settings",
                        headers={"If-Match": str(version)},
                        json={"settings_version": version, "values": values},
                    )
                    st.success("Edge 路径已保存")
                except Exception as exc:
                    st.error(str(exc))
            if edge_cols[1].button("检测 Edge 浏览器", key="recheck-edge-mcp"):
                try:
                    st.session_state["edge_detection"] = _request(
                        "POST", "/api/environment/boss/recheck"
                    )
                except Exception as exc:
                    st.session_state["edge_detection"] = {
                        "servers": {"boss": {"status": "unavailable", "last_error_code": str(exc)}}
                    }
            edge_detection = st.session_state.get("edge_detection")
            if edge_detection:
                edge_ok, edge_message, edge_detail = _edge_detection_result(edge_detection)
                (st.success if edge_ok else st.error)(edge_message)
                if edge_detail:
                    st.caption(edge_detail)
        st.subheader("模型配置")
        if st.button("扫描 Ollama 已下载模型", key="scan-ollama"):
            try:
                st.session_state["ollama_detection"] = _request(
                    "POST", "/api/models/ollama/scan"
                )
            except Exception as exc:
                st.session_state["ollama_detection"] = {
                    "status": "unavailable",
                    "error_code": str(exc),
                }
        ollama_detection = st.session_state.get("ollama_detection")
        if ollama_detection:
            ollama_ok, ollama_message, ollama_detail = _ollama_detection_result(
                ollama_detection
            )
            (st.success if ollama_ok else st.error)(ollama_message)
            if ollama_detail:
                st.caption(ollama_detail)
        try:
            all_models = _request("GET", "/api/models")
            models = _display_model_profiles(all_models, limit=3)
            for model in models:
                cols = st.columns([2, 2, 1, 1])
                cols[0].write(f"{model['role']} / {model['model_name']}")
                cols[1].write(model["base_url"])
                cols[2].write(_friendly_status(model["status"]))
                if cols[3].button("检测", key=f"probe-{model['profile_id']}"):
                    try:
                        st.session_state[f"model_detection_{model['profile_id']}"] = _request(
                            "POST",
                            f"/api/models/{model['profile_id']}/probe",
                            json={"credential_handle_id": st.session_state.get(f"credential-{model['profile_id']}")},
                        )
                    except Exception as exc:
                        st.session_state[f"model_detection_{model['profile_id']}"] = {
                            "model_name": model["model_name"],
                            "status": "probe_failed",
                            "error_code": str(exc),
                        }
                model_detection = st.session_state.get(
                    f"model_detection_{model['profile_id']}"
                )
                if model_detection:
                    model_ok, model_message = _model_detection_result(model_detection)
                    (st.success if model_ok else st.error)(model_message)
                if model.get("provider") == "openai_compatible":
                    try:
                        saved_key_status = _request(
                            "GET",
                            f"/api/credentials/stored/{model['profile_id']}",
                        )
                    except Exception:
                        saved_key_status = {}
                    api_key = st.text_input(
                        f"{model['model_name']} API Key（本机加密保存）",
                        type="password",
                        value=(
                            _MASKED_API_KEY
                            if saved_key_status.get("usable")
                            else ""
                        ),
                        key=f"api-key-{model['profile_id']}",
                    )
                    new_secret = _new_api_key(api_key)
                    if st.button("加密保存 API Key", key=f"credential-button-{model['profile_id']}", disabled=not new_secret):
                        try:
                            credential = _request(
                                "POST",
                                "/api/credentials/session",
                                json={
                                    "profile_id": model["profile_id"],
                                    "api_key": new_secret,
                                    "persist": True,
                                    "ttl_seconds": 31536000,
                                },
                            )
                            st.session_state[f"credential-{model['profile_id']}"] = credential["credential_handle_id"]
                            st.success("API Key 已在本机加密保存，后续启动可直接使用")
                        except Exception as exc:
                            st.error(str(exc))
                    if saved_key_status.get("usable"):
                        st.caption("已保存：••••••••••••（本机加密，可直接使用）")
        except Exception as exc:
            st.error(str(exc))

        with st.expander("新增模型配置"):
            profile_role = st.selectbox("角色", ["chat", "embedding"], key="new-profile-role")
            profile_provider = st.selectbox("Provider", ["ollama", "openai_compatible"], key="new-profile-provider")
            profile_id = st.text_input("Profile ID", key="new-profile-id")
            default_url = "http://127.0.0.1:11434" if profile_provider == "ollama" else "https://api.openai.com/v1"
            profile_url = st.text_input("Base URL", value=default_url, key=f"new-profile-url-{profile_provider}")
            profile_model = st.text_input("模型名称", key="new-profile-model")
            context_window = st.number_input("上下文窗口（chat）", min_value=0, value=32768 if profile_role == "chat" else 0, step=1024)
            max_input = st.number_input("最大输入长度（embedding）", min_value=0, value=8192 if profile_role == "embedding" else 0, step=256)
            make_default = st.checkbox("设为该角色默认模型", value=True, key="new-profile-default")
            if st.button("保存模型配置", key="save-model-profile"):
                try:
                    payload = {
                        "profile_id": profile_id,
                        "role": profile_role,
                        "provider": profile_provider,
                        "base_url": profile_url,
                        "model_name": profile_model,
                        "default_for_role": make_default,
                        "context_window_tokens": int(context_window) or None,
                        "max_input_tokens": int(max_input) or None,
                        "generation_params": (
                            {
                                "max_output_tokens": 8192,
                                "timeout_seconds": 120,
                            }
                            if profile_role == "chat"
                            else {}
                        ),
                    }
                    st.json(_request("POST", "/api/models", json=payload))
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        consent_cols = st.columns(2)
        if consent_cols[0].button("同意向外部模型发送本任务所需数据", key="grant-external-consent"):
            try:
                st.json(_request("POST", "/api/consents/external-model", json={"granted": True}))
            except Exception as exc:
                st.error(str(exc))
        if consent_cols[1].button("撤销外部模型同意", key="revoke-external-consent"):
            try:
                st.json(_request("POST", "/api/consents/external-model", json={"granted": False}))
            except Exception as exc:
                st.error(str(exc))

        with st.expander("匹配评分权重"):
            try:
                saved_settings = _request("GET", "/api/settings")
            except Exception:
                saved_settings = {"settings_version": 0}
            current_dimensions = saved_settings.get("dimension_weights") or {
                "skills": 0.25, "responsibilities": 0.25, "projects": 0.25, "keywords": 0.25
            }
            current_components = saved_settings.get("component_weights") or {"deterministic": 0.5, "semantic": 0.5}
            dimension_values = {
                name: st.number_input(label, min_value=0, max_value=100, value=int(round(float(current_dimensions.get(name, 0.25)) * 100)), key=f"weight-{name}") / 100
                for name, label in {
                    "skills": "技能 %", "responsibilities": "职责 %", "projects": "项目 %", "keywords": "关键词 %"
                }.items()
            }
            component_values = {
                name: st.number_input(label, min_value=0, max_value=100, value=int(round(float(current_components.get(name, 0.5)) * 100)), key=f"component-{name}") / 100
                for name, label in {"deterministic": "确定性指标 %", "semantic": "语义评分 %"}.items()
            }
            if st.button("保存评分权重", key="save-scoring-weights"):
                if abs(sum(dimension_values.values()) - 1.0) > 1e-8 or abs(sum(component_values.values()) - 1.0) > 1e-8:
                    st.error("两组权重必须分别合计 100%")
                else:
                    try:
                        version = int(saved_settings.get("settings_version", 0))
                        ordinary_values = {
                            key: value
                            for key, value in saved_settings.items()
                            if key not in {"settings_version", "configured_in_session"}
                        }
                        ordinary_values.update({"dimension_weights": dimension_values, "component_weights": component_values})
                        st.json(
                            _request(
                                "PUT",
                                "/api/settings",
                                headers={"If-Match": str(version)},
                                json={"settings_version": version, "values": ordinary_values},
                            )
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

    with tabs[1]:
        st.subheader("创建职位快照")
        title = st.text_input("岗位名称", key="job-title")
        city = st.text_input("城市", key="job-city")
        company = st.text_input("公司（可选）", key="job-company")
        manual_job_panel = st.expander("手动粘贴 JD 或公司招聘官网网址（预留）")
        with manual_job_panel:
            raw = st.text_area("粘贴 JD", height=180)
            st.text_input("公司招聘官网网址", key="job-url")
            st.caption("第一版仅预留公司招聘官网入口，暂不访问或解析该网址。")
        risk_cols = st.columns(2)
        if risk_cols[0].button("确认 BOSS 页面访问风险", key="grant-post-time-risk"):
            try:
                _request(
                    "POST",
                    "/api/consents/post-time-risk",
                    json={"granted": True, "policy_version": "boss-post-time-risk-v1"},
                )
                st.success("已记录风险确认，后续沿用该选择")
            except Exception as exc:
                st.error(str(exc))
        if risk_cols[1].button("撤销风险确认", key="revoke-post-time-risk"):
            try:
                _request("POST", "/api/consents/post-time-risk", json={"granted": False})
                st.info("已撤销，后续 BOSS 页面操作将被阻止")
            except Exception as exc:
                st.error(str(exc))

        activity_filter = st.selectbox(
            "HR 活跃状态",
            ["不限", "在线或刚刚活跃", "今日活跃", "近3天活跃", "本周活跃", "活跃时间待解析"],
            key="job-hr-activity-filter",
        )

        def search_payload(cursor: str | None = None) -> dict[str, Any]:
            values: dict[str, Any] = {
                "title": title,
                "city": city,
                "company": company or None,
                "limit": 20,
            }
            if cursor:
                values["cursor"] = cursor
            if activity_filter != "不限":
                values["hr_activity"] = activity_filter
            return values

        if st.button("搜索 BOSS 职位", key="search-boss-jobs"):
            if not title or not city:
                st.warning("搜索时岗位名称和城市必填")
            else:
                try:
                    st.session_state["job_search"] = _request(
                        "POST", "/api/jobs/search", json=search_payload()
                    )
                    st.session_state["job_search_page"] = 0
                    st.session_state.pop("job_detail_view", None)
                except ApiRequestError as exc:
                    if exc.code == "post_time_risk_consent_required":
                        st.warning("请先点击“确认 BOSS 页面访问风险”，确认后再搜索")
                    else:
                        st.error(f"职位搜索失败：{exc.message}")
                except Exception as exc:
                    st.error(str(exc))

        job_search = st.session_state.get("job_search") or {}
        search_jobs = list(job_search.get("jobs") or [])
        search_feedback = _job_search_feedback(job_search) if job_search else None
        if search_feedback:
            level, message = search_feedback
            if level == "warning":
                st.warning(message)
            else:
                st.info(message)
        if search_jobs:
            page_jobs, page_index, page_count = _job_page(
                search_jobs, int(st.session_state.get("job_search_page", 0)), 10
            )
            st.session_state["job_search_page"] = page_index
            header_cols = st.columns([3, 2, 2, 1.5, 1.2, 1.5])
            for column, label in zip(
            header_cols, ["岗位（点击后在采集 Edge 打开）", "公司", "城市", "薪资", "HR", "HR 活跃"]
            ):
                column.markdown(f"**{label}**")
            for item in page_jobs:
                row_cols = st.columns([3, 2, 2, 1.5, 1.2, 1.5])
                if row_cols[0].button(
                    str(item.get("title") or "未命名岗位"),
                    key=f"open-collected-job-{item['job_id']}",
                    use_container_width=True,
                ):
                    try:
                        _request(
                            "POST",
                            "/api/jobs/browser/open",
                            json={"url": item.get("source_url")},
                        )
                        st.success("已在岗位采集 Edge 中打开，不会切换到系统默认浏览器")
                    except ApiRequestError as exc:
                        st.error(f"岗位页面打开失败：{exc.message}")
                    except Exception as exc:
                        st.error(str(exc))
                row_cols[1].write(item.get("company") or "未知公司")
                row_cols[2].write(item.get("city") or "城市未知")
                row_cols[3].write(item.get("salary") or "薪资未知")
                row_cols[4].write(item.get("hr_name") or "未识别")
                row_cols[5].write(
                    item.get("hr_activity") or "活跃时间待解析"
                )

            has_previous = page_index > 0
            has_next = (page_index + 1) * 10 < len(search_jobs) or bool(
                job_search.get("next_cursor")
            )
            pager_cols = st.columns([1, 2, 1, 1.4])
            if pager_cols[0].button(
                "上一页",
                key=f"previous-job-page-{page_index}",
                disabled=not has_previous,
            ):
                st.session_state["job_search_page"] = page_index - 1
                st.rerun()
            pager_cols[1].markdown(
                f"<div style='text-align:center;padding-top:0.45rem'>第 {page_index + 1} / {page_count} 页，已获取 {len(search_jobs)} 条</div>",
                unsafe_allow_html=True,
            )
            if pager_cols[2].button(
                "下一页",
                key=f"next-job-page-{page_index}",
                disabled=not has_next,
            ):
                try:
                    target_page = page_index + 1
                    updated_search = job_search
                    if (
                        len(search_jobs) < (target_page + 1) * 10
                        and job_search.get("next_cursor")
                    ):
                        more = _request(
                            "POST",
                            "/api/jobs/search",
                            json=search_payload(str(job_search["next_cursor"])),
                        )
                        merged = {
                            item["job_id"]: item
                            for item in [*search_jobs, *(more.get("jobs") or [])]
                        }
                        updated_search = {**job_search, **more, "jobs": list(merged.values())}
                        st.session_state["job_search"] = updated_search
                    st.session_state["job_search_page"] = target_page
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            if pager_cols[3].button("关闭 BOSS 浏览器", key="close-boss-browser"):
                try:
                    _request("POST", "/api/jobs/browser/close")
                    st.session_state["job_search"] = {
                        **job_search,
                        "next_cursor": None,
                        "has_more": False,
                    }
                    st.success("BOSS 浏览器已关闭，登录状态仍会保留")
                except Exception as exc:
                    st.error(str(exc))

            labels = {
                item["job_id"]: f"{item.get('title')} · {item.get('company') or '未知公司'} · {item.get('salary') or '薪资未知'}"
                for item in page_jobs
            }
            selected_job_id = st.selectbox(
                "选择需要进一步查看的岗位",
                [""] + list(labels),
                format_func=lambda item: "请选择岗位" if not item else labels[item],
                key=f"selected-search-job-{page_index}",
            )
            if selected_job_id:
                selected_job = next(
                    item for item in page_jobs if item["job_id"] == selected_job_id
                )
                st.session_state["selected_job_id"] = selected_job_id
                st.markdown("#### 已选岗位概要")
                st.markdown(f"**{selected_job.get('title') or '未命名岗位'}**")
                st.write(
                    f"公司：{selected_job.get('company') or '未知'} ｜ "
                    f"城市：{selected_job.get('city') or '未知'} ｜ "
                    f"薪资：{selected_job.get('salary') or '未知'}"
                )
                if st.button(
                    "在岗位采集 Edge 中打开",
                    key=f"open-selected-job-{selected_job_id}",
                ):
                    try:
                        _request(
                            "POST",
                            "/api/jobs/browser/open",
                            json={"url": selected_job.get("source_url")},
                        )
                        st.success("岗位已在采集 Edge 的新标签页中打开")
                    except ApiRequestError as exc:
                        st.error(f"岗位页面打开失败：{exc.message}")
                    except Exception as exc:
                        st.error(str(exc))
                confirm_parse = st.checkbox(
                    "确认进一步解析该岗位的完整 JD",
                    key=f"confirm-job-parse-{selected_job_id}",
                )
                if st.button(
                    "解析并展示岗位详情",
                    key=f"parse-selected-job-{selected_job_id}",
                    disabled=not confirm_parse,
                ):
                    try:
                        detail = _request(
                            "POST",
                            "/api/jobs/from-url",
                            json={"url": selected_job["source_url"]},
                        )
                        if detail.get("status") != "ready":
                            st.warning(detail.get("message") or "岗位详情尚未解析完成")
                        else:
                            st.session_state["selected_job_id"] = detail.get("job_id")
                            st.session_state["job_detail_view"] = detail
                            st.rerun()
                    except ApiRequestError as exc:
                        st.error(f"职位详情解析失败：{exc.message}")
                    except Exception as exc:
                        st.error(str(exc))

                detail = st.session_state.get("job_detail_view") or {}
                detail_job = detail.get("job") if isinstance(detail, dict) else None
                if isinstance(detail_job, dict) and detail_job.get("job_id") == selected_job_id:
                    st.markdown("#### 岗位职责与 JD")
                    responsibilities = list(detail_job.get("responsibilities") or [])
                    requirements = list(detail_job.get("requirements") or [])
                    skills = list(detail_job.get("skills") or [])
                    st.markdown("**岗位职责**")
                    if responsibilities:
                        for value in responsibilities:
                            st.markdown(f"- {value}")
                    else:
                        st.caption("未从页面中拆分出独立的岗位职责")
                    st.markdown("**任职要求**")
                    if requirements:
                        for value in requirements:
                            st.markdown(f"- {value}")
                    else:
                        st.caption("未从页面中拆分出独立的任职要求")
                    st.markdown("**技能要求**")
                    st.write("、".join(str(value) for value in skills) if skills else "未单独识别")
                    with st.expander("查看完整 JD 原文", expanded=True):
                        st.text(detail_job.get("raw_text") or "暂无可复制的 JD 原文")
        with manual_job_panel:
            if st.button("保存粘贴的职位"):
                if not title:
                    st.warning("请填写岗位名称")
                else:
                    result = _request(
                        "POST",
                        "/api/jobs",
                        json={"title": title, "city": city or None, "company": company or None, "raw_text": raw or None},
                    )
                    st.success(f"职位已保存：{result['job_id']}")
            st.button(
                "公司官网解析（第一版暂不可用）",
                key="parse-job-url",
                disabled=True,
            )

        try:
            history = _request("GET", "/api/jobs/history?limit=20")
            if history.get("jobs"):
                with st.expander("查看最近解析的职位"):
                    st.dataframe(
                        [
                            {"job_id": item["job_id"], "岗位": item["title"], "公司": item.get("company"), "城市": item.get("city"), "HR": item.get("hr_name") or "HR 未识别", "HR 活跃": item.get("hr_activity") or "活跃时间待解析"}
                            for item in history["jobs"]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
        except Exception:
            pass

    with tabs[2]:
        st.subheader("上传主简历")
        uploaded = st.file_uploader("仅支持 DOCX、文本型 PDF、Markdown 或 TXT（不超过 10 MB）", type=["docx", "pdf", "md", "markdown", "txt"])
        if st.button("上传并解析", key="upload-resume") and uploaded is not None:
            try:
                result = _request(
                    "POST",
                    "/api/resumes/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                )
                st.session_state["resume"] = result
                st.success(f"已上传：{result['resume_id']}")
                st.json(result)
            except Exception as exc:
                st.error(str(exc))
        resume = st.session_state.get("resume")
        if resume:
            st.write(f"当前简历：{resume.get('filename')} · {resume.get('page_count') or '?'} 页")
            if st.button("标记全部项目区块", key="mark-projects"):
                try:
                    project_ids = [item["section_id"] for item in resume.get("sections", []) if item.get("module") == "projects"]
                    if not project_ids:
                        st.warning("没有自动识别到项目区块，请调用 API 逐段标记")
                    else:
                        result = _request("POST", f"/api/resumes/{resume['resume_id']}/mark-project-region", json={"section_ids": project_ids})
                        st.session_state["resume"] = result["resume"]
                        st.success("项目区域已标记")
                except Exception as exc:
                    st.error(str(exc))

    with tabs[3]:
        st.subheader("中文简历模板")
        github_token = st.text_input(
            "GitHub Token（可选，仅当前进程内存）",
            type="password",
            key="github-token",
        )
        if st.button("暂存 GitHub Token", key="save-github-token", disabled=not github_token):
            try:
                credential = _request(
                    "POST",
                    "/api/credentials/session",
                    json={"scope": "github", "secret": github_token, "ttl_seconds": 300},
                )
                st.session_state["github_credential_handle"] = credential["credential_handle_id"]
                st.success("GitHub Token 已暂存在进程内存，5 分钟后失效")
            except Exception as exc:
                st.error(str(exc))
        template_query = st.text_input("模板关键词", value="中文 技术 简历 模板", key="template-query")
        template_type = st.selectbox("模板格式", ["不限", "docx", "markdown"], key="template-type")
        if st.button("搜索 GitHub 中文模板", key="search-templates"):
            try:
                template_payload = {
                    "query": template_query,
                    "per_page": 5,
                    "credential_handle_id": st.session_state.get("github_credential_handle"),
                }
                if template_type != "不限":
                    template_payload["file_type"] = template_type
                st.session_state["template_search"] = _request(
                    "POST", "/api/templates/search", json=template_payload
                )
            except Exception as exc:
                st.error(str(exc))
        template_search = st.session_state.get("template_search") or {
            "items": [
                {
                    "template_id": "builtin-cn-single-page",
                    "name": "内置中文单页模板",
                    "file_type": "markdown",
                    "license": "internal",
                }
            ]
        }
        template_items = list(template_search.get("items") or [])
        if template_items:
            template_labels = {
                item["template_id"]: (
                    f"{item.get('name') or item.get('file_path') or item['template_id']} · "
                    f"{item.get('file_type') or '未知格式'} · {item.get('license') or '许可证未知'}"
                )
                for item in template_items
            }
            selected_template_id = st.selectbox(
                "选择模板",
                list(template_labels),
                format_func=lambda item: template_labels[item],
                key="selected-template-id",
            )
            selected_template = next(
                item for item in template_items if item["template_id"] == selected_template_id
            )
            template_cols = st.columns(2)
            if template_cols[0].button("预览模板", key="preview-template"):
                try:
                    preview_payload = {
                        "credential_handle_id": st.session_state.get("github_credential_handle")
                    }
                    if selected_template.get("source_url"):
                        preview_payload["source_url"] = selected_template["source_url"]
                    st.session_state["template_preview"] = _request(
                        "POST",
                        f"/api/templates/{selected_template_id}/preview",
                        json=preview_payload,
                    )
                except Exception as exc:
                    st.error(str(exc))
            if template_cols[1].button("下载并选用", key="download-template"):
                try:
                    download_payload = {
                        key: selected_template.get(key)
                        for key in ("source_url", "repository", "file_path", "remote_version", "filename")
                        if selected_template.get(key)
                    }
                    download_payload["credential_handle_id"] = st.session_state.get("github_credential_handle")
                    cached = _request(
                        "POST",
                        f"/api/templates/{selected_template_id}/download",
                        json=download_payload,
                    )
                    st.session_state["task_template_id"] = cached.get("template_id", selected_template_id)
                    st.success(f"已选用模板：{st.session_state['task_template_id']}")
                except Exception as exc:
                    st.error(str(exc))
            if st.session_state.get("template_preview"):
                st.json(st.session_state["template_preview"])

    # The generation workflow belongs to the selected job.  Re-entering the
    # same tab appends this staged section after the job list while keeping the
    # standalone resume/template tabs available for advanced management.
    with tabs[1]:
        job_id = str(st.session_state.get("selected_job_id") or "")
        detail = st.session_state.get("job_detail_view") or {}
        detail_job = detail.get("job") if isinstance(detail, dict) else None
        if not job_id:
            st.info("请先从上方职位列表选择一个感兴趣的岗位。")
            return
        if not isinstance(detail_job, dict) or detail_job.get("job_id") != job_id:
            st.info("请先确认并解析所选岗位的完整 JD，解析完成后再选择生成方式。")
            return

        st.divider()
        st.subheader("下一步：选择处理方式")
        flow_options = {
            "resume_edit": "上传现有简历并新增或替换项目",
            "template_resume": "使用模板生成一份新简历",
            "project_only": "只生成可复制的项目经历纯文本",
            "screenshot_placeholder": "岗位截图（请使用简历项目向导）",
        }
        requested_flow = st.radio(
            "本次希望得到什么？",
            list(flow_options),
            index=None,
            format_func=lambda value: flow_options[value],
            key=f"job-generation-flow-{job_id}",
        )
        if requested_flow is None:
            st.caption("选择后只展示该流程需要的内容，其他设置保持折叠。")
            return
        if requested_flow == "screenshot_placeholder":
            st.info("请在“简历项目向导”中使用岗位截图粘贴、上传、校正和确认流程。")
            return

        resume_id: str | None = None
        template_id: str | None = None
        base_facts: dict[str, Any] = {}
        output_mode = requested_flow
        if requested_flow == "resume_edit":
            st.markdown("#### 上传或选择主简历")
            flow_upload = st.file_uploader(
                "DOCX、文本型 PDF、Markdown 或 TXT（不超过 10 MB）",
                type=["docx", "pdf", "md", "markdown", "txt"],
                key=f"flow-resume-upload-{job_id}",
            )
            if st.button(
                "上传并用于当前岗位",
                key=f"flow-upload-resume-{job_id}",
                disabled=flow_upload is None,
            ):
                try:
                    uploaded_resume = _request(
                        "POST",
                        "/api/resumes/upload",
                        files={
                            "file": (
                                flow_upload.name,
                                flow_upload.getvalue(),
                                flow_upload.type or "application/octet-stream",
                            )
                        },
                    )
                    st.session_state["resume"] = uploaded_resume
                    st.success(f"主简历已解析：{uploaded_resume['resume_id']}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            active_resume = st.session_state.get("resume") or {}
            resume_id = str(active_resume.get("resume_id") or "") or None
            if resume_id:
                st.success(
                    f"当前使用：{active_resume.get('filename') or resume_id}；"
                    "生成后可选择新增项目或替换匹配度较低的项目。"
                )
            else:
                st.info("请先上传一份主简历，随后才会显示模型与生成设置。")
                return
        elif requested_flow == "template_resume":
            st.markdown("#### 选择简历模板")
            cached_template = str(st.session_state.get("task_template_id") or "")
            template_choices = ["builtin-cn-single-page"]
            if cached_template and cached_template not in template_choices:
                template_choices.append(cached_template)
            template_id = st.selectbox(
                "模板",
                template_choices,
                format_func=lambda value: (
                    "系统默认中文单页模板"
                    if value == "builtin-cn-single-page"
                    else f"已下载模板：{value}"
                ),
                key=f"flow-template-{job_id}",
            )
            st.caption("如需选择其他 GitHub 中文模板，可先前往“模板”栏下载，再返回这里选择。")
            st.markdown("#### 填写新简历基础信息")
            base_cols = st.columns(2)
            base_facts = {
                "name": base_cols[0].text_input("姓名", key=f"base-name-{job_id}"),
                "phone": base_cols[1].text_input("电话", key=f"base-phone-{job_id}"),
                "email": base_cols[0].text_input("邮箱", key=f"base-email-{job_id}"),
                "objective": base_cols[1].text_input(
                    "求职目标", value=str(detail_job.get("title") or ""), key=f"base-objective-{job_id}"
                ),
                "location": base_cols[0].text_input("所在地（可选）", key=f"base-location-{job_id}"),
                "certificates": base_cols[1].text_input("证书（可选）", key=f"base-certificates-{job_id}"),
            }
        else:
            st.info("本流程直接生成一份可复制项目经历，不要求上传简历或填写个人信息。")

        st.markdown("#### 生成设置")
        try:
            available_models = _request("GET", "/api/models")
        except Exception:
            available_models = []
        chat_models = _ready_model_profiles(available_models, "chat")
        embedding_models = _ready_model_profiles(available_models, "embedding")
        chat_profile_id = st.selectbox(
            "聊天模型",
            [item["profile_id"] for item in chat_models],
            index=next((index for index, item in enumerate(chat_models) if item.get("default_for_role")), 0),
            format_func=lambda value: next(
                (_model_profile_label(item) for item in chat_models if item["profile_id"] == value),
                value,
            ),
            key="task-chat-profile",
        ) if chat_models else None
        embedding_profile_id = st.selectbox(
            "Embedding 模型",
            [item["profile_id"] for item in embedding_models],
            index=next((index for index, item in enumerate(embedding_models) if item.get("default_for_role")), 0),
            format_func=lambda value: next(
                (_model_profile_label(item) for item in embedding_models if item["profile_id"] == value),
                value,
            ),
            key="task-embedding-profile",
        ) if embedding_models else None
        count = st.slider(
            "候选项目数量",
            1,
            5,
            3,
            key=f"flow-candidate-count-{job_id}-{requested_flow}",
        )
        if requested_flow == "project_only":
            st.caption("每份候选都可直接复制；确认后不会进入简历新增或替换流程。")
        if st.button("开始生成", key=f"start-generation-{job_id}-{requested_flow}"):
            try:
                task = _request(
                    "POST",
                    "/api/tasks",
                    json={
                        "job_id": job_id,
                        "resume_id": resume_id or None,
                        "template_id": template_id,
                        "output_mode": output_mode,
                        "base_facts": base_facts,
                        "candidate_count": count,
                        "chat_profile_id": chat_profile_id,
                        "embedding_profile_id": embedding_profile_id,
                        "credential_handle_id": st.session_state.get(f"credential-{chat_profile_id}"),
                    },
                )
                st.session_state["task_id"] = task["task_id"]
                st.session_state["task"] = task
                st.success("生成流程已启动")
            except Exception as exc:
                st.error(str(exc))
        try:
            task_history = _request("GET", "/api/tasks?limit=20")
        except Exception:
            task_history = {"tasks": []}
        task_options = [item["task_id"] for item in task_history.get("tasks", [])]
        current_task_id = st.session_state.get("task_id", "")
        if current_task_id and current_task_id not in task_options:
            task_options.insert(0, current_task_id)
        if task_options:
            task_id = st.selectbox(
                "历史生成记录",
                task_options,
                index=task_options.index(current_task_id) if current_task_id in task_options else 0,
                format_func=lambda value: next(
                    (
                        f"{item.get('job_title') or '未命名岗位'} · {item.get('status')} · {value}"
                        for item in task_history.get("tasks", [])
                        if item.get("task_id") == value
                    ),
                    value,
                ),
            )
            st.session_state["task_id"] = task_id
        else:
            task_id = st.text_input("生成记录 ID", value=current_task_id)
        if st.button("刷新生成状态") and task_id:
            try:
                st.session_state["task"] = _request("GET", f"/api/tasks/{task_id}")
            except Exception as exc:
                st.error(str(exc))
        task = st.session_state.get("task")
        if task:
            task_project_only = str(
                task.get("requested_output_mode") or task.get("output_mode") or ""
            ) == "project_only"
            st.write(f"状态：{task['status']} / 节点：{task.get('current_node')}")
            st.write(f"checkpoint：{task['checkpoint_version']}")
            if task.get("blocked_reason"):
                st.warning(f"任务被阻断：{task['blocked_reason']}。请按提示修复后重试。")
            if task.get("match"):
                st.metric("匹配分", task["match"]["total_score"])
            if task_project_only and task.get("status") == "completed":
                selected = _selected_candidate(task)
                if selected:
                    plain_text = _candidate_plain_text(selected)
                    st.subheader("可复制项目经历")
                    from app.ui.clipboard_component import clipboard_text_button

                    clipboard_text_button(
                        plain_text,
                        key=f"copy-completed-direct-{task_id}",
                    )
                    st.code(plain_text, language=None)
                    st.download_button(
                        "下载纯文本",
                        data=plain_text,
                        file_name="project.txt",
                        mime="text/plain",
                        key=f"download-completed-direct-{task_id}",
                    )
            for candidate in task.get("candidates", []):
                with st.expander(f"{candidate['candidate_slot_id']} · {candidate['title']} · {candidate['match_score']}"):
                    if task_project_only:
                        plain_text = _candidate_plain_text(candidate)
                        from app.ui.clipboard_component import clipboard_text_button

                        clipboard_text_button(
                            plain_text,
                            key=f"copy-candidate-{candidate['candidate_id']}",
                        )
                        st.code(plain_text, language=None)
                    else:
                        st.write(candidate["introduction"])
                        st.write("技术栈：" + "、".join(candidate["tech_stack"]))
                        st.write("\n".join(f"- {line}" for line in candidate["solutions"]))
                    if st.button("选择", key=f"select-{candidate['candidate_id']}"):
                        try:
                            updated = _request("POST", f"/api/tasks/{task_id}/resume", json=_task_action(task, "select", {"candidate_id": candidate["candidate_id"]}))
                            st.session_state["task"] = updated
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

            if task.get("current_node") == "candidate_confirmation" and task.get("status") == "waiting_user":
                selected = _selected_candidate(task)
                if selected:
                    if task_project_only:
                        plain_text = _candidate_plain_text(selected)
                        st.subheader("可复制项目经历")
                        from app.ui.clipboard_component import clipboard_text_button

                        clipboard_text_button(
                            plain_text,
                            key=f"copy-direct-{task_id}",
                        )
                        st.code(plain_text, language=None)
                        st.info("请核实全部字段；确认后即可将以上纯文本用于粘贴。")
                    else:
                        st.info("请逐项确认候选内容。确认会校验字段哈希，之后再选择新增或替换。")
                    fields = _CANDIDATE_CONFIRMATION_FIELDS
                    master_key = f"confirm-all-{task_id}-{selected['candidate_id']}"
                    field_prefix = f"confirm-{task_id}-{selected['candidate_id']}-"
                    st.checkbox(
                        "全部核实",
                        value=False,
                        key=master_key,
                        on_change=_set_all_candidate_confirmations,
                        kwargs={
                            "session_state": st.session_state,
                            "master_key": master_key,
                            "field_key_prefix": field_prefix,
                        },
                    )
                    confirmations = []
                    for field in fields:
                        checked = st.checkbox(
                            field,
                            value=False,
                            key=f"{field_prefix}{field}",
                        )
                        if checked:
                            confirmations.append({"field_id": field, "value_hash": _field_hash(selected[field]), "confirmed": True})
                    if st.button(
                        "核实并生成可复制纯文本" if task_project_only else "确认候选字段",
                        key=f"confirm-candidate-{task_id}",
                        disabled=len(confirmations) != len(fields),
                    ):
                        try:
                            updated = _request(
                                "POST",
                                f"/api/tasks/{task_id}/confirmations",
                                json={**_task_action(task, "confirm"), "candidate_id": selected["candidate_id"], "confirm": True, "confirmations": confirmations},
                            )
                            st.session_state["task"] = updated
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))
                else:
                    st.error("未能读取已选择候选，请刷新任务后重新选择")

            if task.get("current_node") == "project_application" and task.get("status") == "waiting_user":
                decision = task.get("project_decision") or {}
                st.subheader("应用已确认项目")
                st.write(decision.get("reason") or "请选择如何使用该项目")
                labels = {"add": "新增到简历", "replace": "替换现有项目", "copy_only": "只生成可复制内容"}
                project_only = task.get("output_mode") == "project_only"
                recommended = "copy_only" if project_only else (decision.get("recommended_action") or "add")
                modes = ["copy_only"] if project_only else ["add", "replace", "copy_only"]
                mode = st.radio(
                    "处理方式",
                    modes,
                    index=modes.index(recommended) if recommended in modes else 0,
                    format_func=lambda item: labels[item],
                    horizontal=True,
                    key=f"project-action-{task_id}",
                )
                target_project_id = None
                rankings = decision.get("project_rankings") or []
                if mode == "replace":
                    project_ids = [str(item.get("project_id")) for item in rankings if item.get("project_id")]
                    if project_ids:
                        project_labels = {
                            str(item.get("project_id")): (
                                f"{str(item.get('title') or item.get('project_id'))}"
                                f"（匹配度 {item.get('match_score', 0)}）"
                            )
                            for item in rankings
                            if item.get("project_id")
                        }
                        default_target = str(decision.get("target_project_id") or project_ids[-1])
                        target_project_id = st.selectbox(
                            "替换项目",
                            project_ids,
                            index=project_ids.index(default_target) if default_target in project_ids else 0,
                            format_func=lambda item: project_labels.get(item, item),
                            key=f"replace-project-{task_id}",
                        )
                        selected_meta = next(
                            (item for item in rankings if str(item.get("project_id")) == target_project_id),
                            {},
                        )
                        if selected_meta.get("content_preview"):
                            st.caption(f"原项目摘要：{selected_meta['content_preview']}")
                    else:
                        st.warning("没有可替换的项目，请选择新增或仅复制")
                if st.button("确认处理方式", key=f"apply-project-{task_id}", disabled=mode == "replace" and not target_project_id):
                    try:
                        updated = _request(
                            "POST",
                            f"/api/tasks/{task_id}/resume",
                            json=_task_action(task, "approve", {"project_action": mode, "target_project_id": target_project_id}),
                        )
                        st.session_state["task"] = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            module_labels = {"summary": "个人简介", "skills": "技能清单", "evaluation": "个人评价"}
            current_node = str(task.get("current_node") or "")
            if current_node in {f"{name}_review" for name in module_labels} and task.get("status") == "waiting_user":
                module = current_node.removesuffix("_review")
                assessment = (task.get("module_assessment") or {}).get(module, {})
                st.subheader(f"是否优化{module_labels[module]}")
                st.write(f"当前匹配分：{assessment.get('score', 0)}")
                module_feedback = st.text_area("反馈（可选）", key=f"module-feedback-{task_id}-{module}")
                cols = st.columns(2)
                if cols[0].button("生成优化建议", key=f"module-generate-{task_id}-{module}"):
                    try:
                        updated = _request(
                            "POST",
                            f"/api/tasks/{task_id}/resume",
                            json=_task_action(task, "approve", {"feedback": module_feedback}),
                        )
                        st.session_state["task"] = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
                if cols[1].button("保留原内容", key=f"module-skip-{task_id}-{module}"):
                    try:
                        updated = _request("POST", f"/api/tasks/{task_id}/resume", json=_task_action(task, "skip"))
                        st.session_state["task"] = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            if current_node in {f"{name}_confirmation" for name in module_labels} and task.get("status") == "waiting_user":
                module = current_node.removesuffix("_confirmation")
                draft = (task.get("module_drafts") or {}).get(module) or {}
                st.subheader(f"确认{module_labels[module]}补丁")
                st.text_area("建议内容", value=str(draft.get("new_value") or ""), height=150, disabled=True, key=f"module-draft-{task_id}-{module}")
                revision_feedback = st.text_area("不满意时填写反馈", key=f"module-revise-{task_id}-{module}")
                cols = st.columns(3)
                if cols[0].button("确认采用", key=f"module-confirm-{task_id}-{module}"):
                    try:
                        updated = _request(
                            "POST",
                            f"/api/tasks/{task_id}/resume",
                            json=_task_action(task, "confirm", {"value_hash": draft.get("new_value_hash")}),
                        )
                        st.session_state["task"] = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
                if cols[1].button("按反馈重写", key=f"module-rewrite-{task_id}-{module}", disabled=not revision_feedback.strip()):
                    try:
                        updated = _request(
                            "POST",
                            f"/api/tasks/{task_id}/resume",
                            json=_task_action(task, "revise", {"feedback": revision_feedback}),
                        )
                        st.session_state["task"] = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
                if cols[2].button("拒绝并保留原文", key=f"module-reject-{task_id}-{module}"):
                    try:
                        updated = _request("POST", f"/api/tasks/{task_id}/resume", json=_task_action(task, "reject"))
                        st.session_state["task"] = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            if current_node == "layout_review" and task.get("status") == "waiting_user":
                layout = task.get("layout_decision") or {}
                pages = int(layout.get("estimated_page_count") or 1)
                st.subheader("排版检查")
                st.write(f"当前估算：{pages} 页")
                cols = st.columns(2)
                if cols[0].button("继续排版预览", key=f"layout-approve-{task_id}"):
                    try:
                        updated = _request("POST", f"/api/tasks/{task_id}/resume", json=_task_action(task, "approve", {"page_count": pages}))
                        st.session_state["task"] = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
                if pages > 1 and cols[1].button("允许多页并继续", key=f"layout-multipage-{task_id}"):
                    try:
                        updated = _request("POST", f"/api/tasks/{task_id}/resume", json=_task_action(task, "skip", {"page_count": pages}))
                        st.session_state["task"] = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            if current_node == "compression_review" and task.get("status") == "waiting_user":
                st.subheader("一页压缩确认")
                st.caption("仅改变字体、间距、页边距与区块排布，不修改正文。")
                cols = st.columns(3)
                if cols[0].button("常规无损压缩", key=f"compression-approve-{task_id}"):
                    try:
                        updated = _request(
                            "POST",
                            f"/api/tasks/{task_id}/resume",
                            json=_task_action(
                                task,
                                "approve",
                                {
                                    "accept_compression": True,
                                    "allow_rewrite": False,
                                    "compression_level": "standard",
                                },
                            ),
                        )
                        st.session_state["task"] = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
                if cols[1].button("紧凑重排", key=f"compression-compact-{task_id}"):
                    try:
                        updated = _request(
                            "POST",
                            f"/api/tasks/{task_id}/resume",
                            json=_task_action(
                                task,
                                "approve",
                                {
                                    "accept_compression": True,
                                    "allow_rewrite": False,
                                    "compression_level": "aggressive",
                                },
                            ),
                        )
                        st.session_state["task"] = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
                if cols[2].button("保持多页", key=f"compression-skip-{task_id}"):
                    try:
                        updated = _request("POST", f"/api/tasks/{task_id}/resume", json=_task_action(task, "skip"))
                        st.session_state["task"] = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            if task.get("status") in {"waiting_user", "paused"} and current_node in {"candidate_review", "candidate_confirmation", "feedback_review"}:
                feedback = st.text_area("候选反馈", key=f"feedback-{task_id}", height=100)
                feedback_chat_profile = st.selectbox(
                    "本轮使用的聊天模型",
                    [item["profile_id"] for item in chat_models],
                    index=next(
                        (
                            index
                            for index, item in enumerate(chat_models)
                            if item.get("profile_id") == chat_profile_id
                        ),
                        0,
                    ),
                    key=f"feedback-model-{task_id}",
                ) if chat_models else None
                if st.button("提交反馈并重新生成", key=f"feedback-submit-{task_id}"):
                    try:
                        updated = _request(
                            "POST",
                            f"/api/tasks/{task_id}/feedback",
                            json={
                                "thread_id": task["thread_id"],
                                "checkpoint_version": task["checkpoint_version"],
                                "feedback": feedback,
                                "chat_profile_id": feedback_chat_profile,
                                "credential_handle_id": st.session_state.get(f"credential-{feedback_chat_profile}"),
                            },
                        )
                        st.session_state["task"] = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            if current_node == "preview_review" and task.get("status") == "waiting_user" and task.get("resume_snapshot_id"):
                st.subheader("最终预览确认")
                review_format = st.selectbox("预览格式", ["markdown", "docx", "pdf"], key=f"review-format-{task_id}")
                compression = task.get("compression_decision") or {}
                if st.button("生成当前快照预览", key=f"review-preview-{task_id}"):
                    try:
                        preview = _request(
                            "POST",
                            f"/api/tasks/{task_id}/export-preview",
                            json={
                                "thread_id": task["thread_id"],
                                "checkpoint_version": task["checkpoint_version"],
                                "snapshot_id": task["resume_snapshot_id"],
                                "format": review_format,
                                "accept_compression": bool(compression.get("accepted", False)),
                                "allow_rewrite": bool(compression.get("allow_rewrite", False)),
                                "compression_level": str(compression.get("level") or "standard"),
                            },
                        )
                        st.session_state["review_preview"] = preview
                    except Exception as exc:
                        st.error(str(exc))
                review_preview = st.session_state.get("review_preview")
                review_matches = _preview_matches_task(review_preview, task, review_format)
                if review_matches:
                    _render_export_result(
                        st,
                        review_preview,
                        key_prefix=f"review-{task_id}-{review_format}",
                        content_label="当前快照内容",
                    )
                review_feedback = st.text_area("预览修改反馈", key=f"review-feedback-{task_id}")
                cols = st.columns(2)
                if cols[0].button("确认最终预览", key=f"review-confirm-{task_id}", disabled=not review_matches):
                    try:
                        updated = _request("POST", f"/api/tasks/{task_id}/resume", json=_task_action(task, "confirm"))
                        st.session_state["task"] = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
                if cols[1].button("返回修改", key=f"review-revise-{task_id}", disabled=not review_feedback.strip()):
                    try:
                        updated = _request(
                            "POST",
                            f"/api/tasks/{task_id}/resume",
                            json=_task_action(
                                task,
                                "revise",
                                {
                                    "feedback": review_feedback,
                                    "allow_rewrite": True,
                                },
                            ),
                        )
                        st.session_state["task"] = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            if task.get("status") == "completed" and task.get("resume_snapshot_id"):
                st.subheader("预览与导出")
                export_format = st.selectbox("格式", ["markdown", "docx", "pdf"], key=f"format-{task_id}")
                compression_mode = st.selectbox(
                    "无损压缩方式",
                    ["none", "standard", "aggressive"],
                    format_func=lambda value: {
                        "none": "不压缩",
                        "standard": "常规字体、间距和页边距压缩",
                        "aggressive": "紧凑重排（不改正文）",
                    }[value],
                    key=f"compress-mode-{task_id}",
                )
                accept_compression = compression_mode != "none"
                allow_rewrite = False
                if st.button("生成预览", key=f"preview-{task_id}"):
                    try:
                        preview = _request(
                            "POST",
                            f"/api/tasks/{task_id}/export-preview",
                            json={"thread_id": task["thread_id"], "checkpoint_version": task["checkpoint_version"], "snapshot_id": task["resume_snapshot_id"], "format": export_format, "accept_compression": accept_compression, "allow_rewrite": allow_rewrite, "compression_level": compression_mode if accept_compression else "standard"},
                        )
                        st.session_state["preview"] = preview
                    except Exception as exc:
                        st.error(str(exc))
                preview = st.session_state.get("preview")
                preview_matches = _preview_matches_task(preview, task, export_format)
                if preview_matches:
                    _render_export_result(
                        st,
                        preview,
                        key_prefix=f"preview-{task_id}-{export_format}",
                        content_label="预览内容",
                    )
                    if preview.get("compression", {}).get("warnings"):
                        for warning in preview["compression"]["warnings"]:
                            st.warning(warning)
                preview_reviewed = st.checkbox(
                    "我已检查当前快照的预览内容",
                    value=False,
                    disabled=not preview_matches,
                    key=f"preview-reviewed-{task_id}-{export_format}",
                )
                if st.button("导出文件", key=f"export-{task_id}", disabled=not preview_matches or not preview_reviewed):
                    try:
                        result = _request(
                            "POST",
                            f"/api/tasks/{task_id}/export",
                            json={"thread_id": task["thread_id"], "checkpoint_version": task["checkpoint_version"], "snapshot_id": task["resume_snapshot_id"], "format": export_format, "accept_compression": accept_compression, "allow_rewrite": allow_rewrite, "compression_level": compression_mode if accept_compression else "standard"},
                        )
                        st.session_state["export"] = result
                    except Exception as exc:
                        st.error(str(exc))
                exported = st.session_state.get("export")
                if exported and exported.get("content"):
                    _render_export_result(
                        st,
                        exported,
                        key_prefix=f"export-{task_id}-{export_format}",
                        content_label="最终导出内容",
                    )


def _render_compact_environment(st: Any) -> None:
    """Render only environment controls needed before entering the wizard."""

    st.subheader("环境与模型")
    if st.button("刷新环境状态", key="wizard-refresh-environment"):
        try:
            st.session_state["environment"] = _request("GET", "/api/environment")
        except Exception as exc:
            st.error(f"环境检测失败：{exc}")
    environment = st.session_state.get("environment") or {}
    if environment:
        api_status = str((environment.get("api") or {}).get("status") or "unknown")
        database_status = str((environment.get("database") or {}).get("status") or "unknown")
        mcp_status = str((environment.get("mcp") or {}).get("status") or "unknown")
        cols = st.columns(3)
        cols[0].metric("API", _friendly_status(api_status))
        cols[1].metric("数据库", _friendly_status(database_status))
        cols[2].metric("MCP", _friendly_status(mcp_status))

    with st.expander("Microsoft Edge", expanded=True):
        try:
            edge_settings = _request("GET", "/api/settings")
        except Exception:
            edge_settings = {"settings_version": 0}
        edge_path = st.text_input(
            "msedge.exe 路径（留空自动检测）",
            value=str(edge_settings.get("edge_path") or ""),
            key="wizard-edge-path",
        )
        edge_cols = st.columns(2)
        if edge_cols[0].button("保存 Edge 路径", key="wizard-save-edge"):
            try:
                version = int(edge_settings.get("settings_version", 0))
                values = {
                    key: value
                    for key, value in edge_settings.items()
                    if key not in {"settings_version", "configured_in_session"}
                }
                values["edge_path"] = edge_path or None
                _request(
                    "PUT",
                    "/api/settings",
                    headers={"If-Match": str(version)},
                    json={"settings_version": version, "values": values},
                )
                st.success("Edge 路径已保存")
            except Exception as exc:
                st.error(str(exc))
        if edge_cols[1].button("检测 Edge", key="wizard-detect-edge"):
            try:
                st.session_state["edge_detection"] = _request(
                    "POST", "/api/environment/boss/recheck"
                )
            except Exception as exc:
                st.error(str(exc))
        edge_detection = st.session_state.get("edge_detection")
        if edge_detection:
            edge_ok, edge_message, edge_detail = _edge_detection_result(edge_detection)
            (st.success if edge_ok else st.error)(edge_message)
            if edge_detail:
                st.caption(edge_detail)

    try:
        all_profiles = _request("GET", "/api/models")
    except Exception as exc:
        all_profiles = []
        st.error(str(exc))

    with st.expander("云端聊天模型（OpenAI 兼容）", expanded=True):
        cloud_profiles = [
            item
            for item in all_profiles
            if item.get("role") == "chat"
            and item.get("provider") == "openai_compatible"
        ]
        preferred_cloud_id = st.session_state.get("wizard_cloud_profile_id")
        cloud_profile = next(
            (
                item
                for item in cloud_profiles
                if item.get("profile_id") == preferred_cloud_id
            ),
            next(
                (item for item in cloud_profiles if item.get("default_for_role")),
                cloud_profiles[0] if cloud_profiles else None,
            ),
        )
        cloud_base_url = st.text_input(
            "Base URL",
            value=str(
                (cloud_profile or {}).get("base_url") or "https://api.openai.com/v1"
            ),
            key="wizard-cloud-base-url",
            help="填写兼容 OpenAI /chat/completions 的接口根地址，例如 https://api.openai.com/v1。",
        )
        cloud_model_name = st.text_input(
            "聊天模型名称",
            value=str((cloud_profile or {}).get("model_name") or ""),
            key="wizard-cloud-model-name",
        )
        existing_auth_scheme = str(
            ((cloud_profile or {}).get("generation_params") or {}).get("auth_scheme")
            or "bearer"
        )
        auth_labels = {
            "Bearer Key（标准 OpenAI 兼容接口）": "bearer",
            "直接 Key（DMXAPI 等接口）": "raw",
        }
        auth_options = list(auth_labels)
        cloud_auth_label = st.selectbox(
            "API Key 认证格式",
            auth_options,
            index=1 if existing_auth_scheme == "raw" else 0,
            key="wizard-cloud-auth-scheme",
            help="按服务商示例选择：Authorization: Bearer sk-... 选 Bearer；Authorization: sk-... 选直接 Key。",
        )
        cloud_auth_scheme = auth_labels[cloud_auth_label]
        cloud_candidate_profile_id = ""
        cloud_key_status: dict[str, Any] = {}
        try:
            cloud_candidate_url = _canonical_openai_base_url(cloud_base_url)
            cloud_candidate_digest = hashlib.sha256(
                f"{cloud_candidate_url}|{cloud_model_name.strip()}|{cloud_auth_scheme}".encode()
            ).hexdigest()[:12]
            cloud_candidate_profile_id = f"openai-chat-{cloud_candidate_digest}"
            cloud_key_status = _request(
                "GET", f"/api/credentials/stored/{cloud_candidate_profile_id}"
            )
        except Exception:
            cloud_key_status = {}
        cloud_api_key = st.text_input(
            "API Key（密码输入，本机加密保存）",
            type="password",
            value=_MASKED_API_KEY if cloud_key_status.get("usable") else "",
            key="wizard-cloud-api-key",
            placeholder=(
                "已在本机加密保存，无需重复输入"
                if cloud_key_status.get("usable")
                else "首次使用或旧密钥失效时必须填写"
            ),
        )
        if cloud_key_status.get("needs_reentry"):
            st.warning("旧 API Key 密文无法在当前登录会话解密，请重新输入一次并覆盖保存。")
        consent = st.checkbox(
            "我同意首次使用时向该外部模型发送所选岗位 JD",
            key="wizard-cloud-consent",
            help="同意状态会保存；API Key 使用 AES-GCM 和本机独立密钥加密，不写入数据库、日志或 Git。",
        )

        def ensure_cloud_profile() -> dict[str, Any]:
            if not cloud_base_url.strip() or not cloud_model_name.strip():
                raise ValueError("请填写 Base URL 和聊天模型名称")
            canonical_url = _canonical_openai_base_url(cloud_base_url)
            digest = hashlib.sha256(
                f"{canonical_url}|{cloud_model_name.strip()}|{cloud_auth_scheme}".encode()
            ).hexdigest()[:12]
            profile_id = f"openai-chat-{digest}"
            existing = next(
                (
                    item
                    for item in all_profiles
                    if item.get("profile_id") == profile_id
                ),
                None,
            )
            if existing is None:
                existing = _request(
                    "POST",
                    "/api/models",
                    json={
                        "profile_id": profile_id,
                        "role": "chat",
                        "provider": "openai_compatible",
                        "base_url": canonical_url,
                        "model_name": cloud_model_name.strip(),
                        "default_for_role": True,
                        "context_window_tokens": 32768,
                        "generation_params": {
                            "auth_scheme": cloud_auth_scheme,
                            # Resume/JD/candidate JSON is substantially longer
                            # than a connectivity probe.  A 1200-token default
                            # can be consumed entirely by reasoning models and
                            # looks like a lost credential because content is
                            # empty.  Persist a practical output budget for new
                            # cloud chat profiles.
                            "max_output_tokens": 8192,
                            "timeout_seconds": 120,
                        },
                    },
                )
                all_profiles.append(existing)
            else:
                existing = _request(
                    "POST", f"/api/models/{profile_id}/default"
                )
            st.session_state["wizard_cloud_profile_id"] = profile_id
            if consent:
                _request(
                    "POST",
                    "/api/consents/external-model",
                    json={"granted": True, "policy_version": "external-model-v1"},
                )
            new_secret = _new_api_key(cloud_api_key)
            if new_secret:
                credential = _request(
                    "POST",
                    "/api/credentials/session",
                    json={
                        "profile_id": profile_id,
                        "api_key": new_secret,
                        "persist": True,
                        "ttl_seconds": 31536000,
                    },
                )
                st.session_state[f"credential-{profile_id}"] = credential[
                    "credential_handle_id"
                ]
            else:
                stored = _request("GET", f"/api/credentials/stored/{profile_id}")
                if not stored.get("usable"):
                    raise ValueError("首次使用该模型时必须填写 API Key；保存成功后后续会自动使用")
                credential = _request(
                    "POST",
                    "/api/credentials/session",
                    json={
                        "profile_id": profile_id,
                        "persist": False,
                        "ttl_seconds": 31536000,
                    },
                )
                st.session_state[f"credential-{profile_id}"] = credential[
                    "credential_handle_id"
                ]
            return existing

        cloud_actions = st.columns(2)
        if cloud_actions[0].button(
            "保存配置",
            key="wizard-save-cloud-chat",
            use_container_width=True,
        ):
            try:
                saved_cloud = ensure_cloud_profile()
                st.success(f"已保存 {saved_cloud['model_name']}；API Key 已加密保存并可自动复用")
            except Exception as exc:
                st.error(str(exc))
        if cloud_actions[1].button(
            "保存并检测连接",
            key="wizard-probe-cloud-chat",
            use_container_width=True,
        ):
            try:
                saved_cloud = ensure_cloud_profile()
                profile_id = saved_cloud["profile_id"]
                st.session_state[f"model_detection_{profile_id}"] = _request(
                    "POST",
                    f"/api/models/{profile_id}/probe",
                    json={
                        "credential_handle_id": st.session_state.get(
                            f"credential-{profile_id}"
                        )
                    },
                )
            except Exception as exc:
                st.error(str(exc))
        active_cloud_id = st.session_state.get("wizard_cloud_profile_id") or (
            cloud_profile or {}
        ).get("profile_id")
        cloud_detection = st.session_state.get(
            f"model_detection_{active_cloud_id}"
        )
        if cloud_detection:
            cloud_ok, cloud_message = _model_detection_result(cloud_detection)
            (st.success if cloud_ok else st.error)(cloud_message)
        if active_cloud_id:
            try:
                stored_status = _request(
                    "GET", f"/api/credentials/stored/{active_cloud_id}"
                )
                if stored_status.get("usable"):
                    st.success("该聊天模型已有本机加密 API Key，可直接检测和使用")
                elif stored_status.get("needs_reentry"):
                    st.warning("该聊天模型的旧 API Key 无法解密，请重新输入一次")
            except Exception:
                pass
        st.caption(
            "DMXAPI 示例可填写 Base URL https://www.dmxapi.cn/v1 和实际可用模型 ID。认证格式按服务商文档选择。API Key 不回显，使用 AES-GCM 和本机独立密钥加密保存在 data/credentials（已被 Git 忽略），后续启动可直接使用。"
        )

    with st.expander("云端 Embedding 模型（OpenAI 兼容）", expanded=True):
        embedding_cloud_profiles = [
            item
            for item in all_profiles
            if item.get("role") == "embedding"
            and item.get("provider") == "openai_compatible"
        ]
        preferred_embedding_id = st.session_state.get("wizard_embedding_cloud_profile_id")
        embedding_cloud_profile = next(
            (
                item
                for item in embedding_cloud_profiles
                if item.get("profile_id") == preferred_embedding_id
            ),
            next(
                (item for item in embedding_cloud_profiles if item.get("default_for_role")),
                embedding_cloud_profiles[0] if embedding_cloud_profiles else None,
            ),
        )
        embedding_base_url = st.text_input(
            "Embedding Base URL",
            value=str(
                (embedding_cloud_profile or {}).get("base_url")
                or "https://api.openai.com/v1"
            ),
            key="wizard-embedding-cloud-base-url",
        )
        embedding_model_name = st.text_input(
            "Embedding 模型名称",
            value=str((embedding_cloud_profile or {}).get("model_name") or ""),
            key="wizard-embedding-cloud-model-name",
        )
        embedding_auth_scheme = str(
            ((embedding_cloud_profile or {}).get("generation_params") or {}).get(
                "auth_scheme"
            )
            or "bearer"
        )
        embedding_auth_label = st.selectbox(
            "Embedding API Key 认证格式",
            auth_options,
            index=1 if embedding_auth_scheme == "raw" else 0,
            key="wizard-embedding-cloud-auth-scheme",
        )
        embedding_auth_scheme = auth_labels[embedding_auth_label]
        embedding_candidate_profile_id = ""
        embedding_key_status: dict[str, Any] = {}
        try:
            embedding_candidate_url = _canonical_openai_base_url(embedding_base_url)
            embedding_candidate_digest = hashlib.sha256(
                f"{embedding_candidate_url}|{embedding_model_name.strip()}|{embedding_auth_scheme}".encode()
            ).hexdigest()[:12]
            embedding_candidate_profile_id = (
                f"openai-embedding-{embedding_candidate_digest}"
            )
            embedding_key_status = _request(
                "GET",
                f"/api/credentials/stored/{embedding_candidate_profile_id}",
            )
        except Exception:
            embedding_key_status = {}
        embedding_api_key = st.text_input(
            "Embedding API Key（密码输入，本机加密保存）",
            type="password",
            value=_MASKED_API_KEY if embedding_key_status.get("usable") else "",
            key="wizard-embedding-cloud-api-key",
            placeholder=(
                "已在本机加密保存，无需重复输入"
                if embedding_key_status.get("usable")
                else "首次使用或旧密钥失效时必须填写"
            ),
        )
        if embedding_key_status.get("needs_reentry"):
            st.warning("旧 Embedding API Key 无法解密，请重新输入一次并覆盖保存。")
        embedding_consent = st.checkbox(
            "我同意向该外部 Embedding 模型发送用于匹配和候选去重的文本",
            key="wizard-embedding-cloud-consent",
        )

        def ensure_embedding_cloud_profile() -> dict[str, Any]:
            if not embedding_base_url.strip() or not embedding_model_name.strip():
                raise ValueError("请填写 Embedding Base URL 和模型名称")
            canonical_url = _canonical_openai_base_url(embedding_base_url)
            digest = hashlib.sha256(
                f"{canonical_url}|{embedding_model_name.strip()}|{embedding_auth_scheme}".encode()
            ).hexdigest()[:12]
            profile_id = f"openai-embedding-{digest}"
            existing = next(
                (item for item in all_profiles if item.get("profile_id") == profile_id),
                None,
            )
            if existing is None:
                existing = _request(
                    "POST",
                    "/api/models",
                    json={
                        "profile_id": profile_id,
                        "role": "embedding",
                        "provider": "openai_compatible",
                        "base_url": canonical_url,
                        "model_name": embedding_model_name.strip(),
                        "default_for_role": True,
                        "max_input_tokens": 8192,
                        "generation_params": {"auth_scheme": embedding_auth_scheme},
                    },
                )
                all_profiles.append(existing)
            else:
                existing = _request("POST", f"/api/models/{profile_id}/default")
            st.session_state["wizard_embedding_cloud_profile_id"] = profile_id
            if embedding_consent:
                _request(
                    "POST",
                    "/api/consents/external-model",
                    json={"granted": True, "policy_version": "external-model-v1"},
                )
            new_secret = _new_api_key(embedding_api_key)
            if new_secret:
                credential = _request(
                    "POST",
                    "/api/credentials/session",
                    json={
                        "profile_id": profile_id,
                        "api_key": new_secret,
                        "persist": True,
                        "ttl_seconds": 31536000,
                    },
                )
                st.session_state[f"credential-{profile_id}"] = credential[
                    "credential_handle_id"
                ]
            else:
                stored = _request("GET", f"/api/credentials/stored/{profile_id}")
                if not stored.get("usable"):
                    raise ValueError("首次使用该 Embedding 模型时必须填写 API Key")
                credential = _request(
                    "POST",
                    "/api/credentials/session",
                    json={
                        "profile_id": profile_id,
                        "persist": False,
                        "ttl_seconds": 31536000,
                    },
                )
                st.session_state[f"credential-{profile_id}"] = credential[
                    "credential_handle_id"
                ]
            return existing

        embedding_actions = st.columns(2)
        if embedding_actions[0].button(
            "保存 Embedding 配置",
            key="wizard-save-cloud-embedding",
            use_container_width=True,
        ):
            try:
                saved_embedding = ensure_embedding_cloud_profile()
                st.success(
                    f"已保存 {saved_embedding['model_name']} 为默认 Embedding 模型"
                )
            except Exception as exc:
                st.error(str(exc))
        if embedding_actions[1].button(
            "保存并检测 Embedding",
            key="wizard-probe-cloud-embedding",
            use_container_width=True,
        ):
            try:
                saved_embedding = ensure_embedding_cloud_profile()
                profile_id = saved_embedding["profile_id"]
                st.session_state[f"model_detection_{profile_id}"] = _request(
                    "POST",
                    f"/api/models/{profile_id}/probe",
                    json={
                        "credential_handle_id": st.session_state.get(
                            f"credential-{profile_id}"
                        )
                    },
                )
            except Exception as exc:
                st.error(str(exc))
        active_embedding_id = st.session_state.get(
            "wizard_embedding_cloud_profile_id"
        ) or (embedding_cloud_profile or {}).get("profile_id")
        embedding_detection = st.session_state.get(
            f"model_detection_{active_embedding_id}"
        )
        if embedding_detection:
            embedding_ok, embedding_message = _model_detection_result(
                embedding_detection
            )
            (st.success if embedding_ok else st.error)(embedding_message)
        if active_embedding_id:
            try:
                stored_status = _request(
                    "GET", f"/api/credentials/stored/{active_embedding_id}"
                )
                if stored_status.get("usable"):
                    st.success("该 Embedding 模型已有本机加密 API Key，可直接检测和使用")
                elif stored_status.get("needs_reentry"):
                    st.warning("该 Embedding 模型的旧 API Key 无法解密，请重新输入一次")
            except Exception:
                pass
        st.caption(
            "Embedding 接口使用 /v1/embeddings；聊天模型与 Embedding 模型的 URL、模型名称和 API Key 相互独立。"
        )

    with st.expander("Ollama 本地聊天与 Embedding 模型", expanded=True):
        if st.button("扫描 Ollama 已下载模型", key="wizard-scan-ollama"):
            try:
                st.session_state["ollama_detection"] = _request(
                    "POST", "/api/models/ollama/scan"
                )
            except Exception as exc:
                st.error(str(exc))
        ollama_detection = st.session_state.get("ollama_detection")
        if ollama_detection:
            ok, message, _ = _ollama_detection_result(ollama_detection)
            (st.success if ok else st.error)(message)
            installed_models = _ollama_installed_models(ollama_detection)
            if ok and installed_models:
                st.dataframe(
                    [
                        {
                            "模型名称": item["name"],
                            "能力": "、".join(item["capabilities"]) or "待探测",
                            "参数量": item["parameter_size"],
                            "量化": item["quantization_level"],
                            "上下文": item["context_length"] or "未知",
                            "向量维度": item["embedding_length"] or "—",
                        }
                        for item in installed_models
                    ],
                    width="stretch",
                    hide_index=True,
                )

                model_by_name = {item["name"]: item for item in installed_models}
                ollama_profiles = [
                    item
                    for item in all_profiles
                    if item.get("provider") == "ollama"
                ]

                def select_and_probe_ollama(role: str, model_name: str) -> None:
                    matches = [
                        item
                        for item in ollama_profiles
                        if item.get("role") == role
                        and item.get("model_name") == model_name
                    ]
                    profile = next(
                        (item for item in matches if item.get("default_for_role")),
                        matches[0] if matches else None,
                    )
                    if profile is None:
                        digest = hashlib.sha256(
                            f"{role}|{model_name}".encode("utf-8")
                        ).hexdigest()[:12]
                        scanned = model_by_name[model_name]
                        profile = _request(
                            "POST",
                            "/api/models",
                            json={
                                "profile_id": f"ollama-{role}-{digest}",
                                "role": role,
                                "provider": "ollama",
                                "base_url": str(
                                    ollama_detection.get("base_url")
                                    or "http://127.0.0.1:11434"
                                ),
                                "model_name": model_name,
                                "default_for_role": True,
                                "context_window_tokens": (
                                    int(scanned["context_length"])
                                    if role == "chat"
                                    and scanned.get("context_length")
                                    else None
                                ),
                                "max_input_tokens": (
                                    int(scanned["context_length"])
                                    if role == "embedding"
                                    and scanned.get("context_length")
                                    else None
                                ),
                            },
                        )
                    else:
                        profile = _request(
                            "POST",
                            f"/api/models/{profile['profile_id']}/default",
                        )
                    result = _request(
                        "POST",
                        f"/api/models/{profile['profile_id']}/probe",
                        json={},
                    )
                    st.session_state[f"model_detection_{profile['profile_id']}"] = result
                    st.session_state[f"wizard_selected_ollama_{role}"] = model_name
                    st.session_state[f"wizard_selected_ollama_{role}_profile"] = profile[
                        "profile_id"
                    ]
                    st.rerun()

                role_columns = st.columns(2)
                for column, role, label in (
                    (role_columns[0], "chat", "聊天模型"),
                    (role_columns[1], "embedding", "Embedding 模型"),
                ):
                    names = _ollama_role_model_names(ollama_detection, role)
                    with column:
                        st.markdown(f"#### 选择{label}")
                        if not names:
                            st.warning(f"扫描结果中没有支持{label}能力的模型")
                            continue
                        current_default = next(
                            (
                                str(item.get("model_name"))
                                for item in ollama_profiles
                                if item.get("role") == role
                                and item.get("default_for_role")
                                and item.get("model_name") in names
                            ),
                            None,
                        )
                        remembered = st.session_state.get(
                            f"wizard_selected_ollama_{role}"
                        )
                        selected_name = st.radio(
                            f"已安装的{label}",
                            names,
                            index=(
                                names.index(str(remembered))
                                if remembered in names
                                else (
                                    names.index(current_default)
                                    if current_default in names
                                    else 0
                                )
                            ),
                            key=f"wizard-ollama-{role}-choice",
                        )
                        if st.button(
                            f"选用并检测{label}",
                            key=f"wizard-select-ollama-{role}",
                            use_container_width=True,
                        ):
                            try:
                                select_and_probe_ollama(role, selected_name)
                            except ApiRequestError as exc:
                                st.error(exc.message)
                            except Exception as exc:
                                st.error(str(exc))
                        selected_profile = next(
                            (
                                item
                                for item in ollama_profiles
                                if item.get("role") == role
                                and item.get("model_name") == selected_name
                            ),
                            None,
                        )
                        profile_id = (
                            (selected_profile or {}).get("profile_id")
                            or st.session_state.get(
                                f"wizard_selected_ollama_{role}_profile"
                            )
                        )
                        detection = st.session_state.get(
                            f"model_detection_{profile_id}"
                        )
                        if detection:
                            model_ok, model_message = _model_detection_result(detection)
                            (st.success if model_ok else st.error)(model_message)
            elif ok:
                st.warning("Ollama 当前没有已下载模型，请安装模型后重新扫描")

        with st.expander("新增其他模型配置"):
            role = st.selectbox("模型角色", ["chat", "embedding"], key="wizard-new-role")
            provider = st.selectbox(
                "模型类型", ["ollama", "openai_compatible"], key="wizard-new-provider"
            )
            profile_id = st.text_input("Profile ID", key="wizard-new-profile-id")
            default_url = (
                "http://127.0.0.1:11434"
                if provider == "ollama"
                else "https://api.openai.com/v1"
            )
            base_url = st.text_input("Base URL", value=default_url, key="wizard-new-url")
            model_name = st.text_input("模型名称", key="wizard-new-model")
            if st.button("保存模型配置", key="wizard-save-model"):
                try:
                    _request(
                        "POST",
                        "/api/models",
                        json={
                            "profile_id": profile_id,
                            "role": role,
                            "provider": provider,
                            "base_url": base_url,
                            "model_name": model_name,
                            "default_for_role": True,
                            "context_window_tokens": 32768 if role == "chat" else None,
                            "max_input_tokens": 8192 if role == "embedding" else None,
                            "generation_params": (
                                {
                                    "max_output_tokens": 8192,
                                    "timeout_seconds": 120,
                                }
                                if role == "chat"
                                else {}
                            ),
                        },
                    )
                    st.success("模型配置已保存，请点击检测")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


def _editor_lines(value: str, *, split_commas: bool = False) -> list[str]:
    text = str(value or "")
    if split_commas:
        pieces = re.split(r"[\n,，、]+", text)
    else:
        pieces = text.splitlines()
    result: list[str] = []
    for piece in pieces:
        cleaned = re.sub(r"^\s*(?:[-—–*•·]|\d+[、.．)）])\s*", "", piece).strip()
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result


def _show_job_detail(st: Any, detail: dict[str, Any]) -> dict[str, Any] | None:
    job = detail.get("job") if isinstance(detail, dict) else None
    if not isinstance(job, dict):
        return None
    job_id = str(job.get("job_id") or detail.get("job_id") or "")
    st.success(f"已取得岗位：{job.get('title') or '未命名岗位'}")
    st.caption(
        f"{job.get('company') or '公司未知'} ｜ {job.get('city') or '城市未知'} ｜ "
        f"{job.get('salary') or '薪资未知'} ｜ HR：{job.get('hr_name') or '姓名未识别'} · "
        f"{job.get('hr_activity') or '活跃时间待解析'}"
    )
    parser = detail.get("parser") if isinstance(detail.get("parser"), dict) else {}
    if parser.get("parser_mode") == "model_enhanced":
        st.success(
            f"JD 已由聊天模型增强解析：{parser.get('model_name') or parser.get('model_profile_id')}"
        )
    elif parser.get("parser_mode") == "human_corrected":
        st.success("当前显示已保存的人工校正结果")
    elif parser:
        warnings = list(parser.get("warnings") or [])
        st.warning(warnings[0] if warnings else "聊天模型不可用，当前显示规则解析结果")
        if parser.get("error_code"):
            st.caption(f"错误代码：{parser['error_code']}。下方可直接点击“再次使用大模型解析”。")
    with st.expander("岗位职责、任职要求与完整 JD", expanded=True):
        st.caption("以下内容均可直接校正；职责和任职要求每行一条，技能每行一个。")
        responsibilities_text = st.text_area(
            "岗位职责（可编辑）",
            value="\n".join(job.get("responsibilities") or []),
            height=150,
            key=f"wizard_edit_responsibilities_{job_id}",
            placeholder="每行一条岗位职责",
        )
        requirements_text = st.text_area(
            "任职要求（可编辑）",
            value="\n".join(job.get("requirements") or []),
            height=150,
            key=f"wizard_edit_requirements_{job_id}",
            placeholder="每行一条任职要求",
        )
        skills_text = st.text_area(
            "技能要求（可编辑）",
            value="\n".join(job.get("skills") or []),
            height=100,
            key=f"wizard_edit_skills_{job_id}",
            placeholder="每行一个技能，也支持逗号或顿号分隔",
        )
        raw_text = st.text_area(
            "完整 JD 原文",
            value=str(job.get("raw_text") or ""),
            height=220,
            key=f"wizard_edit_raw_text_{job_id}",
        )
    return {
        "raw_text": raw_text,
        "edited_responsibilities": _editor_lines(responsibilities_text),
        "edited_requirements": _editor_lines(requirements_text),
        "edited_skills": _editor_lines(skills_text, split_commas=True),
    }


def _render_job_search(st: Any, chat_profile_id: str | None = None) -> None:
    title = st.text_input("岗位名称", key="wizard-search-title")
    city = st.text_input("城市", key="wizard-search-city")
    company = st.text_input("公司（可选）", key="wizard-search-company")
    hr_activity = st.selectbox(
        "HR 活跃状态",
        ["不限", "在线或刚刚活跃", "今日活跃", "近3天活跃", "本周活跃", "活跃时间待解析"],
        key="wizard-search-hr-activity",
        help=(
            "系统逐个选择左侧岗位卡片，再从与该岗位同步的右侧 HR 卡片中原子读取"
            "姓名和活跃状态；两项必须来自同一容器，避免异步切换造成岗位间错配。"
        ),
    )
    risk_confirmed = st.checkbox(
        "我确认使用独立采集 Edge 访问 BOSS，并知悉账号风控风险",
        key="wizard-search-risk",
    )
    if st.button("搜索 BOSS 职位", key="wizard-search-button"):
        if not title.strip() or not city.strip():
            st.warning("岗位名称和城市为必填项")
        elif not risk_confirmed:
            st.warning("请先确认 BOSS 访问风险")
        else:
            try:
                _request(
                    "POST",
                    "/api/consents/post-time-risk",
                    json={"granted": True, "policy_version": "boss-post-time-risk-v1"},
                )
                with st.status("正在读取第 1 批岗位卡片…", expanded=False) as status_box:
                    search_result = _request(
                        "POST",
                        "/api/jobs/search",
                        timeout=120,
                        json={
                            "title": title,
                            "city": city,
                            "company": company or None,
                            "hr_activity": None if hr_activity == "不限" else hr_activity,
                            "limit": 10,
                        },
                    )
                    loaded_count = len(search_result.get("jobs") or [])
                    status_box.update(
                        label=f"第 1 批已读取 {loaded_count} 个岗位",
                        state="complete",
                    )
                st.session_state["wizard_job_search"] = search_result
                st.session_state["wizard_job_page"] = 0
                if search_result.get("status") == "waiting_user":
                    for key in (
                        "wizard_selected_job",
                        "wizard_job_detail",
                        "wizard_job_detail_confirmed",
                        "wizard_task",
                        "wizard_preview",
                    ):
                        st.session_state.pop(key, None)
            except ApiRequestError as exc:
                st.error(f"搜索失败：{exc.message}")
            except Exception as exc:
                st.error(str(exc))

    result = st.session_state.get("wizard_job_search") or {}
    retry_notice = st.session_state.pop("wizard_job_page_notice", None)
    if retry_notice:
        st.info(retry_notice)
    feedback = _job_search_feedback(result) if result else None
    if feedback:
        level, message = feedback
        (st.warning if level == "warning" else st.info)(message)
    jobs = list(result.get("jobs") or [])
    if not jobs:
        return
    page_jobs, page_index, page_count = _job_page(
        jobs, int(st.session_state.get("wizard_job_page", 0)), 10
    )
    st.session_state["wizard_job_page"] = page_index
    for item in page_jobs:
        with st.container(border=True):
            cols = st.columns([3, 2, 1.5, 1.2, 1.5])
            cols[0].markdown(f"**{item.get('title') or '未命名岗位'}**")
            cols[1].write(item.get("company") or "公司未知")
            cols[2].write(item.get("salary") or "薪资未知")
            cols[3].write(item.get("hr_name") or "HR 未识别")
            cols[4].write(item.get("hr_activity") or "活跃时间待解析")
            actions = st.columns(2)
            if actions[0].button(
                "查看并选择",
                key=f"wizard-select-job-{item['job_id']}",
                use_container_width=True,
            ):
                st.session_state["wizard_selected_job"] = item
                st.session_state.pop("wizard_job_detail", None)
                st.session_state.pop("wizard_job_parse_error", None)
                st.session_state.pop("wizard_task", None)
                st.session_state.pop("wizard_preview", None)
                st.rerun()
            if actions[1].button(
                "在采集 Edge 查看",
                key=f"wizard-open-job-{item['job_id']}",
                use_container_width=True,
            ):
                try:
                    _request(
                        "POST", "/api/jobs/browser/open", json={"url": item.get("source_url")}
                    )
                    st.success("已在岗位采集 Edge 中打开；此操作只用于查看，不会自动解析")
                except ApiRequestError as exc:
                    st.error(exc.message)
    pager = st.columns([1, 2, 1])
    if pager[0].button("上一页", disabled=page_index == 0, key="wizard-prev-page"):
        st.session_state["wizard_job_page"] = page_index - 1
        st.rerun()
    pager[1].markdown(
        f"<div style='text-align:center'>第 {page_index + 1} / {page_count} 页，共获取 {len(jobs)} 条</div>",
        unsafe_allow_html=True,
    )
    has_next = (page_index + 1) * 10 < len(jobs) or bool(result.get("next_cursor"))
    if pager[2].button("下一页", disabled=not has_next, key="wizard-next-page"):
        target_page = page_index + 1
        if len(jobs) < (target_page + 1) * 10 and result.get("next_cursor"):
            try:
                requested_cursor = str(result["next_cursor"])
                more: dict[str, Any] = {}
                with st.status(
                    f"正在后台加载第 {target_page + 1} 批岗位…",
                    expanded=False,
                ) as status_box:
                    # A minimized BOSS list may need a second trusted-wheel
                    # cycle before its lazy loader appends cards. Keep the
                    # browser hidden and retry the same cursor automatically;
                    # the user never needs to click the collection window.
                    for attempt in range(1, 3):
                        more = _request(
                            "POST",
                            "/api/jobs/search",
                            timeout=120,
                            json={
                                "title": title,
                                "city": city,
                                "company": company or None,
                                "hr_activity": None if hr_activity == "不限" else hr_activity,
                                "limit": 10,
                                "cursor": requested_cursor,
                            },
                        )
                        if (more.get("jobs") or []) or not more.get("load_pending"):
                            break
                        status_box.update(
                            label=f"第 {attempt} 次下拉仍在加载，正在自动继续…",
                            state="running",
                        )
                        requested_cursor = str(
                            more.get("next_cursor") or requested_cursor
                        )
                    status_box.update(
                        label=(
                            f"新增 {len(more.get('jobs') or [])} 个岗位"
                            if more.get("jobs")
                            else "当前批次尚无新岗位，已保留继续加载入口"
                        ),
                        state="complete",
                    )
                merged = {
                    item["job_id"]: item for item in [*jobs, *(more.get("jobs") or [])]
                }
                updated_search = {
                    **result,
                    **more,
                    "jobs": list(merged.values()),
                }
                st.session_state["wizard_job_search"] = updated_search
            except Exception as exc:
                st.error(str(exc))
                return
            merged_jobs = list(updated_search.get("jobs") or [])
            if len(merged_jobs) <= target_page * 10:
                if (
                    not updated_search.get("load_pending")
                    and not updated_search.get("next_cursor")
                ):
                    st.session_state["wizard_job_page_notice"] = "已经到达当前搜索结果末尾"
                elif updated_search.get("next_cursor"):
                    st.session_state["wizard_job_page_notice"] = (
                        "当前批次没有符合筛选条件的新岗位，可继续点击下一页读取后一批"
                    )
                st.session_state["wizard_job_page"] = page_index
                st.rerun()
        st.session_state["wizard_job_page"] = target_page
        st.rerun()

    selected = st.session_state.get("wizard_selected_job")
    if isinstance(selected, dict):
        st.markdown(f"#### 已选择：{selected.get('title')}")
        prior_parse_error = st.session_state.get("wizard_job_parse_error")
        if prior_parse_error:
            st.warning(
                f"上次大模型解析未完成：{prior_parse_error}。"
                "岗位选择和原始卡片仍已保留，可直接再次解析。"
            )
        if st.button(
            "再次使用大模型解析" if prior_parse_error else "解析该岗位的职责与完整 JD",
            key="wizard-parse-selected-job",
        ):
            try:
                detail = _request(
                    "POST",
                    "/api/jobs/from-url",
                    timeout=240,
                    json={
                        "url": selected.get("source_url"),
                        "chat_profile_id": chat_profile_id,
                        "credential_handle_id": st.session_state.get(
                            f"credential-{chat_profile_id}"
                        ),
                    },
                )
                if detail.get("status") == "ready":
                    st.session_state.pop("wizard_job_parse_error", None)
                    _set_wizard_job_detail(st, detail)
                    st.rerun()
                else:
                    message = detail.get("message") or "岗位详情暂未解析完成"
                    st.session_state["wizard_job_parse_error"] = message
                    st.warning(message)
            except ApiRequestError as exc:
                st.session_state["wizard_job_parse_error"] = exc.message
                st.error(exc.message)


def _render_task_progress(st: Any, task: dict[str, Any]) -> None:
    task_id = str(task.get("task_id") or "")
    if not task_id:
        return
    status = str(task.get("status") or "")
    node = str(task.get("current_node") or "")
    project_only = str(
        task.get("requested_output_mode") or task.get("output_mode") or ""
    ) == "project_only"
    if status in {"queued", "running"}:
        st.info(f"正在生成：{node or '准备中'}")
        # Streamlit reruns only this local page session. Polling the task API
        # does not refresh or navigate the BOSS collection browser.
        time.sleep(1.0)
        try:
            st.session_state["wizard_task"] = _request("GET", f"/api/tasks/{task_id}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
        return
    cols = st.columns([3, 1])
    cols[0].write(f"生成状态：{status} · 当前步骤：{node}")
    if cols[1].button("刷新状态", key=f"wizard-refresh-task-{task_id}"):
        try:
            st.session_state["wizard_task"] = _request("GET", f"/api/tasks/{task_id}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    blocked_reason = str(task.get("blocked_reason") or "")
    if blocked_reason == "feedback_clarification_required" or node == "feedback_review":
        classification = task.get("feedback_classification") or {}
        conflicts = "、".join(str(item) for item in classification.get("conflicts") or [])
        st.warning("这次反馈包含相互冲突或不够明确的要求，请补充一条明确的改写要求后继续。")
        if conflicts:
            st.caption(f"需要澄清：{conflicts}")
        clarification = st.text_area(
            "补充反馈",
            key=f"wizard-feedback-clarification-{task_id}",
            placeholder="例如：只突出 Python 和 FastAPI，不新增我没有做过的技术经历。",
        )
        try:
            clarification_models = _ready_model_profiles(_request("GET", "/api/models"), "chat")
        except Exception:
            clarification_models = []
        clarification_profile = (
            st.selectbox(
                "重新生成使用的聊天模型",
                [item["profile_id"] for item in clarification_models],
                format_func=lambda value: next(
                    (_model_profile_label(item) for item in clarification_models if item["profile_id"] == value),
                    value,
                ),
                key=f"wizard-feedback-clarification-model-{task_id}",
            )
            if clarification_models
            else None
        )
        if st.button(
            "提交澄清并重新生成",
            key=f"wizard-feedback-clarification-submit-{task_id}",
            disabled=not clarification.strip(),
        ):
            try:
                updated = _request(
                    "POST",
                    f"/api/tasks/{task_id}/feedback",
                    json={
                        "thread_id": task["thread_id"],
                        "checkpoint_version": task["checkpoint_version"],
                        "feedback": clarification,
                        "chat_profile_id": clarification_profile,
                        "credential_handle_id": st.session_state.get(
                            f"credential-{clarification_profile}"
                        ) if clarification_profile else None,
                    },
                )
                st.session_state["wizard_task"] = updated
                st.rerun()
            except ApiRequestError as exc:
                st.error(f"反馈提交失败：{exc.message}")
    elif blocked_reason:
        st.warning(f"当前流程暂停：{blocked_reason}。请先按提示修复后再重试。")
    if status == "failed":
        failure_message = str((task.get("messages") or ["生成失败"])[-1])
        st.error(f"{failure_message}\n\n可在修复模型或网络后重试当前步骤。")
        retry_step = "generate_candidates" if task.get("match") else "match"
        if st.button("重试失败步骤", key=f"wizard-retry-failed-{task_id}"):
            try:
                updated = _request(
                    "POST",
                    f"/api/tasks/{task_id}/retry-step",
                    json=_task_action(task, "retry", {"step": retry_step}),
                )
                st.session_state["wizard_task"] = updated
                st.rerun()
            except ApiRequestError as exc:
                st.error(f"重试失败：{exc.message}")
    if task.get("match"):
        st.metric("简历与岗位匹配分", task["match"].get("total_score"))

    candidates = [item for item in task.get("candidates", []) if item.get("status") != "duplicate"]
    if node in {"candidate_review", "candidate_confirmation", "project_application"}:
        saved_candidate = st.session_state.pop("wizard_candidate_edit_saved", None)
        if saved_candidate:
            st.success(f"候选项目 {saved_candidate} 的修改已保存，可以继续选择或核实。")
        for candidate in candidates:
            with st.expander(
                f"{candidate.get('candidate_slot_id')} · {candidate.get('title')} · 匹配分 {candidate.get('match_score')}",
                expanded=candidate.get("candidate_id") == task.get("selected_candidate_id"),
            ):
                if project_only:
                    plain_text = _candidate_plain_text(candidate)
                    from app.ui.clipboard_component import clipboard_text_button

                    clipboard_text_button(
                        plain_text,
                        key=f"wizard-copy-candidate-{candidate['candidate_id']}",
                    )
                    st.code(plain_text, language=None)
                else:
                    st.write(candidate.get("introduction"))
                    st.write("技术栈：" + "、".join(candidate.get("tech_stack") or []))
                    for line in candidate.get("solutions") or []:
                        st.markdown(f"- {line}")
                    st.write("结果：" + "；".join(candidate.get("results") or []))
                if (
                    node in {"candidate_review", "candidate_confirmation"}
                    and candidate.get("status") == "draft"
                ):
                    editor_key = (
                        f"wizard-candidate-editor-{task_id}-"
                        f"{candidate['candidate_id']}-{task['checkpoint_version']}"
                    )
                    with st.form(editor_key):
                        st.caption("可直接修改下列内容；保存后仍需重新核实全部字段。")
                        edit_title = st.text_input(
                            "项目名称",
                            value=str(candidate.get("title") or ""),
                        )
                        edit_period = st.text_input(
                            "项目时间",
                            value=str(candidate.get("period") or ""),
                        )
                        edit_introduction = st.text_area(
                            "项目简介",
                            value=str(candidate.get("introduction") or ""),
                            height=120,
                        )
                        edit_stack = st.text_input(
                            "技术栈（使用顿号、逗号或换行分隔）",
                            value="、".join(candidate.get("tech_stack") or []),
                        )
                        edit_solutions = st.text_area(
                            "解决方案（每行一条）",
                            value="\n".join(candidate.get("solutions") or []),
                            height=180,
                        )
                        edit_results = st.text_area(
                            "项目结果（每行一条）",
                            value="\n".join(candidate.get("results") or []),
                            height=120,
                        )
                        save_candidate = st.form_submit_button("保存项目修改")
                    if save_candidate:
                        editable_fields = {
                            "title": edit_title,
                            "period": edit_period,
                            "introduction": edit_introduction,
                            "tech_stack": _candidate_editor_list(
                                edit_stack, tech_stack=True
                            ),
                            "solutions": _candidate_editor_list(edit_solutions),
                            "results": _candidate_editor_list(edit_results),
                        }
                        try:
                            updated = _request(
                                "POST",
                                f"/api/tasks/{task_id}/resume",
                                json=_task_action(
                                    task,
                                    "edit",
                                    {
                                        "candidate_id": candidate["candidate_id"],
                                        "fields": editable_fields,
                                        "old_value_hashes": {
                                            field: _field_hash(candidate.get(field))
                                            for field in editable_fields
                                        },
                                    },
                                ),
                            )
                            st.session_state["wizard_task"] = updated
                            st.session_state["wizard_candidate_edit_saved"] = (
                                candidate.get("candidate_slot_id")
                                or candidate.get("candidate_id")
                            )
                            st.session_state[f"wizard-confirm-all-{task_id}"] = False
                            for field_name in _CANDIDATE_CONFIRMATION_FIELDS:
                                st.session_state[
                                    f"wizard-confirm-{task_id}-{field_name}"
                                ] = False
                            st.rerun()
                        except ApiRequestError as exc:
                            st.error(f"保存项目修改失败：{exc.message}")
                if node == "candidate_review" and st.button(
                    "选择这份纯文本" if project_only else "选择这个项目",
                    key=f"wizard-choose-candidate-{candidate['candidate_id']}",
                ):
                    updated = _request(
                        "POST",
                        f"/api/tasks/{task_id}/resume",
                        json=_task_action(
                            task, "select", {"candidate_id": candidate["candidate_id"]}
                        ),
                    )
                    st.session_state["wizard_task"] = updated
                    st.rerun()

    if node == "candidate_review" and status == "waiting_user":
        feedback = st.text_area("不满意时填写反馈并重新生成", key=f"wizard-feedback-{task_id}")
        try:
            feedback_models = _ready_model_profiles(_request("GET", "/api/models"), "chat")
        except Exception:
            feedback_models = []
        feedback_chat_profile = (
            st.selectbox(
                "本次重新生成使用的聊天模型",
                [item["profile_id"] for item in feedback_models],
                format_func=lambda value: next(
                    (
                        _model_profile_label(item)
                        for item in feedback_models
                        if item["profile_id"] == value
                    ),
                    value,
                ),
                key=f"wizard-feedback-model-{task_id}",
            )
            if feedback_models
            else None
        )
        if st.button(
            "按反馈重新生成",
            key=f"wizard-regenerate-{task_id}",
            disabled=not feedback.strip() or not feedback_chat_profile,
        ):
            updated = _request(
                "POST",
                f"/api/tasks/{task_id}/feedback",
                json={
                    "thread_id": task["thread_id"],
                    "checkpoint_version": task["checkpoint_version"],
                    "feedback": feedback,
                    "chat_profile_id": feedback_chat_profile,
                    "credential_handle_id": st.session_state.get(
                        f"credential-{feedback_chat_profile}"
                    ),
                },
            )
            st.session_state["wizard_task"] = updated
            st.rerun()

    if node == "candidate_confirmation" and status == "waiting_user":
        selected = _selected_candidate(task)
        if selected:
            if project_only:
                st.markdown("#### 可复制项目经历")
                plain_text = _candidate_plain_text(selected)
                from app.ui.clipboard_component import clipboard_text_button

                clipboard_text_button(
                    plain_text,
                    key=f"wizard-copy-direct-{task_id}",
                )
                st.code(plain_text, language=None)
            st.markdown("#### 核实项目字段")
            st.warning("模型推导内容均为待核实项。请确认真实、可解释后再继续。")
            master_key = f"wizard-confirm-all-{task_id}"
            field_prefix = f"wizard-confirm-{task_id}-"
            st.checkbox(
                "全部核实",
                value=False,
                key=master_key,
                on_change=_set_all_candidate_confirmations,
                kwargs={
                    "session_state": st.session_state,
                    "master_key": master_key,
                    "field_key_prefix": field_prefix,
                },
            )
            confirmations = []
            for field in _CANDIDATE_CONFIRMATION_FIELDS:
                if st.checkbox(
                    f"确认 {field}",
                    value=False,
                    key=f"{field_prefix}{field}",
                ):
                    confirmations.append(
                        {"field_id": field, "value_hash": _field_hash(selected[field]), "confirmed": True}
                    )
            if st.button(
                "核实并完成纯文本" if project_only else "确认以上项目内容",
                key=f"wizard-confirm-candidate-{task_id}",
                disabled=len(confirmations) != len(_CANDIDATE_CONFIRMATION_FIELDS),
            ):
                try:
                    updated = _request(
                        "POST",
                        f"/api/tasks/{task_id}/confirmations",
                        json={
                            **_task_action(task, "confirm"),
                            "candidate_id": selected["candidate_id"],
                            "confirm": True,
                            "confirmations": confirmations,
                        },
                    )
                    st.session_state["wizard_task"] = updated
                    st.rerun()
                except ApiRequestError as exc:
                    st.error(f"字段核实失败：{exc.message}")

    if node == "project_application" and status == "waiting_user":
        decision = task.get("project_decision") or {}
        requested = str(task.get("requested_output_mode") or "resume_edit")
        rankings = decision.get("project_rankings") or []
        recommended_action = str(decision.get("recommended_action") or "add")
        target_id = decision.get("target_project_id")
        if requested == "project_only":
            selected = _selected_candidate(task)
            if selected:
                st.markdown("#### 可复制项目经历")
                plain_text = _candidate_plain_text(selected)
                from app.ui.clipboard_component import clipboard_text_button

                clipboard_text_button(
                    plain_text,
                    key=f"wizard-copy-button-{task_id}",
                )
                st.code(plain_text, language=None)
            if st.button("生成可复制纯文本", key=f"wizard-copy-project-{task_id}"):
                updated = _request(
                    "POST",
                    f"/api/tasks/{task_id}/resume",
                    json=_task_action(
                        task,
                        "approve",
                        {"project_action": "copy_only", "final_product": "project_only"},
                    ),
                )
                st.session_state["wizard_task"] = updated
                st.rerun()
        elif requested == "template_resume":
            st.info("已选择模板，将把确认后的项目写入并生成完整简历。")
            if st.button("写入模板并生成完整简历", key=f"wizard-template-apply-{task_id}"):
                updated = _request(
                    "POST",
                    f"/api/tasks/{task_id}/resume",
                    json=_task_action(
                        task,
                        "approve",
                        {"project_action": "add", "final_product": "full_resume"},
                    ),
                )
                st.session_state["wizard_task"] = updated
                st.rerun()
        else:
            final_choice = st.radio(
                "希望如何修改已上传的简历？",
                ["project_patch_only", "full_resume"],
                format_func=lambda value: (
                    "只新增或替换项目，其他内容保持不变"
                    if value == "project_patch_only"
                    else "继续优化不匹配模块并生成完整简历"
                ),
                key=f"wizard-final-product-{task_id}",
            )
            available_actions = ["add", "replace"] if rankings else ["add"]
            default_action = (
                recommended_action if recommended_action in available_actions else "add"
            )
            action = st.radio(
                "项目处理方式",
                available_actions,
                index=available_actions.index(default_action),
                format_func=lambda value: "新增项目" if value == "add" else "替换现有项目",
                horizontal=True,
                key=f"wizard-project-action-{task_id}",
            )
            target_id = None
            if action == "replace":
                project_ids = [
                    str(item.get("project_id"))
                    for item in rankings
                    if item.get("project_id")
                ]
                project_labels = {
                    str(item.get("project_id")): (
                        f"{str(item.get('title') or item.get('project_id'))}"
                        f"（匹配度 {item.get('match_score', 0)}）"
                    )
                    for item in rankings
                    if item.get("project_id")
                }
                recommended_target = str(decision.get("target_project_id") or "")
                target_id = st.selectbox(
                    "选择要替换的项目（已默认选中系统建议项）",
                    project_ids,
                    index=(
                        project_ids.index(recommended_target)
                        if recommended_target in project_ids
                        else 0
                    ),
                    format_func=lambda value: project_labels.get(value, value),
                    key=f"wizard-target-project-{task_id}",
                ) if project_ids else None
                selected_meta = next(
                    (item for item in rankings if str(item.get("project_id")) == target_id),
                    {},
                )
                if selected_meta.get("content_preview"):
                    st.caption(f"将被替换的原项目：{selected_meta['content_preview']}")
            st.caption(decision.get("reason") or f"系统建议：{default_action}")
            if st.button(
                "确认修改方式",
                key=f"wizard-apply-project-{task_id}",
                disabled=action == "replace" and not target_id,
            ):
                updated = _request(
                    "POST",
                    f"/api/tasks/{task_id}/resume",
                    json=_task_action(
                        task,
                        "approve",
                        {
                            "project_action": action,
                            "target_project_id": target_id,
                            "final_product": final_choice,
                        },
                    ),
                )
                st.session_state["wizard_task"] = updated
                st.rerun()

    module_labels = {"summary": "个人简介", "skills": "技能清单", "evaluation": "个人评价"}
    if node in {f"{name}_review" for name in module_labels} and status == "waiting_user":
        module = node.removesuffix("_review")
        assessment = (task.get("module_assessment") or {}).get(module, {})
        st.markdown(f"#### 是否优化{module_labels[module]}？")
        st.write(f"当前匹配分：{assessment.get('score', 0)}")
        feedback = st.text_area("优化要求（可选）", key=f"wizard-module-feedback-{task_id}-{module}")
        actions = st.columns(2)
        if actions[0].button("生成优化建议", key=f"wizard-module-yes-{task_id}-{module}"):
            updated = _request(
                "POST",
                f"/api/tasks/{task_id}/resume",
                json=_task_action(task, "approve", {"feedback": feedback}),
            )
            st.session_state["wizard_task"] = updated
            st.rerun()
        if actions[1].button("保留原内容", key=f"wizard-module-no-{task_id}-{module}"):
            updated = _request(
                "POST", f"/api/tasks/{task_id}/resume", json=_task_action(task, "skip")
            )
            st.session_state["wizard_task"] = updated
            st.rerun()

    if node in {f"{name}_confirmation" for name in module_labels} and status == "waiting_user":
        module = node.removesuffix("_confirmation")
        draft = (task.get("module_drafts") or {}).get(module) or {}
        st.text_area(
            f"{module_labels[module]}建议",
            value=str(draft.get("new_value") or ""),
            height=160,
            disabled=True,
            key=f"wizard-module-draft-{task_id}-{module}",
        )
        feedback = st.text_area("修改反馈", key=f"wizard-module-revision-{task_id}-{module}")
        actions = st.columns(3)
        if actions[0].button("采用", key=f"wizard-module-confirm-{task_id}-{module}"):
            updated = _request(
                "POST",
                f"/api/tasks/{task_id}/resume",
                json=_task_action(task, "confirm", {"value_hash": draft.get("new_value_hash")}),
            )
            st.session_state["wizard_task"] = updated
            st.rerun()
        if actions[1].button(
            "按反馈重写",
            key=f"wizard-module-rewrite-{task_id}-{module}",
            disabled=not feedback.strip(),
        ):
            updated = _request(
                "POST",
                f"/api/tasks/{task_id}/resume",
                json=_task_action(task, "revise", {"feedback": feedback}),
            )
            st.session_state["wizard_task"] = updated
            st.rerun()
        if actions[2].button("拒绝", key=f"wizard-module-reject-{task_id}-{module}"):
            updated = _request(
                "POST", f"/api/tasks/{task_id}/resume", json=_task_action(task, "reject")
            )
            st.session_state["wizard_task"] = updated
            st.rerun()

    if node == "layout_review" and status == "waiting_user":
        pages = int((task.get("layout_decision") or {}).get("estimated_page_count") or 1)
        st.write(f"预计页数：{pages}")
        if st.button("继续生成预览", key=f"wizard-layout-{task_id}"):
            updated = _request(
                "POST",
                f"/api/tasks/{task_id}/resume",
                json=_task_action(task, "approve", {"page_count": pages}),
            )
            st.session_state["wizard_task"] = updated
            st.rerun()

    if node == "compression_review" and status == "waiting_user":
        st.caption("以下压缩只调整排版，不删除、缩短或改写任何简历文字。")
        actions = st.columns(3)
        if actions[0].button("常规无损压缩", key=f"wizard-compress-{task_id}"):
            try:
                updated = _request(
                    "POST",
                    f"/api/tasks/{task_id}/resume",
                    json=_task_action(
                        task,
                        "approve",
                        {
                            "accept_compression": True,
                            "allow_rewrite": False,
                            "compression_level": "standard",
                        },
                    ),
                )
                st.session_state["wizard_task"] = updated
                st.rerun()
            except ApiRequestError as exc:
                st.error(f"压缩设置未能应用：{exc.message}")
        if actions[1].button("紧凑重排压缩", key=f"wizard-compact-{task_id}"):
            try:
                updated = _request(
                    "POST",
                    f"/api/tasks/{task_id}/resume",
                    json=_task_action(
                        task,
                        "approve",
                        {
                            "accept_compression": True,
                            "allow_rewrite": False,
                            "compression_level": "aggressive",
                        },
                    ),
                )
                st.session_state["wizard_task"] = updated
                st.rerun()
            except ApiRequestError as exc:
                st.error(f"紧凑重排未能应用：{exc.message}")
        if actions[2].button("保持多页", key=f"wizard-keep-pages-{task_id}"):
            try:
                updated = _request(
                    "POST", f"/api/tasks/{task_id}/resume", json=_task_action(task, "skip")
                )
                st.session_state["wizard_task"] = updated
                st.rerun()
            except ApiRequestError as exc:
                st.error(f"分页设置未能应用：{exc.message}")

    if node == "preview_review" and status == "waiting_user":
        requested = str(task.get("requested_output_mode") or "resume_edit")
        preview_format = "markdown" if requested == "project_only" else st.selectbox(
            "预览格式", ["markdown", "docx", "pdf"], key=f"wizard-preview-format-{task_id}"
        )
        if st.button("生成预览", key=f"wizard-preview-{task_id}"):
            preview = _request(
                "POST",
                f"/api/tasks/{task_id}/export-preview",
                json={
                    "thread_id": task["thread_id"],
                    "checkpoint_version": task["checkpoint_version"],
                    "snapshot_id": task["resume_snapshot_id"],
                    "format": preview_format,
                    "accept_compression": bool((task.get("compression_decision") or {}).get("accepted")),
                    "allow_rewrite": bool((task.get("compression_decision") or {}).get("allow_rewrite")),
                    "compression_level": str((task.get("compression_decision") or {}).get("level") or "standard"),
                },
            )
            st.session_state["wizard_preview"] = preview
        preview = st.session_state.get("wizard_preview")
        if _preview_matches_task(preview, task, preview_format):
            _render_export_result(
                st,
                preview,
                key_prefix=f"wizard-preview-{task_id}-{preview_format}",
                content_label="预览内容",
            )
            overflow = str(preview.get("compliance_status") or "") == "overflow"
            confirm_label = "接受当前多页结果" if overflow else "确认最终结果"
            if st.button(confirm_label, key=f"wizard-final-confirm-{task_id}"):
                updated = _request(
                    "POST", f"/api/tasks/{task_id}/resume", json=_task_action(task, "confirm")
                )
                st.session_state["wizard_task"] = updated
                st.rerun()
            compression_level = str(
                (task.get("compression_decision") or {}).get("level") or "standard"
            )
            if overflow and compression_level != "aggressive" and st.button(
                "继续进行更强的无损压缩",
                key=f"wizard-preview-compact-{task_id}",
            ):
                try:
                    updated = _request(
                        "POST",
                        f"/api/tasks/{task_id}/resume",
                        json=_task_action(
                            task,
                            "retry",
                            {"compression_level": "aggressive"},
                        ),
                    )
                    st.session_state["wizard_task"] = updated
                    st.session_state.pop("wizard_preview", None)
                    st.rerun()
                except ApiRequestError as exc:
                    st.error(f"无损压缩请求失败：{exc.message}")

    if status == "completed":
        st.success("生成已完成")
        requested = str(task.get("requested_output_mode") or "resume_edit")
        if requested == "project_only":
            selected = _selected_candidate(task)
            if selected:
                plain_text = _candidate_plain_text(selected)
                st.markdown("#### 可复制项目经历")
                from app.ui.clipboard_component import clipboard_text_button

                clipboard_text_button(
                    plain_text,
                    key=f"wizard-copy-completed-{task_id}",
                )
                st.code(plain_text, language=None)
                st.download_button(
                    "下载纯文本",
                    data=plain_text,
                    file_name="project.txt",
                    mime="text/plain",
                    key=f"wizard-download-text-{task_id}",
                )
            return
        export_format = "markdown" if requested == "project_only" else st.selectbox(
            "导出格式", ["markdown", "docx", "pdf"], key=f"wizard-export-format-{task_id}"
        )
        if st.button("生成最终文件", key=f"wizard-export-{task_id}"):
            exported = _request(
                "POST",
                f"/api/tasks/{task_id}/export",
                json={
                    "thread_id": task["thread_id"],
                    "checkpoint_version": task["checkpoint_version"],
                    "snapshot_id": task["resume_snapshot_id"],
                    "format": export_format,
                    "accept_compression": bool((task.get("compression_decision") or {}).get("accepted")),
                    "allow_rewrite": bool((task.get("compression_decision") or {}).get("allow_rewrite")),
                    "compression_level": str((task.get("compression_decision") or {}).get("level") or "standard"),
                },
            )
            st.session_state["wizard_final_preview"] = exported
        final_export = st.session_state.get("wizard_final_preview")
        if _preview_matches_task(final_export, task, export_format) and final_export.get("content"):
            _render_export_result(
                st,
                final_export,
                key_prefix=f"wizard-final-{task_id}-{export_format}",
                content_label="可复制内容" if requested == "project_only" else "最终内容",
            )


def _reset_wizard(st: Any) -> None:
    next_epoch = int(st.session_state.get("_resume_wizard_epoch", 0)) + 1
    prefixes = ("wizard_",)
    exact = {"resume", "selected_job_id", "job_detail_view", "task_id", "task"}
    for key in list(st.session_state):
        if key in exact or any(str(key).startswith(prefix) for prefix in prefixes):
            del st.session_state[key]
    # Give the first radio a new frontend identity so Streamlit cannot restore
    # the answer from before "重新开始" on the next rerun.
    st.session_state["_resume_wizard_epoch"] = next_epoch


def _wizard_transition(
    st: Any,
    marker: str,
    value: Any,
    *,
    clear: tuple[str, ...],
) -> None:
    """Discard downstream answers when an earlier wizard choice changes."""

    previous = st.session_state.get(marker)
    if previous is not None and previous != value:
        for key in clear:
            st.session_state.pop(key, None)
    st.session_state[marker] = value


def _sync_wizard_job_state(st: Any, job: dict[str, Any]) -> None:
    """Write detail fields back without replacing a card's HR/activity pair."""

    def merge_with_card(card: dict[str, Any]) -> dict[str, Any]:
        merged = {**card, **job}
        # The search card binds these two fields to this exact job. A detail
        # pane can briefly contain the previous card after an asynchronous
        # click, so it may only fill missing card metadata, never replace a
        # concrete card value.
        card_activity = str(card.get("hr_activity") or "").strip()
        if card_activity not in {"", "活跃时间待解析"}:
            merged["hr_activity"] = card.get("hr_activity")
        if str(card.get("hr_name") or "").strip():
            merged["hr_name"] = card.get("hr_name")
        return merged

    selected = st.session_state.get("wizard_selected_job")
    if isinstance(selected, dict) and (
        selected.get("job_id") == job.get("job_id")
        or selected.get("source_url") == job.get("source_url")
    ):
        st.session_state["wizard_selected_job"] = merge_with_card(selected)
    # Detail parsing may fill a value only when the card genuinely reported
    # “待解析”. Keep both list views in sync without cross-card contamination.
    for state_key in ("wizard_job_search", "job_search"):
        search = st.session_state.get(state_key)
        if not isinstance(search, dict):
            continue
        jobs = list(search.get("jobs") or [])
        changed = False
        merged_jobs: list[dict[str, Any]] = []
        for item in jobs:
            if isinstance(item, dict) and (
                item.get("job_id") == job.get("job_id")
                or item.get("source_url") == job.get("source_url")
            ):
                merged_jobs.append(merge_with_card(item))
                changed = True
            else:
                merged_jobs.append(item)
        if changed:
            st.session_state[state_key] = {**search, "jobs": merged_jobs}


def _set_wizard_job_detail(st: Any, detail: dict[str, Any]) -> None:
    """Bind one JD and invalidate output that belonged to a previous JD."""

    st.session_state["wizard_job_detail"] = detail
    st.session_state.pop("wizard_job_detail_confirmed", None)
    st.session_state.pop("wizard_screenshot_fallback_accepted", None)
    job = detail.get("job") if isinstance(detail, dict) else None
    if isinstance(job, dict):
        job_id = str(job.get("job_id") or detail.get("job_id") or "")
        for prefix in (
            "wizard_edit_responsibilities_",
            "wizard_edit_requirements_",
            "wizard_edit_skills_",
            "wizard_edit_raw_text_",
        ):
            st.session_state.pop(f"{prefix}{job_id}", None)
        _sync_wizard_job_state(st, job)
    st.session_state.pop("wizard_task", None)
    st.session_state.pop("wizard_preview", None)


def _recognize_wizard_screenshot(st: Any, uploaded: Any) -> None:
    """Submit one pasted/uploaded screenshot and stage editable OCR text."""

    try:
        data = uploaded.getvalue()
        result = _request(
            "POST",
            "/api/jobs/screenshot/ocr",
            files={
                "file": (
                    getattr(uploaded, "name", None) or "clipboard.png",
                    data,
                    getattr(uploaded, "type", None) or "image/png",
                )
            },
            timeout=180,
        )
        st.session_state["wizard_screenshot_ocr"] = result
        st.session_state["wizard_screenshot_preview"] = data
        st.session_state["wizard_ocr_corrected_text"] = str(result.get("text") or "")
        for key in (
            "wizard_ocr_title",
            "wizard_ocr_company",
            "wizard_ocr_city",
            "wizard_ocr_salary",
            "wizard_ocr_hr_activity",
        ):
            st.session_state.pop(key, None)
        st.session_state.pop("wizard_job_detail", None)
        st.session_state.pop("wizard_task", None)
        st.session_state.pop("wizard_preview", None)
        st.rerun()
    except ApiRequestError as exc:
        st.error(f"截图识别失败：{exc.message}")
    except Exception as exc:
        st.error(f"截图识别失败：{exc}")


def _render_screenshot_source(st: Any, chat_profile_id: str | None = None) -> bool:
    """Render the paste/upload -> edit -> explicit confirmation OCR flow."""

    detail = st.session_state.get("wizard_job_detail") or {}
    if detail.get("source") == "screenshot_ocr_confirmed" and isinstance(detail.get("job"), dict):
        job = detail["job"]
        if not detail.get("model_fallback"):
            st.success("岗位截图已经校正确认并解析完成。")
            return True

        st.warning(
            detail.get("model_fallback_message")
            or "大模型暂时无法解析，已回退到 OCR 校正文字和规则解析结果。"
        )
        preview = st.session_state.get("wizard_screenshot_preview")
        if preview:
            st.image(preview, caption="岗位原始截图", width="stretch")
        corrected_text = st.text_area(
            "OCR 校正文字（可继续修改后重试）",
            value=str(job.get("raw_text") or st.session_state.get("wizard_ocr_corrected_text") or ""),
            height=320,
            key=f"wizard_ocr_retry_text_{job.get('job_id') or detail.get('job_id')}",
        )
        with st.expander("当前规则解析结果", expanded=True):
            st.markdown("**岗位职责**")
            st.write("\n".join(f"- {value}" for value in job.get("responsibilities") or []) or "未识别")
            st.markdown("**任职要求**")
            st.write("\n".join(f"- {value}" for value in job.get("requirements") or []) or "未识别")
            st.markdown("**技能要求**")
            st.write("、".join(job.get("skills") or []) or "未识别")

        job_id = str(job.get("job_id") or detail.get("job_id") or "")
        if st.session_state.get("wizard_screenshot_fallback_accepted") == job_id:
            st.success("已接受 OCR/规则结果，下面可继续逐项校正并最终确认。")
            return True
        actions = st.columns(2)
        if actions[0].button(
            "重新使用当前模型解析",
            key=f"wizard_retry_screenshot_model_{job_id}",
            disabled=not bool(chat_profile_id),
            use_container_width=True,
        ):
            if not corrected_text.strip():
                st.warning("OCR 校正文字不能为空")
            else:
                try:
                    reparsed = _request(
                        "POST",
                        f"/api/jobs/{job_id}/reparse",
                        timeout=240,
                        json={
                            "raw_text": corrected_text,
                            "chat_profile_id": chat_profile_id,
                            "credential_handle_id": st.session_state.get(
                                f"credential-{chat_profile_id}"
                            ),
                        },
                    )
                    parser = reparsed.get("parser") if isinstance(reparsed.get("parser"), dict) else {}
                    fallback = parser.get("parser_mode") != "model_enhanced"
                    _set_wizard_job_detail(
                        st,
                        {
                            **reparsed,
                            "source": "screenshot_ocr_confirmed",
                            "model_fallback": fallback,
                            "model_fallback_message": (
                                (parser.get("warnings") or ["所选模型仍不可用，已继续保留规则解析结果"])[0]
                                if fallback
                                else None
                            ),
                        },
                    )
                    st.rerun()
                except ApiRequestError as exc:
                    st.error(f"模型重新解析失败：{exc.message}")
        if actions[1].button(
            "接受 OCR/规则结果继续",
            key=f"wizard_accept_screenshot_fallback_{job_id}",
            use_container_width=True,
        ):
            st.session_state["wizard_screenshot_fallback_accepted"] = job_id
            st.rerun()
        if not chat_profile_id:
            st.info("请先在第一栏配置并检测一个聊天模型，再尝试模型增强解析。")
        return False
    st.caption(
        "可在下方输入框聚焦后直接按 Ctrl+V 粘贴剪贴板截图，也可以选择本地图片；"
        "支持 PNG、JPG/JPEG、WEBP、BMP，最大 10 MB。"
    )
    with st.container(border=True):
        from app.ui.clipboard_component import clipboard_image_input

        pasted = clipboard_image_input(key="wizard_screenshot_paste")
        paste_id = str((pasted or {}).get("paste_id") or "")
        if paste_id and paste_id != st.session_state.get("wizard_last_screenshot_paste_id"):
            st.session_state["wizard_last_screenshot_paste_id"] = paste_id
            data_url = str(pasted.get("data_url") or "")
            try:
                header, encoded = data_url.split(",", 1)
                if not header.startswith("data:image/"):
                    raise ValueError("not an image data URL")
                data = base64.b64decode(encoded, validate=True)
                if len(data) > 10 * 1024 * 1024:
                    raise ValueError("image is too large")
                clipboard_file = io.BytesIO(data)
                clipboard_file.name = str(pasted.get("name") or "clipboard.png")
                clipboard_file.type = str(pasted.get("type") or "image/png")
                _recognize_wizard_screenshot(st, clipboard_file)
            except (ValueError, binascii.Error):
                st.error("剪贴板截图数据无效或超过 10 MB，请重新粘贴或改用图片上传")

        uploaded = st.file_uploader(
            "从本地选择岗位截图",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key="wizard_screenshot_file",
        )
        if st.button(
            "使用 PaddleOCR 识别图片",
            key="wizard_screenshot_ocr_button",
            disabled=uploaded is None,
        ):
            _recognize_wizard_screenshot(st, uploaded)

    ocr = st.session_state.get("wizard_screenshot_ocr") or {}
    if not ocr.get("ocr_id"):
        st.info("截图识别完成后会先显示 OCR 原文，校正并确认前不会创建岗位或启动生成。")
        return False

    preview = st.session_state.get("wizard_screenshot_preview")
    if preview:
        st.image(preview, caption="待确认的岗位截图", width="stretch")
    st.success(
        f"已使用 {ocr.get('model') or 'PP-OCRv6 medium'} 完成识别，"
        f"共识别 {len(ocr.get('lines') or [])} 行，请校正后确认。"
    )
    corrected_text = st.text_area(
        "OCR 识别文字（可直接校正）",
        height=320,
        key="wizard_ocr_corrected_text",
    )
    inferred = ocr.get("inferred") if isinstance(ocr.get("inferred"), dict) else {}
    visible_metadata = [str(value) for value in inferred.values() if value]
    if visible_metadata:
        st.caption("已从截图初步识别：" + " ｜ ".join(visible_metadata))
    if st.button("确认校正并进行初步及模型解析", key="wizard_confirm_screenshot"):
        if not corrected_text.strip():
            st.warning("校正后的岗位文字不能为空")
        else:
            try:
                detail = _request(
                    "POST",
                    "/api/jobs/screenshot/confirm",
                    timeout=240,
                    json={
                        "ocr_id": ocr["ocr_id"],
                        "text": corrected_text,
                        "chat_profile_id": chat_profile_id,
                        "credential_handle_id": st.session_state.get(
                            f"credential-{chat_profile_id}"
                        ),
                    },
                )
                _set_wizard_job_detail(st, detail)
                # Keep OCR text/image in this page session as a retry baseline.
                # The backend job now owns the corrected text, so changing the
                # model and reparsing never requires another OCR upload.
                st.session_state["wizard_screenshot_confirmed"] = True
                st.rerun()
            except ApiRequestError as exc:
                st.warning(
                    f"进一步解析未完成：{exc.message}。"
                    "OCR 图片、校正文字和初步结果仍保留在当前页面，可直接修改或重试，无需重新上传。"
                )
    return False


def _render_resume_wizard(st: Any) -> None:
    header = st.columns([4, 1])
    header[0].subheader("简历项目生成向导")
    if header[1].button("重新开始", key="wizard-reset"):
        close_warning: str | None = None
        try:
            _request("POST", "/api/jobs/browser/close", timeout=30)
        except ApiRequestError as exc:
            close_warning = f"岗位采集浏览器未能自动关闭：{exc.message}"
        except Exception as exc:
            close_warning = f"岗位采集浏览器未能自动关闭：{exc}"
        _reset_wizard(st)
        if close_warning:
            st.session_state["wizard_reset_warning"] = close_warning
        st.rerun()

    reset_warning = st.session_state.pop("wizard_reset_warning", None)
    if reset_warning:
        st.warning(reset_warning)

    st.markdown("### 第 1 步：是否上传已有简历？")
    wizard_epoch = int(st.session_state.get("_resume_wizard_epoch", 0))
    has_resume = st.radio(
        "请选择",
        ["yes", "no"],
        index=None,
        format_func=lambda value: "是，上传并修改已有简历" if value == "yes" else "否",
        key=f"wizard_has_resume_{wizard_epoch}",
    )
    if has_resume is None:
        st.info("请先回答这个问题，系统再显示下一步。")
        return
    _wizard_transition(
        st,
        "wizard_has_resume_marker",
        has_resume,
        clear=(
            "wizard_resume",
            "wizard_resume_source_bytes",
            "wizard_resume_source_name",
            "wizard_resume_parser",
            "wizard_no_resume_choice",
            "wizard_job_source",
            "wizard_job_detail",
            "wizard_selected_job",
            "wizard_job_search",
            "wizard_screenshot_ocr",
            "wizard_screenshot_preview",
            "wizard_task",
            "wizard_preview",
        ),
    )

    requested_output: str
    resume_id: str | None = None
    template_id: str | None = None
    base_facts: dict[str, Any] = {}
    if has_resume == "yes":
        try:
            resume_models = _ready_model_profiles(_request("GET", "/api/models"), "chat")
        except Exception:
            resume_models = []
        resume_chat_profile_id = (
            st.selectbox(
                "简历结构化拆解模型",
                [item["profile_id"] for item in resume_models],
                index=next(
                    (
                        index
                        for index, item in enumerate(resume_models)
                        if item.get("default_for_role")
                    ),
                    0,
                ),
                format_func=lambda value: next(
                    (
                        _model_profile_label(item)
                        for item in resume_models
                        if item["profile_id"] == value
                    ),
                    value,
                ),
                key="wizard_resume_chat_profile",
                help="先提取原始文字，再由所选云端或本地模型按区块拆解；模型不得补造原文外内容。",
            )
            if resume_models
            else None
        )
        if not resume_models:
            st.warning("没有检测成功的聊天模型。可先上传并保留规则拆解结果，配置模型后无需重新上传即可重试。")
        uploaded = st.file_uploader(
            "上传主简历（DOCX、文本型 PDF、Markdown 或 TXT，最多 10 MB）",
            type=["docx", "pdf", "md", "markdown", "txt"],
            key="wizard_resume_file",
        )
        if st.button(
            "上传并使用所选模型拆解简历",
            key="wizard_upload_resume",
            disabled=uploaded is None,
        ):
            try:
                source_bytes = uploaded.getvalue()
                parsed = _request(
                    "POST",
                    "/api/resumes/upload",
                    files={
                        "file": (
                            uploaded.name,
                            source_bytes,
                            uploaded.type or "application/octet-stream",
                        )
                    },
                )
                st.session_state["wizard_resume_source_bytes"] = source_bytes
                st.session_state["wizard_resume_source_name"] = uploaded.name
                model_result: dict[str, Any] = {
                    "status": "fallback",
                    "resume": parsed,
                    "parser": {
                        "parser_mode": "rule_fallback",
                        "error_code": "chat_profile_not_found",
                        "warnings": ["未选择已检测成功的聊天模型，当前显示规则拆解结果"],
                    },
                }
                if resume_chat_profile_id:
                    try:
                        model_result = _request(
                            "POST",
                            f"/api/resumes/{parsed['resume_id']}/model-parse",
                            timeout=180,
                            json={
                                "chat_profile_id": resume_chat_profile_id,
                                "credential_handle_id": st.session_state.get(
                                    f"credential-{resume_chat_profile_id}"
                                ),
                            },
                        )
                    except ApiRequestError as exc:
                        model_result["parser"] = {
                            "parser_mode": "rule_fallback",
                            "error_code": exc.code,
                            "warnings": [f"模型拆解请求失败：{exc.message}；已保留规则结果"],
                        }
                st.session_state["wizard_resume"] = model_result.get("resume") or parsed
                st.session_state["wizard_resume_parser"] = model_result.get("parser") or {}
                st.session_state.pop("wizard_task", None)
                st.session_state.pop("wizard_preview", None)
                st.rerun()
            except ApiRequestError as exc:
                st.error(exc.message)
        resume = st.session_state.get("wizard_resume") or {}
        resume_id = str(resume.get("resume_id") or "") or None
        if not resume_id:
            st.info("上传并解析成功后才能继续。")
            return
        st.success(f"简历已解析：{resume.get('filename')} · {resume.get('page_count') or '?'} 页")
        parser = st.session_state.get("wizard_resume_parser") or {}
        if parser.get("parser_mode") == "model_structured":
            st.success(
                f"已由 {_model_profile_label(next((item for item in resume_models if item.get('profile_id') == parser.get('model_profile_id')), {'model_name': parser.get('model_name'), 'provider': 'openai_compatible'}))} 完成结构化拆解"
            )
            for warning in parser.get("warnings") or []:
                st.info(str(warning))
        elif parser:
            warning = (parser.get("warnings") or ["模型拆解未完成，当前保留规则结果"])[0]
            st.warning(str(warning))
            if parser.get("error_code") == "needs_external_model_consent":
                st.info("请先在第一栏确认外部模型数据发送范围，再使用当前已上传简历重试。")

        retry_label = "改用所选模型重新拆解" if parser.get("parser_mode") == "model_structured" else "使用所选模型重试拆解"
        if st.button(
            retry_label,
            key=f"wizard_retry_resume_model_{resume_id}",
            disabled=not bool(resume_chat_profile_id),
        ):
            try:
                model_result = _request(
                    "POST",
                    f"/api/resumes/{resume_id}/model-parse",
                    timeout=180,
                    json={
                        "chat_profile_id": resume_chat_profile_id,
                        "credential_handle_id": st.session_state.get(
                            f"credential-{resume_chat_profile_id}"
                        ),
                    },
                )
                st.session_state["wizard_resume"] = model_result.get("resume") or resume
                st.session_state["wizard_resume_parser"] = model_result.get("parser") or {}
                st.rerun()
            except ApiRequestError as exc:
                st.error(f"简历模型拆解失败：{exc.message}")

        source_name = str(st.session_state.get("wizard_resume_source_name") or resume.get("filename") or "")
        source_bytes = st.session_state.get("wizard_resume_source_bytes")
        if source_name.lower().endswith(".pdf") and not isinstance(source_bytes, bytes):
            try:
                source_bytes = _request_bytes("GET", f"/api/resumes/{resume_id}/source")
                st.session_state["wizard_resume_source_bytes"] = source_bytes
                st.session_state["wizard_resume_source_name"] = source_name
            except ApiRequestError as exc:
                st.warning(f"原始 PDF 暂时无法载入：{exc.message}")
        if source_name.lower().endswith(".pdf") and isinstance(source_bytes, bytes):
            _render_original_pdf(st, source_bytes, source_name)
        confirmed_resume = _render_resume_parse_confirmation(st, resume)
        if confirmed_resume is None:
            return
        resume = confirmed_resume
        resume_id = str(resume.get("resume_id") or "") or None
        requested_output = "resume_edit"
    else:
        no_resume_choice = st.radio(
            "未上传简历，希望怎样输出？",
            ["template_resume", "project_only"],
            index=None,
            format_func=lambda value: (
                "使用模板生成完整简历" if value == "template_resume" else "只生成可复制的项目经历纯文本"
            ),
            key="wizard_no_resume_choice",
        )
        if no_resume_choice is None:
            st.info("请选择模板简历或纯文本项目，随后再选择 JD 来源。")
            return
        _wizard_transition(
            st,
            "wizard_no_resume_choice_marker",
            no_resume_choice,
            clear=("wizard_task", "wizard_preview"),
        )
        requested_output = no_resume_choice
        if requested_output == "template_resume":
            template_id = _render_wizard_template_picker(st)
            if not template_id:
                return
        else:
            st.success("已选择纯文本模式：不需要填写姓名、电话或上传简历。")

    st.markdown("### 第 2 步：从哪里获取岗位 JD？")
    source = st.radio(
        "请选择一种来源",
        ["screenshot", "website", "search"],
        index=None,
        format_func=lambda value: {
            "screenshot": "粘贴或上传岗位截图",
            "website": "输入公司招聘官网网址（预留）",
            "search": "输入岗位名称和城市自动搜索",
        }[value],
        key="wizard_job_source",
    )
    if source is None:
        return
    _wizard_transition(
        st,
        "wizard_job_source_marker",
        source,
        clear=(
            "wizard_job_detail",
            "wizard_selected_job",
            "wizard_job_search",
            "wizard_job_page",
            "wizard_screenshot_ocr",
            "wizard_screenshot_preview",
            "wizard_task",
            "wizard_preview",
        ),
    )

    try:
        jd_models = _ready_model_profiles(_request("GET", "/api/models"), "chat")
    except Exception:
        jd_models = []
    jd_chat_profile_id = (
        st.selectbox(
            "JD 职责解析模型",
            [item["profile_id"] for item in jd_models],
            index=next(
                (
                    index
                    for index, item in enumerate(jd_models)
                    if item.get("default_for_role")
                ),
                0,
            ),
            format_func=lambda value: next(
                (
                    _model_profile_label(item)
                    for item in jd_models
                    if item["profile_id"] == value
                ),
                value,
            ),
            key="wizard_jd_chat_profile",
            help="先用规则提取，再由所选聊天模型拆分职责、任职要求和技能；模型不可用时自动保留规则结果。",
        )
        if jd_models
        else None
    )
    if not jd_models:
        st.warning("尚无可用的聊天模型配置，JD 将暂时使用规则解析。请先在第一栏配置并检测模型。")

    if source == "screenshot":
        screenshot_ready = _render_screenshot_source(st, jd_chat_profile_id)
        if not screenshot_ready:
            return
    elif source == "website":
        st.text_input(
            "公司招聘官网网址",
            key="wizard_company_career_url",
            placeholder="https://careers.example.com/jobs/...",
        )
        st.info("不同公司招聘官网结构差异较大，第一版仅预留该入口，暂不访问或解析网址。")
        return
    else:
        _render_job_search(st, jd_chat_profile_id)

    detail = st.session_state.get("wizard_job_detail") or {}
    if not isinstance(detail.get("job"), dict):
        return
    edited_job = _show_job_detail(st, detail) or {}

    job_id = str((detail.get("job") or {}).get("job_id") or detail.get("job_id") or "")
    if st.session_state.get("wizard_job_detail_confirmed") != job_id:
        st.markdown("#### 确认岗位解析结果")
        st.info("可直接编辑上方字段并保存；留言只指导模型如何分点和归类，不会被写入 JD 原文。")
        feedback = st.text_area(
            "给解析模型的留言（可选）",
            key=f"wizard_job_parse_feedback_{job_id}",
            placeholder="例如：保持原编号边界；把第二条完整归入任职要求，不要拆成短语。",
        )
        review_actions = st.columns(3)
        if review_actions[0].button(
            "保存人工校正",
            key=f"wizard_save_job_correction_{job_id}",
            use_container_width=True,
        ):
            try:
                corrected = _request(
                    "POST",
                    f"/api/jobs/{job_id}/corrections",
                    json=edited_job,
                )
                if corrected.get("status") != "ready":
                    st.error("岗位校正保存失败，请重试")
                else:
                    _set_wizard_job_detail(
                        st,
                        {**corrected, "source": detail.get("source") or "human_correction"},
                    )
                    st.rerun()
            except ApiRequestError as exc:
                st.error(exc.message)
        model_parse_failed = _job_model_parse_failed(detail)
        if review_actions[1].button(
            "再次使用大模型解析" if model_parse_failed else "结合留言重新解析",
            key=f"wizard_reparse_job_{job_id}",
            use_container_width=True,
            disabled=not bool(jd_chat_profile_id),
        ):
            try:
                reparsed = _request(
                    "POST",
                    f"/api/jobs/{job_id}/reparse",
                    timeout=240,
                    json={
                        **edited_job,
                        "feedback": feedback,
                        "chat_profile_id": jd_chat_profile_id,
                        "credential_handle_id": st.session_state.get(
                            f"credential-{jd_chat_profile_id}"
                        ),
                    },
                )
                if reparsed.get("status") == "ready":
                    _set_wizard_job_detail(
                        st,
                        {**reparsed, "source": detail.get("source") or "user_reparse"},
                    )
                    st.rerun()
                st.warning(reparsed.get("message") or "岗位重新解析暂未完成")
            except ApiRequestError as exc:
                st.error(exc.message)
        if review_actions[2].button(
            "保存并确认",
            key=f"wizard_confirm_job_parse_{job_id}",
            use_container_width=True,
        ):
            try:
                corrected = _request(
                    "POST",
                    f"/api/jobs/{job_id}/corrections",
                    json=edited_job,
                )
                if corrected.get("status") != "ready":
                    st.error("岗位校正保存失败，请重试")
                else:
                    confirmed = _request(
                        "POST",
                        f"/api/jobs/{job_id}/confirm-parse",
                    )
                    if confirmed.get("status") != "confirmed":
                        st.error("岗位解析结果确认失败，请重试")
                    else:
                        st.session_state["wizard_job_detail"] = {
                            **corrected,
                            "source": detail.get("source") or "human_correction",
                        }
                        _sync_wizard_job_state(st, corrected.get("job") or {})
                        st.session_state["wizard_job_detail_confirmed"] = job_id
                        st.rerun()
            except ApiRequestError as exc:
                st.error(exc.message)
        return
    st.success("岗位职责与 JD 解析结果已经确认。")
    if st.button("继续校正该岗位", key=f"wizard_reopen_job_parse_{job_id}"):
        st.session_state.pop("wizard_job_detail_confirmed", None)
        st.rerun()

    st.markdown("### 第 3 步：是否根据该岗位生成项目经历？")
    generate = st.radio(
        "请选择",
        ["yes", "no"],
        index=None,
        format_func=lambda value: "是，生成项目候选" if value == "yes" else "否，暂不生成",
        key="wizard_generate_project",
    )
    if generate != "yes":
        if generate == "no":
            st.info("已保留解析后的 JD，不会启动生成流程。")
        return

    if requested_output == "template_resume":
        st.markdown("### 第 4 步：填写模板简历的基础资料")
        base_cols = st.columns(2)
        base_facts = {
            "name": base_cols[0].text_input("姓名", key="wizard_base_name"),
            "phone": base_cols[1].text_input("电话", key="wizard_base_phone"),
            "email": base_cols[0].text_input("邮箱", key="wizard_base_email"),
            "objective": base_cols[1].text_input(
                "求职目标",
                value=str(detail["job"].get("title") or ""),
                key="wizard_base_objective",
            ),
            "location": base_cols[0].text_input("所在地（可选）", key="wizard_base_location"),
            "certificates": base_cols[1].text_input("证书（可选）", key="wizard_base_certificates"),
            "education": st.text_area("教育经历（可选）", key="wizard_base_education"),
        }

    st.markdown("### 第 4 步：选择模型并开始生成" if requested_output != "template_resume" else "### 第 5 步：选择模型并开始生成")
    try:
        models = _request("GET", "/api/models")
    except Exception:
        models = []
    chat_models = _ready_model_profiles(models, "chat")
    embedding_models = _ready_model_profiles(models, "embedding")
    chat_profile_id = st.selectbox(
        "聊天模型",
        [item["profile_id"] for item in chat_models],
        format_func=lambda value: next(
            (_model_profile_label(item) for item in chat_models if item["profile_id"] == value), value
        ),
        key="wizard_chat_profile",
    ) if chat_models else None
    embedding_profile_id = st.selectbox(
        "Embedding 模型",
        [item["profile_id"] for item in embedding_models],
        format_func=lambda value: next(
            (_model_profile_label(item) for item in embedding_models if item["profile_id"] == value), value
        ),
        key="wizard_embedding_profile",
    ) if embedding_models else None
    candidate_count = st.slider(
        "候选项目数量", 1, 5, 3, key="wizard_candidate_count"
    )
    if requested_output == "project_only":
        st.caption("每份候选均可复制；选择并核实后直接完成，不进入项目替换流程。")
    task = st.session_state.get("wizard_task")
    if not task and st.button("开始生成项目", key="wizard_start_task"):
        try:
            created = _request(
                "POST",
                "/api/tasks",
                json={
                    "job_id": detail["job"]["job_id"],
                    "resume_id": resume_id,
                    "template_id": template_id,
                    "output_mode": requested_output,
                    "base_facts": base_facts,
                    "candidate_count": candidate_count,
                    "chat_profile_id": chat_profile_id,
                    "embedding_profile_id": embedding_profile_id,
                    "credential_handle_id": st.session_state.get(f"credential-{chat_profile_id}"),
                    "embedding_credential_handle_id": st.session_state.get(
                        f"credential-{embedding_profile_id}"
                    ),
                },
            )
            st.session_state["wizard_task"] = created
            st.rerun()
        except ApiRequestError as exc:
            st.error(exc.message)
    task = st.session_state.get("wizard_task")
    if isinstance(task, dict):
        _render_task_progress(st, task)


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("请安装 UI 依赖：pip install -e .[ui]") from exc

    st.set_page_config(page_title="Resume Agent", layout="wide")
    st.title("Resume Agent")
    st.caption("根据岗位 JD 逐步生成项目经历或针对性简历")
    tabs = st.tabs(["环境与模型", "简历项目向导"])
    with tabs[0]:
        _render_compact_environment(st)
    with tabs[1]:
        _render_resume_wizard(st)


if __name__ == "__main__":
    main()
