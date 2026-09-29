"""Minimal Streamlit client.  It deliberately delegates all business work to FastAPI."""
from __future__ import annotations

import base64
import binascii
import io
import os
import hashlib
import json
from pathlib import Path
from typing import Any


_STATUS_LABELS = {
    "auth_failed": "认证失败",
    "capability_mismatch": "模型能力不匹配",
    "credential_missing": "缺少访问凭据",
    "edge_environment_failed": "浏览器环境读取失败",
    "edge_not_found": "未找到 Microsoft Edge",
    "executable_missing": "未找到 Ollama 程序",
    "mcp_process_exited": "检测服务已退出",
    "model_not_installed": "模型未安装",
    "needs_setup": "环境尚未配置完成",
    "in_memory": "本地存储正常",
    "degraded": "部分服务不可用",
    "playwright_not_installed": "未安装 Playwright",
    "probe_failed": "能力检测失败",
    "ready": "正常",
    "service_unreachable": "服务无法连接",
    "unavailable": "服务不可用",
    "unprobed": "未检测",
}


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
        response = httpx.request(method, f"{base}{path}", headers=headers, timeout=timeout, **kwargs)
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


def _field_hash(value: Any) -> str:
    if isinstance(value, str):
        serialized = value
    else:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _friendly_status(value: Any) -> str:
    status = str(value or "unknown")
    return _STATUS_LABELS.get(status, status)


def _edge_detection_result(payload: dict[str, Any]) -> tuple[bool, str, str | None]:
    boss = dict((payload.get("servers") or {}).get("boss") or {})
    browser = dict(boss.get("browser") or {})
    if boss.get("status") == "ready" and browser.get("status") == "ready":
        path = str(browser.get("edge_path") or "Microsoft Edge")
        return True, "Edge 浏览器检测成功", f"Playwright 已就绪；浏览器路径：{path}"
    code = browser.get("error_code") or browser.get("status") or boss.get("last_error_code") or boss.get("status")
    return False, f"Edge 浏览器检测失败：{_friendly_status(code)}", None


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
    return False, f"{model_name} 检测失败：{_friendly_status(code)}"


def _display_model_profiles(
    models: list[dict[str, Any]],
    *,
    limit: int = 3,
    role: str | None = None,
) -> list[dict[str, Any]]:
    """Return a small, stable model list without mutating saved profiles."""

    candidates = [item for item in models if role is None or item.get("role") == role]
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


def _job_search_feedback(payload: dict[str, Any]) -> tuple[str, str] | None:
    """Return a concise UI message for non-result search outcomes."""

    jobs = list(payload.get("jobs") or [])
    if payload.get("load_pending"):
        return "info", "本次下拉暂未加载出新岗位，可再次点击“下一页”继续尝试"
    status = str(payload.get("status") or "")
    code = str(payload.get("error_code") or "")
    if status == "waiting_user" or code:
        messages = {
            "boss_login_required": "请在已打开的 Edge 独立窗口中登录 BOSS，完成后再次点击搜索",
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
                    api_key = st.text_input(
                        f"{model['model_name']} API Key（仅当前进程内存）",
                        type="password",
                        key=f"api-key-{model['profile_id']}",
                    )
                    if st.button("暂存 API Key", key=f"credential-button-{model['profile_id']}", disabled=not api_key):
                        try:
                            credential = _request(
                                "POST",
                                "/api/credentials/session",
                                json={"profile_id": model["profile_id"], "api_key": api_key, "ttl_seconds": 300},
                            )
                            st.session_state[f"credential-{model['profile_id']}"] = credential["credential_handle_id"]
                            st.success("API Key 已暂存在进程内存，5 分钟后失效")
                        except Exception as exc:
                            st.error(str(exc))
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
        manual_job_panel = st.expander("手动粘贴 JD 或输入职位 URL（可选）")
        with manual_job_panel:
            raw = st.text_area("粘贴 JD", height=180)
            url = st.text_input("BOSS 职位 URL", key="job-url")
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
            header_cols = st.columns([3, 2, 2, 1.5, 1.5])
            for column, label in zip(
            header_cols, ["岗位（点击后在采集 Edge 打开）", "公司", "城市", "薪资", "HR 活跃"]
            ):
                column.markdown(f"**{label}**")
            for item in page_jobs:
                row_cols = st.columns([3, 2, 2, 1.5, 1.5])
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
                row_cols[4].write(
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
            if st.button("从 BOSS URL 解析", key="parse-job-url"):
                try:
                    if not url:
                        st.warning("请填写 BOSS 职位 URL")
                    else:
                        result = _request(
                            "POST",
                            "/api/jobs/from-url",
                            json={"url": url, "title": title or None, "company": company or None, "city": city or None, "raw_text": raw or None},
                        )
                        st.json(result)
                except Exception as exc:
                    st.error(str(exc))

        try:
            history = _request("GET", "/api/jobs/history?limit=20")
            if history.get("jobs"):
                with st.expander("查看最近解析的职位"):
                    st.dataframe(
                        [
                            {"job_id": item["job_id"], "岗位": item["title"], "公司": item.get("company"), "城市": item.get("city"), "HR 活跃": item.get("hr_activity") or "活跃时间待解析"}
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
            st.info("本流程只生成项目经历候选和可复制纯文本，不要求上传简历或填写个人信息。")

        st.markdown("#### 生成设置")
        try:
            available_models = _request("GET", "/api/models")
        except Exception:
            available_models = []
        chat_models = _display_model_profiles(available_models, limit=3, role="chat")
        embedding_models = _display_model_profiles(available_models, limit=3, role="embedding")
        chat_profile_id = st.selectbox(
            "聊天模型",
            [item["profile_id"] for item in chat_models],
            index=next((index for index, item in enumerate(chat_models) if item.get("default_for_role")), 0),
            key="task-chat-profile",
        ) if chat_models else None
        embedding_profile_id = st.selectbox(
            "Embedding 模型",
            [item["profile_id"] for item in embedding_models],
            index=next((index for index, item in enumerate(embedding_models) if item.get("default_for_role")), 0),
            key="task-embedding-profile",
        ) if embedding_models else None
        count = st.slider("候选项目数量", 1, 5, 3)
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
            st.write(f"状态：{task['status']} / 节点：{task.get('current_node')}")
            st.write(f"checkpoint：{task['checkpoint_version']}")
            if task.get("blocked_reason"):
                st.warning(f"任务被阻断：{task['blocked_reason']}。请按提示修复后重试。")
            if task.get("match"):
                st.metric("匹配分", task["match"]["total_score"])
            for candidate in task.get("candidates", []):
                with st.expander(f"{candidate['candidate_slot_id']} · {candidate['title']} · {candidate['match_score']}"):
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
                    st.info("请逐项确认候选内容。确认会校验字段哈希，之后再选择新增、替换或仅复制。")
                    fields = ("title", "period", "introduction", "tech_stack", "solutions", "results")
                    confirmations = []
                    for field in fields:
                        checked = st.checkbox(field, value=True, key=f"confirm-{task_id}-{selected['candidate_id']}-{field}")
                        if checked:
                            confirmations.append({"field_id": field, "value_hash": _field_hash(selected[field]), "confirmed": True})
                    if st.button("确认候选字段", key=f"confirm-candidate-{task_id}"):
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
                        default_target = str(decision.get("target_project_id") or project_ids[-1])
                        target_project_id = st.selectbox(
                            "替换项目",
                            project_ids,
                            index=project_ids.index(default_target) if default_target in project_ids else 0,
                            key=f"replace-project-{task_id}",
                        )
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
                allow_text_rewrite = st.checkbox("排版压缩后仍超页时允许缩短正文", key=f"compression-rewrite-{task_id}")
                cols = st.columns(2)
                if cols[0].button("同意压缩", key=f"compression-approve-{task_id}"):
                    action = "revise" if allow_text_rewrite else "approve"
                    try:
                        updated = _request(
                            "POST",
                            f"/api/tasks/{task_id}/resume",
                            json=_task_action(task, action, {"accept_compression": True, "allow_rewrite": allow_text_rewrite}),
                        )
                        st.session_state["task"] = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
                if cols[1].button("保持多页", key=f"compression-skip-{task_id}"):
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
                            },
                        )
                        st.session_state["review_preview"] = preview
                    except Exception as exc:
                        st.error(str(exc))
                review_preview = st.session_state.get("review_preview")
                review_matches = _preview_matches_task(review_preview, task, review_format)
                if review_matches:
                    st.write(f"预览状态：{review_preview.get('status')} · {review_preview.get('page_count')} 页")
                    st.text_area("预览内容", review_preview.get("content", ""), height=300, key=f"review-content-{task_id}")
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
                            json=_task_action(task, "revise", {"feedback": review_feedback}),
                        )
                        st.session_state["task"] = updated
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

            if task.get("status") == "completed" and task.get("resume_snapshot_id"):
                st.subheader("预览与导出")
                export_format = st.selectbox("格式", ["markdown", "docx", "pdf"], key=f"format-{task_id}")
                accept_compression = st.checkbox("同意先调整字体、间距和页边距", key=f"compress-{task_id}")
                allow_rewrite = st.checkbox("若仍超页，允许缩短正文（需再次检查事实）", key=f"rewrite-{task_id}")
                if st.button("生成预览", key=f"preview-{task_id}"):
                    try:
                        preview = _request(
                            "POST",
                            f"/api/tasks/{task_id}/export-preview",
                            json={"thread_id": task["thread_id"], "checkpoint_version": task["checkpoint_version"], "snapshot_id": task["resume_snapshot_id"], "format": export_format, "accept_compression": accept_compression, "allow_rewrite": allow_rewrite},
                        )
                        st.session_state["preview"] = preview
                    except Exception as exc:
                        st.error(str(exc))
                preview = st.session_state.get("preview")
                preview_matches = _preview_matches_task(preview, task, export_format)
                if preview_matches:
                    st.write(f"预览状态：{preview.get('status')} · {preview.get('page_count')} 页")
                    st.text_area("预览内容", preview.get("content", ""), height=280, key=f"preview-content-{task_id}")
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
                            json={"thread_id": task["thread_id"], "checkpoint_version": task["checkpoint_version"], "snapshot_id": task["resume_snapshot_id"], "format": export_format, "accept_compression": accept_compression, "allow_rewrite": allow_rewrite},
                        )
                        st.session_state["export"] = result
                    except Exception as exc:
                        st.error(str(exc))
                exported = st.session_state.get("export")
                if exported and exported.get("content"):
                    output_path = exported.get("output_path") or exported.get("path")
                    if exported.get("format") == "markdown":
                        st.download_button("下载 Markdown", exported["content"], file_name="resume.md", mime="text/markdown", key=f"download-{task_id}-md")
                    elif output_path and Path(output_path).is_file():
                        suffix = Path(output_path).suffix.lower()
                        mime = "application/pdf" if suffix == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        st.download_button(
                            f"下载 {suffix.lstrip('.').upper()}",
                            Path(output_path).read_bytes(),
                            file_name=Path(output_path).name,
                            mime=mime,
                            key=f"download-{task_id}-{suffix}",
                        )
                    st.success(f"导出状态：{exported.get('status')}；路径：{exported.get('output_path') or exported.get('path') or '请查看返回信息'}")


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
        cloud_api_key = st.text_input(
            "API Key（密码输入，仅暂存在后端进程内存）",
            type="password",
            key="wizard-cloud-api-key",
        )
        consent = st.checkbox(
            "我同意首次使用时向该外部模型发送所选岗位 JD",
            key="wizard-cloud-consent",
            help="同意状态会保存；API Key 不会写入配置、数据库、日志或备份。",
        )

        def ensure_cloud_profile() -> dict[str, Any]:
            if not cloud_base_url.strip() or not cloud_model_name.strip():
                raise ValueError("请填写 Base URL 和聊天模型名称")
            digest = hashlib.sha256(
                f"{cloud_base_url.strip().rstrip('/')}|{cloud_model_name.strip()}".encode()
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
                        "base_url": cloud_base_url.strip(),
                        "model_name": cloud_model_name.strip(),
                        "default_for_role": True,
                        "context_window_tokens": 32768,
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
            if cloud_api_key:
                credential = _request(
                    "POST",
                    "/api/credentials/session",
                    json={
                        "profile_id": profile_id,
                        "api_key": cloud_api_key,
                        # Process-local only; use a long TTL so the user's
                        # one-time confirmation lasts until the backend exits.
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
                st.success(
                    f"已保存 {saved_cloud['model_name']} 为默认聊天模型；API Key 未持久化"
                )
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
        st.caption(
            "只保存 Base URL、模型名称和默认模型选择；API Key 页面不回显，后端重启后需重新填写。"
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
            ok, message, detail = _ollama_detection_result(ollama_detection)
            (st.success if ok else st.error)(message)
            if detail:
                st.caption(detail)
        profiles = _display_model_profiles(
            [item for item in all_profiles if item.get("provider") == "ollama"],
            limit=3,
        )
        for profile in profiles:
            cols = st.columns([2, 2, 1])
            cols[0].write(f"{profile['role']} · {profile['model_name']}")
            cols[1].write(_friendly_status(profile.get("status")))
            if cols[2].button("检测", key=f"wizard-probe-{profile['profile_id']}"):
                try:
                    st.session_state[f"model_detection_{profile['profile_id']}"] = _request(
                        "POST",
                        f"/api/models/{profile['profile_id']}/probe",
                        json={
                            "credential_handle_id": st.session_state.get(
                                f"credential-{profile['profile_id']}"
                            )
                        },
                    )
                except Exception as exc:
                    st.error(str(exc))
            model_detection = st.session_state.get(
                f"model_detection_{profile['profile_id']}"
            )
            if model_detection:
                model_ok, model_message = _model_detection_result(model_detection)
                (st.success if model_ok else st.error)(model_message)

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
                        },
                    )
                    st.success("模型配置已保存，请点击检测")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


def _show_job_detail(st: Any, detail: dict[str, Any]) -> None:
    job = detail.get("job") if isinstance(detail, dict) else None
    if not isinstance(job, dict):
        return
    st.success(f"已取得岗位：{job.get('title') or '未命名岗位'}")
    st.caption(
        f"{job.get('company') or '公司未知'} ｜ {job.get('city') or '城市未知'} ｜ "
        f"{job.get('salary') or '薪资未知'} ｜ HR：{job.get('hr_activity') or '活跃时间待解析'}"
    )
    parser = detail.get("parser") if isinstance(detail.get("parser"), dict) else {}
    if parser.get("parser_mode") == "model_enhanced":
        st.success(
            f"JD 已由聊天模型增强解析：{parser.get('model_name') or parser.get('model_profile_id')}"
        )
    elif parser:
        warnings = list(parser.get("warnings") or [])
        st.warning(warnings[0] if warnings else "聊天模型不可用，当前显示规则解析结果")
    with st.expander("岗位职责、任职要求与完整 JD", expanded=True):
        st.markdown("**岗位职责**")
        responsibilities = list(job.get("responsibilities") or [])
        if responsibilities:
            for value in responsibilities:
                st.markdown(f"- {value}")
        else:
            st.caption("未拆分出独立岗位职责")
        st.markdown("**任职要求**")
        requirements = list(job.get("requirements") or [])
        if requirements:
            for value in requirements:
                st.markdown(f"- {value}")
        else:
            st.caption("未拆分出独立任职要求")
        st.markdown("**技能要求**")
        st.write("、".join(job.get("skills") or []) or "未单独识别")
        st.text_area(
            "完整 JD 原文",
            value=str(job.get("raw_text") or ""),
            height=220,
            disabled=True,
            key=f"wizard-jd-{job.get('job_id')}",
        )


def _render_job_search(st: Any, chat_profile_id: str | None = None) -> None:
    title = st.text_input("岗位名称", key="wizard-search-title")
    city = st.text_input("城市", key="wizard-search-city")
    company = st.text_input("公司（可选）", key="wizard-search-company")
    hr_activity = st.selectbox(
        "HR 活跃状态",
        ["不限", "在线或刚刚活跃", "今日活跃", "近3天活跃", "本周活跃", "活跃时间待解析"],
        key="wizard-search-hr-activity",
        help="列表页通常只能直接识别“在线”；选择并解析岗位详情后可取得更精确的活跃状态。",
    )
    risk_confirmed = st.checkbox(
        "我确认使用可见 Edge 访问 BOSS，并知悉账号风控风险",
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
                st.session_state["wizard_job_search"] = _request(
                    "POST",
                    "/api/jobs/search",
                    json={
                        "title": title,
                        "city": city,
                        "company": company or None,
                        "hr_activity": None if hr_activity == "不限" else hr_activity,
                        "limit": 20,
                    },
                )
                st.session_state["wizard_job_page"] = 0
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
            cols = st.columns([3, 2, 2, 1.5])
            cols[0].markdown(f"**{item.get('title') or '未命名岗位'}**")
            cols[1].write(item.get("company") or "公司未知")
            cols[2].write(item.get("salary") or "薪资未知")
            cols[3].write(item.get("hr_activity") or "活跃时间待解析")
            actions = st.columns(2)
            if actions[0].button(
                "查看并选择",
                key=f"wizard-select-job-{item['job_id']}",
                use_container_width=True,
            ):
                st.session_state["wizard_selected_job"] = item
                st.session_state.pop("wizard_job_detail", None)
                st.session_state.pop("wizard_task", None)
                st.session_state.pop("wizard_preview", None)
                st.rerun()
            if actions[1].button(
                "在采集 Edge 打开",
                key=f"wizard-open-job-{item['job_id']}",
                use_container_width=True,
            ):
                try:
                    _request(
                        "POST", "/api/jobs/browser/open", json={"url": item.get("source_url")}
                    )
                    detail = _request(
                        "POST",
                        "/api/jobs/from-url",
                        json={
                            "url": item.get("source_url"),
                            "chat_profile_id": chat_profile_id,
                            "credential_handle_id": st.session_state.get(
                                f"credential-{chat_profile_id}"
                            ),
                        },
                    )
                    if detail.get("status") == "ready" and isinstance(
                        detail.get("job"), dict
                    ):
                        _sync_wizard_job_state(st, detail["job"])
                        st.session_state["wizard_job_page_notice"] = (
                            "已在采集 Edge 打开，并刷新 HR 活跃状态："
                            f"{detail['job'].get('hr_activity') or '活跃时间待解析'}"
                        )
                        st.rerun()
                    st.success("已在岗位采集 Edge 中打开")
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
                more = _request(
                    "POST",
                    "/api/jobs/search",
                    json={
                        "title": title,
                        "city": city,
                        "company": company or None,
                        "hr_activity": None if hr_activity == "不限" else hr_activity,
                        "limit": 20,
                        "cursor": str(result["next_cursor"]),
                    },
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
                if not updated_search.get("load_pending"):
                    st.session_state["wizard_job_page_notice"] = "已经到达当前搜索结果末尾"
                st.session_state["wizard_job_page"] = page_index
                st.rerun()
        st.session_state["wizard_job_page"] = target_page
        st.rerun()

    selected = st.session_state.get("wizard_selected_job")
    if isinstance(selected, dict):
        st.markdown(f"#### 已选择：{selected.get('title')}")
        if st.button("解析该岗位的职责与完整 JD", key="wizard-parse-selected-job"):
            try:
                detail = _request(
                    "POST",
                    "/api/jobs/from-url",
                    json={
                        "url": selected.get("source_url"),
                        "chat_profile_id": chat_profile_id,
                        "credential_handle_id": st.session_state.get(
                            f"credential-{chat_profile_id}"
                        ),
                    },
                )
                if detail.get("status") == "ready":
                    _set_wizard_job_detail(st, detail)
                    st.rerun()
                else:
                    st.warning(detail.get("message") or "岗位详情暂未解析完成")
            except ApiRequestError as exc:
                st.error(exc.message)


def _render_task_progress(st: Any, task: dict[str, Any]) -> None:
    task_id = str(task.get("task_id") or "")
    if not task_id:
        return
    cols = st.columns([3, 1])
    cols[0].write(f"生成状态：{task.get('status')} · 当前步骤：{task.get('current_node')}")
    if cols[1].button("刷新状态", key=f"wizard-refresh-task-{task_id}"):
        try:
            st.session_state["wizard_task"] = _request("GET", f"/api/tasks/{task_id}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if task.get("blocked_reason"):
        st.warning(f"当前流程暂停：{task['blocked_reason']}。请在环境页完成对应配置后刷新。")
    if task.get("match"):
        st.metric("简历与岗位匹配分", task["match"].get("total_score"))

    node = str(task.get("current_node") or "")
    status = str(task.get("status") or "")
    candidates = [item for item in task.get("candidates", []) if item.get("status") != "duplicate"]
    if node in {"candidate_review", "candidate_confirmation", "project_application"}:
        for candidate in candidates:
            with st.expander(
                f"{candidate.get('candidate_slot_id')} · {candidate.get('title')} · 匹配分 {candidate.get('match_score')}",
                expanded=candidate.get("candidate_id") == task.get("selected_candidate_id"),
            ):
                st.write(candidate.get("introduction"))
                st.write("技术栈：" + "、".join(candidate.get("tech_stack") or []))
                for line in candidate.get("solutions") or []:
                    st.markdown(f"- {line}")
                st.write("结果：" + "；".join(candidate.get("results") or []))
                if node == "candidate_review" and st.button(
                    "选择这个项目",
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
        if st.button(
            "按反馈重新生成",
            key=f"wizard-regenerate-{task_id}",
            disabled=not feedback.strip(),
        ):
            updated = _request(
                "POST",
                f"/api/tasks/{task_id}/feedback",
                json={
                    "thread_id": task["thread_id"],
                    "checkpoint_version": task["checkpoint_version"],
                    "feedback": feedback,
                },
            )
            st.session_state["wizard_task"] = updated
            st.rerun()

    if node == "candidate_confirmation" and status == "waiting_user":
        selected = _selected_candidate(task)
        if selected:
            st.markdown("#### 核实项目字段")
            st.warning("模型推导内容均为待核实项。请确认真实、可解释后再继续。")
            confirmations = []
            for field in ("title", "period", "introduction", "tech_stack", "solutions", "results"):
                if st.checkbox(
                    f"确认 {field}",
                    value=False,
                    key=f"wizard-confirm-{task_id}-{field}",
                ):
                    confirmations.append(
                        {"field_id": field, "value_hash": _field_hash(selected[field]), "confirmed": True}
                    )
            if st.button(
                "确认以上项目内容",
                key=f"wizard-confirm-candidate-{task_id}",
                disabled=len(confirmations) != 6,
            ):
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

    if node == "project_application" and status == "waiting_user":
        decision = task.get("project_decision") or {}
        requested = str(task.get("requested_output_mode") or "resume_edit")
        rankings = decision.get("project_rankings") or []
        recommended_action = str(decision.get("recommended_action") or "add")
        target_id = decision.get("target_project_id")
        if requested == "project_only":
            st.info("将直接生成可复制的项目经历纯文本，不修改或生成整份简历。")
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
            action = recommended_action if recommended_action in {"add", "replace"} else "add"
            if action == "replace" and not target_id and rankings:
                target_id = rankings[-1].get("project_id")
            st.caption(decision.get("reason") or f"系统建议：{action}")
            if st.button("确认修改方式", key=f"wizard-apply-project-{task_id}"):
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
        allow_rewrite = st.checkbox("排版压缩后仍超页时允许缩短正文", key=f"wizard-rewrite-{task_id}")
        actions = st.columns(2)
        if actions[0].button("同意压缩到一页", key=f"wizard-compress-{task_id}"):
            updated = _request(
                "POST",
                f"/api/tasks/{task_id}/resume",
                json=_task_action(
                    task,
                    "revise" if allow_rewrite else "approve",
                    {"accept_compression": True, "allow_rewrite": allow_rewrite},
                ),
            )
            st.session_state["wizard_task"] = updated
            st.rerun()
        if actions[1].button("保持多页", key=f"wizard-keep-pages-{task_id}"):
            updated = _request(
                "POST", f"/api/tasks/{task_id}/resume", json=_task_action(task, "skip")
            )
            st.session_state["wizard_task"] = updated
            st.rerun()

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
                },
            )
            st.session_state["wizard_preview"] = preview
        preview = st.session_state.get("wizard_preview")
        if _preview_matches_task(preview, task, preview_format):
            st.text_area(
                "预览内容",
                value=str(preview.get("content") or ""),
                height=320,
                key=f"wizard-preview-content-{task_id}",
            )
            if st.button("确认最终结果", key=f"wizard-final-confirm-{task_id}"):
                updated = _request(
                    "POST", f"/api/tasks/{task_id}/resume", json=_task_action(task, "confirm")
                )
                st.session_state["wizard_task"] = updated
                st.rerun()

    if status == "completed":
        st.success("生成已完成")
        requested = str(task.get("requested_output_mode") or "resume_edit")
        export_format = "markdown" if requested == "project_only" else st.selectbox(
            "导出格式", ["markdown", "docx", "pdf"], key=f"wizard-export-format-{task_id}"
        )
        if st.button("生成最终文件", key=f"wizard-export-{task_id}"):
            preview = _request(
                "POST",
                f"/api/tasks/{task_id}/export-preview",
                json={
                    "thread_id": task["thread_id"],
                    "checkpoint_version": task["checkpoint_version"],
                    "snapshot_id": task["resume_snapshot_id"],
                    "format": export_format,
                },
            )
            st.session_state["wizard_final_preview"] = preview
        final_preview = st.session_state.get("wizard_final_preview")
        if isinstance(final_preview, dict) and final_preview.get("content"):
            content = str(final_preview["content"])
            st.text_area(
                "可复制内容" if requested == "project_only" else "最终内容",
                value=content,
                height=360,
                key=f"wizard-final-content-{task_id}",
            )
            st.download_button(
                "下载 Markdown",
                data=content,
                file_name="project.md" if requested == "project_only" else "resume.md",
                mime="text/markdown",
                key=f"wizard-download-{task_id}",
            )


def _reset_wizard(st: Any) -> None:
    prefixes = ("wizard_",)
    exact = {"resume", "selected_job_id", "job_detail_view", "task_id", "task"}
    for key in list(st.session_state):
        if key in exact or any(str(key).startswith(prefix) for prefix in prefixes):
            del st.session_state[key]


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
    """Write newly parsed detail fields back to cards and the selected job."""

    selected = st.session_state.get("wizard_selected_job")
    if isinstance(selected, dict) and (
        selected.get("job_id") == job.get("job_id")
        or selected.get("source_url") == job.get("source_url")
    ):
        st.session_state["wizard_selected_job"] = {**selected, **job}
    # Detail parsing often discovers the precise recruiter activity after the
    # list card only reported “待解析”. Keep both list views in sync.
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
                merged_jobs.append({**item, **job})
                changed = True
            else:
                merged_jobs.append(item)
        if changed:
            st.session_state[state_key] = {**search, "jobs": merged_jobs}


def _set_wizard_job_detail(st: Any, detail: dict[str, Any]) -> None:
    """Bind one JD and invalidate output that belonged to a previous JD."""

    st.session_state["wizard_job_detail"] = detail
    job = detail.get("job") if isinstance(detail, dict) else None
    if isinstance(job, dict):
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


def _render_screenshot_source(st: Any, chat_profile_id: str | None = None) -> None:
    """Render the paste/upload -> edit -> explicit confirmation OCR flow."""

    detail = st.session_state.get("wizard_job_detail") or {}
    if detail.get("source") == "screenshot_ocr_confirmed" and isinstance(detail.get("job"), dict):
        st.success("岗位截图已经校正确认并解析完成。")
        return
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
        return

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
    metadata = st.columns(2)
    title = metadata[0].text_input("岗位名称（必填）", key="wizard_ocr_title")
    company = metadata[1].text_input("公司（可选）", key="wizard_ocr_company")
    city = metadata[0].text_input("城市（可选）", key="wizard_ocr_city")
    salary = metadata[1].text_input("薪资（可选）", key="wizard_ocr_salary")
    hr_activity = st.text_input(
        "HR 活跃状态（可选，例如：在线、本周活跃）",
        key="wizard_ocr_hr_activity",
    )
    if st.button("确认校正并解析岗位 JD", key="wizard_confirm_screenshot"):
        if not title.strip():
            st.warning("请填写岗位名称")
        elif not corrected_text.strip():
            st.warning("校正后的岗位文字不能为空")
        else:
            try:
                detail = _request(
                    "POST",
                    "/api/jobs/screenshot/confirm",
                    json={
                        "ocr_id": ocr["ocr_id"],
                        "text": corrected_text,
                        "title": title,
                        "company": company or None,
                        "city": city or None,
                        "salary": salary or None,
                        "hr_activity": hr_activity or None,
                        "chat_profile_id": chat_profile_id,
                        "credential_handle_id": st.session_state.get(
                            f"credential-{chat_profile_id}"
                        ),
                    },
                )
                _set_wizard_job_detail(st, detail)
                st.session_state.pop("wizard_screenshot_ocr", None)
                st.session_state.pop("wizard_screenshot_preview", None)
                st.rerun()
            except ApiRequestError as exc:
                st.error(f"截图确认失败：{exc.message}")


def _render_resume_wizard(st: Any) -> None:
    header = st.columns([4, 1])
    header[0].subheader("简历项目生成向导")
    if header[1].button("重新开始", key="wizard-reset"):
        _reset_wizard(st)
        st.rerun()

    st.markdown("### 第 1 步：是否上传已有简历？")
    has_resume = st.radio(
        "请选择",
        ["yes", "no"],
        index=None,
        format_func=lambda value: "是，上传并修改已有简历" if value == "yes" else "否",
        key="wizard_has_resume",
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
        uploaded = st.file_uploader(
            "上传主简历（DOCX、文本型 PDF、Markdown 或 TXT，最多 10 MB）",
            type=["docx", "pdf", "md", "markdown", "txt"],
            key="wizard_resume_file",
        )
        if st.button("上传并解析简历", key="wizard_upload_resume", disabled=uploaded is None):
            try:
                parsed = _request(
                    "POST",
                    "/api/resumes/upload",
                    files={
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            uploaded.type or "application/octet-stream",
                        )
                    },
                )
                st.session_state["wizard_resume"] = parsed
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
            template_id = st.selectbox(
                "选择模板",
                ["builtin-cn-single-page"],
                format_func=lambda _: "系统默认中文单页模板",
                key="wizard_template_id",
            )
            st.caption("第一版向导默认使用系统中文单页模板，生成前会要求填写必要资料。")
        else:
            st.success("已选择纯文本模式：不需要填写姓名、电话或上传简历。")

    st.markdown("### 第 2 步：从哪里获取岗位 JD？")
    source = st.radio(
        "请选择一种来源",
        ["screenshot", "url", "search"],
        index=None,
        format_func=lambda value: {
            "screenshot": "粘贴或上传岗位截图",
            "url": "输入 BOSS 岗位网址",
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
        jd_models = _display_model_profiles(
            _request("GET", "/api/models"), limit=3, role="chat"
        )
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
                    f"{item['model_name']}（{'云端' if item.get('provider') == 'openai_compatible' else '本地'}）"
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
        _render_screenshot_source(st, jd_chat_profile_id)
        if not isinstance((st.session_state.get("wizard_job_detail") or {}).get("job"), dict):
            return
    elif source == "url":
        job_url = st.text_input("BOSS 岗位网址", key="wizard_job_url")
        risk = st.checkbox(
            "我确认使用可见 Edge 访问该网址，并知悉账号风控风险",
            key="wizard_url_risk",
        )
        if st.button("打开并解析岗位 JD", key="wizard_parse_url"):
            if not job_url.strip():
                st.warning("请输入岗位网址")
            elif not risk:
                st.warning("请先确认 BOSS 访问风险")
            else:
                try:
                    _request(
                        "POST",
                        "/api/consents/post-time-risk",
                        json={"granted": True, "policy_version": "boss-post-time-risk-v1"},
                    )
                    detail = _request(
                        "POST",
                        "/api/jobs/from-url",
                        json={
                            "url": job_url,
                            "chat_profile_id": jd_chat_profile_id,
                            "credential_handle_id": st.session_state.get(
                                f"credential-{jd_chat_profile_id}"
                            ),
                        },
                    )
                    if detail.get("status") == "ready":
                        _set_wizard_job_detail(st, detail)
                        st.rerun()
                    else:
                        st.warning(detail.get("message") or "岗位详情暂未解析完成")
                except ApiRequestError as exc:
                    st.error(exc.message)
    else:
        _render_job_search(st, jd_chat_profile_id)

    detail = st.session_state.get("wizard_job_detail") or {}
    if not isinstance(detail.get("job"), dict):
        return
    _show_job_detail(st, detail)

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
    chat_models = _display_model_profiles(models, limit=3, role="chat")
    embedding_models = _display_model_profiles(models, limit=3, role="embedding")
    chat_profile_id = st.selectbox(
        "聊天模型",
        [item["profile_id"] for item in chat_models],
        format_func=lambda value: next(
            (item["model_name"] for item in chat_models if item["profile_id"] == value), value
        ),
        key="wizard_chat_profile",
    ) if chat_models else None
    embedding_profile_id = st.selectbox(
        "Embedding 模型",
        [item["profile_id"] for item in embedding_models],
        format_func=lambda value: next(
            (item["model_name"] for item in embedding_models if item["profile_id"] == value), value
        ),
        key="wizard_embedding_profile",
    ) if embedding_models else None
    candidate_count = st.slider("候选项目数量", 1, 5, 3, key="wizard_candidate_count")
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
