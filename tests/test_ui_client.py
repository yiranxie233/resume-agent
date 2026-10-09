from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from app.ui.streamlit_app import (
    ApiRequestError,
    _candidate_editor_list,
    _canonical_openai_base_url,
    _display_model_profiles,
    _edge_detection_result,
    _job_model_parse_failed,
    _job_page,
    _job_search_feedback,
    _model_detection_result,
    _new_api_key,
    _ollama_detection_result,
    _ollama_installed_models,
    _ollama_role_model_names,
    _ready_model_profiles,
    _request,
    _set_all_candidate_confirmations,
    _template_error_detail,
)


class StreamlitApiClientTests(unittest.TestCase):
    def test_candidate_editor_lists_normalize_user_text(self) -> None:
        self.assertEqual(
            _candidate_editor_list("Python、FastAPI，LangGraph", tech_stack=True),
            ["Python", "FastAPI", "LangGraph"],
        )
        self.assertEqual(
            _candidate_editor_list("1. 检索岗位\n- 生成候选\n\n3、导出简历"),
            ["检索岗位", "生成候选", "导出简历"],
        )

    def test_job_model_retry_state_only_marks_fallback_results(self) -> None:
        self.assertTrue(
            _job_model_parse_failed(
                {"parser": {"parser_mode": "rule_fallback", "error_code": "model_parse_failed"}}
            )
        )
        self.assertFalse(
            _job_model_parse_failed({"parser": {"parser_mode": "model_enhanced"}})
        )
        self.assertFalse(
            _job_model_parse_failed({"parser": {"parser_mode": "human_corrected"}})
        )
        self.assertTrue(_job_model_parse_failed({"model_fallback": True}))
        self.assertFalse(_job_model_parse_failed({}))

    def test_confirm_all_candidate_fields_updates_every_checkbox(self) -> None:
        state = {"confirm-all": True}

        _set_all_candidate_confirmations(
            state,
            master_key="confirm-all",
            field_key_prefix="field-",
        )

        for field in (
            "title",
            "period",
            "introduction",
            "tech_stack",
            "solutions",
            "results",
        ):
            self.assertTrue(state[f"field-{field}"])

    def test_saved_api_key_mask_is_visible_but_never_resubmitted(self) -> None:
        self.assertIsNone(_new_api_key("••••••••••••"))
        self.assertIsNone(_new_api_key(""))
        self.assertEqual(_new_api_key("sk-new-value"), "sk-new-value")

    def test_cloud_base_url_accepts_root_or_full_chat_endpoint(self) -> None:
        self.assertEqual(
            _canonical_openai_base_url("https://api.example.com"),
            "https://api.example.com/v1",
        )
        self.assertEqual(
            _canonical_openai_base_url(
                "https://api.example.com/v1/chat/completions"
            ),
            "https://api.example.com/v1",
        )

    def test_job_pagination(self) -> None:
        jobs = [{"job_id": str(index)} for index in range(23)]

        page, index, count = _job_page(jobs, 1, 10)

        self.assertEqual(
            [item["job_id"] for item in page],
            [str(index) for index in range(10, 20)],
        )
        self.assertEqual((index, count), (1, 3))

    def test_request_raises_compact_api_error(self) -> None:
        request = httpx.Request("POST", "http://127.0.0.1:8000/api/jobs/search")
        response = httpx.Response(
            409,
            request=request,
            json={
                "error_code": "post_time_risk_consent_required",
                "message": "请先确认风险",
            },
        )
        with patch("httpx.request", return_value=response), self.assertRaises(ApiRequestError) as caught:
            _request("POST", "/api/jobs/search", json={})
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.code, "post_time_risk_consent_required")
        self.assertEqual(str(caught.exception), "请先确认风险")

    def test_local_api_ignores_vpn_proxy_environment(self) -> None:
        request = httpx.Request("GET", "http://127.0.0.1:8000/api/environment")
        response = httpx.Response(200, request=request, json={"status": "ready"})
        with patch("httpx.request", return_value=response) as mocked:
            _request("GET", "/api/environment")
        self.assertIs(mocked.call_args.kwargs["trust_env"], False)

    def test_model_timeout_becomes_page_safe_api_error(self) -> None:
        request = httpx.Request("POST", "http://127.0.0.1:8000/api/jobs/a/reparse")
        with patch(
            "httpx.request",
            side_effect=httpx.ReadTimeout("timed out", request=request),
        ), self.assertRaises(ApiRequestError) as caught:
            _request("POST", "/api/jobs/a/reparse", timeout=240, json={})
        self.assertEqual(caught.exception.status_code, 504)
        self.assertEqual(caught.exception.code, "connection_timeout")
    def test_detection_results_are_compact_user_messages(self) -> None:
        edge = _edge_detection_result(
            {
                "servers": {
                    "boss": {
                        "status": "ready",
                        "browser": {
                            "status": "ready",
                            "edge_path": "C:/Microsoft/Edge/msedge.exe",
                        },
                    }
                }
            }
        )
        ollama = _ollama_detection_result(
            {"status": "ready", "installed_names": ["bge-m3:latest", "qwen3.5:2b"]}
        )
        missing_model = _model_detection_result(
            {
                "status": "model_not_installed",
                "error_code": "model_not_installed",
                "model_name": "qwen2.5:7b",
            }
        )

        self.assertEqual(edge[0:2], (True, "Edge 浏览器检测成功"))
        self.assertEqual(ollama[0:2], (True, "Ollama 检测成功，共发现 2 个模型"))
        self.assertEqual(missing_model, (False, "qwen2.5:7b 检测失败：模型未安装"))
        network_denied = _model_detection_result(
            {
                "status": "service_unreachable",
                "error_code": "network_permission_denied",
                "model_name": "cloud-chat",
            }
        )
        self.assertEqual(
            network_denied,
            (False, "cloud-chat 检测失败：后端没有外网访问权限"),
        )
        provider_error = _model_detection_result(
            {
                "status": "probe_failed",
                "error_code": "provider_http_error",
                "error_message": "The model `gpt-unknown` does not exist",
                "model_name": "gpt-unknown",
            }
        )
        self.assertEqual(
            provider_error,
            (
                False,
                "gpt-unknown 检测失败：服务商返回请求错误"
                "（服务商信息：The model `gpt-unknown` does not exist）",
            ),
        )

        many_ollama = _ollama_detection_result(
            {"status": "ready", "installed_names": [f"qwen:test-{index}" for index in range(8)]}
        )
        self.assertEqual(many_ollama[1], "Ollama 检测成功，共发现 8 个模型")
        self.assertNotIn("qwen:test-3", many_ollama[2] or "")

    def test_model_profiles_are_deduplicated_prioritized_and_limited(self) -> None:
        models = [
            {
                "profile_id": f"duplicate-{index}",
                "role": "chat",
                "provider": "ollama",
                "base_url": "http://127.0.0.1:11434",
                "model_name": "qwen:test",
                "status": "unprobed",
                "default_for_role": False,
            }
            for index in range(8)
        ]
        models.extend(
            [
                {
                    "profile_id": "embedding-default",
                    "role": "embedding",
                    "provider": "ollama",
                    "base_url": "http://127.0.0.1:11434",
                    "model_name": "bge-m3:latest",
                    "status": "ready",
                    "default_for_role": True,
                },
                {
                    "profile_id": "chat-ready",
                    "role": "chat",
                    "provider": "ollama",
                    "base_url": "http://127.0.0.1:11434",
                    "model_name": "qwen2.5:7b",
                    "status": "ready",
                    "default_for_role": True,
                },
                {
                    "profile_id": "chat-other",
                    "role": "chat",
                    "provider": "openai_compatible",
                    "base_url": "https://api.example.com/v1",
                    "model_name": "qwen-plus",
                    "status": "ready",
                    "default_for_role": False,
                },
            ]
        )

        visible = _display_model_profiles(models, limit=3)
        chat_visible = _display_model_profiles(models, limit=3, role="chat")

        self.assertEqual(len(visible), 3)
        self.assertEqual(
            {item["profile_id"] for item in visible},
            {"embedding-default", "chat-ready", "chat-other"},
        )
        self.assertEqual(len(chat_visible), 3)
        self.assertEqual(sum(item["model_name"] == "qwen:test" for item in chat_visible), 1)
        self.assertEqual(
            {item["profile_id"] for item in _ready_model_profiles(models, "chat")},
            {"chat-ready", "chat-other"},
        )

    def test_template_errors_distinguish_configuration_rate_limit_and_network(self) -> None:
        auth = _template_error_detail(
            ApiRequestError(401, "github_auth_failed", "bad token"),
            stage="在 GitHub 查找",
        )
        limited = _template_error_detail(
            "github_rate_limited",
            stage="在 GitHub 查找",
        )
        network = _template_error_detail(
            "ConnectError: getaddrinfo failed",
            stage="简历下载",
        )

        self.assertEqual(auth["category"], "配置问题")
        self.assertEqual(limited["category"], "GitHub 限流")
        self.assertEqual(network["category"], "网络问题")

    def test_live_ollama_models_are_listed_and_filtered_by_capability(self) -> None:
        scan = {
            "status": "ready",
            "models": [
                {
                    "name": "qwen3.5:2b",
                    "capabilities": ["completion", "tools"],
                    "details": {"parameter_size": "2.3B", "context_length": 262144},
                },
                {
                    "name": "bge-m3:latest",
                    "capabilities": ["embedding"],
                    "details": {"parameter_size": "566M", "embedding_length": 1024},
                },
                {
                    "name": "qwen3-embedding:0.6b",
                    "capabilities": ["embedding", "tools"],
                    "details": {"embedding_length": 1024},
                },
            ],
        }

        self.assertEqual(
            [item["name"] for item in _ollama_installed_models(scan)],
            ["qwen3.5:2b", "bge-m3:latest", "qwen3-embedding:0.6b"],
        )
        self.assertEqual(_ollama_role_model_names(scan, "chat"), ["qwen3.5:2b"])
        self.assertEqual(
            _ollama_role_model_names(scan, "embedding"),
            ["bge-m3:latest", "qwen3-embedding:0.6b"],
        )

    def test_job_search_feedback_is_concise_and_actionable(self) -> None:
        login = _job_search_feedback(
            {
                "status": "waiting_user",
                "jobs": [],
                "error_code": "boss_login_required",
                "message": "technical details should not be shown",
            }
        )
        empty = _job_search_feedback({"status": "ready", "jobs": []})
        pending = _job_search_feedback(
            {"status": "ready", "jobs": [], "load_pending": True, "next_cursor": "2"}
        )
        closed = _job_search_feedback(
            {"status": "waiting_user", "jobs": [], "error_code": "edge_session_closed"}
        )

        self.assertEqual(
            login,
            ("warning", "请在已打开的岗位采集 Edge 中登录 BOSS，完成后再次点击搜索"),
        )
        self.assertEqual(
            empty,
            ("info", "未找到符合条件的职位，请调整岗位名称或城市后重试"),
        )
        self.assertEqual(closed, ("warning", "Edge 窗口已关闭，请重新点击搜索"))
        self.assertEqual(
            pending,
            ("info", "本次下拉暂未加载出新岗位，可再次点击“下一页”继续尝试"),
        )

    def test_stale_environment_token_retries_with_runtime_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".internal-token").write_text("current-token", encoding="utf-8")
            request = httpx.Request("GET", "http://127.0.0.1:8000/api/environment")
            responses = [
                httpx.Response(401, request=request, json={"error_code": "unauthorized"}),
                httpx.Response(200, request=request, json={"status": "ready"}),
            ]

            with patch.dict(
                "os.environ",
                {
                    "RESUME_AGENT_INTERNAL_TOKEN": "stale-token",
                    "RESUME_AGENT_DATA_ROOT": str(root),
                },
                clear=False,
            ), patch("httpx.request", side_effect=responses) as mocked:
                result = _request("GET", "/api/environment")

            self.assertEqual(result, {"status": "ready"})
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(
                mocked.call_args_list[0].kwargs["headers"]["X-Resume-Agent-Token"],
                "stale-token",
            )
            self.assertEqual(
                mocked.call_args_list[1].kwargs["headers"]["X-Resume-Agent-Token"],
                "current-token",
            )


if __name__ == "__main__":
    unittest.main()
