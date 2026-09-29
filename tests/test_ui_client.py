from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from app.ui.streamlit_app import (
    ApiRequestError,
    _edge_detection_result,
    _job_page,
    _job_search_feedback,
    _display_model_profiles,
    _model_detection_result,
    _ollama_detection_result,
    _request,
)


class StreamlitApiClientTests(unittest.TestCase):
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
            ("warning", "请在已打开的 Edge 独立窗口中登录 BOSS，完成后再次点击搜索"),
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
