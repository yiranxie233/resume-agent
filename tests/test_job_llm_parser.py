from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.schemas import JobUrlRequest, ModelProfile, ModelRole, Provider
from app.core.store import InMemoryStore
from app.main import job_from_url
from app.services.job_llm_parser import enhance_job_with_model
from app.services.job_parser import job_from_text
from app.ui.streamlit_app import _set_wizard_job_detail


class _Gateway:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    def chat(self, profile, messages, **kwargs):
        self.calls.append((profile, messages, kwargs))
        return SimpleNamespace(content=self.content, model_name=profile.model_name)


def _profile(*, provider: Provider = Provider.OLLAMA) -> ModelProfile:
    return ModelProfile(
        profile_id="jd-chat",
        role=ModelRole.CHAT,
        provider=provider,
        base_url=(
            "http://127.0.0.1:11434"
            if provider is Provider.OLLAMA
            else "https://api.example.com/v1"
        ),
        model_name="qwen-test",
        status="ready",
        default_for_role=True,
        credential_required=provider is Provider.OPENAI_COMPATIBLE,
        context_window_tokens=32768,
    )


class JobLLMParserTests(unittest.TestCase):
    def test_model_may_only_return_verbatim_jd_evidence(self) -> None:
        raw_text = """岗位职责
负责 FastAPI 智能体服务开发
任职要求
熟悉 Python、FastAPI
"""
        job = job_from_text("AI 工程师", raw_text)
        gateway = _Gateway(
            """{"responsibilities":["负责 FastAPI 智能体服务开发"],
            "requirements":["熟悉 Python、FastAPI","至少三年工作经验"],
            "skills":["Python","FastAPI"]}"""
        )

        outcome = enhance_job_with_model(job, profile=_profile(), gateway=gateway)

        self.assertEqual(outcome.metadata["parser_mode"], "model_enhanced")
        self.assertEqual(outcome.job.responsibilities, ["负责 FastAPI 智能体服务开发"])
        self.assertEqual(outcome.job.requirements, ["熟悉 Python、FastAPI"])
        self.assertEqual(outcome.job.skills, ["Python", "FastAPI"])
        self.assertIn("已丢弃 1 条", outcome.metadata["warnings"][0])

    def test_all_unsupported_model_content_is_rejected(self) -> None:
        job = job_from_text("AI 工程师", "岗位职责\n负责接口开发")
        gateway = _Gateway(
            '{"responsibilities":["负责大模型训练"],"requirements":[],"skills":[]}'
        )

        with self.assertRaisesRegex(ValueError, "没有通过"):
            enhance_job_with_model(job, profile=_profile(), gateway=gateway)

    def test_privacy_gitignore_covers_runtime_credentials_and_documents(self) -> None:
        ignore = Path(".gitignore").read_text(encoding="utf-8")
        for rule in (
            ".env",
            "!.env.example",
            "data/",
            "photos/",
            "output/",
            "**/edge-profile/",
            "*.docx",
            "*.pdf",
        ):
            self.assertIn(rule, ignore)


class JobLLMParserApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_url_text_uses_default_ready_chat_model(self) -> None:
        store = InMemoryStore()
        store.save_profile(_profile())
        gateway = _Gateway(
            '{"responsibilities":["负责 FastAPI 服务开发"],'
            '"requirements":["熟悉 Python"],"skills":["Python","FastAPI"]}'
        )
        app_state = SimpleNamespace(
            store=store,
            db_mirror=None,
            model_gateway=gateway,
            credentials=SimpleNamespace(has=lambda *args, **kwargs: False),
            settings=SimpleNamespace(external_model_consent_required=True),
            consents={},
        )

        with patch("app.main.state", return_value=app_state):
            result = await job_from_url(
                JobUrlRequest(
                    url="https://www.zhipin.com/job_detail/example.html",
                    title="Python 工程师",
                    raw_text="岗位职责\n负责 FastAPI 服务开发\n任职要求\n熟悉 Python",
                ),
                None,
            )

        self.assertEqual(result["parser"]["parser_mode"], "model_enhanced")
        self.assertEqual(result["job"]["skills"], ["Python", "FastAPI"])
        self.assertEqual(len(gateway.calls), 1)

    async def test_missing_cloud_credential_falls_back_without_losing_job(self) -> None:
        store = InMemoryStore()
        store.save_profile(_profile(provider=Provider.OPENAI_COMPATIBLE))
        gateway = _Gateway("{}")
        app_state = SimpleNamespace(
            store=store,
            db_mirror=None,
            model_gateway=gateway,
            credentials=SimpleNamespace(has=lambda *args, **kwargs: False),
            settings=SimpleNamespace(external_model_consent_required=True),
            consents={"external_model:global": {"granted": True}},
        )

        with patch("app.main.state", return_value=app_state):
            result = await job_from_url(
                JobUrlRequest(
                    url="https://www.zhipin.com/job_detail/example.html",
                    title="Python 工程师",
                    raw_text="岗位职责：负责接口开发",
                ),
                None,
            )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["parser"]["parser_mode"], "rule_fallback")
        self.assertEqual(result["parser"]["error_code"], "credential_missing")
        self.assertEqual(result["job"]["responsibilities"], ["负责接口开发"])
        self.assertEqual(gateway.calls, [])


class _FakeStreamlit:
    def __init__(self, session_state):
        self.session_state = session_state


class JobDetailStateTests(unittest.TestCase):
    def test_detail_hr_activity_is_written_back_to_search_list(self) -> None:
        st = _FakeStreamlit(
            {
                "wizard_job_search": {
                    "jobs": [
                        {
                            "job_id": "job-1",
                            "source_url": "https://www.zhipin.com/job_detail/a.html",
                            "hr_activity": "活跃时间待解析",
                        }
                    ]
                },
                "wizard_selected_job": {
                    "job_id": "job-1",
                    "hr_activity": "活跃时间待解析",
                },
            }
        )

        _set_wizard_job_detail(
            st,
            {
                "job": {
                    "job_id": "job-1",
                    "source_url": "https://www.zhipin.com/job_detail/a.html",
                    "hr_activity": "本周活跃",
                }
            },
        )

        self.assertEqual(
            st.session_state["wizard_job_search"]["jobs"][0]["hr_activity"],
            "本周活跃",
        )
        self.assertEqual(
            st.session_state["wizard_selected_job"]["hr_activity"], "本周活跃"
        )


if __name__ == "__main__":
    unittest.main()
