from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.schemas import JobUrlRequest, ModelProfile, ModelRole, Provider
from app.core.store import InMemoryStore
from app.main import job_from_url
from app.services.job_llm_parser import enhance_job_with_model
from app.services.job_parser import infer_job_metadata, job_from_text
from app.ui.streamlit_app import _reset_wizard, _set_wizard_job_detail


class _Gateway:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    def chat(self, profile, messages, **kwargs):
        self.calls.append((profile, messages, kwargs))
        return SimpleNamespace(content=self.content, model_name=profile.model_name)


class _SequenceGateway(_Gateway):
    def __init__(self, contents: list[str]) -> None:
        super().__init__(contents[-1])
        self.contents = list(contents)

    def chat(self, profile, messages, **kwargs):
        self.calls.append((profile, messages, kwargs))
        value = self.contents.pop(0) if self.contents else self.content
        return SimpleNamespace(content=value, model_name=profile.model_name)


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

    def test_invalid_first_response_is_repaired_within_the_same_user_action(self) -> None:
        job = job_from_text(
            "AI 工程师",
            "岗位职责\n负责接口开发\n任职要求\n熟悉 Python",
        )
        gateway = _SequenceGateway(
            [
                '{"responsibilities":["负责接口',
                '{"responsibilities":["负责接口开发"],'
                '"requirements":["熟悉 Python"],"skills":["Python"]}',
            ]
        )

        outcome = enhance_job_with_model(job, profile=_profile(), gateway=gateway)

        self.assertEqual(outcome.metadata["parse_attempts"], 2)
        self.assertEqual(outcome.job.responsibilities, ["负责接口开发"])
        self.assertEqual(len(gateway.calls), 2)
        repair_prompt = gateway.calls[1][1][-1].content
        self.assertIn("上一轮输出未通过", repair_prompt)

    def test_feedback_reparse_receives_previous_structure(self) -> None:
        job = job_from_text("AI 工程师", "岗位职责\n负责接口开发\n任职要求\n熟悉 Python")
        gateway = _Gateway(
            '{"responsibilities":["负责接口开发"],'
            '"requirements":["熟悉 Python"],"skills":["Python"]}'
        )

        outcome = enhance_job_with_model(
            job,
            profile=_profile(),
            gateway=gateway,
            feedback="请保持职责和要求分开",
            previous_job=job,
        )

        prompt = gateway.calls[0][1][-1].content
        developer_prompt = gateway.calls[0][1][1].content
        self.assertIn("上一次解析结果", prompt)
        self.assertIn("请保持职责和要求分开", prompt)
        self.assertIn("禁止把同一编号拆成多个短语", developer_prompt)
        self.assertIn("用户留言仅用于指定分点、归类、去重方式", developer_prompt)
        self.assertTrue(outcome.metadata["feedback_applied"])

    def test_model_cannot_turn_alternative_group_into_all_required_skills(self) -> None:
        raw_text = """任职要求
熟悉 Python、Java、Go 任选其一
掌握 PostgreSQL
"""
        job = job_from_text("后端工程师", raw_text)
        gateway = _Gateway(
            '{"responsibilities":[],"requirements":["熟悉 Python、Java、Go 任选其一"],'
            '"skills":["Python","Java","Go","PostgreSQL"]}'
        )

        outcome = enhance_job_with_model(job, profile=_profile(), gateway=gateway)

        self.assertEqual(
            outcome.job.skills,
            ["PostgreSQL", "Python、Java、Go 任选其一"],
        )
        self.assertIn("不得拆成多个必需技能", gateway.calls[0][1][1].content)

    def test_screenshot_metadata_is_inferred_without_manual_fields(self) -> None:
        metadata = infer_job_metadata(
            "AI 应用开发工程师\n示例科技\n20-30K·14薪\n工作地点：深圳·南山\n刚刚活跃\n岗位职责\n负责智能体开发"
        )

        self.assertEqual(metadata["title"], "AI 应用开发工程师")
        self.assertEqual(metadata["company"], "示例科技")
        self.assertEqual(metadata["salary"], "20-30K·14薪")
        self.assertEqual(metadata["city"], "深圳·南山")
        self.assertEqual(metadata["hr_activity"], "刚刚活跃")

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
            job_parse_confirmations={},
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
            job_parse_confirmations={},
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
    def test_restart_rotates_first_question_widget_identity(self) -> None:
        st = _FakeStreamlit(
            {
                "wizard_has_resume_0": "yes",
                "wizard_job_detail": {"job": {}},
                "_resume_wizard_epoch": 0,
            }
        )

        _reset_wizard(st)

        self.assertNotIn("wizard_has_resume_0", st.session_state)
        self.assertNotIn("wizard_job_detail", st.session_state)
        self.assertEqual(st.session_state["_resume_wizard_epoch"], 1)

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

    def test_detail_cannot_replace_concrete_card_recruiter_pair(self) -> None:
        st = _FakeStreamlit(
            {
                "wizard_job_search": {
                    "jobs": [
                        {
                            "job_id": "job-1",
                            "hr_name": "曹先生",
                            "hr_activity": "3日内活跃",
                        }
                    ]
                },
                "wizard_selected_job": {
                    "job_id": "job-1",
                    "hr_name": "曹先生",
                    "hr_activity": "3日内活跃",
                },
            }
        )

        _set_wizard_job_detail(
            st,
            {
                "job": {
                    "job_id": "job-1",
                    "hr_name": "其他招聘者",
                    "hr_activity": "今日活跃",
                    "responsibilities": ["负责接口开发"],
                }
            },
        )

        card = st.session_state["wizard_job_search"]["jobs"][0]
        self.assertEqual(card["hr_name"], "曹先生")
        self.assertEqual(card["hr_activity"], "3日内活跃")
        self.assertEqual(card["responsibilities"], ["负责接口开发"])


if __name__ == "__main__":
    unittest.main()
