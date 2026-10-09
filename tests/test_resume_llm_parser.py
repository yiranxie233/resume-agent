from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.schemas import ModelProfile, ModelRole, Provider, ResumeDocument, ResumeSection
from app.core.store import InMemoryStore
from app.main import model_parse_resume
from app.services.resume_llm_parser import structure_resume_with_model


class _Gateway:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[tuple[object, object, object]] = []

    def chat(self, profile, messages, **kwargs):
        self.calls.append((profile, messages, kwargs))
        return SimpleNamespace(content=self.content, model_name=profile.model_name)


class _SequenceGateway(_Gateway):
    def __init__(self, contents: list[str]) -> None:
        super().__init__(contents[-1])
        self.contents = list(contents)

    def chat(self, profile, messages, **kwargs):
        self.calls.append((profile, messages, kwargs))
        content = self.contents.pop(0)
        return SimpleNamespace(content=content, model_name=profile.model_name)


def _profile() -> ModelProfile:
    return ModelProfile(
        profile_id="resume-chat",
        role=ModelRole.CHAT,
        provider=Provider.OLLAMA,
        base_url="http://127.0.0.1:11434",
        model_name="qwen-test",
        status="ready",
        default_for_role=True,
    )


def _resume() -> ResumeDocument:
    raw = "基本信息\n张三 13800000000\n项目经历\n智能助手\n负责 FastAPI 服务开发\n技能清单\nPython、FastAPI"
    return ResumeDocument(
        resume_id="resume-test",
        filename="resume.pdf",
        file_type="pdf",
        file_size=len(raw.encode()),
        page_count=1,
        raw_text=raw,
        sections=[
            ResumeSection(
                section_id="rule-section",
                module="basic",
                title="基本信息",
                content=raw,
            )
        ],
    )


class ResumeLLMParserTests(unittest.TestCase):
    def test_only_original_contiguous_spans_are_persisted(self) -> None:
        gateway = _Gateway(
            '{"sections":['
            '{"module":"basic","title":"基本信息","content":"张三 13800000000"},'
            '{"module":"projects","title":"项目经历","content":"智能助手\\n负责 FastAPI 服务开发"},'
            '{"module":"work","title":"工作经历","content":"曾负责百人团队"}'
            "]}"
        )

        outcome = structure_resume_with_model(_resume(), profile=_profile(), gateway=gateway)

        self.assertEqual(outcome.metadata["parser_mode"], "model_structured")
        self.assertEqual([item.module for item in outcome.resume.sections], ["basic", "projects"])
        self.assertEqual(outcome.resume.sections[1].content, "智能助手\n负责 FastAPI 服务开发")
        self.assertIn("已丢弃 1 个", outcome.metadata["warnings"][0])

    def test_all_hallucinated_sections_are_rejected(self) -> None:
        gateway = _Gateway(
            '{"sections":[{"module":"work","title":null,"content":"管理百人团队"}]}'
        )
        with self.assertRaisesRegex(ValueError, "没有通过"):
            structure_resume_with_model(_resume(), profile=_profile(), gateway=gateway)

    def test_model_reflow_repairs_extracted_line_break_and_chinese_schema(self) -> None:
        resume = _resume().model_copy(
            update={
                "raw_text": "基本信息\n张三 13800000000\n项目经历\n智能助手\n负责 FastAPI\n服务开发\n技能清单\nPython、FastAPI"
            }
        )
        gateway = _Gateway(
            "下面是结构化结果：\n"
            '{"sections":{"项目经历":{"title":"项目经历",'
            '"content":"智能助手\\n负责 FastAPI 服务开发"},'
            '"技能清单":"Python、FastAPI"}}'
        )

        outcome = structure_resume_with_model(resume, profile=_profile(), gateway=gateway)

        self.assertEqual([item.module for item in outcome.resume.sections], ["projects", "skills"])
        self.assertEqual(
            outcome.resume.sections[0].content,
            "智能助手\n负责 FastAPI 服务开发",
        )
        self.assertEqual(
            outcome.metadata["evidence_policy"],
            "dynamic_sections_with_contiguous_character_evidence_v3",
        )

    def test_actual_resume_headings_work_without_sections_wrapper(self) -> None:
        resume = _resume().model_copy(
            update={
                "raw_text": (
                    "基本信息\n姓名：张三\n电话：13800000000\n"
                    "校园活动\n技术协会志愿者\n组织每周分享\n"
                    "项目经历\n智能助手\n负责 FastAPI 服务开发"
                )
            }
        )
        gateway = _Gateway(
            '{"基本信息":{"姓名":"张三","电话":"13800000000"},'
            '"校园活动":{"正文":"技术协会志愿者\\n组织每周分享"},'
            '"项目经历":{"标题":"智能助手","正文":"智能助手\\n负责 FastAPI 服务开发"}}'
        )

        outcome = structure_resume_with_model(resume, profile=_profile(), gateway=gateway)

        self.assertEqual(
            [item.module for item in outcome.resume.sections],
            ["basic", "other", "projects"],
        )
        # A field-oriented model response is resolved back to one verbatim
        # source span, including labels that the model did not repeat.
        self.assertEqual(
            outcome.resume.sections[0].content,
            "张三\n电话：13800000000",
        )
        self.assertEqual(outcome.resume.sections[1].title, "校园活动")
        self.assertEqual(outcome.metadata["parser_version"], "resume-llm-parser-v3")

    def test_common_blocks_wrapper_and_free_form_categories_are_normalized(self) -> None:
        gateway = _Gateway(
            '{"data":{"blocks":['
            '{"category":"项目经验","heading":"项目经历",'
            '"text":"智能助手\\n负责 FastAPI 服务开发"},'
            '{"category":"技术能力","heading":"技能清单",'
            '"text":"Python、FastAPI"}'
            ']}}'
        )

        outcome = structure_resume_with_model(_resume(), profile=_profile(), gateway=gateway)

        self.assertEqual(
            [item.module for item in outcome.resume.sections],
            ["projects", "skills"],
        )
        self.assertEqual([item.title for item in outcome.resume.sections], ["项目经历", "技能清单"])

    def test_sections_may_be_json_encoded_object_instead_of_array(self) -> None:
        gateway = _Gateway(
            '{"sections":"{\\"技能清单\\":{\\"content\\":'
            '\\"Python、FastAPI\\"}}"}'
        )

        outcome = structure_resume_with_model(_resume(), profile=_profile(), gateway=gateway)

        self.assertEqual(len(gateway.calls), 1)
        self.assertEqual(outcome.resume.sections[0].module, "skills")

    def test_title_and_items_section_shape_is_dynamic(self) -> None:
        gateway = _Gateway(
            '{"resume":{"sections":['
            '{"section_name":"技能清单","items":["Python、FastAPI"]},'
            '{"section_name":"项目经历","items":['
            '"智能助手","负责 FastAPI 服务开发"]}'
            ']}}'
        )

        outcome = structure_resume_with_model(_resume(), profile=_profile(), gateway=gateway)

        self.assertEqual(
            [item.module for item in outcome.resume.sections],
            ["skills", "projects"],
        )

    def test_invalid_section_shape_also_gets_format_correction_retry(self) -> None:
        gateway = _SequenceGateway(
            [
                '{"sections":42}',
                '{"项目经历":"智能助手\\n负责 FastAPI 服务开发"}',
            ]
        )

        outcome = structure_resume_with_model(_resume(), profile=_profile(), gateway=gateway)

        self.assertEqual(len(gateway.calls), 2)
        self.assertTrue(outcome.metadata["format_retry_used"])
        self.assertEqual(outcome.resume.sections[0].module, "projects")

    def test_invalid_json_gets_one_format_correction_retry(self) -> None:
        gateway = _SequenceGateway(
            [
                "我先解释一下解析思路，但没有给出 JSON",
                (
                    '{"sections":[{"module":"skills","title":"技能清单",'
                    '"content":"Python、FastAPI"}]}'
                ),
            ]
        )

        outcome = structure_resume_with_model(_resume(), profile=_profile(), gateway=gateway)

        self.assertEqual(len(gateway.calls), 2)
        self.assertTrue(outcome.metadata["format_retry_used"])
        self.assertEqual(outcome.resume.sections[0].module, "skills")


class ResumeLLMParserApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_ready_local_model_updates_editable_sections(self) -> None:
        store = InMemoryStore()
        resume = _resume()
        store.save_resume(resume)
        store.save_profile(_profile())
        gateway = _Gateway(
            '{"sections":['
            '{"module":"projects","title":"项目经历","content":"智能助手\\n负责 FastAPI 服务开发"},'
            '{"module":"skills","title":"技能清单","content":"Python、FastAPI"}'
            "]}"
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
            result = await model_parse_resume(
                resume.resume_id,
                {"chat_profile_id": "resume-chat"},
                None,
            )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["parser"]["model_profile_id"], "resume-chat")
        self.assertEqual(
            [item["module"] for item in result["resume"]["sections"]],
            ["projects", "skills"],
        )


if __name__ == "__main__":
    unittest.main()
