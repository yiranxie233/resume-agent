from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.schemas import ModelProfile, ModelRole, Provider
from app.core.store import InMemoryStore
from app.main import (
    confirm_template_structure_endpoint,
    get_template_structure,
    model_parse_template,
)
from app.services.template_llm_parser import (
    confirm_template_structure,
    load_template_structure,
    structure_template_with_model,
)


class _Gateway:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, profile, messages, **kwargs):
        self.calls.append(dict(kwargs))
        source_text = str(messages[-1].content if messages else "")
        if "## 个人信息" not in source_text:
            return SimpleNamespace(
                content=(
                    '{"sections":['
                    '{"module":"summary","title":"个人简介",'
                    '"content":"[待补充]"},'
                    '{"module":"skills","title":"技能清单",'
                    '"content":"[待补充]"},'
                    '{"module":"projects","title":"项目经历",'
                    '"content":"[待补充]"}'
                    "]}"
                ),
                model_name=profile.model_name,
            )
        return SimpleNamespace(
            content=(
                '{"sections":['
                '{"module":"basic","title":"个人信息",'
                '"content":"姓名：[待填写]\\n电话：[待填写]"},'
                '{"module":"skills","title":"技能清单",'
                '"content":"Python、FastAPI"},'
                '{"module":"projects","title":"项目经历",'
                '"content":"项目名称\\n项目简介：[待填写]"}'
                "]}"
            ),
            model_name=profile.model_name,
        )


def _profile() -> ModelProfile:
    return ModelProfile(
        profile_id="template-chat",
        role=ModelRole.CHAT,
        provider=Provider.OLLAMA,
        base_url="http://127.0.0.1:11434",
        model_name="qwen-template-test",
        status="ready",
        default_for_role=True,
    )


def _template_text() -> str:
    return (
        "# 中文技术简历\n\n"
        "## 个人信息\n姓名：[待填写]\n电话：[待填写]\n\n"
        "## 技能清单\nPython、FastAPI\n\n"
        "## 项目经历\n项目名称\n项目简介：[待填写]\n"
    )


class TemplateLLMParserTests(unittest.TestCase):
    def test_model_mapping_requires_editable_user_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "template.md"
            source.write_text(_template_text(), encoding="utf-8")
            parsed = structure_template_with_model(
                "github-test",
                source,
                profile=_profile(),
                gateway=_Gateway(),
            )

            self.assertEqual(parsed["status"], "ready")
            self.assertFalse(parsed["confirmed"])
            self.assertEqual(
                [item["module"] for item in parsed["sections"]],
                ["basic", "skills", "projects"],
            )
            edited = [dict(item, old_value_hash=item["value_hash"]) for item in parsed["sections"]]
            edited[1]["title"] = "专业技能"
            confirmed = confirm_template_structure(
                "github-test",
                source,
                structure_id=parsed["structure_id"],
                source_sha256=parsed["source_sha256"],
                sections=edited,
            )

            self.assertTrue(confirmed["confirmed"])
            self.assertEqual(confirmed["sections"][1]["title"], "专业技能")
            self.assertTrue(all(item["confirmed"] for item in confirmed["sections"]))
            self.assertEqual(
                load_template_structure("github-test", source),
                confirmed,
            )

    def test_source_update_invalidates_previous_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "template.md"
            source.write_text(_template_text(), encoding="utf-8")
            parsed = structure_template_with_model(
                "github-test",
                source,
                profile=_profile(),
                gateway=_Gateway(),
            )
            edited = [dict(item, old_value_hash=item["value_hash"]) for item in parsed["sections"]]
            confirm_template_structure(
                "github-test",
                source,
                structure_id=parsed["structure_id"],
                source_sha256=parsed["source_sha256"],
                sections=edited,
            )
            source.write_text(_template_text() + "\n## 证书\n[待填写]\n", encoding="utf-8")

            stale = load_template_structure("github-test", source)

            self.assertIsNotNone(stale)
            self.assertEqual(stale["status"], "stale")
            self.assertFalse(stale["confirmed"])


class TemplateLLMParserApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_or_cloud_profile_contract_returns_and_confirms_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            store = InMemoryStore()
            store.save_profile(_profile())
            app_state = SimpleNamespace(
                settings=SimpleNamespace(
                    data_root=data_root,
                    external_model_consent_required=True,
                ),
                store=store,
                model_gateway=_Gateway(),
                credentials=SimpleNamespace(has=lambda *args, **kwargs: False),
                consents={},
            )
            with patch("app.main.state", return_value=app_state):
                parsed = await model_parse_template(
                    "builtin-cn-single-page",
                    {"chat_profile_id": "template-chat"},
                    None,
                )
                self.assertEqual(parsed["status"], "ready")
                sections = [
                    {**item, "old_value_hash": item["value_hash"]}
                    for item in parsed["sections"]
                ]
                confirmed = await confirm_template_structure_endpoint(
                    "builtin-cn-single-page",
                    {
                        "structure_id": parsed["structure_id"],
                        "source_sha256": parsed["source_sha256"],
                        "sections": sections,
                    },
                    None,
                )
                loaded = await get_template_structure("builtin-cn-single-page", None)

            self.assertEqual(confirmed["status"], "confirmed")
            self.assertTrue(loaded["confirmed"])
            self.assertEqual(loaded["parser"]["document_kind"], "resume_template")


if __name__ == "__main__":
    unittest.main()
