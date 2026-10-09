from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.main import _attach_template_visual_preview, download_template, search_templates
from app.services.export_service import PreviewResult


class _GitHubManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    async def start_server(self, name: str) -> None:
        return None

    async def health(self, *, probe: bool = False) -> dict:
        return {"github": {"name": "github", "status": "ready"}}

    async def call_tool(self, server: str, tool: str, arguments: dict) -> dict:
        self.calls.append((server, tool, dict(arguments)))
        if tool == "github_search_resume_templates":
            return {
                "ok": True,
                "data": {
                    "items": [
                        {
                            "template_id": "github-0123456789abcdef01234567",
                            "name": "中文技术简历",
                            "file_type": "markdown",
                            "language": "zh",
                            "license": "MIT",
                            "license_allowed": True,
                            "source_url": "https://raw.githubusercontent.com/example/resume/main/resume.md",
                        }
                    ]
                },
            }
        return {
            "ok": True,
            "data": {
                "status": "cached",
                "template_id": "github-0123456789abcdef01234567",
                "path": "data/templates/cache/template/resume.md",
                "metadata": {
                    "template_id": "github-0123456789abcdef01234567",
                    "repository": "example/resume",
                    "source_url": "https://raw.githubusercontent.com/example/resume/main/resume.md",
                    "file_type": "markdown",
                    "language": "zh",
                    "license": "MIT",
                    "license_allowed": True,
                },
            },
        }


class _TemplateMirror:
    def __init__(self) -> None:
        self.metadata: list[dict] = []

    def persist_template(self, metadata: dict) -> str:
        self.metadata.append(metadata)
        return str(metadata["template_id"])


class TemplateMcpApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_cached_markdown_preview_is_renderable_not_a_text_excerpt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            template = data_root / "templates" / "cache" / "github-test" / "resume.md"
            template.parent.mkdir(parents=True)
            template.write_text("# 中文简历\n\n## 项目经历\n", encoding="utf-8")
            app_state = SimpleNamespace(settings=SimpleNamespace(data_root=data_root))
            with patch("app.main.state", return_value=app_state):
                rendered = await _attach_template_visual_preview(
                    {"path": str(template), "file_type": "markdown"},
                    requested=True,
                )

        self.assertEqual(rendered["visual_preview"]["status"], "ready")
        self.assertEqual(rendered["visual_preview"]["mime_type"], "text/markdown")
        self.assertIn("## 项目经历", rendered["visual_preview"]["content"])

    async def test_docx_preview_falls_back_to_safe_browser_page_without_office(self) -> None:
        from docx import Document

        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            template = data_root / "templates" / "cache" / "github-test" / "中文简历模板.docx"
            template.parent.mkdir(parents=True)
            document = Document()
            document.add_heading("中文技术简历", level=1)
            document.add_paragraph("个人信息 项目经历 专业技能 教育经历")
            document.save(template)
            app_state = SimpleNamespace(settings=SimpleNamespace(data_root=data_root))
            unavailable = PreviewResult(
                "pdf",
                "dependency_missing",
                None,
                None,
                None,
                ("no office",),
            )
            with (
                patch("app.main.state", return_value=app_state),
                patch("app.main.convert_docx_to_pdf", return_value=unavailable),
            ):
                rendered = await _attach_template_visual_preview(
                    {"path": str(template), "file_type": "docx"},
                    requested=True,
                )

        visual = rendered["visual_preview"]
        self.assertEqual(visual["status"], "ready")
        self.assertEqual(visual["mime_type"], "text/html")
        self.assertTrue(visual["approximate_layout"])
        self.assertIn("docx-page", visual["content"])

    async def test_search_and_download_use_github_mcp(self) -> None:
        manager = _GitHubManager()
        mirror = _TemplateMirror()
        app_state = SimpleNamespace(
            mcp_manager=manager,
            mcp_startup_errors={},
            db_mirror=mirror,
        )
        with patch("app.main.state", return_value=app_state):
            searched = await search_templates({"query": "中文技术简历"}, None)
            self.assertEqual(searched["source"], "github_mcp")
            item = searched["items"][0]
            downloaded = await download_template(
                item["template_id"],
                {"source_url": item["source_url"], "repository": "example/resume"},
                None,
            )

        self.assertEqual(downloaded["source"], "github_mcp")
        self.assertEqual(mirror.metadata[0]["license"], "MIT")
        self.assertEqual(
            [tool for _, tool, _ in manager.calls],
            ["github_search_resume_templates", "github_download_template"],
        )


if __name__ == "__main__":
    unittest.main()
