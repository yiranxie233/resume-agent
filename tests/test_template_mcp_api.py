from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.main import download_template, search_templates


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
