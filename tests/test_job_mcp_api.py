from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.schemas import JobInput, JobUrlRequest
from app.core.store import InMemoryStore
from app.main import (
    close_job_browser,
    environment_recheck,
    job_from_url,
    open_job_in_collection_browser,
    save_job_corrections,
    search_jobs,
)


class _BossManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    async def start_server(self, name: str) -> None:
        return None

    async def health(self, *, probe: bool = False) -> dict:
        return {"boss": {"name": "boss", "status": "ready"}}

    async def call_tool(self, server: str, tool: str, arguments: dict) -> dict:
        self.calls.append((server, tool, dict(arguments)))
        if tool == "boss_environment":
            return {
                "ok": True,
                "data": {
                    "status": "ready",
                    "playwright_installed": True,
                    "edge_path": "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
                    "profile_dir": "C:/resume-agent/data/edge-profile",
                    "visible": True,
                    "running": False,
                },
            }
        if tool == "boss_search_jobs":
            return {
                "ok": True,
                "data": {
                    "items": [
                        {
                            "title": "Python后端工程师",
                            "company": "示例公司",
                            "city": "深圳·南山",
                            "salary": "20-30K",
                            "skills": ["Python", "FastAPI"],
                            "hr_activity": "在线",
                            "posted_at": None,
                            "posted_at_label": "发布时间未知",
                            "url": "https://www.zhipin.com/job_detail/example.html",
                        },
                        {
                            "title": "AI 智能体开发工程师",
                            "company": "另一家公司",
                            "city": "深圳·福田",
                            "salary": "25-35K",
                            "skills": ["Python", "LangGraph"],
                            "hr_activity": "活跃时间待解析",
                            "posted_at": None,
                            "posted_at_label": "发布时间未知",
                            "url": "https://www.zhipin.com/job_detail/example-2.html",
                        },
                    ],
                    "next_cursor": "2",
                },
            }
        if tool == "boss_close_browser":
            return {"ok": True, "data": {"status": "closed"}}
        if tool == "boss_open_job_page":
            return {
                "ok": True,
                "data": {
                    "status": "opened",
                    "url": arguments["url"],
                    "browser": "collection_edge",
                },
            }
        return {
            "ok": True,
            "data": {
                "title": "Python 后端工程师",
                "company": "示例公司",
                "city": "深圳·南山",
                "salary": "20-30K",
                "responsibilities": ["负责 FastAPI 服务开发"],
                "requirements": ["熟悉 Python"],
                "skills": ["Python", "FastAPI"],
                "hr_activity": "本周活跃",
                "posted_at": None,
                "posted_at_label": "发布时间未知",
                "source_url": "https://www.zhipin.com/job_detail/example.html",
                "raw_text": "负责 FastAPI 服务开发\n熟悉 Python",
            },
        }


class _Mirror:
    def __init__(self) -> None:
        self.persisted: list[str] = []

    def persist_job(self, job_id: str, job: object) -> str:
        self.persisted.append(job_id)
        return f"snapshot-{job_id}-{len(self.persisted)}"


class _LoginRequiredBossManager(_BossManager):
    async def call_tool(self, server: str, tool: str, arguments: dict) -> dict:
        if tool == "boss_search_jobs":
            return {
                "ok": False,
                "data": {"message": "请先登录"},
                "error_code": "boss_login_required",
                "requires_user": True,
            }
        return await super().call_tool(server, tool, arguments)


class JobMcpApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_boss_recheck_includes_real_browser_prerequisites(self) -> None:
        manager = _BossManager()
        app_state = SimpleNamespace(
            mcp_manager=manager,
            mcp_startup_errors={},
        )

        with patch("app.main.state", return_value=app_state):
            result = await environment_recheck("boss", None)

        self.assertEqual(result["servers"]["boss"]["status"], "ready")
        self.assertEqual(result["servers"]["boss"]["browser"]["status"], "ready")
        self.assertTrue(result["servers"]["boss"]["browser"]["playwright_installed"])
        self.assertIn(("boss", "boss_environment", {}), manager.calls)

    async def test_search_deduplicates_and_detail_updates_same_job(self) -> None:
        manager = _BossManager()
        app_state = SimpleNamespace(
            store=InMemoryStore(),
            db_mirror=_Mirror(),
            mcp_manager=manager,
            mcp_startup_errors={},
            consents={"post_time_risk:global": {"granted": True}},
            job_parse_confirmations={},
        )
        with patch("app.main.state", return_value=app_state):
            first = await search_jobs({"title": "Python 后端", "city": "深圳", "limit": 20}, None)
            second = await search_jobs({"title": "Python 后端", "city": "深圳", "limit": 20}, None)
            self.assertEqual(first["source"], "boss_mcp")
            self.assertEqual(first["next_cursor"], "2")
            self.assertEqual(len(first["jobs"]), 2)
            self.assertEqual(first["jobs"][0]["posted_at_label"], "发布时间未知")
            self.assertEqual(first["jobs"][0]["hr_activity"], "在线")
            self.assertEqual(first["jobs"][0]["job_id"], second["jobs"][0]["job_id"])
            self.assertEqual(len(app_state.store.list_jobs()), 2)

            detail = await job_from_url(
                JobUrlRequest(url="https://www.zhipin.com/job_detail/example.html"), None
            )
            self.assertEqual(detail["status"], "ready")
            self.assertEqual(detail["job_id"], first["jobs"][0]["job_id"])
            current = app_state.store.get_job(detail["job_id"])
            self.assertEqual(current.responsibilities, ["负责 FastAPI 服务开发"])
            # The recruiter state split directly from the search card is
            # authoritative. The asynchronous detail pane may still contain
            # the previously selected job and must not overwrite it.
            self.assertEqual(current.hr_activity, "在线")
            self.assertEqual(len(app_state.store.list_jobs()), 2)

            # A later authoritative card render may contain a newer activity
            # value and must replace the stale card value saved previously.
            manager.calls.clear()
            original_call = manager.call_tool

            async def refreshed_card(server: str, tool: str, arguments: dict) -> dict:
                response = await original_call(server, tool, arguments)
                if tool == "boss_search_jobs":
                    response["data"]["items"][0]["hr_name"] = "曹先生"
                    response["data"]["items"][0]["hr_activity"] = "3日内活跃"
                return response

            manager.call_tool = refreshed_card  # type: ignore[method-assign]
            refreshed = await search_jobs(
                {"title": "Python 后端", "city": "深圳", "limit": 20}, None
            )
            self.assertEqual(refreshed["jobs"][0]["hr_name"], "曹先生")
            self.assertEqual(refreshed["jobs"][0]["hr_activity"], "3日内活跃")

            # A partially rendered virtual card is not an authoritative HR
            # pair.  Never combine its new activity with a name saved from an
            # earlier complete card, because that produces a pair which was
            # never shown together on BOSS.
            async def activity_without_name(server: str, tool: str, arguments: dict) -> dict:
                response = await original_call(server, tool, arguments)
                if tool == "boss_search_jobs":
                    response["data"]["items"][0].pop("hr_name", None)
                    response["data"]["items"][0]["hr_activity"] = "今日活跃"
                return response

            manager.call_tool = activity_without_name  # type: ignore[method-assign]
            partial = await search_jobs(
                {"title": "Python 后端", "city": "深圳", "limit": 20}, None
            )
            self.assertEqual(partial["jobs"][0]["hr_name"], "曹先生")
            self.assertEqual(partial["jobs"][0]["hr_activity"], "3日内活跃")

            # A later card render may omit the recruiter state. It must not
            # erase the precise value already obtained from the detail page.
            third = await search_jobs(
                {"title": "AI 智能体", "city": "深圳", "limit": 20}, None
            )
            unknown_card = next(
                item for item in third["jobs"] if item["source_url"].endswith("example-2.html")
            )
            unknown_job_id = unknown_card["job_id"]
            known = app_state.store.get_job(unknown_job_id).model_copy(
                update={"hr_activity": "3天内活跃"}
            )
            app_state.store.update_job(unknown_job_id, known)
            fourth = await search_jobs(
                {"title": "AI 智能体", "city": "深圳", "limit": 20}, None
            )
            preserved = next(
                item for item in fourth["jobs"] if item["job_id"] == unknown_job_id
            )
            self.assertEqual(preserved["hr_activity"], "3天内活跃")

    async def test_close_browser_delegates_without_clearing_login(self) -> None:
        manager = _BossManager()
        app_state = SimpleNamespace(
            mcp_manager=manager,
            mcp_startup_errors={},
        )

        with patch("app.main.state", return_value=app_state):
            result = await close_job_browser(None)

        self.assertEqual(result, {"status": "closed"})
        self.assertIn(("boss", "boss_close_browser", {}), manager.calls)

    async def test_job_link_uses_collection_edge_tool(self) -> None:
        manager = _BossManager()
        app_state = SimpleNamespace(
            mcp_manager=manager,
            mcp_startup_errors={},
        )
        url = "https://www.zhipin.com/job_detail/example.html"

        with patch("app.main.state", return_value=app_state):
            result = await open_job_in_collection_browser({"url": url}, None)

        self.assertEqual(result["browser"], "collection_edge")
        self.assertIn(("boss", "boss_open_job_page", {"url": url}), manager.calls)

    async def test_logged_out_search_never_returns_stale_cached_jobs(self) -> None:
        manager = _LoginRequiredBossManager()
        store = InMemoryStore()
        store.save_job(
            JobInput(
                title="历史 Python 岗位",
                city="深圳",
                raw_text="这是旧缓存",
            )
        )
        app_state = SimpleNamespace(
            store=store,
            db_mirror=None,
            mcp_manager=manager,
            mcp_startup_errors={},
            consents={"post_time_risk:global": {"granted": True}},
            job_parse_confirmations={},
        )

        with patch("app.main.state", return_value=app_state):
            result = await search_jobs(
                {"title": "Python", "city": "深圳", "limit": 20}, None
            )

        self.assertEqual(result["status"], "waiting_user")
        self.assertEqual(result["error_code"], "boss_login_required")
        self.assertEqual(result["jobs"], [])
        self.assertIsNone(result["next_cursor"])

    async def test_human_job_corrections_update_only_editable_jd_fields(self) -> None:
        store = InMemoryStore()
        job_id = store.save_job(
            JobInput(
                title="AI 工程师",
                company="示例公司",
                city="深圳",
                responsibilities=["旧职责"],
                requirements=["旧要求"],
                skills=["Python"],
                raw_text="旧 JD",
            )
        )
        app_state = SimpleNamespace(
            store=store,
            db_mirror=None,
            job_parse_confirmations={job_id: True},
        )

        with patch("app.main.state", return_value=app_state):
            result = await save_job_corrections(
                job_id,
                {
                    "raw_text": "岗位职责\n负责智能体开发\n任职要求\n熟悉 LangGraph",
                    "edited_responsibilities": ["负责智能体开发"],
                    "edited_requirements": ["熟悉 LangGraph"],
                    "edited_skills": ["Python", "LangGraph"],
                },
                None,
            )

        current = store.get_job(job_id)
        self.assertEqual(result["parser"]["parser_mode"], "human_corrected")
        self.assertEqual(current.company, "示例公司")
        self.assertEqual(current.responsibilities, ["负责智能体开发"])
        self.assertEqual(current.requirements, ["熟悉 LangGraph"])
        self.assertEqual(current.skills, ["Python", "LangGraph"])
        self.assertFalse(app_state.job_parse_confirmations[job_id])

if __name__ == "__main__":
    unittest.main()
