from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.mcp import boss_server
from app.mcp.edge_adapter import (
    EdgeActionRequired,
    EdgeAdapterConfig,
    EdgeBossAdapter,
    PlaywrightEdgeBrowser,
    blank_browser_page,
    decode_boss_obfuscated_text,
    login_required,
    normalize_city_code,
    same_job_search_page,
)


class FakeBrowser:
    def __init__(self) -> None:
        self.running = False
        self.opened: list[str] = []
        self.manual_opened: list[str] = []
        self.cleared = False
        self.close_calls = 0
        self.adopt_calls = 0
        self.detach_calls = 0
        self.auxiliary_opened: list[str] = []
        self.user_tabs_opened: list[str] = []
        self.manual_login_running = False

    def open(self, url: str) -> None:
        self.running = True
        self.opened.append(url)

    def open_manual_login(self, url: str) -> None:
        self.manual_login_running = True
        self.manual_opened.append(url)

    def open_auxiliary(self, url: str) -> None:
        self.auxiliary_opened.append(url)

    def open_user_tab(self, url: str) -> None:
        self.user_tabs_opened.append(url)

    def close_auxiliary(self) -> None:
        return None

    def select_search_page(self, target_url: str) -> bool:
        return any(same_job_search_page(url, target_url) for url in self.manual_opened)

    def adopt_manual_login(self) -> None:
        self.adopt_calls += 1
        self.running = True

    def detach_manual_login(self) -> None:
        self.detach_calls += 1
        self.running = False

    def close_manual_login(self) -> None:
        self.manual_login_running = False

    def page_state(self) -> dict[str, object]:
        current_url = self.opened[-1] if self.opened else self.manual_opened[-1]
        return {"url": current_url, "challenge": None, "needs_login": False, "body_text": ""}

    def search_items(self) -> list[dict[str, object]]:
        return [
            {
                "title": "Python 后端工程师",
                "company": "示例公司",
                "city": "北京·海淀区",
                "salary": "20-30K",
                "url": "https://www.zhipin.com/job_detail/example.html",
                "tags": ["Python", "FastAPI"],
                "hr_activity": "在线",
            }
        ]

    def load_more_items(self, seen_urls: set[str]) -> list[dict[str, object]]:
        return [
            *self.search_items(),
            {
                "title": "AI 智能体开发工程师",
                "company": "新增公司",
                "city": "北京·朝阳区",
                "salary": "25-35K",
                "url": "https://www.zhipin.com/job_detail/example-2.html",
                "tags": ["Python", "LangGraph"],
                "hr_activity": "活跃时间待解析",
            },
        ]

    def detail_fields(self) -> dict[str, object]:
        return {
            "title": "Python 后端工程师",
            "company": "示例公司",
            "city": "北京",
            "salary": "20-30K",
            "description": "岗位职责：开发智能体服务",
            "posted_label": "昨天",
            "hr_activity": "本周活跃",
            "body_text": "",
        }

    def sanitized_html(self) -> str:
        return "<!doctype html><html><body>职位</body></html>"

    def clear_session(self) -> None:
        self.cleared = True
        self.running = False

    def close(self) -> None:
        self.close_calls += 1
        self.running = False
        self.manual_login_running = False


class DelayedLoginBrowser(FakeBrowser):
    def __init__(self) -> None:
        super().__init__()
        self.state_checks = 0

    def page_state(self) -> dict[str, object]:
        self.state_checks += 1
        current_url = self.opened[-1] if self.opened else self.manual_opened[-1]
        return {
            "url": current_url,
            "challenge": None,
            "needs_login": self.state_checks > 1,
            "body_text": "",
        }

    def search_items(self) -> list[dict[str, object]]:
        return []


class LoginBrowser(FakeBrowser):
    def __init__(self) -> None:
        super().__init__()
        self.logged_in = False

    def page_state(self) -> dict[str, object]:
        current_url = self.opened[-1] if self.opened else self.manual_opened[-1]
        return {
            "url": current_url,
            "challenge": None,
            "needs_login": not self.logged_in,
            "body_text": "",
        }


class InitiallyBlankBrowser(FakeBrowser):
    def __init__(self) -> None:
        super().__init__()
        self.first_state = True

    def page_state(self) -> dict[str, object]:
        if self.first_state:
            self.first_state = False
            return {
                "url": "about:blank",
                "challenge": None,
                "needs_login": False,
                "body_text": "",
            }
        return super().page_state()


class RetriableLoadBrowser(FakeBrowser):
    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0

    def load_more_items(self, seen_urls: set[str]) -> dict[str, object]:
        self.load_calls += 1
        items = self.search_items()
        if self.load_calls == 1:
            return {"items": items, "exhausted": False, "load_pending": True}
        return {
            "items": [
                *items,
                {
                    "title": "延迟加载的 AI 岗位",
                    "company": "新公司",
                    "city": "北京",
                    "salary": "30-40K",
                    "url": "https://www.zhipin.com/job_detail/delayed.html",
                    "tags": ["Python"],
                    "hr_activity": "刚刚活跃",
                },
            ],
            "exhausted": False,
            "load_pending": False,
        }


class ExhaustedLoadBrowser(FakeBrowser):
    def load_more_items(self, seen_urls: set[str]) -> dict[str, object]:
        return {"items": self.search_items(), "exhausted": True, "load_pending": False}


class EdgeAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.browser = FakeBrowser()
        self.adapter = EdgeBossAdapter(
            EdgeAdapterConfig(profile_dir=root / "profile", snapshot_dir=root / "snapshots"),
            browser=self.browser,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_search_uses_numeric_city_code_not_chinese_name(self) -> None:
        result = self.adapter.search("Python", "北京", limit=20)
        search_url = self.browser.opened[-1] if self.browser.opened else self.browser.manual_opened[-1]
        self.assertIn("/web/geek/jobs?", search_url)
        self.assertIn("city=101010100", search_url)
        self.assertNotIn("city=%E5%8C%97%E4%BA%AC", search_url)
        self.assertEqual(result["city_resolution"], "builtin")
        self.assertEqual(result["items"][0]["posted_at_label"], "发布时间未知")
        self.assertEqual(result["items"][0]["hr_activity"], "在线")

    def test_explicit_city_code_and_unknown_city_boundary(self) -> None:
        code, source = normalize_city_code("自定义城市", "123456789")
        self.assertEqual((code, source), ("123456789", "explicit"))
        with self.assertRaisesRegex(ValueError, "9 位数字"):
            normalize_city_code("北京", "beijing")
        with self.assertRaises(EdgeActionRequired) as caught:
            normalize_city_code("未收录城市")
        self.assertEqual(caught.exception.code, "boss_city_code_required")

    def test_login_detection_does_not_scan_generic_page_copy(self) -> None:
        self.assertFalse(login_required("https://www.zhipin.com/", visible_login_control=False))
        self.assertTrue(login_required("https://www.zhipin.com/", visible_login_control=True))
        self.assertTrue(
            login_required(
                "https://www.zhipin.com/web/user/?ka=header-login",
                visible_login_control=False,
            )
        )

    def test_search_page_comparison_decodes_query_and_space_variants(self) -> None:
        target = "https://www.zhipin.com/web/geek/jobs?query=Python+%E5%90%8E%E7%AB%AF&city=101010100&page=1"
        current = "https://www.zhipin.com/web/geek/jobs?query=Python%20%E5%90%8E%E7%AB%AF&city=101010100&page=1"
        self.assertTrue(same_job_search_page(current, target))
        self.assertTrue(
            same_job_search_page(
                "https://www.zhipin.com/web/geek/jobs?query=Python%20%E5%90%8E%E7%AB%AF&city=101010100&page=4",
                target,
            )
        )
        self.assertFalse(
            same_job_search_page(
                "https://www.zhipin.com/web/geek/jobs?query=Java&city=101010100&page=1",
                target,
            )
        )

    def test_blank_page_detection_only_matches_empty_startup_pages(self) -> None:
        self.assertTrue(blank_browser_page("about:blank"))
        self.assertTrue(blank_browser_page("edge://newtab/"))
        self.assertFalse(blank_browser_page("https://www.zhipin.com/"))

    def test_search_navigates_an_initial_blank_tab_once(self) -> None:
        root = Path(self.temp.name)
        browser = InitiallyBlankBrowser()
        adapter = EdgeBossAdapter(
            EdgeAdapterConfig(profile_dir=root / "blank-profile", snapshot_dir=root / "blank-snapshots"),
            browser=browser,
        )

        result = adapter.search("Python", "北京")

        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(len(browser.manual_opened), 1)
        self.assertEqual(len(browser.opened), 1)
        self.assertTrue(same_job_search_page(browser.opened[0], browser.manual_opened[0]))

    def test_salary_webfont_digits_are_decoded(self) -> None:
        self.assertEqual(
            decode_boss_obfuscated_text("\ue032\ue036-\ue033\ue031K·\ue032\ue036薪"),
            "15-20K·15薪",
        )
        self.assertEqual(
            decode_boss_obfuscated_text("\ue032\ue031\ue031-\ue033\ue031\ue031元/天"),
            "100-200元/天",
        )

    def test_empty_search_rechecks_delayed_login_gate(self) -> None:
        root = Path(self.temp.name)
        browser = DelayedLoginBrowser()
        adapter = EdgeBossAdapter(
            EdgeAdapterConfig(profile_dir=root / "delayed-profile", snapshot_dir=root / "delayed-snapshots"),
            browser=browser,
        )

        with self.assertRaises(EdgeActionRequired) as caught:
            adapter.search("Python", "北京")

        self.assertEqual(caught.exception.code, "boss_login_required")
        self.assertFalse(browser.running)
        self.assertTrue(browser.manual_login_running)
        self.assertEqual(browser.state_checks, 2)

    def test_login_window_stays_open_after_successful_search(self) -> None:
        root = Path(self.temp.name)
        browser = LoginBrowser()
        adapter = EdgeBossAdapter(
            EdgeAdapterConfig(profile_dir=root / "login-profile", snapshot_dir=root / "login-snapshots"),
            browser=browser,
        )

        with self.assertRaises(EdgeActionRequired) as caught:
            adapter.search("Python", "北京")

        self.assertEqual(caught.exception.code, "boss_login_required")
        self.assertFalse(browser.running)
        self.assertTrue(browser.manual_login_running)
        self.assertEqual(browser.opened, [])
        self.assertEqual(len(browser.manual_opened), 1)
        self.assertIn("/web/geek/jobs?", browser.manual_opened[0])
        self.assertEqual(browser.detach_calls, 1)

        browser.logged_in = True
        result = adapter.search("Python", "北京")

        self.assertEqual(len(result["items"]), 1)
        self.assertFalse(browser.running)
        self.assertTrue(browser.manual_login_running)
        self.assertEqual(browser.adopt_calls, 2)
        self.assertEqual(browser.detach_calls, 2)
        self.assertEqual(browser.close_calls, 0)

    def test_repeated_authenticated_search_reuses_collection_edge(self) -> None:
        self.adapter.search("Python", "北京")
        self.adapter.search("Python", "北京")

        self.assertEqual(len(self.browser.manual_opened), 1)
        self.assertTrue(all("/web/geek/jobs?" in url for url in self.browser.manual_opened))
        self.assertEqual(self.browser.adopt_calls, 2)
        self.assertEqual(self.browser.close_calls, 0)

    def test_changed_search_navigates_existing_collection_edge(self) -> None:
        self.adapter.search("Python", "北京")
        self.adapter.search("Java", "北京")

        self.assertEqual(len(self.browser.manual_opened), 1)
        self.assertEqual(len(self.browser.opened), 1)
        self.assertIn("query=Java", self.browser.opened[0])
        self.assertEqual(self.browser.close_calls, 0)

    def test_load_more_scrolls_same_search_and_returns_only_unseen_jobs(self) -> None:
        first = self.adapter.search("Python", "北京")
        second = self.adapter.search("Python", "北京", cursor=first["next_cursor"])

        self.assertEqual(first["next_cursor"], "2")
        self.assertEqual([item["title"] for item in second["items"]], ["AI 智能体开发工程师"])
        self.assertEqual(second["next_cursor"], "3")
        self.assertEqual(len(self.browser.manual_opened), 1)
        self.assertEqual(self.browser.adopt_calls, 2)
        self.assertEqual(self.browser.detach_calls, 2)
        self.assertTrue(self.browser.manual_login_running)

        closed = self.adapter.close_browser()

        self.assertEqual(closed["status"], "closed")
        self.assertFalse(self.browser.manual_login_running)

    def test_slow_load_keeps_same_cursor_and_retry_can_succeed(self) -> None:
        root = Path(self.temp.name)
        browser = RetriableLoadBrowser()
        adapter = EdgeBossAdapter(
            EdgeAdapterConfig(profile_dir=root / "retry-profile", snapshot_dir=root / "retry-snapshots"),
            browser=browser,
        )
        first = adapter.search("Python", "北京")
        pending = adapter.search("Python", "北京", cursor=first["next_cursor"])

        self.assertEqual(pending["items"], [])
        self.assertTrue(pending["load_pending"])
        self.assertEqual(pending["next_cursor"], "2")

        loaded = adapter.search("Python", "北京", cursor=pending["next_cursor"])
        self.assertEqual([item["title"] for item in loaded["items"]], ["延迟加载的 AI 岗位"])
        self.assertEqual(loaded["next_cursor"], "3")
        self.assertFalse(loaded["load_pending"])

    def test_next_cursor_is_cleared_only_after_explicit_end(self) -> None:
        root = Path(self.temp.name)
        browser = ExhaustedLoadBrowser()
        adapter = EdgeBossAdapter(
            EdgeAdapterConfig(profile_dir=root / "end-profile", snapshot_dir=root / "end-snapshots"),
            browser=browser,
        )
        first = adapter.search("Python", "北京")
        ended = adapter.search("Python", "北京", cursor=first["next_cursor"])

        self.assertTrue(ended["exhausted"])
        self.assertIsNone(ended["next_cursor"])

    def test_detail_uses_temporary_tab_and_preserves_active_search(self) -> None:
        first = self.adapter.search("Python", "北京")

        detail = self.adapter.detail(first["items"][0]["url"])

        self.assertEqual(detail["title"], "Python 后端工程师")
        self.assertEqual(detail["hr_activity"], "本周活跃")
        self.assertEqual(self.browser.auxiliary_opened, [first["items"][0]["url"]])
        self.assertTrue(self.browser.manual_login_running)
        self.assertFalse(self.browser.running)
        second = self.adapter.search("Python", "北京", cursor=first["next_cursor"])
        self.assertEqual(len(second["items"]), 1)

    def test_job_link_opens_in_retained_collection_edge(self) -> None:
        first = self.adapter.search("Python", "北京")

        result = self.adapter.open_job_page(first["items"][0]["url"])

        self.assertEqual(result["browser"], "collection_edge")
        self.assertEqual(self.browser.user_tabs_opened, [first["items"][0]["url"]])
        self.assertTrue(self.browser.manual_login_running)
        self.assertFalse(self.browser.running)

    def test_manual_login_is_adopted_without_closing_it_first(self) -> None:
        browser = PlaywrightEdgeBrowser(Path(self.temp.name) / "adopt-profile", edge_path=__file__)

        class Process:
            @staticmethod
            def poll() -> None:
                return None

        login_process = Process()
        browser._manual_login_process = login_process  # type: ignore[assignment]
        browser._manual_debug_port = 39221
        browser._manual_debug_endpoint = (
            "ws://127.0.0.1:39221/devtools/browser/test-browser"
        )
        blank_page = Mock()
        blank_page.url = "about:blank"
        blank_page.is_closed.return_value = False
        boss_page = Mock()
        boss_page.url = "https://www.zhipin.com/web/user/"
        boss_page.is_closed.return_value = False
        context = Mock()
        context.pages = [blank_page, boss_page]
        controller = Mock()
        controller.contexts = [context]

        with (
            patch.object(
                browser,
                "_discover_debug_endpoint",
                return_value="ws://127.0.0.1:39221/devtools/browser/test-browser",
            ),
            patch.object(browser, "_connect_cdp", return_value=controller) as connect,
        ):
            browser.adopt_manual_login()

        connect.assert_called_once_with(
            "ws://127.0.0.1:39221/devtools/browser/test-browser"
        )
        self.assertIs(browser._page, boss_page)
        self.assertIs(browser._manual_login_process, login_process)
        self.assertTrue(browser.running)

    def test_stale_cached_endpoint_is_not_reported_as_running(self) -> None:
        browser = PlaywrightEdgeBrowser(Path(self.temp.name) / "stale-profile", edge_path=__file__)
        browser.profile_dir.mkdir(parents=True, exist_ok=True)
        browser._manual_debug_port = 39225
        browser._manual_debug_endpoint = (
            "ws://127.0.0.1:39225/devtools/browser/stale-browser"
        )
        browser._debug_port_record_file.write_text("39225", encoding="utf-8")

        with patch.object(browser, "_debug_endpoint_for_port", return_value=None):
            self.assertFalse(browser.manual_login_running)

        self.assertIsNone(browser._manual_debug_endpoint)
        self.assertIsNone(browser._manual_debug_port)
        self.assertFalse(browser._debug_port_record_file.exists())

    def test_adopt_rejects_stale_cached_websocket_without_connecting(self) -> None:
        browser = PlaywrightEdgeBrowser(Path(self.temp.name) / "stale-adopt", edge_path=__file__)
        browser._manual_debug_endpoint = (
            "ws://127.0.0.1:39226/devtools/browser/stale-browser"
        )

        with (
            patch.object(browser, "_discover_debug_endpoint", return_value=None),
            patch.object(browser, "_connect_cdp") as connect,
            self.assertRaises(EdgeActionRequired) as caught,
        ):
            browser.adopt_manual_login()

        self.assertEqual(caught.exception.code, "edge_login_attach_failed")
        connect.assert_not_called()

    def test_manual_login_close_force_kills_tree_when_graceful_close_fails(self) -> None:
        class Process:
            pid = 12345

            @staticmethod
            def poll() -> None:
                return None

            @staticmethod
            def wait(*, timeout: int) -> int:
                return 0

            @staticmethod
            def kill() -> None:
                raise AssertionError("forceful taskkill should finish the process")

        browser = PlaywrightEdgeBrowser(Path(self.temp.name) / "manual-profile", edge_path=__file__)
        browser._manual_login_process = Process()  # type: ignore[assignment]
        responses = [SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)]

        with patch("app.mcp.edge_adapter.subprocess.run", side_effect=responses) as run:
            browser.close_manual_login()

        self.assertEqual(run.call_count, 2)
        self.assertNotIn("/F", run.call_args_list[0].args[0])
        self.assertIn("/F", run.call_args_list[1].args[0])
        self.assertIsNone(browser._manual_login_process)

    def test_manual_login_waits_for_published_debug_port(self) -> None:
        process = Mock()
        process.poll.return_value = None
        browser = PlaywrightEdgeBrowser(Path(self.temp.name) / "settle-profile", edge_path=__file__)

        with (
            patch("app.mcp.edge_adapter.subprocess.Popen", return_value=process) as popen,
            patch.object(browser, "_available_debug_port", return_value=39222),
            patch.object(browser, "_discover_debug_endpoint", return_value=None),
            patch.object(
                browser,
                "_debug_endpoint_for_port",
                return_value="ws://127.0.0.1:39222/devtools/browser/test-browser",
            ),
        ):
            browser.open_manual_login("https://www.zhipin.com/web/geek/jobs?query=Python")

        self.assertEqual(
            browser._manual_debug_endpoint,
            "ws://127.0.0.1:39222/devtools/browser/test-browser",
        )
        command = browser._manual_login_process
        self.assertIs(command, process)
        self.assertIn("--remote-debugging-port=39222", popen.call_args.args[0])
        self.assertNotIn("--remote-debugging-port=0", popen.call_args.args[0])
        self.assertNotIn("--disable-gpu", popen.call_args.args[0])
        self.assertEqual(browser._debug_port_record_file.read_text(encoding="utf-8"), "39222")

    def test_transient_navigation_is_retried_once(self) -> None:
        browser = PlaywrightEdgeBrowser(Path(self.temp.name) / "retry-profile", edge_path=__file__)
        page = Mock()
        page.goto.side_effect = [RuntimeError("net::ERR_ABORTED"), None]
        browser._page = page

        with (
            patch.object(browser, "_ensure_started"),
            patch.object(browser, "close") as close,
            patch("app.mcp.edge_adapter.time.sleep") as sleep,
        ):
            browser.open("https://www.zhipin.com/")

        self.assertEqual(page.goto.call_count, 2)
        sleep.assert_called_once_with(0.3)
        close.assert_not_called()

    def test_search_items_supports_current_boss_card_selectors(self) -> None:
        browser = PlaywrightEdgeBrowser(Path(self.temp.name) / "selector-profile", edge_path=__file__)
        page = Mock()
        page.evaluate.side_effect = [
            [],
            [
                {
                    "title": "Python 后端开发工程师",
                    "company": "示例公司",
                    "city": "北京·海淀区",
                    "salary": "20-30K",
                    "url": "https://www.zhipin.com/job_detail/example.html",
                    "tags": ["1-3年", "本科"],
                }
            ],
        ]
        browser._page = page

        with (
            patch.object(browser, "_ensure_started"),
            patch("app.mcp.edge_adapter.time.sleep"),
        ):
            items = browser.search_items()

        self.assertEqual(items[0]["company"], "示例公司")
        script = page.evaluate.call_args_list[0].args[0]
        self.assertIn(".job-card-box", script)
        self.assertIn(".job-salary", script)
        self.assertIn(".boss-name", script)
        self.assertIn(".boss-online-icon", script)
        self.assertIn(".company-location", script)

    def test_search_items_follows_boss_replacement_tab_without_navigation(self) -> None:
        browser = PlaywrightEdgeBrowser(Path(self.temp.name) / "replacement-profile", edge_path=__file__)
        stale = Mock()
        stale.evaluate.side_effect = RuntimeError("Target page has been closed")
        replacement = Mock()
        replacement.url = "https://www.zhipin.com/web/geek/jobs?query=Python"
        replacement.is_closed.return_value = False
        replacement.evaluate.return_value = [
            {
                "title": "Python 后端",
                "url": "https://www.zhipin.com/job_detail/new.html",
            }
        ]
        browser._page = stale
        browser._context = Mock()
        browser._context.pages = [replacement]

        with (
            patch.object(browser, "_ensure_started"),
            patch("app.mcp.edge_adapter.time.sleep"),
        ):
            items = browser.search_items()

        self.assertEqual(items[0]["title"], "Python 后端")
        self.assertIs(browser._page, replacement)

    def test_page_state_follows_boss_replacement_tab(self) -> None:
        browser = PlaywrightEdgeBrowser(Path(self.temp.name) / "state-profile", edge_path=__file__)
        stale = Mock()
        stale.url = "https://www.zhipin.com/web/geek/jobs?query=Python"
        stale.evaluate.side_effect = RuntimeError(
            "Target page, context or browser has been closed"
        )
        replacement = Mock()
        replacement.url = "https://www.zhipin.com/web/geek/jobs?query=Python"
        replacement.is_closed.return_value = False
        replacement.evaluate.side_effect = [
            {"url": replacement.url, "body": "岗位列表"},
            False,
        ]
        browser._page = stale
        browser._context = Mock()
        browser._context.pages = [replacement]

        with patch.object(browser, "_ensure_started"):
            state = browser.page_state()

        self.assertEqual(state["url"], replacement.url)
        self.assertIs(browser._page, replacement)

    def test_load_more_items_scrolls_until_an_unseen_url_appears(self) -> None:
        browser = PlaywrightEdgeBrowser(Path(self.temp.name) / "scroll-profile", edge_path=__file__)
        browser._page = Mock()
        seen = {
            "title": "Python 后端",
            "url": "https://www.zhipin.com/job_detail/seen.html",
        }
        new = {
            "title": "AI 智能体开发",
            "url": "https://www.zhipin.com/job_detail/new.html",
        }

        with (
            patch.object(browser, "_ensure_started"),
            patch.object(browser, "_evaluate_search_items", side_effect=[[seen], [seen, new]]),
            patch("app.mcp.edge_adapter.time.sleep"),
        ):
            result = browser.load_more_items({seen["url"]})

        self.assertEqual(result["items"], [seen, new])
        self.assertFalse(result["load_pending"])
        browser._page.evaluate.assert_called_once()

    def test_load_more_honors_visible_explicit_end_marker(self) -> None:
        browser = PlaywrightEdgeBrowser(Path(self.temp.name) / "end-marker-profile", edge_path=__file__)
        browser._page = Mock()
        seen = {
            "title": "Python 后端",
            "url": "https://www.zhipin.com/job_detail/seen.html",
        }
        browser._page.evaluate.return_value = {"explicit_end": True}

        with (
            patch.object(browser, "_ensure_started"),
            patch.object(browser, "_evaluate_search_items", return_value=[seen]),
        ):
            result = browser.load_more_items({seen["url"]})

        self.assertTrue(result["exhausted"])
        self.assertFalse(result["load_pending"])

    def test_closed_session_error_reopens_stable_manual_window(self) -> None:
        error = EdgeActionRequired("edge_session_closed", "closed")

        self.adapter._close_for_error(error)

        self.assertFalse(self.browser.running)
        self.assertTrue(self.browser.manual_login_running)
        self.assertEqual(self.browser.manual_opened, ["https://www.zhipin.com/"])

    def test_post_time_reports_independent_method(self) -> None:
        result = self.adapter.extract_post_time(text="发布时间：昨天")
        self.assertEqual(result["method"], "independent_dom_metadata_v1")
        self.assertEqual(result["confidence"], "rule_verified")
        self.assertNotEqual(result["posted_at_label"], "发布时间未知")


class BossMCPDelegationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = boss_server._adapter
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.browser = FakeBrowser()
        boss_server._adapter = EdgeBossAdapter(
            EdgeAdapterConfig(profile_dir=root / "profile", snapshot_dir=root / "snapshots"),
            browser=self.browser,
        )

    def tearDown(self) -> None:
        boss_server._adapter = self.previous
        self.temp.cleanup()

    def test_search_tool_delegates_to_edge_adapter(self) -> None:
        response = boss_server.search_jobs(
            {"title": "Python", "city": "北京", "city_code": "101010100", "limit": 20}
        )
        self.assertTrue(response["ok"])
        self.assertEqual(response["data"]["items"][0]["title"], "Python 后端工程师")
        search_url = self.browser.opened[-1] if self.browser.opened else self.browser.manual_opened[-1]
        self.assertIn("city=101010100", search_url)

    def test_environment_tool_checks_prerequisites_without_navigation(self) -> None:
        response = boss_server.environment({})

        self.assertTrue(response["ok"])
        self.assertIn(response["data"]["status"], {"ready", "needs_setup"})
        self.assertEqual(self.browser.opened, [])

    def test_manual_gate_keeps_machine_readable_error(self) -> None:
        response = boss_server.search_jobs({"title": "Python", "city": "未收录城市"})
        self.assertFalse(response["ok"])
        self.assertEqual(response["error_code"], "boss_city_code_required")
        self.assertTrue(response["requires_user"])

    def test_detail_snapshot_reparse_and_logout_are_real_calls(self) -> None:
        url = "https://www.zhipin.com/job_detail/example.html"
        detail = boss_server.get_detail({"url": url})
        self.assertTrue(detail["ok"])
        self.assertEqual(detail["data"]["title"], "Python 后端工程师")

        snapshot = boss_server.snapshot({"url": url})
        self.assertTrue(snapshot["ok"])
        self.assertTrue(Path(snapshot["data"]["path"]).exists())

        reparsed = boss_server.reparse({"url": url})
        self.assertTrue(reparsed["ok"])
        self.assertEqual(reparsed["data"]["status"], "reparsed")

        logged_out = boss_server.logout({})
        self.assertTrue(logged_out["ok"])
        self.assertTrue(self.browser.cleared)


if __name__ == "__main__":
    unittest.main()
