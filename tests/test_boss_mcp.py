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
    split_recruiter_card_text,
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
        self.selected_search_urls: list[str] = []
        self.restored_search_pages: list[tuple[str, str | None]] = []
        self.manual_login_running = False
        self.minimize_calls = 0

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
        self.selected_search_urls.append(target_url)
        return any(same_job_search_page(url, target_url) for url in self.manual_opened)

    def restore_search_page(
        self, target_url: str, anchor_url: str | None = None
    ) -> bool:
        self.restored_search_pages.append((target_url, anchor_url))
        return self.select_search_page(target_url)

    def adopt_manual_login(self) -> None:
        self.adopt_calls += 1
        self.running = True

    def detach_manual_login(self) -> None:
        self.detach_calls += 1
        self.running = False

    def close_manual_login(self) -> None:
        self.manual_login_running = False

    def minimize_collection_window(self) -> None:
        self.minimize_calls += 1

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


class ActivityEnrichBrowser(FakeBrowser):
    def __init__(self) -> None:
        super().__init__()
        self.enrich_calls = 0

    def search_items(self) -> list[dict[str, object]]:
        items = super().search_items()
        items[0]["hr_name"] = "曹先生"
        items[0]["hr_activity"] = "今日活跃"
        return items

    def enrich_search_activities(
        self, items: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        self.enrich_calls += 1
        return [
            {**item, "hr_name": "曹先生", "hr_activity": "3天内活跃"}
            for item in items
        ]


class SurveyCardBrowser(FakeBrowser):
    def search_items(self) -> list[dict[str, object]]:
        return [
            {
                "title": "对搜索是否满意",
                "company": None,
                "city": None,
                "salary": None,
                "url": "https://www.zhipin.com/web/geek/jobs?query=Python",
                "tags": [],
                "hr_name": None,
                "hr_activity": "活跃时间待解析",
            },
            *super().search_items(),
        ]


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

    def test_page_state_detects_visible_header_login_registration_control(self) -> None:
        browser = PlaywrightEdgeBrowser(
            Path(self.temp.name) / "page-state-profile", edge_path=__file__
        )
        page = Mock()
        page.is_closed.return_value = False
        page.url = "https://www.zhipin.com/web/geek/jobs?query=Python&city=101010100"
        page.evaluate.side_effect = [
            {"url": page.url, "body": "职位列表"},
            True,
        ]
        browser._context = Mock()  # type: ignore[assignment]
        browser._page = page  # type: ignore[assignment]

        state = browser.page_state()

        self.assertTrue(state["needs_login"])
        login_probe = page.evaluate.call_args_list[1].args[0]
        self.assertIn("登录/注册", login_probe)
        self.assertIn("header a", login_probe)

    def test_search_replaces_card_hint_with_verified_detail_recruiter_pair(self) -> None:
        root = Path(self.temp.name)
        browser = ActivityEnrichBrowser()
        adapter = EdgeBossAdapter(
            EdgeAdapterConfig(
                profile_dir=root / "activity-profile",
                snapshot_dir=root / "activity-snapshots",
            ),
            browser=browser,
        )

        result = adapter.search("Python", "北京")

        self.assertEqual(result["items"][0]["hr_name"], "曹先生")
        self.assertEqual(result["items"][0]["hr_activity"], "3天内活跃")
        self.assertEqual(browser.enrich_calls, 1)
        self.assertEqual(result["collection"]["status"], "batch_ready")
        self.assertEqual(result["collection"]["batch_size"], 1)
        self.assertEqual(len(browser.manual_opened), 1)
        self.assertEqual(browser.auxiliary_opened, [])

    def test_combined_card_recruiter_label_is_split_without_reparsing(self) -> None:
        name, activity = split_recruiter_card_text("朴先生  半年前活跃", "今日活跃")

        self.assertEqual(name, "朴先生")
        self.assertEqual(activity, "半年前活跃")

    def test_two_week_activity_is_kept_with_its_card_recruiter(self) -> None:
        name, activity = split_recruiter_card_text("李女士 · 招聘经理", "两周内活跃")

        self.assertEqual(name, "李女士")
        self.assertEqual(activity, "两周内活跃")

    def test_inline_online_badge_is_split_from_recruiter_name(self) -> None:
        name, activity = split_recruiter_card_text(
            "程女士在线 博彦科技 · hr", "活跃时间待解析"
        )

        self.assertEqual(name, "程女士")
        self.assertEqual(activity, "在线")

    def test_recoverable_boss_error_does_not_close_authenticated_edge(self) -> None:
        root = Path(self.temp.name)
        browser = FakeBrowser()
        browser.manual_login_running = True
        browser.running = True
        adapter = EdgeBossAdapter(
            EdgeAdapterConfig(
                profile_dir=root / "recoverable-profile",
                snapshot_dir=root / "recoverable-snapshots",
            ),
            browser=browser,
        )

        adapter._close_for_error(  # noqa: SLF001 - regression at adapter boundary
            EdgeActionRequired("boss_parse_failed", "temporary parse failure")
        )

        self.assertEqual(browser.close_calls, 0)
        self.assertEqual(browser.detach_calls, 1)
        self.assertTrue(browser.manual_login_running)

    def test_survey_card_is_skipped_without_consuming_job_result(self) -> None:
        root = Path(self.temp.name)
        adapter = EdgeBossAdapter(
            EdgeAdapterConfig(
                profile_dir=root / "survey-profile",
                snapshot_dir=root / "survey-snapshots",
            ),
            browser=SurveyCardBrowser(),
        )

        result = adapter.search("Python", "北京", limit=1)

        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["title"], "Python 后端工程师")

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
        self.assertGreaterEqual(self.browser.minimize_calls, 3)

    def test_activity_enrichment_script_uses_same_page_detail_pane(self) -> None:
        browser = PlaywrightEdgeBrowser(
            Path(self.temp.name) / "detail-pane-profile", edge_path=__file__
        )
        page = Mock()
        page.is_closed.return_value = False
        page.evaluate.return_value = {"hr_name": "李女士", "hr_activity": "两周活跃"}
        browser._context = Mock()  # type: ignore[assignment]
        browser._page = page  # type: ignore[assignment]

        result = browser.enrich_search_activities(
            [
                {
                    "url": "https://www.zhipin.com/job_detail/example.html",
                    "title": "Python 工程师",
                    "hr_activity": "活跃时间待解析",
                }
            ]
        )

        self.assertEqual(result[0]["hr_name"], "李女士")
        self.assertEqual(result[0]["hr_activity"], "两周活跃")
        script = page.evaluate.call_args.args[0]
        self.assertIn("job-detail-container", script)
        self.assertIn("两|二", script)
        self.assertIn("correctTitle", script)
        self.assertIn("correctCompany", script)
        self.assertIn("程女士在线", script)
        self.assertIn("semanticRecruiterRoots", script)
        self.assertIn("boss-info-box", script)
        self.assertIn("Atomic pair", script)
        self.assertNotIn("window.open", script)

    def test_collection_shutdown_closes_each_tab_in_reverse_order(self) -> None:
        browser = PlaywrightEdgeBrowser(
            Path(self.temp.name) / "close-tabs-profile", edge_path=__file__
        )
        order: list[str] = []

        def page(name: str) -> Mock:
            value = Mock()
            value.is_closed.return_value = False
            value.close.side_effect = lambda: order.append(name)
            return value

        search_page = page("search")
        detail_page = page("detail")
        user_page = page("user")
        context = SimpleNamespace(pages=[search_page, detail_page, user_page])

        closed = browser._close_context_pages(context)

        self.assertEqual(closed, 3)
        self.assertEqual(order, ["user", "detail", "search"])

    def test_search_target_survives_a_frontmost_detail_tab(self) -> None:
        browser = PlaywrightEdgeBrowser(
            Path(self.temp.name) / "retained-search-profile", edge_path=__file__
        )
        search_page = Mock()
        search_page._target_id = "search-target"
        search_page.url = "https://www.zhipin.com/web/geek/jobs?query=Python&city=101010100&page=3"
        search_page.is_closed.return_value = False
        detail_page = Mock()
        detail_page._target_id = "detail-target"
        detail_page.url = "https://www.zhipin.com/job_detail/example.html"
        detail_page.is_closed.return_value = False
        browser._context = SimpleNamespace(pages=[search_page, detail_page])  # type: ignore[assignment]
        browser._page = detail_page  # type: ignore[assignment]
        browser._search_target_id = "search-target"

        selected = browser.select_search_page(
            "https://www.zhipin.com/web/geek/jobs?query=Python&city=101010100&page=1"
        )

        self.assertTrue(selected)
        self.assertIs(browser._page, search_page)

    def test_adopt_manual_login_prefers_cached_search_target_over_new_front_tab(self) -> None:
        browser = PlaywrightEdgeBrowser(
            Path(self.temp.name) / "adopt-search-profile", edge_path=__file__
        )
        search_page = Mock()
        search_page._target_id = "search-target"
        search_page._url = (
            "https://www.zhipin.com/web/geek/jobs?query=Python&city=101010100"
        )
        search_page.is_closed.return_value = False

        class NewFrontTab:
            _target_id = "new-front-target"
            _url = "https://example.com/another-page"

            @staticmethod
            def is_closed() -> bool:
                return False

            @property
            def url(self) -> str:
                raise AssertionError("the unrelated front tab must not be evaluated")

        context = SimpleNamespace(pages=[search_page, NewFrontTab()])
        controller = SimpleNamespace(contexts=[context], stop=Mock())
        browser._search_target_id = "search-target"

        with (
            patch.object(browser, "_discover_debug_endpoint", return_value="ws://edge"),
            patch.object(browser, "_connect_cdp", return_value=controller),
        ):
            browser.adopt_manual_login()

        self.assertIs(browser._page, search_page)
        search_page.set_default_timeout.assert_called_once_with(browser.timeout_ms)

    def test_restore_search_page_activates_tab_and_scrolls_to_last_card(self) -> None:
        browser = PlaywrightEdgeBrowser(
            Path(self.temp.name) / "restore-search-profile", edge_path=__file__
        )
        search_page = Mock()
        search_page._target_id = "search-target"
        search_page.url = (
            "https://www.zhipin.com/web/geek/jobs?query=Python&city=101010100&page=3"
        )
        search_page.is_closed.return_value = False
        search_page.evaluate.return_value = {"found": True, "card_count": 20}
        detail_page = Mock()
        detail_page._target_id = "detail-target"
        detail_page.url = "https://www.zhipin.com/job_detail/example.html"
        detail_page.is_closed.return_value = False
        browser._context = SimpleNamespace(  # type: ignore[assignment]
            pages=[search_page, detail_page]
        )
        browser._page = detail_page  # type: ignore[assignment]
        browser._search_target_id = "search-target"
        anchor = "https://www.zhipin.com/job_detail/last-seen.html"

        restored = browser.restore_search_page(
            "https://www.zhipin.com/web/geek/jobs?query=Python&city=101010100&page=1",
            anchor,
        )

        self.assertTrue(restored)
        self.assertIs(browser._page, search_page)
        search_page.bring_to_front.assert_not_called()
        anchor_script = search_page.evaluate.call_args.args[0]
        self.assertIn("last-seen.html", anchor_script)
        self.assertIn("scrollIntoView", anchor_script)

    def test_restore_search_page_ignores_unrelated_frontmost_tab_without_attaching(self) -> None:
        browser = PlaywrightEdgeBrowser(
            Path(self.temp.name) / "restore-unrelated-tab-profile", edge_path=__file__
        )
        target_url = (
            "https://www.zhipin.com/web/geek/jobs?query=Python&city=101010100&page=1"
        )
        search_page = Mock()
        search_page._target_id = "search-target"
        search_page._url = target_url
        search_page.url = target_url
        search_page.is_closed.return_value = False
        search_page.evaluate.return_value = {"found": True, "card_count": 20}

        class UnrelatedFrontTab:
            _target_id = "new-tab-target"
            _url = "edge://newtab/"

            @staticmethod
            def is_closed() -> bool:
                return False

            @property
            def url(self) -> str:
                raise AssertionError("an unrelated tab must not be attached/evaluated")

        unrelated = UnrelatedFrontTab()
        browser._context = SimpleNamespace(  # type: ignore[assignment]
            pages=[search_page, unrelated]
        )
        browser._page = unrelated  # type: ignore[assignment]
        browser._search_target_id = "search-target"

        restored = browser.restore_search_page(
            target_url,
            "https://www.zhipin.com/job_detail/last-seen.html",
        )

        self.assertTrue(restored)
        self.assertIs(browser._page, search_page)

    def test_restore_search_page_uses_history_when_search_target_navigated_in_place(
        self,
    ) -> None:
        browser = PlaywrightEdgeBrowser(
            Path(self.temp.name) / "restore-history-profile", edge_path=__file__
        )
        target_url = (
            "https://www.zhipin.com/web/geek/jobs?query=Python&city=101010100&page=1"
        )

        class NavigatedSearchPage:
            _target_id = "search-target"

            def __init__(self) -> None:
                self.url = "https://www.zhipin.com/job_detail/example.html"
                self.goto_calls: list[str] = []
                self.front_calls = 0

            @staticmethod
            def is_closed() -> bool:
                return False

            @staticmethod
            def set_default_timeout(_: int) -> None:
                return None

            def evaluate(self, script: str) -> object:
                if "history.back" in script:
                    self.url = target_url
                    return True
                return {"found": True, "card_count": 20}

            def goto(self, url: str, **_: object) -> None:
                self.goto_calls.append(url)
                self.url = url

            def bring_to_front(self) -> None:
                self.front_calls += 1

        page = NavigatedSearchPage()
        browser._context = SimpleNamespace(pages=[page])  # type: ignore[assignment]
        browser._page = page  # type: ignore[assignment]
        browser._search_target_id = "search-target"

        restored = browser.restore_search_page(
            target_url,
            "https://www.zhipin.com/job_detail/example.html",
        )

        self.assertTrue(restored)
        self.assertEqual(page.goto_calls, [])
        self.assertEqual(page.front_calls, 0)

    def test_restore_prefers_intact_search_tab_over_remembered_detail_target(self) -> None:
        browser = PlaywrightEdgeBrowser(
            Path(self.temp.name) / "restore-intact-search-profile", edge_path=__file__
        )
        target_url = (
            "https://www.zhipin.com/web/geek/jobs?query=Python&city=101010100&page=1"
        )
        search_page = Mock()
        search_page._target_id = "search-target"
        search_page.url = target_url
        search_page.is_closed.return_value = False
        search_page.evaluate.return_value = {"found": True, "card_count": 20}
        detail_page = Mock()
        detail_page._target_id = "remembered-detail-target"
        detail_page.url = "https://www.zhipin.com/job_detail/example.html"
        detail_page.is_closed.return_value = False
        browser._context = SimpleNamespace(pages=[search_page, detail_page])  # type: ignore[assignment]
        browser._page = detail_page  # type: ignore[assignment]
        browser._search_target_id = "remembered-detail-target"

        restored = browser.restore_search_page(target_url)

        self.assertTrue(restored)
        self.assertIs(browser._page, search_page)
        self.assertEqual(browser._search_target_id, "search-target")
        detail_page.evaluate.assert_not_called()

    def test_trusted_background_wheel_emulates_active_target_without_fronting_tab(self) -> None:
        page = Mock()
        page.emulate_background_active = Mock()
        page.dispatch_wheel = Mock()

        dispatched = PlaywrightEdgeBrowser._dispatch_trusted_wheel(page, 1600)

        self.assertTrue(dispatched)
        page.emulate_background_active.assert_called_once_with()
        page.dispatch_wheel.assert_called_once_with(1600)
        page.bring_to_front.assert_not_called()

    def test_restore_search_page_reuses_detail_tab_when_original_search_was_closed(
        self,
    ) -> None:
        browser = PlaywrightEdgeBrowser(
            Path(self.temp.name) / "restore-closed-search-profile", edge_path=__file__
        )
        target_url = (
            "https://www.zhipin.com/web/geek/jobs?query=Python&city=101010100&page=1"
        )

        class DetailPage:
            _target_id = "detail-target"

            def __init__(self) -> None:
                self.url = "https://www.zhipin.com/job_detail/example.html"
                self.goto_calls: list[str] = []

            @staticmethod
            def is_closed() -> bool:
                return False

            @staticmethod
            def set_default_timeout(_: int) -> None:
                return None

            def evaluate(self, script: str) -> object:
                if "history.back" in script:
                    return True
                return {"found": False, "card_count": 1}

            def goto(self, url: str, **_: object) -> None:
                self.goto_calls.append(url)
                self.url = url

        page = DetailPage()
        browser._context = SimpleNamespace(pages=[page])  # type: ignore[assignment]
        browser._page = page  # type: ignore[assignment]
        browser._search_target_id = "closed-search-target"

        with patch(
            "app.mcp.edge_adapter.time.monotonic", side_effect=[0.0, 4.0, 4.0]
        ):
            restored = browser.restore_search_page(target_url)

        self.assertTrue(restored)
        self.assertEqual(page.goto_calls, [target_url])
        self.assertIs(browser._page, page)
        self.assertEqual(browser._search_target_id, "detail-target")

    def test_restore_search_page_keeps_valid_page_when_anchor_is_absent(self) -> None:
        browser = PlaywrightEdgeBrowser(
            Path(self.temp.name) / "restore-missing-anchor-profile", edge_path=__file__
        )
        target_url = (
            "https://www.zhipin.com/web/geek/jobs?query=Python&city=101010100&page=1"
        )
        page = Mock()
        page._target_id = "search-target"
        page.url = target_url
        page.is_closed.return_value = False
        page.evaluate.return_value = {"found": False, "card_count": 20}
        browser._context = SimpleNamespace(pages=[page])  # type: ignore[assignment]
        browser._page = page  # type: ignore[assignment]
        browser._search_target_id = "search-target"

        with patch(
            "app.mcp.edge_adapter.time.monotonic", side_effect=[0.0, 5.0]
        ):
            restored = browser.restore_search_page(
                target_url,
                "https://www.zhipin.com/job_detail/no-longer-rendered.html",
            )

        self.assertTrue(restored)
        page.bring_to_front.assert_not_called()
        self.assertIn("cards.at(-1)", page.evaluate.call_args.args[0])

    def test_pagination_supports_legacy_browser_without_restore_method(self) -> None:
        class LegacyBrowser(FakeBrowser):
            restore_search_page = None  # type: ignore[assignment]

        root = Path(self.temp.name)
        browser = LegacyBrowser()
        adapter = EdgeBossAdapter(
            EdgeAdapterConfig(
                profile_dir=root / "legacy-profile",
                snapshot_dir=root / "legacy-snapshots",
            ),
            browser=browser,  # type: ignore[arg-type]
        )

        first = adapter.search("Python", "北京")
        second = adapter.search("Python", "北京", cursor=first["next_cursor"])

        self.assertEqual([item["title"] for item in second["items"]], ["AI 智能体开发工程师"])
        self.assertGreaterEqual(len(browser.selected_search_urls), 1)

    def test_collection_edge_launch_is_minimized(self) -> None:
        browser = PlaywrightEdgeBrowser(
            Path(self.temp.name) / "minimized-profile", edge_path=__file__
        )

        class Process:
            pid = 45678

            @staticmethod
            def poll() -> None:
                return None

        with (
            patch.object(browser, "_available_debug_port", return_value=39231),
            patch.object(browser, "_launch_visible_edge", return_value=Process()) as launch,
            patch.object(browser, "_write_owned_process_record"),
            patch.object(
                browser,
                "_debug_endpoint_for_port",
                return_value="ws://127.0.0.1:39231/devtools/browser/test",
            ),
        ):
            browser.open_manual_login("https://www.zhipin.com/")

        arguments = launch.call_args.args[0]
        self.assertIn("--start-minimized", arguments)
        self.assertIn("--disable-background-timer-throttling", arguments)
        self.assertIn("--disable-backgrounding-occluded-windows", arguments)
        self.assertIn("--disable-renderer-backgrounding", arguments)

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
        self.assertEqual(len(self.browser.restored_search_pages), 1)
        self.assertEqual(
            self.browser.restored_search_pages[0][1],
            first["items"][-1]["url"],
        )

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

    def test_detail_reselects_retained_search_before_login_gate(self) -> None:
        first = self.adapter.search("Python", "北京")
        self.browser.selected_search_urls.clear()

        self.adapter.detail(first["items"][0]["url"])

        self.assertEqual(len(self.browser.selected_search_urls), 1)
        self.assertTrue(
            same_job_search_page(
                self.browser.selected_search_urls[0],
                self.browser.manual_opened[0],
            )
        )

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

    def test_read_only_stale_port_file_does_not_break_environment_detection(self) -> None:
        browser = PlaywrightEdgeBrowser(
            Path(self.temp.name) / "read-only-profile", edge_path=__file__
        )
        browser._manual_debug_port = 39225
        with (
            patch.object(browser, "_discover_debug_endpoint", return_value=None),
            patch.object(
                browser,
                "_discard_runtime_file",
                side_effect=lambda _: None,
            ),
        ):
            self.assertFalse(browser.manual_login_running)

        self.assertIsNone(browser._manual_debug_endpoint)
        self.assertIsNone(browser._manual_debug_port)

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
        process.pid = 12345
        process.poll.return_value = None
        browser = PlaywrightEdgeBrowser(Path(self.temp.name) / "settle-profile", edge_path=__file__)

        with (
            patch.object(
                browser, "_launch_visible_edge", return_value=process
            ) as launch,
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
        arguments = launch.call_args.args[0]
        self.assertIn("--remote-debugging-port=39222", arguments)
        self.assertNotIn("--remote-debugging-port=0", arguments)
        self.assertNotIn("--disable-gpu", arguments)
        self.assertIn("--no-sandbox", arguments)
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
        self.assertIn(".job-card-footer .boss-name", script)
        self.assertNotIn("'.boss-info'", script)
        self.assertIn("recruiterRoots", script)
        self.assertIn(".company-location", script)

    def test_detail_fields_reads_only_the_jd_content_container(self) -> None:
        browser = PlaywrightEdgeBrowser(
            Path(self.temp.name) / "detail-selector-profile", edge_path=__file__
        )
        page = Mock()
        page.evaluate.return_value = {
            "title": "Python 后端开发工程师",
            "description": "岗位职责：负责 FastAPI 服务开发",
        }
        browser._page = page

        with patch.object(browser, "_ensure_started"):
            fields = browser.detail_fields()

        self.assertEqual(fields["description"], "岗位职责：负责 FastAPI 服务开发")
        script = page.evaluate.call_args.args[0]
        self.assertIn("const description = () =>", script)
        self.assertIn("'.job-sec-text'", script)
        self.assertIn("description: description()", script)
        self.assertNotIn("description: text(['.job-sec-text'", script)
        self.assertNotIn("'.job-detail']", script)

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
        script = browser._page.evaluate.call_args.args[0]
        self.assertIn("/job_detail/", script)
        self.assertIn("对搜索是否", script)
        self.assertNotIn("bodyText", script)
        browser._page.dispatch_wheel.assert_called_once_with(1600)

    def test_auxiliary_detail_uses_background_target_without_activating_edge(self) -> None:
        browser = PlaywrightEdgeBrowser(
            Path(self.temp.name) / "background-detail-profile", edge_path=__file__
        )
        parent = Mock()
        parent.is_closed.return_value = False
        detail = Mock()
        detail.is_closed.return_value = False
        context = Mock()
        context.new_background_page.return_value = detail
        browser._context = context  # type: ignore[assignment]
        browser._page = parent  # type: ignore[assignment]

        with (
            patch.object(browser, "_ensure_started"),
            patch.object(browser, "minimize_collection_window") as minimize,
        ):
            browser.open_auxiliary("https://www.zhipin.com/job_detail/example.html")

        minimize.assert_called_once_with()
        context.new_background_page.assert_called_once_with()
        detail.goto.assert_called_once()
        detail.bring_to_front.assert_not_called()

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

    def test_environment_tool_reports_profile_permission_error(self) -> None:
        with patch(
            "app.mcp.boss_server._get_adapter",
            side_effect=PermissionError("private path"),
        ):
            response = boss_server.environment({})

        self.assertFalse(response["ok"])
        self.assertEqual(response["error_code"], "edge_profile_permission_denied")
        self.assertNotIn("private path", str(response))

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
