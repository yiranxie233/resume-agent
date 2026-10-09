from __future__ import annotations

import unittest
from datetime import datetime, timezone

from app.services.job_parser import (
    job_from_text,
    parse_posted_time,
    sanitize_job_description_text,
)


class JobParserTests(unittest.TestCase):
    def test_requirement_skill_sentences_are_not_discarded(self) -> None:
        job = job_from_text(
            "Python 后端工程师",
            """岗位职责：负责智能体服务开发
任职要求
1. 熟悉 Python、FastAPI
2. 掌握 PostgreSQL 和 Redis
技能要求：LangChain、LangGraph
""",
        )

        self.assertEqual(job.responsibilities, ["负责智能体服务开发"])
        self.assertIn("熟悉 Python、FastAPI", job.requirements)
        self.assertIn("掌握 PostgreSQL 和 Redis", job.requirements)
        for term in ("Python", "FastAPI", "PostgreSQL", "Redis", "LangChain", "LangGraph"):
            self.assertIn(term, job.skills)

    def test_words_like_requirements_inside_body_do_not_change_section(self) -> None:
        job = job_from_text(
            "算法工程师",
            """岗位职责
负责分析业务要求并落地模型服务
负责掌握项目进度和风险
""",
        )
        self.assertEqual(len(job.responsibilities), 2)
        self.assertFalse(job.requirements)

    def test_alternative_skill_options_remain_one_group(self) -> None:
        job = job_from_text(
            "后端工程师",
            """任职要求
1. 熟悉 Python、Java、Go 任选其一
2. 掌握 PostgreSQL
""",
        )

        self.assertIn("Python、Java、Go 任选其一", job.skills)
        self.assertIn("PostgreSQL", job.skills)
        self.assertNotIn("Python", job.skills)
        self.assertNotIn("Java", job.skills)
        self.assertNotIn("Go 任选其一", job.skills)

    def test_screenshot_navigation_noise_before_jd_heading_is_ignored(self) -> None:
        job = job_from_text(
            "AI 应用开发工程师",
            """BOSS直聘
首页 职位 公司 登录/注册
AI 应用开发工程师 20-30K
岗位职责
1. 负责 Python 智能体服务开发
任职要求
1. 熟悉 FastAPI 和 LangGraph
""",
        )

        self.assertEqual(job.responsibilities, ["负责 Python 智能体服务开发"])
        self.assertEqual(job.requirements, ["熟悉 FastAPI 和 LangGraph"])
        self.assertNotIn("BOSS直聘", job.responsibilities)

    def test_hidden_boss_brand_between_heading_and_first_item_is_removed(self) -> None:
        raw = "工作职责BOSS直聘真1. 协助梳理业务需求\n任职要求\n1. 熟悉 Python"

        cleaned = sanitize_job_description_text(raw)
        job = job_from_text("产品工程师", raw)

        self.assertNotIn("BOSS直聘", cleaned)
        self.assertEqual(job.responsibilities, ["协助梳理业务需求"])
        self.assertEqual(job.requirements, ["熟悉 Python"])

    def test_real_sentence_mentioning_boss_platform_is_preserved(self) -> None:
        raw = "岗位职责\n1. 负责 BOSS直聘 渠道数据分析与投放复盘"

        cleaned = sanitize_job_description_text(raw)

        self.assertIn("BOSS直聘", cleaned)

    def test_relative_and_chinese_publication_dates(self) -> None:
        now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
        cases = {
            "昨天": datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc),
            "前天": datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc),
            "2周前": datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc),
            "1个月前": datetime(2026, 8, 26, 12, 0, tzinfo=timezone.utc),
            "发布时间：2026年9月20日": datetime(2026, 9, 20, 0, 0, tzinfo=timezone.utc),
            "9月24日": datetime(2026, 9, 24, 0, 0, tzinfo=timezone.utc),
        }
        for label, expected in cases.items():
            with self.subTest(label=label):
                parsed, status = parse_posted_time(label, now=now)
                self.assertEqual(status, "known")
                self.assertEqual(parsed, expected)


if __name__ == "__main__":
    unittest.main()
