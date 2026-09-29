from __future__ import annotations

import unittest

from app.core.schemas import CandidateProject
from app.services.export_service import compress_markdown, estimate_pages, render_markdown
from app.ui.streamlit_app import _preview_matches_task, _selected_candidate


def _candidate(candidate_id: str, title: str) -> CandidateProject:
    return CandidateProject(
        candidate_id=candidate_id,
        candidate_slot_id=f"slot-{candidate_id[-1]}",
        title=title,
        introduction=f"{title}简介",
        tech_stack=["Python"],
        solutions=[f"{title}方案"],
        results=[f"{title}结果"],
    )


class ExportServiceTests(unittest.TestCase):
    def test_render_markdown_uses_explicit_selected_candidate(self) -> None:
        first = _candidate("candidate-1", "首个候选")
        second = _candidate("candidate-2", "用户选择候选")

        rendered = render_markdown(
            job_title="Python 工程师",
            candidates=[first, second],
            selected_candidate_id=second.candidate_id,
        )

        self.assertIn("用户选择候选", rendered)
        self.assertNotIn("首个候选", rendered)

    def test_compression_changes_layout_before_rewriting(self) -> None:
        content = "正文" * 1100
        self.assertEqual(estimate_pages(content), 2)

        compressed = compress_markdown(content, allow_rewrite=False)

        self.assertEqual(compressed.content, content)
        self.assertFalse(compressed.rewritten)
        self.assertEqual(compressed.page_count, 1)
        self.assertGreater(compressed.density, 1.0)

    def test_ui_resolves_only_the_explicit_selection(self) -> None:
        task = {
            "task_id": "task-1",
            "resume_snapshot_id": "snapshot-1",
            "selected_candidate_id": "candidate-2",
            "candidates": [
                {"candidate_id": "candidate-1", "status": "draft"},
                {"candidate_id": "candidate-2", "status": "draft"},
            ],
        }

        self.assertEqual(_selected_candidate(task)["candidate_id"], "candidate-2")
        self.assertIsNone(_selected_candidate({**task, "selected_candidate_id": "missing"}))
        self.assertTrue(
            _preview_matches_task(
                {"task_id": "task-1", "snapshot_id": "snapshot-1", "format": "md"},
                task,
                "markdown",
            )
        )
        self.assertFalse(
            _preview_matches_task(
                {"task_id": "task-1", "snapshot_id": "stale", "format": "md"},
                task,
                "markdown",
            )
        )


if __name__ == "__main__":
    unittest.main()
