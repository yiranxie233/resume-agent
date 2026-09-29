from __future__ import annotations

import unittest

from fastapi import HTTPException

from app.core.schemas import CandidateProject, JobInput, ResumeDocument, ResumeSection
from app.core.store import TaskRecord
from app.core.utils import sha256_text
from app.main import _render_task_preview


class PreviewMaterializationTests(unittest.TestCase):
    def _candidate(self) -> CandidateProject:
        return CandidateProject(
            candidate_id="candidate-selected",
            candidate_slot_id="slot-1",
            title="Generated Project",
            period="2026.01-2026.03",
            introduction="Generated introduction",
            tech_stack=["Python", "FastAPI"],
            solutions=["Generated solution"],
            results=["Generated result"],
            status="confirmed",
            needs_verification=False,
            match_score=91,
        )

    def _resume(self) -> ResumeDocument:
        return ResumeDocument(
            resume_id="resume-source",
            filename="resume.md",
            file_type="markdown",
            file_size=100,
            page_count=1,
            sections=[
                ResumeSection(
                    section_id="summary-1",
                    module="summary",
                    title="Summary",
                    content="Original summary",
                    value_hash=sha256_text("Original summary"),
                ),
                ResumeSection(
                    section_id="project-keep",
                    module="projects",
                    title="Keep Project",
                    content="Keep this project body",
                    value_hash=sha256_text("Keep this project body"),
                ),
                ResumeSection(
                    section_id="project-replace",
                    module="projects",
                    title="Replace Project",
                    content="Replace this project body",
                    value_hash=sha256_text("Replace this project body"),
                ),
            ],
            raw_text="source",
        )

    def _task(self, *, state: dict[str, object]) -> TaskRecord:
        candidate = self._candidate()
        return TaskRecord(
            task_id="task-preview",
            thread_id="thread-preview",
            job=JobInput(title="Python Engineer", city="Shenzhen"),
            resume=self._resume(),
            candidates=[candidate],
            state={"selected_candidate": candidate.candidate_id, **state},
        )

    def test_replace_removes_only_the_hashed_target_project(self) -> None:
        task = self._task(
            state={
                "output_mode": "resume_edit",
                "project_patch": {
                    "operation": "replace",
                    "target_project_id": "project-replace",
                    "old_value_hash": sha256_text("Replace this project body"),
                },
            }
        )

        rendered = _render_task_preview(task)

        self.assertIn("Keep this project body", rendered)
        self.assertNotIn("Replace this project body", rendered)
        self.assertIn("Generated Project", rendered)

    def test_add_preserves_all_existing_projects(self) -> None:
        task = self._task(
            state={
                "output_mode": "resume_edit",
                "project_patch": {"operation": "add", "old_value_hash": sha256_text("")},
            }
        )

        rendered = _render_task_preview(task)

        self.assertIn("Keep this project body", rendered)
        self.assertIn("Replace this project body", rendered)
        self.assertIn("Generated Project", rendered)

    def test_confirmed_module_patch_changes_only_its_target(self) -> None:
        task = self._task(
            state={
                "output_mode": "resume_edit",
                "module_patches": {
                    "summary": {
                        "patch_id": "patch-summary",
                        "status": "confirmed",
                        "target_item_id": "summary-1",
                        "old_value_hash": sha256_text("Original summary"),
                        "new_value": "Updated summary",
                        "new_value_hash": sha256_text("Updated summary"),
                    }
                },
            }
        )

        rendered = _render_task_preview(task)

        self.assertIn("Updated summary", rendered)
        self.assertNotIn("Original summary", rendered)
        self.assertIn("Keep this project body", rendered)
        self.assertIn("Replace this project body", rendered)

    def test_stale_project_hash_is_rejected(self) -> None:
        task = self._task(
            state={
                "output_mode": "resume_edit",
                "project_patch": {
                    "operation": "replace",
                    "target_project_id": "project-replace",
                    "old_value_hash": sha256_text("stale value"),
                },
            }
        )

        with self.assertRaises(HTTPException) as caught:
            _render_task_preview(task)

        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(caught.exception.detail["error_code"], "patch_hash_conflict")

    def test_copy_only_excludes_the_source_resume(self) -> None:
        task = self._task(
            state={
                "output_mode": "project_only",
                "project_patch": {"operation": "copy_only"},
            }
        )

        rendered = _render_task_preview(task)

        self.assertIn("Generated Project", rendered)
        self.assertNotIn("Original summary", rendered)
        self.assertNotIn("Keep this project body", rendered)
        self.assertNotIn("Replace this project body", rendered)


if __name__ == "__main__":
    unittest.main()
