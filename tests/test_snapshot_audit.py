from __future__ import annotations

import unittest

from app.core.database_store import DatabaseMirror
from app.core.db import create_engine_for_url, init_db, session_factory
from app.core.models import ContentPatch, ResumeSnapshot
from app.core.schemas import CandidateProject, JobInput, ResumeDocument, ResumeSection, TaskStatus
from app.core.store import TaskRecord
from app.core.utils import sha256_text


class SnapshotAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine_for_url("sqlite+pysqlite:///:memory:")
        init_db(self.engine)
        self.factory = session_factory(self.engine)
        self.mirror = DatabaseMirror(self.engine, self.factory)

    def tearDown(self) -> None:
        self.mirror.close()

    def _task(self, *, project_hash: str | None = None) -> TaskRecord:
        job = JobInput(title="Python 工程师", skills=["Python", "FastAPI"])
        job_snapshot_id = self.mirror.persist_job("job-snapshot-audit", job)
        project = ResumeSection(
            section_id="project-old",
            module="projects",
            title="旧项目",
            content="旧项目内容",
            value_hash=sha256_text("旧项目内容"),
        )
        skills = ResumeSection(
            section_id="skills-old",
            module="skills",
            title="技能清单",
            content="Python",
            value_hash=sha256_text("Python"),
        )
        resume = ResumeDocument(
            resume_id="resume-audit",
            filename="resume.md",
            file_type="markdown",
            file_size=32,
            page_count=1,
            sections=[project, skills],
            raw_text="旧项目内容\nPython",
        )
        self.assertEqual(
            self.mirror.persist_resume(resume, path="resume.md", data=resume.raw_text.encode()),
            resume.resume_id,
        )
        candidate = CandidateProject(
            candidate_id="candidate-audit",
            candidate_slot_id="slot-audit",
            title="岗位匹配项目",
            introduction="项目简介",
            tech_stack=["Python", "FastAPI"],
            solutions=["实现接口"],
            results=["结果已确认"],
            status="confirmed",
            needs_verification=False,
        )
        return TaskRecord(
            task_id="task-snapshot-audit",
            thread_id="thread-snapshot-audit",
            status=TaskStatus.COMPLETED,
            current_node="completed",
            checkpoint_version=0,
            job_snapshot_id=job_snapshot_id,
            job=job,
            resume=resume,
            candidates=[candidate],
            state={
                "selected_candidate": candidate.candidate_id,
                "resume_snapshot_id": "resume-snapshot-audit",
                "final_confirmation": True,
                "output_mode": "resume_edit",
                "project_patch": {
                    "patch_id": "project-patch-audit",
                    "target_module": "projects",
                    "target_project_id": project.section_id,
                    "operation": "replace",
                    "old_value_hash": project_hash or project.value_hash,
                    "new_value": candidate.model_dump(mode="json"),
                    "reason": "匹配目标岗位",
                    "status": "approved_pending_apply",
                },
                "module_patches": {
                    "skills": {
                        "patch_id": "skills-patch-audit",
                        "target_module": "skills",
                        "target_item_id": skills.section_id,
                        "old_value_hash": skills.value_hash,
                        "new_value": "Python、FastAPI",
                        "new_value_hash": sha256_text("Python、FastAPI"),
                        "reason": "补齐岗位技能",
                        "status": "confirmed",
                    }
                },
                "module_decisions": {"skills": {"decision": "optimize"}},
                "layout_decision": {"status": "confirmed", "estimated_page_count": 1},
                "compression_decision": {"accepted": False},
            },
        )

    def test_confirmed_patches_and_materialized_snapshot_are_auditable(self) -> None:
        task = self._task()
        self.assertTrue(self.mirror.persist_task(task))
        self.assertEqual(self.mirror.persist_resume_snapshot(task), "resume-snapshot-audit")

        with self.factory() as session:
            patches = session.query(ContentPatch).order_by(ContentPatch.target_module).all()
            self.assertEqual([item.status for item in patches], ["applied", "applied"])
            self.assertTrue(all(item.applied_version_id for item in patches))
            snapshot = session.get(ResumeSnapshot, "resume-snapshot-audit")
            self.assertEqual(snapshot.state, "confirmed")
            self.assertEqual(snapshot.project_order, ["candidate-audit"])
            self.assertIn("materialized", snapshot.module_version_ids)
            self.assertEqual(len(snapshot.module_version_ids["patch_ids"]), 2)

        loaded = self.mirror.load_resume_snapshot("resume-snapshot-audit", task_id=task.task_id)
        self.assertEqual(loaded["materialized"]["state"]["selected_candidate"], "candidate-audit")
        self.assertEqual(loaded["materialized"]["resume"]["resume_id"], "resume-audit")

    def test_stale_resume_patch_is_recorded_as_conflicted(self) -> None:
        task = self._task(project_hash=sha256_text("stale"))
        self.assertTrue(self.mirror.persist_task(task))
        with self.factory() as session:
            project_patch = session.query(ContentPatch).filter_by(target_module="projects").one()
            self.assertEqual(project_patch.status, "conflicted")
            self.assertIsNone(project_patch.applied_version_id)
            self.assertEqual(project_patch.diff["actual_value_hash"], sha256_text("旧项目内容"))


if __name__ == "__main__":
    unittest.main()
