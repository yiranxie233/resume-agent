from __future__ import annotations

import unittest
import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.schemas import CandidateProject, JobInput, TaskStatus
from app.core.store import TaskRecord
from app.main import app
from app.services.export_service import PreviewResult


class ExportApiTests(unittest.TestCase):
    def test_rendered_pdf_page_count_overrides_estimate(self) -> None:
        with TestClient(app) as client:
            suffix = uuid.uuid4().hex
            candidate = CandidateProject(
                candidate_id=f"candidate-{suffix}",
                candidate_slot_id="slot-1",
                title="已选择项目",
                introduction="简短项目简介",
                tech_stack=["Python"],
                solutions=["完成核心链路"],
                results=["结果待核实"],
                status="confirmed",
                needs_verification=False,
            )
            task = TaskRecord(
                task_id=f"task-{suffix}",
                thread_id=f"thread-{suffix}",
                status=TaskStatus.COMPLETED,
                current_node="completed",
                checkpoint_version=3,
                job_snapshot_id=f"job-snapshot-{suffix}",
                job=JobInput(title="Python 工程师", city="深圳"),
                candidates=[candidate],
                state={
                    "selected_candidate": candidate.candidate_id,
                    "resume_snapshot_id": f"resume-snapshot-{suffix}",
                },
            )
            app.state.resume.store.create_task(task)
            headers = {"X-Resume-Agent-Token": app.state.resume.settings.internal_token}
            payload = {
                "thread_id": task.thread_id,
                "checkpoint_version": task.checkpoint_version,
                "snapshot_id": task.state["resume_snapshot_id"],
                "format": "pdf",
            }
            docx_result = PreviewResult("docx", "preview_pending", "fake.docx", None, None)
            pdf_result = PreviewResult("pdf", "ready", "fake.pdf", 2, False)

            with patch("app.main.export_docx", return_value=docx_result), patch(
                "app.main.convert_pdf_with_libreoffice", return_value=pdf_result
            ):
                preview = client.post(f"/api/tasks/{task.task_id}/export-preview", headers=headers, json=payload)
                exported = client.post(f"/api/tasks/{task.task_id}/export", headers=headers, json=payload)

            preview.raise_for_status()
            exported.raise_for_status()
            for response in (preview.json(), exported.json()):
                self.assertEqual(response["estimated_page_count"], 1)
                self.assertEqual(response["page_count"], 2)
                self.assertEqual(response["page_count_source"], "rendered_pdf")
                self.assertEqual(response["compliance_status"], "overflow")


if __name__ == "__main__":
    unittest.main()
