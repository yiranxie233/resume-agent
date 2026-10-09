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
    def test_pdf_export_falls_back_to_local_text_renderer_when_office_fails(self) -> None:
        with TestClient(app) as client:
            suffix = uuid.uuid4().hex
            candidate = CandidateProject(
                candidate_id=f"candidate-fallback-{suffix}",
                candidate_slot_id="slot-1",
                title="中文智能体项目",
                introduction="项目简介",
                tech_stack=["Python"],
                solutions=["实现完整导出链路"],
                results=["结果已确认"],
                status="confirmed",
                needs_verification=False,
            )
            task = TaskRecord(
                task_id=f"task-pdf-fallback-{suffix}",
                thread_id=f"thread-pdf-fallback-{suffix}",
                status=TaskStatus.COMPLETED,
                current_node="completed",
                checkpoint_version=1,
                job=JobInput(title="AI 工程师"),
                candidates=[candidate],
                state={
                    "selected_candidate": candidate.candidate_id,
                    "resume_snapshot_id": f"snapshot-pdf-fallback-{suffix}",
                },
            )
            app.state.resume.store.create_task(task)
            headers = {
                "X-Resume-Agent-Token": app.state.resume.settings.internal_token
            }
            failed_office = PreviewResult(
                "pdf", "conversion_failed", None, None, None, ("Word unavailable",)
            )

            with patch("app.main._resume_pdf_result", return_value=failed_office):
                response = client.post(
                    f"/api/tasks/{task.task_id}/export",
                    headers=headers,
                    json={
                        "thread_id": task.thread_id,
                        "checkpoint_version": task.checkpoint_version,
                        "snapshot_id": task.state["resume_snapshot_id"],
                        "format": "pdf",
                    },
                )

            response.raise_for_status()
            payload = response.json()
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["artifact"]["status"], "ready")
            downloaded = client.get(
                payload["artifact"]["download_url"], headers=headers
            )
            downloaded.raise_for_status()
            self.assertTrue(downloaded.content.startswith(b"%PDF"))

    def test_all_formats_expose_task_scoped_downloads_and_docx_visual_preview(self) -> None:
        with TestClient(app) as client:
            suffix = uuid.uuid4().hex
            candidate = CandidateProject(
                candidate_id=f"candidate-artifact-{suffix}",
                candidate_slot_id="slot-1",
                title="可导出项目",
                introduction="项目简介",
                tech_stack=["Python"],
                solutions=["实现导出链路"],
                results=["结果已确认"],
                status="confirmed",
                needs_verification=False,
            )
            task = TaskRecord(
                task_id=f"task-artifact-{suffix}",
                thread_id=f"thread-artifact-{suffix}",
                status=TaskStatus.COMPLETED,
                current_node="completed",
                checkpoint_version=2,
                job=JobInput(title="Python 工程师"),
                candidates=[candidate],
                state={
                    "selected_candidate": candidate.candidate_id,
                    "resume_snapshot_id": f"resume-snapshot-artifact-{suffix}",
                    "final_confirmation": True,
                },
            )
            app.state.resume.store.create_task(task)
            headers = {"X-Resume-Agent-Token": app.state.resume.settings.internal_token}
            payload = {
                "thread_id": task.thread_id,
                "checkpoint_version": task.checkpoint_version,
                "snapshot_id": task.state["resume_snapshot_id"],
            }

            markdown = client.post(
                f"/api/tasks/{task.task_id}/export",
                headers=headers,
                json={**payload, "format": "markdown"},
            )
            markdown.raise_for_status()
            markdown_artifact = markdown.json()["artifact"]
            downloaded_markdown = client.get(
                markdown_artifact["download_url"], headers=headers
            )
            downloaded_markdown.raise_for_status()
            self.assertIn("可导出项目", downloaded_markdown.content.decode("utf-8"))

            def render_pdf(_docx_path, pdf_path):
                pdf_path.write_bytes(b"%PDF-1.4\n% test preview\n")
                return PreviewResult("pdf", "ready", str(pdf_path), 1, True)

            with patch("app.main._resume_pdf_result", side_effect=render_pdf):
                docx = client.post(
                    f"/api/tasks/{task.task_id}/export-preview",
                    headers=headers,
                    json={**payload, "format": "docx"},
                )
                pdf = client.post(
                    f"/api/tasks/{task.task_id}/export",
                    headers=headers,
                    json={**payload, "format": "pdf"},
                )
            docx.raise_for_status()
            pdf.raise_for_status()
            docx_payload = docx.json()
            self.assertEqual(docx_payload["artifact"]["status"], "ready")
            self.assertEqual(
                docx_payload["visual_preview"]["artifact"]["status"], "ready"
            )
            downloaded_docx = client.get(
                docx_payload["artifact"]["download_url"], headers=headers
            )
            downloaded_docx.raise_for_status()
            self.assertTrue(downloaded_docx.content.startswith(b"PK"))
            pdf_payload = pdf.json()
            downloaded_pdf = client.get(
                pdf_payload["artifact"]["download_url"], headers=headers
            )
            downloaded_pdf.raise_for_status()
            self.assertTrue(downloaded_pdf.content.startswith(b"%PDF"))

            traversal = client.get(
                f"/api/tasks/{task.task_id}/artifacts/exports/%2E%2E%2Fresume.md",
                headers=headers,
            )
            self.assertIn(traversal.status_code, {404, 422})

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
