from __future__ import annotations

import unittest
import uuid
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.core.schemas import TaskStatus
from app.core.store import TaskRecord
from app.main import app


class CheckpointApiTests(unittest.TestCase):
    def _task(self, *, status: TaskStatus = TaskStatus.WAITING_USER) -> TaskRecord:
        suffix = uuid.uuid4().hex
        return TaskRecord(
            task_id=f"task-cas-{suffix}",
            thread_id=f"thread-cas-{suffix}",
            status=status,
            current_node="candidate_review",
            checkpoint_version=5,
            state={"generation_branch_id": "branch-current"},
        )

    def test_stale_cancel_does_not_interrupt_running_work(self) -> None:
        with TestClient(app) as client:
            task = self._task(status=TaskStatus.RUNNING)
            app.state.resume.store.create_task(task)
            running = Mock()
            running.done.return_value = False
            app.state.resume.workflow._running[task.task_id] = running
            headers = {"X-Resume-Agent-Token": app.state.resume.settings.internal_token}

            response = client.post(
                f"/api/tasks/{task.task_id}/cancel",
                headers=headers,
                json={
                    "thread_id": task.thread_id,
                    "checkpoint_version": 4,
                    "action": "cancel",
                },
            )

            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()["error_code"], "checkpoint_conflict")
            self.assertTrue(response.json()["detail"]["refresh_required"])
            running.cancel.assert_not_called()
            app.state.resume.workflow._running.pop(task.task_id, None)

    def test_stale_feedback_wins_over_profile_validation(self) -> None:
        with TestClient(app) as client:
            task = self._task()
            app.state.resume.store.create_task(task)
            headers = {"X-Resume-Agent-Token": app.state.resume.settings.internal_token}

            response = client.post(
                f"/api/tasks/{task.task_id}/feedback",
                headers=headers,
                json={
                    "thread_id": task.thread_id,
                    "checkpoint_version": 4,
                    "feedback": "请调整内容",
                    "chat_profile_id": "missing-profile",
                },
            )

            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()["error_code"], "checkpoint_conflict")

    def test_stale_confirmation_is_rejected_before_terminal_state(self) -> None:
        with TestClient(app) as client:
            task = self._task(status=TaskStatus.COMPLETED)
            app.state.resume.store.create_task(task)
            headers = {"X-Resume-Agent-Token": app.state.resume.settings.internal_token}

            response = client.post(
                f"/api/tasks/{task.task_id}/confirmations",
                headers=headers,
                json={
                    "thread_id": task.thread_id,
                    "checkpoint_version": 4,
                    "field_id": "title",
                    "value_hash": "hash",
                },
            )

            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()["error_code"], "checkpoint_conflict")

    def test_preview_requires_thread_and_checkpoint_binding(self) -> None:
        with TestClient(app) as client:
            task = self._task(status=TaskStatus.COMPLETED)
            app.state.resume.store.create_task(task)
            headers = {"X-Resume-Agent-Token": app.state.resume.settings.internal_token}

            response = client.post(
                f"/api/tasks/{task.task_id}/export-preview",
                headers=headers,
                json={"format": "markdown"},
            )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["error_code"], "checkpoint_version_required")


if __name__ == "__main__":
    unittest.main()
