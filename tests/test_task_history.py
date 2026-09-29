from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.core.schemas import JobInput, TaskStatus
from app.core.store import TaskRecord
from app.main import app


class TaskHistoryTests(unittest.TestCase):
    def test_history_lists_metadata_without_resume_or_job_body(self) -> None:
        with TestClient(app) as client:
            task = TaskRecord(
                task_id="task-history",
                thread_id="thread-history",
                status=TaskStatus.WAITING_USER,
                current_node="candidate_review",
                checkpoint_version=4,
                job=JobInput(
                    title="Python Engineer",
                    company="Example",
                    city="Shenzhen",
                    raw_text="private full JD",
                ),
            )
            app.state.resume.store.create_task(task)
            headers = {"X-Resume-Agent-Token": app.state.resume.settings.internal_token}

            response = client.get("/api/tasks?limit=20", headers=headers)

            response.raise_for_status()
            item = next(value for value in response.json()["tasks"] if value["task_id"] == task.task_id)
            self.assertEqual(item["thread_id"], task.thread_id)
            self.assertEqual(item["checkpoint_version"], 4)
            self.assertEqual(item["job_title"], "Python Engineer")
            self.assertNotIn("job", item)
            self.assertNotIn("resume", item)
            self.assertNotIn("raw_text", item)


if __name__ == "__main__":
    unittest.main()
