from __future__ import annotations

import unittest
import uuid

from fastapi.testclient import TestClient

from app.main import app


class TaskConfigurationTests(unittest.TestCase):
    def test_no_resume_requires_confirmable_base_facts(self) -> None:
        with TestClient(app) as client:
            headers = {"X-Resume-Agent-Token": app.state.resume.settings.internal_token}
            created = client.post(
                "/api/jobs",
                headers=headers,
                json={"title": "Python 工程师", "city": "深圳", "skills": ["Python"]},
            )
            created.raise_for_status()
            response = client.post(
                "/api/tasks",
                headers=headers,
                json={"job_id": created.json()["job_id"]},
            )
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["error_code"], "base_facts_required")

    def test_project_only_task_does_not_require_resume_identity_fields(self) -> None:
        with TestClient(app) as client:
            headers = {"X-Resume-Agent-Token": app.state.resume.settings.internal_token}
            job = client.post(
                "/api/jobs",
                headers=headers,
                json={"title": "Python 智能体工程师", "raw_text": "负责智能体项目开发"},
            ).json()
            response = client.post(
                "/api/tasks",
                headers=headers,
                json={
                    "job_id": job["job_id"],
                    "output_mode": "project_only",
                    "candidate_count": 3,
                },
            )

            self.assertEqual(response.status_code, 202, response.text)
            self.assertEqual(response.json()["output_mode"], "project_only")
            self.assertEqual(response.json()["requested_output_mode"], "project_only")
            self.assertIsNone(response.json()["template_id"])

    def test_task_freezes_profile_and_scoring_configuration(self) -> None:
        with TestClient(app) as client:
            headers = {"X-Resume-Agent-Token": app.state.resume.settings.internal_token}
            suffix = uuid.uuid4().hex
            profile_id = f"chat-{suffix}"
            profile = client.post(
                "/api/models",
                headers=headers,
                json={
                    "profile_id": profile_id,
                    "role": "chat",
                    "provider": "ollama",
                    "base_url": "http://127.0.0.1:11434",
                    "model_name": "qwen-test",
                    "status": "ready",
                    "context_window_tokens": 4096,
                },
            )
            profile.raise_for_status()
            self.assertEqual(profile.json()["status"], "unprobed")
            duplicate = client.post("/api/models", headers=headers, json=profile.json())
            self.assertEqual(duplicate.status_code, 409)
            self.assertEqual(duplicate.json()["error_code"], "model_profile_immutable")

            job = client.post(
                "/api/jobs",
                headers=headers,
                json={"title": "Python 工程师", "city": "深圳", "skills": ["Python"]},
            )
            job.raise_for_status()
            task_response = client.post(
                "/api/tasks",
                headers=headers,
                json={
                    "job_id": job.json()["job_id"],
                    "chat_profile_id": profile_id,
                    "base_facts": {
                        "name": "张三",
                        "phone": "13800000000",
                        "email": "zhangsan@example.com",
                        "objective": "Python 工程师",
                    },
                    "dimension_weights": {
                        "skills": 0.4,
                        "responsibilities": 0.2,
                        "projects": 0.2,
                        "keywords": 0.2,
                    },
                    "component_weights": {"deterministic": 0.6, "semantic": 0.4},
                },
            )
            task_response.raise_for_status()
            task = app.state.resume.store.get_task(task_response.json()["task_id"])
            self.assertIsNotNone(task)
            self.assertEqual(task.state["chat_profile_snapshot"]["profile_id"], profile_id)
            self.assertEqual(task.state["dimension_weights"]["skills"], 0.4)
            self.assertEqual(task.state["component_weights"]["semantic"], 0.4)


if __name__ == "__main__":
    unittest.main()
