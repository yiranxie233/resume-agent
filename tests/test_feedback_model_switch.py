from __future__ import annotations

import unittest
import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.schemas import ModelProfile, ModelRole, Provider, TaskStatus
from app.core.store import TaskRecord
from app.main import app
from app.services.feedback import classify_feedback


class FeedbackModelSwitchTests(unittest.TestCase):
    def test_unfamiliar_but_non_conflicting_feedback_does_not_pause(self) -> None:
        result = classify_feedback("这几个候选太普通了，请换一个更贴近业务的写法")

        self.assertFalse(result.requires_clarification)

    def _task(self) -> TaskRecord:
        suffix = uuid.uuid4().hex
        return TaskRecord(
            task_id=f"task-switch-{suffix}",
            thread_id=f"thread-switch-{suffix}",
            status=TaskStatus.WAITING_USER,
            current_node="candidate_review",
            checkpoint_version=2,
            state={
                "chat_profile_id": "original-chat",
                "generation_branch_id": "branch-original",
            },
        )

    def _post(self, client: TestClient, task: TaskRecord, **overrides: object):
        payload: dict[str, object] = {
            "thread_id": task.thread_id,
            "checkpoint_version": task.checkpoint_version,
            "feedback": "请突出 FastAPI 工程能力",
            **overrides,
        }
        headers = {"X-Resume-Agent-Token": app.state.resume.settings.internal_token}
        with patch.object(app.state.resume, "db_mirror", None), patch.object(
            app.state.resume.workflow, "db_mirror", None
        ), patch.object(app.state.resume.workflow, "launch"):
            return client.post(f"/api/tasks/{task.task_id}/feedback", headers=headers, json=payload)

    def test_wrong_role_and_unprobed_profiles_are_rejected(self) -> None:
        with TestClient(app) as client:
            task = self._task()
            app.state.resume.store.create_task(task)
            embedding = ModelProfile(
                profile_id=f"embedding-{uuid.uuid4().hex}",
                role=ModelRole.EMBEDDING,
                provider=Provider.OLLAMA,
                base_url="http://127.0.0.1:11434",
                model_name="bge-m3",
                status="ready",
            )
            unprobed = ModelProfile(
                profile_id=f"chat-{uuid.uuid4().hex}",
                role=ModelRole.CHAT,
                provider=Provider.OLLAMA,
                base_url="http://127.0.0.1:11434",
                model_name="qwen-test",
                status="unprobed",
            )
            app.state.resume.store.seed_profile(embedding)
            app.state.resume.store.seed_profile(unprobed)

            wrong_role = self._post(client, task, chat_profile_id=embedding.profile_id)
            not_ready = self._post(client, task, chat_profile_id=unprobed.profile_id)

            self.assertEqual(wrong_role.status_code, 422)
            self.assertEqual(wrong_role.json()["error_code"], "invalid_chat_profile_role")
            self.assertEqual(not_ready.status_code, 409)
            self.assertEqual(not_ready.json()["error_code"], "needs_model_recheck")

    def test_external_switch_requires_scoped_credential_and_consent(self) -> None:
        with TestClient(app) as client:
            # A developer's persisted global consent must not make this
            # contract test depend on local database state.
            with patch.dict(app.state.resume.consents, {}, clear=True):
                task = self._task()
                app.state.resume.store.create_task(task)
                profile = ModelProfile(
                    profile_id=f"external-{uuid.uuid4().hex}",
                    role=ModelRole.CHAT,
                    provider=Provider.OPENAI_COMPATIBLE,
                    base_url="https://model.example.test/v1",
                    model_name="chat-model",
                    status="ready",
                    credential_required=True,
                )
                app.state.resume.store.seed_profile(profile)

                missing = self._post(client, task, chat_profile_id=profile.profile_id)
                wrong_handle = app.state.resume.credentials.put(
                    "secret-not-persisted", scope="model:another-profile", ttl_seconds=300
                )
                wrong_scope = self._post(
                    client,
                    task,
                    chat_profile_id=profile.profile_id,
                    credential_handle_id=wrong_handle.handle_id,
                )
                valid_handle = app.state.resume.credentials.put(
                    "secret-not-persisted", scope=f"model:{profile.profile_id}", ttl_seconds=300
                )
                consent_required = self._post(
                    client,
                    task,
                    chat_profile_id=profile.profile_id,
                    credential_handle_id=valid_handle.handle_id,
                )

            self.assertEqual(missing.status_code, 409)
            self.assertEqual(missing.json()["error_code"], "needs_credentials")
            self.assertEqual(wrong_scope.status_code, 409)
            self.assertEqual(wrong_scope.json()["error_code"], "needs_credentials")
            self.assertEqual(consent_required.status_code, 409)
            self.assertEqual(
                consent_required.json()["error_code"], "needs_external_model_consent"
            )

    def test_valid_switch_preserves_thread_and_creates_new_branch(self) -> None:
        with TestClient(app) as client:
            task = self._task()
            app.state.resume.store.create_task(task)
            profile = ModelProfile(
                profile_id=f"chat-ready-{uuid.uuid4().hex}",
                role=ModelRole.CHAT,
                provider=Provider.OLLAMA,
                base_url="http://127.0.0.1:11434",
                model_name="qwen-ready",
                status="ready",
                context_window_tokens=4096,
            )
            app.state.resume.store.seed_profile(profile)

            response = self._post(client, task, chat_profile_id=profile.profile_id)

            response.raise_for_status()
            body = response.json()
            stored = app.state.resume.store.get_task(task.task_id)
            self.assertEqual(body["thread_id"], task.thread_id)
            self.assertEqual(stored.thread_id, task.thread_id)
            self.assertNotEqual(stored.state["generation_branch_id"], "branch-original")
            self.assertEqual(stored.state["chat_profile_id"], profile.profile_id)


if __name__ == "__main__":
    unittest.main()
