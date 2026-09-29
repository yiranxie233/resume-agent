from __future__ import annotations

import unittest
import uuid

from fastapi.testclient import TestClient

from app.core.database_store import DatabaseMirror
from app.core.db import create_engine_for_url, init_db, session_factory
from app.core.models import CandidateSlot, Confirmation, Consent, GeneratedVersion, Task, TaskEvent
from app.core.schemas import TaskStatus
from app.core.store import TaskRecord
from app.main import app


class ConsentPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine_for_url("sqlite+pysqlite:///:memory:")
        init_db(self.engine)
        self.factory = session_factory(self.engine)
        self.mirror = DatabaseMirror(self.engine, self.factory)

    def tearDown(self) -> None:
        self.engine.dispose()

    def test_global_consent_survives_reload_and_revocation(self) -> None:
        granted = self.mirror.persist_consent(
            "external_model", True, policy_version="consent-v1"
        )
        self.assertIsNotNone(granted)
        loaded = self.mirror.load_consents()
        self.assertTrue(loaded["external_model:global"]["granted"])
        self.assertIsNotNone(loaded["external_model:global"]["granted_at"])

        revoked = self.mirror.persist_consent(
            "external_model", False, policy_version="consent-v1"
        )
        self.assertIsNotNone(revoked)
        loaded = self.mirror.load_consents()
        self.assertFalse(loaded["external_model:global"]["granted"])
        self.assertIsNotNone(loaded["external_model:global"]["revoked_at"])
        with self.factory() as session:
            self.assertEqual(session.query(Consent).count(), 1)

    def test_delete_task_removes_owned_rows_but_not_global_consent(self) -> None:
        with self.factory() as session:
            task = Task(
                id="task-delete",
                thread_id="thread-delete",
                status="waiting_user",
                embedding_mode="tfidf_fallback",
            )
            session.add(task)
            session.flush()
            version = GeneratedVersion(
                task_id=task.id,
                module_type="candidate_project",
                version_no=1,
                content={"title": "draft"},
                content_hash="hash",
                source_refs={},
            )
            session.add(version)
            session.flush()
            session.add_all(
                [
                    CandidateSlot(
                        task_id=task.id,
                        candidate_slot_id="slot-1",
                        slot_index=1,
                        generated_version_id=version.id,
                    ),
                    Confirmation(
                        task_id=task.id,
                        field_id="project.title",
                        value_hash="hash",
                        value={"value": "draft"},
                    ),
                    TaskEvent(
                        task_id=task.id,
                        sequence_no=1,
                        event_type="created",
                    ),
                    Consent(
                        scope="post_time_risk",
                        task_id=task.id,
                        granted=True,
                        policy_version="consent-v1",
                    ),
                    Consent(
                        scope="external_model",
                        task_id=None,
                        granted=True,
                        policy_version="consent-v1",
                    ),
                ]
            )
            session.commit()

        self.assertTrue(self.mirror.delete_task("task-delete"))
        with self.factory() as session:
            self.assertIsNone(session.get(Task, "task-delete"))
            self.assertEqual(session.query(GeneratedVersion).count(), 0)
            self.assertEqual(session.query(CandidateSlot).count(), 0)
            self.assertEqual(session.query(Confirmation).count(), 0)
            self.assertEqual(session.query(TaskEvent).count(), 0)
            consents = session.query(Consent).all()
            self.assertEqual(len(consents), 1)
            self.assertEqual(consents[0].scope, "external_model")
            self.assertIsNone(consents[0].task_id)


class TaskDeletionApiTests(unittest.TestCase):
    def test_delete_removes_memory_record_and_task_directory(self) -> None:
        with TestClient(app) as client:
            runtime = app.state.resume
            previous_mirror = runtime.db_mirror
            suffix = uuid.uuid4().hex
            task_id = f"task_{suffix}"
            task = TaskRecord(
                task_id=task_id,
                thread_id=f"thread_{suffix}",
                status=TaskStatus.WAITING_USER,
                current_node="candidate_review",
            )
            runtime.store.create_task(task)
            task_dir = runtime.settings.data_root / "tasks" / task_id
            task_dir.mkdir(parents=True)
            (task_dir / "draft.txt").write_text("draft", encoding="utf-8")
            runtime.db_mirror = None
            try:
                response = client.delete(
                    f"/api/tasks/{task_id}",
                    headers={"X-Resume-Agent-Token": runtime.settings.internal_token},
                )
                response.raise_for_status()
                self.assertEqual(response.json()["status"], "deleted")
                self.assertIsNone(runtime.store.get_task(task_id))
                self.assertFalse(task_dir.exists())
            finally:
                runtime.db_mirror = previous_mirror


if __name__ == "__main__":
    unittest.main()
