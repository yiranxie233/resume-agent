from __future__ import annotations

import unittest
import uuid
from types import SimpleNamespace

from app.core.database_store import DatabaseMirror
from app.core.db import create_engine_for_url, init_db, session_factory
from app.core.db_mirror import SqlAlchemyMirror
from app.core.models import LangGraphCheckpoint, MatchRun, ModelProfileVersion, Task, TaskEvent
from app.core.schemas import JobInput, ModelProfile, ModelRole, Provider, TaskStatus
from app.core.store import TaskRecord
from app.core.persistence import get_or_create_settings, update_settings
from app.services.matching import calculate_match


class DatabaseContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine_for_url("sqlite+pysqlite:///:memory:")
        init_db(self.engine)
        self.factory = session_factory(self.engine)
        self.mirror = DatabaseMirror(self.engine, self.factory)

    def tearDown(self) -> None:
        self.mirror.close()

    @staticmethod
    def _profile(profile_id: str, role: ModelRole, provider: Provider) -> ModelProfile:
        return ModelProfile(
            profile_id=profile_id,
            role=role,
            provider=provider,
            base_url="http://127.0.0.1:11434" if provider is Provider.OLLAMA else "https://example.invalid/v1",
            model_name="qwen-test" if role is ModelRole.CHAT else "bge-test",
            status="ready",
            credential_required=provider is Provider.OPENAI_COMPATIBLE,
            context_window_tokens=8192 if role is ModelRole.CHAT else None,
            dimension=768 if role is ModelRole.EMBEDDING else None,
            max_input_tokens=8192 if role is ModelRole.EMBEDDING else None,
            capabilities={"generation": True} if role is ModelRole.CHAT else {"embedding": True},
            generation_params={"temperature": 0.2} if role is ModelRole.CHAT else {},
        )

    def _job_and_profiles(self, *, external_chat: bool = False, external_embedding: bool = False):
        job_id = f"job_{uuid.uuid4().hex}"
        job = JobInput(
            title="Python 工程师",
            city="深圳",
            responsibilities=["负责服务开发"],
            requirements=["熟悉缓存"],
            skills=["Python", "Redis"],
        )
        snapshot_id = self.mirror.persist_job(job_id, job)
        self.assertIsNotNone(snapshot_id)
        chat = self._profile(
            f"chat_{uuid.uuid4().hex}",
            ModelRole.CHAT,
            Provider.OPENAI_COMPATIBLE if external_chat else Provider.OLLAMA,
        )
        embedding = self._profile(
            f"embedding_{uuid.uuid4().hex}",
            ModelRole.EMBEDDING,
            Provider.OPENAI_COMPATIBLE if external_embedding else Provider.OLLAMA,
        )
        self.assertTrue(self.mirror.persist_model_profile(chat))
        self.assertTrue(self.mirror.persist_model_profile(embedding))
        return job_id, job, snapshot_id, chat, embedding

    def _task(self, *, job: JobInput, snapshot_id: str, chat: ModelProfile, embedding: ModelProfile, status=TaskStatus.WAITING_USER):
        return TaskRecord(
            task_id=f"task_{uuid.uuid4().hex}",
            thread_id=f"thread_{uuid.uuid4().hex}",
            status=status,
            current_node="candidate_review",
            job=job,
            job_snapshot_id=snapshot_id,
            state={
                "chat_profile_id": chat.profile_id,
                "chat_profile_version": chat.config_version,
                "chat_profile_snapshot": chat.model_dump(mode="json"),
                "embedding_profile_id": embedding.profile_id,
                "embedding_profile_version": embedding.config_version,
                "embedding_profile_snapshot": embedding.model_dump(mode="json"),
                "dimension_weights": {"skills": 0.25, "responsibilities": 0.25, "projects": 0.25, "keywords": 0.25},
                "component_weights": {"deterministic": 0.5, "semantic": 0.5},
                "scoring_config_version": "scoring-contract-v1",
                "base_facts": {"name": "张三", "api_key": "do-not-store"},
            },
        )

    def test_job_hr_activity_round_trips_through_record_extra(self) -> None:
        job = JobInput(
            title="AI 应用工程师",
            city="深圳",
            hr_activity="本周活跃",
            responsibilities=["负责智能体应用开发"],
        )
        self.assertIsNotNone(self.mirror.persist_job("job-hr-activity", job))

        loaded = dict(self.mirror.load_jobs())

        self.assertEqual(loaded["job-hr-activity"].hr_activity, "本周活跃")

    def test_task_binds_immutable_profile_versions_and_rejects_missing_binding(self) -> None:
        _, job, snapshot_id, chat, embedding = self._job_and_profiles()
        task = self._task(job=job, snapshot_id=snapshot_id, chat=chat, embedding=embedding)
        self.assertTrue(self.mirror.persist_task(task))
        with self.factory() as session:
            row = session.get(Task, task.task_id)
            self.assertIsNotNone(row.chat_model_profile_version_id)
            self.assertIsNotNone(row.embedding_model_profile_version_id)
            versions = session.query(ModelProfileVersion).all()
            self.assertEqual(len(versions), 2)

        # Mutable model-card metadata may be re-probed, but a version already
        # referenced by a task must remain frozen.
        chat.model_name = "changed-after-task-start"
        chat.capabilities = {"generation": False}
        self.assertTrue(self.mirror.persist_model_profile(chat))
        with self.factory() as session:
            frozen = session.get(
                ModelProfileVersion,
                session.get(Task, task.task_id).chat_model_profile_version_id,
            )
            self.assertEqual(frozen.model_name, "qwen-test")
            self.assertEqual(frozen.capabilities, {"generation": True})

        # A task cannot silently proceed with a profile version that was never
        # persisted.  The mirror rejects the write instead of leaving a NULL FK.
        missing = self._task(job=job, snapshot_id=snapshot_id, chat=chat, embedding=embedding)
        missing.state["chat_profile_version"] = 99
        self.assertFalse(self.mirror.persist_task(missing))
        with self.factory() as session:
            self.assertIsNone(session.get(Task, missing.task_id))

    def test_match_run_hydrates_after_restart_and_public_ids_do_not_collide(self) -> None:
        _, job, snapshot_id, chat, embedding = self._job_and_profiles()
        first = self._task(job=job, snapshot_id=snapshot_id, chat=chat, embedding=embedding)
        first.match = calculate_match(job, None, match_run_id="same-public-id", scoring_config_version="scoring-contract-v1")
        self.assertTrue(self.mirror.persist_task(first))
        first.checkpoint_version = 1
        first.current_node = "candidate_review"
        self.assertTrue(self.mirror.persist_checkpoint(first))

        second = self._task(job=job, snapshot_id=snapshot_id, chat=chat, embedding=embedding)
        second.match = calculate_match(job, None, match_run_id="same-public-id", scoring_config_version="scoring-contract-v1")
        self.assertTrue(self.mirror.persist_task(second))
        second.checkpoint_version = 1
        self.assertTrue(self.mirror.persist_checkpoint(second))

        loaded = {item.task_id: item for item in self.mirror.load_tasks()}
        self.assertEqual(loaded[first.task_id].match.match_run_id, "same-public-id")
        self.assertEqual(loaded[first.task_id].match.scoring_config_version, "scoring-contract-v1")
        self.assertEqual(loaded[first.task_id].match.dimensions["skills"].score, first.match.dimensions["skills"].score)
        self.assertEqual(loaded[second.task_id].match.match_run_id, "same-public-id")
        with self.factory() as session:
            self.assertEqual(session.query(MatchRun).count(), 2)

    def test_restart_pauses_when_either_external_profile_lacks_process_credential(self) -> None:
        _, job, snapshot_id, chat, embedding = self._job_and_profiles(external_chat=True)
        task = self._task(job=job, snapshot_id=snapshot_id, chat=chat, embedding=embedding, status=TaskStatus.RUNNING)
        self.assertTrue(self.mirror.persist_task(task))
        restored = self.mirror.load_tasks()[0]
        self.assertEqual(restored.status, TaskStatus.PAUSED)
        self.assertEqual(restored.blocked_reason, "needs_credentials")

        _, job2, snapshot_id2, chat2, embedding2 = self._job_and_profiles(external_embedding=True)
        task2 = self._task(job=job2, snapshot_id=snapshot_id2, chat=chat2, embedding=embedding2, status=TaskStatus.WAITING_USER)
        self.assertTrue(self.mirror.persist_task(task2))
        restored_by_id = {item.task_id: item for item in self.mirror.load_tasks()}
        self.assertEqual(restored_by_id[task2.task_id].status, TaskStatus.PAUSED)
        self.assertEqual(restored_by_id[task2.task_id].blocked_reason, "needs_credentials")

    def test_compact_checkpoint_excludes_resume_jd_and_credentials(self) -> None:
        engine = create_engine_for_url("sqlite+pysqlite:///:memory:")
        mirror = SqlAlchemyMirror(engine, initialize=True)
        task = SimpleNamespace(
            task_id="task-compact-contract",
            thread_id="thread-compact-contract",
            status="running",
            blocked_reason=None,
            current_node="candidate_review",
            checkpoint_version=1,
            job_snapshot_id="job-1",
            candidates=[],
            state={
                "job": {"requirements": "COMPLETE JD BODY"},
                "resume": {"raw_text": "COMPLETE RESUME BODY"},
                "messages": ["COMPLETE MESSAGE BODY"],
                "base_facts": {
                    "name": "张三",
                    "api_key": "secret",
                    "github_token": "ghp_secret",
                    "resume_text": "COMPLETE RESUME BODY FROM BASE FACTS",
                },
                "project_patch": {"new_value": "COMPLETE PATCH BODY"},
                "module_drafts": {"summary": {"new_value": "COMPLETE DRAFT BODY"}},
                "pending_action": {"action": "confirm", "payload": "COMPLETE ACTION BODY", "token": "secret"},
            },
        )
        self.assertTrue(mirror.persist_checkpoint(task))
        with mirror.factory() as session:
            row = session.get(Task, task.task_id)
            checkpoint = session.query(LangGraphCheckpoint).one()
            serialized = str(row.extra) + str(checkpoint.state_ref) + str(checkpoint.pending_action)
            self.assertNotIn("COMPLETE JD BODY", serialized)
            self.assertNotIn("COMPLETE RESUME BODY", serialized)
            self.assertNotIn("COMPLETE PATCH BODY", serialized)
            self.assertNotIn("COMPLETE DRAFT BODY", serialized)
            self.assertNotIn("COMPLETE RESUME BODY FROM BASE FACTS", serialized)
            self.assertNotIn("secret", serialized)
            self.assertNotIn("ghp_secret", serialized)
            self.assertEqual(row.extra["base_facts"], {"name": "张三"})
        mirror.close()

    def test_profile_metadata_and_events_strip_nested_credentials(self) -> None:
        profile = self._profile("chat-secret-contract", ModelRole.CHAT, Provider.OPENAI_COMPATIBLE)
        profile.generation_params = {
            "temperature": 0.2,
            "api-key": "sk-abcdefghijklmnopqrstuvwxyz",
            "nested": {"githubToken": "ghp_abcdefghijklmnopqrstuvwxyz"},
        }
        self.assertTrue(self.mirror.persist_model_profile(profile))
        _, job, snapshot_id, _, embedding = self._job_and_profiles()
        task = self._task(job=job, snapshot_id=snapshot_id, chat=profile, embedding=embedding)
        self.assertTrue(self.mirror.persist_task(task))
        self.assertTrue(
            self.mirror.persist_event(
                task.task_id,
                {
                    "type": "error",
                    "message": "api_key=sk-abcdefghijklmnopqrstuvwxyz github_token=ghp_abcdefghijklmnopqrstuvwxyz",
                    "payload": {"authorization": "Bearer secret"},
                },
            )
        )
        with self.factory() as session:
            version = session.query(ModelProfileVersion).filter_by(profile_id=profile.profile_id).one()
            serialized = str(version.generation_params)
            self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz", serialized)
            self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz", serialized)
            event = session.query(TaskEvent).filter_by(task_id=task.task_id).one()
            self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz", event.message)
            self.assertNotIn("ghp_abcdefghijklmnopqrstuvwxyz", event.message)

        with self.factory() as session:
            settings = get_or_create_settings(session)
            updated = update_settings(
                session,
                {
                    "theme": "system",
                    "apiKey": "sk-abcdefghijklmnopqrstuvwxyz",
                    "nested": {"github-token": "ghp_abcdefghijklmnopqrstuvwxyz"},
                },
                expected_version=settings.settings_version,
            )
            self.assertEqual(updated.values, {"theme": "system", "nested": {}})


if __name__ == "__main__":
    unittest.main()
