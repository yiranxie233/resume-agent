from datetime import datetime, timedelta, timezone
import unittest

from app.core.db import create_engine_for_url, init_db, session_factory
from app.core.models import ContentPatch, GeneratedVersion
from app.core.persistence import (
    LeaseBusy,
    OptimisticConflict,
    PatchConflict,
    acquire_worker_lease,
    apply_content_patch,
    canonical_hash,
    commit_checkpoint,
    create_candidate_slots,
    create_task,
    get_or_create_operation,
    get_or_create_settings,
    record_feedback,
    renew_worker_lease,
    update_settings,
)


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine_for_url("sqlite+pysqlite:///:memory:")
        init_db(self.engine)
        self.factory = session_factory(self.engine)
        self.session = self.factory()

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()

    def test_task_has_stable_thread_and_checkpoint_compare_and_swap(self):
        task = create_task(self.session)
        self.assertNotEqual(task.id, task.thread_id)
        first = commit_checkpoint(
            self.session,
            task_id=task.id,
            thread_id=task.thread_id,
            expected_version=0,
            node_name="initialize",
            state_ref={"phase": "started"},
        )
        self.assertEqual(first.version, 1)
        with self.assertRaises(OptimisticConflict) as error:
            commit_checkpoint(
                self.session,
                task_id=task.id,
                thread_id=task.thread_id,
                expected_version=0,
                node_name="stale-tab",
            )
        self.assertEqual(error.exception.current_version, 1)

    def test_settings_if_match_and_idempotency(self):
        settings = get_or_create_settings(self.session)
        self.assertEqual(settings.settings_version, 1)
        updated = update_settings(self.session, {"chat_model": "qwen2.5:7b"}, expected_version=1)
        self.assertEqual(updated.settings_version, 2)
        with self.assertRaises(OptimisticConflict):
            update_settings(self.session, {"chat_model": "other"}, expected_version=1)

        task = create_task(self.session)
        operation, created = get_or_create_operation(
            self.session,
            task_id=task.id,
            thread_id=task.thread_id,
            operation_key="task/node/input/action",
            node_name="write",
            input_hash="abc",
        )
        self.assertTrue(created)
        same, created_again = get_or_create_operation(
            self.session,
            task_id=task.id,
            thread_id=task.thread_id,
            operation_key="task/node/input/action",
            node_name="write",
            input_hash="abc",
        )
        self.assertEqual(same.id, operation.id)
        self.assertFalse(created_again)
        with self.assertRaises(OptimisticConflict):
            get_or_create_operation(
                self.session,
                task_id=task.id,
                thread_id=task.thread_id,
                operation_key="task/node/input/action",
                node_name="write",
                input_hash="changed",
            )

    def test_worker_lease_requires_expiry_plus_takeover_grace(self):
        task = create_task(self.session)
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        lease = acquire_worker_lease(
            self.session,
            task.id,
            "worker-a",
            lease_seconds=60,
            takeover_grace_seconds=30,
            now=start,
        )
        with self.assertRaises(LeaseBusy):
            acquire_worker_lease(self.session, task.id, "worker-b", now=start + timedelta(seconds=89))
        replacement = acquire_worker_lease(
            self.session,
            task.id,
            "worker-b",
            lease_seconds=60,
            takeover_grace_seconds=30,
            now=start + timedelta(seconds=91),
        )
        self.assertEqual(replacement.worker_id, "worker-b")
        with self.assertRaises(LeaseBusy):
            renew_worker_lease(self.session, lease.id, "worker-a", now=start + timedelta(seconds=91))

    def test_candidate_slots_feedback_and_field_patch(self):
        task = create_task(self.session)
        slots = create_candidate_slots(self.session, task.id, 3)
        self.assertEqual([slot.candidate_slot_id for slot in slots], ["slot-1", "slot-2", "slot-3"])
        feedback = record_feedback(self.session, task.id, "请突出缓存和检索链路", target_module="projects")
        self.assertEqual(feedback.status, "received")

        original = {"items": [{"id": "project-1", "text": "old"}, {"id": "project-2", "text": "keep"}]}
        version = GeneratedVersion(
            task_id=task.id,
            module_type="projects",
            version_no=1,
            content=original,
            content_hash=canonical_hash(original),
            source_refs={},
        )
        self.session.add(version)
        self.session.flush()
        patch = ContentPatch(
            task_id=task.id,
            target_version_id=version.id,
            target_module="projects",
            target_item_id="project-1",
            old_value_hash=canonical_hash(original["items"][0]),
            new_value={"id": "project-1", "text": "new"},
            reason="岗位要求突出缓存",
        )
        self.session.add(patch)
        self.session.flush()
        applied = apply_content_patch(self.session, patch.id)
        self.assertEqual(applied.version_no, 2)
        self.assertEqual(applied.content["items"][0]["text"], "new")
        self.assertEqual(version.content["items"][0]["text"], "old")

        stale_patch = ContentPatch(
            task_id=task.id,
            target_version_id=applied.id,
            target_module="projects",
            target_item_id="project-1",
            old_value_hash="stale",
            new_value={"id": "project-1", "text": "bad overwrite"},
        )
        self.session.add(stale_patch)
        self.session.flush()
        with self.assertRaises(PatchConflict):
            apply_content_patch(self.session, stale_patch.id)
        self.assertEqual(stale_patch.status, "conflicted")


if __name__ == "__main__":
    unittest.main()
