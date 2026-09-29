from types import SimpleNamespace
import unittest

from app.core.db import create_engine_for_url
from app.core.db_mirror import SqlAlchemyMirror
from app.core.models import LangGraphCheckpoint, Task, TaskEvent


class DatabaseMirrorTests(unittest.TestCase):
    def test_mirror_persists_only_compact_checkpoint_state(self):
        engine = create_engine_for_url("sqlite+pysqlite:///:memory:")
        mirror = SqlAlchemyMirror(engine, initialize=True)
        task = SimpleNamespace(
            task_id="task-1",
            thread_id="thread-1",
            status="running",
            blocked_reason=None,
            current_node="match",
            checkpoint_version=1,
            job_snapshot_id="job-snapshot-1",
            candidates=[],
            state={
                "job": {"requirements": "must not be copied"},
                "resume": {"phone": "must not be copied"},
                "candidates": [{"text": "must not be copied"}],
                "messages": ["must not be copied"],
                "embedding_mode": "tfidf_fallback",
                "selected_candidate": "slot-1",
                "chat_profile_id": "chat-1",
                "embedding_profile_id": "embedding-1",
                "retry_step": "candidate_generation",
                "base_facts": {"name": "张三", "api_key": "must-not-persist"},
            },
        )
        self.assertTrue(mirror.persist_checkpoint(task))
        self.assertTrue(mirror.persist_event("task-1", {"type": "status", "status": "running", "node": "match", "job": "secret"}))

        factory = mirror.factory
        with factory() as session:
            row = session.get(Task, "task-1")
            checkpoint = session.query(LangGraphCheckpoint).one()
            event = session.query(TaskEvent).one()
            self.assertIsNotNone(row)
            self.assertEqual(
                row.extra,
                {
                    "embedding_mode": "tfidf_fallback",
                    "selected_candidate": "slot-1",
                    "chat_profile_id": "chat-1",
                    "embedding_profile_id": "embedding-1",
                    "retry_step": "candidate_generation",
                    "base_facts": {"name": "张三"},
                },
            )
            self.assertEqual(checkpoint.state_ref, row.extra)
            self.assertNotIn("secret", event.payload)
            self.assertEqual(event.event_type, "status")

        mirror.close()


if __name__ == "__main__":
    unittest.main()
