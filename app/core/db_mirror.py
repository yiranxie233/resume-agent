"""Optional PostgreSQL mirror for the first vertical workflow slice.

``WorkflowEngine`` currently keeps rich Pydantic objects in the in-process store
for fast UI interaction.  This mirror persists the durable identifiers and
checkpoint/event audit trail without copying full resume/JD text into graph
state.  It is deliberately optional: callers can inject it only after the
database health check succeeds.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .db import init_db, session_factory, session_scope
from .models import LangGraphCheckpoint, Task
from .persistence import append_event
from .utils import redact_sensitive, strip_sensitive


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _without_secrets(value: Any) -> Any:
    return strip_sensitive(value)


def _compact_pending_action(value: Any) -> dict[str, Any] | None:
    """Keep routing metadata from an interrupt, never its editable payload."""

    if not isinstance(value, dict):
        return None
    allowed = {
        "action",
        "type",
        "module",
        "target_id",
        "target_project_id",
        "candidate_id",
        "candidate_slot_id",
        "checkpoint_version",
    }
    result = {
        str(name): _without_secrets(item)
        for name, item in value.items()
        if str(name) in allowed
    }
    return result or None


def _compact_base_facts(value: Any) -> dict[str, Any]:
    """Keep only the explicit no-upload identity fields.

    ``base_facts`` is user input, so treating it as an unrestricted JSON blob
    would let a client smuggle a complete resume/JD into a checkpoint.  The
    workflow only needs these fields to start a no-upload task; rich material
    belongs in a resume/business version row after parsing.
    """

    if not isinstance(value, dict):
        return {}
    allowed = {
        "name",
        "phone",
        "email",
        "location",
        "city",
        "education",
        "objective",
        "certificates",
        "target_role",
    }
    def bounded(item: Any, depth: int = 0) -> Any:
        item = _without_secrets(item)
        if depth > 3:
            return str(item)[:1000]
        if isinstance(item, dict):
            return {
                str(key)[:64]: bounded(entry, depth + 1)
                for key, entry in list(item.items())[:20]
                if str(key).lower() not in {"resume_text", "raw_text", "jd", "job_description"}
            }
        if isinstance(item, list):
            return [bounded(entry, depth + 1) for entry in item[:20]]
        if isinstance(item, str):
            return item[:4000]
        if isinstance(item, (int, float, bool)) or item is None:
            return item
        return str(item)[:1000]

    result: dict[str, Any] = {}
    for name in allowed:
        if name not in value:
            continue
        result[name] = bounded(value[name])
    return result


def _compact_state(state: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only graph-control fields and stable IDs in a checkpoint.

    Rich fields are intentionally omitted even if a legacy workflow passes them
    under ``job``, ``resume`` or ``candidates``.  The business tables/files are
    the content source of truth per the technical design.
    """

    if not state:
        return {}
    allowed = {
        "task_id",
        "thread_id",
        "current_node",
        "status",
        "blocked_reason",
        "candidate_count",
        "embedding_mode",
        "chat_profile_id",
        "chat_profile_version",
        "embedding_profile_id",
        "embedding_profile_version",
        # A no-upload task needs the user's confirmed base facts to resume
        # after restart.  Keep this object sanitized and compact; rich resume
        # text and binary files remain in their business tables/artifacts.
        "base_facts",
        "template_id",
        "template_structure_snapshot",
        "requested_output_mode",
        "final_product",
        "dimension_weights",
        "component_weights",
        "scoring_config_version",
        "selected_candidate",
        "selected_candidate_id",
        "generation_branch_id",
        "last_feedback_id",
        "retry_step",
        "last_match_run_id",
        "last_snapshot_id",
        "resume_snapshot_id",
        "candidate_generation_status",
        "rejected_candidate_ids",
        "pending_module",
        "confirmed_candidate_id",
        "project_decision",
        "module_assessment",
        "module_decisions",
        "layout_decision",
        "compression_decision",
        "preview_status",
        "previous_snapshot_ids",
        "final_confirmation",
        "output_mode",
        "allow_rewrite",
        "pending_action",
        "feedback_classification",
        "model_audit",
        "model_binding_error",
    }
    compact: dict[str, Any] = {}
    for key in allowed:
        if key in state and key not in {"job", "resume", "candidates", "messages"}:
            value = state[key]
            # Keep only JSON-safe scalar/list/dict references.  Pydantic models
            # or arbitrary objects are never serialized into checkpoint rows.
            try:
                import json

                json.dumps(value, ensure_ascii=False)
            except (TypeError, ValueError):
                continue
            value = _without_secrets(value)
            if key == "base_facts":
                value = _compact_base_facts(value)
            if key == "selected_candidate" and not isinstance(value, (str, int, float, bool, type(None))):
                value = value.get("candidate_id") if isinstance(value, dict) else None
            if key == "pending_action":
                value = _compact_pending_action(value)
            if key == "feedback_classification" and isinstance(value, dict):
                value = {
                    name: value.get(name)
                    for name in (
                        "classifier_type",
                        "category",
                        "confidence",
                        "requires_clarification",
                    )
                    if value.get(name) is not None
                }
            if key == "model_audit" and isinstance(value, dict):
                # Keep audit hashes/metadata, never the rendered prompt or
                # complete provider response.
                value = {
                    name: value.get(name)
                    for name in ("request_key", "context_manifest_hash", "model_name", "profile_version", "fallback_reason")
                    if value.get(name) is not None
                }
            compact[key] = copy.deepcopy(value)
    return compact


class SqlAlchemyMirror:
    """Persist compact workflow state using caller-owned SQLAlchemy sessions."""

    def __init__(
        self,
        engine: Engine,
        *,
        factory: sessionmaker[Session] | None = None,
        initialize: bool = False,
    ) -> None:
        self.engine = engine
        self.factory = factory or session_factory(engine)
        if initialize:
            init_db(engine)

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with session_scope(self.factory) as session:
            yield session

    def persist_checkpoint(self, task: Any) -> bool:
        """Upsert a task's metadata and append its new checkpoint idempotently."""

        task_id = str(task.task_id)
        thread_id = str(task.thread_id)
        version = int(getattr(task, "checkpoint_version", 0))
        status = _enum_value(getattr(task, "status", "queued"))
        state = getattr(task, "state", {}) or {}
        with self._session() as session:
            row = session.get(Task, task_id)
            if row is None:
                row = Task(
                    id=task_id,
                    thread_id=thread_id,
                    status=status,
                    blocked_reason=getattr(task, "blocked_reason", None),
                    current_node=getattr(task, "current_node", None),
                    job_snapshot_id=getattr(task, "job_snapshot_id", None),
                    embedding_mode=str(state.get("embedding_mode", "tfidf_fallback")),
                    latest_checkpoint_version=version,
                    extra=_compact_state(state),
                )
                session.add(row)
                session.flush()
            else:
                if row.thread_id != thread_id:
                    raise ValueError("task/thread_id mismatch while mirroring checkpoint")
                if version < row.latest_checkpoint_version:
                    # Never let an old in-memory snapshot overwrite durable state.
                    raise ValueError("mirror checkpoint version is older than database")
                row.status = status
                row.blocked_reason = getattr(task, "blocked_reason", None)
                row.current_node = getattr(task, "current_node", None)
                row.latest_checkpoint_version = version
                row.extra = _compact_state(state)
                session.flush()

            if version <= 0:
                return
            existing = session.scalar(
                select(LangGraphCheckpoint).where(
                    LangGraphCheckpoint.task_id == task_id,
                    LangGraphCheckpoint.thread_id == thread_id,
                    LangGraphCheckpoint.version == version,
                )
            )
            if existing is None:
                session.add(
                    LangGraphCheckpoint(
                        task_id=task_id,
                        thread_id=thread_id,
                        version=version,
                        node_name=getattr(task, "current_node", None),
                        state_ref=_compact_state(state),
                        pending_action=_compact_pending_action(state.get("pending_action")) if isinstance(state, dict) else None,
                        object_refs={
                            "job_snapshot_id": getattr(task, "job_snapshot_id", None),
                            "candidate_ids": [
                                getattr(candidate, "candidate_id", None)
                                for candidate in (getattr(task, "candidates", None) or [])
                                if getattr(candidate, "candidate_id", None)
                            ],
                        },
                    )
                )
            return True

    def persist_event(self, task_id: str, event: dict[str, Any]) -> bool:
        """Append one sanitized event; full resume/JD payloads are omitted."""

        allowed = {"type", "status", "node", "error_code", "message", "version", "phase"}
        payload = strip_sensitive({key: value for key, value in event.items() if key in allowed})
        event_type = str(payload.pop("type", "progress"))
        with self._session() as session:
            append_event(
                session,
                task_id,
                event_type,
                node_name=payload.pop("node", None),
                status=payload.pop("status", None),
                message=payload.pop("message", None),
                payload=payload,
            )
            return True

    def healthcheck(self) -> bool:
        """Run a lightweight connection check for the environment page."""

        from sqlalchemy import text

        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True

    def close(self) -> None:
        self.engine.dispose()
