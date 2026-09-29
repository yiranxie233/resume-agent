"""Small transaction-oriented persistence services used by API and graph code.

These helpers keep concurrency rules in one place.  They do not know anything
about LangGraph itself; the graph layer passes only stable IDs and references.
All functions expect a caller-owned SQLAlchemy ``Session`` and flush, rather than
commit, so a business write and its checkpoint can share one transaction.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import (
    AppSettings,
    CandidateSlot,
    ContentPatch,
    FeedbackRecord,
    GeneratedVersion,
    LangGraphCheckpoint,
    ModelProfile,
    ModelProfileVersion,
    NodeOperation,
    Task,
    TaskEvent,
    WorkerLease,
    new_id,
    utcnow,
)
from .utils import redact_sensitive, strip_sensitive


class PersistenceError(RuntimeError):
    """Base class for expected persistence-level conflicts."""


class NotFoundError(PersistenceError):
    pass


class OptimisticConflict(PersistenceError):
    """A caller attempted to write using an old version/checkpoint."""

    def __init__(self, message: str, *, current_version: int | None = None) -> None:
        super().__init__(message)
        self.current_version = current_version


class LeaseBusy(PersistenceError):
    """Another worker still owns a task lease."""

    def __init__(self, message: str, *, lease_id: str | None = None, takeover_at: datetime | None = None) -> None:
        super().__init__(message)
        self.lease_id = lease_id
        self.takeover_at = takeover_at


class PatchConflict(PersistenceError):
    """A field-level patch no longer matches the value it was generated from."""

    def __init__(self, message: str, *, patch_id: str | None = None) -> None:
        super().__init__(message)
        self.patch_id = patch_id


def _aware_utc(value: datetime) -> datetime:
    """Normalize timestamps returned by SQLite (which drops tzinfo)."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def canonical_hash(value: Any) -> str:
    """Hash a JSON-compatible value deterministically."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_task(
    session: Session,
    *,
    task_id: str | None = None,
    thread_id: str | None = None,
    status: str = "queued",
    timeout_seconds: int = 300,
    embedding_mode: str = "embedding",
    **fields: Any,
) -> Task:
    """Create a task with a fresh, stable thread ID.

    The caller may provide a thread ID when importing an already-created task,
    but normal new-task flows should omit it.  Existing tasks are never reused
    implicitly; the unique database constraint makes accidental reuse explicit.
    """

    task = Task(
        id=task_id or new_id(),
        thread_id=thread_id or new_id(),
        status=status,
        timeout_seconds=timeout_seconds,
        embedding_mode=embedding_mode,
        **fields,
    )
    session.add(task)
    session.flush()
    return task


def get_task(session: Session, task_id: str, *, for_update: bool = False) -> Task:
    query = select(Task).where(Task.id == task_id)
    if for_update:
        query = query.with_for_update()
    task = session.scalar(query)
    if task is None:
        raise NotFoundError(f"task not found: {task_id}")
    return task


def append_event(
    session: Session,
    task_id: str,
    event_type: str,
    *,
    node_name: str | None = None,
    status: str | None = None,
    message: str | None = None,
    payload: dict[str, Any] | None = None,
) -> TaskEvent:
    """Append a monotonically numbered, non-sensitive task event."""

    # The task row lock serializes sequence assignment on PostgreSQL.  SQLite
    # ignores FOR UPDATE but remains deterministic for the single-worker mode.
    get_task(session, task_id, for_update=True)
    last = session.scalar(select(func.max(TaskEvent.sequence_no)).where(TaskEvent.task_id == task_id))
    event = TaskEvent(
        task_id=task_id,
        sequence_no=(last or 0) + 1,
        event_type=event_type,
        node_name=node_name,
        status=status,
        message=redact_sensitive(message),
        payload=strip_sensitive(payload or {}),
    )
    session.add(event)
    session.flush()
    return event


def commit_checkpoint(
    session: Session,
    *,
    task_id: str,
    thread_id: str,
    expected_version: int,
    node_name: str | None = None,
    state_ref: dict[str, Any] | None = None,
    pending_action: dict[str, Any] | None = None,
    object_refs: dict[str, Any] | None = None,
) -> LangGraphCheckpoint:
    """Append a checkpoint only if the caller saw the latest version.

    The task's ``latest_checkpoint_version`` is the compare-and-swap value.  A
    stale browser tab therefore receives :class:`OptimisticConflict` and cannot
    overwrite the current graph state.
    """

    task = get_task(session, task_id, for_update=True)
    if task.thread_id != thread_id:
        raise OptimisticConflict("thread_id does not belong to task", current_version=task.latest_checkpoint_version)
    if task.latest_checkpoint_version != expected_version:
        raise OptimisticConflict(
            "checkpoint version is stale; refresh before submitting the action",
            current_version=task.latest_checkpoint_version,
        )
    next_version = expected_version + 1
    # Import lazily to avoid an import cycle: db_mirror uses the transaction
    # helpers in this module while exposing the shared compact-state contract.
    from .db_mirror import _compact_pending_action, _compact_state

    checkpoint = LangGraphCheckpoint(
        task_id=task_id,
        thread_id=thread_id,
        version=next_version,
        node_name=node_name,
        state_ref=_compact_state(state_ref),
        pending_action=_compact_pending_action(pending_action),
        object_refs=strip_sensitive(object_refs or {}),
    )
    task.latest_checkpoint_version = next_version
    session.add(checkpoint)
    session.flush()
    return checkpoint


def latest_checkpoint(session: Session, task_id: str) -> LangGraphCheckpoint | None:
    task = get_task(session, task_id)
    return session.scalar(
        select(LangGraphCheckpoint)
        .where(LangGraphCheckpoint.task_id == task_id)
        .order_by(LangGraphCheckpoint.version.desc())
        .limit(1)
    )


def get_or_create_operation(
    session: Session,
    *,
    task_id: str,
    thread_id: str,
    operation_key: str,
    node_name: str,
    input_hash: str,
) -> tuple[NodeOperation, bool]:
    """Return an idempotency record and whether it was newly created."""

    existing = session.scalar(select(NodeOperation).where(NodeOperation.operation_key == operation_key))
    if existing is not None:
        if existing.input_hash != input_hash:
            raise OptimisticConflict("operation_key was reused with a different input")
        return existing, False
    operation = NodeOperation(
        task_id=task_id,
        thread_id=thread_id,
        operation_key=operation_key,
        node_name=node_name,
        input_hash=input_hash,
        status="pending",
    )
    # A concurrent worker may win the unique key between the read and insert.
    # Use a savepoint so handling that race does not roll back the caller's
    # surrounding business transaction.
    try:
        with session.begin_nested():
            session.add(operation)
            session.flush()
    except IntegrityError:
        existing = session.scalar(select(NodeOperation).where(NodeOperation.operation_key == operation_key))
        if existing is None:
            raise
        if existing.input_hash != input_hash:
            raise OptimisticConflict("operation_key was reused with a different input")
        return existing, False
    return operation, True


def finish_operation(
    session: Session,
    operation_id: str,
    *,
    status: str = "committed",
    output_ref: dict[str, Any] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> NodeOperation:
    operation = session.get(NodeOperation, operation_id)
    if operation is None:
        raise NotFoundError(f"node operation not found: {operation_id}")
    operation.status = status
    operation.output_ref = output_ref
    operation.error_code = error_code
    operation.error_message = error_message
    session.flush()
    return operation


def get_or_create_settings(session: Session) -> AppSettings:
    settings = session.get(AppSettings, 1)
    if settings is None:
        candidate = AppSettings(id=1, settings_version=1, values={})
        try:
            with session.begin_nested():
                session.add(candidate)
                session.flush()
            settings = candidate
        except IntegrityError:
            settings = session.get(AppSettings, 1)
            if settings is None:
                raise
    return settings


def update_settings(
    session: Session,
    values: dict[str, Any],
    *,
    expected_version: int,
) -> AppSettings:
    """Atomically update settings using an If-Match-style version check."""

    settings = session.scalar(select(AppSettings).where(AppSettings.id == 1).with_for_update())
    if settings is None:
        if expected_version not in (0, 1):
            raise OptimisticConflict("settings row does not exist", current_version=0)
        settings = AppSettings(id=1, settings_version=1, values=strip_sensitive(copy.deepcopy(values)))
        session.add(settings)
        session.flush()
        return settings
    if settings.settings_version != expected_version:
        raise OptimisticConflict(
            "settings version is stale; refresh before saving",
            current_version=settings.settings_version,
        )
    settings.values = strip_sensitive(copy.deepcopy(values))
    settings.settings_version += 1
    session.flush()
    return settings


def acquire_worker_lease(
    session: Session,
    task_id: str,
    worker_id: str,
    *,
    lease_seconds: int = 60,
    takeover_grace_seconds: int = 30,
    now: datetime | None = None,
) -> WorkerLease:
    """Acquire a PostgreSQL-backed task lease, or raise :class:`LeaseBusy`."""

    now = _aware_utc(now or utcnow())
    get_task(session, task_id, for_update=True)
    active = session.scalar(
        select(WorkerLease)
        .where(WorkerLease.task_id == task_id, WorkerLease.status == "active")
        .order_by(WorkerLease.acquired_at.desc())
        .with_for_update()
    )
    if active is not None:
        expires_at = _aware_utc(active.expires_at)
        takeover_at = _aware_utc(active.takeover_after) if active.takeover_after else (
            expires_at + timedelta(seconds=takeover_grace_seconds)
        )
        if now < takeover_at:
            raise LeaseBusy(
                f"task {task_id} is leased by {active.worker_id}",
                lease_id=active.id,
                takeover_at=takeover_at,
            )
        active.status = "revoked"
        active.released_at = now
    expires_at = now + timedelta(seconds=lease_seconds)
    lease = WorkerLease(
        worker_id=worker_id,
        task_id=task_id,
        status="active",
        acquired_at=now,
        expires_at=expires_at,
        heartbeat_at=now,
        takeover_after=expires_at + timedelta(seconds=takeover_grace_seconds),
    )
    session.add(lease)
    session.flush()
    return lease


def renew_worker_lease(
    session: Session,
    lease_id: str,
    worker_id: str,
    *,
    lease_seconds: int = 60,
    now: datetime | None = None,
) -> WorkerLease:
    now = _aware_utc(now or utcnow())
    lease = session.scalar(select(WorkerLease).where(WorkerLease.id == lease_id).with_for_update())
    if lease is None:
        raise NotFoundError(f"worker lease not found: {lease_id}")
    if lease.worker_id != worker_id or lease.status != "active":
        raise LeaseBusy("worker does not own an active lease", lease_id=lease_id)
    expires_at = _aware_utc(lease.expires_at)
    takeover_after = _aware_utc(lease.takeover_after) if lease.takeover_after else expires_at
    if now >= takeover_after:
        lease.status = "expired"
        session.flush()
        raise LeaseBusy("worker lease has expired", lease_id=lease_id)
    grace = 0
    if lease.takeover_after is not None:
        grace = max(0, int((takeover_after - expires_at).total_seconds()))
    lease.heartbeat_at = now
    lease.expires_at = now + timedelta(seconds=lease_seconds)
    # Preserve the configured takeover grace already stored on the row.
    if lease.takeover_after is not None:
        lease.takeover_after = lease.expires_at + timedelta(seconds=grace)
    session.flush()
    return lease


def release_worker_lease(
    session: Session,
    lease_id: str,
    worker_id: str,
    *,
    now: datetime | None = None,
) -> WorkerLease:
    now = _aware_utc(now or utcnow())
    lease = session.scalar(select(WorkerLease).where(WorkerLease.id == lease_id).with_for_update())
    if lease is None:
        raise NotFoundError(f"worker lease not found: {lease_id}")
    if lease.worker_id != worker_id or lease.status != "active":
        raise LeaseBusy("worker does not own an active lease", lease_id=lease_id)
    lease.status = "released"
    lease.released_at = now
    session.flush()
    return lease


def create_candidate_slots(session: Session, task_id: str, count: int = 3) -> list[CandidateSlot]:
    """Create stable ``slot-1`` … ``slot-N`` rows (bounded to the product limit)."""

    if not 1 <= count <= 5:
        raise ValueError("candidate count must be between 1 and 5")
    existing = list(
        session.scalars(
            select(CandidateSlot)
            .where(CandidateSlot.task_id == task_id)
            .order_by(CandidateSlot.slot_index.asc())
        )
    )
    by_index = {slot.slot_index: slot for slot in existing}
    slots = [by_index[i] for i in range(1, count + 1) if i in by_index]
    missing = [
        CandidateSlot(task_id=task_id, candidate_slot_id=f"slot-{i}", slot_index=i)
        for i in range(1, count + 1)
        if i not in by_index
    ]
    session.add_all(missing)
    session.flush()
    slots.extend(missing)
    return sorted(slots, key=lambda item: item.slot_index)


def record_feedback(
    session: Session,
    task_id: str,
    text: str,
    *,
    target_version_id: str | None = None,
    target_module: str | None = None,
) -> FeedbackRecord:
    if not text.strip():
        raise ValueError("feedback cannot be empty")
    feedback = FeedbackRecord(
        task_id=task_id,
        target_version_id=target_version_id,
        target_module=target_module,
        text=str(redact_sensitive(text)),
    )
    session.add(feedback)
    session.flush()
    return feedback


def snapshot_model_profile(session: Session, profile_id: str) -> ModelProfileVersion:
    """Freeze a model profile for a task or generation branch.

    Profile settings may be edited later, but a task always points at this
    immutable copy so model/tag/tokenizer changes cannot silently alter old
    results.
    """

    profile = session.scalar(select(ModelProfile).where(ModelProfile.id == profile_id))
    if profile is None:
        raise NotFoundError(f"model profile not found: {profile_id}")
    latest = session.scalar(
        select(func.max(ModelProfileVersion.version_no)).where(
            ModelProfileVersion.profile_id == profile_id
        )
    ) or 0
    version = ModelProfileVersion(
        profile_id=profile.id,
        version_no=latest + 1,
        role=profile.role,
        provider=profile.provider,
        base_url=profile.base_url,
        model_name=profile.model_name,
        model_digest=profile.model_digest,
        dimension=profile.dimension,
        context_window_tokens=profile.context_window_tokens,
        tokenizer_id=profile.tokenizer_id,
        tokenizer_version=profile.tokenizer_version,
        tokenizer_source=profile.tokenizer_source,
        estimator_version=profile.estimator_version,
        max_input_tokens=profile.max_input_tokens,
        chunk_policy_version=profile.chunk_policy_version,
        normalize_embeddings=profile.normalize_embeddings,
        distance_metric=profile.distance_metric,
        batch_size=profile.batch_size,
        capabilities=strip_sensitive(copy.deepcopy(profile.capabilities)),
        generation_params=strip_sensitive(copy.deepcopy(profile.generation_params)),
        embedding_algorithm_version=profile.embedding_algorithm_version,
        budget_policy_version=(profile.estimator_version if profile.role == "chat" else None),
        probe_result=strip_sensitive(copy.deepcopy(profile.probe_result)),
    )
    session.add(version)
    session.flush()
    return version


def _replace_item(value: Any, target_id: str, replacement: Any) -> tuple[Any, Any, bool]:
    """Copy a JSON tree and replace the first object carrying ``id``."""

    if isinstance(value, dict):
        if value.get("id") == target_id:
            return copy.deepcopy(replacement), value, True
        result: dict[str, Any] = {}
        for key, child in value.items():
            updated, old, found = _replace_item(child, target_id, replacement)
            result[key] = updated
            if found:
                return result | {k: copy.deepcopy(v) for k, v in value.items() if k not in result}, old, True
        return result, None, False
    if isinstance(value, list):
        result = []
        for child in value:
            updated, old, found = _replace_item(child, target_id, replacement)
            result.append(updated)
            if found:
                # Preserve untouched siblings exactly, while keeping a fresh tree.
                index = len(result) - 1
                result.extend(copy.deepcopy(value[index + 1 :]))
                return result, old, True
        return result, None, False
    return copy.deepcopy(value), None, False


def apply_content_patch(session: Session, patch_id: str) -> GeneratedVersion:
    """Apply one field-level patch to a new immutable draft version.

    The target content convention is a JSON object containing items with stable
    ``id`` fields.  The old value hash is checked before any new version is
    created; a mismatch marks only this patch as conflicted.
    """

    patch = session.get(ContentPatch, patch_id)
    if patch is None:
        raise NotFoundError(f"content patch not found: {patch_id}")
    target = session.get(GeneratedVersion, patch.target_version_id)
    if target is None:
        raise NotFoundError(f"target generated version not found: {patch.target_version_id}")
    sanitized_new_value = strip_sensitive(copy.deepcopy(patch.new_value))
    updated_content, old_value, found = _replace_item(target.content, patch.target_item_id, sanitized_new_value)
    if not found:
        patch.status = "conflicted"
        session.flush()
        raise PatchConflict("patch target item no longer exists", patch_id=patch_id)
    if canonical_hash(old_value) != patch.old_value_hash:
        patch.status = "conflicted"
        session.flush()
        raise PatchConflict("patch old value hash does not match current content", patch_id=patch_id)

    latest_no = session.scalar(
        select(func.max(GeneratedVersion.version_no)).where(
            GeneratedVersion.task_id == target.task_id,
            GeneratedVersion.module_type == target.module_type,
        )
    ) or 0
    new_version = GeneratedVersion(
        task_id=target.task_id,
        module_type=target.module_type,
        version_no=latest_no + 1,
        generation_branch_id=target.generation_branch_id,
        candidate_slot_id=target.candidate_slot_id,
        content=updated_content,
        content_hash=canonical_hash(updated_content),
        source_refs=strip_sensitive(copy.deepcopy(target.source_refs)),
        state="draft",
        is_user_confirmed=False,
    )
    session.add(new_version)
    session.flush()
    patch.status = "applied"
    patch.applied_version_id = new_version.id
    patch.new_value = sanitized_new_value
    patch.diff = {
        "target_item_id": patch.target_item_id,
        "old": redact_sensitive(old_value),
        "new": sanitized_new_value,
    }
    session.flush()
    return new_version
