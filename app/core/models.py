"""SQLAlchemy persistence models.

The models intentionally use portable SQLAlchemy types (including JSON and text
UUIDs) so the same schema can be exercised with SQLite in unit tests and run on
the PostgreSQL Docker service in production.  Business objects are immutable by
convention once referenced by a task; new versions are represented by new rows.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# Business identifiers carry a human-readable prefix in the local API (for
# example ``task_<uuid>`` and ``snapshot_<job_id>``).  Keep the database width
# larger than a bare UUID so PostgreSQL and SQLite enforce the same contract.
IDENTIFIER_LENGTH = 128


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Declarative base with deterministic names for migration constraints."""

    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class Task(TimestampMixin, Base):
    __tablename__ = "tasks"
    __table_args__ = (
        UniqueConstraint("thread_id", name="uq_tasks_thread_id"),
        CheckConstraint(
            "status IN ('queued','running','waiting_user','paused','failed','cancelled','completed','deleted')",
            name="task_status_values",
        ),
        CheckConstraint(
            "embedding_mode IN ('embedding','tfidf_fallback')",
            name="task_embedding_mode_values",
        ),
        Index("ix_tasks_status_queue", "status", "queue_position"),
    )

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", nullable=False, index=True)
    blocked_reason: Mapped[str | None] = mapped_column(String(64))
    current_node: Mapped[str | None] = mapped_column(String(128))
    job_snapshot_id: Mapped[str | None] = mapped_column(ForeignKey("job_snapshots.id"))
    # Deferred on PostgreSQL to avoid a creation cycle with resume_files.task_id.
    resume_file_id: Mapped[str | None] = mapped_column(
        ForeignKey("resume_files.id", use_alter=True, name="fk_tasks_resume_file_id_resume_files")
    )
    chat_model_profile_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_profile_versions.id")
    )
    embedding_model_profile_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_profile_versions.id")
    )
    embedding_mode: Mapped[str] = mapped_column(String(32), default="embedding", nullable=False)
    generation_branch_id: Mapped[str | None] = mapped_column(String(64))
    queue_position: Mapped[int | None] = mapped_column(Integer)
    timeout_seconds: Mapped[int] = mapped_column(Integer, default=300, nullable=False)
    latest_checkpoint_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(96))
    error_message: Mapped[str | None] = mapped_column(Text)
    recovery_note: Mapped[str | None] = mapped_column(Text)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class TaskEvent(Base):
    __tablename__ = "task_events"
    __table_args__ = (
        UniqueConstraint("task_id", "sequence_no", name="uq_task_events_task_sequence"),
        Index("ix_task_events_task_created", "task_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    node_name: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str | None] = mapped_column(String(32))
    message: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class NodeOperation(TimestampMixin, Base):
    __tablename__ = "node_operations"
    __table_args__ = (
        UniqueConstraint("operation_key", name="uq_node_operations_operation_key"),
        CheckConstraint(
            "status IN ('pending','committed','rolled_back','needs_review')",
            name="node_operation_status_values",
        ),
        Index("ix_node_operations_task_status", "task_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    thread_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("tasks.thread_id", ondelete="CASCADE"), nullable=False
    )
    operation_key: Mapped[str] = mapped_column(String(512), nullable=False)
    node_name: Mapped[str] = mapped_column(String(128), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="pending", nullable=False)
    output_ref: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_code: Mapped[str | None] = mapped_column(String(96))
    error_message: Mapped[str | None] = mapped_column(Text)


class NodeAttempt(Base):
    __tablename__ = "node_attempts"
    __table_args__ = (UniqueConstraint("operation_id", "attempt_no", name="uq_node_attempts_number"),)

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    operation_id: Mapped[str | None] = mapped_column(ForeignKey("node_operations.id", ondelete="CASCADE"))
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    node_name: Mapped[str] = mapped_column(String(128), nullable=False)
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="running", nullable=False)
    request_key: Mapped[str | None] = mapped_column(String(512))
    error_code: Mapped[str | None] = mapped_column(String(96))
    error_message: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ContextSnapshot(Base):
    __tablename__ = "context_snapshots"

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    node_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_profile_version_id: Mapped[str | None] = mapped_column(ForeignKey("model_profile_versions.id"))
    tokenizer_id: Mapped[str | None] = mapped_column(String(128))
    tokenizer_version: Mapped[str | None] = mapped_column(String(64))
    estimator_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    budget_policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_context_window_tokens: Mapped[int | None] = mapped_column(Integer)
    max_output_tokens: Mapped[int | None] = mapped_column(Integer)
    safety_margin_tokens: Mapped[int | None] = mapped_column(Integer)
    usable_input_tokens: Mapped[int | None] = mapped_column(Integer)
    estimated_input_tokens: Mapped[int | None] = mapped_column(Integer)
    included_object_ids: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    omitted_object_ids: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    trim_order: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    blocked_reason: Mapped[str | None] = mapped_column(String(96))
    context_manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ModelInvocation(Base):
    __tablename__ = "model_invocations"
    __table_args__ = (UniqueConstraint("request_key", name="uq_model_invocations_request_key"),)

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    node_name: Mapped[str] = mapped_column(String(128), nullable=False)
    request_key: Mapped[str] = mapped_column(String(512), nullable=False)
    model_profile_version_id: Mapped[str | None] = mapped_column(ForeignKey("model_profile_versions.id"))
    context_snapshot_id: Mapped[str | None] = mapped_column(ForeignKey("context_snapshots.id"))
    context_manifest_hash: Mapped[str | None] = mapped_column(String(128))
    evidence_ids: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    response_ref: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    response_schema_version: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="running", nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(96))
    error_summary: Mapped[str | None] = mapped_column(Text)
    debug_prompt_ref: Mapped[str | None] = mapped_column(String(512))
    debug_response_ref: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class WorkerLease(Base):
    __tablename__ = "worker_leases"
    __table_args__ = (Index("ix_worker_leases_task_active", "task_id", "status", "expires_at"),)

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    worker_id: Mapped[str] = mapped_column(String(128), nullable=False)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="active", nullable=False)
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    takeover_after: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LangGraphCheckpoint(Base):
    __tablename__ = "langgraph_checkpoints"
    __table_args__ = (
        UniqueConstraint("thread_id", "version", name="uq_checkpoints_thread_version"),
        Index("ix_checkpoints_thread_latest", "thread_id", "version"),
    )

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    thread_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("tasks.thread_id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    node_name: Mapped[str | None] = mapped_column(String(128))
    state_ref: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    pending_action: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    object_refs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class JobRecord(TimestampMixin, Base):
    __tablename__ = "job_records"
    __table_args__ = (Index("ix_job_records_title_city", "job_title", "city"),)

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_key: Mapped[str | None] = mapped_column(String(256), index=True)
    job_title: Mapped[str] = mapped_column(String(256), nullable=False)
    company_name: Mapped[str | None] = mapped_column(String(256))
    city: Mapped[str | None] = mapped_column(String(128))
    salary: Mapped[str | None] = mapped_column(String(128))
    responsibilities: Mapped[str | None] = mapped_column(Text)
    requirements: Mapped[str | None] = mapped_column(Text)
    skills: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    posted_time_status: Mapped[str] = mapped_column(String(32), default="unknown", nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    detail_snapshot_path: Mapped[str | None] = mapped_column(Text)
    detail_snapshot_hash: Mapped[str | None] = mapped_column(String(128))
    parser_version: Mapped[str | None] = mapped_column(String(64))
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class JobSnapshot(Base):
    __tablename__ = "job_snapshots"

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    job_record_id: Mapped[str | None] = mapped_column(ForeignKey("job_records.id"))
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    job_title: Mapped[str] = mapped_column(String(256), nullable=False)
    company_name: Mapped[str | None] = mapped_column(String(256))
    city: Mapped[str | None] = mapped_column(String(128))
    salary: Mapped[str | None] = mapped_column(String(128))
    responsibilities: Mapped[str | None] = mapped_column(Text)
    requirements: Mapped[str | None] = mapped_column(Text)
    skills: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    posted_time_status: Mapped[str] = mapped_column(String(32), default="unknown", nullable=False)
    evidence_ids: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    raw_snapshot_ref: Mapped[str | None] = mapped_column(Text)
    parser_version: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class JobEvidence(Base):
    __tablename__ = "job_evidence"
    __table_args__ = (UniqueConstraint("snapshot_id", "evidence_id", name="uq_job_evidence_snapshot_id"),)

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("job_snapshots.id", ondelete="CASCADE"), nullable=False)
    evidence_id: Mapped[str] = mapped_column(String(128), nullable=False)
    field_name: Mapped[str | None] = mapped_column(String(128))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    source_locator: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ResumeFile(TimestampMixin, Base):
    __tablename__ = "resume_files"

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id", ondelete="SET NULL"))
    path: Mapped[str] = mapped_column(Text, nullable=False)
    original_name: Mapped[str | None] = mapped_column(String(256))
    media_type: Mapped[str] = mapped_column(String(128), nullable=False)
    format: Mapped[str] = mapped_column(String(16), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(128), nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    file_role: Mapped[str] = mapped_column(String(32), default="source", nullable=False)
    validation_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class ResumeSection(Base):
    __tablename__ = "resume_sections"
    __table_args__ = (UniqueConstraint("resume_file_id", "section_id", name="uq_resume_sections_file_section"),)

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    resume_file_id: Mapped[str] = mapped_column(ForeignKey("resume_files.id", ondelete="CASCADE"), nullable=False)
    section_id: Mapped[str] = mapped_column(String(128), nullable=False)
    module_type: Mapped[str] = mapped_column(String(64), nullable=False)
    item_index: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    page_no: Mapped[int | None] = mapped_column(Integer)
    evidence_id: Mapped[str | None] = mapped_column(String(128))
    source_locator: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    user_corrected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    style_ref: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class TemplateRecord(TimestampMixin, Base):
    __tablename__ = "template_records"

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    repository: Mapped[str | None] = mapped_column(String(256))
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(32), default="zh", nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    license_name: Mapped[str | None] = mapped_column(String(128))
    license_allows_modification: Mapped[bool | None] = mapped_column(Boolean)
    stars: Mapped[int | None] = mapped_column(Integer)
    remote_version: Mapped[str | None] = mapped_column(String(256))
    local_version: Mapped[str | None] = mapped_column(String(256))
    preview_path: Mapped[str | None] = mapped_column(Text)
    update_status: Mapped[str | None] = mapped_column(String(32))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class ScoringConfig(Base):
    __tablename__ = "scoring_configs"

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    version: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    dimension_weights: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    deterministic_weight: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    semantic_weight: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    rubric_version: Mapped[str] = mapped_column(String(64), nullable=False)
    rounding_digits: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, default=80.0, nullable=False)
    algorithm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class MatchRun(Base):
    __tablename__ = "match_runs"

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    job_snapshot_id: Mapped[str] = mapped_column(ForeignKey("job_snapshots.id"), nullable=False)
    resume_file_id: Mapped[str | None] = mapped_column(ForeignKey("resume_files.id"))
    scoring_config_id: Mapped[str | None] = mapped_column(ForeignKey("scoring_configs.id"))
    skill_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    responsibility_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    project_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    keyword_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_score_raw: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    weights: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    embedding_mode: Mapped[str] = mapped_column(String(32), default="embedding", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class GeneratedVersion(Base):
    __tablename__ = "generated_versions"
    __table_args__ = (Index("ix_generated_versions_task_module", "task_id", "module_type", "created_at"),)

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    module_type: Mapped[str] = mapped_column(String(64), nullable=False)
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    generation_branch_id: Mapped[str | None] = mapped_column(String(64))
    candidate_slot_id: Mapped[str | None] = mapped_column(String(64))
    content: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    source_refs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    state: Mapped[str] = mapped_column(String(32), default="draft", nullable=False)
    is_user_confirmed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ContentPatch(Base):
    __tablename__ = "content_patches"

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    target_version_id: Mapped[str] = mapped_column(ForeignKey("generated_versions.id"), nullable=False)
    target_module: Mapped[str] = mapped_column(String(64), nullable=False)
    target_item_id: Mapped[str] = mapped_column(String(128), nullable=False)
    old_value_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    new_value: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    evidence_ids: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    applied_version_id: Mapped[str | None] = mapped_column(ForeignKey("generated_versions.id"))
    diff: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ResumeSnapshot(Base):
    __tablename__ = "resume_snapshots"

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    resume_file_id: Mapped[str | None] = mapped_column(ForeignKey("resume_files.id"))
    module_version_ids: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    project_order: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    template_id: Mapped[str | None] = mapped_column(ForeignKey("template_records.id"))
    layout_version: Mapped[str | None] = mapped_column(String(64))
    confirmation_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    snapshot_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    state: Mapped[str] = mapped_column(String(32), default="draft", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class FeedbackRecord(Base):
    __tablename__ = "feedback_records"

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    target_version_id: Mapped[str | None] = mapped_column(ForeignKey("generated_versions.id"))
    target_module: Mapped[str | None] = mapped_column(String(64))
    text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="received", nullable=False)
    classification_id: Mapped[str | None] = mapped_column(ForeignKey("feedback_classifications.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class FeedbackClassification(Base):
    __tablename__ = "feedback_classifications"

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    classifier_type: Mapped[str] = mapped_column(String(32), nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    spans: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    fact_candidates: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    conflicts: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    requires_clarification: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class Confirmation(Base):
    __tablename__ = "confirmations"

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    field_id: Mapped[str] = mapped_column(String(128), nullable=False)
    value_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    source_version_id: Mapped[str | None] = mapped_column(String(128))
    patch_version_id: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default="confirmed", nullable=False)
    confirmed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class SkillVersion(Base):
    __tablename__ = "skill_versions"

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    diff: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    is_current: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ModelProfile(TimestampMixin, Base):
    __tablename__ = "model_profiles"
    __table_args__ = (
        CheckConstraint("role IN ('chat','embedding')", name="model_profile_role_values"),
        CheckConstraint("provider IN ('ollama','openai_compatible')", name="model_profile_provider_values"),
        Index("ix_model_profiles_role_default", "role", "is_default"),
    )

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    base_url: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String(256), nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    credential_required: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    configured_in_session: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="unknown", nullable=False)
    model_digest: Mapped[str | None] = mapped_column(String(256))
    dimension: Mapped[int | None] = mapped_column(Integer)
    context_window_tokens: Mapped[int | None] = mapped_column(Integer)
    tokenizer_id: Mapped[str | None] = mapped_column(String(128))
    tokenizer_version: Mapped[str | None] = mapped_column(String(64))
    tokenizer_source: Mapped[str | None] = mapped_column(String(64))
    estimator_version: Mapped[str | None] = mapped_column(String(64))
    embedding_algorithm_version: Mapped[str | None] = mapped_column(String(64))
    max_input_tokens: Mapped[int | None] = mapped_column(Integer)
    chunk_policy_version: Mapped[str | None] = mapped_column(String(64))
    normalize_embeddings: Mapped[bool | None] = mapped_column(Boolean)
    distance_metric: Mapped[str | None] = mapped_column(String(32))
    batch_size: Mapped[int | None] = mapped_column(Integer)
    capabilities: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    probe_result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    last_probe_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    generation_params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class ModelProfileVersion(Base):
    __tablename__ = "model_profile_versions"
    __table_args__ = (UniqueConstraint("profile_id", "version_no", name="uq_model_profile_versions_number"),)

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    profile_id: Mapped[str] = mapped_column(ForeignKey("model_profiles.id", ondelete="CASCADE"), nullable=False)
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    base_url: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String(256), nullable=False)
    model_digest: Mapped[str | None] = mapped_column(String(256))
    dimension: Mapped[int | None] = mapped_column(Integer)
    context_window_tokens: Mapped[int | None] = mapped_column(Integer)
    tokenizer_id: Mapped[str | None] = mapped_column(String(128))
    tokenizer_version: Mapped[str | None] = mapped_column(String(64))
    tokenizer_source: Mapped[str | None] = mapped_column(String(64))
    estimator_version: Mapped[str | None] = mapped_column(String(64))
    max_input_tokens: Mapped[int | None] = mapped_column(Integer)
    chunk_policy_version: Mapped[str | None] = mapped_column(String(64))
    normalize_embeddings: Mapped[bool | None] = mapped_column(Boolean)
    distance_metric: Mapped[str | None] = mapped_column(String(32))
    batch_size: Mapped[int | None] = mapped_column(Integer)
    capabilities: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    generation_params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    embedding_algorithm_version: Mapped[str | None] = mapped_column(String(64))
    budget_policy_version: Mapped[str | None] = mapped_column(String(64))
    probe_result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class EmbeddingArtifact(Base):
    __tablename__ = "embedding_artifacts"
    __table_args__ = (
        UniqueConstraint("content_hash", "model_profile_version_id", name="uq_embedding_content_profile"),
    )

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    model_profile_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_profile_versions.id", ondelete="CASCADE"), nullable=False
    )
    model_digest: Mapped[str | None] = mapped_column(String(256))
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[list[Any] | None] = mapped_column(JSON)
    index_ref: Mapped[str | None] = mapped_column(Text)
    algorithm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="ready", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class CandidateSlot(Base):
    __tablename__ = "candidate_slots"
    __table_args__ = (UniqueConstraint("task_id", "candidate_slot_id", name="uq_candidate_slots_task_slot"),)

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    candidate_slot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    slot_index: Mapped[int] = mapped_column(Integer, nullable=False)
    generation_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    generated_version_id: Mapped[str | None] = mapped_column(ForeignKey("generated_versions.id"))
    dedup_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    ranking_score: Mapped[float | None] = mapped_column(Float)
    discard_reason: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(String(96))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class Consent(Base):
    __tablename__ = "consents"
    __table_args__ = (UniqueConstraint("scope", "task_id", name="uq_consents_scope_task"),)

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    scope: Mapped[str] = mapped_column(String(64), nullable=False)
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    granted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    granted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class ExportRun(Base):
    __tablename__ = "export_runs"

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("resume_snapshots.id"), nullable=False)
    format: Mapped[str] = mapped_column(String(16), nullable=False)
    engine: Mapped[str | None] = mapped_column(String(32))
    output_path: Mapped[str | None] = mapped_column(Text)
    preview_path: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    compliance_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    feedback: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(String(96))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class BackupManifest(Base):
    __tablename__ = "backup_manifests"

    id: Mapped[str] = mapped_column(String(IDENTIFIER_LENGTH), primary_key=True, default=new_id)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    file_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="created", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    restored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class AppSettings(Base):
    """Singleton settings row used for If-Match/settings_version updates."""

    __tablename__ = "app_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    settings_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    values: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )
