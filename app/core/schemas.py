"""Pydantic contracts shared by the API, graph and persistence layers.

The schemas deliberately keep generated content separate from confirmed content.  A
caller can therefore inspect a draft without accidentally treating it as exportable.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Provider(StrEnum):
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


class ModelRole(StrEnum):
    CHAT = "chat"
    EMBEDDING = "embedding"


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    PAUSED = "paused"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    DELETED = "deleted"


class BlockedReason(StrEnum):
    NEEDS_CREDENTIALS = "needs_credentials"
    NEEDS_MODEL_RECHECK = "needs_model_recheck"
    REINDEX_REQUIRED = "reindex_required"
    CONTEXT_BUDGET_BLOCKED = "context_budget_blocked"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source_type: Literal["job", "resume", "user", "model"]
    text: str
    location: str | None = None
    text_hash: str


class JobInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    company: str | None = Field(default=None, max_length=200)
    city: str | None = Field(default=None, max_length=100)
    salary: str | None = Field(default=None, max_length=100)
    responsibilities: list[str] = Field(default_factory=list, max_length=100)
    requirements: list[str] = Field(default_factory=list, max_length=100)
    skills: list[str] = Field(default_factory=list, max_length=100)
    hr_name: str | None = Field(default=None, max_length=100)
    hr_activity: str | None = Field(default=None, max_length=100)
    posted_at: datetime | None = None
    posted_at_label: str | None = Field(default=None, max_length=100)
    source_url: str | None = None
    raw_text: str | None = None

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        parsed = urlparse(value)
        if parsed.scheme.lower() == "file":
            raise ValueError("file:// URLs are not allowed")
        if parsed.scheme and parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError("source_url must use http or https")
        return value


class JobCreateRequest(JobInput):
    pass


class JobUrlRequest(BaseModel):
    url: str
    title: str | None = Field(default=None, max_length=200)
    company: str | None = Field(default=None, max_length=200)
    city: str | None = Field(default=None, max_length=100)
    raw_text: str | None = Field(default=None, max_length=200_000)
    posted_time_label: str | None = Field(default=None, max_length=100)
    chat_profile_id: str | None = Field(default=None, max_length=200)
    credential_handle_id: str | None = Field(default=None, max_length=200)

    @field_validator("url")
    @classmethod
    def reject_file_url(cls, value: str) -> str:
        value = value.strip()
        if value.lower().startswith("file:"):
            raise ValueError("file:// URLs are not allowed")
        return value


class ScreenshotConfirmRequest(BaseModel):
    """User-corrected OCR text that may be promoted to a real job record."""

    ocr_id: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=200_000)
    title: str | None = Field(default=None, max_length=200)
    company: str | None = Field(default=None, max_length=200)
    city: str | None = Field(default=None, max_length=100)
    salary: str | None = Field(default=None, max_length=100)
    hr_name: str | None = Field(default=None, max_length=100)
    hr_activity: str | None = Field(default=None, max_length=100)
    chat_profile_id: str | None = Field(default=None, max_length=200)
    credential_handle_id: str | None = Field(default=None, max_length=200)


class ResumeSection(BaseModel):
    section_id: str
    module: Literal[
        "basic",
        "summary",
        "education",
        "work",
        "projects",
        "skills",
        "evaluation",
        "certificates",
        "objective",
        "other",
    ]
    title: str | None = None
    content: str
    confirmed: bool = False
    value_hash: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ResumeDocument(BaseModel):
    resume_id: str
    filename: str
    file_type: Literal["docx", "pdf", "markdown", "text"]
    file_size: int
    page_count: int | None = None
    sections: list[ResumeSection] = Field(default_factory=list)
    raw_text: str = ""


class ModelProfile(BaseModel):
    profile_id: str
    role: ModelRole
    provider: Provider
    base_url: str
    model_name: str
    status: str = "unprobed"
    default_for_role: bool = False
    credential_required: bool = False
    context_window_tokens: int | None = None
    tokenizer_id: str | None = None
    tokenizer_version: str | None = None
    tokenizer_source: str | None = None
    dimension: int | None = None
    max_input_tokens: int | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    generation_params: dict[str, Any] = Field(default_factory=dict)
    config_version: int = Field(default=1, ge=1)
    last_probe_at: datetime | None = None
    error_code: str | None = None


class TaskCreateRequest(BaseModel):
    job_id: str | None = None
    job: JobInput | None = None
    resume_id: str | None = None
    template_id: str | None = Field(default=None, max_length=128)
    output_mode: Literal["resume_edit", "template_resume", "project_only"] = "resume_edit"
    base_facts: dict[str, Any] = Field(default_factory=dict)
    candidate_count: int = Field(default=3, ge=1, le=5)
    allow_embedding_fallback: bool = True
    chat_profile_id: str | None = None
    embedding_profile_id: str | None = None
    credential_handle_id: str | None = None
    embedding_credential_handle_id: str | None = None
    dimension_weights: dict[str, float] | None = None
    component_weights: dict[str, float] | None = None

    @model_validator(mode="after")
    def require_job_source(self) -> "TaskCreateRequest":
        if not self.job_id and self.job is None:
            raise ValueError("job or job_id is required")
        if self.job_id and self.job is not None:
            raise ValueError("provide either job_id or job, not both")
        expected_dimensions = {"skills", "responsibilities", "projects", "keywords"}
        if self.dimension_weights is not None:
            if set(self.dimension_weights) != expected_dimensions:
                raise ValueError("dimension_weights must contain skills, responsibilities, projects and keywords")
            if any(value < 0 or value > 1 for value in self.dimension_weights.values()):
                raise ValueError("dimension_weights values must be between 0 and 1")
            if abs(sum(self.dimension_weights.values()) - 1.0) > 1e-8:
                raise ValueError("dimension_weights must sum to 1")
        if self.component_weights is not None:
            if set(self.component_weights) != {"deterministic", "semantic"}:
                raise ValueError("component_weights must contain deterministic and semantic")
            if any(value < 0 or value > 1 for value in self.component_weights.values()):
                raise ValueError("component_weights values must be between 0 and 1")
            if abs(sum(self.component_weights.values()) - 1.0) > 1e-8:
                raise ValueError("component_weights must sum to 1")
        return self


class ActionRequest(BaseModel):
    thread_id: str
    checkpoint_version: int = Field(ge=0)
    action: Literal[
        "approve",
        "reject",
        "edit",
        "revise",
        "select",
        "skip",
        "resume",
        "cancel",
        "confirm",
        "retry",
    ]
    payload: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    thread_id: str
    checkpoint_version: int = Field(ge=0)
    feedback: str = Field(min_length=1, max_length=5000)
    target_version_id: str | None = None
    chat_profile_id: str | None = None
    credential_handle_id: str | None = None


class PatchRequest(BaseModel):
    thread_id: str
    checkpoint_version: int = Field(ge=0)
    module: str
    target_id: str
    old_value_hash: str
    new_value: str = Field(max_length=100_000)
    reason: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(default_factory=list)


class SettingsUpdate(BaseModel):
    settings_version: int = Field(ge=0)
    values: dict[str, Any] = Field(default_factory=dict)


class MatchDimension(BaseModel):
    deterministic: float = Field(ge=0, le=100)
    semantic: float = Field(ge=0, le=100)
    score: float = Field(ge=0, le=100)
    weight: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    match_run_id: str
    total_score: float = Field(ge=0, le=100)
    dimensions: dict[str, MatchDimension]
    threshold_triggered: bool
    scoring_config_version: str
    embedding_mode: str


class CandidateProject(BaseModel):
    candidate_id: str
    candidate_slot_id: str
    title: str
    period: str = "[待补充]"
    introduction: str
    tech_stack: list[str] = Field(default_factory=list)
    solutions: list[str] = Field(default_factory=list)
    results: list[str] = Field(default_factory=list)
    status: Literal["draft", "confirmed", "duplicate", "failed"] = "draft"
    needs_verification: bool = True
    match_score: float = Field(default=0, ge=0, le=100)
    evidence_ids: list[str] = Field(default_factory=list)


class TaskView(BaseModel):
    task_id: str
    thread_id: str
    status: TaskStatus
    blocked_reason: str | None = None
    current_node: str | None = None
    checkpoint_version: int
    job_snapshot_id: str | None = None
    # Immutable resume snapshot created after content decisions enter preview;
    # export/preview clients must bind to this id rather than the mutable job
    # record or the latest draft.
    resume_snapshot_id: str | None = None
    previous_snapshot_ids: list[str] = Field(default_factory=list)
    template_id: str | None = None
    output_mode: str = "resume_edit"
    requested_output_mode: str = "resume_edit"
    final_product: str | None = None
    selected_candidate_id: str | None = None
    generation_branch_id: str | None = None
    feedback_classification: dict[str, Any] | None = None
    embedding_mode: str | None = None
    pending_module: str | None = None
    project_decision: dict[str, Any] | None = None
    project_patch: dict[str, Any] | None = None
    module_assessment: dict[str, Any] = Field(default_factory=dict)
    module_decisions: dict[str, Any] = Field(default_factory=dict)
    module_drafts: dict[str, Any] = Field(default_factory=dict)
    module_patches: dict[str, Any] = Field(default_factory=dict)
    layout_decision: dict[str, Any] | None = None
    compression_decision: dict[str, Any] | None = None
    preview_status: str | None = None
    final_confirmation: bool = False
    candidates: list[CandidateProject] = Field(default_factory=list)
    match: MatchResult | None = None
    messages: list[str] = Field(default_factory=list)


class ProbeResponse(BaseModel):
    role: ModelRole
    provider: Provider
    status: str
    base_url: str
    endpoint: str | None = None
    model_name: str
    model_digest: str | None = None
    dimension: int | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    checked_at: datetime
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    requires_user: bool = False
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobSearchRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    city: str = Field(min_length=1, max_length=100)
    experience: str | None = None
    education: str | None = None
    salary: str | None = None
    company_size: str | None = None
    industry: str | None = None
    posted_after: datetime | None = None
    posted_before: datetime | None = None
    cursor: str | None = None
    limit: int = Field(default=20, ge=1, le=20)


class SectionUpdateRequest(BaseModel):
    section_id: str
    content: str = Field(min_length=1, max_length=100_000)
    confirmed: bool = False


class ProjectRegionRequest(BaseModel):
    section_ids: list[str] = Field(min_length=1, max_length=100)


class RetryStepRequest(BaseModel):
    thread_id: str
    checkpoint_version: int = Field(ge=0)
    node: str | None = None


class ExportPreviewRequest(BaseModel):
    thread_id: str
    checkpoint_version: int = Field(ge=0)
    snapshot_id: str | None = None
    format: Literal["docx", "pdf", "markdown"] = "markdown"
    accept_compression: bool = False
    allow_rewrite: bool = False


class ExportRequest(ExportPreviewRequest):
    snapshot_id: str | None = None
    confirm_unverified: bool = False


class ConsentRequest(BaseModel):
    thread_id: str | None = None
    scope: Literal["external_model", "post_time_risk"]
    granted: bool


class ConfirmationRequest(BaseModel):
    thread_id: str
    checkpoint_version: int = Field(ge=0)
    candidate_id: str | None = None
    fields: list[dict[str, Any]] = Field(default_factory=list)
    confirm: bool = False


class SkillUpdateRequest(BaseModel):
    content: str = Field(max_length=50_000)
    expected_version: int = Field(default=0, ge=0)
