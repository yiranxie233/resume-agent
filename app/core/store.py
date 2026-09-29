"""Thread-safe repository used by the first local vertical slice.

The application can be switched to the PostgreSQL repository when it is available;
the in-memory implementation keeps the API useful during setup and unit tests and
never pretends that its data survives a process restart.
"""
from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.core.schemas import CandidateProject, JobInput, MatchResult, ModelProfile, ResumeDocument, TaskStatus


class VersionConflict(Exception):
    def __init__(self, expected: int, current: int):
        super().__init__(f"checkpoint version conflict: expected {expected}, current {current}")
        self.expected = expected
        self.current = current


@dataclass
class TaskRecord:
    task_id: str
    thread_id: str
    status: TaskStatus = TaskStatus.QUEUED
    blocked_reason: str | None = None
    current_node: str | None = None
    checkpoint_version: int = 0
    job_snapshot_id: str | None = None
    job: JobInput | None = None
    resume: ResumeDocument | None = None
    candidates: list[CandidateProject] = field(default_factory=list)
    match: MatchResult | None = None
    messages: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class InMemoryStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.tasks: dict[str, TaskRecord] = {}
        self.jobs: dict[str, JobInput] = {}
        # Keep insertion order and lightweight timestamps separately from the
        # Pydantic payload.  This mirrors the immutable history contract without
        # coupling the in-memory fallback to the SQLAlchemy models.
        self.job_created_at: dict[str, float] = {}
        self.job_updated_at: dict[str, float] = {}
        self.job_history: list[str] = []
        self.resumes: dict[str, ResumeDocument] = {}
        self.resume_created_at: dict[str, float] = {}
        self.resume_history: list[str] = []
        self.profiles: dict[str, ModelProfile] = {}
        self.settings: dict[str, Any] = {"settings_version": 0}
        self.events: dict[str, list[dict[str, Any]]] = {}

    def seed_profile(self, profile: ModelProfile) -> None:
        with self._lock:
            if profile.default_for_role:
                for current in self.profiles.values():
                    if current.role == profile.role and current.profile_id != profile.profile_id:
                        current.default_for_role = False
            self.profiles[profile.profile_id] = copy.deepcopy(profile)

    def list_profiles(self, role: str | None = None) -> list[ModelProfile]:
        with self._lock:
            values = list(self.profiles.values())
            if role:
                values = [item for item in values if item.role.value == role]
            return copy.deepcopy(values)

    def get_profile(self, profile_id: str) -> ModelProfile | None:
        with self._lock:
            value = self.profiles.get(profile_id)
            return copy.deepcopy(value) if value else None

    def save_profile(self, profile: ModelProfile) -> ModelProfile:
        with self._lock:
            if profile.default_for_role:
                for current in self.profiles.values():
                    if current.role == profile.role and current.profile_id != profile.profile_id:
                        current.default_for_role = False
            self.profiles[profile.profile_id] = copy.deepcopy(profile)
            return copy.deepcopy(profile)

    def save_job(self, job: JobInput) -> str:
        job_id = f"job_{uuid.uuid4().hex}"
        with self._lock:
            self.jobs[job_id] = job.model_copy(deep=True)
            now = time.time()
            self.job_created_at[job_id] = now
            self.job_updated_at[job_id] = now
            self.job_history.append(job_id)
        return job_id

    def hydrate_job(
        self,
        job_id: str,
        job: JobInput,
        *,
        created_at: float | None = None,
        updated_at: float | None = None,
    ) -> None:
        """Restore a durable job without creating a new business identifier."""

        with self._lock:
            self.jobs[job_id] = job.model_copy(deep=True)
            now = time.time()
            self.job_created_at[job_id] = float(created_at or now)
            self.job_updated_at[job_id] = float(updated_at or created_at or now)
            if job_id not in self.job_history:
                self.job_history.append(job_id)

    def get_job(self, job_id: str) -> JobInput | None:
        with self._lock:
            job = self.jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    def update_job(self, job_id: str, job: JobInput) -> JobInput:
        """Replace the mutable current parse while retaining the same job id.

        Job snapshots used by tasks remain immutable; this method only changes
        the current record returned by history/detail endpoints.
        """
        with self._lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            self.jobs[job_id] = job.model_copy(deep=True)
            self.job_updated_at[job_id] = time.time()
            return job.model_copy(deep=True)

    def list_jobs(self) -> list[tuple[str, JobInput, float, float]]:
        with self._lock:
            # Newest first is stable across pagination and matches the UI's
            # history expectation.
            ids = sorted(
                self.job_history,
                key=lambda item: self.job_updated_at.get(item, 0.0),
                reverse=True,
            )
            return [
                (
                    job_id,
                    self.jobs[job_id].model_copy(deep=True),
                    self.job_created_at.get(job_id, 0.0),
                    self.job_updated_at.get(job_id, 0.0),
                )
                for job_id in ids
                if job_id in self.jobs
            ]

    def find_job_by_source_url(self, source_url: str) -> tuple[str, JobInput] | None:
        """Find the current cached job for an immutable external source URL."""

        needle = str(source_url or "").strip()
        if not needle:
            return None
        with self._lock:
            for job_id in reversed(self.job_history):
                job = self.jobs.get(job_id)
                if job is not None and (job.source_url or "").strip() == needle:
                    return job_id, job.model_copy(deep=True)
        return None

    def save_resume(self, resume: ResumeDocument) -> str:
        with self._lock:
            self.resumes[resume.resume_id] = resume.model_copy(deep=True)
            if resume.resume_id not in self.resume_history:
                self.resume_history.append(resume.resume_id)
            self.resume_created_at.setdefault(resume.resume_id, time.time())
        return resume.resume_id

    def hydrate_resume(self, resume: ResumeDocument, *, created_at: float | None = None) -> None:
        """Restore a durable resume record without changing its ID."""

        with self._lock:
            self.resumes[resume.resume_id] = resume.model_copy(deep=True)
            self.resume_created_at[resume.resume_id] = float(created_at or time.time())
            if resume.resume_id not in self.resume_history:
                self.resume_history.append(resume.resume_id)

    def hydrate_task(self, task: TaskRecord) -> None:
        """Restore a task and initialize its in-memory event stream."""

        with self._lock:
            self.tasks[task.task_id] = copy.deepcopy(task)
            self.events.setdefault(task.task_id, [])

    def get_resume(self, resume_id: str) -> ResumeDocument | None:
        with self._lock:
            resume = self.resumes.get(resume_id)
            return resume.model_copy(deep=True) if resume else None

    def list_resumes(self) -> list[tuple[ResumeDocument, float]]:
        with self._lock:
            ids = sorted(
                self.resume_history,
                key=lambda item: self.resume_created_at.get(item, 0.0),
                reverse=True,
            )
            return [
                (self.resumes[item].model_copy(deep=True), self.resume_created_at.get(item, 0.0))
                for item in ids
                if item in self.resumes
            ]

    def update_resume(self, resume_id: str, resume: ResumeDocument) -> ResumeDocument:
        with self._lock:
            if resume_id not in self.resumes:
                raise KeyError(resume_id)
            if resume.resume_id != resume_id:
                raise ValueError("resume_id cannot change")
            self.resumes[resume_id] = resume.model_copy(deep=True)
            return resume.model_copy(deep=True)

    def create_task(self, task: TaskRecord) -> TaskRecord:
        with self._lock:
            if task.task_id in self.tasks:
                raise ValueError("task already exists")
            self.tasks[task.task_id] = copy.deepcopy(task)
            self.events[task.task_id] = []
            return copy.deepcopy(task)

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            task = self.tasks.get(task_id)
            return copy.deepcopy(task) if task else None

    def list_tasks(self) -> list[TaskRecord]:
        """Return task metadata in newest-first order for local history views."""

        with self._lock:
            values = sorted(
                self.tasks.values(),
                key=lambda item: (item.updated_at, item.created_at, item.task_id),
                reverse=True,
            )
            return copy.deepcopy(values)

    def update_task(self, task_id: str, **changes: Any) -> TaskRecord:
        with self._lock:
            task = self.tasks[task_id]
            for key, value in changes.items():
                setattr(task, key, copy.deepcopy(value))
            task.updated_at = time.time()
            return copy.deepcopy(task)

    def append_event(self, task_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            self.events.setdefault(task_id, []).append({"ts": time.time(), **copy.deepcopy(event)})

    def events_for(self, task_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self.events.get(task_id, []))

    def checkpoint(self, task_id: str, expected_version: int, *, state: dict[str, Any], **changes: Any) -> TaskRecord:
        with self._lock:
            task = self.tasks[task_id]
            if task.checkpoint_version != expected_version:
                raise VersionConflict(expected_version, task.checkpoint_version)
            task.checkpoint_version += 1
            task.state = copy.deepcopy(state)
            for key, value in changes.items():
                setattr(task, key, copy.deepcopy(value))
            task.updated_at = time.time()
            self.events.setdefault(task_id, []).append(
                {"ts": time.time(), "type": "checkpoint", "version": task.checkpoint_version, "node": task.current_node}
            )
            return copy.deepcopy(task)

    def update_settings(self, expected_version: int, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            current = int(self.settings.get("settings_version", 0))
            if current != expected_version:
                raise VersionConflict(expected_version, current)
            self.settings.update(copy.deepcopy(values))
            self.settings["settings_version"] = current + 1
            return copy.deepcopy(self.settings)

    def delete_task(self, task_id: str) -> bool:
        with self._lock:
            existed = task_id in self.tasks
            if existed:
                self.tasks.pop(task_id, None)
                self.events.pop(task_id, None)
            return existed

    def set_consent(self, scope: str, granted: bool, *, task_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            consents = self.settings.setdefault("consents", {})
            key = f"{task_id or '*'}:{scope}"
            value = {"scope": scope, "task_id": task_id, "granted": bool(granted), "updated_at": time.time()}
            consents[key] = value
            return copy.deepcopy(value)

    def get_consent(self, scope: str, *, task_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            consents = self.settings.get("consents", {})
            value = consents.get(f"{task_id or '*'}:{scope}") or consents.get(f"*:{scope}")
            return copy.deepcopy(value) if value else None

    def delete_resume(self, resume_id: str) -> bool:
        with self._lock:
            existed = resume_id in self.resumes
            if existed:
                self.resumes.pop(resume_id, None)
                self.resume_created_at.pop(resume_id, None)
                self.resume_history = [item for item in self.resume_history if item != resume_id]
            return existed

    def update_skill(self, content: str, expected_version: int) -> dict[str, Any]:
        with self._lock:
            current = int(self.settings.get("skill_version", 0))
            if current != expected_version:
                raise VersionConflict(expected_version, current)
            next_version = current + 1
            value = {"version": next_version, "content": str(content), "updated_at": time.time()}
            self.settings["skill_version"] = next_version
            self.settings["skill_content"] = str(content)
            versions = self.settings.setdefault("skill_versions", [])
            versions.append(copy.deepcopy(value))
            return copy.deepcopy(value)

    def get_skill(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": int(self.settings.get("skill_version", 0)),
                "content": str(self.settings.get("skill_content", "")),
            }

    def list_skill_versions(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self.settings.get("skill_versions", []))
