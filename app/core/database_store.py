"""Best-effort SQLAlchemy mirror for the local workflow.

The graph still uses a compact in-process cache for low-latency reads.  This
adapter persists immutable job snapshots, tasks, events and checkpoints when
PostgreSQL/SQLite is available.  A database outage never silently changes the
checkpoint semantics: the caller receives ``available=False`` and can surface a
recovery warning rather than treating the write as durable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

from app.core.db_mirror import _compact_base_facts, _compact_state, _without_secrets
from app.core.models import (
    AppSettings,
    CandidateSlot,
    Confirmation,
    Consent,
    ContentPatch,
    ContextSnapshot,
    ExportRun,
    FeedbackClassification as FeedbackClassificationRow,
    FeedbackRecord,
    GeneratedVersion,
    JobEvidence,
    JobRecord,
    JobSnapshot,
    LangGraphCheckpoint,
    MatchRun,
    ModelInvocation,
    ModelProfile as ModelProfileRow,
    ModelProfileVersion,
    NodeAttempt,
    NodeOperation,
    ResumeFile,
    ResumeSection,
    ResumeSnapshot,
    ScoringConfig,
    Task,
    TaskEvent,
    TemplateRecord,
    WorkerLease,
)
from app.core.persistence import (
    LeaseBusy,
    NotFoundError,
    OptimisticConflict,
    acquire_worker_lease,
    append_event,
    release_worker_lease,
    renew_worker_lease,
)
from app.core.utils import redact_sensitive, sha256_json, sha256_bytes, sha256_text, stable_id


def _safe_source_url(value: str | None) -> str | None:
    """Return a URL accepted by ``JobInput`` when hydrating a DB row.

    Locally entered jobs intentionally use a ``local://`` sentinel when they
    are stored in the non-null SQL source column.  That sentinel is an
    internal persistence detail and must not be fed back through the API URL
    validator (which accepts only HTTP(S) and rejects ``file://``).
    """

    if not value:
        return None
    parsed = urlparse(value)
    return value if parsed.scheme.lower() in {"http", "https"} else None


def _json_value(value: Any) -> Any:
    """Convert Pydantic/domain values into a JSON-column-safe value."""

    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def _sanitized_json(value: Any) -> Any:
    """Return JSON-safe business data with credential-shaped values removed."""

    return _json_value(_without_secrets(value))


def _validated_profile_url(value: Any) -> str:
    """Reject credentials hidden in a persisted model endpoint URL."""

    result = str(value or "")
    parsed = urlparse(result)
    if parsed.username or parsed.password:
        raise ValueError("model base_url must not contain embedded credentials")
    sensitive_query_keys = {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "client_secret",
        "github_token",
        "key",
        "password",
        "secret",
        "token",
        "x_api_key",
    }
    query_names = [name.lower().replace("-", "_") for name, _ in parse_qsl(parsed.query, keep_blank_values=True)]
    fragment_names = [name.lower().replace("-", "_") for name, _ in parse_qsl(parsed.fragment, keep_blank_values=True)]
    if any(name in sensitive_query_keys for name in [*query_names, *fragment_names]):
        raise ValueError("model base_url must not contain credential query parameters")
    return result


def _profile_probe_result(profile: Any) -> dict[str, Any]:
    checked_at = getattr(profile, "last_probe_at", None)
    if hasattr(checked_at, "isoformat"):
        checked_at = checked_at.isoformat()
    return _sanitized_json(
        {
            "status": str(getattr(profile, "status", "unknown")),
            "checked_at": checked_at,
            "error_code": getattr(profile, "error_code", None),
        }
    )


class DatabaseMirror:
    def __init__(self, engine: Any, factory: Any) -> None:
        self.engine = engine
        self.factory = factory
        self.available = True

    @staticmethod
    def _apply_profile_version_values(row: ModelProfileVersion, values: dict[str, Any]) -> None:
        for name in (
            "role",
            "provider",
            "base_url",
            "model_name",
            "model_digest",
            "dimension",
            "context_window_tokens",
            "tokenizer_id",
            "tokenizer_version",
            "tokenizer_source",
            "estimator_version",
            "max_input_tokens",
            "chunk_policy_version",
            "normalize_embeddings",
            "distance_metric",
            "batch_size",
            "capabilities",
            "generation_params",
            "embedding_algorithm_version",
            "budget_policy_version",
            "probe_result",
        ):
            setattr(row, name, values.get(name))

    @staticmethod
    def _profile_values_from_snapshot(snapshot: dict[str, Any], role: str) -> dict[str, Any]:
        snapshot = _sanitized_json(snapshot)
        snapshot_role = str(snapshot.get("role") or role)
        if snapshot_role != role:
            raise ValueError(f"{role} profile snapshot has role {snapshot_role!r}")
        provider = str(snapshot.get("provider") or "")
        if provider not in {"ollama", "openai_compatible"}:
            raise ValueError("unsupported model provider in task snapshot")
        base_url = _validated_profile_url(snapshot.get("base_url"))
        model_name = str(snapshot.get("model_name") or "").strip()
        if not base_url or not model_name:
            raise ValueError("task model snapshot is missing base_url or model_name")
        checked_at = snapshot.get("last_probe_at")
        if hasattr(checked_at, "isoformat"):
            checked_at = checked_at.isoformat()
        return {
            "role": role,
            "provider": provider,
            "base_url": base_url,
            "model_name": model_name,
            "model_digest": snapshot.get("model_digest"),
            "dimension": snapshot.get("dimension"),
            "context_window_tokens": snapshot.get("context_window_tokens"),
            "tokenizer_id": snapshot.get("tokenizer_id"),
            "tokenizer_version": snapshot.get("tokenizer_version"),
            "tokenizer_source": snapshot.get("tokenizer_source"),
            "estimator_version": snapshot.get("estimator_version"),
            "max_input_tokens": snapshot.get("max_input_tokens"),
            "chunk_policy_version": snapshot.get("chunk_policy_version"),
            "normalize_embeddings": snapshot.get("normalize_embeddings"),
            "distance_metric": snapshot.get("distance_metric"),
            "batch_size": snapshot.get("batch_size"),
            "capabilities": _sanitized_json(snapshot.get("capabilities") or {}),
            "generation_params": _sanitized_json(snapshot.get("generation_params") or {}),
            "embedding_algorithm_version": (
                snapshot.get("embedding_algorithm_version") or ("embedding-v1" if role == "embedding" else None)
            ),
            "budget_policy_version": snapshot.get("budget_policy_version") or ("context-budget-v1" if role == "chat" else None),
            "probe_result": _sanitized_json(
                {
                    "status": snapshot.get("status") or "unknown",
                    "checked_at": checked_at,
                    "error_code": snapshot.get("error_code"),
                }
            ),
        }

    @classmethod
    def _ensure_task_profile_version(
        cls,
        session: Any,
        state: dict[str, Any],
        role: str,
    ) -> ModelProfileVersion | None:
        prefix = "chat" if role == "chat" else "embedding"
        snapshot = state.get(f"{prefix}_profile_snapshot")
        snapshot = snapshot if isinstance(snapshot, dict) else None
        profile_id = state.get(f"{prefix}_profile_id") or (snapshot or {}).get("profile_id")
        version_value = state.get(f"{prefix}_profile_version")
        if version_value is None:
            version_value = (snapshot or {}).get("config_version")
        if not profile_id and version_value is None and snapshot is None:
            return None
        if not profile_id or version_value is None:
            raise ValueError(f"{role} profile binding requires profile_id and version")
        profile_id = str(profile_id)
        try:
            version_no = int(str(version_value).removeprefix("v"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {role} profile version") from exc
        if version_no < 1:
            raise ValueError(f"invalid {role} profile version")
        if snapshot is not None:
            snapshot_id = str(snapshot.get("profile_id") or profile_id)
            if snapshot_id != profile_id:
                raise ValueError(f"{role} profile snapshot id does not match task binding")
            if snapshot.get("config_version") is not None:
                try:
                    snapshot_version = int(str(snapshot.get("config_version")).removeprefix("v"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid {role} profile snapshot version") from exc
                if snapshot_version != version_no:
                    raise ValueError(f"task {role} profile version does not match its frozen snapshot")
        existing = session.query(ModelProfileVersion).filter_by(
            profile_id=profile_id,
            version_no=version_no,
        ).one_or_none()
        if existing is not None:
            if existing.role != role:
                raise ValueError(f"task {role} profile points to a {existing.role} version")
            if snapshot is not None:
                values = cls._profile_values_from_snapshot(snapshot, role)
                for name in ("provider", "base_url", "model_name"):
                    if getattr(existing, name) != values[name]:
                        raise ValueError(f"task {role} profile snapshot differs from immutable version")
            return existing
        if snapshot is None:
            raise ValueError(f"missing immutable {role} profile version")
        values = cls._profile_values_from_snapshot(snapshot, role)
        profile = session.get(ModelProfileRow, profile_id)
        if profile is None:
            profile = ModelProfileRow(
                id=profile_id,
                role=role,
                provider=values["provider"],
                base_url=values["base_url"],
                model_name=values["model_name"],
                is_default=bool(snapshot.get("default_for_role", False)),
                credential_required=bool(snapshot.get("credential_required", values["provider"] == "openai_compatible")),
                configured_in_session=False,
                status=str(snapshot.get("status") or "unknown"),
                dimension=values["dimension"],
                context_window_tokens=values["context_window_tokens"],
                tokenizer_id=values["tokenizer_id"],
                tokenizer_version=values["tokenizer_version"],
                tokenizer_source=values["tokenizer_source"],
                max_input_tokens=values["max_input_tokens"],
                capabilities=values["capabilities"],
                generation_params=values["generation_params"],
                probe_result=values["probe_result"],
                version=version_no,
            )
            session.add(profile)
            session.flush()
        elif (
            profile.role != role
            or profile.provider != values["provider"]
            or profile.base_url != values["base_url"]
            or profile.model_name != values["model_name"]
        ):
            raise ValueError(f"task {role} snapshot conflicts with model profile metadata")
        version = ModelProfileVersion(profile_id=profile_id, version_no=version_no)
        cls._apply_profile_version_values(version, values)
        session.add(version)
        session.flush()
        return version

    @staticmethod
    def _persist_state_payload(
        session: Any,
        *,
        task_id: str,
        module_type: str,
        value: Any,
        checkpoint_version: int,
    ) -> GeneratedVersion:
        content = {"value": _sanitized_json(value)}
        content_hash = sha256_json(content)
        version_id = stable_id("state", task_id, module_type, content_hash)
        existing = session.get(GeneratedVersion, version_id)
        if existing is not None:
            return existing
        latest = (
            session.query(GeneratedVersion)
            .filter_by(task_id=task_id, module_type=module_type)
            .order_by(GeneratedVersion.version_no.desc())
            .first()
        )
        row = GeneratedVersion(
            id=version_id,
            task_id=task_id,
            module_type=module_type,
            version_no=(latest.version_no + 1) if latest else 1,
            content=content,
            content_hash=content_hash,
            source_refs={"checkpoint_version": int(checkpoint_version)},
            state="business_state",
            is_user_confirmed=False,
        )
        session.add(row)
        session.flush()
        return row

    @classmethod
    def _persist_business_state(cls, session: Any, task: Any) -> None:
        state = getattr(task, "state", {}) or {}
        checkpoint_version = int(getattr(task, "checkpoint_version", 0))
        for key in (
            "base_facts",
            "project_patch",
            "preview_feedback",
            "module_decisions",
            "layout_decision",
            "compression_decision",
            "output_mode",
            "template_id",
            "previous_snapshot_ids",
            "final_confirmation",
        ):
            if key in state:
                value = _compact_base_facts(state.get(key)) if key == "base_facts" else state.get(key)
                cls._persist_state_payload(
                    session,
                    task_id=str(task.task_id),
                    module_type=f"task_state:{key}",
                    value=value,
                    checkpoint_version=checkpoint_version,
                )
        for state_key, singular in (("module_drafts", "module_draft"), ("module_patches", "module_patch")):
            values = state.get(state_key)
            if not isinstance(values, dict):
                continue
            for module, value in values.items():
                cls._persist_state_payload(
                    session,
                    task_id=str(task.task_id),
                    module_type=f"task_state:{singular}:{str(module)[:24]}",
                    value=value,
                    checkpoint_version=checkpoint_version,
                )
        project_patch = state.get("project_patch")
        if isinstance(project_patch, dict) and project_patch.get("operation") in {"add", "replace"}:
            cls._persist_resume_content_patch(
                session,
                task,
                project_patch,
                module="projects",
                checkpoint_version=checkpoint_version,
            )
        module_patches = state.get("module_patches")
        if isinstance(module_patches, dict):
            for module, patch in module_patches.items():
                if isinstance(patch, dict) and patch.get("status") == "confirmed":
                    cls._persist_resume_content_patch(
                        session,
                        task,
                        patch,
                        module=str(module),
                        checkpoint_version=checkpoint_version,
                    )

    @classmethod
    def _persist_resume_content_patch(
        cls,
        session: Any,
        task: Any,
        patch: dict[str, Any],
        *,
        module: str,
        checkpoint_version: int,
    ) -> ContentPatch:
        """Persist one resume-module patch exactly once with hash validation."""

        patch_key = str(patch.get("patch_id") or sha256_json(_sanitized_json(patch)))
        patch_id = stable_id("content-patch", str(task.task_id), patch_key)
        existing = session.get(ContentPatch, patch_id)
        if existing is not None:
            return existing
        target_item_id = str(
            patch.get("target_project_id")
            or patch.get("target_item_id")
            or f"{module}:new"
        )
        old_value = ""
        resume = getattr(task, "resume", None)
        if resume is not None:
            source_section = next(
                (section for section in resume.sections if section.section_id == target_item_id),
                None,
            )
            if source_section is not None:
                old_value = str(source_section.content)
                actual_hash = str(source_section.value_hash or sha256_text(old_value))
            else:
                actual_hash = sha256_text("")
        else:
            actual_hash = sha256_text("")
        expected_hash = str(patch.get("old_value_hash") or "")
        source = cls._persist_state_payload(
            session,
            task_id=str(task.task_id),
            module_type=f"resume_source:{module}:{target_item_id[:24]}",
            value={"target_item_id": target_item_id, "value": old_value},
            checkpoint_version=checkpoint_version,
        )
        new_value = _sanitized_json(patch.get("new_value"))
        applied = None
        status = "conflicted"
        if expected_hash and expected_hash == actual_hash:
            applied = cls._persist_state_payload(
                session,
                task_id=str(task.task_id),
                module_type=f"resume_applied:{module}:{target_item_id[:24]}",
                value={"target_item_id": target_item_id, "value": new_value},
                checkpoint_version=checkpoint_version,
            )
            applied.is_user_confirmed = True
            applied.state = "confirmed"
            status = "applied"
        row = ContentPatch(
            id=patch_id,
            task_id=str(task.task_id),
            target_version_id=source.id,
            target_module=module,
            target_item_id=target_item_id,
            old_value_hash=expected_hash,
            new_value={"value": new_value},
            reason=str(redact_sensitive(str(patch.get("reason") or ""))),
            evidence_ids=_sanitized_json(patch.get("evidence_ids") or []),
            status=status,
            applied_version_id=applied.id if applied is not None else None,
            diff=(
                {"target_item_id": target_item_id, "old": old_value, "new": new_value}
                if status == "applied"
                else {
                    "target_item_id": target_item_id,
                    "expected_old_value_hash": expected_hash,
                    "actual_value_hash": actual_hash,
                }
            ),
        )
        session.add(row)
        session.flush()
        return row

    @staticmethod
    def _persist_confirmation_rows(session: Any, task_id: str, confirmations: Any) -> None:
        if not isinstance(confirmations, list):
            return
        for item in confirmations:
            if not isinstance(item, dict):
                continue
            field_id = str(item.get("field_id") or item.get("target_id") or "")
            value_hash = str(item.get("value_hash") or "")
            if not field_id or not value_hash:
                continue
            status = "confirmed" if bool(item.get("confirmed", True)) else "rejected"
            row_id = stable_id("confirmation", task_id, field_id, value_hash, status)
            if session.get(Confirmation, row_id) is not None:
                continue
            session.add(
                Confirmation(
                    id=row_id,
                    task_id=task_id,
                    field_id=field_id,
                    value_hash=value_hash,
                    value=_sanitized_json(
                        {
                            "module": item.get("module"),
                            "value": item.get("value"),
                        }
                    ),
                    source_version_id=item.get("source_version_id"),
                    patch_version_id=item.get("patch_version_id"),
                    status=status,
                )
            )

    @staticmethod
    def _ensure_scoring_config(session: Any, state: dict[str, Any]) -> ScoringConfig:
        version = str(state.get("scoring_config_version") or "scoring-v1")
        dimensions = _sanitized_json(
            state.get("dimension_weights")
            or {"skills": 0.25, "responsibilities": 0.25, "projects": 0.25, "keywords": 0.25}
        )
        components = _sanitized_json(
            state.get("component_weights") or {"deterministic": 0.5, "semantic": 0.5}
        )
        existing = session.query(ScoringConfig).filter_by(version=version).one_or_none()
        if existing is not None:
            if dict(existing.dimension_weights or {}) != dimensions or (
                abs(float(existing.deterministic_weight) - float(components.get("deterministic", 0.5))) > 1e-9
                or abs(float(existing.semantic_weight) - float(components.get("semantic", 0.5))) > 1e-9
            ):
                raise ValueError("scoring config version is already bound to different weights")
            return existing
        row = ScoringConfig(
            id=stable_id("scoring-config", version),
            version=version,
            dimension_weights=dimensions,
            deterministic_weight=float(components.get("deterministic", 0.5)),
            semantic_weight=float(components.get("semantic", 0.5)),
            rubric_version="match-rubric-v1",
            rounding_digits=1,
            threshold=80.0,
            algorithm_version="scoring-v1",
            config_json={"component_weights": components},
        )
        session.add(row)
        session.flush()
        return row

    def persist_job(self, job_id: str, job: Any) -> str | None:
        try:
            with self.factory() as session:
                content_hash = sha256_json(job.model_dump(mode="json"))
                snapshot_id = stable_id("snapshot", job_id, content_hash)
                existing_snapshot = session.get(JobSnapshot, snapshot_id)
                record = session.get(JobRecord, job_id)
                if record is None:
                    record = JobRecord(id=job_id)
                    session.add(record)
                record.source_url = job.source_url or "local://job"
                record.job_title = job.title
                record.company_name = job.company
                record.city = job.city
                record.salary = job.salary
                record.responsibilities = "\n".join(job.responsibilities)
                record.requirements = "\n".join(job.requirements)
                record.skills = job.skills
                record.posted_at = job.posted_at
                record.posted_time_status = "known" if job.posted_at else "unknown"
                record.extra = {**dict(record.extra or {}), "hr_activity": job.hr_activity}
                record.parser_version = "job-parser-v1"
                if existing_snapshot is not None:
                    session.commit()
                    return snapshot_id
                snapshot = JobSnapshot(
                    id=snapshot_id,
                    job_record_id=job_id,
                    source_url=job.source_url or "local://job",
                    snapshot_hash=content_hash,
                    job_title=job.title,
                    company_name=job.company,
                    city=job.city,
                    salary=job.salary,
                    responsibilities="\n".join(job.responsibilities),
                    requirements="\n".join(job.requirements),
                    skills=job.skills,
                    posted_at=job.posted_at,
                    posted_time_status="known" if job.posted_at else "unknown",
                    evidence_ids=[],
                    parser_version="job-parser-v1",
                )
                session.add(snapshot)
                from app.services.matching import build_job_evidence

                evidence_rows = build_job_evidence(job)
                snapshot.evidence_ids = [item.evidence_id for item in evidence_rows]
                session.flush()
                session.add_all(
                    [
                        JobEvidence(
                            snapshot_id=snapshot_id,
                            evidence_id=item.evidence_id,
                            field_name=(item.location.split("[", 1)[0] if item.location else None),
                            text=item.text,
                            text_hash=item.text_hash,
                            source_locator={"location": item.location},
                        )
                        for item in evidence_rows
                    ]
                )
                session.commit()
                return snapshot_id
        except Exception:
            self.available = False
            return None

    def persist_template(self, metadata: dict[str, Any]) -> str | None:
        """Upsert non-secret template metadata after a verified cache write."""

        template_id = str(metadata.get("template_id") or "").strip()
        if not template_id:
            return None
        try:
            with self.factory() as session:
                row = session.get(TemplateRecord, template_id)
                if row is None:
                    row = TemplateRecord(
                        id=template_id,
                        source_url=str(metadata.get("source_url") or "internal://template"),
                        file_type=str(metadata.get("file_type") or "markdown"),
                    )
                    session.add(row)
                row.repository = str(metadata.get("repository") or "") or None
                row.source_url = str(metadata.get("source_url") or row.source_url or "internal://template")
                row.file_path = str(metadata.get("file_path") or "") or None
                row.language = str(metadata.get("language") or "zh")
                row.file_type = str(metadata.get("file_type") or row.file_type or "markdown")
                row.license_name = str(metadata.get("license") or metadata.get("license_name") or "") or None
                license_allowed = metadata.get("license_allowed")
                row.license_allows_modification = bool(license_allowed) if license_allowed is not None else None
                row.stars = int(metadata["stars"]) if metadata.get("stars") is not None else None
                row.remote_version = str(metadata.get("remote_version") or "") or None
                row.local_version = str(metadata.get("local_version") or metadata.get("version") or "") or None
                row.preview_path = str(metadata.get("cached_path") or metadata.get("path") or "") or None
                row.update_status = str(metadata.get("update_status") or "cached")
                row.metadata_json = _sanitized_json(metadata)
                session.commit()
                return template_id
        except Exception:
            self.available = False
            return None

    def load_settings(self) -> dict[str, Any] | None:
        """Read persisted non-secret settings for process restart."""

        try:
            with self.factory() as session:
                row = session.get(AppSettings, 1)
                if row is None:
                    return None
                return {**dict(row.values or {}), "settings_version": int(row.settings_version)}
        except Exception:
            self.available = False
            return None

    def persist_consent(
        self,
        scope: str,
        granted: bool,
        *,
        policy_version: str,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Persist a non-secret consent decision and its revocation history.

        API keys and tokens never pass through this method.  A global consent is
        represented by ``task_id IS NULL`` and is updated in place so restarting
        the local application does not prompt again until the user revokes it.
        """

        try:
            with self.factory() as session:
                query = session.query(Consent).filter(Consent.scope == scope)
                query = query.filter(Consent.task_id == task_id) if task_id else query.filter(Consent.task_id.is_(None))
                row = query.one_or_none()
                if row is None:
                    row = Consent(scope=scope, task_id=task_id, policy_version=policy_version)
                    session.add(row)
                now = datetime.now(timezone.utc)
                row.granted = bool(granted)
                row.policy_version = policy_version
                row.granted_at = now if granted else row.granted_at
                row.revoked_at = None if granted else now
                row.metadata_json = _sanitized_json(metadata or {})
                session.commit()
                return {
                    "scope": row.scope,
                    "task_id": row.task_id,
                    "granted": bool(row.granted),
                    "policy_version": row.policy_version,
                    "granted_at": row.granted_at.isoformat() if row.granted_at else None,
                    "revoked_at": row.revoked_at.isoformat() if row.revoked_at else None,
                }
        except Exception:
            self.available = False
            return None

    def load_consents(self) -> dict[str, dict[str, Any]]:
        """Load persisted consent decisions without loading any credentials."""

        try:
            with self.factory() as session:
                rows = session.query(Consent).all()
                return {
                    f"{row.scope}:{row.task_id or 'global'}": {
                        "scope": row.scope,
                        "task_id": row.task_id,
                        "granted": bool(row.granted),
                        "policy_version": row.policy_version,
                        "granted_at": row.granted_at.isoformat() if row.granted_at else None,
                        "revoked_at": row.revoked_at.isoformat() if row.revoked_at else None,
                    }
                    for row in rows
                }
        except Exception:
            self.available = False
            return {}

    def delete_task(self, task_id: str) -> bool:
        """Delete a task and all task-owned durable records.

        Explicit deletes keep source resumes and job history intact because both
        can be reused by other tasks.  Task drafts, checkpoints, confirmations,
        feedback, exports and task-scoped consent are removed.
        """

        try:
            with self.factory() as session:
                row = session.get(Task, task_id)
                if row is None:
                    return True

                feedback_rows = session.query(FeedbackRecord).filter_by(task_id=task_id).all()
                classification_ids = {
                    item.classification_id for item in feedback_rows if item.classification_id
                }

                # Delete in dependency order so this remains correct even when
                # SQLite foreign-key cascades are disabled in a development DB.
                for model in (
                    ExportRun,
                    ContentPatch,
                    CandidateSlot,
                    Confirmation,
                    FeedbackRecord,
                    ResumeSnapshot,
                    ModelInvocation,
                    ContextSnapshot,
                    NodeAttempt,
                    NodeOperation,
                    WorkerLease,
                    LangGraphCheckpoint,
                    TaskEvent,
                    Consent,
                    MatchRun,
                ):
                    session.query(model).filter(model.task_id == task_id).delete(
                        synchronize_session=False
                    )
                session.query(GeneratedVersion).filter(
                    GeneratedVersion.task_id == task_id
                ).delete(synchronize_session=False)
                if classification_ids:
                    session.query(FeedbackClassificationRow).filter(
                        FeedbackClassificationRow.id.in_(classification_ids)
                    ).delete(synchronize_session=False)
                session.query(ResumeFile).filter(ResumeFile.task_id == task_id).update(
                    {ResumeFile.task_id: None}, synchronize_session=False
                )
                session.delete(row)
                session.commit()
                return True
        except Exception:
            self.available = False
            return False

    def persist_settings(self, expected_version: int, values: dict[str, Any]) -> dict[str, Any]:
        """Atomically persist settings with the same CAS semantics as memory."""

        try:
            with self.factory() as session:
                row = session.get(AppSettings, 1)
                if row is None:
                    row = AppSettings(id=1, settings_version=expected_version, values={})
                    session.add(row)
                    session.flush()
                if int(row.settings_version) != int(expected_version):
                    raise OptimisticConflict("settings version is stale", current_version=int(row.settings_version))
                row.values = _sanitized_json(values)
                row.settings_version = int(expected_version) + 1
                session.commit()
                return {**dict(row.values or {}), "settings_version": int(row.settings_version)}
        except OptimisticConflict:
            raise
        except Exception:
            self.available = False
            raise

    def persist_model_profile(self, profile: Any) -> bool:
        """Persist non-secret profile metadata and an immutable config version."""

        try:
            with self.factory() as session:
                profile_id = str(profile.profile_id)
                role = str(profile.role.value if hasattr(profile.role, "value") else profile.role)
                provider = str(profile.provider.value if hasattr(profile.provider, "value") else profile.provider)
                snapshot = _sanitized_json(
                    profile.model_dump(mode="json") if hasattr(profile, "model_dump") else vars(profile)
                )
                values = self._profile_values_from_snapshot(snapshot, role)
                row = session.get(ModelProfileRow, profile_id)
                if row is None:
                    row = ModelProfileRow(id=profile_id)
                    session.add(row)
                if bool(getattr(profile, "default_for_role", False)):
                    session.query(ModelProfileRow).filter(
                        ModelProfileRow.role == role,
                        ModelProfileRow.id != profile_id,
                    ).update({ModelProfileRow.is_default: False}, synchronize_session=False)
                row.role = role
                row.provider = provider
                row.base_url = values["base_url"]
                row.model_name = values["model_name"]
                row.is_default = bool(getattr(profile, "default_for_role", False))
                row.credential_required = bool(profile.credential_required)
                row.configured_in_session = False
                row.status = str(profile.status)
                row.model_digest = values["model_digest"]
                row.dimension = values["dimension"]
                row.context_window_tokens = values["context_window_tokens"]
                row.tokenizer_id = values["tokenizer_id"]
                row.tokenizer_version = values["tokenizer_version"]
                row.tokenizer_source = values["tokenizer_source"]
                row.estimator_version = values["estimator_version"]
                row.embedding_algorithm_version = values["embedding_algorithm_version"]
                row.max_input_tokens = values["max_input_tokens"]
                row.chunk_policy_version = values["chunk_policy_version"]
                row.normalize_embeddings = values["normalize_embeddings"]
                row.distance_metric = values["distance_metric"]
                row.batch_size = values["batch_size"]
                row.capabilities = values["capabilities"]
                row.generation_params = values["generation_params"]
                row.probe_result = _profile_probe_result(profile)
                row.last_probe_at = getattr(profile, "last_probe_at", None)
                row.version = int(profile.config_version)
                session.flush()
                existing_version = session.query(ModelProfileVersion).filter_by(
                    profile_id=profile_id, version_no=int(profile.config_version)
                ).one_or_none()
                if existing_version is None:
                    existing_version = ModelProfileVersion(
                        profile_id=profile_id,
                        version_no=int(profile.config_version),
                    )
                    self._apply_profile_version_values(existing_version, values)
                    session.add(existing_version)
                else:
                    # Probe metadata can be refreshed until a task freezes this
                    # version.  Once referenced, the immutable snapshot remains
                    # untouched even if the mutable settings card is re-probed.
                    referenced = session.query(Task.id).filter(
                        (Task.chat_model_profile_version_id == existing_version.id)
                        | (Task.embedding_model_profile_version_id == existing_version.id)
                    ).first()
                    if referenced is None:
                        self._apply_profile_version_values(existing_version, values)
                session.commit()
                return True
        except Exception:
            self.available = False
            return False

    def load_model_profiles(self) -> list[Any]:
        """Hydrate persisted profiles without ever loading credentials."""

        from app.core.schemas import ModelProfile, ModelRole, Provider

        try:
            with self.factory() as session:
                rows = session.query(ModelProfileRow).order_by(ModelProfileRow.created_at.asc()).all()
                return [
                    ModelProfile(
                        profile_id=row.id,
                        role=ModelRole(row.role),
                        provider=Provider(row.provider),
                        base_url=row.base_url,
                        model_name=row.model_name,
                        status=row.status,
                        default_for_role=bool(row.is_default),
                        credential_required=bool(row.credential_required),
                        context_window_tokens=row.context_window_tokens,
                        tokenizer_id=row.tokenizer_id,
                        tokenizer_version=row.tokenizer_version,
                        tokenizer_source=row.tokenizer_source,
                        dimension=row.dimension,
                        max_input_tokens=row.max_input_tokens,
                        capabilities=dict(row.capabilities or {}),
                        generation_params=dict(row.generation_params or {}),
                        config_version=int(row.version or 1),
                        last_probe_at=row.last_probe_at,
                        error_code=(row.probe_result or {}).get("error_code") if isinstance(row.probe_result, dict) else None,
                    )
                    for row in rows
                ]
        except Exception:
            self.available = False
            return []

    def persist_task(self, task: Any) -> bool:
        if task.job_snapshot_id is None:
            return False
        try:
            with self.factory() as session:
                state = getattr(task, "state", {}) or {}
                resume_obj = getattr(task, "resume", None)
                requested_resume_id = getattr(resume_obj, "resume_id", None) if resume_obj else None
                # A task may be created from an in-memory upload before the
                # upload transaction reaches the mirror.  Do not write a
                # dangling FK; the resume can be linked on a later retry.
                resume_row = session.get(ResumeFile, requested_resume_id) if requested_resume_id else None
                resume_id = resume_row.id if resume_row is not None else None
                chat_version = self._ensure_task_profile_version(session, state, "chat")
                embedding_version = self._ensure_task_profile_version(session, state, "embedding")
                row = session.get(Task, task.task_id)
                status = task.status.value if hasattr(task.status, "value") else str(task.status)
                if row is None:
                    row = Task(
                        id=task.task_id,
                        thread_id=task.thread_id,
                        status=status,
                        current_node=task.current_node,
                        job_snapshot_id=task.job_snapshot_id,
                        resume_file_id=resume_id,
                        chat_model_profile_version_id=chat_version.id if chat_version else None,
                        embedding_model_profile_version_id=embedding_version.id if embedding_version else None,
                        blocked_reason=getattr(task, "blocked_reason", None),
                        generation_branch_id=state.get("generation_branch_id"),
                        timeout_seconds=int(state.get("timeout_seconds", 300)),
                        embedding_mode=str(state.get("embedding_mode", "tfidf_fallback")),
                        latest_checkpoint_version=task.checkpoint_version,
                        extra=_compact_state(state),
                    )
                    session.add(row)
                else:
                    if row.thread_id != task.thread_id:
                        return False
                    incoming_version = int(task.checkpoint_version)
                    current_version = int(row.latest_checkpoint_version or 0)
                    incoming_compact = _compact_state(state)
                    if incoming_version < current_version:
                        return False
                    if incoming_version > current_version + 1:
                        return False
                    if incoming_version == current_version and row.extra != incoming_compact:
                        return False
                    row.status = status
                    row.current_node = task.current_node
                    row.job_snapshot_id = task.job_snapshot_id
                    if resume_id is not None:
                        row.resume_file_id = resume_id
                    if chat_version is not None:
                        row.chat_model_profile_version_id = chat_version.id
                    if embedding_version is not None:
                        row.embedding_model_profile_version_id = embedding_version.id
                    row.blocked_reason = getattr(task, "blocked_reason", None)
                    row.generation_branch_id = state.get("generation_branch_id", row.generation_branch_id)
                    row.latest_checkpoint_version = incoming_version
                    row.extra = incoming_compact
                session.flush()
                # Keep the optional reverse link only for resumes that are not
                # already owned by another task; the same source resume may be
                # reused by multiple independent tasks.
                if resume_row is not None and resume_row.task_id is None:
                    resume_row.task_id = task.task_id
                self._persist_business_state(session, task)
                self._persist_confirmation_rows(session, str(task.task_id), state.get("confirmations"))
                session.commit()
                return True
        except Exception:
            self.available = False
            return False

    def persist_resume(self, resume: Any, *, path: str, data: bytes) -> str | None:
        """Persist an uploaded source file and its structured sections."""

        try:
            with self.factory() as session:
                file_id = str(resume.resume_id)
                existing = session.get(ResumeFile, file_id)
                if existing is not None:
                    return file_id
                row = ResumeFile(
                    id=file_id,
                    path=path,
                    original_name=resume.filename,
                    media_type={"docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "pdf": "application/pdf"}.get(resume.file_type, "text/plain"),
                    format=resume.file_type,
                    size_bytes=resume.file_size,
                    sha256=sha256_bytes(data),
                    page_count=resume.page_count,
                    file_role="source",
                    validation_status="ready",
                    manifest={"resume_id": resume.resume_id},
                )
                session.add(row)
                for index, section in enumerate(resume.sections):
                    session.add(
                        ResumeSection(
                            resume_file_id=file_id,
                            section_id=section.section_id,
                            module_type=section.module,
                            item_index=index,
                            text=section.content,
                            text_hash=section.value_hash or sha256_json(section.content),
                            source_locator={"title": section.title},
                            user_corrected=section.confirmed,
                        )
                    )
                session.commit()
                return file_id
        except Exception:
            self.available = False
            return None

    def persist_resume_sections(self, resume: Any) -> bool:
        """Reconcile the editable structured projection of a source resume.

        The uploaded bytes remain immutable.  User corrections and manual
        project-region labels live in ``resume_sections`` and must survive a
        process restart, so this method updates only that projection.
        """

        try:
            with self.factory() as session:
                row = session.get(ResumeFile, str(resume.resume_id))
                if row is None:
                    return False
                existing = {
                    item.section_id: item
                    for item in session.query(ResumeSection).filter_by(resume_file_id=row.id).all()
                }
                seen: set[str] = set()
                for index, section in enumerate(getattr(resume, "sections", []) or []):
                    section_id = str(section.section_id)
                    seen.add(section_id)
                    target = existing.get(section_id)
                    if target is None:
                        target = ResumeSection(resume_file_id=row.id, section_id=section_id)
                        session.add(target)
                    target.module_type = str(section.module)
                    target.item_index = index
                    target.text = str(section.content)
                    target.text_hash = str(section.value_hash or sha256_text(target.text))
                    target.source_locator = {"title": section.title}
                    target.user_corrected = bool(section.confirmed)
                for section_id, target in existing.items():
                    if section_id not in seen:
                        session.delete(target)
                session.commit()
                return True
        except Exception:
            self.available = False
            return False

    def persist_resume_snapshot(self, task: Any) -> str | None:
        """Create an immutable snapshot that references all confirmed versions."""

        snapshot_id = str(getattr(task, "state", {}).get("resume_snapshot_id") or "")
        if not snapshot_id:
            return None
        try:
            with self.factory() as session:
                task_state = getattr(task, "state", {}) or {}
                task_status = getattr(task, "status", None)
                task_status_value = task_status.value if hasattr(task_status, "value") else str(task_status or "")
                is_final = bool(task_state.get("final_confirmation")) or task_status_value == "completed"
                existing_snapshot = session.get(ResumeSnapshot, snapshot_id)
                if existing_snapshot is not None:
                    if is_final and existing_snapshot.state != "confirmed":
                        refs = dict(existing_snapshot.module_version_ids or {})
                        materialized = session.get(GeneratedVersion, refs.get("materialized")) if refs.get("materialized") else None
                        if materialized is not None:
                            materialized.is_user_confirmed = True
                            materialized.state = "confirmed"
                        existing_snapshot.state = "confirmed"
                        session.commit()
                    return snapshot_id
                if existing_snapshot is None:
                    selected = task_state.get("selected_candidate")
                    selected_value = _json_value(selected)
                    selected_id = selected_value.get("candidate_id") if isinstance(selected_value, dict) else selected_value
                    candidate_ids = [item.candidate_id for item in (getattr(task, "candidates", None) or [])]
                    selected_candidate = next(
                        (
                            item for item in (getattr(task, "candidates", None) or [])
                            if item.candidate_id == selected_id
                        ),
                        None,
                    )
                    selected_version = None
                    if selected_id:
                        versions = (
                            session.query(GeneratedVersion)
                            .filter_by(task_id=task.task_id, module_type="candidate_project")
                            .order_by(GeneratedVersion.created_at.desc())
                            .all()
                        )
                        selected_version = next(
                            (
                                version
                                for version in versions
                                if isinstance(version.content, dict)
                                and version.content.get("candidate_id") == selected_id
                            ),
                            None,
                        )
                    materialized_payload = {
                        "task_id": str(task.task_id),
                        "job": _sanitized_json(getattr(task, "job", None)),
                        "resume": _sanitized_json(getattr(task, "resume", None)),
                        "candidates": _sanitized_json([selected_candidate] if selected_candidate else []),
                        "state": _sanitized_json(
                            {
                                "selected_candidate": selected_id,
                                "selected_candidate_id": selected_id,
                                "project_patch": task_state.get("project_patch"),
                                "module_patches": task_state.get("module_patches"),
                                "output_mode": task_state.get("output_mode"),
                            }
                        ),
                    }
                    materialized_version = self._persist_state_payload(
                        session,
                        task_id=str(task.task_id),
                        module_type="resume_materialized",
                        value=materialized_payload,
                        checkpoint_version=int(getattr(task, "checkpoint_version", 0)),
                    )
                    materialized_version.is_user_confirmed = is_final
                    materialized_version.state = "confirmed" if materialized_version.is_user_confirmed else "draft"
                    state_ref_names = (
                        "project_patch",
                        "module_decisions",
                        "layout_decision",
                        "compression_decision",
                        "output_mode",
                        "template_id",
                        "previous_snapshot_ids",
                        "final_confirmation",
                    )
                    state_refs: dict[str, str] = {}
                    for name in state_ref_names:
                        version = (
                            session.query(GeneratedVersion)
                            .filter_by(task_id=task.task_id, module_type=f"task_state:{name}")
                            .order_by(GeneratedVersion.version_no.desc())
                            .first()
                        )
                        if version is not None:
                            state_refs[name] = str(version.id)
                    applied_patches = (
                        session.query(ContentPatch)
                        .filter_by(task_id=task.task_id, status="applied")
                        .order_by(ContentPatch.created_at.asc())
                        .all()
                    )
                    module_versions: dict[str, Any] = {
                        "projects": selected_version.id if selected_version else None,
                        "selected_candidate_id": selected_id,
                        "materialized": materialized_version.id,
                        "state_refs": state_refs,
                        "patch_ids": [str(item.id) for item in applied_patches],
                        "job_snapshot_id": getattr(task, "job_snapshot_id", None),
                    }
                    for item in applied_patches:
                        if item.applied_version_id:
                            module_versions[str(item.target_module)] = str(item.applied_version_id)
                    project_order = [
                        section.section_id
                        for section in (getattr(task, "resume", None).sections if getattr(task, "resume", None) else [])
                        if section.module == "projects"
                    ]
                    project_patch = task_state.get("project_patch")
                    if isinstance(project_patch, dict) and selected_id:
                        if project_patch.get("operation") == "replace":
                            target_id = str(project_patch.get("target_project_id") or "")
                            project_order = [selected_id if item == target_id else item for item in project_order]
                        elif project_patch.get("operation") == "add" and selected_id not in project_order:
                            project_order.append(selected_id)
                    template_id = str(task_state.get("template_id") or "").strip() or None
                    if template_id and session.get(TemplateRecord, template_id) is None:
                        template_id = None
                    confirmation_hash = sha256_json(_sanitized_json(task_state.get("confirmations", [])))
                    content = {
                        "selected_candidate_id": selected_id,
                        "candidate_ids": candidate_ids,
                        "selected_version_id": selected_version.id if selected_version else None,
                        "job_snapshot_id": getattr(task, "job_snapshot_id", None),
                        "module_version_ids": module_versions,
                        "project_order": project_order,
                        "template_id": template_id,
                        "layout_version": state_refs.get("layout_decision"),
                        "confirmation_hash": confirmation_hash,
                    }
                    resume_file_id = getattr(task, "resume", None).resume_id if getattr(task, "resume", None) else None
                    # A snapshot may refer to a source file only when that file
                    # was persisted successfully; otherwise leave the FK null.
                    if resume_file_id and session.get(ResumeFile, resume_file_id) is None:
                        resume_file_id = None
                    session.add(
                        ResumeSnapshot(
                            id=snapshot_id,
                            task_id=task.task_id,
                            resume_file_id=resume_file_id,
                            module_version_ids=module_versions,
                            project_order=project_order,
                            template_id=template_id,
                            layout_version=state_refs.get("layout_decision"),
                            confirmation_hash=confirmation_hash,
                            snapshot_hash=sha256_json(content),
                            state="confirmed" if is_final else "draft",
                        )
                    )
                    session.commit()
                return snapshot_id
        except Exception:
            self.available = False
            return None

    def load_resume_snapshot(self, snapshot_id: str, *, task_id: str | None = None) -> dict[str, Any] | None:
        """Load an immutable snapshot and its materialized business version."""

        try:
            with self.factory() as session:
                row = session.get(ResumeSnapshot, str(snapshot_id))
                if row is None or (task_id is not None and row.task_id != task_id):
                    return None
                refs = dict(row.module_version_ids or {})
                materialized_id = refs.get("materialized")
                materialized = session.get(GeneratedVersion, materialized_id) if materialized_id else None
                value = None
                if materialized is not None and isinstance(materialized.content, dict):
                    value = materialized.content.get("value")
                return {
                    "snapshot_id": str(row.id),
                    "task_id": str(row.task_id),
                    "state": str(row.state),
                    "snapshot_hash": str(row.snapshot_hash),
                    "confirmation_hash": str(row.confirmation_hash),
                    "module_version_ids": refs,
                    "project_order": list(row.project_order or []),
                    "template_id": row.template_id,
                    "layout_version": row.layout_version,
                    "materialized": _json_value(value) if isinstance(value, dict) else None,
                }
        except Exception:
            self.available = False
            return None

    def persist_feedback(self, task_id: str, text: str, classification: dict[str, Any] | None = None, *, target_version_id: str | None = None) -> str | None:
        """Persist user feedback and its deterministic classification."""

        try:
            with self.factory() as session:
                classification_id = None
                if classification:
                    row = FeedbackClassificationRow(
                        classifier_type=str(classification.get("classifier_type", "rules-v1")),
                        category=str(classification.get("category", "opinion")),
                        confidence=float(classification.get("confidence", 0.0)),
                        spans=_sanitized_json(classification.get("spans", [])),
                        fact_candidates=_sanitized_json(classification.get("fact_candidates", [])),
                        conflicts=_sanitized_json(classification.get("conflicts", [])),
                        requires_clarification=bool(classification.get("requires_clarification", False)),
                    )
                    session.add(row)
                    session.flush()
                    classification_id = row.id
                feedback = FeedbackRecord(
                    task_id=task_id,
                    target_version_id=(target_version_id if target_version_id and session.get(GeneratedVersion, target_version_id) else None),
                    text=str(redact_sensitive(str(text))),
                    classification_id=classification_id,
                    status="received",
                )
                session.add(feedback)
                session.flush()
                feedback_id = str(feedback.id)
                session.commit()
                return feedback_id
        except Exception:
            self.available = False
            return None

    def persist_content_patch(
        self,
        task_id: str,
        *,
        target_id: str,
        module: str,
        old_value_hash: str,
        new_value: Any,
        reason: str,
        evidence_ids: list[str] | None = None,
        old_value: Any = None,
    ) -> bool:
        """Persist an auditable field patch and immutable old/new versions.

        The API writes its candidate checkpoint before this audit call.  When
        that happens the slot already points at the new version, so ``old_value``
        is used to locate the prior immutable version by hash.  Direct callers
        can omit it and the method creates the new version itself.
        """

        try:
            with self.factory() as session:
                slots = session.query(CandidateSlot).filter_by(task_id=task_id).all()
                target = None
                for slot in slots:
                    if not slot.generated_version_id:
                        continue
                    version = session.get(GeneratedVersion, slot.generated_version_id)
                    if version is not None and isinstance(version.content, dict) and version.content.get("candidate_id") == target_id:
                        target = version
                        break
                if target is None:
                    versions = (
                        session.query(GeneratedVersion)
                        .filter_by(task_id=task_id, module_type="candidate_project")
                        .order_by(GeneratedVersion.created_at.desc())
                        .all()
                    )
                    target = next(
                        (version for version in versions if isinstance(version.content, dict) and version.content.get("candidate_id") == target_id),
                        None,
                    )
                if target is None or not isinstance(target.content, dict) or module not in target.content:
                    return False

                def field_hash(value: Any) -> str:
                    serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    return sha256_text(serialized)

                value = _sanitized_json(new_value)
                current_hash = field_hash(target.content[module])
                source = target if current_hash == old_value_hash else None
                if source is None:
                    # Find the historical version the caller actually edited;
                    # this prevents a stale tab from patching a newer branch.
                    history = (
                        session.query(GeneratedVersion)
                        .filter_by(task_id=task_id, module_type="candidate_project", candidate_slot_id=target.candidate_slot_id)
                        .order_by(GeneratedVersion.version_no.asc())
                        .all()
                    )
                    source = next(
                        (
                            version for version in history
                            if isinstance(version.content, dict)
                            and module in version.content
                            and field_hash(version.content[module]) == old_value_hash
                        ),
                        None,
                    )
                if source is None:
                    session.add(
                        ContentPatch(
                            task_id=task_id,
                            target_version_id=target.id,
                            target_module=module,
                            target_item_id=target_id,
                            old_value_hash=old_value_hash,
                            new_value={"value": value},
                            reason=str(redact_sensitive(reason)),
                            evidence_ids=_sanitized_json(evidence_ids or []),
                            status="conflicted",
                            diff={"expected_old_value_hash": old_value_hash, "actual_value_hash": current_hash},
                        )
                    )
                    session.commit()
                    return False

                old_field_value = source.content[module]
                updated_content = dict(source.content)
                updated_content[module] = value
                updated_hash = sha256_json(updated_content)
                version_id = stable_id("gen", task_id, source.candidate_slot_id or source.id, updated_hash)
                applied = session.get(GeneratedVersion, version_id)
                if applied is None:
                    # Reuse a version already created by persist_checkpoint.
                    applied = next(
                        (
                            version for version in session.query(GeneratedVersion).filter_by(task_id=task_id, module_type="candidate_project").all()
                            if version.candidate_slot_id == source.candidate_slot_id and version.content_hash == updated_hash
                        ),
                        None,
                    )
                if applied is None:
                    latest = (
                        session.query(GeneratedVersion)
                        .filter_by(task_id=task_id, module_type=source.module_type)
                        .order_by(GeneratedVersion.version_no.desc())
                        .first()
                    )
                    applied = GeneratedVersion(
                        id=version_id,
                        task_id=task_id,
                        module_type=source.module_type,
                        version_no=(latest.version_no + 1) if latest else 1,
                        generation_branch_id=source.generation_branch_id,
                        candidate_slot_id=source.candidate_slot_id,
                        content=_sanitized_json(updated_content),
                        content_hash=updated_hash,
                        source_refs=_sanitized_json(source.source_refs),
                        state="draft",
                        is_user_confirmed=False,
                    )
                    session.add(applied)
                    session.flush()
                for slot in slots:
                    if slot.generated_version_id in {target.id, source.id}:
                        slot.generated_version_id = applied.id
                        slot.generation_status = "completed"
                        break
                patch = ContentPatch(
                    task_id=task_id,
                    target_version_id=source.id,
                    target_module=module,
                    target_item_id=target_id,
                    old_value_hash=old_value_hash,
                    new_value={"value": value},
                    reason=str(redact_sensitive(reason)),
                    evidence_ids=_sanitized_json(evidence_ids or []),
                    status="applied",
                    applied_version_id=applied.id,
                    diff={"target_item_id": target_id, "field": module, "old": _sanitized_json(old_field_value), "new": value},
                )
                session.add(patch)
                session.commit()
                return True
        except Exception:
            self.available = False
            return False

    def persist_confirmations(self, task_id: str, confirmations: list[dict[str, Any]]) -> bool:
        try:
            with self.factory() as session:
                self._persist_confirmation_rows(session, task_id, confirmations)
                session.commit()
                return True
        except Exception:
            self.available = False
            return False
    def persist_checkpoint(self, task: Any) -> bool:
        try:
            with self.factory() as session:
                row = session.get(Task, task.task_id)
                if row is None:
                    return False
                if row.thread_id != task.thread_id:
                    return False
                checkpoint_version = int(task.checkpoint_version)
                current_version = int(row.latest_checkpoint_version or 0)
                compact = _compact_state(getattr(task, "state", {}))
                existing = session.query(LangGraphCheckpoint).filter_by(
                    task_id=task.task_id,
                    thread_id=task.thread_id,
                    version=checkpoint_version,
                ).one_or_none()
                if checkpoint_version == current_version:
                    # Idempotent replay is accepted only when the durable row
                    # has the exact same compact state.  Do not mutate metadata
                    # before this check: a stale browser tab must not partially
                    # overwrite the task row.
                    if checkpoint_version == 0:
                        return True
                    if existing is not None and existing.state_ref == compact and existing.node_name == task.current_node:
                        return True
                    if existing is None and row.extra == compact:
                        session.add(
                            LangGraphCheckpoint(
                                task_id=task.task_id,
                                thread_id=task.thread_id,
                                version=checkpoint_version,
                                node_name=task.current_node,
                                state_ref=compact,
                                pending_action=(
                                    _compact_state({"pending_action": getattr(task, "state", {}).get("pending_action")}).get("pending_action")
                                    if isinstance(getattr(task, "state", {}), dict)
                                    else None
                                ),
                                object_refs=_sanitized_json({"job_snapshot_id": task.job_snapshot_id}),
                            )
                        )
                        session.commit()
                        return True
                    session.rollback()
                    return False
                if checkpoint_version != current_version + 1:
                    # A caller cannot skip a checkpoint version or jump over a
                    # write committed by another worker.
                    session.rollback()
                    return False
                # Reconcile only forward versions.  Replaying the same workflow
                # result is idempotent instead of turning the mirror unavailable.
                row.status = task.status.value if hasattr(task.status, "value") else str(task.status)
                row.current_node = task.current_node
                row.blocked_reason = getattr(task, "blocked_reason", None)
                state = getattr(task, "state", {}) or {}
                chat_version = self._ensure_task_profile_version(session, state, "chat")
                embedding_version = self._ensure_task_profile_version(session, state, "embedding")
                if chat_version is not None:
                    row.chat_model_profile_version_id = chat_version.id
                if embedding_version is not None:
                    row.embedding_model_profile_version_id = embedding_version.id
                if state.get("generation_branch_id") is not None:
                    row.generation_branch_id = str(state["generation_branch_id"])
                row.latest_checkpoint_version = checkpoint_version
                row.extra = _compact_state(state)
                self._persist_business_state(session, task)
                self._persist_confirmation_rows(session, str(task.task_id), state.get("confirmations"))
                # Candidate bodies belong to business-version rows, not the
                # lightweight graph checkpoint.  Upsert one immutable draft per
                # candidate slot so a restarted process can rebuild the review.
                for candidate in (getattr(task, "candidates", None) or []):
                    slot_id = str(getattr(candidate, "candidate_slot_id", ""))
                    if not slot_id:
                        continue
                    slot = session.query(CandidateSlot).filter_by(task_id=task.task_id, candidate_slot_id=slot_id).one_or_none()
                    if slot is None:
                        slot = CandidateSlot(task_id=task.task_id, candidate_slot_id=slot_id, slot_index=len(session.query(CandidateSlot).filter_by(task_id=task.task_id).all()) + 1)
                        session.add(slot)
                        session.flush()
                    content = _sanitized_json(
                        candidate.model_dump(mode="json") if hasattr(candidate, "model_dump") else dict(candidate)
                    )
                    content_hash = sha256_json(content)
                    # Generated versions are immutable.  Include the content
                    # hash in the deterministic key so a feedback branch can
                    # point the slot at a new version while replaying the same
                    # checkpoint remains idempotent.
                    version_id = stable_id("gen", task.task_id, slot_id, content_hash)
                    generated_version = session.get(GeneratedVersion, version_id)
                    if generated_version is None:
                        latest_version = (
                            session.query(GeneratedVersion)
                            .filter_by(
                                task_id=task.task_id,
                                module_type="candidate_project",
                                candidate_slot_id=slot_id,
                            )
                            .order_by(GeneratedVersion.version_no.desc())
                            .first()
                        )
                        generated_version = GeneratedVersion(
                            id=version_id,
                            task_id=task.task_id,
                            module_type="candidate_project",
                            version_no=(latest_version.version_no + 1) if latest_version else 1,
                            generation_branch_id=state.get("generation_branch_id"),
                            candidate_slot_id=slot_id,
                            content=content,
                            content_hash=content_hash,
                            source_refs={"job_snapshot_id": task.job_snapshot_id},
                            state=str(getattr(candidate, "status", "draft")),
                            is_user_confirmed=not bool(getattr(candidate, "needs_verification", True)),
                        )
                        session.add(generated_version)
                    slot.generated_version_id = version_id
                    slot.generation_status = "completed"
                    slot.dedup_status = str(getattr(candidate, "status", "draft"))
                    slot.ranking_score = float(getattr(candidate, "match_score", 0.0))
                match = getattr(task, "match", None)
                if match is not None:
                    dimensions = match.dimensions
                    public_match_id = str(match.match_run_id)
                    existing_matches = session.query(MatchRun).filter_by(task_id=task.task_id).all()
                    existing_match = next(
                        (
                            item
                            for item in existing_matches
                            if item.id == public_match_id
                            or (
                                isinstance(item.evidence, dict)
                                and isinstance(item.evidence.get("_meta"), dict)
                                and item.evidence["_meta"].get("match_run_id") == public_match_id
                            )
                        ),
                        None,
                    )
                    if existing_match is None:
                        scoring_config = self._ensure_scoring_config(session, state)
                        storage_id = public_match_id
                        occupied = session.get(MatchRun, storage_id)
                        if occupied is not None and occupied.task_id != task.task_id:
                            storage_id = stable_id("match-row", task.task_id, public_match_id)
                        session.add(MatchRun(
                            id=storage_id,
                            task_id=task.task_id,
                            job_snapshot_id=task.job_snapshot_id,
                            resume_file_id=row.resume_file_id,
                            scoring_config_id=scoring_config.id,
                            skill_score=float(dimensions.get("skills").score if dimensions.get("skills") else 0.0),
                            responsibility_score=float(
                                dimensions.get("responsibilities").score
                                if dimensions.get("responsibilities")
                                else 0.0
                            ),
                            project_score=float(dimensions.get("projects").score if dimensions.get("projects") else 0.0),
                            keyword_score=float(dimensions.get("keywords").score if dimensions.get("keywords") else 0.0),
                            total_score_raw=float(match.total_score),
                            total_score=float(match.total_score),
                            weights={name: float(value.weight) for name, value in dimensions.items()},
                            evidence={
                                **{
                                    name: {
                                        "deterministic": float(value.deterministic),
                                        "semantic": float(value.semantic),
                                        "evidence_ids": list(value.evidence_ids),
                                    }
                                    for name, value in dimensions.items()
                                },
                                "_meta": {
                                    "match_run_id": public_match_id,
                                    "scoring_config_version": str(match.scoring_config_version),
                                    "threshold_triggered": bool(match.threshold_triggered),
                                },
                            },
                            embedding_mode=str(match.embedding_mode),
                        ))
                if existing is None and checkpoint_version > 0:
                    session.add(
                        LangGraphCheckpoint(
                            task_id=task.task_id,
                            thread_id=task.thread_id,
                            version=checkpoint_version,
                            node_name=task.current_node,
                            state_ref=_compact_state(getattr(task, "state", {})),
                            pending_action=(
                                _compact_state({"pending_action": getattr(task, "state", {}).get("pending_action")}).get("pending_action")
                                if isinstance(getattr(task, "state", {}), dict)
                                else None
                            ),
                            object_refs=_sanitized_json({"job_snapshot_id": task.job_snapshot_id}),
                        )
                    )
                elif existing is not None:
                    # Do not overwrite an existing checkpoint with a stale or
                    # divergent snapshot from another browser tab.
                    if existing.state_ref != compact:
                        session.rollback()
                        return False
                session.commit()
                return True
        except Exception:
            self.available = False
            return False

    def load_jobs(self) -> list[tuple[str, Any]]:
        """Load the latest immutable job records for process restart hydration."""

        from app.core.schemas import JobInput

        try:
            with self.factory() as session:
                rows = session.query(JobRecord).order_by(JobRecord.updated_at.desc()).all()
                return [
                    (
                        row.id,
                        JobInput(
                            title=row.job_title,
                            company=row.company_name,
                            city=row.city,
                            salary=row.salary,
                            responsibilities=(row.responsibilities or "").splitlines(),
                            requirements=(row.requirements or "").splitlines(),
                            skills=list(row.skills or []),
                            hr_activity=(row.extra or {}).get("hr_activity"),
                            posted_at=row.posted_at,
                            posted_at_label="发布时间未知" if row.posted_at is None else None,
                            source_url=_safe_source_url(row.source_url),
                        ),
                    )
                    for row in rows
                ]
        except Exception:
            self.available = False
            return []

    def load_tasks(self) -> list[Any]:
        """Hydrate task control state and candidate versions after restart."""

        from app.core.schemas import CandidateProject, JobInput, ResumeDocument, ResumeSection as ResumeSectionSchema, TaskStatus
        from app.core.store import TaskRecord

        try:
            with self.factory() as session:
                rows = session.query(Task).order_by(Task.created_at.asc()).all()
                result: list[TaskRecord] = []
                for row in rows:
                    snapshot = session.get(JobSnapshot, row.job_snapshot_id) if row.job_snapshot_id else None
                    if snapshot is None:
                        continue
                    job_record = (
                        session.get(JobRecord, snapshot.job_record_id)
                        if snapshot.job_record_id
                        else None
                    )
                    job = JobInput(
                        title=snapshot.job_title,
                        company=snapshot.company_name,
                        city=snapshot.city,
                        salary=snapshot.salary,
                        responsibilities=(snapshot.responsibilities or "").splitlines(),
                        requirements=(snapshot.requirements or "").splitlines(),
                        skills=list(snapshot.skills or []),
                        hr_activity=(job_record.extra or {}).get("hr_activity") if job_record else None,
                        posted_at=snapshot.posted_at,
                        posted_at_label="发布时间未知" if snapshot.posted_at is None else None,
                        source_url=_safe_source_url(snapshot.source_url),
                    )
                    candidates: list[CandidateProject] = []
                    # CandidateSlot is the mutable pointer to the latest
                    # immutable GeneratedVersion for that slot.  Restoring all
                    # historical versions would surface feedback revisions as
                    # duplicate candidates, so follow the pointers first.
                    slots = (
                        session.query(CandidateSlot)
                        .filter_by(task_id=row.id)
                        .order_by(CandidateSlot.slot_index.asc())
                        .all()
                    )
                    versions: list[GeneratedVersion] = []
                    seen_version_ids: set[str] = set()
                    for slot in slots:
                        if not slot.generated_version_id:
                            continue
                        version = session.get(GeneratedVersion, slot.generated_version_id)
                        if version is not None and version.id not in seen_version_ids:
                            versions.append(version)
                            seen_version_ids.add(version.id)
                    # Keep compatibility with rows written before candidate
                    # slots were introduced: choose the newest version per slot
                    # rather than returning every historical revision.
                    historical = (
                        session.query(GeneratedVersion)
                        .filter_by(task_id=row.id, module_type="candidate_project")
                        .order_by(GeneratedVersion.created_at.desc())
                        .all()
                    )
                    latest_by_slot: dict[str, GeneratedVersion] = {}
                    for version in historical:
                        slot_key = str(version.candidate_slot_id or version.id)
                        latest_by_slot.setdefault(slot_key, version)
                    for version in latest_by_slot.values():
                        if version.id not in seen_version_ids:
                            versions.append(version)
                            seen_version_ids.add(version.id)
                    for version in versions:
                        try:
                            candidates.append(CandidateProject.model_validate(version.content))
                        except Exception:
                            continue
                    resume = None
                    if row.resume_file_id:
                        resume_file = session.get(ResumeFile, row.resume_file_id)
                        if resume_file is not None:
                            sections = [
                                ResumeSectionSchema(
                                    section_id=section.section_id,
                                    module=section.module_type,
                                    title=(section.source_locator or {}).get("title") if isinstance(section.source_locator, dict) else None,
                                    content=section.text,
                                    confirmed=bool(section.user_corrected),
                                    value_hash=section.text_hash,
                                )
                                for section in session.query(ResumeSection).filter_by(resume_file_id=resume_file.id).order_by(ResumeSection.item_index.asc()).all()
                            ]
                            raw_text = ""
                            try:
                                raw_text = Path(resume_file.path).read_text(encoding="utf-8", errors="replace")
                            except (OSError, UnicodeError):
                                raw_text = "\n".join(section.content for section in sections)
                            resume = ResumeDocument(
                                resume_id=resume_file.id,
                                filename=resume_file.original_name or "resume",
                                file_type=resume_file.format if resume_file.format in {"docx", "pdf", "markdown", "text"} else "text",
                                file_size=resume_file.size_bytes,
                                page_count=resume_file.page_count,
                                sections=sections,
                                raw_text=raw_text,
                            )
                    status = row.status if row.status != "running" else "paused"
                    raw_extra = _sanitized_json(dict(row.extra or {}))
                    # Re-apply the compact whitelist to legacy rows as well;
                    # older builds may have stored editable drafts or profile
                    # snapshots directly in ``tasks.extra``.
                    compact_state = _compact_state(raw_extra)
                    profile_versions: dict[str, ModelProfileVersion] = {}
                    for prefix, role, version_id in (
                        ("chat", "chat", row.chat_model_profile_version_id),
                        ("embedding", "embedding", row.embedding_model_profile_version_id),
                    ):
                        if not version_id:
                            continue
                        version = session.get(ModelProfileVersion, version_id)
                        if version is None:
                            # A broken FK/reference must never silently select a
                            # different model.  Leave the task blocked for a
                            # repair/retry rather than hydrating a mutable card.
                            compact_state.setdefault("model_binding_error", []).append(
                                f"{prefix}_profile_version_missing"
                            )
                            continue
                        profile_versions[prefix] = version
                        profile_row = session.get(ModelProfileRow, version.profile_id)
                        probe = dict(version.probe_result or {})
                        compact_state[f"{prefix}_profile_id"] = version.profile_id
                        compact_state[f"{prefix}_profile_version"] = int(version.version_no)
                        compact_state[f"{prefix}_profile_snapshot"] = _sanitized_json(
                            {
                                "profile_id": version.profile_id,
                                "role": version.role,
                                "provider": version.provider,
                                "base_url": version.base_url,
                                "model_name": version.model_name,
                                "status": probe.get("status") or (profile_row.status if profile_row else "unknown"),
                                "credential_required": bool(
                                    profile_row.credential_required
                                    if profile_row is not None
                                    else version.provider == "openai_compatible"
                                ),
                                "context_window_tokens": version.context_window_tokens,
                                "tokenizer_id": version.tokenizer_id,
                                "tokenizer_version": version.tokenizer_version,
                                "tokenizer_source": version.tokenizer_source,
                                "dimension": version.dimension,
                                "max_input_tokens": version.max_input_tokens,
                                "capabilities": dict(version.capabilities or {}),
                                "generation_params": dict(version.generation_params or {}),
                                "config_version": int(version.version_no),
                                "model_digest": version.model_digest,
                                "embedding_algorithm_version": version.embedding_algorithm_version,
                                "budget_policy_version": version.budget_policy_version,
                                "error_code": probe.get("error_code"),
                            }
                        )
                    # Legacy rows may predate the FK columns.  Preserve only a
                    # validated, sanitized profile snapshot long enough to
                    # enforce the restart credential gate; never trust it as a
                    # replacement for a missing immutable version on writes.
                    for prefix, role in (("chat", "chat"), ("embedding", "embedding")):
                        if f"{prefix}_profile_snapshot" in compact_state:
                            continue
                        legacy = raw_extra.get(f"{prefix}_profile_snapshot")
                        if not isinstance(legacy, dict):
                            continue
                        try:
                            self._profile_values_from_snapshot(legacy, role)
                            legacy_version = int(str(legacy.get("config_version") or 1).removeprefix("v"))
                        except (TypeError, ValueError):
                            continue
                        compact_state[f"{prefix}_profile_id"] = str(legacy.get("profile_id") or "")
                        compact_state[f"{prefix}_profile_version"] = legacy_version
                        compact_state[f"{prefix}_profile_snapshot"] = _sanitized_json(legacy)
                    feedback_id = compact_state.get("last_feedback_id")
                    if feedback_id:
                        feedback_row = session.get(FeedbackRecord, str(feedback_id))
                        if feedback_row is not None and feedback_row.task_id == row.id:
                            compact_state["last_feedback"] = feedback_row.text
                    match_row = (
                        session.query(MatchRun)
                        .filter_by(task_id=row.id)
                        .order_by(MatchRun.created_at.desc())
                        .first()
                    )
                    match = None
                    if match_row is not None:
                        from app.core.schemas import MatchDimension, MatchResult

                        evidence = dict(match_row.evidence or {})
                        match_meta = evidence.get("_meta", {}) if isinstance(evidence.get("_meta"), dict) else {}
                        weights = dict(match_row.weights or {})
                        score_values = {
                            "skills": match_row.skill_score,
                            "responsibilities": match_row.responsibility_score,
                            "projects": match_row.project_score,
                            "keywords": match_row.keyword_score,
                        }
                        dimensions = {}
                        for name, score_value in score_values.items():
                            audit = evidence.get(name, {}) if isinstance(evidence.get(name, {}), dict) else {}
                            dimensions[name] = MatchDimension(
                                deterministic=float(audit.get("deterministic", 0.0)),
                                semantic=float(audit.get("semantic", 0.0)),
                                score=float(score_value),
                                weight=float(weights.get(name, 0.25)),
                                evidence_ids=list(audit.get("evidence_ids", [])),
                            )
                        scoring_config = session.get(ScoringConfig, match_row.scoring_config_id) if match_row.scoring_config_id else None
                        match = MatchResult(
                            match_run_id=str(match_meta.get("match_run_id") or match_row.id),
                            total_score=float(match_row.total_score),
                            dimensions=dimensions,
                            threshold_triggered=bool(
                                match_meta.get("threshold_triggered")
                                if "threshold_triggered" in match_meta
                                else float(match_row.total_score_raw)
                                < float(scoring_config.threshold if scoring_config is not None else 80.0)
                            ),
                            scoring_config_version=str(
                                match_meta.get("scoring_config_version")
                                or (scoring_config.version if scoring_config is not None else compact_state.get("scoring_config_version") or "scoring-v1")
                            ),
                            embedding_mode=match_row.embedding_mode,
                        )
                        compact_state["last_match_run_id"] = match.match_run_id
                        if scoring_config is not None:
                            compact_state["scoring_config_version"] = scoring_config.version
                            compact_state["dimension_weights"] = dict(scoring_config.dimension_weights or {})
                            compact_state["component_weights"] = {
                                "deterministic": float(scoring_config.deterministic_weight),
                                "semantic": float(scoring_config.semantic_weight),
                            }
                    # Business-state versions are the durable source for fields
                    # intentionally omitted from compact checkpoints.
                    state_versions = (
                        session.query(GeneratedVersion)
                        .filter_by(task_id=row.id)
                        .filter(GeneratedVersion.module_type.like("task_state:%"))
                        .order_by(GeneratedVersion.created_at.asc())
                        .all()
                    )
                    for version in state_versions:
                        if not isinstance(version.content, dict) or "value" not in version.content:
                            continue
                        suffix = version.module_type.removeprefix("task_state:")
                        if suffix in {"base_facts", "project_patch", "preview_feedback"}:
                            compact_state[suffix] = version.content.get("value")
                        elif suffix.startswith("module_draft:"):
                            compact_state.setdefault("module_drafts", {})[suffix.split(":", 1)[1]] = version.content.get("value")
                        elif suffix.startswith("module_patch:"):
                            compact_state.setdefault("module_patches", {})[suffix.split(":", 1)[1]] = version.content.get("value")
                    confirmation_rows = session.query(Confirmation).filter_by(task_id=row.id).all()
                    if confirmation_rows:
                        compact_state["confirmations"] = [
                            {
                                "field_id": item.field_id,
                                "value_hash": item.value_hash,
                                "confirmed": item.status == "confirmed",
                                "source_version_id": item.source_version_id,
                                "patch_version_id": item.patch_version_id,
                            }
                            for item in confirmation_rows
                        ]
                    restored_blocked_reason = row.blocked_reason or (
                        "needs_restart_review" if row.status == "running" else None
                    )
                    external_profile = any(
                        isinstance(compact_state.get(f"{prefix}_profile_snapshot"), dict)
                        and compact_state[f"{prefix}_profile_snapshot"].get("provider") == "openai_compatible"
                        for prefix in ("chat", "embedding")
                    )
                    if status not in {"completed", "cancelled", "deleted"} and external_profile:
                        status = "paused"
                        restored_blocked_reason = "needs_credentials"
                    elif status not in {"completed", "cancelled", "deleted"} and compact_state.get("model_binding_error"):
                        status = "paused"
                        restored_blocked_reason = "needs_model_recheck"
                    result.append(
                        TaskRecord(
                            task_id=row.id,
                            thread_id=row.thread_id,
                            status=TaskStatus(status),
                            blocked_reason=restored_blocked_reason,
                            current_node=row.current_node,
                            checkpoint_version=row.latest_checkpoint_version,
                            job_snapshot_id=row.job_snapshot_id,
                            job=job,
                            resume=resume,
                            candidates=candidates,
                            match=match,
                            state=compact_state,
                            messages=["任务已从持久层恢复，请确认是否继续"],
                        )
                    )
                return result
        except Exception:
            self.available = False
            return []

    def load_resumes(self) -> list[Any]:
        """Load source resume metadata/sections for process restart hydration."""

        from app.core.schemas import ResumeDocument, ResumeSection as ResumeSectionSchema

        try:
            with self.factory() as session:
                result: list[ResumeDocument] = []
                rows = session.query(ResumeFile).filter(ResumeFile.file_role == "source").order_by(ResumeFile.created_at.asc()).all()
                for row in rows:
                    sections = [
                        ResumeSectionSchema(
                            section_id=section.section_id,
                            module=section.module_type,
                            title=(section.source_locator or {}).get("title") if isinstance(section.source_locator, dict) else None,
                            content=section.text,
                            confirmed=bool(section.user_corrected),
                            value_hash=section.text_hash,
                        )
                        for section in session.query(ResumeSection).filter_by(resume_file_id=row.id).order_by(ResumeSection.item_index.asc()).all()
                    ]
                    raw_text = ""
                    try:
                        raw_text = Path(row.path).read_text(encoding="utf-8", errors="replace")
                    except (OSError, UnicodeError):
                        raw_text = "\n".join(section.content for section in sections)
                    result.append(
                        ResumeDocument(
                            resume_id=row.id,
                            filename=row.original_name or "resume",
                            file_type=row.format if row.format in {"docx", "pdf", "markdown", "text"} else "text",
                            file_size=row.size_bytes,
                            page_count=row.page_count,
                            sections=sections,
                            raw_text=raw_text,
                        )
                    )
                return result
        except Exception:
            self.available = False
            return []

    def persist_event(self, task_id: str, event: dict[str, Any]) -> bool:
        try:
            with self.factory() as session:
                append_event(
                    session,
                    task_id,
                    str(event.get("type", "event")),
                    node_name=event.get("node"),
                    status=event.get("status"),
                    message=redact_sensitive(event.get("message") or event.get("error")),
                    payload=_sanitized_json({k: v for k, v in event.items() if k not in {"type", "node", "status", "message", "error"}}),
                )
                session.commit()
                return True
        except Exception:
            self.available = False
            return False

    def acquire_lease(self, task_id: str, worker_id: str, *, lease_seconds: int = 60, takeover_grace_seconds: int = 30) -> str | None:
        try:
            with self.factory() as session:
                lease = acquire_worker_lease(session, task_id, worker_id, lease_seconds=lease_seconds, takeover_grace_seconds=takeover_grace_seconds)
                session.commit()
                return lease.id
        except LeaseBusy:
            return None
        except Exception:
            self.available = False
            return None

    def renew_lease(self, lease_id: str, worker_id: str, *, lease_seconds: int = 60) -> bool:
        """Renew an owned lease without treating ownership loss as DB outage."""

        try:
            with self.factory() as session:
                renew_worker_lease(
                    session,
                    lease_id,
                    worker_id,
                    lease_seconds=lease_seconds,
                )
                session.commit()
                return True
        except (LeaseBusy, NotFoundError):
            return False
        except Exception:
            self.available = False
            return False

    def release_lease(self, lease_id: str, worker_id: str) -> bool:
        try:
            with self.factory() as session:
                release_worker_lease(session, lease_id, worker_id)
                session.commit()
                return True
        except (LeaseBusy, NotFoundError):
            return False
        except Exception:
            self.available = False
            return False

    def healthcheck(self) -> bool:
        """Probe the configured database and clear a transient outage flag."""

        from sqlalchemy import text

        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            self.available = True
            return True
        except Exception:
            self.available = False
            return False

    def close(self) -> None:
        """Release pooled database connections owned by this adapter."""

        self.engine.dispose()
