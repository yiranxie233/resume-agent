"""LangGraph-backed workflow for the first end-to-end task slice."""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, TypedDict

from app.core.schemas import CandidateProject, JobInput, ResumeDocument, TaskStatus
from app.core.store import InMemoryStore, TaskRecord, VersionConflict
from app.services.candidate_generator import generate_candidates, generate_candidates_with_model
from app.services.matching import calculate_match
from app.services.feedback import classify_feedback

try:  # LangGraph is optional for lightweight utility imports.
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - exercised only in minimal installations
    END = "__end__"
    StateGraph = None  # type: ignore[assignment,misc]


class GraphState(TypedDict, total=False):
    task_id: str
    thread_id: str
    job: dict[str, Any]
    resume: dict[str, Any] | None
    candidate_count: int
    embedding_mode: str
    match: dict[str, Any]
    candidates: list[dict[str, Any]]
    current_node: str
    status: str
    messages: list[str]
    last_feedback: str
    generation_branch_id: str
    chat_profile: dict[str, Any]
    embedding_profile: dict[str, Any]
    chat_profile_snapshot: dict[str, Any]
    embedding_profile_snapshot: dict[str, Any]
    credential_handle_id: str | None
    chat_credential_handle_id: str | None
    embedding_credential_handle_id: str | None
    strict_model_gate: bool
    model_audit: dict[str, Any]
    base_facts: dict[str, Any]
    retry_step: str
    candidate_generation_status: str
    output_mode: str
    requested_output_mode: str
    final_product: str
    template_id: str | None
    template_structure_snapshot: dict[str, Any]
    selected_candidate: str
    selected_candidate_id: str
    dimension_weights: dict[str, float]
    component_weights: dict[str, float]
    scoring_config_version: str


# These errors are safety/configuration gates, not ordinary provider failures.
# Even in the optional-model deployment they must pause the task instead of
# silently switching to deterministic text generation.
_NON_DEGRADABLE_MODEL_ERRORS = {
    "credential_missing",
    "auth_failed",
    "needs_external_model_consent",
    "context_budget_blocked",
    "context_window_unknown",
    "context_exceeds_budget",
    "fixed_context_exceeds_budget",
    "usable_input_non_positive",
    "required_item_missing",
    "required_item_exceeds_budget",
    "token_estimation_failed",
    "invalid_context",
    "duplicate_context_id",
    "reindex_required",
    "embedding_profile_missing",
}


class _WorkerLeaseLost(RuntimeError):
    code = "worker_lease_lost"


def _node_match(state: GraphState) -> GraphState:
    try:
        job = JobInput.model_validate(state["job"])
        resume = ResumeDocument.model_validate(state["resume"]) if state.get("resume") else None
        dimension_weights = state.get("dimension_weights")
        component_weights = state.get("component_weights")
        result = calculate_match(
            job,
            resume,
            weights=dict(dimension_weights) if isinstance(dimension_weights, dict) else None,
            component_weights=dict(component_weights) if isinstance(component_weights, dict) else None,
            embedding_mode=state.get("embedding_mode", "tfidf_fallback"),
            scoring_config_version=str(state.get("scoring_config_version", "scoring-v1")),
        )
    except Exception as exc:
        try:
            setattr(exc, "workflow_node", "match")
        except Exception:
            pass
        raise
    return {
        **state,
        "match": result.model_dump(mode="json"),
        "current_node": "match",
        "messages": [*state.get("messages", []), "匹配评分完成"],
    }


def _node_candidates(state: GraphState, *, gateway: Any | None = None) -> GraphState:
    job = JobInput.model_validate(state["job"])
    resume = ResumeDocument.model_validate(state["resume"]) if state.get("resume") else None
    from app.core.schemas import MatchResult

    match = MatchResult.model_validate(state["match"])
    project_only = str(
        state.get("requested_output_mode") or state.get("output_mode") or ""
    ) == "project_only"
    candidate_count = max(1, min(5, int(state.get("candidate_count", 3))))
    model_audit: dict[str, Any] | None = None
    use_model = bool(
        gateway
        and isinstance(state.get("chat_profile"), dict)
        and state.get("chat_profile", {}).get("status") == "ready"
    )
    if use_model:
        try:
            candidates, model_audit = generate_candidates_with_model(
                job,
                resume,
                match,
                profile=state["chat_profile"],
                gateway=gateway,
                credential_handle_id=state.get("credential_handle_id"),
                embedding_credential_handle_id=state.get("embedding_credential_handle_id"),
                count=candidate_count,
                task_id=state["task_id"],
                thread_id=state.get("thread_id"),
                embedding_mode=state.get("embedding_mode", "tfidf_fallback"),
                embedding_profile=state.get("embedding_profile"),
                feedback=state.get("last_feedback"),
                branch_id=state.get("generation_branch_id", "branch-1"),
                base_facts=state.get("base_facts", {}),
            )
        except Exception as exc:
            # Strict deployments surface provider/schema/budget failures as a
            # recoverable task error.  Development mode keeps the local
            # deterministic path available and records why it was used.
            error_code = str(getattr(exc, "code", "") or "")
            if bool(state.get("strict_model_gate")) or error_code in _NON_DEGRADABLE_MODEL_ERRORS or bool(getattr(exc, "requires_user", False)):
                try:
                    setattr(exc, "workflow_node", "candidate_generation")
                except Exception:
                    pass
                raise
            model_audit = {"fallback_reason": getattr(exc, "code", str(exc))}
            candidates = generate_candidates(
                job,
                resume,
                match,
                count=candidate_count,
                task_id=state["task_id"],
                branch_id=state.get("generation_branch_id", "branch-1"),
                embedding_mode=state.get("embedding_mode", "tfidf_fallback"),
                embedding_profile=state.get("embedding_profile"),
                gateway=gateway,
                credential_handle_id=state.get("credential_handle_id"),
                embedding_credential_handle_id=state.get("embedding_credential_handle_id"),
                thread_id=state.get("thread_id"),
                feedback=state.get("last_feedback"),
            )
    else:
        candidates = generate_candidates(
            job,
            resume,
            match,
            count=candidate_count,
            task_id=state["task_id"],
            branch_id=state.get("generation_branch_id", "branch-1"),
            embedding_mode=state.get("embedding_mode", "tfidf_fallback"),
            embedding_profile=state.get("embedding_profile"),
            gateway=gateway,
            credential_handle_id=state.get("credential_handle_id"),
            embedding_credential_handle_id=state.get("embedding_credential_handle_id"),
            thread_id=state.get("thread_id"),
            feedback=state.get("last_feedback"),
        )
    requested_count = candidate_count
    generation_status = "complete" if len(candidates) >= requested_count else "partial"
    messages = [*state.get("messages", [])]
    if project_only:
        messages.append(
            f"已生成 {len(candidates)} 份可复制项目经历，请选择并核实后直接复制使用"
        )
    else:
        messages.append(f"已生成 {len(candidates)} 个候选项目，等待选择或反馈")
    if generation_status == "partial":
        messages.append(f"当前仅生成 {len(candidates)}/{requested_count} 个候选，用户可选择继续补生成")
    if model_audit and model_audit.get("fallback_reason"):
        messages.append(f"模型生成不可用，已使用本地降级：{model_audit['fallback_reason']}")
    result: GraphState = {
        **state,
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "current_node": (
            "candidate_confirmation"
            if project_only and len(candidates) == 1
            else "candidate_review"
        ),
        "status": TaskStatus.WAITING_USER.value,
        "messages": messages,
        "model_audit": model_audit or {},
        "candidate_generation_status": generation_status,
    }
    if project_only and len(candidates) == 1:
        selected_id = candidates[0].candidate_id
        result.update(
            {
                "selected_candidate": selected_id,
                "selected_candidate_id": selected_id,
                "candidate_count": candidate_count,
                "output_mode": "project_only",
                "requested_output_mode": "project_only",
            }
        )
    return result


def build_resume_graph(*, gateway: Any | None = None):
    """Build a deterministic graph; graph nodes remain independently testable."""
    if StateGraph is None:
        return None
    graph = StateGraph(GraphState)
    graph.add_node("match", _node_match)
    if gateway is None:
        graph.add_node("generate_candidates", _node_candidates)
    else:
        graph.add_node("generate_candidates", lambda state: _node_candidates(state, gateway=gateway))
    graph.set_entry_point("match")
    graph.add_edge("match", "generate_candidates")
    graph.add_edge("generate_candidates", END)
    return graph.compile()


_MODULE_SEQUENCE = ("summary", "skills", "evaluation")
_MODULE_LABELS = {
    "summary": "个人简介",
    "skills": "技能清单",
    "evaluation": "个人评价",
}


def _value_hash(value: Any) -> str:
    from app.core.utils import sha256_text

    if isinstance(value, str):
        return sha256_text(value)
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _project_sections(task: TaskRecord) -> list[Any]:
    if task.resume is None:
        return []
    return [section for section in task.resume.sections if section.module == "projects"]


def _project_recommendation(task: TaskRecord) -> dict[str, Any]:
    projects = _project_sections(task)
    terms: list[str] = []
    if task.job is not None:
        terms = [task.job.title, *task.job.skills, *task.job.requirements, *task.job.responsibilities]
    from app.core.utils import normalize_terms, term_hits

    normalized_terms = normalize_terms(terms)
    rankings = []
    for section in projects:
        hits = term_hits(section.content, normalized_terms)
        score = round(len(hits) / len(normalized_terms) * 100, 1) if normalized_terms else 0.0
        rankings.append(
            {
                "project_id": section.section_id,
                "title": str(section.title or "未命名项目").strip(),
                "content_preview": str(section.content or "").strip()[:120],
                "match_score": score,
                "matched_terms": hits,
            }
        )
    rankings.sort(key=lambda item: (-item["match_score"], item["project_id"]))
    if len(projects) >= 2:
        target_id = rankings[-1]["project_id"]
        return {
            "recommended_action": "replace",
            "target_project_id": target_id,
            "project_count": len(projects),
            "project_rankings": rankings,
            "reason": "现有项目不少于两个，建议替换与目标岗位匹配度最低的项目；正式应用前仍需用户确认目标项目",
        }
    return {
        "recommended_action": "add",
        "target_project_id": None,
        "project_count": len(projects),
        "project_rankings": rankings,
        "reason": "现有项目少于两个，建议新增已确认的候选项目",
    }


def _module_assessment(task: TaskRecord) -> dict[str, dict[str, Any]]:
    """Return deterministic, auditable gates for the three optional modules."""

    threshold_triggered = bool(task.match and task.match.threshold_triggered)
    job = task.job
    terms: list[str] = []
    if job is not None:
        terms = [job.title, *job.skills, *job.requirements, *job.responsibilities]
    normalized_terms = []
    for term in terms:
        value = str(term).strip().casefold()
        if len(value) >= 2 and value not in normalized_terms:
            normalized_terms.append(value)

    assessment: dict[str, dict[str, Any]] = {}
    for module in _MODULE_SEQUENCE:
        sections = [] if task.resume is None else [
            section for section in task.resume.sections if section.module == module
        ]
        text = "\n".join(section.content for section in sections).casefold()
        if module == "skills" and task.match is not None and "skills" in task.match.dimensions:
            score = float(task.match.dimensions["skills"].score)
        elif not normalized_terms:
            score = 100.0 if text else 0.0
        else:
            hits = sum(1 for term in normalized_terms if term in text)
            score = round(hits / len(normalized_terms) * 100, 1)
        needs_optimization = threshold_triggered and (not sections or score < 80.0)
        assessment[module] = {
            "label": _MODULE_LABELS[module],
            "score": score,
            "present": bool(sections),
            "needs_optimization": needs_optimization,
            "reason": (
                "总匹配分低于 80，且该模块缺失或与岗位关键词覆盖不足"
                if needs_optimization
                else "该模块不触发本轮优化"
            ),
        }
    return assessment


def _next_module(
    assessment: dict[str, dict[str, Any]], decisions: dict[str, Any]
) -> str | None:
    for module in _MODULE_SEQUENCE:
        if assessment.get(module, {}).get("needs_optimization") and module not in decisions:
            return module
    return None


def _module_draft(task: TaskRecord, module: str, *, feedback: str = "", new_value: Any = None) -> dict[str, Any]:
    sections = [] if task.resume is None else [
        section for section in task.resume.sections if section.module == module
    ]
    target = sections[0] if sections else None
    old_value = target.content if target else ""
    if new_value is None:
        skills = list(task.job.skills[:6]) if task.job else []
        focus = "、".join(skills) or (task.job.title if task.job else "目标岗位要求")
        label = _MODULE_LABELS[module]
        prefix = f"{old_value}\n" if old_value else ""
        new_value = (
            f"{prefix}[待核实] {label}建议围绕“{focus}”结合本人真实经历补充；"
            "确认前不会写入正式简历。"
        )
        if feedback:
            new_value += f" 用户反馈：{feedback}"
    if not isinstance(new_value, str) or not new_value.strip():
        raise ValueError("new_value must be a non-empty string")
    value = new_value.strip()
    return {
        "patch_id": f"patch-{uuid.uuid4().hex[:12]}",
        "target_module": module,
        "target_item_id": target.section_id if target else f"{module}:new",
        "old_value_hash": target.value_hash or _value_hash(old_value) if target else _value_hash(""),
        "new_value": value,
        "new_value_hash": _value_hash(value),
        "reason": feedback or f"根据岗位匹配结果优化{_MODULE_LABELS[module]}",
        "needs_verification": True,
        "status": "draft",
    }


def _layout_review_state(task: TaskRecord, state: dict[str, Any]) -> dict[str, Any]:
    from app.services.export_service import estimate_pages, render_markdown

    selected_id = state.get("selected_candidate")
    content = render_markdown(
        job_title=task.job.title if task.job else "目标岗位",
        candidates=task.candidates,
        resume=task.resume,
        selected_candidate_id=str(selected_id) if selected_id else None,
    )
    pages = estimate_pages(content)
    return {
        **state,
        "pending_module": "layout",
        "layout_decision": {
            "estimated_page_count": pages,
            "requires_compression": pages > 1,
            "status": "awaiting_layout_review",
        },
    }


def _advance_module_review(
    task: TaskRecord, state: dict[str, Any]
) -> tuple[dict[str, Any], str, str]:
    assessment = state.get("module_assessment")
    if not isinstance(assessment, dict):
        assessment = _module_assessment(task)
    decisions = state.get("module_decisions")
    if not isinstance(decisions, dict):
        decisions = {}
    pending = _next_module(assessment, decisions)
    if pending:
        return (
            {
                **state,
                "module_assessment": assessment,
                "module_decisions": decisions,
                "pending_module": pending,
            },
            f"{pending}_review",
            f"请确认是否优化{_MODULE_LABELS[pending]}",
        )
    return (
        _layout_review_state(
            task,
            {
                **state,
                "module_assessment": assessment,
                "module_decisions": decisions,
            },
        ),
        "layout_review",
        "内容模块已逐项处理，请检查一页排版方案",
    )


class WorkflowEngine:
    def __init__(
        self,
        store: InMemoryStore,
        db_mirror: Any | None = None,
        *,
        model_gateway: Any | None = None,
        strict_model_gate: bool = False,
        task_timeout_seconds: int = 300,
        worker_lease_seconds: int = 60,
        worker_heartbeat_seconds: float = 15,
        worker_takeover_grace_seconds: int = 30,
    ) -> None:
        self.store = store
        self.db_mirror = db_mirror
        self.model_gateway = model_gateway
        self.strict_model_gate = strict_model_gate
        self.task_timeout_seconds = max(30, int(task_timeout_seconds))
        self.worker_lease_seconds = max(1, int(worker_lease_seconds))
        self.worker_heartbeat_seconds = max(0.001, float(worker_heartbeat_seconds))
        self.worker_takeover_grace_seconds = max(0, int(worker_takeover_grace_seconds))
        if self.worker_heartbeat_seconds >= self.worker_lease_seconds:
            self.worker_heartbeat_seconds = max(0.001, self.worker_lease_seconds / 2)
        self.graph = build_resume_graph(gateway=model_gateway)
        self._running: dict[str, asyncio.Task[Any]] = {}
        self.worker_id = f"worker-{uuid.uuid4().hex[:12]}"
        # The first deployment is single-user/single-worker.  A local lock
        # preserves FIFO-like execution even when two browser tabs submit at
        # the same time; PostgreSQL leases still protect multi-process starts.
        self._execution_lock = asyncio.Lock()

    async def _lease_heartbeat(
        self,
        lease_id: str,
        stop: asyncio.Event,
        lost: asyncio.Event,
    ) -> None:
        """Renew an owned lease until the node finishes or ownership is lost."""

        assert self.db_mirror is not None
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.worker_heartbeat_seconds)
                return
            except asyncio.TimeoutError:
                pass
            try:
                renewed = await asyncio.to_thread(
                    self.db_mirror.renew_lease,
                    lease_id,
                    self.worker_id,
                    lease_seconds=self.worker_lease_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                renewed = False
            if not renewed:
                lost.set()
                return

    @staticmethod
    async def _run_with_lease_guard(call: Any, lost: asyncio.Event | None) -> Any:
        computation = asyncio.create_task(asyncio.to_thread(call))
        if lost is None:
            try:
                return await asyncio.shield(computation)
            except asyncio.CancelledError:
                # ``to_thread`` cannot stop a synchronous model/file call once
                # it has started.  Keep the single-worker lock occupied until
                # that call really returns, then propagate cancellation so its
                # result is discarded by the caller.
                while not computation.done():
                    try:
                        await asyncio.shield(computation)
                    except asyncio.CancelledError:
                        continue
                    except BaseException:
                        break
                raise
        lost_waiter = asyncio.create_task(lost.wait())
        try:
            done, _ = await asyncio.wait(
                {computation, lost_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            lost_waiter.cancel()
            try:
                await lost_waiter
            except asyncio.CancelledError:
                pass
            while not computation.done():
                try:
                    await asyncio.shield(computation)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            raise
        if lost_waiter in done and lost.is_set():
            computation.cancel()
            try:
                await computation
            except asyncio.CancelledError:
                pass
            raise _WorkerLeaseLost("worker lease heartbeat failed")
        lost_waiter.cancel()
        try:
            await lost_waiter
        except asyncio.CancelledError:
            pass
        return await computation

    def _pause_after_lease_loss(
        self,
        task_id: str,
        *,
        expected_version: int,
    ) -> TaskRecord:
        latest = self.store.get_task(task_id)
        if latest is None:
            raise ValueError("task not found after worker lease loss")
        if latest.status in {TaskStatus.CANCELLED, TaskStatus.DELETED, TaskStatus.COMPLETED}:
            return latest
        if latest.checkpoint_version != expected_version:
            return latest
        try:
            paused = self.store.checkpoint(
                task_id,
                expected_version,
                state={**latest.state, "blocked_reason": "worker_lease_lost"},
                status=TaskStatus.PAUSED,
                blocked_reason="worker_lease_lost",
                current_node="worker_lease",
                messages=[
                    *latest.messages,
                    "worker 租约续期失败，已丢弃当前节点结果；请确认数据库状态后重试",
                ],
            )
        except VersionConflict:
            return self.store.get_task(task_id) or latest
        self.store.append_event(
            task_id,
            {
                "type": "blocked",
                "status": "paused",
                "blocked_reason": "worker_lease_lost",
            },
        )
        # Losing ownership explicitly forbids this worker from writing the
        # checkpoint/event to the shared database.
        return paused

    def initial_state(self, task: TaskRecord) -> GraphState:
        assert task.job is not None
        profile = self.store.get_profile(str(task.state.get("chat_profile_id", "")))
        embedding_profile = self.store.get_profile(str(task.state.get("embedding_profile_id", "")))
        chat_snapshot = task.state.get("chat_profile_snapshot")
        embedding_snapshot = task.state.get("embedding_profile_snapshot")
        chat_payload = (
            dict(chat_snapshot)
            if isinstance(chat_snapshot, dict) and chat_snapshot
            else (profile.model_dump(mode="json") if profile else {})
        )
        embedding_payload = (
            dict(embedding_snapshot)
            if isinstance(embedding_snapshot, dict) and embedding_snapshot
            else (embedding_profile.model_dump(mode="json") if embedding_profile else {})
        )
        dimension_weights = task.state.get("dimension_weights")
        component_weights = task.state.get("component_weights")
        return {
            # Preserve immutable task choices (output mode, template snapshot,
            # base facts and credential/profile bindings).  Reconstructing a
            # small subset here previously changed project-only tasks back to
            # resume_edit after the first graph checkpoint.
            **task.state,
            "task_id": task.task_id,
            "thread_id": task.thread_id,
            "job": task.job.model_dump(mode="json"),
            "resume": task.resume.model_dump(mode="json") if task.resume else None,
            "candidate_count": int(task.state.get("candidate_count", 3)),
            "embedding_mode": task.state.get("embedding_mode", "tfidf_fallback"),
            "last_feedback": str(task.state.get("last_feedback", "")),
            "generation_branch_id": str(task.state.get("generation_branch_id", "branch-1")),
            # New tasks pin immutable profile snapshots.  ID lookups remain as
            # a compatibility path for tasks created before snapshot pinning.
            "chat_profile": chat_payload,
            "embedding_profile": embedding_payload,
            "chat_profile_snapshot": chat_payload,
            "embedding_profile_snapshot": embedding_payload,
            "credential_handle_id": task.state.get("credential_handle_id"),
            "chat_credential_handle_id": task.state.get("chat_credential_handle_id"),
            "embedding_credential_handle_id": task.state.get("embedding_credential_handle_id"),
            "strict_model_gate": self.strict_model_gate,
            "base_facts": task.state.get("base_facts", {}),
            "retry_step": str(task.state.get("retry_step", "")),
            "dimension_weights": (
                dict(dimension_weights) if isinstance(dimension_weights, dict) else {}
            ),
            "component_weights": (
                dict(component_weights) if isinstance(component_weights, dict) else {}
            ),
            "scoring_config_version": str(
                task.state.get("scoring_config_version", "scoring-v1")
            ),
            "messages": [],
        }

    async def run(self, task_id: str) -> TaskRecord:
        async with self._execution_lock:
            try:
                return await asyncio.wait_for(self._run_once(task_id), timeout=self.task_timeout_seconds)
            except asyncio.TimeoutError:
                current = self.store.get_task(task_id)
                if current is None:
                    raise ValueError("task not found after timeout")
                if current.status in {TaskStatus.CANCELLED, TaskStatus.DELETED}:
                    # A user cancellation won the race with timeout handling.
                    return current
                # A timeout is a state transition and therefore consumes a
                # checkpoint version.  Using update_task here would let an old
                # browser snapshot submit over the paused timeout state.
                try:
                    task = self.store.checkpoint(
                        task_id,
                        current.checkpoint_version,
                        state={**current.state, "blocked_reason": "task_timeout"},
                        status=TaskStatus.PAUSED,
                        blocked_reason="task_timeout",
                        current_node="timeout",
                        messages=[*current.messages, "任务超过总时限，已暂停；可重试当前步骤"],
                    )
                except VersionConflict:
                    # Another worker may have committed a newer checkpoint while
                    # cancellation propagated.  Never overwrite that state.
                    task = self.store.get_task(task_id)
                    if task is None:
                        raise ValueError("task not found after timeout")
                    return task
                self.store.append_event(task_id, {"type": "blocked", "status": "paused", "blocked_reason": "task_timeout"})
                if self.db_mirror:
                    self.db_mirror.persist_checkpoint(task)
                return task

    async def _run_once(self, task_id: str) -> TaskRecord:
        task = self.store.get_task(task_id)
        if not task or not task.job:
            raise ValueError("task or job not found")
        current_task = self.store.get_task(task_id)
        if current_task and current_task.status in {TaskStatus.CANCELLED, TaskStatus.DELETED, TaskStatus.COMPLETED}:
            return current_task
        if current_task is not None:
            task = current_task
        execution_version = int(task.checkpoint_version)
        lease_id: str | None = None
        lease_stop = asyncio.Event()
        lease_lost = asyncio.Event()
        heartbeat_task: asyncio.Task[None] | None = None
        lease_finished = False

        async def stop_heartbeat() -> None:
            lease_stop.set()
            if heartbeat_task is None or heartbeat_task.done():
                if heartbeat_task is not None:
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
                return
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        async def finish_lease(*, release: bool = True) -> None:
            nonlocal lease_finished
            if lease_finished:
                return
            lease_finished = True
            await stop_heartbeat()
            if (
                release
                and lease_id
                and self.db_mirror
                and not lease_lost.is_set()
            ):
                await asyncio.to_thread(
                    self.db_mirror.release_lease,
                    lease_id,
                    self.worker_id,
                )

        if self.db_mirror and getattr(self.db_mirror, "available", False) and hasattr(self.db_mirror, "acquire_lease"):
            lease_id = self.db_mirror.acquire_lease(
                task_id,
                self.worker_id,
                lease_seconds=self.worker_lease_seconds,
                takeover_grace_seconds=self.worker_takeover_grace_seconds,
            )
            if lease_id is None:
                latest = self.store.get_task(task_id)
                if latest is None:
                    raise ValueError("task not found while acquiring worker lease")
                try:
                    paused = self.store.checkpoint(
                        task_id,
                        latest.checkpoint_version,
                        state={**latest.state, "blocked_reason": "worker_lease_busy"},
                        status=TaskStatus.PAUSED,
                        blocked_reason="worker_lease_busy",
                        current_node="worker_lease",
                        messages=[*latest.messages, "任务正在由其他 worker 执行，请稍后重试"],
                    )
                except VersionConflict:
                    return self.store.get_task(task_id) or latest
                self.store.append_event(task_id, {"type": "blocked", "status": "paused", "blocked_reason": "worker_lease_busy"})
                if self.db_mirror:
                    self.db_mirror.persist_checkpoint(paused)
                return paused
            if not hasattr(self.db_mirror, "renew_lease"):
                lease_lost.set()
            else:
                heartbeat_task = asyncio.create_task(
                    self._lease_heartbeat(lease_id, lease_stop, lease_lost)
                )
        # A feedback retry starts at candidate generation and must retain the
        # existing match result; a brand-new task starts at match.
        retrying = bool(
            current_task
            and current_task.match is not None
            and (
                bool(current_task.state.get("last_feedback"))
                or str(current_task.state.get("retry_step", "")) == "candidate_generation"
                or current_task.current_node in {"candidate_generation", "feedback_review"}
            )
        )
        start_node = "candidate_generation" if retrying else "match"
        try:
            self.store.update_task(task_id, status=TaskStatus.RUNNING, current_node=start_node)
            self.store.append_event(task_id, {"type": "status", "status": "running", "node": start_node})
            if self.db_mirror:
                self.db_mirror.persist_event(task_id, {"type": "status", "status": "running", "node": start_node})
            state = self.initial_state(task)
        except BaseException:
            # A failure before the main node try/except (including cancellation
            # during hydration) must not strand a database worker lease.
            await finish_lease(release=not lease_lost.is_set())
            raise
        try:
            def execute_node() -> GraphState:
                if self.graph is not None and not retrying:
                    # LangGraph orders the nodes; durable business writes happen
                    # only after the lease is revalidated below.
                    return self.graph.invoke(
                        state,
                        {"configurable": {"thread_id": task.thread_id}},
                    )
                if retrying:
                    return _node_candidates(
                        {
                            **state,
                            "match": task.match.model_dump(mode="json")
                            if task.match
                            else {},
                        },
                        gateway=self.model_gateway,
                    )
                return _node_candidates(
                    _node_match(state),
                    gateway=self.model_gateway,
                )

            try:
                result = await self._run_with_lease_guard(
                    execute_node,
                    lease_lost if lease_id else None,
                )
            except Exception:
                await stop_heartbeat()
                if lease_lost.is_set():
                    raise _WorkerLeaseLost("worker lease heartbeat failed")
                if lease_id and self.db_mirror:
                    renewed = await asyncio.to_thread(
                        self.db_mirror.renew_lease,
                        lease_id,
                        self.worker_id,
                        lease_seconds=self.worker_lease_seconds,
                    )
                    if not renewed:
                        lease_lost.set()
                        raise _WorkerLeaseLost("worker lease verification failed")
                raise
            await stop_heartbeat()
            if lease_lost.is_set():
                raise _WorkerLeaseLost("worker lease heartbeat failed")
            if lease_id and self.db_mirror:
                renewed = await asyncio.to_thread(
                    self.db_mirror.renew_lease,
                    lease_id,
                    self.worker_id,
                    lease_seconds=self.worker_lease_seconds,
                )
                if not renewed:
                    lease_lost.set()
                    raise _WorkerLeaseLost("worker lease verification failed")
            current = self.store.get_task(task_id)
            if current is None:
                raise ValueError("task disappeared while workflow was running")
            if current.status in {TaskStatus.CANCELLED, TaskStatus.DELETED}:
                # Never let a late model result resurrect a cancelled/deleted
                # task.  A user action that advanced the checkpoint wins.
                await finish_lease()
                return current
            if current.checkpoint_version != execution_version:
                # Another tab/worker committed a newer action while this node
                # was running.  Do not overwrite its state with a stale result.
                self.store.append_event(task_id, {"type": "stale_result_discarded", "version": current.checkpoint_version})
                await finish_lease()
                return current
            expected = current.checkpoint_version
            result_node = str(result.get("current_node") or "candidate_review")
            self.store.checkpoint(
                task_id,
                expected,
                state=result,
                status=TaskStatus.WAITING_USER,
                current_node=result_node,
                candidates=[CandidateProject.model_validate(item) for item in result.get("candidates", [])],
                match=__import__("app.core.schemas", fromlist=["MatchResult"]).MatchResult.model_validate(result["match"]),
                messages=result.get("messages", []),
            )
            persisted = self.store.get_task(task_id)
            if persisted and self.db_mirror:
                self.db_mirror.persist_checkpoint(persisted)
            self.store.append_event(task_id, {"type": "interrupt", "node": result_node})
            if self.db_mirror:
                self.db_mirror.persist_event(task_id, {"type": "interrupt", "node": result_node})
        except _WorkerLeaseLost:
            paused = self._pause_after_lease_loss(
                task_id,
                expected_version=execution_version,
            )
            await finish_lease(release=False)
            return paused
        except asyncio.CancelledError:
            await finish_lease(release=not lease_lost.is_set())
            raise
        except Exception as exc:
            raw_code = getattr(exc, "code", None)
            code = str(raw_code or "workflow_failed")
            # ContextBudgetBlocked carries the more specific snapshot reason;
            # retain it for the UI when available instead of collapsing every
            # case into a generic error.
            snapshot = getattr(exc, "snapshot", None)
            specific_reason = getattr(snapshot, "blocked_reason", None)
            if code == "context_budget_blocked" and specific_reason:
                code = str(specific_reason)
            blocked_codes = {
                "context_budget_blocked",
                "context_window_unknown",
                "context_exceeds_budget",
                "fixed_context_exceeds_budget",
                "usable_input_non_positive",
                "required_item_missing",
                "required_item_exceeds_budget",
                "token_estimation_failed",
                "credential_missing",
                "auth_failed",
                "needs_external_model_consent",
                "invalid_context",
                "duplicate_context_id",
                "reindex_required",
                "embedding_profile_missing",
            }
            blocked = code in blocked_codes or any(
                code.startswith(prefix)
                for prefix in (
                    "required_item_missing:",
                    "required_item_exceeds_budget:",
                )
            )
            status = TaskStatus.PAUSED if blocked else TaskStatus.FAILED
            failed_node = str(getattr(exc, "workflow_node", "") or start_node)
            latest = self.store.get_task(task_id)
            if latest is None:
                await finish_lease(release=not lease_lost.is_set())
                raise
            if latest.status in {TaskStatus.CANCELLED, TaskStatus.DELETED} or latest.checkpoint_version != execution_version:
                # A concurrent user action already advanced the graph.  Keep its
                # state and only release this worker's lease.
                await finish_lease(release=not lease_lost.is_set())
                return latest
            try:
                updated = self.store.checkpoint(
                    task_id,
                    latest.checkpoint_version,
                    state={**latest.state, "blocked_reason": code if blocked else None},
                    status=status,
                    blocked_reason=code if blocked else None,
                    current_node=failed_node,
                    messages=[*latest.messages, str(exc)],
                )
            except VersionConflict:
                current_after_conflict = self.store.get_task(task_id)
                if current_after_conflict is None:
                    await finish_lease(release=not lease_lost.is_set())
                    raise
                await finish_lease(release=not lease_lost.is_set())
                return current_after_conflict
            self.store.append_event(task_id, {"type": "blocked" if blocked else "error", "error_code": code, "error": str(exc), "status": status.value, "node": failed_node})
            if self.db_mirror:
                self.db_mirror.persist_checkpoint(updated)
                self.db_mirror.persist_event(task_id, {"type": "blocked" if blocked else "error", "error_code": code, "message": str(exc), "status": status.value, "node": failed_node})
        await finish_lease(release=not lease_lost.is_set())
        return self.store.get_task(task_id)  # type: ignore[return-value]

    def launch(self, task_id: str) -> None:
        existing = self._running.get(task_id)
        if existing and not existing.done():
            return
        running = asyncio.create_task(self.run(task_id))
        self._running[task_id] = running

        def _forget(done: asyncio.Task[Any]) -> None:
            # Do not remove a newer retry task that replaced this completed one.
            if self._running.get(task_id) is done:
                self._running.pop(task_id, None)

        running.add_done_callback(_forget)

    def _persist_user_transition(
        self,
        result: TaskRecord,
        *,
        create_snapshot: bool = False,
        confirmations: list[dict[str, Any]] | None = None,
    ) -> TaskRecord:
        if not self.db_mirror:
            return result
        self.db_mirror.persist_checkpoint(result)
        if create_snapshot and hasattr(self.db_mirror, "persist_resume_snapshot"):
            self.db_mirror.persist_resume_snapshot(result)
        if confirmations and hasattr(self.db_mirror, "persist_confirmations"):
            self.db_mirror.persist_confirmations(result.task_id, confirmations)
        return result

    @staticmethod
    def _preview_state(task: TaskRecord, state: dict[str, Any], *, compression: dict[str, Any]) -> dict[str, Any]:
        previous = list(state.get("previous_snapshot_ids", []))
        current_snapshot = state.get("resume_snapshot_id")
        if current_snapshot and current_snapshot not in previous:
            previous.append(current_snapshot)
        return {
            **state,
            "pending_module": "preview",
            "previous_snapshot_ids": previous,
            "resume_snapshot_id": f"snapshot-resume-{task.task_id}-{uuid.uuid4().hex[:8]}",
            "compression_decision": compression,
            "preview_status": "awaiting_user_confirmation",
        }

    def resume(self, task_id: str, *, expected_version: int, action: str, payload: dict[str, Any]) -> TaskRecord:
        task = self.store.get_task(task_id)
        if not task:
            raise KeyError(task_id)
        if task.checkpoint_version != expected_version:
            raise VersionConflict(expected_version, task.checkpoint_version)
        candidates = list(task.candidates)
        if action == "cancel":
            if task.status in {TaskStatus.COMPLETED, TaskStatus.DELETED}:
                raise ValueError("completed or deleted task cannot be cancelled; create a new task for changes")
            result = self.store.checkpoint(task_id, expected_version, state=task.state, status=TaskStatus.CANCELLED, current_node="cancelled")
            if self.db_mirror:
                self.db_mirror.persist_checkpoint(result)
                if result.status == TaskStatus.COMPLETED:
                    self.db_mirror.persist_resume_snapshot(result)
            return result
        # Every interrupt exposes a small, explicit action set.  Rejecting an
        # action that belongs to another node prevents a stale browser tab from
        # smuggling a valid payload into the wrong graph state.
        if task.status == TaskStatus.WAITING_USER:
            allowed_actions = {
                "candidate_review": {"select", "approve", "confirm", "edit", "reject", "revise", "retry", "skip"},
                # Selecting a candidate is deliberately separate from confirming
                # its fields.  A selected draft may be inspected/edited, but it
                # cannot become exportable until the confirmation endpoint sends
                # hashes for every generated field.
                "candidate_confirmation": {"confirm", "edit", "revise", "retry", "select", "reject"},
                "feedback_review": {"revise", "retry"},
                "project_application": {"approve", "confirm", "select", "skip", "revise", "retry", "reject"},
                "summary_review": {"approve", "revise", "skip"},
                "skills_review": {"approve", "revise", "skip"},
                "evaluation_review": {"approve", "revise", "skip"},
                "summary_confirmation": {"confirm", "approve", "revise", "reject", "skip"},
                "skills_confirmation": {"confirm", "approve", "revise", "reject", "skip"},
                "evaluation_confirmation": {"confirm", "approve", "revise", "reject", "skip"},
                "layout_review": {"approve", "select", "skip", "retry"},
                "compression_review": {"approve", "confirm", "skip", "revise"},
                "preview_review": {"approve", "confirm", "revise", "retry", "reject"},
            }.get(task.current_node or "candidate_review", set())
            if action not in allowed_actions:
                raise ValueError(f"action {action!r} is not allowed at node {task.current_node or 'candidate_review'}")
        clarification_resume = (
            task.status == TaskStatus.PAUSED
            and task.blocked_reason == "feedback_clarification_required"
            and action in {"revise", "retry"}
        )
        if (
            task.status == TaskStatus.PAUSED
            and task.blocked_reason == "feedback_clarification_required"
            and not str(payload.get("feedback", "")).strip()
        ):
            raise ValueError("请补充明确反馈后再重试")
        if (
            task.status == TaskStatus.PAUSED
            and task.blocked_reason == "feedback_clarification_required"
            and action not in {"revise", "retry"}
        ):
            raise ValueError("反馈澄清节点只允许提交 revise/retry 反馈")
        if task.status != TaskStatus.WAITING_USER and not clarification_resume:
            if action in {"resume", "retry"} and task.status in {TaskStatus.PAUSED, TaskStatus.FAILED}:
                # A user-triggered retry clears a temporary gate while retaining
                # the same thread/checkpoint lineage.
                retry_feedback = str(payload.get("feedback", "")).strip() if "feedback" in payload else str(task.state.get("last_feedback", ""))
                retry_node = "candidate_generation" if task.match is not None else "match"
                credential_refresh = {
                    key: payload[key]
                    for key in (
                        "credential_handle_id",
                        "chat_credential_handle_id",
                        "embedding_credential_handle_id",
                    )
                    if payload.get(key)
                }
                result = self.store.checkpoint(
                    task_id,
                    expected_version,
                    state={
                        **task.state,
                        **credential_refresh,
                        "last_feedback": retry_feedback,
                        "retry_step": retry_node,
                        # A user-triggered retry is a new draft branch even when
                        # the input text is unchanged; historical attempts stay
                        # comparable and auditable.
                        "generation_branch_id": f"branch-{uuid.uuid4().hex[:10]}",
                    },
                    status=TaskStatus.RUNNING,
                    blocked_reason=None,
                    current_node=retry_node,
                    messages=[*task.messages, "已接受用户重试，重新执行当前步骤"],
                )
                if self.db_mirror:
                    self.db_mirror.persist_checkpoint(result)
                self.launch(task_id)
                return result
            raise ValueError("task is not waiting for a user action")
        node = task.current_node or "candidate_review"
        if node in {"candidate_review", "candidate_confirmation"} and action == "edit":
            candidate_id = str(
                payload.get("candidate_id") or payload.get("candidate_slot_id") or ""
            ).strip()
            selected_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.candidate_id == candidate_id
                    or candidate.candidate_slot_id == candidate_id
                ),
                None,
            )
            if selected_candidate is None:
                raise ValueError("candidate_id is required and must belong to this task")
            if selected_candidate.status in {"duplicate", "failed", "confirmed"}:
                raise ValueError(f"{selected_candidate.status} candidate cannot be edited")
            fields = payload.get("fields")
            old_hashes = payload.get("old_value_hashes")
            if not isinstance(fields, dict) or not fields:
                raise ValueError("fields is required for candidate edit")
            if not isinstance(old_hashes, dict):
                raise ValueError("old_value_hashes is required for candidate edit")

            editable = {
                "title",
                "period",
                "introduction",
                "tech_stack",
                "solutions",
                "results",
            }
            unknown = sorted(set(fields) - editable)
            if unknown:
                raise ValueError(f"candidate fields are not editable: {','.join(unknown)}")
            update: dict[str, Any] = {}
            field_patches: list[dict[str, str]] = []
            for field_name, incoming in fields.items():
                old_value = getattr(selected_candidate, field_name)
                supplied_hash = str(old_hashes.get(field_name) or "")
                expected_hash = _value_hash(old_value)
                if supplied_hash != expected_hash:
                    raise ValueError(f"candidate old value hash mismatch: {field_name}")
                if field_name in {"tech_stack", "solutions", "results"}:
                    if not isinstance(incoming, list):
                        raise ValueError(f"{field_name} must be a list")
                    normalized: Any = [
                        str(value).strip()
                        for value in incoming
                        if str(value).strip()
                    ]
                else:
                    normalized = str(incoming or "").strip()
                    if field_name == "period" and not normalized:
                        normalized = "[待补充]"
                if field_name in {"title", "introduction"} and not normalized:
                    raise ValueError(f"{field_name} cannot be empty")
                update[field_name] = normalized
                field_patches.append(
                    {
                        "field": field_name,
                        "old_value_hash": expected_hash,
                        "new_value_hash": _value_hash(normalized),
                    }
                )

            replacement = CandidateProject.model_validate(
                {
                    **selected_candidate.model_dump(mode="json"),
                    **update,
                    "status": "draft",
                    "needs_verification": True,
                }
            )
            candidates = [
                replacement
                if candidate.candidate_id == selected_candidate.candidate_id
                else candidate
                for candidate in candidates
            ]
            edits = list(task.state.get("candidate_edits", []))
            edits.append(
                {
                    "edit_id": f"candidate-edit-{uuid.uuid4().hex[:12]}",
                    "candidate_id": replacement.candidate_id,
                    "checkpoint_base": expected_version,
                    "patches": field_patches,
                }
            )
            next_state = {
                **task.state,
                "candidate_edits": edits,
                # Any earlier confirmations were hashes of the pre-edit draft.
                # They must never promote the newly edited values implicitly.
                "confirmations": [],
            }
            result = self.store.checkpoint(
                task_id,
                expected_version,
                state=next_state,
                status=TaskStatus.WAITING_USER,
                current_node=node,
                candidates=candidates,
                messages=[
                    *task.messages,
                    f"已保存候选项目 {replacement.candidate_slot_id} 的编辑；请重新核实字段后继续",
                ],
            )
            return self._persist_user_transition(result)
        if node == "project_application" and action in {"approve", "confirm", "select", "skip"}:
            selected_id = str(task.state.get("selected_candidate") or "")
            selected_candidate = next(
                (candidate for candidate in candidates if candidate.candidate_id == selected_id),
                None,
            )
            if selected_candidate is None or selected_candidate.status != "confirmed":
                raise ValueError("a field-confirmed candidate is required before project application")
            recommendation = task.state.get("project_decision")
            if not isinstance(recommendation, dict):
                recommendation = _project_recommendation(task)
            project_action = "copy_only" if action == "skip" else str(
                payload.get("project_action")
                or payload.get("mode")
                or recommendation.get("recommended_action")
                or "add"
            )
            if project_action not in {"add", "replace", "copy_only"}:
                raise ValueError("project_action must be add, replace, or copy_only")
            target_id = payload.get("target_project_id") or recommendation.get("target_project_id")
            target = next(
                (section for section in _project_sections(task) if section.section_id == target_id),
                None,
            )
            if project_action == "replace" and target is None:
                raise ValueError("target_project_id must identify an existing project section")
            final_product = str(
                payload.get("final_product")
                or ("project_only" if project_action == "copy_only" else "full_resume")
            )
            if final_product not in {"project_only", "project_patch_only", "full_resume"}:
                raise ValueError("final_product must be project_only, project_patch_only, or full_resume")
            patch = None
            if project_action != "copy_only":
                old_value = target.content if target is not None else ""
                patch = {
                    "patch_id": f"patch-{uuid.uuid4().hex[:12]}",
                    "target_module": "projects",
                    "target_project_id": target.section_id if target is not None else "projects:new",
                    "operation": project_action,
                    "old_value_hash": (
                        target.value_hash or _value_hash(old_value)
                        if target is not None
                        else _value_hash("")
                    ),
                    "new_value": selected_candidate.model_dump(mode="json"),
                    "reason": str(
                        payload.get("reason")
                        or recommendation.get("reason")
                        or "根据目标岗位应用已确认候选项目"
                    ),
                    "status": "approved_pending_apply",
                }
            next_state = {
                **task.state,
                "project_decision": {
                    **recommendation,
                    "selected_action": project_action,
                    "target_project_id": target.section_id if target is not None else None,
                    "status": "approved",
                },
                "project_patch": patch,
                "output_mode": "project_only" if project_action == "copy_only" else "resume_edit",
                "final_product": final_product,
            }
            if project_action == "copy_only":
                next_state = _layout_review_state(task, next_state)
                next_node = "layout_review"
                message = "已保留可复制项目内容；请继续确认预览与输出"
            elif next_state["final_product"] == "project_patch_only":
                next_state = _layout_review_state(task, next_state)
                next_node = "layout_review"
                message = "已生成局部项目补丁并保留其他简历内容；请继续确认预览"
            else:
                next_state, next_node, message = _advance_module_review(task, next_state)
            result = self.store.checkpoint(
                task_id,
                expected_version,
                state=next_state,
                status=TaskStatus.WAITING_USER,
                current_node=next_node,
                candidates=candidates,
                messages=[*task.messages, message],
            )
            return self._persist_user_transition(result)
        if node in {f"{module}_review" for module in _MODULE_SEQUENCE}:
            module = node.removesuffix("_review")
            decisions = dict(task.state.get("module_decisions", {}))
            if action == "skip":
                decisions[module] = {
                    "decision": "keep",
                    "reason": str(payload.get("reason") or "用户选择保留原内容"),
                }
                next_state, next_node, message = _advance_module_review(
                    task,
                    {**task.state, "module_decisions": decisions},
                )
            else:
                feedback = str(payload.get("feedback") or "").strip()
                if action == "revise" and not feedback and not payload.get("new_value"):
                    raise ValueError("feedback or new_value is required for revise")
                draft = _module_draft(
                    task,
                    module,
                    feedback=feedback,
                    new_value=payload.get("new_value"),
                )
                drafts = dict(task.state.get("module_drafts", {}))
                drafts[module] = draft
                next_state = {
                    **task.state,
                    "pending_module": module,
                    "module_drafts": drafts,
                    "module_decisions": decisions,
                }
                next_node = f"{module}_confirmation"
                message = f"已生成{_MODULE_LABELS[module]}字段补丁，确认哈希后才会采用"
            result = self.store.checkpoint(
                task_id,
                expected_version,
                state=next_state,
                status=TaskStatus.WAITING_USER,
                current_node=next_node,
                messages=[*task.messages, message],
            )
            return self._persist_user_transition(result)
        if node in {f"{module}_confirmation" for module in _MODULE_SEQUENCE}:
            module = node.removesuffix("_confirmation")
            drafts = dict(task.state.get("module_drafts", {}))
            draft = drafts.get(module)
            if not isinstance(draft, dict):
                raise ValueError(f"missing {module} draft")
            decisions = dict(task.state.get("module_decisions", {}))
            patches = dict(task.state.get("module_patches", {}))
            if action == "revise":
                feedback = str(payload.get("feedback") or "").strip()
                if not feedback and not payload.get("new_value"):
                    raise ValueError("feedback or new_value is required for revise")
                drafts[module] = _module_draft(
                    task,
                    module,
                    feedback=feedback,
                    new_value=payload.get("new_value"),
                )
                result = self.store.checkpoint(
                    task_id,
                    expected_version,
                    state={**task.state, "module_drafts": drafts, "pending_module": module},
                    status=TaskStatus.WAITING_USER,
                    current_node=node,
                    messages=[*task.messages, f"已根据反馈更新{_MODULE_LABELS[module]}补丁，请重新确认"],
                )
                return self._persist_user_transition(result)
            if action in {"reject", "skip"}:
                draft = {**draft, "status": "rejected"}
                drafts[module] = draft
                decisions[module] = {
                    "decision": "keep",
                    "reason": str(payload.get("reason") or "用户拒绝该模块补丁"),
                }
            else:
                supplied_hash = str(payload.get("value_hash") or "")
                if supplied_hash != str(draft.get("new_value_hash") or ""):
                    raise ValueError("module confirmation value_hash mismatch")
                confirmed_patch = {
                    **draft,
                    "status": "confirmed",
                    "needs_verification": False,
                }
                drafts[module] = confirmed_patch
                patches[module] = confirmed_patch
                decisions[module] = {"decision": "optimize", "patch_id": draft.get("patch_id")}
            next_state, next_node, message = _advance_module_review(
                task,
                {
                    **task.state,
                    "module_drafts": drafts,
                    "module_patches": patches,
                    "module_decisions": decisions,
                },
            )
            result = self.store.checkpoint(
                task_id,
                expected_version,
                state=next_state,
                status=TaskStatus.WAITING_USER,
                current_node=next_node,
                messages=[*task.messages, message],
            )
            return self._persist_user_transition(result)
        if node == "layout_review":
            layout = dict(task.state.get("layout_decision", {}))
            if isinstance(payload.get("layout"), dict):
                layout["layout"] = dict(payload["layout"])
            if action == "retry":
                layout["status"] = "awaiting_layout_review"
                result = self.store.checkpoint(
                    task_id,
                    expected_version,
                    state={**task.state, "layout_decision": layout},
                    status=TaskStatus.WAITING_USER,
                    current_node="layout_review",
                    messages=[*task.messages, "已重新计算排版预览条件"],
                )
                return self._persist_user_transition(result)
            try:
                pages = int(payload.get("page_count", layout.get("estimated_page_count", 1)))
            except (TypeError, ValueError) as exc:
                raise ValueError("page_count must be an integer") from exc
            layout["estimated_page_count"] = max(1, pages)
            layout["requires_compression"] = pages > 1
            if pages > 1 and action != "skip":
                layout["status"] = "needs_compression_consent"
                result = self.store.checkpoint(
                    task_id,
                    expected_version,
                    state={
                        **task.state,
                        "pending_module": "compression",
                        "layout_decision": layout,
                    },
                    status=TaskStatus.WAITING_USER,
                    current_node="compression_review",
                    messages=[*task.messages, "预览超过一页，请确认是否先调整字体、间距和页边距"],
                )
                return self._persist_user_transition(result)
            compression = {
                "accepted": False,
                "allow_rewrite": False,
                "level": "none",
                "compliance_status": "multi_page_allowed" if pages > 1 else "one_page_estimated",
            }
            preview_state = self._preview_state(task, {**task.state, "layout_decision": layout}, compression=compression)
            result = self.store.checkpoint(
                task_id,
                expected_version,
                state=preview_state,
                status=TaskStatus.WAITING_USER,
                current_node="preview_review",
                messages=[*task.messages, "已创建不可变简历快照，请查看预览后最终确认"],
            )
            return self._persist_user_transition(result, create_snapshot=True)
        if node == "compression_review":
            if action in {"approve", "confirm"} and not bool(payload.get("accept_compression", False)):
                raise ValueError("accept_compression=true is required")
            if action == "revise" and not bool(payload.get("allow_rewrite", False)):
                raise ValueError("allow_rewrite=true is required before content rewriting")
            if action == "revise" and not task.state.get("previous_snapshot_ids"):
                raise ValueError("content rewriting is allowed only after reviewing a layout-only preview")
            compression_level = str(payload.get("compression_level") or "standard")
            if compression_level not in {"standard", "aggressive"}:
                raise ValueError("compression_level must be standard or aggressive")
            compression = {
                "accepted": action != "skip",
                "allow_rewrite": bool(payload.get("allow_rewrite", False)),
                "level": compression_level if action != "skip" else "none",
                "compliance_status": "compression_approved" if action != "skip" else "multi_page_allowed",
            }
            preview_state = self._preview_state(task, task.state, compression=compression)
            result = self.store.checkpoint(
                task_id,
                expected_version,
                state=preview_state,
                status=TaskStatus.WAITING_USER,
                current_node="preview_review",
                messages=[*task.messages, "已按选择生成新快照，请查看预览后最终确认"],
            )
            return self._persist_user_transition(result, create_snapshot=True)
        if node == "preview_review":
            if action in {"approve", "confirm"}:
                result = self.store.checkpoint(
                    task_id,
                    expected_version,
                    state={
                        **task.state,
                        "pending_module": None,
                        "preview_status": "confirmed",
                        "final_confirmation": True,
                    },
                    status=TaskStatus.COMPLETED,
                    current_node="completed",
                    messages=[*task.messages, "用户已确认最终预览，任务完成"],
                )
                return self._persist_user_transition(result, create_snapshot=True)
            if action == "retry" and payload.get("compression_level"):
                compression_level = str(payload.get("compression_level"))
                if compression_level not in {"standard", "aggressive"}:
                    raise ValueError("compression_level must be standard or aggressive")
                compression = {
                    "accepted": True,
                    "allow_rewrite": False,
                    "level": compression_level,
                    "compliance_status": "compression_approved",
                }
                result = self.store.checkpoint(
                    task_id,
                    expected_version,
                    state={
                        **task.state,
                        "compression_decision": compression,
                        "preview_status": "awaiting_user_confirmation",
                    },
                    status=TaskStatus.WAITING_USER,
                    current_node="preview_review",
                    messages=[*task.messages, "已切换为更强的无损排版压缩，请重新生成预览"],
                )
                return self._persist_user_transition(result)
            if action == "retry":
                result = self.store.checkpoint(
                    task_id,
                    expected_version,
                    state={**task.state, "preview_status": "awaiting_user_confirmation"},
                    status=TaskStatus.WAITING_USER,
                    current_node="preview_review",
                    messages=[*task.messages, "已请求重新渲染当前快照"],
                )
                return self._persist_user_transition(result)
            previous = list(task.state.get("previous_snapshot_ids", []))
            current_snapshot = task.state.get("resume_snapshot_id")
            if current_snapshot and current_snapshot not in previous:
                previous.append(current_snapshot)
            revised_state = _layout_review_state(
                task,
                {
                    **task.state,
                    "previous_snapshot_ids": previous,
                    "resume_snapshot_id": None,
                    "preview_status": "revision_requested",
                    "preview_feedback": str(payload.get("feedback") or ""),
                    "allow_rewrite": bool(payload.get("allow_rewrite", False)),
                },
            )
            result = self.store.checkpoint(
                task_id,
                expected_version,
                state=revised_state,
                status=TaskStatus.WAITING_USER,
                current_node="layout_review",
                messages=[*task.messages, "已记录预览反馈，将基于现有结果生成新版本"],
            )
            return self._persist_user_transition(result)
        if action == "reject":
            rejected_id = payload.get("candidate_id") or payload.get("candidate_slot_id")
            if not rejected_id:
                raise ValueError("candidate_id is required for reject")
            changed = False
            for candidate in candidates:
                if candidate.candidate_id == rejected_id or candidate.candidate_slot_id == rejected_id:
                    candidate.status = "failed"
                    changed = True
            if not changed:
                raise ValueError("candidate_id is required and must belong to this task")
            result = self.store.checkpoint(
                task_id,
                expected_version,
                state={**task.state, "rejected_candidate_ids": [*task.state.get("rejected_candidate_ids", []), rejected_id]},
                status=TaskStatus.WAITING_USER,
                current_node="candidate_review",
                candidates=candidates,
                messages=[*task.messages, "已拒绝该候选项目，可继续选择其他候选或提交反馈"],
            )
            if self.db_mirror:
                self.db_mirror.persist_checkpoint(result)
            return result
        if action in {"select", "approve"}:
            selected = payload.get("candidate_id") or payload.get("candidate_slot_id")
            selected_candidate = None
            for candidate in candidates:
                if candidate.candidate_id == selected or candidate.candidate_slot_id == selected:
                    if candidate.status in {"duplicate", "failed"}:
                        raise ValueError(f"{candidate.status} candidate cannot be confirmed")
                    selected_candidate = candidate
            if selected_candidate is None:
                raise ValueError("candidate_id is required and must belong to this task")
            # Selection only records which draft the user wants to inspect.  It
            # intentionally does not mark any field as confirmed.
            result = self.store.checkpoint(
                task_id,
                expected_version,
                state={
                    **task.state,
                    "selected_candidate": selected_candidate.candidate_id,
                    "selected_candidate_id": selected_candidate.candidate_id,
                },
                status=TaskStatus.WAITING_USER,
                current_node="candidate_confirmation",
                candidates=candidates,
                messages=[*task.messages, "已选择候选草稿，请逐字段确认后再写入正式简历"],
            )
            if self.db_mirror:
                self.db_mirror.persist_checkpoint(result)
            return result
        if action == "confirm":
            selected = payload.get("candidate_id") or payload.get("candidate_slot_id") or task.state.get("selected_candidate")
            selected_candidate = next(
                (item for item in candidates if item.candidate_id == selected or item.candidate_slot_id == selected),
                None,
            )
            if selected_candidate is None:
                raise ValueError("candidate_id is required and must belong to this task")
            if selected_candidate.status in {"duplicate", "failed"}:
                raise ValueError(f"{selected_candidate.status} candidate cannot be confirmed")
            confirmations = payload.get("confirmations")
            if not isinstance(confirmations, list):
                raise ValueError("confirmations is required for candidate confirmation")
            # Field-only confirmation requests are accumulated in the task
            # checkpoint.  The final promotion may therefore submit just the
            # latest field; merge by field ID and validate the complete set.
            prior_confirmations = task.state.get("confirmations", [])
            if not isinstance(prior_confirmations, list):
                prior_confirmations = []
            merged_by_field: dict[str, dict[str, Any]] = {}
            for item in [*prior_confirmations, *confirmations]:
                if isinstance(item, dict):
                    key = str(item.get("field_id") or item.get("target_id") or "")
                    if key:
                        merged_by_field[key] = item
            confirmations = list(merged_by_field.values())
            if not confirmations:
                raise ValueError("confirmations is required for candidate confirmation")
            from app.core.utils import sha256_text

            def field_hash(value: Any) -> str:
                if isinstance(value, str):
                    return sha256_text(value)
                return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))

            fields = {
                "title": selected_candidate.title,
                "period": selected_candidate.period,
                "introduction": selected_candidate.introduction,
                "tech_stack": selected_candidate.tech_stack,
                "solutions": selected_candidate.solutions,
                "results": selected_candidate.results,
            }
            supplied: dict[str, dict[str, Any]] = {}
            for item in confirmations:
                if not isinstance(item, dict):
                    raise ValueError("each confirmation must be an object")
                field_id = str(item.get("field_id") or item.get("target_id") or "")
                # Accept both the UI's fully-qualified ID and a bare field name.
                field_name = field_id.rsplit(":", 1)[-1].rsplit(".", 1)[-1]
                if field_name in fields:
                    supplied[field_name] = item
            missing = [name for name in fields if name not in supplied]
            if missing:
                raise ValueError(f"missing field confirmations: {','.join(missing)}")
            invalid: list[str] = []
            normalized_confirmations: list[dict[str, Any]] = []
            for name, value in fields.items():
                item = supplied[name]
                if not bool(item.get("confirmed", True)):
                    invalid.append(name)
                    continue
                expected_hash = field_hash(value)
                if str(item.get("value_hash") or "") != expected_hash:
                    invalid.append(name)
                    continue
                normalized_confirmations.append(
                    {
                        "module": "projects",
                        "field_id": f"{selected_candidate.candidate_id}.{name}",
                        "value_hash": expected_hash,
                        "value": value,
                        "confirmed": True,
                        "source_version_id": item.get("source_version_id"),
                        "patch_version_id": item.get("patch_version_id"),
                    }
                )
            if invalid:
                raise ValueError(f"confirmation hash mismatch or rejection: {','.join(invalid)}")
            for candidate in candidates:
                if candidate.candidate_id == selected_candidate.candidate_id:
                    candidate.status = "confirmed"
                    candidate.needs_verification = False
            requested_output = str(
                task.state.get("requested_output_mode")
                or task.state.get("output_mode")
                or ""
            )
            if requested_output == "project_only":
                result = self.store.checkpoint(
                    task_id,
                    expected_version,
                    state={
                        **task.state,
                        "selected_candidate": selected_candidate.candidate_id,
                        "selected_candidate_id": selected_candidate.candidate_id,
                        "confirmed_candidate_id": selected_candidate.candidate_id,
                        "confirmations": normalized_confirmations,
                        "pending_module": None,
                        "project_decision": {
                            "selected_action": "copy_only",
                            "status": "approved",
                        },
                        "project_patch": None,
                        "output_mode": "project_only",
                        "requested_output_mode": "project_only",
                        "final_product": "project_only",
                        "final_confirmation": True,
                    },
                    status=TaskStatus.COMPLETED,
                    current_node="completed",
                    candidates=candidates,
                    messages=[
                        *task.messages,
                        "用户已核实项目字段，可直接复制纯文本使用",
                    ],
                )
                return self._persist_user_transition(
                    result, confirmations=normalized_confirmations
                )
            project_decision = _project_recommendation(task)
            result = self.store.checkpoint(
                task_id,
                expected_version,
                state={
                    **task.state,
                    "selected_candidate": selected_candidate.candidate_id,
                    "selected_candidate_id": selected_candidate.candidate_id,
                    "confirmed_candidate_id": selected_candidate.candidate_id,
                    "confirmations": normalized_confirmations,
                    "pending_module": "projects",
                    "project_decision": project_decision,
                },
                status=TaskStatus.WAITING_USER,
                current_node="project_application",
                candidates=candidates,
                messages=[
                    *task.messages,
                    "用户已逐字段确认候选项目，请确认将其新增到简历、替换指定项目或仅复制使用",
                ],
            )
            return self._persist_user_transition(result, confirmations=normalized_confirmations)
        if action in {"revise", "retry"}:
            feedback = str(payload.get("feedback", "")).strip()
            if not feedback and action == "revise":
                raise ValueError("feedback is required for revise")
            classification = classify_feedback(feedback) if feedback else None
            profile_override = {
                key: payload[key]
                for key in (
                    "chat_profile_id",
                    "chat_profile_version",
                    "chat_profile_snapshot",
                    "credential_handle_id",
                )
                if key in payload
            }
            feedback_id = None
            if feedback and self.db_mirror and hasattr(self.db_mirror, "persist_feedback"):
                feedback_id = self.db_mirror.persist_feedback(
                    task_id,
                    feedback,
                    classification.to_dict() if classification else None,
                    target_version_id=payload.get("target_version_id"),
                )
            if classification is not None and classification.requires_clarification:
                result = self.store.checkpoint(
                    task_id,
                    expected_version,
                    state={
                        **task.state,
                        **profile_override,
                        "last_feedback": feedback,
                        "last_feedback_id": feedback_id or task.state.get("last_feedback_id"),
                        "feedback_classification": classification.to_dict(),
                        "retry_step": "candidate_generation",
                    },
                    status=TaskStatus.PAUSED,
                    blocked_reason="feedback_clarification_required",
                    current_node="feedback_review",
                    messages=[
                        *task.messages,
                        "反馈包含多个类别或含义不明确，请补充说明后再重新生成",
                    ],
                )
                if self.db_mirror:
                    self.db_mirror.persist_checkpoint(result)
                return result
            result = self.store.checkpoint(
                task_id,
                expected_version,
                state={
                    **task.state,
                    **profile_override,
                    "last_feedback": feedback,
                    "last_feedback_id": feedback_id or task.state.get("last_feedback_id"),
                    "feedback_classification": classification.to_dict() if classification else None,
                    "generation_branch_id": f"branch-{uuid.uuid4().hex[:10]}",
                    "retry_step": "candidate_generation",
                },
                status=TaskStatus.RUNNING,
                current_node="candidate_generation",
                messages=[*task.messages, "已记录反馈，准备重新生成"],
            )
            if self.db_mirror:
                self.db_mirror.persist_checkpoint(result)
            self.launch(task_id)
            return result
        if action == "skip":
            if task.resume is None:
                raise ValueError("cannot skip all candidates when no source resume exists")
            next_state = _layout_review_state(
                task,
                {
                    **task.state,
                    "pending_module": "layout",
                    "output_mode": "original_resume",
                },
            )
            result = self.store.checkpoint(
                task_id,
                expected_version,
                state=next_state,
                status=TaskStatus.WAITING_USER,
                current_node="layout_review",
                messages=[*task.messages, "已跳过候选项目，将预览原简历内容"],
            )
            return self._persist_user_transition(result)
        raise ValueError(f"unsupported action: {action}")
