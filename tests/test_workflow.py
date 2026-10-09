from __future__ import annotations

import asyncio
import json
import threading
import unittest

from app.core.model_gateway import ChatResult, GatewayError
from app.core.schemas import JobInput, ResumeDocument, ResumeSection, TaskStatus
from app.core.store import InMemoryStore, TaskRecord
from app.core.utils import stable_id
from app.services.candidate_generator import (
    ModelCandidateGenerationError,
    generate_candidates,
    generate_candidates_with_model,
)
from app.services.workflow import (
    WorkflowEngine,
    _node_candidates,
    _node_match,
    _project_recommendation,
    _value_hash,
)


def _job() -> JobInput:
    return JobInput(
        title="Python 后端工程师",
        responsibilities=["实现服务接口"],
        requirements=["具备工程落地能力"],
        skills=["Python", "FastAPI"],
    )


def _model_state(*, thread_id: str = "thread-a", strict: bool = False) -> dict:
    return {
        "task_id": "task-a",
        "thread_id": thread_id,
        "job": _job().model_dump(mode="json"),
        "resume": None,
        "candidate_count": 1,
        "embedding_mode": "tfidf_fallback",
        "match": {},
        "chat_profile": {
            "profile_id": "chat-a",
            "role": "chat",
            "provider": "ollama",
            "base_url": "http://127.0.0.1:11434",
            "model_name": "qwen-test",
            "profile_version": "v1",
            "status": "ready",
            "context_window_tokens": 4096,
            "generation_params": {"max_output_tokens": 256},
        },
        "strict_model_gate": strict,
        "generation_branch_id": "branch-1",
        "messages": [],
    }


class _ErrorGateway:
    def __init__(self, error: Exception):
        self.error = error

    def chat(self, *args, **kwargs):
        raise self.error


class _CaptureGateway:
    def __init__(self):
        self.request_keys: list[str | None] = []
        self.chat_credentials: list[str | None] = []
        self.embedding_credentials: list[str | None] = []
        self.message_batches: list[list] = []

    def chat(self, *args, **kwargs):
        self.request_keys.append(kwargs.get("request_key"))
        self.chat_credentials.append(kwargs.get("credential_handle_id"))
        self.message_batches.append(list(args[1]))
        evidence_id = stable_id("job", "responsibility", 0, "实现服务接口")
        return ChatResult(
            content=json.dumps(
                {
                    "candidates": [
                        {
                            "title": "服务平台",
                            "period": "[待补充]",
                            "introduction": "建设服务接口平台",
                            "tech_stack": ["Python"],
                            "solutions": ["实现接口"],
                            "results": ["形成闭环"],
                            "evidence_ids": [evidence_id],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            model_name="qwen-test",
            profile_version="v1",
        )

    def embed(self, profile, texts, **kwargs):
        # Orthogonal vectors keep candidates distinct while proving the fixed
        # embedding path is actually invoked.
        from app.core.model_gateway import EmbeddingResult

        self.embedding_credentials.append(kwargs.get("credential_handle_id"))
        return EmbeddingResult(
            vectors=tuple((1.0, 0.0) if index == 0 else (0.0, 1.0) for index, _ in enumerate(texts)),
            model_name="bge-test",
            profile_version="v1",
            dimension=2,
            request_key=kwargs.get("request_key"),
        )


class _MalformedGateway:
    def __init__(self, payload):
        self.payload = payload

    def chat(self, *args, **kwargs):
        return ChatResult(content=json.dumps(self.payload), model_name="qwen-test", profile_version="v1")


class _StringifiedCandidateGateway:
    def __init__(self) -> None:
        self.attempts = 0

    def chat(self, *args, **kwargs):
        self.attempts += 1
        evidence_id = stable_id("job", "responsibility", 0, "实现服务接口")
        candidate = {
            "title": "智能服务协同平台",
            "period": "[待补充]",
            "introduction": "面向服务团队建设请求接入、任务流转和质量复盘平台。",
            "tech_stack": ["Python", "FastAPI"],
            "solutions": ["使用状态机约束任务创建、处理和关闭。"],
            "results": ["[待核实] 完成端到端业务闭环验收。"],
            "evidence_ids": [evidence_id],
        }
        return ChatResult(
            content=json.dumps(
                {"candidates": [json.dumps(candidate, ensure_ascii=False)]},
                ensure_ascii=False,
            ),
            model_name="qwen-test",
            profile_version="v1",
        )


class _FormatRetryGateway(_CaptureGateway):
    def __init__(self):
        super().__init__()
        self.attempts = 0

    def chat(self, *args, **kwargs):
        self.attempts += 1
        if self.attempts == 1:
            self.request_keys.append(kwargs.get("request_key"))
            self.chat_credentials.append(kwargs.get("credential_handle_id"))
            return ChatResult(
                content="这里是候选项目，但上一轮没有输出 JSON",
                model_name="qwen-test",
                profile_version="v1",
            )
        return super().chat(*args, **kwargs)


class _SchemaRetryGateway(_CaptureGateway):
    def __init__(self):
        super().__init__()
        self.attempts = 0

    def chat(self, *args, **kwargs):
        self.attempts += 1
        if self.attempts == 1:
            self.request_keys.append(kwargs.get("request_key"))
            self.chat_credentials.append(kwargs.get("credential_handle_id"))
            return ChatResult(
                content=json.dumps(
                    {"result": {"projects": [{"project_name": "缺少项目正文"}]}},
                    ensure_ascii=False,
                ),
                model_name="qwen-test",
                profile_version="v1",
            )
        return super().chat(*args, **kwargs)


class _AlwaysSparseCandidateGateway:
    def __init__(self):
        self.attempts = 0

    def chat(self, *args, **kwargs):
        self.attempts += 1
        return ChatResult(
            content=json.dumps(
                {
                    "projects": [
                        {
                            "name": "只有标题和简介的平台",
                            "description": "缺少技术栈、解决方案和结果，不能作为完整候选项目。",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            model_name="qwen-test",
            profile_version="v1",
        )


class _IncrementalCandidateGateway:
    """Return 1, then 2, then 1 candidates to exercise bounded top-up."""

    def __init__(self) -> None:
        self.calls: list[str | None] = []
        self.round = 0

    def chat(self, *args, **kwargs):
        self.calls.append(kwargs.get("request_key"))
        projects = {
            "仓储": ("仓储履约协同平台", "仓库拣货、库存锁定和出库履约"),
            "客服": ("智能客服工单平台", "咨询接入、工单路由和服务质检"),
            "风控": ("交易风控审核平台", "规则命中、人工复核和风险处置"),
            "运维": ("设备预测运维平台", "设备采集、异常诊断和维修排程"),
        }
        batches = (("仓储",), ("客服", "风控"), ("运维",))
        labels = batches[min(self.round, len(batches) - 1)]
        self.round += 1
        evidence_id = stable_id("job", "responsibility", 0, "实现服务接口")
        return ChatResult(
            content=json.dumps(
                {
                    "candidates": [
                        {
                            "title": projects[label][0],
                            "period": "[待补充]",
                            "introduction": f"面向企业团队建设{projects[label][1]}的完整业务系统。",
                            "tech_stack": ["Python", "FastAPI", "PostgreSQL"],
                            "solutions": [
                                f"领域建模：围绕{projects[label][1]}拆分领域对象和状态流转。",
                                f"数据链路：为{label}场景设计事件记录、查询索引和异常补偿。",
                                f"服务治理：针对{label}接口加入幂等、限流和可观测性指标。",
                                f"工程交付：完成{label}链路的集成测试和容器化部署。",
                            ],
                            "results": [
                                f"[待核实] 完成{projects[label][1]}的端到端业务验收。",
                                f"[待核实] 建立{label}场景的成功率与延迟指标。",
                            ],
                            "evidence_ids": [evidence_id],
                        }
                        for label in labels
                    ]
                },
                ensure_ascii=False,
            ),
            model_name="qwen-test",
            profile_version="v1",
        )


class _BaseFactsEvidenceGateway(_CaptureGateway):
    def chat(self, *args, **kwargs):
        self.request_keys.append(kwargs.get("request_key"))
        self.chat_credentials.append(kwargs.get("credential_handle_id"))
        self.message_batches.append(list(args[1]))
        return ChatResult(
            content=json.dumps(
                {
                    "candidates": [
                        {
                            "title": "智能服务工单协同平台",
                            "period": "[待补充]",
                            "introduction": "面向客户服务团队建设工单协同平台，覆盖咨询接入、任务流转、异常恢复和质量复盘。",
                            "tech_stack": ["Python", "FastAPI", "PostgreSQL", "Docker"],
                            "solutions": [
                                "领域建模：拆分会话、工单和处理记录，并通过状态机约束任务流转。",
                                "接口服务：使用 FastAPI 与 Pydantic 建立输入校验和统一错误响应。",
                                "一致性治理：使用幂等键、事务和失败补偿避免重复写入与部分成功。",
                                "工程交付：补充接口测试、结构化日志和容器化部署以验证完整链路。",
                            ],
                            "results": [
                                "[待核实] 完成从请求接入到工单关闭的端到端业务闭环。",
                                "[待核实] 建立成功率、延迟和异常恢复率等验收指标。",
                            ],
                            "evidence_ids": ["user:base_facts"],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            model_name="qwen-test",
            profile_version="v1",
        )


class _BlockingGraph:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def invoke(self, state, _config):
        self.entered.set()
        if not self.release.wait(timeout=2):
            raise TimeoutError("test graph was not released")
        return _node_candidates(_node_match(state))


class _LeaseMirror:
    available = True

    def __init__(self, *, renew_succeeds: bool) -> None:
        self.renew_succeeds = renew_succeeds
        self.acquire_calls: list[dict] = []
        self.renew_calls: list[dict] = []
        self.release_calls: list[tuple[str, str]] = []
        self.persisted_checkpoints: list[TaskRecord] = []
        self.persisted_events: list[dict] = []
        self.renewed = threading.Event()

    def acquire_lease(self, task_id, worker_id, **kwargs):
        self.acquire_calls.append({"task_id": task_id, "worker_id": worker_id, **kwargs})
        return "lease-test"

    def renew_lease(self, lease_id, worker_id, **kwargs):
        self.renew_calls.append({"lease_id": lease_id, "worker_id": worker_id, **kwargs})
        self.renewed.set()
        return self.renew_succeeds

    def release_lease(self, lease_id, worker_id):
        self.release_calls.append((lease_id, worker_id))
        return True

    def persist_checkpoint(self, task):
        self.persisted_checkpoints.append(task)
        return True

    def persist_event(self, _task_id, event):
        self.persisted_events.append(dict(event))
        return True


class WorkflowTests(unittest.TestCase):
    def test_stringified_candidate_object_is_normalized_without_resaving_model(self):
        from app.services.matching import calculate_match

        state = _model_state(strict=True)
        gateway = _StringifiedCandidateGateway()

        candidates, audit = generate_candidates_with_model(
            _job(),
            None,
            calculate_match(_job(), None),
            profile=state["chat_profile"],
            gateway=gateway,
            count=1,
            task_id=state["task_id"],
            thread_id=state["thread_id"],
        )

        self.assertEqual(gateway.attempts, 1)
        self.assertEqual(candidates[0].title, "智能服务协同平台")
        self.assertTrue(audit["schema_normalization_used"])

    def test_candidate_generation_keeps_existing_and_tops_up_to_requested_count(self):
        from app.services.matching import calculate_match

        state = _model_state(strict=True)
        gateway = _IncrementalCandidateGateway()
        candidates, audit = generate_candidates_with_model(
            _job(),
            None,
            calculate_match(_job(), None),
            profile=state["chat_profile"],
            gateway=gateway,
            count=4,
            task_id=state["task_id"],
            thread_id=state["thread_id"],
        )

        self.assertEqual(len(candidates), 4)
        self.assertEqual(
            {item.title for item in candidates},
            {
                "仓储履约协同平台",
                "智能客服工单平台",
                "交易风控审核平台",
                "设备预测运维平台",
            },
        )
        self.assertEqual(audit["requested_candidate_count"], 4)
        self.assertEqual(audit["returned_candidate_count"], 4)
        self.assertEqual(audit["local_completion_count"], 0)
        self.assertEqual(
            [item["accepted"] for item in audit["candidate_completion_rounds"]],
            [2, 1],
        )
        self.assertEqual(len(gateway.calls), 3)

    def test_missing_candidate_evidence_defaults_only_to_included_jd_items(self):
        from app.services.matching import build_job_evidence, calculate_match

        state = _model_state(strict=True)
        state["chat_profile"]["context_window_tokens"] = 16_384
        job = _job()
        resume = ResumeDocument(
            resume_id="resume-evidence-boundary",
            filename="resume.md",
            file_type="markdown",
            file_size=100,
            sections=[
                ResumeSection(
                    section_id="project-1",
                    module="projects",
                    title="已有项目",
                    content="用户已经确认的项目事实",
                    confirmed=True,
                    evidence_ids=["resume-evidence-1"],
                )
            ],
            raw_text="用户已经确认的项目事实",
        )
        gateway = _MalformedGateway(
            {
                "projects": [
                    {
                        "name": "智能工单平台",
                        "description": "面向客服团队建设工单流转平台。",
                        "skills": ["Python", "FastAPI"],
                        "implementation": ["使用状态机约束工单流转。"],
                        "outcomes": ["[待核实] 完成主要业务链路验收。"],
                    }
                ]
            }
        )

        candidates, audit = generate_candidates_with_model(
            job,
            resume,
            calculate_match(job, resume),
            profile=state["chat_profile"],
            gateway=gateway,
            count=1,
            task_id=state["task_id"],
            thread_id=state["thread_id"],
            base_facts={"objective": "Python 后端工程师"},
        )

        expected_jd_ids = {item.evidence_id for item in build_job_evidence(job)}
        self.assertTrue(candidates[0].evidence_ids)
        self.assertEqual(set(candidates[0].evidence_ids), expected_jd_ids)
        self.assertNotIn("resume-evidence-1", candidates[0].evidence_ids)
        self.assertIn("resume:project-1", audit["context_snapshot"]["included_object_ids"])
        self.assertIn("user:base_facts", audit["context_snapshot"]["included_object_ids"])

    def test_sparse_candidate_schema_gets_only_one_bounded_repair_attempt(self):
        from app.services.matching import calculate_match

        state = _model_state(strict=True)
        job = _job()
        gateway = _AlwaysSparseCandidateGateway()

        with self.assertRaises(ModelCandidateGenerationError) as caught:
            generate_candidates_with_model(
                job,
                None,
                calculate_match(job, None),
                profile=state["chat_profile"],
                gateway=gateway,
                count=1,
                task_id=state["task_id"],
                thread_id=state["thread_id"],
            )

        self.assertEqual(caught.exception.code, "invalid_model_schema")
        self.assertEqual(gateway.attempts, 2)

    def test_four_and_five_candidate_output_budget_preserves_jd_context(self):
        from app.services.matching import calculate_match

        for count in (4, 5):
            with self.subTest(count=count):
                state = _model_state(strict=True)
                job = _job()
                candidates, audit = generate_candidates_with_model(
                    job,
                    None,
                    calculate_match(job, None),
                    profile=state["chat_profile"],
                    gateway=_CaptureGateway(),
                    count=count,
                    task_id=state["task_id"],
                    thread_id=state["thread_id"],
                )

                snapshot = audit["context_snapshot"]
                self.assertTrue(candidates)
                self.assertIn("jd:responsibility:0", snapshot["included_object_ids"])
                self.assertLessEqual(
                    audit["max_output_tokens"]
                    + snapshot["estimated_input_tokens"]
                    + snapshot["safety_margin_tokens"],
                    snapshot["context_window_tokens"],
                )

    def test_base_facts_context_id_is_canonicalized_as_valid_evidence(self):
        from app.services.matching import calculate_match

        gateway = _BaseFactsEvidenceGateway()
        state = _model_state(strict=True)
        job = _job()
        candidates, _ = generate_candidates_with_model(
            job,
            None,
            calculate_match(job, None),
            profile=state["chat_profile"],
            gateway=gateway,
            count=1,
            task_id=state["task_id"],
            thread_id=state["thread_id"],
            base_facts={"objective": "Python 后端工程师"},
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0].evidence_ids), 1)
        self.assertTrue(candidates[0].evidence_ids[0].startswith("user-facts_"))

    def test_candidate_prompt_requires_concrete_implementation_details(self):
        from app.services.matching import calculate_match

        gateway = _CaptureGateway()
        state = _model_state()
        job = _job()
        generate_candidates_with_model(
            job,
            None,
            calculate_match(job, None),
            profile=state["chat_profile"],
            gateway=gateway,
            count=1,
            task_id=state["task_id"],
            thread_id=state["thread_id"],
        )

        developer_prompt = gateway.message_batches[0][1].content
        self.assertIn("具体业务场景", developer_prompt)
        self.assertIn("实现机制或数据流", developer_prompt)
        self.assertIn("[待核实]", developer_prompt)

    def test_deterministic_fallback_builds_domain_project_not_resume_generator(self):
        from app.services.matching import calculate_match

        job = JobInput(
            title="大模型应用工程师",
            responsibilities=["建设企业知识库问答服务"],
            requirements=["熟悉 RAG 与服务工程化"],
            skills=["Python", "FastAPI", "LangGraph"],
        )
        candidate = generate_candidates(
            job,
            None,
            calculate_match(job, None),
            count=1,
        )[0]

        self.assertNotIn("简历匹配", candidate.title)
        self.assertIn("知识库", candidate.title)
        self.assertGreaterEqual(len(candidate.solutions), 4)
        self.assertTrue(all(len(item) >= 25 for item in candidate.solutions))
        self.assertTrue(all(item.startswith("[待核实]") for item in candidate.results))

    def test_candidate_generation_retries_one_invalid_json_response(self):
        gateway = _FormatRetryGateway()
        state = _node_match(_model_state(strict=True))

        result = _node_candidates(state, gateway=gateway)

        self.assertEqual(gateway.attempts, 2)
        self.assertTrue(result["model_audit"]["format_retry_used"])
        self.assertEqual(len(result["candidates"]), 1)

    def test_candidate_generation_normalizes_wrapped_chinese_project_shape(self):
        from app.services.matching import calculate_match

        state = _model_state(strict=True)
        job = _job()
        gateway = _MalformedGateway(
            {
                "data": {
                    "projects": [
                        {
                            "项目名称": "智能工单协同平台",
                            "项目周期": "[待补充]",
                            "项目简介": "面向服务团队建设工单流转和质量复盘平台。",
                            "技术栈": "Python、FastAPI、PostgreSQL、Docker",
                            "技术方案": [
                                "状态编排：使用状态机约束工单创建、分派、处理和关闭。",
                                "一致性治理：通过幂等键和事务补偿处理重复回调。",
                            ],
                            "项目成果": ["[待核实] 工单异常可追踪率达到 100%。"],
                        }
                    ]
                }
            }
        )

        candidates, audit = generate_candidates_with_model(
            job,
            None,
            calculate_match(job, None),
            profile=state["chat_profile"],
            gateway=gateway,
            count=1,
            task_id=state["task_id"],
            thread_id=state["thread_id"],
        )

        self.assertEqual(candidates[0].title, "智能工单协同平台")
        self.assertEqual(candidates[0].tech_stack[:2], ["Python", "FastAPI"])
        self.assertTrue(candidates[0].evidence_ids)
        self.assertTrue(audit["schema_normalization_used"])
        self.assertEqual(audit["defaulted_evidence_count"], 1)
        self.assertFalse(audit["format_retry_used"])

    def test_candidate_generation_repairs_valid_json_with_incomplete_schema(self):
        gateway = _SchemaRetryGateway()
        state = _node_match(_model_state(strict=True))

        result = _node_candidates(state, gateway=gateway)

        self.assertEqual(gateway.attempts, 2)
        self.assertTrue(result["model_audit"]["format_retry_used"])
        self.assertEqual(result["candidates"][0]["title"], "服务平台")

    def test_candidate_format_failure_is_labelled_as_generation_not_match(self):
        gateway = _ErrorGateway(
            ModelCandidateGenerationError("invalid_model_json", "模型没有返回合法 JSON")
        )
        state = _node_match(_model_state(strict=True))

        with self.assertRaises(Exception) as caught:
            _node_candidates(state, gateway=gateway)

        self.assertEqual(
            getattr(caught.exception, "workflow_node", None), "candidate_generation"
        )

    def test_project_recommendation_replaces_lowest_matching_project(self):
        resume = ResumeDocument(
            resume_id="resume-project-ranking",
            filename="resume.md",
            file_type="markdown",
            file_size=100,
            sections=[
                ResumeSection(
                    section_id="project-python",
                    module="projects",
                    title="Python 服务平台",
                    content="使用 Python 和 FastAPI 实现服务接口",
                ),
                ResumeSection(
                    section_id="project-unrelated",
                    module="projects",
                    title="线下活动项目",
                    content="负责线下活动组织与物料采购",
                ),
            ],
        )
        recommendation = _project_recommendation(
            TaskRecord(
                task_id="task-ranking",
                thread_id="thread-ranking",
                job=_job(),
                resume=resume,
            )
        )
        self.assertEqual(recommendation["recommended_action"], "replace")
        self.assertEqual(recommendation["target_project_id"], "project-unrelated")
        self.assertEqual(recommendation["project_rankings"][0]["project_id"], "project-python")
        self.assertEqual(recommendation["project_rankings"][0]["title"], "Python 服务平台")
        self.assertIn("FastAPI", recommendation["project_rankings"][0]["content_preview"])

    def test_match_uses_task_pinned_scoring_configuration(self):
        state = {
            "task_id": "task-scoring",
            "thread_id": "thread-scoring",
            "job": _job().model_dump(mode="json"),
            "resume": None,
            "embedding_mode": "tfidf_fallback",
            "dimension_weights": {
                "skills": 0.4,
                "responsibilities": 0.3,
                "projects": 0.2,
                "keywords": 0.1,
            },
            "component_weights": {"deterministic": 0.7, "semantic": 0.3},
            "scoring_config_version": "scoring-custom-v2",
        }
        result = _node_match(state)
        self.assertEqual(result["match"]["scoring_config_version"], "scoring-custom-v2")
        self.assertEqual(result["match"]["dimensions"]["skills"]["weight"], 0.4)

    def test_initial_state_prefers_task_pinned_model_snapshots(self):
        store = InMemoryStore()
        task = TaskRecord(
            task_id="task-profile-snapshot",
            thread_id="thread-profile-snapshot",
            job=_job(),
            state={
                "chat_profile_id": "missing-live-profile",
                "embedding_profile_id": "missing-live-embedding",
                "chat_profile_snapshot": {
                    "profile_id": "chat-pinned",
                    "status": "ready",
                    "model_name": "pinned-chat",
                },
                "embedding_profile_snapshot": {
                    "profile_id": "embedding-pinned",
                    "status": "ready",
                    "model_name": "pinned-embedding",
                },
            },
        )
        store.create_task(task)
        state = WorkflowEngine(store).initial_state(task)
        self.assertEqual(state["chat_profile"]["model_name"], "pinned-chat")
        self.assertEqual(state["embedding_profile"]["model_name"], "pinned-embedding")

    def test_non_strict_model_gate_errors_are_not_silently_downgraded(self):
        from app.services.matching import calculate_match

        match = calculate_match(_job(), None)
        state = _model_state()
        state["match"] = match.model_dump(mode="json")
        with self.assertRaises(GatewayError) as raised:
            _node_candidates(
                state,
                gateway=_ErrorGateway(GatewayError("credential_missing", "missing", requires_user=True)),
            )
        self.assertEqual(raised.exception.code, "credential_missing")

    def test_non_strict_schema_error_uses_explicit_fallback(self):
        from app.services.matching import calculate_match

        state = _model_state()
        state["match"] = calculate_match(_job(), None).model_dump(mode="json")
        result = _node_candidates(
            state,
            gateway=_ErrorGateway(RuntimeError("invalid provider payload")),
        )
        self.assertEqual(result["model_audit"]["fallback_reason"], "invalid provider payload")
        self.assertTrue(result["candidates"])

    def test_model_scalar_fields_are_strictly_typed(self):
        from app.services.candidate_generator import ModelCandidateGenerationError
        from app.services.matching import calculate_match

        state = _model_state()
        state["match"] = calculate_match(_job(), None).model_dump(mode="json")
        gateway = _MalformedGateway({"candidates": [{"title": 42, "introduction": "x"}]})
        with self.assertRaises(ModelCandidateGenerationError):
            # strict mode prevents this malformed response from being silently
            # replaced by a deterministic draft.
            _node_candidates({**state, "strict_model_gate": True}, gateway=gateway)

    def test_request_key_follows_thread_id_and_fallback_branch_ids_change(self):
        from app.services.matching import calculate_match

        gateway = _CaptureGateway()
        first = _model_state(thread_id="thread-a")
        first["match"] = calculate_match(_job(), None).model_dump(mode="json")
        _node_candidates(first, gateway=gateway)
        second = _model_state(thread_id="thread-b")
        second["match"] = first["match"]
        _node_candidates(second, gateway=gateway)
        self.assertEqual(len(gateway.request_keys), 2)
        self.assertNotEqual(gateway.request_keys[0], gateway.request_keys[1])

        candidates_a = generate_candidates(_job(), None, calculate_match(_job(), None), task_id="task-a", branch_id="branch-a", count=1)
        candidates_b = generate_candidates(_job(), None, calculate_match(_job(), None), task_id="task-a", branch_id="branch-b", count=1)
        self.assertNotEqual(candidates_a[0].candidate_id, candidates_b[0].candidate_id)

    def test_embedding_mode_uses_embedding_profile_for_deduplication(self):
        from app.services.matching import calculate_match

        gateway = _CaptureGateway()
        state = _model_state()
        state["candidate_count"] = 1
        state["embedding_mode"] = "embedding"
        state["embedding_profile"] = {
            "profile_id": "embedding-a",
            "role": "embedding",
            "provider": "ollama",
            "base_url": "http://127.0.0.1:11434",
            "model_name": "bge-test",
            "profile_version": "v1",
            "status": "ready",
            "max_input_tokens": 8192,
        }
        state["credential_handle_id"] = "chat-handle"
        state["embedding_credential_handle_id"] = "embedding-handle"
        state["match"] = calculate_match(_job(), None, embedding_mode="embedding").model_dump(mode="json")
        result = _node_candidates(state, gateway=gateway)
        self.assertEqual(result["model_audit"]["embedding_mode"], "embedding")
        self.assertEqual(gateway.chat_credentials, ["chat-handle"])
        self.assertEqual(gateway.embedding_credentials, ["embedding-handle"])

    def test_feedback_clarification_can_be_resubmitted_from_paused_state(self):
        async def scenario():
            store = InMemoryStore()
            task = TaskRecord(
                task_id="task-feedback",
                thread_id="thread-feedback",
                job=_job(),
                state={"candidate_count": 1},
            )
            store.create_task(task)
            engine = WorkflowEngine(store)
            current = await engine.run(task.task_id)
            current = engine.resume(
                task.task_id,
                expected_version=current.checkpoint_version,
                action="revise",
                payload={"feedback": "我没有做过 Redis，但请增加一条 Redis 经历"},
            )
            self.assertEqual(current.blocked_reason, "feedback_clarification_required")
            current = engine.resume(
                task.task_id,
                expected_version=current.checkpoint_version,
                action="revise",
                payload={"feedback": "请突出 Python"},
            )
            self.assertEqual(current.status.value, "running")
            self.assertEqual(current.current_node, "candidate_generation")
            await engine._running[task.task_id]
            self.assertEqual(store.get_task(task.task_id).status.value, "waiting_user")

        asyncio.run(scenario())

    def test_timeout_advances_checkpoint_and_pauses_task(self):
        async def scenario():
            store = InMemoryStore()
            task = TaskRecord(task_id="task-timeout", thread_id="thread-timeout", job=_job())
            store.create_task(task)
            engine = WorkflowEngine(store)
            engine.task_timeout_seconds = 0.01

            async def blocked(_task_id: str):
                await asyncio.sleep(1)

            engine._run_once = blocked  # type: ignore[method-assign]
            result = await engine.run(task.task_id)
            self.assertEqual(result.status.value, "paused")
            self.assertEqual(result.blocked_reason, "task_timeout")
            self.assertEqual(result.checkpoint_version, 1)

        asyncio.run(scenario())

    def test_worker_lease_heartbeat_renews_with_configured_intervals(self):
        async def scenario():
            store = InMemoryStore()
            task = TaskRecord(
                task_id="task-lease-heartbeat",
                thread_id="thread-lease-heartbeat",
                job=_job(),
                state={"candidate_count": 1},
            )
            store.create_task(task)
            mirror = _LeaseMirror(renew_succeeds=True)
            graph = _BlockingGraph()
            engine = WorkflowEngine(
                store,
                mirror,
                worker_lease_seconds=2,
                worker_heartbeat_seconds=0.01,
                worker_takeover_grace_seconds=7,
            )
            engine.graph = graph
            running = asyncio.create_task(engine._run_once(task.task_id))
            self.assertTrue(await asyncio.to_thread(graph.entered.wait, 1))
            self.assertTrue(await asyncio.to_thread(mirror.renewed.wait, 1))
            graph.release.set()
            result = await asyncio.wait_for(running, timeout=1)
            self.assertEqual(result.status, TaskStatus.WAITING_USER)
            self.assertEqual(
                mirror.acquire_calls[0]["lease_seconds"],
                2,
            )
            self.assertEqual(
                mirror.acquire_calls[0]["takeover_grace_seconds"],
                7,
            )
            # At least one periodic heartbeat plus the pre-commit validation.
            self.assertGreaterEqual(len(mirror.renew_calls), 2)
            self.assertTrue(
                all(call["lease_seconds"] == 2 for call in mirror.renew_calls)
            )
            self.assertEqual(mirror.release_calls, [("lease-test", engine.worker_id)])
            self.assertTrue(mirror.persisted_checkpoints)

        asyncio.run(scenario())

    def test_worker_lease_loss_discards_late_result_and_pauses_without_db_write(self):
        async def scenario():
            store = InMemoryStore()
            task = TaskRecord(
                task_id="task-lease-lost",
                thread_id="thread-lease-lost",
                job=_job(),
                state={"candidate_count": 1},
            )
            store.create_task(task)
            mirror = _LeaseMirror(renew_succeeds=False)
            graph = _BlockingGraph()
            engine = WorkflowEngine(
                store,
                mirror,
                worker_lease_seconds=2,
                worker_heartbeat_seconds=0.01,
                worker_takeover_grace_seconds=3,
            )
            engine.graph = graph
            running = asyncio.create_task(engine._run_once(task.task_id))
            self.assertTrue(await asyncio.to_thread(graph.entered.wait, 1))
            self.assertTrue(await asyncio.to_thread(mirror.renewed.wait, 1))
            result = await asyncio.wait_for(running, timeout=1)
            graph.release.set()
            await asyncio.sleep(0.05)
            self.assertEqual(result.status, TaskStatus.PAUSED)
            self.assertEqual(result.blocked_reason, "worker_lease_lost")
            self.assertEqual(result.current_node, "worker_lease")
            self.assertEqual(result.checkpoint_version, 1)
            self.assertFalse(result.candidates)
            self.assertFalse(mirror.persisted_checkpoints)
            self.assertFalse(mirror.release_calls)
            self.assertTrue(
                any(
                    event.get("blocked_reason") == "worker_lease_lost"
                    for event in store.events_for(task.task_id)
                )
            )

        asyncio.run(scenario())

    def test_cancel_waits_for_non_interruptible_node_before_next_task_runs(self):
        class PerTaskBlockingGraph:
            def __init__(self) -> None:
                self.entered = {
                    "task-cancel-active": threading.Event(),
                    "task-after-cancel": threading.Event(),
                }
                self.release = {
                    "task-cancel-active": threading.Event(),
                    "task-after-cancel": threading.Event(),
                }

            def invoke(self, state, _config):
                task_id = state["task_id"]
                self.entered[task_id].set()
                if not self.release[task_id].wait(timeout=2):
                    raise TimeoutError(f"test graph was not released: {task_id}")
                return _node_candidates(_node_match(state))

        async def scenario():
            store = InMemoryStore()
            first_task = TaskRecord(
                task_id="task-cancel-active",
                thread_id="thread-cancel-active",
                job=_job(),
                state={"candidate_count": 1},
            )
            second_task = TaskRecord(
                task_id="task-after-cancel",
                thread_id="thread-after-cancel",
                job=_job(),
                state={"candidate_count": 1},
            )
            store.create_task(first_task)
            store.create_task(second_task)
            graph = PerTaskBlockingGraph()
            engine = WorkflowEngine(store)
            engine.graph = graph

            first_run = asyncio.create_task(engine.run(first_task.task_id))
            self.assertTrue(
                await asyncio.to_thread(graph.entered[first_task.task_id].wait, 1)
            )
            second_run = asyncio.create_task(engine.run(second_task.task_id))
            await asyncio.sleep(0)
            self.assertFalse(graph.entered[second_task.task_id].is_set())

            first_run.cancel()
            cancelled = engine.resume(
                first_task.task_id,
                expected_version=0,
                action="cancel",
                payload={},
            )
            self.assertEqual(cancelled.status, TaskStatus.CANCELLED)
            await asyncio.sleep(0.05)
            advanced_before_current_node_finished = graph.entered[
                second_task.task_id
            ].is_set()

            graph.release[first_task.task_id].set()
            try:
                await first_run
            except asyncio.CancelledError:
                pass
            self.assertTrue(
                await asyncio.to_thread(graph.entered[second_task.task_id].wait, 1)
            )
            graph.release[second_task.task_id].set()
            await asyncio.wait_for(second_run, timeout=1)

            persisted_first = store.get_task(first_task.task_id)
            self.assertIsNotNone(persisted_first)
            self.assertEqual(persisted_first.status, TaskStatus.CANCELLED)
            self.assertFalse(persisted_first.candidates)
            self.assertFalse(
                advanced_before_current_node_finished,
                "cooperative cancellation must keep the queue locked until the "
                "already-started synchronous node has returned",
            )

        asyncio.run(scenario())

    def test_feedback_classifier_allows_non_conflicting_mixed_preferences(self):
        from app.services.feedback import classify_feedback

        classification = classify_feedback("请突出 Python 并调整顺序")
        self.assertFalse(classification.requires_clarification)
        self.assertEqual(classification.category, "opinion")

    def test_confirmation_merges_prior_field_only_confirmations(self):
        from app.core.utils import sha256_text

        async def scenario():
            store = InMemoryStore()
            task = TaskRecord(task_id="task-confirm", thread_id="thread-confirm", job=_job(), state={"candidate_count": 1})
            store.create_task(task)
            engine = WorkflowEngine(store)
            current = await engine.run(task.task_id)
            selected = engine.resume(
                task.task_id,
                expected_version=current.checkpoint_version,
                action="select",
                payload={"candidate_id": current.candidates[0].candidate_id},
            )
            candidate = selected.candidates[0]
            values = {
                "title": candidate.title,
                "period": candidate.period,
                "introduction": candidate.introduction,
                "tech_stack": candidate.tech_stack,
                "solutions": candidate.solutions,
                "results": candidate.results,
            }

            def value_hash(value):
                serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                return sha256_text(serialized)

            first = {"field_id": "title", "value_hash": value_hash(values["title"]), "confirmed": True}
            selected = store.checkpoint(
                task.task_id,
                selected.checkpoint_version,
                state={**selected.state, "confirmations": [first]},
                current_node="candidate_confirmation",
                messages=selected.messages,
            )
            rest = [
                {"field_id": name, "value_hash": value_hash(value), "confirmed": True}
                for name, value in values.items()
                if name != "title"
            ]
            confirmed = engine.resume(
                task.task_id,
                expected_version=selected.checkpoint_version,
                action="confirm",
                payload={"candidate_id": candidate.candidate_id, "confirmations": rest},
            )
            self.assertEqual(confirmed.status.value, "waiting_user")
            self.assertEqual(confirmed.current_node, "project_application")
            self.assertEqual(confirmed.state["pending_module"], "projects")
            self.assertNotIn("resume_snapshot_id", confirmed.state)

        asyncio.run(scenario())

    def test_candidate_confirmation_requires_project_modules_layout_and_preview(self):
        from app.core.utils import sha256_text

        async def scenario():
            store = InMemoryStore()
            task = TaskRecord(
                task_id="task-state-machine",
                thread_id="thread-state-machine",
                job=_job(),
                state={"candidate_count": 1},
            )
            store.create_task(task)
            engine = WorkflowEngine(store)
            current = await engine.run(task.task_id)
            current = engine.resume(
                task.task_id,
                expected_version=current.checkpoint_version,
                action="select",
                payload={"candidate_id": current.candidates[0].candidate_id},
            )
            candidate = current.candidates[0]

            def value_hash(value):
                serialized = value if isinstance(value, str) else json.dumps(
                    value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                return sha256_text(serialized)

            fields = {
                "title": candidate.title,
                "period": candidate.period,
                "introduction": candidate.introduction,
                "tech_stack": candidate.tech_stack,
                "solutions": candidate.solutions,
                "results": candidate.results,
            }
            current = engine.resume(
                task.task_id,
                expected_version=current.checkpoint_version,
                action="confirm",
                payload={
                    "candidate_id": candidate.candidate_id,
                    "confirmations": [
                        {
                            "field_id": name,
                            "value_hash": value_hash(value),
                            "confirmed": True,
                        }
                        for name, value in fields.items()
                    ],
                },
            )
            self.assertEqual(current.current_node, "project_application")
            self.assertEqual(current.state["project_decision"]["recommended_action"], "add")

            current = engine.resume(
                task.task_id,
                expected_version=current.checkpoint_version,
                action="approve",
                payload={},
            )
            self.assertEqual(current.current_node, "summary_review")
            self.assertEqual(current.state["project_patch"]["operation"], "add")

            for module in ("summary", "skills", "evaluation"):
                self.assertEqual(current.current_node, f"{module}_review")
                self.assertEqual(current.state["pending_module"], module)
                current = engine.resume(
                    task.task_id,
                    expected_version=current.checkpoint_version,
                    action="skip",
                    payload={},
                )

            self.assertEqual(current.current_node, "layout_review")
            self.assertNotIn("resume_snapshot_id", current.state)
            current = engine.resume(
                task.task_id,
                expected_version=current.checkpoint_version,
                action="approve",
                payload={"page_count": 1},
            )
            self.assertEqual(current.current_node, "preview_review")
            self.assertTrue(current.state["resume_snapshot_id"])
            self.assertEqual(current.status.value, "waiting_user")

            current = engine.resume(
                task.task_id,
                expected_version=current.checkpoint_version,
                action="confirm",
                payload={},
            )
            self.assertEqual(current.status.value, "completed")
            self.assertEqual(current.current_node, "completed")
            self.assertTrue(current.state["final_confirmation"])

        asyncio.run(scenario())

    def test_candidate_can_be_edited_and_saved_before_confirmation(self):
        async def scenario():
            store = InMemoryStore()
            task = TaskRecord(
                task_id="task-candidate-edit",
                thread_id="thread-candidate-edit",
                job=_job(),
                state={
                    "candidate_count": 1,
                    "output_mode": "project_only",
                    "requested_output_mode": "project_only",
                },
            )
            store.create_task(task)
            engine = WorkflowEngine(store)
            current = await engine.run(task.task_id)
            candidate = current.candidates[0]
            original = {
                "title": candidate.title,
                "period": candidate.period,
                "introduction": candidate.introduction,
                "tech_stack": candidate.tech_stack,
                "solutions": candidate.solutions,
                "results": candidate.results,
            }
            fields = {
                **original,
                "title": "用户校正后的智能招聘助手",
                "solutions": [*candidate.solutions, "增加候选内容人工校验闭环"],
            }

            edited = engine.resume(
                task.task_id,
                expected_version=current.checkpoint_version,
                action="edit",
                payload={
                    "candidate_id": candidate.candidate_id,
                    "fields": fields,
                    "old_value_hashes": {
                        name: _value_hash(value) for name, value in original.items()
                    },
                },
            )

            self.assertEqual(edited.current_node, current.current_node)
            self.assertEqual(
                edited.candidates[0].title, "用户校正后的智能招聘助手"
            )
            self.assertEqual(edited.candidates[0].status, "draft")
            self.assertTrue(edited.candidates[0].needs_verification)
            self.assertEqual(edited.state["confirmations"], [])
            self.assertEqual(
                edited.state["candidate_edits"][-1]["checkpoint_base"],
                current.checkpoint_version,
            )

            with self.assertRaisesRegex(ValueError, "old value hash mismatch"):
                engine.resume(
                    task.task_id,
                    expected_version=edited.checkpoint_version,
                    action="edit",
                    payload={
                        "candidate_id": candidate.candidate_id,
                        "fields": {"title": "过期页面的修改"},
                        "old_value_hashes": {"title": _value_hash(candidate.title)},
                    },
                )

        asyncio.run(scenario())

    def test_project_only_honours_draft_count_and_completes_without_resume_patch(self):
        from app.core.utils import sha256_text

        async def scenario():
            store = InMemoryStore()
            task = TaskRecord(
                task_id="task-project-only-direct",
                thread_id="thread-project-only-direct",
                job=_job(),
                state={
                    "candidate_count": 3,
                    "output_mode": "project_only",
                    "requested_output_mode": "project_only",
                },
            )
            store.create_task(task)
            engine = WorkflowEngine(store)

            current = await engine.run(task.task_id)
            self.assertEqual(current.current_node, "candidate_review")
            self.assertEqual(len(current.candidates), 3)
            current = engine.resume(
                task.task_id,
                expected_version=current.checkpoint_version,
                action="select",
                payload={"candidate_id": current.candidates[0].candidate_id},
            )
            self.assertEqual(current.current_node, "candidate_confirmation")
            candidate = current.candidates[0]
            values = {
                "title": candidate.title,
                "period": candidate.period,
                "introduction": candidate.introduction,
                "tech_stack": candidate.tech_stack,
                "solutions": candidate.solutions,
                "results": candidate.results,
            }
            confirmations = []
            for field, value in values.items():
                serialized = (
                    value
                    if isinstance(value, str)
                    else json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                confirmations.append(
                    {
                        "field_id": field,
                        "value_hash": sha256_text(serialized),
                        "confirmed": True,
                    }
                )
            current = engine.resume(
                task.task_id,
                expected_version=current.checkpoint_version,
                action="confirm",
                payload={
                    "candidate_id": candidate.candidate_id,
                    "confirmations": confirmations,
                },
            )

            self.assertEqual(current.status, TaskStatus.COMPLETED)
            self.assertEqual(current.current_node, "completed")
            self.assertEqual(current.state["final_product"], "project_only")
            self.assertIsNone(current.state["project_patch"])
            self.assertNotIn("resume_snapshot_id", current.state)

        asyncio.run(scenario())

    def test_optional_module_reviews_only_trigger_below_eighty(self):
        from app.services.matching import calculate_match

        store = InMemoryStore()
        match = calculate_match(_job(), None)
        match.total_score = 85.0
        match.threshold_triggered = False
        candidate = generate_candidates(_job(), None, match, task_id="task-high-score", count=1)[0]
        candidate.status = "confirmed"
        candidate.needs_verification = False
        task = TaskRecord(
            task_id="task-high-score",
            thread_id="thread-high-score",
            job=_job(),
            match=match,
            candidates=[candidate],
            state={
                "selected_candidate": candidate.candidate_id,
                "project_decision": {
                    "recommended_action": "add",
                    "target_project_id": None,
                    "project_count": 0,
                },
            },
            current_node="project_application",
            status=TaskStatus.WAITING_USER,
        )
        store.create_task(task)
        current = WorkflowEngine(store).resume(
            task.task_id,
            expected_version=0,
            action="approve",
            payload={},
        )
        self.assertEqual(current.current_node, "layout_review")
        self.assertFalse(
            any(
                item["needs_optimization"]
                for item in current.state["module_assessment"].values()
            )
        )

    def test_project_patch_only_skips_optional_module_rewrites(self):
        from app.services.matching import calculate_match

        store = InMemoryStore()
        match = calculate_match(_job(), None)
        candidate = generate_candidates(
            _job(), None, match, task_id="task-project-patch", count=1
        )[0]
        candidate.status = "confirmed"
        candidate.needs_verification = False
        task = TaskRecord(
            task_id="task-project-patch",
            thread_id="thread-project-patch",
            job=_job(),
            match=match,
            candidates=[candidate],
            state={
                "selected_candidate": candidate.candidate_id,
                "requested_output_mode": "resume_edit",
                "project_decision": {
                    "recommended_action": "add",
                    "target_project_id": None,
                    "project_count": 0,
                },
            },
            current_node="project_application",
            status=TaskStatus.WAITING_USER,
        )
        store.create_task(task)

        current = WorkflowEngine(store).resume(
            task.task_id,
            expected_version=0,
            action="approve",
            payload={"project_action": "add", "final_product": "project_patch_only"},
        )

        self.assertEqual(current.current_node, "layout_review")
        self.assertEqual(current.state["final_product"], "project_patch_only")
        self.assertEqual(current.state["project_patch"]["operation"], "add")
        self.assertNotIn("module_drafts", current.state)

    def test_module_patch_requires_matching_value_hash(self):
        async def scenario():
            store = InMemoryStore()
            task = TaskRecord(
                task_id="task-module-hash",
                thread_id="thread-module-hash",
                job=_job(),
                state={
                    "module_assessment": {
                        "summary": {"needs_optimization": True},
                        "skills": {"needs_optimization": False},
                        "evaluation": {"needs_optimization": False},
                    },
                    "module_decisions": {},
                    "pending_module": "summary",
                },
                current_node="summary_review",
                status=TaskStatus.WAITING_USER,
            )
            store.create_task(task)
            engine = WorkflowEngine(store)
            current = engine.resume(
                task.task_id,
                expected_version=0,
                action="approve",
                payload={},
            )
            self.assertEqual(current.current_node, "summary_confirmation")
            with self.assertRaises(ValueError):
                engine.resume(
                    task.task_id,
                    expected_version=current.checkpoint_version,
                    action="confirm",
                    payload={"value_hash": "stale"},
                )
            draft_hash = current.state["module_drafts"]["summary"]["new_value_hash"]
            current = engine.resume(
                task.task_id,
                expected_version=current.checkpoint_version,
                action="confirm",
                payload={"value_hash": draft_hash},
            )
            self.assertEqual(current.current_node, "layout_review")
            self.assertEqual(current.state["module_patches"]["summary"]["status"], "confirmed")

        asyncio.run(scenario())

    def test_rewrite_requires_a_layout_only_preview_first(self):
        store = InMemoryStore()
        task = TaskRecord(
            task_id="task-compression-order",
            thread_id="thread-compression-order",
            job=_job(),
            state={
                "pending_module": "layout",
                "layout_decision": {
                    "estimated_page_count": 2,
                    "requires_compression": True,
                    "status": "awaiting_layout_review",
                },
            },
            current_node="layout_review",
            status=TaskStatus.WAITING_USER,
        )
        store.create_task(task)
        engine = WorkflowEngine(store)
        current = engine.resume(
            task.task_id,
            expected_version=0,
            action="approve",
            payload={"page_count": 2},
        )
        self.assertEqual(current.current_node, "compression_review")
        with self.assertRaises(ValueError):
            engine.resume(
                task.task_id,
                expected_version=current.checkpoint_version,
                action="revise",
                payload={"allow_rewrite": True},
            )
        current = engine.resume(
            task.task_id,
            expected_version=current.checkpoint_version,
            action="approve",
            payload={"accept_compression": True},
        )
        self.assertEqual(current.current_node, "preview_review")
        first_snapshot = current.state["resume_snapshot_id"]
        current = engine.resume(
            task.task_id,
            expected_version=current.checkpoint_version,
            action="revise",
            payload={"feedback": "压缩后仍超过一页", "allow_rewrite": True},
        )
        self.assertEqual(current.current_node, "layout_review")
        self.assertIn(first_snapshot, current.state["previous_snapshot_ids"])
        current = engine.resume(
            task.task_id,
            expected_version=current.checkpoint_version,
            action="approve",
            payload={"page_count": 2},
        )
        current = engine.resume(
            task.task_id,
            expected_version=current.checkpoint_version,
            action="revise",
            payload={"allow_rewrite": True},
        )
        self.assertEqual(current.current_node, "preview_review")
        self.assertTrue(current.state["compression_decision"]["allow_rewrite"])
        self.assertNotEqual(current.state["resume_snapshot_id"], first_snapshot)


if __name__ == "__main__":
    unittest.main()
