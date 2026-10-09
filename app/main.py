"""FastAPI application entry point for the local Resume Agent."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.core.config import Settings, get_settings
from app.core.database_store import DatabaseMirror
from app.core.db import create_engine_from_settings, init_db, session_factory
from app.core.schemas import (
    ActionRequest,
    CandidateProject,
    FeedbackRequest,
    JobCreateRequest,
    JobInput,
    JobUrlRequest,
    ModelProfile,
    ModelRole,
    PatchRequest,
    ProbeResponse,
    Provider,
    ResumeDocument,
    ResumeSection,
    ScreenshotConfirmRequest,
    SettingsUpdate,
    TaskCreateRequest,
    TaskStatus,
    TaskView,
)
from app.core.store import InMemoryStore, TaskRecord, VersionConflict
from app.core.secret_store import EncryptedSecretStore, SecretStoreUnavailable
from app.services.job_llm_parser import enhance_job_with_model, rule_fallback_metadata
from app.services.job_parser import infer_job_metadata, job_from_text, parse_job_detail_text, validate_boss_url
from app.services.resume_parser import ResumeParseError, parse_resume_bytes
from app.services.resume_llm_parser import (
    resume_rule_fallback_metadata,
    structure_resume_with_model,
)
from app.services.screenshot_parser import (
    MAX_SCREENSHOT_BYTES,
    PaddleScreenshotOCR,
    ScreenshotOCRError,
)
from app.services.export_service import (
    compress_markdown,
    convert_docx_to_pdf,
    convert_pdf_with_libreoffice,
    estimate_pages,
    export_docx,
    export_text_pdf,
    render_candidate_project,
    render_markdown,
    write_preview,
)
from app.services.backup_service import create_backup, restore_backup, validate_backup
from app.services.template_service import (
    cache_github_template,
    check_github_template_update,
    preview_cached_template,
    preview_github_template,
    render_docx_html_preview,
    search_github_templates,
)
from app.services.template_llm_parser import (
    confirm_template_structure,
    load_template_structure,
    structure_template_with_model,
)
from app.services.skill_service import list_skill_versions, read_skill, rollback_skill as rollback_skill_file, save_skill
from app.services.workflow import WorkflowEngine
from app.mcp.client import MCPClientError, MCPClientManager, default_server_specs
from app.mcp.edge_adapter import decode_boss_obfuscated_text


def _gateway_profile(profile: ModelProfile):
    """Translate the API schema into the provider-neutral gateway profile."""
    from app.core.model_gateway import (
        EmbeddingParameters,
        GenerationParameters,
        ModelProfile as GatewayProfile,
        ModelRole as GatewayRole,
        Provider as GatewayProvider,
    )

    generation = GenerationParameters(
        temperature=float(profile.generation_params.get("temperature", 0.2)),
        top_p=float(profile.generation_params.get("top_p", 1.0)),
        max_output_tokens=int(profile.generation_params.get("max_output_tokens", 1200)),
        timeout_seconds=float(profile.generation_params.get("timeout_seconds", 120.0)),
        max_retries=int(profile.generation_params.get("max_retries", 1)),
        retry_interval_seconds=float(
            profile.generation_params.get("retry_interval_seconds", 0.5)
        ),
    )
    embedding = EmbeddingParameters(max_input_tokens=profile.max_input_tokens)
    return GatewayProfile(
        profile_id=profile.profile_id,
        role=GatewayRole(profile.role.value),
        provider=GatewayProvider(profile.provider.value),
        base_url=profile.base_url,
        model_name=profile.model_name,
        profile_version=f"v{profile.config_version}",
        credential_required=profile.credential_required,
        auth_scheme=str(profile.generation_params.get("auth_scheme") or "bearer"),
        context_window_tokens=profile.context_window_tokens,
        tokenizer_id=profile.tokenizer_id,
        tokenizer_version=profile.tokenizer_version,
        tokenizer_source=profile.tokenizer_source,
        generation=generation,
        embedding=embedding,
    )


def _probe_response(result: Any) -> ProbeResponse:
    return ProbeResponse(
        role=ModelRole(result.role.value),
        provider=Provider(result.provider.value),
        status=result.status.value,
        base_url=result.base_url,
        endpoint=result.endpoint,
        model_name=result.model_name,
        model_digest=result.model_digest,
        dimension=result.dimension,
        capabilities=dict(result.capabilities),
        checked_at=datetime.fromisoformat(result.checked_at.replace("Z", "+00:00")),
        error_code=result.error_code,
        error_message=result.error_message,
        retryable=result.retryable,
        requires_user=result.requires_user,
        warnings=list(result.warnings),
        metadata=dict(result.metadata),
    )


def _default_profiles(settings: Settings) -> list[ModelProfile]:
    return [
        ModelProfile(
            profile_id="ollama-chat-default",
            role=ModelRole.CHAT,
            provider=Provider.OLLAMA,
            base_url=settings.ollama_base_url,
            model_name=settings.default_chat_model,
            default_for_role=True,
            context_window_tokens=32768,
            tokenizer_id="conservative-char-v1",
            tokenizer_version="1",
            tokenizer_source="fallback",
        ),
        ModelProfile(
            profile_id="ollama-embedding-default",
            role=ModelRole.EMBEDDING,
            provider=Provider.OLLAMA,
            base_url=settings.ollama_base_url,
            model_name=settings.default_embedding_model,
            default_for_role=True,
            dimension=None,
            max_input_tokens=8192,
        ),
    ]


class AppState:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = InMemoryStore()
        self.screenshot_ocr = PaddleScreenshotOCR()
        # OCR images are never persisted. Only the extracted text and its
        # digest live briefly in process memory until the user confirms it.
        self.pending_ocr: dict[str, dict[str, Any]] = {}
        self.pending_ocr_lock = threading.RLock()
        self.job_parse_confirmations: dict[str, bool] = {}
        from app.core.model_gateway import CredentialStore

        self.credentials = CredentialStore()
        self.secret_store = EncryptedSecretStore(
            self.settings.data_root / "credentials" / "model-secrets.json"
        )
        self.credential_scopes: dict[str, set[str]] = {"model": set(), "github": set()}
        # Profile-to-handle bindings are process-local only.  They let every
        # model operation reuse a saved API key without asking Streamlit to
        # keep an opaque handle alive across reruns.  The map never contains
        # plaintext and is rebuilt from the encrypted machine-local store.
        self.model_credential_handles: dict[str, str] = {}
        self.credential_registry_lock = threading.RLock()
        from app.core.model_gateway import ModelGateway

        # One gateway owns the process-local credential store and is shared by
        # probes and graph nodes.  It never serializes the credential entries.
        self.model_gateway = ModelGateway(credential_store=self.credentials)
        # FastAPI owns these official MCP stdio clients for the whole process.
        # Credentials are passed only to individual tool calls; the manager
        # never receives or persists them.
        self.mcp_manager = MCPClientManager(
            default_server_specs(
                project_root=Path(__file__).resolve().parents[1],
                data_root=self.settings.data_root,
                edge_path=self.settings.edge_path,
            )
        )
        self.mcp_startup_errors: dict[str, str] = {}
        self.mcp_start_task: asyncio.Task[Any] | None = None
        # Consent contains no credential material and is persisted separately.
        # Model API keys may be recovered from the machine-local encrypted
        # store into short-lived handles; GitHub tokens remain process-local.
        self.consents: dict[str, dict[str, Any]] = {}
        self.skill_versions: list[dict[str, Any]] = []
        self.skill_path = self.settings.data_root / "skills" / "resume-preferences.md"
        try:
            self.current_skill = self.skill_path.read_text(encoding="utf-8") if self.skill_path.exists() else ""
        except OSError:
            self.current_skill = ""
        try:
            self.skill_versions = list_skill_versions(self.settings.data_root)
        except OSError:
            self.skill_versions = []
        self.db_mirror: DatabaseMirror | None = None
        try:
            engine = create_engine_from_settings(self.settings)
            init_db(engine)
            self.db_mirror = DatabaseMirror(engine, session_factory(engine))
        except Exception:
            # The API can still run in setup mode; environment reports the
            # in-memory fallback and the next process restart retries the DB.
            self.db_mirror = None
        if self.db_mirror:
            persisted_settings = self.db_mirror.load_settings()
            if persisted_settings:
                # Secrets are never loaded from this row; the mirror only
                # contains ordinary preferences and versioned feature flags.
                persisted_settings.pop("api_key", None)
                persisted_settings.pop("github_token", None)
                self.store.settings.update(persisted_settings)
            self.consents.update(self.db_mirror.load_consents())
        for profile in _default_profiles(self.settings):
            self.store.seed_profile(profile)
        if self.db_mirror:
            # Rehydrate durable IDs/content before constructing the workflow so
            # a restarted process can continue an existing thread instead of
            # silently creating a new task lineage.
            try:
                for profile in self.db_mirror.load_model_profiles():
                    self.store.seed_profile(profile)
                for job_id, job in self.db_mirror.load_jobs():
                    self.store.hydrate_job(job_id, job)
                for resume in self.db_mirror.load_resumes():
                    self.store.hydrate_resume(resume)
                for task in self.db_mirror.load_tasks():
                    self.store.hydrate_task(task)
            except Exception:
                # A corrupt/partial row must not prevent setup mode from
                # starting; the database mirror reports the degraded state.
                self.db_mirror.available = False
        # Restore one live handle per saved external-model profile at startup.
        # Failure to decrypt one legacy entry is isolated to that profile and
        # is reported by /api/credentials/stored/{profile_id}; it must not stop
        # the rest of the local application from starting.
        for profile in self.store.list_profiles():
            if profile.provider is not Provider.OPENAI_COMPATIBLE:
                continue
            try:
                secret = self.secret_store.get(profile.profile_id)
                if not secret:
                    continue
                handle = self.credentials.put(
                    secret,
                    scope=f"model:{profile.profile_id}",
                    ttl_seconds=31536000,
                )
            except (SecretStoreUnavailable, TypeError, ValueError):
                continue
            self.model_credential_handles[profile.profile_id] = handle.handle_id
            self.credential_scopes["model"].add(handle.handle_id)
        self.workflow = WorkflowEngine(
            self.store,
            self.db_mirror,
            model_gateway=self.model_gateway,
            strict_model_gate=self.settings.strict_model_gate,
            task_timeout_seconds=self.settings.task_timeout_seconds,
            worker_lease_seconds=self.settings.worker_lease_seconds,
            worker_heartbeat_seconds=self.settings.worker_heartbeat_seconds,
            worker_takeover_grace_seconds=self.settings.worker_takeover_grace_seconds,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.resume = AppState()
    runtime = app.state.resume
    runtime.mcp_startup_errors = {}
    try:
        if runtime.settings.mcp_autostart:
            async def start_mcp_children() -> None:
                async def start_one(server_name: str) -> None:
                    try:
                        await runtime.mcp_manager.start_server(server_name)
                    except MCPClientError as exc:
                        # Missing Playwright/Edge or a missing optional SDK must not
                        # prevent the local resume editor from starting.  The exact,
                        # machine-readable health state is exposed on /environment.
                        runtime.mcp_startup_errors[server_name] = exc.error_code
                    except Exception:
                        runtime.mcp_startup_errors[server_name] = "mcp_startup_failed"

                await asyncio.gather(*(start_one(name) for name in ("boss", "github")))

            # Do not make the API's first response wait for optional browser/model
            # dependencies.  The manager still belongs to the lifespan and every
            # dependent route can await the same per-server startup lock.
            runtime.mcp_start_task = asyncio.create_task(start_mcp_children())
        yield
    finally:
        startup_task = runtime.mcp_start_task
        if startup_task is not None and not startup_task.done():
            startup_task.cancel()
            await asyncio.gather(startup_task, return_exceptions=True)

        await runtime.mcp_manager.close()

        running = list(getattr(runtime.workflow, "_running", {}).values())
        pending = [task for task in running if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        if runtime.db_mirror is not None:
            runtime.db_mirror.close()


app = FastAPI(title="Resume Agent", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8501",
        "http://localhost:8501",
        "http://127.0.0.1:8666",
        "http://localhost:8666",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error_code(detail: Any, status_code: int) -> str:
    """Return a stable public error code without leaking provider internals."""
    if isinstance(detail, dict):
        value = detail.get("error_code") or detail.get("code")
        if value:
            return str(value)
    if status_code == 401:
        return "unauthorized"
    if status_code == 404:
        return "not_found"
    if status_code == 409:
        return "conflict"
    if status_code == 422:
        return "validation_error"
    return "request_error"


def _error_message(detail: Any) -> str:
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("detail") or detail.get("error_code") or "请求失败")
    if isinstance(detail, list):
        return "请求参数校验失败"
    return str(detail or "请求失败")


def _json_safe(value: Any) -> Any:
    """Convert validation internals (notably ValueError in ``ctx``) to JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


_SECRET_SETTING_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "github_token",
    "password",
    "secret",
    "token",
}


def _strip_secret_settings(value: Any, *, key: str | None = None) -> Any:
    """Remove credential-shaped values before settings reach memory/DB.

    Credentials have a dedicated in-memory handle API.  Settings may contain
    nested provider preferences, so filtering only two top-level keys is not
    sufficient to meet the no-persistence guarantee.
    """

    if key and key.lower() in _SECRET_SETTING_KEYS:
        return None
    if isinstance(value, dict):
        return {
            str(name): _strip_secret_settings(item, key=str(name))
            for name, item in value.items()
            if str(name).lower() not in _SECRET_SETTING_KEYS
        }
    if isinstance(value, list):
        return [_strip_secret_settings(item) for item in value]
    return value


def _validate_scoring_settings(values: dict[str, Any]) -> None:
    contracts = {
        "dimension_weights": {"skills", "responsibilities", "projects", "keywords"},
        "component_weights": {"deterministic", "semantic"},
    }
    for key, expected in contracts.items():
        if key not in values:
            continue
        weights = values[key]
        if not isinstance(weights, dict) or set(weights) != expected:
            raise HTTPException(
                status_code=422,
                detail={"error_code": "invalid_scoring_weights", "message": f"{key} 字段不完整"},
            )
        try:
            numeric = [float(weights[name]) for name in expected]
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail={"error_code": "invalid_scoring_weights", "message": f"{key} 必须为数字"},
            ) from exc
        if any(value < 0 or value > 1 for value in numeric) or abs(sum(numeric) - 1.0) > 1e-8:
            raise HTTPException(
                status_code=422,
                detail={"error_code": "invalid_scoring_weights", "message": f"{key} 必须位于 0-1 且合计为 1"},
            )


@app.exception_handler(StarletteHTTPException)
async def http_error_envelope(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Keep legacy ``detail`` while exposing one machine-readable envelope.

    Existing clients used the detail string/dict directly.  Retaining it in the
    response avoids a breaking change while new clients can consistently consume
    ``error.code`` and ``error.message``.
    """
    code = _error_code(exc.detail, exc.status_code)
    message = _error_message(exc.detail)
    details = _json_safe(exc.detail) if isinstance(exc.detail, (dict, list)) else None
    request_id = uuid.uuid4().hex
    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content={
            "ok": False,
            "error": {
                "code": code,
                "message": message,
                "details": details,
                "retryable": code in {"service_unavailable", "probe_exception", "temporary_failure"},
                "requires_user": exc.status_code in {401, 409, 422},
                "request_id": request_id,
            },
            "error_code": code,
            "message": message,
            "detail": _json_safe(exc.detail),
            "request_id": request_id,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_envelope(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = uuid.uuid4().hex
    details = _json_safe(exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "ok": False,
            "error": {
                "code": "validation_error",
                "message": "请求参数校验失败",
                "details": details,
                "retryable": False,
                "requires_user": True,
                "request_id": request_id,
            },
            "error_code": "validation_error",
            "message": "请求参数校验失败",
            "detail": details,
            "request_id": request_id,
        },
    )


def state() -> AppState:
    if not hasattr(app.state, "resume"):
        app.state.resume = AppState()
    return app.state.resume


def _materialize_model_credential(
    profile_id: str,
    handle_id: str | None = None,
) -> str | None:
    """Return a live handle, restoring the machine-local encrypted secret if needed."""

    app_state = state()
    profile_id = str(profile_id or "").strip()
    if not profile_id:
        return None
    scope = f"model:{profile_id}"
    registry = getattr(app_state, "model_credential_handles", None)
    registry_lock = getattr(app_state, "credential_registry_lock", None)

    def resolve() -> str | None:
        active_handle = registry.get(profile_id) if isinstance(registry, dict) else None
        if active_handle and app_state.credentials.has(active_handle, scope=scope):
            return active_handle
        if isinstance(registry, dict):
            registry.pop(profile_id, None)
        if active_handle:
            app_state.credential_scopes.setdefault("model", set()).discard(active_handle)
        if handle_id and app_state.credentials.has(handle_id, scope=scope):
            if isinstance(registry, dict):
                registry[profile_id] = handle_id
            app_state.credential_scopes.setdefault("model", set()).add(handle_id)
            return handle_id
        secret_store = getattr(app_state, "secret_store", None)
        if secret_store is None:
            return None
        try:
            secret = secret_store.get(profile_id)
        except SecretStoreUnavailable:
            return None
        if not secret:
            return None
        handle = app_state.credentials.put(secret, scope=scope, ttl_seconds=31536000)
        if isinstance(registry, dict):
            registry[profile_id] = handle.handle_id
        app_state.credential_scopes.setdefault("model", set()).add(handle.handle_id)
        return handle.handle_id

    if registry_lock is None:
        return resolve()
    with registry_lock:
        return resolve()


async def _ensure_mcp_server(server_name: str) -> dict[str, Any]:
    """Ensure one local MCP child is ready, returning a safe health summary."""

    try:
        await state().mcp_manager.start_server(server_name)
        health = await state().mcp_manager.health(probe=False)
        return dict(health.get(server_name, {}))
    except MCPClientError as exc:
        state().mcp_startup_errors[server_name] = exc.error_code
        return {"name": server_name, "status": "unavailable", "error_code": exc.error_code}
    except Exception:
        state().mcp_startup_errors[server_name] = "mcp_startup_failed"
        return {"name": server_name, "status": "unavailable", "error_code": "mcp_startup_failed"}


async def _apply_runtime_setting_changes(values: dict[str, Any]) -> None:
    """Apply non-secret process settings that require adapter reconstruction."""

    if "edge_path" not in values:
        return
    requested = str(values.get("edge_path") or "").strip() or None
    if requested == state().settings.edge_path:
        return
    await state().mcp_manager.close()
    state().settings.edge_path = requested
    state().mcp_manager = MCPClientManager(
        default_server_specs(
            project_root=Path(__file__).resolve().parents[1],
            data_root=state().settings.data_root,
            edge_path=requested,
        )
    )
    state().mcp_startup_errors.clear()


def require_internal_token(x_resume_agent_token: str | None = Header(default=None)) -> None:
    configured = state().settings.internal_token
    # Browser UI receives the token from the local runtime file in a later slice;
    # allow same-machine requests when no explicit token was configured.
    if configured and x_resume_agent_token != configured:
        raise HTTPException(status_code=401, detail="invalid internal token")


def task_view(task: TaskRecord) -> TaskView:
    return TaskView(
        task_id=task.task_id,
        thread_id=task.thread_id,
        status=task.status,
        blocked_reason=task.blocked_reason,
        current_node=task.current_node,
        checkpoint_version=task.checkpoint_version,
        job_snapshot_id=task.job_snapshot_id,
        resume_snapshot_id=task.state.get("resume_snapshot_id"),
        previous_snapshot_ids=[
            str(value)
            for value in task.state.get("previous_snapshot_ids", [])
            if str(value).strip()
        ],
        template_id=task.state.get("template_id"),
        output_mode=str(task.state.get("output_mode") or "resume_edit"),
        requested_output_mode=str(
            task.state.get("requested_output_mode") or task.state.get("output_mode") or "resume_edit"
        ),
        final_product=task.state.get("final_product"),
        selected_candidate_id=task.state.get("selected_candidate") or task.state.get("selected_candidate_id"),
        generation_branch_id=task.state.get("generation_branch_id"),
        feedback_classification=task.state.get("feedback_classification"),
        embedding_mode=task.state.get("embedding_mode"),
        pending_module=task.state.get("pending_module"),
        project_decision=task.state.get("project_decision"),
        project_patch=task.state.get("project_patch"),
        module_assessment=dict(task.state.get("module_assessment") or {}),
        module_decisions=dict(task.state.get("module_decisions") or {}),
        module_drafts=dict(task.state.get("module_drafts") or {}),
        module_patches=dict(task.state.get("module_patches") or {}),
        layout_decision=task.state.get("layout_decision"),
        compression_decision=task.state.get("compression_decision"),
        preview_status=task.state.get("preview_status"),
        final_confirmation=bool(task.state.get("final_confirmation", False)),
        candidates=task.candidates,
        match=task.match,
        messages=task.messages,
    )


def _require_task_checkpoint(task: TaskRecord, thread_id: str, checkpoint_version: int) -> None:
    """Reject stale browser actions before any validation or side effect."""

    if task.thread_id != thread_id:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "task_thread_not_found", "message": "task/thread not found"},
        )
    if task.checkpoint_version != checkpoint_version:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "checkpoint_conflict",
                "current_version": task.checkpoint_version,
                "refresh_required": True,
            },
        )


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "service": "resume-agent", "version": app.version, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/environment")
async def environment(_: None = Depends(require_internal_token)) -> dict[str, Any]:
    settings = state().settings
    profiles = state().store.list_profiles()
    try:
        mcp_health = await state().mcp_manager.health(probe=False)
    except Exception:
        mcp_health = {}
    return {
        "api": {"status": "ready", "host": settings.api_host, "port": settings.api_port},
        "database": {
            "status": "ready" if state().db_mirror and state().db_mirror.available else "in_memory",
            # Keep credentials embedded in DATABASE_URL out of the environment
            # response; Settings.public_dict() renders a password-redacted URL.
            "configured_url": settings.public_dict().get("database_url"),
        },
        "ollama": {"status": "not_probed", "base_url": settings.ollama_base_url},
        "mcp": {
            "status": "ready" if any(item.get("status") == "ready" for item in mcp_health.values()) else "degraded",
            "servers": mcp_health,
            "startup_errors": dict(state().mcp_startup_errors),
        },
        "models": [profile.model_dump(mode="json") for profile in profiles],
        "data_root": str(settings.data_root),
    }


@app.post("/api/environment/{component}/recheck")
async def environment_recheck(component: str, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    if component not in {"ollama", "database", "models", "mcp", "boss", "github"}:
        raise HTTPException(status_code=404, detail="unknown component")
    if component == "database":
        app_state = state()
        if app_state.db_mirror is None:
            try:
                engine = create_engine_from_settings(app_state.settings)
                init_db(engine)
                app_state.db_mirror = DatabaseMirror(engine, session_factory(engine))
            except Exception as exc:
                return {"component": component, "status": "unavailable", "error_code": "database_unavailable", "message": str(exc)}
        if app_state.db_mirror.healthcheck():
            return {"component": component, "status": "ready", "configured_url": app_state.settings.public_dict().get("database_url")}
        return {"component": component, "status": "unavailable", "error_code": "database_unavailable"}
    if component in {"mcp", "boss", "github"}:
        names = ["boss", "github"] if component == "mcp" else [component]
        results: dict[str, Any] = {}
        for name in names:
            try:
                await state().mcp_manager.start_server(name)
                results[name] = await state().mcp_manager.health(probe=True)
                results[name] = results[name].get(name, {})
                if name == "boss" and results[name].get("status") == "ready":
                    browser = await state().mcp_manager.call_tool(
                        "boss", "boss_environment", {}
                    )
                    if browser.get("ok") and isinstance(browser.get("data"), dict):
                        results[name]["browser"] = dict(browser["data"])
                    else:
                        browser_data = (
                            browser.get("data")
                            if isinstance(browser.get("data"), dict)
                            else {}
                        )
                        results[name]["browser"] = {
                            "status": "needs_setup",
                            "error_code": browser.get("error_code") or "edge_environment_failed",
                            "requires_user": bool(browser.get("requires_user")),
                            "message": browser_data.get("message"),
                        }
                state().mcp_startup_errors.pop(name, None)
            except MCPClientError as exc:
                state().mcp_startup_errors[name] = exc.error_code
                results[name] = {"status": "unavailable", "error_code": exc.error_code}
            except Exception:
                state().mcp_startup_errors[name] = "mcp_recheck_failed"
                results[name] = {"status": "unavailable", "error_code": "mcp_recheck_failed"}
        return {"component": component, "servers": results}
    return {"component": component, "status": "scheduled", "message": "检测将在模型/数据库适配器中执行"}


@app.get("/api/settings")
async def get_settings_api(_: None = Depends(require_internal_token)) -> dict[str, Any]:
    values = dict(state().store.settings)
    values.pop("api_key", None)
    values.pop("github_token", None)
    configured: dict[str, bool] = {}
    for group, handles in state().credential_scopes.items():
        live = {handle for handle in handles if state().credentials.has(handle)}
        state().credential_scopes[group] = live
        configured[group] = bool(live)
    registry = getattr(state(), "model_credential_handles", {})
    lock = getattr(state(), "credential_registry_lock", None)
    if isinstance(registry, dict):
        def prune_registry() -> None:
            for profile_id, handle_id in list(registry.items()):
                if not state().credentials.has(
                    handle_id,
                    scope=f"model:{profile_id}",
                ):
                    registry.pop(profile_id, None)

        if lock is None:
            prune_registry()
        else:
            with lock:
                prune_registry()
    values["configured_in_session"] = configured
    return values


@app.put("/api/settings")
async def update_settings(
    request: SettingsUpdate,
    if_match: str | None = Header(default=None, alias="If-Match"),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    if if_match is None or not if_match.strip():
        raise HTTPException(
            status_code=428,
            detail={"error_code": "settings_if_match_required", "message": "保存设置必须携带 If-Match: settings_version"},
        )
    supplied = if_match.strip().strip('"')
    try:
        if int(supplied) != request.settings_version:
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": "settings_version_conflict",
                    "message": "If-Match 与 settings_version 不一致",
                    "current": state().store.settings.get("settings_version", 0),
                },
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error_code": "invalid_if_match", "message": "If-Match 必须是 settings_version"}) from exc
    try:
        values = _strip_secret_settings(request.values)
        _validate_scoring_settings(values)
        if state().db_mirror and state().db_mirror.available:
            try:
                persisted = state().db_mirror.persist_settings(request.settings_version, values)
                state().store.settings = dict(persisted)
                await _apply_runtime_setting_changes(values)
                return persisted
            except Exception as exc:
                current = getattr(exc, "current_version", None)
                if current is not None:
                    raise HTTPException(status_code=409, detail={"error_code": "settings_version_conflict", "current": current}) from exc
                # A transient database failure falls back only when explicitly
                # allowed; the response says so via the environment endpoint.
                if not state().settings.allow_in_memory_store:
                    raise HTTPException(status_code=503, detail={"error_code": "database_unavailable", "message": str(exc)}) from exc
        persisted = state().store.update_settings(request.settings_version, values)
        await _apply_runtime_setting_changes(values)
        return persisted
    except VersionConflict as exc:
        raise HTTPException(status_code=409, detail={"error_code": "settings_version_conflict", "current": exc.current}) from exc


@app.get("/api/models")
async def list_models(role: str | None = Query(default=None), _: None = Depends(require_internal_token)) -> list[ModelProfile]:
    if role and role not in {"chat", "embedding"}:
        raise HTTPException(status_code=422, detail="role must be chat or embedding")
    return state().store.list_profiles(role)


@app.post("/api/models", response_model=ModelProfile, status_code=201)
async def create_model_profile(payload: dict[str, Any], _: None = Depends(require_internal_token)) -> ModelProfile:
    try:
        profile = ModelProfile.model_validate(_strip_secret_settings(payload))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if profile.provider is Provider.OPENAI_COMPATIBLE:
        from app.core.model_gateway import canonicalize_openai_base_url

        try:
            profile.base_url = canonicalize_openai_base_url(profile.base_url)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        profile.credential_required = True
    if state().store.get_profile(profile.profile_id) is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "model_profile_immutable",
                "message": "已存在同名模型配置；请使用新的 profile_id 创建新版本",
            },
        )
    # Readiness is established only by the role-specific probe endpoint.  A
    # client-supplied `status=ready` must never bypass model/embedding gates.
    profile.status = "unprobed"
    profile.capabilities = {}
    profile.last_probe_at = None
    profile.error_code = None
    # Validate URL scheme/embedded credentials and role-specific gateway fields
    # before putting the profile in the store.  This prevents an invalid profile
    # from being saved and only failing later during a probe.
    try:
        _gateway_profile(profile)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    saved = state().store.save_profile(profile)
    if state().db_mirror and state().db_mirror.available:
        state().db_mirror.persist_model_profile(saved)
    return saved


@app.post("/api/models/ollama/scan")
async def scan_ollama(_: None = Depends(require_internal_token)) -> dict[str, Any]:
    try:
        from app.core.model_gateway import OllamaAdapter

        result = await asyncio.to_thread(OllamaAdapter().scan, state().settings.ollama_base_url)
        return result
    except Exception as exc:
        return {"status": "service_unreachable", "error_code": "ollama_unavailable", "detail": str(exc)}


@app.post("/api/models/{profile_id}/default", response_model=ModelProfile)
async def set_default_model_profile(
    profile_id: str,
    _: None = Depends(require_internal_token),
) -> ModelProfile:
    """Select a saved non-secret profile as the default for its own role."""

    profile = state().store.get_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="model profile not found")
    profile.default_for_role = True
    saved = state().store.save_profile(profile)
    if state().db_mirror and state().db_mirror.available:
        state().db_mirror.persist_model_profile(saved)
    return saved


@app.post("/api/models/{profile_id}/probe", response_model=ProbeResponse)
async def probe_model(profile_id: str, payload: dict[str, Any] | None = Body(default=None), _: None = Depends(require_internal_token)) -> ProbeResponse:
    profile = state().store.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="model profile not found")
    try:
        from app.core.model_gateway import CredentialUnavailable, ModelGateway

        gateway = ModelGateway(credential_store=state().credentials)
        credential_handle_id = _materialize_model_credential(
            profile.profile_id,
            str((payload or {}).get("credential_handle_id") or "") or None,
        )
        result = await asyncio.to_thread(
            gateway.probe,
            _gateway_profile(profile),
            credential_handle_id=credential_handle_id,
        )
        result = _probe_response(result)
    except CredentialUnavailable as exc:
        # Missing/expired handles are an expected user-action state, not a
        # provider probe crash.  Keep the profile unready and expose a stable
        # error code so the UI can ask for a new in-memory credential.
        result = ProbeResponse(
            role=profile.role,
            provider=profile.provider,
            status="credential_missing",
            base_url=profile.base_url,
            model_name=profile.model_name,
            checked_at=datetime.now(timezone.utc),
            error_code=exc.code,
            retryable=False,
        )
    except (TypeError, ValueError) as exc:
        result = ProbeResponse(
            role=profile.role,
            provider=profile.provider,
            status="probe_failed",
            base_url=profile.base_url,
            model_name=profile.model_name,
            checked_at=datetime.now(timezone.utc),
            error_code="profile_invalid",
            retryable=False,
        )
    except Exception as exc:
        result = ProbeResponse(
            role=profile.role,
            provider=profile.provider,
            status="probe_failed",
            base_url=profile.base_url,
            model_name=profile.model_name,
            checked_at=datetime.now(timezone.utc),
            error_code="probe_exception",
            retryable=True,
        )
    profile.status = result.status
    profile.dimension = result.dimension or profile.dimension
    profile.capabilities = result.capabilities
    profile.error_code = result.error_code
    profile.last_probe_at = result.checked_at
    state().store.save_profile(profile)
    if state().db_mirror and state().db_mirror.available:
        state().db_mirror.persist_model_profile(profile)
    return result


@app.post("/api/credentials/session")
async def create_credential_session(payload: dict[str, Any], _: None = Depends(require_internal_token)) -> dict[str, Any]:
    secret = payload.get("api_key") or payload.get("github_token")
    profile_id = str(payload.get("profile_id") or "").strip()
    scope_value = payload.get("scope")
    if not scope_value and profile_id:
        scope_value = f"model:{profile_id}"
    scope = str(scope_value or "")
    if profile_id and scope != f"model:{profile_id}":
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "credential_scope_mismatch",
                "message": "模型凭据 scope 与 profile_id 不匹配",
            },
        )
    if not secret and profile_id:
        try:
            secret = state().secret_store.get(profile_id)
        except SecretStoreUnavailable as exc:
            raise HTTPException(
                status_code=409,
                detail={"error_code": "credential_store_unavailable", "message": str(exc)},
            ) from exc
    if not secret or not scope:
        raise HTTPException(status_code=422, detail="secret and scope are required")
    try:
        if profile_id and bool(payload.get("persist", True)):
            state().secret_store.set(profile_id, str(secret))
        ttl_seconds = int(payload.get("ttl_seconds", 300))
        handle = state().credentials.put(str(secret), scope=scope, ttl_seconds=ttl_seconds)
    except SecretStoreUnavailable as exc:
        raise HTTPException(
            status_code=409,
            detail={"error_code": "credential_store_unavailable", "message": str(exc)},
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    scope_group = "github" if "github" in scope.lower() else "model"
    state().credential_scopes.setdefault(scope_group, set()).add(handle.handle_id)
    if profile_id:
        registry = getattr(state(), "model_credential_handles", None)
        lock = getattr(state(), "credential_registry_lock", None)

        def register_handle() -> None:
            if not isinstance(registry, dict):
                return
            previous = registry.get(profile_id)
            registry[profile_id] = handle.handle_id
            if previous and previous != handle.handle_id:
                state().credentials.revoke(previous)
                state().credential_scopes.setdefault("model", set()).discard(previous)

        if lock is None:
            register_handle()
        else:
            with lock:
                register_handle()
    return {"credential_handle_id": handle.handle_id, "scope": handle.scope, "expires_at": handle.expires_at_iso}


@app.get("/api/credentials/stored/{profile_id}")
async def stored_credential_status(
    profile_id: str,
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    try:
        stored = state().secret_store.has(profile_id)
        if not stored:
            return {"stored": False, "usable": False, "available": True}
        secret = state().secret_store.get(profile_id)
    except SecretStoreUnavailable as exc:
        return {
            "stored": False,
            "usable": False,
            "available": True,
            "needs_reentry": True,
            "message": str(exc),
        }
    return {
        "stored": bool(secret),
        "usable": bool(secret),
        "available": True,
        "needs_reentry": False,
    }


@app.delete("/api/credentials/stored/{profile_id}")
async def delete_stored_credential(
    profile_id: str,
    _: None = Depends(require_internal_token),
) -> dict[str, bool]:
    try:
        removed = state().secret_store.delete(profile_id)
    except SecretStoreUnavailable as exc:
        raise HTTPException(
            status_code=409,
            detail={"error_code": "credential_store_unavailable", "message": str(exc)},
        ) from exc
    registry = getattr(state(), "model_credential_handles", None)
    lock = getattr(state(), "credential_registry_lock", None)

    def revoke_active_handle() -> None:
        if not isinstance(registry, dict):
            return
        handle_id = registry.pop(profile_id, None)
        if handle_id:
            state().credentials.revoke(handle_id)
            state().credential_scopes.setdefault("model", set()).discard(handle_id)

    if lock is None:
        revoke_active_handle()
    else:
        with lock:
            revoke_active_handle()
    return {"removed": removed}


@app.delete("/api/credentials/session/{handle_id}")
async def delete_credential_session(handle_id: str, _: None = Depends(require_internal_token)) -> dict[str, bool]:
    revoked = state().credentials.revoke(handle_id)
    for handles in state().credential_scopes.values():
        handles.discard(handle_id)
    registry = getattr(state(), "model_credential_handles", None)
    lock = getattr(state(), "credential_registry_lock", None)

    def forget_handle() -> None:
        if not isinstance(registry, dict):
            return
        for profile_id, active_handle in list(registry.items()):
            if active_handle == handle_id:
                registry.pop(profile_id, None)

    if lock is None:
        forget_handle()
    else:
        with lock:
            forget_handle()
    return {"revoked": revoked}


async def set_consent(
    scope: str,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    """Record a user decision for an external/risk-gated operation.

    Consent is non-secret audit state and survives restart. Model API keys use
    the separate machine-local encrypted store; GitHub tokens remain
    process-local. Neither credential type enters this persistence path.
    """
    aliases = {
        "external-model": "external_model",
        "external_model": "external_model",
        "post-time-risk": "post_time_risk",
        "post_time_risk": "post_time_risk",
    }
    normalized_scope = aliases.get(scope)
    if normalized_scope is None:
        raise HTTPException(status_code=404, detail={"error_code": "unknown_consent_scope", "message": "未知同意范围"})
    body = payload or {}
    if "granted" not in body:
        raise HTTPException(status_code=422, detail={"error_code": "consent_decision_required", "message": "必须提供 granted"})
    granted = bool(body.get("granted"))
    policy_version = str(body.get("policy_version") or "consent-v1")
    task_id = str(body.get("task_id") or "") or None
    if task_id and state().store.get_task(task_id) is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "task_not_found", "message": "同意记录引用的任务不存在"},
        )
    key = f"{normalized_scope}:{task_id or 'global'}"
    record = {
        "scope": normalized_scope,
        "task_id": task_id,
        "granted": granted,
        "policy_version": policy_version,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if state().db_mirror:
        persisted = state().db_mirror.persist_consent(
            normalized_scope,
            granted,
            policy_version=policy_version,
            task_id=task_id,
            metadata={"source": "api"},
        )
        if persisted is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "consent_persistence_unavailable",
                    "message": "同意状态未能持久化，本次操作未生效",
                },
            )
        record.update(persisted)
    state().consents[key] = record
    return {"status": "granted" if granted else "revoked", **record}


# Keep the two explicit paths from the design document as aliases; clients do
# not need to interpolate a scope string and OpenAPI shows both operations.
@app.post("/api/consents/external-model")
async def consent_external_model(
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    return await set_consent("external-model", payload, None)


@app.post("/api/consents/post-time-risk")
async def consent_post_time_risk(
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    return await set_consent("post-time-risk", payload, None)


# Register the parameterized compatibility route *after* the two explicit
# routes above.  Starlette matches routes in registration order, so placing
# ``/api/consents/{scope}`` first would make the documented explicit paths
# unreachable while still leaving misleading operations in OpenAPI.
app.add_api_route(
    "/api/consents/{scope}",
    set_consent,
    methods=["POST"],
    name="set_consent",
)


@app.get("/api/skills")
async def get_skill(_: None = Depends(require_internal_token)) -> dict[str, Any]:
    current = read_skill(state().settings.data_root)
    state().current_skill = str(current.get("content", ""))
    state().skill_versions = list_skill_versions(state().settings.data_root)
    return {
        "exists": bool(current["content"]),
        "content": current["content"],
        "current_version": current["version"],
        "version_count": len(state().skill_versions),
        "path": current["path"],
        "content_hash": current["content_hash"],
        "deletion_supported": False,
    }


@app.put("/api/skills")
async def put_skill(
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    try:
        body = payload or {}
        content = body.get("content")
        if not isinstance(content, str) or not content.strip():
            raise HTTPException(status_code=422, detail={"error_code": "skill_content_required", "message": "Skill 内容不能为空"})
        current = read_skill(state().settings.data_root)
        expected = int(body.get("expected_version", current["version"]))
        result = save_skill(state().settings.data_root, content, expected_version=expected)
        state().current_skill = content
        state().skill_versions = list_skill_versions(state().settings.data_root)
        return {"status": "saved", **result, "message": "Skill 已保存；第一版不提供删除操作"}
    except HTTPException:
        raise
    except (ValueError, RuntimeError, OSError) as exc:
        code = "skill_version_conflict" if "conflict" in str(exc).lower() else "invalid_skill"
        raise HTTPException(status_code=409 if code.endswith("conflict") else 422, detail={"error_code": code, "message": str(exc)}) from exc


@app.get("/api/skills/versions")
async def get_skill_versions(_: None = Depends(require_internal_token)) -> dict[str, Any]:
    versions = list(reversed(list_skill_versions(state().settings.data_root)))
    return {"versions": versions, "total": len(versions)}


@app.post("/api/skills/rollback")
async def rollback_skill(
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    try:
        body = payload or {}
        requested = body.get("version") or body.get("version_id")
        if isinstance(requested, str) and requested.startswith("v"):
            requested = requested[1:]
        version = int(requested)
        result = rollback_skill_file(state().settings.data_root, version)
        state().current_skill = result["content"]
        state().skill_versions = list_skill_versions(state().settings.data_root)
        return {"status": "rolled_back", **result}
    except (TypeError, ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=422, detail={"error_code": "skill_rollback_failed", "message": str(exc)}) from exc


@app.post("/api/jobs", response_model=dict[str, str])
async def create_job(request: JobCreateRequest, _: None = Depends(require_internal_token)) -> dict[str, str]:
    app_state = state()
    job_id = app_state.store.save_job(request)
    snapshot_id = app_state.db_mirror.persist_job(job_id, request) if app_state.db_mirror else None
    return {"job_id": job_id, "job_snapshot_id": snapshot_id or f"snapshot_{job_id}"}


def _persist_discovered_job(
    job: JobInput,
    *,
    authoritative_recruiter: bool = False,
) -> tuple[str, str | None, JobInput]:
    """Upsert a discovered URL without mixing recruiter data between sources.

    The BOSS search card renders the HR name and activity together and is the
    authoritative source for that pair.  Detail panes are asynchronous, so a
    detail parse must not replace a concrete card value.  Conversely, a fresh
    card render *must* be allowed to replace a stale value saved by an earlier
    search because activity naturally changes over time.
    """

    existing_pair = state().store.find_job_by_source_url(job.source_url or "")
    if existing_pair is None:
        job_id = state().store.save_job(job)
        current = job
    else:
        job_id, existing = existing_pair
        existing_activity = str(existing.hr_activity or "").strip()
        incoming_activity = str(job.hr_activity or "").strip()
        existing_name = str(existing.hr_name or "").strip()
        incoming_name = str(job.hr_name or "").strip()
        # The list card displays HR name and activity as one authoritative
        # label. Detail-pane content is asynchronous and can still describe
        # the previously selected card, so never overwrite a concrete card
        # value with a later detail reparse.
        if (
            authoritative_recruiter
            and incoming_name
            and incoming_activity not in {"", "活跃时间待解析"}
        ):
            merged_hr_name = job.hr_name
            merged_activity = job.hr_activity
        elif existing_name or existing_activity not in {"", "活跃时间待解析"}:
            # Preserve the previous pair as a unit. Combining a newly parsed
            # activity with an old name (or vice versa) fabricates a recruiter
            # state that was never present on one card.
            merged_hr_name = existing.hr_name
            merged_activity = existing.hr_activity
        else:
            merged_hr_name = job.hr_name
            merged_activity = job.hr_activity or existing.hr_activity
        current = JobInput(
            title=job.title or existing.title,
            company=job.company or existing.company,
            city=job.city or existing.city,
            salary=job.salary or existing.salary,
            responsibilities=job.responsibilities or existing.responsibilities,
            requirements=job.requirements or existing.requirements,
            skills=job.skills or existing.skills,
            hr_name=merged_hr_name,
            hr_activity=merged_activity,
            posted_at=job.posted_at or existing.posted_at,
            posted_at_label=job.posted_at_label or existing.posted_at_label,
            source_url=job.source_url or existing.source_url,
            raw_text=job.raw_text or existing.raw_text,
        )
        state().store.update_job(job_id, current)
    snapshot_id = state().db_mirror.persist_job(job_id, current) if state().db_mirror else None
    return job_id, snapshot_id, current


def _job_from_mcp(value: dict[str, Any], *, fallback_url: str | None = None) -> JobInput:
    payload = dict(value)
    payload["source_url"] = payload.pop("url", None) or payload.get("source_url") or fallback_url
    if not payload.get("posted_at") and not payload.get("posted_at_label"):
        payload["posted_at_label"] = "发布时间未知"
    allowed = set(JobInput.model_fields)
    return JobInput.model_validate({key: item for key, item in payload.items() if key in allowed})


async def _enhance_job_detail_with_chat_model(
    job: JobInput,
    *,
    chat_profile_id: str | None = None,
    credential_handle_id: str | None = None,
    feedback: str | None = None,
    previous_job: JobInput | None = None,
) -> tuple[JobInput, dict[str, Any]]:
    """Enhance a rule-parsed JD while preserving a usable fallback on failure."""

    app_state = state()
    store = getattr(app_state, "store", None)
    if store is None:
        return job, rule_fallback_metadata(
            warning="聊天模型配置不可用，已保留规则解析结果",
            error_code="model_store_unavailable",
        )
    profile = store.get_profile(chat_profile_id) if chat_profile_id else next(
        (
            item
            for item in store.list_profiles(ModelRole.CHAT.value)
            if item.default_for_role
        ),
        None,
    )
    if profile is None:
        return job, rule_fallback_metadata(
            warning="未配置默认聊天模型，当前使用规则解析结果",
            error_code="chat_profile_not_found",
        )
    if profile.role is not ModelRole.CHAT:
        return job, rule_fallback_metadata(
            warning="所选配置不是聊天模型，当前使用规则解析结果",
            error_code="invalid_chat_profile_role",
            profile=profile,
        )
    if profile.status != "ready":
        return job, rule_fallback_metadata(
            warning="所选聊天模型尚未检测成功，当前使用规则解析结果",
            error_code="model_not_ready",
            profile=profile,
        )
    if profile.provider is Provider.OPENAI_COMPATIBLE:
        credentials = getattr(app_state, "credentials", None)
        credential_handle_id = _materialize_model_credential(
            profile.profile_id,
            credential_handle_id,
        )
        if (
            credentials is None
            or not credential_handle_id
            or not credentials.has(
                credential_handle_id,
                scope=f"model:{profile.profile_id}",
            )
        ):
            return job, rule_fallback_metadata(
                warning="云端模型 API Key 未配置或无法恢复，当前使用规则解析结果",
                error_code="credential_missing",
                profile=profile,
            )
        settings = getattr(app_state, "settings", None)
        requires_consent = bool(
            getattr(settings, "external_model_consent_required", True)
        )
        consents = getattr(app_state, "consents", {})
        if requires_consent and not consents.get("external_model:global", {}).get(
            "granted", False
        ):
            return job, rule_fallback_metadata(
                warning="尚未确认向外部模型发送 JD，当前使用规则解析结果",
                error_code="needs_external_model_consent",
                profile=profile,
            )
    gateway = getattr(app_state, "model_gateway", None)
    if gateway is None:
        return job, rule_fallback_metadata(
            warning="模型网关不可用，当前使用规则解析结果",
            error_code="model_gateway_unavailable",
            profile=profile,
        )
    try:
        outcome = await asyncio.to_thread(
            enhance_job_with_model,
            job,
            profile=profile,
            gateway=gateway,
            credential_handle_id=credential_handle_id,
            feedback=feedback,
            previous_job=previous_job,
        )
        return outcome.job, outcome.metadata
    except Exception as exc:
        error_code = str(getattr(exc, "code", "model_parse_failed"))
        error_message = str(
            getattr(exc, "message", None) or str(exc) or "模型返回格式不符合要求"
        ).strip()
        if isinstance(exc, json.JSONDecodeError):
            error_message = "模型没有返回完整的 JSON 结构"
        elif isinstance(exc, TypeError):
            error_message = str(exc) or "模型返回字段类型不正确"
        elif isinstance(exc, ValueError):
            error_message = str(exc) or "模型返回内容未通过 JD 原文校验"
        return job, rule_fallback_metadata(
            warning=(
                f"聊天模型解析失败：{error_message[:180]}；"
                "已安全保留规则解析结果，可直接再次使用大模型解析"
            ),
            error_code=error_code,
            profile=profile,
            retryable=bool(getattr(exc, "retryable", True)),
            error_message=error_message[:500],
        )


def _prune_pending_ocr(app_state: Any, *, now: float | None = None) -> None:
    """Drop unconfirmed OCR text after 30 minutes without ever storing images."""

    current = float(now if now is not None else time.time())
    lock = app_state.pending_ocr_lock
    with lock:
        expired = [
            ocr_id
            for ocr_id, item in app_state.pending_ocr.items()
            if current - float(item.get("created_at", 0)) > 1800
        ]
        for ocr_id in expired:
            app_state.pending_ocr.pop(ocr_id, None)


@app.post("/api/jobs/screenshot/ocr")
async def ocr_job_screenshot(
    file: UploadFile = File(...),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    """Recognize one screenshot without creating or changing a job record."""

    data = await file.read(MAX_SCREENSHOT_BYTES + 1)
    if len(data) > MAX_SCREENSHOT_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error_code": "image_too_large", "message": "岗位截图不能超过 10 MB"},
        )
    try:
        result = await asyncio.to_thread(state().screenshot_ocr.recognize, data)
    except ScreenshotOCRError as exc:
        status_code = 503 if exc.code in {
            "ocr_dependency_missing",
            "ocr_model_not_installed",
            "ocr_model_load_failed",
            "ocr_model_permission_denied",
            "ocr_memory_exhausted",
            "ocr_runtime_incompatible",
        } else 422
        raise HTTPException(
            status_code=status_code,
            detail={"error_code": exc.code, "message": str(exc)},
        ) from exc

    app_state = state()
    _prune_pending_ocr(app_state)
    ocr_id = f"ocr_{uuid.uuid4().hex}"
    created_at = time.time()
    with app_state.pending_ocr_lock:
        app_state.pending_ocr[ocr_id] = {
            "created_at": created_at,
            "text": result["text"],
            "text_hash": hashlib.sha256(result["text"].encode("utf-8")).hexdigest(),
            "model": result["model"],
        }
    return {
        "status": "needs_confirmation",
        "ocr_id": ocr_id,
        "text": result["text"],
        "lines": result["lines"],
        "model": result["model"],
        "image": {
            "format": result["image_format"],
            "width": result["width"],
            "height": result["height"],
        },
        "needs_confirmation": True,
        "job_created": False,
        "inferred": infer_job_metadata(result["text"]),
    }


@app.post("/api/jobs/screenshot/confirm")
async def confirm_job_screenshot(
    request: ScreenshotConfirmRequest,
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    """Create a job only after the user has corrected and confirmed OCR text."""

    corrected_text = request.text.strip()
    if not corrected_text:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "ocr_text_empty", "message": "校正后的岗位文字不能为空"},
        )
    app_state = state()
    _prune_pending_ocr(app_state)
    with app_state.pending_ocr_lock:
        pending = app_state.pending_ocr.get(request.ocr_id)
    if pending is None:
        # The image intentionally never leaves the Streamlit page session and
        # OCR staging is in-memory. If FastAPI restarted (or a response was
        # lost after a successful confirmation), the browser still owns the
        # user-corrected OCR text. Recover from that text instead of forcing an
        # expensive image upload/OCR pass again.
        pending = {
            "created_at": time.time(),
            "text": corrected_text,
            "text_hash": hashlib.sha256(corrected_text.encode("utf-8")).hexdigest(),
            "model": "client_recovered_ocr_text",
            "recovered": True,
        }
        with app_state.pending_ocr_lock:
            app_state.pending_ocr[request.ocr_id] = pending
    cached_response = pending.get("confirmed_response")
    if isinstance(cached_response, dict):
        # Confirmation is idempotent. This covers an HTTP timeout after the
        # server already persisted the job and lets the next click restore the
        # OCR fallback screen without creating a duplicate job.
        return {**cached_response, "confirmation_replayed": True}
    inferred = infer_job_metadata(corrected_text)
    parsed = parse_job_detail_text(
        str(request.title or inferred.get("title") or "截图岗位").strip(),
        corrected_text,
        company=request.company or inferred.get("company"),
        city=request.city or inferred.get("city"),
        hr_name=request.hr_name or inferred.get("hr_name"),
        hr_activity=request.hr_activity or inferred.get("hr_activity"),
    ).model_copy(update={"salary": request.salary or inferred.get("salary")})
    parsed, parser = await _enhance_job_detail_with_chat_model(
        parsed,
        chat_profile_id=request.chat_profile_id,
        credential_handle_id=request.credential_handle_id,
    )
    job_id, snapshot_id, parsed = _persist_discovered_job(parsed)
    app_state.job_parse_confirmations[job_id] = False
    response = {
        "status": "ready",
        "source": "screenshot_ocr_confirmed",
        "job_id": job_id,
        "job_snapshot_id": snapshot_id,
        "ocr_id": request.ocr_id,
        "ocr_text_changed": (
            hashlib.sha256(corrected_text.encode("utf-8")).hexdigest()
            != pending["text_hash"]
        ),
        "parser": parser,
        "model_fallback": parser.get("parser_mode") != "model_enhanced",
        "model_fallback_message": (
            "大模型增强解析失败或不可用，已保留 OCR 校正文字和规则解析结果；"
            "配置模型后可直接重新解析，无需再次上传截图"
            if parser.get("parser_mode") != "model_enhanced"
            else None
        ),
        "job": {**_job_summary(job_id, parsed), "raw_text": parsed.raw_text},
    }
    with app_state.pending_ocr_lock:
        staged = app_state.pending_ocr.get(request.ocr_id)
        if staged is not None:
            staged["confirmed_response"] = response
            staged["confirmed_at"] = time.time()
    return response


def _require_post_time_risk_consent() -> None:
    if not state().consents.get("post_time_risk:global", {}).get("granted", False):
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "post_time_risk_consent_required",
                "message": "首次启用 BOSS 页面访问与岗位采集前，必须确认账号访问风险",
                "requires_user": True,
            },
        )


@app.post("/api/jobs/from-url")
async def job_from_url(request: JobUrlRequest, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    try:
        validate_boss_url(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if request.raw_text:
        title = (request.title or "").strip()
        if not title:
            raise HTTPException(status_code=422, detail={"error_code": "job_title_required", "message": "粘贴职位详情时必须提供岗位名称"})
        parsed = parse_job_detail_text(
            title,
            request.raw_text,
            company=request.company,
            city=request.city,
            source_url=request.url,
            posted_label=request.posted_time_label,
        )
        parsed, parser = await _enhance_job_detail_with_chat_model(
            parsed,
            chat_profile_id=request.chat_profile_id,
            credential_handle_id=request.credential_handle_id,
        )
        job_id, snapshot_id, parsed = _persist_discovered_job(parsed)
        state().job_parse_confirmations[job_id] = False
        return {
            "status": "ready",
            "source": "provided_text",
            "job_id": job_id,
            "job_snapshot_id": snapshot_id,
            "parser": parser,
            "job": {**_job_summary(job_id, parsed), "raw_text": parsed.raw_text},
        }

    _require_post_time_risk_consent()
    health = await _ensure_mcp_server("boss")
    if health.get("status") != "ready":
        return {
            "status": "waiting_user",
            "url": request.url,
            "error_code": health.get("error_code", "mcp_unavailable"),
            "requires_user": True,
            "message": "BOSS MCP 暂不可用，请完成本地依赖配置后重新检测",
        }
    try:
        result = await state().mcp_manager.call_tool(
            "boss", "boss_get_job_detail", {"url": request.url}
        )
    except MCPClientError as exc:
        return {
            "status": "waiting_user",
            "url": request.url,
            "error_code": exc.error_code,
            "requires_user": True,
            "message": "BOSS MCP 调用失败，请重新检测后再试",
        }
    if not result.get("ok"):
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        return {
            "status": "waiting_user",
            "url": request.url,
            "error_code": result.get("error_code") or "boss_detail_unavailable",
            "retryable": bool(result.get("retryable")),
            "requires_user": bool(result.get("requires_user")),
            "message": data.get("message") or "请在可见 Edge 中完成登录或验证后重新解析",
        }
    try:
        parsed = _job_from_mcp(dict(result.get("data") or {}), fallback_url=request.url)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail={"error_code": "invalid_boss_detail", "message": "BOSS 返回的职位详情结构无效"},
        ) from exc
    parsed, parser = await _enhance_job_detail_with_chat_model(
        parsed,
        chat_profile_id=request.chat_profile_id,
        credential_handle_id=request.credential_handle_id,
    )
    job_id, snapshot_id, parsed = _persist_discovered_job(parsed)
    state().job_parse_confirmations[job_id] = False
    return {
        "status": "ready",
        "source": "boss_mcp",
        "job_id": job_id,
        "job_snapshot_id": snapshot_id,
        "parser": parser,
        "job": {**_job_summary(job_id, parsed), "raw_text": parsed.raw_text},
    }


def _job_summary(job_id: str, job: JobInput, created_at: float | None = None, updated_at: float | None = None) -> dict[str, Any]:
    """Return a redacted, list-friendly job representation."""
    return {
        "job_id": job_id,
        "title": job.title,
        "company": job.company,
        "city": job.city,
        "salary": decode_boss_obfuscated_text(job.salary) or None,
        "responsibilities": job.responsibilities,
        "requirements": job.requirements,
        "skills": job.skills,
        "hr_name": job.hr_name,
        "hr_activity": job.hr_activity or "活跃时间待解析",
        "posted_at": job.posted_at.isoformat() if job.posted_at else None,
        "posted_at_label": job.posted_at_label or ("发布时间未知" if not job.posted_at else None),
        "source_url": job.source_url,
        "created_at": datetime.fromtimestamp(created_at, timezone.utc).isoformat() if created_at else None,
        "updated_at": datetime.fromtimestamp(updated_at, timezone.utc).isoformat() if updated_at else None,
    }


def _decode_offset(cursor: str | None) -> int:
    if not cursor:
        return 0
    try:
        offset = int(cursor)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail={"error_code": "invalid_cursor", "message": "分页游标无效"}) from exc
    if offset < 0:
        raise HTTPException(status_code=422, detail={"error_code": "invalid_cursor", "message": "分页游标无效"})
    return offset


def _compact_search_text(value: Any) -> str:
    """Normalize spacing/case without weakening the actual BOSS query."""

    return re.sub(r"\s+", "", str(value or "")).casefold()


def _matches_hr_activity(value: str | None, requested: str | None) -> bool:
    """Match the user-facing HR activity groups without inventing precision."""

    wanted = str(requested or "").strip()
    if not wanted or wanted == "不限":
        return True
    activity = _compact_search_text(value or "活跃时间待解析")
    groups = {
        "在线或刚刚活跃": ("在线", "刚刚"),
        "今日活跃": ("在线", "刚刚", "今日", "今天"),
        "近3天活跃": ("在线", "刚刚", "今日", "今天", "昨天", "1天", "2天", "3天"),
        "本周活跃": ("在线", "刚刚", "今日", "今天", "昨天", "本周", "1天", "2天", "3天", "4天", "5天", "6天", "7天"),
        "活跃时间待解析": ("活跃时间待解析",),
    }
    terms = groups.get(wanted, (_compact_search_text(wanted),))
    return any(_compact_search_text(term) in activity for term in terms)


@app.post("/api/jobs/search")
async def search_jobs(
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    """Search BOSS through MCP, with the durable local cache as a fallback."""
    body = payload or {}
    title_query = str(body.get("title") or body.get("keyword") or "").strip().lower()
    city_query = str(body.get("city") or "").strip().lower()
    company_query = str(body.get("company") or "").strip().lower()
    hr_activity_filter = str(body.get("hr_activity") or "").strip() or None
    try:
        limit = min(max(int(body.get("limit", 20) or 20), 1), 20)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail={"error_code": "invalid_limit", "message": "limit 必须是 1-20 的整数"}) from exc
    if not title_query or not city_query:
        raise HTTPException(status_code=422, detail={"error_code": "missing_search_fields", "message": "职位名称和城市为必填项"})
    _require_post_time_risk_consent()
    cursor = str(body.get("cursor") or "").strip() or None
    known_only = bool(body.get("known_posted_at", False))
    max_age_days = body.get("max_age_days", body.get("posted_within_days", body.get("days")))
    posted_after = body.get("posted_after")
    posted_before = body.get("posted_before")
    try:
        posted_after_dt = datetime.fromisoformat(str(posted_after).replace("Z", "+00:00")) if posted_after else None
        posted_before_dt = datetime.fromisoformat(str(posted_before).replace("Z", "+00:00")) if posted_before else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error_code": "invalid_time_filter", "message": "日期筛选格式无效"}) from exc
    if posted_after_dt is not None and posted_after_dt.tzinfo is None:
        posted_after_dt = posted_after_dt.replace(tzinfo=timezone.utc)
    if posted_before_dt is not None and posted_before_dt.tzinfo is None:
        posted_before_dt = posted_before_dt.replace(tzinfo=timezone.utc)
    try:
        max_age_days = int(max_age_days) if max_age_days is not None else None
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail={"error_code": "invalid_time_filter", "message": "max_age_days 必须是整数"}) from exc
    if max_age_days is not None and max_age_days < 0:
        raise HTTPException(status_code=422, detail={"error_code": "invalid_time_filter", "message": "max_age_days 不能为负数"})

    imported_ids: set[str] | None = None
    remote_next_cursor: str | None = None
    remote_load_pending = False
    remote_exhausted = False
    remote_collection: dict[str, Any] = {}
    remote_error_code: str | None = None
    remote_message: str | None = None
    health = await _ensure_mcp_server("boss")
    if health.get("status") == "ready":
        arguments: dict[str, Any] = {
            "title": str(body.get("title") or body.get("keyword") or "").strip(),
            "city": str(body.get("city") or "").strip(),
            "limit": limit,
        }
        if body.get("city_code"):
            arguments["city_code"] = str(body["city_code"])
        if cursor:
            arguments["cursor"] = cursor
        try:
            remote = await state().mcp_manager.call_tool("boss", "boss_search_jobs", arguments)
        except MCPClientError as exc:
            remote_error_code = exc.error_code
            remote_message = "BOSS 实时采集失败，请保持岗位采集 Edge 打开后重试"
        else:
            if remote.get("ok") and isinstance(remote.get("data"), dict):
                imported_ids = set()
                remote_data = dict(remote["data"])
                remote_next_cursor = str(remote_data.get("next_cursor") or "").strip() or None
                remote_load_pending = bool(remote_data.get("load_pending"))
                remote_exhausted = bool(remote_data.get("exhausted"))
                if isinstance(remote_data.get("collection"), dict):
                    remote_collection = dict(remote_data["collection"])
                for item in remote_data.get("items", []):
                    if not isinstance(item, dict):
                        continue
                    try:
                        parsed = _job_from_mcp(item)
                        job_id, _, _ = _persist_discovered_job(
                            parsed,
                            authoritative_recruiter=True,
                        )
                    except (TypeError, ValueError):
                        continue
                    imported_ids.add(job_id)
            else:
                remote_error_code = str(remote.get("error_code") or "boss_search_unavailable")
                data = remote.get("data") if isinstance(remote.get("data"), dict) else {}
                remote_message = data.get("message") or "请在可见 Edge 中完成登录或验证后重试"
    else:
        remote_error_code = str(health.get("error_code") or "mcp_unavailable")
        remote_message = "BOSS MCP 暂不可用，请重新检测后重试"

    # Login, verification and retained-session errors are not cache-fallback
    # conditions. Returning historical jobs here made an explicitly logged-out
    # page look like a successful fresh search and allowed stale cards to be
    # selected. Keep the result empty until the user completes the visible
    # browser action and manually submits the same search again.
    if imported_ids is None and remote_error_code:
        return {
            "status": "waiting_user",
            "source": "boss_mcp",
            "jobs": [],
            "total": 0,
            "next_cursor": None,
            "has_more": False,
            "load_pending": False,
            "exhausted": False,
            "collection": {"status": "waiting_user", "batch_size": 0},
            "error_code": remote_error_code,
            "requires_user": True,
            "message": remote_message,
        }

    now = datetime.now(timezone.utc)
    matches: list[tuple[int, dict[str, Any]]] = []
    for job_id, job, created_at, updated_at in state().store.list_jobs():
        if imported_ids is not None and job_id not in imported_ids:
            continue
        normalized_title_query = _compact_search_text(title_query)
        normalized_city_query = _compact_search_text(city_query)
        normalized_company_query = _compact_search_text(company_query)
        normalized_job_title = _compact_search_text(job.title)
        normalized_job_city = _compact_search_text(job.city)
        normalized_job_company = _compact_search_text(job.company)
        normalized_job_text = _compact_search_text(job.raw_text)
        if imported_ids is None and (
            normalized_title_query
            and normalized_title_query not in normalized_job_title
            and normalized_title_query not in normalized_job_text
        ):
            continue
        if normalized_city_query and normalized_city_query not in normalized_job_city:
            continue
        if normalized_company_query and normalized_company_query not in normalized_job_company:
            continue
        if not _matches_hr_activity(job.hr_activity, hr_activity_filter):
            continue
        if known_only and job.posted_at is None:
            continue
        if job.posted_at is not None:
            posted_value = job.posted_at if job.posted_at.tzinfo else job.posted_at.replace(tzinfo=timezone.utc)
            if posted_after_dt and posted_value < posted_after_dt:
                continue
            if posted_before_dt and posted_value > posted_before_dt:
                continue
        if max_age_days is not None and job.posted_at is not None:
            posted = job.posted_at
            if posted.tzinfo is None:
                posted = posted.replace(tzinfo=timezone.utc)
            if (now - posted).total_seconds() > max_age_days * 86400:
                continue
        # Deterministic relevance score, then newest update and id as ties.
        haystack = _compact_search_text(
            " ".join([job.title, job.company or "", job.city or "", job.raw_text or ""])
        )
        score = 0
        if normalized_title_query:
            score += 50 if normalized_title_query in normalized_job_title else (10 if normalized_title_query in haystack else 0)
        if normalized_city_query and normalized_city_query in normalized_job_city:
            score += 20
        if normalized_company_query and normalized_company_query in normalized_job_company:
            score += 20
        if job.posted_at is not None:
            score += 1
        matches.append((score, _job_summary(job_id, job, created_at, updated_at)))
    matches.sort(key=lambda item: (-item[0], item[1]["updated_at"] or "", item[1]["job_id"]))
    if imported_ids is not None:
        page = [item[1] for item in matches[:limit]]
        next_cursor = remote_next_cursor
        source = "boss_mcp"
    else:
        offset = _decode_offset(cursor)
        page = [item[1] for item in matches[offset : offset + limit]]
        next_offset = offset + len(page)
        next_cursor = str(next_offset) if next_offset < len(matches) else None
        source = "local_cache"
    return {
        "status": "ready" if remote_error_code is None else "waiting_user",
        "source": source,
        "jobs": page,
        "total": len(matches),
        "next_cursor": next_cursor,
        "has_more": next_cursor is not None,
        "load_pending": remote_load_pending,
        "exhausted": remote_exhausted,
        "collection": remote_collection or {
            "status": (
                "complete"
                if remote_exhausted
                else ("waiting_for_more" if remote_load_pending else "batch_ready")
            ),
            "batch_size": len(page),
        },
        "error_code": remote_error_code,
        "requires_user": remote_error_code is not None,
        "message": remote_message,
    }


@app.post("/api/jobs/browser/close")
async def close_job_browser(_: None = Depends(require_internal_token)) -> dict[str, Any]:
    health = await _ensure_mcp_server("boss")
    if health.get("status") != "ready":
        raise HTTPException(
            status_code=503,
            detail={"error_code": "mcp_unavailable", "message": "BOSS MCP 暂不可用"},
        )
    try:
        result = await state().mcp_manager.call_tool("boss", "boss_close_browser", {})
    except MCPClientError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error_code": exc.error_code, "message": "关闭 BOSS 浏览器失败"},
        ) from exc
    if not result.get("ok"):
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": result.get("error_code") or "boss_close_failed",
                "message": "关闭 BOSS 浏览器失败",
            },
        )
    return dict(result.get("data") or {"status": "closed"})


@app.post("/api/jobs/browser/open")
async def open_job_in_collection_browser(
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    """Open a job in the authenticated Edge used by the BOSS collector."""

    url = str((payload or {}).get("url") or "").strip()
    try:
        validate_boss_url(url)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "invalid_boss_url", "message": str(exc)},
        ) from exc
    health = await _ensure_mcp_server("boss")
    if health.get("status") != "ready":
        raise HTTPException(
            status_code=503,
            detail={"error_code": "mcp_unavailable", "message": "BOSS MCP 暂不可用"},
        )
    try:
        result = await state().mcp_manager.call_tool(
            "boss", "boss_open_job_page", {"url": url}
        )
    except MCPClientError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": exc.error_code,
                "message": "岗位采集 Edge 暂时不可用，请保持搜索窗口打开后重试",
            },
        ) from exc
    if not result.get("ok"):
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": result.get("error_code") or "boss_open_failed",
                "message": data.get("message") or "请先打开岗位采集 Edge 并完成登录",
            },
        )
    return dict(result.get("data") or {"status": "opened", "url": url})


@app.get("/api/jobs/history")
async def jobs_history(
    limit: int = Query(default=20, ge=1, le=20),
    cursor: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    jobs = state().store.list_jobs()
    offset = _decode_offset(cursor)
    page = jobs[offset : offset + limit]
    next_offset = offset + len(page)
    return {
        "jobs": [_job_summary(job_id, job, created_at, updated_at) for job_id, job, created_at, updated_at in page],
        "total": len(jobs),
        "next_cursor": str(next_offset) if next_offset < len(jobs) else None,
        "has_more": next_offset < len(jobs),
    }


def _edited_job_list(
    body: dict[str, Any],
    key: str,
    current: list[str],
    *,
    max_item_length: int,
) -> list[str]:
    """Validate one user-edited JD array without silently coercing objects."""

    if key not in body:
        return list(current)
    value = body.get(key)
    if not isinstance(value, list):
        raise HTTPException(
            status_code=422,
            detail={"error_code": "invalid_job_correction", "message": f"{key} 必须是字符串数组"},
        )
    if len(value) > 100:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "invalid_job_correction", "message": f"{key} 最多包含 100 项"},
        )
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise HTTPException(
                status_code=422,
                detail={"error_code": "invalid_job_correction", "message": f"{key} 只能包含字符串"},
            )
        text = item.strip()
        if not text:
            continue
        if len(text) > max_item_length:
            raise HTTPException(
                status_code=422,
                detail={"error_code": "invalid_job_correction", "message": f"{key} 存在过长条目"},
            )
        if text not in result:
            result.append(text)
    return result


def _apply_job_corrections(existing: JobInput, body: dict[str, Any]) -> JobInput:
    """Apply explicit human corrections while preserving unrelated fields."""

    raw_value = body.get("raw_text", existing.raw_text or "")
    if not isinstance(raw_value, str):
        raise HTTPException(
            status_code=422,
            detail={"error_code": "invalid_raw_text", "message": "raw_text 必须是字符串"},
        )
    raw_text = raw_value.strip()
    if not raw_text:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "job_text_unavailable", "message": "完整 JD 原文不能为空"},
        )
    if len(raw_text) > 200_000:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "job_text_too_large", "message": "完整 JD 原文不能超过 20 万字符"},
        )
    responsibilities = _edited_job_list(
        body,
        "edited_responsibilities",
        existing.responsibilities,
        max_item_length=1000,
    )
    requirements = _edited_job_list(
        body,
        "edited_requirements",
        existing.requirements,
        max_item_length=1000,
    )
    skills = _edited_job_list(
        body,
        "edited_skills",
        existing.skills,
        max_item_length=100,
    )
    return existing.model_copy(
        update={
            "raw_text": raw_text,
            "responsibilities": responsibilities,
            "requirements": requirements,
            "skills": skills,
        }
    )


def _updated_job_payload(job_id: str, job: JobInput) -> dict[str, Any]:
    metadata = next(
        (item for item in state().store.list_jobs() if item[0] == job_id), None
    )
    return {
        **(
            _job_summary(job_id, job, metadata[2], metadata[3])
            if metadata
            else _job_summary(job_id, job)
        ),
        "raw_text": job.raw_text,
    }


async def _reparse_job(job_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    existing = state().store.get_job(job_id)
    if existing is None:
        raise HTTPException(status_code=404, detail={"error_code": "job_not_found", "message": "职位不存在"})
    body = payload or {}
    feedback = str(body.get("feedback") or "").strip()
    raw_text = body.get("raw_text")
    if raw_text is not None and not isinstance(raw_text, str):
        raise HTTPException(status_code=422, detail={"error_code": "invalid_raw_text", "message": "raw_text 必须是字符串"})
    raw_text = str(raw_text if raw_text is not None else (existing.raw_text or "")).strip()
    if not raw_text:
        if not existing.source_url:
            return {
                "status": "waiting_user",
                "job_id": job_id,
                "job": _job_summary(job_id, existing),
                "error_code": "job_text_unavailable",
                "requires_user": True,
                "message": "当前职位没有来源 URL 或可重新解析文本",
            }
        _require_post_time_risk_consent()
        health = await _ensure_mcp_server("boss")
        if health.get("status") != "ready":
            return {
                "status": "waiting_user",
                "job_id": job_id,
                "job": _job_summary(job_id, existing),
                "error_code": health.get("error_code", "mcp_unavailable"),
                "requires_user": True,
                "message": "BOSS MCP 暂不可用，请重新检测后再刷新",
            }
        try:
            remote = await state().mcp_manager.call_tool(
                "boss",
                "boss_reparse_job",
                {"job_id": job_id, "url": existing.source_url},
            )
        except MCPClientError as exc:
            remote = {"ok": False, "error_code": exc.error_code, "requires_user": True}
        if not remote.get("ok"):
            data = remote.get("data") if isinstance(remote.get("data"), dict) else {}
            return {
                "status": "waiting_user",
                "job_id": job_id,
                "job": _job_summary(job_id, existing),
                "error_code": remote.get("error_code") or "boss_reparse_unavailable",
                "retryable": bool(remote.get("retryable")),
                "requires_user": bool(remote.get("requires_user")),
                "message": data.get("message") or "请在可见 Edge 中完成登录或验证后重试",
            }
        data = remote.get("data") if isinstance(remote.get("data"), dict) else {}
        job_value = data.get("job") if isinstance(data.get("job"), dict) else data
        try:
            parsed = _job_from_mcp(dict(job_value), fallback_url=existing.source_url)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=502,
                detail={"error_code": "invalid_boss_detail", "message": "BOSS 返回的职位详情结构无效"},
            ) from exc
    else:
        parsed = job_from_text(existing.title, raw_text, company=existing.company, city=existing.city)
        parsed.salary = existing.salary
        parsed.hr_name = existing.hr_name
        parsed.hr_activity = existing.hr_activity
        parsed.source_url = existing.source_url
        parsed.posted_at = existing.posted_at
        parsed.posted_at_label = existing.posted_at_label or ("发布时间未知" if existing.posted_at is None else None)
        if body.get("posted_at_label") is not None:
            from app.services.job_parser import parse_posted_time

            parsed.posted_at, _ = parse_posted_time(str(body.get("posted_at_label")))
            parsed.posted_at_label = str(body.get("posted_at_label"))
    edited_keys = {
        "edited_responsibilities",
        "edited_requirements",
        "edited_skills",
    }
    has_structured_edits = bool(edited_keys.intersection(body))
    if has_structured_edits:
        parsed = _apply_job_corrections(parsed, body)
    parsed, parser = await _enhance_job_detail_with_chat_model(
        parsed,
        chat_profile_id=(str(body.get("chat_profile_id") or "").strip() or None),
        credential_handle_id=(
            str(body.get("credential_handle_id") or "").strip() or None
        ),
        feedback=feedback,
        previous_job=parsed if has_structured_edits else existing,
    )
    updated = state().store.update_job(job_id, parsed)
    state().job_parse_confirmations[job_id] = False
    if state().db_mirror:
        state().db_mirror.persist_job(job_id, updated)
    return {
        "status": "ready",
        "job_id": job_id,
        "job": _updated_job_payload(job_id, updated),
        "parser_version": parser.get("parser_version", "job-parser-v1"),
        "parser": parser,
        "message": "职位文本已重新解析；已创建任务的 job_snapshot 不会被覆盖",
    }


@app.post("/api/jobs/{job_id}/corrections")
async def save_job_corrections(
    job_id: str,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    """Persist an explicit human JD correction without invoking a model."""

    existing = state().store.get_job(job_id)
    if existing is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "job_not_found", "message": "职位不存在"},
        )
    corrected = _apply_job_corrections(existing, payload or {})
    updated = state().store.update_job(job_id, corrected)
    state().job_parse_confirmations[job_id] = False
    if state().db_mirror:
        state().db_mirror.persist_job(job_id, updated)
    return {
        "status": "ready",
        "job_id": job_id,
        "job": _updated_job_payload(job_id, updated),
        "parser": {
            "parser_mode": "human_corrected",
            "parser_version": "human-correction-v1",
            "warnings": [],
        },
        "message": "人工校正已保存，请再次确认解析结果",
    }


@app.post("/api/jobs/{job_id}/reparse")
async def reparse_job(
    job_id: str,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    return await _reparse_job(job_id, payload)


@app.post("/api/jobs/{job_id}/confirm-parse")
async def confirm_job_parse(
    job_id: str,
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    if state().store.get_job(job_id) is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "job_not_found", "message": "职位不存在"},
        )
    state().job_parse_confirmations[job_id] = True
    return {"status": "confirmed", "job_id": job_id}


@app.post("/api/jobs/{job_id}/refresh")
async def refresh_job(
    job_id: str,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    # Refresh is explicit and user-triggered.  In the local first slice the
    # visible Edge/MCP supplies fresh text through the optional request body.
    return await _reparse_job(job_id, payload)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, _: None = Depends(require_internal_token)) -> JobInput:
    job = state().store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.post("/api/resumes/upload")
async def upload_resume(file: UploadFile = File(...), _: None = Depends(require_internal_token)) -> dict[str, Any]:
    data = await file.read()
    try:
        resume = parse_resume_bytes(file.filename or "resume.txt", data)
    except ResumeParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    state().store.save_resume(resume)
    source_dir = state().settings.data_root / "resumes" / resume.resume_id
    source_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(resume.filename).name or "resume.source"
    source_path = source_dir / safe_name
    temporary = source_path.with_suffix(source_path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(source_path)
    if state().db_mirror:
        state().db_mirror.persist_resume(resume, path=str(source_path), data=data)
    return resume.model_dump(mode="json")


@app.get("/api/resumes/{resume_id}/source")
async def get_resume_source(
    resume_id: str,
    _: None = Depends(require_internal_token),
) -> FileResponse:
    """Return the exact uploaded file for a private in-app preview."""

    resume = state().store.get_resume(resume_id)
    if resume is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "resume_not_found", "message": "简历不存在"},
        )
    source_root = (state().settings.data_root / "resumes" / resume_id).resolve()
    candidate = (source_root / Path(resume.filename).name).resolve()
    if candidate.parent != source_root or not candidate.is_file():
        raise HTTPException(
            status_code=404,
            detail={"error_code": "resume_source_missing", "message": "原始简历文件不存在"},
        )
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".md": "text/markdown; charset=utf-8",
        ".markdown": "text/markdown; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
    }
    return FileResponse(
        candidate,
        media_type=media_types.get(candidate.suffix.casefold(), "application/octet-stream"),
        filename=candidate.name,
        content_disposition_type="inline",
    )


@app.post("/api/resumes/{resume_id}/model-parse")
async def model_parse_resume(
    resume_id: str,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    """Structure extracted resume text with a selected ready chat model.

    The rule parse remains available in every failure response, so changing a
    model or credential never requires uploading the private resume again.
    """

    resume = state().store.get_resume(resume_id)
    if resume is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "resume_not_found", "message": "简历不存在"},
        )
    body = payload or {}
    profile_id = str(body.get("chat_profile_id") or "").strip()
    profile = state().store.get_profile(profile_id) if profile_id else next(
        (
            item
            for item in state().store.list_profiles(ModelRole.CHAT.value)
            if item.default_for_role and item.status == "ready"
        ),
        None,
    )

    def fallback(message: str, error_code: str) -> dict[str, Any]:
        return {
            "status": "fallback",
            "resume": resume.model_dump(mode="json"),
            "parser": resume_rule_fallback_metadata(
                warning=message,
                error_code=error_code,
                profile=profile,
            ),
            "retryable": True,
        }

    if profile is None:
        return fallback("未选择已检测成功的聊天模型，已保留规则拆解结果", "chat_profile_not_found")
    if profile.role is not ModelRole.CHAT:
        return fallback("所选配置不是聊天模型，已保留规则拆解结果", "invalid_chat_profile_role")
    if profile.status != "ready":
        return fallback("所选聊天模型尚未检测成功，已保留规则拆解结果", "model_not_ready")

    handle_id = str(body.get("credential_handle_id") or "").strip() or None
    if profile.provider is Provider.OPENAI_COMPATIBLE:
        handle_id = _materialize_model_credential(profile.profile_id, handle_id)
        if not handle_id or not state().credentials.has(
            handle_id, scope=f"model:{profile.profile_id}"
        ):
            return fallback("云端模型 API Key 无法恢复，已保留规则拆解结果", "credential_missing")
        if (
            state().settings.external_model_consent_required
            and not state().consents.get("external_model:global", {}).get("granted", False)
        ):
            return fallback(
                "尚未确认向外部模型发送简历文字，已保留规则拆解结果",
                "needs_external_model_consent",
            )
    try:
        outcome = await asyncio.to_thread(
            structure_resume_with_model,
            resume,
            profile=profile,
            gateway=state().model_gateway,
            credential_handle_id=handle_id,
        )
    except Exception as exc:
        from app.core.utils import redact_sensitive

        reason = str(redact_sensitive(str(exc) or type(exc).__name__)).strip()
        if len(reason) > 240:
            reason = f"{reason[:237]}..."
        return fallback(
            "聊天模型拆解简历失败，已安全保留规则拆解结果"
            + (f"。原因：{reason}" if reason else ""),
            str(getattr(exc, "code", "resume_model_parse_failed")),
        )
    if state().db_mirror and state().db_mirror.available:
        if not state().db_mirror.persist_resume_sections(outcome.resume):
            return fallback("模型拆解完成，但数据库暂时无法保存，已保留原结果", "resume_persistence_unavailable")
    updated = state().store.update_resume(resume_id, outcome.resume)
    return {
        "status": "ready",
        "resume": updated.model_dump(mode="json"),
        "parser": outcome.metadata,
        "retryable": False,
    }


def _resume_summary(resume: Any, created_at: float | None = None) -> dict[str, Any]:
    """Return history metadata without duplicating the complete raw document."""
    return {
        "resume_id": resume.resume_id,
        "filename": resume.filename,
        "file_type": resume.file_type,
        "file_size": resume.file_size,
        "page_count": resume.page_count,
        "section_count": len(resume.sections),
        "modules": sorted({section.module for section in resume.sections}),
        "created_at": datetime.fromtimestamp(created_at, timezone.utc).isoformat() if created_at else None,
    }


@app.get("/api/resumes/history")
async def resumes_history(
    limit: int = Query(default=20, ge=1, le=20),
    cursor: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    resumes = state().store.list_resumes()
    offset = _decode_offset(cursor)
    page = resumes[offset : offset + limit]
    next_offset = offset + len(page)
    return {
        "resumes": [_resume_summary(resume, created_at) for resume, created_at in page],
        "total": len(resumes),
        "next_cursor": str(next_offset) if next_offset < len(resumes) else None,
        "has_more": next_offset < len(resumes),
    }


@app.post("/api/resumes/{resume_id}/parse")
async def parse_resume_endpoint(
    resume_id: str,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    resume = state().store.get_resume(resume_id)
    if resume is None:
        raise HTTPException(status_code=404, detail={"error_code": "resume_not_found", "message": "简历不存在"})
    # Parsing is intentionally idempotent.  The upload step already performed
    # the format/page/OCR gates; this route lets the UI explicitly re-run the
    # sectioning pass without mutating the original file bytes.
    requested_version = (payload or {}).get("parser_version", "resume-parser-v1")
    return {
        "status": "ready",
        "parser_version": str(requested_version),
        "resume": resume.model_dump(mode="json"),
        "message": "简历解析完成，原始文件与版式信息保持不变",
    }


@app.patch("/api/resumes/{resume_id}/sections")
async def patch_resume_sections(
    resume_id: str,
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    resume = state().store.get_resume(resume_id)
    if resume is None:
        raise HTTPException(status_code=404, detail={"error_code": "resume_not_found", "message": "简历不存在"})
    from app.core.utils import sha256_text
    from app.core.schemas import ResumeSection

    updates = payload.get("sections")
    if updates is None:
        updates = [payload]
    if not isinstance(updates, list) or not updates:
        raise HTTPException(status_code=422, detail={"error_code": "sections_required", "message": "sections 不能为空"})
    sections = [item.model_copy(deep=True) for item in resume.sections]
    by_id = {section.section_id: section for section in sections}
    allowed_modules = {
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
    }
    changed: list[str] = []
    for item in updates:
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail={"error_code": "invalid_section_patch", "message": "section patch 必须是对象"})
        section_id = str(item.get("section_id") or "")
        section = by_id.get(section_id)
        if section is None:
            raise HTTPException(status_code=404, detail={"error_code": "section_not_found", "message": f"未找到简历区块：{section_id}"})
        expected_hash = item.get("old_value_hash") or item.get("value_hash")
        if "content" in item and not expected_hash:
            raise HTTPException(status_code=422, detail={"error_code": "old_value_hash_required", "message": "修改简历内容必须携带 old_value_hash"})
        if expected_hash and sha256_text(section.content) != expected_hash:
            raise HTTPException(status_code=409, detail={"error_code": "resume_section_conflict", "message": "简历区块已被修改，请刷新后重试", "section_id": section_id})
        if "content" in item:
            if not isinstance(item["content"], str) or not item["content"].strip():
                raise HTTPException(status_code=422, detail={"error_code": "invalid_section_content", "message": "区块内容不能为空"})
            section.content = item["content"]
            section.value_hash = sha256_text(section.content)
            section.confirmed = False
        if "confirmed" in item:
            section.confirmed = bool(item["confirmed"])
        if "title" in item:
            section.title = str(item["title"]) if item["title"] is not None else None
        if "module" in item:
            module = str(item["module"] or "").strip()
            if module not in allowed_modules:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error_code": "invalid_resume_module",
                        "message": f"不支持的简历区块类型：{module}",
                    },
                )
            section.module = module  # type: ignore[assignment]
        if "evidence_ids" in item and isinstance(item["evidence_ids"], list):
            section.evidence_ids = [str(value) for value in item["evidence_ids"]]
        changed.append(section_id)
    candidate_resume = resume.model_copy(update={"sections": sections})
    if state().db_mirror and state().db_mirror.available:
        if not state().db_mirror.persist_resume_sections(candidate_resume):
            raise HTTPException(
                status_code=503,
                detail={"error_code": "resume_persistence_unavailable", "message": "简历区块已在当前会话更新，但数据库暂不可写，请稍后重试"},
            )
    updated = state().store.update_resume(resume_id, candidate_resume)
    return {"status": "ready", "changed_section_ids": changed, "resume": updated.model_dump(mode="json")}


@app.post("/api/resumes/{resume_id}/mark-project-region")
async def mark_project_region(
    resume_id: str,
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    resume = state().store.get_resume(resume_id)
    if resume is None:
        raise HTTPException(status_code=404, detail={"error_code": "resume_not_found", "message": "简历不存在"})
    ids = payload.get("section_ids") or payload.get("sections") or []
    if not isinstance(ids, list) or not ids:
        raise HTTPException(status_code=422, detail={"error_code": "project_region_required", "message": "请提供 section_ids"})
    ids_set = {str(value) for value in ids}
    sections = [section.model_copy(deep=True) for section in resume.sections]
    missing = sorted(ids_set - {section.section_id for section in sections})
    if missing:
        raise HTTPException(status_code=404, detail={"error_code": "section_not_found", "message": f"未找到简历区块：{missing[0]}"})
    for section in sections:
        if section.section_id in ids_set:
            section.module = "projects"
    candidate_resume = resume.model_copy(update={"sections": sections})
    if state().db_mirror and state().db_mirror.available:
        if not state().db_mirror.persist_resume_sections(candidate_resume):
            raise HTTPException(
                status_code=503,
                detail={"error_code": "resume_persistence_unavailable", "message": "项目区域已在当前会话更新，但数据库暂不可写，请稍后重试"},
            )
    updated = state().store.update_resume(resume_id, candidate_resume)
    return {
        "status": "ready",
        "project_section_ids": sorted(ids_set),
        "resume": updated.model_dump(mode="json"),
        "message": "项目经历区域已标记，后续任务将优先使用这些区块",
    }


@app.get("/api/resumes/{resume_id}")
async def get_resume(resume_id: str, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    resume = state().store.get_resume(resume_id)
    if not resume:
        raise HTTPException(status_code=404, detail="resume not found")
    return resume.model_dump(mode="json")


@app.post("/api/tasks", response_model=TaskView, status_code=202)
async def create_task(request: TaskCreateRequest, _: None = Depends(require_internal_token)) -> TaskView:
    store = state().store
    job = request.job
    if request.job_id:
        job = store.get_job(request.job_id)
    if job is None:
        raise HTTPException(status_code=422, detail="job or job_id is required")
    if request.job_id and state().job_parse_confirmations.get(request.job_id) is False:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "job_parse_confirmation_required",
                "message": "请先确认岗位职责与 JD 解析结果",
            },
        )
    source_job_id = request.job_id
    if source_job_id is None:
        source_job_id = store.save_job(job)
    snapshot_id = f"snapshot_{source_job_id}"
    if state().db_mirror:
        # Persist/reconcile the immutable snapshot for both an inline job and a
        # locally cached job.  The mirror operation is idempotent, so refreshing
        # a job cannot rewrite snapshots already referenced by older tasks.
        snapshot_id = state().db_mirror.persist_job(source_job_id, job) or snapshot_id
    resume = store.get_resume(request.resume_id) if request.resume_id else None
    if request.resume_id and resume is None:
        raise HTTPException(status_code=404, detail={"error_code": "resume_not_found", "message": "简历不存在"})
    if resume is not None and any(not section.confirmed for section in resume.sections):
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "resume_parse_confirmation_required",
                "message": "请先校正并确认简历的机器解析结果",
            },
        )
    if resume is None and request.output_mode != "project_only":
        required_base_facts = {
            "name": "姓名",
            "phone": "电话",
            "email": "邮箱",
            "objective": "求职目标",
        }
        missing = [label for key, label in required_base_facts.items() if not str(request.base_facts.get(key, "")).strip()]
        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "error_code": "base_facts_required",
                    "message": f"未上传简历时必须填写：{'、'.join(missing)}",
                    "missing_fields": [key for key in required_base_facts if not str(request.base_facts.get(key, "")).strip()],
                },
            )
    template_id: str | None = None
    template_structure_snapshot: dict[str, Any] | None = None
    if resume is None and request.output_mode != "project_only":
        template_id = str(request.template_id or "builtin-cn-single-page").strip()
        template_metadata: dict[str, Any]
        if template_id == "builtin-cn-single-page":
            builtin_path = write_preview(
                _BUILTIN_TEMPLATE,
                root=state().settings.data_root / "templates" / "cache" / template_id,
                name="template.md",
            )
            template_metadata = {
                "template_id": template_id,
                "source_url": "internal://builtin-cn-single-page",
                "cached_path": str(builtin_path),
                "file_type": "markdown",
                "language": "zh",
                "license": "internal",
                "license_allowed": True,
                "update_status": "built_in",
            }
        else:
            cache_root = (state().settings.data_root / "templates" / "cache").resolve()
            template_root = (cache_root / template_id).resolve()
            if template_root.parent != cache_root or not (template_root / "metadata.json").is_file():
                raise HTTPException(
                    status_code=409,
                    detail={"error_code": "template_not_cached", "message": "请先下载并预览所选模板"},
                )
            try:
                template_metadata = json.loads((template_root / "metadata.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise HTTPException(
                    status_code=409,
                    detail={"error_code": "template_metadata_invalid", "message": "所选模板缓存元数据损坏"},
                ) from exc
        if request.output_mode == "template_resume":
            template_source = _resolve_cached_template_source(template_id)
            confirmed_structure = load_template_structure(template_id, template_source)
            if not confirmed_structure or not confirmed_structure.get("confirmed"):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error_code": "template_structure_confirmation_required",
                        "message": "请先使用聊天模型解析模板结构，校正并确认后再继续",
                    },
                )
            template_structure_snapshot = {
                "structure_id": confirmed_structure.get("structure_id"),
                "structure_version": confirmed_structure.get("structure_version"),
                "source_sha256": confirmed_structure.get("source_sha256"),
                "confirmed_structure_hash": confirmed_structure.get(
                    "confirmed_structure_hash"
                ),
                "sections": confirmed_structure.get("sections") or [],
            }
        if state().db_mirror and hasattr(state().db_mirror, "persist_template"):
            state().db_mirror.persist_template(template_metadata)
    task_id = f"task_{uuid.uuid4().hex}"
    thread_id = f"thread_{uuid.uuid4().hex}"
    chat_profile_id = request.chat_profile_id
    if chat_profile_id is None:
        chat_profile_id = next(
            (item.profile_id for item in store.list_profiles(ModelRole.CHAT.value) if item.default_for_role),
            "ollama-chat-default",
        )
    embedding_profile_id = request.embedding_profile_id
    if embedding_profile_id is None:
        embedding_profile_id = next(
            (item.profile_id for item in store.list_profiles(ModelRole.EMBEDDING.value) if item.default_for_role),
            "ollama-embedding-default",
        )
    chat_profile = store.get_profile(chat_profile_id)
    embedding_profile = store.get_profile(embedding_profile_id)
    if request.chat_profile_id and chat_profile is None:
        raise HTTPException(status_code=404, detail={"error_code": "chat_profile_not_found", "message": "聊天模型 profile 不存在"})
    if request.embedding_profile_id and embedding_profile is None:
        raise HTTPException(status_code=404, detail={"error_code": "embedding_profile_not_found", "message": "embedding 模型 profile 不存在"})
    if chat_profile is not None and chat_profile.role is not ModelRole.CHAT:
        raise HTTPException(status_code=422, detail={"error_code": "invalid_chat_profile_role", "message": "chat_profile_id 必须引用聊天模型"})
    if embedding_profile is not None and embedding_profile.role is not ModelRole.EMBEDDING:
        raise HTTPException(status_code=422, detail={"error_code": "invalid_embedding_profile_role", "message": "embedding_profile_id 必须引用 embedding 模型"})
    chat_credential_handle_id = request.credential_handle_id
    if chat_profile and chat_profile.provider is Provider.OPENAI_COMPATIBLE:
        chat_credential_handle_id = _materialize_model_credential(
            chat_profile.profile_id,
            chat_credential_handle_id,
        )
    embedding_credential_handle_id = request.embedding_credential_handle_id
    if embedding_profile and embedding_profile.provider is Provider.OPENAI_COMPATIBLE:
        embedding_credential_handle_id = _materialize_model_credential(
            embedding_profile.profile_id,
            embedding_credential_handle_id,
        )
    blocked_reason: str | None = None
    initial_status = TaskStatus.QUEUED
    if state().settings.strict_model_gate and (chat_profile is None or chat_profile.status != "ready"):
        initial_status = TaskStatus.PAUSED
        blocked_reason = "needs_model_recheck"
    if chat_profile and chat_profile.provider is Provider.OPENAI_COMPATIBLE:
        if not chat_credential_handle_id or not state().credentials.has(chat_credential_handle_id, scope=f"model:{chat_profile.profile_id}"):
            initial_status = TaskStatus.PAUSED
            blocked_reason = "needs_credentials"
        elif state().settings.external_model_consent_required and not state().consents.get("external_model:global", {}).get("granted", False):
            initial_status = TaskStatus.PAUSED
            blocked_reason = "needs_external_model_consent"
    embedding_ready = bool(embedding_profile and embedding_profile.status == "ready")
    if embedding_ready and embedding_profile and embedding_profile.provider is Provider.OPENAI_COMPATIBLE:
        embedding_ready = bool(
            embedding_credential_handle_id
            and state().credentials.has(
                embedding_credential_handle_id,
                scope=f"model:{embedding_profile.profile_id}",
            )
        )
    embedding_mode = "embedding" if embedding_ready else "tfidf_fallback"
    if (
        embedding_mode == "embedding"
        and embedding_profile
        and embedding_profile.provider is Provider.OPENAI_COMPATIBLE
        and state().settings.external_model_consent_required
        and not state().consents.get("external_model:global", {}).get("granted", False)
    ):
        initial_status = TaskStatus.PAUSED
        blocked_reason = "needs_external_model_consent"
    if embedding_mode == "tfidf_fallback" and not request.allow_embedding_fallback:
        initial_status = TaskStatus.PAUSED
        blocked_reason = "needs_model_recheck"
    if resume is not None and not any(section.module == "projects" for section in resume.sections):
        initial_status = TaskStatus.PAUSED
        blocked_reason = "project_region_required"
    from app.services.matching import DEFAULT_COMPONENT_WEIGHTS, DEFAULT_WEIGHTS
    from app.core.utils import stable_id

    configured_dimensions = state().store.settings.get("dimension_weights", DEFAULT_WEIGHTS)
    configured_components = state().store.settings.get("component_weights", DEFAULT_COMPONENT_WEIGHTS)
    dimension_weights = dict(request.dimension_weights or configured_dimensions)
    component_weights = dict(request.component_weights or configured_components)
    try:
        # Reuse the scoring function's exact two-layer validation contract.
        from app.services.matching import calculate_match

        calculate_match(
            job,
            resume,
            weights=dimension_weights,
            component_weights=component_weights,
            embedding_mode=embedding_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error_code": "invalid_scoring_weights", "message": str(exc)}) from exc
    scoring_config_version = stable_id("scoring", dimension_weights, component_weights, "scoring-v1")
    task = TaskRecord(
        task_id=task_id,
        thread_id=thread_id,
        status=initial_status,
        blocked_reason=blocked_reason,
        job=job,
        resume=resume,
        job_snapshot_id=snapshot_id,
        state={
            # Every output mode honours the user's 1-5 draft count.  A
            # project-only task still skips resume replacement, but may offer
            # several independently copyable drafts before one is confirmed.
            "candidate_count": request.candidate_count,
            "embedding_mode": embedding_mode,
            "base_facts": request.base_facts,
            "template_id": template_id,
            "template_structure_snapshot": template_structure_snapshot,
            "output_mode": (
                "project_only" if request.output_mode == "project_only" else "resume_edit"
            ),
            "requested_output_mode": request.output_mode,
            "chat_profile_id": chat_profile_id,
            "chat_profile_version": chat_profile.config_version if chat_profile else None,
            "chat_profile_snapshot": chat_profile.model_dump(mode="json") if chat_profile else None,
            "embedding_profile_id": embedding_profile_id,
            "embedding_profile_version": embedding_profile.config_version if embedding_profile else None,
            "embedding_profile_snapshot": embedding_profile.model_dump(mode="json") if embedding_profile else None,
            "credential_handle_id": chat_credential_handle_id,
            "chat_credential_handle_id": chat_credential_handle_id,
            "embedding_credential_handle_id": embedding_credential_handle_id,
            "dimension_weights": dimension_weights,
            "component_weights": component_weights,
            "scoring_config_version": scoring_config_version,
        },
        messages=["任务已创建，等待执行" if initial_status == TaskStatus.QUEUED else f"任务已暂停：{blocked_reason}"],
    )
    store.create_task(task)
    if state().db_mirror:
        state().db_mirror.persist_task(task)
    store.append_event(task_id, {"type": "created", "thread_id": thread_id})
    if initial_status == TaskStatus.QUEUED:
        state().workflow.launch(task_id)
    else:
        store.append_event(task_id, {"type": "blocked", "status": initial_status.value, "blocked_reason": blocked_reason})
    return task_view(task)


@app.get("/api/tasks")
async def tasks_history(
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    """List resumable task metadata without returning resume or JD content."""

    tasks = state().store.list_tasks()
    offset = _decode_offset(cursor)
    page = tasks[offset : offset + limit]
    next_offset = offset + len(page)
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "thread_id": task.thread_id,
                "status": task.status.value,
                "blocked_reason": task.blocked_reason,
                "current_node": task.current_node,
                "checkpoint_version": task.checkpoint_version,
                "job_title": task.job.title if task.job else None,
                "company": task.job.company if task.job else None,
                "created_at": datetime.fromtimestamp(task.created_at, timezone.utc).isoformat(),
                "updated_at": datetime.fromtimestamp(task.updated_at, timezone.utc).isoformat(),
            }
            for task in page
        ],
        "total": len(tasks),
        "next_cursor": str(next_offset) if next_offset < len(tasks) else None,
        "has_more": next_offset < len(tasks),
    }


@app.get("/api/tasks/{task_id}", response_model=TaskView)
async def get_task(task_id: str, _: None = Depends(require_internal_token)) -> TaskView:
    task = state().store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return task_view(task)


@app.get("/api/tasks/{task_id}/events")
async def task_events(task_id: str, _: None = Depends(require_internal_token)) -> StreamingResponse:
    if not state().store.get_task(task_id):
        raise HTTPException(status_code=404, detail="task not found")

    async def stream():
        sent = 0
        for _ in range(120):
            events = state().store.events_for(task_id)
            while sent < len(events):
                payload = events[sent]
                sent += 1
                yield f"data: {__import__('json').dumps(payload, ensure_ascii=False)}\n\n"
            task = state().store.get_task(task_id)
            if task and task.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.WAITING_USER, TaskStatus.DELETED} and sent >= len(events):
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/tasks/{task_id}/resume", response_model=TaskView)
async def resume_task(task_id: str, request: ActionRequest, _: None = Depends(require_internal_token)) -> TaskView:
    task = state().store.get_task(task_id)
    if not task or task.thread_id != request.thread_id:
        raise HTTPException(status_code=404, detail="task/thread not found")
    # Compare the client snapshot before checking business state so an old tab
    # consistently receives the optimistic-concurrency error, even when the
    # current task has since reached a formal/immutable state.
    if task.checkpoint_version != request.checkpoint_version:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "checkpoint_conflict",
                "current_version": task.checkpoint_version,
                "refresh_required": True,
            },
        )
    if task.status == TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail={"error_code": "formal_snapshot_immutable", "message": "已确认的正式快照不可原地修改，请创建新的简历任务"},
        )
    try:
        updated = state().workflow.resume(
            task_id,
            expected_version=request.checkpoint_version,
            action=request.action,
            payload=request.payload,
        )
    except VersionConflict as exc:
        raise HTTPException(status_code=409, detail={"error_code": "checkpoint_conflict", "current_version": exc.current}) from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return task_view(updated)


@app.post("/api/tasks/{task_id}/feedback", response_model=TaskView)
async def task_feedback(task_id: str, request: FeedbackRequest, _: None = Depends(require_internal_token)) -> TaskView:
    task = state().store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task/thread not found")
    _require_task_checkpoint(task, request.thread_id, request.checkpoint_version)
    action = "revise"
    if task.status == TaskStatus.PAUSED and task.blocked_reason == "feedback_clarification_required":
        # A clarification response is a retry of the paused feedback gate; the
        # workflow will retain the same thread and re-run candidate generation.
        action = "retry"
    workflow_payload: dict[str, Any] = {
        "feedback": request.feedback,
        "target_version_id": request.target_version_id,
    }
    if request.chat_profile_id:
        profile = state().store.get_profile(request.chat_profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail={"error_code": "chat_profile_not_found", "message": "聊天模型 profile 不存在"})
        if profile.role is not ModelRole.CHAT:
            raise HTTPException(status_code=422, detail={"error_code": "invalid_chat_profile_role", "message": "反馈阶段只能切换聊天模型"})
        if profile.status != "ready":
            raise HTTPException(status_code=409, detail={"error_code": "needs_model_recheck", "message": "新聊天模型尚未通过连接和角色探测"})
        handle_id = request.credential_handle_id
        if profile.provider is Provider.OPENAI_COMPATIBLE:
            handle_id = _materialize_model_credential(
                profile.profile_id,
                handle_id,
            )
            if not handle_id or not state().credentials.has(handle_id, scope=f"model:{profile.profile_id}"):
                raise HTTPException(status_code=409, detail={"error_code": "needs_credentials", "message": "切换外部模型需要已保存或当前输入的有效 API Key"})
            if state().settings.external_model_consent_required and not state().consents.get("external_model:global", {}).get("granted", False):
                raise HTTPException(status_code=409, detail={"error_code": "needs_external_model_consent", "message": "请先确认外部模型数据发送范围"})
        workflow_payload.update(
            {
                "chat_profile_id": profile.profile_id,
                "chat_profile_version": profile.config_version,
                "chat_profile_snapshot": profile.model_dump(mode="json"),
                "credential_handle_id": handle_id,
            }
        )
    elif request.credential_handle_id:
        raise HTTPException(status_code=422, detail={"error_code": "chat_profile_required", "message": "credential_handle_id 必须与 chat_profile_id 一起提交"})
    try:
        updated = state().workflow.resume(
            task_id,
            expected_version=request.checkpoint_version,
            action=action,
            payload=workflow_payload,
        )
    except VersionConflict as exc:
        raise HTTPException(status_code=409, detail={"error_code": "checkpoint_conflict", "current_version": exc.current}) from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=409, detail={"error_code": "invalid_task_action", "message": str(exc)}) from exc
    if state().db_mirror:
        state().db_mirror.persist_checkpoint(updated)
    return task_view(updated)


@app.post("/api/tasks/{task_id}/cancel", response_model=TaskView)
async def cancel_task(task_id: str, request: ActionRequest, _: None = Depends(require_internal_token)) -> TaskView:
    task = state().store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task/thread not found")
    _require_task_checkpoint(task, request.thread_id, request.checkpoint_version)
    cancel_request = request.model_copy(update={"action": "cancel"})
    running = state().workflow._running.get(task_id)
    if running and not running.done():
        running.cancel()
    return await resume_task(task_id, cancel_request)


def _task_action_payload(payload: dict[str, Any] | None) -> tuple[str, int, dict[str, Any]]:
    """Normalize action-like request bodies used by retry/confirmation routes."""
    body = payload or {}
    thread_id = str(body.get("thread_id") or "")
    try:
        version = int(body.get("checkpoint_version"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "checkpoint_version_required", "message": "必须提供 checkpoint_version"},
        ) from exc
    if not thread_id:
        raise HTTPException(status_code=422, detail={"error_code": "thread_id_required", "message": "必须提供 thread_id"})
    if version < 0:
        raise HTTPException(status_code=422, detail={"error_code": "invalid_checkpoint_version", "message": "checkpoint_version 不能为负数"})
    return thread_id, version, body


@app.post("/api/tasks/{task_id}/confirmations", response_model=TaskView)
async def task_confirmations(
    task_id: str,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> TaskView:
    """Record explicit user confirmations without promoting drafts silently."""
    thread_id, checkpoint_version, body = _task_action_payload(payload)
    task = state().store.get_task(task_id)
    if not task or task.thread_id != thread_id:
        raise HTTPException(status_code=404, detail={"error_code": "task_not_found", "message": "task/thread not found"})
    _require_task_checkpoint(task, thread_id, checkpoint_version)
    if task.status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.DELETED}:
        raise HTTPException(
            status_code=409,
            detail={"error_code": "formal_snapshot_immutable", "message": "当前任务已结束，不能追加确认；请创建新的简历任务"},
        )
    confirmations = body.get("confirmations")
    if confirmations is None:
        confirmations = [body]
    if not isinstance(confirmations, list) or not confirmations:
        raise HTTPException(status_code=422, detail={"error_code": "confirmations_required", "message": "confirmations 不能为空"})
    normalized: list[dict[str, Any]] = []
    for item in confirmations:
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail={"error_code": "invalid_confirmation", "message": "confirmation 必须是对象"})
        field_id = str(item.get("field_id") or item.get("target_id") or "").strip()
        value_hash = str(item.get("value_hash") or "").strip()
        if not field_id or not value_hash:
            raise HTTPException(status_code=422, detail={"error_code": "invalid_confirmation", "message": "每条确认必须包含 field_id 和 value_hash"})
        normalized.append(
            {
                "module": str(item.get("module") or ""),
                "field_id": field_id,
                "value_hash": value_hash,
                "value": item.get("value"),
                "confirmed": bool(item.get("confirmed", True)),
                "source_version_id": item.get("source_version_id"),
                "patch_version_id": item.get("patch_version_id"),
            }
        )
    # A candidate confirmation is the explicit promotion action.  Field-only
    # confirmations remain a waiting checkpoint and cannot silently complete a
    # task or write an exportable resume.
    candidate_id = str(body.get("candidate_id") or body.get("candidate_slot_id") or "")
    if candidate_id and bool(body.get("confirm", body.get("confirmed", False))):
        try:
            updated = state().workflow.resume(
                task_id,
                expected_version=checkpoint_version,
                action="confirm",
                payload={"candidate_id": candidate_id, "confirmations": normalized},
            )
        except VersionConflict as exc:
            raise HTTPException(status_code=409, detail={"error_code": "checkpoint_conflict", "current_version": exc.current}) from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=409, detail={"error_code": "confirmation_failed", "message": str(exc)}) from exc
        if state().db_mirror:
            state().db_mirror.persist_checkpoint(updated)
            if hasattr(state().db_mirror, "persist_confirmations"):
                state().db_mirror.persist_confirmations(task_id, normalized)
        return task_view(updated)
    confirmations_state = [*task.state.get("confirmations", []), *normalized]
    try:
        updated = state().store.checkpoint(
            task_id,
            checkpoint_version,
            state={**task.state, "confirmations": confirmations_state},
            messages=[*task.messages, f"已记录 {len(normalized)} 条字段确认；候选仍需用户明确选择后才可写入正式简历"],
        )
    except VersionConflict as exc:
        raise HTTPException(status_code=409, detail={"error_code": "checkpoint_conflict", "current_version": exc.current}) from exc
    if state().db_mirror:
        state().db_mirror.persist_checkpoint(updated)
        if hasattr(state().db_mirror, "persist_confirmations"):
            state().db_mirror.persist_confirmations(task_id, normalized)
    return task_view(updated)


@app.post("/api/tasks/{task_id}/retry-step", response_model=TaskView)
async def retry_task_step(
    task_id: str,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> TaskView:
    """Retry one graph step on the existing task/thread."""
    thread_id, checkpoint_version, body = _task_action_payload(payload)
    task = state().store.get_task(task_id)
    if not task or task.thread_id != thread_id:
        raise HTTPException(status_code=404, detail={"error_code": "task_not_found", "message": "task/thread not found"})
    _require_task_checkpoint(task, thread_id, checkpoint_version)
    step = str(body.get("step") or body.get("node") or "match")
    if step not in {"match", "generate_candidates", "candidate_review", "export_preview"}:
        raise HTTPException(status_code=422, detail={"error_code": "invalid_retry_step", "message": "不支持重试该步骤"})
    if task.status == TaskStatus.RUNNING:
        raise HTTPException(status_code=409, detail={"error_code": "task_already_running", "message": "任务正在运行，请等待当前步骤完成"})
    if task.status in {TaskStatus.DELETED, TaskStatus.CANCELLED}:
        raise HTTPException(status_code=409, detail={"error_code": "task_not_retryable", "message": "已删除或取消的任务不能重试"})

    # A task/checkpoint may contain an expired process-local handle after an
    # API restart.  Restore fresh handles from the machine-local encrypted
    # store before evaluating retry gates, so users never need to click
    # "保存配置" again merely to run the next model step.
    chat_profile_id = str(task.state.get("chat_profile_id") or "")
    chat_profile = state().store.get_profile(chat_profile_id) if chat_profile_id else None
    if chat_profile is not None and chat_profile.provider is Provider.OPENAI_COMPATIBLE:
        restored = _materialize_model_credential(
            chat_profile_id,
            str(
                body.get("credential_handle_id")
                or task.state.get("credential_handle_id")
                or ""
            )
            or None,
        )
        if restored:
            body["credential_handle_id"] = restored
            body["chat_credential_handle_id"] = restored
    embedding_profile_id = str(task.state.get("embedding_profile_id") or "")
    embedding_profile = (
        state().store.get_profile(embedding_profile_id)
        if embedding_profile_id
        else None
    )
    if (
        embedding_profile is not None
        and embedding_profile.provider is Provider.OPENAI_COMPATIBLE
    ):
        restored_embedding = _materialize_model_credential(
            embedding_profile_id,
            str(
                body.get("embedding_credential_handle_id")
                or task.state.get("embedding_credential_handle_id")
                or ""
            )
            or None,
        )
        if restored_embedding:
            body["embedding_credential_handle_id"] = restored_embedding

    # A blocked reason is a gate, not merely a display label.  Do not let a
    # stale retry request clear credential/model/consent gates and launch a
    # worker with incomplete prerequisites.
    if task.status == TaskStatus.PAUSED:
        blocked = task.blocked_reason or ""
        if blocked == "needs_credentials":
            profile_id = str(task.state.get("chat_profile_id") or "")
            handle_id = body.get("credential_handle_id") or task.state.get(
                "credential_handle_id"
            )
            if not profile_id or not handle_id or not state().credentials.has(str(handle_id), scope=f"model:{profile_id}"):
                raise HTTPException(status_code=409, detail={"error_code": "needs_credentials", "message": "本机未找到该模型已保存的 API Key，请返回环境页重新输入一次"})
        elif blocked == "needs_external_model_consent" and not state().consents.get("external_model:global", {}).get("granted", False):
            raise HTTPException(status_code=409, detail={"error_code": "needs_external_model_consent", "message": "请先确认外部模型数据发送范围"})
        elif blocked == "project_region_required":
            if task.resume is None or not any(section.module == "projects" for section in task.resume.sections):
                raise HTTPException(status_code=409, detail={"error_code": blocked, "message": "请先标记简历中的项目经历区域，再重试当前步骤"})
        elif blocked in {
            "needs_model_recheck",
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
            "embedding_profile_missing",
            "reindex_required",
        } or blocked.startswith("required_item_missing:") or blocked.startswith("required_item_exceeds_budget:"):
            raise HTTPException(status_code=409, detail={"error_code": blocked, "message": "请先完成阻断原因要求的修复/重新检测，再重试当前步骤"})

    if task.status == TaskStatus.WAITING_USER:
        try:
            updated = state().workflow.resume(
                task_id,
                expected_version=checkpoint_version,
                action="retry",
                payload={**body, "step": step},
            )
        except VersionConflict as exc:
            raise HTTPException(status_code=409, detail={"error_code": "checkpoint_conflict", "current_version": exc.current}) from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=409, detail={"error_code": "retry_not_allowed", "message": str(exc)}) from exc
    elif task.status in {TaskStatus.PAUSED, TaskStatus.FAILED}:
        try:
            updated = state().workflow.resume(
                task_id,
                expected_version=checkpoint_version,
                action="retry",
                payload={**body, "step": step},
            )
        except VersionConflict as exc:
            raise HTTPException(status_code=409, detail={"error_code": "checkpoint_conflict", "current_version": exc.current}) from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=409, detail={"error_code": "retry_not_allowed", "message": str(exc)}) from exc
    else:
        try:
            updated = state().store.checkpoint(
                task_id,
                checkpoint_version,
                state={**task.state, "retry_step": step, "last_feedback": body.get("feedback", "")},
                status=TaskStatus.QUEUED,
                blocked_reason=None,
                current_node=step,
                messages=[*task.messages, f"已接受重试请求：{step}"],
            )
        except VersionConflict as exc:
            raise HTTPException(status_code=409, detail={"error_code": "checkpoint_conflict", "current_version": exc.current}) from exc
    if state().db_mirror:
        state().db_mirror.persist_checkpoint(updated)
    if updated.status in {TaskStatus.RUNNING, TaskStatus.QUEUED}:
        state().workflow.launch(task_id)
    return task_view(updated)


def _render_task_preview(task: TaskRecord) -> str:
    from app.core.utils import sha256_text, stable_id

    selected_id = task.state.get("selected_candidate")
    candidates = [item for item in task.candidates if item.candidate_id == selected_id]
    if not candidates:
        candidates = [item for item in task.candidates if item.status == "confirmed"] or task.candidates[:1]
    output_mode = str(task.state.get("output_mode") or "resume_edit")
    resume = task.resume.model_copy(deep=True) if task.resume is not None else None
    if resume is not None:
        sections = list(resume.sections)
        project_patch = task.state.get("project_patch")
        project_operation = (
            str(project_patch.get("operation") or "")
            if isinstance(project_patch, dict)
            else ""
        )
        selected_project = candidates[0] if candidates else None
        if project_operation == "replace":
            target_id = str(project_patch.get("target_project_id") or "")
            target = next((section for section in sections if section.section_id == target_id), None)
            if target is None:
                raise HTTPException(
                    status_code=409,
                    detail={"error_code": "patch_target_missing", "message": "待替换项目已不存在，请刷新后重新选择"},
                )
            current_hash = target.value_hash or sha256_text(target.content)
            if current_hash != str(project_patch.get("old_value_hash") or ""):
                raise HTTPException(
                    status_code=409,
                    detail={"error_code": "patch_hash_conflict", "message": "原项目已发生变化，拒绝应用过期替换补丁"},
                )
            if selected_project is None:
                raise HTTPException(
                    status_code=409,
                    detail={"error_code": "selected_candidate_missing", "message": "已确认的替换项目不存在，请重新选择"},
                )
            rendered_project = render_candidate_project(selected_project)
            rendered_lines = rendered_project.splitlines()
            project_body = "\n".join(rendered_lines[1:]).strip() if rendered_lines else rendered_project
            project_title = "  ".join(
                value
                for value in (selected_project.title.strip(), selected_project.period.strip())
                if value
            )
            replacement = target.model_copy(
                update={
                    "title": project_title or selected_project.title,
                    "content": project_body,
                    "value_hash": sha256_text(project_body),
                    "confirmed": True,
                }
            )
            sections = [replacement if section.section_id == target_id else section for section in sections]
            # The selected project now occupies the exact location of the
            # replaced section; do not append a duplicate candidate later.
            candidates = []
        elif project_operation == "add" and selected_project is not None:
            rendered_project = render_candidate_project(selected_project)
            rendered_lines = rendered_project.splitlines()
            project_body = "\n".join(rendered_lines[1:]).strip() if rendered_lines else rendered_project
            project_title = "  ".join(
                value
                for value in (selected_project.title.strip(), selected_project.period.strip())
                if value
            )
            new_project_section = ResumeSection(
                section_id=stable_id(
                    "section", task.task_id, "projects", selected_project.candidate_id
                ),
                module="projects",
                title=project_title or selected_project.title,
                content=project_body,
                confirmed=True,
                value_hash=sha256_text(project_body),
            )
            project_indexes = [
                index
                for index, section in enumerate(sections)
                if section.module == "projects"
                or bool(re.search(r"项目(?:经历|经验)?", str(section.title or "")))
            ]
            if project_indexes:
                insert_at = project_indexes[-1] + 1
            else:
                # If a resume has no project block, create one before the
                # trailing skills/education/certificate/evaluation modules,
                # never after the entire original document.
                trailing_modules = {"skills", "education", "certificates", "evaluation"}
                insert_at = next(
                    (
                        index
                        for index, section in enumerate(sections)
                        if section.module in trailing_modules
                    ),
                    len(sections),
                )
            sections.insert(insert_at, new_project_section)
            candidates = []
        module_patches = task.state.get("module_patches")
        if isinstance(module_patches, dict):
            for module, patch in module_patches.items():
                if not isinstance(patch, dict) or patch.get("status") != "confirmed":
                    continue
                target_id = str(patch.get("target_item_id") or "")
                target = next((section for section in sections if section.section_id == target_id), None)
                old_hash = str(patch.get("old_value_hash") or "")
                if target is not None:
                    current_hash = target.value_hash or sha256_text(target.content)
                    if current_hash != old_hash:
                        raise HTTPException(
                            status_code=409,
                            detail={
                                "error_code": "patch_hash_conflict",
                                "message": f"{module} 模块已发生变化，拒绝应用过期补丁",
                            },
                        )
                    replacement = target.model_copy(
                        update={
                            "content": str(patch.get("new_value") or ""),
                            "value_hash": str(patch.get("new_value_hash") or ""),
                            "confirmed": True,
                        }
                    )
                    sections = [replacement if item.section_id == target_id else item for item in sections]
                elif target_id.endswith(":new") and old_hash == sha256_text(""):
                    value = str(patch.get("new_value") or "")
                    sections.append(
                        ResumeSection(
                            section_id=stable_id("section", task.task_id, module, patch.get("patch_id")),
                            module=module,
                            title={"summary": "个人简介", "skills": "技能清单", "evaluation": "个人评价"}.get(module),
                            content=value,
                            confirmed=True,
                            value_hash=sha256_text(value),
                        )
                    )
                else:
                    raise HTTPException(
                        status_code=409,
                        detail={"error_code": "patch_target_missing", "message": f"{module} 模块目标不存在"},
                    )
        resume = resume.model_copy(update={"sections": sections})
    if output_mode == "project_only":
        resume = None
    elif output_mode == "original_resume":
        candidates = []
    template_content: str | None = None
    template_path = _task_template_path(task)
    if template_path is not None and template_path.suffix.lower() in {".md", ".markdown"}:
        try:
            template_content = template_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": "template_content_invalid",
                    "message": "所选 Markdown 模板无法读取，请重新下载或选择其他模板",
                },
            ) from exc
    return render_markdown(
        job_title=task.job.title if task.job else "",
        candidates=candidates,
        resume=resume,
        selected_candidate_id=(candidates[0].candidate_id if candidates else None),
        base_facts=(
            dict(task.state.get("base_facts") or {})
            if isinstance(task.state.get("base_facts"), dict)
            else {}
        ),
        template_content=template_content,
    )


def _task_template_path(task: TaskRecord) -> Path | None:
    """Resolve only the template file already pinned to this task."""

    template_id = str(task.state.get("template_id") or "").strip()
    if not template_id:
        return None
    cache_root = (state().settings.data_root / "templates" / "cache").resolve()
    template_root = (cache_root / template_id).resolve()
    if template_root.parent != cache_root:
        raise HTTPException(
            status_code=409,
            detail={"error_code": "template_path_invalid", "message": "所选模板路径不合法"},
        )
    candidates: list[Path] = []
    metadata_path = template_root / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=409,
                detail={"error_code": "template_metadata_invalid", "message": "所选模板缓存元数据损坏"},
            ) from exc
        cached_path = Path(str(metadata.get("cached_path") or ""))
        if str(cached_path):
            candidates.append(cached_path)
    if template_root.is_dir():
        candidates.extend(
            item
            for item in sorted(template_root.iterdir())
            if item.is_file() and item.suffix.lower() in {".md", ".markdown", ".docx"}
        )
    for candidate in candidates:
        resolved = candidate.resolve()
        if (
            resolved.is_file()
            and resolved.suffix.lower() in {".md", ".markdown", ".docx"}
            and template_root in resolved.parents
        ):
            return resolved
    raise HTTPException(
        status_code=409,
        detail={
            "error_code": "template_not_cached",
            "message": "所选模板缓存不存在，请返回模板选择步骤重新下载",
        },
    )


def _task_template_replacements(task: TaskRecord) -> dict[str, str]:
    facts = task.state.get("base_facts")
    facts = dict(facts) if isinstance(facts, dict) else {}
    objective = str(facts.get("objective") or (task.job.title if task.job else ""))
    values = {
        "name": facts.get("name"),
        "姓名": facts.get("name"),
        "phone": facts.get("phone"),
        "电话": facts.get("phone"),
        "email": facts.get("email"),
        "邮箱": facts.get("email"),
        "location": facts.get("location"),
        "所在地": facts.get("location"),
        "objective": objective,
        "求职目标": objective,
        "education": facts.get("education"),
        "教育经历": facts.get("education"),
        "certificates": facts.get("certificates"),
        "证书": facts.get("certificates"),
    }
    return {key: str(value) for key, value in values.items() if str(value or "").strip()}


def _task_bound_to_snapshot(
    task: TaskRecord,
    snapshot_id: str,
    *,
    require_confirmed: bool = False,
) -> TaskRecord:
    """Rebuild render input from the immutable business version when available."""

    mirror = state().db_mirror
    if mirror is None or not hasattr(mirror, "load_resume_snapshot"):
        return task
    snapshot = mirror.load_resume_snapshot(snapshot_id, task_id=task.task_id)
    if snapshot is None and hasattr(mirror, "persist_resume_snapshot"):
        # Compatibility for tasks completed before snapshot materialization was
        # introduced.  Normal workflow completion has already persisted it.
        mirror.persist_resume_snapshot(task)
        snapshot = mirror.load_resume_snapshot(snapshot_id, task_id=task.task_id)
    if snapshot is None:
        if mirror.available:
            raise HTTPException(
                status_code=409,
                detail={"error_code": "snapshot_not_materialized", "message": "不可变简历快照尚未物化，请重新确认最终预览"},
            )
        return task
    if require_confirmed and snapshot.get("state") != "confirmed":
        raise HTTPException(
            status_code=409,
            detail={"error_code": "snapshot_not_confirmed", "message": "简历快照尚未最终确认"},
        )
    payload = snapshot.get("materialized")
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=409,
            detail={"error_code": "snapshot_content_missing", "message": "简历快照缺少物化内容"},
        )
    try:
        job_value = payload.get("job")
        resume_value = payload.get("resume")
        candidates_value = payload.get("candidates") or []
        snapshot_state = payload.get("state") or {}
        return replace(
            task,
            job=JobInput.model_validate(job_value) if isinstance(job_value, dict) else None,
            resume=ResumeDocument.model_validate(resume_value) if isinstance(resume_value, dict) else None,
            candidates=[CandidateProject.model_validate(item) for item in candidates_value if isinstance(item, dict)],
            state=dict(snapshot_state),
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=409,
            detail={"error_code": "snapshot_content_invalid", "message": "简历快照内容无法恢复"},
        ) from exc


def _compressed_layout(layout: dict[str, Any], compression: Any) -> dict[str, Any]:
    """Apply a validated compression scale to the user-selected layout."""

    try:
        font_size = float(layout.get("font_size_pt", 10.5))
        line_spacing = float(layout.get("line_spacing", 1.0))
        margin_cm = float(layout.get("margin_cm", 1.2))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail={"error_code": "invalid_layout", "message": "排版参数必须是数字"}) from exc
    if not (6 <= font_size <= 18 and 0.7 <= line_spacing <= 2.0 and 0.3 <= margin_cm <= 3.0):
        raise HTTPException(status_code=422, detail={"error_code": "invalid_layout", "message": "排版参数超出允许范围"})
    return {
        **layout,
        "font_size_pt": font_size * compression.font_scale,
        "line_spacing": line_spacing * compression.line_spacing,
        "margin_cm": margin_cm * compression.margin_scale,
        "paragraph_spacing_before_pt": 0,
        "paragraph_spacing_after_pt": 0,
        "compact_reflow": bool(getattr(compression, "compact_reflow", False)),
    }


_TASK_ARTIFACT_SCOPES = {"previews", "exports"}
_TASK_ARTIFACT_MEDIA_TYPES = {
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
}


def _task_artifact_root(task_id: str, scope: str) -> Path:
    """Return a task-local artifact directory and reject path traversal."""

    if scope not in _TASK_ARTIFACT_SCOPES:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "artifact_scope_invalid", "message": "文件类型不存在"},
        )
    tasks_root = (state().settings.data_root / "tasks").resolve()
    task_root = (tasks_root / str(task_id)).resolve()
    if task_root.parent != tasks_root:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "artifact_path_invalid", "message": "任务文件路径不合法"},
        )
    return (task_root / scope).resolve()


def _task_artifact_descriptor(
    task_id: str,
    path: str | Path | None,
    *,
    scope: str,
) -> dict[str, Any] | None:
    """Expose a task file by opaque API URL, never by trusting its raw path."""

    if not path:
        return None
    artifact_root = _task_artifact_root(task_id, scope)
    candidate = Path(path).resolve()
    if not candidate.is_file() or candidate.parent != artifact_root:
        return None
    media_type = _TASK_ARTIFACT_MEDIA_TYPES.get(candidate.suffix.casefold())
    if media_type is None:
        return None
    return {
        "status": "ready",
        "file_name": candidate.name,
        "mime_type": media_type,
        "size_bytes": candidate.stat().st_size,
        "download_url": f"/api/tasks/{task_id}/artifacts/{scope}/{candidate.name}",
    }


def _resume_pdf_result(docx_path: Path, pdf_path: Path):
    """Prefer LibreOffice while retaining the existing test seam and Word fallback."""

    result = convert_pdf_with_libreoffice(docx_path, pdf_path)
    if result.status == "ready":
        return result
    fallback = convert_docx_to_pdf(docx_path, pdf_path)
    return fallback if fallback.status == "ready" else result


def _resume_pdf_with_text_fallback(
    docx_path: Path,
    pdf_path: Path,
    *,
    content: str,
    layout: dict[str, Any],
):
    """Always return a downloadable PDF when text rendering is available."""

    office_result = _resume_pdf_result(docx_path, pdf_path)
    if office_result.status == "ready":
        return office_result
    text_result = export_text_pdf(content, target=pdf_path, layout=layout)
    if text_result.status != "ready":
        return office_result
    warnings = tuple(
        dict.fromkeys((*office_result.warnings, *text_result.warnings))
    )
    return type(text_result)(
        text_result.format,
        text_result.status,
        text_result.path,
        text_result.page_count,
        text_result.one_page,
        warnings,
    )


@app.get("/api/tasks/{task_id}/artifacts/{scope}/{filename}")
async def download_task_artifact(
    task_id: str,
    scope: str,
    filename: str,
    _: None = Depends(require_internal_token),
) -> FileResponse:
    """Download only a generated file directly inside this task's artifact root."""

    if state().store.get_task(task_id) is None:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "task_not_found", "message": "任务不存在"},
        )
    if not filename or Path(filename).name != filename:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "artifact_path_invalid", "message": "文件名不合法"},
        )
    artifact_root = _task_artifact_root(task_id, scope)
    candidate = (artifact_root / filename).resolve()
    media_type = _TASK_ARTIFACT_MEDIA_TYPES.get(candidate.suffix.casefold())
    if candidate.parent != artifact_root or media_type is None or not candidate.is_file():
        raise HTTPException(
            status_code=404,
            detail={"error_code": "artifact_not_found", "message": "导出文件不存在或已失效"},
        )
    return FileResponse(
        candidate,
        media_type=media_type,
        filename=candidate.name,
        content_disposition_type="attachment",
    )


@app.post("/api/tasks/{task_id}/export-preview")
async def export_preview(
    task_id: str,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    body = payload or {}
    task = state().store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail={"error_code": "task_not_found", "message": "任务不存在"})
    thread_id, requested_version, body = _task_action_payload(body)
    _require_task_checkpoint(task, thread_id, requested_version)
    format_name = str(body.get("format") or "pdf").lower()
    if format_name == "markdown":
        format_name = "md"
    if format_name not in {"pdf", "docx", "md", "text", "txt"}:
        raise HTTPException(status_code=422, detail={"error_code": "unsupported_export_format", "message": "仅支持 PDF、DOCX 或 Markdown 预览"})
    preview_allowed = task.status == TaskStatus.COMPLETED or (
        task.status == TaskStatus.WAITING_USER and task.current_node == "preview_review"
    )
    if not preview_allowed:
        return {
            "status": "blocked",
            "format": format_name,
            "error_code": "confirmation_required",
            "requires_user": True,
            "message": "任务尚未完成最终候选确认，不能生成正式简历预览",
            "task_id": task_id,
            "checkpoint_version": task.checkpoint_version,
        }
    snapshot_id = str(task.state.get("resume_snapshot_id") or "")
    if not snapshot_id:
        raise HTTPException(status_code=409, detail={"error_code": "snapshot_required", "message": "请先创建不可变简历快照"})
    requested_snapshot = body.get("snapshot_id")
    if not requested_snapshot:
        raise HTTPException(status_code=409, detail={"error_code": "snapshot_required", "message": "预览必须显式绑定 snapshot_id"})
    if str(requested_snapshot) != snapshot_id:
        raise HTTPException(status_code=409, detail={"error_code": "snapshot_conflict", "message": "预览必须绑定当前不可变简历快照"})
    bound_task = _task_bound_to_snapshot(task, snapshot_id)
    content = _render_task_preview(bound_task)
    template_path = _task_template_path(bound_task)
    template_replacements = _task_template_replacements(bound_task)
    pages = estimate_pages(content)
    preview_root = state().settings.data_root / "tasks" / task_id / "previews"
    layout = body.get("layout") if isinstance(body.get("layout"), dict) else {}
    compression: dict[str, Any] | None = None
    saved_compression = task.state.get("compression_decision")
    saved_compression = saved_compression if isinstance(saved_compression, dict) else {}
    accept_compression = bool(body.get("accept_compression", saved_compression.get("accepted", False)))
    allow_rewrite = bool(body.get("allow_rewrite", saved_compression.get("allow_rewrite", False)))
    compression_level = str(
        body.get("compression_level")
        or saved_compression.get("level")
        or "standard"
    )
    if pages > 1 and not accept_compression and saved_compression.get("compliance_status") != "multi_page_allowed":
        markdown_path = write_preview(content, root=preview_root, name="resume-preview.md")
        return {
            "status": "needs_compression_consent",
            "format": format_name,
            "error_code": "compression_consent_required",
            "requires_user": True,
            "message": "预览超过一页，需要用户先同意调整字体、间距和页边距",
            "task_id": task_id,
            "snapshot_id": snapshot_id,
            "checkpoint_version": task.checkpoint_version,
            "content": content,
            "page_count": pages,
            "markdown_path": str(markdown_path),
            "compliance_status": "overflow",
        }
    if pages > 1 and accept_compression:
        compressed = compress_markdown(
            content,
            allow_rewrite=allow_rewrite,
            level=compression_level,
        )
        content = compressed.content
        pages = compressed.page_count
        compression = {
            "font_scale": compressed.font_scale,
            "line_spacing": compressed.line_spacing,
            "margin_scale": compressed.margin_scale,
            "rewritten": compressed.rewritten,
            "warnings": list(compressed.warnings),
            "density": compressed.density,
            "level": compressed.level,
            "compact_reflow": compressed.compact_reflow,
        }
        # Pass the approved style adjustment to DOCX generation while retaining
        # one consistent body/heading font family.
        layout = _compressed_layout(layout, compressed)
    markdown_path = write_preview(content, root=preview_root, name="resume-preview.md")
    result: dict[str, Any] = {
        "status": "ready" if pages <= 1 else "preview_overflow",
        "format": format_name,
        "engine": "native_text" if format_name in {"md", "text", "txt"} else None,
        "task_id": task_id,
        "snapshot_id": snapshot_id,
        "checkpoint_version": task.checkpoint_version,
        "content": content,
        "markdown_path": str(markdown_path),
        "page_count": pages,
        "estimated_page_count": pages,
        "page_count_source": "estimate",
        "compliance_status": "draft_preview" if pages <= 1 else "overflow",
    }
    result["markdown_artifact"] = _task_artifact_descriptor(
        task_id, markdown_path, scope="previews"
    )
    if compression is not None:
        result["compression"] = compression
    if format_name in {"md", "text", "txt"}:
        result["artifact"] = result["markdown_artifact"]
    elif format_name == "docx":
        docx_result = export_docx(
            content,
            target=preview_root / "resume-preview.docx",
            layout=layout,
            template_path=template_path,
            template_replacements=template_replacements,
        )
        result["docx"] = docx_result.__dict__
        result["artifact"] = _task_artifact_descriptor(
            task_id, docx_result.path, scope="previews"
        )
        if docx_result.path:
            pdf_preview = _resume_pdf_with_text_fallback(
                Path(docx_result.path),
                preview_root / "resume-preview.pdf",
                content=content,
                layout=layout,
            )
            result["visual_preview"] = {
                **pdf_preview.__dict__,
                "artifact": _task_artifact_descriptor(
                    task_id, pdf_preview.path, scope="previews"
                ),
            }
        else:
            result["visual_preview"] = {
                "status": "unavailable",
                "artifact": None,
                "warnings": list(docx_result.warnings),
            }
        if docx_result.status != "preview_pending":
            result["status"] = "preview_pending" if docx_result.status in {"dependency_missing", "conversion_failed"} else docx_result.status
            result["error_code"] = "documents_dependency_missing" if docx_result.status == "dependency_missing" else "document_preview_unavailable"
    elif format_name == "pdf":
        docx = export_docx(
            content,
            target=preview_root / "resume-preview.docx",
            layout=layout,
            template_path=template_path,
            template_replacements=template_replacements,
        )
        pdf_result = (
            _resume_pdf_with_text_fallback(
                Path(docx.path),
                preview_root / "resume-preview.pdf",
                content=content,
                layout=layout,
            )
            if docx.path
            else export_text_pdf(
                content,
                target=preview_root / "resume-preview.pdf",
                layout=layout,
            )
        )
        result["pdf"] = pdf_result.__dict__ if pdf_result else {"status": "dependency_missing", "path": None}
        result["artifact"] = _task_artifact_descriptor(
            task_id, pdf_result.path if pdf_result else None, scope="previews"
        )
        result["visual_preview"] = {
            **(pdf_result.__dict__ if pdf_result else {"status": "dependency_missing"}),
            "artifact": result["artifact"],
        }
        if pdf_result is not None and pdf_result.status == "ready" and pdf_result.page_count is not None:
            result["page_count"] = pdf_result.page_count
            result["page_count_source"] = "rendered_pdf"
            result["compliance_status"] = "one_page" if pdf_result.page_count <= 1 else "overflow"
        if pdf_result is None or pdf_result.status != "ready":
            result["status"] = "preview_pending"
            result["error_code"] = "documents_dependency_missing" if pdf_result is None or pdf_result.status == "dependency_missing" else "pdf_conversion_failed"
    return result


@app.post("/api/tasks/{task_id}/export")
async def export_task(
    task_id: str,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    body = payload or {}
    task = state().store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail={"error_code": "task_not_found", "message": "任务不存在"})
    thread_id, supplied_version, body = _task_action_payload(body)
    _require_task_checkpoint(task, thread_id, supplied_version)
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail={"error_code": "confirmation_required", "message": "请先确认候选项目"})
    selected_id = task.state.get("selected_candidate")
    selected = next((item for item in task.candidates if item.candidate_id == selected_id), None)
    if selected is None or (selected.needs_verification and not bool(body.get("confirm_unverified", False))):
        raise HTTPException(status_code=409, detail={"error_code": "unverified_content", "message": "仍有待核实内容"})
    snapshot_id = task.state.get("resume_snapshot_id")
    requested_snapshot = body.get("snapshot_id")
    if not requested_snapshot:
        raise HTTPException(status_code=409, detail={"error_code": "snapshot_required", "message": "导出必须显式绑定 snapshot_id"})
    if requested_snapshot and snapshot_id and str(requested_snapshot) != str(snapshot_id):
        raise HTTPException(status_code=409, detail={"error_code": "snapshot_conflict", "message": "导出必须绑定当前不可变简历快照"})
    if not snapshot_id:
        raise HTTPException(status_code=409, detail={"error_code": "snapshot_required", "message": "请先创建不可变简历快照"})
    bound_task = _task_bound_to_snapshot(
        task, str(snapshot_id), require_confirmed=True
    )
    content = _render_task_preview(bound_task)
    template_path = _task_template_path(bound_task)
    template_replacements = _task_template_replacements(bound_task)
    original_pages = estimate_pages(content)
    layout = body.get("layout") if isinstance(body.get("layout"), dict) else {}
    compression: dict[str, Any] | None = None
    saved_compression = task.state.get("compression_decision")
    saved_compression = saved_compression if isinstance(saved_compression, dict) else {}
    accept_compression = bool(
        body.get("accept_compression", saved_compression.get("accepted", False))
    )
    if original_pages > 1 and accept_compression:
        compressed = compress_markdown(
            content,
            allow_rewrite=bool(
                body.get("allow_rewrite", saved_compression.get("allow_rewrite", False))
            ),
            level=str(
                body.get("compression_level")
                or saved_compression.get("level")
                or "standard"
            ),
        )
        content = compressed.content
        layout = _compressed_layout(layout, compressed)
        compression = {
            "font_scale": compressed.font_scale,
            "line_spacing": compressed.line_spacing,
            "margin_scale": compressed.margin_scale,
            "rewritten": compressed.rewritten,
            "warnings": list(compressed.warnings),
            "density": compressed.density,
            "level": compressed.level,
            "compact_reflow": compressed.compact_reflow,
        }
    final_pages = estimate_pages(content, density=(compressed.density if compression is not None else 1.0))
    root = state().settings.data_root / "tasks" / task_id / "exports"
    root.mkdir(parents=True, exist_ok=True)
    format_name = str(body.get("format") or "markdown").lower()
    if format_name in {"markdown", "md", "text", "txt"}:
        path = write_preview(content, root=root, name="resume.md")
        result = {"status": "ready", "format": "markdown", "task_id": task_id, "checkpoint_version": task.checkpoint_version, "output_path": str(path), "content": content, "page_count": final_pages, "estimated_page_count": final_pages, "page_count_source": "estimate", "snapshot_id": snapshot_id, "compliance_status": "one_page" if final_pages <= 1 else "overflow", "artifact": _task_artifact_descriptor(task_id, path, scope="exports")}
        if compression is not None:
            result["compression"] = compression
        return result
    if format_name == "docx":
        docx_result = export_docx(
            content,
            target=root / "resume.docx",
            layout=layout,
            template_path=template_path,
            template_replacements=template_replacements,
        )
        result = {
            "status": docx_result.status,
            "format": "docx",
            "task_id": task_id,
            "checkpoint_version": task.checkpoint_version,
            "snapshot_id": snapshot_id,
            **docx_result.__dict__,
            "content": content,
            "estimated_page_count": final_pages,
            "page_count_source": "unverified",
            "compliance_status": "preview_pending",
            "artifact": _task_artifact_descriptor(
                task_id, docx_result.path, scope="exports"
            ),
        }
        if docx_result.path:
            pdf_preview = _resume_pdf_with_text_fallback(
                Path(docx_result.path),
                root / "resume-preview.pdf",
                content=content,
                layout=layout,
            )
            result["visual_preview"] = {
                **pdf_preview.__dict__,
                "artifact": _task_artifact_descriptor(
                    task_id, pdf_preview.path, scope="exports"
                ),
            }
        if compression is not None:
            result["compression"] = compression
        if docx_result.status == "dependency_missing":
            result["error_code"] = "documents_dependency_missing"
        return result
    if format_name == "pdf":
        docx = export_docx(
            content,
            target=root / "resume.docx",
            layout=layout,
            template_path=template_path,
            template_replacements=template_replacements,
        )
        pdf_result = (
            _resume_pdf_with_text_fallback(
                Path(docx.path),
                root / "resume.pdf",
                content=content,
                layout=layout,
            )
            if docx.path
            else export_text_pdf(content, target=root / "resume.pdf", layout=layout)
        )
        rendered_pages = pdf_result.page_count if pdf_result.page_count is not None else final_pages
        result = {
            "status": pdf_result.status,
            "format": "pdf",
            "task_id": task_id,
            "checkpoint_version": task.checkpoint_version,
            "snapshot_id": snapshot_id,
            **pdf_result.__dict__,
            "content": content,
            "page_count": rendered_pages,
            "estimated_page_count": final_pages,
            "page_count_source": "rendered_pdf" if pdf_result.page_count is not None else "estimate",
            "compliance_status": "one_page" if rendered_pages <= 1 and pdf_result.status == "ready" else "overflow" if rendered_pages > 1 else "preview_pending",
            "artifact": _task_artifact_descriptor(
                task_id, pdf_result.path, scope="exports"
            ),
        }
        result["visual_preview"] = {
            **pdf_result.__dict__,
            "artifact": result["artifact"],
        }
        if compression is not None:
            result["compression"] = compression
        result["error_code"] = (
            "pdf_conversion_failed"
            if pdf_result.status == "conversion_failed"
            else "documents_dependency_missing"
            if pdf_result.status == "dependency_missing"
            else None
        )
        return result
    raise HTTPException(status_code=422, detail={"error_code": "unsupported_export_format", "message": "仅支持 DOCX、PDF 或 Markdown"})


@app.delete("/api/tasks/{task_id}", response_model=TaskView)
async def delete_task(task_id: str, _: None = Depends(require_internal_token)) -> TaskView:
    task = state().store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail={"error_code": "task_not_found", "message": "任务不存在"})
    running = state().workflow._running.get(task_id)
    if running and not running.done():
        running.cancel()
        try:
            await running
        except asyncio.CancelledError:
            pass

    tasks_root = (state().settings.data_root / "tasks").resolve()
    task_directory = (tasks_root / task_id).resolve()
    if tasks_root not in task_directory.parents:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "invalid_task_path", "message": "任务目录超出允许范围"},
        )
    staged_directory: Path | None = None
    if task_directory.exists():
        staged_directory = tasks_root / f".deleting-{task_id}-{uuid.uuid4().hex[:8]}"
        try:
            task_directory.replace(staged_directory)
        except OSError as exc:
            raise HTTPException(
                status_code=409,
                detail={"error_code": "task_files_busy", "message": "任务文件正在使用，暂时无法删除"},
            ) from exc

    try:
        if state().db_mirror and not state().db_mirror.delete_task(task_id):
            raise HTTPException(
                status_code=503,
                detail={"error_code": "task_delete_persistence_failed", "message": "数据库删除失败，任务未删除"},
            )
        state().store.delete_task(task_id)
        state().consents = {
            key: value
            for key, value in state().consents.items()
            if value.get("task_id") != task_id
        }
    except Exception:
        if staged_directory and staged_directory.exists() and not task_directory.exists():
            try:
                staged_directory.replace(task_directory)
            except OSError:
                pass
        raise

    cleanup_warning = False
    if staged_directory and staged_directory.exists():
        try:
            shutil.rmtree(staged_directory)
        except OSError:
            # The durable task is already gone.  Keep the hidden tombstone for a
            # later maintenance cleanup instead of reporting that deletion did
            # not happen.
            cleanup_warning = True
    task.status = TaskStatus.DELETED
    task.current_node = "deleted"
    task.messages = [
        *task.messages,
        "任务、版本、反馈和导出记录已删除",
        *(["部分任务文件已隔离，等待后续清理"] if cleanup_warning else []),
    ]
    return task_view(task)


@app.post("/api/tasks/{task_id}/patch", response_model=TaskView)
async def apply_patch(task_id: str, request: PatchRequest, _: None = Depends(require_internal_token)) -> TaskView:
    task = state().store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task/thread not found")
    _require_task_checkpoint(task, request.thread_id, request.checkpoint_version)
    if task.status == TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail={"error_code": "formal_snapshot_immutable", "message": "已确认的正式快照不可原地修改，请创建新的简历任务"},
        )
    target = next((item for item in task.candidates if item.candidate_id == request.target_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="patch target not found")
    from app.core.utils import sha256_text

    field_name = request.module.strip().lower()
    if field_name not in {"title", "period", "introduction", "tech_stack", "solutions", "results"}:
        raise HTTPException(status_code=422, detail={"error_code": "unsupported_patch_module", "message": "不支持的项目字段"})
    old_value: Any = getattr(target, field_name)
    old_serialized = old_value if isinstance(old_value, str) else json.dumps(old_value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    old_hashes = {sha256_text(old_serialized)}
    if not isinstance(old_value, str):
        # Accept the ordinary JSON rendering used by browser clients as well as
        # the canonical form persisted by the server; both still bind the patch
        # to the exact pre-edit value.
        old_hashes.add(sha256_text(json.dumps(old_value, ensure_ascii=False)))
    if request.old_value_hash not in old_hashes:
        raise HTTPException(status_code=409, detail={"error_code": "patch_conflict", "message": "目标内容已变化"})
    if isinstance(old_value, list):
        try:
            parsed = json.loads(request.new_value)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail={"error_code": "invalid_patch_value", "message": "列表字段需要 JSON 数组"}) from exc
        if not isinstance(parsed, list):
            raise HTTPException(status_code=422, detail={"error_code": "invalid_patch_value", "message": "列表字段需要 JSON 数组"})
        setattr(target, field_name, [str(item) for item in parsed])
    else:
        setattr(target, field_name, request.new_value)
    target.needs_verification = True
    try:
        updated = state().store.checkpoint(
            task_id,
            request.checkpoint_version,
            state={**task.state, "last_patch": request.model_dump(mode="json")},
            candidates=task.candidates,
            messages=[*task.messages, "字段级补丁已应用到新草稿"],
        )
    except VersionConflict as exc:
        raise HTTPException(status_code=409, detail={"error_code": "checkpoint_conflict", "current_version": exc.current}) from exc
    if state().db_mirror:
        if not state().db_mirror.persist_checkpoint(updated):
            raise HTTPException(status_code=503, detail={"error_code": "checkpoint_persistence_conflict", "message": "数据库中的任务版本已变化，请刷新后重试"})
        if hasattr(state().db_mirror, "persist_content_patch"):
            patched_candidate = next(item for item in updated.candidates if item.candidate_id == request.target_id)
            audited = state().db_mirror.persist_content_patch(
                task_id,
                target_id=request.target_id,
                module=field_name,
                old_value_hash=request.old_value_hash,
                new_value=getattr(patched_candidate, field_name),
                reason=request.reason,
                evidence_ids=request.evidence_ids,
                old_value=old_value,
            )
            if not audited:
                raise HTTPException(
                    status_code=409,
                    detail={"error_code": "patch_audit_conflict", "message": "补丁未能与持久化版本对应，请刷新后重试"},
                )
    return task_view(updated)


@app.get("/api/tasks/{task_id}/events-json")
async def task_events_json(task_id: str, _: None = Depends(require_internal_token)) -> list[dict[str, Any]]:
    return state().store.events_for(task_id)


@app.post("/api/backups/create")
async def create_backup_api(_: None = Depends(require_internal_token)) -> dict[str, Any]:
    return create_backup(state().settings.data_root)


@app.post("/api/backups/validate")
async def validate_backup_api(payload: dict[str, Any] = Body(default_factory=dict), _: None = Depends(require_internal_token)) -> dict[str, Any]:
    archive = Path(str(payload.get("path") or "")).resolve()
    root = state().settings.data_root.resolve()
    if root not in archive.parents:
        raise HTTPException(status_code=422, detail="备份路径必须位于 data_root 内")
    try:
        return validate_backup(archive)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail={"error_code": "backup_invalid", "message": str(exc)}) from exc


@app.post("/api/backups/restore")
async def restore_backup_api(payload: dict[str, Any] = Body(default_factory=dict), _: None = Depends(require_internal_token)) -> dict[str, Any]:
    archive = Path(str(payload.get("path") or "")).resolve()
    root = state().settings.data_root.resolve()
    if root not in archive.parents:
        raise HTTPException(status_code=422, detail="备份路径必须位于 data_root 内")
    try:
        return restore_backup(archive, root, overwrite=bool(payload.get("overwrite", False)))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail={"error_code": "backup_restore_failed", "message": str(exc)}) from exc


_BUILTIN_TEMPLATE = """# 中文技术简历\n\n## 个人简介\n\n[待补充]\n\n## 技能清单\n\n[待补充]\n\n## 项目经历\n\n[待补充]\n"""


def _resolve_cached_template_source(template_id: str) -> Path:
    """Resolve one selected template without accepting arbitrary local paths."""

    template_id = str(template_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", template_id):
        raise HTTPException(
            status_code=422,
            detail={"error_code": "template_id_invalid", "message": "模板 ID 不合法"},
        )
    cache_root = (state().settings.data_root / "templates" / "cache").resolve()
    template_root = (cache_root / template_id).resolve()
    if template_root.parent != cache_root:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "template_path_invalid", "message": "模板缓存路径不合法"},
        )
    if template_id == "builtin-cn-single-page":
        return write_preview(_BUILTIN_TEMPLATE, root=template_root, name="template.md").resolve()
    if not template_root.is_dir():
        raise HTTPException(
            status_code=404,
            detail={"error_code": "template_not_cached", "message": "模板尚未下载到本机"},
        )
    metadata_path = template_root / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        cached_name = Path(str(metadata.get("filename") or "")).name
        if cached_name:
            preferred = (template_root / cached_name).resolve()
            if (
                preferred.parent == template_root
                and preferred.is_file()
                and preferred.suffix.casefold() in {".md", ".markdown", ".docx"}
            ):
                return preferred
    candidates = sorted(
        item.resolve()
        for item in template_root.iterdir()
        if item.is_file() and item.suffix.casefold() in {".md", ".markdown", ".docx"}
    )
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "template_not_cached", "message": "模板缓存文件不存在"},
        )
    return candidates[0]


def _github_token(payload: dict[str, Any]) -> str | None:
    handle_id = payload.get("credential_handle_id")
    if not handle_id:
        return None
    try:
        return state().credentials.get(str(handle_id), scope="github")
    except Exception as exc:
        raise HTTPException(
            status_code=401,
            detail={"error_code": "credential_missing", "message": "GitHub 凭据句柄无效或已过期"},
        ) from exc


def _builtin_template_item() -> dict[str, Any]:
    return {
        "template_id": "builtin-cn-single-page",
        "name": "内置中文单页模板",
        "file_type": "markdown",
        "language": "zh",
        "license": "internal",
        "stars": None,
        "preview_available": True,
    }


def _persist_template_result(result: dict[str, Any]) -> None:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else result
    if state().db_mirror and hasattr(state().db_mirror, "persist_template"):
        state().db_mirror.persist_template(dict(metadata))


async def _attach_template_visual_preview(
    result: dict[str, Any],
    *,
    requested: bool,
) -> dict[str, Any]:
    """Attach a browser-renderable preview for one already cached template.

    A DOCX preview is generated only for the currently selected template, not
    for every search result.  This keeps GitHub discovery responsive and avoids
    returning arbitrary local files: both source and generated PDF must remain
    under the configured template cache directory.
    """

    if not requested:
        return result
    cache_root = (state().settings.data_root / "templates" / "cache").resolve()
    raw_path = str(result.get("path") or "").strip()
    if not raw_path:
        return {
            **result,
            "visual_preview": {
                "status": "unavailable",
                "message": "模板尚未下载到本机，无法生成排版预览",
            },
        }
    source = Path(raw_path).resolve()
    if cache_root != source and cache_root not in source.parents:
        return {
            **result,
            "visual_preview": {
                "status": "unavailable",
                "message": "模板路径不在受信任缓存目录内",
            },
        }
    if not source.is_file():
        return {
            **result,
            "visual_preview": {
                "status": "unavailable",
                "message": "本机模板缓存文件不存在",
            },
        }
    if source.suffix.casefold() in {".md", ".markdown"}:
        content = source.read_text(encoding="utf-8-sig")
        return {
            **result,
            "visual_preview": {
                "status": "ready",
                "mime_type": "text/markdown",
                "content": content,
            },
        }
    if source.suffix.casefold() != ".docx":
        return {
            **result,
            "visual_preview": {
                "status": "unavailable",
                "message": "该模板格式暂不支持排版预览",
            },
        }
    preview_root = (source.parent / "preview").resolve()
    if cache_root not in preview_root.parents:
        return {
            **result,
            "visual_preview": {
                "status": "unavailable",
                "message": "模板预览目录无效",
            },
        }
    preview_root.mkdir(parents=True, exist_ok=True)
    pdf_path = preview_root / f"{source.stem}.pdf"
    needs_render = (
        not pdf_path.is_file()
        or pdf_path.stat().st_mtime_ns < source.stat().st_mtime_ns
    )
    conversion = None
    if needs_render:
        conversion = await asyncio.to_thread(convert_docx_to_pdf, source, pdf_path)
        if conversion.status != "ready" or not conversion.path:
            try:
                browser_html = await asyncio.to_thread(
                    render_docx_html_preview,
                    source.read_bytes(),
                )
            except (OSError, ValueError):
                browser_html = ""
            if browser_html:
                return {
                    **result,
                    "visual_preview": {
                        "status": "ready",
                        "mime_type": "text/html",
                        "content": browser_html,
                        "approximate_layout": True,
                        "message": (
                            "当前环境无法调用 Word/LibreOffice，已生成安全的浏览器排版预览"
                        ),
                    },
                }
            return {
                **result,
                "visual_preview": {
                    "status": conversion.status,
                    "message": "；".join(conversion.warnings)
                    or "无法使用本机 Office 渲染该模板",
                },
            }
    try:
        data = pdf_path.read_bytes()
    except OSError as exc:
        return {
            **result,
            "visual_preview": {
                "status": "unavailable",
                "message": f"模板 PDF 预览读取失败：{type(exc).__name__}",
            },
        }
    if len(data) > 12 * 1024 * 1024:
        return {
            **result,
            "visual_preview": {
                "status": "too_large",
                "message": "模板 PDF 预览超过 12MB，请下载后在 Word 中查看",
            },
        }
    return {
        **result,
        "visual_preview": {
            "status": "ready",
            "mime_type": "application/pdf",
            "filename": pdf_path.name,
            "content_base64": base64.b64encode(data).decode("ascii"),
            "page_count": conversion.page_count if conversion is not None else None,
        },
    }


@app.post("/api/templates/search")
async def search_templates(payload: dict[str, Any] | None = Body(default=None), _: None = Depends(require_internal_token)) -> dict[str, Any]:
    body = payload or {}
    query = str(body.get("query", "中文简历模板"))
    source = str(body.get("source", "github")).lower()
    items: list[dict[str, Any]] = []
    error_code: str | None = None
    result_source = "builtin_fallback"
    if source == "github":
        token = _github_token(body)
        page = max(1, int(body.get("page", 1) or 1))
        per_page = min(5, max(1, int(body.get("per_page", 5) or 5)))
        health = await _ensure_mcp_server("github")
        if health.get("status") == "ready":
            arguments: dict[str, Any] = {
                "query": query,
                "language": "zh",
                "page": page,
                "per_page": per_page,
            }
            if body.get("file_type"):
                arguments["file_type"] = str(body["file_type"])
            if token:
                arguments["github_token"] = token
            try:
                remote_result = await state().mcp_manager.call_tool(
                    "github", "github_search_resume_templates", arguments
                )
            except MCPClientError as exc:
                error_code = exc.error_code
            else:
                if remote_result.get("ok") and isinstance(remote_result.get("data"), dict):
                    items = [
                        item
                        for item in remote_result["data"].get("items", [])
                        if isinstance(item, dict) and item.get("license_allowed") is True
                    ]
                    result_source = "github_mcp"
                else:
                    error_code = str(remote_result.get("error_code") or "github_unavailable")
        else:
            error_code = str(health.get("error_code") or "mcp_unavailable")

        # Keep first-run usable when the optional MCP SDK is not installed.
        if not items and error_code:
            try:
                direct = await asyncio.to_thread(
                    search_github_templates,
                    query=query,
                    language="zh",
                    file_type=str(body.get("file_type") or "") or None,
                    page=page,
                    per_page=per_page,
                    token=token,
                )
                items = [item for item in direct.get("items", []) if item.get("license_allowed") is True]
                if items:
                    result_source = "github_direct_fallback"
            except Exception:
                pass
    if not items:
        items = [_builtin_template_item()]
    return {
        "status": "ready",
        "source": result_source,
        "query": query,
        "items": items[:5],
        "error_code": error_code if result_source == "builtin_fallback" else None,
        "message": "GitHub 模板服务不可用，已使用内置模板" if result_source == "builtin_fallback" and error_code else None,
    }


@app.post("/api/templates/{template_id}/preview")
async def preview_template(
    template_id: str,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    if template_id == "builtin-cn-single-page":
        path = write_preview(_BUILTIN_TEMPLATE, root=state().settings.data_root / "templates" / "cache" / template_id, name="template.md")
        result = {
            "status": "ready",
            "template_id": template_id,
            "path": str(path),
            "content": _BUILTIN_TEMPLATE,
            "source_url": "internal://builtin-cn-single-page",
            "file_type": "markdown",
            "language": "zh",
            "license": "internal",
            "license_allowed": True,
        }
        _persist_template_result(result)
        return await _attach_template_visual_preview(
            result,
            requested=bool((payload or {}).get("render_visual", False)),
        )
    body = payload or {}
    token = _github_token(body)
    health = await _ensure_mcp_server("github")
    if health.get("status") == "ready":
        arguments: dict[str, Any] = {"template_id": template_id}
        if body.get("source_url"):
            arguments = {"source_url": str(body["source_url"])}
        if token:
            arguments["github_token"] = token
        try:
            remote = await state().mcp_manager.call_tool(
                "github", "github_preview_template", arguments
            )
        except MCPClientError:
            remote = {"ok": False, "error_code": "mcp_unavailable"}
        if remote.get("ok") and isinstance(remote.get("data"), dict):
            result = {
                "template_id": template_id,
                **dict(remote["data"]),
                "source": "github_mcp",
            }
            return await _attach_template_visual_preview(
                result,
                requested=bool(body.get("render_visual", False)),
            )

    try:
        if body.get("source_url"):
            direct = await asyncio.to_thread(
                preview_github_template, str(body["source_url"]), token=token
            )
        else:
            cache_root = (state().settings.data_root / "templates" / "cache").resolve()
            template_root = (cache_root / template_id).resolve()
            if template_root.parent != cache_root or not template_root.is_dir():
                raise FileNotFoundError("模板不存在或尚未缓存")
            candidates = sorted(
                item for item in template_root.iterdir()
                if item.is_file() and item.suffix.lower() in {".md", ".markdown", ".docx"}
            )
            if not candidates:
                raise FileNotFoundError("模板缓存文件不存在")
            direct = preview_cached_template(candidates[0], cache_root=cache_root)
    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=404,
            detail={"error_code": "template_preview_unavailable", "message": str(exc)},
        ) from exc
    result = {
        "status": "ready",
        "template_id": template_id,
        **direct,
        "source": "local_fallback",
    }
    return await _attach_template_visual_preview(
        result,
        requested=bool(body.get("render_visual", False)),
    )


@app.get("/api/templates/{template_id}/structure")
async def get_template_structure(
    template_id: str,
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    source = _resolve_cached_template_source(template_id)
    structure = load_template_structure(template_id, source)
    if structure is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error_code": "template_structure_not_found",
                "message": "该模板尚未使用聊天模型解析",
            },
        )
    return structure


@app.post("/api/templates/{template_id}/model-parse")
async def model_parse_template(
    template_id: str,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    """Map a cached template to editable resume modules with a selected model."""

    source = _resolve_cached_template_source(template_id)
    body = payload or {}
    profile_id = str(body.get("chat_profile_id") or "").strip()
    profile = state().store.get_profile(profile_id) if profile_id else next(
        (
            item
            for item in state().store.list_profiles(ModelRole.CHAT.value)
            if item.default_for_role and item.status == "ready"
        ),
        None,
    )

    def unavailable(message: str, error_code: str) -> dict[str, Any]:
        return {
            "status": "fallback",
            "template_id": template_id,
            "structure": None,
            "retryable": True,
            "parser": {
                "parser_mode": "unavailable",
                "model_profile_id": getattr(profile, "profile_id", None),
                "model_name": getattr(profile, "model_name", None),
                "error_code": error_code,
                "warnings": [message],
            },
        }

    if profile is None:
        return unavailable("请选择一个已经检测成功的聊天模型", "chat_profile_not_found")
    if profile.role is not ModelRole.CHAT:
        return unavailable("所选配置不是聊天模型", "invalid_chat_profile_role")
    if profile.status != "ready":
        return unavailable("所选聊天模型尚未检测成功", "model_not_ready")
    handle_id = str(body.get("credential_handle_id") or "").strip() or None
    if profile.provider is Provider.OPENAI_COMPATIBLE:
        handle_id = _materialize_model_credential(profile.profile_id, handle_id)
        if not handle_id or not state().credentials.has(
            handle_id, scope=f"model:{profile.profile_id}"
        ):
            return unavailable("云端模型 API Key 无法恢复", "credential_missing")
        if (
            getattr(state().settings, "external_model_consent_required", True)
            and not state().consents.get("external_model:global", {}).get("granted", False)
        ):
            return unavailable(
                "尚未确认向外部模型发送模板文字",
                "needs_external_model_consent",
            )
    try:
        structure = await asyncio.to_thread(
            structure_template_with_model,
            template_id,
            source,
            profile=profile,
            gateway=state().model_gateway,
            credential_handle_id=handle_id,
        )
    except Exception as exc:
        from app.core.utils import redact_sensitive

        reason = str(redact_sensitive(str(exc) or type(exc).__name__)).strip()
        if len(reason) > 240:
            reason = f"{reason[:237]}..."
        return unavailable(
            "聊天模型解析模板结构失败" + (f"。原因：{reason}" if reason else ""),
            str(getattr(exc, "code", "template_model_parse_failed")),
        )
    return structure


@app.post("/api/templates/{template_id}/confirm-structure")
async def confirm_template_structure_endpoint(
    template_id: str,
    payload: dict[str, Any] = Body(default_factory=dict),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    source = _resolve_cached_template_source(template_id)
    try:
        return confirm_template_structure(
            template_id,
            source,
            structure_id=str(payload.get("structure_id") or ""),
            source_sha256=str(payload.get("source_sha256") or ""),
            sections=payload.get("sections"),
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error_code": "template_structure_conflict", "message": str(exc)},
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error_code": "template_model_parse_required", "message": str(exc)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "template_structure_invalid", "message": str(exc)},
        ) from exc


@app.post("/api/templates/{template_id}/download")
async def download_template(template_id: str, payload: dict[str, Any] | None = Body(default=None), _: None = Depends(require_internal_token)) -> dict[str, Any]:
    if template_id == "builtin-cn-single-page":
        path = write_preview(_BUILTIN_TEMPLATE, root=state().settings.data_root / "templates" / "cache" / template_id, name="template.md")
        result = {
            "status": "ready",
            "template_id": template_id,
            "path": str(path),
            "content": _BUILTIN_TEMPLATE,
            "source_url": "internal://builtin-cn-single-page",
            "file_type": "markdown",
            "language": "zh",
            "license": "internal",
            "license_allowed": True,
        }
        _persist_template_result(result)
        return result
    body = payload or {}
    source_url = str(body.get("source_url") or "")
    if not source_url:
        raise HTTPException(status_code=422, detail={"error_code": "template_source_required", "message": "远程模板需要 source_url"})
    token = _github_token(body)
    arguments = {
        "source_url": source_url,
        "repository": str(body.get("repository") or "") or None,
        "file_path": str(body.get("file_path") or "") or None,
        "filename": str(body.get("filename") or "") or None,
        "remote_version": str(body.get("remote_version") or "") or None,
        "confirm_update": bool(body.get("confirm_update", False)),
    }
    arguments = {key: value for key, value in arguments.items() if value is not None}
    if token:
        arguments["github_token"] = token
    health = await _ensure_mcp_server("github")
    if health.get("status") == "ready":
        try:
            remote = await state().mcp_manager.call_tool(
                "github", "github_download_template", arguments
            )
        except MCPClientError:
            remote = {"ok": False, "error_code": "mcp_unavailable"}
        if remote.get("ok") and isinstance(remote.get("data"), dict):
            result = {**dict(remote["data"]), "source": "github_mcp"}
            _persist_template_result(result)
            return result
        if remote.get("requires_user"):
            raise HTTPException(
                status_code=409,
                detail={
                    "error_code": remote.get("error_code") or "template_download_failed",
                    "message": "模板下载需要用户处理 GitHub 凭据或更新确认",
                },
            )
    try:
        result = await asyncio.to_thread(
            cache_github_template,
            source_url=source_url,
            cache_root=state().settings.data_root / "templates" / "cache",
            token=token,
            repository=str(body.get("repository") or "") or None,
            file_path=str(body.get("file_path") or "") or None,
            remote_version=str(body.get("remote_version") or "") or None,
            filename=str(body.get("filename") or "") or None,
            allow_overwrite=bool(body.get("confirm_update", False)),
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail={"error_code": "template_download_failed", "message": str(exc)}) from exc
    response = {"status": "cached", **result, "source": "local_fallback"}
    _persist_template_result(response)
    return response


@app.post("/api/templates/{template_id}/migrate")
async def migrate_template(template_id: str, payload: dict[str, Any] | None = Body(default=None), _: None = Depends(require_internal_token)) -> dict[str, Any]:
    if template_id != "builtin-cn-single-page":
        raise HTTPException(status_code=404, detail={"error_code": "template_not_found", "message": "模板不存在"})
    return {"status": "waiting_user", "template_id": template_id, "requires_layout_confirmation": True, "message": "模板迁移仅生成预览，确认前不会覆盖原简历"}


@app.post("/api/templates/{template_id}/check-update")
async def check_template_update(
    template_id: str,
    payload: dict[str, Any] | None = Body(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    if template_id == "builtin-cn-single-page":
        return {"template_id": template_id, "status": "up_to_date", "auto_overwrite": False}
    body = payload or {}
    token = _github_token(body)
    health = await _ensure_mcp_server("github")
    if health.get("status") == "ready":
        arguments: dict[str, Any] = {"template_id": template_id}
        if token:
            arguments["github_token"] = token
        try:
            remote = await state().mcp_manager.call_tool(
                "github", "github_check_template_update", arguments
            )
        except MCPClientError:
            remote = {"ok": False}
        if isinstance(remote.get("data"), dict):
            return {**dict(remote["data"]), "source": "github_mcp"}
    try:
        direct = await asyncio.to_thread(
            check_github_template_update,
            cache_root=state().settings.data_root / "templates" / "cache",
            template_id=template_id,
            token=token,
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "template_update_check_failed", "message": str(exc)},
        ) from exc
    return {**direct, "source": "local_fallback"}


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=False)


if __name__ == "__main__":
    run()
