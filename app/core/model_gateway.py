"""Provider-neutral chat and embedding model gateway.

This module deliberately uses only the Python standard library.  The application
can install LangChain later and adapt its ``BaseChatModel``/embedding interfaces to
the ports below, while tests and a minimal local install remain usable without
third-party HTTP clients.

Secrets are represented by short-lived handles from :class:`CredentialStore` and
are never part of a :class:`ModelProfile` or a probe response.
"""

from __future__ import annotations

import copy
import json
import math
import secrets
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence


DEFAULT_CHAT_MODEL = "qwen2.5:7b"
DEFAULT_EMBEDDING_MODEL = "bge-m3"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
MODEL_GATEWAY_VERSION = "model-gateway-v1"


class ModelRole(str, Enum):
    CHAT = "chat"
    EMBEDDING = "embedding"


class Provider(str, Enum):
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


class ProbeStatus(str, Enum):
    READY = "ready"
    EXECUTABLE_MISSING = "executable_missing"
    SERVICE_UNREACHABLE = "service_unreachable"
    MODEL_NOT_INSTALLED = "model_not_installed"
    CAPABILITY_MISMATCH = "capability_mismatch"
    PROBE_FAILED = "probe_failed"
    CREDENTIAL_MISSING = "credential_missing"
    AUTH_FAILED = "auth_failed"


class GatewayError(RuntimeError):
    """A provider-independent error returned by an adapter."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        requires_user: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status_code = status_code
        self.requires_user = requires_user


class CredentialUnavailable(GatewayError):
    def __init__(self, message: str = "credential is missing or expired") -> None:
        super().__init__("credential_missing", message, requires_user=True)


@dataclass(frozen=True)
class CredentialHandle:
    """Opaque identifier returned to the API layer; it contains no secret."""

    handle_id: str
    scope: str
    expires_at: float

    @property
    def expires_at_iso(self) -> str:
        return datetime.fromtimestamp(self.expires_at, tz=timezone.utc).isoformat()


@dataclass
class _CredentialEntry:
    secret: str = field(repr=False)
    scope: str
    expires_at: float


class CredentialStore:
    """Thread-safe in-memory credentials with TTL and explicit scope.

    The store is intentionally process-local.  Do not serialize this object or its
    entries.  A worker can receive only a ``handle_id`` and resolve it while the
    process is alive; expiry/restart naturally causes a credential gate.
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.time
        self._entries: dict[str, _CredentialEntry] = {}
        self._lock = threading.RLock()

    def put(self, secret: str, *, scope: str, ttl_seconds: int = 300) -> CredentialHandle:
        if not isinstance(secret, str) or not secret:
            raise ValueError("secret must be a non-empty string")
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("scope must be a non-empty string")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        now = self._clock()
        handle = CredentialHandle(
            handle_id=secrets.token_urlsafe(24),
            scope=scope.strip(),
            expires_at=now + ttl_seconds,
        )
        with self._lock:
            self._purge_locked(now)
            self._entries[handle.handle_id] = _CredentialEntry(
                secret=secret, scope=handle.scope, expires_at=handle.expires_at
            )
        return handle

    # Explicit aliases keep the API layer readable while preserving ``put/get``
    # as the small internal primitive used by tests and workers.
    def create(self, secret: str, *, scope: str, ttl_seconds: int = 300) -> CredentialHandle:
        return self.put(secret, scope=scope, ttl_seconds=ttl_seconds)

    def get(self, handle_id: str, *, scope: str | None = None) -> str:
        if not handle_id:
            raise CredentialUnavailable()
        now = self._clock()
        with self._lock:
            entry = self._entries.get(handle_id)
            if entry is None or entry.expires_at <= now:
                self._entries.pop(handle_id, None)
                raise CredentialUnavailable()
            if scope is not None and entry.scope != scope:
                raise CredentialUnavailable("credential scope does not match")
            return entry.secret

    def resolve(self, handle_id: str, *, scope: str | None = None) -> str:
        return self.get(handle_id, scope=scope)

    def revoke(self, handle_id: str) -> bool:
        with self._lock:
            return self._entries.pop(handle_id, None) is not None

    def delete(self, handle_id: str) -> bool:
        return self.revoke(handle_id)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def purge(self) -> int:
        with self._lock:
            return self._purge_locked(self._clock())

    def _purge_locked(self, now: float) -> int:
        expired = [key for key, value in self._entries.items() if value.expires_at <= now]
        for key in expired:
            self._entries.pop(key, None)
        return len(expired)

    def has(self, handle_id: str, *, scope: str | None = None) -> bool:
        try:
            self.get(handle_id, scope=scope)
        except CredentialUnavailable:
            return False
        return True


@dataclass(frozen=True)
class GenerationParameters:
    temperature: float = 0.2
    top_p: float = 1.0
    max_output_tokens: int = 1200
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_interval_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be in [0, 2]")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.timeout_seconds <= 0 or self.max_retries < 0 or self.retry_interval_seconds < 0:
            raise ValueError("invalid generation timeout/retry settings")


@dataclass(frozen=True)
class EmbeddingParameters:
    batch_size: int = 16
    normalize: bool = True
    distance: str = "cosine"
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_interval_seconds: float = 0.5
    max_input_tokens: int | None = None
    chunk_policy_version: str = "embedding-chunk-v1"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.distance not in {"cosine", "dot", "euclidean"}:
            raise ValueError("distance must be cosine, dot, or euclidean")
        if self.timeout_seconds <= 0 or self.max_retries < 0 or self.retry_interval_seconds < 0:
            raise ValueError("invalid embedding timeout/retry settings")
        if self.max_input_tokens is not None and self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")


def _default_context_window(role: ModelRole) -> int | None:
    # Ollama metadata often omits this value.  Returning None keeps the task gate
    # conservative; callers may set an explicit, versioned value after probing.
    return None if role is ModelRole.EMBEDDING else None


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    role: ModelRole
    provider: Provider
    base_url: str
    model_name: str
    profile_version: str = "v1"
    credential_required: bool = False
    context_window_tokens: int | None = None
    tokenizer_id: str | None = None
    tokenizer_version: str | None = None
    tokenizer_source: str | None = None
    estimator_version: str = "conservative-char-v1"
    generation: GenerationParameters = field(default_factory=GenerationParameters)
    embedding: EmbeddingParameters = field(default_factory=EmbeddingParameters)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        role = self.role if isinstance(self.role, ModelRole) else ModelRole(self.role)
        provider = self.provider if isinstance(self.provider, Provider) else Provider(self.provider)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "provider", provider)
        if not self.profile_id.strip() or not self.model_name.strip():
            raise ValueError("profile_id and model_name are required")
        normalized = normalize_base_url(self.base_url)
        object.__setattr__(self, "base_url", normalized)
        if provider is Provider.OPENAI_COMPATIBLE:
            _assert_external_https(self)
        if self.context_window_tokens is not None and self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive when set")
        if role is ModelRole.EMBEDDING and self.context_window_tokens is not None:
            # It is harmless to keep the value for metadata, but embedding requests
            # use ``embedding.max_input_tokens`` and never this chat window.
            pass

    @classmethod
    def default_chat(cls, *, profile_id: str = "ollama-chat-default") -> "ModelProfile":
        return cls(
            profile_id=profile_id,
            role=ModelRole.CHAT,
            provider=Provider.OLLAMA,
            base_url=DEFAULT_OLLAMA_URL,
            model_name=DEFAULT_CHAT_MODEL,
        )

    @classmethod
    def default_embedding(cls, *, profile_id: str = "ollama-embedding-default") -> "ModelProfile":
        return cls(
            profile_id=profile_id,
            role=ModelRole.EMBEDDING,
            provider=Provider.OLLAMA,
            base_url=DEFAULT_OLLAMA_URL,
            model_name=DEFAULT_EMBEDDING_MODEL,
        )

    def public_dict(self) -> dict[str, Any]:
        """Return a persistence/API-safe representation with no credentials."""
        return {
            "profile_id": self.profile_id,
            "role": self.role.value,
            "provider": self.provider.value,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "profile_version": self.profile_version,
            "credential_required": self.credential_required,
            "context_window_tokens": self.context_window_tokens,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_version": self.tokenizer_version,
            "tokenizer_source": self.tokenizer_source,
            "estimator_version": self.estimator_version,
            "generation": {
                "temperature": self.generation.temperature,
                "top_p": self.generation.top_p,
                "max_output_tokens": self.generation.max_output_tokens,
                "timeout_seconds": self.generation.timeout_seconds,
                "max_retries": self.generation.max_retries,
                "retry_interval_seconds": self.generation.retry_interval_seconds,
            },
            "embedding": {
                "batch_size": self.embedding.batch_size,
                "normalize": self.embedding.normalize,
                "distance": self.embedding.distance,
                "timeout_seconds": self.embedding.timeout_seconds,
                "max_retries": self.embedding.max_retries,
                "retry_interval_seconds": self.embedding.retry_interval_seconds,
                "max_input_tokens": self.embedding.max_input_tokens,
                "chunk_policy_version": self.embedding.chunk_policy_version,
            },
            "metadata": _redact_metadata(self.metadata),
        }

    def snapshot(self, *, version: str | None = None, **changes: Any) -> "ModelProfile":
        """Create an immutable task profile version instead of mutating this one."""
        if version is None:
            version = f"{self.profile_version}.next"
        changes["profile_version"] = version
        return replace(self, **changes)

    @classmethod
    def from_object(cls, value: Any) -> "ModelProfile":
        """Convert the API/Pydantic profile contract without importing Pydantic.

        The persistence/API layer has its own schema model.  Keeping this adapter
        duck-typed avoids a hard dependency and makes the provider ports usable in
        small scripts and tests as well as the FastAPI application.
        """
        if isinstance(value, cls):
            return value
        if hasattr(value, "model_dump"):
            data = value.model_dump()
        elif isinstance(value, Mapping):
            data = dict(value)
        else:
            data = {
                key: getattr(value, key)
                for key in (
                    "profile_id",
                    "role",
                    "provider",
                    "base_url",
                    "model_name",
                    "profile_version",
                    "config_version",
                    "credential_required",
                    "context_window_tokens",
                    "tokenizer_id",
                    "tokenizer_version",
                    "tokenizer_source",
                    "generation_params",
                    "max_input_tokens",
                    "capabilities",
                )
                if hasattr(value, key)
            }
        generation_data = dict(data.get("generation", data.get("generation_params", {})) or {})
        embedding_data = dict(data.get("embedding", {}) or {})
        if data.get("max_input_tokens") is not None:
            embedding_data.setdefault("max_input_tokens", data["max_input_tokens"])
        version = data.get("profile_version")
        if version is None and data.get("config_version") is not None:
            version = f"v{data['config_version']}"
        return cls(
            profile_id=str(data["profile_id"]),
            role=ModelRole(data["role"]),
            provider=Provider(data["provider"]),
            base_url=str(data["base_url"]),
            model_name=str(data["model_name"]),
            profile_version=str(version or "v1"),
            credential_required=bool(data.get("credential_required", False)),
            context_window_tokens=data.get("context_window_tokens"),
            tokenizer_id=data.get("tokenizer_id"),
            tokenizer_version=data.get("tokenizer_version"),
            tokenizer_source=data.get("tokenizer_source"),
            estimator_version=str(data.get("estimator_version", "conservative-char-v1")),
            generation=GenerationParameters(**{
                key: generation_data[key]
                for key in (
                    "temperature",
                    "top_p",
                    "max_output_tokens",
                    "timeout_seconds",
                    "max_retries",
                    "retry_interval_seconds",
                )
                if key in generation_data
            }),
            embedding=EmbeddingParameters(**{
                key: embedding_data[key]
                for key in (
                    "batch_size",
                    "normalize",
                    "distance",
                    "timeout_seconds",
                    "max_retries",
                    "retry_interval_seconds",
                    "max_input_tokens",
                    "chunk_policy_version",
                )
                if key in embedding_data
            }),
            metadata={
                "capabilities": copy.deepcopy(data.get("capabilities", {})),
                "dimension": data.get("dimension"),
            },
        )


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {self.role}")


@dataclass(frozen=True)
class ProbeResult:
    role: ModelRole
    provider: Provider
    status: ProbeStatus
    base_url: str
    endpoint: str | None
    model_name: str
    model_digest: str | None = None
    dimension: int | None = None
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    checked_at: str = ""
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    requires_user: bool = False
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checked_at:
            object.__setattr__(
                self, "checked_at", datetime.now(timezone.utc).isoformat()
            )

    @property
    def ready(self) -> bool:
        return self.status is ProbeStatus.READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "provider": self.provider.value,
            "status": self.status.value,
            "base_url": self.base_url,
            "endpoint": self.endpoint,
            "model_name": self.model_name,
            "model_digest": self.model_digest,
            "dimension": self.dimension,
            "capabilities": dict(self.capabilities),
            "checked_at": self.checked_at,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "retryable": self.retryable,
            "requires_user": self.requires_user,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ChatResult:
    content: str
    model_name: str
    profile_version: str
    request_key: str | None = None
    finish_reason: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    capabilities: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...]
    model_name: str
    profile_version: str
    dimension: int
    request_key: str | None = None
    normalized: bool = False


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GatewayError("invalid_json", "provider returned invalid JSON") from exc


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Any = None,
        timeout: float = 60.0,
    ) -> HttpResponse:
        ...


class UrllibTransport:
    """Small standard-library HTTP transport with bounded response size."""

    def __init__(self, *, max_response_bytes: int = 8 * 1024 * 1024) -> None:
        self.max_response_bytes = max_response_bytes

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Any = None,
        timeout: float = 60.0,
    ) -> HttpResponse:
        payload = None
        request_headers = {"Accept": "application/json", **dict(headers or {})}
        if json_body is not None:
            payload = json.dumps(json_body, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            request_headers.setdefault("Content-Type", "application/json")
        request = urllib.request.Request(
            url=url, data=payload, headers=request_headers, method=method.upper()
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read(self.max_response_bytes + 1)
                if len(body) > self.max_response_bytes:
                    raise GatewayError("response_too_large", "provider response exceeds limit")
                return HttpResponse(
                    status_code=int(response.status),
                    headers={str(k).lower(): str(v) for k, v in response.headers.items()},
                    body=body,
                )
        except GatewayError:
            raise
        except urllib.error.HTTPError as exc:
            body = exc.read(self.max_response_bytes + 1)
            return HttpResponse(
                status_code=int(exc.code),
                headers={str(k).lower(): str(v) for k, v in exc.headers.items()},
                body=body,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise GatewayError(
                "service_unreachable", str(exc), retryable=True
            ) from exc


def normalize_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("base_url is required")
    value = value.strip().rstrip("/")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("credentials must not be embedded in base_url")
    if parsed.query:
        raise ValueError("base_url must not contain a query string")
    if parsed.fragment:
        raise ValueError("base_url must not contain a fragment")
    return value


def _is_local_url(url: str) -> bool:
    hostname = (urllib.parse.urlparse(url).hostname or "").lower()
    return hostname in {"localhost", "127.0.0.1", "::1"}


def _assert_external_https(profile: ModelProfile) -> None:
    if profile.provider is Provider.OPENAI_COMPATIBLE:
        parsed = urllib.parse.urlparse(profile.base_url)
        if parsed.scheme != "https" and not _is_local_url(profile.base_url):
            raise ValueError("external OpenAI-compatible endpoints must use HTTPS")


def _endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _openai_endpoint(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    # Treat a trailing /v1 as the API root and add it exactly once otherwise.
    if not base.lower().endswith("/v1"):
        base += "/v1"
    return _endpoint(base, path)


def _headers_for(profile: ModelProfile, secret: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if profile.provider is Provider.OPENAI_COMPATIBLE:
        if profile.credential_required and not secret:
            raise CredentialUnavailable()
        if secret:
            headers["Authorization"] = f"Bearer {secret}"
    return headers


def _error_from_status(
    response: HttpResponse,
    *,
    role: ModelRole,
    secret: str | None = None,
) -> GatewayError:
    message = f"provider returned HTTP {response.status_code}"
    try:
        payload = response.json()
        if isinstance(payload, Mapping):
            error = payload.get("error")
            if isinstance(error, Mapping):
                message = str(error.get("message") or message)
            elif error:
                message = str(error)
            elif payload.get("message"):
                message = str(payload["message"])
    except GatewayError:
        pass
    if secret:
        message = message.replace(secret, "[redacted]")
    if response.status_code in {401, 403}:
        return GatewayError("auth_failed", message, status_code=response.status_code, requires_user=True)
    if response.status_code == 404:
        return GatewayError("endpoint_not_found", message, status_code=response.status_code)
    return GatewayError(
        "provider_http_error",
        message,
        retryable=response.status_code >= 500 or response.status_code == 429,
        status_code=response.status_code,
    )


def _safe_float_vector(value: Any) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise GatewayError("invalid_embedding", "embedding vector is not an array")
    result: list[float] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise GatewayError("invalid_embedding", "embedding contains non-numeric value") from exc
        if not math.isfinite(number):
            raise GatewayError("invalid_embedding", "embedding contains non-finite value")
        result.append(number)
    if not result:
        raise GatewayError("invalid_embedding", "embedding vector is empty")
    return tuple(result)


class ModelAdapter(Protocol):
    def probe(self, profile: ModelProfile, *, secret: str | None = None) -> ProbeResult:
        ...

    def chat(
        self,
        profile: ModelProfile,
        messages: Sequence[ChatMessage],
        *,
        secret: str | None = None,
        request_key: str | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        ...

    def embed(
        self,
        profile: ModelProfile,
        texts: Sequence[str],
        *,
        secret: str | None = None,
        request_key: str | None = None,
    ) -> EmbeddingResult:
        ...


class ChatModelPort(Protocol):
    """Provider-neutral chat capability consumed by graph nodes."""

    def chat(
        self,
        profile: ModelProfile,
        messages: Sequence[ChatMessage],
        *,
        secret: str | None = None,
        request_key: str | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        ...


class EmbeddingModelPort(Protocol):
    """Provider-neutral embedding capability consumed by graph nodes."""

    def embed(
        self,
        profile: ModelProfile,
        texts: Sequence[str],
        *,
        secret: str | None = None,
        request_key: str | None = None,
    ) -> EmbeddingResult:
        ...


class _BaseAdapter:
    provider: Provider

    def __init__(self, transport: HttpTransport | None = None) -> None:
        self.transport = transport or UrllibTransport()

    def _request(
        self,
        profile: ModelProfile,
        method: str,
        url: str,
        *,
        body: Any = None,
        secret: str | None = None,
    ) -> HttpResponse:
        _assert_external_https(profile)
        try:
            response = self.transport.request(
                method,
                url,
                headers=_headers_for(profile, secret),
                json_body=body,
                timeout=(
                    profile.generation.timeout_seconds
                    if profile.role is ModelRole.CHAT
                    else profile.embedding.timeout_seconds
                ),
            )
        except CredentialUnavailable:
            raise
        except GatewayError:
            raise
        if response.status_code < 200 or response.status_code >= 300:
            raise _error_from_status(response, role=profile.role, secret=secret)
        return response

    @staticmethod
    def _ensure_role(profile: ModelProfile, role: ModelRole) -> None:
        if profile.role is not role:
            raise ValueError(f"profile role {profile.role.value!r} cannot be used as {role.value!r}")

    @staticmethod
    def _with_retries(fn: Callable[[], Any], *, retries: int, interval: float) -> Any:
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return fn()
            except GatewayError as exc:
                last = exc
                if not exc.retryable or attempt >= retries:
                    raise
                if interval:
                    time.sleep(interval * (2**attempt))
        assert last is not None
        raise last


class OllamaAdapter(_BaseAdapter):
    provider = Provider.OLLAMA

    def __init__(self, transport: HttpTransport | None = None, *, base_url: str | None = None) -> None:
        super().__init__(transport)
        self.base_url = normalize_base_url(base_url) if base_url else DEFAULT_OLLAMA_URL

    def scan(self, base_url: str | None = None) -> dict[str, Any]:
        """Return the exact models reported by Ollama's /api/tags endpoint.

        Scanning is intentionally independent of role: ``/api/show`` and the
        role-specific probe decide whether a selected model is chat or embedding.
        """
        base_url = base_url or getattr(self, "base_url", None) or DEFAULT_OLLAMA_URL
        normalized = normalize_base_url(base_url)
        warnings = [] if shutil.which("ollama") else [ProbeStatus.EXECUTABLE_MISSING.value]
        # Kept for compatibility with the API helper; callers normally construct
        # a temporary profile and use ``probe`` for role-specific readiness.
        try:
            version_response = self.transport.request(
                "GET", _endpoint(normalized, "/api/version"), timeout=10
            )
            tags_response = self.transport.request(
                "GET", _endpoint(normalized, "/api/tags"), timeout=10
            )
            if version_response.status_code < 200 or version_response.status_code >= 300:
                raise GatewayError(
                    "service_unreachable",
                    f"Ollama returned HTTP {version_response.status_code} for /api/version",
                    retryable=version_response.status_code >= 500,
                )
            if tags_response.status_code < 200 or tags_response.status_code >= 300:
                raise GatewayError(
                    "service_unreachable",
                    f"Ollama returned HTTP {tags_response.status_code}",
                    retryable=True,
                )
            payload = tags_response.json()
            models = payload.get("models", []) if isinstance(payload, Mapping) else []
            return {
                "status": "ready",
                "base_url": normalized,
                "version": _safe_json(version_response),
                "models": models,
                "installed_names": [
                    item.get("name") for item in models if isinstance(item, Mapping)
                ],
                "warnings": warnings,
                "error_code": None,
            }
        except GatewayError as exc:
            return {
                "status": ProbeStatus.SERVICE_UNREACHABLE.value
                if exc.code == "service_unreachable"
                else ProbeStatus.PROBE_FAILED.value,
                "base_url": normalized,
                "models": [],
                "installed_names": [],
                "warnings": warnings,
                "error_code": exc.code,
                "error_message": exc.message,
                "retryable": exc.retryable,
            }

    def probe(self, profile: ModelProfile, *, secret: str | None = None) -> ProbeResult:
        profile = coerce_model_profile(profile)
        self._ensure_role(profile, profile.role)
        checked_at = datetime.now(timezone.utc).isoformat()
        warnings: list[str] = []
        if shutil.which("ollama") is None:
            warnings.append(ProbeStatus.EXECUTABLE_MISSING.value)
        try:
            version = self._request(profile, "GET", _endpoint(profile.base_url, "/api/version"))
            tags = self._request(profile, "GET", _endpoint(profile.base_url, "/api/tags"))
        except GatewayError as exc:
            status = ProbeStatus.SERVICE_UNREACHABLE if exc.code == "service_unreachable" else ProbeStatus.PROBE_FAILED
            return ProbeResult(
                role=profile.role,
                provider=self.provider,
                status=status,
                base_url=profile.base_url,
                endpoint=None,
                model_name=profile.model_name,
                checked_at=checked_at,
                error_code=exc.code,
                error_message=exc.message,
                retryable=exc.retryable,
                requires_user=exc.requires_user,
                warnings=tuple(warnings),
            )
        try:
            tag_payload = tags.json()
            installed = tag_payload.get("models", []) if isinstance(tag_payload, Mapping) else []
            names = {
                str(item.get("name"))
                for item in installed
                if isinstance(item, Mapping) and item.get("name")
            }
            matching_tag = next(
                (
                    item
                    for item in installed
                    if isinstance(item, Mapping)
                    and _ollama_model_reference_matches(
                        str(item.get("name") or ""), profile.model_name
                    )
                ),
                None,
            )
            tag_digest = _first_str(matching_tag, "digest") if isinstance(matching_tag, Mapping) else None
            if matching_tag is None:
                return ProbeResult(
                    role=profile.role,
                    provider=self.provider,
                    status=ProbeStatus.MODEL_NOT_INSTALLED,
                    base_url=profile.base_url,
                    endpoint=None,
                    model_name=profile.model_name,
                    checked_at=checked_at,
                    error_code="model_not_installed",
                    error_message=f"model {profile.model_name!r} is not listed by /api/tags",
                    requires_user=True,
                    warnings=tuple(warnings),
                    metadata={"installed_models": sorted(names)},
                )
        except GatewayError as exc:
            return ProbeResult(
                role=profile.role,
                provider=self.provider,
                status=ProbeStatus.PROBE_FAILED,
                base_url=profile.base_url,
                endpoint=None,
                model_name=profile.model_name,
                checked_at=checked_at,
                error_code=exc.code,
                error_message=exc.message,
                retryable=exc.retryable,
                warnings=tuple(warnings),
            )
        digest: str | None = tag_digest
        show_metadata: Mapping[str, Any] = {}
        show_capabilities: Any = None
        try:
            show = self._request(
                profile,
                "POST",
                _endpoint(profile.base_url, "/api/show"),
                body={"name": profile.model_name},
            )
            show_payload = show.json()
            if isinstance(show_payload, Mapping):
                digest = _extract_digest(show_payload) or digest
                show_metadata = dict(show_payload.get("details") or {}) if isinstance(show_payload.get("details"), Mapping) else {}
                show_capabilities = show_payload.get("capabilities")
        except GatewayError:
            # `/api/show` is useful metadata, but the role-specific probe is the
            # readiness authority.  Keep going for older Ollama versions.
            pass

        try:
            if profile.role is ModelRole.CHAT:
                endpoint = "/api/chat"
                payload = {
                    "model": profile.model_name,
                    "messages": [{"role": "user", "content": 'Return exactly {"ok":true}'}],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                }
                try:
                    response = self._request(profile, "POST", _endpoint(profile.base_url, endpoint), body=payload)
                except GatewayError as exc:
                    if exc.code != "endpoint_not_found":
                        raise
                    endpoint = "/api/generate"
                    response = self._request(
                        profile,
                        "POST",
                        _endpoint(profile.base_url, endpoint),
                        body={
                            "model": profile.model_name,
                            "prompt": 'Return exactly {"ok":true}',
                            "stream": False,
                            "format": "json",
                            "options": {"temperature": 0},
                        },
                    )
                data = response.json()
                text = _extract_ollama_chat_text(data)
                if not text:
                    raise GatewayError("capability_mismatch", "chat probe returned no text")
                structured = _is_json_object(text)
                capabilities = {
                    "generation": True,
                    "structured_json": structured,
                    "tool_call": _has_capability(show_capabilities, "tools")
                    or _has_capability(show_metadata.get("capabilities"), "tools"),
                    "streaming": True,
                }
                return ProbeResult(
                    role=profile.role,
                    provider=self.provider,
                    status=ProbeStatus.READY,
                    base_url=profile.base_url,
                    endpoint=endpoint,
                    model_name=profile.model_name,
                    model_digest=digest,
                    capabilities=capabilities,
                    checked_at=checked_at,
                    warnings=tuple(warnings),
                    metadata={"version": _safe_json(version), "show": show_metadata},
                )

            endpoint = "/api/embed"
            body = {"model": profile.model_name, "input": ["resume-agent probe", "resume-agent probe"]}
            try:
                response = self._request(profile, "POST", _endpoint(profile.base_url, endpoint), body=body)
            except GatewayError as exc:
                if exc.code != "endpoint_not_found":
                    raise
                endpoint = "/api/embeddings"
                response = self._request(
                    profile,
                    "POST",
                    _endpoint(profile.base_url, endpoint),
                    body={"model": profile.model_name, "prompt": "resume-agent probe"},
                )
            data = response.json()
            vectors = _extract_ollama_vectors(data)
            if not vectors:
                raise GatewayError("capability_mismatch", "embedding probe returned no vector")
            dimension = len(vectors[0])
            if any(len(vector) != dimension for vector in vectors):
                raise GatewayError("capability_mismatch", "embedding dimension is not stable")
            stable = len(vectors) < 2 or _vectors_close(vectors[0], vectors[1])
            if not stable:
                raise GatewayError("capability_mismatch", "repeated embedding is not stable")
            return ProbeResult(
                role=profile.role,
                provider=self.provider,
                status=ProbeStatus.READY,
                base_url=profile.base_url,
                endpoint=endpoint,
                model_name=profile.model_name,
                model_digest=digest,
                dimension=dimension,
                capabilities={"embedding": True, "batch": len(vectors) > 1},
                checked_at=checked_at,
                warnings=tuple(warnings),
                metadata={"version": _safe_json(version), "show": show_metadata},
            )
        except GatewayError as exc:
            status = ProbeStatus.CAPABILITY_MISMATCH if exc.code == "capability_mismatch" else ProbeStatus.PROBE_FAILED
            return ProbeResult(
                role=profile.role,
                provider=self.provider,
                status=status,
                base_url=profile.base_url,
                endpoint=None,
                model_name=profile.model_name,
                model_digest=digest,
                checked_at=checked_at,
                error_code=exc.code,
                error_message=exc.message,
                retryable=exc.retryable,
                requires_user=exc.requires_user,
                warnings=tuple(warnings),
            )

    def chat(
        self,
        profile: ModelProfile,
        messages: Sequence[ChatMessage],
        *,
        secret: str | None = None,
        request_key: str | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        profile = coerce_model_profile(profile)
        self._ensure_role(profile, ModelRole.CHAT)
        payload: dict[str, Any] = {
            "model": profile.model_name,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": False,
            "options": {
                "temperature": profile.generation.temperature,
                "top_p": profile.generation.top_p,
            },
        }
        if response_format:
            payload["format"] = response_format
        try:
            response = self._with_retries(
                lambda: self._request(profile, "POST", _endpoint(profile.base_url, "/api/chat"), body=payload, secret=secret),
                retries=profile.generation.max_retries,
                interval=profile.generation.retry_interval_seconds,
            )
        except GatewayError as exc:
            # Ollama versions predating /api/chat expose the same operation as
            # /api/generate.  Keep the compatibility branch provider-local.
            if exc.code != "endpoint_not_found":
                raise
            prompt = "\n".join(f"{message.role}: {message.content}" for message in messages)
            response = self._with_retries(
                lambda: self._request(
                    profile,
                    "POST",
                    _endpoint(profile.base_url, "/api/generate"),
                    body={
                        "model": profile.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "format": payload.get("format"),
                        "options": payload.get("options", {}),
                    },
                    secret=secret,
                ),
                retries=profile.generation.max_retries,
                interval=profile.generation.retry_interval_seconds,
            )
        data = response.json()
        content = _extract_ollama_chat_text(data)
        if not content:
            raise GatewayError("invalid_response", "chat response has no content")
        return ChatResult(
            content=content,
            model_name=profile.model_name,
            profile_version=profile.profile_version,
            request_key=request_key,
            finish_reason=(str(data.get("done_reason")) if isinstance(data, Mapping) and data.get("done_reason") else None),
            usage={k: data[k] for k in ("prompt_eval_count", "eval_count") if isinstance(data, Mapping) and k in data},
            capabilities={"generation": True},
        )

    def embed(
        self,
        profile: ModelProfile,
        texts: Sequence[str],
        *,
        secret: str | None = None,
        request_key: str | None = None,
    ) -> EmbeddingResult:
        profile = coerce_model_profile(profile)
        self._ensure_role(profile, ModelRole.EMBEDDING)
        if not texts:
            raise ValueError("texts must not be empty")
        payload = {"model": profile.model_name, "input": list(texts)}
        try:
            response = self._with_retries(
                lambda: self._request(profile, "POST", _endpoint(profile.base_url, "/api/embed"), body=payload, secret=secret),
                retries=profile.embedding.max_retries,
                interval=profile.embedding.retry_interval_seconds,
            )
            data = response.json()
            vectors = _extract_ollama_vectors(data)
        except GatewayError as exc:
            if exc.code != "endpoint_not_found":
                raise
            vectors_list: list[tuple[float, ...]] = []
            for text in texts:
                response = self._with_retries(
                    lambda text=text: self._request(
                        profile,
                        "POST",
                        _endpoint(profile.base_url, "/api/embeddings"),
                        body={"model": profile.model_name, "prompt": text},
                        secret=secret,
                    ),
                    retries=profile.embedding.max_retries,
                    interval=profile.embedding.retry_interval_seconds,
                )
                data = response.json()
                vectors_list.append(_extract_ollama_vectors(data)[0])
            vectors = tuple(vectors_list)
        return _embedding_result(profile, vectors, request_key)


class OpenAICompatibleAdapter(_BaseAdapter):
    provider = Provider.OPENAI_COMPATIBLE

    def __init__(self, transport: HttpTransport | None = None, *, base_url: str | None = None) -> None:
        super().__init__(transport)
        self.base_url = normalize_base_url(base_url) if base_url else None

    def probe(self, profile: ModelProfile, *, secret: str | None = None) -> ProbeResult:
        profile = coerce_model_profile(profile)
        checked_at = datetime.now(timezone.utc).isoformat()
        if profile.credential_required and not secret:
            return ProbeResult(
                role=profile.role,
                provider=self.provider,
                status=ProbeStatus.CREDENTIAL_MISSING,
                base_url=profile.base_url,
                endpoint=None,
                model_name=profile.model_name,
                checked_at=checked_at,
                error_code="credential_missing",
                error_message="an API key is required for this profile",
                requires_user=True,
            )
        try:
            if profile.role is ModelRole.CHAT:
                endpoint = "/chat/completions"
                response = self._request(
                    profile,
                    "POST",
                    _openai_endpoint(profile.base_url, endpoint),
                    body={
                        "model": profile.model_name,
                        "messages": [{"role": "user", "content": 'Return exactly {"ok":true}'}],
                        "temperature": 0,
                        "max_tokens": 16,
                        "response_format": {"type": "json_object"},
                    },
                    secret=secret,
                )
                data = response.json()
                content = _extract_openai_chat_text(data)
                if not content:
                    raise GatewayError("capability_mismatch", "chat probe returned no text")
                capabilities = {
                    "generation": True,
                    "structured_json": _is_json_object(content),
                    "tool_call": bool(_extract_tool_calls(data)),
                    "streaming": True,
                }
                return ProbeResult(
                    role=profile.role,
                    provider=self.provider,
                    status=ProbeStatus.READY,
                    base_url=profile.base_url,
                    endpoint=endpoint,
                    model_name=profile.model_name,
                    capabilities=capabilities,
                    checked_at=checked_at,
                )
            endpoint = "/embeddings"
            response = self._request(
                profile,
                "POST",
                _openai_endpoint(profile.base_url, endpoint),
                body={"model": profile.model_name, "input": ["resume-agent probe", "resume-agent probe"]},
                secret=secret,
            )
            data = response.json()
            vectors = _extract_openai_vectors(data)
            if not vectors:
                raise GatewayError("capability_mismatch", "embedding probe returned no vector")
            dimension = len(vectors[0])
            if any(len(vector) != dimension for vector in vectors):
                raise GatewayError("capability_mismatch", "embedding dimension is not stable")
            if len(vectors) > 1 and not _vectors_close(vectors[0], vectors[1]):
                raise GatewayError("capability_mismatch", "repeated embedding is not stable")
            return ProbeResult(
                role=profile.role,
                provider=self.provider,
                status=ProbeStatus.READY,
                base_url=profile.base_url,
                endpoint=endpoint,
                model_name=profile.model_name,
                dimension=dimension,
                capabilities={"embedding": True, "batch": len(vectors) > 1},
                checked_at=checked_at,
            )
        except CredentialUnavailable as exc:
            return ProbeResult(
                role=profile.role,
                provider=self.provider,
                status=ProbeStatus.CREDENTIAL_MISSING,
                base_url=profile.base_url,
                endpoint=None,
                model_name=profile.model_name,
                checked_at=checked_at,
                error_code=exc.code,
                error_message=exc.message,
                requires_user=True,
            )
        except GatewayError as exc:
            status = {
                "service_unreachable": ProbeStatus.SERVICE_UNREACHABLE,
                "auth_failed": ProbeStatus.AUTH_FAILED,
                "capability_mismatch": ProbeStatus.CAPABILITY_MISMATCH,
            }.get(exc.code, ProbeStatus.PROBE_FAILED)
            return ProbeResult(
                role=profile.role,
                provider=self.provider,
                status=status,
                base_url=profile.base_url,
                endpoint=None,
                model_name=profile.model_name,
                checked_at=checked_at,
                error_code=exc.code,
                error_message=exc.message,
                retryable=exc.retryable,
                requires_user=exc.requires_user,
            )

    def chat(
        self,
        profile: ModelProfile,
        messages: Sequence[ChatMessage],
        *,
        secret: str | None = None,
        request_key: str | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        profile = coerce_model_profile(profile)
        self._ensure_role(profile, ModelRole.CHAT)
        body: dict[str, Any] = {
            "model": profile.model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": profile.generation.temperature,
            "top_p": profile.generation.top_p,
            "max_tokens": profile.generation.max_output_tokens,
        }
        if response_format:
            body["response_format"] = response_format
        response = self._with_retries(
            lambda: self._request(
                profile,
                "POST",
                _openai_endpoint(profile.base_url, "/chat/completions"),
                body=body,
                secret=secret,
            ),
            retries=profile.generation.max_retries,
            interval=profile.generation.retry_interval_seconds,
        )
        data = response.json()
        content = _extract_openai_chat_text(data)
        if not content:
            raise GatewayError("invalid_response", "chat response has no content")
        usage = data.get("usage") if isinstance(data, Mapping) and isinstance(data.get("usage"), Mapping) else {}
        choices = data.get("choices") if isinstance(data, Mapping) else []
        finish = None
        if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
            finish = str(choices[0].get("finish_reason")) if choices[0].get("finish_reason") else None
        return ChatResult(
            content=content,
            model_name=profile.model_name,
            profile_version=profile.profile_version,
            request_key=request_key,
            finish_reason=finish,
            usage=dict(usage),
            capabilities={"generation": True},
        )

    def embed(
        self,
        profile: ModelProfile,
        texts: Sequence[str],
        *,
        secret: str | None = None,
        request_key: str | None = None,
    ) -> EmbeddingResult:
        profile = coerce_model_profile(profile)
        self._ensure_role(profile, ModelRole.EMBEDDING)
        if not texts:
            raise ValueError("texts must not be empty")
        response = self._with_retries(
            lambda: self._request(
                profile,
                "POST",
                _openai_endpoint(profile.base_url, "/embeddings"),
                body={"model": profile.model_name, "input": list(texts)},
                secret=secret,
            ),
            retries=profile.embedding.max_retries,
            interval=profile.embedding.retry_interval_seconds,
        )
        return _embedding_result(profile, _extract_openai_vectors(response.json()), request_key)


class ModelGateway:
    """Selects the correct provider adapter and keeps role boundaries explicit."""

    def __init__(
        self,
        *,
        transport: HttpTransport | None = None,
        credential_store: CredentialStore | None = None,
    ) -> None:
        self.credential_store = credential_store or CredentialStore()
        self._adapters: dict[Provider, ModelAdapter] = {
            Provider.OLLAMA: OllamaAdapter(transport),
            Provider.OPENAI_COMPATIBLE: OpenAICompatibleAdapter(transport),
        }

    def adapter_for(self, profile: ModelProfile) -> ModelAdapter:
        profile = coerce_model_profile(profile)
        try:
            return self._adapters[profile.provider]
        except KeyError as exc:
            raise ValueError(f"unsupported provider: {profile.provider}") from exc

    def resolve_secret(self, profile: ModelProfile, credential_handle_id: str | None) -> str | None:
        profile = coerce_model_profile(profile)
        if not profile.credential_required:
            return None
        if not credential_handle_id:
            raise CredentialUnavailable()
        return self.credential_store.get(credential_handle_id, scope=f"model:{profile.profile_id}")

    def probe(self, profile: ModelProfile, *, credential_handle_id: str | None = None) -> ProbeResult:
        profile = coerce_model_profile(profile)
        try:
            secret = self.resolve_secret(profile, credential_handle_id) if profile.credential_required else None
        except CredentialUnavailable as exc:
            return ProbeResult(
                role=profile.role,
                provider=profile.provider,
                status=ProbeStatus.CREDENTIAL_MISSING,
                base_url=profile.base_url,
                endpoint=None,
                model_name=profile.model_name,
                checked_at=datetime.now(timezone.utc).isoformat(),
                error_code=exc.code,
                error_message=exc.message,
                retryable=False,
                requires_user=True,
            )
        return self.adapter_for(profile).probe(profile, secret=secret)

    def chat(
        self,
        profile: ModelProfile,
        messages: Sequence[ChatMessage],
        *,
        credential_handle_id: str | None = None,
        request_key: str | None = None,
        response_format: Mapping[str, Any] | None = None,
        context_snapshot: Any | None = None,
    ) -> ChatResult:
        profile = coerce_model_profile(profile)
        _validate_context_snapshot(context_snapshot)
        secret = self.resolve_secret(profile, credential_handle_id) if profile.credential_required else None
        adapter = self.adapter_for(profile)
        return adapter.chat(profile, messages, secret=secret, request_key=request_key, response_format=response_format)

    def embed(
        self,
        profile: ModelProfile,
        texts: Sequence[str],
        *,
        credential_handle_id: str | None = None,
        request_key: str | None = None,
    ) -> EmbeddingResult:
        profile = coerce_model_profile(profile)
        secret = self.resolve_secret(profile, credential_handle_id) if profile.credential_required else None
        adapter = self.adapter_for(profile)
        return adapter.embed(profile, texts, secret=secret, request_key=request_key)


def coerce_model_profile(value: Any) -> ModelProfile:
    """Public conversion helper for API/Pydantic profile objects."""
    return ModelProfile.from_object(value)


def _safe_json(response: HttpResponse) -> Any:
    try:
        return response.json()
    except GatewayError:
        return {}


_SENSITIVE_METADATA_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "github_token",
    "password",
    "secret",
    "token",
}


def _redact_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy profile metadata while removing common credential-shaped fields."""
    def walk(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): "[redacted]" if str(key).lower() in _SENSITIVE_METADATA_KEYS else walk(child)
                for key, child in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [walk(child) for child in item]
        return copy.deepcopy(item)

    return walk(value)


def _first_str(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value):
            return str(value)
    return None


def _extract_digest(mapping: Mapping[str, Any]) -> str | None:
    direct = _first_str(mapping, "digest", "model_digest")
    if direct:
        return direct
    for key in ("details", "model_info", "metadata"):
        nested = mapping.get(key)
        if isinstance(nested, Mapping):
            value = _extract_digest(nested)
            if value:
                return value
    return None


def _has_capability(value: Any, name: str) -> bool:
    """Check Ollama metadata across old/new response shapes."""

    if isinstance(value, Mapping):
        if name in value and bool(value[name]):
            return True
        return any(_has_capability(item, name) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_capability(item, name) for item in value)
    if isinstance(value, str):
        return name.lower() in value.lower()
    return False


def _ollama_model_reference_matches(installed_name: str, requested_name: str) -> bool:
    """Match Ollama's implicit ``:latest`` without weakening explicit tags."""

    installed = installed_name.strip()
    requested = requested_name.strip()
    if installed == requested:
        return True
    final_segment = requested.rsplit("/", 1)[-1]
    return ":" not in final_segment and installed == f"{requested}:latest"


def _extract_ollama_chat_text(data: Any) -> str:
    if not isinstance(data, Mapping):
        return ""
    message = data.get("message")
    if isinstance(message, Mapping) and message.get("content") is not None:
        return str(message.get("content"))
    if data.get("response") is not None:
        return str(data.get("response"))
    return ""


def _extract_openai_chat_text(data: Any) -> str:
    if not isinstance(data, Mapping):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    message = first.get("message")
    if isinstance(message, Mapping) and message.get("content") is not None:
        return str(message.get("content"))
    if first.get("text") is not None:
        return str(first.get("text"))
    return ""


def _extract_tool_calls(data: Any) -> list[Any]:
    if not isinstance(data, Mapping):
        return []
    choices = data.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        return []
    result: list[Any] = []
    for choice in choices:
        if isinstance(choice, Mapping):
            message = choice.get("message")
            if isinstance(message, Mapping) and isinstance(message.get("tool_calls"), Sequence):
                result.extend(message["tool_calls"])
    return result


def _extract_ollama_vectors(data: Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(data, Mapping):
        raise GatewayError("invalid_embedding", "embedding response is not an object")
    values = data.get("embeddings")
    if values is None and data.get("embedding") is not None:
        values = [data.get("embedding")]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise GatewayError("invalid_embedding", "embedding response has no embeddings array")
    return tuple(_safe_float_vector(item) for item in values)


def _extract_openai_vectors(data: Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(data, Mapping):
        raise GatewayError("invalid_embedding", "embedding response is not an object")
    values = data.get("data")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise GatewayError("invalid_embedding", "embedding response has no data array")
    ordered: list[tuple[int, tuple[float, ...]]] = []
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise GatewayError("invalid_embedding", "embedding item is not an object")
        try:
            order = int(item.get("index", index))
        except (TypeError, ValueError):
            order = index
        ordered.append((order, _safe_float_vector(item.get("embedding"))))
    ordered.sort(key=lambda pair: pair[0])
    return tuple(vector for _, vector in ordered)


def _vectors_close(left: Sequence[float], right: Sequence[float], tolerance: float = 1e-6) -> bool:
    return len(left) == len(right) and all(abs(a - b) <= tolerance for a, b in zip(left, right))


def _is_json_object(text: str) -> bool:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(value, Mapping)


def _embedding_result(
    profile: ModelProfile,
    vectors: Sequence[Sequence[float]],
    request_key: str | None,
) -> EmbeddingResult:
    if not vectors:
        raise GatewayError("invalid_embedding", "embedding response is empty")
    safe = tuple(_safe_float_vector(vector) for vector in vectors)
    dimension = len(safe[0])
    if any(len(vector) != dimension for vector in safe):
        raise GatewayError("invalid_embedding", "embedding dimensions differ")
    if profile.embedding.normalize:
        normalized: list[tuple[float, ...]] = []
        for vector in safe:
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0 or not math.isfinite(norm):
                raise GatewayError("invalid_embedding", "embedding vector has zero norm")
            normalized.append(tuple(value / norm for value in vector))
        safe = tuple(normalized)
    return EmbeddingResult(
        vectors=safe,
        model_name=profile.model_name,
        profile_version=profile.profile_version,
        dimension=dimension,
        request_key=request_key,
        normalized=profile.embedding.normalize,
    )


def _validate_context_snapshot(snapshot: Any | None) -> None:
    """Refuse a provider call when a caller supplied a blocked budget snapshot."""
    if snapshot is None:
        return
    blocked = bool(getattr(snapshot, "blocked", False))
    reason = getattr(snapshot, "blocked_reason", None)
    if blocked or reason:
        raise GatewayError(
            "context_budget_blocked",
            str(reason or "context budget is blocked"),
            requires_user=True,
        )
    estimated = getattr(snapshot, "estimated_input_tokens", None)
    usable = getattr(snapshot, "usable_input_tokens", None)
    if estimated is not None and usable is not None and estimated > usable:
        raise GatewayError(
            "context_budget_blocked",
            "estimated context exceeds usable input budget",
            requires_user=True,
        )


__all__ = [
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_OLLAMA_URL",
    "MODEL_GATEWAY_VERSION",
    "ChatMessage",
    "ChatModelPort",
    "ChatResult",
    "CredentialHandle",
    "CredentialStore",
    "CredentialUnavailable",
    "EmbeddingParameters",
    "EmbeddingResult",
    "EmbeddingModelPort",
    "GenerationParameters",
    "GatewayError",
    "HttpResponse",
    "HttpTransport",
    "ModelGateway",
    "ModelProfile",
    "ModelRole",
    "OllamaAdapter",
    "OpenAICompatibleAdapter",
    "ProbeResult",
    "ProbeStatus",
    "Provider",
    "UrllibTransport",
    "normalize_base_url",
    "coerce_model_profile",
]
