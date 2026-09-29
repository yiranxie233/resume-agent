"""Deterministic context construction and token-budget gating.

The graph must know whether a request fits *before* it calls a provider.  This
module normalizes the exact payload that will be rendered, applies a versioned
budget policy, and returns an auditable snapshot.  It has no dependency on a
specific tokenizer; an optional exact tokenizer can be injected, otherwise the
versioned conservative byte upper-bound estimator is used.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from enum import IntEnum
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


CONTEXT_BUDGET_POLICY_VERSION = "context-budget-v1"
CONSERVATIVE_ESTIMATOR_VERSION = "conservative-char-v1"
CONTEXT_SOURCE_PRIORITY = {
    "user_confirmed": 600,
    "confirmed_fact": 600,
    "evidence": 500,
    "jd_duty": 400,
    "jd_requirement": 400,
    "feedback": 300,
    "skill": 200,
    "history": 100,
    "untrusted_context": 50,
}


class ContextBudgetError(RuntimeError):
    """Base class for context construction failures."""

    def __init__(self, code: str, message: str, *, snapshot: "ContextSnapshot | None" = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.snapshot = snapshot


class ContextBudgetBlocked(ContextBudgetError):
    def __init__(self, message: str, *, snapshot: "ContextSnapshot") -> None:
        super().__init__("context_budget_blocked", message, snapshot=snapshot)


class TokenEstimatorError(ContextBudgetError):
    def __init__(self, message: str) -> None:
        super().__init__("token_estimation_failed", message)


def normalize_text(value: str) -> str:
    """Apply the canonical NFC/LF normalization required by the design."""
    if not isinstance(value, str):
        raise TypeError("context text must be a string")
    value = unicodedata.normalize("NFC", value)
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "__dict__") and not isinstance(value, type):
        # Keep this intentionally narrow; arbitrary objects must not leak into
        # prompts through their repr.
        return _jsonable(vars(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return normalize_text(value) if isinstance(value, str) else value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    """Stable JSON used both for estimation and context/request hashes."""
    return normalize_text(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class Tokenizer(Protocol):
    tokenizer_id: str
    tokenizer_version: str

    def count(self, rendered: str) -> int:
        ...


@dataclass(frozen=True)
class ConservativeTokenEstimator:
    """Conservative UTF-8 byte upper-bound estimator.

    A byte can represent at most one provider token in the conservative bound;
    this intentionally overestimates Chinese and mixed-language content.  The
    fixed wrapper allowance accounts for provider message/schema framing.
    """

    version: str = CONSERVATIVE_ESTIMATOR_VERSION
    wrapper_overhead_tokens: int = 32

    def __post_init__(self) -> None:
        if self.wrapper_overhead_tokens < 0:
            raise ValueError("wrapper_overhead_tokens must be non-negative")

    def count(self, rendered: str) -> int:
        rendered = normalize_text(rendered)
        if not isinstance(rendered, str):  # pragma: no cover - defensive
            raise TokenEstimatorError("rendered context is not text")
        try:
            byte_count = len(rendered.encode("utf-8"))
        except UnicodeError as exc:
            raise TokenEstimatorError("context cannot be encoded as UTF-8") from exc
        return byte_count + self.wrapper_overhead_tokens


@dataclass(frozen=True)
class BudgetPolicy:
    context_window_tokens: int | None
    max_output_tokens: int
    safety_margin_tokens: int | None = None
    version: str = CONTEXT_BUDGET_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.context_window_tokens is not None and self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive when set")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.safety_margin_tokens is not None and self.safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens must be non-negative")

    @property
    def safety_margin(self) -> int | None:
        if self.context_window_tokens is None:
            return None
        if self.safety_margin_tokens is not None:
            return self.safety_margin_tokens
        return max(512, math.ceil(self.context_window_tokens * 0.10))

    @property
    def usable_input(self) -> int | None:
        if self.context_window_tokens is None or self.safety_margin is None:
            return None
        return self.context_window_tokens - self.max_output_tokens - self.safety_margin

    @classmethod
    def from_profile(
        cls,
        profile: Any,
        *,
        max_output_tokens: int | None = None,
        safety_margin_tokens: int | None = None,
        version: str = CONTEXT_BUDGET_POLICY_VERSION,
    ) -> "BudgetPolicy":
        """Build a chat budget from either gateway or API profile objects."""
        def read(name: str, default: Any = None) -> Any:
            if isinstance(profile, Mapping):
                return profile.get(name, default)
            return getattr(profile, name, default)

        generation = read("generation", None)
        configured_output = getattr(generation, "max_output_tokens", None)
        if isinstance(generation, Mapping):
            configured_output = generation.get("max_output_tokens")
        if configured_output is None:
            params = read("generation_params", {}) or {}
            configured_output = params.get("max_output_tokens", 1200)
        return cls(
            context_window_tokens=read("context_window_tokens", None),
            max_output_tokens=int(max_output_tokens if max_output_tokens is not None else configured_output),
            safety_margin_tokens=safety_margin_tokens,
            version=version,
        )


class ContextPriority(IntEnum):
    USER_CONFIRMED = 600
    EVIDENCE = 500
    JD_DUTY = 400
    FEEDBACK = 300
    SKILL = 200
    HISTORY = 100
    UNTRUSTED = 50


@dataclass(frozen=True)
class ContextItem:
    object_id: str
    content: str
    source_type: str
    required: bool = False
    priority: int | None = None
    trust_level: str = "trusted"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ValueError("context item object_id is required")
        object.__setattr__(self, "content", normalize_text(self.content))
        if self.priority is None:
            object.__setattr__(
                self,
                "priority",
                CONTEXT_SOURCE_PRIORITY.get(self.source_type, ContextPriority.HISTORY),
            )


@dataclass(frozen=True)
class ContextSnapshot:
    tokenizer_id: str | None
    tokenizer_version: str | None
    estimator_mode: str
    estimator_version: str
    budget_policy_version: str
    context_window_tokens: int | None
    max_output_tokens: int
    safety_margin_tokens: int | None
    usable_input_tokens: int | None
    estimated_input_tokens: int | None
    included_object_ids: tuple[str, ...]
    omitted_object_ids: tuple[str, ...]
    trim_order: tuple[str, ...]
    blocked_reason: str | None
    context_manifest_hash: str
    rendered_context: Mapping[str, Any]
    rendered_text: str

    @property
    def blocked(self) -> bool:
        return self.blocked_reason is not None

    def to_dict(self, *, include_rendered: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_version": self.tokenizer_version,
            "estimator_mode": self.estimator_mode,
            "estimator_version": self.estimator_version,
            "budget_policy_version": self.budget_policy_version,
            "context_window_tokens": self.context_window_tokens,
            "max_output_tokens": self.max_output_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "usable_input_tokens": self.usable_input_tokens,
            "estimated_input_tokens": self.estimated_input_tokens,
            "included_object_ids": list(self.included_object_ids),
            "omitted_object_ids": list(self.omitted_object_ids),
            "trim_order": list(self.trim_order),
            "blocked_reason": self.blocked_reason,
            "context_manifest_hash": self.context_manifest_hash,
        }
        if include_rendered:
            result["rendered_context"] = self.rendered_context
            result["rendered_text"] = self.rendered_text
        return result


def _message_payload(message: Any) -> dict[str, str]:
    if isinstance(message, Mapping):
        role = str(message.get("role", "user"))
        content = normalize_text(str(message.get("content", "")))
    elif hasattr(message, "role") and hasattr(message, "content"):
        role = str(message.role)
        content = normalize_text(str(message.content))
    elif isinstance(message, Sequence) and len(message) == 2:
        role = str(message[0])
        content = normalize_text(str(message[1]))
    else:
        raise TypeError("message must be a mapping, ChatMessage-like object, or pair")
    return {"role": role, "content": content}


class ContextBuilder:
    """Build an auditable, atomically trimmed context payload."""

    def __init__(
        self,
        policy: BudgetPolicy,
        *,
        tokenizer: Tokenizer | Callable[[str], int] | None = None,
        fallback_estimator: ConservativeTokenEstimator | None = None,
    ) -> None:
        self.policy = policy
        self.tokenizer = tokenizer
        self.fallback_estimator = fallback_estimator or ConservativeTokenEstimator()

    @classmethod
    def from_profile(
        cls,
        profile: Any,
        *,
        tokenizer: Tokenizer | Callable[[str], int] | None = None,
        max_output_tokens: int | None = None,
        safety_margin_tokens: int | None = None,
        budget_policy_version: str = CONTEXT_BUDGET_POLICY_VERSION,
    ) -> "ContextBuilder":
        return cls(
            BudgetPolicy.from_profile(
                profile,
                max_output_tokens=max_output_tokens,
                safety_margin_tokens=safety_margin_tokens,
                version=budget_policy_version,
            ),
            tokenizer=tokenizer,
        )

    def _count(self, rendered_text: str) -> tuple[int, str, str | None, str | None]:
        if self.tokenizer is not None:
            try:
                counter = self.tokenizer.count if hasattr(self.tokenizer, "count") else self.tokenizer
                value = int(counter(rendered_text))
                if value < 0:
                    raise ValueError("tokenizer returned a negative count")
                return (
                    value,
                    "exact",
                    getattr(self.tokenizer, "tokenizer_id", None),
                    getattr(self.tokenizer, "tokenizer_version", None),
                )
            except Exception:
                # A tokenizer that cannot be loaded is explicitly downgraded to
                # the versioned conservative estimator; no silent truncation.
                pass
        try:
            return (
                self.fallback_estimator.count(rendered_text),
                "conservative_fallback",
                None,
                None,
            )
        except Exception as exc:
            raise TokenEstimatorError(str(exc)) from exc

    def build(
        self,
        *,
        system: str = "",
        developer: str = "",
        messages: Sequence[Any] = (),
        tool_schemas: Sequence[Any] = (),
        tool_results: Sequence[Any] = (),
        items: Sequence[ContextItem] = (),
        required_object_ids: Iterable[str] = (),
    ) -> ContextSnapshot:
        normalized_items = [self._coerce_item(item) for item in items]
        required_ids = set(required_object_ids) | {item.object_id for item in normalized_items if item.required}
        # Duplicate object IDs are ambiguous for field-level evidence references.
        seen: set[str] = set()
        for item in normalized_items:
            if item.object_id in seen:
                raise ContextBudgetError("duplicate_context_id", f"duplicate context item: {item.object_id}")
            seen.add(item.object_id)
        missing_required = sorted(required_ids - seen)
        # Keep caller order as the deterministic tie-break without consulting the
        # list during ``sort`` (CPython temporarily empties the list while sorting).
        normalized_items = [
            item
            for _, item in sorted(
                enumerate(normalized_items),
                key=lambda pair: (-(pair[1].priority or 0), pair[0]),
            )
        ]

        fixed_payload = {
            "system": normalize_text(system),
            "developer": normalize_text(developer),
            "messages": [_message_payload(message) for message in messages],
            "tool_schemas": [_jsonable(schema) for schema in tool_schemas],
            "tool_results": [_jsonable(result) for result in tool_results],
            "context_items": [],
        }
        included: list[ContextItem] = []
        omitted: list[str] = []
        trim_order: list[str] = []
        blocked_reason: str | None = None

        if missing_required:
            blocked_reason = f"required_item_missing:{missing_required[0]}"
        elif self.policy.context_window_tokens is None:
            blocked_reason = "context_window_unknown"
        elif self.policy.usable_input is None or self.policy.usable_input <= 0:
            blocked_reason = "usable_input_non_positive"

        # Count the fixed payload first.  It is never silently truncated.
        if blocked_reason is None:
            try:
                fixed_text = canonical_json(fixed_payload)
                fixed_count, mode, tokenizer_id, tokenizer_version = self._count(fixed_text)
            except (TokenEstimatorError, TypeError, ValueError) as exc:
                blocked_reason = getattr(exc, "code", "invalid_context")
                fixed_count, mode, tokenizer_id, tokenizer_version = None, "conservative_fallback", None, None
        else:
            fixed_count, mode, tokenizer_id, tokenizer_version = None, "conservative_fallback", None, None

        if blocked_reason is not None and not omitted:
            omitted.extend(item.object_id for item in normalized_items)

        if blocked_reason is None and fixed_count is not None and fixed_count > self.policy.usable_input:
            blocked_reason = "fixed_context_exceeds_budget"

        # Add complete atomic items in fixed priority order.  A lower-priority
        # item is omitted when it does not fit; a required item blocks the node.
        if blocked_reason is None:
            for index, item in enumerate(normalized_items):
                trial = dict(fixed_payload)
                trial["context_items"] = [
                    {
                        "id": selected.object_id,
                        "source_type": selected.source_type,
                        "trust_level": selected.trust_level,
                        "content": selected.content,
                        "metadata": _jsonable(selected.metadata),
                    }
                    for selected in [*included, item]
                ]
                try:
                    trial_count, trial_mode, trial_id, trial_version = self._count(canonical_json(trial))
                except (TokenEstimatorError, TypeError, ValueError) as exc:
                    blocked_reason = getattr(exc, "code", "invalid_context")
                    mode, tokenizer_id, tokenizer_version = "conservative_fallback", None, None
                    break
                if trial_count <= self.policy.usable_input:
                    included.append(item)
                    mode, tokenizer_id, tokenizer_version = trial_mode, trial_id, trial_version
                else:
                    omitted.append(item.object_id)
                    trim_order.append(item.object_id)
                    if item.object_id in required_ids:
                        blocked_reason = f"required_item_exceeds_budget:{item.object_id}"
                        break

            if blocked_reason is not None:
                selected_ids = {item.object_id for item in included}
                omitted_set = set(omitted)
                for remaining in normalized_items:
                    if remaining.object_id not in selected_ids and remaining.object_id not in omitted_set:
                        omitted.append(remaining.object_id)

        final_payload = dict(fixed_payload)
        final_payload["context_items"] = [
            {
                "id": item.object_id,
                "source_type": item.source_type,
                "trust_level": item.trust_level,
                "content": item.content,
                "metadata": _jsonable(item.metadata),
            }
            for item in included
        ]
        try:
            rendered_text = canonical_json(final_payload)
        except (TypeError, ValueError) as exc:
            blocked_reason = blocked_reason or "invalid_context"
            # Keep a deterministic, non-provider payload for the audit snapshot;
            # the blocked reason prevents this placeholder from being sent.
            rendered_text = "{}"
        try:
            estimated, final_mode, final_tokenizer_id, final_tokenizer_version = self._count(rendered_text)
            # If the initial fixed pass had a tokenizer failure, final count may
            # still succeed with fallback; preserve the accurate final mode.
            mode = final_mode
            tokenizer_id = final_tokenizer_id
            tokenizer_version = final_tokenizer_version
        except TokenEstimatorError as exc:
            estimated = None
            blocked_reason = blocked_reason or exc.code

        if blocked_reason is None and estimated is not None and estimated > (self.policy.usable_input or 0):
            blocked_reason = "context_exceeds_budget"

        audit_payload: Mapping[str, Any] = final_payload
        manifest_material = {
            "rendered": audit_payload,
            "included": [item.object_id for item in included],
            "omitted": omitted,
            "trim_order": trim_order,
            "estimator_version": self.fallback_estimator.version,
            "estimator_mode": mode,
            "tokenizer_id": tokenizer_id,
            "tokenizer_version": tokenizer_version,
            "budget_policy_version": self.policy.version,
            "context_window_tokens": self.policy.context_window_tokens,
            "max_output_tokens": self.policy.max_output_tokens,
            "safety_margin_tokens": self.policy.safety_margin,
        }
        try:
            manifest_hash = sha256_text(canonical_json(manifest_material))
        except (TypeError, ValueError):
            # Invalid provider payloads must remain persistable as an audit
            # record without retaining a non-JSON value (NaN, unserializable
            # object, etc.).
            audit_payload = {"blocked_reason": blocked_reason or "invalid_context"}
            manifest_material["rendered"] = audit_payload
            manifest_hash = sha256_text(canonical_json(manifest_material))
        snapshot = ContextSnapshot(
            tokenizer_id=tokenizer_id,
            tokenizer_version=tokenizer_version,
            estimator_mode=mode,
            estimator_version=self.fallback_estimator.version,
            budget_policy_version=self.policy.version,
            context_window_tokens=self.policy.context_window_tokens,
            max_output_tokens=self.policy.max_output_tokens,
            safety_margin_tokens=self.policy.safety_margin,
            usable_input_tokens=self.policy.usable_input,
            estimated_input_tokens=estimated,
            included_object_ids=tuple(item.object_id for item in included),
            omitted_object_ids=tuple(omitted),
            trim_order=tuple(trim_order),
            blocked_reason=blocked_reason,
            context_manifest_hash=manifest_hash,
            rendered_context=audit_payload,
            rendered_text=rendered_text,
        )
        return snapshot

    @staticmethod
    def _coerce_item(item: Any) -> ContextItem:
        if isinstance(item, ContextItem):
            return item
        if hasattr(item, "model_dump"):
            return ContextItem(**item.model_dump())
        if isinstance(item, Mapping):
            return ContextItem(**item)
        raise TypeError("context items must be ContextItem, mapping, or Pydantic-like objects")

    def build_or_raise(self, **kwargs: Any) -> ContextSnapshot:
        snapshot = self.build(**kwargs)
        if snapshot.blocked:
            raise ContextBudgetBlocked(snapshot.blocked_reason or "context budget blocked", snapshot=snapshot)
        return snapshot


def make_request_key(
    *,
    thread_id: str,
    node_name: str,
    model_profile_version: str,
    snapshot: ContextSnapshot,
    generation_parameters: Mapping[str, Any] | None = None,
) -> str:
    """Derive the stable deduplication key specified by the design."""
    material = {
        "thread_id": thread_id,
        "node_name": node_name,
        "model_profile_version": model_profile_version,
        "context_manifest_hash": snapshot.context_manifest_hash,
        "rendered_context_hash": sha256_text(snapshot.rendered_text),
        "estimator_version": snapshot.estimator_version,
        "estimator_mode": snapshot.estimator_mode,
        "budget_policy_version": snapshot.budget_policy_version,
        "generation_parameters": _jsonable(generation_parameters or {}),
    }
    return sha256_text(canonical_json(material))


__all__ = [
    "BudgetPolicy",
    "CONSERVATIVE_ESTIMATOR_VERSION",
    "CONTEXT_BUDGET_POLICY_VERSION",
    "ContextBudgetBlocked",
    "ContextBudgetError",
    "ContextBuilder",
    "ContextItem",
    "ContextPriority",
    "ContextSnapshot",
    "ConservativeTokenEstimator",
    "TokenEstimatorError",
    "canonical_json",
    "make_request_key",
    "normalize_text",
    "sha256_text",
]
