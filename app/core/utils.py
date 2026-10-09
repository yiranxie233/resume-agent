"""Small deterministic helpers used by domain services."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from typing import Any


def canonical_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return value.strip()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def extract_json_value(value: str, *, expected_keys: Iterable[str] = ()) -> Any:
    """Extract one JSON value from a chat response without evaluating text.

    OpenAI-compatible relays and local models sometimes wrap an otherwise
    valid JSON object in Markdown, a short explanation, or a ``<think>``
    block.  Taking everything between the first ``{`` and last ``}`` is not
    safe enough because those wrappers can contain their own braces.  Scan
    every possible JSON start with ``JSONDecoder.raw_decode`` and prefer the
    largest value containing the caller's expected top-level keys.
    """

    text = str(value or "").lstrip("\ufeff").strip()
    if not text:
        raise json.JSONDecodeError("empty model response", text, 0)
    try:
        return json.loads(text)
    except json.JSONDecodeError as original:
        decoder = json.JSONDecoder()
        expected = {str(key) for key in expected_keys if str(key)}
        candidates: list[tuple[int, int, int, Any]] = []
        for index, char in enumerate(text):
            if char not in "{[":
                continue
            try:
                payload, end = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                continue
            matched = (
                len(expected.intersection(payload))
                if expected and isinstance(payload, dict)
                else 0
            )
            candidates.append((matched, end - index, index, payload))
        if not candidates:
            raise
        # Expected contract keys are more important than wrapper size.  For
        # equally suitable values, prefer the largest and then the later one,
        # which is commonly the final answer after a reasoning preamble.
        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return candidates[0][3]


def sha256_text(value: str) -> str:
    return hashlib.sha256(canonical_text(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "bearer",
    "client_secret",
    "github_token",
    "password",
    "secret",
    "token",
    "x_api_key",
    "cookie",
    "set-cookie",
}


def _sensitive_key(value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return normalized in _SENSITIVE_KEYS


def redact_sensitive(value: Any) -> Any:
    """Return a JSON-safe copy with credential/session-shaped fields removed."""

    if isinstance(value, dict):
        return {
            str(key): (
                "[redacted]"
                if _sensitive_key(key)
                else redact_sensitive(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, str):
        # Avoid persisting common bearer/cookie forms when they appear in an
        # exception or provider diagnostic string rather than a JSON field.
        value = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[redacted]", value)
        value = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+", r"\1[redacted]", value)
        value = re.sub(r"(?i)(github[_-]?token\s*[:=]\s*)[^\s,;]+", r"\1[redacted]", value)
        value = re.sub(r"(?i)((?:set-)?cookie\s*[:=]\s*)[^\r\n]+", r"\1[redacted]", value)
        value = re.sub(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", "[redacted]", value)
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{20,}\b", "[redacted]", value)
    return value


def strip_sensitive(value: Any) -> Any:
    """Remove credential/session fields and redact recognizable string forms."""

    if isinstance(value, dict):
        return {
            str(key): strip_sensitive(item)
            for key, item in value.items()
            if not _sensitive_key(key)
        }
    if isinstance(value, list):
        return [strip_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [strip_sensitive(item) for item in value]
    return redact_sensitive(value)


def stable_id(prefix: str, *parts: Any) -> str:
    digest = sha256_json(parts)[:20]
    return f"{prefix}_{digest}"


def normalize_terms(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    aliases = {
        "py": "python",
        "py3": "python",
        "postgres": "postgresql",
        "pg": "postgresql",
        "lang-graph": "langgraph",
        "lang chain": "langchain",
    }
    for value in values:
        term = re.sub(r"\s+", " ", canonical_text(str(value)).lower())
        term = aliases.get(term, term)
        if term and term not in seen:
            result.append(term)
            seen.add(term)
    return result


def term_hits(text: str, terms: Iterable[str]) -> list[str]:
    normalized = canonical_text(text).lower()
    return [term for term in normalize_terms(terms) if term in normalized]
