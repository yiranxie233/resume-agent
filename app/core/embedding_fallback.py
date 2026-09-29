"""Deterministic local TF-IDF fallback for semantic comparison.

The fallback is intentionally modest: it is not presented as an embedding model,
and callers must persist ``mode=tfidf_fallback`` with the task/scoring config.  It
uses Chinese character 2--4 grams plus normalized technical terms, then L2
normalizes vectors and computes cosine similarity.  No external tokenizer or
scientific package is required.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


TFIDF_ALGORITHM_VERSION = "tfidf-char2-4-tech-v1"
TFIDF_DEFAULT_THRESHOLD = 0.75
EMBEDDING_DEFAULT_THRESHOLD = 0.85

_ASCII_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#._/-]*")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFC", text)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip().lower()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _cjk_runs(text: str) -> list[str]:
    runs: list[str] = []
    current: list[str] = []
    for char in text:
        if _CJK_RE.fullmatch(char):
            current.append(char)
        elif current:
            runs.append("".join(current))
            current = []
    if current:
        runs.append("".join(current))
    return runs


def tokenize(text: str) -> tuple[str, ...]:
    """Return stable character n-grams and technical terms.

    Single CJK characters are kept for short phrases; longer runs additionally
    receive 2--4 grams.  ASCII terms preserve symbols common in technology names
    (C++, C#, .NET, fastapi, qwen3, etc.).
    """
    text = normalize_text(text)
    features: list[str] = []
    for run in _cjk_runs(text):
        features.extend(f"cjk1:{char}" for char in run)
        for n in (2, 3, 4):
            if len(run) >= n:
                features.extend(f"cjk{n}:{run[index:index+n]}" for index in range(len(run) - n + 1))
    features.extend(f"term:{match.group(0)}" for match in _ASCII_TERM_RE.finditer(text))
    # Keep a small amount of punctuation-independent token coverage for mixed
    # identifiers (e.g. ``Qwen3-Embedding``) while remaining deterministic.
    for raw in re.split(r"[^\w+#.-]+", text, flags=re.UNICODE):
        if raw and not _CJK_RE.search(raw) and len(raw) >= 2:
            features.append(f"word:{raw}")
    return tuple(features)


@dataclass(frozen=True)
class SparseVector:
    values: Mapping[str, float]

    @property
    def dimension(self) -> int:
        return len(self.values)

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)


def cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


class TfidfFallback:
    """A small deterministic vectorizer suitable for local candidate deduplication."""

    algorithm_version = TFIDF_ALGORITHM_VERSION
    mode = "tfidf_fallback"

    def __init__(self, *, max_features: int | None = None) -> None:
        self.max_features = max_features
        self._idf: dict[str, float] = {}
        self._fitted = False

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return tuple(self._idf.keys())

    @property
    def dimension(self) -> int:
        return len(self._idf)

    def fit(self, documents: Sequence[str]) -> "TfidfFallback":
        if not documents:
            raise ValueError("at least one document is required")
        document_tokens = [set(tokenize(document)) for document in documents]
        document_frequency: Counter[str] = Counter()
        for tokens in document_tokens:
            document_frequency.update(tokens)
        features = sorted(document_frequency)
        if self.max_features is not None:
            if self.max_features <= 0:
                raise ValueError("max_features must be positive")
            # Keep the most informative features by DF, with lexical tie-break.
            features = sorted(features, key=lambda key: (-document_frequency[key], key))[: self.max_features]
            features.sort()
        count = len(document_tokens)
        self._idf = {
            feature: math.log((1.0 + count) / (1.0 + document_frequency[feature])) + 1.0
            for feature in features
        }
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit() must be called before transform()")

    def transform(self, documents: Sequence[str]) -> tuple[SparseVector, ...]:
        self._check_fitted()
        result: list[SparseVector] = []
        for document in documents:
            counts = Counter(tokenize(document))
            weighted = {
                feature: float(counts[feature]) * self._idf[feature]
                for feature in self._idf
                if counts.get(feature, 0)
            }
            norm = math.sqrt(sum(value * value for value in weighted.values()))
            if norm:
                weighted = {feature: value / norm for feature, value in weighted.items()}
            result.append(SparseVector(weighted))
        return tuple(result)

    def fit_transform(self, documents: Sequence[str]) -> tuple[SparseVector, ...]:
        return self.fit(documents).transform(documents)

    def similarity(self, left: str | SparseVector, right: str | SparseVector) -> float:
        self._check_fitted()
        left_vector = left if isinstance(left, SparseVector) else self.transform([left])[0]
        right_vector = right if isinstance(right, SparseVector) else self.transform([right])[0]
        return cosine_similarity(left_vector.values, right_vector.values)

    def artifact(self, text: str) -> "EmbeddingArtifact":
        self._check_fitted()
        vector = self.transform([text])[0]
        return EmbeddingArtifact(
            content_hash=content_hash(text),
            mode=self.mode,
            algorithm_version=self.algorithm_version,
            dimension=self.dimension,
            vector=vector.values,
        )


@dataclass(frozen=True)
class EmbeddingArtifact:
    content_hash: str
    mode: str
    algorithm_version: str
    dimension: int
    vector: Mapping[str, float]

    def to_dict(self, *, include_vector: bool = True) -> dict[str, Any]:
        value = {
            "content_hash": self.content_hash,
            "mode": self.mode,
            "algorithm_version": self.algorithm_version,
            "dimension": self.dimension,
        }
        if include_vector:
            value["vector"] = dict(self.vector)
        return value


@dataclass(frozen=True)
class CandidateText:
    candidate_id: str
    title: str
    technology_stack: str = ""
    body: str = ""
    match_score: float = 0.0

    @property
    def normalized_exact_key(self) -> str:
        return "|".join(
            [
                re.sub(r"\s+", "", normalize_text(self.title)),
                re.sub(r"\s+", "", normalize_text(self.technology_stack)),
            ]
        )

    @property
    def comparison_text(self) -> str:
        return "\n".join([self.title, self.technology_stack, self.body])


@dataclass(frozen=True)
class DuplicateDecision:
    dropped_candidate_id: str
    kept_candidate_id: str
    reason: str
    similarity: float
    threshold: float
    mode: str
    algorithm_version: str


@dataclass(frozen=True)
class DeduplicationResult:
    kept: tuple[CandidateText, ...]
    dropped: tuple[CandidateText, ...]
    decisions: tuple[DuplicateDecision, ...]
    mode: str
    threshold: float
    algorithm_version: str


def deduplicate_candidates(
    candidates: Sequence[CandidateText],
    *,
    mode: str = "tfidf_fallback",
    threshold: float = TFIDF_DEFAULT_THRESHOLD,
) -> DeduplicationResult:
    """Deduplicate candidates using exact title/stack keys then TF-IDF cosine.

    ``mode=embedding`` is accepted for a caller that has already chosen a real
    embedding path, but this function only computes fallback vectors; callers must
    use their embedding adapter in that mode.  Keeping the explicit mode here
    prevents accidentally applying the 0.75 fallback threshold to real vectors.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if mode not in {"tfidf_fallback", "embedding"}:
        raise ValueError("mode must be tfidf_fallback or embedding")
    if mode == "embedding":
        raise ValueError("deduplicate_candidates requires real vectors for embedding mode")
    if not candidates:
        return DeduplicationResult((), (), (), mode, threshold, TFIDF_ALGORITHM_VERSION)

    exact_kept: list[CandidateText] = []
    dropped: list[CandidateText] = []
    decisions: list[DuplicateDecision] = []
    exact_index: dict[str, CandidateText] = {}
    for candidate in candidates:
        key = candidate.normalized_exact_key
        previous = exact_index.get(key)
        if previous is None:
            exact_index[key] = candidate
            exact_kept.append(candidate)
            continue
        keep, drop = _prefer_candidate(previous, candidate)
        exact_index[key] = keep
        if keep.candidate_id != previous.candidate_id:
            exact_kept[exact_kept.index(previous)] = keep
        dropped.append(drop)
        decisions.append(
            DuplicateDecision(
                dropped_candidate_id=drop.candidate_id,
                kept_candidate_id=keep.candidate_id,
                reason="exact_title_and_stack",
                similarity=1.0,
                threshold=threshold,
                mode=mode,
                algorithm_version=TFIDF_ALGORITHM_VERSION,
            )
        )

    # Fit on the surviving candidate texts.  A one-document corpus still yields
    # deterministic vectors (IDF=1), and empty text simply has zero similarity.
    vectorizer = TfidfFallback().fit([candidate.comparison_text for candidate in exact_kept])
    kept: list[CandidateText] = []
    for candidate in exact_kept:
        candidate_vector = vectorizer.transform([candidate.comparison_text])[0]
        duplicate_of: CandidateText | None = None
        duplicate_similarity = 0.0
        for previous in kept:
            similarity = vectorizer.similarity(candidate_vector, vectorizer.transform([previous.comparison_text])[0])
            if similarity >= threshold and similarity > duplicate_similarity:
                duplicate_of = previous
                duplicate_similarity = similarity
        if duplicate_of is None:
            kept.append(candidate)
            continue
        keep, drop = _prefer_candidate(duplicate_of, candidate)
        if keep.candidate_id != duplicate_of.candidate_id:
            kept[kept.index(duplicate_of)] = keep
        dropped.append(drop)
        decisions.append(
            DuplicateDecision(
                dropped_candidate_id=drop.candidate_id,
                kept_candidate_id=keep.candidate_id,
                reason="tfidf_cosine_threshold",
                similarity=round(duplicate_similarity, 6),
                threshold=threshold,
                mode=mode,
                algorithm_version=TFIDF_ALGORITHM_VERSION,
            )
        )
    return DeduplicationResult(
        kept=tuple(kept),
        dropped=tuple(dropped),
        decisions=tuple(decisions),
        mode=mode,
        threshold=threshold,
        algorithm_version=TFIDF_ALGORITHM_VERSION,
    )


def deduplicate_with_vectors(
    candidates: Sequence[CandidateText],
    vectors: Mapping[str, Mapping[str, float] | Sequence[float]],
    *,
    threshold: float = EMBEDDING_DEFAULT_THRESHOLD,
    mode: str = "embedding",
    algorithm_version: str = "embedding-cosine-v1",
) -> DeduplicationResult:
    """Deduplicate using caller-supplied real embedding vectors.

    Vectors are keyed by ``candidate_id``.  Dense sequences are converted to a
    sparse index mapping solely for cosine calculation; no model-specific data is
    persisted here.  The separate function makes it impossible to accidentally
    apply the TF-IDF fallback threshold to a real embedding task.
    """
    if mode != "embedding":
        raise ValueError("deduplicate_with_vectors is for embedding mode")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    def as_mapping(vector: Mapping[str, float] | Sequence[float]) -> Mapping[str, float]:
        if isinstance(vector, Mapping):
            values = {str(key): float(value) for key, value in vector.items()}
        else:
            values = {str(index): float(value) for index, value in enumerate(vector)}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("embedding vector contains non-finite values")
        return values

    exact_kept: list[CandidateText] = []
    dropped: list[CandidateText] = []
    decisions: list[DuplicateDecision] = []
    exact_index: dict[str, CandidateText] = {}
    for candidate in candidates:
        if candidate.candidate_id not in vectors:
            raise ValueError(f"missing embedding vector for {candidate.candidate_id}")
        key = candidate.normalized_exact_key
        previous = exact_index.get(key)
        if previous is None:
            exact_index[key] = candidate
            exact_kept.append(candidate)
            continue
        keep, drop = _prefer_candidate(previous, candidate)
        exact_index[key] = keep
        if keep.candidate_id != previous.candidate_id:
            exact_kept[exact_kept.index(previous)] = keep
        dropped.append(drop)
        decisions.append(
            DuplicateDecision(
                dropped_candidate_id=drop.candidate_id,
                kept_candidate_id=keep.candidate_id,
                reason="exact_title_and_stack",
                similarity=1.0,
                threshold=threshold,
                mode=mode,
                algorithm_version=algorithm_version,
            )
        )

    converted = {candidate_id: as_mapping(vector) for candidate_id, vector in vectors.items()}
    kept: list[CandidateText] = []
    for candidate in exact_kept:
        duplicate_of: CandidateText | None = None
        duplicate_similarity = 0.0
        for previous in kept:
            similarity = cosine_similarity(
                converted[candidate.candidate_id], converted[previous.candidate_id]
            )
            if similarity >= threshold and similarity > duplicate_similarity:
                duplicate_of = previous
                duplicate_similarity = similarity
        if duplicate_of is None:
            kept.append(candidate)
            continue
        keep, drop = _prefer_candidate(duplicate_of, candidate)
        if keep.candidate_id != duplicate_of.candidate_id:
            kept[kept.index(duplicate_of)] = keep
        dropped.append(drop)
        decisions.append(
            DuplicateDecision(
                dropped_candidate_id=drop.candidate_id,
                kept_candidate_id=keep.candidate_id,
                reason="embedding_cosine_threshold",
                similarity=round(duplicate_similarity, 6),
                threshold=threshold,
                mode=mode,
                algorithm_version=algorithm_version,
            )
        )
    return DeduplicationResult(
        kept=tuple(kept),
        dropped=tuple(dropped),
        decisions=tuple(decisions),
        mode=mode,
        threshold=threshold,
        algorithm_version=algorithm_version,
    )


def _prefer_candidate(left: CandidateText, right: CandidateText) -> tuple[CandidateText, CandidateText]:
    # Stable tie-break by candidate ID after the backend match score.
    if right.match_score > left.match_score:
        return right, left
    if right.match_score < left.match_score:
        return left, right
    if right.candidate_id < left.candidate_id:
        return right, left
    return left, right


__all__ = [
    "CandidateText",
    "DeduplicationResult",
    "DuplicateDecision",
    "EMBEDDING_DEFAULT_THRESHOLD",
    "EmbeddingArtifact",
    "TFIDF_ALGORITHM_VERSION",
    "TFIDF_DEFAULT_THRESHOLD",
    "SparseVector",
    "TfidfFallback",
    "content_hash",
    "cosine_similarity",
    "deduplicate_candidates",
    "deduplicate_with_vectors",
    "normalize_text",
    "tokenize",
]
