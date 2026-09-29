"""Deterministic matching and evidence construction.

The first vertical slice keeps the scoring formula independent of an LLM.  A model
can later supply semantic rubric values through the same ``MatchResult`` contract.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.core.schemas import Evidence, JobInput, MatchDimension, MatchResult, ResumeDocument
from app.core.utils import normalize_terms, sha256_text, stable_id, term_hits


DEFAULT_WEIGHTS = {"skills": 0.25, "responsibilities": 0.25, "projects": 0.25, "keywords": 0.25}
DEFAULT_COMPONENT_WEIGHTS = {"deterministic": 0.5, "semantic": 0.5}


def build_job_evidence(job: JobInput) -> list[Evidence]:
    evidence: list[Evidence] = []
    groups = {
        "responsibility": job.responsibilities,
        "requirement": job.requirements,
        "skill": job.skills,
    }
    for kind, values in groups.items():
        for index, value in enumerate(values):
            text = str(value).strip()
            if not text:
                continue
            evidence.append(
                Evidence(
                    evidence_id=stable_id("job", kind, index, text),
                    source_type="job",
                    text=text,
                    location=f"{kind}[{index}]",
                    text_hash=sha256_text(text),
                )
            )
    return evidence


def _resume_text(resume: ResumeDocument | None) -> str:
    if not resume:
        return ""
    sections = "\n".join(section.content for section in resume.sections)
    return f"{resume.raw_text}\n{sections}"


def _deterministic_score(text: str, terms: Iterable[str]) -> tuple[float, list[str]]:
    normalized = normalize_terms(terms)
    if not normalized:
        return 0.0, []
    hits = term_hits(text, normalized)
    return round(len(hits) / len(normalized) * 100, 4), hits


def _semantic_proxy(deterministic: float) -> float:
    # A conservative proxy used only when no chat model is configured.  It is
    # deliberately bounded and recorded as a non-LLM semantic estimate.
    if deterministic <= 0:
        return 0.0
    if deterministic < 25:
        return 25.0
    if deterministic < 50:
        return 50.0
    if deterministic < 75:
        return 75.0
    return 100.0


def calculate_match(
    job: JobInput,
    resume: ResumeDocument | None,
    *,
    weights: dict[str, float] | None = None,
    embedding_mode: str = "tfidf_fallback",
    match_run_id: str | None = None,
    scoring_config_version: str = "scoring-v1",
    component_weights: dict[str, float] | None = None,
) -> MatchResult:
    weights = weights or DEFAULT_WEIGHTS.copy()
    if (
        set(weights) != set(DEFAULT_WEIGHTS)
        or any(value < 0 or value > 1 for value in weights.values())
        or abs(sum(weights.values()) - 1.0) > 1e-8
    ):
        raise ValueError("matching weights must contain four dimensions summing to 1")
    component_weights = component_weights or DEFAULT_COMPONENT_WEIGHTS.copy()
    if (
        set(component_weights) != set(DEFAULT_COMPONENT_WEIGHTS)
        or any(value < 0 or value > 1 for value in component_weights.values())
        or abs(sum(component_weights.values()) - 1.0) > 1e-8
    ):
        raise ValueError("component weights must contain deterministic and semantic values summing to 1")
    text = _resume_text(resume)
    skill_score, skill_hits = _deterministic_score(text, job.skills)
    responsibility_score, responsibility_hits = _deterministic_score(
        text, [*job.responsibilities, *job.requirements]
    )
    project_text = "\n".join(
        section.content for section in (resume.sections if resume else []) if section.module == "projects"
    )
    project_score, project_hits = _deterministic_score(
        project_text or text, [*job.skills, *job.responsibilities]
    )
    keywords = normalize_terms([job.title, *job.skills, *job.requirements])
    keyword_score, keyword_hits = _deterministic_score(text, keywords)
    values = {
        "skills": (skill_score, skill_hits),
        "responsibilities": (responsibility_score, responsibility_hits),
        "projects": (project_score, project_hits),
        "keywords": (keyword_score, keyword_hits),
    }
    dimensions: dict[str, MatchDimension] = {}
    for name, (deterministic, hits) in values.items():
        semantic = _semantic_proxy(deterministic)
        score = round(
            (deterministic * component_weights["deterministic"])
            + (semantic * component_weights["semantic"]),
            4,
        )
        dimensions[name] = MatchDimension(
            deterministic=deterministic,
            semantic=semantic,
            score=score,
            weight=weights[name],
            evidence_ids=[stable_id("resume", name, hit) for hit in hits],
        )
    total = sum(dim.score * dim.weight for dim in dimensions.values())
    return MatchResult(
        match_run_id=match_run_id or stable_id("match", job.title, text),
        total_score=round(total, 1),
        dimensions=dimensions,
        threshold_triggered=total < 80,
        scoring_config_version=scoring_config_version,
        embedding_mode=embedding_mode,
    )


def rank_candidate(candidate: dict[str, Any], job: JobInput) -> float:
    text = " ".join(
        [
            str(candidate.get("title", "")),
            str(candidate.get("introduction", "")),
            " ".join(candidate.get("tech_stack", [])),
            " ".join(candidate.get("solutions", [])),
        ]
    )
    score, _ = _deterministic_score(text, [*job.skills, *job.responsibilities])
    return round(score, 1)
