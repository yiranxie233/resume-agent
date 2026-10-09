"""Deterministic feedback classification used before a new generation branch.

The classifier intentionally does not invent facts.  It only labels common
Chinese preference/constraint phrases and asks for clarification when a note is
ambiguous or contains conflicting signals.  A later LLM classifier can implement
the same result contract without changing the workflow boundary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


FEEDBACK_CATEGORIES = ("fact", "style", "structure", "opinion")


@dataclass(frozen=True)
class FeedbackClassification:
    category: str
    confidence: float
    spans: tuple[tuple[int, int], ...] = ()
    fact_candidates: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    classifier_type: str = "rules-v1"

    @property
    def requires_clarification(self) -> bool:
        # Free-form preference feedback is safe to pass through as an
        # untrusted instruction even when the rules cannot confidently label
        # its category.  Pause only for a real contradiction (or an effectively
        # empty/unusable note), not merely because the wording is unfamiliar.
        return self.confidence < 0.5 or bool(self.conflicts)

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "confidence": self.confidence,
            "spans": [list(item) for item in self.spans],
            "fact_candidates": list(self.fact_candidates),
            "conflicts": list(self.conflicts),
            "classifier_type": self.classifier_type,
            "requires_clarification": self.requires_clarification,
        }


_FACT_PATTERNS = (
    r"我(?:实际|确实|曾经|负责过|做过)",
    r"使用过",
    r"掌握",
    r"不要虚构",
    r"不能写",
    r"没有(?:做过|使用过)",
    r"没(?:有)?(?:做过|使用过|掌握)",
    r"(?:未曾|不曾)(?:做过|使用过)",
    r"数据不准确",
)
_STYLE_PATTERNS = (
    r"突出",
    r"简洁",
    r"详细",
    r"量化",
    r"更有说服力",
    r"强调",
    r"语气",
    r"风格",
)
_STRUCTURE_PATTERNS = (
    r"放在开头",
    r"调整顺序",
    r"替换",
    r"删除",
    r"增加一条",
    r"(?:增加|添加|新增)(?:一个|一条|项目|内容|技术|经历)?",
    r"保留",
    r"只要项目",
    r"不要个人",
)


def _matches(text: str, patterns: tuple[str, ...]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pattern in patterns:
        spans.extend((match.start(), match.end()) for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    return sorted(set(spans))


def classify_feedback(text: str) -> FeedbackClassification:
    value = str(text or "").strip()
    if not value:
        return FeedbackClassification("opinion", 0.0, conflicts=("反馈为空",))
    groups = {
        "fact": _matches(value, _FACT_PATTERNS),
        "style": _matches(value, _STYLE_PATTERNS),
        "structure": _matches(value, _STRUCTURE_PATTERNS),
    }
    active = [name for name, spans in groups.items() if spans]
    conflicts: list[str] = []
    negative_fact = bool(re.search(r"(?:没有|没|未曾|不曾)", value))
    structural_add = bool(re.search(r"(?:增加|添加|新增|替换)", value))
    explicit_do_not_add = bool(re.search(r"(?:不要|别|无需)\s*(?:增加|添加|新增|替换)", value))
    if "fact" in active and "structure" in active and negative_fact and structural_add and not explicit_do_not_add:
        conflicts.append("事实限制与结构修改要求同时出现")
    if len(active) == 1:
        category = active[0]
        confidence = 0.96
        spans = tuple(groups[category])
    elif len(active) > 1:
        category = "opinion"
        # Mixed instructions are not inherently contradictory (for example,
        # "突出 Python 并调整顺序").  Keep them as one user opinion with high
        # confidence; only the explicit conflict rules above should pause the
        # workflow.
        confidence = 0.72 if conflicts else 0.90
        spans = tuple(sorted({span for name in active for span in groups[name]}))
    else:
        category = "opinion"
        confidence = 0.80
        spans = ()
    fact_candidates: tuple[str, ...] = ()
    if "fact" in active:
        # Preserve the user's exact sentence for later field-level confirmation;
        # never convert it into a confirmed fact here.
        fact_candidates = (value,)
    return FeedbackClassification(
        category=category,
        confidence=confidence,
        spans=spans,
        fact_candidates=fact_candidates,
        conflicts=tuple(conflicts),
    )


__all__ = ["FEEDBACK_CATEGORIES", "FeedbackClassification", "classify_feedback"]
