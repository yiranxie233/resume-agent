"""Model-assisted, user-confirmed structure mapping for cached resume templates.

The visual template remains the source of truth.  Model output is only a
proposed mapping from the template's existing text to the resume modules used
by the workflow.  A mapping is tied to the exact template bytes and is not
usable by task creation until the user explicitly confirms the editable
sections.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.schemas import ResumeDocument, ResumeSection
from app.core.utils import sha256_bytes, sha256_json, sha256_text, stable_id
from app.services.resume_llm_parser import structure_resume_with_model
from app.services.template_service import extract_template_text

TEMPLATE_STRUCTURE_VERSION = "template-llm-structure-v1"
TEMPLATE_STRUCTURE_FILENAME = "template-structure.json"
_ALLOWED_MODULES = {
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source(path: Path) -> tuple[bytes, str, str]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError("模板缓存文件不存在")
    data = source.read_bytes()
    text = extract_template_text(data, source.suffix)
    if not text.strip():
        raise ValueError("模板中没有可解析的文字结构")
    return data, text, sha256_bytes(data)


def _sidecar_path(path: Path) -> Path:
    return Path(path).resolve().parent / TEMPLATE_STRUCTURE_FILENAME


def _write_sidecar(path: Path, payload: dict[str, Any]) -> None:
    target = _sidecar_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(target)


def _section_payload(
    *, source_sha256: str, index: int, section: ResumeSection
) -> dict[str, Any]:
    content = str(section.content or "").strip()
    title = str(section.title or "").strip() or None
    section_id = stable_id(
        "template_section",
        source_sha256,
        index,
        section.module,
        title,
        content,
    )
    value = {"module": section.module, "title": title, "content": content}
    return {
        "section_id": section_id,
        **value,
        "value_hash": sha256_json(value),
        "confirmed": False,
    }


def structure_template_with_model(
    template_id: str,
    path: Path,
    *,
    profile: Any,
    gateway: Any,
    credential_handle_id: str | None = None,
) -> dict[str, Any]:
    """Parse one cached template with either an Ollama or cloud chat profile."""

    data, raw_text, source_sha256 = _source(path)
    suffix = Path(path).suffix.casefold()
    file_type = "docx" if suffix == ".docx" else "markdown"
    source_document = ResumeDocument(
        resume_id=f"template_{template_id}_{source_sha256[:12]}",
        filename=Path(path).name,
        file_type=file_type,
        file_size=len(data),
        page_count=None,
        raw_text=raw_text,
        sections=[
            ResumeSection(
                section_id="template_source",
                module="other",
                title="模板原文",
                content=raw_text,
                value_hash=sha256_text(raw_text),
            )
        ],
    )
    outcome = structure_resume_with_model(
        source_document,
        profile=profile,
        gateway=gateway,
        credential_handle_id=credential_handle_id,
    )
    sections = [
        _section_payload(source_sha256=source_sha256, index=index, section=section)
        for index, section in enumerate(outcome.resume.sections)
    ]
    if not sections:
        raise ValueError("模型没有识别出可确认的模板区块")
    structure_id = stable_id(
        "template_structure",
        TEMPLATE_STRUCTURE_VERSION,
        template_id,
        source_sha256,
        sections,
    )
    payload = {
        "status": "ready",
        "template_id": template_id,
        "source_filename": Path(path).name,
        "source_sha256": source_sha256,
        "structure_id": structure_id,
        "structure_version": TEMPLATE_STRUCTURE_VERSION,
        "sections": sections,
        "confirmed": False,
        "parsed_at": _utc_now(),
        "confirmed_at": None,
        "parser": {
            **outcome.metadata,
            "parser_version": TEMPLATE_STRUCTURE_VERSION,
            "source_parser_version": outcome.metadata.get("parser_version"),
            "document_kind": "resume_template",
        },
    }
    # Persist the unconfirmed proposal so a Streamlit rerun or API restart does
    # not force a second paid model call.  It remains unusable until confirmed.
    _write_sidecar(path, payload)
    return payload


def load_template_structure(template_id: str, path: Path) -> dict[str, Any] | None:
    """Load a structure only when it still belongs to the current source file."""

    sidecar = _sidecar_path(path)
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("template_id") != template_id:
        return None
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    if payload.get("source_sha256") != sha256_bytes(data):
        return {
            **payload,
            "status": "stale",
            "confirmed": False,
            "message": "模板文件已更新，请重新使用模型解析并确认结构",
        }
    return payload


def confirm_template_structure(
    template_id: str,
    path: Path,
    *,
    structure_id: str,
    source_sha256: str,
    sections: Any,
) -> dict[str, Any]:
    """Validate editable section mappings and atomically mark them confirmed."""

    pending = load_template_structure(template_id, path)
    if pending is None:
        raise FileNotFoundError("尚未使用聊天模型解析该模板")
    if pending.get("status") == "stale":
        raise ValueError("模板文件已更新，请重新解析")
    if pending.get("structure_id") != structure_id:
        raise RuntimeError("模板解析结果已变化，请刷新后重新确认")
    if pending.get("source_sha256") != source_sha256:
        raise RuntimeError("模板源文件版本不一致，请刷新后重新确认")
    if not isinstance(sections, list) or not sections:
        raise ValueError("模板结构至少需要一个区块")
    if len(sections) > 100:
        raise ValueError("模板结构区块过多")
    pending_sections = {
        str(item.get("section_id")): item
        for item in pending.get("sections") or []
        if isinstance(item, dict) and item.get("section_id")
    }
    if len(sections) != len(pending_sections):
        raise ValueError("确认时不能增加或删除模型识别的模板区块")
    confirmed_sections: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in sections:
        if not isinstance(item, dict):
            raise ValueError("模板区块必须是对象")
        section_id = str(item.get("section_id") or "").strip()
        original = pending_sections.get(section_id)
        if original is None or section_id in seen:
            raise ValueError("模板区块 ID 无效或重复")
        seen.add(section_id)
        expected_hash = str(item.get("old_value_hash") or "").strip()
        if not expected_hash or expected_hash != str(original.get("value_hash") or ""):
            raise RuntimeError("模板区块已变化，请刷新后重新确认")
        module = str(item.get("module") or "").strip()
        if module not in _ALLOWED_MODULES:
            raise ValueError(f"不支持的模板区块类型：{module}")
        title = str(item.get("title") or "").strip() or None
        content = str(item.get("content") or "").strip()
        if title is not None and len(title) > 200:
            raise ValueError("模板区块标题不能超过 200 个字符")
        if not content:
            raise ValueError("模板区块内容不能为空")
        if len(content) > 100_000:
            raise ValueError("模板区块内容过长")
        value = {"module": module, "title": title, "content": content}
        confirmed_sections.append(
            {
                "section_id": section_id,
                **value,
                "value_hash": sha256_json(value),
                "confirmed": True,
            }
        )
    confirmed = {
        **pending,
        "status": "confirmed",
        "sections": confirmed_sections,
        "confirmed": True,
        "confirmed_at": _utc_now(),
        "confirmed_structure_hash": sha256_json(confirmed_sections),
    }
    _write_sidecar(path, confirmed)
    return confirmed


__all__ = [
    "TEMPLATE_STRUCTURE_FILENAME",
    "TEMPLATE_STRUCTURE_VERSION",
    "confirm_template_structure",
    "load_template_structure",
    "structure_template_with_model",
]
