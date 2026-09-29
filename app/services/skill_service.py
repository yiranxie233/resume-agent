"""Versioned, user-editable resume preference Skill stored as Markdown."""
from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any

from app.core.utils import sha256_text


SKILL_FILENAME = "resume-preferences.md"
_HEADING_RE = re.compile(r"^#{1,3}\s+.+$", re.MULTILINE)


def validate_skill(content: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(content, str):
        return False, ["Skill 必须是 Markdown 文本"]
    if len(content) > 50_000:
        errors.append("Skill 不得超过 50000 个字符")
    if content and not _HEADING_RE.search(content):
        errors.append("Skill 至少需要一个 Markdown 标题")
    forbidden = ("匹配权重", "安全确认规则", "系统提示词")
    for phrase in forbidden:
        if phrase in content:
            errors.append(f"Skill 不得修改系统规则：{phrase}")
    return not errors, errors


def skill_path(root: Path) -> Path:
    path = root / "skills" / SKILL_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_skill(root: Path) -> dict[str, Any]:
    path = skill_path(root)
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    versions_dir = root / "skills" / "versions"
    versions = []
    if versions_dir.exists():
        for item in versions_dir.glob("v*.md"):
            try:
                versions.append(int(item.stem[1:]))
            except ValueError:
                continue
    version = max(versions, default=(1 if content else 0))
    return {"version": version, "content": content, "path": str(path), "content_hash": sha256_text(content)}


def save_skill(root: Path, content: str, *, expected_version: int = 0) -> dict[str, Any]:
    valid, errors = validate_skill(content)
    if not valid:
        raise ValueError("；".join(errors))
    current = read_skill(root)
    if int(current["version"]) != expected_version:
        raise RuntimeError(f"skill version conflict: expected {expected_version}, current {current['version']}")
    version = expected_version + 1
    path = skill_path(root)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)
    history_dir = root / "skills" / "versions"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"v{version}.md"
    history_path.write_text(content, encoding="utf-8", newline="\n")
    previous = str(current.get("content", ""))
    diff = "".join(difflib.unified_diff(previous.splitlines(True), content.splitlines(True), fromfile=f"v{version-1}", tofile=f"v{version}"))
    return {"version": version, "content": content, "path": str(path), "content_hash": sha256_text(content), "diff": diff}


def list_skill_versions(root: Path) -> list[dict[str, Any]]:
    directory = root / "skills" / "versions"
    if not directory.exists():
        return []
    result: list[dict[str, Any]] = []
    for path in sorted(directory.glob("v*.md"), key=lambda item: item.name):
        content = path.read_text(encoding="utf-8")
        result.append({"version": int(path.stem[1:]), "content": content, "content_hash": sha256_text(content), "path": str(path)})
    return result


def rollback_skill(root: Path, version: int) -> dict[str, Any]:
    versions = {item["version"]: item for item in list_skill_versions(root)}
    if version not in versions:
        raise FileNotFoundError(f"skill version not found: {version}")
    current = read_skill(root)
    return save_skill(root, versions[version]["content"], expected_version=int(current["version"]))


__all__ = ["list_skill_versions", "read_skill", "rollback_skill", "save_skill", "skill_path", "validate_skill"]
