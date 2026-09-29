"""Recoverable local backup manifests for the single-user deployment."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXCLUDED_PARTS = {"edge-profile", ".internal-token", ".env"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return not any(part in EXCLUDED_PARTS or part.startswith(".tmp") for part in relative.parts)


def create_backup(data_root: Path, *, destination: Path | None = None) -> dict[str, Any]:
    data_root = data_root.resolve()
    backup_dir = destination or (data_root / "backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = backup_dir / f"resume-agent-{timestamp}.zip"
    entries: list[dict[str, Any]] = []
    fd, temporary_name = tempfile.mkstemp(prefix=".backup-", suffix=".zip", dir=backup_dir)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(data_root.rglob("*")):
                if not path.is_file() or not _included(path, data_root) or "backups" in path.relative_to(data_root).parts:
                    continue
                relative = path.relative_to(data_root).as_posix()
                digest = _sha256(path)
                entries.append({"path": relative, "sha256": digest, "size": path.stat().st_size})
                bundle.write(path, relative)
            manifest = {"format": "resume-agent-backup-v1", "created_at": datetime.now(timezone.utc).isoformat(), "files": entries}
            bundle.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    manifest["archive"] = str(archive)
    manifest["manifest_hash"] = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return manifest


def validate_backup(archive: Path) -> dict[str, Any]:
    if not archive.exists() or not archive.is_file():
        raise FileNotFoundError(str(archive))
    with zipfile.ZipFile(archive) as bundle:
        if "manifest.json" not in bundle.namelist():
            raise ValueError("备份缺少 manifest.json")
        manifest = json.loads(bundle.read("manifest.json").decode("utf-8"))
        errors: list[str] = []
        for item in manifest.get("files", []):
            name = str(item.get("path", ""))
            if name not in bundle.namelist():
                errors.append(f"缺少文件：{name}")
                continue
            digest = hashlib.sha256(bundle.read(name)).hexdigest()
            if digest != item.get("sha256"):
                errors.append(f"校验失败：{name}")
        return {"ok": not errors, "manifest": manifest, "errors": errors}


def restore_backup(archive: Path, data_root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    validation = validate_backup(archive)
    if not validation["ok"]:
        raise ValueError("备份校验失败")
    data_root = data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    restored: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        for item in validation["manifest"].get("files", []):
            relative = Path(str(item["path"]))
            target = (data_root / relative).resolve()
            if data_root not in target.parents:
                raise ValueError("备份路径越界")
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(relative.as_posix()))
            restored.append(relative.as_posix())
    return {"ok": True, "restored": restored, "skipped": len(validation["manifest"].get("files", [])) - len(restored)}


__all__ = ["create_backup", "restore_backup", "validate_backup"]
