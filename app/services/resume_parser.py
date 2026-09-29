"""Safe, dependency-optional resume ingestion for DOCX/text PDFs/Markdown."""
from __future__ import annotations

import re
import uuid
import math
from pathlib import Path

from app.core.schemas import ResumeDocument, ResumeSection
from app.core.utils import sha256_text

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_PAGES = 3


class ResumeParseError(ValueError):
    """Raised when a file cannot be safely imported."""


def _parse_sections(text: str) -> list[ResumeSection]:
    patterns = {
        "projects": r"项目(?:经历|经验)?",
        "skills": r"(?:专业技能|技能清单|技术栈|技能)",
        "summary": r"(?:个人简介|个人概述|简介)",
        "evaluation": r"(?:个人评价|自我评价)",
        "education": r"教育经历",
        "work": r"工作经历|实习经历",
        "certificates": r"证书|资格证",
        "objective": r"求职目标|求职意向",
        "basic": r"基本信息|联系方式",
    }
    lines = text.splitlines()
    sections: list[ResumeSection] = []
    current_module = "basic"
    current_title: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        content = "\n".join(buffer).strip()
        if content:
            section_id = f"section_{uuid.uuid4().hex[:12]}"
            sections.append(
                ResumeSection(
                    section_id=section_id,
                    module=current_module,  # type: ignore[arg-type]
                    title=current_title,
                    content=content,
                    value_hash=sha256_text(content),
                )
            )
        buffer = []

    for line in lines:
        stripped = line.strip().lstrip("#*- ")
        found = next(((module, title) for module, pattern in patterns.items() if re.search(pattern, stripped) for title in [stripped]), None)
        if found:
            flush()
            current_module, current_title = found
        else:
            buffer.append(line)
    flush()
    return sections


def parse_resume_bytes(filename: str, data: bytes) -> ResumeDocument:
    if len(data) > MAX_FILE_SIZE:
        raise ResumeParseError("文件超过 10 MB 限制")
    suffix = Path(filename).suffix.lower()
    if suffix == ".doc":
        raise ResumeParseError("不支持 .doc，请转换为 .docx")
    if suffix not in {".docx", ".pdf", ".md", ".markdown", ".txt"}:
        raise ResumeParseError("仅支持 DOCX、文本型 PDF、Markdown 或 TXT")
    text = ""
    page_count: int | None = None
    if suffix == ".docx":
        try:
            from docx import Document  # type: ignore
        except ImportError as exc:
            raise ResumeParseError("缺少 python-docx，请安装 documents 依赖") from exc
        try:
            document = Document(__import__("io").BytesIO(data))
            text = "\n".join(p.text for p in document.paragraphs)
            text += "\n" + "\n".join(" | ".join(cell.text for cell in row.cells) for table in document.tables for row in table.rows)
        except Exception as exc:
            raise ResumeParseError("DOCX 文件无法解析") from exc
        # DOCX has no reliable page count without a layout engine.  Use a
        # conservative text estimate for the import gate and expose it as an
        # estimate; Word/LibreOffice remains authoritative during export.
        page_count = max(1, math.ceil(len(text) / 2000))
        if page_count > MAX_PAGES:
            raise ResumeParseError("简历最多支持 3 页")
    elif suffix == ".pdf":
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise ResumeParseError("缺少 PyMuPDF，请安装 documents 依赖") from exc
        try:
            document = fitz.open(stream=data, filetype="pdf")
            page_count = document.page_count
            if page_count > MAX_PAGES:
                raise ResumeParseError("简历最多支持 3 页")
            text = "\n".join(page.get_text("text") for page in document)
            if not text.strip():
                raise ResumeParseError("PDF 不包含可复制文本，不支持 OCR")
        except ResumeParseError:
            raise
        except Exception as exc:
            raise ResumeParseError("PDF 无法解析或已损坏") from exc
    else:
        text = data.decode("utf-8-sig", errors="replace")
        page_count = 1
    if not text.strip():
        raise ResumeParseError("简历内容为空")
    return ResumeDocument(
        resume_id=f"resume_{uuid.uuid4().hex}",
        filename=filename,
        file_type={".docx": "docx", ".pdf": "pdf", ".md": "markdown", ".markdown": "markdown", ".txt": "text"}[suffix],  # type: ignore[arg-type]
        file_size=len(data),
        page_count=page_count,
        sections=_parse_sections(text),
        raw_text=text,
    )
