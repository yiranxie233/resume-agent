"""Local resume preview/export helpers.

The first release deliberately keeps the export boundary conservative.  Markdown
is always available; DOCX is produced only when ``python-docx`` is installed and
PDF conversion is reported as pending unless a local office converter is found.
No exporter mutates the source resume or a confirmed version in place.
"""
from __future__ import annotations

import html
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

from app.core.schemas import CandidateProject, ResumeDocument


@dataclass(frozen=True)
class PreviewResult:
    format: str
    status: str
    path: str | None
    page_count: int | None
    one_page: bool | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompressionResult:
    """A non-destructive one-page adjustment attempt.

    ``rewritten`` is deliberately explicit.  Font/spacing adjustments are
    represented as layout metadata and never change the text.  Text changes are
    only made when the caller has separately opted into rewriting.
    """

    content: str
    page_count: int
    font_scale: float
    line_spacing: float
    margin_scale: float
    rewritten: bool = False
    warnings: tuple[str, ...] = ()
    density: float = 1.0


def _safe_name(value: str) -> str:
    value = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    return value.strip("._")[:80] or "resume"


def render_markdown(
    *,
    job_title: str,
    candidates: list[CandidateProject],
    resume: ResumeDocument | None = None,
    selected_candidate_id: str | None = None,
) -> str:
    """Render a reviewable Chinese resume draft without claiming facts."""

    lines: list[str] = ["# 中文简历草稿", "", f"> 目标岗位：{job_title}", ""]
    if resume:
        lines.extend(["## 原简历模块", ""])
        for section in resume.sections:
            lines.extend([f"### {section.title or section.module}", "", section.content, ""])
    selected = next((item for item in candidates if item.candidate_id == selected_candidate_id), None)
    if selected is not None:
        lines.extend(["## 项目经历（草稿）", "", f"### {selected.title}  {selected.period}", ""])
        lines.extend([selected.introduction, "", f"**技术栈：** {'、'.join(selected.tech_stack)}", ""])
        lines.append("**解决方案：**")
        lines.extend([f"- {item}" for item in selected.solutions])
        lines.append("")
        lines.append("**结果：**")
        lines.extend([f"- {item}" for item in selected.results])
        lines.append("")
        if selected.needs_verification:
            lines.extend(["> [示例/待核实] 以上项目内容未经逐字段确认，不得视为正式简历事实。", ""])
    else:
        lines.extend(["## 候选项目", ""])
        for item in candidates:
            lines.extend([f"### {item.title}", "", item.introduction, ""])
    return "\n".join(lines).strip() + "\n"


def estimate_pages(markdown: str, *, density: float = 1.0) -> int:
    """Conservative page estimate for preview when no office renderer exists."""

    # A Chinese technical resume usually fits roughly 2,000 visible characters
    # per dense page.  This is only a warning, never a compliance assertion.
    capacity = max(800, int(2000 * max(0.5, min(2.0, float(density)))))
    return max(1, (len(markdown) + capacity - 1) // capacity)


def compress_markdown(content: str, *, allow_rewrite: bool = False) -> CompressionResult:
    """Try the approved one-page compression sequence.

    The first pass only changes layout density.  If that cannot fit one page,
    text is shortened only when ``allow_rewrite`` is true.  Headings and the
    relative order of sections are retained, and the caller still receives the
    resulting preview for user approval.
    """

    original = str(content or "")
    for font_scale, line_spacing, margin_scale, density in (
        (0.95, 0.95, 0.95, 1.15),
        (0.90, 0.90, 0.90, 1.28),
        (0.85, 0.85, 0.85, 1.42),
    ):
        pages = estimate_pages(original, density=density)
        if pages <= 1:
            return CompressionResult(
                original,
                pages,
                font_scale,
                line_spacing,
                margin_scale,
                False,
                ("已先调整字号、行距和页边距，正文文字未改写",),
                density,
            )

    if not allow_rewrite:
        return CompressionResult(
            original,
            estimate_pages(original, density=1.42),
            0.85,
            0.85,
            0.85,
            False,
            ("仅调整排版后仍超过一页；需要用户另行允许改写或手动删减",),
            1.42,
        )

    # Deterministic, conservative shortening for the explicit rewrite path.
    # It does not invent facts or merge unrelated sections.  Bullets are clipped
    # at sentence boundaries where possible; headings and labels are untouched.
    rewritten_lines: list[str] = []
    for line in original.splitlines():
        stripped = line.strip()
        if not stripped:
            if rewritten_lines and rewritten_lines[-1] != "":
                rewritten_lines.append("")
            continue
        if stripped.startswith(("#", ">", "- ", "* ")):
            prefix = line[: len(line) - len(line.lstrip())]
            marker = ""
            body = stripped
            if stripped.startswith("> "):
                marker, body = "> ", stripped[2:]
            elif stripped.startswith(("- ", "* ")):
                marker, body = stripped[:2], stripped[2:]
            elif stripped.startswith("#"):
                heading = re.match(r"^(#+\s*)", stripped)
                marker = heading.group(1) if heading else ""
                body = stripped[len(marker) :]
            if marker and not marker.startswith("#") and len(body) > 150:
                # Prefer a complete Chinese/English sentence over a hard cut.
                cut = body[:150]
                boundary = max(cut.rfind("。"), cut.rfind("；"), cut.rfind("."), cut.rfind(";"))
                if boundary >= 60:
                    cut = cut[: boundary + 1]
                body = cut.rstrip("，,；; ") + "…"
            rewritten_lines.append(prefix + marker + body)
        else:
            rewritten_lines.append(line)
    rewritten = "\n".join(rewritten_lines).strip() + "\n"
    pages = estimate_pages(rewritten, density=1.28)
    return CompressionResult(
        rewritten,
        pages,
        0.90,
        0.90,
        0.90,
        True,
        ("已按用户授权缩短过长条目；请检查预览和事实准确性",) if pages <= 1 else ("自动改写后仍超过一页，请手动删减模块",),
        1.28,
    )


def write_preview(content: str, *, root: Path, name: str = "preview.md") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / _safe_name(name)
    fd, temporary_name = tempfile.mkstemp(prefix=".preview-", suffix=".tmp", dir=root)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def export_docx(content: str, *, target: Path, layout: dict[str, Any] | None = None) -> PreviewResult:
    try:
        from docx import Document  # type: ignore
    except ImportError:
        return PreviewResult("docx", "dependency_missing", None, None, None, ("请安装 documents 依赖",))
    document = Document()
    layout = dict(layout or {})
    font_name = str(layout.get("font_name") or "Microsoft YaHei")
    try:
        font_size = float(layout.get("font_size_pt", 10.5))
        line_spacing = float(layout.get("line_spacing", 1.0))
        margin_cm = float(layout.get("margin_cm", 1.2))
        if not (6 <= font_size <= 18 and 0.7 <= line_spacing <= 2.0 and 0.3 <= margin_cm <= 3.0):
            raise ValueError
    except (TypeError, ValueError):
        return PreviewResult("docx", "invalid_layout", None, None, None, ("排版参数超出允许范围",))
    for section in document.sections:
        from docx.shared import Cm

        section.top_margin = Cm(margin_cm)
        section.bottom_margin = Cm(margin_cm)
        section.left_margin = Cm(margin_cm)
        section.right_margin = Cm(margin_cm)
    from docx.shared import Pt

    normal = document.styles["Normal"]
    normal.font.name = font_name
    normal.font.size = Pt(font_size)
    normal._element.rPr.rFonts.set(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", font_name
    )
    normal.paragraph_format.line_spacing = line_spacing
    normal.paragraph_format.space_after = Pt(0)
    for style_name in ("Heading 1", "Heading 2", "Heading 3", "List Bullet"):
        try:
            style = document.styles[style_name]
            style.font.name = font_name
            style.font.size = Pt(font_size if style_name == "List Bullet" else font_size + (2 if style_name == "Heading 1" else 1))
            style._element.rPr.rFonts.set(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", font_name
            )
            style.paragraph_format.line_spacing = line_spacing
            style.paragraph_format.space_after = Pt(0)
        except KeyError:
            continue
    for line in content.splitlines():
        if line.startswith("# "):
            document.add_heading(line[2:].strip(), level=1)
        elif line.startswith("## "):
            document.add_heading(line[3:].strip(), level=2)
        elif line.startswith("### "):
            document.add_heading(line[4:].strip(), level=3)
        elif line.startswith("> "):
            document.add_paragraph(line[2:].strip())
        elif line.startswith("- "):
            document.add_paragraph(line[2:].strip(), style="List Bullet")
        elif line.strip():
            document.add_paragraph(line)
    target.parent.mkdir(parents=True, exist_ok=True)
    document.save(target)
    # python-docx cannot certify pagination; keep the result explicitly pending.
    return PreviewResult("docx", "preview_pending", str(target), None, None, ("尚未使用 Word/LibreOffice 渲染分页",))


def convert_pdf_with_libreoffice(docx_path: Path, pdf_path: Path) -> PreviewResult:
    executable = shutil.which("soffice") or shutil.which("libreoffice")
    if not executable:
        return PreviewResult("pdf", "dependency_missing", None, None, None, ("未检测到 LibreOffice，请手动安装后重新检测",))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [executable, "--headless", "--convert-to", "pdf", "--outdir", str(pdf_path.parent), str(docx_path)],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return PreviewResult("pdf", "conversion_failed", None, None, None, (str(exc),))
    produced = pdf_path.parent / f"{docx_path.stem}.pdf"
    if completed.returncode != 0 or not produced.exists():
        return PreviewResult("pdf", "conversion_failed", None, None, None, (completed.stderr[-500:],))
    page_count: int | None = None
    try:
        import fitz  # type: ignore

        page_count = fitz.open(produced).page_count
    except ImportError:
        pass
    return PreviewResult("pdf", "ready", str(produced), page_count, page_count == 1 if page_count is not None else None)


def office_status() -> dict[str, Any]:
    return {
        "word": bool(shutil.which("WINWORD.EXE") or shutil.which("winword")),
        "libreoffice": bool(shutil.which("soffice") or shutil.which("libreoffice")),
        "pdf_parser": _has_fitz(),
    }


def _has_fitz() -> bool:
    try:
        import fitz  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


__all__ = [
    "CompressionResult",
    "PreviewResult",
    "compress_markdown",
    "convert_pdf_with_libreoffice",
    "estimate_pages",
    "export_docx",
    "office_status",
    "render_markdown",
    "write_preview",
]
