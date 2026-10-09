"""Local resume preview/export helpers.

The first release deliberately keeps the export boundary conservative.  Markdown
is always available; DOCX is produced only when ``python-docx`` is installed and
PDF conversion is reported as pending unless a local office converter is found.
No exporter mutates the source resume or a confirmed version in place.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from contextlib import suppress
from dataclasses import dataclass
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
    level: str = "standard"
    compact_reflow: bool = False


def _safe_name(value: str) -> str:
    value = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    return value.strip("._")[:80] or "resume"


def render_candidate_project(candidate: CandidateProject, *, heading_level: int = 3) -> str:
    """Render one project once so preview, copy and template output stay aligned."""

    heading = "#" * max(1, min(6, int(heading_level)))
    lines = [f"{heading} {candidate.title}  {candidate.period}".rstrip(), ""]
    if candidate.introduction.strip():
        lines.extend([f"**项目简介：** {candidate.introduction.strip()}", ""])
    if candidate.tech_stack:
        lines.extend([f"**技术栈：** {'、'.join(candidate.tech_stack)}", ""])
    if candidate.solutions:
        lines.append("**解决方案：**")
        lines.extend(f"{index}. {value}" for index, value in enumerate(candidate.solutions, 1))
        lines.append("")
    if candidate.results:
        lines.extend([f"**结果：** {'；'.join(candidate.results)}", ""])
    if candidate.needs_verification:
        lines.extend(["> [示例/待核实] 以上项目内容未经逐字段确认，不得视为正式简历事实。", ""])
    return "\n".join(lines).strip()


def _replace_markdown_section(template: str, headings: tuple[str, ...], body: str) -> tuple[str, bool]:
    """Replace one common Markdown section while retaining its heading line."""

    names = "|".join(re.escape(value) for value in headings)
    pattern = re.compile(
        rf"(?ms)^(?P<marks>#{{1,6}})\s*(?P<title>{names})\s*$"
        rf"(?P<body>.*?)(?=^#{{1,6}}\s|\Z)"
    )
    match = pattern.search(template)
    if not match:
        return template, False
    replacement = f"{match.group('marks')} {match.group('title')}\n\n{body.strip()}\n\n"
    return template[: match.start()] + replacement + template[match.end() :], True


def apply_markdown_template(
    template: str,
    *,
    base_facts: dict[str, Any],
    project_content: str,
    job_title: str,
) -> str:
    """Fill a downloaded Markdown template without executing embedded content."""

    values = {
        "name": str(base_facts.get("name") or "[待补充]"),
        "phone": str(base_facts.get("phone") or "[待补充]"),
        "email": str(base_facts.get("email") or "[待补充]"),
        "location": str(base_facts.get("location") or "[待补充]"),
        "objective": str(base_facts.get("objective") or job_title or "[待补充]"),
        "education": str(base_facts.get("education") or "[待补充]"),
        "certificates": str(base_facts.get("certificates") or "[待补充]"),
        "project_experience": project_content or "[待补充]",
    }
    rendered = str(template or "")
    aliases = {
        "name": ("name", "姓名"),
        "phone": ("phone", "电话"),
        "email": ("email", "邮箱"),
        "location": ("location", "所在地"),
        "objective": ("objective", "求职目标", "求职意向"),
        "education": ("education", "教育经历"),
        "certificates": ("certificates", "证书"),
        "project_experience": ("project_experience", "projects", "项目经历"),
    }
    for key, names in aliases.items():
        for name in names:
            rendered = rendered.replace("{{" + name + "}}", values[key])

    basic = " ｜ ".join(
        value
        for value in (values["phone"], values["email"], values["location"])
        if value != "[待补充]"
    ) or "[待补充]"
    section_values = (
        (("个人信息", "基本信息", "联系方式"), f"{values['name']}\n\n{basic}"),
        (("求职目标", "求职意向"), values["objective"]),
        (("教育经历",), values["education"]),
        (("证书", "证书与荣誉"), values["certificates"]),
        (("项目经历",), values["project_experience"]),
    )
    project_replaced = False
    for headings, body in section_values:
        rendered, replaced = _replace_markdown_section(rendered, headings, body)
        if "项目经历" in headings:
            project_replaced = replaced
    if not project_replaced and project_content:
        rendered = rendered.rstrip() + "\n\n## 项目经历\n\n" + project_content.strip() + "\n"
    return rendered.strip() + "\n"


def render_markdown(
    *,
    job_title: str,
    candidates: list[CandidateProject],
    resume: ResumeDocument | None = None,
    selected_candidate_id: str | None = None,
    base_facts: dict[str, Any] | None = None,
    template_content: str | None = None,
) -> str:
    """Render a reviewable Chinese resume draft without claiming facts."""

    facts = dict(base_facts or {})
    lines: list[str] = ["# 中文简历草稿", ""]
    if facts:
        lines[0] = f"# {str(facts.get('name') or '中文简历').strip()}"
        contacts = [
            str(facts.get(key) or "").strip()
            for key in ("phone", "email", "location")
        ]
        contacts = [value for value in contacts if value]
        if contacts:
            lines.extend([" ｜ ".join(contacts), ""])
    lines.extend([f"> 目标岗位：{str(facts.get('objective') or job_title).strip()}", ""])
    if facts.get("education"):
        lines.extend(["## 教育经历", "", str(facts["education"]).strip(), ""])
    if facts.get("certificates"):
        lines.extend(["## 证书", "", str(facts["certificates"]).strip(), ""])
    if resume:
        lines.extend(["## 原简历模块", ""])
        for section in resume.sections:
            lines.extend([f"### {section.title or section.module}", "", section.content, ""])
    selected = next((item for item in candidates if item.candidate_id == selected_candidate_id), None)
    if selected is not None:
        project_content = render_candidate_project(selected)
        lines.extend(["## 项目经历", "", project_content, ""])
    elif candidates:
        lines.extend(["## 候选项目", ""])
        for item in candidates:
            lines.extend([f"### {item.title}", "", item.introduction, ""])
    content = "\n".join(lines).strip() + "\n"
    if template_content is not None and resume is None:
        return apply_markdown_template(
            template_content,
            base_facts=facts,
            project_content=render_candidate_project(selected) if selected is not None else "",
            job_title=job_title,
        )
    return content


def estimate_pages(markdown: str, *, density: float = 1.0) -> int:
    """Conservative page estimate for preview when no office renderer exists."""

    # A Chinese technical resume usually fits roughly 2,000 visible characters
    # per dense page.  This is only a warning, never a compliance assertion.
    capacity = max(800, int(2000 * max(0.5, min(2.0, float(density)))))
    return max(1, (len(markdown) + capacity - 1) // capacity)


def compress_markdown(
    content: str,
    *,
    allow_rewrite: bool = False,
    level: str = "standard",
) -> CompressionResult:
    """Try the approved one-page compression sequence.

    The first pass only changes layout density.  If that cannot fit one page,
    text is shortened only when ``allow_rewrite`` is true.  Headings and the
    relative order of sections are retained, and the caller still receives the
    resulting preview for user approval.
    """

    original = str(content or "")
    compression_level = str(level or "standard").strip().casefold()
    if compression_level not in {"standard", "aggressive"}:
        compression_level = "standard"
    profiles = [
        (0.95, 0.95, 0.95, 1.15),
        (0.90, 0.90, 0.90, 1.28),
        (0.85, 0.85, 0.85, 1.42),
    ]
    if compression_level == "aggressive":
        # Text remains byte-for-byte identical. The extra density represents
        # tighter paragraph spacing/list indents and compact section flow in
        # addition to smaller fonts, line spacing and margins.
        profiles.extend(
            [
                (0.80, 0.78, 0.65, 1.72),
                (0.75, 0.72, 0.55, 2.05),
            ]
        )
    for font_scale, line_spacing, margin_scale, density in profiles:
        pages = estimate_pages(original, density=density)
        if pages <= 1:
            return CompressionResult(
                original,
                pages,
                font_scale,
                line_spacing,
                margin_scale,
                False,
                (
                    "已采用紧凑重排、字号、行距和页边距压缩，正文文字未改写"
                    if compression_level == "aggressive"
                    else "已先调整字号、行距和页边距，正文文字未改写",
                ),
                density,
                compression_level,
                compression_level == "aggressive",
            )

    if not allow_rewrite:
        font_scale, line_spacing, margin_scale, density = profiles[-1]
        return CompressionResult(
            original,
            estimate_pages(original, density=density),
            font_scale,
            line_spacing,
            margin_scale,
            False,
            (
                "最强无损压缩后仍超过一页；可接受当前页数，正文内容未作任何修改"
                if compression_level == "aggressive"
                else "常规无损压缩后仍超过一页；可继续选择紧凑重排，正文内容不会修改",
            ),
            density,
            compression_level,
            compression_level == "aggressive",
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
        compression_level,
        compression_level == "aggressive",
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


def _export_docx_impl(
    content: str,
    *,
    target: Path,
    layout: dict[str, Any] | None = None,
    template_path: Path | None = None,
    template_replacements: dict[str, str] | None = None,
) -> PreviewResult:
    try:
        from docx import Document  # type: ignore
        from docx.opc.exceptions import PackageNotFoundError  # type: ignore
    except ImportError:
        return PreviewResult("docx", "dependency_missing", None, None, None, ("请安装 documents 依赖",))
    source_template = Path(template_path).resolve() if template_path else None
    if source_template is not None and source_template.suffix.lower() == ".docx":
        if not source_template.is_file():
            return PreviewResult("docx", "template_missing", None, None, None, ("所选 DOCX 模板缓存不存在",))
        try:
            document = Document(str(source_template))
        except (KeyError, OSError, PackageNotFoundError, ValueError):
            return PreviewResult("docx", "template_invalid", None, None, None, ("所选 DOCX 模板无法打开",))
    else:
        document = Document()
    layout = dict(layout or {})
    template_applied = source_template is not None and source_template.suffix.lower() == ".docx"
    apply_layout = not template_applied or bool(layout)
    font_name = str(layout.get("font_name") or "Microsoft YaHei")
    try:
        font_size = float(layout.get("font_size_pt", 10.5))
        line_spacing = float(layout.get("line_spacing", 1.0))
        margin_cm = float(layout.get("margin_cm", 1.2))
        if not (6 <= font_size <= 18 and 0.7 <= line_spacing <= 2.0 and 0.3 <= margin_cm <= 3.0):
            raise ValueError
    except (TypeError, ValueError):
        return PreviewResult("docx", "invalid_layout", None, None, None, ("排版参数超出允许范围",))
    if apply_layout:
        for section in document.sections:
            from docx.shared import Cm

            section.top_margin = Cm(margin_cm)
            section.bottom_margin = Cm(margin_cm)
            section.left_margin = Cm(margin_cm)
            section.right_margin = Cm(margin_cm)
    from docx.enum.style import WD_STYLE_TYPE
    from docx.shared import Pt

    # Several popular Chinese DOCX templates rename/remove Word's English
    # built-in heading and list styles. ``Document.add_heading`` otherwise
    # raises KeyError and the API becomes a generic 500. Re-create only the
    # missing paragraph styles while leaving every existing template style
    # untouched.
    required_styles = ("Normal", "Heading 1", "Heading 2", "Heading 3", "List Bullet", "List Number")
    for style_name in required_styles:
        try:
            document.styles[style_name]
        except KeyError:
            style = document.styles.add_style(style_name, WD_STYLE_TYPE.PARAGRAPH)
            if style_name != "Normal":
                try:
                    style.base_style = document.styles["Normal"]
                except KeyError:
                    pass

    if apply_layout:
        try:
            normal = document.styles["Normal"]
        except KeyError:
            normal = document.styles.add_style("Normal", WD_STYLE_TYPE.PARAGRAPH)
        normal.font.name = font_name
        normal.font.size = Pt(font_size)
        normal._element.rPr.rFonts.set(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", font_name
        )
        normal.paragraph_format.line_spacing = line_spacing
        normal.paragraph_format.space_before = Pt(
            float(layout.get("paragraph_spacing_before_pt", 0))
        )
        normal.paragraph_format.space_after = Pt(
            float(layout.get("paragraph_spacing_after_pt", 0))
        )
        for style_name in ("Heading 1", "Heading 2", "Heading 3", "List Bullet"):
            try:
                style = document.styles[style_name]
                style.font.name = font_name
                style.font.size = Pt(
                    font_size
                    if style_name == "List Bullet"
                    else font_size + (2 if style_name == "Heading 1" else 1)
                )
                style._element.rPr.rFonts.set(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia",
                    font_name,
                )
                style.paragraph_format.line_spacing = line_spacing
                style.paragraph_format.space_after = Pt(0)
                if style_name.startswith("Heading"):
                    style.paragraph_format.keep_with_next = True
            except KeyError:
                continue
    replacements = {
        str(key): str(value)
        for key, value in (template_replacements or {}).items()
        if str(key)
    }
    replaced_resume_content = False
    if template_applied:
        paragraphs = list(document.paragraphs)
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    paragraphs.extend(cell.paragraphs)
        for paragraph in paragraphs:
            original = paragraph.text
            updated = original
            for key, value in replacements.items():
                for token in ("{{" + key + "}}", "${" + key + "}"):
                    updated = updated.replace(token, value)
            for token in ("{{resume_content}}", "${resume_content}"):
                if token in updated:
                    updated = updated.replace(token, content)
                    replaced_resume_content = True
            if updated != original:
                paragraph.text = updated

    # Templates with an explicit whole-resume placeholder receive all content
    # in place. A downloaded sample without placeholders must not leak its
    # example person's name/photo/history into the generated resume. Clear its
    # body while retaining page setup, headers/footers and reusable styles,
    # then rebuild the confirmed content inside that template package.
    if not template_applied or not replaced_resume_content:
        if template_applied and not replaced_resume_content:
            from docx.oxml.ns import qn

            body = document._element.body
            for child in list(body):
                if child.tag != qn("w:sectPr"):
                    body.remove(child)
        for line in content.splitlines():
            plain_line = line.replace("**", "")
            if plain_line.startswith("# "):
                document.add_heading(plain_line[2:].strip(), level=1)
            elif plain_line.startswith("## "):
                document.add_heading(plain_line[3:].strip(), level=2)
            elif plain_line.startswith("### "):
                document.add_heading(plain_line[4:].strip(), level=3)
            elif plain_line.startswith("> "):
                document.add_paragraph(plain_line[2:].strip())
            elif re.match(r"^\d+\.\s", plain_line):
                try:
                    document.add_paragraph(re.sub(r"^\d+\.\s*", "", plain_line), style="List Number")
                except KeyError:
                    document.add_paragraph(plain_line)
            elif plain_line.startswith("- "):
                try:
                    document.add_paragraph(plain_line[2:].strip(), style="List Bullet")
                except KeyError:
                    document.add_paragraph(plain_line)
            elif plain_line.strip():
                document.add_paragraph(plain_line)
    if bool(layout.get("compact_reflow")):
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.shared import Cm

        for section in document.sections:
            section.header_distance = Cm(0.3)
            section.footer_distance = Cm(0.3)
        all_paragraphs = list(document.paragraphs)
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    all_paragraphs.extend(cell.paragraphs)
                    tc_pr = cell._tc.get_or_add_tcPr()
                    tc_mar = tc_pr.first_child_found_in("w:tcMar")
                    if tc_mar is None:
                        tc_mar = OxmlElement("w:tcMar")
                        tc_pr.append(tc_mar)
                    for edge in ("top", "left", "bottom", "right"):
                        node = tc_mar.find(qn(f"w:{edge}"))
                        if node is None:
                            node = OxmlElement(f"w:{edge}")
                            tc_mar.append(node)
                        node.set(qn("w:w"), "36")
                        node.set(qn("w:type"), "dxa")
        for paragraph in all_paragraphs:
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(0)
            style_name = str(getattr(paragraph.style, "name", "") or "")
            if style_name.startswith("Heading"):
                paragraph.paragraph_format.keep_with_next = True
            elif style_name in {"List Bullet", "List Number"}:
                paragraph.paragraph_format.left_indent = Cm(0.35)
                paragraph.paragraph_format.first_line_indent = Cm(-0.25)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        document.save(target)
    except (KeyError, OSError, PackageNotFoundError, TypeError, ValueError) as exc:
        target.unlink(missing_ok=True)
        return PreviewResult(
            "docx",
            "conversion_failed",
            None,
            None,
            None,
            (f"DOCX 模板生成失败：{type(exc).__name__}",),
        )
    # python-docx cannot certify pagination; keep the result explicitly pending.
    warnings = ["尚未使用 Word/LibreOffice 渲染分页"]
    if template_applied and not replaced_resume_content:
        warnings.append("模板没有完整内容占位符，已清除示例人物正文并保留页面设置与可复用样式")
    return PreviewResult("docx", "preview_pending", str(target), None, None, tuple(warnings))


def export_docx(
    content: str,
    *,
    target: Path,
    layout: dict[str, Any] | None = None,
    template_path: Path | None = None,
    template_replacements: dict[str, str] | None = None,
) -> PreviewResult:
    """Guard the untrusted DOCX template boundary from generic API 500s."""

    try:
        return _export_docx_impl(
            content,
            target=target,
            layout=layout,
            template_path=template_path,
            template_replacements=template_replacements,
        )
    except Exception as exc:
        target.unlink(missing_ok=True)
        return PreviewResult(
            "docx",
            "conversion_failed",
            None,
            None,
            None,
            (f"DOCX 模板生成失败：{type(exc).__name__}",),
        )


def convert_pdf_with_libreoffice(docx_path: Path, pdf_path: Path) -> PreviewResult:
    executable = shutil.which("soffice") or shutil.which("libreoffice")
    if not executable and os.name == "nt":
        for candidate in (
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
            / "LibreOffice/program/soffice.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))
            / "LibreOffice/program/soffice.exe",
        ):
            if candidate.is_file():
                executable = str(candidate)
                break
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


def convert_pdf_with_word(docx_path: Path, pdf_path: Path) -> PreviewResult:
    """Render a DOCX with an installed Microsoft Word without showing a window.

    This is a Windows-only fallback for local machines that have Word but not
    LibreOffice.  Automation and link updates are disabled before opening the
    untrusted, already validated ``.docx`` template.
    """

    if os.name != "nt":
        return PreviewResult(
            "pdf",
            "dependency_missing",
            None,
            None,
            None,
            ("当前系统未检测到可用的 DOCX 排版渲染器",),
        )
    try:
        import pythoncom  # type: ignore
        import win32com.client  # type: ignore
    except ImportError:
        return PreviewResult(
            "pdf",
            "dependency_missing",
            None,
            None,
            None,
            ("未安装 pywin32，无法调用本机 Microsoft Word 生成预览",),
        )

    source = Path(docx_path).resolve()
    target = Path(pdf_path).resolve()
    if not source.is_file() or source.suffix.casefold() != ".docx":
        return PreviewResult(
            "pdf", "conversion_failed", None, None, None, ("DOCX 模板不存在",)
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    word = None
    document = None
    pythoncom.CoInitialize()
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        # msoAutomationSecurityForceDisable. DOCX cannot normally contain VBA,
        # but forcing this flag also protects against malformed/mislabelled input.
        try:
            word.AutomationSecurity = 3
            word.Options.UpdateLinksAtOpen = False
        except (pythoncom.com_error, AttributeError) as exc:
            return PreviewResult(
                "pdf",
                "conversion_failed",
                None,
                None,
                None,
                (f"Microsoft Word 无法启用安全预览模式：{type(exc).__name__}",),
            )
        document = word.Documents.Open(
            str(source),
            ConfirmConversions=False,
            ReadOnly=True,
            AddToRecentFiles=False,
            Visible=False,
            OpenAndRepair=True,
            NoEncodingDialog=True,
        )
        document.ExportAsFixedFormat(
            OutputFileName=str(target),
            ExportFormat=17,  # wdExportFormatPDF
            OpenAfterExport=False,
            OptimizeFor=0,
            CreateBookmarks=0,
            DocStructureTags=True,
            BitmapMissingFonts=True,
        )
    except (pythoncom.com_error, AttributeError, OSError) as exc:
        target.unlink(missing_ok=True)
        return PreviewResult(
            "pdf",
            "conversion_failed",
            None,
            None,
            None,
            (f"Microsoft Word 生成模板预览失败：{type(exc).__name__}",),
        )
    finally:
        if document is not None:
            with suppress(pythoncom.com_error, AttributeError):
                document.Close(False)
        if word is not None:
            with suppress(pythoncom.com_error, AttributeError):
                word.Quit(False)
        pythoncom.CoUninitialize()
    if not target.is_file():
        return PreviewResult(
            "pdf", "conversion_failed", None, None, None, ("Word 未生成 PDF 文件",)
        )
    page_count: int | None = None
    try:
        import fitz  # type: ignore

        with fitz.open(target) as document_pdf:
            page_count = document_pdf.page_count
    except (ImportError, OSError, ValueError):
        pass
    return PreviewResult(
        "pdf",
        "ready",
        str(target),
        page_count,
        page_count == 1 if page_count is not None else None,
    )


def convert_docx_to_pdf(docx_path: Path, pdf_path: Path) -> PreviewResult:
    """Use LibreOffice first and Microsoft Word as the local Windows fallback."""

    libreoffice = convert_pdf_with_libreoffice(docx_path, pdf_path)
    if libreoffice.status == "ready":
        return libreoffice
    word = convert_pdf_with_word(docx_path, pdf_path)
    if word.status == "ready":
        return word
    warnings = tuple(dict.fromkeys((*libreoffice.warnings, *word.warnings)))
    return PreviewResult(
        "pdf",
        (
            "dependency_missing"
            if libreoffice.status == word.status == "dependency_missing"
            else "conversion_failed"
        ),
        None,
        None,
        None,
        warnings,
    )


def export_text_pdf(
    content: str,
    *,
    target: Path,
    layout: dict[str, Any] | None = None,
) -> PreviewResult:
    """Generate a readable local PDF when no Office renderer is available.

    This fallback preserves every text character and pagination, but it cannot
    reproduce floating graphics from a downloaded DOCX template. The caller
    surfaces that limitation explicitly instead of returning an HTTP 500 or a
    missing download.
    """

    try:
        import fitz  # type: ignore
    except ImportError:
        return PreviewResult(
            "pdf", "dependency_missing", None, None, None, ("缺少 PyMuPDF，无法生成 PDF",)
        )
    options = dict(layout or {})
    try:
        font_size = float(options.get("font_size_pt", 10.5))
        line_spacing = float(options.get("line_spacing", 1.0))
        margin_cm = float(options.get("margin_cm", 1.2))
    except (TypeError, ValueError):
        return PreviewResult("pdf", "invalid_layout", None, None, None, ("PDF 排版参数无效",))
    font_candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/Deng.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    )
    bold_candidates = (
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/Dengb.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    )
    font_path = next((path for path in font_candidates if path.is_file()), None)
    bold_path = next((path for path in bold_candidates if path.is_file()), font_path)
    if font_path is None:
        return PreviewResult(
            "pdf", "dependency_missing", None, None, None, ("未检测到可嵌入的中文字体",)
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page_width, page_height = fitz.paper_size("a4")
    margin = max(18.0, margin_cm * 72 / 2.54)
    right = page_width - margin
    bottom = page_height - margin
    page = document.new_page(width=page_width, height=page_height)
    y = margin
    body_font = fitz.Font(fontfile=str(font_path))
    heading_font = fitz.Font(fontfile=str(bold_path or font_path))

    def clean_line(value: str) -> tuple[str, bool, float]:
        stripped = value.rstrip()
        heading = bool(re.match(r"^#{1,6}\s+", stripped))
        stripped = re.sub(r"^#{1,6}\s+", "", stripped)
        stripped = stripped.replace("**", "")
        size = min(font_size + 2.0, 13.0) if heading else font_size
        return stripped, heading, size

    def wrap(value: str, font: Any, size: float, width: float) -> list[str]:
        if not value:
            return [""]
        rows: list[str] = []
        buffer = ""
        for character in value:
            candidate = buffer + character
            if buffer and font.text_length(candidate, fontsize=size) > width:
                rows.append(buffer)
                buffer = character
            else:
                buffer = candidate
        if buffer or not rows:
            rows.append(buffer)
        return rows

    try:
        for source_line in str(content or "").splitlines():
            value, heading, size = clean_line(source_line)
            font = heading_font if heading else body_font
            rows = wrap(value, font, size, right - margin)
            line_height = max(size * line_spacing * 1.25, size + 1.5)
            if not value:
                rows = [""]
                line_height = max(3.0, size * 0.45)
            for row in rows:
                if y + line_height > bottom:
                    page = document.new_page(width=page_width, height=page_height)
                    y = margin
                if row:
                    page.insert_text(
                        (margin, y + size),
                        row,
                        fontsize=size,
                        fontname="resume-heading" if heading else "resume-body",
                        fontfile=str(bold_path if heading and bold_path else font_path),
                        color=(0, 0, 0),
                    )
                y += line_height
        try:
            document.subset_fonts()
        except (AttributeError, RuntimeError, ValueError):
            pass
        document.save(target, garbage=4, deflate=True)
        page_count = document.page_count
    except (OSError, OverflowError, RuntimeError, TypeError, ValueError) as exc:
        target.unlink(missing_ok=True)
        return PreviewResult(
            "pdf", "conversion_failed", None, None, None, (f"基础 PDF 生成失败：{type(exc).__name__}",)
        )
    finally:
        document.close()
    return PreviewResult(
        "pdf",
        "ready",
        str(target),
        page_count,
        page_count == 1,
        ("Office 渲染器不可用，已生成保留全部文字的基础排版 PDF；DOCX 模板图形仅保留在 DOCX 文件中",),
    )


def office_status() -> dict[str, Any]:
    word_candidates = (
        shutil.which("WINWORD.EXE"),
        shutil.which("winword"),
        str(Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Microsoft Office/root/Office16/WINWORD.EXE"),
        str(Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft Office/root/Office16/WINWORD.EXE"),
    )
    return {
        "word": any(value and Path(value).is_file() for value in word_candidates),
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
    "apply_markdown_template",
    "compress_markdown",
    "convert_docx_to_pdf",
    "convert_pdf_with_libreoffice",
    "convert_pdf_with_word",
    "estimate_pages",
    "export_docx",
    "export_text_pdf",
    "office_status",
    "render_candidate_project",
    "render_markdown",
    "write_preview",
]
