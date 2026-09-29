"""Job input normalization and URL safety checks."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
import re

from app.core.schemas import JobInput


def validate_boss_url(url: str) -> str:
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("职位 URL 必须使用 http 或 https")
    host = (parsed.hostname or "").lower()
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("职位 URL 不得包含凭据或 fragment")
    if not (host == "zhipin.com" or host.endswith(".zhipin.com")):
        raise ValueError("只允许 BOSS 直聘官方网页 URL")
    return url


def parse_posted_time(value: str | None, *, now: datetime | None = None) -> tuple[datetime | None, str]:
    """Normalize common BOSS/boss-show-time labels without guessing unknown dates."""

    if not value or not str(value).strip():
        return None, "unknown"
    text = re.sub(r"^(?:发布于|发布时间|更新于|更新时间)\s*[:：]?\s*", "", str(value).strip())
    current = now or datetime.now(timezone.utc)
    if text in {"刚刚", "刚刚发布"}:
        return current, "known"
    if text.startswith("今天"):
        return current, "known"
    if text.startswith("昨天"):
        return current - timedelta(days=1), "known"
    if text.startswith("前天"):
        return current - timedelta(days=2), "known"
    match = re.search(r"(\d+)\s*分钟前", text)
    if match:
        return current - timedelta(minutes=int(match.group(1))), "known"
    match = re.search(r"(\d+)\s*小时前", text)
    if match:
        return current - timedelta(hours=int(match.group(1))), "known"
    match = re.search(r"(\d+)\s*天前", text)
    if match:
        return current - timedelta(days=int(match.group(1))), "known"
    match = re.search(r"(\d+)\s*周前", text)
    if match:
        return current - timedelta(weeks=int(match.group(1))), "known"
    match = re.search(r"(\d+)\s*个月前", text)
    if match:
        return current - timedelta(days=30 * int(match.group(1))), "known"
    match = re.search(r"(\d+)\s*年前", text)
    if match:
        return current - timedelta(days=365 * int(match.group(1))), "known"
    normalized = text.replace("年", "-").replace("月", "-").replace("日", "").replace("号", "")
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
    ):
        try:
            parsed = datetime.strptime(normalized, fmt).replace(tzinfo=current.tzinfo or timezone.utc)
            return parsed, "known"
        except ValueError:
            continue
    month_day = re.fullmatch(r"(\d{1,2})-(\d{1,2})(?:\s+(\d{1,2}):(\d{2}))?", normalized)
    if month_day:
        month, day = int(month_day.group(1)), int(month_day.group(2))
        hour, minute = int(month_day.group(3) or 0), int(month_day.group(4) or 0)
        try:
            parsed = current.replace(month=month, day=day, hour=hour, minute=minute, second=0, microsecond=0)
            if parsed > current + timedelta(days=1):
                parsed = parsed.replace(year=parsed.year - 1)
            return parsed, "known"
        except ValueError:
            return None, "unknown"
    return None, "unknown"


def parse_job_detail_text(
    title: str,
    text: str,
    *,
    company: str | None = None,
    city: str | None = None,
    source_url: str | None = None,
    posted_label: str | None = None,
    hr_activity: str | None = None,
) -> JobInput:
    """Parse a user-selected BOSS detail snapshot into the stable job schema."""

    job = job_from_text(title, text, company=company, city=city)
    posted_at, status = parse_posted_time(posted_label)
    return job.model_copy(
        update={
            "source_url": source_url,
            "posted_at": posted_at,
            "posted_at_label": posted_label if status == "unknown" else posted_label,
            "hr_activity": str(hr_activity).strip() if hr_activity else None,
        }
    )


_SECTION_HEADINGS = {
    "responsibilities": (
        "岗位职责",
        "工作职责",
        "职位职责",
        "职位描述",
        "工作内容",
        "岗位描述",
    ),
    "requirements": (
        "任职要求",
        "岗位要求",
        "职位要求",
        "任职资格",
        "岗位资格",
        "基本要求",
    ),
    "skills": ("技能要求", "技术要求", "技术栈", "专业技能", "加分项"),
}


def _section_line(line: str) -> tuple[str | None, str]:
    """Return a section name only when the line starts with a real heading."""

    normalized = re.sub(r"^[一二三四五六七八九十\d]+[、.．)）]\s*", "", line).strip()
    for section, headings in _SECTION_HEADINGS.items():
        for heading in headings:
            match = re.match(rf"^{re.escape(heading)}(?:\s*[:：]\s*(.*))?$", normalized)
            if match:
                return section, (match.group(1) or "").strip()
    return None, normalized


def _skill_terms(line: str) -> list[str]:
    """Extract conservative skill terms without turning an entire duty into one keyword."""

    value = re.sub(r"^(?:熟悉|掌握|精通|了解|具备|使用|优先考虑)\s*", "", line.strip())
    value = re.split(r"[；;。]", value, maxsplit=1)[0]
    parts = re.split(r"[、,，/]|\s+(?:及|和|与)\s+", value)
    terms: list[str] = []
    for part in parts:
        candidate = part.strip(" ：:()（）")
        candidate = re.sub(r"^(?:熟悉|掌握|精通|了解|具备|使用)\s*", "", candidate)
        if not candidate or len(candidate) > 80:
            continue
        if candidate not in terms:
            terms.append(candidate)
    return terms


def job_from_text(title: str, text: str, *, company: str | None = None, city: str | None = None) -> JobInput:
    lines = [line.strip(" -*•\t") for line in text.splitlines() if line.strip()]
    skills: list[str] = []
    responsibilities: list[str] = []
    requirements: list[str] = []
    mode = "responsibilities"
    for line in lines:
        section, remainder = _section_line(line)
        if section:
            mode = section
            if not remainder:
                continue
        line = remainder
        if mode == "skills":
            skills.extend(_skill_terms(line) or [line])
        elif mode == "requirements":
            requirements.append(line)
            if re.search(r"(?:熟悉|掌握|精通|了解|技术栈|技能)", line):
                skills.extend(_skill_terms(line))
        else:
            responsibilities.append(line)
    skills = list(dict.fromkeys(item for item in skills if item))
    return JobInput(
        title=title,
        company=company,
        city=city,
        responsibilities=responsibilities[:100],
        requirements=requirements[:100],
        skills=skills[:100],
        raw_text=text,
    )
