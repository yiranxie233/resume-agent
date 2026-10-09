"""Job input normalization and URL safety checks."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from app.core.schemas import JobInput


def infer_job_metadata(text: str) -> dict[str, str | None]:
    """Conservatively infer visible card metadata from corrected screenshot text."""

    lines = [re.sub(r"\s+", " ", line).strip(" -*•\t") for line in str(text or "").splitlines()]
    lines = [line for line in lines if line]
    section_names = {item for values in _SECTION_HEADINGS.values() for item in values}
    ignored_title_lines = {
        "boss直聘",
        "首页",
        "职位",
        "公司",
        "校园·海归",
        "app",
        "搜索",
        "登录/注册",
        "我要招聘",
        "我要找工作",
    }
    title = next(
        (
            re.split(r"\s+\d+(?:\.\d+)?\s*[-–—~至]", line, maxsplit=1)[0].strip()
            for line in lines[:12]
            if 2 <= len(line) <= 80
            and line.casefold() not in ignored_title_lines
            and line not in section_names
            and not re.search(r"(?:\d+[Kk]|元/|薪|经验|学历|本科|大专|招聘者|活跃)", line)
        ),
        None,
    )
    company = next(
        (
            line
            for line in lines[:20]
            if len(line) <= 100 and re.search(r"(?:公司|科技|集团|工作室|事务所|研究院)$", line)
        ),
        None,
    )
    salary_match = re.search(
        r"(?:\d+(?:\.\d+)?\s*[-–—~至]\s*\d+(?:\.\d+)?\s*[Kk万](?:[·・]\d+薪)?|"
        r"\d+\s*[-–—~至]\s*\d+\s*元/(?:天|时|小时|月))",
        str(text or ""),
    )
    activity_match = re.search(
        r"(刚刚活跃|当前在线|在线|今日活跃|今天活跃|昨日活跃|昨天活跃|"
        r"[一二两三四五六七八九十\d]+\s*(?:分钟|小时|天|日|周|个月|月)(?:内|前)?活跃|"
        r"半年(?:内|前)?活跃|本周活跃|本月活跃|近(?:一|两|二)?周活跃|近(?:一)?月活跃)",
        str(text or ""),
    )
    hr_name_match = re.search(
        r"(?:招聘者|招聘人|HR|hr)\s*[:：]?\s*([\u4e00-\u9fffA-Za-z·•]{1,30})",
        str(text or ""),
    )
    city_match = re.search(
        r"(?:工作地点|城市|地点)\s*[:：]?\s*([\u4e00-\u9fff]{2,12}(?:[·・][\u4e00-\u9fff]{1,12})?)",
        str(text or ""),
    )
    return {
        "title": title,
        "company": company,
        "city": city_match.group(1) if city_match else None,
        "salary": salary_match.group(0).replace(" ", "") if salary_match else None,
        "hr_name": hr_name_match.group(1) if hr_name_match else None,
        "hr_activity": activity_match.group(1) if activity_match else None,
    }


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
    hr_name: str | None = None,
    hr_activity: str | None = None,
) -> JobInput:
    """Parse a user-selected BOSS detail snapshot into the stable job schema."""

    job = job_from_text(title, sanitize_job_description_text(text), company=company, city=city)
    posted_at, _ = parse_posted_time(posted_label)
    return job.model_copy(
        update={
            "source_url": source_url,
            "posted_at": posted_at,
            "posted_at_label": posted_label,
            "hr_name": str(hr_name).strip() if hr_name else None,
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


_JOB_SECTION_HEADING_PATTERN = (
    r"(?:岗位职责|工作职责|职位职责|职位描述|工作内容|岗位描述|"
    r"任职要求|岗位要求|职位要求|任职资格|岗位资格|基本要求|"
    r"技能要求|技术要求|技术栈|专业技能|加分项)"
)


def sanitize_job_description_text(text: str) -> str:
    """Remove BOSS page chrome/watermark text without rewriting JD evidence.

    The detail DOM sometimes exposes the site brand as a hidden sibling between
    a section heading and its first numbered item.  Only standalone brand lines
    and that exact heading-boundary pattern are removed.  A genuine sentence
    mentioning the BOSS platform elsewhere remains untouched.
    """

    value = str(text or "").replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(
        rf"(?P<heading>{_JOB_SECTION_HEADING_PATTERN})\s*BOSS\s*直聘(?:\s*真)?\s*"
        r"(?=(?:\d+[.、．)）]|[-*•·]))",
        r"\g<heading>\n",
        value,
        flags=re.IGNORECASE,
    )
    cleaned_lines: list[str] = []
    for line in value.splitlines():
        compact = re.sub(r"\s+", "", line).casefold()
        if compact in {"boss直聘", "boss直聘真"}:
            continue
        cleaned_lines.append(line.rstrip())
    return "\n".join(cleaned_lines).strip()


_ALTERNATIVE_SKILL_PATTERN = re.compile(
    r"(?:任选(?:其)?一|任[一意](?:一项|一种|其一)?|至少(?:掌握|熟悉|会|具备)?(?:其中)?(?:一项|一种)|"
    r"其中(?:一项|一种|之一)|[二三四五六七八九十\d]+选一|(?:一项|一种)即可|之一)",
    flags=re.IGNORECASE,
)


def is_alternative_skill_requirement(value: str) -> bool:
    """Return whether one JD phrase describes alternatives, not cumulative skills."""

    return bool(_ALTERNATIVE_SKILL_PATTERN.search(str(value or "")))


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
    # An alternative list is one requirement group. Splitting it would make
    # every option look mandatory in the UI, scoring and generation prompt.
    if is_alternative_skill_requirement(value):
        return [value.strip(" ：:()（）")]
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
    text = sanitize_job_description_text(text)
    lines = [
        re.sub(r"^\s*(?:[-—–*•·]|[lI]\s+)", "", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]
    lines = list(dict.fromkeys(line for line in lines if line))
    skills: list[str] = []
    responsibilities: list[str] = []
    requirements: list[str] = []
    has_section_heading = any(_section_line(line)[0] is not None for line in lines)
    mode: str | None = None if has_section_heading else "responsibilities"
    for line in lines:
        section, remainder = _section_line(line)
        if section:
            mode = section
            if not remainder:
                continue
        line = remainder
        if mode is None:
            # Search-page screenshots contain navigation, salary and company
            # text before the first JD heading. Metadata inference handles
            # those values; they must not become fake responsibilities.
            continue
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
