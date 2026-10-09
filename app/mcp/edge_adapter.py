"""Visible Edge/Playwright adapter used only by explicit BOSS MCP calls.

The browser is launched lazily with an isolated persistent profile.  This
module never attempts to solve login challenges, CAPTCHAs, sliders, or access
restrictions; those states are returned to the caller as manual gates.
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from app.services.job_parser import parse_job_detail_text, parse_posted_time, validate_boss_url


# BOSS uses numeric city identifiers in the ``city`` query parameter.  Keep a
# deliberately small, reviewed list for common destinations and require an
# explicit code for every other city.  Passing a Chinese name as ``city`` can
# silently fall back to an unrelated location, which is worse than asking the
# user for a code.
COMMON_CITY_CODES: dict[str, str] = {
    "全国": "100010000",
    "北京": "101010100",
    "上海": "101020100",
    "天津": "101030100",
    "重庆": "101040100",
    "广州": "101280100",
    "深圳": "101280600",
    "杭州": "101210100",
    "南京": "101190100",
    "苏州": "101190400",
    "成都": "101270100",
    "武汉": "101200100",
    "西安": "101110100",
    "长沙": "101250100",
    "郑州": "101180100",
    "青岛": "101120200",
    "济南": "101120100",
    "合肥": "101220100",
    "厦门": "101230200",
    "福州": "101230100",
    "宁波": "101210400",
    "无锡": "101190200",
    "佛山": "101280800",
    "东莞": "101281600",
    "珠海": "101280700",
}

BOSS_HOME_URL = "https://www.zhipin.com/"
_MANUAL_BROWSER_CODES = {
    "boss_login_required",
    "boss_verification_required",
    "boss_search_session_expired",
    "edge_login_attach_failed",
}


class EdgeActionRequired(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def normalize_city_code(city: str, city_code: str | None = None) -> tuple[str, str]:
    """Resolve a reviewed BOSS city code without treating a name as a code."""

    display_city = str(city or "").strip()
    supplied = str(city_code or "").strip()
    if supplied:
        if not re.fullmatch(r"\d{9}", supplied):
            raise ValueError("city_code 必须是 9 位数字 BOSS 城市代码")
        return supplied, "explicit"
    normalized = display_city.removesuffix("市")
    resolved = COMMON_CITY_CODES.get(normalized)
    if resolved:
        return resolved, "builtin"
    raise EdgeActionRequired(
        "boss_city_code_required",
        f"暂未内置“{display_city}”的 BOSS 城市代码，请提供显式 city_code 后重试",
    )


def login_required(url: str, *, visible_login_control: bool) -> bool:
    """Infer login state from a login route or a specific visible control."""

    parsed = urllib.parse.urlparse(str(url or ""))
    path = parsed.path.rstrip("/").lower()
    query = urllib.parse.parse_qs(parsed.query)
    login_route = path in {"/login", "/web/user", "/web/user/login"}
    login_query = any("login" in value.lower() for value in query.get("ka", []))
    return bool(login_route or login_query or visible_login_control)


def same_job_search_page(current_url: str, target_url: str) -> bool:
    """Compare a loaded BOSS search page by its stable search parameters.

    BOSS may update the ``page`` query parameter while infinite scrolling.  A
    page-number change is still the same retained search tab and must never
    cause the browser session to be discarded or relaunched.
    """

    current = urllib.parse.urlparse(str(current_url or ""))
    target = urllib.parse.urlparse(str(target_url or ""))
    if current.netloc.lower() != "www.zhipin.com":
        return False
    if current.path.rstrip("/") != "/web/geek/jobs":
        return False
    current_query = urllib.parse.parse_qs(current.query)
    target_query = urllib.parse.parse_qs(target.query)
    for key in ("query", "city"):
        if current_query.get(key, [""])[0] != target_query.get(key, [""])[0]:
            return False
    return True


def blank_browser_page(url: str) -> bool:
    """Return whether Edge is still showing an empty startup/new-tab page."""

    value = str(url or "").strip().lower().rstrip("/")
    return value in {"", "about:blank", "edge://newtab", "edge://new-tab-page"}


_BOSS_DIGIT_TRANSLATION = str.maketrans(
    {chr(0xE031 + digit): str(digit) for digit in range(10)}
)


def decode_boss_obfuscated_text(value: Any) -> str:
    """Decode digit glyphs rendered by BOSS's ``kanzhun-mix`` webfont."""

    return str(value or "").translate(_BOSS_DIGIT_TRANSLATION).strip()


_HR_ACTIVITY_RE = re.compile(
    r"(刚刚活跃|当前在线|在线|今日活跃|今天活跃|昨日活跃|昨天活跃|"
    r"[一二两三四五六七八九十\d]+\s*(?:分钟|小时|天|日|周|个月|月)(?:内|前)?活跃|"
    r"半年(?:内|前)?活跃|本周活跃|本月活跃|近(?:一|两|二)?周活跃|"
    r"近(?:一)?月活跃|月内活跃|较少活跃)"
)


def split_recruiter_card_text(
    hr_name: Any, hr_activity: Any = None
) -> tuple[str | None, str]:
    """Split BOSS's authoritative card label into name and activity columns."""

    combined = re.sub(r"\s+", " ", str(hr_name or "")).strip()
    explicit = re.sub(r"\s+", " ", str(hr_activity or "")).strip()
    match = _HR_ACTIVITY_RE.search(combined)
    if match:
        explicit = match.group(1)
        combined = f"{combined[:match.start()]} {combined[match.end():]}"
        combined = re.sub(r"[|｜·•,，:：\s]+", " ", combined).strip()
    honorific = re.search(r"([\u4e00-\u9fffA-Za-z·]{1,20}(?:先生|女士))", combined)
    if honorific:
        combined = honorific.group(1)
    else:
        combined = re.sub(
            r"(?:招聘者|招聘经理|招聘专员|人事经理|人事专员|人事|HRBP|HR|猎头)\s*$",
            "",
            combined,
            flags=re.IGNORECASE,
        ).strip(" |｜·•,，:：")
    return combined or None, explicit or "活跃时间待解析"


class EdgeBrowserPort(Protocol):
    @property
    def running(self) -> bool: ...

    @property
    def manual_login_running(self) -> bool: ...

    def open(self, url: str) -> None: ...

    def open_auxiliary(self, url: str) -> None: ...

    def open_user_tab(self, url: str) -> None: ...

    def close_auxiliary(self) -> None: ...

    def select_search_page(self, target_url: str) -> bool: ...

    def restore_search_page(
        self, target_url: str, anchor_url: str | None = None
    ) -> bool: ...

    def open_manual_login(self, url: str) -> None: ...

    def adopt_manual_login(self) -> None: ...

    def detach_manual_login(self) -> None: ...

    def close_manual_login(self) -> None: ...

    def show_for_user(self) -> None: ...

    def minimize_collection_window(self) -> None: ...

    def page_state(self) -> dict[str, Any]: ...

    def search_items(self) -> list[dict[str, Any]]: ...

    def load_more_items(
        self, seen_urls: set[str], *, minimum_new_items: int = 1
    ) -> dict[str, Any]: ...

    def enrich_search_activities(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    def detail_fields(self) -> dict[str, Any]: ...

    def sanitized_html(self) -> str: ...

    def clear_session(self) -> None: ...

    def close(self) -> None: ...


def detect_edge_path(explicit: str | None = None) -> str | None:
    candidates = [explicit, shutil.which("msedge"), shutil.which("msedge.exe")]
    program_files = [os.getenv("PROGRAMFILES"), os.getenv("PROGRAMFILES(X86)"), os.getenv("LOCALAPPDATA")]
    for root in program_files:
        if root:
            candidates.append(str(Path(root) / "Microsoft" / "Edge" / "Application" / "msedge.exe"))
    for value in candidates:
        if value and Path(value).is_file():
            return str(Path(value).resolve())
    return None


class _DirectCDPController:
    """Small synchronous CDP client for an already visible Edge process.

    Recent Edge builds on some Windows systems reset Playwright's Node-based
    CDP handshake even though the same loopback WebSocket works from Python.
    This adapter covers only the browser primitives used by the BOSS MCP and
    leaves login/CAPTCHA work entirely in the visible browser.
    """

    def __init__(self, endpoint: str, *, timeout_ms: int) -> None:
        try:
            from websocket import create_connection  # type: ignore
        except ImportError as exc:
            raise EdgeActionRequired(
                "playwright_not_installed",
                "缺少 Edge 接管依赖，请重新安装项目 mcp 依赖",
            ) from exc
        self.timeout_seconds = max(timeout_ms / 1000, 0.5)
        self._connection = create_connection(
            endpoint,
            origin="http://localhost",
            timeout=self.timeout_seconds,
            enable_multithread=True,
            http_proxy_host=None,
            http_proxy_port=None,
        )
        self._lock = threading.RLock()
        self._next_id = 0
        self._closed = False
        self._pages: dict[str, _DirectCDPPage] = {}
        self._context = _DirectCDPContext(self)

    @property
    def contexts(self) -> list["_DirectCDPContext"]:
        return [] if self._closed else [self._context]

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Edge CDP connection is closed")
            self._next_id += 1
            request_id = self._next_id
            payload: dict[str, Any] = {
                "id": request_id,
                "method": method,
                "params": params or {},
            }
            if session_id:
                payload["sessionId"] = session_id
            self._connection.send(json.dumps(payload, ensure_ascii=False))
            deadline = time.monotonic() + (timeout or self.timeout_seconds)
            while time.monotonic() < deadline:
                remaining = max(0.05, deadline - time.monotonic())
                self._connection.settimeout(remaining)
                message = self._connection.recv()
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                response = json.loads(message)
                if response.get("id") != request_id:
                    continue
                if response.get("error"):
                    error = response["error"]
                    raise RuntimeError(str(error.get("message") or error))
                result = response.get("result")
                return result if isinstance(result, dict) else {}
            raise TimeoutError(f"Edge CDP command timed out: {method}")

    def target_infos(self) -> list[dict[str, Any]]:
        values = self.call("Target.getTargets").get("targetInfos") or []
        return [value for value in values if isinstance(value, dict)]

    def pages(self) -> list["_DirectCDPPage"]:
        result: list[_DirectCDPPage] = []
        active_ids: set[str] = set()
        for info in self.target_infos():
            if info.get("type") != "page":
                continue
            target_id = str(info.get("targetId") or "")
            if not target_id:
                continue
            active_ids.add(target_id)
            page = self._pages.get(target_id)
            if page is None:
                page = _DirectCDPPage(self, target_id, str(info.get("url") or ""))
                self._pages[target_id] = page
            else:
                page._url = str(info.get("url") or page._url)
            result.append(page)
        for target_id in set(self._pages) - active_ids:
            self._pages[target_id]._closed = True
        return result

    def new_page(self, *, background: bool = False) -> "_DirectCDPPage":
        params: dict[str, Any] = {"url": "about:blank"}
        if background:
            # ``background`` is a Chromium Target-domain option.  Creating the
            # temporary detail target this way prevents a programmatic JD
            # parse from selecting a new Edge tab or stealing focus from the
            # Streamlit page.  Explicit user-view actions still call
            # ``bring_to_front`` below.
            params["background"] = True
        try:
            result = self.call("Target.createTarget", params)
        except RuntimeError:
            # Older Chromium builds may not understand ``background``.  Keep
            # compatibility while the owning window remains minimised.
            if not background:
                raise
            result = self.call("Target.createTarget", {"url": "about:blank"})
        target_id = str(result.get("targetId") or "")
        if not target_id:
            raise RuntimeError("Edge did not create a new tab")
        page = _DirectCDPPage(self, target_id, "about:blank")
        self._pages[target_id] = page
        return page

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._connection.close()
        except Exception:
            pass

    def close(self) -> None:
        if not self._closed:
            try:
                self.call("Browser.close", timeout=2)
            except Exception:
                pass
        self.stop()


class _DirectCDPContext:
    def __init__(self, controller: _DirectCDPController) -> None:
        self._controller = controller
        self.browser = controller

    @property
    def pages(self) -> list["_DirectCDPPage"]:
        return self._controller.pages()

    def new_page(self) -> "_DirectCDPPage":
        return self._controller.new_page()

    def new_background_page(self) -> "_DirectCDPPage":
        return self._controller.new_page(background=True)

    def clear_cookies(self) -> None:
        pages = self.pages
        if not pages:
            return
        pages[0]._call("Network.clearBrowserCookies")

    def close(self) -> None:
        self._controller.close()


class _DirectCDPPage:
    def __init__(self, controller: _DirectCDPController, target_id: str, url: str) -> None:
        self._controller = controller
        self._target_id = target_id
        self._url = url
        self._session_id: str | None = None
        self._closed = False
        self._timeout_ms = 20_000

    def _session(self) -> str:
        if self._session_id is None:
            attached = self._controller.call(
                "Target.attachToTarget",
                {"targetId": self._target_id, "flatten": True},
            )
            self._session_id = str(attached.get("sessionId") or "")
            if not self._session_id:
                raise RuntimeError("Edge did not attach to the selected tab")
        return self._session_id

    def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._controller.call(
            method,
            params,
            session_id=self._session(),
            timeout=max(self._timeout_ms / 1000, 0.5),
        )

    @property
    def url(self) -> str:
        if self._closed:
            return self._url
        try:
            self._url = str(self.evaluate("() => location.href") or self._url)
        except Exception:
            pass
        return self._url

    def set_default_timeout(self, timeout_ms: int) -> None:
        self._timeout_ms = timeout_ms

    def is_closed(self) -> bool:
        if self._closed:
            return True
        try:
            return not any(
                str(info.get("targetId") or "") == self._target_id
                for info in self._controller.target_infos()
            )
        except Exception:
            return True

    def evaluate(self, script: str) -> Any:
        expression = script.strip()
        if expression.startswith("(") or "=>" in expression:
            expression = f"({expression})()"
        result = self._call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": True,
            },
        ).get("result") or {}
        if result.get("subtype") == "error":
            raise RuntimeError(str(result.get("description") or "JavaScript evaluation failed"))
        return result.get("value")

    def goto(self, url: str, **options: Any) -> None:
        self._call("Page.enable")
        result = self._call("Page.navigate", {"url": url})
        if result.get("errorText"):
            raise RuntimeError(str(result["errorText"]))
        self._url = url
        # A committed navigation is enough for callers that immediately poll
        # a specific asynchronously rendered DOM field. BOSS detail pages can
        # keep subresources pending for a long time, so waiting for the whole
        # document here would serialize into an MCP timeout.
        if str(options.get("wait_until") or "").lower() == "commit":
            return
        deadline = time.monotonic() + max(self._timeout_ms / 1000, 0.5)
        while time.monotonic() < deadline:
            try:
                ready = self.evaluate("() => document.readyState")
                self._url = str(self.evaluate("() => location.href") or url)
                if ready in {"interactive", "complete"}:
                    return
            except Exception:
                pass
            time.sleep(0.1)
        raise TimeoutError(f"Edge page navigation timed out: {url}")

    def bring_to_front(self) -> None:
        self._call("Page.bringToFront")

    def dispatch_wheel(self, delta_y: float) -> None:
        """Send trusted wheel input to this target without activating it."""

        viewport = self.evaluate(
            """
            () => {
              const roots = Array.from(document.querySelectorAll(
                '.job-list-container, .rec-job-list, .job-list-box, [class*="job-list"]'
              ));
              const root = roots.find((node) => {
                const rect = node.getBoundingClientRect();
                const style = getComputedStyle(node);
                return rect.width > 0 && rect.height > 0 &&
                  style.display !== 'none' && style.visibility !== 'hidden';
              });
              const rect = root?.getBoundingClientRect();
              const width = window.innerWidth || 1024;
              const height = window.innerHeight || 768;
              return {
                // A wheel event scrolls the element beneath the pointer.  BOSS
                // uses a left-hand virtual list beside its detail pane, so the
                // generic viewport centre can target the wrong scroller.
                x: Math.min(width - 1, Math.max(1, Math.floor(
                  rect ? rect.left + rect.width / 2 : width / 2
                ))),
                y: Math.min(height - 1, Math.max(1, Math.floor(rect ? Math.min(
                  rect.bottom - 8, rect.top + rect.height * 0.75
                ) : height / 2)))
              };
            }
            """
        )
        point = viewport if isinstance(viewport, dict) else {}
        self._call(
            "Input.dispatchMouseEvent",
            {
                "type": "mouseWheel",
                "x": float(point.get("x") or 512),
                "y": float(point.get("y") or 384),
                "deltaX": 0,
                "deltaY": float(delta_y),
                "pointerType": "mouse",
            },
        )

    def emulate_background_active(self) -> None:
        """Keep a background search target eligible for lazy loading.

        BOSS pauses its virtual-list loader when Chromium reports the search
        tab as hidden. CDP focus emulation changes the lifecycle for only this
        target and does not call ``Page.bringToFront``.
        """

        self._call("Emulation.setFocusEmulationEnabled", {"enabled": True})
        try:
            self._call("Page.setWebLifecycleState", {"state": "active"})
        except Exception:
            # Optional on some Edge builds; focus emulation is sufficient.
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._controller.call("Target.closeTarget", {"targetId": self._target_id})
        self._closed = True


class _DetachedEdgeProcess:
    """Small Popen-compatible handle for Edge launched by Start-Process."""

    def __init__(self, pid: int, identity: tuple[str, int] | None) -> None:
        self.pid = pid
        self._identity = identity
        self._started_at = time.monotonic()

    def poll(self) -> int | None:
        current = PlaywrightEdgeBrowser._process_identity(self.pid)
        if current is not None and current == self._identity:
            return None
        # Process identity queries may be temporarily denied while Edge moves
        # from its launcher to the browser root. Do not declare a freshly
        # launched interactive window dead during that settling interval.
        if time.monotonic() - self._started_at < 20.0:
            return None
        return 0

    def wait(self, *, timeout: int) -> int:
        deadline = time.monotonic() + timeout
        while self.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.poll() is None:
            raise subprocess.TimeoutExpired("msedge.exe", timeout)
        return 0

    def kill(self) -> None:
        PlaywrightEdgeBrowser._stop_owned_process_tree(self.pid)


class PlaywrightEdgeBrowser:
    """Thin synchronous Playwright wrapper with lazy visible Edge startup."""

    def __init__(self, profile_dir: Path, *, edge_path: str | None = None, timeout_ms: int = 20_000) -> None:
        self.profile_dir = profile_dir.resolve()
        self.edge_path = detect_edge_path(edge_path)
        self.timeout_ms = timeout_ms
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        # The BOSS search tab must survive user-opened detail tabs and CDP
        # detach/reattach cycles. Direct CDP target ids remain stable for the
        # lifetime of the Edge tab even though Python page wrappers do not.
        self._search_target_id: str | None = None
        self._auxiliary_parent_page: Any = None
        self._manual_login_process: subprocess.Popen[bytes] | None = None
        self._manual_debug_port: int | None = None
        self._manual_debug_endpoint: str | None = None

    @property
    def _devtools_port_file(self) -> Path:
        return self.profile_dir / "DevToolsActivePort"

    @property
    def _debug_port_record_file(self) -> Path:
        """Application-owned port record for fixed-port CDP sessions."""

        return self.profile_dir / "ResumeAgentDebugPort"

    @property
    def _owned_pid_record_file(self) -> Path:
        """PID of the Edge root process launched for this isolated profile."""

        return self.profile_dir / "ResumeAgentEdgePid"

    @staticmethod
    def _process_identity(pid: int) -> tuple[str, int] | None:
        """Return executable path and creation ticks for a Windows process."""

        if os.name != "nt" or not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return None
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.QueryFullProcessImageNameW.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.LPWSTR,
                ctypes.POINTER(wintypes.DWORD),
            ]
            kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
            kernel32.GetProcessTimes.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
            ]
            kernel32.GetProcessTimes.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return None
            try:
                size = wintypes.DWORD(32768)
                buffer = ctypes.create_unicode_buffer(size.value)
                if not kernel32.QueryFullProcessImageNameW(
                    handle, 0, buffer, ctypes.byref(size)
                ):
                    return None
                creation = wintypes.FILETIME()
                exit_time = wintypes.FILETIME()
                kernel_time = wintypes.FILETIME()
                user_time = wintypes.FILETIME()
                if not kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel_time),
                    ctypes.byref(user_time),
                ):
                    return None
                ticks = (int(creation.dwHighDateTime) << 32) | int(
                    creation.dwLowDateTime
                )
                return str(Path(buffer.value).resolve()), ticks
            finally:
                kernel32.CloseHandle(handle)
        except (OSError, TypeError, ValueError):
            return None

    def _write_owned_process_record(self, pid: int) -> None:
        identity = self._process_identity(pid)
        if identity is None:
            return
        path, creation_ticks = identity
        self._owned_pid_record_file.write_text(
            json.dumps(
                {
                    "pid": pid,
                    "path": path,
                    "creation_ticks": creation_ticks,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _recorded_owned_pid(self) -> int | None:
        try:
            record = json.loads(
                self._owned_pid_record_file.read_text(encoding="utf-8")
            )
            pid = int(record["pid"])
            recorded_path = str(Path(str(record["path"])).resolve()).casefold()
            recorded_ticks = int(record["creation_ticks"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        identity = self._process_identity(pid)
        if identity is None:
            return None
        current_path, current_ticks = identity
        if (
            pid <= 0
            or pid == os.getpid()
            or current_path.casefold() != recorded_path
            or current_ticks != recorded_ticks
            or Path(current_path).name.casefold() != "msedge.exe"
        ):
            return None
        return pid

    @staticmethod
    def _stop_owned_process_tree(pid: int) -> None:
        """Stop only the exact app-recorded Edge process tree."""

        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.run(
                ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
                check=False,
                creationflags=creation_flags,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    @staticmethod
    def _discard_runtime_file(path: Path) -> None:
        """Best-effort stale CDP cleanup; environment checks stay read-only."""

        try:
            path.unlink(missing_ok=True)
        except OSError:
            # A read-only explicit profile is reported when Edge is launched;
            # merely checking whether Edge is installed must not crash.
            pass

    @staticmethod
    def _debug_endpoint_for_port(port: int, *, timeout: float = 1.0) -> str | None:
        """Resolve a ready loopback CDP WebSocket endpoint for one port."""

        if not (0 < int(port) < 65536):
            return None
        connection: http.client.HTTPConnection | None = None
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
            connection.request("GET", "/json/version")
            response = connection.getresponse()
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
            endpoint = str(payload.get("webSocketDebuggerUrl") or "").strip()
            parsed = urllib.parse.urlparse(endpoint)
            if (
                parsed.scheme != "ws"
                or parsed.hostname not in {"127.0.0.1", "localhost"}
                or parsed.port != port
                or not parsed.path.startswith("/devtools/browser/")
            ):
                return None
            # Always connect through the explicitly loopback-bound address.
            return urllib.parse.urlunparse(
                ("ws", f"127.0.0.1:{port}", parsed.path, "", parsed.query, "")
            )
        except (OSError, http.client.HTTPException, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        finally:
            if connection is not None:
                connection.close()

    def _discover_debug_endpoint(self) -> str | None:
        candidate_ports: list[int] = []
        if self._manual_debug_port is not None:
            candidate_ports.append(self._manual_debug_port)
        for path in (self._debug_port_record_file, self._devtools_port_file):
            try:
                port = int(path.read_text(encoding="utf-8").splitlines()[0].strip())
            except (OSError, IndexError, TypeError, ValueError):
                continue
            if port not in candidate_ports:
                candidate_ports.append(port)
        for port in candidate_ports:
            endpoint = self._debug_endpoint_for_port(port)
            if endpoint is not None:
                self._manual_debug_port = port
                return endpoint
        return None

    @property
    def running(self) -> bool:
        if self._context is None or self._page is None:
            return False
        try:
            return not bool(self._page.is_closed())
        except Exception:
            return False

    @property
    def manual_login_running(self) -> bool:
        # Never trust a cached WebSocket URL or a still-present Popen object by
        # itself.  Edge may have closed while the MCP process stayed alive;
        # treating that stale state as running prevents the next search from
        # launching a replacement window.
        endpoint = self._discover_debug_endpoint()
        if endpoint is not None:
            self._manual_debug_endpoint = endpoint
            return True
        self._manual_debug_endpoint = None
        self._manual_debug_port = None
        self._discard_runtime_file(self._debug_port_record_file)
        self._discard_runtime_file(self._devtools_port_file)
        return False

    @staticmethod
    def _available_debug_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])

    def _launch_visible_edge(self, arguments: list[str]) -> Any:
        """Launch the collection Edge minimized outside the MCP child job.

        On managed Windows sessions, launching Edge directly from a nested
        Python stdio child can terminate its browser root with 0x80000003 a
        moment after the debugging port is published. PowerShell's native
        Start-Process creates the same normal interactive process users get
        from the desktop and returns its exact PID for safe ownership checks.
        """

        if os.name == "nt":
            powershell = shutil.which("pwsh.exe") or shutil.which("powershell.exe")
            if powershell:
                quote = lambda value: "'" + str(value).replace("'", "''") + "'"
                values = ",".join(quote(value) for value in arguments)
                stdout_path = self.profile_dir / "edge-launch.stdout.log"
                stderr_path = self.profile_dir / "edge-launch.stderr.log"
                pid_path = self.profile_dir / "ResumeAgentLaunchPid.tmp"
                self._discard_runtime_file(pid_path)
                script = (
                    f"$p=Start-Process -FilePath {quote(self.edge_path or '')} "
                    f"-ArgumentList @({values}) "
                    f"-RedirectStandardOutput {quote(stdout_path)} "
                    f"-RedirectStandardError {quote(stderr_path)} "
                    f"-WindowStyle Minimized -PassThru; "
                    f"[System.IO.File]::WriteAllText({quote(pid_path)}, $p.Id.ToString())"
                )
                encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
                launcher = subprocess.Popen(
                    [
                        powershell,
                        "-NoLogo",
                        "-NoProfile",
                        "-NonInteractive",
                        "-EncodedCommand",
                        encoded,
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                deadline = time.monotonic() + 15.0
                pid_text = ""
                while time.monotonic() < deadline:
                    try:
                        pid_text = pid_path.read_text(encoding="utf-8").strip()
                    except OSError:
                        pid_text = ""
                    if pid_text.isdigit() or launcher.poll() is not None:
                        break
                    time.sleep(0.05)
                self._discard_runtime_file(pid_path)
                if launcher.poll() is None:
                    launcher.terminate()
                    try:
                        launcher.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        launcher.kill()
                if not pid_text.isdigit():
                    raise OSError("PowerShell 未返回岗位采集 Edge 的进程号")
                pid = int(pid_text)
                deadline = time.monotonic() + 2.0
                identity = self._process_identity(pid)
                while identity is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                    identity = self._process_identity(pid)
                return _DetachedEdgeProcess(pid, identity)
        return subprocess.Popen(
            [self.edge_path or "", *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

    def open_manual_login(self, url: str) -> None:
        if self.manual_login_running:
            return
        # A previous MCP process may have timed out while its visible Edge was
        # still alive. Only an app-owned PID recorded beside this isolated
        # profile may be terminated; the user's normal Edge is never touched.
        stale_pid = self._recorded_owned_pid()
        if stale_pid is not None:
            self._stop_owned_process_tree(stale_pid)
            self._discard_runtime_file(self._owned_pid_record_file)
        # A launcher can remain alive even after its CDP endpoint failed.  Stop
        # that exact process tree before reusing the isolated profile.
        if self._manual_login_process is not None and self._manual_login_process.poll() is None:
            self.close_manual_login()
        else:
            self._manual_login_process = None
        if not self.edge_path:
            raise EdgeActionRequired("edge_not_found", "未检测到 Microsoft Edge，请在设置中指定 msedge.exe 路径")
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        # Chromium exposes navigator.webdriver when remote-debugging-port=0.
        # BOSS treats that as automation and can enter a one-second security
        # redirect loop.  Reserve an explicit loopback port instead; the page
        # then behaves like a normal user-controlled Edge tab while the local
        # MCP can still attach on demand.
        debug_port = self._available_debug_port()
        self._discard_runtime_file(self._devtools_port_file)
        self._discard_runtime_file(self._debug_port_record_file)
        try:
            self._manual_login_process = self._launch_visible_edge(
                [
                    f"--user-data-dir={self.profile_dir}",
                    "--remote-debugging-address=127.0.0.1",
                    f"--remote-debugging-port={debug_port}",
                    "--remote-allow-origins=http://localhost",
                    "--new-window",
                    "--start-minimized",
                    "--no-first-run",
                    "--disable-background-mode",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--no-sandbox",
                    "--disable-features=msEdgeStartupBoost",
                    url,
                ]
            )
            self._write_owned_process_record(self._manual_login_process.pid)
            self._manual_debug_port = debug_port
            deadline = time.monotonic() + min(max(self.timeout_ms / 1000, 8.0), 20.0)
            while time.monotonic() < deadline:
                endpoint = self._debug_endpoint_for_port(debug_port)
                if endpoint is not None:
                    time.sleep(0.25)
                    confirmed = self._debug_endpoint_for_port(
                        debug_port, timeout=1.5
                    )
                    if confirmed is not None:
                        self._manual_debug_endpoint = confirmed
                        self._debug_port_record_file.write_text(
                            str(debug_port), encoding="utf-8"
                        )
                        # Edge may briefly restore a just-created window even
                        # when Start-Process and Chromium both requested a
                        # minimized launch. Minimize again as soon as the owned
                        # CDP process is ready; the login gate explicitly
                        # restores it only when human action is actually needed.
                        self.minimize_collection_window()
                        return
                if self._manual_login_process.poll() is not None:
                    break
                time.sleep(0.1)
            raise EdgeActionRequired(
                "edge_login_attach_failed",
                "Edge 已启动但调试端口尚未就绪，请保持窗口打开并重新搜索",
            )
        except EdgeActionRequired:
            self._manual_debug_port = None
            self._manual_debug_endpoint = None
            raise
        except OSError as exc:
            self._manual_login_process = None
            self._manual_debug_port = None
            self._manual_debug_endpoint = None
            raise EdgeActionRequired("edge_launch_failed", "无法启动 BOSS 登录窗口") from exc

    def _set_owned_window_state(
        self,
        command: int,
        *,
        foreground: bool = False,
        send_to_back: bool = False,
    ) -> None:
        """Best-effort window state change for this app-owned Edge only."""

        if os.name != "nt":
            return
        pid = self._recorded_owned_pid()
        if pid is None and self._manual_login_process is not None:
            pid = self._manual_login_process.pid
        if not pid:
            return
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)
            callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
            handles: list[int] = []

            @callback_type
            def collect(hwnd: int, _: int) -> bool:
                process_id = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
                if int(process_id.value) == int(pid) and user32.IsWindowVisible(hwnd):
                    handles.append(int(hwnd))
                return True

            user32.EnumWindows(collect, 0)
            for hwnd in handles:
                user32.ShowWindow(hwnd, command)
                if send_to_back:
                    # Keep the renderer active behind the user's web app. A
                    # truly minimised Chromium window can postpone BOSS's
                    # IntersectionObserver/infinite-list work until clicked.
                    user32.SetWindowPos(hwnd, 1, 0, 0, 0, 0, 0x53)
                if foreground:
                    user32.SetForegroundWindow(hwnd)
        except (OSError, TypeError, ValueError):
            return

    def show_for_user(self) -> None:
        # SW_RESTORE. Used only for an explicit login/view action.
        self._set_owned_window_state(9, foreground=True)

    def minimize_collection_window(self) -> None:
        # SW_SHOWNOACTIVATE + HWND_BOTTOM: do not steal focus, but keep the page
        # rendered so lazy loading also works beyond the first two UI pages.
        self._set_owned_window_state(4, send_to_back=True)

    def _connect_cdp(self, endpoint: str) -> _DirectCDPController:
        return _DirectCDPController(endpoint, timeout_ms=self.timeout_ms)

    def adopt_manual_login(self) -> None:
        """Attach a local Python CDP client to the visible Edge window."""

        if self.running:
            return
        # Immediately after this instance launches Edge, use the endpoint just
        # returned by /json/version. A second HTTP probe during Edge's startup
        # can race its debugging server on recent Windows builds. For an Edge
        # inherited from an earlier action/process, re-resolve it normally.
        recently_launched = bool(
            self._manual_login_process is not None
            and self._manual_login_process.poll() is None
            and self._manual_debug_endpoint
        )
        endpoint = (
            self._manual_debug_endpoint
            if recently_launched
            else self._discover_debug_endpoint()
        )
        if endpoint is None and self._manual_debug_port is not None:
            # Edge can briefly stop answering /json/version while the first
            # BOSS document initializes. Retry the already bound loopback port
            # instead of treating one transient probe as a closed window.
            deadline = time.monotonic() + 2.0
            while endpoint is None and time.monotonic() < deadline:
                time.sleep(0.1)
                endpoint = self._debug_endpoint_for_port(
                    self._manual_debug_port, timeout=0.5
                )
        if endpoint is None:
            self._manual_debug_endpoint = None
            raise EdgeActionRequired(
                "edge_login_attach_failed",
                "登录窗口尚未准备好，请保持 Edge 打开并重新搜索",
            )
        controller: _DirectCDPController | None = None
        try:
            controller = self._connect_cdp(endpoint)
            browser = controller
            contexts = list(browser.contexts)
            if not contexts:
                raise RuntimeError("Edge 调试会话中没有可用的浏览器上下文")
            context = contexts[0]
            pages = [page for page in context.pages if not page.is_closed()]
            if not pages:
                pages = [context.new_page()]
            retained = next(
                (
                    candidate
                    for candidate in pages
                    if self._search_target_id
                    and str(getattr(candidate, "_target_id", "")) == self._search_target_id
                ),
                None,
            )
            # ``Target.getTargets`` has already refreshed ``_url`` for direct
            # CDP pages.  Do not attach to/evaluate an unrelated foreground
            # tab merely to inspect its URL: Edge new-tab/extension pages can
            # stall that evaluation and previously prevented the intact BOSS
            # search tab from being recovered for pagination.
            boss_pages = [
                page
                for page in pages
                if "zhipin.com" in self._known_page_url(page)
            ]
            search_pages = [
                candidate
                for candidate in boss_pages
                if urllib.parse.urlparse(self._known_page_url(candidate)).path.rstrip("/")
                == "/web/geek/jobs"
            ]
            retained_search = (
                retained
                if retained is not None
                and urllib.parse.urlparse(self._known_page_url(retained)).path.rstrip("/")
                == "/web/geek/jobs"
                else None
            )
            page = retained_search or (
                search_pages[-1]
                if search_pages
                else (retained or (boss_pages[-1] if boss_pages else pages[-1]))
            )
            page.set_default_timeout(self.timeout_ms)
        except Exception as exc:
            if controller is not None:
                try:
                    controller.stop()
                except Exception:
                    pass
            raise EdgeActionRequired(
                "edge_login_attach_failed",
                "暂时无法接管登录窗口，请保持 Edge 打开并重新搜索",
            ) from exc

        self._manual_debug_endpoint = endpoint
        self._playwright = controller
        self._browser = browser
        self._context = context
        self._page = page

    def detach_manual_login(self) -> None:
        """Release Playwright control while leaving the visible Edge open."""

        playwright = self._playwright
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._auxiliary_parent_page = None
        if playwright is not None:
            try:
                playwright.stop()
            except Exception:
                pass

    @staticmethod
    def _close_context_pages(context: Any) -> int:
        """Close every app-owned tab individually before stopping Edge.

        Chromium records an unclean browser exit when its process tree is
        terminated while page targets still exist.  Closing each target first
        prevents the next isolated-profile launch from showing “恢复页面”.
        Failures are best-effort because the final owned-process shutdown still
        has to run when one target has already crashed.
        """

        if context is None:
            return 0
        try:
            pages = list(context.pages)
        except Exception:
            return 0
        closed = 0
        # Auxiliary/detail tabs are normally appended after the retained search
        # page.  Close them first and the original collection page last.
        for page in reversed(pages):
            try:
                is_closed = getattr(page, "is_closed", None)
                if callable(is_closed) and is_closed():
                    continue
                page.close()
                closed += 1
            except Exception:
                pass
        return closed

    def close_manual_login(self) -> None:
        process = self._manual_login_process
        recorded_pid = self._recorded_owned_pid()

        # Search calls detach their CDP client so the browser can stay open.
        # Reattach only for shutdown, close every tab one by one, then release
        # the controller.  If Edge/CDP is already unavailable the exact
        # app-recorded PID fallback below still prevents an orphan process.
        if self._context is None and (
            process is not None
            or recorded_pid is not None
            or self._manual_debug_endpoint is not None
            or self._manual_debug_port is not None
        ):
            try:
                self.adopt_manual_login()
            except Exception:
                pass
        self._close_context_pages(self._context)
        self.detach_manual_login()

        self._manual_login_process = None
        self._manual_debug_port = None
        self._manual_debug_endpoint = None
        self._search_target_id = None
        self._discard_runtime_file(self._devtools_port_file)
        self._discard_runtime_file(self._debug_port_record_file)
        self._discard_runtime_file(self._owned_pid_record_file)
        if process is None:
            if recorded_pid is not None:
                self._stop_owned_process_tree(recorded_pid)
            return
        if process.poll() is not None:
            if recorded_pid is not None and recorded_pid != process.pid:
                self._stop_owned_process_tree(recorded_pid)
            return
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            graceful = subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
                creationflags=creation_flags,
            )
            if graceful.returncode == 0:
                process.wait(timeout=3)
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
                creationflags=creation_flags,
            )
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _ensure_started(self) -> None:
        if self.running:
            return
        if self._context is not None or self._playwright is not None or self._page is not None:
            self.close()
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError as exc:
            raise EdgeActionRequired(
                "playwright_not_installed",
                "未安装 Playwright；请手动安装 playwright 后重新检测",
            ) from exc
        if not self.edge_path:
            raise EdgeActionRequired("edge_not_found", "未检测到 Microsoft Edge，请在设置中指定 msedge.exe 路径")
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._playwright = sync_playwright().start()
            self._context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                executable_path=self.edge_path,
                headless=False,
                viewport=None,
                chromium_sandbox=True,
                args=[
                    "--start-maximized",
                    "--disable-background-mode",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                ],
            )
            self._browser = self._context.browser
            self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
            self._page.set_default_timeout(self.timeout_ms)
        except Exception as exc:
            self.close()
            raise EdgeActionRequired("edge_launch_failed", f"无法启动可见 Edge：{exc}") from exc

    def open(self, url: str) -> None:
        for attempt in range(2):
            self._ensure_started()
            try:
                self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                if urllib.parse.urlparse(str(url)).path.rstrip("/") == "/web/geek/jobs":
                    target_id = str(getattr(self._page, "_target_id", "") or "")
                    if target_id:
                        self._search_target_id = target_id
                return
            except Exception as exc:
                message = str(exc)
                session_closed = "Target page, context or browser has been closed" in message
                network_error = any(
                    marker in message
                    for marker in (
                        "ERR_NETWORK_ACCESS_DENIED",
                        "ERR_INTERNET_DISCONNECTED",
                        "ERR_PROXY_CONNECTION_FAILED",
                        "ERR_NAME_NOT_RESOLVED",
                    )
                )
                transient_navigation = session_closed or any(
                    marker in message
                    for marker in (
                        "ERR_ABORTED",
                        "interrupted by another navigation",
                        "Navigation is interrupted",
                        "frame was detached",
                    )
                )
                if transient_navigation and attempt == 0:
                    # BOSS may redirect immediately after login/navigation and
                    # abort the original request.  Keep the adopted page alive
                    # and retry there instead of discarding its authenticated
                    # session.  A genuinely closed page still needs a restart.
                    if session_closed:
                        self.close()
                    time.sleep(0.3)
                    continue
                if network_error:
                    # Preserve the owned authenticated Edge window. A proxy or
                    # network outage may be temporary, and closing it here
                    # discards both the user's login and the retained search
                    # target needed for a later retry.
                    raise EdgeActionRequired(
                        "boss_network_unavailable",
                        "Edge 无法访问 BOSS，请检查系统网络或代理是否允许访问 zhipin.com",
                    ) from exc
                self.close()
                raise EdgeActionRequired(
                    "boss_navigation_failed",
                    "BOSS 页面打开失败，浏览器会话已重置，请重新搜索",
                ) from exc

    def open_auxiliary(self, url: str) -> None:
        """Open one background detail target while preserving the search page."""

        self._ensure_started()
        self.close_auxiliary()
        # Parsing is not a user-view action.  Keep the owned collection window
        # behind the web app even if Edge restored it after a prior explicit
        # "在采集 Edge 查看" click.
        self.minimize_collection_window()
        parent = self._page
        background_factory = getattr(self._context, "new_background_page", None)
        page = (
            background_factory()
            if callable(background_factory)
            else self._context.new_page()
        )
        page.set_default_timeout(self.timeout_ms)
        self._auxiliary_parent_page = parent
        self._page = page
        try:
            page.goto(url, wait_until="commit", timeout=min(self.timeout_ms, 5_000))
        except Exception as exc:
            self.close_auxiliary()
            raise EdgeActionRequired(
                "boss_navigation_failed",
                "BOSS 职位详情打开失败，请重新解析",
            ) from exc

    def open_user_tab(self, url: str) -> None:
        """Open a user-visible tab in the retained collection Edge window.

        The active search page remains selected internally so later pagination
        can return to it without navigation, refresh, or a browser restart.
        """

        self._ensure_started()
        page = self._context.new_page()
        page.set_default_timeout(self.timeout_ms)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            page.bring_to_front()
        except Exception as exc:
            try:
                if not page.is_closed():
                    page.close()
            except Exception:
                pass
            raise EdgeActionRequired(
                "boss_navigation_failed",
                "无法在岗位采集 Edge 中打开该职位页面",
            ) from exc

    def close_auxiliary(self) -> None:
        parent = self._auxiliary_parent_page
        if parent is None:
            return
        page = self._page
        self._auxiliary_parent_page = None
        try:
            if page is not None and page is not parent and not page.is_closed():
                page.close()
        except Exception:
            pass
        try:
            self._page = parent if not parent.is_closed() else None
        except Exception:
            self._page = None

    @staticmethod
    def _keep_target_active(page: Any) -> None:
        """Resume a background page without making its Edge tab visible."""

        activate = getattr(page, "emulate_background_active", None)
        if callable(activate):
            try:
                activate()
            except Exception:
                pass

    @staticmethod
    def _dispatch_trusted_wheel(page: Any, delta_y: float = 1400) -> bool:
        """Dispatch real browser input while leaving the Edge window hidden.

        BOSS currently has list loaders that react to a trusted wheel event,
        not just JavaScript ``scrollTop`` changes.  The normal Playwright page
        exposes ``mouse.wheel``; the direct CDP page implements
        ``dispatch_wheel`` above.  Neither path brings the target to front.
        """

        PlaywrightEdgeBrowser._keep_target_active(page)
        direct_dispatch = getattr(page, "dispatch_wheel", None)
        if callable(direct_dispatch):
            try:
                direct_dispatch(delta_y)
                return True
            except Exception:
                pass
        try:
            page.mouse.wheel(0, delta_y)
            return True
        except Exception:
            return False

    def select_search_page(self, target_url: str) -> bool:
        """Select the retained search tab without navigating or refreshing it."""

        if self._context is None:
            return False
        try:
            pages = [page for page in self._context.pages if not page.is_closed()]
            retained = next(
                (
                    page
                    for page in pages
                    if self._search_target_id
                    and str(getattr(page, "_target_id", "")) == self._search_target_id
                    and urllib.parse.urlparse(self._known_page_url(page)).path.rstrip("/")
                    == "/web/geek/jobs"
                ),
                None,
            )
            candidates = [
                page
                for page in pages
                if same_job_search_page(self._known_page_url(page), target_url)
            ]
            retained_search = (
                retained
                if retained is not None
                and urllib.parse.urlparse(self._known_page_url(retained)).path.rstrip("/")
                == "/web/geek/jobs"
                else None
            )
            selected = retained_search or (candidates[-1] if candidates else None)
            if selected is None:
                return False
            self._page = selected
            self._page.set_default_timeout(self.timeout_ms)
            self._keep_target_active(self._page)
            self._keep_target_active(self._page)
            target_id = str(getattr(self._page, "_target_id", "") or "")
            if target_id:
                self._search_target_id = target_id
            return True
        except Exception:
            return False

    @staticmethod
    def _known_page_url(page: Any, *, live: bool = False) -> str:
        """Read a page URL without activating unrelated direct-CDP targets."""

        if live:
            try:
                return str(page.url or "").strip()
            except Exception:
                pass
        cached = getattr(page, "_url", None)
        if isinstance(cached, str) and cached.strip():
            return cached.strip()
        try:
            return str(page.url or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _job_anchor_script(anchor_url: str | None) -> str:
        """Build a DOM-only restore script for the last collected job card."""

        encoded_anchor = json.dumps(str(anchor_url or ""), ensure_ascii=False)
        return rf"""
        () => {{
          const expected = {encoded_anchor};
          const normalize = (value) => {{
            try {{
              const parsed = new URL(value, location.href);
              return parsed.origin + parsed.pathname;
            }} catch (_) {{ return ''; }}
          }};
          const cards = Array.from(document.querySelectorAll(
            '.job-card-box, .job-card-wrapper, .job-list-box > li'
          )).filter((card) =>
            card.querySelector('a.job-card-left[href*="/job_detail/"], a[href*="/job_detail/"]')
          );
          const expectedKey = normalize(expected);
          const target = (expectedKey && cards.find((card) => {{
            const anchor = card.querySelector(
              'a.job-card-left[href*="/job_detail/"], a[href*="/job_detail/"]'
            );
            return normalize(anchor?.href || '') === expectedKey;
          }})) || cards.at(-1) || null;
          target?.scrollIntoView({{block: 'center', behavior: 'instant'}});
          return {{found: Boolean(target && expectedKey && normalize(
            target.querySelector('a[href*="/job_detail/"]')?.href || ''
          ) === expectedKey), card_count: cards.length}};
        }}
        """

    def restore_search_page(
        self, target_url: str, anchor_url: str | None = None
    ) -> bool:
        """Restore the retained search tab and its last collected card.

        A user can activate another tab or follow a job link in the original
        search tab while the MCP is detached. Pagination must continue in the
        existing Edge process: prefer an intact matching search target, then
        recover the remembered target through browser history, and only as a
        final in-process fallback navigate a BOSS tab/new tab to the immutable
        search URL. No browser process is closed or relaunched here.
        """

        if self._context is None:
            return False
        try:
            pages = [page for page in self._context.pages if not page.is_closed()]
            retained = next(
                (
                    page
                    for page in pages
                    if self._search_target_id
                    and str(getattr(page, "_target_id", ""))
                    == self._search_target_id
                ),
                None,
            )
            matching = [
                page
                for page in pages
                if same_job_search_page(self._known_page_url(page), target_url)
            ]
            # An old target id can now point at a detail page when the user
            # followed a card in-place. Prefer any still-intact matching
            # search document before attempting history recovery on that old
            # target. This also ignores whichever unrelated tab is frontmost.
            retained_search = (
                retained
                if retained is not None
                and same_job_search_page(self._known_page_url(retained), target_url)
                else None
            )
            selected = retained_search or (matching[-1] if matching else retained)
            created_recovery_page = False
            if selected is None:
                # Preserve a user-opened job detail tab. Restore collection in
                # a background target inside the same authenticated context.
                background_factory = getattr(self._context, "new_background_page", None)
                regular_factory = getattr(self._context, "new_page", None)
                if callable(background_factory):
                    selected = background_factory()
                    created_recovery_page = True
                elif callable(regular_factory):
                    selected = regular_factory()
                    created_recovery_page = True
                else:
                    # Compatibility for minimal browser ports: reuse a BOSS
                    # tab only when the context cannot create a new target.
                    selected = next(
                        (
                            page
                            for page in pages
                            if urllib.parse.urlparse(self._known_page_url(page)).netloc.lower()
                            == "www.zhipin.com"
                        ),
                        None,
                    )
                    if selected is None:
                        return False
            selected.set_default_timeout(self.timeout_ms)

            if not same_job_search_page(self._known_page_url(selected), target_url):
                # The remembered target may have been navigated in-place to a
                # job detail. First use its history so BOSS can restore the
                # virtual list and scroll state without another search request.
                if not created_recovery_page:
                    try:
                        selected.evaluate("() => { history.back(); return true; }")
                    except Exception:
                        pass
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    if same_job_search_page(
                        self._known_page_url(selected, live=True), target_url
                    ):
                        break
                    time.sleep(0.1)
                if not same_job_search_page(
                    self._known_page_url(selected, live=True), target_url
                ):
                    # History may not contain the search (for example the user
                    # closed its tab). Reuse this authenticated Edge context;
                    # never restart the browser or discard its login profile.
                    selected.goto(
                        target_url,
                        wait_until="domcontentloaded",
                        timeout=self.timeout_ms,
                    )
            if not same_job_search_page(
                self._known_page_url(selected, live=True), target_url
            ):
                return False

            self._page = selected
            self._keep_target_active(selected)
            target_id = str(getattr(selected, "_target_id", "") or "")
            if target_id:
                self._search_target_id = target_id

            # On a history-restored virtual list the prior card normally
            # exists immediately. If BOSS is still hydrating it, briefly walk
            # the list viewport until the anchor appears; otherwise leave the
            # last real card visible so load_more_items can continue from it.
            anchor_deadline = time.monotonic() + 8.0
            try:
                while True:
                    state = selected.evaluate(self._job_anchor_script(anchor_url))
                    if not anchor_url or (
                        isinstance(state, dict) and bool(state.get("found"))
                    ):
                        break
                    if time.monotonic() >= anchor_deadline:
                        break
                    selected.evaluate(
                        """
                        () => {
                          const roots = Array.from(document.querySelectorAll(
                            '.job-list-container, .rec-job-list, .job-list-box, [class*="job-list"]'
                          ));
                          const viewport = roots.find((node) =>
                            node.scrollHeight > node.clientHeight
                          );
                          if (viewport) {
                            viewport.scrollBy({top: Math.max(viewport.clientHeight * 0.9, 600), behavior: 'instant'});
                            viewport.dispatchEvent(new Event('scroll', {bubbles: true}));
                          } else {
                            window.scrollBy({top: Math.max(window.innerHeight * 0.9, 600), behavior: 'instant'});
                          }
                        }
                        """
                    )
                    self._dispatch_trusted_wheel(selected, 1200)
                    time.sleep(0.2)
            except Exception:
                # Restoring the correct search document is the required part;
                # a transient virtual-list hydration failure must not turn a
                # valid retained session into "session expired".
                pass
            return True
        except Exception:
            return False

    def page_state(self) -> dict[str, Any]:
        self._ensure_started()
        deadline = time.monotonic() + 2.0
        while True:
            try:
                snapshot = self._page.evaluate(
                    """
                    () => ({
                      url: location.href,
                      body: document.body?.innerText || ''
                    })
                    """
                )
                snapshot = snapshot if isinstance(snapshot, dict) else {}
                url = str(snapshot.get("url") or self._page.url or "")
                body = str(snapshot.get("body") or "")[:200_000]
                break
            except Exception as exc:
                message = str(exc)
                target_closed = "Target page, context or browser has been closed" in message
                if target_closed and time.monotonic() < deadline:
                    if not self._recover_live_search_page():
                        time.sleep(0.1)
                    continue
                if target_closed:
                    self.close()
                    raise EdgeActionRequired("edge_session_closed", "Edge 窗口已关闭，请重新搜索") from exc
                raise EdgeActionRequired("boss_page_unavailable", "暂时无法读取 BOSS 页面，请重新搜索") from exc
        challenge_words = ("安全验证", "请完成验证", "验证码", "拖动滑块", "访问异常", "访问受限")
        challenge = next((word for word in challenge_words if word in body), None)
        try:
            visible_login_control = bool(
                self._page.evaluate(
                    r"""
                    () => {
                      const selectors = [
                        'a[ka="header-login"]',
                        'a[href*="/web/user/?ka=header-login"]',
                        'a[href*="/web/user/login"]',
                        'a[href*="/login"]',
                        '.header-login-btn',
                        '.btn-login',
                        '[data-ka="header-login"]',
                        'header a', 'header button', 'header [role="button"]',
                        '.header a', '.header button', '.header [role="button"]',
                        '[class*="header"] a', '[class*="header"] button',
                        'nav a', 'nav button'
                      ];
                      const nodes = new Set(selectors.flatMap((selector) =>
                        Array.from(document.querySelectorAll(selector))
                      ));
                      const exactLoginLabels = new Set([
                        '登录/注册', '登录／注册', '登录|注册', '登录', '注册'
                      ]);
                      return Array.from(nodes).some((node) => {
                        const style = window.getComputedStyle(node);
                        const rect = node.getBoundingClientRect();
                        const visible = style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
                        if (!visible) return false;
                        const label = (node.textContent || '').replace(/\s+/g, '').trim();
                        const href = String(node.getAttribute?.('href') || '');
                        const marker = String(node.getAttribute?.('ka') || node.getAttribute?.('data-ka') || '');
                        return exactLoginLabels.has(label) ||
                          /(?:^|[-_])header-login(?:$|[-_])/.test(marker) ||
                          /\/web\/user\/(?:login)?(?:\?|$)|\/login(?:\?|$)/.test(href);
                      });
                    }
                    """
                )
            )
        except Exception:
            # A DOM probe failure must not turn generic page copy containing
            # “登录” into a false logout signal.  URL routes remain reliable.
            visible_login_control = False
        needs_login = login_required(url, visible_login_control=visible_login_control)
        return {
            "url": url,
            "challenge": challenge,
            "needs_login": needs_login,
            "body_text": body,
        }

    @staticmethod
    def _search_items_script() -> str:
        return r"""
        () => Array.from(document.querySelectorAll('.job-card-box, .job-card-wrapper, .job-list-box > li')).map((card) => {
          const text = (root, ...selectors) => {
            for (const selector of selectors) {
              const value = root?.querySelector(selector)?.textContent?.replace(/\s+/g, ' ')?.trim();
              if (value) return value;
            }
            return null;
          };
          const anchor = card.querySelector('a.job-card-left, a[href*="/job_detail/"]');
          if (!anchor) return null;
          const resolved = new URL(anchor.href, location.href);
          if (!resolved.pathname.includes('/job_detail/')) return null;
          const activityPattern = /(刚刚活跃|当前在线|在线|今日活跃|今天活跃|昨日活跃|昨天活跃|[一二两三四五六七八九十\d]+\s*(?:分钟|小时|天|日|周|个月|月)(?:内|前)?活跃|半年(?:内|前)?活跃|本周活跃|本月活跃|近(?:一|两|二)?周活跃|近(?:一)?月活跃|月内活跃|较少活跃)/;
          const cleanRecruiterName = (value) => {
            let candidate = String(value || '').replace(/\s+/g, ' ').trim();
            candidate = candidate.replace(activityPattern, ' ');
            const honorific = candidate.match(/([\u4e00-\u9fffA-Za-z·]{1,20}(?:先生|女士))/);
            if (honorific) return honorific[1];
            candidate = candidate
              .replace(/(?:招聘者|招聘经理|招聘专员|人事经理|人事专员|人事|HRBP|HR|猎头|经理)/gi, ' ')
              .replace(/[|｜·•,，:：()（）\s]+/g, ' ')
              .trim();
            const parts = candidate.split(' ').filter((part) =>
              /^[\u4e00-\u9fffA-Za-z·]{2,24}$/.test(part) &&
              !/(公司|科技|集团|岗位|职位|薪|经验|学历)/.test(part)
            );
            return parts[0] || '';
          };
          // On the current BOSS search page ``.job-card-footer .boss-info``
          // is the *company* block (logo + company name), not a recruiter.
          // Only accept an HR pair when this exact card exposes a dedicated
          // recruiter subtree containing both the name and textual activity.
          const recruiterRoots = Array.from(card.querySelectorAll([
            '.job-boss-info', '.job-card-boss', '[class*="job-boss-info"]',
            '[class*="recruiter-info"]', '[class*="recruiter-card"]'
          ].join(',')));
          let recruiterText = '';
          let activity = null;
          for (const recruiterRoot of recruiterRoots) {
            const rootText = String(recruiterRoot.innerText || recruiterRoot.textContent || '')
              .replace(/\s+/g, ' ').trim();
            const rootActivity = rootText.match(activityPattern)?.[1] || null;
            let rootName = text(
              recruiterRoot, '.boss-name', '[class*="boss-name"]',
              '.recruiter-name', '[class*="recruiter-name"]',
              '.name', '[class*="name"]'
            ) || cleanRecruiterName(rootText);
            rootName = cleanRecruiterName(rootName);
            if (rootName && rootActivity) {
              recruiterText = rootName;
              activity = rootActivity;
              break;
            }
          }
          return {
             title: text(card, '.job-name'),
             company: text(
               card, '.company-name', '.company-info .name', '[class*="company-name"]',
               '.job-card-footer .boss-name', '.job-card-footer [class*="boss-name"]'
             ),
             city: text(card, '.job-area', '.company-location'),
             salary: text(card, '.salary', '.job-salary'),
             url: resolved.href,
             tags: Array.from(card.querySelectorAll('.tag-list li, .job-card-footer li')).map(x => x.textContent.trim()).filter(Boolean),
             hr_name: recruiterText || null,
             hr_activity: activity || '活跃时间待解析'
          };
        }).filter(item => item.title && item.url)
        """

    def _evaluate_search_items(self) -> list[dict[str, Any]]:
        self._keep_target_active(self._page)
        target_id = str(getattr(self._page, "_target_id", "") or "")
        if target_id:
            self._search_target_id = target_id
        value = self._page.evaluate(self._search_items_script())
        return value if isinstance(value, list) else []

    def _recover_live_search_page(self) -> bool:
        """Follow a BOSS-created replacement tab without causing navigation."""

        if self._context is None:
            return False
        try:
            pages = [page for page in self._context.pages if not page.is_closed()]
            retained = next(
                (
                    page
                    for page in pages
                    if self._search_target_id
                    and str(getattr(page, "_target_id", "")) == self._search_target_id
                ),
                None,
            )
            candidates = [
                page
                for page in pages
                if urllib.parse.urlparse(self._known_page_url(page)).path.rstrip("/")
                == "/web/geek/jobs"
            ]
            selected = retained or (candidates[-1] if candidates else None)
            if selected is None:
                return False
            self._page = selected
            self._page.set_default_timeout(self.timeout_ms)
            target_id = str(getattr(self._page, "_target_id", "") or "")
            if target_id:
                self._search_target_id = target_id
            return True
        except Exception:
            return False

    def search_items(self) -> list[dict[str, Any]]:
        self._ensure_started()
        deadline = time.monotonic() + min(self.timeout_ms / 1000, 5.0)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                value = self._evaluate_search_items()
                if value:
                    return value
                last_error = None
            except Exception as exc:
                # BOSS replaces the document while loading its result list.
                # Retry only until the first stable card batch appears.
                last_error = exc
                self._recover_live_search_page()
            time.sleep(0.1)
        if last_error is not None:
            raise EdgeActionRequired("boss_parse_failed", "职位列表加载期间页面上下文持续变化") from last_error
        return []

    def enrich_search_activities(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Atomically verify recruiter name/activity in the matching detail pane.

        Each card is selected in place, but its pair is accepted only when the
        rendered detail can be tied to the target URL or to both the target
        title and company. Name and activity must then come from the same
        recruiter subtree. A stale asynchronous pane therefore cannot mix one
        job's HR with another job's activity.
        """

        self._ensure_started()
        self._keep_target_active(self._page)
        enriched = [dict(item) for item in items]
        overall_deadline = time.monotonic() + 40.0
        for item in enriched:
            if time.monotonic() >= overall_deadline:
                break
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            target = json.dumps(url, ensure_ascii=False)
            expected_title = json.dumps(str(item.get("title") or ""), ensure_ascii=False)
            expected_company = json.dumps(str(item.get("company") or ""), ensure_ascii=False)
            script = rf"""
            async () => {{
              const targetUrl = new URL({target}, location.href);
              const expectedTitle = String({expected_title} || '').replace(/\s+/g, '').trim();
              const expectedCompany = String({expected_company} || '').replace(/\s+/g, '').trim();
              const normalize = (value) => {{
                try {{
                  const parsed = new URL(value, location.href);
                  return parsed.origin + parsed.pathname;
                }} catch (_) {{ return ''; }}
              }};
              const cards = Array.from(document.querySelectorAll(
                '.job-card-box, .job-card-wrapper, .job-list-box > li'
              ));
              const card = cards.find((node) => {{
                const anchor = node.querySelector('a.job-card-left, a[href*="/job_detail/"]');
                return anchor && normalize(anchor.href) === normalize(targetUrl.href);
              }});
              if (!card) return null;
              card.scrollIntoView({{block: 'nearest', behavior: 'instant'}});
              const clickTarget = card.querySelector(
                '.job-info, .job-card-body, .job-card-info, .job-name'
              ) || card;
              clickTarget.click();
              const activityFrom = (value) => {{
                const text = String(value || '').replace(/\s+/g, ' ').trim();
                // BOSS renders the green `在线` badge inline with the name on
                // some builds, so innerText can be `程女士在线` without a
                // whitespace boundary.  This function is only called for a
                // recruiter-card subtree; accepting the bare token here is
                // therefore both necessary and scoped.
                const labelled = text.match(/刚刚活跃|当前在线|今日活跃|今天活跃|昨日活跃|昨天活跃|[一二两三四五六七八九十\d]+\s*(?:分钟|小时|天|日|周|个月|月)(?:内|前)?活跃|本周活跃|本月活跃|近(?:一|两|二)?周活跃|近(?:一)?月活跃|月内活跃|半年(?:内|前)?活跃|较少活跃|在线/);
                return labelled ? labelled[0] : null;
              }};
              const recruiterNameFrom = (root) => {{
                for (const selector of [
                  '.boss-name', '.job-boss-info .name', '.boss-info .name',
                  '.job-detail-boss .name', '[class*="boss-name"]',
                  '.recruiter-name', '[class*="recruiter-name"]', '.name'
                ]) {{
                  const source = root.querySelector(selector);
                  if (!source) continue;
                  const copy = source.cloneNode(true);
                  copy.querySelectorAll(
                    '.boss-active-time, .boss-status, [class*="active-time"], '
                    + '[class*="boss-status"], [class*="online"], i, svg'
                  ).forEach((node) => node.remove());
                  const value = copy.textContent
                    ?.replace(/刚刚活跃|当前在线|在线|今日活跃|今天活跃|昨日活跃|昨天活跃|[一二两三四五六七八九十\d]+\s*(?:分钟|小时|天|日|周|个月|月)(?:内|前)?活跃|本周活跃|本月活跃|近(?:一|两|二)?周活跃|近(?:一)?月活跃|月内活跃|半年(?:内|前)?活跃|较少活跃/g, ' ')
                    ?.replace(/\s+/g, ' ')?.trim();
                  const honorific = value?.match(/([\u4e00-\u9fffA-Za-z·]{{1,20}}(?:先生|女士))/);
                  if (honorific) return honorific[1];
                  if (value && value.length <= 40) return value;
                }}
                // Current BOSS markup may rename the `.name` class while the
                // human-readable card text remains stable.  Keep this scoped
                // fallback inside the same recruiter subtree.
                const rootText = String(root.innerText || root.textContent || '')
                  .replace(/\s+/g, ' ').trim();
                const honorific = rootText.match(/([\u4e00-\u9fffA-Za-z·]{{1,20}}(?:先生|女士))/);
                if (honorific) return honorific[1];
                return null;
              }};
              const deadline = Date.now() + 2200;
              while (Date.now() < deadline) {{
                const roots = Array.from(document.querySelectorAll(
                  '.job-detail-container, .job-detail-box, .job-detail-wrap, '
                  + '.job-detail-content, .job-detail, [class*="job-detail"]'
                )).filter((node) => {{
                  const rect = node.getBoundingClientRect();
                  const style = getComputedStyle(node);
                  return rect.width > 0 && rect.height > 0 && style.display !== 'none' &&
                    style.visibility !== 'hidden';
                }});
                for (const root of roots) {{
                  const heading = root.querySelector(
                    '.job-name, .name h1, .job-banner .name, h1, h2'
                  )?.textContent?.replace(/\s+/g, '')?.trim() || '';
                  const correctLink = Array.from(
                    root.querySelectorAll('a[href*="/job_detail/"]')
                  ).some((anchor) => normalize(anchor.href) === normalize(targetUrl.href));
                  const correctTitle = expectedTitle && heading &&
                    (heading.includes(expectedTitle) || expectedTitle.includes(heading));
                  const company = root.querySelector(
                    '.company-name, .company-info .name, .company-info a, '
                    + '.company-info, [class*="company-name"], [class*="company-info"] a'
                  )?.textContent?.replace(/\s+/g, '')?.trim() || '';
                  const rootText = String(root.innerText || root.textContent || '')
                    .replace(/\s+/g, '');
                  const correctCompany = expectedCompany && (
                    (company && (company.includes(expectedCompany) || expectedCompany.includes(company))) ||
                    rootText.includes(expectedCompany)
                  );
                  const selectedCard = card.matches(
                    '.active, .selected, [aria-selected="true"], [class*="selected"]'
                  ) || Boolean(card.querySelector(
                    '.active, .selected, [aria-selected="true"], [class*="selected"]'
                  ));
                  // Title alone is not unique enough (many cards share it).
                  // URL is strongest; otherwise require title plus the
                  // expected company, or the exact selected card after its
                  // right-hand heading has switched.
                  if (!correctLink && !(correctTitle && (correctCompany || selectedCard))) continue;

                  const explicitRecruiterRoots = Array.from(root.querySelectorAll([
                    '.job-boss-info', '.boss-info', '.job-detail-boss',
                    '.boss-info-box', '.boss-card', '.recruiter-card',
                    '[class*="job-boss"]', '[class*="boss-info"]',
                    '[class*="recruiter"]'
                  ].join(','))).filter((node) => !node.closest(
                    '.job-list-container, .rec-job-list, .job-list-box'
                  ));
                  // DOM class names have changed across BOSS releases.  A
                  // semantic fallback finds the smallest visible subtree that
                  // contains both a recruiter honorific and an activity label.
                  const semanticRecruiterRoots = Array.from(root.querySelectorAll('div, section, li'))
                    .filter((node) => {{
                      if (node.closest('.job-list-container, .rec-job-list, .job-list-box')) return false;
                      const rect = node.getBoundingClientRect();
                      const style = getComputedStyle(node);
                      if (rect.width <= 0 || rect.height <= 0 || style.display === 'none' || style.visibility === 'hidden') return false;
                      const value = String(node.innerText || node.textContent || '')
                        .replace(/\s+/g, ' ').trim();
                      return value.length <= 240 &&
                        /[\u4e00-\u9fffA-Za-z·]{{1,20}}(?:先生|女士)/.test(value) &&
                        Boolean(activityFrom(value));
                    }})
                    .sort((left, right) =>
                      String(left.innerText || left.textContent || '').length -
                      String(right.innerText || right.textContent || '').length
                    );
                  const recruiterRoots = [...new Set([
                    ...explicitRecruiterRoots,
                    ...semanticRecruiterRoots
                  ])];
                  for (const recruiterRoot of recruiterRoots) {{
                    const hrName = recruiterNameFrom(recruiterRoot);
                    const activity = activityFrom(
                      recruiterRoot.innerText || recruiterRoot.textContent || ''
                    );
                    // Atomic pair only: partial data can never overwrite the
                    // corresponding values taken from the search card.
                    if (activity && hrName) {{
                      return {{hr_name: hrName, hr_activity: activity}};
                    }}
                  }}
                }}
                await new Promise((resolve) => setTimeout(resolve, 120));
              }}
              return null;
            }}
            """
            try:
                recruiter = self._page.evaluate(script)
                if isinstance(recruiter, dict):
                    hr_name = str(recruiter.get("hr_name") or "").strip()
                    activity = str(recruiter.get("hr_activity") or "").strip()
                    if hr_name and activity:
                        item["hr_name"] = hr_name
                        item["hr_activity"] = activity
            except Exception:
                # Best-effort only. Login/CAPTCHA handling stays at the
                # surrounding search boundary and is never bypassed here.
                pass
            time.sleep(0.05)
        return enriched

    def load_more_items(
        self, seen_urls: set[str], *, minimum_new_items: int = 1
    ) -> dict[str, Any]:
        """Progressively scroll until new cards appear or the page explicitly ends.

        A timeout is deliberately reported as ``load_pending`` instead of
        ``exhausted``. BOSS often delays its infinite-list request, and treating
        one slow response as the end would incorrectly disable pagination.
        """

        self._ensure_started()
        minimum_new_items = max(1, int(minimum_new_items))
        deadline = time.monotonic() + min(max(self.timeout_ms / 1000, 15.0), 20.0)
        next_scroll_at = 0.0
        last_error: Exception | None = None
        last_items: list[dict[str, Any]] = []
        successful_probe = False
        while time.monotonic() < deadline:
            try:
                value = self._evaluate_search_items()
                last_items = value
                successful_probe = True
                unseen_count = sum(
                    1
                    for item in value
                    if str(item.get("url") or "") not in seen_urls
                )
                if unseen_count >= minimum_new_items:
                    return {"items": value, "exhausted": False, "load_pending": False}
                now = time.monotonic()
                if now >= next_scroll_at:
                    scroll_state = self._page.evaluate(
                        """
                        () => {
                          const allCards = Array.from(document.querySelectorAll(
                            '.job-card-box, .job-card-wrapper, .job-list-box > li'
                          ));
                          const cards = allCards.filter((card) =>
                            card.querySelector('a.job-card-left[href*="/job_detail/"], a[href*="/job_detail/"]')
                          );
                          const lastJobCard = cards.at(-1);
                          const listRoot = lastJobCard?.closest(
                            '.job-list-box, .job-list-container, .rec-job-list, ul, ol'
                          ) || lastJobCard?.parentElement || null;
                          // The survey/promotion node is commonly the last
                          // child after the last real job. Scroll that sentinel
                          // into view as well, otherwise the lazy loader below
                          // it never intersects and no later jobs are appended.
                          const lastListNode = listRoot?.lastElementChild || lastJobCard;
                          lastListNode?.scrollIntoView({block: 'end', behavior: 'instant'});
                          // BOSS sometimes inserts a survey card (“对搜索是否
                          // 满意”) between real jobs. Scroll the nearest list
                          // viewport past that non-job node instead of treating
                          // it as the end of the result set.
                          let scrollParent = listRoot;
                          while (scrollParent && scrollParent !== document.body) {
                            const style = getComputedStyle(scrollParent);
                            if (/(auto|scroll)/.test(style.overflowY || '') &&
                                scrollParent.scrollHeight > scrollParent.clientHeight) break;
                            scrollParent = scrollParent.parentElement;
                          }
                          if (scrollParent && scrollParent !== document.body) {
                            scrollParent.scrollBy({
                              top: Math.max(scrollParent.clientHeight * 0.9, 600),
                              behavior: 'instant'
                            });
                            if (scrollParent.scrollHeight - scrollParent.scrollTop -
                                scrollParent.clientHeight < 32) {
                              scrollParent.scrollTop = scrollParent.scrollHeight;
                            }
                            scrollParent.dispatchEvent(new Event('scroll', {bubbles: true}));
                            scrollParent.dispatchEvent(new WheelEvent('wheel', {
                              deltaY: 1200, bubbles: true, cancelable: true
                            }));
                          }
                          const root = document.scrollingElement || document.documentElement;
                          window.scrollTo({top: root.scrollHeight, behavior: 'instant'});
                          document.querySelectorAll(
                            '.job-list-container, .rec-job-list, .job-list-box, [class*="job-list"]'
                          ).forEach((node) => {
                            if (node.scrollHeight > node.clientHeight) {
                              node.scrollTop = node.scrollHeight;
                              node.dispatchEvent(new Event('scroll', {bubbles: true}));
                            }
                          });
                          const loadMore = Array.from(document.querySelectorAll('button, a')).find((node) =>
                            /查看更多职位|继续加载|加载更多/.test(node.textContent || '') &&
                            node.getBoundingClientRect().height > 0
                          );
                          loadMore?.click();
                          const endNodes = Array.from(document.querySelectorAll(
                            '.no-more, .no-more-data, .job-list-empty, [class*="no-more"]'
                          ));
                          const visibleEndNode = endNodes.some((node) => {
                            const rect = node.getBoundingClientRect();
                            const style = window.getComputedStyle(node);
                            return rect.width > 0 && rect.height > 0 && style.display !== 'none' &&
                              style.visibility !== 'hidden' &&
                              /没有更多|暂无更多|到底了|暂无职位/.test(node.textContent || '');
                          });
                          return {
                            card_count: cards.length,
                            scroll_height: root.scrollHeight,
                            at_bottom: root.scrollTop + window.innerHeight >= root.scrollHeight - 8,
                            explicit_end: visibleEndNode
                          };
                        }
                        """
                    )
                    # Some BOSS lazy loaders listen for a trusted wheel input
                    # instead of only observing scrollTop mutations.  The
                    # collection runtime normally uses our direct CDP page
                    # (which has no Playwright ``mouse`` property), so route the
                    # input through the shared dispatcher.  This works while
                    # the owned Edge window remains minimised and never calls
                    # Page.bringToFront.
                    self._dispatch_trusted_wheel(self._page, 1600)
                    if isinstance(scroll_state, dict) and scroll_state.get("explicit_end"):
                        return {"items": value, "exhausted": True, "load_pending": False}
                    next_scroll_at = now + 0.6
                last_error = None
            except Exception as exc:
                last_error = exc
                self._recover_live_search_page()
            time.sleep(0.2)
        if last_error is not None and not successful_probe:
            raise EdgeActionRequired("boss_parse_failed", "向下加载期间页面上下文持续变化") from last_error
        return {"items": last_items, "exhausted": False, "load_pending": True}

    def detail_fields(self) -> dict[str, Any]:
        self._ensure_started()
        script = r"""
        () => {
          const text = (selectors) => {
            for (const selector of selectors) {
              const node = document.querySelector(selector);
              if (node?.textContent?.trim()) return node.textContent.trim();
            }
            return null;
          };
          const description = () => {
            const exactSelectors = [
              '.job-sec-text',
              '.job-detail-section .job-sec-text',
              '.job-detail-section .text',
              '.job-description',
              '[class*="job-sec-text"]'
            ];
            for (const selector of exactSelectors) {
              const nodes = Array.from(document.querySelectorAll(selector));
              const node = nodes.find((candidate) => {
                const value = (candidate.innerText || candidate.textContent || '').trim();
                return value.length >= 20 && value.length <= 30000;
              });
              if (node) return (node.innerText || node.textContent || '').trim();
            }
            // Last resort: start at an exact JD heading and choose the smallest
            // nearby section containing meaningful text. Never read the broad
            // .job-detail page shell, which also contains the BOSS brand/nav.
            const headingPattern = /^(岗位职责|工作职责|职位职责|职位描述|工作内容|岗位描述)$/;
            const headings = Array.from(document.querySelectorAll('h2, h3, .title, [class*="title"]'))
              .filter((node) => headingPattern.test((node.textContent || '').replace(/\s+/g, '')));
            for (const heading of headings) {
              let node = heading.parentElement;
              for (let depth = 0; node && depth < 3; depth += 1, node = node.parentElement) {
                const value = (node.innerText || node.textContent || '').trim();
                if (value.length >= 20 && value.length <= 30000) return value;
              }
            }
            return null;
          };
          const activity = () => {
            const selectors = [
              '.boss-active-time', '.job-boss-info .boss-active-time',
              '.job-boss-info .boss-status', '.boss-info-attr .boss-status',
              '.job-boss-info', '.boss-info', '.job-detail-boss',
              '.info-publis', '.job-publish', '.boss-info-attr',
              '[class*="active-time"]', '[class*="boss-status"]',
              '[class*="boss-online"]', '[class*="recruiter"]'
            ];
            const pattern = /(刚刚活跃|当前在线|在线|今日活跃|今天活跃|昨日活跃|昨天活跃|[一二两三四五六七八九十\d]+\s*(?:分钟|小时|天|日|周|个月|月)(?:内|前)?活跃|本周活跃|本月活跃|近(?:一|两|二)?周活跃|近(?:一)?月活跃|月内活跃|半年内活跃|较少活跃)/;
            for (const selector of selectors) {
              for (const node of document.querySelectorAll(selector)) {
                const value = (node.textContent || '').trim();
                const match = value.match(pattern);
                if (match) return match[1];
              }
            }
            for (const selector of ['.job-boss-info', '.boss-info', '.job-detail-boss', '.job-detail-container']) {
              const value = document.querySelector(selector)?.innerText || '';
              const match = value.match(pattern);
              if (match) return match[1];
            }
            return null;
          };
          return {
            title: text(['.name h1', '.job-banner .name', 'h1']),
            company: text(['.company-info .name', '.company-name']),
            city: text(['.job-address .location-address', '.job-banner .text-desc']),
            salary: text(['.salary']),
            description: description(),
            hr_name: text(['.boss-name', '.job-boss-info .name', '.boss-info .name', '.job-detail-boss .name', '[class*="boss-name"]']),
            hr_activity: activity(),
            posted_label: text(['.job-time', '.job-publish-time', '.publish-time', 'time[datetime]']) ||
              document.querySelector('meta[property="article:published_time"]')?.content || null,
            body_text: document.body?.innerText || ''
          };
        }
        """
        try:
            value = self._page.evaluate(script)
            return value if isinstance(value, dict) else {}
        except Exception as exc:
            raise EdgeActionRequired("boss_parse_failed", f"职位详情解析失败：{exc}") from exc

    def sanitized_html(self) -> str:
        self._ensure_started()
        script = """
        () => {
          const root = document.documentElement.cloneNode(true);
          root.querySelectorAll('script, style, form, input, textarea, meta[http-equiv="set-cookie"]').forEach(n => n.remove());
          root.querySelectorAll('*').forEach((node) => {
            [...node.attributes].forEach((attr) => {
              if (/^(on|data-token|data-cookie|value)/i.test(attr.name)) node.removeAttribute(attr.name);
            });
          });
          return '<!doctype html>\n' + root.outerHTML;
        }
        """
        try:
            return str(self._page.evaluate(script) or "")
        except Exception as exc:
            raise EdgeActionRequired("boss_snapshot_failed", f"职位页面快照失败：{exc}") from exc

    def clear_session(self) -> None:
        if self._context is None:
            return
        try:
            self._context.clear_cookies()
            for page in self._context.pages:
                try:
                    page.evaluate("() => { localStorage.clear(); sessionStorage.clear(); }")
                except Exception:
                    pass
        finally:
            self.close()

    def close(self) -> None:
        context, browser, playwright = self._context, self._browser, self._playwright
        self._close_context_pages(context)
        self._context = None
        self._browser = None
        self._playwright = None
        self._page = None
        self._auxiliary_parent_page = None
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        elif context is not None:
            try:
                context.close()
            except Exception:
                pass
        if playwright is not None:
            try:
                playwright.stop()
            except Exception:
                pass
        # A CDP-attached login window belongs to this isolated profile.  Once
        # the explicit operation finishes, ensure its process tree is gone.
        self.close_manual_login()


@dataclass(frozen=True)
class EdgeAdapterConfig:
    profile_dir: Path
    snapshot_dir: Path
    edge_path: str | None = None
    max_results: int = 20


class EdgeBossAdapter:
    def __init__(self, config: EdgeAdapterConfig, browser: EdgeBrowserPort | None = None) -> None:
        self.config = config
        self.browser = browser or PlaywrightEdgeBrowser(config.profile_dir, edge_path=config.edge_path)
        self._lock = threading.RLock()
        self._active_search_key: tuple[str, str] | None = None
        self._active_search_url: str | None = None
        self._seen_search_urls: set[str] = set()
        self._last_search_job_url: str | None = None

    def _restore_active_search_page(self) -> bool:
        """Return to the retained collection page at its last job anchor."""

        target_url = self._active_search_url or ""
        restore = getattr(self.browser, "restore_search_page", None)
        if callable(restore):
            return bool(restore(target_url, self._last_search_job_url))
        # Backward compatibility for browser ports implementing the original
        # interface before anchored page restoration was introduced.
        return self.browser.select_search_page(target_url)

    def environment(self) -> dict[str, Any]:
        try:
            import importlib.util

            playwright_installed = importlib.util.find_spec("playwright") is not None
        except (ImportError, ValueError):
            playwright_installed = False
        edge_path = getattr(self.browser, "edge_path", None) or self.config.edge_path
        automated_running = self.browser.running
        manual_login_running = self.browser.manual_login_running
        return {
            "status": "ready" if playwright_installed and edge_path else "needs_setup",
            "playwright_installed": playwright_installed,
            "edge_path": edge_path,
            "profile_dir": str(self.config.profile_dir.resolve()),
            "visible": True,
            "running": automated_running or manual_login_running,
            "mode": "automated" if automated_running else ("manual_login" if manual_login_running else "stopped"),
        }

    def _manual_gate(self, state: dict[str, Any]) -> None:
        if state.get("challenge"):
            if self.browser.manual_login_running:
                self.browser.detach_manual_login()
            raise EdgeActionRequired(
                "boss_verification_required",
                f"页面要求人工处理：{state['challenge']}。请在可见 Edge 中完成后再重试",
            )
        if state.get("needs_login"):
            # Disconnect control but preserve the exact visible window in
            # which the user logs in.  The next explicit action reconnects to
            # this same CDP endpoint and page.
            show_for_user = getattr(self.browser, "show_for_user", None)
            if callable(show_for_user):
                show_for_user()
            if self.browser.manual_login_running:
                self.browser.detach_manual_login()
            else:
                self.browser.close()
                self.browser.open_manual_login(BOSS_HOME_URL)
            raise EdgeActionRequired(
                "boss_login_required",
                "请在已打开的岗位采集 Edge 中登录 BOSS，完成后重新点击搜索",
            )

    def _authentication_state(self, initial_url: str = BOSS_HOME_URL) -> dict[str, Any]:
        if not self.browser.running:
            # Always start BOSS in a normal Edge process and attach over the
            # loopback CDP endpoint.  Playwright's persistent-context launcher
            # adds automation startup switches that can make BOSS repeatedly
            # reload the page on later searches.
            if not self.browser.manual_login_running:
                self.browser.open_manual_login(initial_url)
                minimize = getattr(self.browser, "minimize_collection_window", None)
                if callable(minimize):
                    minimize()
            self.browser.adopt_manual_login()
        state = self.browser.page_state()
        # Edge can publish its CDP endpoint before the URL passed on the
        # command line has produced a page target.  In that short window the
        # controller attaches to about:blank/new-tab.  Navigate that empty tab
        # exactly once; never touch an already loaded BOSS page, so repeated
        # searches and pagination do not refresh it.
        if blank_browser_page(str(state.get("url") or "")):
            self.browser.open(initial_url)
            state = self.browser.page_state()
        return state

    def _close_for_error(self, exc: BaseException) -> None:
        if isinstance(exc, EdgeActionRequired):
            # A collection/parser failure is not authorization to kill the
            # app-owned authenticated Edge process.  Doing so made Chromium
            # show “ME 意外关闭/还原页面” and discarded the retained virtual list.
            # Release only our CDP controller; an explicit reset remains the
            # sole operation that closes the collection browser.
            if self.browser.manual_login_running:
                try:
                    if self.browser.running:
                        self.browser.detach_manual_login()
                except Exception:
                    pass
                return
            if exc.code in {
                "boss_navigation_failed",
                "boss_page_unavailable",
                "edge_session_closed",
            }:
                self.browser.close()
                try:
                    self.browser.open_manual_login(BOSS_HOME_URL)
                except EdgeActionRequired:
                    pass
            return
        self.browser.close()

    def check_login(self) -> dict[str, Any]:
        with self._lock:
            state: dict[str, Any] = {}
            try:
                state = self._authentication_state()
                self._manual_gate(state)
            except EdgeActionRequired as exc:
                if exc.code == "boss_login_required":
                    return {
                        "status": "needs_login",
                        "url": state.get("url") or BOSS_HOME_URL,
                        "profile_dir": str(self.config.profile_dir.resolve()),
                    }
                self._close_for_error(exc)
                raise
            except Exception as exc:
                self._close_for_error(exc)
                raise
            else:
                result = {
                    "status": "logged_in",
                    "url": state.get("url"),
                    "profile_dir": str(self.config.profile_dir.resolve()),
                }
                self.browser.close()
                return result

    def search(
        self,
        title: str,
        city: str,
        *,
        city_code: str | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        title, city = title.strip(), city.strip()
        if not title or not city:
            raise ValueError("职位名称和城市为必填项")
        resolved_city_code, city_resolution = normalize_city_code(city, city_code)
        page = max(1, int(cursor or "1"))
        limit = min(max(1, int(limit)), self.config.max_results)
        search_key = (title.casefold(), resolved_city_code)
        query = urllib.parse.urlencode({"query": title, "city": resolved_city_code, "page": 1})
        search_url = f"https://www.zhipin.com/web/geek/jobs?{query}"
        with self._lock:
            try:
                # An earlier explicit "在采集 Edge 查看" action may have
                # restored the isolated browser. Automated collection always
                # puts that same owned window back in the background first;
                # the login gate restores it only when human input is needed.
                minimize = getattr(self.browser, "minimize_collection_window", None)
                if callable(minimize) and self.browser.manual_login_running:
                    minimize()
                loading_more = page > 1
                exhausted = False
                load_pending = False
                if loading_more:
                    if self._active_search_key != search_key or not self.browser.manual_login_running:
                        raise EdgeActionRequired(
                            "boss_search_session_expired",
                            "当前岗位搜索窗口已关闭，请重新搜索后再加载下一页",
                        )
                    self.browser.adopt_manual_login()
                    retained_search_url = self._active_search_url or search_url
                    if not self._restore_active_search_page():
                        raise EdgeActionRequired(
                            "boss_search_session_expired",
                            "无法恢复原岗位采集页，请保持采集 Edge 打开后重试",
                        )
                    state = self.browser.page_state()
                    self._manual_gate(state)
                    if not same_job_search_page(str(state.get("url") or ""), retained_search_url):
                        raise EdgeActionRequired(
                            "boss_search_session_expired",
                            "当前 Edge 页面已离开该岗位搜索，请重新搜索",
                        )
                    try:
                        load_result = self.browser.load_more_items(
                            set(self._seen_search_urls),
                            minimum_new_items=limit,
                        )
                    except TypeError as exc:
                        # External/test browser ports built for the previous MCP
                        # contract may not expose the optional batch target yet.
                        if "minimum_new_items" not in str(exc):
                            raise
                        load_result = self.browser.load_more_items(
                            set(self._seen_search_urls)
                        )
                    if isinstance(load_result, dict):
                        all_items = list(load_result.get("items") or [])
                        exhausted = bool(load_result.get("exhausted"))
                        load_pending = bool(load_result.get("load_pending"))
                    else:  # compatibility with external/test browser ports
                        all_items = list(load_result)
                    raw_items = [
                        item
                        for item in all_items
                        if (
                            str(item.get("url") or "") not in self._seen_search_urls
                            and "/job_detail/"
                            in urllib.parse.urlparse(str(item.get("url") or "")).path
                        )
                    ][:limit]
                else:
                    # Reuse the retained, authenticated collection window.
                    # Closing it here made every repeated submit launch a new
                    # Edge process and could send BOSS through another redirect
                    # or login cycle.  ``_authentication_state`` reconnects to
                    # the existing CDP endpoint; navigation happens only when
                    # the requested title/city really changed.
                    self._active_search_key = None
                    self._active_search_url = None
                    self._seen_search_urls.clear()
                    self._last_search_job_url = None
                    state = self._authentication_state(search_url)
                    self._manual_gate(state)
                    if not same_job_search_page(str(state.get("url") or ""), search_url):
                        self.browser.open(search_url)
                    state = self.browser.page_state()
                    self._manual_gate(state)
                    raw_items = [
                        item
                        for item in self.browser.search_items()
                        if "/job_detail/"
                        in urllib.parse.urlparse(str(item.get("url") or "")).path
                    ][:limit]
                    if not raw_items:
                        self._manual_gate(self.browser.page_state())
                self._manual_gate(self.browser.page_state())
                # Current BOSS cards expose the company in ``.boss-info``;
                # recruiter name/activity live in the right-hand detail pane.
                # Click each returned card in-place and accept the pair only
                # after that pane matches the card URL or title+company. The
                # browser implementation reads both values from the same DOM
                # subtree, preventing stale cross-card combinations.
                enrich_recruiters = getattr(
                    self.browser, "enrich_search_activities", None
                )
                if raw_items and callable(enrich_recruiters):
                    raw_items = enrich_recruiters(raw_items)
                # Header login controls are often rendered after the public
                # card list. Recheck once cards/details have settled so an
                # initially blank header cannot leak a logged-out result list.
                self._manual_gate(self.browser.page_state())
            except Exception as exc:
                self._close_for_error(exc)
                raise
            else:
                for item in raw_items:
                    value = str(item.get("url") or "")
                    if value and "/job_detail/" in urllib.parse.urlparse(value).path:
                        self._seen_search_urls.add(value)
                        self._last_search_job_url = value
                self._active_search_key = search_key
                self._active_search_url = search_url
                minimize = getattr(self.browser, "minimize_collection_window", None)
                if callable(minimize):
                    minimize()
                self.browser.detach_manual_login()
        items: list[dict[str, Any]] = []
        for item in raw_items:
            try:
                url = validate_boss_url(str(item.get("url") or ""))
            except ValueError:
                continue
            if "/job_detail/" not in urllib.parse.urlparse(url).path:
                # Survey, promotion and “对搜索是否满意” cards are not jobs.
                # Ignoring them here also prevents their URL from consuming a
                # pagination slot or being persisted as a selectable record.
                continue
            hr_name, hr_activity = split_recruiter_card_text(
                item.get("hr_name"), item.get("hr_activity")
            )
            items.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "company": item.get("company"),
                    "city": item.get("city"),
                    "salary": decode_boss_obfuscated_text(item.get("salary")) or None,
                    "skills": [str(value) for value in item.get("tags", []) if str(value).strip()],
                    "hr_name": hr_name,
                    "hr_activity": hr_activity,
                    "posted_at": None,
                    "posted_at_label": "发布时间未知",
                    "url": url,
                }
            )
        if (
            loading_more
            and not items
            and not exhausted
            and not load_pending
        ):
            # A survey/lazy-loading boundary is not proof that the result set
            # ended. Keep the same cursor enabled so the user can continue the
            # retained-page scroll; only an explicit end marker clears it.
            load_pending = True
        return {
            "items": items,
            "next_cursor": (
                None
                if exhausted or not self.browser.manual_login_running
                else (
                    str(page + 1)
                    if items
                    else (str(page) if loading_more and load_pending else None)
                )
            ),
            "page": page,
            "exhausted": exhausted,
            "load_pending": load_pending,
            "collection": {
                "status": (
                    "complete"
                    if exhausted
                    else ("waiting_for_more" if load_pending else "batch_ready")
                ),
                "batch": page,
                "batch_size": len(items),
                "seen_count": len(self._seen_search_urls),
            },
            "requested_city": city,
            "city_code": resolved_city_code,
            "city_resolution": city_resolution,
        }

    def close_browser(self) -> dict[str, Any]:
        with self._lock:
            self.browser.close()
            self._active_search_key = None
            self._active_search_url = None
            self._seen_search_urls.clear()
            self._last_search_job_url = None
        return {"status": "closed", "profile_dir": str(self.config.profile_dir.resolve())}

    def open_job_page(self, url: str) -> dict[str, Any]:
        """Open a selected job in the already authenticated collection Edge."""

        url = validate_boss_url(url)
        with self._lock:
            try:
                if not self.browser.manual_login_running:
                    raise EdgeActionRequired(
                        "boss_search_session_expired",
                        "岗位采集 Edge 已关闭，请重新搜索后再打开岗位页面",
                    )
                self.browser.adopt_manual_login()
                if self._active_search_url and not self._restore_active_search_page():
                    raise EdgeActionRequired(
                        "boss_search_session_expired",
                        "无法恢复原岗位采集页，请保持采集 Edge 打开后重试",
                    )
                self._manual_gate(self.browser.page_state())
                self.browser.open_user_tab(url)
                show_for_user = getattr(self.browser, "show_for_user", None)
                if callable(show_for_user):
                    show_for_user()
            except Exception as exc:
                self._close_for_error(exc)
                raise
            else:
                self.browser.detach_manual_login()
        return {"status": "opened", "url": url, "browser": "collection_edge"}

    def detail(self, url: str) -> dict[str, Any]:
        url = validate_boss_url(url)
        with self._lock:
            preserve_search = bool(
                self._active_search_key is not None
                and self._active_search_url
                and self.browser.manual_login_running
            )
            try:
                if preserve_search:
                    minimize = getattr(self.browser, "minimize_collection_window", None)
                    if callable(minimize):
                        minimize()
                    self.browser.adopt_manual_login()
                    if not self._restore_active_search_page():
                        raise EdgeActionRequired(
                            "boss_search_session_expired",
                            "无法恢复原岗位采集页，请保持采集 Edge 打开后重试",
                        )
                    self._manual_gate(self.browser.page_state())
                    self.browser.open_auxiliary(url)
                else:
                    self._manual_gate(self._authentication_state())
                    self.browser.open(url)
                state = self.browser.page_state()
                self._manual_gate(state)
                fields = self.browser.detail_fields()
                # Recruiter activity is rendered asynchronously after the main
                # description. Poll the DOM briefly without refreshing or
                # navigating so a visible value is not persisted as unknown.
                activity_deadline = time.monotonic() + 3.0
                while not str(fields.get("hr_activity") or "").strip() and time.monotonic() < activity_deadline:
                    time.sleep(0.2)
                    latest = self.browser.detail_fields()
                    fields.update({key: value for key, value in latest.items() if value})
            except Exception as exc:
                keep_manual_tab = (
                    isinstance(exc, EdgeActionRequired)
                    and exc.code in _MANUAL_BROWSER_CODES
                )
                if preserve_search and not keep_manual_tab:
                    self.browser.close_auxiliary()
                self._close_for_error(exc)
                raise
            else:
                if preserve_search:
                    self.browser.close_auxiliary()
                    self.browser.detach_manual_login()
                else:
                    self.browser.close()
        text = str(fields.get("description") or "").strip()
        if not text:
            raise EdgeActionRequired("boss_parse_failed", "职位详情中未找到可复制的岗位描述")
        posted_label = str(fields.get("posted_label") or "").strip() or self._extract_posted_label(
            str(fields.get("body_text") or "")
        )
        parsed = parse_job_detail_text(
            str(fields.get("title") or "未命名职位"),
            text,
            company=fields.get("company"),
            city=fields.get("city"),
            source_url=url,
            posted_label=posted_label,
            hr_name=fields.get("hr_name"),
            hr_activity=fields.get("hr_activity"),
        )
        parsed = parsed.model_copy(
            update={"salary": decode_boss_obfuscated_text(fields.get("salary")) or None}
        )
        return parsed.model_dump(mode="json")

    @staticmethod
    def _extract_posted_label(text: str) -> str | None:
        patterns = (
            r"(?:发布于|发布时间[:：]?|更新于|更新时间[:：]?)\s*(刚刚(?:发布)?|今天|昨天|前天|\d+\s*(?:分钟|小时|天|周|个月|年)前|\d{4}(?:年|[-/.])\d{1,2}(?:月|[-/.])\d{1,2}(?:日)?(?:\s+\d{1,2}:\d{2})?)",
            r"(刚刚发布|今天|昨天|前天|\d+\s*(?:分钟|小时|天|周|个月|年)前)",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None

    def extract_post_time(self, *, url: str | None = None, text: str | None = None) -> dict[str, Any]:
        label = self._extract_posted_label(text or "") if text else None
        if label is None and url:
            detail = self.detail(url)
            label = detail.get("posted_at_label")
        posted_at, status = parse_posted_time(label)
        return {
            "posted_at": posted_at.isoformat() if posted_at else None,
            "posted_at_label": label if status == "known" else "发布时间未知",
            "method": "independent_dom_metadata_v1",
            "confidence": "rule_verified" if status == "known" else "unknown",
        }

    def snapshot(self, url: str) -> dict[str, Any]:
        url = validate_boss_url(url)
        with self._lock:
            try:
                self._manual_gate(self._authentication_state())
                self.browser.open(url)
                state = self.browser.page_state()
                self._manual_gate(state)
                html = self.browser.sanitized_html()
            except Exception as exc:
                self._close_for_error(exc)
                raise
            else:
                self.browser.close()
        if not html.strip():
            raise EdgeActionRequired("boss_snapshot_failed", "页面快照内容为空")
        digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
        self.config.snapshot_dir.mkdir(parents=True, exist_ok=True)
        target = self.config.snapshot_dir / f"boss-{digest[:20]}.html"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(html, encoding="utf-8")
        temporary.replace(target)
        return {
            "path": str(target.resolve()),
            "sha256": digest,
            "source_url": url,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "sanitized": True,
        }

    def logout(self) -> dict[str, Any]:
        with self._lock:
            self.browser.close_manual_login()
            if not self.browser.running:
                self.browser.open("about:blank")
            self.browser.clear_session()
        return {"status": "logged_out", "profile_dir": str(self.config.profile_dir.resolve())}

    def close(self) -> None:
        with self._lock:
            self.browser.close_manual_login()
            self.browser.close()


def default_edge_adapter() -> EdgeBossAdapter:
    data_root = Path(os.getenv("RESUME_AGENT_DATA_ROOT", "./data"))
    profile_dir = Path(
        os.getenv("RESUME_AGENT_EDGE_PROFILE_DIR", str(data_root / "edge-profile"))
    )
    return EdgeBossAdapter(
        EdgeAdapterConfig(
            profile_dir=profile_dir,
            snapshot_dir=data_root / "job-snapshots",
            edge_path=os.getenv("RESUME_AGENT_EDGE_PATH") or None,
        )
    )


__all__ = [
    "EdgeActionRequired",
    "EdgeAdapterConfig",
    "EdgeBossAdapter",
    "PlaywrightEdgeBrowser",
    "COMMON_CITY_CODES",
    "default_edge_adapter",
    "detect_edge_path",
    "blank_browser_page",
    "login_required",
    "normalize_city_code",
]
