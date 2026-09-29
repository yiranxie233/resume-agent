"""Visible Edge/Playwright adapter used only by explicit BOSS MCP calls.

The browser is launched lazily with an isolated persistent profile.  This
module never attempts to solve login challenges, CAPTCHAs, sliders, or access
restrictions; those states are returned to the caller as manual gates.
"""
from __future__ import annotations

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

    def open_manual_login(self, url: str) -> None: ...

    def adopt_manual_login(self) -> None: ...

    def detach_manual_login(self) -> None: ...

    def close_manual_login(self) -> None: ...

    def page_state(self) -> dict[str, Any]: ...

    def search_items(self) -> list[dict[str, Any]]: ...

    def load_more_items(self, seen_urls: set[str]) -> dict[str, Any]: ...

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
            from websockets.sync.client import connect  # type: ignore
        except ImportError as exc:
            raise EdgeActionRequired(
                "playwright_not_installed",
                "缺少 Edge 接管依赖，请重新安装项目 mcp 依赖",
            ) from exc
        self.timeout_seconds = max(timeout_ms / 1000, 0.5)
        self._connection = connect(
            endpoint,
            origin=None,
            open_timeout=self.timeout_seconds,
            close_timeout=1,
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
                message = self._connection.recv(timeout=remaining)
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

    def new_page(self) -> "_DirectCDPPage":
        target_id = str(
            self.call("Target.createTarget", {"url": "about:blank"}).get("targetId") or ""
        )
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

    def goto(self, url: str, **_: Any) -> None:
        self._call("Page.enable")
        result = self._call("Page.navigate", {"url": url})
        if result.get("errorText"):
            raise RuntimeError(str(result["errorText"]))
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

    def close(self) -> None:
        if self._closed:
            return
        self._controller.call("Target.closeTarget", {"targetId": self._target_id})
        self._closed = True


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

    @staticmethod
    def _debug_endpoint_for_port(port: int, *, timeout: float = 0.2) -> str | None:
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
        self._debug_port_record_file.unlink(missing_ok=True)
        self._devtools_port_file.unlink(missing_ok=True)
        return False

    @staticmethod
    def _available_debug_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])

    def open_manual_login(self, url: str) -> None:
        if self.manual_login_running:
            return
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
        self._devtools_port_file.unlink(missing_ok=True)
        self._debug_port_record_file.unlink(missing_ok=True)
        try:
            self._manual_login_process = subprocess.Popen(
                [
                    self.edge_path,
                    f"--user-data-dir={self.profile_dir}",
                    "--remote-debugging-address=127.0.0.1",
                    f"--remote-debugging-port={debug_port}",
                    "--new-window",
                    "--no-first-run",
                    "--disable-background-mode",
                    url,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            self._manual_debug_port = debug_port
            deadline = time.monotonic() + min(max(self.timeout_ms / 1000, 5.0), 15.0)
            while time.monotonic() < deadline:
                endpoint = self._debug_endpoint_for_port(debug_port)
                if endpoint is not None:
                    self._manual_debug_endpoint = endpoint
                    self._debug_port_record_file.write_text(str(debug_port), encoding="utf-8")
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

    def _connect_cdp(self, endpoint: str) -> _DirectCDPController:
        return _DirectCDPController(endpoint, timeout_ms=self.timeout_ms)

    def adopt_manual_login(self) -> None:
        """Attach a local Python CDP client to the visible Edge window."""

        if self.running:
            return
        # Re-resolve through the live HTTP endpoint on every attach.  A cached
        # WebSocket URL is not evidence that the browser still exists.
        endpoint = self._discover_debug_endpoint()
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
            boss_pages = [page for page in pages if "zhipin.com" in str(page.url or "")]
            page = boss_pages[-1] if boss_pages else pages[-1]
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

    def close_manual_login(self) -> None:
        process = self._manual_login_process
        self._manual_login_process = None
        self._manual_debug_port = None
        self._manual_debug_endpoint = None
        self._devtools_port_file.unlink(missing_ok=True)
        self._debug_port_record_file.unlink(missing_ok=True)
        if process is None or process.poll() is not None:
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
                args=["--start-maximized", "--disable-background-mode"],
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
                self.close()
                if network_error:
                    raise EdgeActionRequired(
                        "boss_network_unavailable",
                        "Edge 无法访问 BOSS，请检查系统网络或代理是否允许访问 zhipin.com",
                    ) from exc
                raise EdgeActionRequired(
                    "boss_navigation_failed",
                    "BOSS 页面打开失败，浏览器会话已重置，请重新搜索",
                ) from exc

    def open_auxiliary(self, url: str) -> None:
        """Open one temporary detail tab while preserving the search page."""

        self._ensure_started()
        self.close_auxiliary()
        parent = self._page
        page = self._context.new_page()
        page.set_default_timeout(self.timeout_ms)
        self._auxiliary_parent_page = parent
        self._page = page
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
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

    def select_search_page(self, target_url: str) -> bool:
        """Select the retained search tab without navigating or refreshing it."""

        if self._context is None:
            return False
        try:
            candidates = [
                page
                for page in self._context.pages
                if not page.is_closed()
                and same_job_search_page(str(page.url or ""), target_url)
            ]
            if not candidates:
                return False
            self._page = candidates[-1]
            self._page.set_default_timeout(self.timeout_ms)
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
                    """
                    () => {
                      const selectors = [
                        'a[ka="header-login"]',
                        'a[href*="/web/user/?ka=header-login"]',
                        '.header-login-btn',
                        '.btn-login',
                        '[data-ka="header-login"]'
                      ];
                      return selectors.some((selector) => Array.from(document.querySelectorAll(selector)).some((node) => {
                        const style = window.getComputedStyle(node);
                        const rect = node.getBoundingClientRect();
                        const visible = style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
                        return visible && /登录|注册/.test(node.textContent || '');
                      }));
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
        return """
        () => Array.from(document.querySelectorAll('.job-card-box, .job-card-wrapper, .job-list-box > li')).map((card) => {
          const text = (...selectors) => {
            for (const selector of selectors) {
              const value = card.querySelector(selector)?.textContent?.trim();
              if (value) return value;
            }
            return null;
          };
          const anchor = card.querySelector('a.job-card-left, a[href*="/job_detail/"]');
          const activityNode = card.querySelector('.boss-active-time, .boss-status');
          const online = Boolean(card.querySelector('.boss-online-icon'));
          return {
            title: text('.job-name'), company: text('.company-name', '.boss-name'),
            city: text('.job-area', '.company-location'),
            salary: text('.salary', '.job-salary'), url: anchor ? new URL(anchor.href, location.href).href : null,
            tags: Array.from(card.querySelectorAll('.tag-list li, .job-card-footer li')).map(x => x.textContent.trim()).filter(Boolean),
            hr_activity: online ? '在线' : (activityNode?.textContent?.trim() || '活跃时间待解析')
          };
        }).filter(item => item.title && item.url)
        """

    def _evaluate_search_items(self) -> list[dict[str, Any]]:
        value = self._page.evaluate(self._search_items_script())
        return value if isinstance(value, list) else []

    def _recover_live_search_page(self) -> bool:
        """Follow a BOSS-created replacement tab without causing navigation."""

        if self._context is None:
            return False
        try:
            candidates = [
                page
                for page in self._context.pages
                if not page.is_closed()
                and urllib.parse.urlparse(str(page.url or "")).path.rstrip("/")
                == "/web/geek/jobs"
            ]
            if not candidates:
                return False
            self._page = candidates[-1]
            self._page.set_default_timeout(self.timeout_ms)
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

    def load_more_items(self, seen_urls: set[str]) -> dict[str, Any]:
        """Progressively scroll until new cards appear or the page explicitly ends.

        A timeout is deliberately reported as ``load_pending`` instead of
        ``exhausted``. BOSS often delays its infinite-list request, and treating
        one slow response as the end would incorrectly disable pagination.
        """

        self._ensure_started()
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
                if any(str(item.get("url") or "") not in seen_urls for item in value):
                    return {"items": value, "exhausted": False, "load_pending": False}
                now = time.monotonic()
                if now >= next_scroll_at:
                    scroll_state = self._page.evaluate(
                        """
                        () => {
                          const cards = Array.from(document.querySelectorAll(
                            '.job-card-box, .job-card-wrapper, .job-list-box > li'
                          ));
                          cards.at(-1)?.scrollIntoView({block: 'end', behavior: 'instant'});
                          const root = document.scrollingElement || document.documentElement;
                          window.scrollTo({top: root.scrollHeight, behavior: 'instant'});
                          document.querySelectorAll('.job-list-container, .rec-job-list').forEach((node) => {
                            if (node.scrollHeight > node.clientHeight) node.scrollTop = node.scrollHeight;
                          });
                          const bodyText = document.body?.innerText || '';
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
                          const explicitEnd = /没有更多(?:职位|岗位|数据)?|暂无更多(?:职位|岗位|数据)?|已经到底了/.test(bodyText) || visibleEndNode;
                          return {
                            card_count: cards.length,
                            scroll_height: root.scrollHeight,
                            at_bottom: root.scrollTop + window.innerHeight >= root.scrollHeight - 8,
                            explicit_end: explicitEnd
                          };
                        }
                        """
                    )
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
          const activity = () => {
            const selectors = [
              '.boss-active-time', '.job-boss-info .boss-active-time',
              '.job-boss-info .boss-status', '.boss-info-attr .boss-status',
              '[class*="active-time"]', '[class*="boss-online"]'
            ];
            const pattern = /(刚刚活跃|当前在线|在线|今日活跃|今天活跃|昨日活跃|昨天活跃|\d+\s*(?:分钟|小时|天|日)内?活跃|本周活跃|本月活跃|近(?:一)?周活跃|近(?:一)?月活跃|半年内活跃)/;
            for (const selector of selectors) {
              for (const node of document.querySelectorAll(selector)) {
                const value = (node.textContent || '').trim();
                const match = value.match(pattern);
                if (match) return match[1] === '当前在线' ? '在线' : match[1];
              }
            }
            for (const selector of ['.job-boss-info', '.boss-info', '.job-detail-boss']) {
              const value = document.querySelector(selector)?.innerText || '';
              const match = value.match(pattern);
              if (match) return match[1] === '当前在线' ? '在线' : match[1];
            }
            return null;
          };
          return {
            title: text(['.name h1', '.job-banner .name', 'h1']),
            company: text(['.company-info .name', '.company-name']),
            city: text(['.job-address .location-address', '.job-banner .text-desc']),
            salary: text(['.salary']),
            description: text(['.job-sec-text', '.job-detail-section .text', '.job-detail']),
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
        self._seen_search_urls: set[str] = set()

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
            if self.browser.manual_login_running:
                self.browser.detach_manual_login()
            else:
                self.browser.close()
                self.browser.open_manual_login(BOSS_HOME_URL)
            raise EdgeActionRequired("boss_login_required", "请在可见 Edge 独立 profile 中手动登录 BOSS 后再重试")

    def _authentication_state(self, initial_url: str = BOSS_HOME_URL) -> dict[str, Any]:
        if not self.browser.running:
            # Always start BOSS in a normal Edge process and attach over the
            # loopback CDP endpoint.  Playwright's persistent-context launcher
            # adds automation startup switches that can make BOSS repeatedly
            # reload the page on later searches.
            if not self.browser.manual_login_running:
                self.browser.open_manual_login(initial_url)
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
        if isinstance(exc, EdgeActionRequired) and exc.code in _MANUAL_BROWSER_CODES:
            return
        self.browser.close()
        if isinstance(exc, EdgeActionRequired) and exc.code in {
            "boss_navigation_failed",
            "boss_page_unavailable",
            "edge_session_closed",
        }:
            try:
                self.browser.open_manual_login(BOSS_HOME_URL)
            except EdgeActionRequired:
                pass

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
                    if not self.browser.select_search_page(search_url):
                        raise EdgeActionRequired(
                            "boss_search_session_expired",
                            "当前 Edge 页面已离开该岗位搜索，请重新搜索",
                        )
                    state = self.browser.page_state()
                    self._manual_gate(state)
                    if not same_job_search_page(str(state.get("url") or ""), search_url):
                        raise EdgeActionRequired(
                            "boss_search_session_expired",
                            "当前 Edge 页面已离开该岗位搜索，请重新搜索",
                        )
                    load_result = self.browser.load_more_items(set(self._seen_search_urls))
                    if isinstance(load_result, dict):
                        all_items = list(load_result.get("items") or [])
                        exhausted = bool(load_result.get("exhausted"))
                        load_pending = bool(load_result.get("load_pending"))
                    else:  # compatibility with external/test browser ports
                        all_items = list(load_result)
                    raw_items = [
                        item
                        for item in all_items
                        if str(item.get("url") or "") not in self._seen_search_urls
                    ][:limit]
                else:
                    # Reuse the retained, authenticated collection window.
                    # Closing it here made every repeated submit launch a new
                    # Edge process and could send BOSS through another redirect
                    # or login cycle.  ``_authentication_state`` reconnects to
                    # the existing CDP endpoint; navigation happens only when
                    # the requested title/city really changed.
                    self._active_search_key = None
                    self._seen_search_urls.clear()
                    state = self._authentication_state(search_url)
                    self._manual_gate(state)
                    if not same_job_search_page(str(state.get("url") or ""), search_url):
                        self.browser.open(search_url)
                    state = self.browser.page_state()
                    self._manual_gate(state)
                    raw_items = self.browser.search_items()[:limit]
                    if not raw_items:
                        self._manual_gate(self.browser.page_state())
            except Exception as exc:
                self._close_for_error(exc)
                raise
            else:
                for item in raw_items:
                    value = str(item.get("url") or "")
                    if value:
                        self._seen_search_urls.add(value)
                self._active_search_key = search_key
                self.browser.detach_manual_login()
        items: list[dict[str, Any]] = []
        for item in raw_items:
            try:
                url = validate_boss_url(str(item.get("url") or ""))
            except ValueError:
                continue
            items.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "company": item.get("company"),
                    "city": item.get("city"),
                    "salary": decode_boss_obfuscated_text(item.get("salary")) or None,
                    "skills": [str(value) for value in item.get("tags", []) if str(value).strip()],
                    "hr_activity": str(item.get("hr_activity") or "活跃时间待解析").strip(),
                    "posted_at": None,
                    "posted_at_label": "发布时间未知",
                    "url": url,
                }
            )
        return {
            "items": items,
            "next_cursor": (
                None
                if exhausted or not self.browser.manual_login_running
                else (str(page + 1) if items else (str(page) if load_pending else None))
            ),
            "page": page,
            "exhausted": exhausted,
            "load_pending": load_pending,
            "requested_city": city,
            "city_code": resolved_city_code,
            "city_resolution": city_resolution,
        }

    def close_browser(self) -> dict[str, Any]:
        with self._lock:
            self.browser.close()
            self._active_search_key = None
            self._seen_search_urls.clear()
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
                self._manual_gate(self.browser.page_state())
                self.browser.open_user_tab(url)
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
                self._active_search_key is not None and self.browser.manual_login_running
            )
            try:
                self._manual_gate(self._authentication_state())
                if preserve_search:
                    self.browser.open_auxiliary(url)
                else:
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
