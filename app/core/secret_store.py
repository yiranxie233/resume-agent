"""Machine-local encrypted credential storage.

Model API keys use AES-GCM with a separately permission-restricted local key.
DPAPI ciphertext written by older builds remains readable when Windows permits
it, but new values deliberately avoid DPAPI: services launched from a managed
terminal can receive a different logon token after restart, making otherwise
valid DPAPI ciphertext impossible to decrypt.  The persisted JSON always
contains ciphertext only.
"""
from __future__ import annotations

import base64
import ctypes
import json
import os
import secrets
import stat
import threading
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path


class SecretStoreUnavailable(RuntimeError):
    pass


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _blob(value: bytes) -> tuple[_DataBlob, ctypes.Array[ctypes.c_char]]:
    buffer = ctypes.create_string_buffer(value)
    return (
        _DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))),
        buffer,
    )


def _dpapi_transform(value: bytes, *, decrypt: bool) -> bytes:
    if os.name != "nt":
        raise SecretStoreUnavailable("持久化 API Key 当前仅支持 Windows DPAPI")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    source, source_buffer = _blob(value)
    entropy, entropy_buffer = _blob(b"resume-agent:model-credentials:v1")
    output = _DataBlob()
    flags = 0x01  # CRYPTPROTECT_UI_FORBIDDEN
    if decrypt:
        function = crypt32.CryptUnprotectData
        function.argtypes = [
            ctypes.POINTER(_DataBlob),
            ctypes.POINTER(ctypes.c_wchar_p),
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(_DataBlob),
        ]
        description = ctypes.c_wchar_p()
        ok = function(
            ctypes.byref(source),
            ctypes.byref(description),
            ctypes.byref(entropy),
            None,
            None,
            flags,
            ctypes.byref(output),
        )
    else:
        function = crypt32.CryptProtectData
        function.argtypes = [
            ctypes.POINTER(_DataBlob),
            ctypes.c_wchar_p,
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(_DataBlob),
        ]
        ok = function(
            ctypes.byref(source),
            "Resume Agent model credential",
            ctypes.byref(entropy),
            None,
            None,
            flags,
            ctypes.byref(output),
        )
    # Keep buffers alive until the native call above has returned.
    _ = source_buffer, entropy_buffer
    if not ok:
        raise SecretStoreUnavailable(f"Windows DPAPI 操作失败：{ctypes.get_last_error()}")
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        kernel32.LocalFree(output.pbData)


def _protect(value: bytes) -> bytes:
    return _dpapi_transform(value, decrypt=False)


def _unprotect(value: bytes) -> bytes:
    return _dpapi_transform(value, decrypt=True)


class EncryptedSecretStore:
    """Small atomic ciphertext store keyed by model profile ID."""

    def __init__(
        self,
        path: Path,
        *,
        protect: Callable[[bytes], bytes] | None = None,
        unprotect: Callable[[bytes], bytes] | None = None,
    ) -> None:
        self.path = Path(path)
        self.key_path = self.path.with_suffix(".key")
        self._protect = protect or self._protect_default
        self._unprotect = unprotect or self._unprotect_default
        self._lock = threading.RLock()

    def _local_key(self) -> bytes:
        if self.key_path.is_file():
            try:
                value = base64.b64decode(
                    self.key_path.read_text(encoding="ascii"), validate=True
                )
            except (OSError, ValueError) as exc:
                raise SecretStoreUnavailable("本机凭据加密密钥无法读取") from exc
            if len(value) != 32:
                raise SecretStoreUnavailable("本机凭据加密密钥格式无效")
            return value
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        value = secrets.token_bytes(32)
        temporary = self.key_path.with_suffix(self.key_path.suffix + ".tmp")
        temporary.write_text(base64.b64encode(value).decode("ascii"), encoding="ascii")
        try:
            os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        temporary.replace(self.key_path)
        return value

    def _protect_default(self, value: bytes) -> bytes:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise SecretStoreUnavailable(
                "未安装 cryptography，无法加密保存 API Key"
            ) from exc
        nonce = secrets.token_bytes(12)
        ciphertext = AESGCM(self._local_key()).encrypt(
            nonce, value, b"resume-agent:model-credentials:v1"
        )
        return b"AESGCM1" + nonce + ciphertext

    def _unprotect_default(self, value: bytes) -> bytes:
        if value.startswith(b"DPAPI1"):
            return _unprotect(value[6:])
        if value.startswith(b"AESGCM1"):
            try:
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM

                payload = value[7:]
                return AESGCM(self._local_key()).decrypt(
                    payload[:12],
                    payload[12:],
                    b"resume-agent:model-credentials:v1",
                )
            except Exception as exc:
                raise SecretStoreUnavailable("本机保存的 API Key 无法解密，请重新填写") from exc
        raise SecretStoreUnavailable("不支持的本机凭据加密格式")

    def _read(self) -> dict[str, str]:
        if not self.path.is_file():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SecretStoreUnavailable("本机加密凭据文件无法读取") from exc
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, dict):
            raise SecretStoreUnavailable("本机加密凭据文件格式无效")
        return {str(key): str(value) for key, value in entries.items()}

    def _write(self, entries: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "entries": entries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        temporary.replace(self.path)

    def set(self, profile_id: str, secret: str) -> None:
        if not str(profile_id).strip() or not secret:
            raise ValueError("profile_id and secret are required")
        ciphertext = base64.b64encode(self._protect(secret.encode("utf-8"))).decode("ascii")
        with self._lock:
            entries = self._read()
            entries[str(profile_id)] = ciphertext
            self._write(entries)
        # A successful write is not enough: fail the request immediately if
        # this process cannot recover the value it just saved.
        if self.get(profile_id) != secret:
            raise SecretStoreUnavailable("本机保存的 API Key 回读校验失败")

    def get(self, profile_id: str) -> str | None:
        with self._lock:
            ciphertext = self._read().get(str(profile_id))
        if not ciphertext:
            return None
        try:
            protected = base64.b64decode(ciphertext, validate=True)
            return self._unprotect(protected).decode("utf-8")
        except (ValueError, UnicodeDecodeError, SecretStoreUnavailable) as exc:
            raise SecretStoreUnavailable("本机保存的 API Key 无法解密，请重新填写") from exc

    def has(self, profile_id: str) -> bool:
        with self._lock:
            return str(profile_id) in self._read()

    def delete(self, profile_id: str) -> bool:
        with self._lock:
            entries = self._read()
            removed = entries.pop(str(profile_id), None) is not None
            if removed:
                self._write(entries)
            return removed
