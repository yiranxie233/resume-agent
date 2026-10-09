from __future__ import annotations

import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.model_gateway import CredentialStore
from app.core.secret_store import EncryptedSecretStore
from app.main import (
    _materialize_model_credential,
    create_credential_session,
    delete_stored_credential,
    stored_credential_status,
)
from app.services.backup_service import create_backup


class EncryptedSecretStoreTests(unittest.TestCase):
    def test_secret_is_encrypted_and_round_trips_without_plaintext(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials" / "model-secrets.json"
            store = EncryptedSecretStore(
                path,
                protect=lambda value: bytes(item ^ 0xA5 for item in value),
                unprotect=lambda value: bytes(item ^ 0xA5 for item in value),
            )

            store.set("chat-profile", "sk-private-value")

            self.assertTrue(store.has("chat-profile"))
            self.assertEqual(store.get("chat-profile"), "sk-private-value")
            self.assertNotIn("sk-private-value", path.read_text(encoding="utf-8"))
            self.assertTrue(store.delete("chat-profile"))
            self.assertIsNone(store.get("chat-profile"))

    def test_saved_secret_can_rebuild_handle_after_process_state_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials" / "model-secrets.json"

            def protect(value: bytes) -> bytes:
                return bytes(item ^ 0xA5 for item in value)

            first_store = EncryptedSecretStore(
                path,
                protect=protect,
                unprotect=protect,
            )
            first_store.set("chat-profile", "sk-private-value")

            restarted_state = SimpleNamespace(
                secret_store=EncryptedSecretStore(
                    path,
                    protect=protect,
                    unprotect=protect,
                ),
                credentials=CredentialStore(),
                credential_scopes={"model": set(), "github": set()},
            )
            with patch("app.main.state", return_value=restarted_state):
                handle_id = _materialize_model_credential("chat-profile")

            self.assertIsNotNone(handle_id)
            self.assertTrue(
                restarted_state.credentials.has(
                    str(handle_id),
                    scope="model:chat-profile",
                )
            )
            self.assertIn(handle_id, restarted_state.credential_scopes["model"])

    def test_materialize_reuses_profile_handle_without_rereading_secret(self) -> None:
        credentials = CredentialStore()
        handle = credentials.put(
            "sk-live-value",
            scope="model:chat-profile",
            ttl_seconds=300,
        )
        app_state = SimpleNamespace(
            secret_store=SimpleNamespace(
                get=lambda profile_id: (_ for _ in ()).throw(
                    AssertionError("active handle should be reused")
                )
            ),
            credentials=credentials,
            credential_scopes={"model": {handle.handle_id}, "github": set()},
            model_credential_handles={"chat-profile": handle.handle_id},
            credential_registry_lock=threading.RLock(),
        )

        with patch("app.main.state", return_value=app_state):
            restored = _materialize_model_credential("chat-profile")

        self.assertEqual(restored, handle.handle_id)

    def test_default_store_uses_restart_safe_aes_ciphertext(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials" / "model-secrets.json"
            first = EncryptedSecretStore(path)

            first.set("chat-profile", "sk-restart-safe")

            payload = path.read_text(encoding="utf-8")
            self.assertNotIn("sk-restart-safe", payload)
            second = EncryptedSecretStore(path)
            self.assertEqual(second.get("chat-profile"), "sk-restart-safe")

    def test_backup_excludes_encrypted_credentials_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "credentials").mkdir()
            (root / "credentials" / "model-secrets.json").write_text(
                "ciphertext", encoding="utf-8"
            )
            (root / "tasks").mkdir()
            (root / "tasks" / "safe.json").write_text("{}", encoding="utf-8")

            result = create_backup(root)

            with zipfile.ZipFile(result["archive"]) as archive:
                self.assertIn("tasks/safe.json", archive.namelist())
                self.assertNotIn("credentials/model-secrets.json", archive.namelist())


class StoredCredentialApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_status_requires_ciphertext_to_be_decryptable(self) -> None:
        broken_store = SimpleNamespace(
            has=lambda profile_id: True,
            get=lambda profile_id: (_ for _ in ()).throw(
                Exception("should be normalized below")
            ),
        )

        def unreadable(_: str) -> str:
            from app.core.secret_store import SecretStoreUnavailable

            raise SecretStoreUnavailable("旧密钥无法解密")

        broken_store.get = unreadable
        app_state = SimpleNamespace(secret_store=broken_store)
        with patch("app.main.state", return_value=app_state):
            result = await stored_credential_status("chat-profile", None)

        self.assertFalse(result["stored"])
        self.assertFalse(result["usable"])
        self.assertTrue(result["needs_reentry"])

    async def test_saving_new_key_rotates_only_the_process_handle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials" / "model-secrets.json"

            def transform(value: bytes) -> bytes:
                return bytes(item ^ 0xA5 for item in value)

            app_state = SimpleNamespace(
                secret_store=EncryptedSecretStore(
                    path,
                    protect=transform,
                    unprotect=transform,
                ),
                credentials=CredentialStore(),
                credential_scopes={"model": set(), "github": set()},
                model_credential_handles={},
                credential_registry_lock=threading.RLock(),
            )
            with patch("app.main.state", return_value=app_state):
                first = await create_credential_session(
                    {
                        "profile_id": "chat-profile",
                        "api_key": "sk-first",
                        "persist": True,
                        "ttl_seconds": 300,
                    },
                    None,
                )
                second = await create_credential_session(
                    {
                        "profile_id": "chat-profile",
                        "api_key": "sk-second",
                        "persist": True,
                        "ttl_seconds": 300,
                    },
                    None,
                )

            self.assertFalse(app_state.credentials.has(first["credential_handle_id"]))
            self.assertEqual(
                app_state.credentials.get(
                    second["credential_handle_id"],
                    scope="model:chat-profile",
                ),
                "sk-second",
            )
            self.assertEqual(app_state.secret_store.get("chat-profile"), "sk-second")
            self.assertEqual(
                app_state.model_credential_handles["chat-profile"],
                second["credential_handle_id"],
            )

    async def test_deleting_saved_key_revokes_active_handle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials" / "model-secrets.json"

            def transform(value: bytes) -> bytes:
                return bytes(item ^ 0xA5 for item in value)

            secret_store = EncryptedSecretStore(
                path,
                protect=transform,
                unprotect=transform,
            )
            secret_store.set("chat-profile", "sk-value")
            credentials = CredentialStore()
            handle = credentials.put(
                "sk-value",
                scope="model:chat-profile",
                ttl_seconds=300,
            )
            app_state = SimpleNamespace(
                secret_store=secret_store,
                credentials=credentials,
                credential_scopes={"model": {handle.handle_id}, "github": set()},
                model_credential_handles={"chat-profile": handle.handle_id},
                credential_registry_lock=threading.RLock(),
            )

            with patch("app.main.state", return_value=app_state):
                result = await delete_stored_credential("chat-profile", None)

            self.assertTrue(result["removed"])
            self.assertFalse(credentials.has(handle.handle_id))
            self.assertNotIn("chat-profile", app_state.model_credential_handles)
            self.assertIsNone(secret_store.get("chat-profile"))


if __name__ == "__main__":
    unittest.main()
