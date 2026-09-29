from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.core.config import Settings


class SettingsRuntimeTests(unittest.TestCase):
    def test_runtime_token_file_matches_each_process_token(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = Settings(data_root=root, internal_token="first-token").ensure_runtime()
            self.assertEqual((root / ".internal-token").read_text(encoding="utf-8"), first.internal_token)

            second = Settings(data_root=root, internal_token="second-token").ensure_runtime()
            self.assertEqual((root / ".internal-token").read_text(encoding="utf-8"), second.internal_token)
            self.assertNotEqual(first.internal_token, second.internal_token)

    def test_generated_token_reuses_existing_runtime_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner = Settings(data_root=root, internal_token="running-api-token").ensure_runtime()

            helper = Settings(data_root=root).ensure_runtime()

            self.assertEqual(helper.internal_token, owner.internal_token)
            self.assertEqual(
                (root / ".internal-token").read_text(encoding="utf-8"),
                "running-api-token",
            )


if __name__ == "__main__":
    unittest.main()
