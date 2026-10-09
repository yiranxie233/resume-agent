from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app


class ResumeSourceApiTests(unittest.TestCase):
    def test_uploaded_pdf_can_be_loaded_again_for_visual_preview(self) -> None:
        import fitz  # type: ignore

        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "Resume preview source")
        source = document.tobytes()
        document.close()

        with TestClient(app) as client:
            headers = {
                "X-Resume-Agent-Token": app.state.resume.settings.internal_token
            }
            uploaded = client.post(
                "/api/resumes/upload",
                headers=headers,
                files={"file": ("resume.pdf", source, "application/pdf")},
            )
            uploaded.raise_for_status()
            resume_id = uploaded.json()["resume_id"]

            preview = client.get(
                f"/api/resumes/{resume_id}/source",
                headers=headers,
            )

        preview.raise_for_status()
        self.assertEqual(preview.headers["content-type"], "application/pdf")
        self.assertEqual(preview.content, source)


if __name__ == "__main__":
    unittest.main()
