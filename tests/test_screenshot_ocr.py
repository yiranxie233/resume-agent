from __future__ import annotations

import io
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import UploadFile
from PIL import Image

from app.core.schemas import ScreenshotConfirmRequest
from app.core.store import InMemoryStore
from app.main import confirm_job_screenshot, ocr_job_screenshot
from app.services.screenshot_parser import (
    MAX_SCREENSHOT_BYTES,
    PaddleScreenshotOCR,
    ScreenshotOCRError,
    extract_ocr_lines,
)


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (120, 60), "white").save(output, format="PNG")
    return output.getvalue()


class _FakeModel:
    @staticmethod
    def predict(image: object) -> list[dict[str, object]]:
        return [
            {
                "rec_texts": ["岗位职责", "负责 Python 智能体开发", "任职要求", "熟悉 FastAPI"],
                "rec_scores": [0.99, 0.95, 0.98, 0.94],
                "rec_boxes": [[0, 0, 20, 10], [0, 11, 80, 22], [0, 23, 30, 33], [0, 34, 60, 45]],
            }
        ]


class _FakeOCRService:
    @staticmethod
    def recognize(data: bytes) -> dict[str, object]:
        return {
            "text": "岗位职责\n负责 Python 智能体开发\n任职要求\n熟悉 FastAPI",
            "lines": [
                {"text": "岗位职责", "score": 0.99, "box": [0, 0, 20, 10]},
                {"text": "负责 Python 智能体开发", "score": 0.95, "box": [0, 11, 80, 22]},
            ],
            "model": "PP-OCRv6_medium",
            "image_format": "PNG",
            "width": 120,
            "height": 60,
        }


class ScreenshotOCRServiceTests(unittest.TestCase):
    def test_extracts_text_scores_and_boxes(self) -> None:
        lines = extract_ocr_lines(_FakeModel.predict(None))

        self.assertEqual(lines[1]["text"], "负责 Python 智能体开发")
        self.assertEqual(lines[1]["score"], 0.95)
        self.assertEqual(lines[1]["box"], [0, 11, 80, 22])

    def test_detector_batch_output_is_sorted_into_reading_order(self) -> None:
        lines = extract_ocr_lines(
            {
                "rec_texts": ["第二行", "第一行右", "第一行左"],
                "rec_scores": [0.9, 0.9, 0.9],
                "rec_boxes": [[0, 30, 80, 45], [120, 5, 200, 20], [0, 5, 80, 20]],
            }
        )

        self.assertEqual([item["text"] for item in lines], ["第一行左", "第一行右", "第二行"])

    def test_small_vertical_jitter_stays_on_the_same_visual_row(self) -> None:
        lines = extract_ocr_lines(
            {
                "rec_texts": ["第一行右", "第一行左", "第二行"],
                "rec_scores": [0.9, 0.9, 0.9],
                "rec_boxes": [[120, 6, 200, 22], [0, 4, 80, 20], [0, 35, 80, 51]],
            }
        )

        self.assertEqual([item["text"] for item in lines], ["第一行左", "第一行右", "第二行"])

    def test_recognize_uses_v6_medium_result_without_disk_file(self) -> None:
        service = PaddleScreenshotOCR()
        service._model = _FakeModel()

        result = service.recognize(_png_bytes())

        self.assertEqual(result["model"], "PP-OCRv6_medium")
        self.assertIn("负责 Python 智能体开发", result["text"])
        self.assertEqual(result["image_format"], "PNG")

    def test_invalid_and_oversized_images_are_rejected(self) -> None:
        service = PaddleScreenshotOCR()
        with self.assertRaises(ScreenshotOCRError) as invalid:
            service.recognize(b"not-an-image")
        self.assertEqual(invalid.exception.code, "invalid_image")

        with self.assertRaises(ScreenshotOCRError) as oversized:
            service.recognize(b"x" * (MAX_SCREENSHOT_BYTES + 1))
        self.assertEqual(oversized.exception.code, "image_too_large")

    def test_missing_local_medium_models_returns_setup_error(self) -> None:
        service = PaddleScreenshotOCR()
        missing = Path("Z:/definitely-missing-paddle-model")
        with (
            patch(
                "app.services.screenshot_parser._model_directories",
                return_value=(missing / "det", missing / "rec"),
            ),
            self.assertRaises(ScreenshotOCRError) as caught,
        ):
            service._load_model()
        self.assertEqual(caught.exception.code, "ocr_model_not_installed")


class ScreenshotOCRApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_ocr_requires_confirmation_before_job_is_created(self) -> None:
        app_state = SimpleNamespace(
            screenshot_ocr=_FakeOCRService(),
            pending_ocr={},
            pending_ocr_lock=threading.RLock(),
            store=InMemoryStore(),
            db_mirror=None,
            job_parse_confirmations={},
        )
        upload = UploadFile(filename="job.png", file=io.BytesIO(_png_bytes()))

        with patch("app.main.state", return_value=app_state):
            staged = await ocr_job_screenshot(upload, None)
            self.assertFalse(staged["job_created"])
            self.assertTrue(staged["needs_confirmation"])
            self.assertEqual(app_state.store.list_jobs(), [])

            confirmed = await confirm_job_screenshot(
                ScreenshotConfirmRequest(
                    ocr_id=staged["ocr_id"],
                    text=(staged["text"] + "\n技能要求：LangGraph"),
                    title="AI 智能体工程师",
                    company="示例公司",
                    city="深圳",
                    salary="20-30K",
                    hr_activity="在线",
                ),
                None,
            )

        self.assertEqual(confirmed["status"], "ready")
        self.assertTrue(confirmed["ocr_text_changed"])
        self.assertEqual(confirmed["job"]["hr_activity"], "在线")
        self.assertEqual(len(app_state.store.list_jobs()), 1)
        self.assertIn(staged["ocr_id"], app_state.pending_ocr)
        self.assertIn(
            "confirmed_response", app_state.pending_ocr[staged["ocr_id"]]
        )
        self.assertTrue(confirmed["model_fallback"])
        self.assertIn("无需再次上传截图", confirmed["model_fallback_message"])
        self.assertIn("负责 Python 智能体开发", confirmed["job"]["raw_text"])

        with patch("app.main.state", return_value=app_state):
            replayed = await confirm_job_screenshot(
                ScreenshotConfirmRequest(
                    ocr_id=staged["ocr_id"],
                    text=(staged["text"] + "\n技能要求：LangGraph"),
                ),
                None,
            )
        self.assertTrue(replayed["confirmation_replayed"])
        self.assertEqual(replayed["job_id"], confirmed["job_id"])
        self.assertEqual(len(app_state.store.list_jobs()), 1)

    async def test_expired_backend_stage_recovers_from_client_ocr_text(self) -> None:
        app_state = SimpleNamespace(
            pending_ocr={},
            pending_ocr_lock=threading.RLock(),
            store=InMemoryStore(),
            db_mirror=None,
            job_parse_confirmations={},
        )

        with patch("app.main.state", return_value=app_state):
            confirmed = await confirm_job_screenshot(
                ScreenshotConfirmRequest(
                    ocr_id="ocr_expired_after_restart",
                    text="岗位职责\n负责智能体服务开发\n任职要求\n熟悉 Python",
                ),
                None,
            )

        self.assertEqual(confirmed["status"], "ready")
        self.assertTrue(confirmed["model_fallback"])
        self.assertIn("ocr_expired_after_restart", app_state.pending_ocr)
        self.assertEqual(len(app_state.store.list_jobs()), 1)

    async def test_oversized_upload_is_rejected_before_ocr(self) -> None:
        app_state = SimpleNamespace(
            screenshot_ocr=_FakeOCRService(),
            pending_ocr={},
            pending_ocr_lock=threading.RLock(),
        )
        upload = UploadFile(
            filename="large.png",
            file=io.BytesIO(b"x" * (MAX_SCREENSHOT_BYTES + 1)),
        )
        with (
            patch("app.main.state", return_value=app_state),
            self.assertRaises(Exception) as caught,
        ):
            await ocr_job_screenshot(upload, None)
        self.assertEqual(getattr(caught.exception, "status_code", None), 413)


if __name__ == "__main__":
    unittest.main()
