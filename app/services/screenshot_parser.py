"""Local screenshot OCR backed by PaddleOCR v6 medium models.

The model is loaded lazily because importing and initializing Paddle is
expensive. Images are validated in memory and are never written to disk.
"""
from __future__ import annotations

import io
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MAX_SCREENSHOT_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}
OCR_MODEL_NAME = "PP-OCRv6_medium"


class ScreenshotOCRError(ValueError):
    """A safe, machine-readable OCR failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _official_model_root() -> Path:
    configured = os.getenv("PADDLEX_HOME", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".paddlex"


def _model_directories() -> tuple[Path, Path]:
    root = _official_model_root() / "official_models"
    return root / "PP-OCRv6_medium_det", root / "PP-OCRv6_medium_rec"


def _validate_image(data: bytes) -> tuple[Any, str]:
    if not data:
        raise ScreenshotOCRError("empty_image", "截图文件为空")
    if len(data) > MAX_SCREENSHOT_BYTES:
        raise ScreenshotOCRError("image_too_large", "岗位截图不能超过 10 MB")
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - guarded by environment checks
        raise ScreenshotOCRError("ocr_dependency_missing", "缺少 Pillow，请安装 OCR 依赖") from exc

    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = str(probe.format or "").upper()
            width, height = probe.size
            probe.verify()
        if image_format not in ALLOWED_IMAGE_FORMATS:
            raise ScreenshotOCRError(
                "unsupported_image_type", "仅支持 PNG、JPG/JPEG、WEBP 和 BMP 岗位截图"
            )
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise ScreenshotOCRError(
                "image_dimensions_too_large", "截图尺寸无效或像素总数超过 4000 万"
            )
        with Image.open(io.BytesIO(data)) as source:
            image = source.convert("RGB")
            image.load()
        return image, image_format
    except ScreenshotOCRError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise ScreenshotOCRError("invalid_image", "无法读取截图，请选择有效的图片文件") from exc


def _result_mapping(value: Any) -> dict[str, Any]:
    """Normalize PaddleX result objects across compatible 3.x releases."""

    candidates = [value]
    for name in ("json", "res"):
        try:
            candidates.append(getattr(value, name))
        except (AttributeError, TypeError, ValueError):
            continue
    for candidate in candidates:
        if callable(candidate):
            try:
                candidate = candidate()
            except (AttributeError, TypeError, ValueError):
                continue
        if isinstance(candidate, Mapping):
            payload = dict(candidate)
            nested = payload.get("res")
            if isinstance(nested, Mapping):
                return dict(nested)
            return payload
    return {}


def _json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (AttributeError, TypeError, ValueError):
            return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def extract_ocr_lines(results: Any) -> list[dict[str, Any]]:
    """Extract ordered text, confidence and boxes from PaddleOCR predictions."""

    values = results if isinstance(results, (list, tuple)) else [results]
    lines: list[dict[str, Any]] = []
    for result in values:
        payload = _result_mapping(result)
        texts_value = payload.get("rec_texts")
        scores_value = payload.get("rec_scores")
        boxes_value = payload.get("rec_boxes")
        if boxes_value is None:
            boxes_value = payload.get("rec_polys")
        texts = list(texts_value) if texts_value is not None else []
        scores = list(scores_value) if scores_value is not None else []
        boxes = list(boxes_value) if boxes_value is not None else []
        for index, text in enumerate(texts):
            normalized = str(text or "").strip()
            if not normalized:
                continue
            score: float | None = None
            if index < len(scores):
                try:
                    score = round(float(scores[index]), 6)
                except (TypeError, ValueError):
                    score = None
            lines.append(
                {
                    "text": normalized,
                    "score": score,
                    "box": _json_value(boxes[index]) if index < len(boxes) else None,
                }
            )
    return lines


class PaddleScreenshotOCR:
    """Thread-safe lazy PaddleOCR v6 medium adapter."""

    def __init__(self) -> None:
        self._model: Any = None
        self._lock = threading.RLock()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        detection_dir, recognition_dir = _model_directories()
        if not detection_dir.is_dir() or not recognition_dir.is_dir():
            raise ScreenshotOCRError(
                "ocr_model_not_installed",
                "未检测到 PP-OCRv6 medium 本地模型，请先下载检测与识别模型后重新尝试",
            )
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ScreenshotOCRError(
                "ocr_dependency_missing", "缺少 PaddleOCR/PaddlePaddle，请安装项目 OCR 依赖"
            ) from exc
        try:
            self._model = PaddleOCR(
                text_detection_model_name="PP-OCRv6_medium_det",
                text_recognition_model_name="PP-OCRv6_medium_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
                # PaddlePaddle 3.3 on Windows can raise
                # ConvertPirAttribute2RuntimeAttribute for PP-OCRv6 through
                # oneDNN. The regular CPU kernels are stable for this local UI.
                enable_mkldnn=False,
            )
        except Exception as exc:
            raise ScreenshotOCRError("ocr_model_load_failed", "PaddleOCR 模型加载失败") from exc
        return self._model

    def recognize(self, data: bytes) -> dict[str, Any]:
        image, image_format = _validate_image(data)
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - Paddle installs numpy
            raise ScreenshotOCRError("ocr_dependency_missing", "缺少 NumPy，请安装 OCR 依赖") from exc
        with self._lock:
            model = self._load_model()
            try:
                predictions = model.predict(np.asarray(image))
                lines = extract_ocr_lines(list(predictions))
            except ScreenshotOCRError:
                raise
            except Exception as exc:
                raise ScreenshotOCRError("ocr_failed", "岗位截图识别失败，请重试") from exc
        text = "\n".join(item["text"] for item in lines).strip()
        if not text:
            raise ScreenshotOCRError("ocr_text_empty", "截图中未识别到可用文字")
        return {
            "text": text,
            "lines": lines,
            "model": OCR_MODEL_NAME,
            "image_format": image_format,
            "width": image.width,
            "height": image.height,
        }


__all__ = [
    "ALLOWED_IMAGE_FORMATS",
    "MAX_SCREENSHOT_BYTES",
    "OCR_MODEL_NAME",
    "PaddleScreenshotOCR",
    "ScreenshotOCRError",
    "extract_ocr_lines",
]
