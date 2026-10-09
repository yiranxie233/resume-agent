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
    def geometry(item: dict[str, Any]) -> tuple[float, float, float] | None:
        box = item.get("box")
        if not isinstance(box, list) or not box:
            return None
        try:
            if isinstance(box[0], list):
                points = [point for point in box if isinstance(point, list) and len(point) >= 2]
                if not points:
                    return None
                ys = [float(point[1]) for point in points]
                xs = [float(point[0]) for point in points]
                return (min(xs), (min(ys) + max(ys)) / 2, max(1.0, max(ys) - min(ys)))
            if len(box) >= 4:
                top, bottom = float(box[1]), float(box[3])
                return (float(box[0]), (top + bottom) / 2, max(1.0, bottom - top))
        except (TypeError, ValueError):
            pass
        return None

    # Paddle normally returns reading order, but long screenshots can be
    # emitted in detector batch order. Cluster boxes into visual rows before
    # sorting by X; a one-pixel Y difference must not reverse two columns on
    # the same line.
    located = [(item, geometry(item), index) for index, item in enumerate(lines)]
    positioned = [value for value in located if value[1] is not None]
    positioned.sort(key=lambda value: (value[1][1], value[1][0]))  # type: ignore[index]
    rows: list[dict[str, Any]] = []
    for item, box, index in positioned:
        assert box is not None
        x, center_y, height = box
        target = next(
            (
                row
                for row in reversed(rows[-3:])
                if abs(center_y - float(row["center_y"]))
                <= max(4.0, min(height, float(row["height"])) * 0.6)
            ),
            None,
        )
        if target is None:
            rows.append(
                {
                    "center_y": center_y,
                    "height": height,
                    "items": [(x, index, item)],
                }
            )
        else:
            count = len(target["items"])
            target["center_y"] = (float(target["center_y"]) * count + center_y) / (count + 1)
            target["height"] = max(float(target["height"]), height)
            target["items"].append((x, index, item))
    ordered: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: float(value["center_y"])):
        ordered.extend(item for _, _, item in sorted(row["items"], key=lambda value: (value[0], value[1])))
    ordered.extend(item for item, box, _ in located if box is None)
    return ordered


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
                text_detection_model_dir=str(detection_dir.resolve()),
                text_recognition_model_name="PP-OCRv6_medium_rec",
                text_recognition_model_dir=str(recognition_dir.resolve()),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_recognition_batch_size=1,
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
                # A failed native predictor can remain in a poisoned state.
                # Reload it on the next explicit attempt without requiring an
                # API restart or another upload in the browser session.
                self._model = None
                message = str(exc).casefold()
                if isinstance(exc, PermissionError) or "permission denied" in message or "拒绝访问" in message:
                    raise ScreenshotOCRError(
                        "ocr_model_permission_denied",
                        "PaddleOCR 模型目录没有读取权限，请修复权限后重试当前截图",
                    ) from exc
                if isinstance(exc, MemoryError) or any(
                    marker in message
                    for marker in ("out of memory", "bad allocation", "resource exhausted")
                ):
                    raise ScreenshotOCRError(
                        "ocr_memory_exhausted",
                        "截图解析所需内存不足，请关闭占用内存的程序或裁剪图片后重试",
                    ) from exc
                if "convertpirattribute2runtimeattribute" in message or "onednn" in message:
                    raise ScreenshotOCRError(
                        "ocr_runtime_incompatible",
                        "PaddleOCR 运行时兼容失败，请确认使用项目锁定的 PaddlePaddle 版本",
                    ) from exc
                raise ScreenshotOCRError(
                    "ocr_failed", "岗位截图识别失败；图片仍保留在当前页面，可直接重试"
                ) from exc
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
