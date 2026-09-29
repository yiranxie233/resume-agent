"""Small zero-build Streamlit component for pasting clipboard images."""
from __future__ import annotations

from pathlib import Path
from typing import Any

_component: Any = None


def clipboard_image_input(*, key: str) -> dict[str, Any] | None:
    """Return one pasted image as a data URL payload."""

    global _component
    if _component is None:
        import streamlit.components.v1 as components

        component_dir = Path(__file__).resolve().parent / "components" / "clipboard_image"
        _component = components.declare_component("resume_agent_clipboard_image", path=str(component_dir))
    value = _component(key=key, default=None)
    return value if isinstance(value, dict) else None


__all__ = ["clipboard_image_input"]
