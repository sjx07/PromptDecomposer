"""Shared heuristics for recursive prompt refinement decisions."""

from __future__ import annotations

import re
from typing import Any, Dict, List


def string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_confidence(value: Any) -> str:
    confidence = str(value or "").strip().lower()
    if confidence in {"low", "medium", "high"}:
        return confidence
    return "medium"


def looks_like_list_item(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith(("- ", "* ")) or bool(re.match(r"^\d+\.\s", stripped))


def default_should_refine(raw: Dict[str, Any], content: str) -> bool:
    cues = set(string_list(raw.get("boundary_cues")))
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    body = list(lines)
    if body and body[0].endswith(":") and not looks_like_list_item(body[0]):
        body = body[1:]
    if len(body) < 2:
        return False

    list_like = sum(1 for line in body if looks_like_list_item(line))
    if list_like >= 2 and list_like == len(body):
        return True
    if "header" in cues and list_like >= 2:
        return True
    return False


def should_refine(raw: Dict[str, Any], content: str) -> bool:
    explicit = raw.get("should_refine")
    if isinstance(explicit, bool):
        return explicit
    return default_should_refine(raw, content)
