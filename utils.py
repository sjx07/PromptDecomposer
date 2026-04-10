"""Shared utilities for decomposition: text splitting, LLM calling, helpers."""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .component import Component

logger = logging.getLogger(__name__)


# ── Unit / span helpers ────────────────────────────────────────────────


def split_units(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Split text into non-empty lines with their character spans.

    Returns:
        units: List of stripped line strings.
        spans: List of (start, end) char offsets into original text.
    """
    units: List[str] = []
    spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"[^\n]*\n?", text):
        line = m.group()
        stripped = line.strip()
        if stripped:
            # Use the full line span (preserves indentation context)
            units.append(stripped)
            spans.append((m.start(), m.start() + len(line.rstrip("\n"))))
    return units, spans


def units_to_char_span(
    unit_spans: List[Tuple[int, int]],
    start_unit: int,
    end_unit: int,
) -> Tuple[int, int]:
    """Convert unit index range to character span."""
    return (unit_spans[start_unit][0], unit_spans[end_unit][1])


# ── LLM caller ─────────────────────────────────────────────────────────


def estimate_tokens(text: str, model: Optional[str] = None) -> int:
    """Estimate token count for text, preferring tiktoken when available."""
    if not text:
        return 0

    try:
        import tiktoken

        if model:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, math.ceil(len(text) / 4))


def call_llm_json(
    client: Any,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict:
    """Call LLM and parse JSON response."""
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or "{}"

    usage = getattr(resp, "usage", None)
    if usage is not None:
        usage_data = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            "estimated": False,
        }
    else:
        prompt_tokens = estimate_tokens(system, model=model) + estimate_tokens(user, model=model)
        completion_tokens = estimate_tokens(text, model=model)
        usage_data = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated": True,
        }

    if usage_callback is not None:
        usage_callback(usage_data)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown fences
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        logger.warning("Failed to parse LLM JSON response")
        return {}


# ── Component helpers ──────────────────────────────────────────────────


def components_from_dicts(records: List[Dict]) -> List[Component]:
    """Create components from a list of dicts (manual / pre-parsed input).

    Each dict should have at least "text". Optional: "id", "component_type",
    "source_prompt_idx", "metadata".
    """
    return [
        Component(
            id=r.get("id", f"c{i}"),
            span=tuple(r["span"]) if "span" in r else (0, 0),
            component_type=r.get("component_type", "unknown"),
            source_prompt_idx=r.get("source_prompt_idx", 0),
            metadata=r.get("metadata", {}),
            _text=r.get("text", ""),
        )
        for i, r in enumerate(records)
    ]


def deduplicate(components: List[Component]) -> List[Component]:
    """Remove duplicate components by normalized text, keeping first occurrence."""
    seen: set = set()
    result: List[Component] = []
    for c in components:
        key = c.text.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result
