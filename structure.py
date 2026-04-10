"""Lightweight structural hints for recursive prompt decomposition."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence


_PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}|\{[^}]+\}|\$\{[^}]+\}")


def line_cues(text: str) -> List[str]:
    """Infer deterministic structural cues from one line of prompt text."""
    stripped = text.strip()
    if not stripped:
        return []

    cues: List[str] = []
    if re.match(r"^\s{0,3}#{1,6}\s+\S", text):
        cues.append("markdown_heading")
        cues.append("header")
    elif stripped.endswith(":") and not re.match(r"^\d+\.", stripped):
        cues.append("header")

    if re.match(r"^\s*\d+\.\s", text):
        cues.append("numbered_list")
    if re.match(r"^\s*[-*]\s", text):
        cues.append("bullet_list")
    if re.match(r"^\s*```", text):
        cues.append("code_fence")
    if re.match(r"^\s*>\s", text):
        cues.append("blockquote")
    if _PLACEHOLDER_RE.search(text):
        cues.append("placeholder")
    if not cues:
        cues.append("paragraph")
    return cues


def describe_units(units: Sequence[str]) -> List[Dict[str, object]]:
    """Return per-line structural hints for a unit sequence."""
    return [
        {"index": idx, "text": unit, "cues": line_cues(unit)}
        for idx, unit in enumerate(units)
    ]


def summarize_cues(units: Sequence[str]) -> List[str]:
    """Return unique cue types observed across a unit sequence."""
    seen: List[str] = []
    for record in describe_units(units):
        for cue in record["cues"]:
            if cue not in seen:
                seen.append(cue)
    return seen


def format_structure_hints(units: Sequence[str]) -> str:
    """Format deterministic structural hints for an LLM user message."""
    lines: List[str] = []
    for record in describe_units(units):
        cues = [cue for cue in record["cues"] if cue != "paragraph"]
        if not cues:
            continue
        snippet = str(record["text"]).replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        lines.append(f"[{record['index']}] cues={','.join(cues)} :: {snippet}")
    if not lines:
        return "(no strong structural cues detected)"
    return "\n".join(lines)
