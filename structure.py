"""Lightweight structural hints for recursive prompt decomposition."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence


_PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}|\{[^}]+\}|\$\{[^}]+\}")
_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+\S")
_XML_TAG_RE = re.compile(r"^</?([A-Za-z][\w:.-]*)(?:\s+[^>]*)?/?>$")
_XML_OPEN_RE = re.compile(r"^<([A-Za-z][\w:.-]*)(?:\s+[^>]*)?>$")
_XML_CLOSE_RE = re.compile(r"^</([A-Za-z][\w:.-]*)>$")


def line_cues(text: str) -> List[str]:
    """Infer deterministic structural cues from one line of prompt text."""
    stripped = text.strip()
    if not stripped:
        return []

    cues: List[str] = []
    if _MD_HEADING_RE.match(text):
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


def _looks_like_list_item(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith(("- ", "* ")) or bool(re.match(r"^\d+\.\s", stripped))


def _is_code_fence(text: str) -> bool:
    return bool(re.match(r"^\s*```", text))


def _code_fence_mask(units: Sequence[str]) -> List[bool]:
    mask: List[bool] = []
    in_fence = False
    for unit in units:
        if _is_code_fence(unit):
            mask.append(True)
            in_fence = not in_fence
        else:
            mask.append(in_fence)
    return mask


def _heading_level(text: str) -> int | None:
    match = _MD_HEADING_RE.match(text)
    if match is None:
        return None
    return len(match.group(1))


def _is_colon_header(text: str) -> bool:
    stripped = text.strip()
    if re.match(r"^(\*\*|__).+:(\*\*|__)$", stripped):
        return True
    return stripped.endswith(":") and not _looks_like_list_item(stripped)


def _is_list_boundary(units: Sequence[str], idx: int) -> bool:
    unit = units[idx]
    return (
        _is_code_fence(unit)
        or _heading_level(unit) is not None
        or (
            _is_colon_header(unit)
            and idx + 1 < len(units)
            and _looks_like_list_item(units[idx + 1])
        )
    )


def _has_future_list_item(units: Sequence[str], start: int) -> bool:
    idx = start + 1
    while idx < len(units):
        if _is_list_boundary(units, idx):
            return False
        if _looks_like_list_item(units[idx]):
            return True
        idx += 1
    return False


def _is_list_continuation(units: Sequence[str], idx: int) -> bool:
    if _looks_like_list_item(units[idx]):
        return False
    if _is_list_boundary(units, idx):
        return False
    return _has_future_list_item(units, idx)


def _advance_list_span(units: Sequence[str], idx: int) -> int:
    while idx < len(units):
        if _looks_like_list_item(units[idx]):
            idx += 1
            continue
        if _is_list_continuation(units, idx):
            idx += 1
            continue
        break
    return idx


def _xml_open_tag(text: str) -> str | None:
    stripped = text.strip()
    match = _XML_OPEN_RE.match(stripped)
    if match is None or stripped.endswith("/>"):
        return None
    return match.group(1)


def _xml_close_tag(text: str) -> str | None:
    match = _XML_CLOSE_RE.match(text.strip())
    if match is None:
        return None
    return match.group(1)


def _is_xml_tag_line(text: str) -> bool:
    return bool(_XML_TAG_RE.match(text.strip()))


def _candidate(
    units: Sequence[str],
    start: int,
    end: int,
    kind: str,
    confidence: str = "medium",
) -> Dict[str, object]:
    local = units[start : end + 1]
    return {
        "id": f"S{start}_{end}",
        "start_unit": start,
        "end_unit": end,
        "kind": kind,
        "confidence": confidence,
        "line_count": end - start + 1,
        "cues": summarize_cues(local),
    }


def _candidate_is_whole_scope(candidate: Dict[str, object], unit_count: int) -> bool:
    return int(candidate["start_unit"]) == 0 and int(candidate["end_unit"]) == unit_count - 1


def _clean_candidates(candidates: List[Dict[str, object]], unit_count: int) -> List[Dict[str, object]]:
    if len(candidates) < 2:
        return []
    if len(candidates) == 1 and _candidate_is_whole_scope(candidates[0], unit_count):
        return []
    candidates = sorted(candidates, key=lambda item: (int(item["start_unit"]), int(item["end_unit"])))
    for idx, candidate in enumerate(candidates):
        candidate["id"] = f"S{idx}"
    return candidates


def _markdown_section_candidates(units: Sequence[str]) -> List[Dict[str, object]]:
    fenced = _code_fence_mask(units)
    headings = [
        (idx, _heading_level(unit))
        for idx, unit in enumerate(units)
        if not fenced[idx]
    ]
    headings = [(idx, level) for idx, level in headings if level is not None]
    if not headings:
        return []
    levels = sorted({int(level) for _idx, level in headings})
    active_level = levels[0]
    if sum(1 for _idx, level in headings if level == active_level) == 1 and len(levels) > 1:
        active_level = levels[1]
    section_headings = [
        (idx, int(level))
        for idx, level in headings
        if int(level) == active_level
    ]

    candidates: List[Dict[str, object]] = []
    first_heading = section_headings[0][0]
    if first_heading > 0:
        candidates.append(_candidate(units, 0, first_heading - 1, "preamble", "medium"))

    for pos, (start, level) in enumerate(section_headings):
        end = len(units) - 1
        for next_start, next_level in headings:
            if next_start <= start:
                continue
            if next_level <= level:
                end = next_start - 1
                break
        candidates.append(_candidate(units, start, end, "markdown_section", "high"))

    return _clean_candidates(candidates, len(units))


def _find_xml_section_end(
    units: Sequence[str],
    start: int,
    tag: str,
    fenced: Sequence[bool] | None = None,
) -> int | None:
    depth = 0
    fenced = fenced or [False] * len(units)
    for idx in range(start, len(units)):
        if fenced[idx]:
            continue
        open_tag = _xml_open_tag(units[idx])
        close_tag = _xml_close_tag(units[idx])
        if open_tag == tag:
            depth += 1
        if close_tag == tag:
            depth -= 1
            if depth == 0:
                return idx
    return None


def _xml_section_candidates(units: Sequence[str]) -> List[Dict[str, object]]:
    fenced = _code_fence_mask(units)
    if not any(not fenced[idx] and _is_xml_tag_line(unit) for idx, unit in enumerate(units)):
        return []

    candidates: List[Dict[str, object]] = []
    idx = 0
    while idx < len(units):
        tag = None if fenced[idx] else _xml_open_tag(units[idx])
        if tag is None:
            start = idx
            idx += 1
            while idx < len(units) and (fenced[idx] or _xml_open_tag(units[idx]) is None):
                idx += 1
            candidates.append(_candidate(units, start, idx - 1, "preamble", "medium"))
            continue

        end = _find_xml_section_end(units, idx, tag, fenced)
        if end is None:
            end = idx
        candidates.append(_candidate(units, idx, end, "xml_section", "high"))
        idx = end + 1

    return _clean_candidates(candidates, len(units))


def _xml_wrapper_child_candidates(units: Sequence[str]) -> List[Dict[str, object]]:
    if len(units) < 4:
        return []

    tag = _xml_open_tag(units[0])
    if tag is None:
        return []
    fenced = _code_fence_mask(units)
    if _find_xml_section_end(units, 0, tag, fenced) != len(units) - 1:
        return []

    inner = units[1:-1]
    inner_candidates = _xml_section_candidates(inner)
    if not inner_candidates:
        return []

    candidates = [_candidate(units, 0, 0, "xml_wrapper_open", "high")]
    for candidate in inner_candidates:
        copied = dict(candidate)
        copied["start_unit"] = int(copied["start_unit"]) + 1
        copied["end_unit"] = int(copied["end_unit"]) + 1
        candidates.append(copied)
    candidates.append(_candidate(units, len(units) - 1, len(units) - 1, "xml_wrapper_close", "high"))
    return _clean_candidates(candidates, len(units))


def _scan_block_candidates(units: Sequence[str]) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    idx = 0
    while idx < len(units):
        unit = units[idx]
        if _is_code_fence(unit):
            start = idx
            idx += 1
            while idx < len(units) and not _is_code_fence(units[idx]):
                idx += 1
            if idx < len(units):
                idx += 1
            candidates.append(_candidate(units, start, idx - 1, "code_fence", "high"))
            continue

        if _is_colon_header(unit) and idx + 1 < len(units) and _looks_like_list_item(units[idx + 1]):
            start = idx
            idx += 1
            idx = _advance_list_span(units, idx)
            candidates.append(_candidate(units, start, idx - 1, "header_list", "high"))
            continue

        if _looks_like_list_item(unit):
            start = idx
            idx = _advance_list_span(units, idx)
            candidates.append(_candidate(units, start, idx - 1, "list_block", "medium"))
            continue

        start = idx
        idx += 1
        while (
            idx < len(units)
            and not _is_code_fence(units[idx])
            and not _looks_like_list_item(units[idx])
            and not (_is_colon_header(units[idx]) and idx + 1 < len(units) and _looks_like_list_item(units[idx + 1]))
        ):
            idx += 1
        candidates.append(_candidate(units, start, idx - 1, "paragraph_block", "medium"))

    return _clean_candidates(candidates, len(units))


def build_structure_candidates(units: Sequence[str]) -> List[Dict[str, object]]:
    """Return conservative structural candidate spans for one decomposition scope.

    These spans are intended to constrain semantic labeling, not to replace
    semantic labels. The function only returns multi-span skeletons when clear
    structural boundaries are present.
    """
    if len(units) < 2:
        return []

    for builder in (
        _markdown_section_candidates,
        _xml_section_candidates,
        _xml_wrapper_child_candidates,
        _scan_block_candidates,
    ):
        candidates = builder(units)
        if candidates:
            return candidates
    return []


def format_structure_candidates(units: Sequence[str], candidates: Sequence[Dict[str, object]]) -> str:
    """Format candidate spans for a structure-constrained LLM request."""
    if not candidates:
        return "(no structural candidate spans)"

    lines: List[str] = []
    for candidate in candidates:
        start = int(candidate["start_unit"])
        end = int(candidate["end_unit"])
        snippet = " / ".join(str(unit).strip() for unit in units[start : min(end + 1, start + 3)])
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        cues = ",".join(str(cue) for cue in candidate.get("cues", []))
        lines.append(
            f"[{candidate['id']}] lines={start}-{end} "
            f"kind={candidate['kind']} confidence={candidate['confidence']} "
            f"cues={cues or 'none'} :: {snippet}"
        )
    return "\n".join(lines)


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
