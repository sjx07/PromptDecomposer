"""Apply deterministic structural candidate spans to LLM segmentation output."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .align import align_to_source_details
from .prompts import normalize_free_label, normalize_label
from .refine import default_should_refine, normalize_confidence, should_refine, string_list
from .structure import build_structure_candidates
from .utils import units_to_char_span


def candidate_unit_range(candidate: Dict[str, Any]) -> Tuple[int, int]:
    return int(candidate["start_unit"]), int(candidate["end_unit"])


def candidate_text(units: List[str], candidate: Dict[str, Any]) -> str:
    start, end = candidate_unit_range(candidate)
    return "\n".join(units[start : end + 1])


def merge_cues(*cue_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for cues in cue_lists:
        for cue in cues:
            if cue not in merged:
                merged.append(cue)
    return merged


def ranges_overlap(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def normalize_segment_label(mode: str, label: str) -> str:
    if mode == "guided":
        return normalize_label(label)
    if mode == "free":
        return normalize_free_label(label)
    raise ValueError(f"Unsupported decomposition mode: {mode!r}")


def default_candidate_label(mode: str, units: List[str], candidate: Dict[str, Any]) -> str:
    text = candidate_text(units, candidate).lower()
    kind = str(candidate.get("kind", "structural_section"))

    if mode == "free":
        return normalize_free_label(kind)

    if "example" in text:
        return "examples"
    if "format" in text or "schema" in text or "json" in text:
        return "output_format"
    if "style" in text or "tone" in text or "respond" in text:
        return "style_constraints"
    if "input" in text or "argument" in text:
        return "input_description"
    if "role" in text or "you are" in text:
        return "role"
    if "step" in text or "workflow" in text or "process" in text:
        return "procedure"
    if kind in {"header_list", "list_block"}:
        first_lines = [line.strip() for line in candidate_text(units, candidate).splitlines() if line.strip()]
        numbered = any(re.match(r"^\d+\.\s", line) for line in first_lines)
        return "procedure" if numbered else "rules"
    if "rule" in text or "guideline" in text or "constraint" in text:
        return "rules"
    return "unknown"


def candidate_record(
    units: List[str],
    unit_spans: List[Tuple[int, int]],
    candidate: Dict[str, Any],
    *,
    mode: str,
    label: str | None = None,
    raw_label: str | None = None,
    reason: str = "",
    confidence: str | None = None,
    boundary_cues: List[str] | None = None,
    anchor_phrases: List[str] | None = None,
    refine: bool | None = None,
) -> Dict[str, Any]:
    start_unit, end_unit = candidate_unit_range(candidate)
    content = candidate_text(units, candidate)
    node_label = label or default_candidate_label(mode, units, candidate)
    cues = merge_cues(list(candidate.get("cues", [])), boundary_cues or [])
    nested_candidates = build_structure_candidates(units[start_unit : end_unit + 1])
    if nested_candidates:
        should_refine_value = True
    elif refine is not None:
        should_refine_value = bool(should_refine({"should_refine": refine}, content))
    else:
        should_refine_value = default_should_refine({"boundary_cues": cues}, content)

    return {
        "type": node_label,
        "span": list(units_to_char_span(unit_spans, start_unit, end_unit)),
        "unit_range": [start_unit, end_unit],
        "score": 1.0,
        "alignment_exact": True,
        "alignment_margin": 1.0,
        "alignment_ambiguous": False,
        "raw_label": raw_label or node_label,
        "reason": reason or "structural candidate span",
        "confidence": confidence or str(candidate.get("confidence", "medium")),
        "should_refine": should_refine_value,
        "boundary_cues": cues,
        "anchor_phrases": anchor_phrases or [],
        "structural_candidate": {
            "id": candidate.get("id"),
            "kind": candidate.get("kind"),
            "confidence": candidate.get("confidence"),
            "cues": list(candidate.get("cues", [])),
        },
    }


def snap_record_to_structure_candidate(
    record: Dict[str, Any],
    structural_candidates: List[Dict[str, Any]],
    used_candidate_ids: set[str],
    units: List[str],
    unit_spans: List[Tuple[int, int]],
) -> Dict[str, Any] | None:
    record_range = tuple(record["unit_range"])
    candidates = [
        candidate
        for candidate in structural_candidates
        if str(candidate.get("id")) not in used_candidate_ids
        and ranges_overlap(record_range, candidate_unit_range(candidate))
    ]
    if len(candidates) != 1:
        return None

    candidate = candidates[0]
    candidate_range = candidate_unit_range(candidate)
    if not (
        record_range == candidate_range
        or (candidate_range[0] <= record_range[0] and record_range[1] <= candidate_range[1])
    ):
        return None

    used_candidate_ids.add(str(candidate["id"]))
    record["unit_range"] = [candidate_range[0], candidate_range[1]]
    record["span"] = list(units_to_char_span(unit_spans, candidate_range[0], candidate_range[1]))
    record["boundary_cues"] = merge_cues(
        list(candidate.get("cues", [])),
        record.get("boundary_cues", []),
    )
    record["structural_candidate"] = {
        "id": candidate.get("id"),
        "kind": candidate.get("kind"),
        "confidence": candidate.get("confidence"),
        "cues": list(candidate.get("cues", [])),
    }
    return record


def align_segments_to_source(
    units: List[str],
    unit_tokens: List[List[str]],
    unit_spans: List[Tuple[int, int]],
    llm_segments: List[Dict],
    *,
    mode: str,
    reject_ambiguous: bool,
    structural_candidates: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Align LLM segments, optionally snapping them to structural candidates."""
    aligned: List[Dict[str, Any]] = []
    cursor = 0
    candidate_by_id = {
        str(candidate["id"]): candidate
        for candidate in structural_candidates or []
    }
    used_candidate_ids: set[str] = set()

    for raw in llm_segments:
        if not isinstance(raw, dict):
            continue

        raw_label = str(raw.get("label", "unknown")).strip()
        label = normalize_segment_label(mode, raw_label)
        boundary_cues = string_list(raw.get("boundary_cues"))
        anchor_phrases = string_list(raw.get("anchor_phrases"))
        span_id = str(raw.get("span_id", "")).strip()
        candidate = candidate_by_id.get(span_id)
        if candidate is not None and span_id not in used_candidate_ids:
            used_candidate_ids.add(span_id)
            aligned.append(candidate_record(
                units,
                unit_spans,
                candidate,
                mode=mode,
                label=label,
                raw_label=raw_label,
                reason=str(raw.get("reason", "")).strip(),
                confidence=normalize_confidence(raw.get("confidence")),
                boundary_cues=boundary_cues,
                anchor_phrases=anchor_phrases,
                refine=raw.get("should_refine") if isinstance(raw.get("should_refine"), bool) else None,
            ))
            continue

        content = str(raw.get("content", "")).strip()
        if not content:
            continue

        match = align_to_source_details(
            content,
            units,
            unit_tokens,
            min_start=cursor,
            anchor_phrases=anchor_phrases,
            boundary_cues=boundary_cues,
        )
        if match is None and cursor > 0:
            match = align_to_source_details(
                content,
                units,
                unit_tokens,
                min_start=0,
                anchor_phrases=anchor_phrases,
                boundary_cues=boundary_cues,
            )
        if match is None:
            continue
        if reject_ambiguous and bool(match["ambiguous"]):
            continue

        start_unit = int(match["start"])
        end_unit = int(match["end"])
        char_span = units_to_char_span(unit_spans, start_unit, end_unit)
        cursor = max(cursor, end_unit + 1)

        aligned.append({
            "type": label,
            "span": list(char_span),
            "unit_range": [start_unit, end_unit],
            "score": float(match["score"]),
            "alignment_exact": bool(match["exact"]),
            "alignment_margin": float(match["margin"]),
            "alignment_ambiguous": bool(match["ambiguous"]),
            "raw_label": raw_label,
            "reason": str(raw.get("reason", "")).strip(),
            "confidence": normalize_confidence(raw.get("confidence")),
            "should_refine": should_refine(raw, content),
            "boundary_cues": boundary_cues,
            "anchor_phrases": anchor_phrases,
        })
        if structural_candidates:
            snapped = snap_record_to_structure_candidate(
                aligned[-1],
                structural_candidates,
                used_candidate_ids,
                units,
                unit_spans,
            )
            if snapped is None:
                aligned.pop()

    if structural_candidates:
        covered = [
            tuple(record["unit_range"])
            for record in aligned
        ]
        for candidate in structural_candidates:
            candidate_id = str(candidate["id"])
            candidate_range = candidate_unit_range(candidate)
            if candidate_id in used_candidate_ids:
                continue
            if any(ranges_overlap(candidate_range, record_range) for record_range in covered):
                continue
            aligned.append(candidate_record(units, unit_spans, candidate, mode=mode))

    aligned.sort(key=lambda record: (record["unit_range"][0], record["unit_range"][1]))
    return aligned
