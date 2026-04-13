"""PromptDecomposer: orchestrates recursive prompt decomposition.

Pipeline:
1. Split prompt into semantic units (non-empty lines)
2. Ask the LLM for child segments at the current scope
3. Align the returned child text back to the source span
4. Recurse into refinable child nodes within their parent span only

Output is a span-based tree: the original prompt is stored once,
and every node references it by [start, end) character offsets.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from .align import _tokenize, align_to_source_details
from .prompts import (
    FREE_CHILD_SYSTEM,
    FREE_SEGMENT_SYSTEM,
    GUIDED_CHILD_SYSTEM,
    GUIDED_SEGMENT_SYSTEM,
    normalize_free_label,
    normalize_label,
)
from .structure import format_structure_hints, summarize_cues
from .utils import call_llm_json, split_units, units_to_char_span

logger = logging.getLogger(__name__)


def _span_str(span: List[int]) -> str:
    return f"{span[0]}:{span[1]}"


def _node_id(prompt_id: str, node_type: str, span: List[int]) -> str:
    return f"{prompt_id}:{node_type}[{_span_str(span)}]"


def _child_id(parent_id: str, node_type: str, span: List[int]) -> str:
    return f"{parent_id}/{node_type}[{_span_str(span)}]"


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_confidence(value: Any) -> str:
    confidence = str(value or "").strip().lower()
    if confidence in {"low", "medium", "high"}:
        return confidence
    return "medium"


def _looks_like_list_item(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith(("- ", "* ")) or bool(re.match(r"^\d+\.\s", stripped))


def _looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    return (
        bool(re.match(r"^\s{0,3}#{1,6}\s+\S", text))
        or (stripped.startswith("<") and stripped.endswith(">") and " " not in stripped)
        or (stripped.endswith(":") and not _looks_like_list_item(stripped))
    )


def _is_procedure_like_anchor(text: str) -> bool:
    lowered = text.strip().lower()
    return bool(
        lowered
        and (
            "procedure" in lowered
            or "workflow" in lowered
            or "step-by-step" in lowered
            or re.search(r"\bsteps?\b", lowered) is not None
            or re.search(r"\bhow to\b", lowered) is not None
        )
    )


def _strip_leading_anchor(units: List[str]) -> Tuple[List[str], List[str]]:
    body = [unit.strip() for unit in units if unit.strip()]
    anchor: List[str] = []
    while body and _looks_like_heading(body[0]) and not _looks_like_list_item(body[0]):
        anchor.append(body.pop(0))
    return anchor, body


def _default_should_refine(raw: Dict[str, Any], content: str) -> bool:
    cues = set(_string_list(raw.get("boundary_cues")))
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    if _is_procedure_like_anchor(lines[0]):
        return False

    body = list(lines)
    if body and _looks_like_heading(body[0]):
        body = body[1:]
    if len(body) < 2:
        return False

    list_like = sum(1 for line in body if _looks_like_list_item(line))
    if list_like >= 2 and list_like == len(body):
        return True
    if "header" in cues and list_like >= 2:
        return True
    return False


def _should_refine(raw: Dict[str, Any], content: str) -> bool:
    explicit = raw.get("should_refine")
    if isinstance(explicit, bool):
        return explicit
    return _default_should_refine(raw, content)


class PromptDecomposer:
    """Domain-agnostic recursive prompt decomposer with align-back to source."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        temperature: float = 0.0,
        atomize: bool = True,
        mode: str = "guided",
        max_depth: int = 4,
        min_span_chars: int = 80,
    ):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.atomize = atomize
        self.mode = mode
        self.max_depth = max_depth
        self.min_span_chars = min_span_chars
        self._client = None
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "requests": 0,
            "estimated_requests": 0,
        }

    @property
    def client(self):
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self):
        if self.provider == "openai":
            from openai import OpenAI
            return OpenAI()
        if self.provider == "gemini":
            from openai import OpenAI
            import os
            return OpenAI(
                api_key=os.environ.get("GEMINI_API_KEY", ""),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        raise ValueError(f"Unsupported provider: {self.provider}")

    def usage_snapshot(self) -> Dict[str, int]:
        """Return cumulative usage observed by this decomposer instance."""
        return dict(self._usage)

    def _record_usage(self, usage: Dict[str, Any]) -> None:
        self._usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        self._usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        self._usage["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        self._usage["requests"] += 1
        if usage.get("estimated"):
            self._usage["estimated_requests"] += 1

    def decompose(self, prompt: str, prompt_id: str = "p0") -> Dict[str, Any]:
        """Decompose a prompt into a span-based tree."""
        units, unit_spans = split_units(prompt)
        if not units:
            return {"prompt": prompt, "tree": []}

        unit_tokens = [_tokenize(unit) for unit in units]
        tree = self._decompose_scope(
            prompt_id,
            prompt,
            units,
            unit_tokens,
            unit_spans,
            depth=0,
            parent_id=None,
            parent_type=None,
            top_level=True,
        )
        return {"prompt": prompt, "tree": tree}

    def _segment_system(self, depth: int = 0) -> str:
        if self.mode == "guided":
            return GUIDED_SEGMENT_SYSTEM if depth == 0 else GUIDED_CHILD_SYSTEM
        if self.mode == "free":
            return FREE_SEGMENT_SYSTEM if depth == 0 else FREE_CHILD_SYSTEM
        raise ValueError(f"Unsupported decomposition mode: {self.mode!r}")

    def _segment_user_message(
        self,
        units: List[str],
        depth: int,
        parent_type: str | None,
    ) -> str:
        scope = "top-level prompt" if depth == 0 else f"child scope inside parent label={parent_type or 'unknown'}"
        structural_hints = format_structure_hints(units)
        numbered = "\n".join(f"[{i}] {unit}" for i, unit in enumerate(units))
        return (
            f"Current scope: {scope}\n"
            f"Depth: {depth}\n"
            f"Observed structural hints:\n{structural_hints}\n\n"
            f"Prompt lines:\n{numbered}"
        )

    def _request_segments(
        self,
        units: List[str],
        *,
        depth: int,
        parent_type: str | None,
    ) -> List[Dict]:
        """Ask the LLM to segment the current scope into child blocks."""
        result = call_llm_json(
            self.client,
            self.model,
            self._segment_system(depth),
            self._segment_user_message(units, depth, parent_type),
            self.temperature,
            usage_callback=self._record_usage,
        )
        return result.get("segments", [])

    def _normalize_segment_label(self, label: str) -> str:
        if self.mode == "guided":
            return normalize_label(label)
        if self.mode == "free":
            return normalize_free_label(label)
        raise ValueError(f"Unsupported decomposition mode: {self.mode!r}")

    def _align_segments(
        self,
        units: List[str],
        unit_tokens: List[List[str]],
        unit_spans: List[Tuple[int, int]],
        llm_segments: List[Dict],
        *,
        reject_ambiguous: bool,
    ) -> List[Dict[str, Any]]:
        """Align LLM child segments to the current local scope."""
        aligned: List[Dict[str, Any]] = []
        cursor = 0

        for raw in llm_segments:
            if not isinstance(raw, dict):
                continue
            content = str(raw.get("content", "")).strip()
            if not content:
                continue

            raw_label = str(raw.get("label", "unknown")).strip()
            label = self._normalize_segment_label(raw_label)
            boundary_cues = _string_list(raw.get("boundary_cues"))
            anchor_phrases = _string_list(raw.get("anchor_phrases"))
            content_lines = [line.strip() for line in content.splitlines() if line.strip()]
            if (
                len(content_lines) == 1
                and _looks_like_heading(content_lines[0])
                and (
                    "markdown_heading" in boundary_cues
                    or "header" in boundary_cues
                    or raw_label.endswith("_heading")
                )
            ):
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
                "confidence": _normalize_confidence(raw.get("confidence")),
                "should_refine": _should_refine(raw, content),
                "boundary_cues": boundary_cues,
                "anchor_phrases": anchor_phrases,
            })

        return aligned

    def _is_valid_split(
        self,
        aligned: List[Dict[str, Any]],
        *,
        top_level: bool,
        parent_span: Tuple[int, int] | None,
        units: List[str],
    ) -> bool:
        if not aligned:
            return False
        if not top_level and len(aligned) < 2:
            anchor, _ = _strip_leading_anchor(units)
            if len(aligned) == 1 and parent_span is not None and anchor:
                child_span = tuple(aligned[0]["span"])
                parent_tuple = tuple(parent_span)
                if child_span != parent_tuple and child_span[0] >= parent_tuple[0] and child_span[1] <= parent_tuple[1]:
                    return True
            return False

        prev_end = None
        for record in aligned:
            start, end = record["span"]
            if prev_end is not None and start < prev_end:
                return False
            prev_end = end

        if not top_level and parent_span is not None:
            if len(aligned) == 1 and tuple(aligned[0]["span"]) == tuple(parent_span):
                return False
        return True

    def _build_node(
        self,
        prompt_id: str,
        record: Dict[str, Any],
        *,
        depth: int,
        parent_id: str | None,
        local_units: List[str],
    ) -> Dict[str, Any]:
        span = record["span"]
        node_type = record["type"]
        return {
            "id": _node_id(prompt_id, node_type, span),
            "type": node_type,
            "span": span,
            "metadata": {
                "depth": depth,
                "mode": self.mode,
                "parent_id": parent_id,
                "raw_label": record.get("raw_label", node_type),
                "reason": record.get("reason", ""),
                "confidence": record.get("confidence", "medium"),
                "should_refine": record.get("should_refine", False),
                "boundary_cues": record.get("boundary_cues", []),
                "anchor_phrases": record.get("anchor_phrases", []),
                "alignment_score": record.get("score", 0.0),
                "alignment_exact": record.get("alignment_exact", False),
                "alignment_margin": record.get("alignment_margin", 0.0),
                "alignment_ambiguous": record.get("alignment_ambiguous", False),
                "structural_hints": summarize_cues(local_units),
            },
            "children": [],
        }

    def _looks_like_flat_atomic_block(self, units: List[str]) -> bool:
        anchor, body = _strip_leading_anchor(units)
        if len(body) < 2:
            return False
        if anchor and _is_procedure_like_anchor(anchor[0]):
            return False
        list_like = sum(1 for unit in body if _looks_like_list_item(unit))
        return list_like >= 1 and list_like == len(body)

    def _should_recurse(self, node: Dict[str, Any], units: List[str], depth: int) -> bool:
        if not self.atomize:
            return False
        if depth + 1 >= self.max_depth:
            return False
        if len(units) < 2:
            return False
        if (node["span"][1] - node["span"][0]) < self.min_span_chars and not self._looks_like_flat_atomic_block(units):
            return False
        return bool(node.get("metadata", {}).get("should_refine", False))

    def _decompose_scope(
        self,
        prompt_id: str,
        prompt: str,
        units: List[str],
        unit_tokens: List[List[str]],
        unit_spans: List[Tuple[int, int]],
        *,
        depth: int,
        parent_id: str | None,
        parent_type: str | None,
        top_level: bool,
    ) -> List[Dict[str, Any]]:
        if not units:
            return []

        llm_segments = self._request_segments(units, depth=depth, parent_type=parent_type)
        aligned = self._align_segments(
            units,
            unit_tokens,
            unit_spans,
            llm_segments,
            reject_ambiguous=not top_level,
        )
        parent_span = (
            (unit_spans[0][0], unit_spans[-1][1])
            if unit_spans else None
        )
        if not self._is_valid_split(aligned, top_level=top_level, parent_span=parent_span, units=units):
            return []

        tree: List[Dict[str, Any]] = []
        for record in aligned:
            start_unit, end_unit = record["unit_range"]
            child_units = units[start_unit : end_unit + 1]
            child_tokens = unit_tokens[start_unit : end_unit + 1]
            child_spans = unit_spans[start_unit : end_unit + 1]

            node = self._build_node(
                prompt_id,
                record,
                depth=depth,
                parent_id=parent_id,
                local_units=child_units,
            )
            if self._should_recurse(node, child_units, depth):
                children = self._decompose_scope(
                    prompt_id,
                    prompt,
                    child_units,
                    child_tokens,
                    child_spans,
                    depth=depth + 1,
                    parent_id=node["id"],
                    parent_type=node["type"],
                    top_level=False,
                )
                if children:
                    node["children"] = children
            tree.append(node)

        return tree
