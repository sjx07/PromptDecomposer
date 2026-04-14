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
from typing import Any, Dict, List, Tuple

from .align import _tokenize, align_to_source_details
from .postprocess import postprocess_tree
from .prompts import (
    ATOMIZE_SYSTEM,
    FREE_CHILD_SYSTEM,
    FREE_SEGMENT_SYSTEM,
    GUIDED_CHILD_SYSTEM,
    GUIDED_SEGMENT_SYSTEM,
)
from .refine import (
    looks_like_list_item as _is_list_item_start,
    normalize_confidence as _normalize_confidence,
    string_list as _string_list,
)
from .structure import (
    build_structure_candidates,
    format_structure_candidates,
    format_structure_hints,
    summarize_cues,
)
from .structure_constraints import align_segments_to_source
from .utils import call_llm_json, split_units, units_to_char_span

logger = logging.getLogger(__name__)

_STRUCTURAL_LIST_KINDS = {"header_list", "list_block"}
_ATOMIZABLE_LIST_TERMS = {
    "constraint",
    "critical",
    "do_not",
    "guidance",
    "guideline",
    "exclusion",
    "inclusion",
    "instruction",
    "practice",
    "prohibition",
    "requirement",
    "rule",
    "tip",
}
_COHERENT_LIST_TERMS = {
    "catalog",
    "demo",
    "domain_scope",
    "example",
    "few_shot",
    "format",
    "mechanic",
    "phase",
    "priority",
    "schema",
    "source",
    "step",
    "strength",
    "table",
    "tool_list",
    "workflow",
    "xml",
}


def _span_str(span: List[int]) -> str:
    return f"{span[0]}:{span[1]}"


def _node_id(prompt_id: str, node_type: str, span: List[int]) -> str:
    return f"{prompt_id}:{node_type}[{_span_str(span)}]"


def _child_id(parent_id: str, node_type: str, span: List[int]) -> str:
    return f"{parent_id}/{node_type}[{_span_str(span)}]"


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
        tree = self._postprocess_tree(prompt_id, prompt, tree)
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
        structural_candidates: List[Dict[str, Any]] | None = None,
    ) -> str:
        scope = "top-level prompt" if depth == 0 else f"child scope inside parent label={parent_type or 'unknown'}"
        structural_hints = format_structure_hints(units)
        structural_candidate_block = ""
        if structural_candidates:
            structural_candidate_block = (
                "Structural candidate spans (hard boundaries, not labels):\n"
                f"{format_structure_candidates(units, structural_candidates)}\n\n"
                "Return exactly one segment per structural candidate span_id. "
                "Use the span_id values as given, copy each candidate's content "
                "verbatim, and choose the semantic label yourself. The candidate "
                "kind is only boundary evidence.\n\n"
            )
        numbered = "\n".join(f"[{i}] {unit}" for i, unit in enumerate(units))
        return (
            f"Current scope: {scope}\n"
            f"Depth: {depth}\n"
            f"Observed structural hints:\n{structural_hints}\n\n"
            f"{structural_candidate_block}"
            f"Prompt lines:\n{numbered}"
        )

    def _request_segments(
        self,
        units: List[str],
        *,
        depth: int,
        parent_type: str | None,
        structural_candidates: List[Dict[str, Any]] | None = None,
    ) -> List[Dict]:
        """Ask the LLM to segment the current scope into child blocks."""
        result = call_llm_json(
            self.client,
            self.model,
            self._segment_system(depth),
            self._segment_user_message(units, depth, parent_type, structural_candidates),
            self.temperature,
            usage_callback=self._record_usage,
        )
        return result.get("segments", [])

    def _align_segments(
        self,
        units: List[str],
        unit_tokens: List[List[str]],
        unit_spans: List[Tuple[int, int]],
        llm_segments: List[Dict],
        *,
        reject_ambiguous: bool,
        structural_candidates: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        return align_segments_to_source(
            units,
            unit_tokens,
            unit_spans,
            llm_segments,
            mode=self.mode,
            reject_ambiguous=reject_ambiguous,
            structural_candidates=structural_candidates,
        )

    def _is_valid_split(
        self,
        aligned: List[Dict[str, Any]],
        *,
        top_level: bool,
        parent_span: Tuple[int, int] | None,
    ) -> bool:
        if not aligned:
            return False
        if not top_level and len(aligned) < 2:
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
                "structural_candidate": record.get("structural_candidate"),
            },
            "children": [],
        }

    def _list_item_start_count(self, units: List[str]) -> int:
        return sum(1 for unit in units if _is_list_item_start(unit))

    def _structural_candidate_kind(self, node: Dict[str, Any]) -> str:
        candidate = node.get("metadata", {}).get("structural_candidate") or {}
        return str(candidate.get("kind", ""))

    def _atomization_label_text(self, node: Dict[str, Any]) -> str:
        metadata = node.get("metadata", {})
        parts = [
            str(node.get("type", "")),
            str(metadata.get("raw_label", "")),
            str(metadata.get("title", "")),
        ]
        return " ".join(parts).lower().replace("-", "_").replace(" ", "_")

    def _is_coherent_structural_list(self, node: Dict[str, Any]) -> bool:
        label_text = self._atomization_label_text(node)
        return any(term in label_text for term in _COHERENT_LIST_TERMS)

    def _has_atomizable_list_label(self, node: Dict[str, Any]) -> bool:
        label_text = self._atomization_label_text(node)
        return any(term in label_text for term in _ATOMIZABLE_LIST_TERMS)

    def _should_atomize_structural_list(
        self,
        node: Dict[str, Any],
        units: List[str],
        depth: int,
    ) -> bool:
        if not self.atomize:
            return False
        if depth + 1 >= self.max_depth:
            return False
        if self._structural_candidate_kind(node) not in _STRUCTURAL_LIST_KINDS:
            return False
        if self._list_item_start_count(units) < 2:
            return False
        if self._is_coherent_structural_list(node):
            return False
        return self._has_atomizable_list_label(node)

    def _should_legacy_atomize(self, node: Dict[str, Any], units: List[str]) -> bool:
        return bool(node.get("metadata", {}).get("should_refine", False)) and self._list_item_start_count(units) >= 2

    def _should_recurse(self, node: Dict[str, Any], units: List[str], depth: int) -> bool:
        if not self.atomize:
            return False
        if depth + 1 >= self.max_depth:
            return False
        if len(units) < 2:
            return False
        if (
            self._structural_candidate_kind(node) in _STRUCTURAL_LIST_KINDS
            and self._is_coherent_structural_list(node)
        ):
            return False
        if (node["span"][1] - node["span"][0]) < self.min_span_chars and self._list_item_start_count(units) < 2:
            return False
        return bool(node.get("metadata", {}).get("should_refine", False))

    def _request_atomize(self, section_text: str) -> List[Dict]:
        """Ask the LLM to extract individual rules from a local section."""
        result = call_llm_json(
            self.client,
            self.model,
            ATOMIZE_SYSTEM,
            f"Section text:\n{section_text}",
            self.temperature,
            usage_callback=self._record_usage,
        )
        return result.get("rules", [])

    def _legacy_atomize_children(
        self,
        prompt_id: str,
        prompt: str,
        parent: Dict[str, Any],
        units: List[str],
        unit_tokens: List[List[str]],
        unit_spans: List[Tuple[int, int]],
        depth: int,
    ) -> List[Dict[str, Any]]:
        """Fallback atomization for simple rule/procedure blocks."""
        section_text = prompt[parent["span"][0]:parent["span"][1]]
        if not section_text.strip():
            return []

        llm_rules = self._request_atomize(section_text)
        children: List[Dict[str, Any]] = []
        cursor = 0
        seen_spans: set[tuple[int, int]] = set()

        for raw in llm_rules:
            if not isinstance(raw, dict):
                continue
            content = str(raw.get("content", "")).strip()
            if not content:
                continue

            boundary_cues = _string_list(raw.get("boundary_cues"))
            anchor_phrases = _string_list(raw.get("anchor_phrases"))
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
            if match is None or bool(match["ambiguous"]):
                continue

            start_unit = int(match["start"])
            end_unit = int(match["end"])
            char_span = units_to_char_span(unit_spans, start_unit, end_unit)
            span_key = (char_span[0], char_span[1])
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            cursor = max(cursor, end_unit + 1)

            node_type = parent["type"]
            child = {
                "id": _child_id(parent["id"], node_type, list(char_span)),
                "type": node_type,
                "kind": str(raw.get("kind", "rule")).lower(),
                "span": list(char_span),
                "metadata": {
                    "depth": depth,
                    "mode": self.mode,
                    "parent_id": parent["id"],
                    "reason": str(raw.get("reason", "")).strip(),
                    "confidence": _normalize_confidence(raw.get("confidence")),
                    "boundary_cues": boundary_cues,
                    "anchor_phrases": anchor_phrases,
                    "alignment_score": float(match["score"]),
                    "alignment_exact": bool(match["exact"]),
                    "alignment_margin": float(match["margin"]),
                    "alignment_ambiguous": bool(match["ambiguous"]),
                    "structural_hints": summarize_cues(units[start_unit : end_unit + 1]),
                },
            }
            children.append(child)

        return children

    def _postprocess_tree(
        self,
        prompt_id: str,
        prompt: str,
        tree: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return postprocess_tree(prompt_id, prompt, tree)

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

        structural_candidates = build_structure_candidates(units)
        llm_segments = self._request_segments(
            units,
            depth=depth,
            parent_type=parent_type,
            structural_candidates=structural_candidates,
        )
        aligned = self._align_segments(
            units,
            unit_tokens,
            unit_spans,
            llm_segments,
            reject_ambiguous=not top_level,
            structural_candidates=structural_candidates,
        )
        parent_span = (
            (unit_spans[0][0], unit_spans[-1][1])
            if unit_spans else None
        )
        if not self._is_valid_split(aligned, top_level=top_level, parent_span=parent_span):
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
            children: List[Dict[str, Any]] = []
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
                if not children and self._should_legacy_atomize(node, child_units):
                    children = self._legacy_atomize_children(
                        prompt_id,
                        prompt,
                        node,
                        child_units,
                        child_tokens,
                        child_spans,
                        depth + 1,
                    )
            if not children and self._should_atomize_structural_list(node, child_units, depth):
                children = self._legacy_atomize_children(
                    prompt_id,
                    prompt,
                    node,
                    child_units,
                    child_tokens,
                    child_spans,
                    depth + 1,
                )
            if children:
                node["children"] = children
            tree.append(node)

        return tree
