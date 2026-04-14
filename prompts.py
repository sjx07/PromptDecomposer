"""LLM prompt templates and functional component taxonomy for decomposition."""

from __future__ import annotations

from typing import Dict

# ── Functional component taxonomy (domain-agnostic) ────────────────────

FUNCTIONAL_COMPONENTS = [
    "role",
    "task",
    "input_description",
    "procedure",
    "rules",
    "examples",
    "output_format",
    "style_constraints",
    "cot_trigger",
    "input_slots",
]

# Legacy labels kept for compatibility; recursive refinement is guided by
# should_refine plus structural evidence rather than by node type alone.
ATOMIZABLE_TYPES = {"rules", "procedure"}

# ── LLM prompts ────────────────────────────────────────────────────────

_GUIDED_LABELS = """\
Allowed labels and their meanings:
  role              - The agent's identity, persona, or area of expertise.
  task              - The core objective or goal the agent must accomplish.
  input_description - Describes the structure or meaning of input fields.
  procedure         - Ordered instructions or workflow the agent must follow.
  rules             - Unordered constraints, guidelines, or heuristics.
  examples          - Concrete few-shot input/output demonstrations.
  output_format     - Required response structure or schema.
  style_constraints - Tone, length, language, or presentation requirements.
  cot_trigger       - Instructions that elicit chain-of-thought reasoning.
  input_slots       - Template placeholders for runtime values, e.g. {field}.
  unknown           - Content that does not fit any category above."""

_COMMON_SEGMENT_CONTRACT = """\
Your job is the semantic part of prompt decomposition: name the ordered child
blocks in the current scope and explain the boundary evidence. Deterministic
code handles header cleanup, structural snapping, label normalization, and
postprocessing after your response.

Contract:
- Every segment must cover consecutive source lines from the provided scope.
- Copy segment content verbatim from the input; do not paraphrase or summarize.
- Prefer coherent child blocks at the current scope over tiny fragments.
- If candidate span_ids are provided, keep those span_ids and assign semantic
  labels; candidate kinds are boundary cues, not valid labels.
- Set should_refine=true only when a block contains multiple independently
  meaningful child blocks that should be decomposed in a later call.
- boundary_cues are short evidence tags such as markdown_heading, numbered_list,
  bullet_list, code_fence, placeholder, example_block, or paragraph.
- anchor_phrases are short exact substrings copied from the segment."""

_SEGMENT_JSON_SCHEMA = """\
Return strict JSON:
{"segments": [
  {
    "span_id": "<structural span id if candidates were provided>",
    "label": "<label>",
    "content": "<verbatim text>",
    "reason": "<brief boundary/label reason>",
    "should_refine": <true|false>,
    "confidence": "<low|medium|high>",
    "boundary_cues": ["<cue>", "..."],
    "anchor_phrases": ["<exact phrase>", "..."]
  }
]}"""

GUIDED_SEGMENT_SYSTEM = "\n\n".join(
    [
        "You are a prompt structure analyst.",
        _GUIDED_LABELS,
        _COMMON_SEGMENT_CONTRACT,
        _SEGMENT_JSON_SCHEMA,
    ]
)

FREE_SEGMENT_SYSTEM = "\n\n".join(
    [
        "You are a prompt structure analyst.",
        "Use short lowercase snake_case semantic labels that are reusable across similar prompts.",
        _COMMON_SEGMENT_CONTRACT,
        _SEGMENT_JSON_SCHEMA,
    ]
)

# Backward-compatible alias for the original guided segmentation prompt name.
SEGMENT_SYSTEM = GUIDED_SEGMENT_SYSTEM

GUIDED_CHILD_SYSTEM = GUIDED_SEGMENT_SYSTEM
FREE_CHILD_SYSTEM = FREE_SEGMENT_SYSTEM

ATOMIZE_SYSTEM = """\
You are a rule extraction agent. Extract independently toggleable atomic rules,
instructions, or constraints from the provided prompt block.

Contract:
- Copy each item verbatim from the source.
- Preserve numbering or bullets when they are part of the item.
- Skip wrapper-only section headers.
- Classify kind as constraint, requirement, guideline, or heuristic.
- Provide boundary_cues and exact anchor_phrases for source alignment.

Atomize lists whose items can be toggled independently, such as rules,
prohibitions, requirements, verification practices, and practical tips.
Do not atomize coherent examples, schemas, tables, tool catalogs, or workflows.

Return strict JSON:
{"rules": [
  {
    "content": "<verbatim rule text>",
    "kind": "<constraint|requirement|guideline|heuristic>",
    "reason": "<brief structural reason>",
    "confidence": "<low|medium|high>",
    "boundary_cues": ["<cue>", "..."],
    "anchor_phrases": ["<exact phrase>", "..."]
  }
]}"""

# ── Label normalization ────────────────────────────────────────────────

# Common aliases for functional components
_LABEL_ALIASES: Dict[str, str] = {
    "persona": "role",
    "instruction": "task",
    "instructions": "task",
    "input": "input_description",
    "inputs": "input_description",
    "output": "output_format",
    "format": "output_format",
    "example": "examples",
    "few_shot": "examples",
    "steps": "procedure",
    "workflow": "procedure",
    "constraints": "rules",
    "guidelines": "rules",
}


def normalize_label(label: str) -> str:
    """Normalize a functional component label."""
    label = label.strip().lower().replace(" ", "_").replace("-", "_")
    if label in FUNCTIONAL_COMPONENTS:
        return label
    return _LABEL_ALIASES.get(label, "unknown")


def normalize_free_label(label: str) -> str:
    """Normalize a free-style semantic label into snake_case."""
    label = label.strip().lower().replace("-", "_").replace(" ", "_")
    label = "".join(ch for ch in label if ch.isalnum() or ch == "_")
    label = label.strip("_")
    while "__" in label:
        label = label.replace("__", "_")
    return label or "unknown"
