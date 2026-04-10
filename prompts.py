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

GUIDED_SEGMENT_SYSTEM = """\
You are a prompt structure analyst. Given a prompt template split into \
numbered lines, identify the ordered child segments at the current scope and \
label each one.

Allowed labels and their meanings:
  role              — The agent's identity, persona, or area of expertise.
  task              — The core objective or goal the agent must accomplish.
  input_description — Describes the structure or meaning of the input fields.
  procedure         — Ordered, step-by-step instructions the agent must follow.
  rules             — Unordered constraints, guidelines, or heuristics.
  examples          — Concrete few-shot input/output demonstrations.
  output_format     — Specifies the required structure or schema of the response.
  style_constraints — Tone, length, language, or presentation requirements.
  cot_trigger       — Instructions that elicit chain-of-thought reasoning.
  input_slots       — Template placeholders for runtime values (e.g. {field}).
  unknown           — Content that does not fit any category above.

Rules:
- Each segment spans one or more consecutive lines.
- Copy the content VERBATIM from the input lines (no paraphrase).
- A header line (e.g. "Rules:") should merge with the block it introduces.
- Prefer maximal coherent sections at the current scope, not tiny fragments.
- Respect markdown and formatting cues such as headings, bullets, numbered lists,
  blockquotes, fenced code blocks, and placeholder-heavy lines.
- If you are decomposing inside an existing parent block, produce child segments
  that are meaningful within that parent rather than re-segmenting the entire prompt.
- When a child block is a smaller piece of a parent rules/procedure section,
  keep the functional label aligned with that parent rather than inventing subtypes.
- Mark should_refine=true only when the segment contains multiple
  independently meaningful sub-instructions that should become child nodes.
- Example: a flat constraint list such as "Rules:\n1. ...\n2. ..." should usually
  set should_refine=true.
- Example: a coherent workflow such as "Procedure:\n1. ...\n2. ...\n3. ..."
  should usually stay whole with should_refine=false unless the steps split into
  larger independently meaningful sub-blocks rather than one workflow.
- Example: a flat constraint list such as "Rules:\n1. ...\n2. ..." should usually
  set should_refine=true.
- Example: a coherent workflow such as "Procedure:\n1. ...\n2. ...\n3. ..."
  should usually stay whole with should_refine=false unless the steps split into
  larger independently meaningful sub-blocks rather than one workflow.
- Use boundary_cues to explain why the boundaries are trustworthy
  (e.g. header, markdown_heading, numbered_list, bullet_list, blockquote,
  code_fence, placeholder, example_block, paragraph).
- anchor_phrases must be short exact substrings copied from the segment that
  would help align the segment back to the source. Prefer headings or rare phrases.
- Few-shot examples with concrete values → "examples".
- Template placeholders (e.g. {question}, {{schema}}) → "input_slots".
- Ordered step-by-step instructions → "procedure".
- Constraint lists without strict order → "rules".

Return strict JSON:
{"segments": [
  {
    "label": "<component>",
    "content": "<verbatim text>",
    "reason": "<brief structural reason>",
    "should_refine": <true|false>,
    "confidence": "<low|medium|high>",
    "boundary_cues": ["<cue>", "..."],
    "anchor_phrases": ["<exact phrase>", "..."]
  }
]}"""

FREE_SEGMENT_SYSTEM = """\
You are a prompt structure analyst. Given a prompt template split into \
numbered lines, identify the ordered coherent child sections at the current \
scope without forcing them into a fixed taxonomy.

Rules:
- Each segment spans one or more consecutive lines.
- Copy the content VERBATIM from the input lines (no paraphrase).
- Prefer maximal coherent sections at the current scope, not tiny fragments.
- Use short semantic labels that describe the section's role in the prompt.
- Labels must be lowercase snake_case and should be stable enough to reuse
  across similar prompts.
- Respect markdown and formatting cues such as headings, bullets, numbered lists,
  blockquotes, fenced code blocks, and placeholder-heavy lines.
- If you are decomposing inside an existing parent block, produce child segments
  that are meaningful within that parent rather than re-segmenting the entire prompt.
- Mark should_refine=true only when the segment contains multiple
  independently meaningful sub-instructions that should become child nodes.
- Example: a flat constraint list such as "Rules:\n1. ...\n2. ..." should usually
  set should_refine=true.
- Example: a coherent workflow such as "Procedure:\n1. ...\n2. ...\n3. ..."
  should usually stay whole with should_refine=false unless the steps split into
  larger independently meaningful sub-blocks rather than one workflow.
- Use boundary_cues to explain why the boundaries are trustworthy
  (e.g. header, markdown_heading, numbered_list, bullet_list, blockquote,
  code_fence, placeholder, example_block, paragraph).
- anchor_phrases must be short exact substrings copied from the segment that
  would help align the segment back to the source. Prefer headings or rare phrases.

Return strict JSON:
{"segments": [
  {
    "label": "<snake_case semantic label>",
    "content": "<verbatim text>",
    "reason": "<brief structural reason>",
    "should_refine": <true|false>,
    "confidence": "<low|medium|high>",
    "boundary_cues": ["<cue>", "..."],
    "anchor_phrases": ["<exact phrase>", "..."]
  }
]}"""

# Backward-compatible alias for the original guided segmentation prompt name.
SEGMENT_SYSTEM = GUIDED_SEGMENT_SYSTEM

GUIDED_CHILD_SYSTEM = GUIDED_SEGMENT_SYSTEM
FREE_CHILD_SYSTEM = FREE_SEGMENT_SYSTEM

ATOMIZE_SYSTEM = """\
You are a rule extraction agent. Given a text block from a prompt, extract \
each individual rule, instruction, or constraint as a separate item.

Rules:
- Copy each rule VERBATIM from the source (no paraphrase, no summary).
- Classify each rule's kind: constraint, requirement, guideline, heuristic.
- Preserve numbering/bullets if present.
- If a line is a section header (not a rule), skip it.
- Return one item only when it is independently meaningful if toggled on/off.
- Use boundary_cues to describe structural evidence such as numbered_list,
  bullet_list, imperative_sentence, example_block, or inline_constraint.
- anchor_phrases must be short exact substrings copied from the rule.

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
