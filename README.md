# PromptDecomposer

Domain-agnostic, span-based decomposition of natural-language prompts into a
functional tree. Each node points back to the source prompt by
`[start, end)` character offsets, so the original text is stored once and
every child is verifiable against it.

## Install / requirements

Python 3.10+. Requires an LLM client:

- `openai` SDK (used for both OpenAI and Gemini via the OpenAI-compatible
  Gemini endpoint). Provide `OPENAI_API_KEY` or `GEMINI_API_KEY`.
- `tiktoken` is optional; token estimates fall back to a char/4 heuristic
  if it isn't importable.

## Quick start

```python
from PromptDecomposer import decompose_prompt

result = decompose_prompt(
    prompt_text,
    model="gpt-4o-mini",       # or a gemini model via provider="gemini"
    provider="openai",          # or "gemini"
    mode="guided",              # "guided" (10-label taxonomy) or "free"
    atomize=True,               # recurse into rule/procedure-like blocks
    max_depth=4,
    min_span_chars=80,
)

print(result.summary())
tree = result.trees[0]                 # list of root nodes
leaves = result.select_nodes("leaves") # -> List[DecompositionNode]
```

For batch jobs, use `decompose_prompts([...])` or instantiate
`PromptDecomposer` directly and call `.decompose(prompt, prompt_id=...)` —
the instance also exposes a cumulative `usage_snapshot()` (prompt /
completion / total tokens, request count, and estimated-request count when
the SDK doesn't return usage).

## Output shape

`decompose` returns `{"prompt": <original>, "tree": [root, ...]}`. Every node:

```python
{
    "id": "p0:<type>[start:end]",
    "type": "<functional label>",
    "span": [start, end],           # char offsets into the prompt
    "metadata": {
        "depth": ..., "mode": ..., "parent_id": ...,
        "raw_label": ..., "reason": ..., "confidence": "low|medium|high",
        "should_refine": ..., "boundary_cues": [...], "anchor_phrases": [...],
        "alignment_score": ..., "alignment_exact": ...,
        "alignment_margin": ..., "alignment_ambiguous": ...,
        "structural_hints": [...],
    },
    "children": [...],
}
```

`DecompositionResult.select_nodes(selector)` supports `"top_level"`,
`"leaves"` (default), or `"depth:N"`.

## Modes and taxonomy

**`guided` mode** uses a fixed 10-label functional taxonomy (defined in
`prompts.py`):

```
role, task, input_description, procedure, rules, examples,
output_format, style_constraints, cot_trigger, input_slots
```

Unknown or unaligned labels normalize to `unknown`. A small alias table
folds common synonyms (`persona` → `role`, `instructions` → `task`,
`constraints` → `rules`, …).

**`free` mode** lets the LLM emit its own `snake_case` semantic labels —
useful for exploratory analysis across heterogeneous prompts.

## How decomposition works

`PromptDecomposer.decompose` runs this loop per scope:

1. **Split** the prompt into semantic units (`utils.split_units` — non-empty
   lines with their original char spans).
2. **Summarize structural hints** from surface cues per line (markdown
   headings, bullets, numbered lists, code fences, blockquotes, placeholders;
   `structure.line_cues` / `format_structure_hints`). These are passed to the
   LLM as observed cues only — they don't impose boundaries.
3. **Ask the LLM** for ordered child segments in the current scope, with a
   verbatim `content`, a label, `should_refine`, `boundary_cues`, and
   `anchor_phrases` (`GUIDED_SEGMENT_SYSTEM` / `FREE_SEGMENT_SYSTEM`;
   child scopes reuse the same system prompt).
4. **Align** each `content` back to the source with a weighted token-F1 +
   `SequenceMatcher` + containment + structural-bonus score
   (`align.align_to_source_details`). `anchor_phrases` / `boundary_cues`
   bump the structural bonus. A monotonic `cursor` keeps segments ordered
   within the scope; falls back to a full-scope search when no match is
   found after the cursor. Ambiguous matches (tight margin to the
   runner-up) are rejected in child scopes but accepted at the top level.
   Heading-only child segments (single-line content that looks like a
   heading, tagged with `header` / `markdown_heading` cues or a
   `*_heading` label) are dropped — headings belong with the block they
   introduce, not as standalone children.
5. **Gate recursion** via `_should_recurse`: requires `atomize=True`,
   `depth + 1 < max_depth`, `len(units) ≥ 2`, and either a span larger
   than `min_span_chars` or a block that looks like a flat list of items
   (`_looks_like_flat_atomic_block`), combined with
   `metadata.should_refine == True`. `should_refine` is taken from the LLM
   when it's an explicit bool, otherwise derived from boundary cues +
   list-like line ratio (`_default_should_refine`). Procedure-like leading
   anchors (`procedure`, `workflow`, `steps`, `step-by-step`, `how to`)
   suppress auto-refine so that coherent workflows stay whole.
6. **Recurse** into each qualifying child in its own sub-scope only —
   segmentation is driven entirely by the one LLM segment call per scope;
   there is no separate rule-extraction pass.

There is no post-processing pass: the tree returned by `_decompose_scope`
is what you get.

## Reconstruction

`reconstruct_from_tree(prompt, tree, enabled_ids)` rebuilds a prompt from a
subset of node ids: enabling a node includes its whole subtree; disabled
nodes disappear, and orphaned sub-section headers / empty numbering are
cleaned up. `reconstruct(components, enabled_ids)` is the legacy flat-list
variant used by `extract_components(...)`.

## Caching (batch jobs)

`batch.py` provides deterministic cache keys and simple disk-backed storage
for decomposition runs:

```python
from pathlib import Path
from PromptDecomposer.batch import (
    decompose_cache_key, load_cached_decomposition, store_cached_decomposition,
    pricing_for_model, usage_cost,
)

key = decompose_cache_key(
    prompt, model="gpt-4o-mini", provider="openai",
    mode="guided", atomize=True, temperature=0.0,
    max_depth=4, min_span_chars=80,
)
cached = load_cached_decomposition(Path(".cache"), key)
# ... if None: run decompose, then store_cached_decomposition(...)
```

The cache key includes every knob that materially changes the output
(including `max_depth` / `min_span_chars`), and is versioned via
`DECOMPOSE_CACHE_VERSION` — bump it after algorithmic changes.

`pricing_for_model(name)` + `usage_cost(usage, pricing)` turn
`usage_snapshot()` into dollar estimates for the currently supported
models in `MODEL_PRICING_PER_MILLION`.

## Module map

| File | Role |
|---|---|
| `__init__.py` | Public entrypoints: `decompose_prompt`, `decompose_prompts`, `decompose_corpus`, `PromptDecomposer` |
| `pipeline.py` | `PromptDecomposer` — recursive segment → align → gate-recurse loop, plus inline `should_refine` / heading / procedure-anchor heuristics |
| `prompts.py` | `GUIDED_SEGMENT_SYSTEM` / `FREE_SEGMENT_SYSTEM`, 10-label taxonomy, label normalization |
| `structure.py` | Per-line structural cues and `format_structure_hints` / `summarize_cues` helpers |
| `align.py` | Fuzzy token/char alignment of LLM text back to source units with anchor / cue bonuses |
| `models.py` | `DecompositionNode`, `DecompositionResult` (tree-only) |
| `tree.py` | Tree traversal, selectors (`top_level`, `leaves`, `depth:N`), node → `Component` projection |
| `extract.py` | Flatten a tree into `Component` records |
| `reconstruct.py` | Rebuild prompt text from a selection of node / component ids |
| `component.py` | `Component` dataclass — internal shape for the flat `reconstruct` / `extract_components` path; kept alongside the span-tree API |
| `batch.py` | Cache keys, disk cache, usage / cost helpers |
| `utils.py` | `split_units`, `units_to_char_span`, `call_llm_json`, token estimator |
