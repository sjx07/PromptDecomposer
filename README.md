# PromptDecomposer

`decompose/` turns raw prompt text into a tree. This package is only about decomposition.

If you are using the full repo, the decomposition settings are usually passed through the top-level runner. If you are using this package on its own, the Python API below is the main entrypoint.

## The Files That Matter

- `pipeline.py`
  Main recursive decomposition logic.
- `models.py`
  `DecompositionNode` and `DecompositionResult`.
- `prompts.py`
  LLM prompts used for segmentation and refinement.
- `align.py`
  Maps model output back to exact text spans.
- `structure.py`
  Extracts structural hints such as headings and lists.
- `tree.py`
  Tree traversal and selector helpers.
- `reconstruct.py`
  Rebuilds prompt text from selected nodes.
- `batch.py`
  Batch processing, caching, and usage accounting helpers.

## Direct Python Usage

```python
from decompose import decompose_prompt

result = decompose_prompt(
    prompt,
    model="gpt-5.4",
    provider="openai",
    mode="free",
    atomize=True,
    max_depth=5,
    min_span_chars=60,
)
```

Useful result methods:

- `summary()`
- `root_nodes()`
- `select_nodes(selector)`
- `to_dict()`

## The Main Knobs

### `mode`

- `guided`
  More stable labels and structure.
- `free`
  More open labels and freer grouping.

Use `guided` when you want consistency across prompts.
Use `free` when you want the model to discover structure more openly.

### `atomize`

If `True`, the decomposer can keep refining nodes into smaller parts.

### `max_depth`

Caps recursion depth.
Increase it when the tree stops too early.
Lower it when the tree is getting too fragmented.

### `min_span_chars`

Prevents the decomposer from trying to split very small spans.
Increase it when the tree is over-splitting.

## Using It Through The Repo Scripts

In the host repo, decomposition is usually run either through:

- [`../scripts/run_prompt_shapley.py`](../scripts/run_prompt_shapley.py)
- [`../scripts/run_decompose_corpus.py`](../scripts/run_decompose_corpus.py)

The decomposition config controls:

- input source
- output location
- model/provider
- recursion behavior

Example fields:

- `input_file`
- `output_dir`
- `artifact_name`
- `prompt_field`
- `decompose_model`
- `decompose_provider`
- `mode`
- `atomize`
- `max_depth`
- `min_span_chars`

Use `output_dir` as the base folder for one run. Use `artifact_name` for the saved decomposition JSON inside that folder. If omitted, it defaults to `decomposition.json`.

When the full runner is used, it also writes `decomposition.html` in the same `output_dir`.

## Visualization

Decomposition itself only produces data.
HTML inspection belongs to the visualization layer in the host repo.

For stored results, the useful tools are:

- [`../scripts/render_decomposition_html.py`](../scripts/render_decomposition_html.py)
- [`../scripts/serve_decompositions.py`](../scripts/serve_decompositions.py)

Use the viewer to check:

- whether headings became their own nodes incorrectly
- whether procedures were split too aggressively
- whether the tree depth is sensible

## Practical Debugging Order

When decomposition looks wrong, inspect files in this order:

1. `prompts.py`
2. `pipeline.py`
3. `align.py`
4. `structure.py`

That is usually where behavior issues come from.
