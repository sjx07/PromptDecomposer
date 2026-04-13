# Decompose

`prompt_shapley.decompose` turns raw prompts into a tree-shaped decomposition IR.
It is standalone: it does not require attribution or Shapley computation.

## What It Produces

The decomposition result is a prompt-aligned tree with span offsets:

- `DecompositionResult.prompts`: original prompt strings
- `DecompositionResult.trees`: serialized span trees
- `DecompositionResult.root_nodes()`: root nodes as `DecompositionNode`
- `DecompositionResult.select_nodes(selector)`: projected frontier for a given selector

Each node stores:

- `id`
- `span = [start, end)`
- `type`
- `kind` when the extractor emits a finer label
- `children`
- optional `metadata`

Spans always refer back to the original prompt text. The tree is the source of truth.

## Public API

Use the package entrypoints from `prompt_shapley.decompose`:

- `decompose_prompt(prompt, ...)`
- `decompose_prompts(prompts, ...)`
- `decompose_corpus(prompts, ...)`
- `PromptDecomposer`
- `DecompositionResult`

Typical configuration knobs:

- `model`: decomposition model
- `provider`: model provider
- `mode`: `guided` or `free`
- `atomize`: enable recursive refinement
- `max_depth`: recursion limit
- `min_span_chars`: minimum span size before refinement

## Quick Start

### Decompose one prompt

```python
from pathlib import Path
from prompt_shapley.decompose import decompose_prompt

prompt = Path("template.txt").read_text()
result = decompose_prompt(
    prompt,
    model="gpt-5.4",
    provider="openai",
    mode="free",
    atomize=True,
    max_depth=5,
    min_span_chars=60,
)

print(result.summary())
print(result.root_nodes())
```

### Render the tree to HTML

```python
html = result.to_tree_html(title="Template decomposition")
Path("decomposition.html").write_text(html)
```

### Run over a corpus

```python
from prompt_shapley.decompose import decompose_corpus

prompts = ["...", "..."]
result = decompose_corpus(prompts, mode="guided", atomize=True)
```

## Batch Decomposition

For larger corpora, use the batch runner:

```bash
python scripts/run_decompose_corpus.py \
  --input-file decompose/tasks/claudecode.json \
  --prompt-field full_text \
  --model gpt-5.4 \
  --mode free \
  --batch-size 10
```

The batch runner writes:

- `decompositions.jsonl` for per-prompt results
- `summary.json` for aggregate metrics
- cached decomposition entries under `.cache/decompose/`

Use `--force` to re-send prompts even if a cache entry exists.

## Inspecting Results

For a saved decomposition record, render an HTML view with:

```bash
python scripts/render_decomposition_html.py \
  --input-file results/.../decompositions.jsonl \
  --source-index 0 \
  --output-file results/.../view.html
```

The HTML view shows:

- the original prompt
- the decomposition tree
- node spans and metadata
- navigation and source metadata when available

## Design Notes

- Recursive decomposition is parent-local. Each refinement step works inside the current span.
- `guided` mode biases labels toward a taxonomy.
- `free` mode lets the model discover labels more freely while keeping the same tree contract.
- The tree is generic; attribution can choose a frontier later, but decomposition itself does not depend on it.

## Testing

Run the decomposition tests with:

```bash
pytest prompt_shapley/tests/test_decompose.py
```

