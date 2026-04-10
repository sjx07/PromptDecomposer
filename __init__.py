"""Decompose raw prompts into atomic components.

Uses PromptDecomposer (domain-agnostic, span-based) by default.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from .component import Component
from .models import DecompositionResult
from .extract import extract_components
from .pipeline import PromptDecomposer
from .reconstruct import reconstruct, reconstruct_from_tree
from .utils import components_from_dicts, deduplicate

logger = logging.getLogger(__name__)


def decompose_prompt(
    prompt: str,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    atomize: bool = True,
    mode: str = "guided",
    max_depth: int = 4,
    min_span_chars: int = 80,
) -> DecompositionResult:
    """Decompose a single prompt into a standalone tree artifact."""
    return decompose_prompts(
        [prompt],
        model=model,
        provider=provider,
        atomize=atomize,
        mode=mode,
        max_depth=max_depth,
        min_span_chars=min_span_chars,
    )


def decompose_prompts(
    prompts: List[str],
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    atomize: bool = True,
    mode: str = "guided",
    max_depth: int = 4,
    min_span_chars: int = 80,
    deduplicate_components: bool = False,
) -> DecompositionResult:
    """Decompose prompts into atomic components via LLM with align-back.

    Each prompt is segmented into functional groups, atomized into
    individual rules, and aligned back to the original text as spans.

    Args:
        prompts: Raw prompt strings.
        model: LLM model for decomposition.
        provider: "openai" or "gemini".
        atomize: Allow recursive refinement beyond the top level.
        mode: "guided" for taxonomy-aware segmentation, "free" for open labels.
        max_depth: Maximum decomposition depth including top-level sections.
        min_span_chars: Minimum node size before recursive refinement is attempted.
        deduplicate_components: Remove duplicate components by normalized text.

    Returns:
        DecompositionResult with span-based trees and flat components.
    """
    decomposer = PromptDecomposer(
        model=model,
        provider=provider,
        atomize=atomize,
        mode=mode,
        max_depth=max_depth,
        min_span_chars=min_span_chars,
    )

    trees: List[Dict] = []
    for prompt_idx, prompt_text in enumerate(prompts):
        result = decomposer.decompose(prompt_text, prompt_id=f"p{prompt_idx}")
        trees.append(result)

    if deduplicate_components:
        components = deduplicate(extract_components(prompts, trees))
        return DecompositionResult(
            prompts=prompts, trees=trees, components=components,
        )

    return DecompositionResult(
        prompts=prompts, trees=trees,
    )


def decompose_corpus(
    prompts: List[str],
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    atomize: bool = True,
    mode: str = "guided",
    max_depth: int = 4,
    min_span_chars: int = 80,
    deduplicate_components: bool = False,
) -> DecompositionResult:
    """Alias for decomposition-focused callers working over prompt batches."""
    return decompose_prompts(
        prompts,
        model=model,
        provider=provider,
        atomize=atomize,
        mode=mode,
        max_depth=max_depth,
        min_span_chars=min_span_chars,
        deduplicate_components=deduplicate_components,
    )


__all__ = [
    "PromptDecomposer",
    "decompose_prompt",
    "decompose_corpus",
    "decompose_prompts",
    "extract_components",
    "reconstruct",
    "reconstruct_from_tree",
    "components_from_dicts",
    "deduplicate",
]
