"""Extract flat Component lists from span-based decomposition trees."""

from __future__ import annotations

from typing import Any, Dict, List

from .component import Component
from .tree import component_from_node, select_nodes


def extract_components(
    prompts: List[str],
    trees: List[Dict[str, Any]],
    selector: str = "leaves",
) -> List[Component]:
    """Flatten span trees into a list of Components with resolved text.

    By default, leaf nodes become components. Other projections can be
    requested with selectors such as ``top_level`` or ``depth:N``.
    """
    components: List[Component] = []
    for prompt_idx, tree_data in enumerate(trees):
        prompt = prompts[prompt_idx]
        for node, depth, parent, resolved_type in select_nodes(tree_data, selector):
            components.append(
                component_from_node(
                    prompt,
                    prompt_idx,
                    node,
                    depth,
                    parent,
                    resolved_type,
                )
            )
    return components
