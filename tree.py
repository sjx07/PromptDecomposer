"""Helpers for traversing and projecting generic decomposition trees."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from .component import Component
from .models import DecompositionNode


TreeNode = Dict[str, Any]
NodeRecord = Tuple[TreeNode, int, Optional[TreeNode], str]


def get_roots(tree_data: Any) -> List[TreeNode]:
    """Return the top-level nodes from a tree payload or raw node list."""
    if isinstance(tree_data, dict):
        return tree_data.get("tree", [])
    if isinstance(tree_data, list):
        return tree_data
    return []


def canonical_selector(selector: str) -> str:
    """Normalize legacy selector names to the generic tree selectors."""
    aliases = {
        "segment": "top_level",
        "rule": "leaves",
        "leaf": "leaves",
    }
    return aliases.get(selector, selector)


def iter_nodes(
    tree_data: Any,
    depth: int = 0,
    parent: Optional[TreeNode] = None,
    inherited_type: Optional[str] = None,
) -> Iterable[NodeRecord]:
    """Yield every node in the tree with depth and resolved component type."""
    for node in get_roots(tree_data):
        resolved_type = node.get("type") or inherited_type or "unknown"
        yield node, depth, parent, resolved_type
        yield from iter_nodes(
            node.get("children", []),
            depth=depth + 1,
            parent=node,
            inherited_type=resolved_type,
        )


def _matches_selector(node: TreeNode, depth: int, selector: str) -> bool:
    selector = canonical_selector(selector)
    children = node.get("children", [])

    if selector == "top_level":
        return depth == 0
    if selector == "leaves":
        return not children
    if selector.startswith("depth:"):
        try:
            target_depth = int(selector.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid selector: {selector!r}.") from exc
        return depth == target_depth

    raise ValueError(
        f"Unknown selector: {selector!r}. "
        "Use 'top_level', 'leaves', 'depth:N', or legacy 'segment'/'rule'."
    )


def select_nodes(tree_data: Any, selector: str = "leaves") -> List[NodeRecord]:
    """Select a projection of the tree by selector."""
    return [
        record
        for record in iter_nodes(tree_data)
        if _matches_selector(record[0], record[1], selector)
    ]


def build_node(
    prompt: str,
    prompt_idx: int,
    node: TreeNode,
    inherited_type: Optional[str] = None,
) -> DecompositionNode:
    """Convert one raw tree node into a typed decomposition node."""
    node_type = node.get("type") or inherited_type or "unknown"
    children = [
        build_node(prompt, prompt_idx, child, node_type)
        for child in node.get("children", [])
    ]
    span = tuple(node["span"])
    metadata = dict(node.get("metadata", {}))
    return DecompositionNode(
        id=node["id"],
        span=span,
        node_type=node_type,
        source_prompt_idx=prompt_idx,
        kind=node.get("kind"),
        children=children,
        metadata=metadata,
        _text=prompt[span[0]:span[1]],
    )


def build_nodes(prompt: str, prompt_idx: int, tree_data: Any) -> List[DecompositionNode]:
    """Convert the raw roots for one prompt into decomposition nodes."""
    return [build_node(prompt, prompt_idx, node) for node in get_roots(tree_data)]


def select_decomposition_nodes(
    prompt: str,
    prompt_idx: int,
    tree_data: Any,
    selector: str = "leaves",
) -> List[DecompositionNode]:
    """Project raw tree data into decomposition nodes with the given selector."""
    return [
        build_node(prompt, prompt_idx, node, resolved_type)
        for node, _depth, _parent, resolved_type in select_nodes(tree_data, selector)
    ]


def component_from_node(
    prompt: str,
    prompt_idx: int,
    node: TreeNode,
    depth: int,
    parent: Optional[TreeNode],
    resolved_type: str,
) -> Component:
    """Convert a tree node into a Shapley component."""
    span = tuple(node["span"])
    metadata: Dict[str, Any] = {"depth": depth}
    if "kind" in node:
        metadata["kind"] = node["kind"]
    if parent is not None:
        parent_span = list(parent["span"])
        metadata["parent_id"] = parent["id"]
        metadata["parent_span"] = parent_span
        metadata["segment_text"] = prompt[parent_span[0]:parent_span[1]]

    return Component(
        id=node["id"],
        span=span,
        component_type=resolved_type,
        source_prompt_idx=prompt_idx,
        metadata=metadata,
        _text=prompt[span[0]:span[1]],
    )
