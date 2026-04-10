
"""Data structures for prompt decomposition."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DecompositionNode:
    """A generic node in a prompt decomposition tree."""

    id: str
    span: Tuple[int, int]
    node_type: str
    source_prompt_idx: int = 0
    kind: Optional[str] = None
    children: List["DecompositionNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _text: str = field(default="", repr=False)

    @property
    def text(self) -> str:
        return self._text

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "type": self.node_type,
            "span": list(self.span),
        }
        if self.kind is not None:
            data["kind"] = self.kind
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        if self.children:
            data["children"] = [child.to_dict() for child in self.children]
        else:
            data["children"] = []
        return data


def _coerce_legacy_components(records: Optional[List[Any]]) -> Optional[List[Dict[str, Any]]]:
    if records is None:
        return None
    coerced: List[Dict[str, Any]] = []
    for record in records:
        if isinstance(record, dict):
            coerced.append(dict(record))
        elif is_dataclass(record):
            coerced.append(asdict(record))
        elif hasattr(record, "to_dict"):
            coerced.append(record.to_dict())
        else:
            coerced.append({
                "id": getattr(record, "id", None),
                "span": list(getattr(record, "span", (0, 0))),
                "component_type": getattr(record, "component_type", "unknown"),
                "source_prompt_idx": getattr(record, "source_prompt_idx", 0),
                "metadata": dict(getattr(record, "metadata", {}) or {}),
                "text": getattr(record, "text", ""),
            })
    return coerced


@dataclass(init=False)
class DecompositionResult:
    """Tree-first decomposition result.

    The span tree is the source of truth. ``legacy_components`` only exists
    for compatibility with older flat serializations.
    """

    prompts: List[str]
    trees: List[Dict[str, Any]]
    legacy_components: Optional[List[Dict[str, Any]]] = field(default=None, repr=False)

    def __init__(
        self,
        prompts: List[str],
        trees: Optional[List[Dict[str, Any]]] = None,
        components: Optional[List[Any]] = None,
        legacy_components: Optional[List[Dict[str, Any]]] = None,
    ):
        self.prompts = prompts
        self.trees = trees or []
        raw_components = legacy_components if legacy_components is not None else components
        self.legacy_components = _coerce_legacy_components(raw_components)

    def root_nodes(self, prompt_idx: int = 0) -> List[DecompositionNode]:
        """Return the root decomposition nodes for one prompt."""
        if not self.trees or not self.prompts:
            return []

        from .tree import build_nodes

        return build_nodes(self.prompts[prompt_idx], prompt_idx, self.trees[prompt_idx])

    def select_nodes(
        self,
        selector: str = "leaves",
        prompt_idx: Optional[int] = None,
    ) -> List[DecompositionNode]:
        """Project the stored trees into generic decomposition nodes."""
        if not self.trees or not self.prompts:
            return []

        from .tree import select_decomposition_nodes

        if prompt_idx is not None:
            return select_decomposition_nodes(
                self.prompts[prompt_idx],
                prompt_idx,
                self.trees[prompt_idx],
                selector,
            )

        nodes: List[DecompositionNode] = []
        for idx, prompt in enumerate(self.prompts):
            nodes.extend(select_decomposition_nodes(prompt, idx, self.trees[idx], selector))
        return nodes

    def to_dict(self) -> Dict[str, Any]:
        """Serialize prompts + span trees only."""
        data: Dict[str, Any] = {
            "prompts": self.prompts,
            "trees": self.trees,
        }
        if not self.trees and self.legacy_components is not None:
            data["components"] = self.legacy_components
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecompositionResult":
        """Deserialize from span-based JSON or legacy flat format."""
        prompts = data.get("prompts", [])
        trees = data.get("trees", [])

        if trees:
            return cls(prompts=prompts, trees=trees)

        flat = data.get("components", [])
        return cls(prompts=prompts, legacy_components=flat)

    def summary(self) -> str:
        if self.trees:
            leaf_nodes = self.select_nodes("leaves")
            type_counts: Dict[str, int] = {}
            for node in leaf_nodes:
                type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1
            lines = [
                f"Decomposed {len(self.prompts)} prompt(s) into {len(leaf_nodes)} leaf node(s):"
            ]
            for node_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {node_type}: {count}")
            return "\n".join(lines)

        if self.legacy_components is not None:
            type_counts: Dict[str, int] = {}
            for record in self.legacy_components:
                component_type = str(record.get("component_type", "unknown"))
                type_counts[component_type] = type_counts.get(component_type, 0) + 1
            lines = [
                f"Loaded {len(self.prompts)} prompt(s) with {len(self.legacy_components)} legacy component(s):"
            ]
            for component_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {component_type}: {count}")
            return "\n".join(lines)

        return f"Decomposed {len(self.prompts)} prompt(s) with no tree nodes."

