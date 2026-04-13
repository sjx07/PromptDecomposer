
"""Data structures for prompt decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class DecompositionResult:
    """Tree-first decomposition result."""

    prompts: List[str]
    trees: List[Dict[str, Any]]

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
        """Serialize prompts + span trees."""
        return {
            "prompts": self.prompts,
            "trees": self.trees,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecompositionResult":
        """Deserialize from span-based JSON."""
        prompts = data.get("prompts", [])
        trees = data.get("trees", [])
        if not trees:
            raise ValueError("DecompositionResult requires 'prompts' and 'trees'.")
        return cls(prompts=prompts, trees=trees)

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

        return f"Decomposed {len(self.prompts)} prompt(s) with no tree nodes."
