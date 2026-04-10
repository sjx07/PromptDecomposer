"""Shared span-based component type for decomposition and attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class Component:
    """A span-referenced prompt unit used as a Shapley player."""

    id: str
    span: Tuple[int, int]
    component_type: str
    source_prompt_idx: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    _text: str = field(default="", repr=False)

    @classmethod
    def make(cls, text: str, label: str = "rules", id: Optional[str] = None) -> "Component":
        if id is None:
            id = f"c{abs(hash(text)) % 10000}"
        return cls(id=id, span=(0, 0), component_type=label, _text=text)

    @property
    def text(self) -> str:
        return self._text

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Component):
            return NotImplemented
        return self.id == other.id
