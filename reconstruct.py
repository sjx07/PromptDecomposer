"""Reconstruct prompt text from a subset of enabled components."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .component import Component

# Standard rendering order of functional component types.
SECTION_ORDER = [
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


def reconstruct_from_tree(
    prompt: str,
    tree: List[Dict],
    enabled_ids: Set[str],
    section_separator: str = "\n\n",
) -> str:
    """Reconstruct prompt from a generic span tree.

    Enabled IDs can refer to any nodes in the tree. Enabling a node
    includes its full subtree. If a node is not enabled, reconstruction
    recurses into its children and preserves only text around enabled
    descendants.

    Args:
        prompt: Original prompt text.
        tree: Span tree from PromptDecomposer.
        enabled_ids: Set of component IDs to include.
        section_separator: Separator between rendered sections.

    Returns:
        Reconstructed prompt string.
    """
    if not enabled_ids:
        return ""

    parts: List[str] = []

    for node in tree:
        rendered = _render_node(prompt, node, enabled_ids)
        rendered = rendered.strip()
        if rendered:
            parts.append(rendered)

    return section_separator.join(parts)


def _render_node(prompt: str, node: Dict, enabled_ids: Set[str]) -> str:
    """Render one node, preserving only enabled descendants when needed."""
    start, end = node["span"]
    children = sorted(node.get("children", []), key=lambda child: child["span"][0])

    if node["id"] in enabled_ids:
        return prompt[start:end]

    if not children:
        return ""

    pieces: List[str] = []
    cursor = start
    has_enabled_descendant = False

    for child in children:
        child_start, child_end = child["span"]
        rendered_child = _render_node(prompt, child, enabled_ids)
        if rendered_child:
            has_enabled_descendant = True
            if child_start > cursor:
                pieces.append(prompt[cursor:child_start])
            pieces.append(rendered_child)
        cursor = max(cursor, child_end)

    if not has_enabled_descendant:
        return ""

    if cursor < end:
        pieces.append(prompt[cursor:end])

    text = "".join(pieces)
    text = re.sub(r"\n\s*\n", "\n", text)
    text = _remove_orphaned_headers(text)
    text = _renumber_items(text)
    return text


# ── Post-processing helpers ────────────────────────────────────────────


def _remove_orphaned_headers(text: str) -> str:
    """Remove sub-section header lines that have no content below them.

    A header is a non-indented, non-bullet line followed by only blank
    lines (or end of text) before the next header or content.
    """
    lines = text.split("\n")
    result: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this looks like a header: non-empty, not indented,
        # not a bullet/numbered item, and not a single-word label with colon
        is_header = (
            stripped
            and not line.startswith((" ", "\t"))
            and not re.match(r"^\s*[-*]\s", line)
            and not re.match(r"^\s*\d+\.\s", line)
        )

        if is_header:
            # Look ahead: is there any indented/bullet content before
            # next header or end?
            has_content = False
            j = i + 1
            while j < len(lines):
                next_stripped = lines[j].strip()
                if not next_stripped:
                    j += 1
                    continue
                # Next non-blank line: is it content (indented/bullet)?
                if lines[j].startswith((" ", "\t")) or re.match(r"^\s*[-*]\s", lines[j]) or re.match(r"^\s*\d+\.\s", lines[j]):
                    has_content = True
                break
            if has_content:
                result.append(line)
            # else: orphaned header, skip it
        else:
            result.append(line)
        i += 1
    return "\n".join(result)


def _renumber_items(text: str) -> str:
    """Renumber ordered list items (1., 2., ...) to be sequential."""
    lines = text.split("\n")
    counter = 0
    prev_was_numbered = False

    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)\d+\.\s", line)
        if m:
            if not prev_was_numbered:
                counter = 0
            counter += 1
            lines[i] = re.sub(r"^(\s*)\d+\.", rf"\g<1>{counter}.", line, count=1)
            prev_was_numbered = True
        else:
            if line.strip():
                prev_was_numbered = False

    return "\n".join(lines)


def reconstruct(
    components: List[Component],
    enabled_ids: Set[str],
    section_separator: str = "\n\n",
    rule_separator: str = "\n",
) -> str:
    """Reconstruct prompt from flat components (backward compatible).

    If components have parent_span metadata (from span-based decomposition),
    uses span-based reconstruction for those groups.  Otherwise falls back
    to simple text joining.

    Args:
        components: All available components.
        enabled_ids: Component IDs to include.
        section_separator: Separator between functional sections.
        rule_separator: Separator between items within a section.

    Returns:
        Reconstructed prompt string.
    """
    if not enabled_ids:
        return ""

    # Group by (component_type, parent_id or segment_idx)
    groups = _group_components(components)

    # Collect groups under each component type
    type_to_groups: Dict[str, List[Tuple]] = defaultdict(list)
    for key, group in groups.items():
        ctype, group_key = key
        type_to_groups[ctype].append((group_key, group))

    for ctype in type_to_groups:
        type_to_groups[ctype].sort(key=lambda x: x[0] if x[0] is not None else float("inf"))

    parts: List[str] = []
    rendered_types: set = set()

    def _render_type(section_type: str) -> None:
        if section_type not in type_to_groups:
            return
        for _group_key, group in type_to_groups[section_type]:
            enabled_in_group = [c for c in group if c.id in enabled_ids]
            if not enabled_in_group:
                continue

            # Try span-based reconstruction via parent_span metadata
            parent_span = None
            for c in group:
                ps = c.metadata.get("parent_span")
                if ps:
                    parent_span = ps
                    break

            if parent_span is not None:
                # Span-based: use segment_text approach
                # Find the component that had parent_span to get segment_text
                span_comp = next((gc for gc in group if gc.metadata.get("parent_span")), None)
                segment_text = span_comp.metadata.get("segment_text") if span_comp else None
                if segment_text:
                    rendered = segment_text
                    for gc in group:
                        if gc.id not in enabled_ids:
                            rendered = rendered.replace(gc.text, "", 1)
                    rendered = re.sub(r"\n\s*\n\s*\n", "\n\n", rendered).strip()
                    if rendered:
                        parts.append(rendered)
                    rendered_types.add(section_type)
                    return

            # Fallback: simple text join
            rendered = rule_separator.join(c.text for c in enabled_in_group)
            if rendered:
                parts.append(rendered)
        rendered_types.add(section_type)

    for section_type in SECTION_ORDER:
        _render_type(section_type)

    for section_type in sorted(type_to_groups.keys()):
        if section_type not in rendered_types:
            _render_type(section_type)

    return section_separator.join(parts)


def _group_components(
    components: List[Component],
) -> Dict[Tuple[str, Optional[str]], List[Component]]:
    """Group components by (type, parent_id or segment_idx)."""
    groups: Dict[Tuple[str, Optional[str]], List[Component]] = defaultdict(list)
    for c in components:
        group_key = c.metadata.get("parent_id") or c.metadata.get("segment_idx")
        groups[(c.component_type, group_key)].append(c)
    return groups
