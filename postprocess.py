"""Postprocess span trees to remove common granularity artifacts."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .refine import looks_like_list_item


def _span_str(span: List[int]) -> str:
    return f"{span[0]}:{span[1]}"


def _node_id(prompt_id: str, node_type: str, span: List[int]) -> str:
    return f"{prompt_id}:{node_type}[{_span_str(span)}]"


def _child_id(parent_id: str, node_type: str, span: List[int]) -> str:
    return f"{parent_id}/{node_type}[{_span_str(span)}]"


def _node_text(prompt: str, node: Dict[str, Any]) -> str:
    start, end = node["span"]
    return prompt[start:end]


def _nonempty_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _nodes_are_all_list_items(prompt: str, nodes: List[Dict[str, Any]]) -> bool:
    if not nodes:
        return False
    for node in nodes:
        lines = _nonempty_lines(_node_text(prompt, node))
        if not lines or not looks_like_list_item(lines[0]):
            return False
    return True


def _is_xml_tag_line(line: str) -> bool:
    return bool(re.match(r"^</?[A-Za-z][\w:.-]*(?:\s+[^>]*)?/?>$", line.strip()))


def _is_xml_closing_line(line: str) -> bool:
    return bool(re.match(r"^</[A-Za-z][\w:.-]*>$", line.strip()))


def _is_header_line(line: str) -> bool:
    stripped = line.strip()
    if re.match(r"^#{1,6}\s+\S", stripped):
        return True
    if re.match(r"^(\*\*|__).+:(\*\*|__)$", stripped):
        return True
    if _is_xml_tag_line(stripped):
        return True
    return stripped.endswith(":") and not looks_like_list_item(stripped)


def _is_header_only_text(text: str) -> bool:
    lines = _nonempty_lines(text)
    return bool(lines) and all(_is_header_line(line) for line in lines)


def _is_xml_closing_only_text(text: str) -> bool:
    lines = _nonempty_lines(text)
    return bool(lines) and all(_is_xml_closing_line(line) for line in lines)


def _is_minor_wrapper_prefix(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return _is_header_only_text(stripped)


def _is_minor_wrapper_suffix(text: str) -> bool:
    stripped = text.strip()
    return not stripped or _is_xml_closing_only_text(stripped)


def _can_absorb_wrapper_text(prefix: str, suffix: str) -> bool:
    return _is_minor_wrapper_prefix(prefix) and _is_minor_wrapper_suffix(suffix)


def _record_postprocess(node: Dict[str, Any], action: str, **payload: Any) -> None:
    metadata = node.setdefault("metadata", {})
    actions = metadata.setdefault("postprocess_actions", [])
    actions.append({"action": action, **payload})


def _line_spans(prompt: str, span: List[int]) -> List[tuple[int, int, str]]:
    start, end = span
    lines: List[tuple[int, int, str]] = []
    for match in re.finditer(r"[^\n]*\n?", prompt[start:end]):
        raw = match.group().rstrip("\n")
        if not raw.strip():
            continue
        line_start = start + match.start()
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        lines.append((line_start + leading, line_start + trailing, raw.strip()))
    return lines


def _attach_span_metadata(
    node: Dict[str, Any],
    *,
    field: str,
    span: List[int],
    text: str,
    source_type: str = "header",
) -> None:
    metadata = node.setdefault("metadata", {})
    spans_key = f"{field}_spans"
    texts_key = f"{field}_texts"
    metadata.setdefault(spans_key, []).append(list(span))
    metadata.setdefault(texts_key, []).append(text)
    metadata.setdefault(f"{field}_source_types", []).append(source_type)
    if field not in metadata:
        metadata[field] = text
    singular_span_key = f"{field}_span"
    if singular_span_key not in metadata:
        metadata[singular_span_key] = list(span)


def _next_line_start(lines: List[tuple[int, int, str]], current_index: int, fallback_end: int) -> int | None:
    if current_index + 1 >= len(lines):
        return None
    return lines[current_index + 1][0]


def _trim_leading_header_from_first_child(prompt: str, parent: Dict[str, Any]) -> None:
    children = parent.get("children", [])
    if not children:
        return

    first = children[0]
    if first["span"][0] != parent["span"][0]:
        return

    lines = _line_spans(prompt, first["span"])
    if not lines:
        return
    line_start, line_end, text = lines[0]
    if not _is_header_line(text):
        return

    _attach_span_metadata(
        parent,
        field="title",
        span=[line_start, line_end],
        text=text,
        source_type=first.get("type", "header"),
    )
    old_span = list(first["span"])
    next_start = _next_line_start(lines, 0, first["span"][1])
    if next_start is None:
        children.pop(0)
    else:
        first["span"][0] = next_start
    _record_postprocess(
        parent,
        "promoted_child_header_to_parent_metadata",
        child_type=first.get("type", "unknown"),
        header_span=[line_start, line_end],
        old_child_span=old_span,
        new_child_span=list(first["span"]) if first in children else None,
    )


def _attach_wrapper_text_to_child(
    prompt: str,
    child: Dict[str, Any],
    *,
    field: str,
    span: List[int],
    source_type: str,
) -> None:
    text = prompt[span[0]:span[1]].strip()
    if not text:
        return
    _attach_span_metadata(child, field=field, span=span, text=text, source_type=source_type)


def _transfer_metadata_field(source: Dict[str, Any], target: Dict[str, Any], field: str) -> None:
    source_metadata = source.get("metadata", {})
    target_metadata = target.setdefault("metadata", {})
    for suffix in ("", "_span", "_spans", "_texts", "_source_types"):
        key = f"{field}{suffix}"
        if key not in source_metadata:
            continue
        value = source_metadata[key]
        if suffix in {"_spans", "_texts", "_source_types"}:
            existing = list(target_metadata.get(key, []))
            incoming = list(value)
            target_metadata[key] = incoming + existing if field == "title" else existing + incoming
        else:
            target_metadata[key] = value if field == "title" else target_metadata.setdefault(key, value)


def _span_contains_recorded(node: Dict[str, Any], field: str, span: List[int]) -> bool:
    for recorded in node.get("metadata", {}).get(f"{field}_spans", []):
        if span[0] <= recorded[0] and recorded[1] <= span[1]:
            return True
    return False


def postprocess_tree(
    prompt_id: str,
    prompt: str,
    tree: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize common LLM granularity drift without another model call."""
    cleaned = _postprocess_siblings(prompt, tree)
    cleaned = _hoist_prompt_wrapper(prompt, cleaned)
    _refresh_tree_ids(prompt_id, cleaned, parent_id=None, depth=0)
    return cleaned


def _postprocess_siblings(
    prompt: str,
    nodes: List[Dict[str, Any]],
    parent: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    for node in nodes:
        children = node.get("children", [])
        if children:
            node["children"] = _postprocess_siblings(prompt, children, parent=node)
            _mark_list_item_children(prompt, node)
            node = _collapse_single_child_wrappers(prompt, node)
        processed.append(node)

    if parent is not None:
        parent["children"] = processed
        _trim_leading_header_from_first_child(prompt, parent)
        processed = parent.get("children", [])
    return _merge_header_siblings(prompt, processed)


def _merge_header_siblings(prompt: str, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach heading/label-only siblings to the adjacent content block."""
    merged: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(nodes):
        node = nodes[idx]
        if not node.get("children") and _is_header_only_text(_node_text(prompt, node)):
            if merged and _is_xml_closing_only_text(_node_text(prompt, node)):
                previous = merged[-1]
                _attach_wrapper_text_to_child(
                    prompt,
                    previous,
                    field="suffix",
                    span=list(node["span"]),
                    source_type=node.get("type", "unknown"),
                )
                _record_postprocess(
                    previous,
                    "attached_header_suffix",
                    header_type=node.get("type", "unknown"),
                    header_span=list(node["span"]),
                )
                idx += 1
                continue
            if idx + 1 < len(nodes):
                next_node = nodes[idx + 1]
                _attach_wrapper_text_to_child(
                    prompt,
                    next_node,
                    field="title",
                    span=list(node["span"]),
                    source_type=node.get("type", "unknown"),
                )
                _record_postprocess(
                    next_node,
                    "attached_header_prefix",
                    header_type=node.get("type", "unknown"),
                    header_span=list(node["span"]),
                )
                idx += 1
                continue

        merged.append(node)
        idx += 1
    return merged


def _collapse_single_child_wrappers(prompt: str, node: Dict[str, Any]) -> Dict[str, Any]:
    """Remove wrapper levels that add only a heading, preamble, or tag."""
    while len(node.get("children", [])) == 1:
        child = node["children"][0]
        parent_start, parent_end = node["span"]
        child_start, child_end = child["span"]
        prefix = prompt[parent_start:child_start]
        suffix = prompt[child_end:parent_end]
        same_span = (parent_start, parent_end) == (child_start, child_end)

        if not same_span and not _can_absorb_wrapper_text(prefix, suffix):
            break

        _transfer_metadata_field(node, child, "title")
        _transfer_metadata_field(node, child, "preamble")
        _transfer_metadata_field(node, child, "suffix")

        prefix_span = [parent_start, child_start]
        suffix_span = [child_end, parent_end]
        if prefix.strip() and not _span_contains_recorded(node, "title", prefix_span):
            _attach_wrapper_text_to_child(
                prompt,
                child,
                field="preamble",
                span=prefix_span,
                source_type=node.get("type", "unknown"),
            )
        if suffix.strip() and not _span_contains_recorded(node, "suffix", suffix_span):
            _attach_wrapper_text_to_child(
                prompt,
                child,
                field="suffix",
                span=suffix_span,
                source_type=node.get("type", "unknown"),
            )
        _record_postprocess(
            child,
            "collapsed_single_child_wrapper",
            parent_type=node.get("type", "unknown"),
            parent_span=list(node["span"]),
            child_span=list(child["span"]),
        )
        node = child
    return node


def _mark_list_item_children(prompt: str, node: Dict[str, Any]) -> None:
    """Preserve same-label rule/procedure trees while marking item leaves."""
    children = node.get("children", [])
    if len(children) < 2:
        return

    parent_type = node.get("type")
    for child in children:
        child_lines = _nonempty_lines(_node_text(prompt, child))
        if (
            child.get("type") == parent_type
            and child_lines
            and looks_like_list_item(child_lines[0])
        ):
            child.setdefault("metadata", {})["structural_role"] = "list_item"
            child["kind"] = child.get("kind") or "item"
            _record_postprocess(
                child,
                "marked_list_item",
                parent_type=parent_type or "unknown",
            )


def _hoist_prompt_wrapper(prompt: str, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Use the prompt itself as the root when a single root is just a title wrapper."""
    if len(nodes) != 1:
        return nodes

    wrapper = nodes[0]
    children = wrapper.get("children", [])
    if not children:
        return nodes
    if _nodes_are_all_list_items(prompt, children):
        return nodes

    first = children[0]
    last = children[-1]
    wrapper_start, wrapper_end = wrapper["span"]
    prefix = prompt[wrapper_start:first["span"][0]]
    suffix = prompt[last["span"][1]:wrapper_end]
    if not (prefix.strip() or suffix.strip()):
        return nodes
    if not _can_absorb_wrapper_text(prefix, suffix):
        return nodes

    if prefix.strip():
        _attach_wrapper_text_to_child(
            prompt,
            first,
            field="title" if _is_header_only_text(prefix) else "preamble",
            span=[wrapper_start, first["span"][0]],
            source_type=wrapper.get("type", "unknown"),
        )
        _record_postprocess(
            first,
            "attached_prompt_wrapper_prefix",
            wrapper_type=wrapper.get("type", "unknown"),
            wrapper_span=list(wrapper["span"]),
        )
    if suffix.strip():
        _attach_wrapper_text_to_child(
            prompt,
            last,
            field="suffix",
            span=[last["span"][1], wrapper_end],
            source_type=wrapper.get("type", "unknown"),
        )
        _record_postprocess(
            last,
            "attached_prompt_wrapper_suffix",
            wrapper_type=wrapper.get("type", "unknown"),
            wrapper_span=list(wrapper["span"]),
        )

    for child in children:
        _record_postprocess(
            child,
            "hoisted_prompt_wrapper",
            wrapper_type=wrapper.get("type", "unknown"),
            wrapper_span=list(wrapper["span"]),
        )
    return children


def _refresh_tree_ids(
    prompt_id: str,
    nodes: List[Dict[str, Any]],
    *,
    parent_id: str | None,
    depth: int,
) -> None:
    for node in nodes:
        node_type = node.get("type") or "unknown"
        span = list(node["span"])
        node["span"] = span
        if parent_id is None:
            node["id"] = _node_id(prompt_id, node_type, span)
        else:
            node["id"] = _child_id(parent_id, node_type, span)

        metadata = node.setdefault("metadata", {})
        metadata["depth"] = depth
        metadata["parent_id"] = parent_id

        children = node.get("children", [])
        node["children"] = children
        _refresh_tree_ids(
            prompt_id,
            children,
            parent_id=node["id"],
            depth=depth + 1,
        )
