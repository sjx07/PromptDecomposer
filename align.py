"""Fuzzy alignment engine for matching LLM output back to source text spans."""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Sequence, Tuple


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens for fuzzy matching."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalize(text: str) -> str:
    """Normalize for character-level comparison."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _looks_like_header(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and stripped.endswith(":") and not re.match(r"^\d+\.", stripped)


def _looks_like_markdown_heading(text: str) -> bool:
    return bool(re.match(r"^\s{0,3}#{1,6}\s+\S", text))


def _looks_like_numbered_item(text: str) -> bool:
    return bool(re.match(r"^\s*\d+\.\s", text))


def _looks_like_bullet_item(text: str) -> bool:
    return bool(re.match(r"^\s*[-*]\s", text))


def _looks_like_code_fence(text: str) -> bool:
    return bool(re.match(r"^\s*```", text))


def _looks_like_blockquote(text: str) -> bool:
    return bool(re.match(r"^\s*>\s", text))


def _token_f1(left: List[str], right: List[str]) -> float:
    """Token-level F1 between two token lists."""
    if not left or not right:
        return 0.0
    left_counts: Dict[str, int] = {}
    right_counts: Dict[str, int] = {}
    for t in left:
        left_counts[t] = left_counts.get(t, 0) + 1
    for t in right:
        right_counts[t] = right_counts.get(t, 0) + 1
    overlap = sum(min(c, right_counts.get(t, 0)) for t, c in left_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2 * precision * recall / (precision + recall)


def _match_score(
    content: str,
    content_tokens: List[str],
    window: str,
    window_tokens: List[str],
    *,
    anchor_phrases: Optional[Sequence[str]] = None,
    boundary_cues: Optional[Sequence[str]] = None,
    first_unit: str = "",
    last_unit: str = "",
) -> float:
    """Score how well a text window matches target content.

    Weighted combination:
      45% token F1
      20% character-level SequenceMatcher ratio
      16% substring containment
      19% structural bonus from anchors / boundary cues
    """
    if not window_tokens:
        return 0.0
    f1 = _token_f1(content_tokens, window_tokens)
    cn = _normalize(content)
    wn = _normalize(window)
    if not cn or not wn:
        return f1
    char_ratio = difflib.SequenceMatcher(None, cn, wn).ratio()
    contains = 1.0 if (cn in wn or wn in cn) else 0.0
    structural = _structural_bonus(
        window=window,
        anchor_phrases=anchor_phrases,
        boundary_cues=boundary_cues,
        first_unit=first_unit,
        last_unit=last_unit,
    )
    return 0.45 * f1 + 0.20 * char_ratio + 0.16 * contains + 0.19 * structural


def _structural_bonus(
    window: str,
    anchor_phrases: Optional[Sequence[str]],
    boundary_cues: Optional[Sequence[str]],
    first_unit: str,
    last_unit: str,
) -> float:
    """Reward matches that satisfy explicit structural evidence."""
    cues = {cue.strip().lower() for cue in boundary_cues or [] if cue}
    bonus = 0.0
    normalized_window = _normalize(window)

    anchors = [phrase.strip() for phrase in anchor_phrases or [] if phrase and phrase.strip()]
    if anchors:
        matched = sum(1 for phrase in anchors if _normalize(phrase) in normalized_window)
        bonus += 0.6 * (matched / len(anchors))

    if "header" in cues and _looks_like_header(first_unit):
        bonus += 0.15
    if "markdown_heading" in cues and _looks_like_markdown_heading(first_unit):
        bonus += 0.15
    if "numbered_list" in cues and (
        _looks_like_numbered_item(first_unit) or _looks_like_numbered_item(last_unit)
    ):
        bonus += 0.15
    if "bullet_list" in cues and (
        _looks_like_bullet_item(first_unit) or _looks_like_bullet_item(last_unit)
    ):
        bonus += 0.15
    if "placeholder" in cues and ("{" in window or "}" in window):
        bonus += 0.1
    if "example_block" in cues and "example" in normalized_window:
        bonus += 0.1
    if "code_fence" in cues and (_looks_like_code_fence(first_unit) or _looks_like_code_fence(last_unit)):
        bonus += 0.1
    if "blockquote" in cues and (_looks_like_blockquote(first_unit) or _looks_like_blockquote(last_unit)):
        bonus += 0.1
    if "paragraph" in cues and not cues.intersection({"header", "numbered_list", "bullet_list"}):
        bonus += 0.05

    return min(bonus, 1.0)


def align_to_source_details(
    content: str,
    units: List[str],
    unit_tokens: List[List[str]],
    min_start: int = 0,
    threshold: float = 0.30,
    max_window_ratio: float = 2.5,
    min_window_tokens: int = 24,
    anchor_phrases: Optional[Sequence[str]] = None,
    boundary_cues: Optional[Sequence[str]] = None,
    ambiguity_margin: float = 0.05,
) -> Optional[Dict[str, float | int | bool]]:
    """Find the best matching unit window and return score diagnostics."""
    n = len(units)
    if n == 0:
        return None
    content_tokens = _tokenize(content)
    if not content_tokens:
        return None

    max_tok = max(min_window_tokens, int(len(content_tokens) * max_window_ratio))
    start_floor = max(0, min(min_start, n - 1))
    normalized_content = _normalize(content)

    candidates: List[Dict[str, float | int | bool]] = []
    for start in range(start_floor, n):
        win_tokens: List[str] = []
        for end in range(start, n):
            win_tokens.extend(unit_tokens[end])
            if len(win_tokens) > max_tok and end > start:
                break
            window_text = "\n".join(units[start : end + 1])
            exact = _normalize(window_text) == normalized_content
            score = _match_score(
                content,
                content_tokens,
                window_text,
                win_tokens,
                anchor_phrases=anchor_phrases,
                boundary_cues=boundary_cues,
                first_unit=units[start],
                last_unit=units[end],
            )
            if exact:
                score = max(score, 0.99)
            candidates.append({
                "start": start,
                "end": end,
                "score": score,
                "exact": exact,
            })

    if not candidates:
        return None

    candidates.sort(
        key=lambda candidate: (
            bool(candidate["exact"]),
            float(candidate["score"]),
            -int(candidate["end"]) + int(candidate["start"]),
        ),
        reverse=True,
    )
    best = candidates[0]
    if float(best["score"]) < threshold:
        return None

    second = next(
        (
            candidate for candidate in candidates[1:]
            if candidate["start"] != best["start"] or candidate["end"] != best["end"]
        ),
        None,
    )
    second_score = float(second["score"]) if second is not None else 0.0
    margin = float(best["score"]) - second_score
    ambiguous = second is not None and second_score >= threshold and margin < ambiguity_margin
    if bool(best["exact"]) and second is not None and not bool(second["exact"]):
        ambiguous = False

    return {
        "start": int(best["start"]),
        "end": int(best["end"]),
        "score": float(best["score"]),
        "exact": bool(best["exact"]),
        "margin": margin,
        "ambiguous": ambiguous,
    }


def align_to_source(
    content: str,
    units: List[str],
    unit_tokens: List[List[str]],
    min_start: int = 0,
    threshold: float = 0.30,
    max_window_ratio: float = 2.5,
    min_window_tokens: int = 24,
    anchor_phrases: Optional[Sequence[str]] = None,
    boundary_cues: Optional[Sequence[str]] = None,
) -> Optional[Tuple[int, int, float]]:
    """Find the best contiguous window of units matching content.

    Returns (start_unit, end_unit, score) or None if below threshold.
    """
    details = align_to_source_details(
        content,
        units,
        unit_tokens,
        min_start=min_start,
        threshold=threshold,
        max_window_ratio=max_window_ratio,
        min_window_tokens=min_window_tokens,
        anchor_phrases=anchor_phrases,
        boundary_cues=boundary_cues,
    )
    if details is None:
        return None
    return (int(details["start"]), int(details["end"]), float(details["score"]))
