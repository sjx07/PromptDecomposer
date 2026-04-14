"""Batch-processing helpers for decomposition jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence


MODEL_PRICING_PER_MILLION: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
}
DECOMPOSE_CACHE_VERSION = "decompose-cache-v8"


def normalize_model_name(model: str) -> str:
    """Normalize provider/model strings into a pricing lookup key."""
    model = model.strip().lower()
    if "/" in model:
        model = model.rsplit("/", 1)[-1]
    return model


def pricing_for_model(model: str) -> Optional[Dict[str, float]]:
    """Return pricing per 1M tokens for a supported model."""
    key = normalize_model_name(model)
    pricing = MODEL_PRICING_PER_MILLION.get(key)
    if pricing is None:
        return None
    return dict(pricing)


def chunked(items: Sequence, size: int) -> Iterator[Sequence]:
    """Yield fixed-size batches from a sequence."""
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def usage_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    """Compute usage delta between two decomposer snapshots."""
    keys = {"prompt_tokens", "completion_tokens", "total_tokens", "requests", "estimated_requests"}
    return {
        key: int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
        for key in keys
    }


def merge_usage(total: Dict[str, int], delta: Dict[str, int]) -> Dict[str, int]:
    """Add a usage delta into a cumulative usage dict."""
    merged = dict(total)
    for key, value in delta.items():
        merged[key] = int(merged.get(key, 0) or 0) + int(value or 0)
    return merged


def usage_cost(usage: Dict[str, int], pricing: Optional[Dict[str, float]]) -> Optional[float]:
    """Compute dollar cost for usage totals using per-1M-token pricing."""
    if pricing is None:
        return None
    prompt_cost = usage.get("prompt_tokens", 0) * pricing["input"] / 1_000_000
    completion_cost = usage.get("completion_tokens", 0) * pricing["output"] / 1_000_000
    return prompt_cost + completion_cost


def zero_usage() -> Dict[str, int]:
    """Return a zero-valued usage payload."""
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "requests": 0,
        "estimated_requests": 0,
    }


def decompose_cache_key(
    prompt: str,
    *,
    model: str,
    provider: str,
    mode: str,
    atomize: bool,
    temperature: float,
    max_depth: int = 4,
    min_span_chars: int = 80,
) -> str:
    """Build a deterministic cache key for one decomposition request."""
    payload = {
        "version": DECOMPOSE_CACHE_VERSION,
        "prompt": prompt,
        "model": model,
        "provider": provider,
        "mode": mode,
        "atomize": bool(atomize),
        "temperature": float(temperature),
        "max_depth": int(max_depth),
        "min_span_chars": int(min_span_chars),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_path(cache_dir: Path, key: str) -> Path:
    """Return the on-disk path for a cache entry."""
    return cache_dir / key[:2] / f"{key}.json"


def load_cached_decomposition(cache_dir: Path, key: str) -> Optional[Dict[str, Any]]:
    """Load a cached decomposition payload if present."""
    path = cache_path(cache_dir, key)
    if not path.exists():
        return None
    with path.open() as f:
        record = json.load(f)
    decomposition = record.get("decomposition")
    if not isinstance(decomposition, dict):
        return None
    return decomposition


def store_cached_decomposition(cache_dir: Path, key: str, decomposition: Dict[str, Any]) -> Path:
    """Persist a decomposition payload to the local cache."""
    path = cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(
            {
                "cache_version": DECOMPOSE_CACHE_VERSION,
                "key": key,
                "decomposition": decomposition,
            },
            f,
        )
    return path
