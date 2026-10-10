"""Plan budgeted HOT capacity reduction and MRU-preserving compaction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class KVDuoHotMergePlan:
    kept_pages: tuple[int, ...]
    released_pages: tuple[int, ...]
    source_slots: tuple[int, ...]
    destination_slots: tuple[int, ...]
    tokens: tuple[int, ...]
    lru_slots: tuple[int, ...]


@dataclass(frozen=True)
class KVDuoHotMergeTensors:
    kept_pages: tuple[int, ...]
    released_pages: tuple[int, ...]
    tokens: torch.Tensor
    locations: torch.Tensor
    lru_slots: torch.Tensor
    source_locations: torch.Tensor
    destination_locations: torch.Tensor


def prepare_kvduo_hot_merge(
    tokens: torch.Tensor,
    locations: torch.Tensor,
    lru_slots: torch.Tensor,
    capacity: int,
    k: int,
    top_k: int,
    target_pages: int,
) -> KVDuoHotMergeTensors:
    import torch

    if k <= 0 or top_k <= 0 or capacity <= 0 or capacity % k:
        raise ValueError("HOT merge requires positive, page-aligned geometry")
    pages = capacity // k
    if not (top_k + k - 1) // k <= target_pages <= pages <= 8:
        raise ValueError("HOT merge target is outside the minimum/current capacity")
    if (
        tokens.ndim != 1
        or locations.shape != tokens.shape
        or lru_slots.ndim != 1
        or tokens.numel() < lru_slots.numel()
        or lru_slots.numel() < capacity
    ):
        raise ValueError("HOT merge metadata has incompatible capacities")
    if tokens.is_cuda:
        from sglang.kernels.ops.kvcache.hisparse import build_kvduo_hot_merge_plan

        summary, new_tokens, new_locs, new_lru, sources, destinations = (
            build_kvduo_hot_merge_plan(
                tokens, locations, lru_slots, capacity, k, target_pages
            )
        )
        status, moved, mask = summary.cpu().tolist()
        if status & 1:
            raise ValueError("HOT merge requires the current LRU slot permutation")
        if status & 2:
            raise RuntimeError("KVDuo HOT merge has invalid physical slot locations")
        kept_pages = tuple(page for page in range(pages) if mask & (1 << page))
        if (
            status
            or mask >> pages
            or len(kept_pages) != target_pages
            or not 0 <= moved <= target_pages * k
        ):
            raise RuntimeError("KVDuo GPU merge planning returned invalid results")
        released_pages = tuple(page for page in range(pages) if page not in kept_pages)
        sources, destinations = sources[:moved], destinations[:moved]
    else:
        plan = plan_kvduo_hot_merge(
            tokens[:capacity].tolist(),
            lru_slots[:capacity].tolist(),
            k,
            top_k,
            target_pages,
        )
        locs = locations[:capacity].tolist()
        if any(loc < 0 for loc in locs) or len(set(locs)) != capacity:
            raise RuntimeError("KVDuo HOT merge has invalid physical slot locations")
        kept_pages, released_pages = plan.kept_pages, plan.released_pages
        new_tokens = torch.full_like(tokens, -1)
        new_locs = torch.full_like(locations, -1)
        new_lru = torch.full_like(lru_slots, -1)
        target = target_pages * k
        new_tokens[:target] = torch.tensor(plan.tokens, dtype=tokens.dtype)
        new_locs[:target] = torch.tensor(
            [locs[page * k + offset] for page in kept_pages for offset in range(k)],
            dtype=locations.dtype,
        )
        new_lru[:target] = torch.tensor(plan.lru_slots, dtype=lru_slots.dtype)
        sources = torch.tensor(
            [locs[slot] for slot in plan.source_slots], dtype=torch.int64
        )
        destinations = torch.tensor(
            [locs[slot] for slot in plan.destination_slots], dtype=torch.int64
        )
    return KVDuoHotMergeTensors(
        kept_pages, released_pages, new_tokens, new_locs, new_lru, sources, destinations
    )


def kvduo_shrink_target(segment_hits, misses, steps, current_pages, minimum_pages, rho):
    """Smallest mature capacity whose discarded LRU contribution fits the budget."""
    if not 1 <= minimum_pages <= current_pages <= 8:
        raise ValueError("Invalid HOT capacity geometry")
    if steps <= 64:
        return current_pages
    values = list(segment_hits[:current_pages])
    if len(values) != current_pages or any(
        not math.isfinite(v) or v < 0 for v in (*values, misses, rho)
    ):
        return current_pages
    for m in range(minimum_pages, current_pages):
        if sum(values[m:]) <= rho * misses:
            return m
    return current_pages


def plan_kvduo_hot_merge(
    tokens: Sequence[int],
    lru_slots: Sequence[int],
    k: int,
    top_k: int,
    target_pages: int,
) -> KVDuoHotMergePlan:
    """Keep the MRU valid entries in relative LRU order, minimizing payload moves.

    Slot numbers in the copy plan refer to the original capacity. The new LRU
    refers to compacted slots; moving a survivor never changes its access rank.
    This plan is transient and does not model future cache accesses.
    """
    capacity = len(tokens)
    if k <= 0 or top_k <= 0 or capacity % k:
        raise ValueError("HOT merge requires positive, page-aligned geometry")
    n = capacity // k
    minimum = (top_k + k - 1) // k
    if not minimum <= n <= 8:
        raise ValueError("HOT merge capacity is outside its minimum/8-page limits")
    if sorted(lru_slots) != list(range(capacity)):
        raise ValueError("HOT merge requires the current LRU slot permutation")
    m = target_pages
    if not minimum <= m <= n:
        raise ValueError("HOT merge target is outside the minimum/current capacity")
    survivors = [slot for slot in lru_slots if tokens[slot] >= 0][-m * k :]
    survivor_set = set(survivors)
    counts = [0] * n
    for slot in survivors:
        counts[slot // k] += 1
    kept_pages = tuple(sorted(sorted(range(n), key=lambda p: (-counts[p], p))[:m]))
    kept_set = set(kept_pages)
    released_pages = tuple(p for p in range(n) if p not in kept_set)
    retained_slots = [p * k + offset for p in kept_pages for offset in range(k)]
    old_to_new = {slot: new for new, slot in enumerate(retained_slots)}
    available = [slot for slot in retained_slots if slot not in survivor_set]
    sources = [slot for slot in survivors if slot // k not in kept_set]
    destinations = available[: len(sources)]

    new_tokens = [-1] * (m * k)
    survivor_locations = {}
    for slot in survivors:
        if slot in old_to_new:
            new = old_to_new[slot]
            new_tokens[new] = tokens[slot]
            survivor_locations[slot] = new
    for source, destination in zip(sources, destinations):
        new = old_to_new[destination]
        new_tokens[new] = tokens[source]
        survivor_locations[source] = new
    new_lru = [slot for slot, token in enumerate(new_tokens) if token < 0]
    new_lru.extend(survivor_locations[slot] for slot in survivors)
    return KVDuoHotMergePlan(
        kept_pages,
        released_pages,
        tuple(sources),
        tuple(destinations),
        tuple(new_tokens),
        tuple(new_lru),
    )
