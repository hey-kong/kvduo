"""A pressure-boundary plan for compacting the real HOT cache's LRU survivors."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class KVDuoHotMergePlan:
    kept_pages: tuple[int, ...]
    released_pages: tuple[int, ...]
    source_slots: tuple[int, ...]
    destination_slots: tuple[int, ...]
    tokens: tuple[int, ...]
    lru_slots: tuple[int, ...]


def plan_kvduo_hot_merge(
    tokens: Sequence[int], lru_slots: Sequence[int], k: int, top_k: int
) -> KVDuoHotMergePlan | None:
    """Keep the MRU valid entries, then minimize moves by choosing backing pages.

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
    m = max(minimum, (n + 1) // 2)
    if m == n:
        return None

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
