"""Plan no-copy prefill reclamation using a fixed relative-miss budget."""

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class KVDuoPrefillPolicy:
    max_relative_miss_increase: float = 0.10

    def __post_init__(self):
        value = self.max_relative_miss_increase
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                "max_relative_miss_increase must be a finite nonnegative number"
            )


DEFAULT_KVDUO_PREFILL_MIN_OBSERVATIONS = 64


@dataclass(frozen=True)
class KVDuoPrefillObject:
    owner: tuple[int, int]  # (logical layer, request slot)
    base_pages: int
    current_pages: int
    counts: tuple[float, float]  # extra HOT hits, host misses
    steps: int
    processed: int
    eligible: bool = True
    base_valid: bool = True

    @property
    def extra_hits(self) -> float:
        return self.counts[0]

    @property
    def misses(self) -> float:
        return self.counts[1]

    @property
    def valid_counts(self) -> bool:
        return all(math.isfinite(v) and v >= 0 for v in self.counts)

    @property
    def mature(self) -> bool:
        return (
            self.base_valid
            and 0 < self.base_pages < self.current_pages <= 8
            and self.steps > DEFAULT_KVDUO_PREFILL_MIN_OBSERVATIONS
            and self.valid_counts
        )


def plan_kvduo_prefill_rollback(
    objects: Sequence[KVDuoPrefillObject],
    required_pages: int,
    policy: KVDuoPrefillPolicy,
    reference_misses: Mapping[tuple[int, int], float] | None = None,
    reserved_cost: float = 0,
) -> tuple[tuple[int, int], ...] | None:
    """Select HOT objects by most recent Hand processing and a fixed round budget."""
    if required_pages <= 0:
        return ()
    if reference_misses is None:
        reference_misses = {obj.owner: obj.misses for obj in objects if obj.mature}
    if (
        not math.isfinite(reserved_cost)
        or reserved_cost < 0
        or any(not math.isfinite(v) or v < 0 for v in reference_misses.values())
    ):
        return None
    rho = policy.max_relative_miss_increase
    budget = rho * sum(reference_misses.values())
    candidates = sorted(
        objects,
        key=lambda obj: (-obj.processed, obj.owner[1], obj.owner[0]),
    )
    selected = []
    seen = set()
    freed_pages = 0
    cost = reserved_cost
    for obj in candidates:
        if obj.owner in seen:
            continue
        seen.add(obj.owner)
        if (
            not obj.eligible
            or not obj.mature
            or obj.processed <= 0
            or obj.owner not in reference_misses
            or obj.extra_hits > rho * obj.misses
            or cost + obj.extra_hits > budget
        ):
            continue
        selected.append(obj.owner)
        freed_pages += obj.current_pages - obj.base_pages
        cost += obj.extra_hits
        if freed_pages >= required_pages:
            return tuple(selected)
    return None
