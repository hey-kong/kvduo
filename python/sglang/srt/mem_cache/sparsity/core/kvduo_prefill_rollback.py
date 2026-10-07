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


@dataclass(frozen=True)
class KVDuoPrefillObject:
    owner: tuple[int, int]
    base_pages: int
    current_pages: int
    counts: tuple[float, float, float, float]  # accesses, direct, base, extra
    steps: int
    reference: float
    reference_steps: int
    eligible: bool = True

    @property
    def misses(self) -> float:
        d, direct, base, extra = self.counts
        return max(0.0, d - direct - base - extra)

    @property
    def valid_counts(self) -> bool:
        d, direct, base, extra = self.counts
        return (
            all(math.isfinite(v) and v >= 0 for v in self.counts)
            and d >= 0
            and direct + base + extra <= d + 1e-4 * max(d, 1)
        )


def plan_kvduo_prefill_rollback(
    objects: Sequence[KVDuoPrefillObject],
    required_pages: int,
    policy: KVDuoPrefillPolicy,
    min_observations: float = 64,
    reference_misses: Mapping[tuple[int, int], float] | None = None,
    reserved_cost: float = 0,
) -> tuple[tuple[int, int], ...] | None:
    """Evaluate a complete stable-score plan against one fixed round budget.

    Without a round, actual counts for every active decode object establish M0.
    Later arrivals cannot enlarge it; committed costs never decay with EMA.
    """
    if required_pages <= 0:
        return ()
    if reference_misses is None:
        if not objects or any(not obj.valid_counts for obj in objects):
            return None
        reference_misses = {obj.owner: obj.misses for obj in objects}
    rho = policy.max_relative_miss_increase
    budget = rho * sum(reference_misses.values())
    candidates = []
    for obj in objects:
        freed = obj.current_pages - obj.base_pages
        if (
            not obj.eligible
            or obj.owner not in reference_misses
            or obj.base_pages <= 0
            or freed <= 0
            or obj.steps <= min_observations
            or obj.reference_steps <= min_observations
            or not obj.valid_counts
            or obj.counts[0] <= 0
            or not math.isfinite(obj.reference)
            or not 0 <= obj.reference <= 1
        ):
            continue
        d, direct, base, extra = obj.counts
        if extra <= rho * obj.misses and (1 - (direct + base) / d) <= (1 + rho) * (
            1 - obj.reference
        ):
            candidates.append((extra / freed, obj.owner, freed, extra))
    selected = []
    freed_pages = 0
    cost = reserved_cost
    for _, owner, freed, extra in sorted(candidates, key=lambda item: item[0]):
        selected.append(owner)
        freed_pages += freed
        cost += extra
        if freed_pages >= required_pages:
            return tuple(selected) if cost <= budget else None
    return None
