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
) -> tuple[int, ...] | None:
    """Select whole requests by most recent Hand merge, with a fixed round budget.

    All established HOT layers must have safe bases. Only expanded layers add
    observations, miss budget, or rollback cost; a base-only layer can be young.
    """
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
    requests = {}
    for obj in objects:
        requests.setdefault(obj.owner[1], []).append(obj)
    candidates = []
    for req, layers in requests.items():
        if any(
            not obj.eligible or not 0 < obj.base_pages <= obj.current_pages <= 8
            for obj in layers
        ):
            continue
        expanded = [obj for obj in layers if obj.current_pages > obj.base_pages]
        if not expanded or any(
            not obj.mature or obj.owner not in reference_misses for obj in expanded
        ):
            continue
        extra = sum(obj.extra_hits for obj in expanded)
        misses = sum(obj.misses for obj in expanded)
        sequence = max(obj.processed for obj in layers)
        if sequence <= 0 or extra > rho * misses:
            continue
        freed = sum(obj.current_pages - obj.base_pages for obj in expanded)
        candidates.append((sequence, req, freed, extra))
    selected = []
    freed_pages = 0
    cost = reserved_cost
    for _, req, freed, extra in sorted(candidates, key=lambda item: -item[0]):
        # A request that exceeds the remaining cumulative budget is not
        # eligible for this plan. Continue checking older requests safely.
        if cost + extra > budget:
            continue
        selected.append(req)
        freed_pages += freed
        cost += extra
        if freed_pages >= required_pages:
            return tuple(selected)
    return None
