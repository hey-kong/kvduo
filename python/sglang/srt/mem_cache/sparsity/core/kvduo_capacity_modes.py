"""Request-atomic elasticity recovery with a protected prefill space plan."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KVDuoRecoveryRequest:
    key: tuple[int, int]
    order: int
    quantum: int
    ready: bool


def plan_kvduo_recovery(requests, available_pages, prefill_pages):
    """FIFO recovery; skip protected requests, stop at a safe request that cannot fit."""
    remaining = available_pages - prefill_pages
    if remaining < 0:
        return ()
    selected = []
    for request in sorted(requests, key=lambda r: (r.order, r.key)):
        if not request.ready:
            continue
        if request.quantum > remaining:
            break
        selected.append(request.key)
        remaining -= request.quantum
    return tuple(selected)
