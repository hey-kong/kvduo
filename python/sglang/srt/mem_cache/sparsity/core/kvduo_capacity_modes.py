"""Object-level elasticity recovery with a protected prefill space plan."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KVDuoRecoveryObject:
    key: tuple[int, int, int]
    order: int
    ready: bool


def plan_kvduo_recovery(objects, available_pages, prefill_pages):
    """FIFO recovery of ready objects, reserving one page for each."""
    remaining = available_pages - prefill_pages
    if remaining < 0:
        return ()
    selected = []
    for obj in sorted(objects, key=lambda o: (o.order, o.key[1], o.key[0], o.key[2])):
        if not obj.ready:
            continue
        if remaining < 1:
            break
        selected.append(obj.key)
        remaining -= 1
    return tuple(selected)
