"""Object-level unlock after associated prefill completion."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KVDuoRecoveryObject:
    key: tuple[int, int, int]
    order: int
    ready: bool


def plan_kvduo_recovery(objects):
    """Unlock ready objects in stable FIFO order without allocating pages."""
    return tuple(
        obj.key
        for obj in sorted(
            objects, key=lambda o: (o.order, o.key[1], o.key[0], o.key[2])
        )
        if obj.ready
    )
