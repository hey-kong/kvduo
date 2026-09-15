"""Host-prefix ownership and residency planning for KVDuo."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Hashable, Iterable, Mapping, Tuple


class KVDuoPrefixResidency(Enum):
    FULL_GPU_HIT = auto()
    PARTIAL_GPU_HIT = auto()
    HOST_ONLY_HIT = auto()
    RESTORE_REQUIRED = auto()
    INVALID_VERSION = auto()


@dataclass
class HostPrefixRecord:
    """Host-owned page record independent of a recyclable request slot."""

    identity: Hashable
    domains: Tuple[Hashable, ...]
    host_locations: Dict[Hashable, Tuple[int, ...]]
    data_versions: Dict[Hashable, Hashable]
    host_versions: Dict[Hashable, Hashable]
    # Optional model-event timestamps.  Copies and prefix attachment preserve
    # these values instead of fabricating a new write/attention touch.
    model_touches: Dict[Hashable, Hashable] = field(default_factory=dict)
    request_references: set[Hashable] = field(default_factory=set)
    cache_reference: bool = False
    restore_pins: int = 0
    dma_pins: int = 0
    last_access: int = 0
    tie_break_key: Tuple = ()

    def __post_init__(self) -> None:
        self.domains = tuple(dict.fromkeys(self.domains))
        known = set(self.domains)
        if (
            set(self.host_locations) != known
            or set(self.data_versions) != known
            or not set(self.host_versions) <= known
            or not set(self.model_touches) <= known
        ):
            raise ValueError(
                "Host prefix domain metadata is incomplete or inconsistent"
            )

    def domain_valid(self, domain: Hashable) -> bool:
        return self.host_versions.get(domain) == self.data_versions[domain]

    @property
    def fully_valid(self) -> bool:
        return all(self.domain_valid(domain) for domain in self.domains)

    @property
    def pinned(self) -> bool:
        return (
            bool(self.request_references) or self.restore_pins > 0 or self.dma_pins > 0
        )

    @property
    def physical_slots(self) -> int:
        # Domain identity has already deduplicated shared-layer storage.
        return sum(len(self.host_locations[domain]) for domain in self.domains)


class KVDuoHostPrefixCache:
    """Capacity-bounded host records with request/cache ownership and LRU."""

    def __init__(self, capacity_slots: int):
        if capacity_slots < 0:
            raise ValueError("Host prefix capacity cannot be negative")
        self.capacity_slots = capacity_slots
        self.records: Dict[Hashable, HostPrefixRecord] = {}

    @property
    def used_slots(self) -> int:
        return sum(record.physical_slots for record in self.records.values())

    def insert(self, record: HostPrefixRecord) -> None:
        if record.identity in self.records:
            raise ValueError(f"Duplicate host prefix identity: {record.identity!r}")
        if self.used_slots + record.physical_slots > self.capacity_slots:
            raise MemoryError("Host prefix record exceeds the admitted cache capacity")
        self.records[record.identity] = record

    def acquire(
        self, identity: Hashable, request_id: Hashable, clock: int
    ) -> HostPrefixRecord:
        record = self.records[identity]
        record.request_references.add(request_id)
        record.last_access = max(record.last_access, clock)
        return record

    def release(self, identity: Hashable, request_id: Hashable) -> None:
        self.records[identity].request_references.discard(request_id)

    def retain_for_cache(self, identity: Hashable) -> None:
        self.records[identity].cache_reference = True

    def evict_lru(self, required_slots: int) -> Tuple[HostPrefixRecord, ...]:
        """Revoke cache ownership and return unpinned victims to free physically."""
        if required_slots <= 0:
            return ()
        candidates = sorted(
            (record for record in self.records.values() if not record.pinned),
            key=lambda record: (
                record.last_access,
                record.tie_break_key,
                repr(record.identity),
            ),
        )
        victims = []
        released = 0
        for record in candidates:
            record.cache_reference = False
            victims.append(record)
            released += record.physical_slots
            del self.records[record.identity]
            if released >= required_slots:
                break
        return tuple(victims)


@dataclass(frozen=True)
class KVDuoPrefixPageView:
    identity: Hashable
    domains: Tuple[Hashable, ...]
    gpu_full_domains: frozenset[Hashable] = frozenset()


@dataclass(frozen=True)
class KVDuoPrefixRestorePlan:
    match_status: KVDuoPrefixResidency
    execution_status: KVDuoPrefixResidency
    reusable_gpu: Tuple[Tuple[Hashable, Hashable], ...]
    restore_from_host: Tuple[Tuple[Hashable, Hashable], ...]
    missing_bytes: int


@dataclass
class KVDuoPrefixRestoreCommitment:
    """Pins host sources between prefill admission and H2D publication."""

    identities: Tuple[Hashable, ...]
    active: bool = True


def begin_kvduo_prefix_restore(
    plan: KVDuoPrefixRestorePlan, host_cache: KVDuoHostPrefixCache
) -> KVDuoPrefixRestoreCommitment:
    if plan.execution_status is not KVDuoPrefixResidency.RESTORE_REQUIRED:
        raise ValueError("KVDuo restore commitment requires a restore plan")
    identities = tuple(
        dict.fromkeys(identity for identity, _ in plan.restore_from_host)
    )
    for identity in identities:
        record = host_cache.records.get(identity)
        if record is None or not record.fully_valid:
            raise ValueError("KVDuo restore source changed before commitment")
    for identity in identities:
        host_cache.records[identity].restore_pins += 1
    return KVDuoPrefixRestoreCommitment(identities)


def finish_kvduo_prefix_restore(
    commitment: KVDuoPrefixRestoreCommitment,
    host_cache: KVDuoHostPrefixCache,
) -> None:
    if not commitment.active:
        raise ValueError("KVDuo restore commitment already finished")
    for identity in commitment.identities:
        record = host_cache.records.get(identity)
        if record is None or record.restore_pins <= 0:
            raise ValueError("KVDuo restore ownership was lost while pinned")
        record.restore_pins -= 1
    commitment.active = False


def plan_kvduo_prefix_restore(
    pages: Iterable[KVDuoPrefixPageView],
    host_cache: KVDuoHostPrefixCache,
    domain_bytes: Mapping[Hashable, int],
) -> KVDuoPrefixRestorePlan:
    """Classify a logical Radix match and build its deduplicated restore plan."""
    reusable = []
    restore = []
    any_gpu = False
    any_host_only = False
    invalid = False
    for page in pages:
        record = host_cache.records.get(page.identity)
        for domain in tuple(dict.fromkeys(page.domains)):
            if domain in page.gpu_full_domains:
                reusable.append((page.identity, domain))
                any_gpu = True
            elif (
                record is not None
                and domain in record.data_versions
                and record.domain_valid(domain)
            ):
                restore.append((page.identity, domain))
                any_host_only = True
            else:
                invalid = True

    if invalid:
        status = KVDuoPrefixResidency.INVALID_VERSION
    elif restore and any_gpu:
        status = KVDuoPrefixResidency.PARTIAL_GPU_HIT
    elif restore and any_host_only:
        status = KVDuoPrefixResidency.HOST_ONLY_HIT
    else:
        status = KVDuoPrefixResidency.FULL_GPU_HIT
    execution = (
        KVDuoPrefixResidency.RESTORE_REQUIRED if restore and not invalid else status
    )
    return KVDuoPrefixRestorePlan(
        status,
        execution,
        tuple(reusable),
        tuple(restore),
        sum(domain_bytes[domain] for _, domain in restore),
    )
