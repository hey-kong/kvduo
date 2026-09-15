"""Control-plane state primitives for KVDuo mixed KV residency.

This module deliberately has no torch dependency.  It is the authoritative
metadata vocabulary used by later KVDuo lifecycle work; CUDA-visible mirrors
remain an implementation detail of the coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Hashable, Iterable, Mapping, Optional, Tuple

PageIdentity = Hashable
DomainIdentity = Hashable
OwnerIdentity = Hashable


class KVDuoPageResidency(Enum):
    """Coarse residency of a cross-domain logical page."""

    GPU_FULL = auto()
    HOST_ONLY = auto()
    MIXED = auto()
    ABSENT = auto()


class KVDuoPressureStatus(Enum):
    """Outcome of a physical-pool allocation plan."""

    READY = auto()
    RECLAIM_REQUIRED = auto()
    WAIT_FOR_IO = auto()
    WAIT_FOR_MEMORY = auto()
    RETRACT_REQUIRED = auto()


@dataclass
class KVDuoPagePinState:
    """References that make a full or hot physical page non-reclaimable."""

    attention: int = 0
    dma: int = 0
    prefill: int = 0

    @property
    def pinned(self) -> bool:
        return self.attention > 0 or self.dma > 0 or self.prefill > 0

    def validate(self) -> None:
        if min(self.attention, self.dma, self.prefill) < 0:
            raise ValueError("KVDuo page pin counts cannot be negative")


@dataclass
class KVDuoFullPageState:
    """Ownership and validity of one cross-domain logical full page.

    ``domains`` contains physical storage identities, not necessarily model
    layer ids.  Layers sharing KV therefore name the same domain and are
    accounted for once.  A host copy is valid only when its version equals the
    corresponding data version.
    """

    page_id: PageIdentity
    domains: Tuple[DomainIdentity, ...]
    data_versions: Dict[DomainIdentity, int] = field(default_factory=dict)
    gpu_full_domains: set[DomainIdentity] = field(default_factory=set)
    host_versions: Dict[DomainIdentity, int] = field(default_factory=dict)
    request_owners: set[OwnerIdentity] = field(default_factory=set)
    prefix_owners: set[OwnerIdentity] = field(default_factory=set)
    sealed: bool = False
    stable: bool = False
    tail_protected: bool = False
    last_touch: int = 0
    pins: KVDuoPagePinState = field(default_factory=KVDuoPagePinState)

    def __post_init__(self) -> None:
        # Preserve declaration order while deduplicating shared storage.
        self.domains = tuple(dict.fromkeys(self.domains))
        for domain in self.domains:
            self.data_versions.setdefault(domain, 0)
        self.validate()

    @property
    def gpu_full(self) -> bool:
        return bool(self.domains) and self.gpu_full_domains == set(self.domains)

    @property
    def host_full_valid(self) -> bool:
        return bool(self.domains) and all(
            self.host_versions.get(domain) == self.data_versions[domain]
            for domain in self.domains
        )

    @property
    def residency(self) -> KVDuoPageResidency:
        if self.gpu_full:
            return KVDuoPageResidency.GPU_FULL
        if self.host_full_valid and not self.gpu_full_domains:
            return KVDuoPageResidency.HOST_ONLY
        if self.gpu_full_domains or self.host_versions:
            return KVDuoPageResidency.MIXED
        return KVDuoPageResidency.ABSENT

    @property
    def reclaimable(self) -> bool:
        return (
            self.gpu_full
            and self.host_full_valid
            and self.sealed
            and self.stable
            and not self.tail_protected
            and not self.pins.pinned
        )

    def record_model_write(self, domain: DomainIdentity, clock: int) -> None:
        """Record a real model write and invalidate an older host copy."""
        self._require_domain(domain)
        if clock < self.last_touch:
            raise ValueError("KVDuo logical clocks must be monotonic")
        self.data_versions[domain] += 1
        self.host_versions.pop(domain, None)
        self.gpu_full_domains.add(domain)
        self.last_touch = clock
        self.stable = False

    def record_attention_touch(self, clock: int) -> None:
        if clock < self.last_touch:
            raise ValueError("KVDuo logical clocks must be monotonic")
        self.last_touch = clock

    def mark_host_copy_complete(self, domain: DomainIdentity) -> None:
        """Validate a host copy without fabricating a model touch event."""
        self._require_domain(domain)
        self.host_versions[domain] = self.data_versions[domain]

    def attach_request(self, owner: OwnerIdentity) -> None:
        self.request_owners.add(owner)

    def detach_request(self, owner: OwnerIdentity) -> None:
        self.request_owners.discard(owner)

    def validate(self) -> None:
        if not self.domains:
            raise ValueError("A KVDuo full page must contain at least one KV domain")
        known = set(self.domains)
        if set(self.data_versions) != known:
            raise ValueError("Every KVDuo domain must have exactly one data version")
        if not self.gpu_full_domains <= known or not set(self.host_versions) <= known:
            raise ValueError("KVDuo page metadata contains an unknown KV domain")
        if any(version < 0 for version in self.data_versions.values()):
            raise ValueError("KVDuo data versions cannot be negative")
        self.pins.validate()

    def _require_domain(self, domain: DomainIdentity) -> None:
        if domain not in self.data_versions:
            raise KeyError(f"Unknown KVDuo KV domain: {domain!r}")


@dataclass
class KVDuoHotCapacity:
    """Per-request, per-domain hot-cache capacity in physical bytes."""

    minimum_bytes: int
    current_bytes: int
    maximum_bytes: int

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_bytes <= self.current_bytes <= self.maximum_bytes:
            raise ValueError("Expected 0 <= minimum <= current <= maximum hot capacity")

    @property
    def shrinkable_bytes(self) -> int:
        return self.current_bytes - self.minimum_bytes


@dataclass
class KVDuoRequestResidency:
    """Request view over ordered full pages and private hot capacities."""

    request_id: OwnerIdentity
    full_pages: list[PageIdentity] = field(default_factory=list)
    hot_capacity: Dict[DomainIdentity, KVDuoHotCapacity] = field(default_factory=dict)

    def longest_contiguous_gpu_prefix(
        self, pages: Mapping[PageIdentity, KVDuoFullPageState]
    ) -> int:
        length = 0
        for page_id in self.full_pages:
            page = pages.get(page_id)
            if page is None or not page.gpu_full:
                break
            length += 1
        return length


@dataclass(frozen=True)
class KVDuoAllocationPlan:
    """Byte-accurate request against one specific physical KV pool."""

    available_bytes: int
    new_full_bytes: int = 0
    new_hot_bytes: int = 0
    reserved_bytes: int = 0
    turnover_bytes: int = 0

    def __post_init__(self) -> None:
        if (
            min(
                self.available_bytes,
                self.new_full_bytes,
                self.new_hot_bytes,
                self.reserved_bytes,
                self.turnover_bytes,
            )
            < 0
        ):
            raise ValueError("KVDuo allocation-plan byte counts cannot be negative")

    @property
    def required_bytes(self) -> int:
        return (
            self.new_full_bytes
            + self.new_hot_bytes
            + self.reserved_bytes
            + self.turnover_bytes
        )

    @property
    def shortfall_bytes(self) -> int:
        return max(0, self.required_bytes - self.available_bytes)

    @property
    def status(self) -> KVDuoPressureStatus:
        if self.shortfall_bytes:
            return KVDuoPressureStatus.RECLAIM_REQUIRED
        return KVDuoPressureStatus.READY


@dataclass(frozen=True)
class KVDuoPressureResult:
    """Structured pressure outcome used instead of an ambiguous bool."""

    status: KVDuoPressureStatus
    requested_bytes: int
    reclaimed_bytes: int = 0
    remaining_shortfall_bytes: int = 0
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        if (
            min(
                self.requested_bytes,
                self.reclaimed_bytes,
                self.remaining_shortfall_bytes,
            )
            < 0
        ):
            raise ValueError("KVDuo pressure-result byte counts cannot be negative")
        if self.status is KVDuoPressureStatus.READY and self.remaining_shortfall_bytes:
            raise ValueError("A READY KVDuo pressure result cannot have a shortfall")


class KVDuoResidencyCatalog:
    """Own full-page metadata and keep request references balanced."""

    def __init__(self) -> None:
        self.pages: Dict[PageIdentity, KVDuoFullPageState] = {}
        self.requests: Dict[OwnerIdentity, KVDuoRequestResidency] = {}

    def add_page(self, page: KVDuoFullPageState) -> None:
        if page.page_id in self.pages:
            raise ValueError(f"Duplicate KVDuo page identity: {page.page_id!r}")
        self.pages[page.page_id] = page

    def attach_request(
        self, request_id: OwnerIdentity, page_ids: Iterable[PageIdentity]
    ) -> KVDuoRequestResidency:
        if request_id in self.requests:
            raise ValueError(f"Duplicate KVDuo request identity: {request_id!r}")
        page_ids = list(page_ids)
        missing = [page_id for page_id in page_ids if page_id not in self.pages]
        if missing:
            raise KeyError(f"Unknown KVDuo page identities: {missing!r}")
        request = KVDuoRequestResidency(request_id, page_ids)
        self.requests[request_id] = request
        for page_id in page_ids:
            self.pages[page_id].attach_request(request_id)
        return request

    def detach_request(self, request_id: OwnerIdentity) -> None:
        request = self.requests.pop(request_id)
        for page_id in request.full_pages:
            self.pages[page_id].detach_request(request_id)
