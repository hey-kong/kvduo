"""Control-plane state primitives for KVDuo mixed KV residency.

This module deliberately has no torch dependency.  It is the authoritative
metadata vocabulary used by later KVDuo lifecycle work; CUDA-visible mirrors
remain an implementation detail of the coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, Hashable, Iterable, Mapping, Optional, Tuple

PageIdentity = Hashable
DomainIdentity = Hashable
OwnerIdentity = Hashable


class KVDuoPhysicalPageKind(Enum):
    """Exclusive state of one cross-storage-layer physical page number."""

    FREE = auto()
    FULL = auto()
    HOT = auto()


@dataclass(frozen=True)
class KVDuoHotPageGeometry:
    """Runtime-derived addressing geometry for a whole KVDuo HOT page."""

    storage_layers: int
    compressed_page_size: int
    layer_slot_stride: int

    def __post_init__(self) -> None:
        if (
            min(self.storage_layers, self.compressed_page_size, self.layer_slot_stride)
            <= 0
        ):
            raise ValueError("KVDuo hot-page geometry dimensions must be positive")

    @property
    def slots_per_page(self) -> int:
        return self.storage_layers * self.compressed_page_size

    def encode(self, page_start: int, logical_slot: int) -> int:
        """Encode HOT(page, slot) into the flat layer/page/offset backing."""
        if page_start < 0 or not 0 <= logical_slot < self.slots_per_page:
            raise IndexError("KVDuo hot-page address is outside the physical page")
        storage_layer, offset = divmod(logical_slot, self.compressed_page_size)
        return storage_layer * self.layer_slot_stride + page_start + offset


@dataclass(frozen=True)
class KVDuoWritebackTicket:
    """Version captured when an asynchronous host writeback is submitted."""

    generation: int
    data_version: int


@dataclass
class KVDuoEntryVersionState:
    """Reference write/touch/version lifecycle for one stored KV identity.

    A single aggregate ``last_touch`` is sufficient for page-LRU because max is
    associative: ``max(max(write, attention) for entries)`` equals the maximum
    of this field over the page. It avoids duplicating timestamps per layer when
    layers share the same physical KV storage identity.
    """

    generation: int = 0
    data_version: int = 0
    host_version: int = -1
    last_touch: int = 0

    @property
    def host_valid(self) -> bool:
        return self.generation > 0 and self.host_version == self.data_version

    def initialize_generation(self, generation: int, clock: int) -> None:
        if generation <= self.generation or clock < 0:
            raise ValueError("KVDuo generation must advance with a valid clock")
        self.generation = generation
        self.data_version = 1
        self.host_version = -1
        self.last_touch = clock

    def model_update(self, clock: int) -> None:
        if self.generation == 0 or clock < self.last_touch:
            raise ValueError(
                "KVDuo update requires a live generation and monotonic clock"
            )
        self.data_version += 1
        self.host_version = -1
        self.last_touch = clock

    def attention_complete(self, clock: int) -> None:
        if self.generation == 0 or clock < self.last_touch:
            raise ValueError("KVDuo attention touch requires a live generation")
        self.last_touch = clock

    def begin_writeback(self) -> KVDuoWritebackTicket:
        if self.generation == 0:
            raise ValueError("Cannot write back an unused KVDuo address")
        return KVDuoWritebackTicket(self.generation, self.data_version)

    def complete_writeback(self, ticket: KVDuoWritebackTicket) -> bool:
        """Publish exactly the copied version; return whether it is current."""
        if ticket.generation != self.generation:
            return False
        self.host_version = ticket.data_version
        return self.host_valid

    def attach_or_restore(self) -> None:
        """Prefix attachment and physical copies intentionally change nothing."""

    def scan_or_score(self) -> None:
        """Indexer-only activity intentionally changes no model timestamp."""


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


class KVDuoPressureAction(Enum):
    """Action the allocation caller must take."""

    SUCCESS = auto()
    WAIT = auto()
    RETRACT_REQUIRED = auto()
    ERROR = auto()


class KVDuoPressureReason(Enum):
    """Cause of a pressure action, kept separate from the action itself."""

    NONE = auto()
    DMA_PENDING = auto()
    MEMORY_SHORTAGE = auto()
    INSUFFICIENT_RECLAIMABLE_PAGES = auto()
    INVALID_STATE = auto()


class KVDuoResourceKind(Enum):
    """Independently accounted resource pools relevant to KVDuo admission."""

    MAIN_KV_PHYSICAL = auto()
    LOGICAL_SLOTS = auto()
    SWA = auto()
    INDEXER = auto()
    COMPRESSION_STATE = auto()
    HOST_KV = auto()


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
class KVDuoHotSlotState:
    logical_position: Optional[int] = None
    last_touch: int = 0
    pin_epoch: int = 0


@dataclass
class KVDuoHotPageState:
    """One request/domain-private hot physical page."""

    physical_page_id: int
    request_id: OwnerIdentity
    domain_id: DomainIdentity
    slots: list[KVDuoHotSlotState]
    last_touch: int
    dma_pins: int = 0


class KVDuoHotPageAllocator:
    """Deterministic metadata allocator for request/domain-private hot pages.

    Physical page ids are supplied by the owning KV pool. Pages are created
    only by :meth:`insert_host_miss`; full-page demotion never calls this API.
    """

    def __init__(self, page_size: int, domain_aliases: Optional[Mapping] = None):
        if page_size <= 0:
            raise ValueError("KVDuo hot page size must be positive")
        self.page_size = page_size
        self.domain_aliases = dict(domain_aliases or {})
        self.pages: Dict[int, KVDuoHotPageState] = {}
        self.by_owner: Dict[Tuple[OwnerIdentity, DomainIdentity], list[int]] = {}
        self.locations: Dict[
            Tuple[OwnerIdentity, DomainIdentity, int], Tuple[int, int]
        ] = {}

    def storage_domain(self, domain: DomainIdentity) -> DomainIdentity:
        return self.domain_aliases.get(domain, domain)

    def insert_host_miss(
        self,
        request_id: OwnerIdentity,
        domain_id: DomainIdentity,
        logical_position: int,
        *,
        clock: int,
        allocate_page: Callable[[], int],
        protected_epoch: int = 0,
    ) -> Tuple[int, int]:
        """Insert a real host miss, allocating a new MRU page if necessary."""
        domain_id = self.storage_domain(domain_id)
        key = (request_id, domain_id, logical_position)
        existing = self.locations.get(key)
        if existing is not None:
            page = self.pages[existing[0]]
            slot = page.slots[existing[1]]
            slot.last_touch = clock
            slot.pin_epoch = protected_epoch
            page.last_touch = max(page.last_touch, clock)
            return existing

        owner = (request_id, domain_id)
        target = None
        for page_id in self.by_owner.get(owner, ()):
            page = self.pages[page_id]
            if any(slot.logical_position is None for slot in page.slots):
                target = page
                break
        if target is None:
            page_id = allocate_page()
            if page_id in self.pages:
                raise ValueError("KVDuo physical hot page is already owned")
            target = KVDuoHotPageState(
                page_id,
                request_id,
                domain_id,
                [KVDuoHotSlotState() for _ in range(self.page_size)],
                clock,
            )
            self.pages[page_id] = target
            self.by_owner.setdefault(owner, []).append(page_id)

        slot_index = next(
            i for i, slot in enumerate(target.slots) if slot.logical_position is None
        )
        target.slots[slot_index] = KVDuoHotSlotState(
            logical_position, clock, protected_epoch
        )
        target.last_touch = max(target.last_touch, clock)
        self.locations[key] = (target.physical_page_id, slot_index)
        return self.locations[key]

    def lookup(
        self,
        request_id: OwnerIdentity,
        domain_id: DomainIdentity,
        logical_position: int,
    ) -> Optional[Tuple[int, int]]:
        return self.locations.get(
            (request_id, self.storage_domain(domain_id), logical_position)
        )


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
class KVDuoTailTransition:
    """Observable result of appending one entry to a logical tail page."""

    page_id: PageIdentity
    offset: int
    new_page: bool
    sealed_page: Optional[PageIdentity]
    writeback_candidates: Tuple[PageIdentity, ...]
    protected_pages: Tuple[PageIdentity, ...]


class KVDuoTailRotation:
    """Model-independent append/seal/protect lifecycle for logical full pages.

    Adapters choose the logical position passed to :meth:`append`, so compressed
    models retain control over when a new KV entry exists. Pages are protected
    by logical page ordinal, never by slicing the last ``N * page_size`` tokens.
    """

    def __init__(
        self,
        *,
        page_size: int,
        tail_protected_pages: int = 2,
        adapter_extra_protected_pages: int = 0,
    ) -> None:
        if page_size <= 0 or tail_protected_pages <= 0:
            raise ValueError("KVDuo page size and tail protection must be positive")
        if adapter_extra_protected_pages < 0:
            raise ValueError("Adapter tail protection cannot be negative")
        self.page_size = page_size
        self.protected_count = max(tail_protected_pages, adapter_extra_protected_pages)
        self.pages: Dict[int, KVDuoFullPageState] = {}
        self._published: Dict[int, Dict[DomainIdentity, set[int]]] = {}
        self._writeback_submitted: set[int] = set()

    def append(
        self,
        logical_position: int,
        domains: Iterable[DomainIdentity],
        *,
        clock: int,
    ) -> KVDuoTailTransition:
        if logical_position < 0:
            raise ValueError("KVDuo logical positions cannot be negative")
        ordinal, offset = divmod(logical_position, self.page_size)
        unique_domains = tuple(dict.fromkeys(domains))
        new_page = ordinal not in self.pages
        if new_page:
            if self.pages and ordinal != max(self.pages) + 1:
                raise ValueError("KVDuo tail pages must be appended contiguously")
            self.pages[ordinal] = KVDuoFullPageState(ordinal, unique_domains)
            self._published[ordinal] = {domain: set() for domain in unique_domains}
        page = self.pages[ordinal]
        if page.domains != unique_domains:
            raise ValueError("KV storage domains cannot change within a tail")

        for domain in page.domains:
            if offset in self._published[ordinal][domain]:
                raise ValueError("A KVDuo tail entry cannot be published twice")
            self._published[ordinal][domain].add(offset)
            page.record_model_write(domain, clock)
            if len(self._published[ordinal][domain]) == self.page_size:
                page.gpu_full_domains.add(domain)

        sealed_page = None
        if all(
            len(offsets) == self.page_size
            for offsets in self._published[ordinal].values()
        ):
            page.sealed = True
            sealed_page = page.page_id
        self._refresh_protection()
        return KVDuoTailTransition(
            page_id=page.page_id,
            offset=offset,
            new_page=new_page,
            sealed_page=sealed_page,
            writeback_candidates=self.writeback_candidates(),
            protected_pages=self.protected_pages,
        )

    def entry_readable(self, page_id: int, domain: DomainIdentity, offset: int) -> bool:
        """Whether a generated tail entry is valid before the page is full."""
        return offset in self._published.get(page_id, {}).get(domain, set())

    @property
    def protected_pages(self) -> Tuple[PageIdentity, ...]:
        ordinals = sorted(self.pages)
        return tuple(ordinals[-self.protected_count :])

    def mark_stable(self, page_id: int) -> None:
        page = self.pages[page_id]
        if not page.sealed:
            raise ValueError("An unsealed KVDuo page cannot become stable")
        page.stable = True

    def writeback_candidates(self) -> Tuple[PageIdentity, ...]:
        """Stable sealed protected pages older than the current append page."""
        if not self.pages:
            return ()
        newest = max(self.pages)
        return tuple(
            ordinal
            for ordinal in sorted(self.pages)
            if ordinal != newest
            and self.pages[ordinal].sealed
            and self.pages[ordinal].stable
            and not self.pages[ordinal].host_full_valid
            and ordinal not in self._writeback_submitted
        )

    def submit_writeback(self, page_id: int) -> None:
        if page_id not in self.writeback_candidates():
            raise ValueError("KVDuo page is not ready for host writeback")
        self._writeback_submitted.add(page_id)
        self.pages[page_id].pins.dma += 1

    def complete_writeback(self, page_id: int) -> None:
        if page_id not in self._writeback_submitted:
            raise ValueError("KVDuo page writeback was not submitted")
        page = self.pages[page_id]
        for domain in page.domains:
            page.mark_host_copy_complete(domain)
        page.pins.dma -= 1
        self._writeback_submitted.remove(page_id)

    def _refresh_protection(self) -> None:
        protected = set(self.protected_pages)
        for ordinal, page in self.pages.items():
            page.tail_protected = ordinal in protected


@dataclass(frozen=True)
class KVDuoAllocationPlan:
    """Byte-accurate request against one specific physical KV pool."""

    available_bytes: int
    new_full_bytes: int = 0
    new_hot_bytes: int = 0
    reserved_bytes: int = 0
    turnover_bytes: int = 0
    # Commitments not already reflected in ``available_bytes``. Allocated or
    # allocator-reserved memory must not be listed here a second time.
    pending_commitment_bytes: int = 0
    resource_kind: KVDuoResourceKind = KVDuoResourceKind.MAIN_KV_PHYSICAL
    independent_budgets: Tuple["KVDuoResourceBudget", ...] = ()

    def __post_init__(self) -> None:
        if (
            min(
                self.available_bytes,
                self.new_full_bytes,
                self.new_hot_bytes,
                self.reserved_bytes,
                self.turnover_bytes,
                self.pending_commitment_bytes,
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
            + self.pending_commitment_bytes
        )

    @property
    def shortfall_bytes(self) -> int:
        return max(0, self.required_bytes - self.available_bytes)

    @property
    def status(self) -> KVDuoPressureStatus:
        if self.shortfall_bytes or self.blocking_resources:
            return KVDuoPressureStatus.RECLAIM_REQUIRED
        return KVDuoPressureStatus.READY

    @property
    def blocking_resources(self) -> Tuple[KVDuoResourceKind, ...]:
        """Non-main pools that must be handled by their own managers."""
        return tuple(
            budget.kind for budget in self.independent_budgets if budget.shortfall_bytes
        )


@dataclass(frozen=True)
class KVDuoPressureResult:
    """Structured pressure outcome used instead of an ambiguous bool."""

    status: KVDuoPressureStatus
    requested_bytes: int
    reclaimed_bytes: int = 0
    remaining_shortfall_bytes: int = 0
    detail: Optional[str] = None
    action: Optional[KVDuoPressureAction] = None
    reason: Optional[KVDuoPressureReason] = None

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

        expected_action = {
            KVDuoPressureStatus.READY: KVDuoPressureAction.SUCCESS,
            KVDuoPressureStatus.RECLAIM_REQUIRED: KVDuoPressureAction.ERROR,
            KVDuoPressureStatus.WAIT_FOR_IO: KVDuoPressureAction.WAIT,
            KVDuoPressureStatus.WAIT_FOR_MEMORY: KVDuoPressureAction.WAIT,
            KVDuoPressureStatus.RETRACT_REQUIRED: KVDuoPressureAction.RETRACT_REQUIRED,
        }[self.status]
        if self.action is None:
            object.__setattr__(self, "action", expected_action)
        if self.reason is None:
            object.__setattr__(
                self,
                "reason",
                (
                    KVDuoPressureReason.NONE
                    if self.status is KVDuoPressureStatus.READY
                    else KVDuoPressureReason.MEMORY_SHORTAGE
                ),
            )


@dataclass(frozen=True)
class KVDuoResourceBudget:
    """Snapshot of one independently managed resource pool."""

    kind: KVDuoResourceKind
    available_bytes: int
    requested_bytes: int
    pending_commitment_bytes: int = 0

    def __post_init__(self) -> None:
        if (
            min(
                self.available_bytes,
                self.requested_bytes,
                self.pending_commitment_bytes,
            )
            < 0
        ):
            raise ValueError("KVDuo resource-budget byte counts cannot be negative")

    @property
    def shortfall_bytes(self) -> int:
        return max(
            0,
            self.requested_bytes + self.pending_commitment_bytes - self.available_bytes,
        )


def plan_kvduo_allocation(
    *,
    available_main_kv_bytes: int,
    requested_full_bytes: int = 0,
    requested_hot_bytes: int = 0,
    incremental_reserved_bytes: int = 0,
    incremental_turnover_bytes: int = 0,
    pending_commitment_bytes: int = 0,
    independent_budgets: Iterable[KVDuoResourceBudget] = (),
) -> KVDuoAllocationPlan:
    """Build a physical-byte plan without double-counting existing allocations.

    ``pending_commitment_bytes`` is intentionally limited to promises not yet
    reflected by the allocator's available count. Other pools are snapshots
    used only to report blockers; they never contribute to main-KV reclamation.
    """
    budgets = tuple(independent_budgets)
    if any(budget.kind is KVDuoResourceKind.MAIN_KV_PHYSICAL for budget in budgets):
        raise ValueError("Main KV must be described by the plan's main-pool fields")
    kinds = [budget.kind for budget in budgets]
    if len(kinds) != len(set(kinds)):
        raise ValueError("Each independent resource pool may appear only once")
    return KVDuoAllocationPlan(
        available_bytes=available_main_kv_bytes,
        new_full_bytes=requested_full_bytes,
        new_hot_bytes=requested_hot_bytes,
        reserved_bytes=incremental_reserved_bytes,
        turnover_bytes=incremental_turnover_bytes,
        pending_commitment_bytes=pending_commitment_bytes,
        independent_budgets=budgets,
    )


def execute_kvduo_pressure_plan(
    plan: KVDuoAllocationPlan,
    reclaim_full_pages: Callable[[int], int],
    available_bytes: Callable[[], int],
    *,
    dma_can_make_progress: bool = False,
) -> KVDuoPressureResult:
    """Execute and recheck a plan against the main sparse-KV physical pool.

    The callback boundary keeps selection/mutation in the coordinator while
    making the budgeting rules independently testable.  No other resource pool
    can accidentally trigger full-page eviction.
    """
    if (
        plan.resource_kind is not KVDuoResourceKind.MAIN_KV_PHYSICAL
        or plan.blocking_resources
    ):
        blocked = ", ".join(kind.name for kind in plan.blocking_resources)
        return KVDuoPressureResult(
            KVDuoPressureStatus.RECLAIM_REQUIRED,
            plan.required_bytes,
            remaining_shortfall_bytes=plan.shortfall_bytes,
            action=KVDuoPressureAction.ERROR,
            reason=KVDuoPressureReason.INVALID_STATE,
            detail=(
                "Full-page reclaim is only valid for the main KV physical pool"
                if not blocked
                else f"Independent resource shortage: {blocked}"
            ),
        )

    initial_shortfall = plan.shortfall_bytes
    if initial_shortfall == 0:
        return KVDuoPressureResult(
            KVDuoPressureStatus.READY,
            plan.required_bytes,
            action=KVDuoPressureAction.SUCCESS,
            reason=KVDuoPressureReason.NONE,
        )
    reclaimed = reclaim_full_pages(initial_shortfall)
    remaining = max(0, plan.required_bytes - available_bytes())
    if remaining == 0:
        return KVDuoPressureResult(
            KVDuoPressureStatus.READY,
            plan.required_bytes,
            reclaimed_bytes=reclaimed,
            action=KVDuoPressureAction.SUCCESS,
            reason=KVDuoPressureReason.NONE,
        )
    if dma_can_make_progress:
        return KVDuoPressureResult(
            KVDuoPressureStatus.WAIT_FOR_IO,
            plan.required_bytes,
            reclaimed_bytes=reclaimed,
            remaining_shortfall_bytes=remaining,
            action=KVDuoPressureAction.WAIT,
            reason=KVDuoPressureReason.DMA_PENDING,
        )
    return KVDuoPressureResult(
        KVDuoPressureStatus.RECLAIM_REQUIRED,
        plan.required_bytes,
        reclaimed_bytes=reclaimed,
        remaining_shortfall_bytes=remaining,
        action=KVDuoPressureAction.ERROR,
        reason=KVDuoPressureReason.INSUFFICIENT_RECLAIMABLE_PAGES,
    )


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
