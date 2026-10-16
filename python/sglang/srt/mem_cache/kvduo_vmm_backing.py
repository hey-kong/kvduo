"""Transactional, address-stable backing exchanges for KVDuo.

This is a storage primitive, not an admission policy. The caller must remove
donor blocks from its allocators and finish their previous GPU accesses before
preparing an exchange. Preparation maps aliases without destroying the donor;
TP ranks can therefore agree on success before committing. No KV is copied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class MappingBackend(Protocol):
    def create(self, size: int): ...
    def map(self, address: int, size: int, handle) -> None: ...
    def unmap(self, address: int, size: int) -> None: ...
    def release(self, handle) -> None: ...


class MappingRollbackError(RuntimeError):
    """The destination is still mapped after a failed permission update."""


@dataclass(frozen=True)
class BackingExchange:
    epoch: int
    donors: tuple[int, ...]
    destinations: tuple[int, ...]


class VMMBacking:
    """A fixed byte budget shared by independently reserved virtual buffers.

    Addresses passed here are block aligned, reserved VAs owned by the caller.
    They must not overlap other backing managers. A block can span multiple model
    pages: admission must account for every page touched by its byte interval.
    The serving allocator remains responsible for page ownership and lifetimes.
    """

    def __init__(self, backend: MappingBackend, *, block_bytes: int, budget_bytes: int):
        if block_bytes <= 0 or budget_bytes < block_bytes:
            raise ValueError("history VMM budget must fit at least one block")
        self.backend = backend
        self.block_bytes = block_bytes
        self.max_blocks = budget_bytes // block_bytes
        self.mappings: dict[int, object] = {}
        self._idle: list[object] = []
        self._handles: list[object] = []
        self._allocated = 0
        self._epoch = 0
        self._pending: PreparedExchange | None = None
        self._poisoned = False
        self._closed = False

    def _check(self):
        if self._closed or self._poisoned:
            raise RuntimeError(
                "history VMM manager is closed or has an incomplete unmap"
            )

    @property
    def allocated_bytes(self):
        return self._allocated * self.block_bytes

    def plan(self, *, donors=(), destinations=()) -> BackingExchange:
        self._check()
        if self._pending is not None:
            raise RuntimeError("another history VMM exchange is pending")
        donors, destinations = tuple(donors), tuple(destinations)
        for addresses in (donors, destinations):
            if len(set(addresses)) != len(addresses):
                raise ValueError("duplicate block address")
            if any(a <= 0 or a % self.block_bytes for a in addresses):
                raise ValueError("block address must be positive and block aligned")
        if any(a not in self.mappings for a in donors):
            raise ValueError("donor has no physical backing")
        if any(a in self.mappings for a in destinations):
            raise ValueError("destination already has physical backing")
        available = self.max_blocks - len(self.mappings) + len(donors)
        if len(destinations) > available:
            raise MemoryError("history VMM exchange exceeds physical budget")
        return BackingExchange(self._epoch, donors, destinations)

    def prepare(self, plan: BackingExchange, *, accesses_complete: bool):
        self._check()
        if not accesses_complete:
            raise RuntimeError("previous GPU accesses have not completed")
        if plan.epoch != self._epoch:
            raise RuntimeError("stale history VMM plan")
        # Validate again: callers may have retained a plan while other work ran.
        if self.plan(donors=plan.donors, destinations=plan.destinations) != plan:
            raise RuntimeError("invalid history VMM plan")
        transaction = PreparedExchange(self, plan)
        self._pending = transaction
        try:
            transaction._prepare()
        except BaseException:
            transaction.abort()
            raise
        return transaction

    def release_idle(self):
        """Return unmapped allocations to CUDA rather than retaining a reserve."""
        self._check()
        if self._pending is not None:
            raise RuntimeError("cannot release handles during a mapping transaction")
        while self._idle:
            handle = self._idle[-1]
            self.backend.release(handle)
            self._idle.pop()
            self._handles.remove(handle)
            self._allocated -= 1

    def close(self, *, accesses_complete: bool):
        if self._closed:
            return
        if not accesses_complete:
            raise RuntimeError("cannot release backing with outstanding GPU accesses")
        if self._pending is not None:
            self._pending.abort()
        # Remove entries only after successful driver operations. A failed
        # cleanup can be retried without unmapping/releasing successful entries.
        for address in list(self.mappings):
            self.backend.unmap(address, self.block_bytes)
            self._idle.append(self.mappings.pop(address))
        while self._handles:
            self.backend.release(self._handles[-1])
            self._handles.pop()
            self._allocated -= 1
        self._idle.clear()
        self._closed = True


class PreparedExchange:
    def __init__(self, pool: VMMBacking, plan: BackingExchange):
        self.pool, self.plan = pool, plan
        self.aliases: dict[int, object] = {}
        self._borrowed_idle: list[object] = []
        self._done = False

    def _prepare(self):
        pool = self.pool
        handles = [pool.mappings[a] for a in self.plan.donors]
        for i, address in enumerate(self.plan.destinations):
            if i < len(handles):
                handle = handles[i]
            else:
                if pool._idle:
                    handle = pool._idle.pop()
                else:
                    handle = pool.backend.create(pool.block_bytes)
                    pool._allocated += 1
                    pool._handles.append(handle)
                self._borrowed_idle.append(handle)
            try:
                pool.backend.map(address, pool.block_bytes, handle)
            except MappingRollbackError:
                self.aliases[address] = handle
                pool._poisoned = True
                raise
            self.aliases[address] = handle

    def abort(self):
        if self._done:
            return
        pool = self.pool
        try:
            for address in list(self.aliases):
                pool.backend.unmap(address, pool.block_bytes)
                del self.aliases[address]
        except BaseException:
            pool._poisoned = True
            raise
        pool._idle.extend(self._borrowed_idle)
        self._borrowed_idle.clear()
        pool._pending = None
        self._done = True

    def commit(self):
        if self._done:
            raise RuntimeError("history VMM transaction already finished")
        pool = self.pool
        pool._check()
        try:
            for i, address in enumerate(self.plan.donors):
                pool.backend.unmap(address, pool.block_bytes)
                handle = pool.mappings.pop(address)
                if i >= len(self.plan.destinations):
                    pool._idle.append(handle)
        except BaseException:
            # A partial commit is fatal for serving. Never publish destinations
            # or return donor pages to an allocator after this failure.
            pool._poisoned = True
            raise
        pool.mappings.update(self.aliases)
        self.aliases.clear()
        self._borrowed_idle.clear()
        pool._epoch += 1
        pool._pending = None
        self._done = True


class CudaMappingBackend:
    """CUDA driver operations; use inside the owning device's context."""

    def __init__(self, device_id: int):
        from sglang.srt.mem_cache.kv_vmm_backing import _check, _driver

        self.driver, self.check = _driver(), _check
        drv = self.driver
        self.prop = drv.CUmemAllocationProp()
        self.prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        self.prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        self.prop.location.id = device_id
        self.access = drv.CUmemAccessDesc()
        self.access.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        self.access.location.id = device_id
        self.access.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    def create(self, size):
        return self.check(self.driver.cuMemCreate(size, self.prop, 0), "cuMemCreate")

    def map(self, address, size, handle):
        self.check(self.driver.cuMemMap(address, size, 0, handle, 0), "cuMemMap")
        try:
            self.check(
                self.driver.cuMemSetAccess(address, size, [self.access], 1),
                "cuMemSetAccess",
            )
        except BaseException:
            try:
                self.check(self.driver.cuMemUnmap(address, size), "cuMemUnmap rollback")
            except BaseException as exc:
                raise MappingRollbackError(
                    "destination mapping could not be rolled back"
                ) from exc
            raise

    def unmap(self, address, size):
        self.check(self.driver.cuMemUnmap(address, size), "cuMemUnmap")

    def release(self, handle):
        self.check(self.driver.cuMemRelease(handle), "cuMemRelease")
