"""Coarse FULL/history backing loans for the CUDA DeepSeek-V4 layout."""

import logging
from dataclasses import dataclass
from math import ceil

import torch
from sglang.srt.mem_cache.kv_vmm_backing import KvVmmArena, align_up, query_granularity
from sglang.srt.mem_cache.kvduo_shared_hbm import CudaMappingBackend, SharedHBMBlocks

logger = logging.getLogger(__name__)


@dataclass
class Region:
    tensor: torch.Tensor
    kind: str
    rows: int
    layers: int
    row_bytes: int

    def spans(self, start, end):
        for layer in range(self.layers):
            base = self.tensor.data_ptr() + layer * self.rows * self.row_bytes
            yield base + start * self.row_bytes, base + end * self.row_bytes


@dataclass
class Loan:
    start: int
    end: int
    physical_pages: tuple[int, ...]
    donors: tuple[int, ...]
    destinations: tuple[int, ...]


def contiguous_runs(pages):
    runs = []
    for page in sorted(pages):
        if runs and runs[-1][1] == page:
            runs[-1] = (runs[-1][0], page + 1)
        else:
            runs.append((page, page + 1))
    return runs


class KVDuoSharedLayout:
    def __init__(self, *, device, initial_pages, maximum_pages, chunk_mb):
        self.device = torch.device(device)
        device_id = self.device.index
        if device_id is None:
            device_id = torch.cuda.current_device()
        self.block_bytes = align_up(2 << 20, query_granularity(device_id))
        self.arena = KvVmmArena(
            device_id, reserve_bytes=1 << 40, allocation_alignment=self.block_bytes
        )
        self.initial_pages = initial_pages
        self.maximum_pages = maximum_pages
        self.current_pages = initial_pages
        self.chunk_bytes = chunk_mb << 20
        self.regions = []
        self.loans = []
        self.backing = None
        self.logical_allocator = None
        self.physical_allocator = None
        self._device_id = device_id

    def allocate(self, *, rows, layers, row_bytes, kind):
        with torch.cuda.use_mem_pool(self.arena.pool):
            tensor = torch.empty(
                (rows * layers, row_bytes), dtype=torch.uint8, device=self.device
            )
        self.regions.append(Region(tensor, kind, rows, layers, row_bytes))
        return tensor

    def _blocks(self, kind, start, end, *, interior=False):
        blocks = set()
        quantum = self.block_bytes
        for region in self.regions:
            if region.kind != kind:
                continue
            for lo, hi in region.spans(start, end):
                if interior:
                    lo, hi = align_up(lo, quantum), hi // quantum * quantum
                else:
                    lo, hi = lo // quantum * quantum, align_up(hi, quantum)
                blocks.update(range(lo, hi, quantum))
        return blocks

    def initialize(self):
        addresses = self._blocks("full", 0, self.initial_pages + 1)
        addresses |= self._blocks("history", 0, self.initial_pages + 1)
        self.backing = SharedHBMBlocks(
            CudaMappingBackend(self._device_id),
            block_bytes=self.block_bytes,
            budget_bytes=len(addresses) * self.block_bytes,
        )
        self.backing.prepare(
            self.backing.plan(destinations=sorted(addresses)), accesses_complete=True
        ).commit()
        logger.info(
            "KVDuo shared HBM initialized: initial_history_pages=%d, virtual_history_pages=%d, committed_bytes=%d, loan_target_MiB=%d",
            self.initial_pages,
            self.maximum_pages,
            self.backing.allocated_bytes,
            self.chunk_bytes >> 20,
        )
        for region in self.regions:
            region.tensor.view(region.layers, region.rows, -1)[
                :, : self.initial_pages + 1
            ].zero_()

    def bind(self, logical_allocator, physical_allocator):
        self.logical_allocator = logical_allocator
        self.physical_allocator = physical_allocator
        self.reset_free_lists()

    def reset_free_lists(self):
        # The allocators' size/num_pages remain the virtual address ceiling.
        # Only backed page IDs may be advertised as available for execution.
        logical = self.logical_allocator
        logical.free_pages = logical.free_pages[
            logical.free_pages <= self.current_pages
        ]
        if self.loans:
            withheld = torch.tensor(
                [p for loan in self.loans for p in loan.physical_pages],
                dtype=torch.int64,
                device=self.device,
            )
            physical = self.physical_allocator
            physical.free_pages = physical.free_pages[
                ~torch.isin(physical.free_pages, withheld)
            ]

    def minimum_donor_pages(self, required_free_pages, available_pages):
        """Byte lower bound for a mandatory history allocation, not feasibility.

        Fragmentation/alignment are still checked transactionally by grow().
        """
        missing = max(0, required_free_pages - available_pages)
        if not missing or self.current_pages + missing > self.maximum_pages:
            return 0
        destinations = (
            self._blocks(
                "history", self.current_pages + 1, self.current_pages + missing + 1
            )
            - self.backing.mappings.keys()
        )
        idle = self.backing.max_blocks - len(self.backing.mappings)
        blocks = max(0, len(destinations) - idle)
        full_bytes = sum(
            r.row_bytes * r.layers for r in self.regions if r.kind == "full"
        )
        return ceil(blocks * self.block_bytes / full_bytes)

    def grow(
        self,
        *,
        required_free_pages,
        protected_full_pages,
        agree,
        common_available=lambda value: value,
        minimum_only=False,
    ):
        logical, physical = self.logical_allocator, self.physical_allocator
        logical.merge_and_sort_free()
        physical.merge_and_sort_free()
        available = common_available(logical.available_size() // logical.page_size)
        missing = max(0, required_free_pages - available)
        if not agree((missing > 0, self.current_pages, self.maximum_pages)):
            return 0
        if not missing or self.current_pages == self.maximum_pages:
            return 0
        bytes_per_page = sum(
            r.row_bytes * r.layers for r in self.regions if r.kind == "history"
        )
        growth = (
            missing
            if minimum_only
            else max(missing, ceil(self.chunk_bytes / bytes_per_page))
        )
        end = min(self.maximum_pages, self.current_pages + growth)
        if end - self.current_pages < missing:
            return 0
        destinations = self._blocks("history", self.current_pages + 1, end + 1)
        destinations.difference_update(self.backing.mappings)
        idle_count = self.backing.max_blocks - len(self.backing.mappings)
        need = max(0, len(destinations) - idle_count)
        free = physical.free_pages.cpu().tolist()
        allowance = max(0, len(free) - protected_full_pages)
        donors, withheld = set(), []
        # Prefer long contiguous runs to avoid lending partly occupied blocks.
        for start, stop in sorted(
            contiguous_runs(free), key=lambda r: (r[0] - r[1], r[0])
        ):
            if len(donors) >= need:
                break
            stop = min(stop, start + allowance - len(withheld))
            if stop <= start:
                continue
            blocks = self._blocks("full", start, stop, interior=True)
            blocks.intersection_update(self.backing.mappings)
            if not blocks:
                continue
            if len(donors | blocks) >= need:
                lo, hi = start + 1, stop
                while lo < hi:
                    mid = (lo + hi) // 2
                    trial = self._blocks("full", start, mid, interior=True)
                    trial.intersection_update(self.backing.mappings)
                    if len(donors | trial) >= need:
                        hi = mid
                    else:
                        lo = mid + 1
                stop = lo
                blocks = self._blocks("full", start, stop, interior=True)
                blocks.intersection_update(self.backing.mappings)
            donors.update(blocks)
            withheld.extend(range(start, stop))
        possible = len(donors) >= need
        # Physical IDs/free-list fragmentation may differ between TP ranks.
        # Agree on the published logical range and feasibility, not local VAs.
        signature = (possible, self.current_pages, end)
        if not agree(signature) or not possible:
            return 0
        loan = Loan(
            self.current_pages + 1,
            end + 1,
            tuple(withheld),
            tuple(sorted(donors)),
            tuple(sorted(destinations)),
        )
        prepared = False
        try:
            new_physical = physical.free_pages[
                ~torch.isin(
                    physical.free_pages,
                    torch.tensor(withheld, dtype=torch.int64, device=self.device),
                )
            ]
            new_logical = torch.cat(
                (
                    logical.free_pages,
                    torch.arange(
                        loan.start, loan.end, device=self.device, dtype=torch.int64
                    ),
                )
            )
            prepared = True
        except torch.OutOfMemoryError:
            pass
        if not agree(prepared) or not prepared:
            return 0
        if not self._exchange(loan.donors, loan.destinations, agree):
            return 0
        # Initialize only newly published logical rows, after donors are revoked.
        for region in self.regions:
            if region.kind == "history":
                region.tensor.view(region.layers, region.rows, -1)[
                    :, loan.start : loan.end
                ].zero_()
        physical.free_pages, logical.free_pages = new_physical, new_logical
        self.current_pages = end
        self.loans.append(loan)
        return len(withheld)

    def trim(self, *, agree):
        """Return wholly unused trailing loans; never move live history entries."""
        logical, physical = self.logical_allocator, self.physical_allocator
        logical.merge_and_sort_free()
        physical.merge_and_sort_free()
        free = set(logical.free_pages.cpu().tolist())
        returned = 0
        while self.loans:
            loan = self.loans[-1]
            unused = all(p in free for p in range(loan.start, loan.end))
            if not agree((unused, loan.start, loan.end)) or not unused:
                break
            prepared = False
            try:
                new_logical = logical.free_pages[logical.free_pages < loan.start]
                new_physical = torch.cat(
                    (
                        physical.free_pages,
                        torch.tensor(
                            loan.physical_pages, device=self.device, dtype=torch.int64
                        ),
                    )
                )
                prepared = True
            except torch.OutOfMemoryError:
                pass
            if not agree(prepared) or not prepared:
                break
            if not self._exchange(loan.destinations, loan.donors, agree):
                break
            logical.free_pages, physical.free_pages = new_logical, new_physical
            self.current_pages = loan.start - 1
            returned += len(loan.physical_pages)
            self.loans.pop()
        return returned

    def _exchange(self, donors, destinations, agree):
        # Caller has joined decode/backup streams. This is not run in steady
        # state: only at an actual loan/return boundary.
        torch.cuda.current_stream(self.device).synchronize()
        transaction = None
        error = None
        try:
            transaction = self.backing.prepare(
                self.backing.plan(donors=donors, destinations=destinations),
                accesses_complete=True,
            )
        except Exception as exc:
            error = exc
        same = agree(error is None)
        if not same or error is not None:
            if transaction is not None:
                transaction.abort()
            if self.backing._poisoned:
                raise RuntimeError(
                    "shared HBM mapping rollback failed; serving cannot continue"
                ) from error
            if error is not None:
                logger.warning(
                    "KVDuo backing loan preparation failed; original mappings retained: %s",
                    error,
                )
            return False
        transaction.commit()
        return True

    def close(self):
        with torch.cuda.device(self._device_id):
            torch.cuda.synchronize(self._device_id)
            if self.backing is not None:
                self.backing.close(accesses_complete=True)
            self.arena.close()
