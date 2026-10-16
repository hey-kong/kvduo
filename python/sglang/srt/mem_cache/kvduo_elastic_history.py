"""Address-stable indexer/C128 growth using additional free CUDA memory."""

import logging
from dataclasses import dataclass
from math import ceil

import torch
from sglang.srt.mem_cache.kv_vmm_backing import KvVmmArena, align_up, query_granularity
from sglang.srt.mem_cache.kvduo_vmm_backing import CudaMappingBackend, VMMBacking

logger = logging.getLogger(__name__)


@dataclass
class Region:
    tensor: torch.Tensor
    rows: int
    layers: int
    row_bytes: int

    def spans(self, start, end):
        for layer in range(self.layers):
            base = self.tensor.data_ptr() + layer * self.rows * self.row_bytes
            yield base + start * self.row_bytes, base + end * self.row_bytes


@dataclass
class Growth:
    start: int
    end: int
    addresses: tuple[int, ...]


class KVDuoElasticHistory:
    def __init__(
        self, *, device, initial_pages, maximum_pages, chunk_mb, reserve_bytes
    ):
        self.device = torch.device(device)
        self._device_id = self.device.index
        if self._device_id is None:
            self._device_id = torch.cuda.current_device()
        if not 0 < initial_pages <= maximum_pages:
            raise ValueError("invalid initial/virtual history capacities")
        self.block_bytes = align_up(2 << 20, query_granularity(self._device_id))
        self.arena = KvVmmArena(
            self._device_id,
            reserve_bytes=1 << 40,
            allocation_alignment=self.block_bytes,
        )
        self.initial_pages = self.current_pages = initial_pages
        self.maximum_pages = maximum_pages
        self.chunk_bytes = chunk_mb << 20
        self.reserve_bytes = reserve_bytes
        self.regions = []
        self.growths = []
        self.backing = None
        self.logical_allocator = None

    def allocate(self, *, rows, layers, row_bytes):
        with torch.cuda.use_mem_pool(self.arena.pool):
            tensor = torch.empty(
                (rows * layers, row_bytes), dtype=torch.uint8, device=self.device
            )
        self.regions.append(Region(tensor, rows, layers, row_bytes))
        return tensor

    def _blocks(self, start, end):
        blocks = set()
        for region in self.regions:
            for lo, hi in region.spans(start, end):
                blocks.update(
                    range(
                        lo // self.block_bytes * self.block_bytes,
                        align_up(hi, self.block_bytes),
                        self.block_bytes,
                    )
                )
        return blocks

    def initialize(self):
        ceiling = self._blocks(0, self.maximum_pages + 1)
        addresses = self._blocks(0, self.initial_pages + 1)
        self.backing = VMMBacking(
            CudaMappingBackend(self._device_id),
            block_bytes=self.block_bytes,
            budget_bytes=len(ceiling) * self.block_bytes,
        )
        self.backing.prepare(
            self.backing.plan(destinations=sorted(addresses)), accesses_complete=True
        ).commit()
        self._zero_rows(0, self.initial_pages + 1)
        logger.info(
            "KVDuo elastic history: initial_pages=%d, virtual_pages=%d, committed_bytes=%d, growth_MiB=%d, runtime_reserve_bytes=%d",
            self.initial_pages,
            self.maximum_pages,
            self.backing.allocated_bytes,
            self.chunk_bytes >> 20,
            self.reserve_bytes,
        )

    def _zero_rows(self, start, end):
        for region in self.regions:
            region.tensor.view(region.layers, region.rows, -1)[:, start:end].zero_()

    def bind(self, logical_allocator):
        self.logical_allocator = logical_allocator
        self.reset_free_lists()

    def reset_free_lists(self):
        logical = self.logical_allocator
        logical.free_pages = logical.free_pages[
            logical.free_pages <= self.current_pages
        ]

    def _free_bytes(self):
        # Driver-free memory excludes PyTorch's cached blocks. Do not flush that
        # cache: it can contain working sets needed by subsequent prefill steps.
        return max(0, torch.cuda.mem_get_info(self.device)[0] - self.reserve_bytes)

    def grow(
        self,
        *,
        required_free_pages,
        agree,
        common_available=lambda value: value,
        minimum_only=False,
        prepare_boundary=lambda: None,
    ):
        logical = self.logical_allocator
        logical.merge_and_sort_free()
        available = common_available(logical.available_size() // logical.page_size)
        missing = max(0, required_free_pages - available)
        if not agree((missing, self.current_pages, self.maximum_pages)):
            return 0
        if not missing or self.current_pages == self.maximum_pages:
            return 0
        bytes_per_page = sum(r.row_bytes * r.layers for r in self.regions)
        growth = (
            missing
            if minimum_only
            else max(missing, ceil(self.chunk_bytes / bytes_per_page))
        )
        end = min(self.maximum_pages, self.current_pages + growth)
        if end - self.current_pages < missing:
            return 0
        destinations = tuple(
            sorted(
                self._blocks(self.current_pages + 1, end + 1)
                - self.backing.mappings.keys()
            )
        )
        required_bytes = len(destinations) * self.block_bytes
        if common_available(self._free_bytes()) < required_bytes:
            return 0
        prepare_boundary()
        prepared = False
        try:
            new_free = torch.cat(
                (
                    logical.free_pages,
                    torch.arange(
                        self.current_pages + 1,
                        end + 1,
                        device=self.device,
                        dtype=torch.int64,
                    ),
                )
            )
            prepared = True
        except torch.OutOfMemoryError:
            pass
        if not agree(prepared) or not prepared:
            return 0
        # The host allocator is published only after every rank has mapped and
        # initialized the new rows. Failure releases fresh handles, never FULL.
        if not self._exchange(
            (), destinations, agree, zero_rows=(self.current_pages + 1, end + 1)
        ):
            return 0
        self.growths.append(Growth(self.current_pages + 1, end + 1, destinations))
        added = end - self.current_pages
        self.current_pages = end
        logical.free_pages = new_free
        return added

    def trim(self, *, agree):
        logical = self.logical_allocator
        logical.merge_and_sort_free()
        free = set(logical.free_pages.cpu().tolist())
        returned = 0
        while self.growths:
            growth = self.growths[-1]
            unused = all(p in free for p in range(growth.start, growth.end))
            if not agree((unused, growth.start, growth.end)) or not unused:
                break
            prepared = False
            try:
                new_free = logical.free_pages[logical.free_pages < growth.start]
                prepared = True
            except torch.OutOfMemoryError:
                pass
            if not agree(prepared) or not prepared:
                break
            if not self._exchange(growth.addresses, (), agree):
                break
            logical.free_pages = new_free
            self.current_pages = growth.start - 1
            returned += growth.end - growth.start
            self.growths.pop()
        return returned

    def _exchange(self, donors, destinations, agree, zero_rows=None):
        torch.cuda.current_stream(self.device).synchronize()
        transaction = None
        error = None
        try:
            transaction = self.backing.prepare(
                self.backing.plan(donors=donors, destinations=destinations),
                accesses_complete=True,
            )
            if zero_rows is not None:
                self._zero_rows(*zero_rows)
                torch.cuda.current_stream(self.device).synchronize()
        except Exception as exc:
            error = exc
        same = agree(error is None)
        if not same or error is not None:
            if transaction is not None:
                transaction.abort()
            if self.backing._poisoned:
                raise RuntimeError(
                    "history mapping rollback failed; serving cannot continue"
                ) from error
            self.backing.release_idle()
            if error is not None:
                logger.warning(
                    "KVDuo history growth failed; previous capacity retained: %s", error
                )
            return False
        transaction.commit()
        self.backing.release_idle()
        return True

    def close(self):
        with torch.cuda.device(self._device_id):
            torch.cuda.synchronize(self._device_id)
            if self.backing is not None:
                self.backing.close(accesses_complete=True)
            self.arena.close()
