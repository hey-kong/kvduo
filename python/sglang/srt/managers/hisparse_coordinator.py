# to be combined with the sparse coordinator class and sparse algorithm family

import logging
from typing import List, NamedTuple, Sequence, Union

import torch

from sglang.kernels.ops.kvcache.hisparse import (
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseDSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.sparsity.core.kvduo_state import (
    KVDuoPressureAction,
    KVDuoPressureReason,
    KVDuoPressureResult,
    KVDuoPressureStatus,
    KVDuoResidencyCatalog,
    execute_kvduo_pressure_plan,
    plan_kvduo_allocation,
)
from sglang.srt.mem_cache.sparsity.core.kvduo_prefix_cache import (
    HostPrefixRecord,
    KVDuoHostPrefixCache,
    KVDuoPrefixPageView,
    begin_kvduo_prefix_restore,
    finish_kvduo_prefix_restore,
    plan_kvduo_prefix_restore,
)
from sglang.srt.utils import get_device_module, is_hip
from sglang.srt.utils.common import get_num_new_pages, is_pin_memory_available

device_module = get_device_module()

_is_hip = is_hip()

logger = logging.getLogger(__name__)

KVDUO_STATS_POLL_INTERVAL = 8


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseTokenStats(NamedTuple):
    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float


class HiSparseCoordinator:
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: Union[
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ],
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
        host_to_device_ratio: int = 2,
        swap_in_block_size: int = 960,
        tail_protected_pages: int = 0,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device
        self.swap_in_block_size = swap_in_block_size
        self.tail_protected_pages = tail_protected_pages
        self.enable_mixed_residency = tail_protected_pages > 0
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        if self.is_dsv4_hisparse:
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
            page_size = self.mem_pool_device.page_size
            num_host_pages = (
                self.token_to_kv_pool_allocator.size_full // self.compress_ratio
                + page_size
                - 1
            ) // page_size
            self.mem_pool_host = DeepSeekV4PagedHostPool(
                pool_name="dsv4_hisparse_c4",
                device_buffers=self.mem_pool_device.kv_buffer,
                item_bytes=self.mem_pool_device.bytes_per_page_padded,
                num_host_pages=num_host_pages,
                slot_page_size=page_size,
                layout="layer_first",
            )
            self.item_size_bytes = (
                self.mem_pool_device.kv_cache_total_dim
                * self.mem_pool_device.store_dtype.itemsize
            )
        else:
            assert isinstance(
                self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
            )
            self.mem_pool_device: HiSparseDSATokenToKVPool = (
                self.token_to_kv_pool_allocator.get_kvcache()
            )
            self.mem_pool_host = MLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=self.mem_pool_device.page_size,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
            self.item_size_bytes = self.mem_pool_host.token_stride_size
        self.page_size = self.mem_pool_device.page_size
        if self.enable_mixed_residency:
            # KVDuo's resolver has one compiled, capture-stable metadata width.
            # This is deliberately not a physical KV reservation.
            max_hot_capacity = 16 * top_k
            if max_hot_capacity > torch.iinfo(torch.int16).max:
                raise ValueError(
                    "KVDuo 16K capacity exceeds the int16 resolver slot type"
                )
            self.device_buffer_size = (
                (max_hot_capacity + self.page_size - 1) // self.page_size
            ) * self.page_size

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        max_compressed_context_len = (
            max_context_len + self.compress_ratio - 1
        ) // self.compress_ratio

        # to have an extra page for new tokens
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_req_slots, self.padded_buffer_size),
            dtype=torch.int64,
            device=device,
        )
        self.req_device_buffer_size = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        # Capture-stable dynamic capacity consumed by the resolver kernel. It
        # excludes the separate turnover/newest page.
        self.req_device_buffer_size_gpu = torch.zeros(
            max_num_req_slots, dtype=torch.int32, device=device
        )
        self.req_to_host_pool = torch.full(
            (max_num_req_slots, max_compressed_context_len + self.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.req_to_host_pool_allocated_len = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        # Capture-stable logical-address rows consumed by KVDuo's fused resolver.
        # Unlike ReqToTokenPool this is always int64 and, for DSV4, is indexed in
        # compressed C4 position space rather than original-token space.
        if self.enable_mixed_residency:
            # Authoritative control-plane ownership/state catalog. CUDA tensors
            # below are capture-stable mirrors, not independent ownership data.
            self.kvduo_residency = KVDuoResidencyCatalog()
            self.req_to_full_lookup = torch.full(
                (max_num_req_slots, max_compressed_context_len + self.page_size),
                -1,
                dtype=torch.int64,
                device=device,
            )
            self.req_reserved_logical = torch.full(
                (max_num_req_slots,), -1, dtype=torch.int64, device=device
            )
            mapping_size = (
                self.mem_pool_device.full_to_hisparse_device_index_mapping.numel()
            )
            self.full_last_touch = torch.zeros(
                mapping_size, dtype=torch.int64, device=device
            )
            # One aggregate timestamp per physical storage identity is enough
            # for page-LRU (page max of entry max). Version mirrors make async
            # host completion generation-safe without per-layer duplicates.
            self.full_generation = torch.zeros(
                mapping_size, dtype=torch.int64, device=device
            )
            self.full_data_version = torch.zeros(
                mapping_size, dtype=torch.int64, device=device
            )
            self.full_host_version = torch.full(
                (mapping_size,), -1, dtype=torch.int64, device=device
            )
            self.full_touch_clock = torch.zeros(1, dtype=torch.int64, device=device)
            self.kvduo_swap_status = torch.zeros(
                (self.mem_pool_device.layer_num, max_num_req_slots),
                dtype=torch.int32,
                device=device,
            )
            # Fixed-address [host_misses, valid_accesses] counters. The resolver
            # only performs atomic updates; graph-external policy consumes them.
            self.kvduo_resolver_stats = torch.zeros(
                (self.mem_pool_device.layer_num, max_num_req_slots, 2),
                dtype=torch.int32,
                device=device,
            )
            self._kvduo_stats_snapshot = torch.empty_like(self.kvduo_resolver_stats)
            # Growth is opportunistic: the current 2K-or-larger workset can
            # always resolve misses through entry-LRU. Poll cumulative counters
            # only every few replays and stage them asynchronously so the CPU
            # never blocks the next replay merely to decide optional growth.
            self.kvduo_stats_poll_interval = KVDUO_STATS_POLL_INTERVAL
            self._kvduo_replay_count = 0
            self._kvduo_stats_host = torch.empty(
                self.kvduo_resolver_stats.shape,
                dtype=self.kvduo_resolver_stats.dtype,
                device="cpu",
                pin_memory=is_pin_memory_available(device),
            )
            self._kvduo_stats_stream = device_module.Stream()
            self._kvduo_stats_snapshot_ready_event = device_module.Event()
            self._kvduo_stats_event = device_module.Event()
            self._kvduo_stats_pending = False
            self._kvduo_stats_pending_owners = {}
            self._active_kvduo_reqs = {}
            self._kvduo_host_valid_len = [0] * max_num_req_slots
            self._pending_kvduo_host_valid = []
            self._kvduo_pressure_protected = None
            self._kvduo_hot_pressure_protected = set()
            # Physical buffers are layer-first.  A carrier reserves one row
            # page from the legacy cross-layer allocator; ownership of each
            # layer fragment is then tracked independently.  Consequently a
            # hot page can be reused/released for one layer without affecting
            # the same row range in other layers.  A carrier is returned to the
            # legacy allocator only after every layer fragment is free.
            self.kvduo_req_hot_capacity = torch.zeros(
                (self.mem_pool_device.layer_num, max_num_req_slots),
                dtype=torch.int64,
                device="cpu",
            )
            self.kvduo_req_hot_capacity_gpu = torch.zeros(
                (self.mem_pool_device.layer_num, max_num_req_slots),
                dtype=torch.int32,
                device=device,
            )
            self._kvduo_hot_carriers = {}
            self._kvduo_free_layer_pages = [
                set() for _ in range(self.mem_pool_device.layer_num)
            ]
            self._kvduo_req_layer_pages = {}
            self.host_prefix_cache = KVDuoHostPrefixCache(self.mem_pool_host.size)
            self._req_host_prefix_records = [set() for _ in range(max_num_req_slots)]
        else:
            self.kvduo_residency = None
            self.req_to_full_lookup = None
            self.req_reserved_logical = None
            self.full_last_touch = None
            self.full_generation = None
            self.full_data_version = None
            self.full_host_version = None
            self.full_touch_clock = None
            self.kvduo_swap_status = None
            self.kvduo_resolver_stats = None
            self._active_kvduo_reqs = None
            self._kvduo_host_valid_len = None
            self._pending_kvduo_host_valid = None
            self._kvduo_pressure_protected = None
            self._kvduo_hot_pressure_protected = None
            self.kvduo_req_hot_capacity = None
            self.kvduo_req_hot_capacity_gpu = None
            self._kvduo_hot_carriers = None
            self._kvduo_free_layer_pages = None
            self._kvduo_req_layer_pages = None
            self.host_prefix_cache = None
            self._req_host_prefix_records = None

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        self.ack_staging_queue: List[HiSparseAct] = []
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # initialize data structures for swap-in kernel
        layer_num = self.mem_pool_device.layer_num
        if self.enable_mixed_residency:
            max_hot_pages = (
                self.device_buffer_size + self.page_size - 1
            ) // self.page_size
            self.hot_page_last_touch = torch.zeros(
                (layer_num, max_num_req_slots, max_hot_pages),
                dtype=torch.int64,
                device=device,
            )
        else:
            self.hot_page_last_touch = None
        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_req_slots, 1)
            .contiguous()
        )
        self._device_buffer_arange_i32 = torch.arange(
            self.device_buffer_size, dtype=torch.int32, device=device
        )

        # Pre-allocated output buffer for swap_in_selected_pages (CUDA-graph safe)
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.raw_indices_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        self._skip_first_backup = [False] * max_num_req_slots
        self._mixed_slots = [False] * max_num_req_slots
        if self.enable_mixed_residency:
            # Decode allocation is owned by the token allocator, but pressure
            # policy belongs here.  The callback runs immediately before a
            # real sparse-KV page allocation and therefore cannot confuse a
            # logical/SWA/indexer shortage with physical main-KV pressure.
            self.token_to_kv_pool_allocator.enable_kvduo_full_decode = True
            self.token_to_kv_pool_allocator.kvduo_physical_allocation_guard = (
                self._guard_kvduo_physical_allocation
            )

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def destroy(self) -> None:
        # Drain in-flight transfers so the buffer is idle, then unregister it.
        # See HostKVCache.destroy for why the explicit unregister matters.
        self.write_staging_stream.synchronize()
        self.decode_backup_stream.synchronize()
        if self.enable_mixed_residency:
            self._kvduo_stats_stream.synchronize()
        self.mem_pool_host.destroy()

    def get_token_stats(self) -> HiSparseTokenStats:
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        host_capacity = self.mem_pool_host.size
        host_tokens = host_capacity - self.mem_pool_host.available_size()
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

    def _host_prefix_identity(self, req: Req, page_ordinal: int):
        fill_ids = req.get_fill_ids()
        token_end = min(
            len(fill_ids),
            (page_ordinal + 1) * self.page_size * self.compress_ratio,
        )
        return (self.compress_ratio, tuple(fill_ids[:token_end]))

    def augment_kvduo_prefix_match(self, req: Req, match_result):
        """Extend a GPU Radix hit with the longest valid host-owned prefix.

        The Radix node remains the lock anchor for its GPU portion.  Host pages
        use content identities and request references, so they remain pinned
        even before a recyclable request-pool slot is assigned.
        """
        if not self.enable_mixed_residency:
            return match_result
        for identity in getattr(req, "kvduo_host_prefix_records", ()):
            if identity in self.host_prefix_cache.records:
                self.host_prefix_cache.release(identity, req.rid)
        req.kvduo_host_prefix_records = set()
        gpu_len = len(match_result.device_indices)
        page_tokens = self.page_size * self.compress_ratio
        max_prefix = req._compute_max_prefix_len(len(req.get_fill_ids()))
        max_prefix = max_prefix // page_tokens * page_tokens
        host_end = gpu_len // page_tokens * page_tokens
        pages = []
        identities = []
        for ordinal in range(max_prefix // page_tokens):
            identity = self._host_prefix_identity(req, ordinal)
            record = self.host_prefix_cache.records.get(identity)
            gpu_full = (ordinal + 1) * page_tokens <= gpu_len
            if not gpu_full and (record is None or not record.fully_valid):
                break
            pages.append(
                KVDuoPrefixPageView(
                    identity=identity,
                    domains=("main_kv",),
                    gpu_full_domains=(
                        frozenset({"main_kv"}) if gpu_full else frozenset()
                    ),
                )
            )
            host_end = (ordinal + 1) * page_tokens
            if not gpu_full:
                self.host_prefix_cache.acquire(identity, req.rid, 0)
                identities.append(identity)

        if host_end <= gpu_len:
            return match_result
        # Repeated scheduling matches are idempotent because request references
        # are sets.  Keep the identities on Req rather than a request-slot row.
        req.kvduo_host_prefix_records = set(identities)
        plan = plan_kvduo_prefix_restore(
            pages,
            self.host_prefix_cache,
            {"main_kv": self._physical_page_bytes()},
        )
        return match_result._replace(
            last_host_node=match_result.last_device_node,
            best_match_node=match_result.last_device_node,
            host_hit_length=host_end - gpu_len,
            full_kv_hit_length=host_end,
            cache_protected_len=gpu_len,
            kvduo_residency_plan=plan,
        )

    def init_kvduo_load_back(self, req: Req, host_hit_length: int):
        """Allocate and restore the host-only suffix before prefill executes."""
        plan = getattr(req, "kvduo_residency_plan", None)
        if plan is None or host_hit_length <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        commitment = begin_kvduo_prefix_restore(plan, self.host_prefix_cache)
        prefix_len = len(req.prefix_indices)
        target_len = prefix_len + host_hit_length
        prefix_cpu = torch.tensor([prefix_len], dtype=torch.int64)
        target_cpu = torch.tensor([target_len], dtype=torch.int64)
        prefix_gpu = prefix_cpu.to(self.device)
        target_gpu = target_cpu.to(self.device)
        last_loc = (
            req.prefix_indices[-1:].to(torch.int64)
            if prefix_len
            else torch.full((1,), -1, dtype=torch.int64, device=self.device)
        )
        restored = None
        try:
            allocator = self.token_to_kv_pool_allocator
            logical_pages = get_num_new_pages(
                seq_lens=target_cpu,
                page_size=allocator.page_size,
                prefix_lens=prefix_cpu,
            )
            if self.is_dsv4_hisparse:
                physical_pages = get_num_new_pages(
                    seq_lens=target_cpu // self.compress_ratio,
                    page_size=self.page_size,
                    prefix_lens=prefix_cpu // self.compress_ratio,
                )
            else:
                physical_pages = logical_pages

            logical_allocator = allocator.logical_attn_allocator
            physical_allocator = allocator.hisparse_attn_allocator
            logical_before = logical_allocator.available_size() // allocator.page_size
            physical_before = physical_allocator.available_size() // self.page_size
            local_ready = logical_before >= logical_pages
            if self.tp_world_size > 1:
                ready = torch.tensor(
                    int(local_ready), dtype=torch.int32, device=self.device
                )
                torch.distributed.all_reduce(
                    ready, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
                )
                local_ready = bool(ready.item())
            if not local_ready:
                logger.warning(
                    "KVDuo host-prefix restore deferred for req %s: logical pool "
                    "needs %d pages, had %d before allocation; physical pool needs "
                    "%d pages, had %d",
                    req.rid,
                    logical_pages,
                    logical_before,
                    physical_pages,
                    physical_before,
                )
                return None

            # A restore can need a full physical page even though its tokens were
            # discounted from the prefill-input budget. Reclaim before entering
            # alloc_extend, while protecting every GPU prefix location that this
            # request is about to append to.
            protected = self.mem_pool_device.translate_loc_from_full_to_compressed(
                req.prefix_indices
            )
            self._kvduo_pressure_protected = protected
            try:
                pressure = self._reclaim_for_physical_allocation(
                    physical_pages * self.page_size
                )
            finally:
                self._kvduo_pressure_protected = None
            local_ready = pressure.action is KVDuoPressureAction.SUCCESS
            if self.tp_world_size > 1:
                ready = torch.tensor(
                    int(local_ready), dtype=torch.int32, device=self.device
                )
                torch.distributed.all_reduce(
                    ready, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
                )
                local_ready = bool(ready.item())
            if not local_ready:
                logger.warning(
                    "KVDuo host-prefix restore deferred for req %s: physical pool "
                    "reclaim failed (needs %d pages, had %d before reclaim, "
                    "action=%s, reason=%s, remaining_shortfall_bytes=%d); logical "
                    "pool had %d pages",
                    req.rid,
                    physical_pages,
                    physical_before,
                    pressure.action.name,
                    pressure.reason.name,
                    pressure.remaining_shortfall_bytes,
                    logical_before,
                )
                return None

            restored = self.token_to_kv_pool_allocator.alloc_extend(
                prefix_gpu,
                prefix_cpu,
                target_gpu,
                target_cpu,
                last_loc,
                host_hit_length,
            )
            allocation_ready = restored is not None
            if self.tp_world_size > 1:
                ready = torch.tensor(
                    int(allocation_ready), dtype=torch.int32, device=self.device
                )
                torch.distributed.all_reduce(
                    ready, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
                )
                allocation_ready = bool(ready.item())
            if not allocation_ready:
                if restored is not None:
                    self.token_to_kv_pool_allocator.free(restored)
                    restored = None
                logger.warning(
                    "KVDuo host-prefix restore deferred for req %s: alloc_extend "
                    "failed after admission (logical pool had %d pages; physical "
                    "pool had %d pages before reclaim; needed logical=%d, physical=%d)",
                    req.rid,
                    logical_before,
                    physical_before,
                    logical_pages,
                    physical_pages,
                )
                return None

            host_locs = []
            versions = []
            touches = []
            first_page = prefix_len // (self.page_size * self.compress_ratio)
            page_count = host_hit_length // (self.page_size * self.compress_ratio)
            for ordinal in range(first_page, first_page + page_count):
                record = self.host_prefix_cache.records[
                    self._host_prefix_identity(req, ordinal)
                ]
                if not record.fully_valid:
                    raise RuntimeError(
                        "KVDuo host-prefix version changed during restore"
                    )
                host_locs.extend(record.host_locations["main_kv"])
                versions.extend(record.host_versions["main_kv"])
                touches.extend(
                    record.model_touches.get(
                        "main_kv", (0,) * len(record.host_locations["main_kv"])
                    )
                )

            compressed = self.mem_pool_device.translate_loc_from_full_to_compressed(
                restored
            )
            device_locs = (
                self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                    restored
                )
            )
            host_tensor = torch.tensor(host_locs, dtype=torch.int64, device=self.device)
            if len(host_tensor) != len(device_locs):
                raise RuntimeError(
                    "KVDuo restore adapter produced mismatched locations"
                )
            for layer_id in range(self.mem_pool_device.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_tensor,
                    device_locs,
                    layer_id,
                    io_backend="kernel",
                )
            version_tensor = torch.tensor(
                versions, dtype=torch.int64, device=self.device
            )
            self.full_generation[compressed] += 1
            self.full_data_version[compressed] = version_tensor
            self.full_host_version[compressed] = version_tensor
            # Restore is a copy, not a model write or attention touch.
            self.full_last_touch[compressed] = torch.tensor(
                touches, dtype=torch.int64, device=self.device
            )
            req.kvduo_restored_prefix_len = target_len
            return restored
        except Exception:
            if restored is not None:
                self.token_to_kv_pool_allocator.free(restored)
            raise
        finally:
            finish_kvduo_prefix_restore(commitment, self.host_prefix_cache)

    def _evict_host_prefix_for_slots(self, required_slots: int) -> None:
        if not self.enable_mixed_residency:
            return
        shortfall = max(0, required_slots - self.mem_pool_host.available_size())
        victims = self.host_prefix_cache.evict_lru(shortfall)
        locations = {
            location
            for record in victims
            for domain in record.domains
            for location in record.host_locations[domain]
        }
        if locations:
            self.mem_pool_host.free(
                torch.tensor(sorted(locations), dtype=torch.int64, device=self.device)
            )

    def _attach_cached_host_prefix(self, req: Req, host_len: int) -> int:
        """Attach the longest contiguous valid host prefix to this request row."""
        if not self.enable_mixed_residency:
            return 0
        owner = (req.rid, req.req_pool_idx)
        attached = 0
        for ordinal in range(host_len // self.page_size):
            identity = self._host_prefix_identity(req, ordinal)
            record = self.host_prefix_cache.records.get(identity)
            if record is None or not record.fully_valid:
                break
            locations = record.host_locations["main_kv"]
            start = ordinal * self.page_size
            self.req_to_host_pool[req.req_pool_idx, start : start + self.page_size] = (
                torch.tensor(locations, dtype=torch.int64, device=self.device)
            )
            self.host_prefix_cache.acquire(
                identity, owner, int(self.full_touch_clock[0])
            )
            self._req_host_prefix_records[req.req_pool_idx].add(identity)
            attached += self.page_size
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = attached
        return attached

    def _release_host_prefix_refs(self, req: Req) -> None:
        if not self.enable_mixed_residency:
            return
        owner = (req.rid, req.req_pool_idx)
        if req.req_pool_idx is not None:
            for identity in self._req_host_prefix_records[req.req_pool_idx]:
                if identity in self.host_prefix_cache.records:
                    self.host_prefix_cache.release(identity, owner)
            self._req_host_prefix_records[req.req_pool_idx].clear()
        for identity in getattr(req, "kvduo_host_prefix_records", ()):
            if identity in self.host_prefix_cache.records:
                self.host_prefix_cache.release(identity, req.rid)
        req.kvduo_host_prefix_records = set()

    def release_kvduo_match_refs(self, req: Req) -> None:
        """Drop host match pins for a request that never reached execution."""
        if not self.enable_mixed_residency:
            return
        for identity in getattr(req, "kvduo_host_prefix_records", ()):
            if identity in self.host_prefix_cache.records:
                self.host_prefix_cache.release(identity, req.rid)
        req.kvduo_host_prefix_records = set()

    def _host_record_locations_for_request(self, req: Req) -> set[int]:
        locations = set()
        for identity in self._req_host_prefix_records[req.req_pool_idx]:
            record = self.host_prefix_cache.records.get(identity)
            if record is not None:
                for domain in record.domains:
                    locations.update(record.host_locations[domain])
        return locations

    def _retain_request_host_prefix(self, req: Req) -> set[int]:
        """Transfer complete host pages from request ownership to cache ownership."""
        valid_len = self._kvduo_host_valid_len[req.req_pool_idx]
        valid_len = valid_len // self.page_size * self.page_size
        if valid_len == 0:
            return set()
        host_locs = self.req_to_host_pool[req.req_pool_idx, :valid_len]
        versions = self.full_host_version[
            self.req_to_full_lookup[req.req_pool_idx, :valid_len]
        ]
        touches = self.full_last_touch[
            self.req_to_full_lookup[req.req_pool_idx, :valid_len]
        ]
        # One compact transfer, not one synchronization per page.
        host_locs_cpu = host_locs.cpu().tolist()
        versions_cpu = versions.cpu().tolist()
        touches_cpu = touches.cpu().tolist()
        retained = set()
        for ordinal, start in enumerate(range(0, valid_len, self.page_size)):
            identity = self._host_prefix_identity(req, ordinal)
            locations = tuple(host_locs_cpu[start : start + self.page_size])
            version = tuple(versions_cpu[start : start + self.page_size])
            touch = tuple(touches_cpu[start : start + self.page_size])
            existing = self.host_prefix_cache.records.get(identity)
            if existing is None:
                record = HostPrefixRecord(
                    identity=identity,
                    domains=("main_kv",),
                    host_locations={"main_kv": locations},
                    data_versions={"main_kv": version},
                    host_versions={"main_kv": version},
                    model_touches={"main_kv": touch},
                    cache_reference=True,
                    last_access=int(self.full_touch_clock[0]),
                    tie_break_key=(ordinal, repr(identity)),
                )
                self.host_prefix_cache.insert(record)
                retained.update(locations)
            else:
                # Existing shared storage remains authoritative; this request's
                # duplicate physical page is intentionally not retained.
                retained.update(existing.host_locations["main_kv"])
        return retained

    def admit_request_into_staging(self, req: Req) -> None:
        req.hisparse_staging = True

        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.extend_range.end
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        prefill_len = len(device_indices)
        attached = self._attach_cached_host_prefix(req, prefill_len)
        new_host_slots = max(
            0,
            ((prefill_len + self.page_size - 1) // self.page_size * self.page_size)
            - attached,
        )
        self._evict_host_prefix_for_slots(new_host_slots)
        host_indices = self.mem_pool_host.alloc_paged_token_slots(
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            0,
            prefill_len,
        )

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

    def admit_request_direct(self, req: Req) -> None:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging DMA entirely. Only allocates a small device buffer
        (4KB) for decode-time swap-in, then marks the request as ready.
        Host indices were already written to req_to_host_pool.

        Metadata fixups after alloc_device_buffer():
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.
        """
        self.alloc_device_buffer(req)

        host_len = self.host_token_len(req.kv.kv_allocated_len)
        if host_len <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            self._preload_to_device_buffer(req)
        else:
            # Long sequence: reset device_buffer_tokens to -1 so the kernel
            # sees all slots as empty -> every top-k lookup is a miss -> host load.
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1

        req.hisparse_staging = False
        self._skip_first_backup[req.req_pool_idx] = True
        logger.debug("HiSparse: admitting request %s directly", req.rid)

    def host_token_len(self, kv_allocated_len: int) -> int:
        if self.is_dsv4_hisparse:
            return kv_allocated_len // self.compress_ratio
        return kv_allocated_len

    def _preload_to_device_buffer(self, req: Req) -> None:
        """Preload all tokens from host pool into the device buffer."""
        n = self.host_token_len(req.kv.kv_allocated_len)
        host_indices = self.req_to_host_pool[req.req_pool_idx, :n]
        device_locs = self.req_to_device_buffer[req.req_pool_idx, :n]

        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def _physical_page_bytes(self) -> int:
        """Actual cross-layer bytes represented by one allocator page."""
        per_layer = getattr(
            self.mem_pool_device,
            "bytes_per_page_padded",
            self.item_size_bytes * self.page_size,
        )
        return per_layer * self.mem_pool_device.layer_num

    def _physical_bytes(self, num_slots: int) -> int:
        """Convert an aligned allocator request to actual physical bytes."""
        if num_slots % self.page_size:
            raise ValueError(
                f"KVDuo physical allocation must be page aligned: {num_slots=}"
            )
        return num_slots // self.page_size * self._physical_page_bytes()

    def _reclaim_for_physical_allocation(
        self, need_size: int, *, turnover_size: int = 0
    ) -> KVDuoPressureResult:
        """Reclaim full pages for an imminent sparse-pool allocation only.

        This intentionally consults the dedicated physical allocator rather
        than ``available_size()`` on the composite allocator, whose minimum may
        instead reflect logical KV, SWA, indexer, or compression-state pressure.
        """
        if need_size < 0 or turnover_size < 0 or turnover_size > need_size:
            return KVDuoPressureResult(
                KVDuoPressureStatus.RECLAIM_REQUIRED,
                0,
                action=KVDuoPressureAction.ERROR,
                reason=KVDuoPressureReason.INVALID_STATE,
                detail=(
                    "invalid sparse-KV allocation sizes: "
                    f"{need_size=}, {turnover_size=}"
                ),
            )
        physical_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        available_slots = (
            physical_allocator.available_size() // self.page_size * self.page_size
        )
        plan = plan_kvduo_allocation(
            available_main_kv_bytes=self._physical_bytes(available_slots),
            requested_hot_bytes=self._physical_bytes(need_size - turnover_size),
            incremental_turnover_bytes=self._physical_bytes(turnover_size),
        )
        if not self.enable_mixed_residency:
            remaining = plan.shortfall_bytes
            return KVDuoPressureResult(
                (
                    KVDuoPressureStatus.READY
                    if remaining == 0
                    else KVDuoPressureStatus.RECLAIM_REQUIRED
                ),
                plan.required_bytes,
                remaining_shortfall_bytes=remaining,
                action=(
                    KVDuoPressureAction.SUCCESS
                    if remaining == 0
                    else KVDuoPressureAction.ERROR
                ),
                reason=(
                    KVDuoPressureReason.NONE
                    if remaining == 0
                    else KVDuoPressureReason.MEMORY_SHORTAGE
                ),
            )

        def reclaim_bytes(shortfall_bytes: int) -> int:
            pages = (
                shortfall_bytes + self._physical_page_bytes() - 1
            ) // self._physical_page_bytes()
            reclaimed_slots = self.reclaim_kvduo_full_pages(pages * self.page_size)
            # Full pages are always the first victims. Only after every legal
            # non-tail full victim has been consumed may pressure spill into
            # request/layer-private hot pages.
            if reclaimed_slots < pages * self.page_size:
                reclaimed_slots += self.reclaim_kvduo_hot_pages(
                    pages * self.page_size - reclaimed_slots
                )
            return reclaimed_slots // self.page_size * self._physical_page_bytes()

        return execute_kvduo_pressure_plan(
            plan,
            reclaim_full_pages=reclaim_bytes,
            available_bytes=lambda: self._physical_bytes(
                physical_allocator.available_size() // self.page_size * self.page_size
            ),
            # This path waits for outstanding backup before page selection, so
            # no pending DMA can independently make a failed plan progress.
            dma_can_make_progress=False,
        )

    @staticmethod
    def _require_allocation_ready(result: KVDuoPressureResult) -> None:
        if result.action is KVDuoPressureAction.SUCCESS:
            return
        raise RuntimeError(
            "KVDuo physical allocation cannot proceed: "
            f"action={result.action.name}, reason={result.reason.name}, "
            f"remaining_shortfall_bytes={result.remaining_shortfall_bytes}"
        )

    def _guard_kvduo_physical_allocation(self, physical_slots: int) -> bool:
        """Make an imminent decode allocation possible without partial state."""
        if physical_slots == 0:
            return True
        result = self._reclaim_for_physical_allocation(physical_slots)
        return result.action is KVDuoPressureAction.SUCCESS

    def ensure_kvduo_decode_capacity(self, requests) -> KVDuoPressureResult:
        """Reserve the exact main sparse-KV pages needed by ordinary decode.

        This is deliberately separate from the composite allocator's
        ``available_size``: logical slots, SWA, indexer, and compression state
        remain the responsibility of their own admission paths.
        """
        if not self.enable_mixed_residency:
            return self._reclaim_for_physical_allocation(0)
        pages = 0
        if self.is_dsv4_hisparse:
            page = self.page_size
            for req in requests:
                next_len = req.kv_committed_len + 1
                if next_len % self.compress_ratio != 0:
                    continue
                compressed_len = next_len // self.compress_ratio
                pages += int((compressed_len - 1) % page == 0)
        else:
            logical_page = self.token_to_kv_pool_allocator.page_size
            pages = sum(
                int(req.kv_committed_len % logical_page == 0) for req in requests
            )
        return self._reclaim_for_physical_allocation(pages * self.page_size)

    def alloc_device_buffer(self, req: Req) -> None:
        if self.is_dsv4_hisparse:
            allocated_len = req.extend_range.end
            alloc_size = self.padded_buffer_size
        else:
            allocated_len = req.kv.kv_allocated_len
            page_size = self.mem_pool_device.page_size
            # Allocate only enough for current tokens (page-aligned).
            # When prefill already fills device_buffer_size, include the reserved page.
            alloc_size = min(
                ((allocated_len + page_size - 1) // page_size) * page_size,
                self.device_buffer_size,
            )
            if alloc_size == self.device_buffer_size:
                alloc_size = self.padded_buffer_size
            if self.enable_mixed_residency:
                minimum = (self.top_k + page_size - 1) // page_size * page_size
                alloc_size = max(alloc_size, minimum)
                if alloc_size == self.device_buffer_size:
                    alloc_size = self.padded_buffer_size

        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.req_pool_idx, :allocated_len]
            )
        )
        compressed_len = len(compressed_logical_indices)
        if self.enable_mixed_residency:
            self.req_to_full_lookup[req.req_pool_idx, :compressed_len] = (
                compressed_logical_indices
            )
            prefix_indices = self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, : req.cache_protected_len
                ]
            )
            new_indices = compressed_logical_indices[len(prefix_indices) :]
            if len(new_indices):
                # Prefix attachment leaves historical generation/touch intact.
                # Only newly generated request-owned positions start a new
                # generation. Staging has already completed their host copy.
                self.full_touch_clock.add_(1)
                clock = self.full_touch_clock[0]
                self.full_generation[new_indices] += 1
                self.full_data_version[new_indices] = 1
                self.full_host_version[new_indices] = self.full_data_version[
                    new_indices
                ]
                self.full_last_touch[new_indices] = clock
            self._active_kvduo_reqs[req.req_pool_idx] = req
            # Staging materializes the complete prefill range. Appending to an
            # incomplete final page will lower this boundary to its page start.
            self._kvduo_host_valid_len[req.req_pool_idx] = compressed_len
            radix_page_size = self.token_to_kv_pool_allocator.page_size
            req.kvduo_radix_insert_len = (
                allocated_len // radix_page_size * radix_page_size
            )

            # A fully-resident KVDuo request does not own a speculative hot
            # buffer.  Physical hot pages are acquired by
            # ``_ensure_kvduo_hot_workset`` only after a real sparse selection
            # contains entries which cannot be resolved through the full-page
            # mapping.  In particular, do not demote/copy full KV while merely
            # admitting the request into decode.
            self.req_to_device_buffer[req.req_pool_idx, :] = 0
            self.req_device_buffer_size[req.req_pool_idx] = 0
            self.req_device_buffer_size_gpu[req.req_pool_idx] = 0
            self.kvduo_req_hot_capacity[:, req.req_pool_idx] = 0
            self.kvduo_req_hot_capacity_gpu[:, req.req_pool_idx] = 0
            self.kvduo_resolver_stats[:, req.req_pool_idx, :] = 0
            self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
            self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
            return

        preserve_indices = None
        if self.enable_mixed_residency:
            turnover_size = (
                self.page_size if alloc_size == self.padded_buffer_size else 0
            )
            pressure = self._reclaim_for_physical_allocation(
                alloc_size, turnover_size=turnover_size
            )
            self._require_allocation_ready(pressure)
            # Preserve exactly the mappings that survived page-level LRU. The
            # allocator obtains hot slots from the physical free list; no full
            # entry is copied into or tagged as hot during this transition.
            resident = self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_logical_indices
            ]
            preserve_indices = compressed_logical_indices[resident > 0]
        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            compressed_logical_indices, alloc_size, preserve_indices=preserve_indices
        )
        if buffer_indices is None:
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(compressed_len=%d, alloc_size=%d)",
                req.rid,
                compressed_len,
                alloc_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

        buffer_indices = buffer_indices.to(torch.int32)
        self.req_to_device_buffer[req.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.req_pool_idx] = alloc_size
        self.req_device_buffer_size_gpu[req.req_pool_idx] = min(
            alloc_size, self.device_buffer_size
        )

        if self.enable_mixed_residency:
            # Demotion never seeds hot entries: all first accesses to a
            # non-resident full page are real misses.
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = -1
        else:
            self.req_device_buffer_tokens[
                :, req.req_pool_idx, : self.device_buffer_size
            ] = self._device_buffer_arange_i32
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )

    def _grow_device_buffers(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Grow device buffers for requests whose sequence length exceeds current capacity."""
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]
        short_reqs_cpu = seq_lens_cpu <= self.device_buffer_size
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)

        if torch.any(needs_grow_cpu):
            page_size = self.mem_pool_device.page_size
            grow_indices = torch.where(needs_grow_cpu)[0]

            # Compute all grow sizes on CPU, then do a single bulk allocation
            req_idxs = []
            old_caps = []
            new_caps = []
            grow_sizes = []
            total_grow = 0
            turnover_grow = 0
            for i in grow_indices.tolist():
                req_idx = int(req_pool_indices_cpu[i])
                current_cap = int(current_caps[i])
                seq_len = int(seq_lens_cpu[i])

                new_cap = min(
                    ((seq_len + page_size - 1) // page_size) * page_size,
                    self.device_buffer_size,
                )
                if new_cap == self.device_buffer_size:
                    new_cap = self.padded_buffer_size
                grow_size = new_cap - current_cap
                if grow_size <= 0:
                    continue
                req_idxs.append(req_idx)
                old_caps.append(current_cap)
                new_caps.append(new_cap)
                grow_sizes.append(grow_size)
                total_grow += grow_size
                if current_cap <= self.device_buffer_size < new_cap:
                    turnover_grow += self.page_size

            if total_grow > 0:
                # ``new_cap`` includes the extra reserved append page when a
                # request reaches the configured hot-buffer capacity, so the
                # real allocation amount below is also the admission amount.
                pressure = self._reclaim_for_physical_allocation(
                    total_grow, turnover_size=turnover_grow
                )
                self._require_allocation_ready(pressure)
                all_new_indices = (
                    self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                        total_grow
                    )
                )
                if all_new_indices is None:
                    logger.error(
                        "HiSparse: _grow_device_buffers bulk alloc failed "
                        "(total_grow=%d)",
                        total_grow,
                    )
                    raise RuntimeError(
                        f"HiSparse _grow_device_buffers failed (total_grow={total_grow})"
                    )

                offset = 0
                for req_idx, current_cap, new_cap, grow_size in zip(
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]
                    offset += grow_size
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk
                    self.req_device_buffer_size[req_idx] = new_cap
                    self.req_device_buffer_size_gpu[req_idx] = min(
                        new_cap, self.device_buffer_size
                    )

        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]

    def _ensure_kvduo_hot_workset(
        self,
        req_pool_indices: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        allocate_mixed_minimum: bool = False,
        requested_capacities: dict[int, int] | None = None,
    ) -> None:
        """Materialize hot pages only for a selected non-full workset.

        The backing metadata has a fixed address for CUDA graph replay, but no
        physical KV slot is owned until this method sees a valid Top-k entry
        whose full mapping is absent.  Capacity grows by allocator pages and is
        owned independently by ``(request, layer)``.  A small carrier layer
        adapts the legacy cross-layer row allocator to the layer-first physical
        layout without allowing a future full-page allocation to alias a hot
        fragment.

        This is a control-plane allocation boundary.  It intentionally performs
        one batched device-to-host copy of the required capacities before a
        replay; the swap-in kernel itself never consumes an incomplete view.
        """
        if not self.enable_mixed_residency or req_pool_indices.numel() == 0:
            return

        valid = top_k_result >= 0
        safe_positions = top_k_result.clamp(min=0).to(torch.int64)
        logical = self.req_to_full_lookup[req_pool_indices[:, None], safe_positions]
        full_locs = self.mem_pool_device.full_to_hisparse_device_index_mapping[
            logical.clamp(min=0)
        ]
        non_full = valid & (logical >= 0) & (full_locs <= 0)
        req_cpu = req_pool_indices.to(device="cpu", dtype=torch.int64)

        page_size = self.page_size
        demand = []
        max_required = self.device_buffer_size
        for row, req_idx in enumerate(req_cpu.tolist()):
            current = int(self.kvduo_req_hot_capacity[layer_id, req_idx])
            hot_tokens = self.req_device_buffer_tokens[layer_id, req_idx, :current]
            occupied = int(torch.count_nonzero(hot_tokens >= 0).item())
            selected = top_k_result[row][non_full[row]]
            if current:
                cached = hot_tokens[hot_tokens >= 0]
                new_misses = int((~torch.isin(selected, cached)).sum().item())
            else:
                new_misses = int(selected.numel())
            required_slots = occupied + new_misses
            if allocate_mixed_minimum and self._mixed_slots[req_idx]:
                required_slots = max(required_slots, 2 * self.top_k)
            required_slots = max(
                required_slots,
                (requested_capacities or {}).get(req_idx, 0),
            )
            mandatory_slots = (
                2 * self.top_k
                if self._mixed_slots[req_idx] and current == 0
                else int(selected.numel())
            )
            demand.append((req_idx, current, required_slots, mandatory_slots))
            max_required = max(max_required, required_slots)

        if max_required > self.device_buffer_size:
            raise RuntimeError("KVDuo attention workset exceeds fixed 16K metadata")
        requests = []
        for req_idx, current, required_slots, _ in demand:
            if not required_slots:
                continue
            tiers = [
                ((multiplier * self.top_k + page_size - 1) // page_size) * page_size
                for multiplier in (2, 4, 8, 16)
            ]
            target = next((tier for tier in tiers if tier >= required_slots), tiers[-1])
            # Capacity changes one tier at a graph boundary. A failed growth is
            # harmless: the resolver continues with entry-LRU replacement.
            if current:
                target = min(
                    target, next((tier for tier in tiers if tier > current), tiers[-1])
                )
            if target <= current:
                continue
            grow = target - current
            requests.append((req_idx, current, target, grow))

        mandatory_targets = {
            req_idx: ((slots + page_size - 1) // page_size) * page_size
            for req_idx, _, _, slots in demand
            if slots > 0
        }
        self._materialize_kvduo_hot_growth(
            layer_id,
            requests,
            mandatory_targets,
            req_cpu.tolist(),
            logical[valid & (logical >= 0)].unique(),
        )

    def _ensure_kvduo_hot_capacity_targets(
        self, layer_id: int, requested_capacities: dict[int, int]
    ) -> None:
        """Materialize CPU-known capacity targets without inspecting GPU tags."""
        if not requested_capacities:
            return
        page_size = self.page_size
        tiers = [
            ((multiplier * self.top_k + page_size - 1) // page_size) * page_size
            for multiplier in (2, 4, 8, 16)
        ]
        requests = []
        mandatory_targets = {}
        for req_idx, required_slots in requested_capacities.items():
            if required_slots > self.device_buffer_size:
                raise RuntimeError("KVDuo hot target exceeds fixed 16K metadata")
            current = int(self.kvduo_req_hot_capacity[layer_id, req_idx])
            target = next((tier for tier in tiers if tier >= required_slots), tiers[-1])
            if current:
                target = min(
                    target, next((tier for tier in tiers if tier > current), tiers[-1])
                )
            if target <= current:
                continue
            requests.append((req_idx, current, target, target - current))
            if current == 0 and self._mixed_slots[req_idx]:
                mandatory_targets[req_idx] = tiers[0]
        self._materialize_kvduo_hot_growth(
            layer_id,
            requests,
            mandatory_targets,
            requested_capacities,
            None,
        )

    def _materialize_kvduo_hot_growth(
        self,
        layer_id: int,
        requests: list[tuple[int, int, int, int]],
        mandatory_targets: dict[int, int],
        protected_req_indices,
        protected_logical,
    ) -> None:
        """Allocate layer-hot pages for precomputed targets without tag readback."""
        if not requests:
            return
        page_size = self.page_size
        total_grow = sum(grow for _, _, _, grow in requests)
        pages_needed = total_grow // page_size
        free_pages = self._kvduo_free_layer_pages[layer_id]
        missing_carriers = max(0, pages_needed - len(free_pages))
        if missing_carriers:
            carrier_slots = missing_carriers * page_size
            self._kvduo_pressure_protected = protected_logical
            self._kvduo_hot_pressure_protected = {
                (layer_id, int(req_idx), page_index)
                for req_idx in protected_req_indices
                for page_index in range(
                    len(self._kvduo_req_layer_pages.get((layer_id, int(req_idx)), ()))
                )
            }
            reserved_free_starts = sorted(free_pages)[
                : min(pages_needed, len(free_pages))
            ]
            for start in reserved_free_starts:
                for owner_layer, owner_req in enumerate(
                    self._kvduo_hot_carriers[start]
                ):
                    if owner_req is None:
                        continue
                    owner_pages = self._kvduo_req_layer_pages.get(
                        (owner_layer, int(owner_req)), ()
                    )
                    if start in owner_pages:
                        self._kvduo_hot_pressure_protected.add(
                            (owner_layer, int(owner_req), owner_pages.index(start))
                        )
            try:
                pressure = self._reclaim_for_physical_allocation(carrier_slots)
                if pressure.action is not KVDuoPressureAction.SUCCESS:
                    requests = [
                        (req_idx, current, target, target - current)
                        for req_idx, target in mandatory_targets.items()
                        if (
                            current := int(
                                self.kvduo_req_hot_capacity[layer_id, req_idx]
                            )
                        )
                        < target
                    ]
                    if not requests:
                        return
                    total_grow = sum(grow for _, _, _, grow in requests)
                    pages_needed = total_grow // page_size
                    missing_carriers = max(0, pages_needed - len(free_pages))
                    carrier_slots = missing_carriers * page_size
                    pressure = self._reclaim_for_physical_allocation(carrier_slots)
                    self._require_allocation_ready(pressure)
                missing_carriers = max(0, pages_needed - len(free_pages))
                carrier_slots = missing_carriers * page_size
                if carrier_slots:
                    pressure = self._reclaim_for_physical_allocation(carrier_slots)
                    self._require_allocation_ready(pressure)
                physical = (
                    self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                        carrier_slots
                    )
                    if carrier_slots
                    else torch.empty(0, dtype=torch.int64, device=self.device)
                )
            finally:
                self._kvduo_pressure_protected = None
                self._kvduo_hot_pressure_protected = set()
            if physical is None:
                raise RuntimeError(
                    "KVDuo layer-page carrier allocation failed after pressure plan"
                )
            for carrier in physical.view(-1, page_size):
                start = int(carrier[0])
                if start in self._kvduo_hot_carriers:
                    raise RuntimeError("KVDuo hot carrier aliases existing ownership")
                self._kvduo_hot_carriers[start] = [
                    None
                ] * self.mem_pool_device.layer_num
                for domain in range(self.mem_pool_device.layer_num):
                    self._kvduo_free_layer_pages[domain].add(start)

        for req_idx, current, target, grow in requests:
            chunks = []
            for _ in range(grow // page_size):
                start = min(self._kvduo_free_layer_pages[layer_id])
                self._kvduo_free_layer_pages[layer_id].remove(start)
                owners = self._kvduo_hot_carriers[start]
                if owners[layer_id] is not None:
                    raise RuntimeError("KVDuo layer page is already owned")
                owners[layer_id] = req_idx
                self._kvduo_req_layer_pages.setdefault((layer_id, req_idx), []).append(
                    start
                )
                chunks.append(
                    torch.arange(
                        start,
                        start + page_size,
                        dtype=torch.int64,
                        device=self.device,
                    )
                )
            page_locs = torch.cat(chunks)
            self.req_device_buffer_token_locs[layer_id, req_idx, current:target] = (
                page_locs.to(torch.int32)
            )
            self.req_device_buffer_tokens[layer_id, req_idx, current:target] = -1
            old_lru = self.lru_slots[layer_id, req_idx, :current].clone()
            self.lru_slots[layer_id, req_idx, :target] = torch.cat(
                [
                    torch.arange(
                        current,
                        target,
                        dtype=torch.int16,
                        device=self.device,
                    ),
                    old_lru,
                ]
            )
            self.kvduo_req_hot_capacity[layer_id, req_idx] = target
            self.kvduo_req_hot_capacity_gpu[layer_id, req_idx] = target

    def _consume_kvduo_stats_snapshot(self):
        """Return a completed asynchronous statistics snapshot, if available."""
        if not self._kvduo_stats_pending or not self._kvduo_stats_event.query():
            return None, None
        self._kvduo_stats_pending = False
        owners = self._kvduo_stats_pending_owners
        self._kvduo_stats_pending_owners = {}
        return self._kvduo_stats_host, owners

    def _schedule_kvduo_stats_snapshot(self) -> None:
        """Stage cumulative resolver counters without blocking the CPU.

        The producer stream first snapshots and resets the live counters. D2H
        then runs on a side stream from the fixed GPU snapshot, so the following
        replay may update the live bank without waiting for host transfer.
        """
        self._kvduo_stats_pending_owners = {
            req_idx: req for req_idx, req in self._active_kvduo_reqs.items()
        }
        self._kvduo_stats_snapshot.copy_(self.kvduo_resolver_stats)
        self.kvduo_resolver_stats.zero_()
        self._kvduo_stats_snapshot_ready_event.record()
        with device_module.stream(self._kvduo_stats_stream):
            self._kvduo_stats_stream.wait_event(self._kvduo_stats_snapshot_ready_event)
            self._kvduo_stats_host.copy_(self._kvduo_stats_snapshot, non_blocking=True)
            self._kvduo_stats_event.record()
        self._kvduo_stats_pending = True

    def prepare_kvduo_graph_replay(
        self,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Perform physical capacity management at the graph replay boundary.

        This is intentionally host-side and must run only after the preceding
        producer stream is complete.  Fully resident requests retain capacity
        zero; newly mixed requests atomically acquire their page-aligned 2K
        tier before the next resolver replay.
        """
        if not self.enable_mixed_residency or req_pool_indices.numel() == 0:
            return
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        # Unit tests construct a minimal coordinator with ``__new__``. Keep that
        # path synchronous while production GPU coordinators use the pre-created
        # pinned snapshot and side stream.
        if not hasattr(self, "_kvduo_replay_count"):
            self.kvduo_stats_poll_interval = KVDUO_STATS_POLL_INTERVAL
            self._kvduo_replay_count = 0
        self._kvduo_replay_count += 1
        poll_due = self._kvduo_replay_count % self.kvduo_stats_poll_interval == 0
        if self.kvduo_resolver_stats.device.type == "cpu":
            stats_cpu = None
            stats_owners = None
            if poll_due:
                stats_cpu = self.kvduo_resolver_stats.clone()
                self.kvduo_resolver_stats.zero_()
                stats_owners = dict(self._active_kvduo_reqs)
        else:
            stats_cpu, stats_owners = self._consume_kvduo_stats_snapshot()
            if poll_due and not self._kvduo_stats_pending and stats_cpu is None:
                self._schedule_kvduo_stats_snapshot()

        if req_pool_indices_cpu is None:
            # Compatibility fallback for direct/unit-test callers. Runtime paths
            # pass ScheduleBatch's existing CPU mirror and avoid this D2H read.
            req_pool_indices_cpu = req_pool_indices.to(device="cpu", dtype=torch.int64)
        requested = set(
            req_pool_indices_cpu.tolist()
            if isinstance(req_pool_indices_cpu, torch.Tensor)
            else req_pool_indices_cpu
        )
        requested.update(
            req_idx for req_idx in self._active_kvduo_reqs if self._mixed_slots[req_idx]
        )

        initial_mixed = {req_idx for req_idx in requested if self._mixed_slots[req_idx]}
        optional_growth = {
            (layer_id, req_idx): min(
                int(self.kvduo_req_hot_capacity[layer_id, req_idx]) * 2,
                16 * self.top_k,
            )
            for req_idx in initial_mixed
            for layer_id in range(self.mem_pool_device.layer_num)
            if stats_cpu is not None
            and stats_owners.get(req_idx) is self._active_kvduo_reqs.get(req_idx)
            and int(self.kvduo_req_hot_capacity[layer_id, req_idx]) > 0
            and int(stats_cpu[layer_id, req_idx, 0]) > 0
        }

        # Pressure reclamation can demote another active request. Iterate to a
        # capacity fixed point so every mixed request has 2K in every storage
        # group. Historical optional growth is consumed on the first pass only.
        pending = initial_mixed
        while True:
            mixed = sorted(pending)
            if not mixed:
                return
            before = {
                (layer_id, req_idx): int(self.kvduo_req_hot_capacity[layer_id, req_idx])
                for req_idx in mixed
                for layer_id in range(self.mem_pool_device.layer_num)
            }
            for layer_id in range(self.mem_pool_device.layer_num):
                targets = {}
                for req_idx in mixed:
                    current = int(self.kvduo_req_hot_capacity[layer_id, req_idx])
                    if current == 0:
                        targets[req_idx] = 2 * self.top_k
                    elif (layer_id, req_idx) in optional_growth:
                        targets[req_idx] = optional_growth[layer_id, req_idx]
                if targets:
                    self._ensure_kvduo_hot_capacity_targets(layer_id, targets)
            optional_growth.clear()
            requested.update(
                req_idx
                for req_idx in self._active_kvduo_reqs
                if self._mixed_slots[req_idx]
            )
            pending = {
                req_idx
                for req_idx in requested
                if self._mixed_slots[req_idx]
                and any(
                    int(self.kvduo_req_hot_capacity[layer_id, req_idx]) < 2 * self.top_k
                    for layer_id in range(self.mem_pool_device.layer_num)
                )
            }
            if not pending:
                break
            made_progress = any(
                int(self.kvduo_req_hot_capacity[layer_id, req_idx])
                > before.get((layer_id, req_idx), -1)
                for req_idx in pending
                for layer_id in range(self.mem_pool_device.layer_num)
            )
            if not made_progress:
                raise RuntimeError(
                    "KVDuo mixed request lacks mandatory 2K capacity before replay"
                )

        for req_idx in requested:
            if not self._mixed_slots[req_idx]:
                continue
            for layer_id in range(self.mem_pool_device.layer_num):
                if int(self.kvduo_req_hot_capacity[layer_id, req_idx]) < 2 * self.top_k:
                    raise RuntimeError(
                        "KVDuo mixed request lacks mandatory 2K capacity before replay"
                    )

    def _grow_kvduo_hot_metadata(self, required_slots: int) -> None:
        """Validate against the immutable, capture-stable resolver metadata."""
        if required_slots > self.device_buffer_size:
            raise RuntimeError("KVDuo hot metadata cannot grow beyond fixed 16K")

    def _release_kvduo_layer_hot_pages(self, req_idx: int) -> None:
        """Release request-private hot pages at single-layer page granularity."""
        if not self.enable_mixed_residency:
            return
        coalesce = set()
        for layer_id in range(self.mem_pool_device.layer_num):
            for start in self._kvduo_req_layer_pages.pop((layer_id, req_idx), []):
                owners = self._kvduo_hot_carriers[start]
                if owners[layer_id] != req_idx:
                    raise RuntimeError("KVDuo layer-page ownership mismatch")
                owners[layer_id] = None
                self._kvduo_free_layer_pages[layer_id].add(start)
                if all(owner is None for owner in owners):
                    coalesce.add(start)
            self.kvduo_req_hot_capacity[layer_id, req_idx] = 0
            self.kvduo_req_hot_capacity_gpu[layer_id, req_idx] = 0

        # Returning a carrier to the legacy allocator is safe only when no
        # layer fragment remains owned.  Until then, other requests may reuse
        # each free layer fragment independently.
        if coalesce:
            indices = []
            for start in sorted(coalesce):
                for layer_pages in self._kvduo_free_layer_pages:
                    layer_pages.remove(start)
                del self._kvduo_hot_carriers[start]
                indices.extend(range(start, start + self.page_size))
            self.token_to_kv_pool_allocator.free_hisparse_indices(
                torch.tensor(indices, dtype=torch.int64, device=self.device)
            )

    def _evict_kvduo_hot_fragment(
        self, layer_id: int, req_idx: int, page_index: int
    ) -> None:
        """Drop one layer page without moving any KV payload."""
        pages = self._kvduo_req_layer_pages[(layer_id, req_idx)]
        last_index = len(pages) - 1
        victim_start = pages[page_index]
        slot = page_index * self.page_size
        last_slot = last_index * self.page_size
        if page_index != last_index:
            # Only page-table metadata moves. The surviving KV payload stays at
            # its original physical addresses carried by token_locs.
            self.req_device_buffer_tokens[
                layer_id, req_idx, slot : slot + self.page_size
            ] = self.req_device_buffer_tokens[
                layer_id, req_idx, last_slot : last_slot + self.page_size
            ].clone()
            self.req_device_buffer_token_locs[
                layer_id, req_idx, slot : slot + self.page_size
            ] = self.req_device_buffer_token_locs[
                layer_id, req_idx, last_slot : last_slot + self.page_size
            ].clone()
            self.hot_page_last_touch[layer_id, req_idx, page_index] = (
                self.hot_page_last_touch[layer_id, req_idx, last_index]
            )
            pages[page_index] = pages[last_index]
        self.req_device_buffer_tokens[
            layer_id, req_idx, last_slot : last_slot + self.page_size
        ] = -1
        self.req_device_buffer_token_locs[
            layer_id, req_idx, last_slot : last_slot + self.page_size
        ] = -1
        self.hot_page_last_touch[layer_id, req_idx, last_index] = 0
        pages.pop()
        if not pages:
            del self._kvduo_req_layer_pages[(layer_id, req_idx)]
        owners = self._kvduo_hot_carriers[victim_start]
        owners[layer_id] = None
        self._kvduo_free_layer_pages[layer_id].add(victim_start)
        new_capacity = last_slot
        self.kvduo_req_hot_capacity[layer_id, req_idx] = new_capacity
        self.kvduo_req_hot_capacity_gpu[layer_id, req_idx] = new_capacity
        # Rebuild a deterministic permutation after metadata page removal;
        # payload compaction is deliberately not performed. Empty slots are
        # kept at the LRU/front side, which is the resolver's empty-first
        # allocation contract.
        tokens = self.req_device_buffer_tokens[layer_id, req_idx, :new_capacity]
        slots = torch.arange(new_capacity, dtype=torch.int16, device=self.device)
        self.lru_slots[layer_id, req_idx, :new_capacity] = torch.cat(
            [slots[tokens < 0], slots[tokens >= 0]]
        )

    def reclaim_kvduo_hot_pages(self, num_tokens: int) -> int:
        """Evict layer-hot pages by LRU after full-page reclaim is exhausted."""
        if not self.enable_mixed_residency or num_tokens <= 0:
            return 0
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        protected = getattr(self, "_kvduo_hot_pressure_protected", set())
        candidates = []
        for (layer_id, req_idx), pages in self._kvduo_req_layer_pages.items():
            minimum_pages = (2 * self.top_k + self.page_size - 1) // self.page_size
            for page_index, start in enumerate(pages):
                if len(pages) <= minimum_pages:
                    continue
                if (layer_id, req_idx, page_index) in protected:
                    continue
                candidates.append(
                    (
                        int(self.hot_page_last_touch[layer_id, req_idx, page_index]),
                        layer_id,
                        req_idx,
                        page_index,
                        start,
                    )
                )
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[4]))
        reclaimed = 0
        # Indices change after removing a non-final page; locate each physical
        # fragment again immediately before eviction.
        for _, layer_id, req_idx, _, start in candidates:
            pages = self._kvduo_req_layer_pages.get((layer_id, req_idx))
            if pages is None or start not in pages:
                continue
            minimum_pages = (2 * self.top_k + self.page_size - 1) // self.page_size
            if len(pages) <= minimum_pages:
                continue
            self._evict_kvduo_hot_fragment(layer_id, req_idx, pages.index(start))
            owners = self._kvduo_hot_carriers[start]
            if all(owner is None for owner in owners):
                for layer_pages in self._kvduo_free_layer_pages:
                    layer_pages.remove(start)
                del self._kvduo_hot_carriers[start]
                self.token_to_kv_pool_allocator.free_hisparse_indices(
                    torch.arange(
                        start,
                        start + self.page_size,
                        dtype=torch.int64,
                        device=self.device,
                    )
                )
                reclaimed += self.page_size
                if reclaimed >= num_tokens:
                    break
        return reclaimed

    def has_ongoing_staging(self) -> bool:
        return len(self.ack_staging_queue) > 0

    def collect_ready_reqs(self) -> List[Req]:
        ready_reqs: List[Req] = []
        if len(self.ack_staging_queue) == 0:
            return ready_reqs

        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            # prepare device buffer and update req
            self.alloc_device_buffer(req)
            self._skip_first_backup[req.req_pool_idx] = True
            req.hisparse_staging = False
            finish_count -= 1
            ready_reqs.append(req)
        return ready_reqs

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        if self.enable_mixed_residency:
            self._backup_kvduo_sealed_pages(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
        else:
            self._eager_backup_previous_token(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )

        if not self.is_dsv4_hisparse:
            if self.enable_mixed_residency:
                # KVDuo appends into the allocator-provided full tail page.
                # A temporary hot append slot must not be allocated or exposed
                # as a durable full mapping.
                compressed_locs = (
                    self.token_to_kv_pool_allocator.get_last_loc_compressed(
                        out_cache_loc
                    )
                )
                positions = seq_lens - 1
                self.req_to_full_lookup[req_pool_indices, positions] = compressed_locs
                self._publish_kvduo_model_write(compressed_locs)
                return
            # Grow device buffers if needed and resolve the latest-token slot.
            reserved_buffer_loc = self._grow_device_buffers(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)

            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
                out_cache_loc
            )
            # ROCm: the decode remap creates a temporary hisparse device slot per
            # new token (via the page_size==1 allocator path). Free the stale
            # slot before pointing the mapping at the reserved device-buffer slot,
            # otherwise the temporary slots leak and corrupt later swap-in lookups.
            # CUDA keeps the original behavior: the swap-in kernel consumes only
            # top_k_device_locs, so stale mapping entries are harmless there.
            if _is_hip:
                previous_locs = self.mem_pool_device._translate_loc_to_hisparse_device(
                    compressed_locs
                )
                stale_locs = previous_locs[
                    (previous_locs > 0) & (previous_locs != reserved_buffer_loc)
                ]
                if stale_locs.numel() > 0:
                    self.token_to_kv_pool_allocator.free_hisparse_indices(stale_locs)

            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = reserved_buffer_loc
            return

        active_reqs = seq_lens % self.compress_ratio == 0
        if not torch.any(active_reqs):
            return

        active_seq_lens = seq_lens[active_reqs]
        active_out_cache_loc = out_cache_loc[active_reqs]
        active_req_pool_indices = req_pool_indices[active_reqs]

        compressed_seq_lens = active_seq_lens // self.compress_ratio
        if self.enable_mixed_residency:
            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
                active_out_cache_loc
            )
            positions = compressed_seq_lens - 1
            self.req_to_full_lookup[active_req_pool_indices, positions] = (
                compressed_locs
            )
            self._publish_kvduo_model_write(compressed_locs)
            return

        reserved_positions = (compressed_seq_lens - 1).clamp(
            max=self.device_buffer_size
        )
        reserved_buffer_loc = self.req_to_device_buffer[
            active_req_pool_indices, reserved_positions
        ]

        self.req_device_buffer_token_locs[
            :, active_req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)

        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            active_out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            reserved_buffer_loc
        )

    def _backup_kvduo_sealed_pages(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Write back a sealed page when the next real tail page starts.

        The compressed-position calculation is the adapter boundary for DSV4.
        Copying does not update model touch clocks. Tail pages remain resident
        and protected; completion only makes an older page legally reclaimable.
        """
        page_jobs = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            compressed_len = self.host_token_len(int(seq_lens_cpu[i]))
            if compressed_len <= 0:
                continue
            current_pos = compressed_len - 1
            page_start = current_pos // self.page_size * self.page_size
            # A write to a partial page invalidates the previously staged host
            # suffix. The page is written back only after it becomes sealed.
            self._kvduo_host_valid_len[req_idx] = min(
                self._kvduo_host_valid_len[req_idx], page_start
            )
            if current_pos % self.page_size != 0 or page_start == 0:
                continue
            sealed_start = page_start - self.page_size
            sealed_end = page_start
            if self._kvduo_host_valid_len[req_idx] >= sealed_end:
                continue
            page_jobs.append((req_idx, sealed_start, sealed_end))

        if not page_jobs:
            return
        self.wait_for_pending_backup()
        host_locs = []
        logical_locs = []
        for req_idx, start, end in page_jobs:
            host_locs.append(
                self.mem_pool_host.alloc_paged_token_slots(
                    self.req_to_host_pool,
                    self.req_to_host_pool_allocated_len,
                    req_idx,
                    start,
                    end - start,
                )
            )
            logical = self.req_to_full_lookup[req_idx, start:end]
            logical_locs.append(logical)

        host_locs = torch.cat(host_locs)
        logical_locs = torch.cat(logical_locs)
        device_locs = self.mem_pool_device.full_to_hisparse_device_index_mapping[
            logical_locs
        ]
        # Page turnover is a control-plane boundary. Validate the entire batch
        # with one device synchronization, never one sync per candidate page.
        valid = torch.all(logical_locs >= 0) & torch.all(device_locs > 0)
        if not bool(valid.item()):
            raise RuntimeError("KVDuo sealed page lost full residency before backup")
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._pending_kvduo_host_valid.extend(
            (req_idx, end, logical, versions)
            for (req_idx, _, end), logical, versions in zip(
                page_jobs,
                logical_locs.split(self.page_size),
                self.full_data_version[logical_locs].clone().split(self.page_size),
            )
        )
        self._has_pending_backup = True

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Back up the previous compressed token to host memory.

        Each newly produced compressed token (one per `compress_ratio` decode
        steps) must be backed up to host so the swap-in kernel can later
        recover it.

        Two cases are skipped:
        - The first decode step right after staging: all prefill tokens were
          already backed up during staging, so there is nothing new to save.
        - Steps where `(seq_len - 1) % compress_ratio != 0`: no new compressed
          token was produced this step.
        """
        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up),
        # and skip non-aligned steps that did not produce a new compressed token.
        backup_indices = []
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                self._skip_first_backup[req_idx] = False
                continue
            if (int(seq_lens_cpu[i]) - 1) % self.compress_ratio == 0:
                backup_indices.append(i)

        if not backup_indices:
            return

        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device=self.device
        )
        backup_req_indices = req_pool_indices[backup_indices_gpu]

        # The previous compressed token's position and its device buffer slot:
        #  compressed_pos = (seq_len - 1) // compress_ratio - 1
        #  - short: slot = compressed_pos          (within the regular buffer)
        #  - long:  slot = device_buffer_size      (the reserved slot)
        prev_seq_lens = seq_lens[backup_indices_gpu] - 1
        compressed_prev_seq_lens = prev_seq_lens // self.compress_ratio
        actual_compressed_pos = compressed_prev_seq_lens - 1

        buffer_slot = actual_compressed_pos.clamp(max=self.device_buffer_size)

        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]

        host_locs_list = []
        for i in backup_indices:
            req_idx = int(req_pool_indices_cpu[i])
            start_pos = (int(seq_lens_cpu[i]) - 1) // self.compress_ratio - 1
            host_locs = self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                req_idx,
                start_pos,
                1,
            )
            host_locs_list.append(host_locs)
        host_locs = torch.cat(host_locs_list)

        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
            if actual_compressed_pos.is_cuda:
                actual_compressed_pos.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    def wait_for_pending_backup(self) -> None:
        if not self._has_pending_backup:
            return
        self._backup_done_event.wait(device_module.current_stream())
        self._has_pending_backup = False
        if self.enable_mixed_residency:
            for (
                req_idx,
                valid_len,
                logical,
                copied_versions,
            ) in self._pending_kvduo_host_valid:
                # Publish the captured version, not whatever version is current
                # when the asynchronous transfer completes.
                self.full_host_version[logical] = copied_versions
                self._kvduo_host_valid_len[req_idx] = max(
                    self._kvduo_host_valid_len[req_idx], valid_len
                )
            self._pending_kvduo_host_valid.clear()

    def _publish_kvduo_model_write(self, logical_locs: torch.Tensor) -> None:
        """Publish a newly generated KV identity in the unified address space."""
        self.full_touch_clock.add_(1)
        self.full_generation[logical_locs] += 1
        self.full_data_version[logical_locs] = 1
        self.full_host_version[logical_locs] = -1
        self.full_last_touch[logical_locs] = self.full_touch_clock[0]

    def record_kvduo_model_update(self, logical_locs: torch.Tensor) -> None:
        """Record an in-place model KV update without changing its generation."""
        if not self.enable_mixed_residency:
            return
        self.full_touch_clock.add_(1)
        self.full_data_version[logical_locs] += 1
        self.full_host_version[logical_locs] = -1
        self.full_last_touch[logical_locs] = self.full_touch_clock[0]

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA kernel) instead.

        Note: dsv4 hisparse is not supported — DeepSeekV4SingleKVPoolHost has no
        load_to_device_per_layer and indices live in compressed space. Currently
        only used as a kernel oracle in test_hisparse_unit.py (non-dsv4 path).

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
        """
        assert not self.is_dsv4_hisparse, (
            "naive_load_topk is not implemented for dsv4 hisparse"
        )
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)

            assert torch.all(selected_tokens >= 0), (
                f"Req {req_idx}: selected tokens contain negative positions"
            )
            assert torch.all(selected_tokens < seq_len), (
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                device_indices = torch.empty(
                    top_n, dtype=torch.int64, device=self.device
                )

                is_latest_token = selected_tokens == (seq_len - 1)
                needs_host_load = ~is_latest_token

                device_indices[is_latest_token] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]

                num_to_load = int(needs_host_load.sum().item())
                if num_to_load > 0:
                    tokens_to_load = selected_tokens[needs_host_load]
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]

                    invalid_mask = host_locs < 0
                    if torch.any(invalid_mask):
                        bad_positions = tokens_to_load[invalid_mask].tolist()
                        raise AssertionError(
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]
                    device_indices[needs_host_load] = buffer_locs

                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)

        return top_k_indices

    def abort_staging_request(self, req: Req) -> None:
        """Remove a request from the staging queue and free its host + device resources.

        Must be called when aborting a request that has been admitted into staging
        but has not yet completed (i.e. req.hisparse_staging is True).
        """
        # Remove from staging queue
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        # Wait for any in-flight staging DMA to complete before freeing
        self.write_staging_stream.synchronize()

        prefill_len = req.extend_range.end
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefill_len
        ]
        # The incoming prefix is owned by RadixTree (and may be shared by other
        # active requests). Only the request-private suffix may be returned here.
        owned_locs = allocated_locs[req.cache_protected_len :]
        self.token_to_kv_pool_allocator.free_hisparse(owned_locs)

        # Free host memory that was allocated during admit_request_into_staging
        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if self.enable_mixed_residency and host_indices.numel() > 0:
            shared = self._host_record_locations_for_request(req)
            if shared:
                shared_tensor = torch.tensor(
                    sorted(shared), dtype=torch.int64, device=host_indices.device
                )
                host_indices = host_indices[~torch.isin(host_indices, shared_tensor)]
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self._release_host_prefix_refs(req)
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        if self.enable_mixed_residency:
            self.req_to_full_lookup[req.req_pool_idx, :] = -1
            self.req_reserved_logical[req.req_pool_idx] = -1
            self._kvduo_host_valid_len[req.req_pool_idx] = 0
            self.kvduo_resolver_stats[:, req.req_pool_idx, :] = 0
        self._mixed_slots[req.req_pool_idx] = False
        if self.enable_mixed_residency:
            self._active_kvduo_reqs.pop(req.req_pool_idx, None)
        self._skip_first_backup[req.req_pool_idx] = False
        req.hisparse_staging = False

    def retract_req(self, req: Req) -> None:
        if req.hisparse_staging:
            self.abort_staging_request(req)
        else:
            self.request_finished(req)

    def request_finished(self, req: Req):
        # release resources only after the execution of a potential overlapped batch
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()

        # Use kv_allocated_len (not seqlen): under speculative decoding the
        # allocator can over-allocate beyond the committed seqlen, and those
        # extra slots may carry stale mapping entries pointing at buffer slots
        # we just freed via free_hisparse_indices(all_hi). If left set, the
        # subsequent release_kv_cache -> allocator.free -> free_hisparse path
        # re-frees them (double-free into the page allocator's free list).
        allocated_len = req.kv.kv_allocated_len

        # release memory -- only free actually-allocated buffer indices
        current_cap = int(self.req_device_buffer_size[req.req_pool_idx])
        all_hi = torch.empty(0, dtype=torch.int64, device=self.device)
        if self.enable_mixed_residency:
            self._release_kvduo_layer_hot_pages(req.req_pool_idx)
        elif current_cap > 0:
            side_buf_hi = self.req_to_device_buffer[req.req_pool_idx, :current_cap]
            all_hi = torch.unique(side_buf_hi[side_buf_hi > 0])
            if all_hi.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(all_hi)

        allocated_locs = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :allocated_len
        ]
        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            allocated_locs
        )
        is_mixed = self.enable_mixed_residency and self._mixed_slots[req.req_pool_idx]
        if is_mixed:
            req.kvduo_mixed_residency = True
            # RadixCache cannot represent host-only holes. Free every surviving
            # request-owned fragment beyond kvduo_radix_insert_len; the cache
            # implementations retain only the already-contiguous full prefix.
            release_start = max(
                req.cache_protected_len,
                getattr(req, "kvduo_radix_insert_len", req.cache_protected_len),
            )
            owned_full_locs = allocated_locs[release_start:]
            owned_compressed_locs = (
                self.mem_pool_device.translate_loc_from_full_to_compressed(
                    owned_full_locs
                )
            )
            full_hi = self.mem_pool_device.full_to_hisparse_device_index_mapping[
                owned_compressed_locs
            ]
            full_hi = torch.unique(full_hi[full_hi > 0])
            if all_hi.numel() > 0:
                full_hi = full_hi[~torch.isin(full_hi, all_hi)]
            if full_hi.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(full_hi)
        if is_mixed:
            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                owned_compressed_locs
            ] = 0
        elif not self.enable_mixed_residency:
            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = 0
        elif self.req_reserved_logical[req.req_pool_idx] >= 0:
            # A decode append slot is transient, not a sealed full page. Keep
            # the prefill full mappings for Radix insertion, but invalidate the
            # final append owner before its private buffer is returned.
            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                self.req_reserved_logical[req.req_pool_idx]
            ] = 0

        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if self.enable_mixed_residency:
            retained = self._retain_request_host_prefix(req)
            self._release_host_prefix_refs(req)
            if retained and host_indices.numel() > 0:
                retained_tensor = torch.tensor(
                    sorted(retained), dtype=torch.int64, device=host_indices.device
                )
                host_indices = host_indices[~torch.isin(host_indices, retained_tensor)]
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        # clear req info
        self.req_device_buffer_tokens[:, req.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.req_pool_idx] = 0
        self.req_device_buffer_size_gpu[req.req_pool_idx] = 0
        if self.enable_mixed_residency:
            self.hot_page_last_touch[:, req.req_pool_idx, :] = 0
            self.kvduo_resolver_stats[:, req.req_pool_idx, :] = 0
        self.req_to_host_pool[req.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.req_pool_idx] = 0
        if self.enable_mixed_residency:
            self.req_to_full_lookup[req.req_pool_idx, :] = -1
            self.req_reserved_logical[req.req_pool_idx] = -1
            self._kvduo_host_valid_len[req.req_pool_idx] = 0
        self.lru_slots[:, req.req_pool_idx, :].copy_(self._lru_init)
        self._skip_first_backup[req.req_pool_idx] = False
        self._mixed_slots[req.req_pool_idx] = False
        if self.enable_mixed_residency:
            self._active_kvduo_reqs.pop(req.req_pool_idx, None)

    def reclaim_kvduo_full_pages(self, num_tokens: int) -> int:
        """Demote cold request-owned full pages to satisfy decode pressure.

        The host copy is complete before requests leave staging. This control
        path may synchronize only when allocation pressure occurs; steady-state
        full/hot/host resolution remains entirely device-driven and graph-safe.
        """
        if not self.enable_mixed_residency or num_tokens <= 0:
            return 0
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()
        candidate_pages = []
        candidate_meta = []
        for req_idx, req in tuple(self._active_kvduo_reqs.items()):
            allocated_len = req.kv.kv_allocated_len
            compressed = self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req_idx, :allocated_len]
            )
            prefix_len = len(
                self.mem_pool_device.translate_loc_from_full_to_compressed(
                    self.req_to_token_pool.req_to_token[
                        req_idx, : req.cache_protected_len
                    ]
                )
            )
            num_pages = (len(compressed) + self.page_size - 1) // self.page_size
            tail_first = max(0, num_pages - self.tail_protected_pages)
            first_owned_page = (prefix_len + self.page_size - 1) // self.page_size
            start = first_owned_page * self.page_size
            host_valid_end = (
                self._kvduo_host_valid_len[req_idx] // self.page_size * self.page_size
            )
            end = min(tail_first * self.page_size, host_valid_end)
            if end <= start:
                continue
            pages = compressed[start:end].view(-1, self.page_size)
            candidate_pages.append(pages)
            candidate_meta.extend(
                (req_idx, offset) for offset in range(start, end, self.page_size)
            )

        if not candidate_pages:
            return 0
        logical_pages = torch.cat(candidate_pages, dim=0)
        physical_pages = self.mem_pool_device.full_to_hisparse_device_index_mapping[
            logical_pages
        ]
        valid = torch.all(physical_pages > 0, dim=1)
        pressure_protected = getattr(self, "_kvduo_pressure_protected", None)
        if pressure_protected is not None:
            valid &= ~torch.any(torch.isin(logical_pages, pressure_protected), dim=1)
        host_current = torch.all(
            self.full_host_version[logical_pages]
            == self.full_data_version[logical_pages],
            dim=1,
        )
        valid &= host_current
        page_touch = torch.max(self.full_last_touch[logical_pages], dim=1).values
        # Pressure handling is a control-plane barrier. Transfer the compact
        # page summaries once, never one synchronization per candidate page.
        summaries = torch.stack([page_touch, valid.to(torch.int64)], dim=1).cpu()
        order = sorted(
            (i for i in range(len(candidate_meta)) if summaries[i, 1]),
            key=lambda i: (
                int(summaries[i, 0]),
                candidate_meta[i][0],
                candidate_meta[i][1],
            ),
        )

        reclaimed = 0
        selected = []
        for i in order:
            if reclaimed >= num_tokens:
                break
            selected.append(i)
            reclaimed += self.page_size

        if not selected:
            return 0
        selected_gpu = torch.tensor(selected, dtype=torch.int64, device=self.device)
        selected_logical = logical_pages[selected_gpu]
        selected_physical = physical_pages[selected_gpu]
        self.mem_pool_device.full_to_hisparse_device_index_mapping[
            selected_logical.flatten()
        ] = 0
        self.token_to_kv_pool_allocator.free_hisparse_indices(
            selected_physical.flatten()
        )
        for i in selected:
            req_idx, start = candidate_meta[i]
            req = self._active_kvduo_reqs[req_idx]
            req.kvduo_mixed_residency = True
            req.kvduo_radix_insert_len = min(
                req.kvduo_radix_insert_len, start * self.compress_ratio
            )
            self._mixed_slots[req_idx] = True
        return reclaimed

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices."""
        num_reqs = req_pool_indices.size(0)

        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]

        swap_in_fn = (
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
        )
        if self.enable_mixed_residency:
            self.full_touch_clock.add_(1)
        swap_in_fn(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=self.page_size if self.enable_mixed_residency else 1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            req_hot_buffer_sizes=(
                self.kvduo_req_hot_capacity_gpu[layer_id]
                if self.enable_mixed_residency
                else self.req_device_buffer_size_gpu
            ),
            hot_page_last_touch=(
                self.hot_page_last_touch[layer_id]
                if self.enable_mixed_residency
                else None
            ),
            req_to_logical_token=(
                self.req_to_full_lookup if self.enable_mixed_residency else None
            ),
            full_to_device_loc=(
                self.mem_pool_device.full_to_hisparse_device_index_mapping
                if self.enable_mixed_residency
                else None
            ),
            full_last_touch=(
                self.full_last_touch if self.enable_mixed_residency else None
            ),
            full_data_version=(
                self.full_data_version if self.enable_mixed_residency else None
            ),
            full_host_version=(
                self.full_host_version if self.enable_mixed_residency else None
            ),
            swap_status=(
                self.kvduo_swap_status[layer_id]
                if self.enable_mixed_residency
                else None
            ),
            resolver_stats=(
                self.kvduo_resolver_stats[layer_id]
                if self.enable_mixed_residency
                else None
            ),
            touch_clock=(
                self.full_touch_clock if self.enable_mixed_residency else None
            ),
        )
        if self.enable_mixed_residency:
            # Capturable execution boundary: an invalid/mismatched host source
            # must stop this stream before sparse attention consumes the table.
            torch._assert_async(
                torch.all(self.kvduo_swap_status[layer_id, req_pool_indices] == 0),
                "KVDuo resolver produced an incomplete address table",
            )
        return top_k_indices
