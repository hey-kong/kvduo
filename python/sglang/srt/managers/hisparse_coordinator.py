# to be combined with the sparse coordinator class and sparse algorithm family

import hashlib
import logging
from contextlib import contextmanager
from typing import List, NamedTuple, Sequence, Union

import torch
from sglang.kernels.ops.kvcache.hisparse import (
    allocate_hisparse_resolver_scratch,
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
    transfer_cache_dsv4_mla,
)
from sglang.kernels.ops.kvcache.kvduo_control import pack_kvduo_feedback
from sglang.kernels.ops.kvcache.kvduo_full_decode import (
    KVDuoFullTopK,
    begin_kvduo_decode,
)
from sglang.kernels.ops.kvcache.kvduo_full_reclaim import (
    select_kvduo_full_pages,
    select_kvduo_full_requests,
    summarize_kvduo_full_pages,
)
from sglang.kernels.ops.kvcache.kvduo_full_write import publish_kvduo_full_write
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseDSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.sparsity.core.kvduo_hot_merge import (
    free_kvduo_whole_pages,
)
from sglang.srt.mem_cache.sparsity.core.kvduo_prefix_cache import (
    HostPrefixRecord,
    KVDuoHostPrefixCache,
    KVDuoPrefixPageView,
    begin_kvduo_prefix_restore,
    finish_kvduo_prefix_restore,
    plan_kvduo_prefix_restore,
)
from sglang.srt.mem_cache.sparsity.core.kvduo_state import (
    KVDuoPressureAction,
    KVDuoPressureReason,
    KVDuoPressureResult,
    KVDuoPressureStatus,
    KVDuoResidencyCatalog,
    KVDuoResidencyMode,
    execute_kvduo_pressure_plan,
    plan_kvduo_allocation,
)
from sglang.srt.runtime_context import get_spec
from sglang.srt.utils import get_device_module, is_hip
from sglang.srt.utils.common import get_num_new_pages, is_pin_memory_available

device_module = get_device_module()

_is_hip = is_hip()

logger = logging.getLogger(__name__)


def _kvduo_stable_sort_atom(value):
    """Return an address-free ordering key for supported prefix-key values."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return (type(value).__name__, value)
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_kvduo_stable_sort_atom(v) for v in value))
    return (type(value).__qualname__, repr(value))


def _kvduo_stable_sort_digest(parent_digest: bytes, namespace, tokens) -> bytes:
    """Build a fixed-width, process-stable digest used only for LRU ordering."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(parent_digest)
    namespace_bytes = repr(_kvduo_stable_sort_atom(namespace)).encode("utf-8")
    digest.update(len(namespace_bytes).to_bytes(8, "big"))
    digest.update(namespace_bytes)
    for token in tokens:
        token_bytes = str(int(token)).encode("ascii")
        digest.update(len(token_bytes).to_bytes(8, "big"))
        digest.update(token_bytes)
    return digest.digest()


class _KVDuoPrefixIdentity:
    """An exact, persistent identity for one page-aligned token prefix.

    Each page adds only its own tokens and points at the preceding page.  The
    cached hash makes dictionary lookup constant-time in the usual case, while
    ``__eq__`` still walks and compares every chunk when hashes collide.
    """

    __slots__ = ("namespace", "parent", "tokens", "_hash", "stable_sort_digest")

    def __init__(self, namespace, parent, tokens):
        self.namespace = namespace
        self.parent = parent
        self.tokens = tokens
        self._hash = hash((namespace, hash(parent), tokens))
        self.stable_sort_digest = _kvduo_stable_sort_digest(
            parent.stable_sort_digest if parent is not None else b"",
            namespace,
            tokens,
        )

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, _KVDuoPrefixIdentity) or self._hash != other._hash:
            return False
        left = self
        right = other
        while left is not right:
            if (
                left is None
                or right is None
                or left.namespace != right.namespace
                or left.tokens != right.tokens
            ):
                return False
            left = left.parent
            right = right.parent
        return True


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
        full_reclaim_policy: str = "request",
        decode_emergency_reclaim: bool = False,
        kvduo_host_size: float | None = None,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        # ``server_args.device`` intentionally contains only the device type
        # (for example, ``"cuda"``).  Do not use that unindexed value for
        # stream/event selection: on a scheduler worker thread it resolves to
        # that thread's current device, which starts as logical GPU 0.  The
        # already allocated request-pool tensor carries the authoritative TP
        # rank index (``cuda:0``, ``cuda:1``, ...).
        device = req_to_token_pool.req_to_token.device
        self.device = device
        self.swap_in_block_size = swap_in_block_size
        self.tail_protected_pages = tail_protected_pages
        self.enable_mixed_residency = tail_protected_pages > 0
        if full_reclaim_policy not in ("request", "page"):
            raise ValueError("full_reclaim_policy must be 'request' or 'page'")
        self.full_reclaim_policy = full_reclaim_policy
        if not isinstance(decode_emergency_reclaim, bool):
            raise ValueError("decode_emergency_reclaim must be a boolean")
        self.decode_emergency_reclaim = decode_emergency_reclaim
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        host_budget = None
        if self.enable_mixed_residency and kvduo_host_size is None:
            kvduo_host_size = 0
        if self.is_dsv4_hisparse:
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
            page_size = self.mem_pool_device.page_size
            num_host_pages = (
                self.token_to_kv_pool_allocator.size_full // self.compress_ratio
                + page_size
                - 1
            ) // page_size
            if self.enable_mixed_residency:
                from sglang.srt.mem_cache.sparsity.core.kvduo_host_budget import (
                    plan_dsv4_host_budget,
                )

                host_budget = plan_dsv4_host_budget(
                    token_to_kv_pool_allocator, kvduo_host_size
                )
                num_host_pages = host_budget.full_pages
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
            host_kwargs = {}
            if self.enable_mixed_residency:
                from sglang.srt.mem_cache.sparsity.core.kvduo_host_budget import (
                    plan_host_budget,
                )

                pool = self.mem_pool_device
                host_budget = plan_host_budget(
                    kvduo_host_size,
                    pool.size // pool.page_size,
                    0,
                    {
                        "kv": pool.page_size
                        * pool.kv_cache_dim
                        * pool.store_dtype.itemsize
                        * pool.layer_num
                    },
                    {},
                )
                host_kwargs["num_host_pages"] = host_budget.full_pages
            self.mem_pool_host = MLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=self.mem_pool_device.page_size,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
                **host_kwargs,
            )
            self.item_size_bytes = self.mem_pool_host.token_stride_size
        if host_budget is not None:
            logger.info(
                "KVDuo Host KV payload per rank: host_size=%s GB, budget_bytes=%d, "
                "allocated_bytes=%d, full_pages=%d, swa_pages=%d, pools=%s",
                kvduo_host_size,
                host_budget.budget_bytes,
                host_budget.allocated_bytes,
                host_budget.full_pages,
                host_budget.swa_pages,
                host_budget.pool_bytes,
            )
        self.page_size = self.mem_pool_device.page_size
        if self.enable_mixed_residency:
            # A KVDuo physical page is the same page number across every KV
            # storage layer.  A HOT(request, logical_layer) owner may use all
            # of those fragments through global flat locations.
            self.hot_storage_layers = self.mem_pool_device.layer_num
            self.hot_page_size = self.hot_storage_layers * self.page_size
            self.min_hot_pages = (top_k + self.hot_page_size - 1) // self.hot_page_size
            # Legacy HiSparse buffers are unused in KVDuo. Actual HOT metadata
            # grows per owner behind a fixed, graph-visible pointer table.
            self.device_buffer_size = self.min_hot_pages * self.hot_page_size

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        self.resolver_scratch = (
            None
            if self.enable_mixed_residency
            else allocate_hisparse_resolver_scratch(
                max_num_req_slots,
                self.top_k,
                self.device_buffer_size,
                self.swap_in_block_size,
                device,
                use_lru_scratch=False,
            )
        )
        self.resolver_lru_scratch = None
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
            self.kvduo_full_clock_base = torch.zeros(
                1, dtype=torch.int64, device=device
            )
            self.kvduo_swap_status = torch.zeros(
                (self.mem_pool_device.layer_num, max_num_req_slots),
                dtype=torch.int32,
                device=device,
            )
            # Diagnostic counters do not decide HOT capacity.
            self.kvduo_resolver_stats = torch.zeros(
                (self.mem_pool_device.layer_num, max_num_req_slots, 2),
                dtype=torch.int32,
                device=device,
            )
            self._init_kvduo_capacity_state(max_num_req_slots)
            # Capture/warmup forwards use synthetic requests, not observations.
            # A graph reads this shared flag at execution time, so replay can
            # resume observation without recapturing or changing metadata pointers.
            self.kvduo_observe = torch.ones(1, dtype=torch.int32, device=device)
            # One event per resolver invocation that must replace an occupied
            # HOT slot for a new H2D miss. Fixed addresses survive graph replay;
            # the next graph boundary consumes every event without miss-rate
            # thresholds or observation windows.
            self.kvduo_hot_growth_events = torch.zeros(
                (self.mem_pool_device.layer_num, max_num_req_slots),
                dtype=torch.int32,
                device=device,
            )
            self._kvduo_hot_growth_events_host = torch.empty(
                self.kvduo_hot_growth_events.shape,
                dtype=torch.int32,
                device="cpu",
                pin_memory=is_pin_memory_available(device),
            )
            self._init_kvduo_feedback_workspace()
            self._active_kvduo_reqs = {}
            # Requests enter this set exactly when pressure first demotes a full
            # page.  It avoids rescanning every active mixed request and every
            # layer at each CUDA graph replay once the minimum HOT tier exists.
            self._kvduo_pending_hot_minimum = set()
            self._kvduo_host_valid_len = [0] * max_num_req_slots
            self._pending_kvduo_host_valid = []
            self._kvduo_pressure_protected = None
            self._kvduo_hot_pressure_protected = set()
            # One page number has exactly one state: FREE, FULL (the normal
            # allocator), or HOT(request, logical_layer).  There are no
            # independently-owned layer fragments.
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
            self._kvduo_hot_page_owners = {}
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
            self.kvduo_observe = None
            self.kvduo_hot_growth_events = None
            self._active_kvduo_reqs = None
            self._kvduo_pending_hot_minimum = None
            self._kvduo_host_valid_len = None
            self._pending_kvduo_host_valid = None
            self._kvduo_pressure_protected = None
            self._kvduo_hot_pressure_protected = None
            self.kvduo_req_hot_capacity = None
            self.kvduo_req_hot_capacity_gpu = None
            self._kvduo_hot_page_owners = None
            self._kvduo_req_layer_pages = None
            self.host_prefix_cache = None
            self._req_host_prefix_records = None

        self.write_staging_stream = device_module.Stream(device=device)
        self.decode_backup_stream = device_module.Stream(device=device)
        self.ack_staging_queue: List[HiSparseAct] = []
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # initialize data structures for swap-in kernel
        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (
                layer_num,
                max_num_req_slots,
                0 if self.enable_mixed_residency else self.padded_buffer_size,
            ),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (
                layer_num,
                max_num_req_slots,
                0 if self.enable_mixed_residency else self.padded_buffer_size,
            ),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(
            0 if self.enable_mixed_residency else self.device_buffer_size,
            dtype=(
                torch.int32
                if self.device_buffer_size > torch.iinfo(torch.int16).max
                else torch.int16
            ),
            device=device,
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
        self.kvduo_prefill_host_cache = None
        if self.enable_mixed_residency:
            if self.is_dsv4_hisparse:
                from sglang.srt.mem_cache.sparsity.core.kvduo_prefill_state import (
                    KVDuoPrefillHostCache,
                )

                self.kvduo_prefill_host_cache = KVDuoPrefillHostCache.create(
                    self, num_swa_pages=host_budget.swa_pages
                )
            # Decode allocation is owned by the token allocator, but pressure
            # policy belongs here.  The callback runs immediately before a
            # real sparse-KV page allocation and therefore cannot confuse a
            # logical/SWA/indexer shortage with physical main-KV pressure.
            self.token_to_kv_pool_allocator.enable_kvduo_full_decode = True
            self.token_to_kv_pool_allocator.kvduo_physical_allocation_guard = (
                self._guard_kvduo_physical_allocation
            )
            self.kvduo_hot_buffers = {}
            self.kvduo_hot_pointers = torch.zeros(
                (layer_num, max_num_req_slots, 4), dtype=torch.int64, device=device
            )
            self.kvduo_frequency_scratch = torch.empty(
                (max_num_req_slots, 9 * self.top_k), dtype=torch.int32, device=device
            )

    def warmup_reclamation(self) -> None:
        if self.enable_mixed_residency and self.kvduo_hot_pointers.is_cuda:
            from sglang.kernels.ops.kvcache.kvduo_warmup import warmup_kvduo_reclaim

            warmup_kvduo_reclaim(
                backing=self.mem_pool_device.get_flat_key_buffer(),
                page_size=self.page_size,
                storage_layers=self.hot_storage_layers,
                top_k=self.top_k,
                paged=self.is_dsv4_hisparse,
                block_size=self.swap_in_block_size,
            )

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def destroy(self) -> None:
        # Drain in-flight transfers so the buffer is idle, then unregister it.
        # See HostKVCache.destroy for why the explicit unregister matters.
        self.write_staging_stream.synchronize()
        self.decode_backup_stream.synchronize()
        if self.kvduo_prefill_host_cache is not None:
            self.kvduo_prefill_host_cache.destroy()
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

    def _host_prefix_identities(self, req: Req, page_ordinal: int, fill_ids=None):
        """Build exact page identities once, sharing all preceding token pages."""
        if fill_ids is None:
            fill_ids = req.get_fill_ids()
        page_tokens = self.page_size * self.compress_ratio
        namespace = (self.compress_ratio, getattr(req, "extra_key", None))
        state = getattr(req, "_kvduo_prefix_identity_state", None)
        if state is None or state[0] != namespace:
            identities = []
            req._kvduo_prefix_identity_state = (namespace, identities)
        else:
            identities = state[1]
        while len(identities) <= page_ordinal:
            ordinal = len(identities)
            token_start = ordinal * page_tokens
            token_end = min(len(fill_ids), token_start + page_tokens)
            identities.append(
                _KVDuoPrefixIdentity(
                    namespace,
                    identities[-1] if identities else None,
                    tuple(fill_ids[token_start:token_end]),
                )
            )
        return identities

    def _host_prefix_identity(self, req: Req, page_ordinal: int, fill_ids=None):
        return self._host_prefix_identities(req, page_ordinal, fill_ids)[page_ordinal]

    def augment_kvduo_prefix_match(
        self,
        req: Req,
        match_result,
        max_prefix_len: int | None = None,
        common_gpu_matcher=None,
    ):
        """Extend a GPU Radix hit with the longest valid host-owned prefix.

        The Radix node remains the lock anchor for its GPU portion.  Host pages
        use content identities and request references, so they remain pinned
        even before a recyclable request-pool slot is assigned.
        DeepSeek-V4 also requires a valid SWA window and boundary state.
        """
        if not self.enable_mixed_residency:
            return match_result
        state_cache = getattr(self, "kvduo_prefill_host_cache", None)
        if state_cache is not None:
            state_cache.poll()
            state_cache.release_match(req)
        for identity in getattr(req, "kvduo_host_prefix_records", ()):
            if identity in self.host_prefix_cache.records:
                self.host_prefix_cache.release(identity, req.rid)
        req.kvduo_host_prefix_records = set()
        gpu_len = len(match_result.device_indices)
        page_tokens = self.page_size * self.compress_ratio
        fill_ids = req.get_fill_ids()
        max_prefix = req._compute_max_prefix_len(len(fill_ids))
        if max_prefix_len is not None:
            max_prefix = min(max_prefix, max_prefix_len)
        if self.is_dsv4_hisparse and state_cache is None:
            # C4-only Host records cannot restore the preceding SWA window or
            # the other state required at a new prefill boundary. Recomputing
            # a window after that boundary does not supply its first query.
            # Keep the tree-validated GPU hit and its TP convergence below.
            max_prefix = min(max_prefix, gpu_len)
            req.kvduo_residency_plan = None
        max_prefix = max_prefix // page_tokens * page_tokens
        host_end = gpu_len // page_tokens * page_tokens
        first_host_page = gpu_len // page_tokens
        for ordinal in range(first_host_page, max_prefix // page_tokens):
            identity = self._host_prefix_identity(req, ordinal, fill_ids)
            record = self.host_prefix_cache.records.get(identity)
            if record is None or not record.fully_valid:
                break
            # Canonicalize the chain as it matches. The next page then compares
            # only its own token chunk; its parent is pointer-identical to the
            # cached record instead of re-walking the historical prefix.
            self._host_prefix_identities(req, ordinal, fill_ids)[ordinal] = (
                record.identity
            )
            host_end = (ordinal + 1) * page_tokens

        # GPU and host eviction are rank-local. Coordinate the final reusable
        # prefix end, not the per-rank incremental host hit: ranks can start
        # with different GPU prefix lengths. Only extend when the common host
        # end is beyond every rank's GPU end, ensuring every rank enters restore
        # and finishes with exactly the same prefix length.
        common_host_end = host_end
        min_gpu_len = gpu_len
        max_gpu_len = gpu_len
        if self.tp_world_size > 1:
            boundaries = torch.tensor(
                [host_end, gpu_len, -gpu_len], dtype=torch.int64, device="cpu"
            )
            torch.distributed.all_reduce(
                boundaries, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
            common_host_end = int(boundaries[0].item())
            min_gpu_len = int(boundaries[1].item())
            max_gpu_len = -int(boundaries[2].item())
        if state_cache is not None:
            # Window eviction is not prefix-monotonic. Intersect checkpoint
            # readiness, rather than taking the minimum of rank-local ends.
            local_ready = state_cache.ready_boundaries(req, gpu_len, max_prefix)
            ready = torch.tensor(
                [
                    int(valid and end <= host_end and end > max_gpu_len)
                    for valid, end in zip(
                        local_ready, range(page_tokens, max_prefix + 1, page_tokens)
                    )
                ],
                dtype=torch.int32,
                device="cpu",
            )
            if self.tp_world_size > 1 and ready.numel():
                torch.distributed.all_reduce(
                    ready, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
                )
            candidates = torch.nonzero(ready).flatten()
            common_host_end = (
                (int(candidates[-1]) + 1) * page_tokens if candidates.numel() else 0
            )
        if common_host_end <= max_gpu_len:
            if min_gpu_len != max_gpu_len:
                if common_gpu_matcher is None:
                    raise RuntimeError(
                        "KVDuo TP GPU prefix lengths diverged without a common "
                        "host-covered extension or common-prefix matcher"
                    )
                return self._converge_common_gpu_match(min_gpu_len, common_gpu_matcher)
            return match_result
        host_end = common_host_end
        state_plan = (
            state_cache.plan(req, gpu_len, host_end)
            if state_cache is not None
            else None
        )
        # Identity construction is unnecessary for a pure GPU hit.  Once a
        # host extension is confirmed, reuse the persistent chain already
        # built during probing to classify both GPU and host pages.
        page_identities = self._host_prefix_identities(req, host_end // page_tokens - 1)
        pages = [
            KVDuoPrefixPageView(
                identity=page_identities[ordinal],
                domains=("main_kv",),
                gpu_full_domains=(
                    frozenset({"main_kv"})
                    if (ordinal + 1) * page_tokens <= gpu_len
                    else frozenset()
                ),
            )
            for ordinal in range(host_end // page_tokens)
        ]
        identities = []
        for ordinal in range(first_host_page, host_end // page_tokens):
            identity = page_identities[ordinal]
            self.host_prefix_cache.acquire(identity, req.rid, 0)
            identities.append(identity)
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
            swa_host_hit_length=(
                state_plan.swa_tokens
                if state_plan is not None
                else match_result.swa_host_hit_length
            ),
            full_kv_hit_length=host_end,
            cache_protected_len=gpu_len,
            kvduo_residency_plan=plan,
        )

    def _converge_common_gpu_match(self, target_len: int, common_gpu_matcher):
        """Rematch until every TP rank has the same actual GPU prefix length."""
        for _ in range(8):
            result = common_gpu_matcher(target_len)
            actual_len = len(result.device_indices)
            if self.tp_world_size <= 1:
                return result
            lengths = torch.tensor(
                [actual_len, -actual_len], dtype=torch.int64, device="cpu"
            )
            torch.distributed.all_reduce(
                lengths, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
            min_actual = int(lengths[0].item())
            max_actual = -int(lengths[1].item())
            if min_actual == max_actual:
                return result
            # A bounded rematch should not grow beyond its target. If a cache
            # validator violates that assumption, converge conservatively.
            target_len = min_actual if min_actual < target_len else 0

        result = common_gpu_matcher(0)
        actual_len = len(result.device_indices)
        lengths = torch.tensor(
            [actual_len, -actual_len], dtype=torch.int64, device="cpu"
        )
        torch.distributed.all_reduce(
            lengths, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
        )
        if int(lengths[0].item()) != -int(lengths[1].item()):
            raise RuntimeError("KVDuo TP GPU prefix rematch failed to converge")
        return result

    def init_kvduo_load_back(
        self,
        req: Req,
        host_hit_length: int,
        radix_reclaimer=None,
        *,
        prefill_reservation: int | None = None,
    ):
        """Allocate and restore the host-only suffix before prefill executes."""
        plan = getattr(req, "kvduo_residency_plan", None)
        if plan is None or host_hit_length <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        state_cache = getattr(self, "kvduo_prefill_host_cache", None)
        state_plan = getattr(req, "kvduo_prefill_state_plan", None)
        if self.is_dsv4_hisparse and (state_cache is None or state_plan is None):
            raise RuntimeError(
                "KVDuo cannot restore a DeepSeek-V4 prefill prefix from C4-only "
                "Host records: required SWA and boundary state are not backed up"
            )
        commitment = begin_kvduo_prefix_restore(plan, self.host_prefix_cache)
        prefix_len = len(req.prefix_indices)
        target_len = prefix_len + host_hit_length
        swa_tail_len = state_plan.swa_tokens if state_plan is not None else 0
        if state_plan is not None and (
            state_plan.prefix_len != prefix_len
            or state_plan.target_len != target_len
            or state_plan.owner != state_cache._owner(req, "match")
            or not state_cache.ready(req, prefix_len, target_len)
        ):
            finish_kvduo_prefix_restore(commitment, self.host_prefix_cache)
            raise RuntimeError("KVDuo prefill checkpoint changed before restore")
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
        main_dma_pins = []
        try:
            allocator = self.token_to_kv_pool_allocator
            logical_pages, swa_pages, physical_pages = (
                self._kvduo_restore_page_requirements(
                    prefix_cpu, target_cpu, swa_tail_len=swa_tail_len
                )
            )

            logical_allocator = allocator.logical_attn_allocator
            physical_allocator = allocator.hisparse_attn_allocator
            if self.is_dsv4_hisparse:
                full_before = (
                    logical_allocator.full_available_size() // allocator.page_size
                )
                swa_before = (
                    logical_allocator.swa_available_size() // allocator.page_size
                )
            else:
                full_before = logical_allocator.available_size() // allocator.page_size
                swa_before = 0
            physical_before = physical_allocator.available_size() // self.page_size
            full_initial = full_before
            swa_initial = swa_before
            physical_initial = physical_before
            radix_evicted_full_tokens = 0
            radix_evicted_swa_tokens = 0

            # Prefill admission includes evictable Radix pages. Materialize that
            # budget before judging the restore impossible. A physical-only
            # shortage may require evicting more than the logical shortfall;
            # repeat while eviction makes progress because an evicted host-only
            # Radix page need not release a sparse physical page.
            if radix_reclaimer is not None:
                while (
                    full_before < logical_pages
                    or swa_before < swa_pages
                    or physical_before < physical_pages
                ):
                    full_shortfall = max(0, logical_pages - full_before)
                    swa_shortfall = max(0, swa_pages - swa_before)
                    physical_shortfall = max(0, physical_pages - physical_before)
                    full_evict_tokens = max(
                        full_shortfall * allocator.page_size,
                        physical_shortfall * self.page_size * self.compress_ratio,
                    )
                    swa_evict_tokens = swa_shortfall * allocator.page_size
                    if full_evict_tokens == 0 and swa_evict_tokens == 0:
                        break
                    evicted_full, evicted_swa = radix_reclaimer(
                        full_evict_tokens, swa_evict_tokens
                    )
                    if evicted_full <= 0 and evicted_swa <= 0:
                        break
                    radix_evicted_full_tokens += evicted_full
                    radix_evicted_swa_tokens += evicted_swa
                    if self.is_dsv4_hisparse:
                        full_before = (
                            logical_allocator.full_available_size()
                            // allocator.page_size
                        )
                        swa_before = (
                            logical_allocator.swa_available_size()
                            // allocator.page_size
                        )
                    else:
                        full_before = (
                            logical_allocator.available_size() // allocator.page_size
                        )
                    physical_before = (
                        physical_allocator.available_size() // self.page_size
                    )
            local_ready = full_before >= logical_pages and swa_before >= swa_pages
            if self.tp_world_size > 1:
                # ``tp_group`` is deliberately the Gloo/CPU group.  Keep its
                # control-plane scalar on CPU as well: passing a CUDA tensor to
                # this request-time collective makes Gloo initialize CUDA's
                # default logical device (GPU 0) in every TP process.
                ready = torch.tensor(int(local_ready), dtype=torch.int32)
                torch.distributed.all_reduce(
                    ready, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
                )
                local_ready = bool(ready.item())
            if not local_ready:
                logger.warning(
                    "KVDuo host-prefix restore deferred for req %s: logical Full/SWA "
                    "need Full=%d/SWA=%d pages, had Full=%d/SWA=%d initially and "
                    "Full=%d/SWA=%d after evicting Full=%d/SWA=%d Radix tokens; "
                    "physical pool needs %d pages, had %d initially and %d after "
                    "Radix eviction",
                    req.rid,
                    logical_pages,
                    swa_pages,
                    full_initial,
                    swa_initial,
                    full_before,
                    swa_before,
                    radix_evicted_full_tokens,
                    radix_evicted_swa_tokens,
                    physical_pages,
                    physical_initial,
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
                    physical_pages * self.page_size,
                    reserve_hot_minimum=self.enable_mixed_residency,
                    prefill_reservation=prefill_reservation,
                    allow_hot_reclaim=False,
                    allow_full_reclaim=False,
                )
            finally:
                self._kvduo_pressure_protected = None
            local_ready = pressure.action is KVDuoPressureAction.SUCCESS
            if self.tp_world_size > 1:
                ready = torch.tensor(int(local_ready), dtype=torch.int32)
                torch.distributed.all_reduce(
                    ready, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
                )
                local_ready = bool(ready.item())
            if not local_ready:
                logger.warning(
                    "KVDuo host-prefix restore deferred for req %s: physical pool "
                    "reclaim failed (needs %d pages, had %d initially and %d after "
                    "evicting Full=%d/SWA=%d Radix tokens, "
                    "action=%s, reason=%s, remaining_shortfall_bytes=%d); logical "
                    "Full/SWA pools had %d/%d initially and %d/%d after Radix eviction",
                    req.rid,
                    physical_pages,
                    physical_initial,
                    physical_before,
                    radix_evicted_full_tokens,
                    radix_evicted_swa_tokens,
                    pressure.action.name,
                    pressure.reason.name,
                    pressure.remaining_shortfall_bytes,
                    full_initial,
                    swa_initial,
                    full_before,
                    swa_before,
                )
                return None

            restore_kwargs = (
                {"swa_tail_len": swa_tail_len} if self.is_dsv4_hisparse else {}
            )
            restored = self.token_to_kv_pool_allocator.alloc_kvduo_restore(
                prefix_gpu,
                prefix_cpu,
                target_gpu,
                target_cpu,
                last_loc,
                host_hit_length,
                **restore_kwargs,
            )
            allocation_ready = restored is not None
            if self.tp_world_size > 1:
                ready = torch.tensor(int(allocation_ready), dtype=torch.int32)
                torch.distributed.all_reduce(
                    ready, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
                )
                allocation_ready = bool(ready.item())
            if not allocation_ready:
                if restored is not None:
                    # Roll back every pool touched by alloc_kvduo_restore (Full
                    # and C4 for DeepSeek V4) rather than using a component-only
                    # free method. The allocator also safely clears any SWA map.
                    self.token_to_kv_pool_allocator.rollback_restore_allocation(
                        restored, **restore_kwargs
                    )
                    restored = None
                logger.warning(
                    "KVDuo host-prefix restore deferred for req %s: alloc_extend "
                    "failed after admission (logical pool had %d pages; physical "
                    "pool had %d pages before reclaim; needed logical=%d, physical=%d)",
                    req.rid,
                    min(full_before, swa_before),
                    physical_before,
                    logical_pages,
                    physical_pages,
                )
                return None

            host_locs = []
            versions = []
            touches = []
            fill_ids = req.get_fill_ids()
            first_page = prefix_len // (self.page_size * self.compress_ratio)
            page_count = host_hit_length // (self.page_size * self.compress_ratio)
            for ordinal in range(first_page, first_page + page_count):
                record = self.host_prefix_cache.records[
                    self._host_prefix_identity(req, ordinal, fill_ids)
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
            if state_cache is not None:
                main_dma_pins = state_cache.pin_main_restore(state_plan)
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
            if state_cache is not None:
                state_cache.load(req, restored)
                if get_spec().speculative_algorithm is None:
                    req.kvduo_swa_live_from = target_len - swa_tail_len
            req.kvduo_shared_prefix_len = prefix_len
            req.kvduo_restored_prefix_len = target_len
            return restored
        except Exception:
            if restored is not None:
                self.token_to_kv_pool_allocator.rollback_restore_allocation(
                    restored,
                    **({"swa_tail_len": swa_tail_len} if self.is_dsv4_hisparse else {}),
                )
            raise
        finally:
            if state_cache is not None:
                state_cache.defer_unpin(main_dma_pins)
            finish_kvduo_prefix_restore(commitment, self.host_prefix_cache)

    def _kvduo_restore_page_requirements(
        self,
        prefix_lens_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        *,
        swa_tail_len: int = 0,
    ) -> tuple[int, int, int]:
        """Return Full/logical, SWA, and sparse-physical pages for restore.

        DeepSeek-V4 allocates SWA only for the restored checkpoint window.
        """
        allocator = self.token_to_kv_pool_allocator
        logical_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=allocator.page_size,
            prefix_lens=prefix_lens_cpu,
        )
        if self.is_dsv4_hisparse:
            physical_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu // self.compress_ratio,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu // self.compress_ratio,
            )
        else:
            physical_pages = logical_pages
        swa_pages = (swa_tail_len + allocator.page_size - 1) // allocator.page_size
        return logical_pages, swa_pages, physical_pages

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

    def _kvduo_host_capacity_ready(
        self, slots: int, *, prefill_key=None, _batch_capacity=None
    ) -> bool:
        reserved = sum(
            reservation[1]
            for key, reservation in self._kvduo_prefill_host_reservations.items()
            if key != prefill_key
        )
        required = slots + reserved
        self._evict_host_prefix_for_slots(required)
        ready = int(self.mem_pool_host.available_size() >= required)
        if _batch_capacity is None:
            return bool(self._kvduo_common_available(ready))
        values = torch.tensor(
            [ready, int(_batch_capacity["local_covered"])], dtype=torch.int64
        )
        if getattr(self, "tp_world_size", 1) > 1:
            torch.distributed.all_reduce(
                values, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
            )
        _batch_capacity["covered"] = bool(values[1].item())
        return bool(values[0].item())

    def _reserve_kvduo_prefill_host(self, req, key) -> bool:
        existing = self._kvduo_prefill_host_reservations.get(key)
        if existing is not None:
            return existing[0] is req
        # Reserve the complete prompt even when this execution is only a chunk.
        # Completed chunks consume this reservation as Host copies are created.
        fill_ids = req.full_untruncated_fill_ids
        length = len(fill_ids) // self.compress_ratio
        capacity = (length + self.page_size - 1) // self.page_size * self.page_size
        if capacity > self.mem_pool_host.size:
            req.set_finish_with_abort(
                "KVDuo prompt exceeds total Host main-KV capacity "
                f"(required={capacity}, capacity={self.mem_pool_host.size} slots); "
                "reduce the prompt length or increase kvduo host_size"
            )
            return False
        records = []
        owner = ("kvduo-prefill", key)
        try:
            for ordinal in range(length // self.page_size):
                identities = self._host_prefix_identities(req, ordinal, fill_ids)
                identity = identities[ordinal]
                record = self.host_prefix_cache.records.get(identity)
                if record is None or not record.fully_valid:
                    break
                identities[ordinal] = record.identity
                self.host_prefix_cache.acquire(identity, owner, 0)
                records.append(identity)
            slots = capacity - len(records) * self.page_size
            if not self._kvduo_host_capacity_ready(slots):
                return False
            self._kvduo_prefill_host_reservations[key] = (req, slots, tuple(records))
            return True
        finally:
            if key not in self._kvduo_prefill_host_reservations:
                for identity in records:
                    self.host_prefix_cache.release(identity, owner)

    def _release_kvduo_prefill_host(self, key) -> None:
        reservation = self._kvduo_prefill_host_reservations.pop(key, None)
        if reservation is not None:
            for identity in reservation[2]:
                if identity in self.host_prefix_cache.records:
                    self.host_prefix_cache.release(identity, ("kvduo-prefill", key))

    def _kvduo_decode_host_slots(self, requests) -> int:
        slots = 0
        for req in requests:
            length = self.host_token_len(req.kv_committed_len + 1)
            if length <= 1 or (length - 1) % self.page_size:
                continue
            slots += max(
                0,
                length - 1 - int(self.req_to_host_pool_allocated_len[req.req_pool_idx]),
            )
        return slots

    def _attach_cached_host_prefix(self, req: Req, host_len: int) -> int:
        """Attach the longest contiguous valid host prefix to this request row."""
        if not self.enable_mixed_residency:
            return 0
        owner = (req.rid, req.req_pool_idx)
        records = []
        locations = []
        fill_ids = req.get_fill_ids()
        for ordinal in range(host_len // self.page_size):
            identities = self._host_prefix_identities(req, ordinal, fill_ids)
            identity = identities[ordinal]
            record = self.host_prefix_cache.records.get(identity)
            state_cache = getattr(self, "kvduo_prefill_host_cache", None)
            if record is None or not (
                record.fully_valid
                or (
                    state_cache is not None
                    and record.dma_pins > 0
                    and state_cache.owns_pending_main(req, identity)
                )
            ):
                break
            identities[ordinal] = record.identity
            locations.extend(record.host_locations["main_kv"])
            records.append(identity)
        attached = len(locations)
        if attached:
            self.req_to_host_pool[req.req_pool_idx, :attached] = torch.tensor(
                locations, dtype=torch.int64, device=self.device
            )
            clock = self.host_prefix_cache.next_clock()
            for identity in records:
                self.host_prefix_cache.acquire(identity, owner, clock)
                self._req_host_prefix_records[req.req_pool_idx].add(identity)
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
        state_cache = getattr(self, "kvduo_prefill_host_cache", None)
        if state_cache is not None:
            state_cache.release_match(req)
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
        summary = (
            torch.cat((host_locs, versions, touches, self.full_touch_clock[:1]))
            .cpu()
            .tolist()
        )
        host_locs_cpu = summary[:valid_len]
        versions_cpu = summary[valid_len : 2 * valid_len]
        touches_cpu = summary[2 * valid_len : 3 * valid_len]
        clock = summary[-1]
        retained = set()
        fill_ids = req.get_fill_ids()
        for ordinal, start in enumerate(range(0, valid_len, self.page_size)):
            identities = self._host_prefix_identities(req, ordinal, fill_ids)
            identity = identities[ordinal]
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
                    last_access=clock,
                    tie_break_key=(ordinal, identity.stable_sort_digest),
                )
                self.host_prefix_cache.insert(record)
                retained.update(locations)
            else:
                identities[ordinal] = existing.identity
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
        prefill_key = (
            self._kvduo_prefill_key(req) if self.enable_mixed_residency else None
        )
        if self.enable_mixed_residency and not self._kvduo_host_capacity_ready(
            new_host_slots, prefill_key=prefill_key
        ):
            raise RuntimeError("KVDuo prefill lost its reserved Host capacity")
        host_indices = self.mem_pool_host.alloc_paged_token_slots(
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            0,
            prefill_len,
        )
        if self.enable_mixed_residency:
            # Canonical prefix pages are immutable and may be shared by an
            # in-flight restore. Only the newly allocated suffix needs staging.
            host_indices = host_indices[attached:]
            device_indices = device_indices[attached:]
        if self.enable_mixed_residency:
            self._release_kvduo_prefill_host(prefill_key)

        start_event = device_module.Event()
        finish_event = device_module.Event()
        schedule_stream = device_module.current_stream(self.device)
        start_event.record(schedule_stream)
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            if host_indices.numel():
                self.mem_pool_host.backup_from_device_all_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    io_backend="kernel",
                )
            finish_event.record(self.write_staging_stream)
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
        self,
        need_size: int,
        *,
        turnover_size: int = 0,
        reserve_hot_minimum: bool = False,
        prefill_reservation: int | None = None,
        allow_hot_reclaim: bool = True,
        allow_full_reclaim: bool = True,
        hot_minimum_credit: int = 0,
        allow_emergency_hot_reclaim: bool = False,
        preferred_full_requests=(),
    ) -> KVDuoPressureResult:
        """Reclaim physical KV pages while protecting pending allocations.

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
        initial_minimum_slots = (
            self._kvduo_mandatory_hot_pages() * self.page_size
            if reserve_hot_minimum
            else 0
        )
        minimum_slots = max(0, initial_minimum_slots - hot_minimum_credit)
        reserved_slots = self._kvduo_reserved_prefill_pages() * self.page_size
        if prefill_reservation is not None:
            reservation = self._kvduo_prefill_reservations.get(prefill_reservation)
            if reservation is None or need_size > reservation[0] * self.page_size:
                return KVDuoPressureResult(
                    KVDuoPressureStatus.RECLAIM_REQUIRED,
                    0,
                    action=KVDuoPressureAction.ERROR,
                    reason=KVDuoPressureReason.INVALID_STATE,
                    detail="Host-prefix restore exceeds its live prefill reservation",
                )
            # Restore and its fresh prefill tail were admitted together. Credit
            # only this allocation, keeping the tail and other prefills reserved.
            # The caller consumes this credit only after restore succeeds.
            reserved_slots -= need_size
            allow_hot_reclaim = False
            allow_full_reclaim = False

        def available_slots():
            # FULL reclamation may make another request mixed. Account for
            # the newly introduced minimum in addition to the initial demand.
            added_minimum = (
                max(
                    0,
                    self._kvduo_mandatory_hot_pages() * self.page_size
                    - initial_minimum_slots,
                )
                if reserve_hot_minimum
                else 0
            )
            return self._kvduo_common_available(
                max(
                    0,
                    physical_allocator.available_size()
                    - reserved_slots
                    - added_minimum,
                )
                // self.page_size
                * self.page_size
            )

        plan = plan_kvduo_allocation(
            available_main_kv_bytes=self._physical_bytes(available_slots()),
            requested_hot_bytes=self._physical_bytes(
                need_size - turnover_size + minimum_slots
            ),
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
            if not allow_full_reclaim:
                return 0
            if reserve_hot_minimum:
                reclaimed_slots = 0
                remaining = max(0, need_size + minimum_slots - available_slots())
                if self.full_reclaim_policy == "page":
                    for req_idx, demand in preferred_full_requests:
                        if remaining <= 0:
                            break
                        reclaimed_slots += self.reclaim_kvduo_full_pages(
                            min(remaining, demand),
                            reserve_hot_minimum=True,
                            request_indices={req_idx},
                        )
                        remaining = max(
                            0, need_size + minimum_slots - available_slots()
                        )
                if remaining > 0:
                    reclaimed_slots += self.reclaim_kvduo_full_pages(
                        remaining, reserve_hot_minimum=True
                    )
                    remaining = max(0, need_size + minimum_slots - available_slots())
            else:
                reclaimed_slots = self.reclaim_kvduo_full_pages(pages * self.page_size)
                remaining = pages * self.page_size - reclaimed_slots
            # Decode may continue through ordinary Hand merge. Prefill restore
            # must fit its physical reservation instead of this HOT fallback.
            if remaining > 0 and allow_hot_reclaim:
                reclaimed_slots += self.reclaim_kvduo_hot_pages(remaining)
            return reclaimed_slots // self.page_size * self._physical_page_bytes()

        return execute_kvduo_pressure_plan(
            plan,
            reclaim_full_pages=reclaim_bytes,
            available_bytes=lambda: self._physical_bytes(available_slots()),
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
        result = self._reclaim_for_physical_allocation(
            physical_slots,
            reserve_hot_minimum=self.enable_mixed_residency,
            allow_emergency_hot_reclaim=self.decode_emergency_reclaim,
        )
        return result.action is KVDuoPressureAction.SUCCESS

    def _reclaim_kvduo_radix_pages(self, required_pages: int, tree_cache) -> None:
        """Materialize evictable FULL budget before considering active owners.

        Host-only cache entries may release logical slots without releasing a
        physical page. Keep evicting while the cache makes progress, and always
        measure the dedicated physical pool rather than the wrapper's minimum.
        """
        if tree_cache is None:
            return
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        while True:
            available_pages = self._kvduo_common_available(
                allocator.available_size() // self.page_size
            )
            if available_pages >= required_pages:
                break
            evictable = (
                tree_cache.full_evictable_size()
                if self.is_dsv4_hisparse
                else tree_cache.evictable_size()
            )
            if self._kvduo_common_available(evictable) <= 0:
                break
            self._wait_kvduo_hot_merge_boundary()
            shortage = required_pages - available_pages
            evicted = tree_cache.evict(
                EvictParams(num_tokens=shortage * self.page_size * self.compress_ratio)
            )
            if self._kvduo_common_available(evicted.num_tokens_evicted) <= 0:
                break

    @contextmanager
    def kvduo_decode_reservation_scope(self, requests, *, consume=True):
        # Admission only borrows credit; actual decode allocation consumes it.
        future = getattr(self, "_kvduo_prefill_future_pages", {})
        if not future:
            yield
            return
        released = {}
        for req in requests:
            generation = getattr(req, "kvduo_prefill_generation", None)
            key = (getattr(req, "rid", None), generation)
            if key in future:
                released[key] = future.pop(key)
        try:
            yield
        except Exception:
            future.update(released)
            raise
        else:
            if not consume:
                future.update(released)

    def ensure_kvduo_decode_capacity(self, requests, *, tree_cache=None):
        requests = list(requests)
        with self.kvduo_decode_reservation_scope(requests, consume=False):
            return self._ensure_kvduo_decode_capacity(requests, tree_cache=tree_cache)

    def _ensure_kvduo_decode_capacity(
        self, requests, *, tree_cache=None
    ) -> KVDuoPressureResult:
        """Reserve the exact main sparse-KV pages needed by ordinary decode.

        This is deliberately separate from the composite allocator's
        ``available_size``: logical slots, SWA, indexer, and compression state
        remain the responsibility of their own admission paths.
        """
        if not self.enable_mixed_residency:
            return self._reclaim_for_physical_allocation(0)
        pages = 0
        preferred = {}
        if self.is_dsv4_hisparse:
            page = self.page_size
            for req in requests:
                next_len = req.kv_committed_len + 1
                if next_len % self.compress_ratio != 0:
                    continue
                compressed_len = next_len // self.compress_ratio
                if (compressed_len - 1) % page == 0:
                    pages += 1
                    preferred[req.req_pool_idx] = self.page_size
        else:
            logical_page = self.token_to_kv_pool_allocator.page_size
            for req in requests:
                if req.kv_committed_len % logical_page == 0:
                    pages += 1
                    preferred[req.req_pool_idx] = self.page_size
        for req_idx in sorted(self._kvduo_pending_hot_minimum):
            if req_idx not in self._active_kvduo_reqs:
                continue
            missing = sum(
                max(
                    0,
                    self.min_hot_pages
                    - int(self.kvduo_req_hot_capacity[layer, req_idx])
                    // self.hot_page_size,
                )
                for layer in range(self.mem_pool_device.layer_num)
            )
            if missing:
                preferred[req_idx] = (
                    preferred.get(req_idx, 0) + missing * self.page_size
                )
        host_slots = self._kvduo_decode_host_slots(requests)
        required_host_slots = host_slots + sum(
            reservation[1]
            for reservation in self._kvduo_prefill_host_reservations.values()
        )
        minimum_pages = self._kvduo_mandatory_hot_pages()
        required_pages = pages + minimum_pages + self._kvduo_reserved_prefill_pages()
        physical_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        local_ready = (
            self.mem_pool_host.available_size() >= required_host_slots
            and physical_allocator.available_size() // self.page_size >= required_pages
        )
        # A single collective covers both pools when no reclamation is needed.
        # All TP ranks must agree before any rank skips the pressure path.
        if self._kvduo_common_available(int(local_ready)):
            return KVDuoPressureResult(
                KVDuoPressureStatus.READY,
                self._physical_bytes((pages + minimum_pages) * self.page_size),
                action=KVDuoPressureAction.SUCCESS,
                reason=KVDuoPressureReason.NONE,
            )
        if not self._kvduo_host_capacity_ready(host_slots):
            return KVDuoPressureResult(
                KVDuoPressureStatus.RECLAIM_REQUIRED,
                0,
                action=KVDuoPressureAction.ERROR,
                reason=KVDuoPressureReason.MEMORY_SHORTAGE,
                detail="KVDuo Host pool cannot cover decode backup and prefill reservations",
            )
        self._reclaim_kvduo_radix_pages(required_pages, tree_cache)
        return self._reclaim_for_physical_allocation(
            pages * self.page_size,
            reserve_hot_minimum=True,
            preferred_full_requests=tuple(sorted(preferred.items())),
            allow_emergency_hot_reclaim=(
                self.decode_emergency_reclaim and pages + minimum_pages > 0
            ),
        )

    def log_kvduo_decode_pressure(self, requests, logical_need, pressure=None) -> None:
        """Report initial admission failures without synchronizing cache tensors."""
        logical = self.token_to_kv_pool_allocator.logical_attn_allocator
        full_available = getattr(
            logical, "full_available_size", logical.available_size
        )()
        swa_available = getattr(logical, "swa_available_size", lambda: -1)()
        logger.warning(
            "KVDuo decode admission failed before request retraction: "
            "reason=%s, detail=%s, requests=%d, logical_need=%d, "
            "logical_available=%d, logical_full_available=%d, swa_available=%d, "
            "physical_free_pages=%d, pending_hot_minimum_pages=%d, "
            "prefill_reserved_pages=%d, host_available=%d, "
            "host_prefill_reserved=%d, reclaimed_bytes=%d, shortfall_bytes=%d, "
            "full_reclaim_policy=%s, decode_emergency_reclaim=%s",
            pressure.reason.name if pressure is not None else "LOGICAL_OR_SWA_SHORTAGE",
            (
                pressure.detail
                if pressure is not None
                else "independent logical/SWA allocation cannot fit"
            ),
            len(requests),
            logical_need,
            logical.available_size(),
            full_available,
            swa_available,
            self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
            // self.page_size,
            self._kvduo_mandatory_hot_pages(),
            self._kvduo_reserved_prefill_pages(),
            self.mem_pool_host.available_size(),
            sum(
                reservation[1]
                for reservation in self._kvduo_prefill_host_reservations.values()
            ),
            pressure.reclaimed_bytes if pressure is not None else 0,
            pressure.remaining_shortfall_bytes if pressure is not None else 0,
            self.full_reclaim_policy,
            self.decode_emergency_reclaim,
        )

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
            req.kvduo_radix_insert_len = (
                compressed_len // self.page_size * self.page_size * self.compress_ratio
            )

            # A fully-resident KVDuo request does not own a speculative hot
            # buffer. The first FULL demotion schedules the whole-page minimum
            # for the next replay boundary.
            self.req_to_device_buffer[req.req_pool_idx, :] = 0
            self.req_device_buffer_size[req.req_pool_idx] = 0
            self.req_device_buffer_size_gpu[req.req_pool_idx] = 0
            self.kvduo_req_hot_capacity[:, req.req_pool_idx] = 0
            self.kvduo_req_hot_capacity_gpu[:, req.req_pool_idx] = 0
            self.kvduo_resolver_stats[:, req.req_pool_idx, :] = 0
            self.kvduo_hot_growth_events[:, req.req_pool_idx] = 0
            self._reset_kvduo_hot_metadata(req.req_pool_idx)
            self._kvduo_decode_sequence += 1
            self._kvduo_decode_order[req.req_pool_idx] = (
                self._kvduo_request_generation[req.req_pool_idx],
                self._kvduo_decode_sequence,
            )
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

    def _ensure_kvduo_hot_capacity_targets(
        self,
        layer_id: int,
        requested_capacities: dict[int, int],
        *,
        allow_reclaim: bool = True,
    ) -> None:
        """Materialize page-aligned CPU-known targets without exponential tiers.

        This path serves minimum admission and H2D-triggered growth. Targets
        retain their whole-page alignment without exponential capacity tiers.
        """
        if not requested_capacities:
            return
        self._ensure_kvduo_hot_capacity_targets_batch(
            {
                (layer_id, req): capacity
                for req, capacity in requested_capacities.items()
            },
            allow_reclaim=allow_reclaim,
        )

    def _ensure_kvduo_hot_capacity_targets_batch(
        self,
        requested_capacities: dict[tuple[int, int], int],
        *,
        allow_reclaim: bool = True,
    ) -> None:
        """Materialize targets for all request/layer pairs in one allocation.

        The first transition to mixed residency commonly needs the minimum HOT
        tier in every C4 logical layer.  Keeping that round as one transaction
        avoids one allocator result D2H synchronization per logical layer.
        """
        if not requested_capacities:
            return
        K = self.hot_page_size
        requests = []
        for (layer_id, req_idx), required_slots in requested_capacities.items():
            current = int(self.kvduo_req_hot_capacity[layer_id, req_idx])
            target = ((required_slots + K - 1) // K) * K
            if target > current:
                requests.append(
                    (int(layer_id), int(req_idx), current, target, target - current)
                )
        self._materialize_kvduo_hot_growth_batch(
            requests,
            allow_reclaim=allow_reclaim,
        )

    def _materialize_kvduo_hot_growth_batch(
        self,
        requests: list[tuple[int, int, int, int, int]],
        allow_reclaim: bool = True,
    ) -> None:
        """Allocate and publish a cross-layer HOT growth transaction."""
        if not requests:
            return
        K = self.hot_page_size
        requests = [
            (layer, req, current, target, grow)
            for layer, req, current, target, grow in requests
            if grow > 0
        ]
        pages_needed = sum(grow // K for _, _, _, _, grow in requests)
        physical_slots = pages_needed * self.page_size
        if physical_slots == 0:
            return
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        if not allow_reclaim and (
            allocator.available_size()
            - self._kvduo_optional_protected_pages() * self.page_size
            < physical_slots
        ):
            return
        if allow_reclaim:
            previous_full_protected = getattr(self, "_kvduo_pressure_protected", None)
            previous_hot_protected = getattr(
                self, "_kvduo_hot_pressure_protected", set()
            )
            self._kvduo_hot_pressure_protected = previous_hot_protected | {
                (layer_id, req_idx, page_index)
                for layer_id, req_idx, *_ in requests
                for page_index in range(
                    len(self._kvduo_req_layer_pages.get((layer_id, req_idx), ()))
                )
            }
            try:
                # This transaction already includes these owners' compulsory
                # minima. Protect every OTHER minimum, including owners made
                # mixed by the FULL reclaim itself, without charging ours twice.
                credit = (
                    sum(
                        min(grow // K, max(0, self.min_hot_pages - current // K))
                        for _, req, current, _, grow in requests
                        if req in getattr(self, "_kvduo_pending_hot_minimum", ())
                    )
                    * self.page_size
                )
                pressure = self._reclaim_for_physical_allocation(
                    physical_slots, reserve_hot_minimum=True, hot_minimum_credit=credit
                )
                self._require_allocation_ready(pressure)
            finally:
                self._kvduo_pressure_protected = previous_full_protected
                self._kvduo_hot_pressure_protected = previous_hot_protected
        physical = allocator.alloc(physical_slots)
        if physical is None:
            if allow_reclaim:
                raise RuntimeError("KVDuo whole-page HOT allocation failed")
            return
        from sglang.srt.mem_cache.sparsity.core.kvduo_frequency import FrequencyBuffer

        self._wait_kvduo_hot_merge_boundary()
        staged = []
        published = []
        safe_to_free = True
        try:
            starts = physical.view(-1, self.page_size)[:, 0].cpu().tolist()
            if len(starts) != len(set(starts)) or any(
                p <= 0 or p % self.page_size or p in self._kvduo_hot_page_owners
                for p in starts
            ):
                safe_to_free = False
                raise RuntimeError("KVDuo allocator returned duplicate/owned HOT pages")
            offset = 0
            layer_stride = self.mem_pool_device.flat_layer_slot_stride
            storage = (
                torch.arange(self.hot_storage_layers, device=self.device) * layer_stride
            )
            within = torch.arange(self.page_size, device=self.device)
            all_locations = (
                physical.view(-1, self.page_size)[:, 0, None, None]
                + storage[None, :, None]
                + within
            ).flatten()
            for layer, req, current, target, grow in requests:
                owner = (layer, req)
                pages = starts[offset : offset + grow // K]
                locations = all_locations[offset * K : (offset + len(pages)) * K]
                offset += len(pages)
                old = self.kvduo_hot_buffers.get(owner)
                if (old.tokens.numel() if old is not None else 0) != current:
                    raise RuntimeError("KVDuo HOT metadata/capacity mismatch")
                new = (
                    old.grown(locations)
                    if old is not None
                    else FrequencyBuffer.empty(locations)
                )
                staged.append((owner, old, new, pages, current, target))
            published = [
                (owner, old, current)
                for owner, old, new, pages, current, target in staged
            ]
            self._publish_kvduo_frequency_buffers(
                [(owner, new) for owner, _, new, *_ in staged]
            )
            self._finish_kvduo_metadata_update()
        except Exception:
            self._publish_kvduo_frequency_buffers(
                [(owner, old) for owner, old, _ in published]
            )
            self._finish_kvduo_metadata_update()
            # Metadata is restored before returning any newly allocated page.
            if safe_to_free:
                self.token_to_kv_pool_allocator.free_hisparse_indices(physical)
            raise
        for owner, old, new, pages, current, target in staged:
            self.kvduo_hot_buffers[owner] = new
            self.kvduo_req_hot_capacity[owner] = target
            self._kvduo_req_layer_pages.setdefault(owner, []).extend(pages)
            self._kvduo_hot_generation[owner] = self._kvduo_request_generation[owner[1]]
            for start in pages:
                self._kvduo_hot_page_owners[start] = owner

    def _publish_kvduo_frequency_buffers(self, updates):
        if not updates:
            return
        requests = self.kvduo_hot_pointers.size(1)
        packet = torch.tensor(
            [
                [
                    layer * requests + req,
                    buffer.tokens.numel() if buffer is not None else 0,
                    *(buffer.pointers() if buffer is not None else [0] * 4),
                ]
                for (layer, req), buffer in updates
            ],
            dtype=torch.int64,
            pin_memory=self.kvduo_hot_pointers.is_cuda,
        ).to(self.device, non_blocking=True)
        indices = packet[:, 0]
        self.kvduo_hot_pointers.view(-1, 4).index_copy_(0, indices, packet[:, 2:])
        self.kvduo_req_hot_capacity_gpu.view(-1).index_copy_(
            0, indices, packet[:, 1].int()
        )

    def _publish_kvduo_frequency_buffer(self, owner, buffer):
        pointers = buffer.pointers() if buffer is not None else [0] * 4
        self.kvduo_hot_pointers[owner].copy_(
            torch.tensor(pointers, dtype=torch.int64, device=self.device)
        )
        self.kvduo_req_hot_capacity_gpu[owner] = (
            buffer.tokens.numel() if buffer is not None else 0
        )

    def _finish_kvduo_metadata_update(self):
        if self.kvduo_hot_pointers.is_cuda:
            device_module.current_stream(self.device).synchronize()

    def kvduo_batch_fully_resident(self, requests_cpu) -> bool:
        if not self.enable_mixed_residency or requests_cpu is None:
            return False
        if isinstance(requests_cpu, torch.Tensor):
            if requests_cpu.device.type != "cpu":
                return False
            requests_cpu = requests_cpu.tolist()
        return bool(len(requests_cpu)) and all(
            int(req) in self._active_kvduo_reqs
            and not self._mixed_slots[int(req)]
            and int(req) not in self._kvduo_pending_hot_minimum
            and not getattr(
                self._active_kvduo_reqs[int(req)], "hisparse_staging", False
            )
            for req in requests_cpu
        )

    def kvduo_full_topk_metadata(self, layer, requests, output):
        return KVDuoFullTopK(
            output,
            requests,
            self.req_to_full_lookup,
            self.mem_pool_device.full_to_hisparse_device_index_mapping,
            self.full_last_touch,
            self.kvduo_swap_status[layer],
            self.kvduo_resolver_stats[layer],
            self.kvduo_hot_growth_events[layer],
            self.kvduo_observe,
            self.num_real_reqs,
            self.kvduo_full_clock_base,
            layer,
            self.mem_pool_device.flat_layer_slot_stride,
            self.page_size,
        )

    def publish_kvduo_decode_controls(self, batch_size, *, fully_resident=False):
        begin_kvduo_decode(
            self.full_touch_clock,
            self.kvduo_full_clock_base,
            self.kvduo_observe,
            self.num_real_reqs,
            batch_size=batch_size,
            full_layers=self.mem_pool_device.layer_num if fully_resident else 0,
        )

    def _init_kvduo_feedback_workspace(self):
        layers, requests = self.kvduo_hot_growth_events.shape
        self._kvduo_feedback_positions = torch.empty(
            requests, dtype=torch.int32, device=self.kvduo_hot_growth_events.device
        )
        self._kvduo_feedback_packet = torch.empty(
            layers * requests,
            dtype=torch.int32,
            device=self.kvduo_hot_growth_events.device,
        )
        self._kvduo_feedback_bindings = None

    def _consume_kvduo_hot_growth_events(self) -> torch.Tensor | dict:
        """Read each H2D capacity event before the immediately following replay.

        A blocking D2H read at this graph-external boundary is necessary: an
        asynchronous, possibly stale snapshot could postpone an event for
        several replays. The caller has ordered this stream after the preceding
        producer. Reset the same fixed-address bank before its next replay.
        """
        if self.kvduo_hot_growth_events.is_cuda:
            layers, requests = self.kvduo_hot_growth_events.shape
            bindings = tuple(
                (req, self._kvduo_request_generation[req])
                for req in sorted(self._active_kvduo_reqs)
                if self._mixed_slots[req]
            )
            if not hasattr(self, "_kvduo_feedback_positions"):
                self._init_kvduo_feedback_workspace()
            if bindings != self._kvduo_feedback_bindings:
                positions = torch.full((requests,), -1, dtype=torch.int32)
                for i, (req, _) in enumerate(bindings):
                    positions[req] = i
                self._kvduo_feedback_positions.copy_(positions)
                self._kvduo_feedback_bindings = bindings
            count = len(bindings)
            pack_kvduo_feedback(
                self.kvduo_hot_growth_events,
                self.kvduo_resolver_stats,
                self._kvduo_feedback_positions,
                self._kvduo_feedback_packet,
                count,
            )
            if not count:
                return {}
            packet = self._kvduo_hot_growth_events_host.view(-1)[: layers * count]
            packet.copy_(self._kvduo_feedback_packet[: layers * count])
            result = {}
            for layer, position in torch.nonzero(
                packet.view(layers, count) > 0
            ).tolist():
                req, generation = bindings[position]
                if (
                    generation == self._kvduo_request_generation[req]
                    and req in self._active_kvduo_reqs
                ):
                    result[layer, req] = int(packet[layer * count + position])
            return result
        self._kvduo_hot_growth_events_host.copy_(self.kvduo_hot_growth_events)
        self.kvduo_hot_growth_events.zero_()
        self.kvduo_resolver_stats.zero_()
        return self._kvduo_hot_growth_events_host

    def prepare_kvduo_graph_replay(
        self,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: Sequence[int] | torch.Tensor | None = None,
        *,
        publish_observation: bool = True,
    ) -> None:
        """Perform physical capacity management at the graph replay boundary.

        This is intentionally host-side and must run only after the preceding
        producer stream is complete.  Fully resident requests retain capacity
        zero; newly mixed requests atomically acquire their whole-page minimum
        tier before the next resolver replay.
        """
        if not self.enable_mixed_residency or req_pool_indices.numel() == 0:
            return
        if self.decode_producer_stream is not None:
            device_module.current_stream(self.device).wait_stream(
                self.decode_producer_stream
            )
        if publish_observation:
            self.kvduo_observe.fill_(1)
        tracked_pending_minimum = getattr(self, "_kvduo_pending_hot_minimum", None)
        pending_minimum = tracked_pending_minimum
        if pending_minimum is None:
            # Compatibility for focused unit tests that construct the
            # coordinator with ``__new__``.
            pending_minimum = {
                req_idx
                for req_idx in self._active_kvduo_reqs
                if self._mixed_slots[req_idx]
                and any(
                    int(self.kvduo_req_hot_capacity[layer_id, req_idx])
                    < self.min_hot_pages * self.hot_page_size
                    for layer_id in range(self.mem_pool_device.layer_num)
                )
            }
        if not pending_minimum and not any(
            self._mixed_slots[req_idx] for req_idx in self._active_kvduo_reqs
        ):
            return
        events_cpu = self._consume_kvduo_hot_growth_events()
        growth_events = (
            events_cpu
            if isinstance(events_cpu, dict)
            else {
                (layer_id, req_idx): int(events_cpu[layer_id, req_idx])
                for layer_id, req_idx in torch.nonzero(events_cpu > 0).tolist()
                if req_idx in self._active_kvduo_reqs and self._mixed_slots[req_idx]
            }
        )
        event_capacities = {
            owner: int(self.kvduo_req_hot_capacity[owner]) for owner in growth_events
        }
        # Events are rank-local. Mixed TP ranks must all reach the optional
        # growth collective, including ranks with no event in this replay.
        # Otherwise the next control collective can match a different step.
        if (
            not pending_minimum
            and not growth_events
            and getattr(self, "tp_world_size", 1) <= 1
        ):
            return
        requested = set(pending_minimum)
        requested.update(req_idx for _, req_idx in growth_events)

        initial_mixed = {req_idx for req_idx in requested if self._mixed_slots[req_idx]}

        # Pressure reclamation can demote another active request. Iterate to a
        # capacity fixed point so every mixed request has its minimum whole-page
        # capacity in every logical layer before event-driven growth.
        pending = initial_mixed
        while True:
            mixed = sorted(pending)
            if not mixed:
                break
            before = {
                (layer_id, req_idx): int(self.kvduo_req_hot_capacity[layer_id, req_idx])
                for req_idx in mixed
                for layer_id in range(self.mem_pool_device.layer_num)
            }
            targets = {
                (layer_id, req_idx): self.min_hot_pages * self.hot_page_size
                for req_idx in mixed
                for layer_id in range(self.mem_pool_device.layer_num)
                if int(self.kvduo_req_hot_capacity[layer_id, req_idx])
                < self.min_hot_pages * self.hot_page_size
            }
            if targets:
                self._ensure_kvduo_hot_capacity_targets_batch(targets)
            requested.update(
                (
                    self._kvduo_pending_hot_minimum
                    if tracked_pending_minimum is not None
                    else (
                        req_idx
                        for req_idx in self._active_kvduo_reqs
                        if self._mixed_slots[req_idx]
                    )
                )
            )
            pending = {
                req_idx
                for req_idx in requested
                if self._mixed_slots[req_idx]
                and any(
                    int(self.kvduo_req_hot_capacity[layer_id, req_idx])
                    < self.min_hot_pages * self.hot_page_size
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
                    "KVDuo mixed request lacks mandatory HOT capacity before replay"
                )

        for req_idx in requested:
            if not self._mixed_slots[req_idx]:
                continue
            for layer_id in range(self.mem_pool_device.layer_num):
                if (
                    int(self.kvduo_req_hot_capacity[layer_id, req_idx])
                    < self.min_hot_pages * self.hot_page_size
                ):
                    raise RuntimeError(
                        "KVDuo mixed request lacks mandatory HOT capacity before replay"
                    )
        if getattr(self, "_kvduo_pending_hot_minimum", None) is not None:
            self._kvduo_pending_hot_minimum.difference_update(requested)

        # Mandatory allocations may already have shrunk an applicant after its
        # event was consumed. Cancel that event at this boundary as well.
        growth_events = {
            owner: events
            for owner, events in growth_events.items()
            if int(self.kvduo_req_hot_capacity[owner]) >= event_capacities[owner]
        }
        # Decode growth may reclaim eligible FULL, then Hand-eligible HOT.
        # Prefill protection is covered before any optional reclamation.
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        reserved_pages = self._kvduo_optional_protected_pages()
        applicants = {owner: 1 for owner, events in growth_events.items() if events > 0}
        applicants = {owner: pages for owner, pages in applicants.items() if pages > 0}
        common_free, consistent = self._kvduo_optional_growth_resources(
            reserved_pages, applicants
        )
        if not consistent:
            return
        if not applicants:
            return
        # Protection only limits optional allocation. Its own shortfall belongs
        # to prefill admission; optional growth cannot consume its reservation.
        if common_free < reserved_pages:
            return

        def growth_shortfall():
            return max(
                0,
                sum(applicants.values())
                + self._kvduo_optional_protected_pages()
                - self._kvduo_common_available(
                    allocator.available_size() // self.page_size
                ),
            )

        shortfall = max(0, sum(applicants.values()) - (common_free - reserved_pages))
        if shortfall > 0:
            if self.full_reclaim_policy == "page":
                for req_idx in sorted({req for _, req in applicants}):
                    if shortfall == 0:
                        break
                    demand = sum(
                        pages
                        for (_, req), pages in applicants.items()
                        if req == req_idx
                    )
                    self.reclaim_kvduo_full_pages(
                        min(shortfall, demand) * self.page_size,
                        reserve_hot_minimum=True,
                        request_indices={req_idx},
                    )
                    shortfall = growth_shortfall()
            if shortfall > 0:
                self.reclaim_kvduo_full_pages(
                    shortfall * self.page_size, reserve_hot_minimum=True
                )

            # A FULL donor can be in this replay batch. Publish every donor's
            # compulsory HOT minimum before optional growth or graph execution.
            donors = set(self._kvduo_pending_hot_minimum)
            minimum_targets = {
                (layer, req): self.min_hot_pages * self.hot_page_size
                for req in donors
                if req in self._active_kvduo_reqs
                for layer in range(self.mem_pool_device.layer_num)
                if int(self.kvduo_req_hot_capacity[layer, req])
                < self.min_hot_pages * self.hot_page_size
            }
            if minimum_targets:
                self._ensure_kvduo_hot_capacity_targets_batch(minimum_targets)
            self._kvduo_pending_hot_minimum.difference_update(donors)
            shortfall = growth_shortfall()
            if shortfall > 0:
                # Pending growth is not an in-flight use. Hand may shrink an
                # applicant; the capacity check below then cancels its event.
                self.reclaim_kvduo_hot_pages(shortfall * self.page_size)
            applicants = {
                owner: pages
                for owner, pages in applicants.items()
                if int(self.kvduo_req_hot_capacity[owner]) >= event_capacities[owner]
            }
            reserved_pages = self._kvduo_optional_protected_pages()
            common_free = self._kvduo_common_available(
                allocator.available_size() // self.page_size
            )
        free_pages = max(
            0,
            common_free - reserved_pages,
        )
        growth_targets = {}
        for (layer_id, req_idx), requested_pages in sorted(applicants.items()):
            current = int(self.kvduo_req_hot_capacity[layer_id, req_idx])
            pages = min(requested_pages, free_pages)
            if pages > 0:
                growth_targets[(layer_id, req_idx)] = (
                    current + pages * self.hot_page_size
                )
                free_pages -= pages
        self._ensure_kvduo_hot_capacity_targets_batch(
            growth_targets, allow_reclaim=False
        )

    def _reset_kvduo_hot_metadata(self, req_idx: int) -> None:
        """Clear statistics and generation-bound state after old readers."""
        if not self.enable_mixed_residency:
            return
        self._wait_kvduo_hot_merge_boundary()
        self._reset_kvduo_prefill_request(req_idx)

    def _release_kvduo_layer_hot_pages(self, req_idx: int) -> None:
        """Return every HOT page of a request directly to the common allocator."""
        if not self.enable_mixed_residency:
            return
        self._end_kvduo_prefill_lifecycle(req_idx)
        self._reset_kvduo_hot_metadata(req_idx)
        starts = []
        for layer_id in range(self.mem_pool_device.layer_num):
            owner = (layer_id, req_idx)
            for start in self._kvduo_req_layer_pages.pop(owner, []):
                if self._kvduo_hot_page_owners.pop(start, None) != owner:
                    raise RuntimeError("KVDuo whole-page ownership mismatch")
                starts.append(start)
            self._publish_kvduo_frequency_buffer(owner, None)
            self._finish_kvduo_metadata_update()
            self.kvduo_hot_buffers.pop(owner, None)
            self.kvduo_req_hot_capacity[owner] = 0
        if starts:
            indices = [
                i
                for start in sorted(starts)
                for i in range(start, start + self.page_size)
            ]
            self.token_to_kv_pool_allocator.free_hisparse_indices(
                torch.tensor(indices, dtype=torch.int64, device=self.device)
            )

    def _wait_kvduo_hot_merge_boundary(self) -> None:
        """Order pressure work after decode readers and pending KV transfers."""
        if self.decode_producer_stream is not None:
            device_module.current_stream(self.device).wait_stream(
                self.decode_producer_stream
            )
        if getattr(self, "_has_pending_backup", False):
            self.wait_for_pending_backup()

    def _move_kvduo_hot_entries(
        self, layer_id: int, src_locs: torch.Tensor, dst_locs: torch.Tensor
    ) -> None:
        """Copy disjoint released-page sources into retained-page destinations."""
        if src_locs.numel() == 0:
            return
        if self.is_dsv4_hisparse:
            # Treat the contiguous layer/page backing as ONE cache: indices
            # already encode storage layers and may move across those layers.
            backing = self.mem_pool_device.get_flat_key_buffer()
            ptrs = torch.tensor(
                [backing.data_ptr()], dtype=torch.uint64, device=self.device
            )
            transfer_cache_dsv4_mla(ptrs, ptrs, src_locs, dst_locs)
        else:
            cache = self.mem_pool_device.get_flat_key_buffer()
            cache.index_copy_(0, dst_locs, cache.index_select(0, src_locs))

    def _merge_kvduo_hot_pages(self, layer_id, req_idx):
        from sglang.srt.mem_cache.sparsity.core.kvduo_frequency import (
            FrequencyBuffer,
            pack_frequency,
            plan_frequency_merge,
            unpack_frequency,
        )

        owner = (layer_id, req_idx)
        if not self._kvduo_can_merge(owner):
            return 0
        self._wait_kvduo_hot_merge_boundary()
        old = self.kvduo_hot_buffers[owner]
        pages = self._kvduo_req_layer_pages[owner]
        capacity = int(self.kvduo_req_hot_capacity[owner])
        if (
            old.tokens.numel() != capacity
            or len(pages) * self.hot_page_size != capacity
            or any(self._kvduo_hot_page_owners.get(page) != owner for page in pages)
        ):
            raise RuntimeError("KVDuo HOT merge ownership mismatch")
        if old.tokens.is_cuda:
            from sglang.kernels.ops.kvcache.kvduo_frequency import merge

            new, kept = merge(
                old,
                self.mem_pool_device.get_flat_key_buffer(),
                k=self.hot_page_size,
                minimum=self.min_hot_pages,
                paged=self.is_dsv4_hisparse,
                top_k=self.top_k,
                item_bytes=self.item_size_bytes,
                publication=(
                    self.kvduo_hot_pointers[owner],
                    self.kvduo_req_hot_capacity_gpu[layer_id, req_idx : req_idx + 1],
                    self.kvduo_hot_growth_events[layer_id, req_idx : req_idx + 1],
                ),
            )
            # Payloads may already have moved. Disagreement is fatal, never
            # resume with the old metadata or release pages on a single rank.
            if not self._kvduo_consensus((owner, kept)):
                raise RuntimeError("KVDuo frequency merge differs across TP ranks")
        else:
            plan = plan_frequency_merge(
                old.tokens.tolist(),
                unpack_frequency(old.frequency, capacity),
                self.hot_page_size,
                self.min_hot_pages,
            )
            if not self._kvduo_consensus(
                (owner, plan.kept_pages, plan.sources, plan.destinations)
            ):
                return 0
            kept = plan.kept_pages
            indices = torch.tensor(
                [
                    p * self.hot_page_size + i
                    for p in kept
                    for i in range(self.hot_page_size)
                ],
                dtype=torch.int64,
                device=self.device,
            )
            new = FrequencyBuffer.empty(old.locations[indices])
            new.tokens.copy_(
                torch.tensor(plan.tokens, dtype=torch.int32, device=self.device)
            )
            new.frequency.copy_(pack_frequency(plan.frequency, device=self.device))
            src = torch.tensor(plan.sources, dtype=torch.int64, device=self.device)
            dst = torch.tensor(plan.destinations, dtype=torch.int64, device=self.device)
            self._move_kvduo_hot_entries(
                layer_id, old.locations[src].long(), old.locations[dst].long()
            )
        # Keep the replacement alive even if a CUDA completion check raises.
        self.kvduo_hot_buffers[owner] = new
        if not old.tokens.is_cuda:
            self._publish_kvduo_frequency_buffer(owner, new)
            self.kvduo_hot_growth_events[owner] = 0
        self.kvduo_req_hot_capacity[owner] = new.tokens.numel()
        self._kvduo_req_layer_pages[owner] = [pages[p] for p in kept]
        released = [page for p, page in enumerate(pages) if p not in kept]
        for start in released:
            del self._kvduo_hot_page_owners[start]
        free_kvduo_whole_pages(
            self.token_to_kv_pool_allocator.hisparse_attn_allocator,
            released,
            self.page_size,
        )
        return len(released) * self.page_size

    def reclaim_kvduo_hot_pages(self, num_tokens):
        if not self.enable_mixed_residency or num_tokens <= 0:
            return 0
        self._wait_kvduo_hot_merge_boundary()
        layers = self.mem_pool_device.layer_num
        owners = [
            (layer, req)
            for req in self._kvduo_hot_request_order
            for layer in range(layers)
        ]
        hand = self._kvduo_hand % len(owners) if owners else 0
        eligible = [self._kvduo_can_merge(owner) for owner in owners]
        # Agree once on the complete scan, including empty/ineligible ranks.
        # No request lifecycle or decode updates interleave at this boundary.
        snapshot = tuple(
            (
                owner,
                self._kvduo_request_generation[owner[1]],
                len(self._kvduo_req_layer_pages.get(owner, ())),
                can_merge,
            )
            for owner, can_merge in zip(owners, eligible)
        )
        if not self._kvduo_consensus((hand, num_tokens, self.min_hot_pages, snapshot)):
            return 0
        candidate_count = sum(eligible)
        if not candidate_count:
            return 0
        # No decode interleaves: a first rotation decays threes to ones; the
        # second can reduce every eligible object to its minimum. Further
        # rotations cannot release anything and must not deadlock the scheduler.
        reclaimed = 0
        for _ in range(2 * len(owners)):
            self._kvduo_hand %= len(owners)
            index = self._kvduo_hand
            owner = owners[index]
            self._kvduo_hand = (index + 1) % len(owners)
            if eligible[index]:
                reclaimed += self._merge_kvduo_hot_pages(*owner)
                # Merge retains its TP plan check; agreed capacity changes are
                # the only eligibility changes possible during this scan.
                eligible[index] = (
                    len(self._kvduo_req_layer_pages[owner]) > self.min_hot_pages
                )
                candidate_count -= not eligible[index]
            if reclaimed >= num_tokens or not candidate_count:
                break
        return reclaimed

    def _kvduo_can_merge(self, owner) -> bool:
        layer, req = owner
        request = self._active_kvduo_reqs.get(req)
        return (
            request is not None
            and not getattr(request, "hisparse_staging", False)
            and not (
                callable(getattr(request, "finished", None)) and request.finished()
            )
            and len(self._kvduo_req_layer_pages.get(owner, ())) > self.min_hot_pages
            and self._kvduo_hot_generation.get(owner)
            == self._kvduo_request_generation[req]
            and not any(
                (l, r) == owner for l, r, _ in self._kvduo_hot_pressure_protected
            )
        )

    def _reclaim_kvduo_emergency_hot_pages(self, num_tokens):
        return self.reclaim_kvduo_hot_pages(num_tokens)

    def _init_kvduo_capacity_state(self, requests: int) -> None:
        self._kvduo_hot_request_order = []
        self._kvduo_next_prefill_pages = 0
        self._kvduo_next_prefill_key = None
        self._kvduo_prefill_lifecycle = {}
        self._kvduo_prefill_host_reservations = {}
        # Remaining FULL footprint of admitted chunks, including first decode.
        self._kvduo_prefill_future_pages = {}
        self._kvduo_prefill_previous_future = {}
        self._kvduo_prefill_new_host_reservations = set()
        self._kvduo_prefill_generation = 0
        self._kvduo_prefill_reservation_requests = {}
        self._kvduo_hand = 0
        self._kvduo_hot_generation = {}
        self._kvduo_request_generation = [0] * requests
        self._kvduo_hot_tail_starts = {}
        self._kvduo_released_full_offsets = {}
        self._kvduo_tail_retry_pages = {}
        self._kvduo_tail_cleanup_pending = {}
        self._kvduo_decode_order = {}
        self._kvduo_decode_sequence = 0
        self._kvduo_prefill_reservations = {}
        self._kvduo_prefill_next_reservation = 0

    def _reset_kvduo_prefill_request(self, req_idx: int) -> None:
        self._kvduo_released_full_offsets.pop(req_idx, None)
        self._kvduo_hot_tail_starts.pop(req_idx, None)
        self._kvduo_tail_retry_pages.pop(req_idx, None)
        self._kvduo_tail_cleanup_pending.pop(req_idx, None)
        self._kvduo_request_generation[req_idx] += 1
        self._kvduo_decode_order.pop(req_idx, None)
        if req_idx in self._kvduo_hot_request_order:
            removed = self._kvduo_hot_request_order.index(req_idx)
            layers = self.mem_pool_device.layer_num
            if removed * layers < self._kvduo_hand:
                self._kvduo_hand = max(removed * layers, self._kvduo_hand - layers)
            self._kvduo_hot_request_order.remove(req_idx)
        for owner in list(self._kvduo_hot_generation):
            if owner[1] == req_idx:
                del self._kvduo_hot_generation[owner]

    def _kvduo_common_available(self, pages: int) -> int:
        if getattr(self, "tp_world_size", 1) <= 1:
            return pages
        value = torch.tensor(pages, dtype=torch.int64)
        torch.distributed.all_reduce(
            value, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
        )
        return int(value.item())

    def _kvduo_optional_growth_resources(self, reserved, applicants):
        """Check exact applicant bits, protection, and common free pages once."""
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        free = allocator.available_size() // self.page_size
        if getattr(self, "tp_world_size", 1) <= 1:
            return free, True
        requests = len(self._kvduo_request_generation)
        objects = self.mem_pool_device.layer_num * requests
        words = [0] * ((objects + 62) // 63)
        for (layer, req), pages in applicants.items():
            if (
                pages != 1
                or not 0 <= req < requests
                or not 0 <= layer < self.mem_pool_device.layer_num
            ):
                raise ValueError(
                    "Optional growth must request one page per valid owner"
                )
            index = layer * requests + req
            words[index // 63] |= 1 << (index % 63)
        fields = [reserved, *words]
        packet = torch.tensor(
            [free, *(item for value in fields for item in (value, -value))],
            dtype=torch.int64,
        )
        torch.distributed.all_reduce(
            packet, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
        )
        consistent = all(
            packet[i].item() == -packet[i + 1].item() for i in range(1, len(packet), 2)
        )
        return int(packet[0]), consistent

    def _kvduo_consensus(self, value) -> bool:
        """Every TP rank must publish the same request/capacity batch."""
        if getattr(self, "tp_world_size", 1) <= 1:
            return True
        values = [None] * self.tp_world_size
        torch.distributed.all_gather_object(values, value, group=self.tp_group)
        return all(other == value for other in values)

    def _kvduo_prefill_key(self, req):
        # Prefill may not have a request-pool slot yet. Its own monotonic
        # incarnation is bound to the slot generation when staging publishes.
        if not hasattr(req, "kvduo_prefill_generation"):
            self._kvduo_prefill_generation += 1
            req.kvduo_prefill_generation = self._kvduo_prefill_generation
        key = (getattr(req, "rid", None), req.kvduo_prefill_generation)
        self._kvduo_prefill_lifecycle.setdefault(key, req)
        return key

    def protect_kvduo_prefill_plan(self, req, end: int) -> None:
        """Protect only the scheduler's next plan, including blocked independent pools."""
        key = self._kvduo_prefill_key(req)
        if self._kvduo_next_prefill_key not in (None, key):
            return
        self._kvduo_next_prefill_key = key
        pages = self.kvduo_prefill_pages(len(req.prefix_indices), end)
        self._kvduo_next_prefill_pages = max(0, pages)

    def kvduo_residency_mode(self, req_idx: int) -> KVDuoResidencyMode:
        return (
            KVDuoResidencyMode.HOT_RESIDENT
            if self._mixed_slots[req_idx]
            else KVDuoResidencyMode.FULL_RESIDENT
        )

    def _publish_kvduo_tail_progress(self, progress) -> None:
        for req_idx, generation, cursor, retry in progress:
            self._kvduo_hot_tail_starts[req_idx] = (generation, cursor)
            if retry:
                self._kvduo_tail_retry_pages[req_idx] = set(retry)
                self._kvduo_tail_cleanup_pending[req_idx] = generation
            else:
                self._kvduo_tail_retry_pages.pop(req_idx, None)
                self._kvduo_tail_cleanup_pending.pop(req_idx, None)

    def _cleanup_kvduo_old_tails(self) -> None:
        if not self.enable_mixed_residency:
            return
        pending = self._kvduo_tail_cleanup_pending
        for req_idx, generation in list(pending.items()):
            if (
                req_idx not in self._active_kvduo_reqs
                or generation != self._kvduo_request_generation[req_idx]
                or self.kvduo_residency_mode(req_idx)
                is not KVDuoResidencyMode.HOT_RESIDENT
            ):
                pending.pop(req_idx)
                record = self._kvduo_hot_tail_starts.get(req_idx)
                if record is not None and record[0] == generation:
                    self._kvduo_hot_tail_starts.pop(req_idx, None)
                    self._kvduo_tail_retry_pages.pop(req_idx, None)
        if not pending:
            return
        # Poll queued turnover/retry jobs; all ranks defer an unfinished backup.
        # The reclaimer orders old readers with a stream wait, rather than
        # requiring an idle producer (which overlap scheduling may never offer).
        ready = (
            not getattr(self, "_has_pending_backup", False)
            or self._backup_done_event.query()
        )
        if (
            not self._kvduo_consensus((tuple(sorted(pending.items())), ready))
            or not ready
        ):
            return
        owners = set(pending)
        self.reclaim_kvduo_full_pages(
            self.page_size, request_indices=owners, release_all=True, tail_cleanup=True
        )

    def begin_kvduo_schedule_boundary(self) -> None:
        state_cache = getattr(self, "kvduo_prefill_host_cache", None)
        if state_cache is not None:
            state_cache.poll()
        self._kvduo_next_prefill_pages = 0
        self._kvduo_next_prefill_key = None
        self._cleanup_kvduo_old_tails()

    def kvduo_prefill_batch_allocated(self) -> None:
        # Allocations have completed. They now occupy physical pages rather
        # than requiring an additional next-plan protection.
        self._kvduo_next_prefill_pages = 0
        self._kvduo_next_prefill_key = None

    def _kvduo_prefill_plan_uncovered_pages(self) -> int:
        key = self._kvduo_next_prefill_key
        covered = sum(
            self._kvduo_prefill_reservations[token][0]
            for token, request in self._kvduo_prefill_reservation_requests.items()
            if request == key and token in self._kvduo_prefill_reservations
        )
        covered += self._kvduo_prefill_future_pages.get(key, 0)
        return max(0, self._kvduo_next_prefill_pages - covered)

    def _end_kvduo_prefill_lifecycle(self, req_idx, request=None) -> None:
        if request is None:
            request = self._active_kvduo_reqs.get(req_idx)
        state_cache = getattr(self, "kvduo_prefill_host_cache", None)
        if state_cache is not None and request is not None:
            state_cache.end_request(request)
        for key, req in list(self._kvduo_prefill_lifecycle.items()):
            if req is not request:
                continue
            for token, prefill in list(
                self._kvduo_prefill_reservation_requests.items()
            ):
                if prefill == key:
                    self.finish_kvduo_prefill_reservation(token, cancelled=True)
            self._release_kvduo_prefill_host(key)
            self._kvduo_prefill_future_pages.pop(key, None)
            del self._kvduo_prefill_lifecycle[key]
        if request is not None and hasattr(request, "kvduo_prefill_generation"):
            del request.kvduo_prefill_generation

    def end_kvduo_prefill(self, req: Req) -> None:
        if not self.enable_mixed_residency:
            return
        self._end_kvduo_prefill_lifecycle(req.req_pool_idx, req)
        self.release_kvduo_match_refs(req)

    def _kvduo_optional_protected_pages(self) -> int:
        return (
            self._kvduo_reserved_prefill_pages()
            + self._kvduo_mandatory_hot_pages()
            + self._kvduo_prefill_plan_uncovered_pages()
        )

    def _kvduo_reserved_prefill_pages(self) -> int:
        return sum(
            reservation[0]
            for reservation in getattr(self, "_kvduo_prefill_reservations", {}).values()
        ) + sum(getattr(self, "_kvduo_prefill_future_pages", {}).values())

    def _kvduo_mandatory_hot_pages(self) -> int:
        return sum(
            max(
                0,
                self.min_hot_pages
                - int(self.kvduo_req_hot_capacity[layer, req]) // self.hot_page_size,
            )
            for req in getattr(self, "_kvduo_pending_hot_minimum", ())
            if req in self._active_kvduo_reqs
            for layer in range(self.mem_pool_device.layer_num)
        )

    def kvduo_prefill_pages(self, prefix_len: int, end: int) -> int:
        """Match alloc_extend's compression rounding and partial-page reuse."""
        prefix = prefix_len // self.compress_ratio
        end = end // self.compress_ratio
        return (end + self.page_size - 1) // self.page_size - (
            prefix + self.page_size - 1
        ) // self.page_size

    @contextmanager
    def kvduo_prefill_reservation_scope(self):
        before = set(self._kvduo_prefill_reservations)
        try:
            yield
        finally:
            for token in set(self._kvduo_prefill_reservations) - before:
                self.finish_kvduo_prefill_reservation(token, cancelled=True)

    def finish_kvduo_prefill_reservation(self, token: int, *, cancelled: bool) -> None:
        reservation = self._kvduo_prefill_reservations.pop(token, None)
        if reservation is None:
            return
        prefill_key = self._kvduo_prefill_reservation_requests.pop(token, None)
        previous = self._kvduo_prefill_previous_future.pop(token, 0)
        new_host = token in self._kvduo_prefill_new_host_reservations
        self._kvduo_prefill_new_host_reservations.discard(token)
        if cancelled and prefill_key is not None:
            if previous:
                self._kvduo_prefill_future_pages[prefill_key] = previous
            else:
                self._kvduo_prefill_future_pages.pop(prefill_key, None)
        if cancelled and prefill_key is not None and new_host:
            self._release_kvduo_prefill_host(prefill_key)
            request = self._kvduo_prefill_lifecycle[prefill_key]
            if getattr(request, "kvduo_prefill_generation", None) == prefill_key[1]:
                del request.kvduo_prefill_generation

    def consume_kvduo_prefill_reservation(self, token: int, pages: int) -> None:
        reservation = self._kvduo_prefill_reservations[token]
        if not 0 <= pages <= reservation[0]:
            raise RuntimeError("KVDuo prefill allocation exceeds its reservation")
        reservation[0] -= pages

    def check_kvduo_prefill_total_capacity(self, req) -> bool:
        """Reject impossible prompts even when scheduler headroom blocks admission."""
        prompt_end = len(req.full_untruncated_fill_ids)
        remaining = getattr(
            getattr(req, "sampling_params", None), "max_new_tokens", 2
        ) - len(getattr(req, "output_ids", ()))
        full_end = prompt_end + int(remaining > 1)
        physical = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        if self.kvduo_prefill_pages(0, full_end) > physical.size // self.page_size:
            req.set_finish_with_abort(
                "KVDuo FULL prefill and first decode exceed physical pool capacity; "
                "reduce the prompt length or increase KV pool capacity"
            )
            return False
        host_slots = prompt_end // self.compress_ratio
        host_slots = (
            (host_slots + self.page_size - 1) // self.page_size * self.page_size
        )
        if host_slots > self.mem_pool_host.size:
            req.set_finish_with_abort(
                "KVDuo prompt exceeds total Host main-KV capacity "
                f"(required={host_slots}, capacity={self.mem_pool_host.size} slots); "
                "reduce the prompt length or increase kvduo host_size"
            )
            return False
        return True

    def reserve_kvduo_prefill(
        self,
        pages: int,
        *,
        tree_cache=None,
        prefix_indices: torch.Tensor | None = None,
        prefill_req=None,
        prefill_end: int | None = None,
    ) -> int | None:
        self._wait_kvduo_hot_merge_boundary()
        prefill_key = (
            self._kvduo_prefill_key(prefill_req) if prefill_req is not None else None
        )
        needs_host = prefill_req is not None and hasattr(
            prefill_req, "full_untruncated_fill_ids"
        )
        total = pages
        if needs_host:
            remaining_output = getattr(
                getattr(prefill_req, "sampling_params", None), "max_new_tokens", 2
            ) - len(getattr(prefill_req, "output_ids", ()))
            # Prefill itself emits one token, including after request retraction.
            full_end = len(prefill_req.full_untruncated_fill_ids) + int(
                remaining_output > 1
            )
            prompt_end = len(prefill_req.full_untruncated_fill_ids)
            execution_end = prompt_end if prefill_end is None else prefill_end
            if not len(prefill_req.prefix_indices) <= execution_end <= prompt_end:
                raise ValueError("Invalid KVDuo prefill execution boundary")
            # Intermediate chunks reserve only their execution footprint. The
            # last chunk additionally holds the first decode allocation.
            if execution_end == prompt_end:
                total = max(
                    pages,
                    self.kvduo_prefill_pages(len(prefill_req.prefix_indices), full_end),
                )
            if not self.check_kvduo_prefill_total_capacity(prefill_req):
                return None
        new_host_reservation = (
            needs_host and prefill_key not in self._kvduo_prefill_host_reservations
        )
        if needs_host and not self._reserve_kvduo_prefill_host(
            prefill_req, prefill_key
        ):
            return None
        if needs_host:
            self.protect_kvduo_prefill_plan(prefill_req, execution_end)
        token = None
        future = self._kvduo_prefill_future_pages
        previous = future.pop(prefill_key, 0)
        try:
            token = self._reserve_kvduo_prefill_physical(
                total,
                tree_cache=tree_cache,
                prefix_indices=prefix_indices,
                prefill_req=prefill_req,
            )
            if token is not None and prefill_key is not None:
                self._kvduo_prefill_reservations[token][0] = pages
                future[prefill_key] = total - pages
                self._kvduo_prefill_previous_future[token] = previous
                if new_host_reservation or (not needs_host and not previous):
                    self._kvduo_prefill_new_host_reservations.add(token)
            return token
        finally:
            if token is None:
                if previous:
                    future[prefill_key] = previous
                if new_host_reservation:
                    self._release_kvduo_prefill_host(prefill_key)

    def _kvduo_next_decode_pages(self) -> int:
        pages = 0
        for req in self._active_kvduo_reqs.values():
            length = getattr(req, "kv_committed_len", None)
            if length is None or getattr(req, "hisparse_staging", False):
                continue
            if callable(getattr(req, "finished", None)) and req.finished():
                continue
            key = (
                getattr(req, "rid", None),
                getattr(req, "kvduo_prefill_generation", None),
            )
            # First-decode holds are already included in reserved prefill pages.
            held = self._kvduo_prefill_future_pages.get(key, 0)
            pages += max(0, self.kvduo_prefill_pages(length, length + 1) - held)
        return pages

    def _reserve_kvduo_prefill_physical(
        self,
        pages: int,
        *,
        tree_cache=None,
        prefix_indices: torch.Tensor | None = None,
        prefill_req=None,
    ) -> int | None:
        """Admit against actual free pages after mandatory capacity and holds.

        Decode-owned FULL reclaim may leave surplus for the next boundary;
        this path never increases that reclaim to fit a waiting prefill.
        """
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        prefill_key = (
            self._kvduo_prefill_key(prefill_req) if prefill_req is not None else None
        )
        required = (
            pages
            + self._kvduo_reserved_prefill_pages()
            + self._kvduo_mandatory_hot_pages()
            + self._kvduo_next_decode_pages()
        )
        self._reclaim_kvduo_radix_pages(required, tree_cache)
        shortage = max(
            0,
            required
            - self._kvduo_common_available(
                allocator.available_size() // self.page_size
            ),
        )
        ready = shortage == 0
        if not self._kvduo_consensus((ready, required)) or not ready:
            return None
        token = self._kvduo_prefill_next_reservation
        self._kvduo_prefill_next_reservation += 1
        if prefill_key is not None:
            self._kvduo_prefill_reservation_requests[token] = prefill_key
        self._kvduo_prefill_reservations[token] = [pages]
        return token

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
            self._publish_kvduo_decode_write(out_cache_loc, req_pool_indices, seq_lens)
            return
        else:
            self._eager_backup_previous_token(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )

        if not self.is_dsv4_hisparse:
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
        required_host_slots = sum(
            max(0, end - int(self.req_to_host_pool_allocated_len[req_idx]))
            for req_idx, _, end in page_jobs
        )
        if not self._kvduo_host_capacity_ready(required_host_slots):
            raise RuntimeError("KVDuo decode backup exceeds available Host capacity")
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
        schedule_stream = device_module.current_stream(self.device)
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
            self._backup_done_event.record(self.decode_backup_stream)
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
        for req_idx, _, _ in page_jobs:
            if self._mixed_slots[req_idx]:
                self._kvduo_tail_cleanup_pending[req_idx] = (
                    self._kvduo_request_generation[req_idx]
                )

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
        schedule_stream = device_module.current_stream(self.device)
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
            self._backup_done_event.record(self.decode_backup_stream)
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
        self._backup_done_event.wait(device_module.current_stream(self.device))
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

    def _publish_kvduo_decode_write(
        self,
        out_cache_loc: torch.Tensor,
        req_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Publish a newly generated KV identity in the unified address space."""
        publish_kvduo_full_write(
            out_cache_loc,
            self.full_generation,
            self.full_data_version,
            self.full_host_version,
            self.full_last_touch,
            self.full_touch_clock,
            lookup=self.req_to_full_lookup,
            req_indices=req_indices,
            positions=seq_lens,
            decode_compress_ratio=self.compress_ratio,
        )

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
        if self.enable_mixed_residency:
            self._end_kvduo_prefill_lifecycle(req.req_pool_idx, req)
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
            self.kvduo_hot_growth_events[:, req.req_pool_idx] = 0
            self._reset_kvduo_hot_metadata(req.req_pool_idx)
        self._mixed_slots[req.req_pool_idx] = False
        if self.enable_mixed_residency:
            self._active_kvduo_reqs.pop(req.req_pool_idx, None)
            self._kvduo_pending_hot_minimum.discard(req.req_pool_idx)
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
            device_module.current_stream(self.device).wait_stream(
                self.decode_producer_stream
            )
        self.wait_for_pending_backup()
        retained = None
        if self.enable_mixed_residency:
            state_cache = getattr(self, "kvduo_prefill_host_cache", None)
            if state_cache is not None:
                retained = state_cache.backup_finished(req)
            self._end_kvduo_prefill_lifecycle(req.req_pool_idx, req)

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
        if self.enable_mixed_residency:
            req.kvduo_mixed_residency = is_mixed
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
            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                owned_compressed_locs
            ] = 0
            if full_hi.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(full_hi)
        elif not self.enable_mixed_residency:
            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = 0
        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.req_pool_idx],
        )
        if self.enable_mixed_residency:
            if retained is None:
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
            self.kvduo_resolver_stats[:, req.req_pool_idx, :] = 0
            self.kvduo_hot_growth_events[:, req.req_pool_idx] = 0
            self._reset_kvduo_hot_metadata(req.req_pool_idx)
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
            self._kvduo_pending_hot_minimum.discard(req.req_pool_idx)

    def reclaim_kvduo_full_pages(
        self,
        num_tokens: int,
        *,
        reserve_hot_minimum: bool = False,
        request_indices=None,
        release_all=False,
        tail_cleanup=False,
    ) -> int:
        """Demote eligible request-owned FULL pages using the configured policy.

        The host copy is complete before requests leave staging. This control
        path runs under allocation pressure or sealed-tail cleanup; steady-state
        full/hot/host resolution remains entirely device-driven and graph-safe.
        """
        if not self.enable_mixed_residency or num_tokens <= 0:
            return 0
        if self.decode_producer_stream is not None:
            device_module.current_stream(self.device).wait_stream(
                self.decode_producer_stream
            )
        self.wait_for_pending_backup()
        tree_cache = getattr(self, "kvduo_tree_cache", None)
        plan_prefix = getattr(tree_cache, "kvduo_reclaimable_prefix", None)
        can_transfer = (
            not tail_cleanup
            and callable(plan_prefix)
            and getattr(self, "_kvduo_pressure_protected", None) is None
        )
        prefix_transfers = {}
        tail_progress = {}
        candidate_pages = []
        candidate_meta = []
        for req_idx, req in sorted(self._active_kvduo_reqs.items()):
            if request_indices is not None and req_idx not in request_indices:
                continue
            if getattr(req, "hisparse_staging", False) or (
                callable(getattr(req, "finished", None)) and req.finished()
            ):
                continue
            protected_len = req.cache_protected_len
            if can_transfer:
                length = plan_prefix(req)
                if length:
                    tree_len = tree_cache.kvduo_prefix_length(req)
                    protected_len = tree_len - length
                    prefix_transfers[req_idx] = (length, tree_len)
            allocated_len = req.kv.kv_allocated_len
            compressed_len = self.host_token_len(allocated_len)
            prefix_len = self.host_token_len(protected_len)
            generation = self._kvduo_request_generation[req_idx]
            released_record = self._kvduo_released_full_offsets.get(req_idx)
            released_offsets = (
                released_record[1]
                if released_record is not None and released_record[0] == generation
                else set()
            )
            num_pages = (compressed_len + self.page_size - 1) // self.page_size
            tail_first = max(0, num_pages - self.tail_protected_pages)
            first_owned_page = (prefix_len + self.page_size - 1) // self.page_size
            start = first_owned_page * self.page_size
            host_valid_end = (
                self._kvduo_host_valid_len[req_idx] // self.page_size * self.page_size
            )
            end = min(tail_first * self.page_size, host_valid_end)
            if tail_cleanup:
                record = self._kvduo_hot_tail_starts.get(req_idx)
                if (
                    record is None
                    or record[0] != self._kvduo_request_generation[req_idx]
                    or not self._mixed_slots[req_idx]
                ):
                    continue
                # Advance over newly expired pages once, but retain blocked
                # offsets independently of the high-water mark for later retry.
                expired_end = min(
                    tail_first * self.page_size,
                    compressed_len // self.page_size * self.page_size,
                )
                offsets = set(self._kvduo_tail_retry_pages.get(req_idx, ()))
                offsets.update(range(record[1], expired_end, self.page_size))
                offsets.difference_update(released_offsets)
                tail_progress[req_idx] = (
                    record[0],
                    max(record[1], expired_end),
                    offsets,
                )
                candidates = sorted(
                    offset for offset in offsets if start <= offset < end
                )
                if not candidates:
                    continue
            else:
                if end <= start:
                    continue
                candidates = []
                cursor = start
                intervals = released_record[2] if released_offsets else ()
                for lo, hi in intervals:
                    if hi <= cursor:
                        continue
                    if lo >= end:
                        break
                    candidates.extend(range(cursor, min(lo, end), self.page_size))
                    cursor = max(cursor, hi)
                    if cursor >= end:
                        break
                candidates.extend(range(cursor, end, self.page_size))
            if not candidates:
                continue
            # Translate only surviving candidate intervals, not the long prefix
            # whose physical mappings were removed at an earlier boundary.
            positions = torch.tensor(candidates, dtype=torch.int64, device=self.device)
            token_positions = (
                positions[:, None] * self.compress_ratio
                + torch.arange(self.page_size * self.compress_ratio, device=self.device)
            ).flatten()
            pages = self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req_idx, token_positions]
            ).view(-1, self.page_size)
            candidate_pages.append(pages)
            candidate_meta.extend((req_idx, offset) for offset in candidates)

        def progress_after(discarded):
            remaining = {req: set(record[2]) for req, record in tail_progress.items()}
            for req_idx, offset in discarded:
                remaining[req_idx].discard(offset)
            return tuple(
                (req_idx, generation, cursor, tuple(sorted(remaining[req_idx])))
                for req_idx, (generation, cursor, _) in sorted(tail_progress.items())
            )

        if not candidate_pages:
            if tail_cleanup:
                progress = progress_after(())
                if self._kvduo_consensus(((), progress)):
                    self._publish_kvduo_tail_progress(progress)
            else:
                self._kvduo_consensus(())
            return 0
        logical_pages = torch.cat(candidate_pages, dim=0)
        physical_pages, page_touch, valid = summarize_kvduo_full_pages(
            logical_pages,
            self.mem_pool_device.full_to_hisparse_device_index_mapping,
            self.full_data_version,
            self.full_host_version,
            self.full_last_touch,
            getattr(self, "_kvduo_pressure_protected", None),
        )

        minimum_costs = {}
        owners = sorted({req_idx for req_idx, _ in candidate_meta})
        request_policy = self.full_reclaim_policy == "request"
        if reserve_hot_minimum or (request_policy and not release_all):
            # The first FULL demotion needs minimum HOT capacity in EVERY
            # logical layer. An allocation cannot count these as free space,
            # or make a fully resident request mixed without room to execute.
            for req_idx in owners:
                minimum_costs[req_idx] = (
                    0
                    if req_idx in self._kvduo_pending_hot_minimum
                    else sum(
                        max(
                            0,
                            self.min_hot_pages
                            - int(self.kvduo_req_hot_capacity[layer, req_idx])
                            // self.hot_page_size,
                        )
                        for layer in range(self.mem_pool_device.layer_num)
                    )
                )
        owner_indices = {req_idx: i for i, req_idx in enumerate(owners)}
        page_owners = [owner_indices[req_idx] for req_idx, _ in candidate_meta]
        costs = [minimum_costs.get(req_idx, 0) for req_idx in owners]
        needed = (num_tokens + self.page_size - 1) // self.page_size
        if request_policy:
            admission_order = []
            for req_idx in owners:
                record = self._kvduo_decode_order.get(req_idx)
                admission_order.append(
                    record[1]
                    if record is not None
                    and record[0] == self._kvduo_request_generation[req_idx]
                    else torch.iinfo(torch.int64).max
                )
            selected = select_kvduo_full_requests(
                valid,
                page_owners,
                costs,
                admission_order,
                needed,
                release_all=release_all,
                mixed_owners=[self._mixed_slots[req_idx] for req_idx in owners],
            )
        else:
            selected = select_kvduo_full_pages(
                page_touch,
                valid,
                page_owners,
                costs,
                needed,
                release_all=release_all,
            )

        selection = tuple(candidate_meta[i] for i in selected)
        progress = ()
        if tail_cleanup:
            # A pressure reclaim may already have removed a deferred page.
            # Retire absent mappings too, without counting them as newly freed.
            absent = (
                (
                    (physical_pages == 0).all(dim=1)
                    & (logical_pages >= 0).all(dim=1)
                    & (logical_pages < self.full_data_version.numel()).all(dim=1)
                )
                .cpu()
                .tolist()
            )
            retired = selection + tuple(
                meta for meta, missing in zip(candidate_meta, absent) if missing
            )
            progress = progress_after(retired)
            if not self._kvduo_consensus((selection, progress)):
                return 0
        elif not self._kvduo_consensus(selection):
            return 0
        if not selected:
            if tail_cleanup:
                self._publish_kvduo_tail_progress(progress)
            return 0
        selected_gpu = torch.tensor(selected, dtype=torch.int64, device=self.device)
        selected_logical = logical_pages[selected_gpu]
        selected_physical = physical_pages[selected_gpu]
        published = device_module.Event() if selected_logical.is_cuda else None
        first_holes = {}
        for i in selected:
            req_idx, start = candidate_meta[i]
            first_holes[req_idx] = min(first_holes.get(req_idx, start), start)
        transfers = tuple(
            (req_idx, prefix_transfers[req_idx][0])
            for req_idx, start in sorted(first_holes.items())
            if req_idx in prefix_transfers
            and start * self.compress_ratio < prefix_transfers[req_idx][1]
        )
        if can_transfer and not self._kvduo_consensus(transfers):
            return 0
        for req_idx, length in transfers:
            tree_cache.detach_kvduo_prefix(self._active_kvduo_reqs[req_idx], length)
        self.mem_pool_device.full_to_hisparse_device_index_mapping[
            selected_logical.flatten()
        ] = 0
        for req_idx, start in first_holes.items():
            req = self._active_kvduo_reqs[req_idx]
            if not self._mixed_slots[req_idx]:
                compressed_len = self.host_token_len(req.kv.kv_allocated_len)
                pages = (compressed_len + self.page_size - 1) // self.page_size
                self._kvduo_hot_tail_starts[req_idx] = (
                    self._kvduo_request_generation[req_idx],
                    max(0, pages - self.tail_protected_pages) * self.page_size,
                )
            if req_idx not in self._kvduo_hot_request_order:
                self._kvduo_hot_request_order.append(req_idx)
            req.kvduo_mixed_residency = True
            req.kvduo_radix_insert_len = min(
                getattr(req, "kvduo_radix_insert_len", start * self.compress_ratio),
                start * self.compress_ratio,
            )
            self._mixed_slots[req_idx] = True
            if not tail_cleanup:
                self._kvduo_pending_hot_minimum.add(req_idx)
        if published is not None:
            published.record(device_module.current_stream(self.device))
            published.synchronize()
        self.token_to_kv_pool_allocator.free_hisparse_indices(
            selected_physical.flatten()
        )
        newly_released = {}
        for req_idx, offset in selection:
            newly_released.setdefault(req_idx, []).append(offset)
            generation = self._kvduo_request_generation[req_idx]
            record = self._kvduo_released_full_offsets.get(req_idx)
            if record is None or record[0] != generation:
                record = (generation, set(), [])
                self._kvduo_released_full_offsets[req_idx] = record
            record[1].add(offset)
        for req_idx, offsets in newly_released.items():
            intervals = self._kvduo_released_full_offsets[req_idx][2]
            segments = sorted(
                intervals + [(offset, offset + self.page_size) for offset in offsets]
            )
            intervals.clear()
            for lo, hi in segments:
                if intervals and intervals[-1][1] >= lo:
                    intervals[-1] = (intervals[-1][0], max(intervals[-1][1], hi))
                else:
                    intervals.append((lo, hi))
        if tail_cleanup:
            self._publish_kvduo_tail_progress(progress)
        elif request_policy:
            logger.info(
                "KVDuo FULL request reclaim completed: requests=%d, "
                "released_physical_blocks=%d, physical_block_tokens=%d",
                len(first_holes),
                len(selected),
                self.page_size,
            )
        return len(selected) * self.page_size

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        *,
        touch_clock_advanced: bool = False,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices."""
        if not self.is_dsv4_hisparse:
            layer_id -= getattr(self.mem_pool_device, "start_layer", 0)
        num_reqs = req_pool_indices.size(0)

        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]

        if self.enable_mixed_residency:
            from sglang.kernels.ops.kvcache.kvduo_frequency import resolve

            if not touch_clock_advanced:
                self.full_touch_clock.add_(1)
            resolve(
                self,
                layer_id,
                req_pool_indices,
                compressed_seq_lens,
                top_k_result,
                top_k_indices,
            )
            return top_k_indices

        swap_in_fn = (
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
        )
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
            page_size=1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            req_hot_buffer_sizes=self.req_device_buffer_size_gpu,
            resolver_scratch=self.resolver_scratch,
        )
        return top_k_indices
