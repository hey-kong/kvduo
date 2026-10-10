# to be combined with the sparse coordinator class and sparse algorithm family

import hashlib
import logging
import math
from contextlib import contextmanager
from typing import List, NamedTuple, Sequence, Union

import torch

from sglang.kernels.ops.kvcache.hisparse import (
    allocate_hisparse_resolver_scratch,
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
    rollback_kvduo_hot_metadata,
    transfer_cache_dsv4_mla,
)
from sglang.kernels.ops.kvcache.kvduo_full_reclaim import (
    select_kvduo_full_pages,
    summarize_kvduo_full_pages,
)
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
    kvduo_shrink_target,
    prepare_kvduo_hot_merge,
)
from sglang.srt.mem_cache.sparsity.core.kvduo_capacity_modes import (
    KVDuoRecoveryObject,
    plan_kvduo_recovery,
)
from sglang.srt.mem_cache.sparsity.core.kvduo_prefill_rollback import (
    KVDuoPrefillObject,
    KVDuoPrefillPolicy,
    plan_kvduo_prefill_rollback,
)
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
        prefill_policy: KVDuoPrefillPolicy | None = None,
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
        self.kvduo_prefill_policy = prefill_policy or KVDuoPrefillPolicy()
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
            # A KVDuo physical page is the same page number across every KV
            # storage layer.  A HOT(request, logical_layer) owner may use all
            # of those fragments through global flat locations.
            self.hot_storage_layers = self.mem_pool_device.layer_num
            self.hot_page_size = self.hot_storage_layers * self.page_size
            self.min_hot_pages = (top_k + self.hot_page_size - 1) // self.hot_page_size
            if self.min_hot_pages > 8:
                raise ValueError(
                    f"KVDuo top_k={top_k} needs {self.min_hot_pages} full pages; "
                    "the per-layer hot-cache limit is 8 pages"
                )
            # Fixed graph metadata covers the promised 1K..8K physical pages.
            max_hot_capacity = 8 * self.hot_page_size
            self.device_buffer_size = max_hot_capacity

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        self.resolver_scratch = allocate_hisparse_resolver_scratch(
            max_num_req_slots,
            self.top_k,
            self.device_buffer_size,
            self.swap_in_block_size,
            device,
        )
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
            # Diagnostic counters do not decide HOT capacity.
            self.kvduo_resolver_stats = torch.zeros(
                (self.mem_pool_device.layer_num, max_num_req_slots, 2),
                dtype=torch.int32,
                device=device,
            )
            self._init_kvduo_prefill_rollback_state(max_num_req_slots)
            # Capture/warmup forwards use synthetic requests, not observations.
            # A graph reads this shared flag at execution time, so replay can
            # resume observation without recapturing or changing EMA pointers.
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
            self.device_buffer_size,
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
        """
        if not self.enable_mixed_residency:
            return match_result
        for identity in getattr(req, "kvduo_host_prefix_records", ()):
            if identity in self.host_prefix_cache.records:
                self.host_prefix_cache.release(identity, req.rid)
        req.kvduo_host_prefix_records = set()
        gpu_len = len(match_result.device_indices)
        page_tokens = self.page_size * self.compress_ratio
        fill_ids = req.get_fill_ids()
        max_prefix = req._compute_max_prefix_len(len(fill_ids))
        if max_prefix_len is not None:
            # Preserve the cache's own match cap. In particular, DeepSeek V4's
            # unified SWA cache holds back a trailing window for re-prefill;
            # KVDuo must not extend a main/C4-only host hit into that window.
            max_prefix = min(max_prefix, max_prefix_len)
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
            self._host_prefix_identities(req, ordinal, fill_ids)[
                ordinal
            ] = record.identity
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
            logical_pages, swa_pages, physical_pages = (
                self._kvduo_restore_page_requirements(prefix_cpu, target_cpu)
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

            restored = self.token_to_kv_pool_allocator.alloc_kvduo_restore(
                prefix_gpu,
                prefix_cpu,
                target_gpu,
                target_cpu,
                last_loc,
                host_hit_length,
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
                        restored
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
                self.token_to_kv_pool_allocator.rollback_restore_allocation(restored)
            raise
        finally:
            finish_kvduo_prefix_restore(commitment, self.host_prefix_cache)

    def _kvduo_restore_page_requirements(
        self, prefix_lens_cpu: torch.Tensor, seq_lens_cpu: torch.Tensor
    ) -> tuple[int, int, int]:
        """Return Full/logical, SWA, and sparse-physical pages for restore.

        Both supported KVDuo restore allocators own no SWA pages: generic
        HiSparse has no SWA sub-pool, while DeepSeek V4 deliberately leaves its
        trailing SWA window to normal prefill.
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
        return logical_pages, 0, physical_pages

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

    def _kvduo_host_capacity_ready(self, slots: int, *, prefill_key=None) -> bool:
        reserved = sum(
            reservation[1]
            for key, reservation in self._kvduo_prefill_host_reservations.items()
            if key != prefill_key
        )
        required = slots + reserved
        self._evict_host_prefix_for_slots(required)
        return bool(
            self._kvduo_common_available(
                int(self.mem_pool_host.available_size() >= required)
            )
        )

    def _reserve_kvduo_prefill_host(self, req, key) -> bool:
        existing = self._kvduo_prefill_host_reservations.get(key)
        if existing is not None:
            return existing[0] is req
        # Reserve the complete prompt even when this execution is only a chunk.
        # No host backup exists until the final prefill-to-decode transition.
        fill_ids = req.full_untruncated_fill_ids
        length = len(fill_ids) // self.compress_ratio
        capacity = (length + self.page_size - 1) // self.page_size * self.page_size
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
            if record is None or not record.fully_valid:
                break
            identities[ordinal] = record.identity
            locations.extend(record.host_locations["main_kv"])
            records.append(identity)
        attached = len(locations)
        if attached:
            self.req_to_host_pool[req.req_pool_idx, :attached] = torch.tensor(
                locations, dtype=torch.int64, device=self.device
            )
            clock = int(self.full_touch_clock[0])
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
        if self.enable_mixed_residency:
            self.kvduo_complete_prefill(req)
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
            self._release_kvduo_prefill_host(prefill_key)

        start_event = device_module.Event()
        finish_event = device_module.Event()
        schedule_stream = device_module.current_stream(self.device)
        start_event.record(schedule_stream)
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
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
        hot_minimum_credit: int = 0,
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
        if not reserve_hot_minimum:
            reserved_slots += (
                sum(getattr(self, "_kvduo_prefill_hot_commitments", {}).values())
                * self.page_size
            )
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
            if reserve_hot_minimum:
                reclaimed_slots = self.reclaim_kvduo_full_pages(
                    pages * self.page_size, reserve_hot_minimum=True
                )
                remaining = max(0, need_size + minimum_slots - available_slots())
            else:
                reclaimed_slots = self.reclaim_kvduo_full_pages(pages * self.page_size)
                remaining = pages * self.page_size - reclaimed_slots
            # Decode may continue through ordinary Hand merge. Prefill restore
            # must use its admitted rollback budget instead of this HOT fallback.
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
            physical_slots, reserve_hot_minimum=self.enable_mixed_residency
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

    def ensure_kvduo_decode_capacity(
        self, requests, *, tree_cache=None
    ) -> KVDuoPressureResult:
        """Reserve the exact main sparse-KV pages needed by ordinary decode.

        This is deliberately separate from the composite allocator's
        ``available_size``: logical slots, SWA, indexer, and compression state
        remain the responsibility of their own admission paths.
        """
        if not self.enable_mixed_residency:
            return self._reclaim_for_physical_allocation(0)
        if not self._kvduo_host_capacity_ready(self._kvduo_decode_host_slots(requests)):
            return KVDuoPressureResult(
                KVDuoPressureStatus.RECLAIM_REQUIRED,
                0,
                action=KVDuoPressureAction.ERROR,
                reason=KVDuoPressureReason.MEMORY_SHORTAGE,
                detail="KVDuo Host pool cannot cover decode backup and prefill reservations",
            )
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
        self._reclaim_kvduo_radix_pages(
            pages
            + self._kvduo_mandatory_hot_pages()
            + self._kvduo_reserved_prefill_pages(),
            tree_cache,
        )
        return self._reclaim_for_physical_allocation(
            pages * self.page_size, reserve_hot_minimum=True
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
            if required_slots > self.device_buffer_size:
                raise RuntimeError("KVDuo hot target exceeds fixed 8-page metadata")
            current = int(self.kvduo_req_hot_capacity[layer_id, req_idx])
            target = ((required_slots + K - 1) // K) * K
            if target > self.device_buffer_size:
                raise RuntimeError(
                    "KVDuo page-aligned target exceeds fixed 8-page metadata"
                )
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
            and (
                not self._kvduo_object_limited((layer, req))
                or (layer, req) in self._kvduo_recovering_objects
            )
        ]
        pages_needed = sum(grow // K for _, _, _, _, grow in requests)
        physical_slots = pages_needed * self.page_size
        if physical_slots == 0:
            return
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        if not allow_reclaim and (
            allocator.available_size()
            - (self._kvduo_optional_protected_pages() - self._kvduo_recovery_credits)
            * self.page_size
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
        # Staging can allocate CUDA tensors; do all of it before publishing any
        # ownership or capacity. A failure after allocator.alloc must return the
        # acquired physical pages, including failures in address construction.
        staged = []
        published = []
        safe_to_free = True
        evaluation_snapshots = {}
        try:
            # This control path is outside graph replay. Read all page numbers
            # with one D2H synchronization, then construct flat KV locations.
            physical_pages = physical.view(-1, self.page_size)
            page_starts_device = physical_pages[:, 0].to(torch.int64)
            page_starts = page_starts_device.to(device="cpu").tolist()
            if len(page_starts) != pages_needed:
                raise RuntimeError(
                    "KVDuo HOT allocation returned invalid physical pages"
                )
            if len(set(page_starts)) != pages_needed or any(
                start in self._kvduo_hot_page_owners for start in page_starts
            ):
                # The allocator returned a page that may still be owned. Do
                # not put suspect indices back on its free list a second time.
                safe_to_free = False
                raise RuntimeError(
                    "KVDuo HOT allocation returned an owned or duplicate page"
                )
            for layer_id, req_idx, current, _, _ in requests:
                if int(self.kvduo_req_hot_capacity[layer_id, req_idx]) != current:
                    raise RuntimeError("KVDuo HOT capacity changed during allocation")

            layer_stride = getattr(self.mem_pool_device, "flat_layer_slot_stride", 0)
            storage_offsets = (
                torch.arange(
                    self.hot_storage_layers, dtype=torch.int64, device=self.device
                )
                * layer_stride
            )
            within_page = torch.arange(
                self.page_size, dtype=torch.int64, device=self.device
            )
            all_page_locs = (
                page_starts_device[:, None, None]
                + storage_offsets[None, :, None]
                + within_page[None, None, :]
            ).reshape(pages_needed, K)

            next_page = 0
            for layer_id, req_idx, current, target, grow in requests:
                owner = (int(layer_id), int(req_idx))
                request_page_count = grow // K
                starts = page_starts[next_page : next_page + request_page_count]
                page_locs = (
                    all_page_locs[next_page : next_page + request_page_count]
                    .reshape(-1)
                    .to(torch.int32)
                )
                next_page += request_page_count
                previous_lru = self.lru_slots[layer_id, req_idx, :target].clone()
                staged.append(
                    (
                        owner,
                        current,
                        target,
                        starts,
                        page_locs,
                        torch.cat(
                            [
                                torch.arange(
                                    current,
                                    target,
                                    dtype=self.lru_slots.dtype,
                                    device=self.device,
                                ),
                                previous_lru[:current],
                            ]
                        ),
                        self.req_device_buffer_token_locs[
                            layer_id, req_idx, current:target
                        ].clone(),
                        self.req_device_buffer_tokens[
                            layer_id, req_idx, current:target
                        ].clone(),
                        previous_lru,
                        list(self._kvduo_req_layer_pages.get(owner, ())),
                    )
                )
            if next_page != pages_needed:
                raise RuntimeError("KVDuo HOT allocation did not consume every page")

            for (layer_id, req_idx), current, target, *_ in staged:
                owner = (layer_id, req_idx)
                evaluation_snapshots[owner] = (
                    [
                        bank[owner].clone()
                        for bank in (
                            self.kvduo_capacity_ema,
                            self.kvduo_capacity_steps,
                            self.kvduo_prefill_ema,
                            self.kvduo_prefill_steps,
                        )
                    ],
                    self._kvduo_prefill_budget_reference.get(
                        self._kvduo_owner_key(owner)
                    ),
                )
            for entry in staged:
                (
                    (layer_id, req_idx),
                    current,
                    target,
                    starts,
                    page_locs,
                    new_lru,
                    _,
                    _,
                    _,
                    _,
                ) = entry
                # Record the pair before its first write so even an exception
                # halfway through publication restores this pair as well.
                published.append(entry)
                self.req_device_buffer_token_locs[layer_id, req_idx, current:target] = (
                    page_locs
                )
                self.req_device_buffer_tokens[layer_id, req_idx, current:target] = -1
                self.lru_slots[layer_id, req_idx, :target] = new_lru
                self._observe_kvduo_hot_growth((layer_id, req_idx), current, target)
                self.kvduo_req_hot_capacity_gpu[layer_id, req_idx] = target
                self.kvduo_req_hot_capacity[layer_id, req_idx] = target
                self._kvduo_req_layer_pages.setdefault((layer_id, req_idx), []).extend(
                    starts
                )
                self._kvduo_hot_generation[layer_id, req_idx] = (
                    self._kvduo_request_generation[req_idx]
                )
                for start in starts:
                    self._kvduo_hot_page_owners[start] = (layer_id, req_idx)
        except Exception:
            try:
                for owner, (values, previous_misses) in evaluation_snapshots.items():
                    for bank, value in zip(
                        (
                            self.kvduo_capacity_ema,
                            self.kvduo_capacity_steps,
                            self.kvduo_prefill_ema,
                            self.kvduo_prefill_steps,
                        ),
                        values,
                    ):
                        bank[owner] = value
                    if previous_misses is None:
                        self._kvduo_prefill_budget_reference.pop(
                            self._kvduo_owner_key(owner), None
                        )
                    else:
                        self._kvduo_prefill_budget_reference[
                            self._kvduo_owner_key(owner)
                        ] = previous_misses
                for (
                    (layer_id, req_idx),
                    current,
                    target,
                    starts,
                    _,
                    _,
                    previous_locs,
                    previous_tokens,
                    previous_lru,
                    previous_pages,
                ) in reversed(published):
                    self.req_device_buffer_token_locs[
                        layer_id, req_idx, current:target
                    ] = previous_locs
                    self.req_device_buffer_tokens[layer_id, req_idx, current:target] = (
                        previous_tokens
                    )
                    self.lru_slots[layer_id, req_idx, :target] = previous_lru
                    self.kvduo_req_hot_capacity_gpu[layer_id, req_idx] = current
                    self.kvduo_req_hot_capacity[layer_id, req_idx] = current
                    if not previous_pages:
                        self._kvduo_hot_generation.pop((layer_id, req_idx), None)
                    if previous_pages:
                        self._kvduo_req_layer_pages[layer_id, req_idx] = previous_pages
                    else:
                        self._kvduo_req_layer_pages.pop((layer_id, req_idx), None)
                    for start in starts:
                        self._kvduo_hot_page_owners.pop(start, None)
            except Exception as rollback_error:
                # Retain the physical allocation if a GPU write cannot be
                # rolled back; freeing it could leave a live address dangling.
                raise RuntimeError(
                    "KVDuo HOT transaction rollback failed"
                ) from rollback_error
            if safe_to_free:
                self.token_to_kv_pool_allocator.free_hisparse_indices(physical)
            raise

    def _consume_kvduo_hot_growth_events(self) -> torch.Tensor:
        """Read each H2D capacity event before the immediately following replay.

        A blocking D2H read at this graph-external boundary is necessary: an
        asynchronous, possibly stale snapshot could postpone an event for
        several replays. The caller has ordered this stream after the preceding
        producer. Reset the same fixed-address bank before its next replay.
        """
        self._kvduo_hot_growth_events_host.copy_(self.kvduo_hot_growth_events)
        self.kvduo_hot_growth_events.zero_()
        self.kvduo_resolver_stats.zero_()
        return self._kvduo_hot_growth_events_host

    def prepare_kvduo_graph_replay(
        self,
        req_pool_indices: torch.Tensor,
        req_pool_indices_cpu: Sequence[int] | torch.Tensor | None = None,
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
        growth_events = {
            (layer_id, req_idx): int(events_cpu[layer_id, req_idx])
            for layer_id, req_idx in torch.nonzero(events_cpu > 0).tolist()
            if req_idx in self._active_kvduo_reqs and self._mixed_slots[req_idx]
        }
        event_capacities = {
            owner: int(self.kvduo_req_hot_capacity[owner]) for owner in growth_events
        }
        if (
            not pending_minimum
            and not growth_events
            and not self._kvduo_prefill_limited_objects
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
        # Optional growth can reclaim any Hand-eligible HOT owner, including
        # applicants, never FULL pages. Released surplus stays in the free pool.
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        self._recover_kvduo_objects(growth_events)
        reserved_pages = self._kvduo_optional_protected_pages()
        applicants = {
            owner: min(
                1,
                int(self.kvduo_hot_page_limit[owner])
                - int(self.kvduo_req_hot_capacity[owner]) // self.hot_page_size,
            )
            for owner, events in growth_events.items()
            if events > 0 and not self._kvduo_object_limited(owner)
        }
        applicants = {owner: pages for owner, pages in applicants.items() if pages > 0}
        if not self._kvduo_consensus(
            (reserved_pages, tuple(sorted(applicants.items())))
        ):
            return
        if not applicants:
            return
        common_free = self._kvduo_common_available(
            allocator.available_size() // self.page_size
        )
        # Protection only limits optional allocation. Its own shortfall belongs
        # to prefill admission, which evaluates object rollback and the
        # frozen miss budget before releasing active HOT pages.
        if common_free < reserved_pages:
            return
        shortfall = sum(applicants.values()) - (common_free - reserved_pages)
        if shortfall > 0:
            self.reclaim_kvduo_hot_pages(shortfall * self.page_size)
            applicants = {
                owner: pages
                for owner, pages in applicants.items()
                if int(self.kvduo_req_hot_capacity[owner]) >= event_capacities[owner]
            }
        free_pages = max(
            0,
            self._kvduo_common_available(allocator.available_size() // self.page_size)
            - reserved_pages,
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

    def _grow_kvduo_hot_metadata(self, required_slots: int) -> None:
        """Validate against the immutable, capture-stable resolver metadata."""
        if required_slots > self.device_buffer_size:
            raise RuntimeError("KVDuo hot metadata cannot grow beyond fixed 8-page")

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
            self.kvduo_req_hot_capacity[layer_id, req_idx] = 0
            self.kvduo_req_hot_capacity_gpu[layer_id, req_idx] = 0
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

    def _merge_kvduo_hot_pages(
        self, layer_id: int, req_idx: int, target_pages=None
    ) -> int:
        """Retain budgeted MRU entries, preserving LRU and minimizing payload moves.

        No physical page becomes available until payload copies and metadata
        publication complete. A failed copy aborts reclamation without freeing
        its source pages. All tensor addresses/shapes remain capture-stable.
        """
        self._wait_kvduo_hot_merge_boundary()
        owner = (layer_id, req_idx)
        pages = self._kvduo_req_layer_pages.get(owner)
        if not pages or not self._kvduo_can_merge(owner):
            return 0
        if any(
            (layer_id, req_idx, index) in self._kvduo_hot_pressure_protected
            for index in range(len(pages))
        ):
            return 0
        capacity = int(self.kvduo_req_hot_capacity[owner])
        if capacity != len(pages) * self.hot_page_size:
            raise RuntimeError("KVDuo HOT capacity/ownership mismatch before merge")
        if any(self._kvduo_hot_page_owners.get(start) != owner for start in pages):
            raise RuntimeError("KVDuo whole-page ownership mismatch before merge")

        if target_pages is None:
            counts = self.kvduo_capacity_ema[owner].cpu().tolist()
            target_pages = kvduo_shrink_target(
                counts[:8],
                counts[8],
                int(self.kvduo_capacity_steps[owner]),
                len(pages),
                self.min_hot_pages,
                self.kvduo_prefill_policy.max_relative_miss_increase,
            )
        if target_pages == len(pages):
            # An unchanged scan records the base without discarding entries,
            # changing LRU, or erasing the measured capacity cost.
            self._establish_kvduo_hot_base(owner, pages)
            return 0

        plan = prepare_kvduo_hot_merge(
            self.req_device_buffer_tokens[layer_id, req_idx],
            self.req_device_buffer_token_locs[layer_id, req_idx],
            self.lru_slots[layer_id, req_idx],
            capacity,
            self.hot_page_size,
            self.top_k,
            target_pages,
        )
        kept_starts = [pages[index] for index in plan.kept_pages]
        released_starts = [pages[index] for index in plan.released_pages]
        new_capacity = target_pages * self.hot_page_size
        new_tokens, new_locs, new_lru = plan.tokens, plan.locations, plan.lru_slots
        src_locs, dst_locs = plan.source_locations, plan.destination_locations
        released = torch.tensor(
            [
                slot
                for start in released_starts
                for slot in range(start, start + self.page_size)
            ],
            dtype=torch.int64,
            device=self.device,
        )

        finished = device_module.Event() if new_tokens.is_cuda else None
        self._move_kvduo_hot_entries(layer_id, src_locs, dst_locs)
        self.req_device_buffer_tokens[layer_id, req_idx].copy_(new_tokens)
        self.req_device_buffer_token_locs[layer_id, req_idx].copy_(new_locs)
        self.lru_slots[layer_id, req_idx].copy_(new_lru)
        self.kvduo_req_hot_capacity_gpu[owner] = new_capacity
        # A Hand scan can precede replay's event consumption (e.g. mandatory
        # decode allocation). Shrinking consumes any pending event as well as
        # the host-side event filtered by prepare_kvduo_graph_replay.
        self.kvduo_hot_growth_events[owner] = 0
        if finished is not None:
            finished.record(device_module.current_stream(self.device))
            finished.synchronize()
        self.kvduo_req_hot_capacity[owner] = new_capacity
        self._kvduo_req_layer_pages[owner] = kept_starts
        self._reset_kvduo_capacity_stats(owner)
        self._establish_kvduo_hot_base(owner, kept_starts)
        for start in released_starts:
            del self._kvduo_hot_page_owners[start]
        if released_starts:
            self.token_to_kv_pool_allocator.free_hisparse_indices(released)
        return len(released_starts) * self.page_size

    def reclaim_kvduo_hot_pages(self, num_tokens: int) -> int:
        """Scan request-major Hand order, including zero-release and repeated merges."""
        if not self.enable_mixed_residency or num_tokens <= 0:
            return 0
        self._wait_kvduo_hot_merge_boundary()
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        target = allocator.available_size() + num_tokens
        layers = self.mem_pool_device.layer_num
        objects = len(self._kvduo_request_generation) * layers
        reclaimed = 0
        # A single compact snapshot at the safe boundary covers the whole scan.
        counts = self.kvduo_capacity_ema.cpu().tolist()
        steps = self.kvduo_capacity_steps.cpu().tolist()
        plans = []
        for offset in range(objects):
            req, layer = divmod((self._kvduo_hand + offset) % objects, layers)
            owner = (layer, req)
            if not self._kvduo_can_merge(owner):
                continue
            values = counts[layer][req]
            pages = len(self._kvduo_req_layer_pages[owner])
            target_pages = kvduo_shrink_target(
                values[:8],
                values[8],
                steps[layer][req],
                pages,
                self.min_hot_pages,
                self.kvduo_prefill_policy.max_relative_miss_increase,
            )
            plans.append((offset, owner, pages, target_pages))
        if not self._kvduo_consensus(tuple(plans)):
            return 0
        start_hand = self._kvduo_hand
        for offset, (layer, req), _, target_pages in plans:
            self._kvduo_hand = (start_hand + offset + 1) % objects
            reclaimed += self._merge_kvduo_hot_pages(layer, req, target_pages)
            if allocator.available_size() >= target:
                break
        else:
            self._kvduo_hand = start_hand
        # No decode can interleave here. A further rotation sees exactly the
        # same costs or immature reset stats, so it cannot release more pages.
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
            and bool(self._kvduo_req_layer_pages.get(owner))
            and self._kvduo_hot_generation.get(owner)
            == self._kvduo_request_generation[req]
            and not self._kvduo_object_limited(owner)
            and not any(
                (l, r) == owner for l, r, _ in self._kvduo_hot_pressure_protected
            )
        )

    def _init_kvduo_prefill_rollback_state(self, requests: int) -> None:
        shape = (self.mem_pool_device.layer_num, requests)
        self.kvduo_capacity_ema = torch.zeros(
            (*shape, 9), dtype=torch.float32, device=self.device
        )
        self.kvduo_capacity_steps = torch.zeros(
            shape, dtype=torch.int64, device=self.device
        )
        self.kvduo_hot_page_limit = torch.full(
            shape, 8, dtype=torch.int32, device="cpu"
        )
        self._kvduo_next_prefill_pages = 0
        self._kvduo_next_prefill_key = None
        self._kvduo_recovery_credits = 0
        self._kvduo_recovering_objects = set()
        self._kvduo_fixed_queue = {}
        self._kvduo_fixed_sequence = 0
        self._kvduo_prefill_lifecycle = {}
        self._kvduo_prefill_host_reservations = {}
        self._kvduo_prefill_generation = 0
        self._kvduo_full_conversion_pending = {}
        self._kvduo_prefill_hot_commitments = {}
        self._kvduo_prefill_reservation_requests = {}
        self._kvduo_prefill_associations = {}
        self.kvduo_hot_base_capacity = torch.zeros(
            shape, dtype=torch.int32, device=self.device
        )
        self.kvduo_prefill_ema = torch.zeros(
            (*shape, 2), dtype=torch.float32, device=self.device
        )
        self.kvduo_prefill_steps = torch.zeros(
            shape, dtype=torch.int64, device=self.device
        )
        self._kvduo_hand = 0
        self._kvduo_hand_sequence = 0
        self._kvduo_hot_processed = {}
        self._kvduo_hot_base_pages = {}
        self._kvduo_hot_generation = {}
        self._kvduo_request_generation = [0] * requests
        self._kvduo_hot_base_generation = {}
        self._kvduo_prefill_budget_reference = {}
        self._kvduo_prefill_budget_cost = {}
        self._kvduo_prefill_limited_objects = set()
        self._kvduo_prefill_block_leases = {}
        self._kvduo_prefill_reservations = {}
        self._kvduo_prefill_next_reservation = 0

    def _reset_kvduo_prefill_request(self, req_idx: int) -> None:
        if not hasattr(self, "kvduo_prefill_ema"):
            return
        old_keys = {
            self._kvduo_owner_key((layer, req_idx))
            for layer in range(self.mem_pool_device.layer_num)
        }
        self._kvduo_request_generation[req_idx] += 1
        for bank in (
            self.kvduo_hot_base_capacity,
            self.kvduo_capacity_ema,
            self.kvduo_capacity_steps,
            self.kvduo_prefill_ema,
            self.kvduo_prefill_steps,
        ):
            bank[:, req_idx] = 0
        self._kvduo_prefill_limited_objects.difference_update(old_keys)
        self._kvduo_recovering_objects.difference_update(
            (layer, req_idx) for layer in range(self.mem_pool_device.layer_num)
        )
        for key in old_keys:
            self._kvduo_prefill_budget_cost.pop(key, None)
            self._kvduo_prefill_block_leases.pop(key, None)
            self._kvduo_fixed_queue.pop(key, None)
            self._kvduo_prefill_associations.pop(key, None)
        self.kvduo_hot_page_limit[:, req_idx] = 8
        for records in (
            self._kvduo_hot_base_pages,
            self._kvduo_hot_base_generation,
            self._kvduo_hot_generation,
            self._kvduo_hot_processed,
        ):
            for owner in list(records):
                if owner[1] == req_idx:
                    del records[owner]
        for key in list(self._kvduo_prefill_budget_reference):
            if key[1] == req_idx:
                del self._kvduo_prefill_budget_reference[key]
        for reservation in self._kvduo_prefill_reservations.values():
            reservation[1].difference_update(old_keys)
        self._prune_kvduo_prefill_budget()

    def _establish_kvduo_hot_base(self, owner, pages) -> None:
        """Every ordinary merge, including no-release, starts a new base/phase."""
        self._kvduo_hot_base_pages[owner] = tuple(pages)
        self._kvduo_hot_base_generation[owner] = self._kvduo_request_generation[
            owner[1]
        ]
        self.kvduo_hot_base_capacity[owner] = len(pages) * self.hot_page_size
        self.kvduo_prefill_ema[owner] = 0
        self.kvduo_prefill_steps[owner] = 0
        self._kvduo_prefill_budget_reference.pop(self._kvduo_owner_key(owner), None)
        self._kvduo_hand_sequence += 1
        self._kvduo_hot_processed[owner] = self._kvduo_hand_sequence

    def _observe_kvduo_hot_growth(self, owner, old_capacity, new_capacity) -> None:
        if old_capacity == 0:
            self._reset_kvduo_capacity_stats(owner)
        else:
            # Empty slots are prepended at the LRU end. Existing MRU ranks and
            # segments stay unchanged; miss EMA and maturity remain a prior.
            first_new_segment = old_capacity // self.hot_page_size
            end_segment = new_capacity // self.hot_page_size
            # Seed new tails with the previous LRU-most segment's hit EMA.
            # This is a prior, not an observed hit in the newly allocated slots.
            counts = self.kvduo_capacity_ema[owner]
            counts[first_new_segment:end_segment] = counts[first_new_segment - 1]
        base = self._kvduo_hot_base_pages.get(owner)
        if (
            base
            and self._kvduo_hot_base_generation[owner]
            == self._kvduo_request_generation[owner[1]]
            and old_capacity == len(base) * self.hot_page_size
            and new_capacity > old_capacity
        ):
            self.kvduo_prefill_ema[owner] = 0
            self.kvduo_prefill_steps[owner] = 0
            self._kvduo_prefill_budget_reference.pop(self._kvduo_owner_key(owner), None)

    def _reset_kvduo_capacity_stats(self, owner) -> None:
        self.kvduo_capacity_ema[owner] = 0
        self.kvduo_capacity_steps[owner] = 0

    def _kvduo_common_available(self, pages: int) -> int:
        if getattr(self, "tp_world_size", 1) <= 1:
            return pages
        value = torch.tensor(pages, dtype=torch.int64)
        torch.distributed.all_reduce(
            value, op=torch.distributed.ReduceOp.MIN, group=self.tp_group
        )
        return int(value.item())

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
        self._kvduo_prefill_lifecycle.setdefault(key, [req, "prefilling"])
        return key

    def protect_kvduo_prefill_plan(self, req, end: int) -> None:
        """Protect only the scheduler's next plan, including blocked independent pools."""
        key = self._kvduo_prefill_key(req)
        if self._kvduo_next_prefill_key not in (None, key):
            return
        self._kvduo_next_prefill_key = key
        pages = self.kvduo_prefill_pages(len(req.prefix_indices), end)
        # The selected execution's allocations replace this protection once
        # committed; compulsory future HOT minima remain a separate obligation.
        self._kvduo_next_prefill_pages = max(0, pages)
        if (
            self._kvduo_prefill_limited_objects
            and key not in self._kvduo_prefill_hot_commitments
        ):
            self._kvduo_next_prefill_pages += (
                self.min_hot_pages * self.mem_pool_device.layer_num
            )

    def begin_kvduo_schedule_boundary(self) -> None:
        self.process_kvduo_full_conversions()
        self._kvduo_next_prefill_pages = 0
        self._kvduo_next_prefill_key = None

    def bind_kvduo_prefill_batch(self, requests, reservation_tokens) -> None:
        keys = {self._kvduo_prefill_key(req) for req in requests}
        affected = set()
        for token in reservation_tokens:
            reservation = self._kvduo_prefill_reservations.get(token)
            if reservation is not None:
                affected.update(reservation[1])
        for key in affected:
            self._kvduo_prefill_associations.setdefault(key, set()).update(keys)
        # Allocations have completed. They now occupy physical pages rather
        # than requiring an additional next-plan protection.
        self._kvduo_next_prefill_pages = 0
        self._kvduo_next_prefill_key = None

    def kvduo_complete_prefill(self, req) -> None:
        """Only the complete-prefill transition calls this, before any unlock."""
        key = self._kvduo_prefill_key(req)
        if self._kvduo_prefill_lifecycle[key][1] == "awaiting_backup":
            return
        if self._kvduo_prefill_limited_objects:
            self._kvduo_prefill_lifecycle[key][1] = "awaiting_backup"
            self._kvduo_full_conversion_pending[key] = None
        else:
            self._kvduo_prefill_lifecycle[key][1] = "converted"
            self._kvduo_prefill_hot_commitments.pop(key, None)

    def _kvduo_conversion_remaining(self, req) -> bool:
        req_idx = req.req_pool_idx
        compressed = self.mem_pool_device.translate_loc_from_full_to_compressed(
            self.req_to_token_pool.req_to_token[req_idx, : req.kv.kv_allocated_len]
        )
        prefix = self.host_token_len(req.cache_protected_len)
        start = (prefix + self.page_size - 1) // self.page_size * self.page_size
        pages = (len(compressed) + self.page_size - 1) // self.page_size
        end = max(0, pages - self.tail_protected_pages) * self.page_size
        historical = compressed[start:end]
        if not historical.numel():
            return False
        # Still-resident history includes stale/missing host backups and
        # temporary transfer protection. Permanent prefix/tail pages were excluded.
        mapping = self.mem_pool_device.full_to_hisparse_device_index_mapping
        return bool(torch.any(mapping[historical] > 0))

    def process_kvduo_full_conversions(self) -> None:
        """Finish every requested conversion, even if other requests already unlocked."""
        if not self.enable_mixed_residency or not self._kvduo_full_conversion_pending:
            return
        pending = [
            (key, *self._kvduo_prefill_lifecycle[key])
            for key in self._kvduo_full_conversion_pending
            if key in self._kvduo_prefill_lifecycle
        ]
        if not any(
            not getattr(req, "hisparse_staging", False) for _, req, _ in pending
        ):
            return
        self._wait_kvduo_hot_merge_boundary()
        for key, req, state in pending:
            if state != "awaiting_backup":
                continue
            req_idx = getattr(req, "req_pool_idx", None)
            if callable(getattr(req, "finished", None)) and req.finished():
                self._kvduo_prefill_lifecycle[key][1] = "cancelled"
                self._kvduo_prefill_hot_commitments.pop(key, None)
                self._kvduo_full_conversion_pending.pop(key, None)
                continue
            if req_idx is None or self._active_kvduo_reqs.get(req_idx) is not req:
                continue
            if getattr(req, "hisparse_staging", False):
                continue
            generation = self._kvduo_request_key(req_idx)
            previous = self._kvduo_full_conversion_pending[key]
            if previous is None:
                self._kvduo_full_conversion_pending[key] = generation
            elif previous != generation:
                self._kvduo_prefill_lifecycle[key][1] = "cancelled"
                self._kvduo_prefill_hot_commitments.pop(key, None)
                self._kvduo_full_conversion_pending.pop(key, None)
                continue
            self.reclaim_kvduo_full_pages(
                self.page_size, request_indices={req_idx}, release_all=True
            )
            if self._kvduo_conversion_remaining(req):
                continue
            # A mixed request's pending minimum is now counted by its live
            # generation. Avoid charging the same minimum twice.
            self._kvduo_prefill_hot_commitments.pop(key, None)
            self._kvduo_prefill_lifecycle[key][1] = "converted"
            self._kvduo_full_conversion_pending.pop(key, None)

    def _kvduo_prefill_plan_uncovered_pages(self) -> int:
        key = self._kvduo_next_prefill_key
        covered = sum(
            self._kvduo_prefill_reservations[token][0]
            for token, request in self._kvduo_prefill_reservation_requests.items()
            if request == key and token in self._kvduo_prefill_reservations
        )
        covered += self._kvduo_prefill_hot_commitments.get(key, 0)
        return max(0, self._kvduo_next_prefill_pages - covered)

    def _end_kvduo_prefill_lifecycle(self, req_idx, request=None) -> None:
        if request is None:
            request = self._active_kvduo_reqs.get(req_idx)
        for key, (req, _) in list(self._kvduo_prefill_lifecycle.items()):
            if req is not request:
                continue
            self._release_kvduo_prefill_host(key)
            self._kvduo_prefill_hot_commitments.pop(key, None)
            self._kvduo_full_conversion_pending.pop(key, None)
            for associations in self._kvduo_prefill_associations.values():
                associations.discard(key)
            del self._kvduo_prefill_lifecycle[key]
        if request is not None and hasattr(request, "kvduo_prefill_generation"):
            del request.kvduo_prefill_generation

    def _kvduo_optional_protected_pages(self) -> int:
        return (
            self._kvduo_reserved_prefill_pages()
            + self._kvduo_mandatory_hot_pages()
            + self._kvduo_prefill_plan_uncovered_pages()
            + self._kvduo_recovery_credits
        )

    def _kvduo_recovery_ready(self, key) -> bool:
        owner = key[:2]
        req = owner[1]
        pages = self._kvduo_req_layer_pages.get(owner, ())
        base = self._kvduo_hot_base_pages.get(owner, ())
        associations = self._kvduo_prefill_associations.get(key, ())
        request = self._active_kvduo_reqs.get(req)
        return (
            key in self._kvduo_prefill_limited_objects
            and key == self._kvduo_owner_key(owner)
            and request is not None
            and not (
                callable(getattr(request, "finished", None)) and request.finished()
            )
            and not getattr(request, "hisparse_staging", False)
            and not self._kvduo_prefill_block_leases.get(key)
            and all(
                self._kvduo_prefill_lifecycle.get(prefill, [None, "prefilling"])[1]
                in ("converted", "cancelled")
                for prefill in associations
            )
            and not any(
                (l, r) == owner for l, r, _ in self._kvduo_hot_pressure_protected
            )
            and self.min_hot_pages <= len(base) < 8
            and tuple(pages) == base
            and self._kvduo_hot_base_generation.get(owner) == key[2]
            and self._kvduo_hot_generation.get(owner) == key[2]
            and int(self.kvduo_req_hot_capacity[owner])
            == len(pages) * self.hot_page_size
            and all(self._kvduo_hot_page_owners.get(page) == owner for page in pages)
        )

    def _recover_kvduo_objects(self, events) -> None:
        self._prune_kvduo_prefill_budget()
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        objects = []
        for key in sorted(self._kvduo_prefill_limited_objects):
            objects.append(
                KVDuoRecoveryObject(
                    key,
                    self._kvduo_fixed_queue.get(key, 0),
                    self._kvduo_recovery_ready(key),
                )
            )
        available = (
            self._kvduo_common_available(allocator.available_size() // self.page_size)
            - self._kvduo_reserved_prefill_pages()
            - self._kvduo_mandatory_hot_pages()
        )
        selected = plan_kvduo_recovery(
            objects, available, self._kvduo_prefill_plan_uncovered_pages()
        )
        targets = {
            key[:2]: int(self.kvduo_req_hot_capacity[key[:2]]) + self.hot_page_size
            for key in selected
            if events.get(key[:2], 0) > 0
        }
        if not self._kvduo_consensus((selected, tuple(sorted(targets.items())))):
            return
        if not selected:
            return
        quantum = len(selected)
        redeemable = (
            allocator.available_size() // self.page_size
            - self._kvduo_optional_protected_pages()
            >= quantum
            and all(self._kvduo_recovery_ready(key) for key in selected)
        )
        if not self._kvduo_consensus(redeemable) or not redeemable:
            return
        self._kvduo_recovery_credits = quantum
        self._kvduo_recovering_objects = {key[:2] for key in selected}
        try:
            self._ensure_kvduo_hot_capacity_targets_batch(targets, allow_reclaim=False)
            if any(
                int(self.kvduo_req_hot_capacity[owner]) != target
                for owner, target in targets.items()
            ):
                raise RuntimeError(
                    "KVDuo object recovery could not redeem its reserved credits"
                )
        finally:
            self._kvduo_recovery_credits = 0
            self._kvduo_recovering_objects = set()
        for key in selected:
            events.pop(key[:2], None)
            self._release_kvduo_prefill_hold(key)

    def _kvduo_reserved_prefill_pages(self) -> int:
        return sum(
            pages
            for pages, _ in getattr(self, "_kvduo_prefill_reservations", {}).values()
        )

    def _kvduo_mandatory_hot_pages(self) -> int:
        return sum(self._kvduo_prefill_hot_commitments.values()) + sum(
            max(
                0,
                self.min_hot_pages
                - int(self.kvduo_req_hot_capacity[layer, req]) // self.hot_page_size,
            )
            for req in getattr(self, "_kvduo_pending_hot_minimum", ())
            if req in self._active_kvduo_reqs
            for layer in range(self.mem_pool_device.layer_num)
        )

    def _kvduo_owner_key(self, owner):
        return (*owner, self._kvduo_request_generation[owner[1]])

    def _release_kvduo_prefill_hold(self, key) -> None:
        self._kvduo_prefill_limited_objects.discard(key)
        self._kvduo_prefill_budget_cost.pop(key, None)
        self._kvduo_prefill_block_leases.pop(key, None)
        self._kvduo_fixed_queue.pop(key, None)
        self._kvduo_prefill_associations.pop(key, None)
        if key == self._kvduo_owner_key(key[:2]):
            self.kvduo_hot_page_limit[key[:2]] = 8
            self.kvduo_hot_growth_events[key[:2]] = 0
        self._prune_kvduo_prefill_budget()

    def _kvduo_request_key(self, req_idx):
        return (req_idx, self._kvduo_request_generation[req_idx])

    def _kvduo_object_limited(self, owner):
        return self._kvduo_owner_key(owner) in self._kvduo_prefill_limited_objects

    def _prune_kvduo_prefill_budget(self) -> None:
        for key in list(self._kvduo_prefill_limited_objects):
            if (
                key != self._kvduo_owner_key(key[:2])
                or key[1] not in self._active_kvduo_reqs
            ):
                self._kvduo_prefill_limited_objects.discard(key)
                self._kvduo_prefill_budget_cost.pop(key, None)
                self._kvduo_prefill_block_leases.pop(key, None)
                self._kvduo_fixed_queue.pop(key, None)
                self._kvduo_prefill_associations.pop(key, None)
        for key in list(self._kvduo_prefill_budget_reference):
            if (
                key[2] != self._kvduo_request_generation[key[1]]
                or key[1] not in self._active_kvduo_reqs
            ):
                del self._kvduo_prefill_budget_reference[key]
        if not self._kvduo_prefill_limited_objects:
            self._kvduo_prefill_budget_reference.clear()
            self._kvduo_prefill_budget_cost.clear()

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
        _, owners = reservation
        prefill_key = self._kvduo_prefill_reservation_requests.pop(token, None)
        if cancelled and prefill_key is not None:
            self._release_kvduo_prefill_host(prefill_key)
            self._kvduo_prefill_hot_commitments.pop(prefill_key, None)
            request = self._kvduo_prefill_lifecycle[prefill_key][0]
            self._kvduo_prefill_lifecycle[prefill_key][1] = "cancelled"
            if getattr(request, "kvduo_prefill_generation", None) == prefill_key[1]:
                del request.kvduo_prefill_generation
        for key in owners:
            leases = self._kvduo_prefill_block_leases.get(key, set())
            leases.discard(token)
            if not leases:
                self._kvduo_prefill_block_leases.pop(key, None)
                if cancelled:
                    self._release_kvduo_prefill_hold(key)
        self._prune_kvduo_prefill_budget()

    def consume_kvduo_prefill_reservation(self, token: int, pages: int) -> None:
        reservation = self._kvduo_prefill_reservations[token]
        if not 0 <= pages <= reservation[0]:
            raise RuntimeError("KVDuo prefill allocation exceeds its reservation")
        reservation[0] -= pages

    def _kvduo_prefill_objects(self, req_indices: Sequence[int]):
        owners = [
            (layer, req)
            for req in dict.fromkeys(req_indices)
            if req in self._active_kvduo_reqs
            for layer in range(self.mem_pool_device.layer_num)
            if self._kvduo_req_layer_pages.get((layer, req))
        ]
        if not owners:
            return []
        indices = torch.tensor(owners, dtype=torch.int64, device=self.device).T
        snapshot = (
            torch.cat(
                (
                    self.kvduo_prefill_ema[indices[0], indices[1]].double(),
                    self.kvduo_prefill_steps[indices[0], indices[1], None].double(),
                ),
                dim=1,
            )
            .cpu()
            .tolist()
        )
        result = []
        for owner, (extra, misses, steps) in zip(owners, snapshot):
            base = self._kvduo_hot_base_pages.get(owner, ())
            pages = self._kvduo_req_layer_pages[owner]
            base_valid = (
                len(base) >= self.min_hot_pages
                and self._kvduo_hot_processed.get(owner, 0) > 0
                and tuple(pages[: len(base)]) == base
                and self._kvduo_hot_base_generation.get(owner)
                == self._kvduo_request_generation[owner[1]]
                and int(self.kvduo_req_hot_capacity[owner])
                == len(pages) * self.hot_page_size
                and all(
                    self._kvduo_hot_page_owners.get(page) == owner for page in pages
                )
                and self._kvduo_hot_generation.get(owner)
                == self._kvduo_request_generation[owner[1]]
            )
            eligible = base_valid and self._kvduo_can_merge(owner)
            result.append(
                KVDuoPrefillObject(
                    owner,
                    len(base),
                    len(pages),
                    (extra, misses),
                    int(steps),
                    self._kvduo_hot_processed.get(owner, 0),
                    eligible,
                    base_valid,
                )
            )
        return result

    def reserve_kvduo_prefill(
        self,
        pages: int,
        decode_req_indices: Sequence[int],
        *,
        tree_cache=None,
        prefix_indices: torch.Tensor | None = None,
        prefill_req=None,
    ) -> int | None:
        self._wait_kvduo_hot_merge_boundary()
        prefill_key = (
            self._kvduo_prefill_key(prefill_req) if prefill_req is not None else None
        )
        needs_host = prefill_req is not None and hasattr(
            prefill_req, "full_untruncated_fill_ids"
        )
        new_host_reservation = (
            needs_host and prefill_key not in self._kvduo_prefill_host_reservations
        )
        if needs_host and not self._reserve_kvduo_prefill_host(
            prefill_req, prefill_key
        ):
            return None
        token = None
        try:
            token = self._reserve_kvduo_prefill_physical(
                pages,
                decode_req_indices,
                tree_cache=tree_cache,
                prefix_indices=prefix_indices,
                prefill_req=prefill_req,
            )
            return token
        finally:
            if token is None and new_host_reservation:
                self._release_kvduo_prefill_host(prefill_key)

    def _reserve_kvduo_prefill_physical(
        self,
        pages: int,
        decode_req_indices: Sequence[int],
        *,
        tree_cache=None,
        prefix_indices: torch.Tensor | None = None,
        prefill_req=None,
    ) -> int | None:
        """Check one fixed round's miss budget before any no-copy rollback."""
        self._prune_kvduo_prefill_budget()
        allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        prefill_key = (
            self._kvduo_prefill_key(prefill_req) if prefill_req is not None else None
        )
        commitment_keys = set(self._kvduo_prefill_reservation_requests.values())
        if prefill_key is not None:
            commitment_keys.add(prefill_key)
        commitment_keys.difference_update(self._kvduo_prefill_hot_commitments)
        prospective_minimum = (
            len(commitment_keys) * self.min_hot_pages * self.mem_pool_device.layer_num
        )
        # Existing locks already require the incoming complete prefill to
        # become mixed. Include its minimum in ordinary Radix/FULL reclaim,
        # before asking to roll back any additional HOT objects.
        locked_minimum = (
            prospective_minimum if self._kvduo_prefill_limited_objects else 0
        )
        self._reclaim_kvduo_radix_pages(
            pages
            + self._kvduo_reserved_prefill_pages()
            + self._kvduo_mandatory_hot_pages()
            + locked_minimum,
            tree_cache,
        )

        def physical_shortage():
            return max(
                0,
                pages
                + self._kvduo_reserved_prefill_pages()
                + self._kvduo_mandatory_hot_pages()
                + locked_minimum
                - self._kvduo_common_available(
                    allocator.available_size() // self.page_size
                ),
            )

        shortage = physical_shortage()
        if shortage:
            # Preserve the incoming GPU prefix, including a partial page.
            previous = self._kvduo_pressure_protected
            if prefix_indices is not None:
                protected = self.mem_pool_device.translate_loc_from_full_to_compressed(
                    prefix_indices
                )
                self._kvduo_pressure_protected = (
                    protected if previous is None else torch.cat((previous, protected))
                )
            try:
                # Prefill may reclaim legal FULL pages, but generic HOT merge
                # must never bypass the rollback loss budget below. New mixed
                # owners' minimum capacities are charged before demotion.
                self.reclaim_kvduo_full_pages(
                    shortage * self.page_size, reserve_hot_minimum=True
                )
            finally:
                self._kvduo_pressure_protected = previous
            shortage = physical_shortage()
        # A rollback locks objects and therefore obligates the incoming
        # prefill's later mixed-decode minimum. Charge it before selecting HOT.
        if prospective_minimum and (shortage or self._kvduo_prefill_limited_objects):
            shortage = max(
                0,
                pages
                + self._kvduo_reserved_prefill_pages()
                + self._kvduo_mandatory_hot_pages()
                + prospective_minimum
                - self._kvduo_common_available(
                    allocator.available_size() // self.page_size
                ),
            )
        else:
            prospective_minimum = 0
        plan = ()
        objects = []
        if shortage:
            objects = self._kvduo_prefill_objects(decode_req_indices)
            references = (
                {
                    key[:2]: value
                    for key, value in self._kvduo_prefill_budget_reference.items()
                }
                if self._kvduo_prefill_limited_objects
                else None
            )
            plan = plan_kvduo_prefill_rollback(
                objects,
                shortage,
                self.kvduo_prefill_policy,
                references,
                sum(self._kvduo_prefill_budget_cost.values()),
            )
        owners = plan or ()
        ready = plan is not None
        if plan:
            check = {
                obj.owner: obj
                for obj in self._kvduo_prefill_objects([owner[1] for owner in plan])
                if obj.owner in plan
            }
            planned = {obj.owner: obj for obj in objects if obj.owner in plan}
            ready = check == planned and all(obj.eligible for obj in check.values())
        ready = self._kvduo_consensus((ready, plan, owners)) and ready
        if not ready:
            return None
        token = self._kvduo_prefill_next_reservation
        self._kvduo_prefill_next_reservation += 1
        if plan and not self._kvduo_prefill_limited_objects:
            self._kvduo_prefill_budget_reference = {
                self._kvduo_owner_key(obj.owner): obj.misses
                for obj in objects
                if obj.mature
            }
        costs = {obj.owner: obj.extra_hits for obj in objects if obj.owner in plan}
        if prefill_key is not None:
            self._kvduo_prefill_reservation_requests[token] = prefill_key
            if prospective_minimum:
                for pending_key in commitment_keys:
                    self._kvduo_prefill_hot_commitments[pending_key] = (
                        self.min_hot_pages * self.mem_pool_device.layer_num
                    )
        self._kvduo_prefill_reservations[token] = [
            pages,
            {self._kvduo_owner_key(owner) for owner in plan},
        ]
        for owner in plan:
            key = self._kvduo_owner_key(owner)
            self._kvduo_prefill_budget_cost[key] = costs[owner]
            self._kvduo_prefill_limited_objects.add(key)
            self._kvduo_prefill_block_leases.setdefault(key, set()).add(token)
            self.kvduo_hot_growth_events[owner] = 0
            self.kvduo_hot_page_limit[owner] = len(self._kvduo_hot_base_pages[owner])
            self._kvduo_fixed_sequence += 1
            self._kvduo_fixed_queue[key] = self._kvduo_fixed_sequence
            if prefill_key is not None:
                self._kvduo_prefill_associations.setdefault(key, set()).add(prefill_key)
        if prefill_key is not None:
            for key in self._kvduo_prefill_limited_objects:
                if self._kvduo_prefill_block_leases.get(key):
                    self._kvduo_prefill_associations.setdefault(key, set()).add(
                        prefill_key
                    )
        try:
            self._rollback_kvduo_hot_batch(owners)
        except Exception:
            self.finish_kvduo_prefill_reservation(token, cancelled=True)
            raise
        return token

    def _rollback_kvduo_hot_batch(self, owners) -> None:
        if not owners:
            return
        self._wait_kvduo_hot_merge_boundary()
        released = []
        for owner in owners:
            pages = self._kvduo_req_layer_pages[owner]
            base = self._kvduo_hot_base_pages[owner]
            if (
                tuple(pages[: len(base)]) != base
                or len(base) < self.min_hot_pages
                or self._kvduo_hot_base_generation.get(owner)
                != self._kvduo_request_generation[owner[1]]
            ):
                raise RuntimeError("KVDuo rollback base-page identity changed")
            if (
                self._kvduo_hot_generation.get(owner)
                != self._kvduo_request_generation[owner[1]]
                or int(self.kvduo_req_hot_capacity[owner])
                != len(pages) * self.hot_page_size
                or owner[1] not in self._active_kvduo_reqs
                or any(
                    (l, r) == owner for l, r, _ in self._kvduo_hot_pressure_protected
                )
                or any(
                    self._kvduo_hot_page_owners.get(start) != owner for start in pages
                )
            ):
                raise RuntimeError("KVDuo rollback ownership or protection changed")
            for start in pages[len(base) :]:
                released.extend(range(start, start + self.page_size))
        owner_tensor = torch.tensor(owners, dtype=torch.int64, device=self.device)
        targets = torch.tensor(
            [len(self._kvduo_hot_base_pages[o]) * self.hot_page_size for o in owners],
            dtype=torch.int32,
            device=self.device,
        )
        free_indices = torch.tensor(released, dtype=torch.int64, device=self.device)
        done = device_module.Event() if self.req_device_buffer_tokens.is_cuda else None
        rollback_kvduo_hot_metadata(
            owner_tensor,
            targets,
            self.req_device_buffer_tokens,
            self.req_device_buffer_token_locs,
            self.lru_slots,
            self.kvduo_req_hot_capacity_gpu,
            self.kvduo_capacity_ema,
            self.kvduo_capacity_steps,
            self.kvduo_hot_growth_events,
            self.kvduo_prefill_ema,
            self.kvduo_prefill_steps,
        )
        for owner in owners:
            # Resetting the evaluation stage invalidates its frozen reference;
            # the object's admitted cost remains reserved until recovery/end.
            self._kvduo_prefill_budget_reference.pop(self._kvduo_owner_key(owner), None)
        if done is not None:
            done.record(device_module.current_stream(self.device))
            done.synchronize()
        for owner in owners:
            pages = self._kvduo_req_layer_pages[owner]
            base = self._kvduo_hot_base_pages[owner]
            self.kvduo_req_hot_capacity[owner] = len(base) * self.hot_page_size
            for start in pages[len(base) :]:
                del self._kvduo_hot_page_owners[start]
            self._kvduo_req_layer_pages[owner] = list(base)
        self.token_to_kv_pool_allocator.free_hisparse_indices(free_indices)

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
        if self.enable_mixed_residency:
            self.process_kvduo_full_conversions()
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
        assert (
            not self.is_dsv4_hisparse
        ), "naive_load_topk is not implemented for dsv4 hisparse"
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

            assert torch.all(
                selected_tokens >= 0
            ), f"Req {req_idx}: selected tokens contain negative positions"
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
        if self.enable_mixed_residency:
            self._end_kvduo_prefill_lifecycle(req.req_pool_idx, req)
        # release resources only after the execution of a potential overlapped batch
        if self.decode_producer_stream is not None:
            device_module.current_stream(self.device).wait_stream(
                self.decode_producer_stream
            )
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
    ) -> int:
        """Demote cold request-owned full pages for mandatory physical pressure.

        The host copy is complete before requests leave staging. This control
        path may synchronize only when allocation pressure occurs; steady-state
        full/hot/host resolution remains entirely device-driven and graph-safe.
        """
        if not self.enable_mixed_residency or num_tokens <= 0:
            return 0
        if self.decode_producer_stream is not None:
            device_module.current_stream(self.device).wait_stream(
                self.decode_producer_stream
            )
        self.wait_for_pending_backup()
        candidate_pages = []
        candidate_meta = []
        for req_idx, req in sorted(self._active_kvduo_reqs.items()):
            if request_indices is not None and req_idx not in request_indices:
                continue
            if getattr(req, "hisparse_staging", False) or (
                callable(getattr(req, "finished", None)) and req.finished()
            ):
                continue
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
        if reserve_hot_minimum:
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
        selected = select_kvduo_full_pages(
            page_touch,
            valid,
            [owner_indices[req_idx] for req_idx, _ in candidate_meta],
            [minimum_costs.get(req_idx, 0) for req_idx in owners],
            (num_tokens + self.page_size - 1) // self.page_size,
            release_all=release_all,
        )

        if (
            not self._kvduo_consensus(tuple(candidate_meta[i] for i in selected))
            or not selected
        ):
            return 0
        selected_gpu = torch.tensor(selected, dtype=torch.int64, device=self.device)
        selected_logical = logical_pages[selected_gpu]
        selected_physical = physical_pages[selected_gpu]
        self.mem_pool_device.full_to_hisparse_device_index_mapping[
            selected_logical.flatten()
        ] = 0
        for i in selected:
            req_idx, start = candidate_meta[i]
            req = self._active_kvduo_reqs[req_idx]
            req.kvduo_mixed_residency = True
            req.kvduo_radix_insert_len = min(
                getattr(req, "kvduo_radix_insert_len", start * self.compress_ratio),
                start * self.compress_ratio,
            )
            self._mixed_slots[req_idx] = True
            self._kvduo_pending_hot_minimum.add(req_idx)
        if selected_logical.is_cuda:
            published = device_module.Event()
            published.record(device_module.current_stream(self.device))
            published.synchronize()
        self.token_to_kv_pool_allocator.free_hisparse_indices(
            selected_physical.flatten()
        )
        return len(selected) * self.page_size

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices."""
        if not self.is_dsv4_hisparse:
            layer_id -= getattr(self.mem_pool_device, "start_layer", 0)
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
            device_buffer=(
                self.mem_pool_device.get_flat_key_buffer()
                if self.enable_mixed_residency
                else self.mem_pool_device.kv_buffer[layer_id]
            ),
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=self.hot_page_size if self.enable_mixed_residency else 1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            req_hot_buffer_sizes=(
                self.kvduo_req_hot_capacity_gpu[layer_id]
                if self.enable_mixed_residency
                else self.req_device_buffer_size_gpu
            ),
            capacity_ema=(
                self.kvduo_capacity_ema[layer_id]
                if self.enable_mixed_residency
                else None
            ),
            capacity_steps=(
                self.kvduo_capacity_steps[layer_id]
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
            hot_growth_events=(
                self.kvduo_hot_growth_events[layer_id]
                if self.enable_mixed_residency
                else None
            ),
            observe_cache=self.kvduo_observe if self.enable_mixed_residency else None,
            hot_base_capacities=(
                self.kvduo_hot_base_capacity[layer_id]
                if self.enable_mixed_residency
                else None
            ),
            prefill_ema=(
                self.kvduo_prefill_ema[layer_id]
                if self.enable_mixed_residency
                else None
            ),
            prefill_steps=(
                self.kvduo_prefill_steps[layer_id]
                if self.enable_mixed_residency
                else None
            ),
            touch_clock=(
                self.full_touch_clock if self.enable_mixed_residency else None
            ),
            full_location_base=(
                layer_id * self.mem_pool_device.flat_layer_slot_stride
                if self.enable_mixed_residency
                else 0
            ),
            resolver_scratch=getattr(self, "resolver_scratch", None),
        )
        if self.enable_mixed_residency:
            # Capturable execution boundary: an invalid/mismatched host source
            # must stop this stream before sparse attention consumes the table.
            torch._assert_async(
                torch.all(self.kvduo_swap_status[layer_id, req_pool_indices] == 0),
                "KVDuo resolver produced an incomplete address table",
            )
        return top_k_indices
