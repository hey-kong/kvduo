from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def hisparse_resolver_scratch_words(
    num_top_k: int,
    hot_buffer_size: int,
    *,
    warp_size: int | None = None,
    use_lru_scratch: bool = False,
) -> int:
    """Size of the full or compact resolver workspace in int32 words."""
    if warp_size is None:
        warp_size = 64 if torch.version.hip else 32
    chunks = (hot_buffer_size + warp_size - 1) // warp_size
    hash_size = 2 * num_top_k
    int32_count = num_top_k + 2 * (chunks + 1) + hash_size + 4
    slot_bytes = 4 if hot_buffer_size > torch.iinfo(torch.int16).max else 2
    slots = hash_size + (0 if use_lru_scratch else hot_buffer_size)
    required_bytes = 4 * int32_count + slot_bytes * slots
    return (required_bytes + 3) // 4


def allocate_hisparse_resolver_scratch(
    requests: int,
    num_top_k: int,
    hot_buffer_size: int,
    block_size: int,
    device: torch.device,
    *,
    use_lru_scratch: bool = False,
) -> torch.Tensor | None:
    """Check launch resources and allocate before capture when shared memory is too small.

    One row per request is reused across serialized layer calls, rather than
    reserving scratch for every (request, layer). The resolver fully initializes
    each used row on every invocation, including after request-slot reuse.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return None
    properties = torch.cuda.get_device_properties(device)
    warp_size = 64 if torch.version.hip else 32
    max_threads = getattr(properties, "max_threads_per_block", 1024)
    if block_size < warp_size or block_size % warp_size or block_size > max_threads:
        raise ValueError(
            f"HiSparse resolver block_size={block_size} must be a multiple of "
            f"{warp_size} and at most {max_threads} on {device}"
        )
    shared_limit = max(
        properties.shared_memory_per_block,
        getattr(properties, "shared_memory_per_block_optin", 0),
    )
    words = hisparse_resolver_scratch_words(
        num_top_k,
        hot_buffer_size,
        warp_size=warp_size,
        use_lru_scratch=use_lru_scratch,
    )
    # Four small shared counters remain even on the global path. Reserve
    # compiler alignment headroom; the CUDA wrapper checks actual static usage.
    if words * 4 + 256 <= shared_limit:
        return None
    words = hisparse_resolver_scratch_words(
        num_top_k, hot_buffer_size, warp_size=warp_size
    )
    return torch.empty((requests, words), dtype=torch.int32, device=device)


@functools.cache
def _jit_sparse_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    is_mla: bool = False,
    is_dsv4_layout: bool = False,
    use_global_scratch: bool = False,
    use_lru_scratch: bool = False,
    validate_status: bool = False,
) -> Module:
    template_args = make_cpp_args(
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla,
        is_dsv4_layout,
        use_global_scratch,
        use_lru_scratch,
        validate_status,
    )
    cache_args = make_cpp_args(
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla,
        is_dsv4_layout,
        use_global_scratch,
        use_lru_scratch,
        validate_status,
    )
    return load_jit(
        "sparse_cache",
        *cache_args,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer",
                f"load_cache_to_device_buffer<{template_args}>",
            )
        ],
    )


@functools.cache
def _jit_dsv4_transfer_module(block_size: int) -> Module:
    template_args = make_cpp_args(block_size)
    return load_jit(
        "sparse_cache_dsv4_transfer",
        block_size,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "transfer_cache_dsv4_mla",
                f"transfer_cache_dsv4_mla<{template_args}>",
            )
        ],
    )


def transfer_cache_dsv4_mla(
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    block_size: int = 1024,
) -> None:
    """Transfer DSv4 C4 tokens between page-padded C4 buffers."""
    module = _jit_dsv4_transfer_module(block_size)
    module.transfer_cache_dsv4_mla(
        src_ptrs,
        dst_ptrs,
        src_indices,
        dst_indices,
    )


def _load_cache_to_device_buffer_mla(
    *,
    is_dsv4_layout: bool,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int,
    block_size: int,
    num_real_reqs: torch.Tensor | None,
    req_hot_buffer_sizes: torch.Tensor | None = None,
    capacity_ema: torch.Tensor | None = None,
    capacity_steps: torch.Tensor | None = None,
    req_to_logical_token: torch.Tensor | None = None,
    full_to_device_loc: torch.Tensor | None = None,
    full_last_touch: torch.Tensor | None = None,
    full_data_version: torch.Tensor | None = None,
    full_host_version: torch.Tensor | None = None,
    swap_status: torch.Tensor | None = None,
    resolver_stats: torch.Tensor | None = None,
    touch_clock: torch.Tensor | None = None,
    full_location_base: int = 0,
    hot_growth_events: torch.Tensor | None = None,
    observe_cache: torch.Tensor | None = None,
    hot_base_capacities: torch.Tensor | None = None,
    prefill_ema: torch.Tensor | None = None,
    prefill_steps: torch.Tensor | None = None,
    resolver_scratch: torch.Tensor | None = None,
    lru_scratch: torch.Tensor | None = None,
    validate_status: bool = False,
) -> None:
    assert (
        hot_buffer_size >= num_top_k
    ), f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"
    assert lru_slots.dtype == (
        torch.int32 if hot_buffer_size > torch.iinfo(torch.int16).max else torch.int16
    ), "LRU slot dtype must match the compiled HOT capacity"
    if resolver_scratch is not None:
        assert resolver_scratch.dtype == torch.int32
        assert resolver_scratch.ndim == 2 and resolver_scratch.is_contiguous()
        assert resolver_scratch.device == top_k_tokens.device
        assert resolver_scratch.size(0) == device_buffer_tokens.size(0)
        assert resolver_scratch.size(1) >= hisparse_resolver_scratch_words(
            num_top_k, hot_buffer_size
        )
    if lru_scratch is not None:
        assert resolver_scratch is None
        assert lru_scratch.dtype == lru_slots.dtype
        assert lru_scratch.shape == (device_buffer_tokens.size(0), hot_buffer_size)
        assert lru_scratch.is_contiguous() and lru_scratch.device == top_k_tokens.device
        assert lru_scratch.data_ptr() != lru_slots.data_ptr()

    module = _jit_sparse_module(
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla=True,
        is_dsv4_layout=is_dsv4_layout,
        use_global_scratch=resolver_scratch is not None,
        use_lru_scratch=lru_scratch is not None,
        validate_status=validate_status,
    )

    # Pass only scalars through TVM FFI.  Its PyTorch fallback can reject tensor
    # arguments while CUDA graph capture is active, before the C++ wrapper is
    # entered.  All allocations below outlive the captured graph, so passing
    # their stable addresses is safe and also avoids allocations during capture.
    enable_full_lookup = (
        req_to_logical_token is not None
        and full_to_device_loc is not None
        and full_last_touch is not None
        and full_data_version is not None
        and full_host_version is not None
        and swap_status is not None
        and resolver_stats is not None
        and touch_clock is not None
    )
    if validate_status and not enable_full_lookup:
        raise ValueError("Fused resolver validation requires complete KVDuo metadata")
    if enable_full_lookup:
        assert req_to_logical_token.dtype == torch.int64
        assert full_to_device_loc.dtype == torch.int64
        assert full_last_touch.dtype == torch.int64
        assert touch_clock.dtype == torch.int64
        assert full_data_version.dtype == torch.int64
        assert full_host_version.dtype == torch.int64
        assert swap_status.dtype == torch.int32
        assert resolver_stats.dtype == torch.int32
        assert resolver_stats.ndim == 2 and resolver_stats.size(1) == 2
        assert req_to_logical_token.device == top_k_tokens.device
        assert full_to_device_loc.device == top_k_tokens.device
        assert full_last_touch.device == top_k_tokens.device
        assert touch_clock.device == top_k_tokens.device
        assert full_data_version.device == top_k_tokens.device
        assert full_host_version.device == top_k_tokens.device
        assert swap_status.device == top_k_tokens.device
        assert resolver_stats.device == top_k_tokens.device
    if touch_clock is None:
        touch_clock = torch.zeros(1, dtype=torch.int64, device=top_k_tokens.device)
    assert touch_clock.dtype == torch.int64
    assert touch_clock.device == top_k_tokens.device

    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [top_k_tokens.size(0)], dtype=torch.int32, device=top_k_tokens.device
        )
    enable_dynamic_hot_view = enable_full_lookup and req_hot_buffer_sizes is not None
    if lru_scratch is not None:
        assert enable_dynamic_hot_view
    if enable_dynamic_hot_view:
        assert req_hot_buffer_sizes.dtype == torch.int32
        assert req_hot_buffer_sizes.device == top_k_tokens.device
    if hot_growth_events is not None:
        assert enable_full_lookup and enable_dynamic_hot_view
        assert hot_growth_events.dtype == torch.int32
        assert hot_growth_events.ndim == 1 and hot_growth_events.is_contiguous()
        assert hot_growth_events.numel() == device_buffer_tokens.size(0)
        assert hot_growth_events.device == top_k_tokens.device
    if capacity_ema is not None or capacity_steps is not None:
        assert capacity_ema is not None and capacity_steps is not None
        assert enable_full_lookup and enable_dynamic_hot_view
        assert (
            capacity_ema.dtype == torch.float32 and capacity_steps.dtype == torch.int64
        )
        assert capacity_ema.shape == (device_buffer_tokens.size(0), 9)
        assert capacity_steps.shape == (device_buffer_tokens.size(0),)
        for tensor in (capacity_ema, capacity_steps):
            assert tensor.is_contiguous() and tensor.device == top_k_tokens.device
    if observe_cache is not None:
        assert observe_cache.dtype == torch.int32 and observe_cache.numel() == 1
        assert (
            observe_cache.is_contiguous()
            and observe_cache.device == top_k_tokens.device
        )
    if any(t is not None for t in (hot_base_capacities, prefill_ema, prefill_steps)):
        assert all(
            t is not None for t in (hot_base_capacities, prefill_ema, prefill_steps)
        )
        assert enable_full_lookup and enable_dynamic_hot_view
        assert hot_base_capacities.dtype == torch.int32
        assert prefill_ema.dtype == torch.float32
        assert prefill_steps.dtype == torch.int64
        requests = device_buffer_tokens.size(0)
        assert hot_base_capacities.shape == (requests,)
        assert prefill_steps.shape == (requests,)
        assert prefill_ema.shape == (requests, 2)
        for tensor in (hot_base_capacities, prefill_ema, prefill_steps):
            assert tensor.is_contiguous() and tensor.device == top_k_tokens.device
    module.load_cache_to_device_buffer(
        top_k_tokens.data_ptr(),
        device_buffer_tokens.data_ptr(),
        host_cache_locs.data_ptr(),
        device_buffer_locs.data_ptr(),
        host_cache.data_ptr(),
        device_buffer.data_ptr(),
        top_k_device_locs.data_ptr(),
        req_pool_indices.data_ptr(),
        seq_lens.data_ptr(),
        lru_slots.data_ptr(),
        num_real_reqs.data_ptr(),
        req_hot_buffer_sizes.data_ptr() if enable_dynamic_hot_view else 0,
        capacity_ema.data_ptr() if capacity_ema is not None else 0,
        req_to_logical_token.data_ptr() if enable_full_lookup else 0,
        full_to_device_loc.data_ptr() if enable_full_lookup else 0,
        full_last_touch.data_ptr() if enable_full_lookup else 0,
        full_data_version.data_ptr() if enable_full_lookup else 0,
        full_host_version.data_ptr() if enable_full_lookup else 0,
        swap_status.data_ptr() if enable_full_lookup else 0,
        resolver_stats.data_ptr() if enable_full_lookup else 0,
        hot_growth_events.data_ptr() if hot_growth_events is not None else 0,
        touch_clock.data_ptr(),
        torch.cuda.current_stream(top_k_tokens.device).cuda_stream,
        top_k_tokens.size(0),
        host_cache_locs.size(1),
        device_buffer_tokens.stride(0),
        lru_slots.stride(0),
        capacity_steps.data_ptr() if capacity_steps is not None else 0,
        top_k_tokens.stride(0),
        top_k_device_locs.stride(0),
        req_to_logical_token.stride(0) if enable_full_lookup else 0,
        resolver_stats.stride(0) if enable_full_lookup else 0,
        seq_lens.dtype == torch.int64,
        req_pool_indices.dtype == torch.int64,
        enable_full_lookup,
        enable_dynamic_hot_view,
        page_size,
        item_size_bytes,
        full_location_base,
        observe_cache.data_ptr() if observe_cache is not None else 0,
        hot_base_capacities.data_ptr() if hot_base_capacities is not None else 0,
        prefill_ema.data_ptr() if prefill_ema is not None else 0,
        prefill_steps.data_ptr() if prefill_steps is not None else 0,
        resolver_scratch.data_ptr() if resolver_scratch is not None else 0,
        resolver_scratch.stride(0) if resolver_scratch is not None else 0,
        lru_scratch.data_ptr() if lru_scratch is not None else 0,
        lru_scratch.stride(0) if lru_scratch is not None else 0,
    )


def load_cache_to_device_buffer_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
    req_hot_buffer_sizes: torch.Tensor | None = None,
    capacity_ema: torch.Tensor | None = None,
    capacity_steps: torch.Tensor | None = None,
    req_to_logical_token: torch.Tensor | None = None,
    full_to_device_loc: torch.Tensor | None = None,
    full_last_touch: torch.Tensor | None = None,
    full_data_version: torch.Tensor | None = None,
    full_host_version: torch.Tensor | None = None,
    swap_status: torch.Tensor | None = None,
    resolver_stats: torch.Tensor | None = None,
    touch_clock: torch.Tensor | None = None,
    full_location_base: int = 0,
    hot_growth_events: torch.Tensor | None = None,
    observe_cache: torch.Tensor | None = None,
    hot_base_capacities: torch.Tensor | None = None,
    prefill_ema: torch.Tensor | None = None,
    prefill_steps: torch.Tensor | None = None,
    resolver_scratch: torch.Tensor | None = None,
    lru_scratch: torch.Tensor | None = None,
    validate_status: bool = False,
) -> None:
    """Generic MLA hisparse swap-in: device + host both linear (stride=item_size_bytes)."""
    _load_cache_to_device_buffer_mla(
        is_dsv4_layout=False,
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=page_size,
        block_size=block_size,
        num_real_reqs=num_real_reqs,
        req_hot_buffer_sizes=req_hot_buffer_sizes,
        capacity_ema=capacity_ema,
        capacity_steps=capacity_steps,
        req_to_logical_token=req_to_logical_token,
        full_to_device_loc=full_to_device_loc,
        full_last_touch=full_last_touch,
        full_data_version=full_data_version,
        full_host_version=full_host_version,
        swap_status=swap_status,
        resolver_stats=resolver_stats,
        touch_clock=touch_clock,
        full_location_base=full_location_base,
        hot_growth_events=hot_growth_events,
        observe_cache=observe_cache,
        hot_base_capacities=hot_base_capacities,
        prefill_ema=prefill_ema,
        prefill_steps=prefill_steps,
        resolver_scratch=resolver_scratch,
        lru_scratch=lru_scratch,
        validate_status=validate_status,
    )


def load_cache_to_device_buffer_dsv4_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
    req_hot_buffer_sizes: torch.Tensor | None = None,
    capacity_ema: torch.Tensor | None = None,
    capacity_steps: torch.Tensor | None = None,
    req_to_logical_token: torch.Tensor | None = None,
    full_to_device_loc: torch.Tensor | None = None,
    full_last_touch: torch.Tensor | None = None,
    full_data_version: torch.Tensor | None = None,
    full_host_version: torch.Tensor | None = None,
    swap_status: torch.Tensor | None = None,
    resolver_stats: torch.Tensor | None = None,
    touch_clock: torch.Tensor | None = None,
    full_location_base: int = 0,
    hot_growth_events: torch.Tensor | None = None,
    observe_cache: torch.Tensor | None = None,
    hot_base_capacities: torch.Tensor | None = None,
    prefill_ema: torch.Tensor | None = None,
    prefill_steps: torch.Tensor | None = None,
    resolver_scratch: torch.Tensor | None = None,
    lru_scratch: torch.Tensor | None = None,
    validate_status: bool = False,
) -> None:
    """DSv4 hisparse swap-in: page-padded device + page-padded host C4 layout."""
    _load_cache_to_device_buffer_mla(
        is_dsv4_layout=True,
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=top_k_device_locs,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=num_top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=page_size,
        block_size=block_size,
        num_real_reqs=num_real_reqs,
        req_hot_buffer_sizes=req_hot_buffer_sizes,
        capacity_ema=capacity_ema,
        capacity_steps=capacity_steps,
        req_to_logical_token=req_to_logical_token,
        full_to_device_loc=full_to_device_loc,
        full_last_touch=full_last_touch,
        full_data_version=full_data_version,
        full_host_version=full_host_version,
        swap_status=swap_status,
        resolver_stats=resolver_stats,
        touch_clock=touch_clock,
        full_location_base=full_location_base,
        hot_growth_events=hot_growth_events,
        observe_cache=observe_cache,
        hot_base_capacities=hot_base_capacities,
        prefill_ema=prefill_ema,
        prefill_steps=prefill_steps,
        resolver_scratch=resolver_scratch,
        lru_scratch=lru_scratch,
        validate_status=validate_status,
    )


@functools.cache
def _jit_kvduo_merge_module(capacity: int, block_size: int):
    args = make_cpp_args(capacity, block_size)
    return load_jit(
        "kvduo_hot_merge",
        capacity,
        block_size,
        cuda_files=["kvduo_hot_merge.cuh"],
        cuda_wrappers=[("plan", f"kvduo_hot_merge_plan<{args}>")],
    )


def build_kvduo_hot_merge_plan(
    tokens: torch.Tensor,
    locations: torch.Tensor,
    lru: torch.Tensor,
    capacity: int,
    k: int,
    target_pages: int,
) -> tuple[torch.Tensor, ...]:
    assert tokens.is_cuda
    assert tokens.ndim == locations.ndim == lru.ndim == 1
    assert tokens.shape == locations.shape and tokens.numel() >= lru.numel()
    assert tokens.dtype == locations.dtype == torch.int32
    assert lru.dtype == (
        torch.int32 if lru.numel() > torch.iinfo(torch.int16).max else torch.int16
    )
    for tensor in (tokens, locations, lru):
        assert tensor.device == tokens.device and tensor.is_contiguous()
    hash_size = 1 << (2 * capacity - 1).bit_length()
    workspace = torch.empty(
        4 * capacity + hash_size, dtype=torch.int32, device=tokens.device
    )
    summary = torch.empty(3, dtype=torch.int32, device=tokens.device)
    new_tokens = torch.empty_like(tokens)
    new_locations = torch.empty_like(locations)
    new_lru = torch.empty_like(lru)
    sources = torch.empty(capacity, dtype=torch.int64, device=tokens.device)
    destinations = torch.empty_like(sources)
    _jit_kvduo_merge_module(lru.numel(), 256).plan(
        tokens,
        locations,
        lru,
        new_tokens,
        new_locations,
        new_lru,
        sources,
        destinations,
        workspace,
        summary,
        capacity,
        k,
        target_pages,
        hash_size,
    )
    return summary, new_tokens, new_locations, new_lru, sources, destinations


@functools.cache
def _jit_kvduo_rollback_module(capacity: int, block_size: int):
    args = make_cpp_args(capacity, block_size)
    return load_jit(
        "kvduo_prefill_rollback",
        capacity,
        block_size,
        cuda_files=["kvduo_prefill_rollback.cuh"],
        cuda_wrappers=[("rollback", f"kvduo_prefill_rollback<{args}>")],
    )


def rollback_kvduo_hot_metadata(
    owners: torch.Tensor,
    targets: torch.Tensor,
    tokens: torch.Tensor,
    locations: torch.Tensor,
    lru: torch.Tensor,
    capacities: torch.Tensor,
    capacity_ema: torch.Tensor | None = None,
    capacity_steps: torch.Tensor | None = None,
    growth_events: torch.Tensor | None = None,
    prefill_ema: torch.Tensor | None = None,
    prefill_steps: torch.Tensor | None = None,
) -> None:
    """Batch metadata compaction and optional statistics reset without moving KV."""
    if owners.numel() == 0:
        return
    statistics = (
        capacity_ema,
        capacity_steps,
        growth_events,
        prefill_ema,
        prefill_steps,
    )
    reset_stats = any(bank is not None for bank in statistics)
    if reset_stats:
        shapes = (
            (*capacities.shape, 9),
            capacities.shape,
            capacities.shape,
            (*capacities.shape, 2),
            capacities.shape,
        )
        dtypes = (
            torch.float32,
            torch.int64,
            torch.int32,
            torch.float32,
            torch.int64,
        )
        for bank, shape, dtype in zip(statistics, shapes, dtypes):
            if (
                bank is None
                or bank.shape != shape
                or bank.dtype != dtype
                or bank.device != capacities.device
                or not bank.is_contiguous()
            ):
                raise ValueError(
                    "Rollback statistics must be supplied together with matching "
                    "shapes, dtypes, devices and contiguous layout"
                )
    if not tokens.is_cuda:
        # CPU reference also exercises publication/free ordering in unit tests.
        for (layer, req), target in zip(owners.tolist(), targets.tolist()):
            old = int(capacities[layer, req])
            order = lru[layer, req, :old].clone()
            filtered = order[(order >= 0) & (order < target)]
            lru[layer, req, :target] = filtered
            lru[layer, req, target:] = -1
            tokens[layer, req, target:] = -1
            locations[layer, req, target:] = -1
            if reset_stats:
                for bank in statistics:
                    bank[layer, req] = 0
            capacities[layer, req] = target
        return
    assert owners.dtype == torch.int64 and owners.is_contiguous()
    assert targets.dtype == torch.int32 and targets.is_contiguous()
    assert owners.ndim == 2 and owners.size(1) == 2
    assert targets.numel() == owners.size(0)
    for tensor in (tokens, locations, lru, capacities):
        assert tensor.is_contiguous() and tensor.device == owners.device
    assert lru.dtype == (
        torch.int32 if lru.size(2) > torch.iinfo(torch.int16).max else torch.int16
    ), "LRU slot dtype must match the compiled HOT capacity"
    # The coordinator pads token/location rows by one page, but LRU
    # contains only HOT slots. Compile the scan for LRU's logical capacity;
    # the wrapper passes each bank's actual strides and padded width.
    _jit_kvduo_rollback_module(lru.size(2), 256).rollback(
        owners,
        targets,
        tokens,
        locations,
        lru,
        capacities,
        *statistics,
    )
