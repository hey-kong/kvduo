from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@functools.cache
def _jit_sparse_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    is_mla: bool = False,
    is_dsv4_layout: bool = False,
) -> Module:
    template_args = make_cpp_args(
        block_size, num_top_k, hot_buffer_size, is_mla, is_dsv4_layout
    )
    cache_args = make_cpp_args(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla, is_dsv4_layout
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
    hot_page_last_touch: torch.Tensor | None = None,
    req_to_logical_token: torch.Tensor | None = None,
    full_to_device_loc: torch.Tensor | None = None,
    full_last_touch: torch.Tensor | None = None,
    full_data_version: torch.Tensor | None = None,
    full_host_version: torch.Tensor | None = None,
    swap_status: torch.Tensor | None = None,
    resolver_stats: torch.Tensor | None = None,
    touch_clock: torch.Tensor | None = None,
) -> None:
    assert hot_buffer_size >= num_top_k, (
        f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"
    )

    module = _jit_sparse_module(
        item_size_bytes,
        block_size,
        num_top_k,
        hot_buffer_size,
        is_mla=True,
        is_dsv4_layout=is_dsv4_layout,
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
    enable_dynamic_hot_view = (
        req_hot_buffer_sizes is not None and hot_page_last_touch is not None
    )
    if enable_dynamic_hot_view:
        assert req_hot_buffer_sizes.dtype == torch.int32
        assert req_hot_buffer_sizes.device == top_k_tokens.device
        assert hot_page_last_touch.dtype == torch.int64
        assert hot_page_last_touch.device == top_k_tokens.device
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
        hot_page_last_touch.data_ptr() if enable_dynamic_hot_view else 0,
        req_to_logical_token.data_ptr() if enable_full_lookup else 0,
        full_to_device_loc.data_ptr() if enable_full_lookup else 0,
        full_last_touch.data_ptr() if enable_full_lookup else 0,
        full_data_version.data_ptr() if enable_full_lookup else 0,
        full_host_version.data_ptr() if enable_full_lookup else 0,
        swap_status.data_ptr() if enable_full_lookup else 0,
        resolver_stats.data_ptr() if enable_full_lookup else 0,
        touch_clock.data_ptr(),
        torch.cuda.current_stream(top_k_tokens.device).cuda_stream,
        top_k_tokens.size(0),
        host_cache_locs.size(1),
        device_buffer_tokens.stride(0),
        lru_slots.stride(0),
        hot_page_last_touch.stride(0) if enable_dynamic_hot_view else 0,
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
    hot_page_last_touch: torch.Tensor | None = None,
    req_to_logical_token: torch.Tensor | None = None,
    full_to_device_loc: torch.Tensor | None = None,
    full_last_touch: torch.Tensor | None = None,
    full_data_version: torch.Tensor | None = None,
    full_host_version: torch.Tensor | None = None,
    swap_status: torch.Tensor | None = None,
    resolver_stats: torch.Tensor | None = None,
    touch_clock: torch.Tensor | None = None,
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
        hot_page_last_touch=hot_page_last_touch,
        req_to_logical_token=req_to_logical_token,
        full_to_device_loc=full_to_device_loc,
        full_last_touch=full_last_touch,
        full_data_version=full_data_version,
        full_host_version=full_host_version,
        swap_status=swap_status,
        resolver_stats=resolver_stats,
        touch_clock=touch_clock,
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
    hot_page_last_touch: torch.Tensor | None = None,
    req_to_logical_token: torch.Tensor | None = None,
    full_to_device_loc: torch.Tensor | None = None,
    full_last_touch: torch.Tensor | None = None,
    full_data_version: torch.Tensor | None = None,
    full_host_version: torch.Tensor | None = None,
    swap_status: torch.Tensor | None = None,
    resolver_stats: torch.Tensor | None = None,
    touch_clock: torch.Tensor | None = None,
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
        hot_page_last_touch=hot_page_last_touch,
        req_to_logical_token=req_to_logical_token,
        full_to_device_loc=full_to_device_loc,
        full_last_touch=full_last_touch,
        full_data_version=full_data_version,
        full_host_version=full_host_version,
        swap_status=swap_status,
        resolver_stats=resolver_stats,
        touch_clock=touch_clock,
    )
