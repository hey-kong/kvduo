from __future__ import annotations

import functools

import torch

from sglang.kernels.jit.utils import load_jit


@functools.cache
def _jit_hot_growth_module():
    return load_jit(
        "kvduo_hot_growth",
        cuda_files=["kvduo_hot_growth.cuh"],
        cuda_wrappers=[("publish", "kvduo_publish_hot_growth<256>")],
    )


def publish_kvduo_hot_growth(
    page_starts: torch.Tensor,
    tokens: torch.Tensor,
    locations: torch.Tensor,
    lru: torch.Tensor,
    previous_lru: torch.Tensor,
    capacity: torch.Tensor,
    capacity_ema: torch.Tensor,
    capacity_steps: torch.Tensor,
    prefill_ema: torch.Tensor,
    prefill_steps: torch.Tensor,
    *,
    current: int,
    target: int,
    page_size: int,
    storage_layers: int,
    layer_stride: int,
    reset_prefill: bool,
) -> None:
    """Publish one prepared HOT growth transaction without allocating KV scratch."""
    K = page_size * storage_layers
    if (
        page_size <= 0
        or storage_layers <= 0
        or current < 0
        or target <= current
        or current % K
        or target % K
        or target > lru.numel()
        or target > tokens.numel()
        or tokens.shape != locations.shape
        or previous_lru.numel() < current
        or page_starts.numel() != (target - current) // K
        or target > 8 * K
        or (lru.dtype == torch.int16 and target > 32767)
    ):
        raise ValueError("Invalid whole-page HOT growth geometry")
    expected = (
        (page_starts, torch.int64, page_starts.shape),
        (tokens, torch.int32, tokens.shape),
        (locations, torch.int32, tokens.shape),
        (lru, lru.dtype, lru.shape),
        (previous_lru, lru.dtype, previous_lru.shape),
        (capacity, torch.int32, (1,)),
        (capacity_ema, torch.float32, (9,)),
        (capacity_steps, torch.int64, (1,)),
        (prefill_ema, torch.float32, (2,)),
        (prefill_steps, torch.int64, (1,)),
    )
    if lru.dtype not in (torch.int16, torch.int32) or any(
        bank.ndim != 1
        or bank.shape != shape
        or bank.dtype != dtype
        or (bank is not page_starts and bank.stride(0) != 1)
        or bank.device != tokens.device
        for bank, dtype, shape in expected
    ):
        raise ValueError("HOT growth requires matching contiguous metadata rows")
    if current and (
        previous_lru.data_ptr() < lru.data_ptr() + target * lru.element_size()
        and lru.data_ptr()
        < previous_lru.data_ptr() + current * previous_lru.element_size()
    ):
        raise ValueError("HOT growth requires an independent LRU snapshot")
    if tokens.is_cuda:
        _jit_hot_growth_module().publish(
            *(bank for bank, _, _ in expected),
            current,
            target,
            page_size,
            storage_layers,
            layer_stride,
            reset_prefill,
        )
    else:
        offsets = torch.arange(K, dtype=torch.int64)
        locations[current:target] = (
            (
                page_starts[:, None]
                + (offsets // page_size)[None, :] * layer_stride
                + (offsets % page_size)[None, :]
            )
            .flatten()
            .int()
        )
        tokens[current:target] = -1
        lru[:target] = torch.cat(
            (torch.arange(current, target, dtype=lru.dtype), previous_lru[:current])
        )
        if current == 0:
            capacity_ema.zero_()
            capacity_steps.zero_()
        else:
            first, end = current // K, target // K
            capacity_ema[first:end] = capacity_ema[first - 1]
        if reset_prefill:
            prefill_ema.zero_()
            prefill_steps.zero_()
        capacity.fill_(target)
