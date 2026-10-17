"""Warm pressure-only KVDuo kernels using private startup allocations."""

import logging
import time

import torch

from sglang.kernels.ops.kvcache.kvduo_full_reclaim import (
    select_kvduo_full_requests,
    summarize_kvduo_full_pages,
)

logger = logging.getLogger(__name__)


@torch.inference_mode()
def warmup_kvduo_reclaim(
    *, backing, page_size, storage_layers, top_k, paged, block_size=256
):
    """Compile, load and execute reclamation before the worker becomes ready.

    `backing` supplies only layout; real payload and allocator stay untouched.
    """
    if backing.device.type != "cuda":
        return
    started = time.perf_counter()
    logger.info("KVDuo reclamation kernel warmup started")
    with torch.cuda.device(backing.device):
        _warmup(backing, page_size, storage_layers, top_k, paged, block_size)
        torch.cuda.current_stream(backing.device).synchronize()
    logger.info(
        "KVDuo reclamation kernel warmup completed in %.2f s",
        time.perf_counter() - started,
    )


def _warmup(backing, page_size, storage_layers, top_k, paged, block_size=256):
    device = backing.device
    k = page_size * storage_layers
    # Both FULL index types are supported by runtime callers. Exercise request selection, including a nonzero mandatory HOT charge and protected page.
    indices = torch.arange(4 * page_size, dtype=torch.int64, device=device)
    mapping = indices + page_size
    versions = torch.ones_like(indices)
    for dtype in (torch.int32, torch.int64):
        pages = indices.to(dtype).view(4, page_size)
        _, scores, valid = summarize_kvduo_full_pages(
            pages, mapping, versions, versions, indices, indices[:1]
        )
        if not select_kvduo_full_requests(valid, [0, 0, 1, 1], [0, 1], [0, 1], 1):
            raise RuntimeError("KVDuo FULL request warmup failed")

    from sglang.kernels.ops.kvcache.kvduo_frequency import launch_resolve, merge
    from sglang.srt.mem_cache.sparsity.core.kvduo_frequency import (
        FrequencyBuffer,
        pack_frequency,
    )

    minimum = (top_k + k - 1) // k
    capacity = (minimum + 1) * k
    rows = (
        (capacity + page_size + page_size - 1) // page_size
        if paged
        else capacity + page_size
    )
    payload = torch.zeros(
        (rows, *backing.shape[1:]), dtype=backing.dtype, device=device
    )
    host_rows = (top_k + page_size - 1) // page_size if paged else top_k
    host_payload = torch.ones(
        (host_rows, *backing.shape[1:]), dtype=backing.dtype, pin_memory=True
    )
    buffer = FrequencyBuffer.empty(
        torch.arange(page_size, page_size + capacity, dtype=torch.int32, device=device)
    )
    pointers = torch.tensor([buffer.pointers()], dtype=torch.int64, device=device)
    positions = torch.arange(top_k, dtype=torch.int64, device=device)
    status = torch.zeros(1, dtype=torch.int32, device=device)
    output = torch.empty((1, top_k), dtype=torch.int32, device=device)
    stats = torch.zeros((1, 2), dtype=torch.int32, device=device)
    i32 = lambda value: torch.tensor([value], dtype=torch.int32, device=device)
    i64 = lambda value: torch.tensor([value], dtype=torch.int64, device=device)
    published_capacity, growth = i32(capacity), i32(0)
    launch_resolve(
        top_k,
        paged,
        positions.int().view(1, -1),
        i64(0),
        i64(top_k),
        pointers,
        published_capacity,
        i32(1),
        positions.view(1, -1),
        torch.zeros_like(positions),
        torch.zeros_like(positions),
        torch.ones_like(positions),
        torch.ones_like(positions),
        positions.view(1, -1),
        host_payload,
        payload,
        backing[0].numel() * backing.element_size(),
        0,
        i64(1),
        i32(1),
        output,
        status,
        stats,
        growth,
        torch.empty((1, 9 * top_k), dtype=torch.int32, device=device),
        block_size=block_size,
    )
    if status.item() or stats[0, 0].item() != top_k:
        raise RuntimeError("KVDuo frequency resolver warmup failed")
    # Force survivors on both kept and released pages so merge warms copying.
    buffer.tokens.copy_(torch.arange(capacity, dtype=torch.int32, device=device))
    freq = [0] * capacity
    survivors = [page * k for page in range(minimum + int(k > 1))]
    for slot in survivors:
        freq[slot] = 3
    buffer.frequency.copy_(pack_frequency(freq, device=device))
    merged, kept = merge(
        buffer,
        payload,
        k=k,
        minimum=minimum,
        paged=paged,
        top_k=top_k,
        item_bytes=backing[0].numel() * backing.element_size(),
        publication=(pointers[0], published_capacity, growth),
    )
    if (
        int(published_capacity.item()) != minimum * k
        or merged.tokens.numel() != minimum * k
        or sorted(merged.tokens[merged.tokens >= 0].tolist()) != survivors
    ):
        raise RuntimeError("KVDuo frequency merge warmup failed")
