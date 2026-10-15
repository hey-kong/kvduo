"""Warm pressure-only KVDuo kernels using private startup allocations."""

import logging
import time

import torch

from sglang.kernels.ops.kvcache.kvduo_full_reclaim import (
    select_kvduo_full_pages,
    select_kvduo_full_requests,
    summarize_kvduo_full_pages,
)
from sglang.kernels.ops.kvcache.kvduo_hand_plan import (
    allocate_kvduo_hand_workspace,
    prepare_kvduo_hand,
)
from sglang.kernels.ops.kvcache.kvduo_hot_growth import publish_kvduo_hot_growth
from sglang.kernels.ops.kvcache.kvduo_hot_merge import (
    allocate_kvduo_merge_workspace,
    execute_kvduo_hot_merge,
    finish_kvduo_hot_merge,
)

logger = logging.getLogger(__name__)


@torch.inference_mode()
def warmup_kvduo_reclaim(*, backing, page_size, storage_layers, maximum, padded, paged):
    """Compile, load and execute reclamation before the worker becomes ready.

    `backing` supplies only device/dtype/layout; no real payload, metadata or
    allocator is mutated. Use the production maximum for merge specialization.
    """
    if backing.device.type != "cuda":
        return
    started = time.perf_counter()
    logger.info("KVDuo reclamation kernel warmup started")
    with torch.cuda.device(backing.device):
        _warmup(backing, page_size, storage_layers, maximum, padded, paged)
        torch.cuda.current_stream(backing.device).synchronize()
    logger.info(
        "KVDuo reclamation kernel warmup completed in %.2f s",
        time.perf_counter() - started,
    )


def _warmup(backing, page_size, storage_layers, maximum, padded, paged):
    device = backing.device
    k = page_size * storage_layers
    if maximum != 8 * k or padded < maximum or k < 2 or (paged and page_size != 64):
        raise ValueError("Invalid KVDuo warmup geometry")
    # Both FULL index types are supported by runtime callers. Exercise both
    # policies, including a nonzero mandatory HOT charge and protected page.
    indices = torch.arange(4 * page_size, dtype=torch.int64, device=device)
    mapping = indices + page_size
    versions = torch.ones_like(indices)
    for dtype in (torch.int32, torch.int64):
        pages = indices.to(dtype).view(4, page_size)
        _, scores, valid = summarize_kvduo_full_pages(
            pages, mapping, versions, versions, indices, indices[:1]
        )
        if not select_kvduo_full_pages(scores, valid, [0, 0, 1, 1], [0, 1], 1):
            raise RuntimeError("KVDuo FULL page warmup failed")
        if not select_kvduo_full_requests(valid, [0, 0, 1, 1], [0, 1], [0, 1], 1):
            raise RuntimeError("KVDuo FULL request warmup failed")

    counts = torch.zeros((1, 2, 9), dtype=torch.float32, device=device)
    steps = torch.full((1, 2), 65, dtype=torch.int64, device=device)
    counts[:, :, 8] = 1
    hand = allocate_kvduo_hand_workspace(counts)
    hand.pages_host.fill_(2)
    snapshot = prepare_kvduo_hand(counts, steps, hand, hand=0, minimum=1, rho=0.1)
    if not snapshot.next(0, 1):
        raise RuntimeError("KVDuo Hand warmup failed")

    tokens = torch.full((padded,), -1, dtype=torch.int32, device=device)
    locations = torch.full_like(tokens, -1)
    slot_dtype = torch.int32 if maximum > 32767 else torch.int16
    lru = torch.full((maximum,), -1, dtype=slot_dtype, device=device)
    capacity = torch.zeros(1, dtype=torch.int32, device=device)
    ema = torch.zeros(9, dtype=torch.float32, device=device)
    observations = torch.zeros(1, dtype=torch.int64, device=device)
    events = torch.ones(1, dtype=torch.int32, device=device)
    # Three physical pages per storage layer: sentinel plus two HOT pages.
    stride = 3 * page_size
    for current in (0, k):
        publish_kvduo_hot_growth(
            torch.tensor(
                [page_size * (current // k + 1)], dtype=torch.int64, device=device
            ),
            tokens,
            locations,
            lru,
            lru.clone(),
            capacity,
            ema,
            observations,
            current=current,
            target=current + k,
            page_size=page_size,
            storage_layers=storage_layers,
            layer_stride=stride,
        )
    tokens[: 2 * k] = torch.arange(2 * k, dtype=torch.int32, device=device)
    # Interleaving LRU retains entries on BOTH pages, forcing a real copy when
    # one page is released instead of merely publishing a smaller capacity.
    lru[: 2 * k] = (
        torch.arange(2 * k, device=device).view(2, k).T.flatten().to(slot_dtype)
    )
    # C4 uses packed 64-token pages; generic DSA addresses individual rows.
    rows = 3 * storage_layers if paged else stride * storage_layers
    payload = torch.empty(
        (rows, *backing.shape[1:]), dtype=backing.dtype, device=device
    )
    payload.copy_(
        torch.arange(rows, device=device)
        .remainder(251)
        .view(rows, *([1] * (backing.ndim - 1)))
    )
    workspace = allocate_kvduo_merge_workspace(tokens, lru)
    execute_kvduo_hot_merge(
        tokens,
        locations,
        lru,
        payload,
        workspace,
        capacity=2 * k,
        k=k,
        top_k=1,
        target_pages=1,
        paged=paged,
        published_capacity=capacity,
        capacity_ema=ema,
        capacity_steps=observations,
        growth_events=events,
    )
    kept, released = finish_kvduo_hot_merge(workspace, 2, 1, k)
    if len(kept) != 1 or len(released) != 1 or int(workspace.summary_host[1]) <= 0:
        raise RuntimeError("KVDuo merge warmup did not exercise payload movement")
