"""Execute one KVDuo HOT merge without a planning round trip to the host."""

from dataclasses import dataclass

import torch

from sglang.kernels.ops.kvcache.hisparse import _jit_kvduo_merge_module


@dataclass
class KVDuoMergeWorkspace:
    tokens: torch.Tensor
    locations: torch.Tensor
    lru: torch.Tensor
    sources: torch.Tensor
    destinations: torch.Tensor
    scratch: torch.Tensor
    summary: torch.Tensor
    summary_host: torch.Tensor
    page_ids: torch.Tensor
    page_ids_host: torch.Tensor
    hash_size: int
    completion: torch.cuda.Event
    free_pending: bool = False


def allocate_kvduo_merge_workspace(tokens, lru) -> KVDuoMergeWorkspace:
    maximum = lru.numel()
    hash_size = 1 << (2 * maximum - 1).bit_length()
    pinned = tokens.device.type == "cuda"
    return KVDuoMergeWorkspace(
        tokens=torch.empty_like(tokens),
        locations=torch.empty_like(tokens),
        lru=torch.empty_like(lru),
        sources=torch.empty(maximum, dtype=torch.int64, device=tokens.device),
        destinations=torch.empty(maximum, dtype=torch.int64, device=tokens.device),
        scratch=torch.empty(
            4 * maximum + hash_size, dtype=torch.int32, device=tokens.device
        ),
        summary=torch.empty(3, dtype=torch.int32, device=tokens.device),
        summary_host=torch.empty(3, dtype=torch.int32, device="cpu", pin_memory=pinned),
        page_ids=torch.empty(8, dtype=torch.int64, device=tokens.device),
        page_ids_host=torch.empty(
            8, dtype=torch.int64, device="cpu", pin_memory=pinned
        ),
        hash_size=hash_size,
        completion=torch.cuda.Event(),
    )


def execute_kvduo_hot_merge(
    tokens,
    locations,
    lru,
    backing,
    workspace,
    *,
    capacity,
    k,
    top_k,
    target_pages,
    paged,
    published_capacity,
    capacity_ema,
    capacity_steps,
    growth_events,
    base_capacity,
    prefill_ema,
    prefill_steps,
):
    """Queue planning, disjoint payload moves, publication, and compact feedback.

    The caller waits on completion only when it needs ownership and free-page
    information. Failed validation skips both payload and metadata mutation.
    Workspace reuse must follow the previous merge's completion.
    """
    maximum = lru.numel()
    slot_dtype = torch.int32 if maximum > torch.iinfo(torch.int16).max else torch.int16
    if (
        not tokens.is_cuda
        or tokens.ndim != 1
        or locations.shape != tokens.shape
        or lru.ndim != 1
        or tokens.numel() < maximum
        or tokens.dtype != torch.int32
        or locations.dtype != torch.int32
        or lru.dtype != slot_dtype
        or k <= 0
        or capacity <= 0
        or capacity % k
        or top_k <= 0
        or not (top_k + k - 1) // k <= target_pages <= capacity // k <= 8
        or capacity > maximum
        or backing.ndim < 2
        or backing.size(0) == 0
    ):
        raise ValueError("Invalid KVDuo merge geometry or metadata")
    banks = (
        (published_capacity, torch.int32, 1),
        (capacity_ema, torch.float32, 9),
        (capacity_steps, torch.int64, 1),
        (growth_events, torch.int32, 1),
        (base_capacity, torch.int32, 1),
        (prefill_ema, torch.float32, 2),
        (prefill_steps, torch.int64, 1),
    )
    for bank, dtype, count in banks:
        if bank.dtype != dtype or bank.numel() != count:
            raise ValueError("Invalid KVDuo merge publication metadata")
    for bank in (tokens, locations, lru, backing, *(bank for bank, _, _ in banks)):
        if bank.device != tokens.device or not bank.is_contiguous():
            raise ValueError("KVDuo merge banks must be contiguous on one device")
    if (
        workspace.tokens.shape != tokens.shape
        or workspace.lru.shape != lru.shape
        or workspace.tokens.device != tokens.device
        or workspace.lru.dtype != lru.dtype
    ):
        raise ValueError("KVDuo merge workspace does not match the fixed metadata")
    module = _jit_kvduo_merge_module(maximum, 256)
    execute = module.execute_c4 if paged else module.execute_linear
    stream = torch.cuda.current_stream(tokens.device)
    if workspace.free_pending:
        # Include the preceding page-ID upload before reusing pinned staging,
        # even if this merge was dispatched on a different control stream.
        stream.wait_event(workspace.completion)
    execute(
        tokens,
        locations,
        lru,
        workspace.tokens,
        workspace.locations,
        workspace.lru,
        workspace.sources,
        workspace.destinations,
        workspace.scratch,
        workspace.summary,
        capacity,
        k,
        target_pages,
        workspace.hash_size,
        backing.data_ptr(),
        backing.size(0),
        backing.stride(0) * backing.element_size(),
        *(bank.data_ptr() for bank, _, _ in banks),
    )
    workspace.summary_host.copy_(workspace.summary, non_blocking=True)
    workspace.completion.record(stream)
    workspace.free_pending = False


def finish_kvduo_hot_merge(workspace, pages, target_pages, k):
    workspace.completion.synchronize()
    status, moved, mask = workspace.summary_host.tolist()
    if status & 1:
        raise ValueError("HOT merge requires the current LRU slot permutation")
    if status & 2:
        raise RuntimeError("KVDuo HOT merge has invalid physical slot locations")
    kept = tuple(page for page in range(pages) if mask & (1 << page))
    if (
        status
        or mask >> pages
        or len(kept) != target_pages
        or not 0 <= moved <= target_pages * k
    ):
        raise RuntimeError("KVDuo GPU merge execution returned invalid results")
    return kept, tuple(page for page in range(pages) if page not in kept)
