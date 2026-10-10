from __future__ import annotations

import functools
from collections.abc import Sequence

import torch

from sglang.kernels.jit.utils import load_jit


@functools.cache
def _jit_full_reclaim_module():
    return load_jit(
        "kvduo_full_reclaim",
        cuda_files=["kvduo_full_reclaim.cuh"],
        cuda_wrappers=[
            ("summarize", "kvduo_full_page_summary<128>"),
            ("select", "kvduo_full_page_select<128>"),
        ],
    )


def summarize_kvduo_full_pages(
    logical_pages: torch.Tensor,
    mapping: torch.Tensor,
    data_versions: torch.Tensor,
    host_versions: torch.Tensor,
    touches: torch.Tensor,
    protected: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse FULL address, version, and page-age gathering without mutating inputs."""
    if logical_pages.ndim != 2 or logical_pages.size(1) <= 0:
        raise ValueError("FULL candidates must contain whole logical pages")
    banks = (mapping, data_versions, host_versions, touches)
    if any(bank.ndim != 1 or bank.shape != mapping.shape for bank in banks):
        raise ValueError("FULL metadata banks must share the logical address space")
    if mapping.numel() == 0 or any(
        bank.device != logical_pages.device
        or bank.dtype != torch.int64
        or not bank.is_contiguous()
        for bank in (logical_pages, *banks)
    ):
        raise ValueError("FULL metadata must be contiguous int64 on one device")
    protected_pages = None
    if protected is not None:
        if protected.device != logical_pages.device or protected.dtype != torch.int64:
            raise ValueError("FULL protection must use the logical address space")
        protected_pages = torch.isin(logical_pages, protected).any(dim=1).int()
    physical = torch.empty_like(logical_pages)
    scores = torch.empty(
        logical_pages.size(0), dtype=torch.int64, device=mapping.device
    )
    valid = torch.empty_like(scores, dtype=torch.int32)
    if logical_pages.is_cuda:
        _jit_full_reclaim_module().summarize(
            logical_pages,
            mapping,
            data_versions,
            host_versions,
            touches,
            protected_pages,
            physical,
            scores,
            valid,
        )
    else:
        in_range = (logical_pages >= 0) & (logical_pages < mapping.numel())
        safe = logical_pages.clamp(0, mapping.numel() - 1)
        physical.copy_(torch.where(in_range, mapping[safe], 0))
        valid.copy_(
            (
                in_range & (physical > 0) & (host_versions[safe] == data_versions[safe])
            ).all(dim=1)
        )
        scores.copy_(
            touches[safe]
            .masked_fill(~in_range, torch.iinfo(torch.int64).min)
            .amax(dim=1)
        )
        if protected_pages is not None:
            valid.bitwise_and_(1 - protected_pages)
    return physical, scores, valid


def select_kvduo_full_pages(
    scores: torch.Tensor,
    valid: torch.Tensor,
    owner_indices: Sequence[int],
    minimum_costs: Sequence[int],
    needed_pages: int,
    *,
    release_all: bool = False,
) -> list[int]:
    """Choose the coldest prefix, charging each new mixed owner's HOT minimum once.

    Input order breaks age ties by request slot/page offset. Sorting, owner
    eligibility, and net-space prefix selection stay on the tensor device.
    Only a bounded selection packet crosses back to the CPU control plane.
    """
    n = scores.numel()
    if scores.ndim != 1 or valid.shape != scores.shape or valid.device != scores.device:
        raise ValueError("FULL scores and validity must share one page vector")
    if (
        len(owner_indices) != n
        or any(owner < 0 or owner >= len(minimum_costs) for owner in owner_indices)
        or any(cost < 0 for cost in minimum_costs)
    ):
        raise ValueError("Invalid FULL owner indices or HOT minimum costs")
    if n == 0 or needed_pages <= 0:
        return []
    device = scores.device
    owners = torch.tensor(owner_indices, dtype=torch.int64, device=device)
    costs = torch.tensor(minimum_costs, dtype=torch.int64, device=device)
    bound = n if release_all else min(n, needed_pages + sum(minimum_costs))
    if scores.is_cuda:
        order = torch.argsort(scores, stable=True)
        workspace = torch.empty(
            2 * len(minimum_costs), dtype=torch.int32, device=device
        )
        packet = torch.empty(bound + 1, dtype=torch.int64, device=device)
        _jit_full_reclaim_module().select(
            order,
            valid.int(),
            owners,
            costs,
            workspace,
            packet,
            needed_pages,
            release_all,
            any(minimum_costs),
        )
        result = packet.cpu().tolist()
        if not 0 <= result[0] <= bound:
            raise RuntimeError("FULL reclaim plan exceeded its selection bound")
        return result[1 : 1 + result[0]]
    valid = valid.bool()
    counts = torch.zeros_like(costs).scatter_add_(0, owners, valid.long())
    eligible = valid & (counts[owners] > costs[owners])
    order = torch.argsort(scores, stable=True)

    # Stable partition after sorting also handles valid INT64_MAX touch clocks.
    ranks = torch.arange(n, device=device)
    ordered_eligible = eligible[order]
    prefix = ordered_eligible.long().cumsum(0)
    eligible_count = prefix[-1]
    destinations = torch.where(
        ordered_eligible, prefix - 1, eligible_count + ranks - prefix
    )
    compact_order = torch.empty_like(order).scatter_(0, destinations, order)
    compact_owners = owners[compact_order]
    active = ranks < eligible_count
    first = torch.full_like(costs, n).scatter_reduce_(
        0, compact_owners, torch.where(active, ranks, n), reduce="amin"
    )
    charge = torch.where(
        active & (ranks == first[compact_owners]), costs[compact_owners], 0
    )
    net_pages = (active.long() - charge).cumsum(0)
    count = eligible_count
    if not release_all:
        boundary = torch.where(
            active & (net_pages >= needed_pages), ranks + 1, n
        ).amin()
        count = torch.minimum(count, boundary)

    # Before reaching the target, at most all owners' mandatory minima have
    # been charged. This bounds the D2H packet without reading a CUDA scalar.
    packet = torch.empty(bound + 1, dtype=torch.int64, device=device)
    packet[0] = count
    packet[1:] = compact_order[:bound]
    result = packet.cpu().tolist()
    return result[1 : 1 + result[0]]
