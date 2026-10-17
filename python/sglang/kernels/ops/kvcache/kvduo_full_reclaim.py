from __future__ import annotations

import functools
from collections import Counter
from collections.abc import Sequence

import torch

from sglang.kernels.jit.utils import load_jit


@functools.cache
def _jit_full_reclaim_module():
    return load_jit(
        "kvduo_full_reclaim",
        cuda_files=["kvduo_full_reclaim.cuh"],
        cuda_wrappers=[
            ("summarize_i32", "kvduo_full_page_summary<128, int32_t>"),
            ("summarize_i64", "kvduo_full_page_summary<128, int64_t>"),
            ("select_requests", "kvduo_full_request_select<128>"),
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
    if (
        logical_pages.dtype not in (torch.int32, torch.int64)
        or not logical_pages.is_contiguous()
        or mapping.numel() == 0
        or any(
            bank.device != logical_pages.device
            or bank.dtype != torch.int64
            or not bank.is_contiguous()
            for bank in banks
        )
    ):
        raise ValueError(
            "FULL tables must be contiguous int64 with int32/int64 page indices on one device"
        )
    protected_pages = None
    if protected is not None:
        if protected.device != logical_pages.device or protected.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("FULL protection must use the logical address space")
        protected_pages = torch.isin(logical_pages, protected).any(dim=1).int()
    physical = torch.empty(
        logical_pages.shape, dtype=torch.int64, device=mapping.device
    )
    scores = torch.empty(
        logical_pages.size(0), dtype=torch.int64, device=mapping.device
    )
    valid = torch.empty_like(scores, dtype=torch.int32)
    if logical_pages.is_cuda:
        module = _jit_full_reclaim_module()
        summarize = (
            module.summarize_i32
            if logical_pages.dtype == torch.int32
            else module.summarize_i64
        )
        summarize(
            logical_pages,
            mapping.data_ptr(),
            data_versions.data_ptr(),
            host_versions.data_ptr(),
            touches.data_ptr(),
            protected_pages.data_ptr() if protected_pages is not None else 0,
            physical.data_ptr(),
            scores.data_ptr(),
            valid.data_ptr(),
            mapping.numel(),
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


def select_kvduo_full_requests(
    valid: torch.Tensor,
    owner_indices: Sequence[int],
    minimum_costs: Sequence[int],
    admission_order: Sequence[int],
    needed_pages: int,
    *,
    release_all: bool = False,
    mixed_owners: Sequence[bool] | None = None,
) -> list[int]:
    """Prefer largest net gain, then mixed owners at equal gain.

    Equal gain and residency prefer earlier decode admission, then owner index.
    Page validity and one-time HOT minimum costs match the page policy. GPU
    planning never sorts page ages or downloads the full validity vector.
    """
    if mixed_owners is None:
        mixed_owners = [False] * len(minimum_costs)
    n = valid.numel()
    if (
        len(mixed_owners) != len(minimum_costs)
        or valid.ndim != 1
        or len(owner_indices) != n
        or len(admission_order) != len(minimum_costs)
        or any(owner < 0 or owner >= len(minimum_costs) for owner in owner_indices)
        or any(cost < 0 for cost in minimum_costs)
        or any(order < 0 for order in admission_order)
    ):
        raise ValueError("Invalid FULL request selection geometry")
    if n == 0 or needed_pages <= 0:
        return []
    if valid.is_cuda:
        device = valid.device
        owners = torch.tensor(owner_indices, dtype=torch.int64, device=device)
        costs = torch.tensor(minimum_costs, dtype=torch.int64, device=device)
        order = torch.tensor(admission_order, dtype=torch.int64, device=device)
        mixed = torch.tensor(mixed_owners, dtype=torch.int32, device=device)
        largest_owner = max(Counter(owner_indices).values())
        bound = (
            n
            if release_all
            else min(n, needed_pages - 1 + sum(minimum_costs) + largest_owner)
        )
        workspace = torch.empty(
            2 * len(minimum_costs), dtype=torch.int32, device=device
        )
        packet = torch.empty(bound + 1, dtype=torch.int64, device=device)
        _jit_full_reclaim_module().select_requests(
            valid.int(),
            owners,
            costs,
            order,
            mixed,
            workspace,
            packet,
            needed_pages,
            release_all,
        )
        result = packet.cpu().tolist()
        if not 0 <= result[0] <= bound:
            raise RuntimeError("FULL request plan exceeded its selection bound")
        return result[1 : 1 + result[0]]

    validity = valid.bool().tolist()
    counts = Counter(owner for owner, ok in zip(owner_indices, validity) if ok)
    eligible = sorted(
        (owner for owner, count in counts.items() if count > minimum_costs[owner]),
        key=lambda owner: (
            -(counts[owner] - minimum_costs[owner]),
            not mixed_owners[owner],
            admission_order[owner],
            owner,
        ),
    )
    chosen, net = set(), 0
    for owner in eligible:
        if not release_all and net >= needed_pages:
            break
        chosen.add(owner)
        net += counts[owner] - minimum_costs[owner]
    return [
        i
        for i, (owner, ok) in enumerate(zip(owner_indices, validity))
        if ok and owner in chosen
    ]
