from __future__ import annotations

import functools

import torch

from sglang.kernels.jit.utils import load_jit


@functools.cache
def _jit_full_write_module():
    return load_jit(
        "kvduo_full_write",
        cuda_files=["kvduo_full_write.cuh"],
        cuda_wrappers=[("publish", "kvduo_publish_full_write<256>")],
    )


def publish_kvduo_full_write(
    logical_locs: torch.Tensor,
    generation: torch.Tensor,
    data_versions: torch.Tensor,
    host_versions: torch.Tensor,
    touches: torch.Tensor,
    clock: torch.Tensor,
    *,
    lookup: torch.Tensor | None = None,
    req_indices: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> None:
    """Publish distinct newly allocated FULL locations and their request mapping."""
    banks = (generation, data_versions, host_versions, touches)
    if logical_locs.ndim != 1 or logical_locs.dtype not in (torch.int32, torch.int64):
        raise ValueError("FULL locations must be an int32/int64 vector")
    if any(
        bank.ndim != 1
        or bank.shape != generation.shape
        or bank.dtype != torch.int64
        or not bank.is_contiguous()
        or bank.device != logical_locs.device
        for bank in banks
    ):
        raise ValueError("FULL metadata must share one contiguous int64 address space")
    if (
        clock.numel() != 1
        or clock.dtype != torch.int64
        or clock.device != logical_locs.device
    ):
        raise ValueError("FULL clock must be an int64 scalar on the metadata device")
    mapped = lookup is not None
    if mapped != (req_indices is not None) or mapped != (positions is not None):
        raise ValueError(
            "FULL lookup, request indices and positions are required together"
        )
    if mapped:
        if (
            lookup.ndim != 2
            or lookup.dtype != torch.int64
            or lookup.stride(1) != 1
            or lookup.device != logical_locs.device
            or any(
                index.shape != logical_locs.shape
                or index.dtype not in (torch.int32, torch.int64)
                or index.device != logical_locs.device
                for index in (req_indices, positions)
            )
        ):
            raise ValueError(
                "FULL lookup requires matching request and position vectors"
            )
    if logical_locs.is_cuda:
        # Fixed tensor addresses and scalar FFI arguments are capture safe.
        _jit_full_write_module().publish(
            logical_locs.data_ptr(),
            generation.data_ptr(),
            data_versions.data_ptr(),
            host_versions.data_ptr(),
            touches.data_ptr(),
            clock.data_ptr(),
            lookup.data_ptr() if mapped else 0,
            req_indices.data_ptr() if mapped else 0,
            positions.data_ptr() if mapped else 0,
            torch.cuda.current_stream(logical_locs.device).cuda_stream,
            logical_locs.numel(),
            generation.numel(),
            logical_locs.stride(0),
            lookup.stride(0) if mapped else 0,
            lookup.size(0) if mapped else 0,
            lookup.size(1) if mapped else 0,
            req_indices.stride(0) if mapped else 0,
            positions.stride(0) if mapped else 0,
            logical_locs.dtype == torch.int64,
            req_indices.dtype == torch.int64 if mapped else True,
            positions.dtype == torch.int64 if mapped else True,
        )
    else:
        clock.add_(1)
        generation[logical_locs] += 1
        data_versions[logical_locs] = 1
        host_versions[logical_locs] = -1
        touches[logical_locs] = clock.reshape(-1)[0]
        if mapped:
            lookup[req_indices, positions] = logical_locs.to(torch.int64)
