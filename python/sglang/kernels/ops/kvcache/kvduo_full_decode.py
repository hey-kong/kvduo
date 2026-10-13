from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

from sglang.kernels.jit.utils import load_jit


@dataclass(frozen=True)
class KVDuoFullTopK:
    output: torch.Tensor
    requests: torch.Tensor
    lookup: torch.Tensor
    mapping: torch.Tensor
    touches: torch.Tensor
    status: torch.Tensor
    stats: torch.Tensor
    growth: torch.Tensor
    observe: torch.Tensor
    real_requests: torch.Tensor
    clock_base: torch.Tensor
    prefill_ema: torch.Tensor
    prefill_steps: torch.Tensor
    layer: int
    layer_stride: int
    page_size: int


@functools.cache
def _jit_full_decode_module():
    return load_jit(
        "kvduo_full_decode",
        cuda_files=["kvduo_full_decode.cuh"],
        cuda_wrappers=[("begin", "kvduo_begin_decode")],
    )


def begin_kvduo_decode(
    clock: torch.Tensor,
    clock_base: torch.Tensor,
    observe: torch.Tensor,
    real_requests: torch.Tensor,
    *,
    batch_size: int,
    full_layers: int = 0,
) -> None:
    """Publish replay controls and reserve deterministic per-layer FULL stamps."""
    if batch_size < 0 or full_layers < 0:
        raise ValueError("Decode controls must be nonnegative")
    for tensor, dtype in (
        (clock, torch.int64),
        (clock_base, torch.int64),
        (observe, torch.int32),
        (real_requests, torch.int32),
    ):
        if (
            tensor.shape != (1,)
            or tensor.dtype != dtype
            or tensor.device != clock.device
        ):
            raise ValueError("Decode controls require same-device typed scalars")
    if clock.is_cuda:
        _jit_full_decode_module().begin(
            clock, clock_base, observe, real_requests, batch_size, full_layers
        )
    else:
        clock_base.copy_(clock)
        clock.add_(full_layers)
        observe.fill_(1)
        real_requests.fill_(batch_size)
