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
    layer: int
    layer_stride: int
    page_size: int

    def native_metadata(self, scores, raw_indices):
        """Validate FULL banks and transport their addresses as FFI scalars."""
        if self.lookup.ndim != 2 or self.mapping.ndim != 1:
            raise ValueError("KVDuo FULL lookup/mapping require matrix/vector layouts")
        requests, context = self.lookup.shape
        logical = self.mapping.numel()
        if not requests or not context or not logical:
            raise ValueError("KVDuo FULL metadata cannot have empty address spaces")
        if (
            self.layer < 0
            or self.layer_stride < 0
            or self.page_size <= 0
            or self.page_size & (self.page_size - 1)
        ):
            raise ValueError("KVDuo FULL metadata has invalid layer/page geometry")
        fields = (
            ("output", torch.int32, raw_indices.shape),
            ("requests", torch.int64, (scores.shape[0],)),
            ("lookup", torch.int64, (requests, context)),
            ("mapping", torch.int64, (logical,)),
            ("touches", torch.int64, (logical,)),
            ("status", torch.int32, (requests,)),
            ("stats", torch.int32, (requests, 2)),
            ("growth", torch.int32, (requests,)),
            ("observe", torch.int32, (1,)),
            ("real_requests", torch.int32, (1,)),
            ("clock_base", torch.int64, (1,)),
        )
        pointers = []
        for name, dtype, shape in fields:
            tensor = getattr(self, name)
            try:
                valid = (
                    tensor.shape == shape
                    and tensor.dtype == dtype
                    and tensor.device == scores.device
                    and (
                        tensor.stride(1) == 1 and tensor.stride(0) >= context
                        if name == "lookup"
                        else tensor.is_contiguous()
                    )
                )
                pointer = tensor.data_ptr() if valid else 0
            except RuntimeError as error:
                raise ValueError(f"KVDuo FULL metadata {name} is invalid") from error
            if not pointer:
                raise ValueError(f"KVDuo FULL metadata {name} has incompatible layout")
            pointers.append(pointer)
        return (
            *pointers,
            requests,
            context,
            self.lookup.stride(0),
            logical,
            self.layer,
            self.layer_stride,
            self.page_size,
        )


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
