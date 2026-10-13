from __future__ import annotations

import functools

import torch

from sglang.kernels.jit.utils import load_jit


@functools.cache
def _jit_control_module():
    return load_jit(
        "kvduo_control",
        cuda_files=["kvduo_control.cuh"],
        cuda_wrappers=[("feedback", "kvduo_pack_feedback<256>")],
    )


def pack_kvduo_feedback(events, stats, positions, packet, active_requests):
    layers, requests = events.shape
    if (
        events.dtype != torch.int32
        or stats.shape != (layers, requests, 2)
        or stats.dtype != torch.int32
        or positions.shape != (requests,)
        or positions.dtype != torch.int32
        or packet.ndim != 1
        or packet.numel() < layers * active_requests
        or packet.dtype != torch.int32
        or not 0 <= active_requests <= requests
        or any(
            t.device != events.device or not t.is_contiguous()
            for t in (stats, positions, packet, events)
        )
    ):
        raise ValueError("Invalid KVDuo feedback workspace")
    if events.is_cuda:
        _jit_control_module().feedback(
            events, stats, positions, packet, active_requests
        )
    else:
        output = packet[: layers * active_requests].view(layers, active_requests)
        selected = positions >= 0
        output[:, positions[selected].long()] = events[:, selected]
        events.zero_()
        stats.zero_()
