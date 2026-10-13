from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

from sglang.kernels.jit.utils import load_jit


@functools.cache
def _jit_hand_module():
    return load_jit(
        "kvduo_hand",
        cuda_files=["kvduo_hand_plan.cuh"],
        cuda_wrappers=[
            ("targets", "kvduo_hand_targets<256>"),
            ("packet", "kvduo_hand_packet<256>"),
        ],
    )


@dataclass
class KVDuoHandWorkspace:
    pages_host: torch.Tensor
    pages: torch.Tensor
    targets: torch.Tensor
    packet: torch.Tensor
    header: torch.Tensor


def allocate_kvduo_hand_workspace(counts):
    shape = counts.shape[:2]
    objects = shape[0] * shape[1]
    return KVDuoHandWorkspace(
        torch.empty(shape, dtype=torch.int32, device="cpu", pin_memory=counts.is_cuda),
        torch.empty(shape, dtype=torch.int32, device=counts.device),
        torch.empty(objects, dtype=torch.int32, device=counts.device),
        torch.empty(2 + 4 * objects, dtype=torch.int64, device=counts.device),
        torch.empty(2, dtype=torch.int64, device="cpu", pin_memory=counts.is_cuda),
    )


@dataclass
class KVDuoHandSnapshot:
    workspace: KVDuoHandWorkspace
    hand: int

    def next(self, cursor, needed):
        w = self.workspace
        layers, requests = w.pages.shape
        objects = layers * requests
        if not 0 <= cursor <= objects or needed <= 0:
            raise ValueError("Invalid Hand continuation")
        if w.pages.is_cuda:
            _jit_hand_module().packet(
                w.pages, w.targets, w.packet, self.hand, cursor, needed
            )
            w.header.copy_(w.packet[:2])
            count, status = w.header.tolist()
            if status or not 0 <= count <= objects:
                raise ValueError("Invalid HOT capacity geometry in Hand snapshot")
            return [
                tuple(row)
                for row in w.packet[2 : 2 + 4 * count].view(count, 4).cpu().tolist()
            ]
        result = []
        released = 0
        for offset in range(cursor, objects):
            position = (self.hand + offset) % objects
            req, layer = divmod(position, layers)
            target = int(w.targets[offset])
            if target < 0:
                continue
            pages = int(w.pages[layer, req])
            result.append((offset, layer * requests + req, pages, target))
            released += pages - target
            if released >= needed:
                break
        return result


def prepare_kvduo_hand(counts, steps, workspace, *, hand, minimum, rho):
    from sglang.srt.mem_cache.sparsity.core.kvduo_hot_merge import kvduo_shrink_target

    w = workspace
    if counts.ndim != 3 or counts.shape[2] != 9:
        raise ValueError("Invalid Hand observation geometry")
    shape = counts.shape[:2]
    objects = shape[0] * shape[1]
    if (
        counts.ndim != 3
        or counts.shape[2] != 9
        or counts.dtype != torch.float32
        or steps.shape != shape
        or steps.dtype != torch.int64
        or w.pages.shape != shape
        or w.pages.dtype != torch.int32
        or w.targets.shape != (objects,)
        or w.targets.dtype != torch.int32
        or w.packet.numel() < 2 + 4 * objects
        or w.packet.dtype != torch.int64
        or w.pages_host.shape != shape
        or w.pages_host.dtype != torch.int32
        or w.header.shape != (2,)
        or w.header.dtype != torch.int64
        or any(
            t.device.type != "cpu" or not t.is_contiguous()
            for t in (w.pages_host, w.header)
        )
        or not 1 <= minimum <= 8
        or not objects
        or any(
            t.device != counts.device or not t.is_contiguous()
            for t in (counts, steps, w.pages, w.targets, w.packet)
        )
    ):
        raise ValueError("Invalid Hand planning workspace")
    hand = hand % objects
    w.pages.copy_(w.pages_host, non_blocking=True)
    if counts.is_cuda:
        _jit_hand_module().targets(
            counts, steps, w.pages, w.targets, hand, minimum, rho
        )
    else:
        layers, requests = shape
        for offset in range(objects):
            req, layer = divmod((hand + offset) % objects, layers)
            pages = int(w.pages[layer, req])
            w.targets[offset] = (
                kvduo_shrink_target(
                    counts[layer, req, :8].tolist(),
                    float(counts[layer, req, 8]),
                    int(steps[layer, req]),
                    pages,
                    minimum,
                    rho,
                )
                if pages
                else -1
            )
    return KVDuoHandSnapshot(w, hand)
