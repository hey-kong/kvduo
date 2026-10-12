from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FixedShapeTensor(torch.Tensor):
    def __getitem__(self, key):
        if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
            raise AssertionError("GPU boolean indexing requires a host wait")
        return super().__getitem__(key)


def make_allocator(lengths, *, enabled=True, free_slots=512, allowed=True):
    allocator = DeepSeekV4HiSparseTokenToKVPoolAllocator.__new__(
        DeepSeekV4HiSparseTokenToKVPoolAllocator
    )
    allocator.compress_ratio = 4
    allocator.hisparse_page_size = 64
    allocator.enable_kvduo_full_decode = enabled
    allocator.kvduo_physical_allocation_guard = MagicMock(return_value=allowed)
    logical = (torch.arange(len(lengths)) * 4 + 7).as_subclass(FixedShapeTensor)
    allocator.logical_attn_allocator = SimpleNamespace(
        alloc_decode=MagicMock(return_value=logical)
    )
    active = lengths % 4 == 0
    physical = torch.arange(active.sum()) + 100
    allocator.hisparse_attn_allocator = SimpleNamespace(
        available_size=lambda: free_slots,
        alloc_decode=MagicMock(return_value=physical),
    )
    pool = HiSparseC4DevicePool.__new__(HiSparseC4DevicePool)
    pool.compress_ratio = 4
    pool.full_to_hisparse_device_index_mapping = torch.arange(32) + 1000
    allocator.hisparse_kvcache = pool
    allocator.full_to_hisparse_device_index_mapping = (
        pool.full_to_hisparse_device_index_mapping
    )
    return allocator, logical, physical


@pytest.mark.parametrize(
    "lengths", [[255, 256, 257, 260, 261], [256, 260, 264], [255, 257, 259], []]
)
def test_decode_allocation_uses_fixed_shape_indices_and_preserves_request_order(
    lengths,
):
    cpu_lengths = torch.tensor(lengths, dtype=torch.int64)
    allocator, logical, physical = make_allocator(cpu_lengths)
    gpu_lengths = cpu_lengths.clone().as_subclass(FixedShapeTensor)
    last = logical - 1
    original_mapping = allocator.full_to_hisparse_device_index_mapping.clone()
    result = allocator.alloc_decode(gpu_lengths, cpu_lengths, last)
    assert result is logical
    allocator.logical_attn_allocator.alloc_decode.assert_called_once_with(
        gpu_lengths, cpu_lengths, last
    )
    active = cpu_lengths % 4 == 0
    active_indices = torch.nonzero(active, as_tuple=True)[0]
    if active_indices.numel():
        args = allocator.hisparse_attn_allocator.alloc_decode.call_args.args
        expected_lengths = cpu_lengths[active] // 4
        assert torch.equal(args[0], expected_lengths)
        assert torch.equal(args[1], expected_lengths)
        expected_last = original_mapping[
            (last.index_select(0, active_indices) - 3) // 4
        ]
        assert torch.equal(args[2], expected_last)
        original_mapping[logical.index_select(0, active_indices) // 4] = physical
    else:
        allocator.hisparse_attn_allocator.alloc_decode.assert_not_called()
    assert torch.equal(
        allocator.full_to_hisparse_device_index_mapping, original_mapping
    )


@pytest.mark.parametrize("free_slots,allowed", [(0, True), (64, False)])
def test_decode_allocation_keeps_physical_capacity_and_reservation_checks(
    free_slots, allowed
):
    lengths = torch.tensor([256, 260])
    allocator, logical, _ = make_allocator(
        lengths, free_slots=free_slots, allowed=allowed
    )
    before = allocator.full_to_hisparse_device_index_mapping.clone()
    assert allocator.alloc_decode(lengths, lengths, logical - 1) is None
    allocator.kvduo_physical_allocation_guard.assert_called_once_with(64)
    allocator.logical_attn_allocator.alloc_decode.assert_not_called()
    allocator.hisparse_attn_allocator.alloc_decode.assert_not_called()
    assert torch.equal(allocator.full_to_hisparse_device_index_mapping, before)


def test_decode_logical_allocation_failure_does_not_allocate_physical_pages():
    lengths = torch.tensor([256, 260])
    allocator, logical, _ = make_allocator(lengths)
    allocator.logical_attn_allocator.alloc_decode.return_value = None
    before = allocator.full_to_hisparse_device_index_mapping.clone()
    assert allocator.alloc_decode(lengths, lengths, logical - 1) is None
    allocator.hisparse_attn_allocator.alloc_decode.assert_not_called()
    assert torch.equal(allocator.full_to_hisparse_device_index_mapping, before)


def test_standalone_hisparse_decode_keeps_logical_allocation_only():
    lengths = torch.tensor([256, 260])
    allocator, logical, _ = make_allocator(lengths, enabled=False)
    assert allocator.alloc_decode(lengths, lengths, logical - 1) is logical
    allocator.kvduo_physical_allocation_guard.assert_not_called()
    allocator.hisparse_attn_allocator.alloc_decode.assert_not_called()


def test_decode_index_copy_uses_pinned_memory_and_nonblocking_transfer(monkeypatch):
    import sglang.srt.mem_cache.allocator.hisparse as module

    lengths = torch.tensor([256, 257, 260])
    allocator, logical, _ = make_allocator(lengths)
    monkeypatch.setattr(module, "is_pin_memory_available", lambda _: True)
    pinned = []
    copied = []
    original_to = torch.Tensor.to

    def pin(tensor):
        pinned.append(tensor.clone())
        return tensor

    def copy(tensor, *args, **kwargs):
        copied.append(kwargs)
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    monkeypatch.setattr(torch.Tensor, "to", copy)
    allocator.alloc_decode(lengths, lengths, logical - 1)
    assert len(pinned) == 1 and torch.equal(pinned[0], torch.tensor([0, 2]))
    assert {"device": lengths.device, "non_blocking": True} in copied
