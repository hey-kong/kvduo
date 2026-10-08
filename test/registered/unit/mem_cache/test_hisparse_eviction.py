"""HiSparse eviction must fund both full-attention and sliding-window pools."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.mem_cache.allocation import alloc_paged_token_slots_extend
from sglang.srt.mem_cache.allocator import paged
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictResult
from sglang.srt.mem_cache.common import evict_from_tree_cache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_allocator(*, full=121600, swa=2560, physical=30400):
    # Real page inventories and production budget/allocation methods. Only GPU
    # KV storage construction is omitted; slot metadata lives on CPU.
    def pool(size, page_size, available):
        allocator = paged.PagedTokenToKVPoolAllocator(
            size, page_size, torch.int64, "cpu", None, False
        )
        held = allocator.alloc(size - available)
        return allocator, held

    logical = object.__new__(SWATokenToKVPoolAllocator)
    logical.page_size = 256
    logical.full_attn_allocator, full_held = pool(121600, 256, full)
    logical.swa_attn_allocator, swa_held = pool(7680, 256, swa)
    logical.full_to_swa_index_mapping = torch.zeros(121600 + 257, dtype=torch.int64)
    logical.full_to_swa_index_mapping[-1] = -1
    logical._kvcache = SimpleNamespace(
        full_to_swa_index_mapping=logical.full_to_swa_index_mapping,
        translate_loc_from_full_to_swa=lambda indices: logical.full_to_swa_index_mapping[
            indices
        ],
    )

    allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
    allocator.page_size = 256
    allocator.hisparse_page_size = 64
    allocator.compress_ratio = 4
    allocator.logical_attn_allocator = logical
    allocator.hisparse_attn_allocator, _ = pool(30400, 64, physical)
    allocator.full_to_hisparse_device_index_mapping = torch.zeros(
        30400 + 65, dtype=torch.int64
    )
    allocator.full_to_hisparse_device_index_mapping[-1] = -1
    allocator.hisparse_kvcache = SimpleNamespace(
        translate_loc_from_full_to_compressed=lambda indices: indices[indices % 4 == 3]
        // 4,
        _translate_loc_to_hisparse_device=lambda indices: allocator.full_to_hisparse_device_index_mapping[
            indices
        ],
    )
    return allocator, full_held, swa_held


def make_cache(allocator, full_held, swa_held, *, reclaim=True):
    def evict(params):
        full_count = min(
            len(full_held),
            -(-params.num_tokens // allocator.page_size) * allocator.page_size,
        )
        swa_count = min(
            len(swa_held),
            -(-params.swa_num_tokens // allocator.page_size) * allocator.page_size,
        )
        if not reclaim:
            return EvictResult()
        allocator.full_attn_allocator.free(full_held[:full_count])
        allocator.swa_attn_allocator.free(swa_held[:swa_count])
        return EvictResult(
            num_tokens_evicted=full_count, swa_num_tokens_evicted=swa_count
        )

    return SimpleNamespace(
        token_to_kv_pool_allocator=allocator,
        is_chunk_cache=lambda: False,
        evict=MagicMock(side_effect=evict),
        pretty_print=MagicMock(),
        available_and_evictable_str=lambda: (
            f"Available swa: {allocator.swa_available_size() + len(swa_held)} "
            f"(available_size={allocator.swa_available_size()} + "
            f"component_evictable_size_={len(swa_held)})"
        ),
    )


@pytest.fixture
def cpu_extend_kernel(monkeypatch):
    # Replace only the Triton slot-writing kernel with its existing CPU
    # reference. Wrapper/SWA checks and page consumption remain production code.
    kernel = MagicMock()

    def launch(prefix, seq, last, free_pages, out, block_size, page_size):
        paged.alloc_extend_naive(prefix, seq, last, free_pages, out, page_size, "cpu")

    kernel.__getitem__.return_value = launch
    monkeypatch.setattr(paged, "alloc_extend_kernel", kernel)


@pytest.mark.parametrize("lengths", [[7424], [3841, 3583]])
def test_prefill_materializes_swa_evictable_capacity_before_real_allocation(
    lengths, cpu_extend_kernel
):
    allocator, full_held, swa_held = make_allocator()
    cache = make_cache(allocator, full_held, swa_held)
    assert allocator.full_available_size() == 121600
    assert allocator.available_size() == allocator.swa_available_size() == 2560
    assert len(swa_held) == 5120
    prefix = torch.zeros(len(lengths), dtype=torch.int64)
    seq = torch.tensor(lengths, dtype=torch.int64)
    last = torch.full_like(prefix, -1)

    result = alloc_paged_token_slots_extend(
        cache, prefix, prefix, seq, seq, last, sum(lengths)
    )

    assert len(result) == 7424
    params = cache.evict.call_args.args[0]
    assert params.num_tokens == 0
    assert params.swa_num_tokens == 7424 + len(lengths) * 256 - 2560
    used_pages = sum(-(-length // 256) for length in lengths)
    assert allocator.swa_available_size() == 7680 - used_pages * 256
    assert (
        allocator.logical_attn_allocator.full_available_size()
        == 121600 - used_pages * 256
    )
    compressed_pages = sum(-(-(length // 4) // 64) for length in lengths)
    assert (
        allocator.hisparse_attn_allocator.available_size()
        == 30400 - compressed_pages * 64
    )
    assert torch.all(allocator.full_to_swa_index_mapping[result] > 0)
    compressed = allocator.hisparse_kvcache.translate_loc_from_full_to_compressed(
        result
    )
    assert torch.all(allocator.full_to_hisparse_device_index_mapping[compressed] > 0)
    cache.pretty_print.assert_not_called()


@pytest.mark.parametrize(
    "full,swa,physical,expected_full,expected_swa",
    [
        (256, 7680, 30400, 512, 0),
        (121600, 256, 30400, 0, 512),
        (256, 256, 30400, 512, 512),
        (121600, 7680, 64, 512, 0),
        (121600, 7680, 30400, 0, 0),
    ],
)
def test_wrapped_allocator_keeps_full_physical_and_swa_shortfalls_separate(
    full, swa, physical, expected_full, expected_swa
):
    allocator, full_held, swa_held = make_allocator(
        full=full, swa=swa, physical=physical
    )
    cache = make_cache(allocator, full_held, swa_held)
    evict_from_tree_cache(cache, 768)
    if not expected_full and not expected_swa:
        cache.evict.assert_not_called()
    else:
        cache.evict.assert_called_once()
        params = cache.evict.call_args.args[0]
        assert (params.num_tokens, params.swa_num_tokens) == (
            expected_full,
            expected_swa,
        )


def test_prefill_still_fails_when_swa_eviction_cannot_supply_pages():
    allocator, full_held, swa_held = make_allocator()
    cache = make_cache(allocator, full_held, swa_held, reclaim=False)
    prefix = torch.tensor([0], dtype=torch.int64)
    seq = torch.tensor([7424], dtype=torch.int64)
    last = torch.tensor([-1], dtype=torch.int64)
    with pytest.raises(RuntimeError, match="Prefill out of memory"):
        alloc_paged_token_slots_extend(cache, prefix, prefix, seq, seq, last, 7424)
    assert allocator.swa_available_size() == 2560
    assert allocator.logical_attn_allocator.full_available_size() == 121600
    assert allocator.hisparse_attn_allocator.available_size() == 30400
    cache.pretty_print.assert_called_once()


def test_logical_override_funds_swa_without_evicting_for_physical_shortfall():
    allocator, full_held, swa_held = make_allocator(swa=256, physical=0)
    cache = make_cache(allocator, full_held, swa_held)
    evict_from_tree_cache(cache, 768, allocator=allocator.logical_attn_allocator)
    params = cache.evict.call_args.args[0]
    assert params.num_tokens == 0
    assert params.swa_num_tokens == 512
    assert allocator.swa_available_size() == 768
    assert allocator.hisparse_attn_allocator.available_size() == 0
