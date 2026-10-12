import pytest
import torch

from sglang.kernels.ops.kvcache.kvduo_full_write import publish_kvduo_full_write
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_cuda(),
    reason="C4 allocation synchronization checks require CUDA",
)


@pytest.mark.parametrize("requests", [1, 4, 32])
def test_real_paged_c4_decode_and_publication_without_host_wait(requests):
    logical = PagedTokenToKVPoolAllocator(
        requests * 1024, 256, torch.int8, "cuda", None, False
    )
    physical = PagedTokenToKVPoolAllocator(
        requests * 256, 64, torch.int8, "cuda", None, False
    )
    logical.debug_mode = physical.debug_mode = False
    pool = HiSparseC4DevicePool.__new__(HiSparseC4DevicePool)
    pool.compress_ratio = 4
    mapping_size = (logical.size + logical.page_size) // 4 + 1
    mapping = torch.zeros(mapping_size, dtype=torch.int64, device="cuda")
    mapping[-1] = -1
    pool.full_to_hisparse_device_index_mapping = mapping
    allocator = DeepSeekV4HiSparseTokenToKVPoolAllocator.__new__(
        DeepSeekV4HiSparseTokenToKVPoolAllocator
    )
    allocator.compress_ratio = 4
    allocator.hisparse_page_size = 64
    allocator.enable_kvduo_full_decode = True
    allocator.kvduo_physical_allocation_guard = None
    allocator.logical_attn_allocator = logical
    allocator.hisparse_attn_allocator = physical
    allocator.hisparse_kvcache = pool
    allocator.full_to_hisparse_device_index_mapping = mapping

    lengths_cpu = 255 + torch.arange(requests) % 8
    lengths = lengths_cpu.cuda()
    zero_cpu = torch.zeros_like(lengths_cpu)
    zero = zero_cpu.cuda()
    missing = torch.full_like(lengths, -1)
    prefix = logical.alloc_extend(
        zero, zero_cpu, lengths, lengths_cpu, missing, int(lengths_cpu.sum())
    )
    compressed = pool.translate_loc_from_full_to_compressed(prefix)
    mapping[compressed] = physical.alloc_extend(
        zero, zero_cpu, lengths // 4, lengths_cpu // 4, missing, compressed.numel()
    )
    last = prefix.index_select(0, (lengths_cpu.cumsum(0) - 1).cuda())
    generation = torch.ones(mapping_size, dtype=torch.int64, device="cuda")
    data = generation.clone()
    host = generation.clone()
    touches = torch.full_like(generation, 10)
    clock = torch.tensor([10], dtype=torch.int64, device="cuda")
    lookup = torch.full((requests, 80), -1, dtype=torch.int64, device="cuda")
    rows = torch.arange(requests, device="cuda")
    banks = (generation, data, host, touches, clock, lookup, mapping)
    expected = [bank.cpu().clone() for bank in banks]
    allocate_physical = physical.alloc_decode
    physical_outputs = []

    def record_allocation(*args):
        result = allocate_physical(*args)
        physical_outputs.append(result)
        return result

    physical.alloc_decode = record_allocation
    for step in range(9):
        lengths_cpu += 1
        lengths = lengths + 1
        free_before = physical.available_size()
        physical_outputs.clear()
        # Warm the JIT kernels once before checking the steady-state path.
        previous_debug = torch.cuda.get_sync_debug_mode()
        if step:
            torch.cuda.set_sync_debug_mode("error")
        try:
            out = allocator.alloc_decode(lengths, lengths_cpu, last)
            publish_kvduo_full_write(
                out,
                generation,
                data,
                host,
                touches,
                clock,
                lookup=lookup,
                req_indices=rows,
                positions=lengths,
                decode_compress_ratio=4,
            )
        finally:
            torch.cuda.set_sync_debug_mode(previous_debug)
        torch.cuda.synchronize()
        active = lengths_cpu % 4 == 0
        g, d, h, t, c, l, m = expected
        if active.any():
            assert len(physical_outputs) == 1
            raw = out.cpu()[active]
            assert torch.all(raw % 4 == 3)
            ids = (raw - 3) // 4
            c += 1
            g[ids] += 1
            d[ids] = 1
            h[ids] = -1
            t[ids] = c[0]
            l[torch.arange(requests)[active], lengths_cpu[active] // 4 - 1] = ids
            m[ids] = physical_outputs[0].cpu()
        else:
            assert not physical_outputs
        new_pages = (lengths_cpu[active] // 4 % 64 == 1).sum().item()
        assert physical.available_size() == free_before - new_pages * 64
        assert all(
            torch.equal(bank.cpu(), reference)
            for bank, reference in zip(banks, expected)
        )
        last = out
