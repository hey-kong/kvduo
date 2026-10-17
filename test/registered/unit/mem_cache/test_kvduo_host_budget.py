"""Host ratios must not change HBM allocation and must size real payloads."""

import json
from types import SimpleNamespace

import pytest

from sglang.srt.mem_cache.sparsity import parse_hisparse_config, parse_kvduo_config
from sglang.srt.mem_cache.sparsity.core.kvduo_host_budget import (
    plan_dsv4_host_budget,
    plan_host_budget,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("value", [0.5, 2, 4])
def test_host_ratio_config_and_independent_hbm_alias(value):
    args = ServerArgs(
        model_path="dummy",
        enable_kvduo=True,
        kvduo_config=json.dumps({"host_ratio": value}),
    )
    args._handle_kvduo_aliases()
    assert parse_kvduo_config(args).host_ratio == value
    assert parse_hisparse_config(args).host_to_device_ratio == 2.5


def test_defaults_match_hicache_auto_semantics():
    assert ServerArgs(model_path="dummy").hicache_ratio == 2.0
    assert parse_kvduo_config(SimpleNamespace(kvduo_config=None)).host_ratio == 2.0
    plan = plan_host_budget(2, 100, 20, {"c4": 1000, "indexer": 100}, {"swa": 50})
    assert (plan.full_pages, plan.swa_pages) == (200, 40)
    assert plan.allocated_bytes == 200 * 1100 + 40 * 50


@pytest.mark.parametrize("raw", ["0", "-1", "true", "null", '"64"', "NaN", "Infinity"])
def test_invalid_host_ratio(raw):
    with pytest.raises(ValueError, match="host_ratio"):
        parse_kvduo_config(SimpleNamespace(kvduo_config='{"host_ratio":' + raw + "}"))


def test_removed_ratio_rejected_only_for_kvduo():
    with pytest.raises(ValueError, match="host_to_device_ratio"):
        parse_kvduo_config(SimpleNamespace(kvduo_config='{"host_to_device_ratio":2}'))
    cfg = parse_hisparse_config(
        SimpleNamespace(hisparse_config='{"host_to_device_ratio":4}')
    )
    assert cfg.host_to_device_ratio == 4


@pytest.mark.parametrize("ratio", [0.75, 1, 1.5, 2, 3.25])
def test_ratio_scales_each_pool_by_device_page_capacity(ratio):
    plan = plan_host_budget(ratio, 101, 21, {"c4": 1000, "indexer": 100}, {"swa": 20})
    assert plan.full_pages == int(101 * ratio)
    assert plan.swa_pages == int(21 * ratio)
    assert plan.allocated_bytes == plan.full_pages * 1100 + plan.swa_pages * 20


def test_too_small_ratio_fails_before_allocation():
    with pytest.raises(ValueError, match="cannot fit"):
        plan_host_budget(0.001, 100, 20, {"c4": 1100}, {"swa": 20})


def test_removed_host_size_is_rejected():
    with pytest.raises(ValueError, match="host_size"):
        parse_kvduo_config(SimpleNamespace(kvduo_config='{"host_size": 0}'))


def test_dsv4_budget_matches_actual_host_buffers(monkeypatch):
    from test_kvduo_prefill_state import stack
    from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
    from sglang.srt.mem_cache.sparsity.core.kvduo_prefill_state import (
        KVDuoPrefillHostCache,
    )

    c = stack(monkeypatch)
    allocator = c.token_to_kv_pool_allocator
    mapping = allocator.full_to_hisparse_device_index_mapping.clone()
    gpu_free = allocator.hisparse_attn_allocator.available_size()
    for budget in (0.5, 2, 3):
        plan = plan_dsv4_host_budget(allocator, budget)
        c.mem_pool_host = DeepSeekV4PagedHostPool(
            "budget_test_c4",
            c.mem_pool_device.kv_buffer,
            c.mem_pool_device.bytes_per_page_padded,
            plan.full_pages,
            c.page_size,
            pin_memory=False,
        )
        cache = KVDuoPrefillHostCache.create(c, num_swa_pages=plan.swa_pages)
        pools = {"c4": c.mem_pool_host, **cache.history.pools, **cache.window.pools}
        actual = {
            name: sum(t.nbytes for t in pool.kv_buffer) for name, pool in pools.items()
        }
        assert actual == plan.pool_bytes
        assert sum(actual.values()) <= plan.budget_bytes
        assert allocator.hisparse_attn_allocator.available_size() == gpu_free
        assert allocator.full_to_hisparse_device_index_mapping.equal(mapping)


def test_generic_host_exact_pages_and_existing_default(monkeypatch):
    import torch
    from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
    from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

    pool = HiSparseDSATokenToKVPool(
        size=256,
        page_size=64,
        kv_lora_rank=12,
        dtype=torch.bfloat16,
        qk_rope_head_dim=4,
        layer_num=2,
        device="cpu",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=16,
        enable_kvduo=True,
        host_to_device_ratio=1,
    )
    page_bytes = 64 * 16 * 2 * 2
    plan = plan_host_budget(0.75, 4, 0, {"kv": page_bytes}, {})
    kw = dict(
        device_pool=pool,
        host_to_device_ratio=2,
        host_size=0,
        page_size=64,
        layout="layer_first",
        pin_memory=False,
        override_kv_cache_dim=16,
    )
    host = MLATokenToKVPoolHost(**kw, num_host_pages=plan.full_pages)
    assert host.kv_buffer.nbytes == plan.allocated_bytes == 3 * page_bytes
    indices = host.alloc(3 * 64)
    assert indices.numel() == 3 * 64
    assert host.alloc(64) is None
    host.free(indices)
    assert host.available_size() == 3 * 64
    # Ordinary HiSparse/HiCache retain the existing alignment policy.
    legacy = MLATokenToKVPoolHost(**kw)
    assert legacy.size == (2 * 256 // 64 + 1) * 64


def test_dsv4_hbm_layout_is_independent_of_host_ratio():
    from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator

    results = []
    for host_ratio in (0.5, 2, 4):
        args = ServerArgs(
            model_path="dummy",
            enable_kvduo=True,
            kvduo_config=json.dumps({"host_ratio": host_ratio}),
        )
        args._handle_kvduo_aliases()
        args.swa_full_tokens_ratio = 0.1
        kvc = SimpleNamespace(
            kv_cache_dtype_str="fp8_e4m3",
            model_config=SimpleNamespace(
                qk_nope_head_dim=448,
                qk_rope_head_dim=64,
                index_head_dim=128,
                context_len=16384,
                compress_ratios=[4, 128],
                window_size=128,
            ),
            server_args=args,
            layer_info=SimpleNamespace(start_layer=0, end_layer=2),
            ps=SimpleNamespace(pp_size=1, attn_dp_size=1),
        )
        plan = DSV4PoolConfigurator(kvc)
        assert plan.c4_shrink_factor == 2.5
        results.append((plan.bytes_per_full_token, plan._compute_dsv4_sizes(8192, 256)))
    assert results[0] == results[1] == results[2]
    sizes = results[0][1]
    assert sizes.c4_max_total_num_tokens == int(sizes.full_max_total_num_tokens // (2.5 * 4))


@pytest.mark.parametrize("raw", ["0", "-1", "true", "0.5", "NaN", "Infinity", '"2"', "null"])
def test_invalid_logical_capacity_ratio(raw):
    with pytest.raises(ValueError, match="logical_capacity_ratio"):
        parse_kvduo_config(
            SimpleNamespace(kvduo_config='{"logical_capacity_ratio":' + raw + "}")
        )


@pytest.mark.parametrize("host_ratio", [0.5, 2, 4])
@pytest.mark.parametrize("logical_ratio", [1.5, 2, 2.5, 4])
def test_generic_logical_slots_survive_physical_offload(host_ratio, logical_ratio):
    import torch
    from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
    from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator

    args = ServerArgs(
        model_path="dummy",
        enable_kvduo=True,
        kvduo_config=json.dumps(
            {"host_ratio": host_ratio, "logical_capacity_ratio": logical_ratio}
        ),
    )
    args._handle_kvduo_aliases()
    ratio = parse_hisparse_config(args).host_to_device_ratio
    pool = HiSparseDSATokenToKVPool(
        size=1024,
        page_size=64,
        kv_lora_rank=12,
        dtype=torch.bfloat16,
        qk_rope_head_dim=4,
        layer_num=2,
        device="cpu",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=16,
        enable_kvduo=True,
        host_to_device_ratio=ratio,
    )
    allocator = HiSparseTokenToKVPoolAllocator(
        1024, 64, pool.dtype, "cpu", pool, False, host_to_device_ratio=ratio
    )
    logical = allocator.logical_attn_allocator
    physical = allocator.hisparse_attn_allocator
    occupied = logical.alloc(1024)
    locations = physical.alloc(1024)
    allocator.full_to_hisparse_device_index_mapping[occupied] = locations
    allocator.free_hisparse(occupied[:768])
    assert physical.available_size() == 768
    assert logical.available_size() == (logical_ratio - 1) * 1024
    next_page = logical.alloc(64)
    assert next_page is not None and int(next_page.min()) > 1024
    # An offloaded token's indexer remains GPU-resident at its logical address.
    indexer = pool.index_k_with_scale_buffer[0]
    assert indexer.shape[0] * pool.page_size >= (logical_ratio * 1024 + 64)
    assert physical.alloc(64) is not None


def test_dsv4_logical_and_indexer_capacity_survive_full_offload(monkeypatch):
    import torch
    from test_kvduo_prefill_state import stack

    c = stack(monkeypatch)
    allocator = c.token_to_kv_pool_allocator
    logical = allocator.logical_attn_allocator.full_attn_allocator
    physical = allocator.hisparse_attn_allocator
    kv = allocator.get_kvcache()
    assert logical.size == physical.size * 4 * 2
    full = logical.alloc(physical.size * 4)
    locations = physical.alloc(physical.size)
    compressed = torch.arange(int(full[0]) // 4, (int(full[-1]) + 1) // 4)
    allocator.full_to_hisparse_device_index_mapping[compressed] = locations
    allocator.free_compressed(compressed[:768])
    assert physical.available_size() == 768
    assert logical.available_size() == 4096
    next_page = logical.alloc(256)
    assert next_page is not None
    assert kv.c4_logical_size == logical.size // 4
    indexer = kv.c4_indexer_kv_pool.index_k_with_scale_buffer[0]
    assert indexer.shape[0] * 64 >= kv.c4_logical_size
    assert physical.alloc(64) is not None
    # Automatic Host sizing uses physical capacity, not logical overcommit.
    budget = plan_dsv4_host_budget(allocator, 2)
    assert budget.full_pages == 2 * physical.size // physical.page_size


@pytest.mark.parametrize("raw", ["-1", "true", "1.5", '"2"', "null"])
def test_invalid_logical_capacity_tokens(raw):
    with pytest.raises(ValueError, match="logical_capacity_tokens"):
        parse_kvduo_config(
            SimpleNamespace(kvduo_config='{"logical_capacity_tokens":' + raw + "}")
        )
