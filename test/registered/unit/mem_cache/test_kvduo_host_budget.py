"""Explicit Host budgets must not change HBM allocation or exceed payload bytes."""

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


@pytest.mark.parametrize("value", [0, 0.5, 64])
def test_host_size_config_and_independent_hbm_alias(value):
    args = ServerArgs(
        model_path="dummy",
        enable_kvduo=True,
        kvduo_config=json.dumps({"host_size": value}),
    )
    args._handle_kvduo_aliases()
    assert parse_kvduo_config(args).host_size == value
    assert parse_hisparse_config(args).host_to_device_ratio == 1


def test_defaults_match_hicache_auto_semantics():
    assert ServerArgs(model_path="dummy").hicache_size == 0
    assert parse_kvduo_config(SimpleNamespace(kvduo_config=None)).host_size == 0
    plan = plan_host_budget(0, 100, 20, {"c4": 1000, "indexer": 100}, {"swa": 50})
    assert (plan.full_pages, plan.swa_pages) == (200, 40)
    assert plan.allocated_bytes == 200 * 1100 + 40 * 50


@pytest.mark.parametrize("raw", ["-1", "true", "null", '"64"', "NaN", "Infinity"])
def test_invalid_host_size(raw):
    with pytest.raises(ValueError, match="host_size"):
        parse_kvduo_config(SimpleNamespace(kvduo_config='{"host_size":' + raw + "}"))


def test_removed_ratio_rejected_only_for_kvduo():
    with pytest.raises(ValueError, match="host_to_device_ratio"):
        parse_kvduo_config(SimpleNamespace(kvduo_config='{"host_to_device_ratio":2}'))
    cfg = parse_hisparse_config(
        SimpleNamespace(hisparse_config='{"host_to_device_ratio":4}')
    )
    assert cfg.host_to_device_ratio == 4


@pytest.mark.parametrize("budget", [1120, 5000, 12521, 1000000])
def test_fixed_budget_rounds_down_all_pools(budget):
    plan = plan_host_budget(
        budget / 1e9, 100, 20, {"c4": 1000, "indexer": 100}, {"swa": 20}
    )
    assert plan.allocated_bytes <= budget
    assert plan.full_pages >= 1 and plan.swa_pages >= 1
    n = plan.full_pages + 1
    assert n * 1100 + max(1, n * 20 // 100) * 20 > budget


def test_too_small_budget_fails_before_allocation():
    with pytest.raises(ValueError, match="cannot fit"):
        plan_host_budget(1119 / 1e9, 100, 20, {"c4": 1100}, {"swa": 20})


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
    for budget in (0, 0.01, 0.02):
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
    plan = plan_host_budget((page_bytes * 3 + 100) / 1e9, 4, 0, {"kv": page_bytes}, {})
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


def test_fixed_budget_pp_sync_stays_within_limit(monkeypatch):
    import sglang.srt.mem_cache.pool_host.base as base

    monkeypatch.setattr(base, "sync_fixed_hicache_size", lambda n, size: n // 2)
    plan = plan_host_budget(0.001, 100, 20, {"c4": 1000}, {"swa": 20})
    assert plan.allocated_bytes <= plan.budget_bytes


def test_dsv4_hbm_layout_is_independent_of_host_size():
    from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator

    results = []
    for host_size in (0, 16, 128):
        args = ServerArgs(
            model_path="dummy",
            enable_kvduo=True,
            kvduo_config=json.dumps({"host_size": host_size}),
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
        assert plan.c4_shrink_factor == 1
        results.append((plan.bytes_per_full_token, plan._compute_dsv4_sizes(8192, 256)))
    assert results[0] == results[1] == results[2]
