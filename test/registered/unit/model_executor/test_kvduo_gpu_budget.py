"""Physical-first GPU sizing, logical backing and safe extra headroom."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.mem_cache.sparsity.core.kvduo_gpu_budget import plan_gpu_budget
from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def dsv4(*, kvduo=True, ratio=2, extra=0, logical_tokens=0):
    args = ServerArgs(
        model_path="dummy",
        enable_kvduo=kvduo,
        kvduo_config=(
            json.dumps(
                {
                    "logical_capacity_ratio": ratio,
                    "logical_capacity_tokens": logical_tokens,
                }
            )
            if kvduo
            else None
        ),
    )
    if kvduo:
        args._handle_kvduo_aliases()
    args.swa_full_tokens_ratio = 0.1
    args.max_running_requests = 64
    kvc = SimpleNamespace(
        kv_cache_dtype_str="fp8_e4m3",
        kvduo_extra_budget_bytes=extra,
        model_config=SimpleNamespace(
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            index_head_dim=128,
            context_len=16384,
            compress_ratios=[4, 4, 128],
            window_size=128,
        ),
        server_args=args,
        layer_info=SimpleNamespace(start_layer=0, end_layer=3),
        ps=SimpleNamespace(pp_size=1, attn_dp_size=1),
    )
    return DSV4PoolConfigurator(kvc)


def test_extra_headroom_preserves_physical_baseline():
    result = plan_gpu_budget(1000, 200, 8, 2, 10, 2, 1)
    assert (result.physical_tokens, result.logical_tokens) == (100, 200)
    assert result.allocated_bytes == 1200
    assert plan_gpu_budget(1000, 100000, 8, 2, 10, 2, 1) == result


def test_no_extra_keeps_logical_overcommit_and_fits_budget():
    result = plan_gpu_budget(1000, 0, 8, 2, 10, 2, 1)
    assert result.physical_tokens == 75
    assert result.logical_tokens == 200
    assert result.allocated_bytes <= 1000


@pytest.mark.parametrize("extra", [0, 1 << 26, 1 << 28])
def test_dsv4_physical_swa_independent_of_logical_history(extra):
    planner = dsv4(extra=extra)
    budget = 1 << 28
    config = planner.calculate_pool_sizes(budget, 256)
    physical = config.c4_max_total_num_tokens * 4
    logical = config.full_max_total_num_tokens
    assert logical >= 2 * physical
    assert config.c128_max_total_num_tokens * 128 == logical
    assert config.swa_max_total_num_tokens == int(physical * 0.1) // 256 * 256
    actual = (
        physical * planner._kvduo_physical_bytes
        + logical * planner._kvduo_logical_bytes
    )
    actual += planner._get_c128_state_fixed_bytes(64)
    assert actual <= budget + extra
    # The old shared ratio inflated SWA/state with the logical address space.
    old_full = int(
        (budget - planner._get_c128_state_fixed_bytes(64))
        / planner.bytes_per_full_token
    )
    old = planner._compute_dsv4_sizes(old_full, 256)
    assert config.c4_max_total_num_tokens > old.c4_max_total_num_tokens


def test_dsv4_extra_reaches_ordinary_physical_target():
    budget = 1 << 28
    baseline = dsv4(kvduo=False).calculate_pool_sizes(budget, 256)
    result = dsv4(extra=budget).calculate_pool_sizes(budget, 256)
    assert result.c4_max_total_num_tokens == baseline.c4_max_total_num_tokens
    assert result.swa_max_total_num_tokens == baseline.swa_max_total_num_tokens
    assert result.full_max_total_num_tokens == 2 * baseline.full_max_total_num_tokens


def test_dsv4_token_cap_does_not_expand_physical_pool():
    planner = dsv4()
    config = planner.calculate_pool_sizes(1 << 28, 256)
    smaller = planner.calculate_pool_sizes_from_max_tokens(
        config.max_total_num_tokens // 2, 256
    )
    assert smaller.c4_max_total_num_tokens <= config.c4_max_total_num_tokens
    assert smaller.full_max_total_num_tokens >= smaller.c4_max_total_num_tokens * 4


def test_too_small_budget_rejected():
    with pytest.raises(ValueError, match="cannot fit"):
        plan_gpu_budget(1, 0, 8, 2, 10, 2, 64)


@pytest.mark.parametrize(
    "kvduo,slack,expected_extra", [(True, 0.75, 18), (True, 0.99, 0), (False, 0.75, 0)]
)
def test_profile_only_borrows_above_runtime_headroom(
    monkeypatch, kvduo, slack, expected_extra
):
    import sglang.srt.mem_cache.kv_cache_configurator as module

    group = SimpleNamespace(world_size=4, cpu_group=object())
    monkeypatch.setattr(module, "get_world_group", lambda: group)
    free = MagicMock(return_value=60)
    monkeypatch.setattr(module, "get_available_gpu_memory", free)
    monkeypatch.setattr(module, "get_device_memory_capacity", lambda device: 80 * 1024)
    monkeypatch.setattr(
        module, "get_schedule", lambda: SimpleNamespace(mem_fraction_static=slack)
    )
    runner = SimpleNamespace(
        device="cuda",
        gpu_id=0,
        mambaish_config=None,
        server_args=SimpleNamespace(
            enable_kvduo=kvduo,
            pre_capture_activation_reserve_mb=lambda capacity: 1024,
            reserve_for_graph_mb=lambda: 512,
            reserve_for_deepep_a2a_mb=lambda: 512,
        ),
    )
    budget = module.KVCacheConfigurator._profile_available_bytes(runner, 80)
    assert runner.kvduo_extra_budget_bytes == expected_extra * (1 << 30)
    if expected_extra:
        assert budget + runner.kvduo_extra_budget_bytes == 58 * (1 << 30)
    assert free.call_args.kwargs["distributed"] is True
    assert free.call_args.kwargs["cpu_group"] is group.cpu_group


@pytest.mark.parametrize("logical_tokens", [0, 1000])
def test_generic_planning_preserves_distinct_physical_and_logical_sizes(logical_tokens):
    from test_pool_configurator import _make_model_runner, mock_cpu_env
    from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
    from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool

    kvc = _make_model_runner(use_mla_backend=True, num_layers=2)
    kvc.kv_cache_dtype = torch.bfloat16
    kvc.kv_cache_dtype_str = "bf16"
    kvc.model_config.hf_config = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"], index_topk=2048, index_head_dim=128
    )
    kvc.model_config.hf_config.get_text_config = lambda: kvc.model_config.hf_config
    kvc.server_args.enable_kvduo = kvc.server_args.enable_hisparse = True
    kvc.server_args.kvduo_config = json.dumps(
        {"logical_capacity_tokens": logical_tokens}
    )
    kvc.server_args.hisparse_config = '{"host_to_device_ratio":2}'
    kvc.server_args.dsa_prefill_backend = kvc.server_args.dsa_decode_backend = (
        "flashmla_sparse"
    )
    kvc.kvduo_extra_budget_bytes = 1 << 20
    with mock_cpu_env():
        planner = DefaultPoolConfigurator(kvc)
    config = planner.calculate_pool_sizes(1 << 20, 64)
    if logical_tokens:
        assert config.kvduo_logical_tokens == 1024
    else:
        assert config.kvduo_logical_tokens == 2 * config.max_total_num_tokens
    pool = HiSparseDSATokenToKVPool(
        size=config.max_total_num_tokens,
        logical_size=config.kvduo_logical_tokens,
        page_size=64,
        kv_lora_rank=512,
        dtype=torch.bfloat16,
        qk_rope_head_dim=64,
        layer_num=2,
        device="cpu",
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=576,
        enable_kvduo=True,
    )
    allocator = HiSparseTokenToKVPoolAllocator(
        pool.size,
        64,
        pool.dtype,
        "cpu",
        pool,
        False,
        logical_size=config.kvduo_logical_tokens,
    )
    assert allocator.logical_attn_allocator.size == config.kvduo_logical_tokens
    assert allocator.hisparse_attn_allocator.size == config.max_total_num_tokens
    assert (
        pool.index_k_with_scale_buffer[0].shape[0] * 64 >= config.kvduo_logical_tokens
    )
    logical = allocator.logical_attn_allocator
    physical = allocator.hisparse_attn_allocator
    occupied = logical.alloc(physical.size)
    locations = physical.alloc(physical.size)
    allocator.full_to_hisparse_device_index_mapping[occupied] = locations
    allocator.free_hisparse(occupied[:64])
    # Logical IDs of offloaded history remain occupied, while a new logical
    # page and the returned physical page can be allocated independently.
    assert logical.alloc(64) is not None
    assert physical.alloc(64) is not None
    smaller = planner.calculate_pool_sizes_from_max_tokens(64, 64)
    assert smaller.max_total_num_tokens == 64
    assert smaller.kvduo_logical_tokens == config.kvduo_logical_tokens


def test_logical_target_survives_smaller_physical_budget():
    small = plan_gpu_budget(1000, 0, 8, 2, 10, 2, 1, logical_tokens=240)
    large = plan_gpu_budget(1000, 400, 8, 2, 10, 2, 1, logical_tokens=240)
    assert small.logical_tokens == large.logical_tokens == 240
    assert small.physical_tokens < large.physical_tokens == 100
    assert small.allocated_bytes <= 1000
    assert large.allocated_bytes <= 1400


def test_explicit_logical_target_rounds_up_and_bounds_physical():
    plan = plan_gpu_budget(10000, 0, 8, 2, 10, 2, 64, logical_tokens=65)
    assert plan.logical_tokens == plan.physical_tokens == 128
    with pytest.raises(ValueError, match="logical KV backing target"):
        plan_gpu_budget(1000, 0, 8, 2, 10, 2, 64, logical_tokens=1000)


def test_dsv4_absolute_logical_capacity_drives_all_history_pools():
    planner = dsv4(logical_tokens=16385)
    result = planner.calculate_pool_sizes(1 << 28, 256)
    assert result.full_max_total_num_tokens == 16640
    assert result.c128_max_total_num_tokens * 128 == 16640
    assert result.c4_max_total_num_tokens * 4 <= 16640


def test_generic_pp_synchronizes_independent_logical_capacity(monkeypatch):
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    import sglang.srt.mem_cache.kv_cache_configurator as module
    import sglang.srt.model_executor.pool_configurator as planners

    config = planners.MemoryPoolConfig(
        max_total_num_tokens=64, kvduo_logical_tokens=256
    )
    planner = SimpleNamespace(calculate_pool_sizes=lambda *args: config)
    monkeypatch.setattr(planners, "create_memory_pool_configurator", lambda _: planner)
    monkeypatch.setattr(module, "get_schedule", lambda: SimpleNamespace(page_size=64))
    monkeypatch.setattr(
        module, "get_world_group", lambda: SimpleNamespace(cpu_group="pp")
    )

    def reduce(tensor, *, op, group):
        assert group == "pp" and op == torch.distributed.ReduceOp.MIN
        tensor.fill_(192)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    runner = SimpleNamespace(
        server_args=SimpleNamespace(pp_size=2),
        _apply_token_constraints=lambda value: value,
    )
    result = KVCacheConfigurator.config_from_budget(runner, 1000)
    assert result.max_total_num_tokens == 64
    assert result.kvduo_logical_tokens == 192


def test_dsv4_request_state_budget_uses_aligned_logical_target():
    planner = dsv4(logical_tokens=16385)
    planner.requested_max_running_requests_per_worker = None
    state_budget = MagicMock(return_value=0)
    planner._get_c128_state_fixed_bytes_for_token_capacity = state_budget
    config = planner.calculate_pool_sizes(1 << 28, 256)
    state_budget.assert_called_once_with(config.full_max_total_num_tokens)
    assert config.full_max_total_num_tokens == 16640


def test_physical_reduction_reports_missing_headroom_without_shrinking_logical(caplog):
    caplog.set_level(
        logging.INFO, logger="sglang.srt.mem_cache.sparsity.core.kvduo_gpu_budget"
    )
    result = plan_gpu_budget(1000, 100, 8, 2, 10, 2, 1)
    assert result.logical_tokens == 200
    assert "target=100, allocated=87, logical=200" in caplog.text
    assert "extra_required_bytes=200" in caplog.text
    assert "additional_headroom_needed_bytes=100" in caplog.text
    caplog.clear()
    restored = plan_gpu_budget(1000, 200, 8, 2, 10, 2, 1)
    assert restored.physical_tokens == 100
    assert restored.logical_tokens == 200
    assert not caplog.records
