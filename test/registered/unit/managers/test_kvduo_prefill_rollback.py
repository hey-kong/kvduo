"""Production prefill rollback policy, ownership, and admission regressions."""

import ctypes
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.sparsity.core.kvduo_prefill_rollback import (
    KVDuoPrefillObject,
    KVDuoPrefillPolicy,
    plan_kvduo_prefill_rollback,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_hot_merge import make_merge_coordinator
from test_kvduo_hot_tail_ema import native_ema


register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def obj(req=0, **changes):
    value = KVDuoPrefillObject((0, req), 4, 7, (100, 20, 60, 1), 65, 0.8, 65)
    return replace(value, **changes)


def test_example_and_per_released_page_order():
    assert plan_kvduo_prefill_rollback([obj()], 3, KVDuoPrefillPolicy()) == ((0, 0),)
    a = obj(0, base_pages=1, current_pages=8, counts=(100, 20, 65, 1))
    b = obj(1, base_pages=1, current_pages=3)
    assert plan_kvduo_prefill_rollback([b, a], 1, KVDuoPrefillPolicy()) == ((0, 0),)
    assert plan_kvduo_prefill_rollback([obj(1), obj(0)], 1, KVDuoPrefillPolicy()) == (
        (0, 1),
    )
    assert plan_kvduo_prefill_rollback([obj()], 4, KVDuoPrefillPolicy()) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": 64},
        {"reference_steps": 64},
        {"reference_steps": 0},
        {"base_pages": 0},
        {"current_pages": 4},
        {"eligible": False},
        {"reference": 0.9},
        {"counts": (100, 20, 57, 1)},
        {"counts": (100, 20, 60, 6)},
        {"counts": (0, 0, 0, 0)},
        {"counts": (100, 80, 60, 0)},
        {"reference": float("nan")},
    ],
)
def test_candidate_requires_both_mature_phases_and_all_thresholds(changes):
    assert (
        plan_kvduo_prefill_rollback([obj(**changes)], 1, KVDuoPrefillPolicy()) is None
    )


def test_fixed_round_cost_never_renews_on_miss_growth_or_ema_decay():
    policy = KVDuoPrefillPolicy()
    references = {(0, 0): 10, (0, 1): 10}
    objects = [
        obj(0, counts=(100, 10, 80, 0), steps=0, current_pages=4),
        obj(1, counts=(100, 10, 79, 1), reference=0.8),
    ]
    # Total original miss budget = 2, already committed cost = 1.5.
    assert (
        plan_kvduo_prefill_rollback(
            objects, 1, policy, reference_misses=references, reserved_cost=1.5
        )
        is None
    )
    assert plan_kvduo_prefill_rollback(
        objects, 1, policy, reference_misses=references, reserved_cost=1
    ) == ((0, 1),)
    newcomer = obj(2, counts=(100, 10, 80, 0))
    assert (
        plan_kvduo_prefill_rollback([newcomer], 1, policy, reference_misses=references)
        is None
    )


def test_revised_examples_and_zero_miss_have_no_hidden_tolerance():
    assert (
        plan_kvduo_prefill_rollback(
            [obj(counts=(100, 20, 60, 3))], 1, KVDuoPrefillPolicy()
        )
        is None
    )
    assert plan_kvduo_prefill_rollback([obj()], 1, KVDuoPrefillPolicy()) == ((0, 0),)
    assert plan_kvduo_prefill_rollback(
        [obj(counts=(100, 20, 80, 0), reference=1)], 1, KVDuoPrefillPolicy()
    ) == ((0, 0),)
    assert (
        plan_kvduo_prefill_rollback(
            [obj(counts=(100, 20, 79, 1), reference=1)], 1, KVDuoPrefillPolicy()
        )
        is None
    )


def ready_coordinator(extra=3, requests=1):
    coordinator, physical = make_merge_coordinator(
        {(0, 0): 8, **{(0, req): 1 for req in range(1, requests)}}
    )
    coordinator._merge_kvduo_hot_pages(0, 0)
    coordinator.kvduo_prefill_hit_ema[0, 0] = torch.tensor([100, 20, 60, 0])
    coordinator.kvduo_prefill_hit_steps[0, 0] = 65
    coordinator._ensure_kvduo_hot_capacity_targets_batch({(0, 0): (4 + extra) * 4})
    coordinator.kvduo_prefill_hit_ema[0, 0] = torch.tensor([100, 20, 60, 1])
    coordinator.kvduo_prefill_hit_steps[0, 0] = 65
    for req in range(1, requests):
        coordinator.kvduo_prefill_hit_ema[0, req] = torch.tensor([100, 80, 0, 0])
        coordinator.kvduo_prefill_hit_steps[0, req] = 65
        coordinator.kvduo_hot_tail_observation_steps[0, req] = 0
    physical.free_pages = 0
    return coordinator, physical


def test_merge_builds_reference_point_growth_freezes_reference_and_new_merge_resets():
    coordinator, physical = make_merge_coordinator({(0, 0): 8})
    coordinator._merge_kvduo_hot_pages(0, 0)
    base = coordinator._kvduo_hot_base_pages[(0, 0)]
    assert len(base) == 4
    assert int(coordinator.kvduo_hot_base_capacity[0, 0]) == 16
    coordinator.kvduo_prefill_hit_ema[0, 0] = torch.tensor([10, 2, 6, 0])
    coordinator.kvduo_prefill_hit_steps[0, 0] = 65
    coordinator._ensure_kvduo_hot_capacity_targets_batch({(0, 0): 20})
    assert float(coordinator.kvduo_prefill_reference[0, 0]) == pytest.approx(0.8)
    assert int(coordinator.kvduo_prefill_reference_steps[0, 0]) == 65
    assert int(coordinator.kvduo_prefill_hit_steps[0, 0]) == 0
    coordinator.kvduo_prefill_hit_ema[0, 0] = torch.tensor([10, 2, 6, 1])
    coordinator.kvduo_prefill_hit_steps[0, 0] = 1
    coordinator._ensure_kvduo_hot_capacity_targets_batch({(0, 0): 24})
    assert coordinator._kvduo_hot_base_pages[(0, 0)] == base
    assert int(coordinator.kvduo_prefill_hit_steps[0, 0]) == 1
    coordinator._merge_kvduo_hot_pages(0, 0)
    assert len(coordinator._kvduo_hot_base_pages[(0, 0)]) == 3
    assert int(coordinator.kvduo_prefill_reference_steps[0, 0]) == 0


def test_early_expansion_has_insufficient_reference_forever_until_new_merge():
    coordinator, physical = ready_coordinator()
    coordinator.kvduo_prefill_reference_steps[0, 0] = 64
    coordinator.kvduo_prefill_hit_steps[0, 0] = 1000
    assert coordinator.reserve_kvduo_prefill(1, [0]) is None
    assert physical.free_pages == 0


def test_rollback_stably_filters_real_lru_without_moving_payload_and_frees_after_publish():
    coordinator, physical = ready_coordinator(requests=2)
    owner = (0, 0)
    base = coordinator._kvduo_hot_base_pages[owner]
    order = list(range(27, -1, -1))
    coordinator.lru_slots[owner][:28] = torch.tensor(order, dtype=torch.int16)
    expected = [slot for slot in order if slot < 16]
    payload = coordinator.mem_pool_device.kv_buffer[0].clone()
    addresses = [
        bank.data_ptr()
        for bank in (
            coordinator.req_device_buffer_tokens,
            coordinator.req_device_buffer_token_locs,
            coordinator.lru_slots,
            coordinator.kvduo_req_hot_capacity_gpu,
            coordinator.kvduo_prefill_hit_ema,
        )
    ]
    free = coordinator.token_to_kv_pool_allocator.free_hisparse_indices.side_effect

    def verify_then_free(indices):
        assert coordinator._kvduo_req_layer_pages[owner] == list(base)
        assert coordinator.lru_slots[owner][:16].tolist() == expected
        assert int(coordinator.kvduo_req_hot_capacity_gpu[owner]) == 16
        assert coordinator._kvduo_reserved_prefill_pages() == 3
        assert owner in coordinator._kvduo_prefill_growth_blocked
        free(indices)

    coordinator.token_to_kv_pool_allocator.free_hisparse_indices.side_effect = (
        verify_then_free
    )
    coordinator._move_kvduo_hot_entries = MagicMock()
    token = coordinator.reserve_kvduo_prefill(3, [0, 1])
    assert token is not None and physical.free_pages == 3
    coordinator._move_kvduo_hot_entries.assert_not_called()
    assert torch.equal(coordinator.mem_pool_device.kv_buffer[0], payload)
    assert torch.all(coordinator.req_device_buffer_tokens[owner][16:] == -1)
    assert torch.all(coordinator.req_device_buffer_token_locs[owner][16:] == -1)
    assert torch.all(coordinator.lru_slots[owner][16:] == -1)
    assert int(coordinator.kvduo_hot_tail_observation_steps[owner]) == 0
    assert coordinator.kvduo_prefill_hit_ema[owner].tolist() == [0, 0, 0, 0]
    assert sum(coordinator._kvduo_prefill_budget_cost.values()) == 1
    assert float(coordinator.kvduo_prefill_reference[owner]) == pytest.approx(0.8)
    assert addresses == [
        bank.data_ptr()
        for bank in (
            coordinator.req_device_buffer_tokens,
            coordinator.req_device_buffer_token_locs,
            coordinator.lru_slots,
            coordinator.kvduo_req_hot_capacity_gpu,
            coordinator.kvduo_prefill_hit_ema,
        )
    ]


def test_reservation_blocks_other_growth_and_resume_discards_old_misses():
    coordinator, physical = ready_coordinator(requests=2)
    token = coordinator.reserve_kvduo_prefill(3, [0, 1])
    coordinator.kvduo_hot_growth_events[0, 1] = 1
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    assert int(coordinator.kvduo_req_hot_capacity[0, 1]) == 4
    coordinator.consume_kvduo_prefill_reservation(token, 3)
    physical.free_pages -= 3  # prefill allocation
    coordinator.finish_kvduo_prefill_reservation(token, cancelled=False)
    coordinator.kvduo_hot_growth_events[0, 0] = 1
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 16
    physical.free_pages = 1
    coordinator.kvduo_hot_growth_events[0, 0] = 1
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    assert (0, 0) not in coordinator._kvduo_prefill_growth_blocked
    assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 16
    coordinator.kvduo_hot_growth_events[0, 0] = 1
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 20
    assert int(coordinator.kvduo_prefill_reference_steps[0, 0]) == 65


def test_growth_hold_ends_with_free_headroom_even_without_a_new_miss():
    coordinator, physical = ready_coordinator()
    token = coordinator.reserve_kvduo_prefill(3, [0])
    coordinator.consume_kvduo_prefill_reservation(token, 3)
    physical.free_pages -= 3
    coordinator.finish_kvduo_prefill_reservation(token, cancelled=False)
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert (0, 0) in coordinator._kvduo_prefill_growth_blocked
    physical.free_pages = 1
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert not coordinator._kvduo_prefill_growth_blocked
    assert not coordinator._kvduo_prefill_budget_reference
    assert not coordinator._kvduo_prefill_budget_cost
    assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 16


def test_cancellation_and_request_reuse_clear_reservations_holds_and_reference():
    coordinator, physical = ready_coordinator()
    with pytest.raises(RuntimeError):
        with coordinator.kvduo_prefill_reservation_scope():
            coordinator.reserve_kvduo_prefill(3, [0])
            raise RuntimeError("cancel admission")
    assert not coordinator._kvduo_prefill_reservations
    assert not coordinator._kvduo_prefill_growth_blocked
    coordinator._reset_kvduo_hot_tail_ema(0)
    assert not coordinator._kvduo_hot_base_pages
    assert not coordinator._kvduo_prefill_reference_frozen
    assert not torch.any(coordinator.kvduo_prefill_hit_ema)
    assert not torch.any(coordinator.kvduo_prefill_reference_steps)


def test_tp_rejection_and_failed_metadata_update_release_no_pages(monkeypatch):
    coordinator, physical = ready_coordinator()
    coordinator.tp_world_size = 2
    coordinator.tp_group = object()

    def reject(ready, **kwargs):
        assert ready.device.type == "cpu"
        ready.zero_()

    monkeypatch.setattr(torch.distributed, "all_reduce", reject)
    assert coordinator.reserve_kvduo_prefill(3, [0]) is None
    assert physical.free_pages == 0
    coordinator.tp_world_size = 1
    monkeypatch.setattr(
        "sglang.srt.managers.hisparse_coordinator.rollback_kvduo_hot_metadata",
        MagicMock(side_effect=RuntimeError("failed")),
    )
    with pytest.raises(RuntimeError):
        coordinator.reserve_kvduo_prefill(3, [0])
    assert physical.free_pages == 0
    assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 28
    assert not coordinator._kvduo_prefill_reservations


@pytest.mark.parametrize(
    "prefix,end,ratio,expected",
    [
        (3, 4, 1, 0),
        (3, 5, 1, 1),
        (1, 3, 4, 0),
        (3, 4, 4, 1),
        (4, 9, 4, 0),
        (4, 12, 4, 1),
    ],
)
def test_physical_deficit_matches_allocator_rounding(prefix, end, ratio, expected):
    coordinator, _ = ready_coordinator()
    coordinator.compress_ratio = ratio
    assert coordinator.kvduo_prefill_pages(prefix, end) == expected


def budget_allocator(coordinator, *, dsv4=False, logical_size=1024, swa_size=1024):
    # Use real budget methods and real logical/SWA page inventories. Only the
    # GPU backing construction is omitted; the physical pool is the same one
    # that production rollback returns pages to in this fixture.
    page_size = coordinator.page_size * (4 if dsv4 else 1)
    full = PagedTokenToKVPoolAllocator(
        logical_size, page_size, torch.int64, "cpu", None, False
    )
    if dsv4:
        logical = object.__new__(SWATokenToKVPoolAllocator)
        logical.full_attn_allocator = full
        logical.swa_attn_allocator = PagedTokenToKVPoolAllocator(
            swa_size, page_size, torch.int64, "cpu", None, False
        )
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        coordinator.compress_ratio = 4
        allocator.hisparse_page_size = coordinator.page_size
    else:
        logical = full
        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
    allocator.logical_attn_allocator = logical
    allocator.hisparse_attn_allocator = (
        coordinator.token_to_kv_pool_allocator.hisparse_attn_allocator
    )
    allocator.compress_ratio = coordinator.compress_ratio
    allocator.page_size = page_size
    return allocator


def adder_and_req(
    coordinator, *, host_hit=0, chunk=4, dsv4=False, logical_size=1024, swa_size=1024
):
    adder = PrefillAdder.__new__(PrefillAdder)
    adder.dsa_prefill_cp_in_seq_split = False
    adder.prefill_max_requests = None
    adder.can_run_list = []
    adder.new_chunked_req = None
    adder.token_to_kv_pool_allocator = budget_allocator(
        coordinator, dsv4=dsv4, logical_size=logical_size, swa_size=swa_size
    )
    adder.page_size = adder.token_to_kv_pool_allocator.page_size
    adder.rem_chunk_tokens = chunk
    adder.rem_input_tokens = 100
    adder.rem_total_token_offset = adder.cur_rem_token_offset = 0
    adder._mamba_slot_cost = 0
    adder.is_all_swa = adder.is_hybrid_ssm_cache = False
    adder.is_hybrid_swa = dsv4
    adder.rem_swa_token_offset = 0
    adder.dllm_config = adder.prefill_delayer_single_pass = None
    adder.tree_cache = MagicMock(
        disable=False,
        evictable_size=MagicMock(return_value=0),
        full_evictable_size=MagicMock(return_value=0),
        swa_evictable_size=MagicMock(return_value=0),
        sliding_window_size=adder.page_size,
        is_tree_cache=MagicMock(return_value=False),
    )
    adder._req_inc_lock_ref = MagicMock()
    adder._update_prefill_budget = MagicMock()
    adder.budget_state = MagicMock(return_value=AddReqResult.CONTINUE)
    adder.kvduo_coordinator = coordinator
    adder.kvduo_prefill_reservations = []
    adder.running_batch = SimpleNamespace(
        reqs=[SimpleNamespace(req_pool_idx=0, finished=lambda: False)]
    )
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(ignore_eos=False, max_new_tokens=0),
        output_ids=[],
        full_untruncated_fill_ids=list(range(16)),
        prefix_indices=torch.tensor([0, 1], dtype=torch.int64),
        host_hit_length=host_hit,
        swa_host_hit_length=0,
        mamba_pool_idx=None,
        last_node=object(),
        best_match_node=object(),
        cache_protected_len=2,
        retracted_stain=False,
        kvduo_residency_plan=object(),
        needs_host_load_back=lambda: bool(host_hit),
    )
    req.set_extend_range = lambda start, end: setattr(
        req, "extend_range", Range(start, end)
    )
    return adder, req


def actual_budget_adder_and_req(
    coordinator, *, path="new", running_max_new=0, **kwargs
):
    fixture, req = adder_and_req(coordinator, dsv4=True, **kwargs)
    fixture.tree_cache.supports_mamba.return_value = False
    running_req = fixture.running_batch.reqs[0]
    running_req.sampling_params = SimpleNamespace(max_new_tokens=running_max_new)
    running_req.output_ids = []
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    adder = PrefillAdder(
        fixture.page_size,
        fixture.tree_cache,
        fixture.token_to_kv_pool_allocator,
        fixture.running_batch,
        new_token_ratio=1,
        rem_input_tokens=100,
        rem_chunk_tokens=16 if path == "chunk" else None,
        kvduo_coordinator=coordinator,
    )
    req.prefix_indices = torch.arange(8, dtype=torch.int64)
    req.full_untruncated_fill_ids = list(range(40 if path == "chunk" else 24))
    req.cache_protected_len = 8
    return adder, req


@pytest.mark.parametrize("path", ["new", "chunk"])
def test_dsv4_full_physical_pool_reaches_rollback_with_real_allocator_budgets(path):
    coordinator, physical = ready_coordinator()
    adder, req = actual_budget_adder_and_req(coordinator, path=path)
    allocator = adder.token_to_kv_pool_allocator
    assert allocator.logical_attn_allocator.full_available_size() == 1024
    assert allocator.swa_available_size() == 1024
    assert allocator.full_available_size() == 0
    assert allocator.available_size() == 0
    reserve = MagicMock(wraps=coordinator.reserve_kvduo_prefill)
    coordinator.reserve_kvduo_prefill = reserve
    if path == "new":
        assert adder.add_one_req(req, False, None) is AddReqResult.CONTINUE
    else:
        assert adder.add_chunked_req(req) is req
    reserve.assert_called_once_with(
        2, [0], tree_cache=adder.tree_cache, prefix_indices=req.prefix_indices
    )
    assert adder.can_run_list == [req]
    assert req.extend_range == Range(8, 24)
    assert physical.free_pages == 3
    assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 16
    assert coordinator._kvduo_reserved_prefill_pages() == 2
    assert adder.rem_total_tokens == adder.cur_rem_tokens == 1000
    # Actual allocator budgeting still bounds allocations by physical capacity.
    assert allocator.full_available_size() == 24
    assert adder.rem_swa_token_offset > 0


@pytest.mark.parametrize("path", ["new", "chunk"])
@pytest.mark.parametrize("pool", ["logical", "swa"])
def test_rollback_admission_preserves_logical_and_swa_budget_rejections(path, pool):
    coordinator, physical = ready_coordinator()
    adder, req = actual_budget_adder_and_req(
        coordinator,
        path=path,
        logical_size=0 if pool == "logical" else 1024,
        swa_size=0 if pool == "swa" else 1024,
    )
    coordinator.reserve_kvduo_prefill = MagicMock(
        wraps=coordinator.reserve_kvduo_prefill
    )
    if path == "new":
        assert adder.add_one_req(req, False, None) is AddReqResult.NO_TOKEN
    else:
        assert adder.add_chunked_req(req) is req
    coordinator.reserve_kvduo_prefill.assert_not_called()
    assert physical.free_pages == 0
    assert not adder.can_run_list and not hasattr(req, "extend_range")
    assert not coordinator._kvduo_prefill_growth_blocked


@pytest.mark.parametrize("path", ["new", "chunk"])
@pytest.mark.parametrize("reason", ["immature", "miss_budget", "space"])
def test_real_allocator_budget_separation_does_not_bypass_rollback_conditions(
    path, reason
):
    coordinator, physical = ready_coordinator()
    adder, req = actual_budget_adder_and_req(coordinator, path=path)
    if reason == "immature":
        coordinator.kvduo_prefill_reference_steps.zero_()
    elif reason == "miss_budget":
        coordinator.kvduo_prefill_hit_ema[0, 0, 3] = 3
    else:
        req.full_untruncated_fill_ids = list(range(40))
        adder.rem_chunk_tokens = 32 if path == "chunk" else None
    reserve = MagicMock(wraps=coordinator.reserve_kvduo_prefill)
    coordinator.reserve_kvduo_prefill = reserve
    if path == "new":
        assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    else:
        assert adder.add_chunked_req(req) is req
    reserve.assert_called_once()
    assert adder.kvduo_prefill_blocked
    assert physical.free_pages == 0
    assert not adder.can_run_list and not hasattr(req, "extend_range")
    assert not coordinator._kvduo_prefill_reservations
    assert not coordinator._kvduo_prefill_growth_blocked


@pytest.mark.parametrize("path", ["new", "chunk"])
@pytest.mark.parametrize("coordinator_state", ["absent", "disabled"])
def test_non_kvduo_hisparse_keeps_composite_physical_budget(path, coordinator_state):
    coordinator, physical = ready_coordinator()
    adder, req = actual_budget_adder_and_req(coordinator, path=path)
    coordinator.reserve_kvduo_prefill = MagicMock()
    if coordinator_state == "absent":
        adder.kvduo_coordinator = None
    else:
        coordinator.enable_mixed_residency = False
    assert adder.rem_total_tokens == adder.cur_rem_tokens == 0
    if path == "new":
        assert adder.add_one_req(req, False, None) is AddReqResult.NO_TOKEN
    else:
        assert adder.add_chunked_req(req) is req
    coordinator.reserve_kvduo_prefill.assert_not_called()
    assert physical.free_pages == 0
    assert not adder.can_run_list


def test_generic_kvduo_chunk_cannot_use_fallback_to_bypass_empty_logical_pool():
    coordinator, physical = ready_coordinator()
    adder, req = adder_and_req(coordinator, logical_size=0)
    coordinator.reserve_kvduo_prefill = MagicMock()
    assert adder.add_chunked_req(req) is req
    coordinator.reserve_kvduo_prefill.assert_not_called()
    assert not adder.can_run_list and not hasattr(req, "extend_range")
    assert physical.free_pages == 0


def test_logical_budget_is_rechecked_after_prefix_lock_before_rollback():
    coordinator, physical = ready_coordinator()
    adder, req = actual_budget_adder_and_req(coordinator, logical_size=0)
    adder.tree_cache.full_evictable_size.return_value = 32
    coordinator.reserve_kvduo_prefill = MagicMock()

    def lock(_):
        adder.tree_cache.full_evictable_size.return_value = 0

    adder.tree_cache.inc_lock_ref.side_effect = lock
    assert adder.add_one_req(req, False, None) is AddReqResult.NO_TOKEN
    adder.tree_cache.inc_lock_ref.assert_called_once()
    coordinator.reserve_kvduo_prefill.assert_not_called()
    assert physical.free_pages == 0
    assert not adder.can_run_list


@pytest.mark.parametrize("path", ["new", "chunk"])
def test_kvduo_keeps_logical_headroom_for_future_decode_before_rollback(path):
    coordinator, physical = ready_coordinator()
    adder, req = actual_budget_adder_and_req(
        coordinator, path=path, running_max_new=1024 if path == "chunk" else 0
    )
    coordinator.reserve_kvduo_prefill = MagicMock()
    assert (
        adder.token_to_kv_pool_allocator.logical_attn_allocator.full_available_size()
        == 1024
    )
    if path == "new":
        req.sampling_params.max_new_tokens = 1000  # extend + decode + page = 1024
        assert adder.add_one_req(req, False, None) is AddReqResult.NO_TOKEN
    else:
        assert adder.rem_total_token_offset == 1024
        assert adder.add_chunked_req(req) is req
    coordinator.reserve_kvduo_prefill.assert_not_called()
    assert not adder.can_run_list and not hasattr(req, "extend_range")
    assert physical.free_pages == 0


def test_multiple_prefills_charge_logical_swa_offsets_and_separate_physical_reservations():
    coordinator, physical = ready_coordinator()
    adder, first = actual_budget_adder_and_req(coordinator)
    _, second = actual_budget_adder_and_req(coordinator)
    second.full_untruncated_fill_ids = list(range(16))
    assert adder.add_one_req(first, False, None) is AddReqResult.CONTINUE
    assert adder.add_one_req(second, False, None) is AddReqResult.CONTINUE
    assert adder.can_run_list == [first, second]
    assert adder.rem_total_tokens == adder.cur_rem_tokens == 984
    assert adder.rem_swa_token_offset == 40
    assert coordinator._kvduo_reserved_prefill_pages() == physical.free_pages == 3
    assert len(adder.kvduo_prefill_reservations) == 2
    # A third prefill cannot use the already-reserved physical pages.
    _, third = actual_budget_adder_and_req(coordinator)
    third.full_untruncated_fill_ids = list(range(16))
    assert adder.add_one_req(third, False, None) is AddReqResult.OTHER
    assert adder.can_run_list == [first, second]
    assert coordinator._kvduo_reserved_prefill_pages() == 3


def test_prefill_waits_without_mutating_chunk_or_loading_host_when_no_complete_plan():
    coordinator, physical = ready_coordinator()
    coordinator.kvduo_prefill_reference_steps.zero_()
    adder, req = adder_and_req(coordinator, host_hit=4)
    assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    assert not adder.can_run_list and not hasattr(req, "extend_range")
    adder.tree_cache.init_load_back.assert_not_called()
    assert physical.free_pages == 0
    assert adder.add_chunked_req(req) is req
    assert not adder.can_run_list and not hasattr(req, "extend_range")


def test_host_restore_and_chunk_share_exact_reservation_and_cancellation_releases_hold():
    coordinator, physical = ready_coordinator(extra=4)
    adder, req = adder_and_req(coordinator, host_hit=4)

    def restore(_):
        assert coordinator._kvduo_reserved_prefill_pages() == 4
        physical.free_pages -= 2
        return torch.tensor([2, 3, 4, 5]), object()

    adder.tree_cache.init_load_back.side_effect = restore
    assert adder.add_one_req(req, False, None) is AddReqResult.CONTINUE
    assert req.extend_range == Range(6, 10)
    assert coordinator._kvduo_reserved_prefill_pages() == 2
    assert physical.free_pages == 2
    token = adder.kvduo_prefill_reservations[0]
    coordinator.finish_kvduo_prefill_reservation(token, cancelled=False)
    assert coordinator._kvduo_reserved_prefill_pages() == 0
    coordinator, physical = ready_coordinator(extra=4)
    adder, req = adder_and_req(coordinator, host_hit=4)
    adder.tree_cache.init_load_back.return_value = (None, None)
    assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    assert not coordinator._kvduo_prefill_reservations
    assert not coordinator._kvduo_prefill_growth_blocked


def test_actual_cpp_contribution_ema_initializes_zero_hit_steps_and_derives_misses(
    native_ema,
):
    counts = (ctypes.c_float * 4)(999, 999, 999, 999)
    steps = ctypes.c_int64(0)
    native_ema.observe_prefill(100, 20, 60, 1, 1 / 64, counts, ctypes.byref(steps))
    assert list(counts) == [100, 20, 60, 1] and steps.value == 1
    native_ema.observe_prefill(100, 0, 0, 0, 1 / 64, counts, ctypes.byref(steps))
    assert list(counts) == pytest.approx([100, 20 * 63 / 64, 60 * 63 / 64, 63 / 64])
    assert counts[0] - sum(counts[1:]) == pytest.approx(19 * 63 / 64 + 100 / 64)
    old = list(counts)
    native_ema.observe_prefill(0, 0, 0, 0, 1 / 64, counts, ctypes.byref(steps))
    assert list(counts) == old and steps.value == 2


def test_round_reference_and_cost_survive_decay_and_request_reuse_cannot_inherit_them():
    coordinator, physical = ready_coordinator(requests=2)
    token = coordinator.reserve_kvduo_prefill(3, [0, 1])
    key = coordinator._kvduo_owner_key((0, 0))
    assert coordinator._kvduo_prefill_budget_reference[key] == 19
    assert coordinator._kvduo_prefill_budget_cost[key] == 1
    coordinator.kvduo_prefill_hit_ema[0, 0] = torch.tensor([100, 20, 10, 0])
    coordinator.kvduo_prefill_hit_steps[0, 0] = 1000
    coordinator._prune_kvduo_prefill_budget()
    assert coordinator._kvduo_prefill_budget_cost[key] == 1
    coordinator._reset_kvduo_hot_tail_ema(0)
    assert coordinator._kvduo_owner_key((0, 0)) != key
    assert not coordinator._kvduo_prefill_budget_reference
    assert not coordinator._kvduo_prefill_budget_cost
    # Finishing an old token must not clear a newer generation's growth hold.
    new_key = coordinator._kvduo_owner_key((0, 0))
    coordinator._kvduo_prefill_growth_blocked.add((0, 0))
    coordinator._kvduo_prefill_block_generation[(0, 0)] = new_key[2]
    coordinator.finish_kvduo_prefill_reservation(token, cancelled=True)
    assert (0, 0) in coordinator._kvduo_prefill_growth_blocked


def test_growth_limited_objects_cannot_be_merged_again():
    coordinator, physical = ready_coordinator()
    coordinator.reserve_kvduo_prefill(3, [0])
    coordinator.kvduo_hot_tail_observation_steps.fill_(1000)
    coordinator.kvduo_hot_tail_ema.zero_()
    assert coordinator.reclaim_kvduo_hot_pages(2) == 0
    assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 16


def test_mandatory_minimum_is_reserved_before_admitting_prefill():
    coordinator, physical = ready_coordinator(requests=2)
    coordinator.kvduo_req_hot_capacity[0, 1] = 0
    coordinator._kvduo_pending_hot_minimum.add(1)
    # Available three extra pages cannot meet prefill's three plus the minimum.
    assert coordinator.reserve_kvduo_prefill(3, [0, 1]) is None
    assert physical.free_pages == 0


def test_zero_budget_allows_only_zero_extra_and_equal_actual_base_reference():
    coordinator, physical = ready_coordinator()
    coordinator.kvduo_prefill_policy = KVDuoPrefillPolicy(0)
    coordinator.kvduo_prefill_hit_ema[0, 0, 3] = 0
    assert coordinator.reserve_kvduo_prefill(1, [0]) is not None
    assert sum(coordinator._kvduo_prefill_budget_cost.values()) == 0


def test_second_admission_cannot_renew_budget_from_later_misses_or_new_arrivals():
    coordinator, physical = make_merge_coordinator({(0, req): 8 for req in range(3)})
    for req in range(3):
        coordinator._merge_kvduo_hot_pages(0, req)
        coordinator.kvduo_prefill_hit_ema[0, req] = torch.tensor([100, 20, 60, 0])
        coordinator.kvduo_prefill_hit_steps[0, req] = 65
    coordinator._ensure_kvduo_hot_capacity_targets_batch(
        {(0, req): 28 for req in range(3)}
    )
    coordinator.kvduo_prefill_hit_ema[0, 0] = torch.tensor([100, 20, 69, 1])
    coordinator.kvduo_prefill_hit_steps[0, 0] = 65
    coordinator.kvduo_prefill_hit_ema[0, 1] = torch.tensor([100, 20, 80, 0])
    coordinator.kvduo_prefill_hit_steps[0, 1] = 64
    physical.free_pages = 0
    token = coordinator.reserve_kvduo_prefill(3, [0, 1])
    assert token is not None
    assert sum(coordinator._kvduo_prefill_budget_reference.values()) == 10
    coordinator.kvduo_prefill_hit_ema[0, 1] = torch.tensor([100, 20, 69, 1])
    coordinator.kvduo_prefill_hit_steps[0, 1] = 65
    coordinator.kvduo_prefill_hit_ema[0, 2] = torch.tensor([100, 0, 80, 0])
    coordinator.kvduo_prefill_hit_steps[0, 2] = 65
    assert coordinator.reserve_kvduo_prefill(1, [0, 1, 2]) is None
    assert physical.free_pages == 3  # no partial release of otherwise qualified owners
    assert int(coordinator.kvduo_req_hot_capacity[0, 1]) == 28
    coordinator.finish_kvduo_prefill_reservation(token, cancelled=True)
    assert not coordinator._kvduo_prefill_budget_reference
    assert not coordinator._kvduo_prefill_budget_cost


def test_reference_request_end_removes_budget_without_undoing_existing_rollback():
    coordinator, physical = ready_coordinator(requests=2)
    coordinator.kvduo_prefill_hit_ema[0, 1] = torch.tensor([100, 0, 0, 0])
    token = coordinator.reserve_kvduo_prefill(3, [0, 1])
    assert sum(coordinator._kvduo_prefill_budget_reference.values()) == 119
    coordinator._reset_kvduo_hot_tail_ema(1)
    assert sum(coordinator._kvduo_prefill_budget_reference.values()) == 19
    assert sum(coordinator._kvduo_prefill_budget_cost.values()) == 1
    assert (0, 0) in coordinator._kvduo_prefill_growth_blocked
    assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 16


def test_failed_first_growth_restores_reference_phase_and_cache_metadata(monkeypatch):
    coordinator, physical = make_merge_coordinator({(0, 0): 8})
    coordinator._merge_kvduo_hot_pages(0, 0)
    coordinator.kvduo_prefill_hit_ema[0, 0] = torch.tensor([100, 20, 60, 0])
    coordinator.kvduo_prefill_hit_steps[0, 0] = 65
    old_counts = coordinator.kvduo_prefill_hit_ema.clone()
    old_lru = coordinator.lru_slots.clone()
    old_free = physical.free_pages
    observe = coordinator._observe_kvduo_hot_growth

    def fail_after_freeze(*args):
        observe(*args)
        raise RuntimeError("publication failed")

    monkeypatch.setattr(coordinator, "_observe_kvduo_hot_growth", fail_after_freeze)
    with pytest.raises(RuntimeError):
        coordinator._ensure_kvduo_hot_capacity_targets_batch({(0, 0): 20})
    assert physical.free_pages == old_free
    assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 16
    assert torch.equal(coordinator.kvduo_prefill_hit_ema, old_counts)
    assert torch.equal(coordinator.lru_slots, old_lru)
    assert int(coordinator.kvduo_prefill_hit_steps[0, 0]) == 65
    assert int(coordinator.kvduo_prefill_reference_steps[0, 0]) == 0
    assert not coordinator._kvduo_prefill_reference_frozen


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
