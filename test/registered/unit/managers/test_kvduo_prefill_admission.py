"""Physical prefill admission, reservations, and independent-pool budgets."""

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
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_hot_merge import make_merge_coordinator

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def ready_coordinator(extra=3, requests=1):
    coordinator, physical = make_merge_coordinator(
        {(0, 0): 8, **{(0, req): 1 for req in range(1, requests)}}
    )
    coordinator.kvduo_hot_buffers[0, 0].frequency.copy_(
        torch.tensor([-1, 0], dtype=torch.int32)
    )
    coordinator._merge_kvduo_hot_pages(0, 0)
    coordinator._ensure_kvduo_hot_capacity_targets_batch({(0, 0): (4 + extra) * 4})
    physical.free_pages = 0
    return coordinator, physical


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
    c, _ = ready_coordinator()
    c.compress_ratio = ratio
    assert c.kvduo_prefill_pages(prefix, end) == expected


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


@pytest.mark.parametrize("pool", ["logical", "swa"])
def test_physical_blocked_chunk_does_not_reserve_or_protect_a_plan(pool):
    coordinator, physical = ready_coordinator()
    adder, req = adder_and_req(
        coordinator,
        dsv4=True,
        chunk=16,
        logical_size=8 if pool == "logical" else 1024,
        swa_size=16 if pool == "swa" else 1024,
    )
    req.prefix_indices = torch.arange(8, dtype=torch.int64)
    req.full_untruncated_fill_ids = list(range(40))
    reserve = MagicMock(wraps=coordinator.reserve_kvduo_prefill)
    coordinator.reserve_kvduo_prefill = reserve

    assert adder.add_chunked_req(req) is req

    reserve.assert_not_called()
    assert coordinator._kvduo_prefill_plan_uncovered_pages() == 0
    assert physical.free_pages == 0 and not adder.can_run_list


@pytest.mark.parametrize("path", ["new", "chunk"])
@pytest.mark.parametrize("pool", ["logical", "swa"])
def test_physical_admission_preserves_logical_and_swa_budget_rejections(path, pool):
    coordinator, physical = ready_coordinator()
    physical.free_pages = 32
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
    assert coordinator._kvduo_prefill_plan_uncovered_pages() == 0
    assert physical.free_pages == 32
    assert not adder.can_run_list and not hasattr(req, "extend_range")


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


def test_logical_budget_is_rechecked_after_prefix_lock_before_reservation():
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
def test_kvduo_keeps_logical_headroom_for_future_decode_before_reservation(path):
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
    physical.free_pages = 6
    adder, first = actual_budget_adder_and_req(coordinator)
    _, second = actual_budget_adder_and_req(coordinator)
    second.full_untruncated_fill_ids = list(range(16))
    assert adder.add_one_req(first, False, None) is AddReqResult.CONTINUE
    assert adder.add_one_req(second, False, None) is AddReqResult.CONTINUE
    assert adder.can_run_list == [first, second]
    assert adder.rem_total_tokens == adder.cur_rem_tokens == 984
    assert adder.rem_swa_token_offset == 40
    assert coordinator._kvduo_reserved_prefill_pages() == 3
    assert physical.free_pages == 6
    assert coordinator._kvduo_mandatory_hot_pages() == 0
    assert len(adder.kvduo_prefill_reservations) == 2
    # A third prefill cannot consume the scheduler headroom and reservations.
    _, third = actual_budget_adder_and_req(coordinator)
    third.full_untruncated_fill_ids = list(range(16))
    assert adder.add_one_req(third, False, None) is AddReqResult.OTHER
    assert adder.can_run_list == [first, second]
    assert coordinator._kvduo_reserved_prefill_pages() == 3


def test_prefill_waits_without_mutating_chunk_or_loading_host_when_no_complete_plan():
    coordinator, physical = ready_coordinator()
    adder, req = adder_and_req(coordinator, host_hit=4)
    assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    assert not adder.can_run_list and not hasattr(req, "extend_range")
    adder.tree_cache.init_load_back.assert_not_called()
    assert physical.free_pages == 0
    assert adder.add_chunked_req(req) is req
    assert not adder.can_run_list and not hasattr(req, "extend_range")


def test_host_restore_and_chunk_share_exact_reservation_and_cancellation_releases_hold():
    coordinator, physical = ready_coordinator(extra=4)
    physical.free_pages = 7
    adder, req = adder_and_req(coordinator, host_hit=4)
    req.full_untruncated_fill_ids = list(range(10))
    req.sampling_params.max_new_tokens = 2

    def restore(_):
        assert coordinator._kvduo_reserved_prefill_pages() == 5
        physical.free_pages -= 2
        return torch.tensor([2, 3, 4, 5]), object()

    adder.tree_cache.init_load_back.side_effect = restore
    assert adder.add_one_req(req, False, None) is AddReqResult.CONTINUE
    assert req.extend_range == Range(6, 10)
    assert coordinator._kvduo_reserved_prefill_pages() == 3
    assert physical.free_pages == 5
    assert coordinator._kvduo_mandatory_hot_pages() == 0
    token = adder.kvduo_prefill_reservations[0]
    coordinator.finish_kvduo_prefill_reservation(token, cancelled=False)
    assert coordinator._kvduo_reserved_prefill_pages() == 1
    coordinator, physical = ready_coordinator(extra=4)
    physical.free_pages = 7
    adder, req = adder_and_req(coordinator, host_hit=4)
    req.full_untruncated_fill_ids = list(range(10))
    req.sampling_params.max_new_tokens = 2
    adder.tree_cache.init_load_back.return_value = (None, None)
    assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    assert not coordinator._kvduo_prefill_reservations


@pytest.mark.parametrize("path", ["new", "chunk"])
@pytest.mark.parametrize("dsv4", [False, True])
def test_physical_shortage_waits_without_reclaiming_hot_or_active_full(path, dsv4):
    c, physical = ready_coordinator(extra=4)
    adder, req = adder_and_req(c, dsv4=dsv4, chunk=16 if path == "chunk" else None)
    c.reclaim_kvduo_hot_pages = MagicMock(
        side_effect=AssertionError("prefill HOT reclaim")
    )
    c.reclaim_kvduo_full_pages = MagicMock(
        side_effect=AssertionError("prefill FULL reclaim")
    )
    banks = (
        c.req_device_buffer_tokens,
        c.req_device_buffer_token_locs,
        c.lru_slots,
        c.kvduo_req_hot_capacity,
    )
    before = [bank.clone() for bank in banks]
    if path == "new":
        assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    else:
        assert adder.add_chunked_req(req) is req
    assert not adder.can_run_list
    assert not c._kvduo_prefill_reservations
    assert not c._kvduo_prefill_host_reservations
    assert physical.free_pages == 0
    assert all(torch.equal(bank, old) for bank, old in zip(banks, before))
    c.reclaim_kvduo_hot_pages.assert_not_called()
    c.reclaim_kvduo_full_pages.assert_not_called()


def test_admission_retries_real_surplus_and_does_not_hold_hot_growth_after_cancel():
    c, physical = ready_coordinator(extra=3)
    assert c.reserve_kvduo_prefill(2) is None
    physical.free_pages = 2
    token = c.reserve_kvduo_prefill(2)
    assert token is not None
    assert c._kvduo_reserved_prefill_pages() == 2
    c.kvduo_hot_growth_events[0, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    # Hand may now shrink an applicant, but cannot consume the admitted hold.
    capacity = int(c.kvduo_req_hot_capacity[0, 0])
    assert capacity == c.min_hot_pages * c.hot_page_size
    assert physical.free_pages >= c._kvduo_reserved_prefill_pages() == 2
    free_before_growth = physical.free_pages
    c.finish_kvduo_prefill_reservation(token, cancelled=True)
    assert c._kvduo_reserved_prefill_pages() == 0
    c.kvduo_hot_growth_events[0, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c.kvduo_req_hot_capacity[0, 0] == capacity + c.hot_page_size
    assert physical.free_pages == free_before_growth - 1


@pytest.mark.parametrize("path", ["new", "chunk"])
def test_host_rejected_prefill_does_not_protect_physical_pages(path):
    c, physical = ready_coordinator()
    physical.free_pages = 32
    adder, req = adder_and_req(c, chunk=16 if path == "chunk" else None)
    c._reserve_kvduo_prefill_host = MagicMock(return_value=False)
    if path == "chunk":
        assert adder.add_chunked_req(req) is req
    else:
        adder.add_one_req(req, False, None)
    assert c._reserve_kvduo_prefill_host.called
    assert c._kvduo_optional_protected_pages() == 0
    assert not adder.can_run_list


@pytest.mark.parametrize("dsv4", [False, True])
def test_new_prefill_uses_default_physical_headroom_and_retries_after_reclaim(dsv4):
    c, physical = ready_coordinator()
    adder, req = adder_and_req(c, dsv4=dsv4, chunk=8 if dsv4 else 2)
    if dsv4:
        req.prefix_indices = torch.arange(8, dtype=torch.int64)
        req.full_untruncated_fill_ids = list(range(32))
    # Current chunk fits, but the ordinary whole-input + page budget does not.
    physical.free_pages = 1
    c.reserve_kvduo_prefill = MagicMock(wraps=c.reserve_kvduo_prefill)
    total = (
        len(req.full_untruncated_fill_ids) - len(req.prefix_indices) + adder.page_size
    )
    assert adder.rem_total_tokens > total
    assert physical.free_pages > 0
    assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    c.reserve_kvduo_prefill.assert_not_called()
    assert not adder.can_run_list and not hasattr(req, "extend_range")
    assert not c._kvduo_prefill_host_reservations
    # Decode can release space; a later pass does not require a request to finish.
    physical.free_pages = 16
    assert adder.add_one_req(req, False, None) is AddReqResult.CONTINUE
    assert adder.can_run_list == [req]


def test_chunk_keeps_running_decode_headroom_in_physical_pool():
    c, physical = ready_coordinator()
    physical.free_pages = 1
    adder, req = actual_budget_adder_and_req(c, path="chunk", running_max_new=8)
    c.reserve_kvduo_prefill = MagicMock(wraps=c.reserve_kvduo_prefill)
    assert adder.rem_total_tokens == 1016
    assert adder._kvduo_scheduler_physical_budget() == 0
    assert adder.add_chunked_req(req) is req
    c.reserve_kvduo_prefill.assert_not_called()
    assert not hasattr(req, "extend_range")
    physical.free_pages = 3
    assert adder.add_chunked_req(req) is req
    assert req.extend_range == Range(8, 24)


@pytest.mark.parametrize("pool", ["physical", "host"])
def test_impossible_prompt_is_rejected_before_physical_headroom_wait(pool):
    c, physical = ready_coordinator()
    adder, req = adder_and_req(c)
    req.set_finish_with_abort = MagicMock()
    if pool == "physical":
        physical.size = 4
    else:
        c.mem_pool_host.size = 4
    assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    req.set_finish_with_abort.assert_called_once()
    assert not c._kvduo_prefill_reservations


@pytest.mark.parametrize("free_pages,admitted", [(4, False), (5, False), (6, True)])
def test_kvduo_matches_default_swa_admission_with_same_physical_inventory(
    free_pages, admitted
):
    c, physical = ready_coordinator()
    physical.free_pages = free_pages
    kvduo, req = actual_budget_adder_and_req(c, running_max_new=8)
    ordinary, ordinary_req = actual_budget_adder_and_req(c, running_max_new=8)
    ordinary.kvduo_coordinator = None
    # The ordinary SWA composite has full storage matching physical capacity;
    # KVDuo retains a larger logical inventory behind its physical composite.
    ordinary.token_to_kv_pool_allocator = (
        ordinary.token_to_kv_pool_allocator.logical_attn_allocator
    )
    ordinary.token_to_kv_pool_allocator.full_attn_allocator = (
        PagedTokenToKVPoolAllocator(free_pages * 8, 8, torch.int64, "cpu", None, False)
    )
    req.sampling_params.max_new_tokens = 8
    ordinary_req.sampling_params.max_new_tokens = 8
    assert kvduo._kvduo_scheduler_physical_budget() == ordinary.rem_total_tokens
    kvduo.add_one_req(req, False, None)
    ordinary.add_one_req(ordinary_req, False, None)
    assert bool(kvduo.can_run_list) == bool(ordinary.can_run_list) == admitted


def test_physical_budget_is_rechecked_after_prefix_lock():
    c, physical = ready_coordinator()
    physical.free_pages = 5
    adder, req = actual_budget_adder_and_req(c)
    c.reserve_kvduo_prefill = MagicMock()

    def lock(_):
        physical.free_pages = 1

    adder.tree_cache.inc_lock_ref.side_effect = lock
    assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    adder.tree_cache.inc_lock_ref.assert_called_once()
    c.reserve_kvduo_prefill.assert_not_called()
    assert not adder.can_run_list


@pytest.mark.parametrize("local_free_pages", [1, 16])
@pytest.mark.parametrize("path", ["new", "chunk"])
def test_physical_gate_uses_tp_minimum_before_reservation(
    monkeypatch, local_free_pages, path
):
    c, physical = ready_coordinator()
    physical.free_pages = local_free_pages
    adder, req = actual_budget_adder_and_req(c, path=path, running_max_new=8)
    c.tp_world_size = 2
    c.tp_group = object()
    c.reserve_kvduo_prefill = MagicMock()
    c.preview_kvduo_prefill_reclaim = MagicMock()

    def reduce(value, *, op, group):
        assert op == torch.distributed.ReduceOp.MIN and group is c.tp_group
        value.fill_(8)

    collective = MagicMock(side_effect=reduce)
    monkeypatch.setattr(torch.distributed, "all_reduce", collective)
    if path == "new":
        assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    else:
        assert adder.add_chunked_req(req) is req
    collective.assert_called_once()
    c.reserve_kvduo_prefill.assert_not_called()
    assert not adder.can_run_list


@pytest.mark.parametrize("path", ["new", "chunk"])
def test_physical_block_records_prefill_hint_without_admission_or_host_pin(path):
    c, physical = ready_coordinator()
    adder, req = actual_budget_adder_and_req(c, path=path, running_max_new=8)
    if path == "new":
        assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    else:
        assert adder.add_chunked_req(req) is req
    assert c._kvduo_full_prefill_hint is not None
    assert not c._kvduo_prefill_host_reservations
    assert not c._kvduo_prefill_reservations
    assert not adder.can_run_list
    assert physical.free_pages == 0


@pytest.mark.parametrize("gate", ["history", "host"])
def test_prefill_hint_requires_independent_capacity(gate):
    c, _ = ready_coordinator()
    adder, req = adder_and_req(c)
    if gate == "history":
        c._kvduo_prefill_history_ready = MagicMock(return_value=False)
    else:
        c._reserve_kvduo_prefill_host = MagicMock(return_value=False)
    c.preview_kvduo_prefill_reclaim(req, 4, 16, adder.tree_cache)
    assert c._kvduo_full_reclaim_prefill_plan() is None
    assert not c._kvduo_prefill_host_reservations


def test_blocked_prefill_preview_does_not_synchronize_budgets():
    c, _ = ready_coordinator()
    adder, req = adder_and_req(c)
    c._kvduo_prefill_history_ready = MagicMock(
        side_effect=AssertionError("early history check")
    )
    c._reserve_kvduo_prefill_host = MagicMock(
        side_effect=AssertionError("early Host check")
    )
    c.preview_kvduo_prefill_reclaim(req, 4, 16, adder.tree_cache)
    assert c._kvduo_full_prefill_hint is not None
    c._kvduo_prefill_history_ready.assert_not_called()
    c._reserve_kvduo_prefill_host.assert_not_called()
