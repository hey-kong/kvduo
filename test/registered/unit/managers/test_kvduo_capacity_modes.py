"""Exercise space hysteresis, complete-prefill conversion and rollback obligations."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.mem_cache.sparsity.core.kvduo_capacity_modes import (
    KVDuoRecoveryObject,
    plan_kvduo_recovery,
)
from test_kvduo_hot_growth import make_host_pool
from test_kvduo_prefill_rollback import (
    adder_and_req,
    merged_and_expanded,
    ready_coordinator,
)
from test_kvduo_pressure_admission import make_full_coordinator


def recovery(layer, order, ready=True, req=0):
    return KVDuoRecoveryObject((layer, req, 0), order, ready)


def test_fifo_recovery_reserves_one_page_per_object_and_skips_unready_objects():
    objects = [recovery(0, 1), recovery(1, 2)]
    assert plan_kvduo_recovery(objects, 4, 4) == ()
    assert plan_kvduo_recovery(objects, 5, 4) == ((0, 0, 0),)
    assert plan_kvduo_recovery(objects, 6, 4) == ((0, 0, 0), (1, 0, 0))
    objects[0] = recovery(0, 1, False)
    assert plan_kvduo_recovery(objects, 5, 4) == ((1, 0, 0),)
    assert plan_kvduo_recovery([recovery(1, 1), recovery(0, 1, req=1)], 1, 0) == (
        (1, 0, 0),
    )


def fixed_stack(layers=4):
    owners = {(layer, 0): 3 for layer in range(layers)}
    c, p = merged_and_expanded(owners, owners, layers=layers)
    for layer in range(layers):
        c._kvduo_hot_processed[layer, 0] = layers - layer
    token = c.reserve_kvduo_prefill(layers, [0])
    assert token is not None
    c.consume_kvduo_prefill_reservation(token, layers)
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    p.free_pages = 0
    return c, p


@pytest.mark.parametrize("dsv4", [False, True])
def test_host_prefix_chunk_wait_keeps_recovery_credit_out_of_prefill_space(dsv4):
    c, p = fixed_stack(layers=1)
    adder, req = adder_and_req(c, dsv4=dsv4)
    page = adder.page_size
    adder.rem_chunk_tokens = page
    req.rid = "waiting-prefill"
    req.extra_key = None
    req.prefix_indices = torch.arange(page, dtype=torch.int64)
    req.host_hit_length = page
    req.full_untruncated_fill_ids = list(range(4 * page))
    c.mem_pool_host = make_host_pool(0, c.page_size)
    p.free_pages = 2
    before = c.kvduo_req_hot_capacity.clone()
    costs = dict(c._kvduo_prefill_budget_cost)

    assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    assert c._kvduo_optional_protected_pages() == 2  # Restore and current chunk.

    c.kvduo_hot_growth_events[0, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))

    assert c._kvduo_object_limited((0, 0))
    assert torch.equal(c.kvduo_req_hot_capacity, before)
    assert c._kvduo_prefill_budget_cost == costs
    assert p.free_pages == 2 and c._kvduo_recovery_credits == 0

    # Recovery gets one page only after the complete prefill demand is covered.
    p.free_pages = 4
    c.kvduo_hot_growth_events[0, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert not c._kvduo_object_limited((0, 0))
    assert int(c.kvduo_req_hot_capacity[0, 0]) == int(before[0, 0]) + c.hot_page_size
    assert p.free_pages == 3 and not c._kvduo_prefill_budget_cost


@pytest.mark.parametrize("dsv4", [False, True])
def test_waiting_host_prefix_chunk_defers_optional_growth_without_merge(dsv4):
    c, p = ready_coordinator(extra=3)
    adder, req = adder_and_req(c, dsv4=dsv4)
    page = adder.page_size
    adder.rem_chunk_tokens = page
    req.rid = "waiting-prefill"
    req.extra_key = None
    req.prefix_indices = torch.arange(page, dtype=torch.int64)
    req.host_hit_length = page
    req.full_untruncated_fill_ids = list(range(4 * page))
    c.mem_pool_host = make_host_pool(0, c.page_size)
    p.free_pages = 1
    before = c.kvduo_req_hot_capacity.clone()
    c.reclaim_kvduo_hot_pages = MagicMock(wraps=c.reclaim_kvduo_hot_pages)

    assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    assert c._kvduo_optional_protected_pages() == 2
    c.kvduo_hot_growth_events[0, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))

    assert torch.equal(c.kvduo_req_hot_capacity, before)
    assert p.free_pages == 1
    assert not c._kvduo_prefill_limited_objects
    c.reclaim_kvduo_hot_pages.assert_not_called()


def test_one_page_headroom_recovers_only_one_layer_without_replaying_old_events():
    c, p = fixed_stack()
    key = c._kvduo_owner_key((0, 0))
    c._kvduo_next_prefill_pages = 4
    p.free_pages = 4
    c.kvduo_hot_growth_events[:, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert len(c._kvduo_prefill_limited_objects) == 4
    assert c.kvduo_req_hot_capacity[:, 0].tolist() == [8] * 4
    assert not c.kvduo_hot_growth_events.any()
    p.free_pages = 5
    c.kvduo_hot_growth_events[0, 0] = 1
    c.kvduo_capacity_ema[:, 0, :2] = torch.tensor([8, 4])
    c.kvduo_capacity_ema[:, 0, 8] = 2
    c.kvduo_capacity_steps[:, 0] = 100
    c.kvduo_prefill_ema[:, 0] = 7
    c.kvduo_prefill_steps[:, 0] = 66
    costs = dict(c._kvduo_prefill_budget_cost)
    allocate = p.alloc.side_effect

    def redeem(slots):
        assert c._kvduo_recovery_credits == 1
        assert c._kvduo_object_limited((0, 0))
        assert c._kvduo_prefill_budget_cost == costs
        return allocate(slots)

    p.alloc.side_effect = redeem
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert not c._kvduo_object_limited((0, 0))
    assert c.kvduo_req_hot_capacity[:, 0].tolist() == [12, 8, 8, 8]
    assert c.kvduo_hot_page_limit[:, 0].tolist() == [8, 2, 2, 2]
    assert c.kvduo_hot_base_capacity[:, 0].tolist() == [8] * 4
    assert p.free_pages == 4 and c._kvduo_recovery_credits == 0
    assert key not in c._kvduo_prefill_budget_cost
    assert len(c._kvduo_prefill_budget_cost) == 3
    assert c.kvduo_capacity_steps[:, 0].tolist() == [100] * 4
    assert c.kvduo_capacity_ema[:, 0, :3].tolist() == [[8, 4, 4]] + [[8, 4, 0]] * 3
    assert c.kvduo_capacity_ema[:, 0, 8].tolist() == [2] * 4
    assert c.kvduo_prefill_steps[:, 0].tolist() == [0, 66, 66, 66]
    assert c.kvduo_prefill_ema[:, 0].tolist() == [[0, 0]] + [[7, 7]] * 3
    p.alloc.side_effect = allocate
    p.free_pages = 5
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert not c._kvduo_object_limited((1, 0))
    assert c.kvduo_req_hot_capacity[:, 0].tolist() == [12, 8, 8, 8]
    assert c._kvduo_recovery_credits == 0
    assert len(c._kvduo_prefill_budget_cost) == 2
    assert c.kvduo_prefill_steps[1, 0].item() == 66


def test_recovery_failure_keeps_each_selected_object_fixed_and_cost_reserved():
    c, p = fixed_stack(layers=2)
    p.free_pages = 2
    p.alloc.side_effect = lambda _: None
    c.kvduo_hot_growth_events[:, 0] = 1
    with pytest.raises(RuntimeError, match="reserved credits"):
        c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert len(c._kvduo_prefill_limited_objects) == 2
    assert c.kvduo_hot_page_limit[:, 0].tolist() == [2, 2]
    assert len(c._kvduo_prefill_budget_cost) == 2
    assert c._kvduo_recovery_credits == 0


def test_recovery_skips_only_protected_object_and_preserves_its_budget_cost():
    c, p = fixed_stack(layers=2)
    c._kvduo_hot_pressure_protected.add((0, 0, 0))
    p.free_pages = 1
    c.kvduo_hot_growth_events[:, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c._kvduo_object_limited((0, 0))
    assert not c._kvduo_object_limited((1, 0))
    assert c.kvduo_req_hot_capacity[:, 0].tolist() == [8, 12]
    assert c._kvduo_prefill_budget_cost == {(0, 0, 0): 1}
    assert list(c._kvduo_fixed_queue) == [(0, 0, 0)]
    assert not c.kvduo_hot_growth_events.any()


def test_recovery_waits_only_for_its_own_associated_prefill():
    c, p = fixed_stack(layers=2)
    first = c._kvduo_prefill_key(SimpleNamespace(rid="first"))
    second = c._kvduo_prefill_key(SimpleNamespace(rid="second"))
    c._kvduo_prefill_associations[(0, 0, 0)] = {first}
    c._kvduo_prefill_associations[(1, 0, 0)] = {second}
    c._kvduo_prefill_lifecycle[second][1] = "ready"
    p.free_pages = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c._kvduo_object_limited((0, 0))
    assert not c._kvduo_object_limited((1, 0))
    assert p.free_pages == 1 and c.kvduo_req_hot_capacity[:, 0].tolist() == [8, 8]
    assert c._kvduo_prefill_associations[(0, 0, 0)] == {first}


def test_recovery_revalidates_protection_after_preparing_selection():
    c, p = fixed_stack(layers=2)
    p.free_pages = 2
    c.kvduo_hot_growth_events[:, 0] = 1
    before = c.req_device_buffer_tokens.clone()

    def agree(state):
        if isinstance(state, tuple) and len(state) == 2 and state[0]:
            c._kvduo_hot_pressure_protected.add((1, 0, 0))
        return True

    c._kvduo_consensus = agree
    c._recover_kvduo_objects({(0, 0): 1, (1, 0): 1})
    assert len(c._kvduo_prefill_limited_objects) == 2
    assert len(c._kvduo_prefill_budget_cost) == 2
    assert torch.equal(c.req_device_buffer_tokens, before)
    assert c._kvduo_recovery_credits == 0 and p.free_pages == 2


def test_full_prefill_waits_for_all_grouped_prefills():
    c, p = fixed_stack(layers=2)
    key = c._kvduo_owner_key((0, 0))
    first, second = SimpleNamespace(rid="a"), SimpleNamespace(rid="b")
    first_key = c._kvduo_prefill_key(first)
    second_key = c._kvduo_prefill_key(second)
    c._kvduo_prefill_associations[key] = {first_key, second_key}
    p.free_pages = 20
    c.kvduo_complete_prefill(first)
    assert c._kvduo_prefill_lifecycle[first_key][1] == "ready"
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c._kvduo_object_limited((0, 0))
    c._kvduo_prefill_lifecycle[first_key][1] = "ready"
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c._kvduo_object_limited((0, 0))
    c._kvduo_prefill_lifecycle[second_key][1] = "cancelled"
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert not c._kvduo_object_limited((0, 0))
    assert c.kvduo_req_hot_capacity[:, 0].tolist() == [8, 8]


@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("locked", [False, True])
def test_prefill_completion_and_schedule_boundary_keep_full_residency(ratio, locked):
    c, p, req = make_full_coordinator(pages=8, initial_pages=8, ratio=ratio)
    req.rid = "new"
    req.decode_batch_idx = 0
    if locked:
        c._kvduo_prefill_limited_objects.add((0, 0, 99))
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    before = mapping.clone()
    c.kvduo_complete_prefill(req)
    c.begin_kvduo_schedule_boundary()
    assert c.reclaim_kvduo_full_pages(c.page_size, reserve_hot_minimum=True) == 0
    assert torch.equal(mapping, before)
    assert p.free_pages == 0 and not c._mixed_slots[0]
    req.decode_batch_idx = 1
    assert c.reclaim_kvduo_full_pages(c.page_size, reserve_hot_minimum=True) > 0
    assert c._mixed_slots[0]


def test_prefill_reserves_full_pages_without_future_mixed_minimum():
    c, p = ready_coordinator(extra=4)
    req = SimpleNamespace(rid="new", prefix_indices=torch.empty(0, dtype=torch.int64))
    # A complete four-page rollback fits four FULL compute pages.
    token = c.reserve_kvduo_prefill(4, [0], prefill_req=req)
    assert token is not None and c._kvduo_mandatory_hot_pages() == 0
    assert c._kvduo_optional_protected_pages() == 4
    c.finish_kvduo_prefill_reservation(token, cancelled=True)
    assert c._kvduo_optional_protected_pages() == 0
    assert not c._kvduo_object_limited((0, 0))


def test_plan_protection_does_not_double_count_reservation():
    c, p = ready_coordinator(extra=4)
    req = SimpleNamespace(rid="new", prefix_indices=torch.empty(0, dtype=torch.int64))
    c.protect_kvduo_prefill_plan(req, 6)  # three compute pages
    token = c.reserve_kvduo_prefill(3, [0], prefill_req=req)
    assert token is not None
    assert c._kvduo_optional_protected_pages() == 3
    c.bind_kvduo_prefill_batch([req], [token])
    c.consume_kvduo_prefill_reservation(token, 3)
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    assert c._kvduo_optional_protected_pages() == 0


def test_tp_disagreement_never_rolls_back_or_unlocks_local_request():
    c, p = fixed_stack(layers=2)
    p.free_pages = 20
    c._kvduo_consensus = MagicMock(return_value=False)
    c.kvduo_hot_growth_events[:, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c._kvduo_object_limited((0, 0))
    assert c.kvduo_req_hot_capacity[:, 0].tolist() == [8, 8]


def test_slot_reuse_clears_limits_and_queue_without_restarting_scan():
    c, p = fixed_stack(layers=2)
    c._kvduo_hand = 1
    c._kvduo_hand_sequence = 10
    c._release_kvduo_layer_hot_pages(0)
    assert c._kvduo_request_generation[0] == 1
    assert c.kvduo_hot_page_limit[:, 0].tolist() == [8, 8]
    assert not c._kvduo_fixed_queue and not c._kvduo_prefill_budget_cost
    assert not c._kvduo_prefill_limited_objects
    assert not c._kvduo_prefill_associations and not c._kvduo_prefill_block_leases
    assert c._kvduo_hand == 1 and c._kvduo_hand_sequence == 10


def _tp_modes_worker(rank, init_file):
    import torch.distributed as dist
    from datetime import timedelta

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        owners = {(layer, 0): 3 for layer in range(2)}
        c, p = merged_and_expanded(owners, owners, layers=2)
        c.tp_world_size = 2
        c.tp_group = dist.group.WORLD
        # Radix reclamation must also use the common shortfall; a rank with
        # enough local pages cannot skip the other ranks' reclaim collectives.
        p.free_pages = 2 * rank
        evictions = []

        def evict(params):
            evictions.append(params.num_tokens)
            p.free_pages += 2
            return SimpleNamespace(num_tokens_evicted=4)

        cache = SimpleNamespace(evictable_size=lambda: 4, evict=evict)
        c._reclaim_kvduo_radix_pages(2, cache)
        assert evictions == [4]
        p.free_pages = rank
        token = c.reserve_kvduo_prefill(2, [0])
        assert token is not None and c._kvduo_object_limited((0, 0))
        assert c.kvduo_req_hot_capacity[:, 0].tolist() == [8, 8]
        c.consume_kvduo_prefill_reservation(token, 2)
        c.finish_kvduo_prefill_reservation(token, cancelled=False)
        # No applicant means no ordinary merge, even if only one TP rank
        # locally has enough pages to cover the waiting prefill protection.
        c._kvduo_next_prefill_pages = 5
        p.free_pages = 6 * rank
        reclaim = MagicMock(wraps=c.reclaim_kvduo_hot_pages)
        c.reclaim_kvduo_hot_pages = reclaim
        c.prepare_kvduo_graph_replay(torch.tensor([0]))
        reclaim.assert_not_called()
        assert c._kvduo_object_limited((0, 0))
        assert c.kvduo_req_hot_capacity[:, 0].tolist() == [8, 8]
        c._kvduo_next_prefill_pages = 1
        p.free_pages = 2 + rank
        c.kvduo_hot_growth_events[:, 0] = 1
        c.prepare_kvduo_graph_replay(torch.tensor([0]))
        assert c._kvduo_object_limited((0, 0))
        assert not c._kvduo_object_limited((1, 0))
        assert c.kvduo_req_hot_capacity[:, 0].tolist() == [8, 12]
        assert c._kvduo_recovery_credits == 0
        p.free_pages = 3 + rank
        c.kvduo_hot_growth_events[0, 0] = 1
        c.prepare_kvduo_graph_replay(torch.tensor([0]))
        assert not c._kvduo_object_limited((0, 0))
        assert c.kvduo_req_hot_capacity[:, 0].tolist() == [12, 12]
        assert c._kvduo_recovery_credits == 0

        from test_kvduo_hot_merge import prefill_protected_growth_coordinator

        c, p = prefill_protected_growth_coordinator(free_pages=4 + 2 * rank)
        c.tp_world_size = 2
        c.tp_group = dist.group.WORLD
        c.kvduo_hot_growth_events[0, 1] = 1
        # Rank1 locally covers protection plus one growth page; the common
        # inventory does not cover protection, so both ranks must defer.
        c.prepare_kvduo_graph_replay(torch.tensor([0, 1, 2]))
        c.reclaim_kvduo_hot_pages.assert_not_called()
        p.alloc.assert_not_called()
        assert c.kvduo_req_hot_capacity[0].tolist() == [8, 8, 32]
        assert p.free_pages == 4 + 2 * rank

        from test_kvduo_hot_merge import make_merge_coordinator

        c, p = make_merge_coordinator({(0, 0): 7, (0, 1): 2})
        c.tp_world_size = 2
        c.tp_group = dist.group.WORLD
        p.free_pages = rank
        c.kvduo_hot_growth_events[0] = 1
        c.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
        assert c.kvduo_req_hot_capacity[0].tolist() == [8, 12]
        assert c.kvduo_capacity_steps[0].tolist() == [0, 65]
        assert p.free_pages == 4 + rank
    finally:
        dist.destroy_process_group()


def test_two_tp_ranks_use_common_available_pages_for_admission_and_recovery(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _tp_modes_worker,
        args=(str(tmp_path / "gloo-init"),),
        nprocs=2,
        join=True,
        start_method="spawn",
    )


def test_locked_mode_with_enough_space_admits_prefill_without_extra_rollback():
    c, p = fixed_stack(layers=2)
    p.free_pages = 8
    req = SimpleNamespace(rid="new", prefix_indices=torch.empty(0, dtype=torch.int64))
    before = c.kvduo_req_hot_capacity.clone()
    token = c.reserve_kvduo_prefill(2, [0], prefill_req=req)
    assert token is not None and p.free_pages == 8
    assert torch.equal(c.kvduo_req_hot_capacity, before)
    assert c._kvduo_mandatory_hot_pages() == 0
    c.consume_kvduo_prefill_reservation(token, 2)
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    for layer in range(2):
        c._release_kvduo_prefill_hold(c._kvduo_owner_key((layer, 0)))
    c.kvduo_complete_prefill(req)
    assert c._kvduo_mandatory_hot_pages() == 0
    assert c._kvduo_prefill_lifecycle[c._kvduo_prefill_key(req)][1] == "ready"


def test_locked_prefill_never_reclaims_full_or_charges_future_hot_minimum():
    c, p = fixed_stack(layers=2)
    p.free_pages = 2
    req = SimpleNamespace(rid="new", prefix_indices=torch.empty(0, dtype=torch.int64))
    before = c.kvduo_req_hot_capacity.clone()
    c.reclaim_kvduo_full_pages = MagicMock()
    assert c.reserve_kvduo_prefill(3, [0], prefill_req=req) is None
    token = c.reserve_kvduo_prefill(2, [0], prefill_req=req)
    assert token is not None
    c.reclaim_kvduo_full_pages.assert_not_called()
    assert torch.equal(c.kvduo_req_hot_capacity, before)
    assert c._kvduo_optional_protected_pages() == 2
