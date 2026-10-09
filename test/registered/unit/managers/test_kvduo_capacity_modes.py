"""Exercise space hysteresis, complete-prefill conversion and rollback obligations."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.mem_cache.sparsity.core.kvduo_capacity_modes import (
    KVDuoRecoveryRequest,
    plan_kvduo_recovery,
)
from test_kvduo_prefill_rollback import merged_and_expanded, ready_coordinator
from test_kvduo_pressure_admission import make_full_coordinator


def recovery(key, order, quantum, ready=True):
    return KVDuoRecoveryRequest((key, 0), order, quantum, ready)


def test_fifo_quantum_protects_prefill_and_does_not_bypass_large_ready_request():
    requests = [recovery(0, 1, 4), recovery(1, 2, 1)]
    assert plan_kvduo_recovery(requests, 7, 4) == ()
    assert plan_kvduo_recovery(requests, 10, 4) == ((0, 0), (1, 0))
    requests[0] = recovery(0, 1, 4, False)
    assert plan_kvduo_recovery(requests, 7, 4) == ((1, 0),)


def fixed_stack(layers=4):
    owners = {(layer, 0): 3 for layer in range(layers)}
    c, p = merged_and_expanded(owners, owners, layers=layers)
    token = c.reserve_kvduo_prefill(1, [0])
    assert token is not None
    c.consume_kvduo_prefill_reservation(token, 1)
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    p.free_pages = 0
    return c, p


def test_recovery_needs_all_established_layers_and_uses_only_latest_events():
    c, p = fixed_stack()
    key = c._kvduo_request_key(0)
    c._kvduo_next_prefill_pages = 4
    p.free_pages = 7
    c.kvduo_hot_growth_events[:, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c._kvduo_request_limited(0)
    assert c.kvduo_req_hot_capacity[:, 0].tolist() == [8] * 4
    assert not c.kvduo_hot_growth_events.any()
    p.free_pages = 10
    # The earlier event in layers 1 and 3 cannot be replayed on recovery.
    c.kvduo_hot_growth_events[[0, 2], 0] = 1
    cost = c._kvduo_prefill_budget_cost[key]
    allocate = p.alloc.side_effect

    def redeem(slots):
        assert c._kvduo_recovery_credits == 4
        assert c._kvduo_request_limited(0)
        assert c._kvduo_prefill_budget_cost[key] == cost
        return allocate(slots)

    p.alloc.side_effect = redeem
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert not c._kvduo_request_limited(0)
    assert c.kvduo_req_hot_capacity[:, 0].tolist() == [12, 8, 12, 8]
    assert c.kvduo_hot_page_limit[:, 0].tolist() == [8] * 4
    assert c.kvduo_hot_base_capacity[:, 0].tolist() == [8] * 4
    assert p.free_pages == 8 and c._kvduo_recovery_credits == 0
    assert not c._kvduo_prefill_budget_cost


def test_recovery_failure_keeps_request_fixed_and_cost_reserved():
    c, p = fixed_stack(layers=2)
    p.free_pages = 2
    p.alloc.side_effect = lambda _: None
    c.kvduo_hot_growth_events[:, 0] = 1
    with pytest.raises(RuntimeError, match="reserved credits"):
        c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c._kvduo_request_limited(0)
    assert c.kvduo_hot_page_limit[:, 0].tolist() == [2, 2]
    assert c._kvduo_prefill_budget_cost
    assert c._kvduo_recovery_credits == 0


def test_full_prefill_waits_for_conversion_and_all_grouped_prefills():
    c, p = fixed_stack(layers=2)
    key = c._kvduo_request_key(0)
    first, second = SimpleNamespace(rid="a"), SimpleNamespace(rid="b")
    first_key = c._kvduo_prefill_key(first)
    second_key = c._kvduo_prefill_key(second)
    c._kvduo_prefill_associations[key] = {first_key, second_key}
    p.free_pages = 20
    c.kvduo_complete_prefill(first)
    assert c._kvduo_prefill_lifecycle[first_key][1] == "awaiting_backup"
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c._kvduo_request_limited(0)
    c._kvduo_prefill_lifecycle[first_key][1] = "converted"
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c._kvduo_request_limited(0)
    c._kvduo_prefill_lifecycle[second_key][1] = "cancelled"
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert not c._kvduo_request_limited(0)
    assert c.kvduo_req_hot_capacity[:, 0].tolist() == [8, 8]


@pytest.mark.parametrize("ratio", [1, 4])
def test_conversion_releases_all_eligible_history_despite_abundant_space(ratio):
    c, p, req = make_full_coordinator(pages=8, initial_pages=8, ratio=ratio)
    req.rid = "new"
    req.cache_protected_len = 3 * ratio  # include the partially overlapped page
    c._kvduo_prefill_limited_requests.add((0, 99))
    c.kvduo_complete_prefill(req)
    key = c._kvduo_prefill_key(req)
    p.free_pages = 100
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    c.process_kvduo_full_conversions()
    assert mapping[:4].tolist() == [2, 3, 4, 5]
    assert not mapping[4:12].any()
    assert mapping[12:].tolist() == [14, 15, 16, 17]
    assert p.free_pages == 104
    assert c._kvduo_prefill_lifecycle[key][1] == "converted"
    assert c._mixed_slots[0] and c._kvduo_mandatory_hot_pages() == 1


def test_conversion_waits_for_host_version_and_transfer_protection_then_finishes():
    c, p, req = make_full_coordinator(pages=8, initial_pages=8)
    req.rid = "new"
    c._kvduo_prefill_limited_requests.add((0, 99))
    c.kvduo_complete_prefill(req)
    key = c._kvduo_prefill_key(req)
    c.full_host_version[4:6] = 0
    c._kvduo_pressure_protected = torch.tensor([6, 7])
    c.process_kvduo_full_conversions()
    assert c._kvduo_prefill_lifecycle[key][1] == "awaiting_backup"
    assert c._kvduo_full_conversion_pending[key] == c._kvduo_request_key(0)
    assert p.free_pages == 4
    # A requested conversion survives the disappearance of other locks.
    c._kvduo_prefill_limited_requests.clear()
    c.full_host_version[4:6] = 1
    c._kvduo_pressure_protected = None
    c.process_kvduo_full_conversions()
    assert p.free_pages == 6
    assert c._kvduo_prefill_lifecycle[key][1] == "converted"


def test_no_locked_request_keeps_prefill_full_resident():
    c, p, req = make_full_coordinator(pages=8, initial_pages=8)
    req.rid = "new"
    c.kvduo_complete_prefill(req)
    c.process_kvduo_full_conversions()
    assert p.free_pages == 0
    assert torch.all(c.mem_pool_device.full_to_hisparse_device_index_mapping > 0)


def test_prefill_mixed_minimum_cannot_be_spent_by_optional_growth():
    c, p = ready_coordinator(extra=4)
    req = SimpleNamespace(rid="new", prefix_indices=torch.empty(0, dtype=torch.int64))
    # Four compute pages plus its one-page HOT minimum cannot fit the rollback.
    before = c.req_device_buffer_tokens.clone()
    assert c.reserve_kvduo_prefill(4, [0], prefill_req=req) is None
    assert torch.equal(c.req_device_buffer_tokens, before) and p.free_pages == 0
    token = c.reserve_kvduo_prefill(3, [0], prefill_req=req)
    assert token is not None and c._kvduo_mandatory_hot_pages() == 1
    assert c._kvduo_optional_protected_pages() == 4
    c.finish_kvduo_prefill_reservation(token, cancelled=True)
    assert c._kvduo_optional_protected_pages() == 0
    assert not c._kvduo_request_limited(0)


def test_plan_protection_does_not_double_count_reservation_or_hot_commitment():
    c, p = ready_coordinator(extra=4)
    req = SimpleNamespace(rid="new", prefix_indices=torch.empty(0, dtype=torch.int64))
    c.protect_kvduo_prefill_plan(req, 6)  # three compute pages
    token = c.reserve_kvduo_prefill(3, [0], prefill_req=req)
    assert token is not None
    assert c._kvduo_optional_protected_pages() == 4
    c.bind_kvduo_prefill_batch([req], [token])
    c.consume_kvduo_prefill_reservation(token, 3)
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    assert c._kvduo_optional_protected_pages() == 1


def test_tp_disagreement_never_rolls_back_or_unlocks_local_request():
    c, p = fixed_stack(layers=2)
    p.free_pages = 20
    c._kvduo_consensus = MagicMock(return_value=False)
    c.kvduo_hot_growth_events[:, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c._kvduo_request_limited(0)
    assert c.kvduo_req_hot_capacity[:, 0].tolist() == [8, 8]


def test_slot_reuse_clears_limits_and_queue_without_restarting_scan():
    c, p = fixed_stack(layers=2)
    c._kvduo_hand = 1
    c._kvduo_hand_sequence = 10
    c._release_kvduo_layer_hot_pages(0)
    assert c._kvduo_request_generation[0] == 1
    assert c.kvduo_hot_page_limit[:, 0].tolist() == [8, 8]
    assert not c._kvduo_fixed_queue and not c._kvduo_prefill_budget_cost
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
        assert token is not None and c._kvduo_request_limited(0)
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
        assert c._kvduo_request_limited(0)
        assert c.kvduo_req_hot_capacity[:, 0].tolist() == [8, 8]
        c._kvduo_next_prefill_pages = 1
        p.free_pages = 2 + rank  # Only rank1 could fund the whole request locally.
        c.kvduo_hot_growth_events[:, 0] = 1
        c.prepare_kvduo_graph_replay(torch.tensor([0]))
        assert c._kvduo_request_limited(0)
        p.free_pages = 3 + rank
        c.kvduo_hot_growth_events[0, 0] = 1
        c.prepare_kvduo_graph_replay(torch.tensor([0]))
        assert not c._kvduo_request_limited(0)
        assert c.kvduo_req_hot_capacity[:, 0].tolist() == [12, 8]
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
    assert c._kvduo_mandatory_hot_pages() == 2
    c.consume_kvduo_prefill_reservation(token, 2)
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    c._release_kvduo_prefill_hold(c._kvduo_request_key(0))
    c.kvduo_complete_prefill(req)
    assert c._kvduo_mandatory_hot_pages() == 0
    assert c._kvduo_prefill_lifecycle[c._kvduo_prefill_key(req)][1] == "converted"


def test_locked_prefill_reclaims_full_for_its_future_hot_minimum_before_hot_rollback():
    c, p = fixed_stack(layers=2)
    p.free_pages = 2
    req = SimpleNamespace(rid="new", prefix_indices=torch.empty(0, dtype=torch.int64))
    before = c.kvduo_req_hot_capacity.clone()

    def reclaim(slots, *, reserve_hot_minimum):
        assert slots == 4 and reserve_hot_minimum
        p.free_pages += 2
        return slots

    c.reclaim_kvduo_full_pages = MagicMock(side_effect=reclaim)
    token = c.reserve_kvduo_prefill(2, [0], prefill_req=req)
    assert token is not None
    c.reclaim_kvduo_full_pages.assert_called_once()
    assert torch.equal(c.kvduo_req_hot_capacity, before)
    assert c._kvduo_optional_protected_pages() == 4
