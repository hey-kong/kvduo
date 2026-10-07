"""Exercise the production merge policy, payload movement, and free-page loop."""

import itertools
import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.mem_cache.sparsity.core.kvduo_hot_merge import plan_kvduo_hot_merge
from test_kvduo_hot_growth import make_coordinator


def make_merge_coordinator(page_counts, *, layers=1, top_k=2):
    requests = max(req for _, req in page_counts) + 1
    coordinator, physical = make_coordinator(
        layers=layers, requests=requests, top_k=top_k
    )
    coordinator._ensure_kvduo_hot_capacity_targets_batch(
        {owner: pages * 4 for owner, pages in page_counts.items()}
    )
    coordinator._kvduo_pending_hot_minimum.clear()
    # Linear backing lets CPU tests exercise actual production index copies.
    # Locations include the synthetic storage-layer stride used by the fixture.
    coordinator.mem_pool_device.kv_buffer = [
        torch.full((12000, 4), -999, dtype=torch.int64) for _ in range(layers)
    ]
    coordinator.kvduo_hot_tail_observation_steps.fill_(65)
    coordinator.kvduo_hot_tail_ema.fill_(1)
    for (layer, req), pages in page_counts.items():
        capacity = pages * 4
        coordinator.req_device_buffer_tokens[layer, req, :capacity] = torch.arange(
            req * 100 + 10, req * 100 + 10 + capacity, dtype=torch.int32
        )
        coordinator.lru_slots[layer, req, :capacity] = torch.arange(
            capacity, dtype=torch.int16
        )
        seed_payloads(coordinator, layer, req)
    physical.free_pages = 0
    coordinator.token_to_kv_pool_allocator.free_hisparse_indices.reset_mock()
    return coordinator, physical


def seed_payloads(coordinator, layer, req):
    capacity = int(coordinator.kvduo_req_hot_capacity[layer, req])
    tokens = coordinator.req_device_buffer_tokens[layer, req, :capacity]
    locs = coordinator.req_device_buffer_token_locs[layer, req, :capacity].long()
    # Every payload word matters; movement must preserve the complete row.
    coordinator.mem_pool_device.kv_buffer[layer][locs] = tokens.long()[
        :, None
    ] * 10 + torch.arange(4)


def live_lru(coordinator, layer, req):
    capacity = int(coordinator.kvduo_req_hot_capacity[layer, req])
    tokens = coordinator.req_device_buffer_tokens[layer, req]
    return [
        int(tokens[slot])
        for slot in coordinator.lru_slots[layer, req, :capacity]
        if tokens[slot] >= 0
    ]


def assert_payloads_match_tokens(coordinator, layer, req):
    capacity = int(coordinator.kvduo_req_hot_capacity[layer, req])
    tokens = coordinator.req_device_buffer_tokens[layer, req, :capacity]
    locs = coordinator.req_device_buffer_token_locs[layer, req, :capacity].long()
    valid = tokens >= 0
    assert torch.equal(
        coordinator.mem_pool_device.kv_buffer[layer][locs[valid]],
        tokens[valid].long()[:, None] * 10 + torch.arange(4),
    )


def test_merge_keeps_entry_mru_selects_nonleading_pages_and_copies_before_free():
    coordinator, physical = make_merge_coordinator({(0, 0): 4, (1, 0): 2}, layers=2)
    owner = (0, 0)
    pages = list(coordinator._kvduo_req_layer_pages[owner])
    survivors = [5, 1, 12, 6, 13, 4, 14, 7]
    coordinator.lru_slots[owner][:16] = torch.tensor(
        [0, 2, 3, 8, 9, 10, 11, 15] + survivors, dtype=torch.int16
    )
    expected = live_lru(coordinator, *owner)[-8:]
    other = coordinator.req_device_buffer_tokens[1].clone()
    old_locs = coordinator.req_device_buffer_token_locs[owner].clone()
    coordinator._move_kvduo_hot_entries = MagicMock(
        wraps=coordinator._move_kvduo_hot_entries
    )
    free = coordinator.token_to_kv_pool_allocator.free_hisparse_indices.side_effect

    def verify_then_free(indices):
        assert coordinator._kvduo_req_layer_pages[owner] == [pages[1], pages[3]]
        assert live_lru(coordinator, *owner) == expected
        assert_payloads_match_tokens(coordinator, *owner)
        free(indices)

    coordinator.token_to_kv_pool_allocator.free_hisparse_indices.side_effect = (
        verify_then_free
    )
    assert coordinator._merge_kvduo_hot_pages(*owner) == 4  # 2 allocator pages
    assert physical.free_pages == 2
    assert coordinator.lru_slots[owner][:8].tolist() == [1, 7, 4, 2, 5, 0, 6, 3]
    src, dst = coordinator._move_kvduo_hot_entries.call_args.args[1:]
    assert src.tolist() == [int(old_locs[1])]
    assert dst.tolist() == [int(old_locs[15])]  # crosses storage-layer fragments
    assert torch.equal(coordinator.req_device_buffer_tokens[1], other)
    assert coordinator.kvduo_hot_tail_observation_steps.tolist() == [[0], [65]]
    assert coordinator.kvduo_hot_tail_ema.tolist() == [[0], [1]]
    for field in (
        coordinator.req_device_buffer_tokens,
        coordinator.req_device_buffer_token_locs,
        coordinator.lru_slots,
    ):
        assert torch.all(field[owner][8:] == -1)
    assert not torch.any(coordinator.hot_page_last_touch[owner][2:])
    assert all(
        start not in coordinator._kvduo_hot_page_owners
        for start in [pages[0], pages[2]]
    )


def test_merge_retains_all_valid_entries_when_underfilled_and_prioritizes_empty_slots():
    coordinator, _ = make_merge_coordinator({(0, 0): 4})
    tokens = coordinator.req_device_buffer_tokens[0, 0]
    tokens[:16] = -1
    tokens[[1, 5, 13]] = torch.tensor([30, 40, 50], dtype=torch.int32)
    coordinator.lru_slots[0, 0, :16] = torch.tensor(
        [slot for slot in range(16) if slot not in (1, 5, 13)] + [13, 1, 5],
        dtype=torch.int16,
    )
    seed_payloads(coordinator, 0, 0)
    coordinator._merge_kvduo_hot_pages(0, 0)
    assert live_lru(coordinator, 0, 0) == [50, 30, 40]
    lru = coordinator.lru_slots[0, 0, :8].long()
    assert torch.all(tokens[lru[:5]] < 0)
    assert_payloads_match_tokens(coordinator, 0, 0)


def test_optional_growth_selects_raw_q_even_for_one_page_need_and_pools_surplus():
    coordinator, physical = make_merge_coordinator({(0, 0): 8, (0, 1): 4, (0, 2): 1})
    coordinator.kvduo_hot_tail_ema[0] = torch.tensor([0.5, 0.75, 0.1])
    coordinator.hot_page_last_touch[0, 0].fill_(1000)
    coordinator.hot_page_last_touch[0, 1].zero_()
    released = list(coordinator._kvduo_req_layer_pages[(0, 0)][:4])
    coordinator.kvduo_hot_growth_events[0, 2] = 1
    addresses = [
        tensor.data_ptr()
        for tensor in (
            coordinator.req_device_buffer_tokens,
            coordinator.req_device_buffer_token_locs,
            coordinator.lru_slots,
            coordinator.kvduo_req_hot_capacity_gpu,
        )
    ]
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1, 2]))
    assert coordinator.kvduo_req_hot_capacity.tolist() == [[16, 16, 8]]
    assert physical.free_pages == 3
    assert coordinator._kvduo_req_layer_pages[(0, 2)][-1] in released
    assert coordinator.kvduo_hot_tail_observation_steps.tolist() == [[0, 65, 65]]
    assert coordinator.kvduo_hot_tail_ema.tolist() == [[0, 0.75, pytest.approx(0.1)]]
    assert (
        coordinator._reclaim_for_physical_allocation.call_count == 1
    )  # initial admission only
    assert addresses == [
        tensor.data_ptr()
        for tensor in (
            coordinator.req_device_buffer_tokens,
            coordinator.req_device_buffer_token_locs,
            coordinator.lru_slots,
            coordinator.kvduo_req_hot_capacity_gpu,
        )
    ]
    assert_payloads_match_tokens(coordinator, 0, 0)


def test_selection_does_not_divide_ema_by_release_pages_again():
    coordinator, physical = make_merge_coordinator({(0, 0): 8, (0, 1): 4})
    coordinator.kvduo_hot_tail_ema[0] = torch.tensor([0.8, 0.5])
    assert coordinator.reclaim_kvduo_hot_pages(2) == 4
    assert coordinator.kvduo_req_hot_capacity.tolist() == [[32, 8]]
    assert physical.free_pages == 2


def test_loop_merges_other_owners_in_score_order_stops_when_enough_and_resets_each():
    coordinator, physical = make_merge_coordinator({(0, 0): 4, (0, 1): 4, (0, 2): 4})
    coordinator.kvduo_hot_tail_ema[0] = torch.tensor([2.0, 3.0, 1.0])
    coordinator._merge_kvduo_hot_pages = MagicMock(
        wraps=coordinator._merge_kvduo_hot_pages
    )
    assert coordinator.reclaim_kvduo_hot_pages(6) == 8
    assert physical.free_pages == 4
    assert [
        call.args for call in coordinator._merge_kvduo_hot_pages.call_args_list
    ] == [(0, 2), (0, 0)]
    assert coordinator.kvduo_req_hot_capacity.tolist() == [[8, 16, 8]]
    assert coordinator.kvduo_hot_tail_observation_steps.tolist() == [[0, 65, 0]]
    # The two merged owners can still release a page geometrically, but must
    # regain >64 observations. They cannot be repeatedly halved in this pass.
    assert coordinator.reclaim_kvduo_hot_pages(20, excluded_owners={(0, 1)}) == 0


def test_eligibility_excludes_minimum_insufficient_observations_applicants_protection_and_staging():
    coordinator, physical = make_merge_coordinator(
        {(0, req): pages for req, pages in enumerate([1, 4, 4, 4, 4, 3, 4])}
    )
    coordinator.kvduo_hot_tail_ema[0] = torch.tensor([0, 0, 0.1, 0.2, 0.3, 0.9, 0])
    coordinator.kvduo_hot_tail_observation_steps[0, 1] = 64
    coordinator.kvduo_hot_tail_observation_steps[0, 6] = 0
    coordinator._kvduo_hot_pressure_protected = {(0, 3, 1)}
    coordinator._active_kvduo_reqs[4] = SimpleNamespace(hisparse_staging=True)
    assert coordinator.reclaim_kvduo_hot_pages(20, excluded_owners={(0, 2)}) == 2
    assert physical.free_pages == 1
    assert coordinator.kvduo_req_hot_capacity.tolist() == [[4, 16, 16, 16, 16, 8, 16]]
    assert coordinator.kvduo_hot_tail_observation_steps[0, 5] == 0
    coordinator.kvduo_hot_tail_observation_steps[0, 5] = 65
    assert coordinator.reclaim_kvduo_hot_pages(2, excluded_owners={(0, 2)}) == 2
    assert coordinator.kvduo_req_hot_capacity[0, 5] == 4


def test_same_request_other_layer_can_merge_but_all_growth_applicants_are_excluded():
    coordinator, physical = make_merge_coordinator({(0, 0): 4, (1, 0): 4}, layers=2)
    coordinator.kvduo_hot_tail_ema[:, 0] = torch.tensor([0, 1.0])
    coordinator.kvduo_hot_growth_events[0, 0] = 1
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert coordinator.kvduo_req_hot_capacity.tolist() == [[20], [8]]
    assert physical.free_pages == 1
    assert coordinator.kvduo_hot_tail_observation_steps.tolist() == [[65], [0]]

    physical.free_pages = 0
    coordinator.kvduo_hot_tail_observation_steps.fill_(65)
    coordinator.kvduo_hot_growth_events.fill_(1)
    before = coordinator.kvduo_req_hot_capacity.clone()
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert torch.equal(coordinator.kvduo_req_hot_capacity, before)


def test_topk_minimum_clamps_merge_and_prevents_further_reclaim():
    coordinator, physical = make_merge_coordinator({(0, 0): 3}, top_k=7)
    assert coordinator.reclaim_kvduo_hot_pages(20) == 2
    assert physical.free_pages == 1
    assert coordinator.kvduo_req_hot_capacity[0, 0] == 8
    coordinator.kvduo_hot_tail_observation_steps.fill_(65)
    assert coordinator.reclaim_kvduo_hot_pages(2) == 0


def test_merge_orders_decode_and_pending_backup_before_movement(monkeypatch):
    import sglang.srt.managers.hisparse_coordinator as module

    coordinator, _ = make_merge_coordinator({(0, 0): 4})
    coordinator.decode_producer_stream = object()
    coordinator._has_pending_backup = True
    trace = []
    stream = SimpleNamespace(wait_stream=lambda _: trace.append("producer"))
    monkeypatch.setattr(module.device_module, "current_stream", lambda _: stream)

    def backup():
        trace.append("backup")
        coordinator._has_pending_backup = False

    coordinator.wait_for_pending_backup = backup
    move = coordinator._move_kvduo_hot_entries

    def ordered_move(*args):
        assert "producer" in trace and "backup" in trace
        trace.append("move")
        move(*args)

    coordinator._move_kvduo_hot_entries = ordered_move
    coordinator._merge_kvduo_hot_pages(0, 0)
    assert trace.index("producer") < trace.index("backup") < trace.index("move")


def test_failed_payload_move_does_not_release_pages_or_publish_compaction():
    coordinator, physical = make_merge_coordinator({(0, 0): 4})
    coordinator.lru_slots[0, 0, :16] = torch.tensor(
        [0, 2, 3, 8, 9, 10, 11, 15, 5, 1, 12, 6, 13, 4, 14, 7],
        dtype=torch.int16,
    )
    capacity = coordinator.kvduo_req_hot_capacity.clone()
    tokens = coordinator.req_device_buffer_tokens.clone()
    pages = list(coordinator._kvduo_req_layer_pages[(0, 0)])
    coordinator._move_kvduo_hot_entries = MagicMock(
        side_effect=RuntimeError("copy failed")
    )
    with pytest.raises(RuntimeError, match="copy failed"):
        coordinator._merge_kvduo_hot_pages(0, 0)
    coordinator.token_to_kv_pool_allocator.free_hisparse_indices.assert_not_called()
    assert physical.free_pages == 0
    assert torch.equal(coordinator.kvduo_req_hot_capacity, capacity)
    assert torch.equal(coordinator.req_device_buffer_tokens, tokens)
    assert coordinator._kvduo_req_layer_pages[(0, 0)] == pages


def test_mandatory_allocation_reclaims_full_first_then_merges_mature_hot_owner():
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

    coordinator, physical = make_merge_coordinator({(0, 0): 4, (0, 1): 1})
    coordinator.item_size_bytes = 32
    full_protected = torch.tensor([77])
    coordinator._kvduo_pressure_protected = full_protected

    def full_reclaim(slots):
        assert coordinator._kvduo_pressure_protected is full_protected
        return 0

    coordinator.reclaim_kvduo_full_pages = MagicMock(side_effect=full_reclaim)
    coordinator._reclaim_for_physical_allocation = (
        HiSparseCoordinator._reclaim_for_physical_allocation.__get__(coordinator)
    )
    coordinator._require_allocation_ready = (
        HiSparseCoordinator._require_allocation_ready
    )
    protected = {(0, 1, 0)}
    coordinator._kvduo_hot_pressure_protected = protected
    coordinator._ensure_kvduo_hot_capacity_targets_batch({(0, 1): 8})
    coordinator.reclaim_kvduo_full_pages.assert_called_once_with(2)
    assert coordinator.kvduo_req_hot_capacity.tolist() == [[8, 8]]
    assert physical.free_pages == 1
    assert coordinator._kvduo_hot_pressure_protected is protected
    assert coordinator._kvduo_pressure_protected is full_protected
    assert coordinator.kvduo_hot_tail_observation_steps.tolist() == [[0, 65]]


def test_plans_preserve_actual_lru_survivors_and_minimize_moves_across_varied_layouts():
    rng = random.Random(4107)
    for _ in range(250):
        n, k = rng.randint(2, 8), rng.randint(1, 5)
        top_k = rng.randint(1, n * k)
        minimum = (top_k + k - 1) // k
        m = max(minimum, (n + 1) // 2)
        tokens = [slot + 10 if rng.random() > 0.3 else -1 for slot in range(n * k)]
        lru = rng.sample(range(n * k), n * k)
        plan = plan_kvduo_hot_merge(tokens, lru, k, top_k)
        if m == n:
            assert plan is None
            continue
        expected = [tokens[slot] for slot in lru if tokens[slot] >= 0][-m * k :]
        assert [
            plan.tokens[slot] for slot in plan.lru_slots if plan.tokens[slot] >= 0
        ] == expected
        counts = [
            sum(
                token in expected for token in tokens[p * k : (p + 1) * k] if token >= 0
            )
            for p in range(n)
        ]
        best = max(
            sum(counts[p] for p in selected)
            for selected in itertools.combinations(range(n), m)
        )
        assert len(plan.source_slots) == len(expected) - best
        assert sorted(plan.lru_slots) == list(range(m * k))
        assert set(plan.kept_pages).isdisjoint(plan.released_pages)
        assert len(plan.released_pages) == n - m
        payload = list(tokens)
        for src, dst in zip(plan.source_slots, plan.destination_slots):
            assert src // k in plan.released_pages and dst // k in plan.kept_pages
            payload[dst] = tokens[src]
        compacted = [payload[p * k + i] for p in plan.kept_pages for i in range(k)]
        assert all(
            token < 0 or token == compacted[slot]
            for slot, token in enumerate(plan.tokens)
        )
