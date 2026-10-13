"""Exercise the production merge policy, payload movement, and free-page loop."""

import itertools
import random
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.mem_cache.sparsity.core.kvduo_hot_merge import (
    plan_kvduo_hot_merge,
    prepare_kvduo_hot_merge,
    kvduo_shrink_target,
)
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
    backing = torch.full((12000, 4), -999, dtype=torch.int64)
    coordinator.mem_pool_device.get_flat_key_buffer = lambda: backing
    coordinator.mem_pool_device.kv_buffer = [backing for _ in range(layers)]
    for (layer, req), pages in page_counts.items():
        capacity = pages * 4
        coordinator.req_device_buffer_tokens[layer, req, :capacity] = torch.arange(
            req * 100 + 10, req * 100 + 10 + capacity, dtype=torch.int32
        )
        coordinator.lru_slots[layer, req, :capacity] = torch.arange(
            capacity, dtype=torch.int16
        )
        seed_payloads(coordinator, layer, req)
        coordinator.kvduo_capacity_steps[layer, req] = 65
        coordinator.kvduo_capacity_ema[layer, req, : min(2, pages)] = 100
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


def prefill_protected_growth_coordinator(*, free_pages, applicant_pages=2):
    c, p = make_merge_coordinator({(0, 0): 2, (0, 1): applicant_pages, (0, 2): 8})
    key = c._kvduo_owner_key((0, 0))
    c._establish_kvduo_hot_base((0, 0), c._kvduo_req_layer_pages[0, 0])
    c._kvduo_prefill_limited_objects.add(key)
    c._kvduo_prefill_budget_cost[key] = 1.0
    c.kvduo_hot_page_limit[:, 0] = 2
    # Keep the lock pending complete-prefill conversion even when pages appear.
    prefill_key = c._kvduo_prefill_key(SimpleNamespace(rid="prefilling"))
    c._kvduo_prefill_associations[key] = {prefill_key}
    c._kvduo_next_prefill_pages = 5
    p.free_pages = free_pages
    p.alloc.reset_mock()
    c.reclaim_kvduo_hot_pages = MagicMock(wraps=c.reclaim_kvduo_hot_pages)
    return c, p


@pytest.mark.parametrize(
    "events,free_pages,applicant_pages",
    [(0, 0, 2), (1, 0, 2), (1, 4, 2), (0, 5, 2), (1, 0, 8)],
)
def test_optional_growth_never_merges_to_cover_waiting_prefill(
    events, free_pages, applicant_pages
):
    c, p = prefill_protected_growth_coordinator(
        free_pages=free_pages, applicant_pages=applicant_pages
    )
    banks = (
        c.req_device_buffer_tokens,
        c.req_device_buffer_token_locs,
        c.lru_slots,
        c.kvduo_req_hot_capacity,
        c.kvduo_req_hot_capacity_gpu,
        c.kvduo_capacity_ema,
        c.kvduo_capacity_steps,
        c.kvduo_prefill_ema,
        c.kvduo_prefill_steps,
    )
    before = [bank.clone() for bank in banks]
    c.kvduo_hot_growth_events[0, 1] = events

    c.prepare_kvduo_graph_replay(torch.tensor([0, 1, 2]))

    c.reclaim_kvduo_hot_pages.assert_not_called()
    p.alloc.assert_not_called()
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_not_called()
    assert all(torch.equal(bank, old) for bank, old in zip(banks, before))
    assert p.free_pages == free_pages and c._kvduo_hand == 0
    assert c._kvduo_object_limited((0, 0))
    assert c._kvduo_prefill_budget_cost[c._kvduo_owner_key((0, 0))] == 1.0
    assert not c.kvduo_hot_growth_events.any()
    # A deferred capacity event cannot be replayed when space later appears.
    p.free_pages = 6
    c.prepare_kvduo_graph_replay(torch.tensor([0, 1, 2]))
    c.reclaim_kvduo_hot_pages.assert_not_called()
    p.alloc.assert_not_called()
    assert all(torch.equal(bank, old) for bank, old in zip(banks, before))


@pytest.mark.parametrize(
    "free_pages,protected_pages,merge_needed",
    [(5, 5, True), (6, 5, False), (0, 0, True)],
)
def test_optional_growth_merges_only_for_real_applicants_after_prefill_is_covered(
    free_pages, protected_pages, merge_needed
):
    c, p = prefill_protected_growth_coordinator(free_pages=free_pages)
    c._kvduo_next_prefill_pages = protected_pages
    c.kvduo_hot_growth_events[0, 1] = 1
    applicant_lru = live_lru(c, 0, 1)

    c.prepare_kvduo_graph_replay(torch.tensor([0, 1, 2]))

    if merge_needed:
        c.reclaim_kvduo_hot_pages.assert_called_once_with(2)
        # Scanning without shrinking the applicant preserves its event.
        assert (0, 1) in c._kvduo_hot_processed
        assert len(c._kvduo_hot_base_pages[0, 1]) == 2
        assert c.kvduo_req_hot_capacity[0, 2] == 8
        assert_payloads_match_tokens(c, 0, 2)
    else:
        c.reclaim_kvduo_hot_pages.assert_not_called()
        assert c.kvduo_req_hot_capacity[0, 2] == 32
    p.alloc.assert_called_once_with(2)
    assert c.kvduo_req_hot_capacity[0, 1] == 12
    assert live_lru(c, 0, 1) == applicant_lru
    assert p.free_pages == free_pages + (6 if merge_needed else 0) - 1
    assert p.free_pages >= protected_pages
    assert c._kvduo_object_limited((0, 0))


@pytest.mark.parametrize(
    "steps,misses,hits,target",
    [
        (64, 100, [100, 30, 5, 1], 4),
        (65, 100, [100, 30, 5, 1], 2),
        (65, 0, [100, 0, 0, 0], 1),
        (65, 0, [100, 0, 0, 1], 4),
        (65, 1, [100, 0, 0, 0.1], 1),
        (65, 100, [float("nan"), 0, 0, 0], 4),
    ],
)
def test_budget_selects_smallest_mature_capacity(steps, misses, hits, target):
    assert kvduo_shrink_target(hits, misses, steps, 4, 1, 0.10) == target


def test_target_respects_multi_page_minimum():
    assert kvduo_shrink_target([0] * 8, 0, 65, 8, 3, 0.10) == 3


def test_mru_merge_moves_minimum_payload_and_publishes_before_free():
    c, physical = make_merge_coordinator({(0, 0): 4, (1, 0): 2}, layers=2)
    survivors = [5, 1, 12, 6, 13, 4, 14, 7]
    c.lru_slots[0, 0, :16] = torch.tensor(
        [0, 2, 3, 8, 9, 10, 11, 15] + survivors, dtype=torch.int16
    )
    pages = list(c._kvduo_req_layer_pages[0, 0])
    other = c.req_device_buffer_tokens[1].clone()
    c._move_kvduo_hot_entries = MagicMock(wraps=c._move_kvduo_hot_entries)
    free = c.token_to_kv_pool_allocator.free_hisparse_indices.side_effect

    def check(indices):
        assert c._kvduo_req_layer_pages[0, 0] == [pages[1], pages[3]]
        assert live_lru(c, 0, 0) == [10 + s for s in survivors]
        assert_payloads_match_tokens(c, 0, 0)
        assert not c.kvduo_capacity_ema[0, 0].any()
        free(indices)

    c.token_to_kv_pool_allocator.free_hisparse_indices.side_effect = check
    assert c._merge_kvduo_hot_pages(0, 0) == 4
    assert c._move_kvduo_hot_entries.call_args.args[1].numel() == 1
    assert torch.equal(c.req_device_buffer_tokens[1], other)
    for bank in (
        c.req_device_buffer_tokens,
        c.req_device_buffer_token_locs,
        c.lru_slots,
    ):
        assert torch.all(bank[0, 0, 8:] == -1)


def test_no_release_scan_keeps_contents_lru_and_capacity_cost_but_rebuilds_base():
    c, p = make_merge_coordinator({(0, 0): 4})
    c.kvduo_capacity_ema[0, 0, :4] = 1
    before = [
        b.clone()
        for b in (
            c.req_device_buffer_tokens,
            c.lru_slots,
            c.kvduo_capacity_ema,
            c.kvduo_capacity_steps,
        )
    ]
    assert c.reclaim_kvduo_hot_pages(2) == 0
    assert all(
        torch.equal(b, x)
        for b, x in zip(
            (
                c.req_device_buffer_tokens,
                c.lru_slots,
                c.kvduo_capacity_ema,
                c.kvduo_capacity_steps,
            ),
            before,
        )
    )
    assert c._kvduo_hand_sequence == 1
    assert len(c._kvduo_hot_base_pages[0, 0]) == 4
    assert c.reclaim_kvduo_hot_pages(2) == 0
    assert c._kvduo_hand_sequence == 2


def test_hand_resumes_request_major_and_never_erases_cost_to_force_progress():
    c, p = make_merge_coordinator(
        {(layer, req): 3 for req in range(2) for layer in range(2)}, layers=2
    )
    assert c.reclaim_kvduo_hot_pages(2) == 2
    assert c._kvduo_hand == 1 and c._kvduo_hot_processed == {(0, 0): 1}
    assert c.reclaim_kvduo_hot_pages(2) == 2
    assert c._kvduo_hand == 2 and c._kvduo_hot_processed[(1, 0)] == 2


def test_scan_skips_generation_staging_protection_and_fixed_mode():
    c, p = make_merge_coordinator({(0, req): 3 for req in range(5)})
    c._kvduo_request_generation[0] += 1
    c._kvduo_hot_pressure_protected.add((0, 1, 0))
    c._active_kvduo_reqs[2] = SimpleNamespace(hisparse_staging=True)
    c._kvduo_prefill_limited_objects.add(c._kvduo_owner_key((0, 3)))
    assert c.reclaim_kvduo_hot_pages(2) == 2
    assert c._kvduo_hot_processed == {(0, 4): 1}
    c._kvduo_request_generation[4] += 1
    assert c.reclaim_kvduo_hot_pages(2) == 0


def test_applicant_can_shrink_without_growing_back_at_same_boundary():
    c, p = make_merge_coordinator({(0, 0): 7, (0, 1): 2})
    c.kvduo_hot_growth_events[0, 0] = 1
    p.alloc.reset_mock()
    c.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    assert c.kvduo_req_hot_capacity[0].tolist() == [8, 8]
    assert p.free_pages == 5
    p.alloc.assert_not_called()
    assert not c.kvduo_hot_growth_events.any()
    assert not c.kvduo_capacity_ema[0, 0].any()
    assert c.kvduo_capacity_steps[0, 0] == 0
    assert_payloads_match_tokens(c, 0, 0)
    c.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    p.alloc.assert_not_called()


def test_only_shrunk_applicants_lose_their_events():
    c, p = make_merge_coordinator({(0, 0): 7, (0, 1): 2})
    c.kvduo_hot_growth_events[0] = 1
    p.alloc.reset_mock()
    c.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    assert c.kvduo_req_hot_capacity[0].tolist() == [8, 12]
    assert p.free_pages == 4
    p.alloc.assert_called_once_with(2)
    assert c.kvduo_capacity_steps[0].tolist() == [0, 65]
    assert c.kvduo_capacity_ema[0, 1].tolist() == [100, 100, 100, 0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("consume_before_merge", [False, True])
def test_mandatory_merge_consumes_event_before_or_after_replay_reads_it(
    consume_before_merge,
):
    c, p = make_merge_coordinator({(0, 0): 7, (0, 1): 0})
    c.item_size_bytes = 1
    c._kvduo_pending_hot_minimum.add(1)
    c._reclaim_for_physical_allocation = MethodType(
        type(c)._reclaim_for_physical_allocation, c
    )
    c._require_allocation_ready = type(c)._require_allocation_ready
    c.kvduo_hot_growth_events[0, 0] = 1
    p.alloc.reset_mock()
    if not consume_before_merge:
        # A scheduler allocation can scan Hand before replay reads GPU events.
        c._ensure_kvduo_hot_capacity_targets_batch({(0, 1): 4})
        assert not c.kvduo_hot_growth_events.any()
    c.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    assert c.kvduo_req_hot_capacity[0].tolist() == [8, 4]
    assert p.free_pages == 4
    p.alloc.assert_called_once_with(2)
    c.reclaim_kvduo_full_pages.assert_called_once()
    assert not c.kvduo_hot_growth_events.any()


def test_grown_tail_inherits_hit_prior_instead_of_appearing_free_to_shrink():
    c, p = make_merge_coordinator({(0, 0): 2})
    p.free_pages = 1
    old_lru = live_lru(c, 0, 0)
    c.kvduo_hot_growth_events[0, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c.kvduo_req_hot_capacity.item() == 12
    assert c.kvduo_capacity_steps.item() == 65
    assert c.kvduo_capacity_ema[0, 0, :3].tolist() == [100] * 3
    assert c.reclaim_kvduo_hot_pages(2) == 0
    assert c.kvduo_req_hot_capacity.item() == 12
    assert c.kvduo_capacity_steps.item() == 65
    assert live_lru(c, 0, 0) == old_lru
    assert_payloads_match_tokens(c, 0, 0)
    # Later zero contribution permits reclamation without a new maturity phase.
    c.kvduo_capacity_ema[0, 0, 2] = 0
    assert c.reclaim_kvduo_hot_pages(2) == 2
    assert c.kvduo_req_hot_capacity.item() == 8
    assert live_lru(c, 0, 0) == old_lru
    assert not c.kvduo_capacity_ema.any() and not c.kvduo_capacity_steps.any()
    assert_payloads_match_tokens(c, 0, 0)


def test_zero_old_tail_keeps_zero_prior_and_can_still_shrink():
    c, p = make_merge_coordinator({(0, 0): 2}, top_k=5)
    c.kvduo_capacity_ema[0, 0, 1] = 0
    p.free_pages = 1
    c.kvduo_hot_growth_events[0, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c.kvduo_capacity_ema[0, 0, :3].tolist() == [100, 0, 0]
    assert c.kvduo_capacity_steps.item() == 65
    assert c.reclaim_kvduo_hot_pages(2) == 2
    assert c.kvduo_req_hot_capacity.item() == 8
    assert_payloads_match_tokens(c, 0, 0)


def test_copy_failure_does_not_publish_or_free():
    c, p = make_merge_coordinator({(0, 0): 4})
    survivors = list(range(8, 16))
    c.lru_slots[0, 0, :16] = torch.tensor(
        [0, 2, 3, 5, 6, 7, 9, 10] + [1, 4, 8, 11, 12, 13, 14, 15], dtype=torch.int16
    )
    before = c.req_device_buffer_tokens.clone()
    c._move_kvduo_hot_entries = MagicMock(side_effect=RuntimeError("copy failed"))
    with pytest.raises(RuntimeError, match="copy failed"):
        c._merge_kvduo_hot_pages(0, 0)
    assert torch.equal(before, c.req_device_buffer_tokens)
    assert len(c._kvduo_req_layer_pages[0, 0]) == 4 and p.free_pages == 0


def test_random_plans_preserve_mru_set_relative_order_and_optimal_page_choice():
    rng = random.Random(9)
    for _ in range(250):
        n = rng.randrange(1, 9)
        k = 4
        topk = rng.randrange(1, n * k + 1)
        m = rng.randrange((topk + k - 1) // k, n + 1)
        tokens = [i if rng.random() < 0.8 else -1 for i in range(n * k)]
        lru = list(range(n * k))
        rng.shuffle(lru)
        plan = plan_kvduo_hot_merge(tokens, lru, k, topk, m)
        survivors = [i for i in lru if tokens[i] >= 0][-m * k :]
        assert [plan.tokens[i] for i in plan.lru_slots if plan.tokens[i] >= 0] == [
            tokens[i] for i in survivors
        ]
        counts = [sum(i // k == p for i in survivors) for p in range(n)]
        assert list(plan.kept_pages) == sorted(
            sorted(range(n), key=lambda p: (-counts[p], p))[:m]
        )
        assert sorted(plan.lru_slots) == list(range(m * k))


def test_prepared_tensor_plans_match_reference_and_leave_padded_inputs_unchanged():
    rng = random.Random(12)
    for _ in range(80):
        k = rng.choice([1, 4, 31])
        n = rng.randrange(1, 9)
        m = rng.randrange(1, n + 1)
        capacity = n * k
        tokens = torch.full((8 * k + 3,), 999, dtype=torch.int32)
        tokens[:capacity] = torch.tensor(
            [slot if rng.random() < 0.6 else -1 for slot in range(capacity)]
        )
        locs = torch.randperm(tokens.numel(), dtype=torch.int32) + 100
        order = list(range(capacity))
        rng.shuffle(order)
        lru = torch.full((8 * k,), -1, dtype=torch.int16)
        lru[:capacity] = torch.tensor(order, dtype=torch.int16)
        before = [bank.clone() for bank in (tokens, locs, lru)]
        expected = plan_kvduo_hot_merge(tokens[:capacity].tolist(), order, k, 1, m)
        plan = prepare_kvduo_hot_merge(tokens, locs, lru, capacity, k, 1, m)
        assert plan.kept_pages == expected.kept_pages
        assert plan.released_pages == expected.released_pages
        assert plan.tokens[: m * k].tolist() == list(expected.tokens)
        assert plan.lru_slots[: m * k].tolist() == list(expected.lru_slots)
        assert plan.source_locations.tolist() == [
            int(locs[s]) for s in expected.source_slots
        ]
        assert plan.destination_locations.tolist() == [
            int(locs[s]) for s in expected.destination_slots
        ]
        assert plan.locations[: m * k].tolist() == [
            int(locs[page * k + offset])
            for page in expected.kept_pages
            for offset in range(k)
        ]
        assert all(
            torch.all(bank[m * k :] == -1)
            for bank in (plan.tokens, plan.locations, plan.lru_slots)
        )
        assert all(
            torch.equal(bank, saved) for bank, saved in zip((tokens, locs, lru), before)
        )


@pytest.mark.parametrize(
    "status,match",
    [(1, "LRU slot permutation"), (2, "invalid physical slot locations")],
)
def test_gpu_plan_reads_only_summary_and_rejects_invalid_metadata_before_copy(
    monkeypatch, status, match
):
    import sglang.kernels.ops.kvcache.hisparse as ops

    c, p = make_merge_coordinator({(0, 0): 4})
    rows = (
        c.req_device_buffer_tokens[0, 0],
        c.req_device_buffer_token_locs[0, 0],
        c.lru_slots[0, 0],
    )
    before = [row.clone() for row in rows]
    staged = (
        torch.empty_like(rows[0]),
        torch.empty_like(rows[1]),
        torch.empty_like(rows[2]),
        torch.empty(16, dtype=torch.int64),
        torch.empty(16, dtype=torch.int64),
    )
    summary = torch.tensor([status, 0, 3], dtype=torch.int32)
    launch = MagicMock(return_value=(summary, *staged))
    monkeypatch.setattr(ops, "build_kvduo_hot_merge_plan", launch)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    cpu = torch.Tensor.cpu
    reads = []

    def read_summary(tensor, *args, **kwargs):
        assert tensor is summary
        reads.append(tensor)
        return cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", read_summary)
    c._move_kvduo_hot_entries = MagicMock()
    with pytest.raises((ValueError, RuntimeError), match=match):
        prepare_kvduo_hot_merge(*rows, 16, 4, 2, 2)
    assert len(reads) == 1 and launch.call_count == 1
    assert all(torch.equal(row, saved) for row, saved in zip(rows, before))
    c._move_kvduo_hot_entries.assert_not_called()
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_not_called()
    assert p.free_pages == 0 and c.kvduo_req_hot_capacity.item() == 16


def test_gpu_plan_returns_device_metadata_and_only_small_host_ownership_result(
    monkeypatch,
):
    import sglang.kernels.ops.kvcache.hisparse as ops

    tokens = torch.arange(34, dtype=torch.int32)
    locs = torch.arange(34, dtype=torch.int32) + 100
    lru = torch.arange(32, dtype=torch.int16)
    summary = torch.tensor([0, 1, 10], dtype=torch.int32)
    outputs = (
        torch.full_like(tokens, -1),
        torch.full_like(locs, -1),
        torch.full_like(lru, -1),
    )
    sources = torch.tensor([101] * 16, dtype=torch.int64)
    destinations = torch.tensor([107] * 16, dtype=torch.int64)
    monkeypatch.setattr(
        ops,
        "build_kvduo_hot_merge_plan",
        MagicMock(return_value=(summary, *outputs, sources, destinations)),
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    cpu = torch.Tensor.cpu

    def read_summary(tensor):
        assert tensor is summary and tensor.numel() == 3
        return cpu(tensor)

    monkeypatch.setattr(torch.Tensor, "cpu", read_summary)
    plan = prepare_kvduo_hot_merge(tokens, locs, lru, 16, 4, 2, 2)
    assert plan.kept_pages == (1, 3) and plan.released_pages == (0, 2)
    assert (
        plan.tokens is outputs[0]
        and plan.locations is outputs[1]
        and plan.lru_slots is outputs[2]
    )
    assert plan.source_locations.tolist() == [101]
    assert plan.destination_locations.tolist() == [107]


def test_gpu_merge_dispatch_compiles_real_lru_width_and_uses_global_workspace(
    monkeypatch,
):
    import sglang.kernels.ops.kvcache.hisparse as ops

    k = 78 * 64
    capacity = 4 * k
    tokens = torch.empty(8 * k + 64, dtype=torch.int32)
    locs = torch.empty_like(tokens)
    lru = torch.empty(8 * k, dtype=torch.int32)
    module = MagicMock()
    load = MagicMock(return_value=module)
    monkeypatch.setattr(ops, "_jit_kvduo_merge_module", load)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    outputs = ops.build_kvduo_hot_merge_plan(tokens, locs, lru, capacity, k, 2)
    load.assert_called_once_with(39936, 256)
    args = module.plan.call_args.args
    assert args[:3] == (tokens, locs, lru)
    assert args[8].numel() == 4 * capacity + 65536
    assert args[9].numel() == 3
    assert args[10:] == (capacity, k, 2, 65536, torch.iinfo(torch.int32).max)
    assert outputs[1].shape == tokens.shape and outputs[3].shape == lru.shape
