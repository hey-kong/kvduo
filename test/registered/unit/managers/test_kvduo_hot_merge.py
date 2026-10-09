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


@pytest.mark.parametrize(
    "visited_count,target", [(0, 1), (4, 1), (5, 2), (12, 3), (16, 4)]
)
def test_visited_count_drives_capacity_without_halving(visited_count, target):
    c, physical = make_merge_coordinator({(0, 0): 4})
    c.hot_visited[0, 0, :visited_count] = True
    expected = live_lru(c, 0, 0)[:visited_count]
    assert c._merge_kvduo_hot_pages(0, 0) == (4 - target) * c.page_size
    assert physical.free_pages == 4 - target
    assert int(c.kvduo_req_hot_capacity[0, 0]) == target * 4
    assert live_lru(c, 0, 0) == expected
    assert_payloads_match_tokens(c, 0, 0)
    assert not torch.any(c.hot_visited)
    assert len(c._kvduo_hot_base_pages[0, 0]) == target
    assert c._kvduo_hot_processed[0, 0] == 1


def test_merge_keeps_visited_even_if_lru_and_minimizes_payload_moves_before_free():
    c, physical = make_merge_coordinator({(0, 0): 4, (1, 0): 2}, layers=2)
    survivors = [5, 1, 12, 6, 13, 4, 14, 7]
    c.lru_slots[0, 0, :16] = torch.tensor(
        survivors + [0, 2, 3, 8, 9, 10, 11, 15], dtype=torch.int16
    )
    c.hot_visited[0, 0, survivors] = True
    pages = list(c._kvduo_req_layer_pages[0, 0])
    expected = live_lru(c, 0, 0)[:8]
    other = c.req_device_buffer_tokens[1].clone()
    c._move_kvduo_hot_entries = MagicMock(wraps=c._move_kvduo_hot_entries)
    free = c.token_to_kv_pool_allocator.free_hisparse_indices.side_effect

    def check(indices):
        assert c._kvduo_req_layer_pages[0, 0] == [pages[1], pages[3]]
        assert live_lru(c, 0, 0) == expected
        assert_payloads_match_tokens(c, 0, 0)
        assert not torch.any(c.hot_visited[0, 0])
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


def test_minimum_capacity_is_still_cleaned_and_no_release_rebuilds_base():
    c, physical = make_merge_coordinator({(0, 0): 1})
    c.hot_visited[0, 0, 2] = True
    c.kvduo_prefill_ema[0, 0] = torch.tensor([5, 7])
    assert c.reclaim_kvduo_hot_pages(2) == 0
    assert live_lru(c, 0, 0) == [12]
    assert c.lru_slots[0, 0, :4].tolist() == [0, 1, 3, 2]
    assert not torch.any(c.hot_visited)
    assert not torch.any(c.kvduo_prefill_ema)
    assert c._kvduo_hand_sequence == 1
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_not_called()


def test_hand_wraps_and_second_visit_reclaims_after_zero_release_first_rotation():
    c, physical = make_merge_coordinator(
        {(layer, req): 2 for req in range(2) for layer in range(2)}, layers=2
    )
    c.hot_visited[:, :, :8] = True
    merge = c._merge_kvduo_hot_pages
    calls = []

    def trace(layer, req):
        calls.append((layer, req))
        return merge(layer, req)

    c._merge_kvduo_hot_pages = trace
    assert c.reclaim_kvduo_hot_pages(2) == 2
    assert calls == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 0)]
    assert c._kvduo_hand == 1
    assert c._kvduo_hand_sequence == 5
    assert not live_lru(c, 0, 0)
    # The next invocation resumes at the following layer, not slot zero.
    assert c.reclaim_kvduo_hot_pages(2) == 2
    assert calls[-1] == (1, 0)
    assert c._kvduo_hand == 2


def test_hand_skips_protected_staging_applicants_and_limited_requests_without_clearing_flags():
    c, physical = make_merge_coordinator({(0, req): 2 for req in range(5)})
    c.hot_visited[0, :, 0] = True
    c._kvduo_hot_pressure_protected.add((0, 1, 0))
    c._active_kvduo_reqs[2] = SimpleNamespace(hisparse_staging=True)
    c._kvduo_prefill_limited_requests.add(c._kvduo_request_key(3))
    assert c.reclaim_kvduo_hot_pages(2, excluded_owners={(0, 0)}) == 2
    assert torch.all(c.hot_visited[0, :4, 0])
    assert c._kvduo_hot_processed == {(0, 4): 1}
    assert c._kvduo_hand == 0


def test_stale_generation_and_all_protected_finish_without_no_progress_loop():
    c, physical = make_merge_coordinator({(0, 0): 4})
    c.hot_visited[0, 0, :16] = True
    c._kvduo_request_generation[0] += 1
    assert c.reclaim_kvduo_hot_pages(2) == 0
    assert torch.all(c.hot_visited[0, 0, :16])
    assert not c._kvduo_hot_processed
    c._kvduo_hot_generation[0, 0] = c._kvduo_request_generation[0]
    c._kvduo_hot_pressure_protected.add((0, 0, 0))
    assert c.reclaim_kvduo_hot_pages(2) == 0
    assert not c._kvduo_hot_processed


def test_failed_payload_move_never_publishes_or_releases_source_pages():
    c, physical = make_merge_coordinator({(0, 0): 4})
    # One survivor in each page forces a retained/released-page move.
    c.hot_visited[0, 0, [0, 4, 8, 12]] = True
    before = c.req_device_buffer_tokens.clone()
    c._move_kvduo_hot_entries = MagicMock(side_effect=RuntimeError("copy failed"))
    with pytest.raises(RuntimeError, match="copy failed"):
        c._merge_kvduo_hot_pages(0, 0)
    assert torch.equal(before, c.req_device_buffer_tokens)
    assert len(c._kvduo_req_layer_pages[0, 0]) == 4
    assert physical.free_pages == 0
    assert not c._kvduo_hot_base_pages


def test_random_merge_plans_preserve_exact_visited_set_lru_and_page_selection():
    rng = random.Random(9)
    for _ in range(250):
        n = rng.randrange(1, 9)
        k = 4
        top_k = rng.randrange(1, n * k + 1)
        tokens = [i if rng.random() < 0.8 else -1 for i in range(n * k)]
        visited = [rng.random() < 0.6 for i in range(n * k)]
        lru = list(range(n * k))
        rng.shuffle(lru)
        plan = plan_kvduo_hot_merge(tokens, lru, k, top_k, visited)
        survivors = [i for i in lru if tokens[i] >= 0 and visited[i]]
        assert len(plan.kept_pages) == max(
            (top_k + k - 1) // k, (len(survivors) + k - 1) // k
        )
        assert [plan.tokens[i] for i in plan.lru_slots if plan.tokens[i] >= 0] == [
            tokens[i] for i in survivors
        ]
        counts = [sum(i // k == p for i in survivors) for p in range(n)]
        expected = sorted(
            sorted(range(n), key=lambda p: (-counts[p], p))[: len(plan.kept_pages)]
        )
        assert list(plan.kept_pages) == expected
        assert all(i // k in plan.released_pages for i in plan.source_slots)
        assert all(i // k in plan.kept_pages for i in plan.destination_slots)
        assert sorted(plan.lru_slots) == list(range(len(plan.tokens)))
