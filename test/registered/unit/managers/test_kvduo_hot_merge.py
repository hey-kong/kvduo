"""Exercise the production merge policy, payload movement, and free-page loop."""

import random
from unittest.mock import MagicMock

import pytest
import torch
from sglang.srt.mem_cache.sparsity.core.kvduo_frequency import (
    pack_frequency,
    plan_frequency_merge,
    unpack_frequency,
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
        buffer = coordinator.kvduo_hot_buffers[layer, req]
        buffer.tokens.copy_(
            torch.arange(req * 100 + 10, req * 100 + 10 + capacity, dtype=torch.int32)
        )
        buffer.frequency.copy_(
            pack_frequency([3] * min(2, pages) * 4 + [0] * max(0, pages - 2) * 4)
        )
        seed_payloads(coordinator, layer, req)
    physical.free_pages = 0
    coordinator.token_to_kv_pool_allocator.free_hisparse_indices.reset_mock()
    return coordinator, physical


def seed_payloads(coordinator, layer, req):
    buffer = coordinator.kvduo_hot_buffers[layer, req]
    coordinator.mem_pool_device.kv_buffer[layer][buffer.locations.long()] = (
        buffer.tokens.long()[:, None] * 10 + torch.arange(4)
    )


def assert_payloads_match_tokens(coordinator, layer, req):
    buffer = coordinator.kvduo_hot_buffers[layer, req]
    valid = buffer.tokens >= 0
    assert torch.equal(
        coordinator.mem_pool_device.kv_buffer[layer][buffer.locations[valid].long()],
        buffer.tokens[valid].long()[:, None] * 10 + torch.arange(4),
    )


@pytest.mark.parametrize("pages", [2, 4, 8, 12, 31])
def test_merge_retains_only_threes_and_moves_complete_payloads(pages):
    c, physical = make_merge_coordinator({(0, 0): pages})
    buffer = c.kvduo_hot_buffers[0, 0]
    frequencies = [(i * 7 + i // 4) % 4 for i in range(pages * 4)]
    buffer.frequency.copy_(pack_frequency(frequencies))
    survivors = sorted(int(t) for t, f in zip(buffer.tokens, frequencies) if f == 3)
    pointer = c.kvduo_hot_pointers.data_ptr()
    released = c._merge_kvduo_hot_pages(0, 0)
    new = c.kvduo_hot_buffers[0, 0]
    assert sorted(new.tokens[new.tokens >= 0].tolist()) == survivors
    assert unpack_frequency(new.frequency, new.tokens.numel()) == [
        int(t >= 0) for t in new.tokens
    ]
    assert new.tokens.numel() == max(c.min_hot_pages, (len(survivors) + 3) // 4) * 4
    assert released == (pages - new.tokens.numel() // 4) * 2
    assert c.kvduo_hot_pointers.data_ptr() == pointer
    assert_payloads_match_tokens(c, 0, 0)
    assert len(c._kvduo_hot_page_owners) == new.tokens.numel() // 4


def test_hand_revisits_after_zero_release_decay_and_stops_at_minimum():
    c, physical = make_merge_coordinator({(0, 0): 2, (0, 1): 2})
    assert c.reclaim_kvduo_hot_pages(2) == 2
    assert c.kvduo_req_hot_capacity.tolist() == [[4, 8]]
    assert c._kvduo_hand == 1
    # No eligible capacity beyond the minimum: return instead of spinning.
    assert c.reclaim_kvduo_hot_pages(100) == 2
    assert c.reclaim_kvduo_hot_pages(100) == 0


def test_hand_uses_hot_join_order_then_layer_and_skips_protected():
    c, p = make_merge_coordinator(
        {(0, 0): 3, (1, 0): 3, (0, 1): 3, (1, 1): 3}, layers=2
    )
    c._kvduo_hot_request_order = [1, 0]
    c._kvduo_hot_pressure_protected.add((0, 1, c._kvduo_request_generation[1]))
    assert c.reclaim_kvduo_hot_pages(2) == 2
    assert c.kvduo_req_hot_capacity.tolist() == [[12, 12], [12, 8]]
    assert c._kvduo_hand == 2


def test_growth_preserves_frequency_and_initializes_new_slots_zero():
    c, p = make_merge_coordinator({(0, 0): 2})
    before = unpack_frequency(c.kvduo_hot_buffers[0, 0].frequency, 8)
    p.free_pages = 10
    c._ensure_kvduo_hot_capacity_targets_batch({(0, 0): 40})
    buffer = c.kvduo_hot_buffers[0, 0]
    assert unpack_frequency(buffer.frequency, 40) == before + [0] * 32
    assert buffer.tokens[8:].tolist() == [-1] * 32


def test_request_reuse_clears_pointers_ownership_and_hand_entry():
    c, p = make_merge_coordinator({(0, 0): 3, (0, 1): 3})
    c._release_kvduo_layer_hot_pages(0)
    assert (0, 0) not in c.kvduo_hot_buffers
    assert not c.kvduo_hot_pointers[:, 0].any()
    assert c._kvduo_hot_request_order == [1]
    assert c._kvduo_request_generation[0] == 1
    assert all(owner != (0, 0) for owner in c._kvduo_hot_page_owners.values())


@pytest.mark.parametrize("free,events", [(0, 0), (0, 1), (4, 1), (5, 0)])
def test_optional_growth_does_not_merge_for_prefill_protection(free, events):
    c, p = make_merge_coordinator({(0, 0): 2, (0, 1): 3})
    p.free_pages = free
    c._kvduo_next_prefill_pages = 5
    c.kvduo_hot_growth_events[0, 0] = events
    c.reclaim_kvduo_hot_pages = MagicMock(wraps=c.reclaim_kvduo_hot_pages)
    c.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    c.reclaim_kvduo_hot_pages.assert_not_called()
    assert c.kvduo_req_hot_capacity.tolist() == [[8, 12]]


@pytest.mark.parametrize("seed", range(20))
def test_frequency_plan_randomized(seed):
    rng = random.Random(seed)
    pages, k = rng.randint(1, 24), 16
    tokens = [i if rng.randrange(4) else -1 for i in range(pages * k)]
    freq = [rng.randrange(4) for _ in tokens]
    plan = plan_frequency_merge(tokens, freq, k, 1)
    survivors = [t for t, f in zip(tokens, freq) if t >= 0 and f == 3]
    assert sorted(t for t in plan.tokens if t >= 0) == survivors
    assert set(plan.kept_pages).isdisjoint(plan.released_pages)
    assert len(plan.kept_pages) + len(plan.released_pages) == pages
    assert len(plan.sources) == len(plan.destinations)
    assert set(plan.sources).isdisjoint(plan.destinations)
    assert all(freq[s] == 3 and tokens[s] >= 0 for s in plan.sources)
    assert all(tokens[d] < 0 or freq[d] != 3 for d in plan.destinations)
    assert unpack_frequency(pack_frequency(freq), len(freq)) == freq


@pytest.mark.parametrize("state", ["minimum", "protected", "empty"])
def test_hand_batches_ineligible_tp_scan(monkeypatch, state):
    c, _ = make_coordinator(layers=21, requests=32)
    c.tp_world_size = 4
    c.tp_group = object()
    if state == "empty":
        c._kvduo_hot_request_order = []
    for req in range(32):
        for layer in range(21):
            owner = (layer, req)
            c._kvduo_req_layer_pages[owner] = [2] * (
                c.min_hot_pages + (state == "protected")
            )
            c._kvduo_hot_generation[owner] = c._kvduo_request_generation[req]
            if state == "protected":
                c._kvduo_hot_pressure_protected.add((layer, req, 0))

    def gather(values, value, **kwargs):
        values[:] = [value] * 4

    collective = MagicMock(side_effect=gather)
    monkeypatch.setattr(torch.distributed, "all_gather_object", collective)
    c._merge_kvduo_hot_pages = MagicMock()
    assert c.reclaim_kvduo_hot_pages(2) == 0
    assert collective.call_count == 1
    c._merge_kvduo_hot_pages.assert_not_called()


def test_hand_batches_scan_but_retains_each_merge_consensus(monkeypatch):
    c, _ = make_merge_coordinator({(0, 0): 2, (0, 1): 2})
    c.tp_world_size = 4
    c.tp_group = object()

    def gather(values, value, **kwargs):
        values[:] = [value] * 4

    collective = MagicMock(side_effect=gather)
    monkeypatch.setattr(torch.distributed, "all_gather_object", collective)
    # First rotation ages both objects without releasing pages; the second
    # releases one page from the first object and stops at the original Hand.
    assert c.reclaim_kvduo_hot_pages(2) == 2
    assert collective.call_count == 4  # One snapshot plus three merge plans.
    assert c._kvduo_hand == 1


@pytest.mark.parametrize("initial_free_tokens", [0, 64])
def test_hand_reclaims_same_amount_with_different_tp_free_space(
    monkeypatch, initial_free_tokens
):
    transcript = []
    for rank_free_tokens in (0, initial_free_tokens):
        c, physical = make_merge_coordinator({(0, 0): 20, (0, 1): 20})
        physical.free_pages = rank_free_tokens // c.page_size
        physical.available_size.reset_mock()
        c.tp_world_size = 2
        c.tp_group = object()
        observed = []

        def gather(values, value, **kwargs):
            index = len(observed)
            observed.append(value)
            peer = transcript[index] if transcript else value
            values[:] = [value, peer]

        collective = MagicMock(side_effect=gather)
        monkeypatch.setattr(torch.distributed, "all_gather_object", collective)
        assert c.reclaim_kvduo_hot_pages(64) == 72
        assert physical.free_pages * c.page_size == rank_free_tokens + 72
        assert c.kvduo_req_hot_capacity.tolist() == [[8, 8]]
        assert c._kvduo_hand == 0
        assert collective.call_count == 3  # One snapshot and two merge plans.
        physical.available_size.assert_not_called()
        if transcript:
            assert observed == transcript
        else:
            transcript = observed


def test_hand_snapshot_disagreement_does_not_mutate_or_merge(monkeypatch):
    c, _ = make_merge_coordinator({(0, 0): 2})
    c.tp_world_size = 2
    c.tp_group = object()

    def gather(values, value, **kwargs):
        values[:] = [value, (0, 0, 1, ())]  # Other rank has no candidates.

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    c._merge_kvduo_hot_pages = MagicMock()
    assert c.reclaim_kvduo_hot_pages(2) == 0
    c._merge_kvduo_hot_pages.assert_not_called()
    assert c._kvduo_hand == 0


def test_all_growth_applicants_can_donate_and_shrunk_event_is_cancelled():
    c, p = make_merge_coordinator({(0, 0): 4, (0, 1): 4})
    for b in c.kvduo_hot_buffers.values():
        b.frequency.zero_()
    c.kvduo_hot_growth_events.fill_(1)
    c.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    assert c.kvduo_req_hot_capacity.tolist() == [[4, 20]]
    assert p.free_pages == 2
    assert c._kvduo_hand == 1


def test_growth_respects_actual_inflight_protection():
    c, p = make_merge_coordinator({(0, 0): 4, (0, 1): 4})
    c._kvduo_hot_pressure_protected = {(0, 0, 0), (0, 1, 0)}
    c.kvduo_hot_growth_events.fill_(1)
    c.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    assert c.kvduo_req_hot_capacity.tolist() == [[16, 16]]
    assert p.free_pages == 0


def test_geometric_metadata_reuses_storage_and_clears_retried_suffix():
    from sglang.srt.mem_cache.sparsity.core.kvduo_frequency import FrequencyBuffer

    b = FrequencyBuffer.empty(torch.arange(16, dtype=torch.int32))
    b.tokens.copy_(torch.arange(16, dtype=torch.int32))
    b.frequency.copy_(pack_frequency([3] * 16))
    grown = b.grown(torch.arange(16, 20, dtype=torch.int32))
    assert grown._storage[0].numel() == 32
    pointers = grown.pointers()
    next_buffer = grown.grown(torch.arange(20, 24, dtype=torch.int32))
    assert next_buffer.pointers() == pointers
    assert next_buffer.tokens[:16].tolist() == list(range(16))
    assert unpack_frequency(next_buffer.frequency, 24) == [3] * 16 + [0] * 8
    # A rolled-back growth can leave stale data beyond the published capacity.
    next_buffer.tokens[20:].fill_(99)
    next_buffer.frequency.copy_(pack_frequency([3] * 24))
    retried = grown.grown(torch.arange(40, 44, dtype=torch.int32))
    assert retried.tokens[20:].tolist() == [-1] * 4
    assert retried.locations[20:].tolist() == list(range(40, 44))
    assert unpack_frequency(retried.frequency, 24) == [3] * 20 + [0] * 4


def test_growth_publishes_all_owners_once_and_reuses_reserved_arrays(monkeypatch):
    from sglang.srt.mem_cache.sparsity.core.kvduo_frequency import FrequencyBuffer

    c, p = make_merge_coordinator(
        {(l, r): 2 for l in range(2) for r in range(2)}, layers=2
    )
    p.free_pages = 16
    publish = MagicMock(wraps=c._publish_kvduo_frequency_buffers)
    c._publish_kvduo_frequency_buffers = publish
    c._ensure_kvduo_hot_capacity_targets_batch(
        {owner: 12 for owner in c.kvduo_hot_buffers}, allow_reclaim=False
    )
    assert publish.call_count == 1
    assert len(publish.call_args.args[0]) == 4
    pointers = {owner: b.pointers() for owner, b in c.kvduo_hot_buffers.items()}
    monkeypatch.setattr(
        FrequencyBuffer,
        "empty",
        MagicMock(side_effect=AssertionError("unnecessary metadata allocation")),
    )
    c._ensure_kvduo_hot_capacity_targets_batch(
        {owner: 16 for owner in c.kvduo_hot_buffers}, allow_reclaim=False
    )
    assert publish.call_count == 2
    for owner, b in c.kvduo_hot_buffers.items():
        assert b.pointers() == pointers[owner]
        assert c.kvduo_hot_pointers[owner].tolist() == b.pointers()
        assert c.kvduo_req_hot_capacity_gpu[owner].item() == 16
