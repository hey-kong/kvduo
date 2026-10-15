from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.kernels.ops.kvcache.kvduo_full_reclaim import (
    select_kvduo_full_pages,
    select_kvduo_full_requests,
    summarize_kvduo_full_pages,
)
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache.sparsity.core.kvduo_prefix_cache import (
    HostPrefixRecord,
    KVDuoHostPrefixCache,
)
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_pressure_admission import make_full_coordinator

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_request_selection_uses_net_gain_then_decode_age_and_stable_owner():
    owners = [0] * 6 + [1] * 5 + [2] * 4
    valid = torch.ones(len(owners), dtype=torch.int32)
    # Owner 1 releases fewer pages than 0, but needs no new HOT minimum.
    assert select_kvduo_full_requests(valid, owners, [2, 0, 1], [1, 2, 3], 1) == list(
        range(6, 11)
    )
    # All three have the same net gain; earlier decode admission wins.
    assert select_kvduo_full_requests(valid, owners, [3, 2, 1], [3, 2, 1], 1) == list(
        range(11, 15)
    )
    assert select_kvduo_full_requests(valid, owners, [3, 2, 1], [1, 1, 1], 1) == list(
        range(6)
    )


@pytest.mark.parametrize("release_all", [False, True])
def test_request_selection_preserves_validity_and_releases_complete_owner_sets(
    release_all,
):
    generator = torch.Generator().manual_seed(712)
    owners = torch.randint(0, 9, (521,), generator=generator).tolist()
    for _ in range(50):
        valid = (torch.rand(521, generator=generator) > 0.25).int()
        costs = torch.randint(0, 65, (9,), generator=generator).tolist()
        age = torch.randint(0, 3, (9,), generator=generator).tolist()
        mixed = (torch.rand(9, generator=generator) > 0.5).tolist()
        need = int(torch.randint(1, 300, (), generator=generator))
        counts = Counter(owner for owner, ok in zip(owners, valid.tolist()) if ok)
        ordered = sorted(
            (owner for owner in counts if counts[owner] > costs[owner]),
            key=lambda owner: (
                not mixed[owner],
                costs[owner] - counts[owner],
                age[owner],
                owner,
            ),
        )
        chosen, net = set(), 0
        for owner in ordered:
            if not release_all and net >= need:
                break
            chosen.add(owner)
            net += counts[owner] - costs[owner]
        expected = [
            i
            for i, (owner, ok) in enumerate(zip(owners, valid.tolist()))
            if ok and owner in chosen
        ]
        assert (
            select_kvduo_full_requests(
                valid,
                owners,
                costs,
                age,
                need,
                release_all=release_all,
                mixed_owners=mixed,
            )
            == expected
        )


def make_request_reclaim_coordinator(counts, *, layers=1, ratio=1, policy="request"):
    c, physical, prototype = make_full_coordinator(
        pages=max(counts) + 2, layers=layers, ratio=ratio, full_reclaim_policy=policy
    )
    width = (max(counts) + 2) * c.page_size
    requests = len(counts)
    c.req_to_token_pool.req_to_token = torch.arange(
        requests * width * ratio, dtype=torch.int32
    ).view(requests, -1)
    mapping = torch.arange(requests * width) + c.page_size
    c.mem_pool_device.full_to_hisparse_device_index_mapping = mapping
    c.full_data_version = torch.ones_like(mapping)
    c.full_host_version = torch.ones_like(mapping)
    c.full_last_touch = torch.arange(mapping.numel())
    c.kvduo_req_hot_capacity = torch.zeros((layers, requests), dtype=torch.int64)
    c._kvduo_host_valid_len = [(n + 2) * c.page_size for n in counts]
    c._mixed_slots = [False] * requests
    c._kvduo_request_generation = [1] * requests
    c._kvduo_decode_order = {i: (1, i + 1) for i in range(requests)}
    c._kvduo_pending_hot_minimum.clear()
    c._active_kvduo_reqs = {}
    for i, n in enumerate(counts):
        req = SimpleNamespace(**vars(prototype))
        req.req_pool_idx = i
        req.kv = SimpleNamespace(kv_allocated_len=(n + 2) * c.page_size * ratio)
        req.kvduo_radix_insert_len = req.kv.kv_allocated_len
        c._active_kvduo_reqs[i] = req
    c._kvduo_consensus = MagicMock(return_value=True)
    return c, physical, width


@pytest.mark.parametrize("ratio", [1, 4])
def test_request_reclaim_releases_largest_owner_once_and_funds_all_layer_minima(
    ratio, monkeypatch
):
    c, physical, width = make_request_reclaim_coordinator([3, 6], layers=2, ratio=ratio)
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    original = mapping.clone()
    # Request selection needs counts, not page-age sorting.
    monkeypatch.setattr(
        torch, "argsort", MagicMock(side_effect=AssertionError("page sort"))
    )
    assert (
        c.reclaim_kvduo_full_pages(c.page_size, reserve_hot_minimum=True)
        == 6 * c.page_size
    )
    assert torch.equal(mapping[:width], original[:width])
    assert torch.all(mapping[width : width + 6 * c.page_size] == 0)
    assert torch.equal(
        mapping[width + 6 * c.page_size :], original[width + 6 * c.page_size :]
    )
    assert c._mixed_slots == [False, True]
    assert c._kvduo_pending_hot_minimum == {1}
    assert c._kvduo_mandatory_hot_pages() == 2
    assert physical.free_pages == 6
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_called_once()


@pytest.mark.parametrize(
    "policy,expected_pages,expected_mixed",
    [
        ("request", 6, [False, True]),
        ("page", 2, [True, False]),
    ],
)
def test_full_reclaim_policy_switch_preserves_page_lru(
    policy, expected_pages, expected_mixed
):
    c, _, _ = make_request_reclaim_coordinator([3, 6], policy=policy)
    assert (
        c.reclaim_kvduo_full_pages(c.page_size, reserve_hot_minimum=True)
        == expected_pages * c.page_size
    )
    assert c._mixed_slots == expected_mixed


def test_request_reclaim_stops_after_enough_net_space_and_skips_zero_gain():
    c, physical, _ = make_request_reclaim_coordinator([3, 5, 4])
    assert (
        c.reclaim_kvduo_full_pages(6 * c.page_size, reserve_hot_minimum=True)
        == 9 * c.page_size
    )
    assert c._mixed_slots == [False, True, True]
    assert physical.free_pages - c._kvduo_mandatory_hot_pages() == 7
    c, physical, _ = make_request_reclaim_coordinator([1, 2], layers=2)
    original = c.mem_pool_device.full_to_hisparse_device_index_mapping.clone()
    assert c.reclaim_kvduo_full_pages(c.page_size) == 0
    assert physical.free_pages == 0 and c._mixed_slots == [False, False]
    assert torch.equal(
        c.mem_pool_device.full_to_hisparse_device_index_mapping, original
    )


@pytest.mark.parametrize("stale", [False, True])
def test_request_reclaim_age_is_generation_bound(stale):
    c, _, _ = make_request_reclaim_coordinator([4, 4])
    c._kvduo_decode_order = {0: (1, 2), 1: (0 if stale else 1, 1)}
    assert c.reclaim_kvduo_full_pages(c.page_size) == 4 * c.page_size
    assert c._mixed_slots == ([True, False] if stale else [False, True])


def test_request_reclaim_preserves_prefix_tail_stale_host_and_protected_pages_before_free():
    c, _, width = make_request_reclaim_coordinator([8])
    req = c._active_kvduo_reqs[0]
    req.cache_protected_len = c.page_size
    c.full_host_version[2 * c.page_size] = 0
    c._kvduo_pressure_protected = torch.tensor([4 * c.page_size], dtype=torch.int32)
    # The last historical page has not completed Host backup yet.
    c._kvduo_host_valid_len[0] = 7 * c.page_size
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    original = mapping.clone()
    released = [
        i
        for page in (1, 3, 5, 6)
        for i in range(page * c.page_size, (page + 1) * c.page_size)
    ]
    untouched = [i for i in range(width) if i not in released]

    def free(indices):
        assert torch.all(mapping[released] == 0)
        assert torch.equal(mapping[untouched], original[untouched])
        assert indices.tolist() == original[released].tolist()

    c.token_to_kv_pool_allocator.free_hisparse_indices.side_effect = free
    assert c.reclaim_kvduo_full_pages(c.page_size) == len(released)
    assert req.kvduo_radix_insert_len == c.page_size


def test_request_reclaim_tp_disagreement_does_not_publish_or_free():
    c, physical, _ = make_request_reclaim_coordinator([3, 6])
    original = c.mem_pool_device.full_to_hisparse_device_index_mapping.clone()
    c._kvduo_consensus.return_value = False
    assert c.reclaim_kvduo_full_pages(c.page_size) == 0
    assert torch.equal(
        c.mem_pool_device.full_to_hisparse_device_index_mapping, original
    )
    assert c._mixed_slots == [False, False] and physical.free_pages == 0
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_not_called()


def test_decode_admission_order_is_cleared_on_finish_and_recreated_on_slot_reuse():
    c, _, req = make_full_coordinator(full_reclaim_policy="request")
    generation, sequence = c._kvduo_decode_order[0]
    c.request_finished(req)
    assert 0 not in c._kvduo_decode_order
    c.alloc_device_buffer(req)
    new_generation, new_sequence = c._kvduo_decode_order[0]
    assert new_generation > generation and new_sequence > sequence


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_full_summary_rejects_partial_mapping_stale_version_and_partial_protection(
    index_dtype,
):
    pages = torch.arange(12, dtype=index_dtype).view(6, 2)
    mapping = torch.arange(12) + 10
    mapping[3] = 0
    versions = torch.ones(12, dtype=torch.int64)
    host = versions.clone()
    host[5] = 0
    touches = torch.tensor([2, 8, 7, 1, 3, 6, 4, 9, 5, 0, 11, 10])
    originals = [bank.clone() for bank in (pages, mapping, versions, host, touches)]
    physical, scores, valid = summarize_kvduo_full_pages(
        pages, mapping, versions, host, touches, torch.tensor([7], dtype=index_dtype)
    )
    assert physical.tolist() == mapping[pages].tolist()
    assert scores.tolist() == [8, 7, 6, 9, 5, 11]
    assert valid.tolist() == [1, 0, 0, 0, 1, 1]
    assert all(
        torch.equal(bank, saved)
        for bank, saved in zip((pages, mapping, versions, host, touches), originals)
    )


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_full_summary_rejects_invalid_logical_addresses_without_indexing_them(
    index_dtype,
):
    pages = torch.tensor([[-1, 0], [2, 3], [99, -2]], dtype=index_dtype)
    mapping = torch.arange(4) + 10
    versions = torch.ones(4, dtype=torch.int64)
    _, scores, valid = summarize_kvduo_full_pages(
        pages, mapping, versions, versions, torch.arange(4)
    )
    assert valid.tolist() == [0, 1, 0]
    assert scores.tolist() == [0, 3, torch.iinfo(torch.int64).min]


@pytest.mark.parametrize("release_all", [False, True])
def test_device_selection_matches_lru_policy_and_one_time_hot_minimum_charges(
    release_all,
):
    generator = torch.Generator().manual_seed(321)
    owners = [0] * 13 + [1] * 8 + [2] * 15 + [3] * 5
    for _ in range(100):
        scores = torch.randint(0, 12, (len(owners),), generator=generator)
        valid = torch.rand(len(owners), generator=generator) > 0.25
        costs = torch.randint(0, 9, (4,), generator=generator).tolist()
        need = int(torch.randint(1, 20, (), generator=generator))
        counts = Counter(owner for owner, ok in zip(owners, valid.tolist()) if ok)
        order = sorted(
            (
                i
                for i, ok in enumerate(valid.tolist())
                if ok and counts[owners[i]] > costs[owners[i]]
            ),
            key=lambda i: (int(scores[i]), i),
        )
        expected, charged, net = [], set(), 0
        for i in order:
            if not release_all and net >= need:
                break
            expected.append(i)
            net += 1
            if owners[i] not in charged:
                charged.add(owners[i])
                net -= costs[owners[i]]
        assert (
            select_kvduo_full_pages(
                scores, valid, owners, costs, need, release_all=release_all
            )
            == expected
        )


def test_full_selection_zero_gain_and_maximum_clock_ties():
    maximum = torch.iinfo(torch.int64).max
    scores = torch.tensor([maximum, 1, maximum, maximum, maximum])
    valid = torch.tensor([0, 1, 1, 1, 1])
    # Owner 0 cannot fund its own minimum; invalid pages cannot affect ties.
    assert select_kvduo_full_pages(scores, valid, [0, 0, 1, 1, 1], [1, 1], 1) == [2, 3]
    assert select_kvduo_full_pages(scores, valid, [0, 0, 1, 1, 1], [1, 3], 1) == []


def test_selection_download_is_bounded_by_need_and_mandatory_minima(monkeypatch):
    downloads = []
    original = torch.Tensor.cpu

    def download(tensor, *args, **kwargs):
        downloads.append(tensor.numel())
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", download)
    result = select_kvduo_full_pages(
        torch.arange(1000), torch.ones(1000), [0] * 1000, [2], 1
    )
    assert result == [0, 1, 2]
    assert downloads == [4]


def test_full_reclaim_keeps_request_page_ties_protection_and_publication_before_free():
    c, _, req0 = make_full_coordinator()
    count = c.full_last_touch.numel()
    req1 = SimpleNamespace(**vars(req0))
    req1.req_pool_idx = 1
    req0.cache_protected_len = 1
    c.req_to_token_pool.req_to_token = torch.arange(2 * count, dtype=torch.int32).view(
        2, -1
    )
    mapping = torch.arange(2 * count) + 2
    c.mem_pool_device.full_to_hisparse_device_index_mapping = mapping
    c.full_data_version = torch.ones(2 * count, dtype=torch.int64)
    c.full_host_version = c.full_data_version.clone()
    c.full_last_touch = torch.ones(2 * count, dtype=torch.int64)
    c.kvduo_req_hot_capacity = torch.zeros((1, 2), dtype=torch.int64)
    c._kvduo_host_valid_len = [count, count - 6]
    c._mixed_slots = [False, False]
    c._active_kvduo_reqs = {1: req1, 0: req0}
    c._kvduo_pressure_protected = torch.tensor([3], dtype=torch.int32)
    c.full_host_version[count + 1] = 0
    c._kvduo_consensus = MagicMock(return_value=True)
    released = list(range(4, 12)) + list(range(count + 2, count + 10))
    original_mapping = mapping.clone()

    def free(indices):
        assert torch.all(mapping[released] == 0)
        assert indices.tolist() == original_mapping[released].tolist()

    c.token_to_kv_pool_allocator.free_hisparse_indices.side_effect = free
    assert (
        c.reclaim_kvduo_full_pages(2, reserve_hot_minimum=True, release_all=True) == 16
    )
    c._kvduo_consensus.assert_called_once_with(
        tuple(
            [(0, offset) for offset in range(4, 12, 2)]
            + [(1, offset) for offset in range(2, 10, 2)]
        )
    )
    untouched = [i for i in range(2 * count) if i not in released]
    assert torch.equal(mapping[untouched], original_mapping[untouched])
    assert c._kvduo_pending_hot_minimum == {0, 1}
    assert c._mixed_slots == [True, True]


def test_full_plan_tp_rejection_leaves_all_residency_untouched():
    c, _, _ = make_full_coordinator()
    original = c.mem_pool_device.full_to_hisparse_device_index_mapping.clone()
    c._kvduo_consensus = MagicMock(return_value=False)
    assert c.reclaim_kvduo_full_pages(2) == 0
    assert torch.equal(
        c.mem_pool_device.full_to_hisparse_device_index_mapping, original
    )
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_not_called()
    assert not c._mixed_slots[0]


@pytest.mark.parametrize("ratio", [1, 4])
def test_full_reclaim_accepts_real_int32_request_table_for_generic_and_c4(ratio):
    c, _, _ = make_full_coordinator(ratio=ratio)
    c.req_to_token_pool.req_to_token = c.req_to_token_pool.req_to_token.int()
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    original = mapping.clone()
    assert c.reclaim_kvduo_full_pages(4) == 4
    assert torch.all(mapping[:4] == 0)
    assert torch.equal(mapping[4:], original[4:])
    indices = c.token_to_kv_pool_allocator.free_hisparse_indices.call_args.args[0]
    assert indices.dtype == torch.int64
    assert torch.equal(indices, original[:4])


def prefix_coordinator():
    c = HiSparseCoordinator.__new__(HiSparseCoordinator)
    c.enable_mixed_residency = True
    c.device = "cpu"
    c.compress_ratio = 1
    c.page_size = 2
    c.host_prefix_cache = KVDuoHostPrefixCache(8)
    c.req_to_host_pool = torch.full((1, 8), -1, dtype=torch.int64)
    c.req_to_host_pool_allocated_len = torch.zeros(1, dtype=torch.int64)
    c._req_host_prefix_records = [set()]
    c.full_touch_clock = torch.tensor([77])
    req = SimpleNamespace(
        rid="prefix",
        req_pool_idx=0,
        extra_key=None,
        get_fill_ids=lambda: list(range(8)),
    )
    return c, req


def test_host_prefix_attachment_batches_locations_without_reading_gpu_clock():
    c, req = prefix_coordinator()
    identities = []
    for ordinal in range(3):
        identity = c._host_prefix_identity(req, ordinal)
        identities.append(identity)
        c.host_prefix_cache.insert(
            HostPrefixRecord(
                identity=identity,
                domains=("main_kv",),
                host_locations={"main_kv": (20 + 2 * ordinal, 21 + 2 * ordinal)},
                data_versions={"main_kv": (1, 1)},
                host_versions={"main_kv": (1, 1) if ordinal < 2 else (0, 1)},
            )
        )
    c.full_touch_clock = MagicMock()
    c.full_touch_clock.__getitem__.return_value = torch.tensor(77)
    assert c._attach_cached_host_prefix(req, 8) == 4
    assert c.req_to_host_pool[0].tolist() == [20, 21, 22, 23, -1, -1, -1, -1]
    c.full_touch_clock.__getitem__.assert_not_called()
    assert c._req_host_prefix_records[0] == set(identities[:2])
    for identity in identities[:2]:
        record = c.host_prefix_cache.records[identity]
        assert record.last_access == 1
        assert (req.rid, 0) in record.request_references


def test_host_prefix_retention_downloads_once_and_keeps_versions_touches(monkeypatch):
    c, req = prefix_coordinator()
    c._kvduo_host_valid_len = [7]
    c.req_to_host_pool[0] = torch.arange(20, 28)
    c.req_to_full_lookup = torch.arange(8).view(1, -1)
    c.full_host_version = torch.arange(10, 18)
    c.full_last_touch = torch.arange(30, 38)
    downloads = []
    original = torch.Tensor.cpu

    def download(tensor, *args, **kwargs):
        downloads.append(tensor.numel())
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", download)
    assert c._retain_request_host_prefix(req) == set(range(20, 26))
    assert downloads == [19]
    for ordinal in range(3):
        record = c.host_prefix_cache.records[c._host_prefix_identity(req, ordinal)]
        start = 2 * ordinal
        assert record.host_locations["main_kv"] == (20 + start, 21 + start)
        assert (
            record.data_versions["main_kv"]
            == record.host_versions["main_kv"]
            == (10 + start, 11 + start)
        )
        assert record.model_touches["main_kv"] == (30 + start, 31 + start)
        assert record.last_access == 77


def local_pressure_coordinator(counts, *, ratio=1):
    from types import MethodType

    c, physical, width = make_request_reclaim_coordinator(
        counts, ratio=ratio, policy="page"
    )
    c._reclaim_for_physical_allocation = MethodType(
        HiSparseCoordinator._reclaim_for_physical_allocation, c
    )
    c._kvduo_decode_host_slots = lambda requests: 0
    for req in c._active_kvduo_reqs.values():
        req.kv_committed_len = req.kv.kv_allocated_len + (ratio - 1)
    return c, physical, width


@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("mixed", [False, True])
def test_decode_prefers_own_lru_pages_over_colder_other_request(ratio, mixed):
    c, physical, width = local_pressure_coordinator([5, 5], ratio=ratio)
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    original = mapping.clone()
    c.full_last_touch[:width] = 0
    for page, clock in enumerate([500, 100, 400, 300, 200]):
        c.full_last_touch[
            width + page * c.page_size : width + (page + 1) * c.page_size
        ] = clock
    if mixed:
        c._mixed_slots[1] = True
        c.kvduo_req_hot_capacity[:, 1] = c.hot_page_size
    result = c.ensure_kvduo_decode_capacity([c._active_kvduo_reqs[1]])
    assert result.action.name == "SUCCESS"
    assert torch.equal(mapping[:width], original[:width])
    expected = [1] if mixed else [1, 4]
    for page in range(7):
        slots = slice(width + page * c.page_size, width + (page + 1) * c.page_size)
        assert torch.equal(
            mapping[slots],
            torch.zeros_like(mapping[slots]) if page in expected else original[slots],
        )
    assert physical.free_pages == len(expected)


@pytest.mark.parametrize("blocked", ["minimum", "host", "prefix"])
def test_decode_local_shortage_falls_back_to_global_lru(blocked):
    c, physical, width = local_pressure_coordinator(
        [5, 1 if blocked == "minimum" else 5]
    )
    req = c._active_kvduo_reqs[1]
    if blocked == "host":
        c.full_host_version[width:] = 0
    elif blocked == "prefix":
        req.cache_protected_len = req.kv.kv_allocated_len
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    original = mapping.clone()
    result = c.ensure_kvduo_decode_capacity([req])
    assert result.action.name == "SUCCESS"
    assert torch.equal(mapping[width:], original[width:])
    assert torch.all(mapping[: 2 * c.page_size] == 0)
    assert physical.free_pages == 2


def test_partial_local_release_recomputes_global_shortfall_with_hot_minima():
    c, physical, width = local_pressure_coordinator([5, 2])
    result = c._reclaim_for_physical_allocation(
        2 * c.page_size,
        reserve_hot_minimum=True,
        preferred_full_requests=((1, 2 * c.page_size),),
    )
    assert result.action.name == "SUCCESS"
    assert physical.free_pages == 4
    assert c._kvduo_mandatory_hot_pages() == 2
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    assert torch.all(mapping[: 2 * c.page_size] == 0)
    assert torch.all(mapping[width : width + 2 * c.page_size] == 0)


def test_decode_batch_assigns_local_reclaim_only_to_allocation_owners():
    c, physical, width = local_pressure_coordinator([5, 5, 5])
    c._active_kvduo_reqs[0].kv_committed_len -= 1
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    original = mapping.clone()
    result = c.ensure_kvduo_decode_capacity(list(c._active_kvduo_reqs.values()))
    assert result.action.name == "SUCCESS"
    assert torch.equal(mapping[:width], original[:width])
    for owner in (1, 2):
        assert torch.all(mapping[owner * width : owner * width + 2 * c.page_size] == 0)
    assert physical.free_pages == 4
    assert c._kvduo_mandatory_hot_pages() == 2


@pytest.mark.parametrize("protection", ["none", "shared", "prefill", "tp"])
def test_full_reclaim_transfers_exclusive_radix_prefix_before_demotion(protection):
    from array import array
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

    c, physical, width = local_pressure_coordinator([5, 5])
    allocator = c.token_to_kv_pool_allocator
    allocator.device = "cpu"
    allocator.free_segment = MagicMock()
    cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=c.req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=c.page_size,
        )
    )
    req = c._active_kvduo_reqs[1]
    values = c.req_to_token_pool.req_to_token[1, : req.kv.kv_allocated_len].long()
    key = RadixKey(array("q", range(len(values))))
    cache.insert(InsertParams(key=key, value=values))
    match = cache.match_prefix(MatchPrefixParams(key=key))
    req.last_node = match.last_device_node
    req.prefix_indices = match.device_indices
    req.cache_protected_len = len(values)
    cache.inc_lock_ref(req.last_node)
    if protection == "shared":
        cache.inc_lock_ref(req.last_node)
    elif protection == "prefill":
        c._kvduo_pressure_protected = torch.empty(0, dtype=torch.int64)
    elif protection == "tp":
        c._kvduo_consensus = MagicMock(return_value=False)
    c.kvduo_tree_cache = cache
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    original = mapping.clone()
    result = c.ensure_kvduo_decode_capacity([req])
    assert (result.action.name == "SUCCESS") is (protection != "tp")
    match = cache.match_prefix(MatchPrefixParams(key=key))
    if protection != "none":
        assert torch.equal(mapping[width:], original[width:])
        assert len(match.device_indices) == len(values)
        assert req.last_node.lock_ref == (2 if protection == "shared" else 1)
    else:
        assert req.cache_protected_len == 0
        assert len(match.device_indices) == 0
        assert torch.all(mapping[width : width + 2 * c.page_size] == 0)
        assert torch.equal(mapping[:width], original[:width])
        allocator.free_segment.assert_not_called()


@pytest.mark.parametrize(
    "blocked", ["tail", "host", "minimum", "unselected", "owned_suffix"]
)
def test_full_reclaim_does_not_detach_unused_prefix(blocked):
    from array import array
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

    c, physical, width = local_pressure_coordinator([5, 5])
    allocator = c.token_to_kv_pool_allocator
    allocator.device = "cpu"
    allocator.free_segment = MagicMock()
    cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=c.req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=c.page_size,
        )
    )
    req = c._active_kvduo_reqs[1]
    values = c.req_to_token_pool.req_to_token[1, : req.kv.kv_allocated_len].long()
    if blocked == "owned_suffix":
        values = values[: c.page_size]
    key = RadixKey(array("q", range(len(values))))
    cache.insert(InsertParams(key=key, value=values))
    match = cache.match_prefix(MatchPrefixParams(key=key))
    req.last_node = match.last_device_node
    req.prefix_indices = match.device_indices
    req.cache_protected_len = len(values)
    cache.inc_lock_ref(req.last_node)
    c.kvduo_tree_cache = cache
    if blocked == "tail":
        c.tail_protected_pages = 7
    elif blocked == "host":
        c.full_host_version.zero_()
    elif blocked == "minimum":
        c.min_hot_pages = 8
    elif blocked == "owned_suffix":
        c.full_last_touch[width : width + c.page_size] = 999999
    reclaimed = c.reclaim_kvduo_full_pages(
        c.page_size,
        reserve_hot_minimum=True,
        request_indices=None if blocked == "unselected" else {1},
    )
    assert (reclaimed > 0) == (blocked in ("unselected", "owned_suffix"))
    match = cache.match_prefix(MatchPrefixParams(key=key))
    assert len(match.device_indices) == len(values)
    assert req.cache_protected_len == (
        c.page_size if blocked == "owned_suffix" else len(values)
    )
    allocator.free_segment.assert_not_called()


@pytest.mark.parametrize("needed,expected", [(1, [5, 6]), (3, list(range(7)))])
def test_mixed_owner_precedes_larger_full_owner(needed, expected):
    assert (
        select_kvduo_full_requests(
            torch.ones(7, dtype=torch.int32),
            [0] * 5 + [1] * 2,
            [1, 0],
            [0, 99],
            needed,
            mixed_owners=[False, True],
        )
        == expected
    )


def test_mixed_owner_ties_use_net_gain_decode_age_then_slot():
    valid = torch.ones(12, dtype=torch.int32)
    owners = [0] * 4 + [1] * 4 + [2] * 4
    assert select_kvduo_full_requests(
        valid,
        owners,
        [0, 0, 0],
        [2, 1, 1],
        1,
        mixed_owners=[True, True, True],
    ) == [4, 5, 6, 7]


def test_coordinator_passes_actual_mixed_state_to_request_selection():
    c, physical, width = make_request_reclaim_coordinator([6, 2], policy="request")
    c._mixed_slots[1] = True
    c.kvduo_req_hot_capacity[:, 1] = c.hot_page_size
    before = c.mem_pool_device.full_to_hisparse_device_index_mapping.clone()
    assert (
        c.reclaim_kvduo_full_pages(c.page_size, reserve_hot_minimum=True)
        == 2 * c.page_size
    )
    assert torch.equal(
        c.mem_pool_device.full_to_hisparse_device_index_mapping[:width], before[:width]
    )
    assert physical.free_pages == 2


@pytest.mark.parametrize("prefill_pages,admitted", [(3, True), (4, False)])
def test_prefill_rechecks_only_real_surplus_after_decode_request_reclaim(
    prefill_pages, admitted
):
    from types import MethodType

    c, p, req = make_full_coordinator(layers=2, full_reclaim_policy="request")
    req.decode_batch_idx = 0
    c.reclaim_kvduo_full_pages = MagicMock(wraps=c.reclaim_kvduo_full_pages)
    assert c.reserve_kvduo_prefill(prefill_pages, [0]) is None
    c.reclaim_kvduo_full_pages.assert_not_called()
    c._reclaim_for_physical_allocation = MethodType(
        HiSparseCoordinator._reclaim_for_physical_allocation, c
    )
    c._require_allocation_ready = HiSparseCoordinator._require_allocation_ready
    c._guard_kvduo_physical_allocation(c.page_size)
    assert p.free_pages == 6
    p.alloc(c.page_size)
    assert c._kvduo_mandatory_hot_pages() == 2
    calls = c.reclaim_kvduo_full_pages.call_count
    token = c.reserve_kvduo_prefill(prefill_pages, [0])
    assert (token is not None) == admitted
    assert c.reclaim_kvduo_full_pages.call_count == calls
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c.kvduo_req_hot_capacity[:, 0].tolist() == [4, 4]
    assert p.free_pages == 3
