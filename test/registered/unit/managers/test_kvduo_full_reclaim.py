from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.kernels.ops.kvcache.kvduo_full_reclaim import (
    select_kvduo_full_pages,
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


def test_host_prefix_attachment_batches_locations_and_reads_clock_once():
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
    c.full_touch_clock.__getitem__.assert_called_once_with(0)
    assert c._req_host_prefix_records[0] == set(identities[:2])
    for identity in identities[:2]:
        record = c.host_prefix_cache.records[identity]
        assert record.last_access == 77
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
