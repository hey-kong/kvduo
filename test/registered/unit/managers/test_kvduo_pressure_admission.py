"""KVDuo scheduler admission and FULL-prefix lifecycle regressions."""

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.mem_cache.base_prefix_cache import EvictResult, InsertResult
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.sparsity.core.kvduo_state import KVDuoPressureAction
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_hot_growth import make_coordinator
from test_kvduo_prefill_rollback import (
    actual_budget_adder_and_req,
    budget_allocator,
    ready_coordinator,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_cache(allocator):
    return SimpleNamespace(
        token_to_kv_pool_allocator=allocator,
        is_chunk_cache=lambda: False,
        full_evictable_size=MagicMock(return_value=0),
        evictable_size=MagicMock(return_value=0),
        evict=MagicMock(return_value=EvictResult()),
    )


def make_decode_batch(lengths, *, free_pages=0, logical_size=4096, swa_size=4096):
    coordinator, physical = make_coordinator(
        requests=len(lengths), free_pages=free_pages
    )
    coordinator.page_size = 64
    coordinator.mem_pool_host.page_size = coordinator.mem_pool_host.slot_page_size = 64
    coordinator.item_size_bytes = 1
    coordinator.is_dsv4_hisparse = True
    coordinator._kvduo_pending_hot_minimum.clear()
    coordinator._mixed_slots[0] = False
    coordinator._reclaim_for_physical_allocation = MethodType(
        HiSparseCoordinator._reclaim_for_physical_allocation, coordinator
    )
    coordinator.reclaim_kvduo_hot_pages = MagicMock(return_value=0)
    allocator = budget_allocator(
        coordinator, dsv4=True, logical_size=logical_size, swa_size=swa_size
    )
    coordinator.token_to_kv_pool_allocator = allocator
    batch = ScheduleBatch(
        reqs=[
            SimpleNamespace(kv_committed_len=n, req_pool_idx=i)
            for i, n in enumerate(lengths)
        ]
    )
    batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    batch.token_to_kv_pool_allocator = allocator
    batch.hisparse_coordinator = coordinator
    batch.tree_cache = make_cache(allocator)
    return batch, physical


@pytest.mark.parametrize(
    "length,free_pages,expected", [(256, 0, True), (259, 0, False), (259, 1, True)]
)
def test_decode_uses_actual_compressed_physical_need_and_reclaim_result(
    length, free_pages, expected
):
    batch, _ = make_decode_batch([length], free_pages=free_pages)
    assert (
        batch.token_to_kv_pool_allocator.logical_attn_allocator.available_size() == 4096
    )
    assert batch.check_decode_mem() is expected
    batch.tree_cache.evict.assert_not_called()


@pytest.mark.parametrize("pool", ["logical", "swa"])
def test_decode_retains_independent_logical_and_swa_rejections(pool):
    batch, _ = make_decode_batch(
        [256],
        logical_size=0 if pool == "logical" else 4096,
        swa_size=0 if pool == "swa" else 4096,
    )
    assert not batch.check_decode_mem()
    params = batch.tree_cache.evict.call_args.args[0]
    assert params.num_tokens == (256 if pool == "logical" else 0)
    assert params.swa_num_tokens == (256 if pool == "swa" else 0)
    batch.hisparse_coordinator.reclaim_kvduo_full_pages.assert_not_called()


def test_decode_retraction_subset_checks_only_remaining_physical_needs():
    batch, _ = make_decode_batch([256, 259])
    assert not batch.check_decode_mem()
    assert batch.check_decode_mem(selected_indices=[0])
    assert not batch.check_decode_mem(selected_indices=[1])


def test_non_kvduo_decode_keeps_existing_composite_budget():
    batch, _ = make_decode_batch([256])
    batch.hisparse_coordinator = None
    assert not batch.check_decode_mem()


def test_decode_does_not_spend_existing_prefill_reservations():
    batch, _ = make_decode_batch([259], free_pages=1)
    coordinator = batch.hisparse_coordinator
    coordinator._kvduo_prefill_reservations[0] = [1, set()]
    assert not batch.check_decode_mem()
    assert coordinator._kvduo_reserved_prefill_pages() == 1


def test_decode_requires_pending_hot_minimum_even_without_a_new_c4_page():
    batch, physical = make_decode_batch([256])
    coordinator = batch.hisparse_coordinator
    coordinator._kvduo_pending_hot_minimum.add(0)
    coordinator._mixed_slots[0] = True
    assert not batch.check_decode_mem()
    physical.free_pages = 1
    assert batch.check_decode_mem()


def test_decode_reclaims_radix_against_physical_shortfall_without_logical_overeviction():
    batch, physical = make_decode_batch([259])
    batch.tree_cache.full_evictable_size.return_value = 256

    def evict(params):
        assert params.num_tokens == 256 and params.swa_num_tokens == 0
        physical.free_pages += 1
        return EvictResult(num_tokens_evicted=256)

    batch.tree_cache.evict.side_effect = evict
    assert batch.check_decode_mem()
    batch.tree_cache.evict.assert_called_once()
    batch.hisparse_coordinator.reclaim_kvduo_full_pages.assert_not_called()


@pytest.mark.parametrize("path", ["new", "chunk"])
def test_prefill_evicts_radix_before_rejecting_an_immature_hot_rollback(path):
    coordinator, physical = ready_coordinator()
    coordinator.is_dsv4_hisparse = True
    coordinator.kvduo_prefill_steps.zero_()
    adder, req = actual_budget_adder_and_req(coordinator, path=path)
    adder.tree_cache.full_evictable_size.return_value = 16

    def evict(params):
        assert params.num_tokens == 16 and params.swa_num_tokens == 0
        physical.free_pages += 2
        return EvictResult(num_tokens_evicted=16)

    adder.tree_cache.evict.side_effect = evict
    if path == "new":
        assert adder.add_one_req(req, False, None) is AddReqResult.CONTINUE
    else:
        assert adder.add_chunked_req(req) is req
    adder.tree_cache.evict.assert_called_once()
    coordinator.reclaim_kvduo_full_pages.assert_not_called()
    assert physical.free_pages == coordinator._kvduo_reserved_prefill_pages() == 2
    assert not coordinator._kvduo_prefill_limited_objects
    assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 28


def test_radix_reclaim_continues_past_host_only_entries_and_stops_without_progress():
    batch, physical = make_decode_batch([259])
    batch.tree_cache.full_evictable_size.return_value = 512

    def evict(params):
        if batch.tree_cache.evict.call_count == 2:
            physical.free_pages += 1
        return EvictResult(num_tokens_evicted=256)

    batch.tree_cache.evict.side_effect = evict
    assert batch.check_decode_mem()
    assert batch.tree_cache.evict.call_count == 2
    physical.free_pages = 0
    batch.tree_cache.evict.side_effect = None
    batch.tree_cache.evict.return_value = EvictResult()
    batch.tree_cache.evict.reset_mock()
    assert not batch.check_decode_mem()
    batch.tree_cache.evict.assert_called_once()


def make_full_coordinator(*, pages=8, layers=1, ratio=1, initial_pages=2):
    coordinator, physical = make_coordinator(layers=layers, free_pages=0)
    coordinator.compress_ratio = ratio
    coordinator.is_dsv4_hisparse = ratio > 1
    coordinator.mem_pool_device.page_size = coordinator.page_size
    coordinator.padded_buffer_size = (
        coordinator.device_buffer_size + coordinator.page_size
    )
    coordinator.tail_protected_pages = 2
    coordinator._kvduo_pending_hot_minimum.clear()
    coordinator._mixed_slots[0] = False
    count = pages * coordinator.page_size
    full_count = count * ratio
    coordinator.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(full_count).view(1, -1)
    )
    coordinator.mem_pool_device.translate_loc_from_full_to_compressed = (
        lambda indices: indices[ratio - 1 :: ratio] // ratio
    )
    coordinator.mem_pool_device.full_to_hisparse_device_index_mapping = torch.arange(
        coordinator.page_size, coordinator.page_size + count
    )
    coordinator.full_generation = torch.ones(count, dtype=torch.int64)
    coordinator.full_data_version = torch.ones(count, dtype=torch.int64)
    coordinator.full_host_version = torch.ones(count, dtype=torch.int64)
    coordinator.full_last_touch = torch.arange(count, dtype=torch.int64) + 100
    coordinator.full_touch_clock = torch.tensor([100], dtype=torch.int64)
    coordinator.req_to_full_lookup = torch.full((1, count), -1, dtype=torch.int64)
    coordinator._kvduo_host_valid_len = [count]
    coordinator.req_to_device_buffer = torch.zeros((1, 33), dtype=torch.int32)
    coordinator.req_device_buffer_size = torch.zeros(1, dtype=torch.int64)
    coordinator.req_device_buffer_size_gpu = torch.zeros(1, dtype=torch.int32)
    coordinator.req_reserved_logical = torch.full((1,), -1, dtype=torch.int64)
    coordinator.req_to_host_pool = torch.full((1, count), -1, dtype=torch.int64)
    coordinator.req_to_host_pool_allocated_len = torch.zeros(1, dtype=torch.int64)
    coordinator._skip_first_backup = [False]
    coordinator._lru_init = torch.arange(32, dtype=torch.int16)
    coordinator.mem_pool_host.allocated_host_indices = lambda *args: torch.empty(
        0, dtype=torch.int64
    )
    coordinator.mem_pool_host.free = MagicMock()
    coordinator.wait_for_pending_backup = MagicMock()
    coordinator._retain_request_host_prefix = MagicMock(return_value=set())
    coordinator._release_host_prefix_refs = MagicMock()
    coordinator.reclaim_kvduo_full_pages = MethodType(
        HiSparseCoordinator.reclaim_kvduo_full_pages, coordinator
    )
    coordinator.reclaim_kvduo_hot_pages = MagicMock(return_value=0)
    initial_count = initial_pages * coordinator.page_size * ratio
    req = SimpleNamespace(
        req_pool_idx=0,
        cache_protected_len=0,
        hisparse_staging=False,
        kv=SimpleNamespace(kv_allocated_len=initial_count),
        extend_range=Range(0, initial_count),
        origin_input_ids=list(range(initial_count)),
        output_ids=list(range(initial_count, full_count)),
        extra_key=None,
        last_node=None,
        # A reused/retracted Req must lose its previous first-hole marker.
        kvduo_radix_insert_len=0,
    )
    coordinator.alloc_device_buffer(req)
    req.kv.kv_allocated_len = full_count
    coordinator._kvduo_host_valid_len[0] = count
    coordinator._active_kvduo_reqs = {0: req}
    coordinator.token_to_kv_pool_allocator.page_size = coordinator.page_size * ratio
    coordinator.item_size_bytes = 1
    return coordinator, physical, req


def test_decode_full_reclaim_funds_both_append_and_new_mixed_minima():
    coordinator, physical, req = make_full_coordinator(layers=2)
    coordinator._reclaim_for_physical_allocation = MethodType(
        HiSparseCoordinator._reclaim_for_physical_allocation, coordinator
    )
    req.kv_committed_len = 16
    result = coordinator.ensure_kvduo_decode_capacity([req])
    assert result.action is KVDuoPressureAction.SUCCESS
    assert physical.free_pages == 3
    assert coordinator._kvduo_mandatory_hot_pages() == 2
    assert coordinator._guard_kvduo_physical_allocation(2)
    physical.free_pages -= 1  # the admitted FULL append page
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert physical.free_pages == 0
    assert coordinator.kvduo_req_hot_capacity[:, 0].tolist() == [4, 4]


def test_decode_defers_when_full_reclaim_cannot_fund_new_mixed_minima():
    coordinator, physical, req = make_full_coordinator(pages=4, layers=2)
    coordinator._reclaim_for_physical_allocation = MethodType(
        HiSparseCoordinator._reclaim_for_physical_allocation, coordinator
    )
    req.kv_committed_len = 8
    result = coordinator.ensure_kvduo_decode_capacity([req])
    assert result.action is KVDuoPressureAction.ERROR
    assert physical.free_pages == 0
    assert not coordinator._mixed_slots[0]
    assert not coordinator._kvduo_pending_hot_minimum


@pytest.mark.parametrize("ratio", [1, 4])
def test_prefill_full_demotion_reserves_new_mixed_owners_minimum_in_every_layer(ratio):
    coordinator, physical, req = make_full_coordinator(layers=2, ratio=ratio)
    token = coordinator.reserve_kvduo_prefill(1, [0])
    assert token is not None
    assert physical.free_pages == 3  # two HOT minima plus one prefill page
    assert coordinator._kvduo_mandatory_hot_pages() == 2
    assert coordinator._kvduo_reserved_prefill_pages() == 1
    assert req.kvduo_mixed_residency
    coordinator.reclaim_kvduo_hot_pages.assert_not_called()
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert physical.free_pages == 1
    assert coordinator._kvduo_mandatory_hot_pages() == 0
    assert coordinator.kvduo_req_hot_capacity[:, 0].tolist() == [4, 4]


def test_prefill_does_not_demote_full_when_new_hot_minimum_consumes_all_gain():
    coordinator, physical, req = make_full_coordinator(pages=4, layers=2)
    assert coordinator.reserve_kvduo_prefill(1, [0]) is None
    assert physical.free_pages == 0
    assert not coordinator._mixed_slots[0]
    assert not coordinator._kvduo_pending_hot_minimum
    assert req.kvduo_radix_insert_len == 4
    coordinator.reclaim_kvduo_hot_pages.assert_not_called()


def test_prefill_full_reclaim_preserves_incoming_prefix_and_earlier_reservations():
    coordinator, physical, req = make_full_coordinator(layers=2)
    physical.free_pages = 1
    first = coordinator.reserve_kvduo_prefill(1, [0])
    second = coordinator.reserve_kvduo_prefill(1, [0], prefix_indices=torch.arange(4))
    assert first is not None and second is not None
    mapping = coordinator.mem_pool_device.full_to_hisparse_device_index_mapping
    assert torch.all(mapping[:4] > 0)
    assert req.kvduo_radix_insert_len == 4
    assert physical.free_pages == 4
    assert coordinator._kvduo_mandatory_hot_pages() == 2
    assert coordinator._kvduo_reserved_prefill_pages() == 2


@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("protected_pages", [0, 1])
def test_finish_retains_only_prefill_prefix_even_when_decode_prefix_is_resident(
    ratio, protected_pages
):
    coordinator, physical, req = make_full_coordinator(ratio=ratio)
    req.cache_protected_len = protected_pages * 2 * ratio
    coordinator.full_last_touch[8:10] = 0
    assert coordinator.reclaim_kvduo_full_pages(2) == 2
    assert req.kvduo_radix_insert_len == 4 * ratio
    coordinator.request_finished(req)
    mapping = coordinator.mem_pool_device.full_to_hisparse_device_index_mapping
    assert torch.all(mapping[:4] > 0)
    assert torch.all(mapping[4:] == 0)
    assert physical.free_pages == 6
    assert not coordinator._active_kvduo_reqs

    cache = RadixCache.__new__(RadixCache)
    cache.disable = cache.disable_finished_insert = cache.is_eagle = False
    cache.page_size = 2 * ratio
    cache.req_to_token_pool = coordinator.req_to_token_pool
    cache.token_to_kv_pool_allocator = SimpleNamespace(free_segments=MagicMock())
    cache.insert = MagicMock(return_value=InsertResult(prefix_len=0))
    cache.cache_finished_req(req, kv_len_to_handle=16 * ratio)
    params = cache.insert.call_args.args[0]
    assert len(params.key) == len(params.value) == 4 * ratio


@pytest.mark.parametrize("committed,expected", [(15, 4), (16, 4)])
def test_fully_resident_finish_caches_only_prefill_pages(committed, expected):
    coordinator, _, req = make_full_coordinator()
    coordinator.request_finished(req)
    cache = RadixCache.__new__(RadixCache)
    cache.disable = cache.disable_finished_insert = cache.is_eagle = False
    cache.page_size = 2
    cache.req_to_token_pool = coordinator.req_to_token_pool
    cache.token_to_kv_pool_allocator = SimpleNamespace(free_segments=MagicMock())
    cache.insert = MagicMock(return_value=InsertResult(prefix_len=0))
    cache.cache_finished_req(req, kv_len_to_handle=committed)
    assert len(cache.insert.call_args.args[0].key) == expected


@pytest.mark.parametrize("allow_reclaim", [False, True])
def test_mandatory_hot_allocation_cannot_spend_reserved_prefill_pages(allow_reclaim):
    coordinator, physical, _ = make_full_coordinator(pages=4)
    physical.free_pages = 1
    token = coordinator.reserve_kvduo_prefill(1, [0])
    assert token is not None
    coordinator._mixed_slots[0] = True
    coordinator._kvduo_pending_hot_minimum.add(0)
    coordinator._reclaim_for_physical_allocation = MethodType(
        HiSparseCoordinator._reclaim_for_physical_allocation, coordinator
    )
    coordinator._require_allocation_ready = (
        HiSparseCoordinator._require_allocation_ready
    )
    coordinator.reclaim_kvduo_full_pages = MagicMock(return_value=0)
    physical.alloc.reset_mock()
    if allow_reclaim:
        with pytest.raises(RuntimeError, match="physical allocation cannot proceed"):
            coordinator._ensure_kvduo_hot_capacity_targets_batch(
                {(0, 0): 4}, allow_reclaim=True
            )
    else:
        coordinator._ensure_kvduo_hot_capacity_targets_batch(
            {(0, 0): 4}, allow_reclaim=False
        )
    physical.alloc.assert_not_called()
    assert physical.free_pages == 1
    assert coordinator.kvduo_req_hot_capacity[0, 0].item() == 0
    assert coordinator._kvduo_reserved_prefill_pages() == 1
