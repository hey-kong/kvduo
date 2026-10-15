"""Host admission uses the paged allocator's actual free-slot inventory."""

import threading
from contextlib import nullcontext
from datetime import timedelta
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.sparsity.core.kvduo_prefix_cache import (
    HostPrefixRecord,
    KVDuoHostPrefixCache,
)
from sglang.srt.mem_cache.sparsity.core.kvduo_state import KVDuoPressureAction
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_hot_growth import make_host_pool
from test_kvduo_prefill_rollback import actual_budget_adder_and_req, ready_coordinator
from test_kvduo_pressure_admission import make_full_coordinator

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def host_inventory(c, size):
    if c.is_dsv4_hisparse:
        c.mem_pool_host = make_host_pool(size, c.page_size)
    else:
        pool = MLATokenToKVPoolHost.__new__(MLATokenToKVPoolHost)
        pool.size, pool.page_size, pool.device = size, c.page_size, "cpu"
        pool.lock = threading.RLock()
        pool.clear()
        c.mem_pool_host = pool
    c.host_prefix_cache = KVDuoHostPrefixCache(size)
    return c.mem_pool_host


def cached_page(c, locations, *, identity=None, pinned=False, valid=True):
    record = HostPrefixRecord(
        identity=identity if identity is not None else ("old", int(locations[0])),
        domains=("main_kv",),
        host_locations={"main_kv": tuple(locations.tolist())},
        data_versions={"main_kv": (1,) * len(locations)},
        host_versions={"main_kv": (1 if valid else 0,) * len(locations)},
        cache_reference=True,
        request_references={"waiting"} if pinned else set(),
    )
    c.host_prefix_cache.insert(record)
    return record


def prompt(size=8, rid="incoming"):
    req = SimpleNamespace(
        rid=rid,
        full_untruncated_fill_ids=list(range(size)),
        prefix_indices=torch.empty(0, dtype=torch.int64),
    )
    req.get_fill_ids = lambda: req.full_untruncated_fill_ids
    return req


@pytest.mark.parametrize("path", ["new", "chunk"])
def test_host_full_defers_prefill_before_hot_rollback_with_real_allocator_budgets(path):
    c, physical = ready_coordinator()
    adder, req = actual_budget_adder_and_req(c, path=path)
    c.is_dsv4_hisparse = True
    pool = host_inventory(c, 256)
    for locs in pool.alloc(pool.size).split(c.page_size):
        cached_page(c, locs, pinned=True)
    assert (
        adder.token_to_kv_pool_allocator.logical_attn_allocator.available_size() == 1024
    )
    assert adder.token_to_kv_pool_allocator.swa_available_size() == 1024
    assert pool.available_size() == physical.free_pages == 0
    before = c.req_device_buffer_tokens.clone()
    if path == "new":
        assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    else:
        assert adder.add_chunked_req(req) is req
    assert adder.kvduo_prefill_blocked and not adder.can_run_list
    assert not hasattr(req, "extend_range")
    assert int(c.kvduo_req_hot_capacity[0, 0]) == 28
    assert torch.equal(c.req_device_buffer_tokens, before)
    assert not c._kvduo_prefill_limited_objects
    assert not c._kvduo_prefill_host_reservations
    assert not c._kvduo_prefill_reservations
    c.reclaim_kvduo_full_pages.assert_not_called()
    adder.tree_cache.evict.assert_not_called()


@pytest.mark.parametrize("path,slots", [("new", 6), ("chunk", 10)])
def test_prefill_evicts_host_cache_and_keeps_complete_prompt_hold_after_chunk(
    path, slots
):
    c, _ = ready_coordinator(extra=4 if path == "chunk" else 3)
    adder, req = actual_budget_adder_and_req(c, path=path)
    c.is_dsv4_hisparse = True
    pool = host_inventory(c, 256)
    for locs in pool.alloc(pool.size).split(c.page_size):
        cached_page(c, locs)
    if path == "new":
        assert adder.add_one_req(req, False, None) is AddReqResult.CONTINUE
    else:
        assert adder.add_chunked_req(req) is req
    key = c._kvduo_prefill_key(req)
    assert c._kvduo_prefill_host_reservations[key][1] == slots
    assert pool.available_size() == slots
    assert int(c.kvduo_req_hot_capacity[0, 0]) == 16
    for token in adder.kvduo_prefill_reservations:
        c.finish_kvduo_prefill_reservation(token, cancelled=False)
    assert c._kvduo_prefill_host_reservations[key][1] == slots
    assert not c._kvduo_prefill_reservations


@pytest.mark.parametrize("ratio,length,expected", [(1, 5, 6), (4, 21, 6), (4, 7, 2)])
def test_host_hold_rounds_compressed_complete_prompt_not_current_chunk(
    ratio, length, expected
):
    c, physical = ready_coordinator()
    c.compress_ratio = ratio
    c.is_dsv4_hisparse = ratio > 1
    physical.free_pages = 64
    host_inventory(c, expected)
    req = prompt(length)
    req.extend_range = Range(0, 1)
    token = c.reserve_kvduo_prefill(1, [0], prefill_req=req)
    assert token is not None
    assert c._kvduo_prefill_host_reservations[c._kvduo_prefill_key(req)][1] == expected


def test_other_prefills_and_decode_cannot_spend_host_hold_and_cancellation_releases_it():
    c, physical = ready_coordinator()
    physical.free_pages = 64
    pool = host_inventory(c, 8)
    first, second = prompt(6, "first"), prompt(4, "second")
    token = c.reserve_kvduo_prefill(1, [0], prefill_req=first)
    assert token is not None and pool.available_size() == 8
    assert c.reserve_kvduo_prefill(1, [0], prefill_req=second) is None
    assert c._kvduo_host_capacity_ready(2)
    assert not c._kvduo_host_capacity_ready(4)
    decode = SimpleNamespace(req_pool_idx=0, kv_committed_len=4)
    c.token_to_kv_pool_allocator.page_size = c.page_size
    c._reclaim_kvduo_radix_pages = MagicMock()
    result = c.ensure_kvduo_decode_capacity([decode])
    assert result.action is KVDuoPressureAction.ERROR
    c._reclaim_kvduo_radix_pages.assert_not_called()
    c.finish_kvduo_prefill_reservation(token, cancelled=True)
    assert not c._kvduo_prefill_host_reservations
    assert not hasattr(first, "kvduo_prefill_generation")
    assert c.reserve_kvduo_prefill(1, [0], prefill_req=second) is not None


@pytest.mark.parametrize("valid,expected_slots", [(True, 2), (False, 4)])
def test_prefill_pins_only_reusable_contiguous_valid_host_pages(valid, expected_slots):
    c, physical = ready_coordinator()
    physical.free_pages = 64
    pool = host_inventory(c, 6)
    req = prompt(4)
    identity = c._host_prefix_identity(req, 0)
    record = cached_page(c, pool.alloc(2), identity=identity, valid=valid)
    unrelated = cached_page(c, pool.alloc(2))
    token = c.reserve_kvduo_prefill(1, [0], prefill_req=req)
    assert token is not None
    key = c._kvduo_prefill_key(req)
    assert c._kvduo_prefill_host_reservations[key][1] == expected_slots
    assert record.pinned is valid
    if valid:
        assert c._kvduo_host_capacity_ready(2)
        assert unrelated.identity not in c.host_prefix_cache.records
        assert record.identity in c.host_prefix_cache.records
    c.finish_kvduo_prefill_reservation(token, cancelled=True)
    assert not record.pinned


@pytest.mark.parametrize("failure", ["plan", "exception", "scope"])
def test_failed_prefill_releases_new_host_hold_and_pins(failure):
    c, physical = ready_coordinator()
    pool = host_inventory(c, 8)
    req = prompt(8)
    record = cached_page(c, pool.alloc(2), identity=c._host_prefix_identity(req, 0))
    if failure == "exception":
        c._reserve_kvduo_prefill_physical = MagicMock(
            side_effect=RuntimeError("allocation")
        )
        with pytest.raises(RuntimeError, match="allocation"):
            c.reserve_kvduo_prefill(1, [0], prefill_req=req)
    elif failure == "plan":
        c.kvduo_prefill_steps.zero_()
        assert c.reserve_kvduo_prefill(1, [0], prefill_req=req) is None
    else:
        physical.free_pages = 64
        with c.kvduo_prefill_reservation_scope():
            assert c.reserve_kvduo_prefill(1, [0], prefill_req=req) is not None
    assert not c._kvduo_prefill_host_reservations
    assert not record.pinned


def test_later_chunk_retry_keeps_existing_host_hold_until_request_ends():
    c, physical = ready_coordinator()
    physical.free_pages = 64
    host_inventory(c, 8)
    req = prompt(8)
    token = c.reserve_kvduo_prefill(1, [0], prefill_req=req)
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    key = c._kvduo_prefill_key(req)
    c._reserve_kvduo_prefill_physical = MagicMock(return_value=None)
    assert c.reserve_kvduo_prefill(1, [0], prefill_req=req) is None
    assert c._kvduo_prefill_host_reservations[key][1] == 8
    c._end_kvduo_prefill_lifecycle(0, prompt(8, req.rid))
    assert key in c._kvduo_prefill_host_reservations
    c._end_kvduo_prefill_lifecycle(0, req)
    assert not c._kvduo_prefill_host_reservations
    assert not hasattr(req, "kvduo_prefill_generation")
    assert c._kvduo_prefill_key(req) != key


def test_request_finish_releases_host_hold_before_slot_reuse():
    c, physical, req = make_full_coordinator()
    physical.free_pages = 64
    host_inventory(c, 8)
    req.rid = "ending"
    req.full_untruncated_fill_ids = req.origin_input_ids
    token = c.reserve_kvduo_prefill(1, [0], prefill_req=req)
    assert token is not None
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    assert c._kvduo_prefill_host_reservations
    c.request_finished(req)
    assert not c._kvduo_prefill_host_reservations
    assert not hasattr(req, "kvduo_prefill_generation")


def test_staging_converts_hold_to_real_host_allocation_and_request_prefix_pin(
    monkeypatch,
):
    import sglang.srt.managers.hisparse_coordinator as module

    c, physical = ready_coordinator(requests=2)
    physical.free_pages = 64
    pool = host_inventory(c, 8)
    req = prompt(8)
    record = cached_page(c, pool.alloc(2), identity=c._host_prefix_identity(req, 0))
    token = c.reserve_kvduo_prefill(1, [0], prefill_req=req)
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    assert c._kvduo_prefill_host_reservations[c._kvduo_prefill_key(req)][1] == 6
    req.req_pool_idx = 1
    req.extend_range = Range(0, 8)
    c.req_to_token_pool = SimpleNamespace(req_to_token=torch.arange(32).repeat(2, 1))
    c.mem_pool_device.translate_loc_from_full_to_hisparse_device = lambda locs: locs + 2
    c.req_to_host_pool = torch.full((2, 32), -1, dtype=torch.int64)
    c._req_host_prefix_records = [set(), set()]
    c.full_touch_clock = torch.tensor([10], dtype=torch.int64)
    c.write_staging_stream = MagicMock()
    c.ack_staging_queue = []
    pool.backup_from_device_all_layer = MagicMock()
    monkeypatch.setattr(module, "device_module", MagicMock())
    module.device_module.stream.side_effect = lambda stream: nullcontext()
    c.admit_request_into_staging(req)
    assert not c._kvduo_prefill_host_reservations
    assert pool.available_size() == 0
    assert int(c.req_to_host_pool_allocated_len[1]) == 8
    assert c.req_to_host_pool[1, :2].tolist() == list(record.host_locations["main_kv"])
    assert record.request_references == {(req.rid, 1)}
    assert c.ack_staging_queue[0].req is req
    pool.backup_from_device_all_layer.assert_called_once()


@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("mixed", [False, True])
def test_decode_sealed_backup_evicts_only_unpinned_host_prefix_before_allocating(
    monkeypatch, ratio, mixed
):
    import sglang.srt.managers.hisparse_coordinator as module

    c, physical, req = make_full_coordinator(pages=6, ratio=ratio)
    physical.free_pages = 64
    c._reclaim_for_physical_allocation = MethodType(
        HiSparseCoordinator._reclaim_for_physical_allocation, c
    )
    c.req_to_full_lookup[0, :] = torch.arange(12)
    pool = host_inventory(c, 8)
    old = pool.alloc(8)
    c.req_to_host_pool[0, :4] = old[:4]
    c.req_to_host_pool_allocated_len[0] = 4
    c._kvduo_host_valid_len[0] = 4
    pinned = cached_page(c, old[4:6], pinned=True)
    evictable = cached_page(c, old[6:8])
    req.kv_committed_len = 7 * ratio - 1
    assert c.ensure_kvduo_decode_capacity([req]).action is KVDuoPressureAction.SUCCESS
    assert pinned.identity in c.host_prefix_cache.records
    assert evictable.identity not in c.host_prefix_cache.records
    assert pool.available_size() == 2
    c.decode_backup_stream = MagicMock()
    c._backup_done_event = MagicMock()
    c._pending_kvduo_host_valid = []
    pool.backup_from_device_all_layer = MagicMock()
    monkeypatch.setattr(module, "device_module", MagicMock())
    module.device_module.stream.side_effect = lambda stream: nullcontext()
    lengths = torch.tensor([7 * ratio])
    indices = torch.tensor([0])
    c._mixed_slots[0] = mixed
    c._backup_kvduo_sealed_pages(lengths, indices, lengths, indices)
    assert c._kvduo_tail_cleanup_pending == (
        {0: c._kvduo_request_generation[0]} if mixed else {}
    )
    assert pool.available_size() == 0
    assert int(c.req_to_host_pool_allocated_len[0]) == 6
    assert c._has_pending_backup
    host_locs = pool.backup_from_device_all_layer.call_args.args[1]
    assert set(host_locs.tolist()) == set(evictable.host_locations["main_kv"])


def test_decode_batch_host_shortage_prevents_partial_allocations_and_dma():
    c, _ = ready_coordinator(requests=2)
    pool = host_inventory(c, 10)
    c.req_to_host_pool = torch.full((2, 16), -1, dtype=torch.int64)
    c.req_to_host_pool[:, :4] = pool.alloc(8).view(2, 4)
    c.req_to_host_pool_allocated_len[:] = 4
    c._kvduo_host_valid_len = [4, 4]
    c.wait_for_pending_backup = MagicMock()
    allocate = MagicMock(wraps=pool.alloc_paged_token_slots)
    pool.alloc_paged_token_slots = allocate
    pool.backup_from_device_all_layer = MagicMock()
    lengths, indices = torch.tensor([7, 7]), torch.tensor([0, 1])
    with pytest.raises(RuntimeError, match="decode backup exceeds"):
        c._backup_kvduo_sealed_pages(lengths, indices, lengths, indices)
    allocate.assert_not_called()
    pool.backup_from_device_all_layer.assert_not_called()
    assert pool.available_size() == 2
    assert c.req_to_host_pool_allocated_len.tolist() == [4, 4]


def _tp_host_worker(rank, init_file):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        c, physical = ready_coordinator()
        physical.free_pages = 2
        c.tp_world_size = 2
        c.tp_group = dist.group.WORLD
        pool = host_inventory(c, 8)
        if rank == 0:
            pool.alloc(8)
        req = prompt(8)
        assert c.reserve_kvduo_prefill(1, [0], prefill_req=req) is None
        assert not c._kvduo_prefill_host_reservations
        assert not c._kvduo_prefill_limited_objects
        assert int(c.kvduo_req_hot_capacity[0, 0]) == 28
        pool.clear()
        if rank == 0:
            for ordinal, locs in enumerate(pool.alloc(8).split(2)):
                cached_page(c, locs, identity=c._host_prefix_identity(req, ordinal))
        token = c.reserve_kvduo_prefill(1, [0], prefill_req=req)
        assert token is not None
        assert (
            c._kvduo_prefill_host_reservations[c._kvduo_prefill_key(req)][1] == 8 * rank
        )
        c.finish_kvduo_prefill_reservation(token, cancelled=True)
        assert not c._kvduo_prefill_host_reservations
        assert all(not record.pinned for record in c.host_prefix_cache.records.values())
    finally:
        dist.destroy_process_group()


def test_host_readiness_and_zero_private_slots_reach_consensus_across_tp_ranks(
    tmp_path,
):
    import torch.multiprocessing as mp

    mp.start_processes(
        _tp_host_worker,
        args=(str(tmp_path / "host-gloo"),),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
