"""Decode budget checks share one TP collective when both pools have room."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from sglang.srt.mem_cache.base_prefix_cache import EvictResult
from sglang.srt.mem_cache.sparsity.core.kvduo_state import (
    KVDuoPressureAction,
    KVDuoPressureReason,
    KVDuoPressureStatus,
)
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_host_capacity import cached_page, host_inventory
from test_kvduo_prefill_admission import budget_allocator
from test_kvduo_pressure_admission import make_decode_batch

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def decode_case(
    *,
    dsv4=True,
    lengths=None,
    free_pages=1,
    host_size=128,
    minimum=False,
    reserved_pages=0,
    reserved_host_slots=0,
):
    if lengths is None:
        lengths = [259 if dsv4 else 64]
    batch, physical = make_decode_batch(lengths, free_pages=free_pages)
    c = batch.hisparse_coordinator
    c._mixed_slots = [False] * len(batch.reqs)
    c.is_dsv4_hisparse = dsv4
    if not dsv4:
        c.compress_ratio = 1
        batch.token_to_kv_pool_allocator = c.token_to_kv_pool_allocator = (
            budget_allocator(c, logical_size=4096)
        )
    pool = host_inventory(c, host_size)
    if minimum:
        c._kvduo_pending_hot_minimum.add(0)
        c._mixed_slots[0] = True
    if reserved_pages:
        c._kvduo_prefill_reservations[0] = [reserved_pages, set()]
    if reserved_host_slots:
        c._kvduo_prefill_host_reservations[0] = (object(), reserved_host_slots, set())
    for name in (
        "_kvduo_common_available",
        "_kvduo_host_capacity_ready",
        "_reclaim_kvduo_radix_pages",
        "_reclaim_for_physical_allocation",
    ):
        setattr(c, name, MagicMock(wraps=getattr(c, name)))
    return batch, c, physical, pool


@pytest.mark.parametrize("dsv4", [False, True])
@pytest.mark.parametrize("needs_page", [False, True])
def test_decode_with_free_budgets_uses_one_common_check(dsv4, needs_page):
    length = (259 if needs_page else 256) if dsv4 else (64 if needs_page else 63)
    batch, c, physical, pool = decode_case(
        dsv4=dsv4,
        lengths=[length],
        free_pages=int(needs_page),
        host_size=64 * int(needs_page),
    )
    result = c.ensure_kvduo_decode_capacity(batch.reqs, tree_cache=batch.tree_cache)
    assert result.status is KVDuoPressureStatus.READY
    assert result.action is KVDuoPressureAction.SUCCESS
    assert result.reason is KVDuoPressureReason.NONE
    assert result.requested_bytes == c._physical_bytes(64 * int(needs_page))
    c._kvduo_common_available.assert_called_once_with(1)
    c._kvduo_host_capacity_ready.assert_not_called()
    c._reclaim_kvduo_radix_pages.assert_not_called()
    c._reclaim_for_physical_allocation.assert_not_called()
    c.reclaim_kvduo_full_pages.assert_not_called()
    c.reclaim_kvduo_hot_pages.assert_not_called()
    batch.tree_cache.evict.assert_not_called()
    physical.alloc.assert_not_called()
    assert physical.free_pages == int(needs_page)
    assert pool.available_size() == 64 * int(needs_page)


@pytest.mark.parametrize("dsv4", [False, True])
def test_decode_common_check_includes_all_host_and_physical_commitments(dsv4):
    lengths = [259 if dsv4 else 64] * 32
    batch, c, _, _ = decode_case(
        dsv4=dsv4,
        lengths=lengths,
        free_pages=36,
        host_size=2112,
        minimum=True,
        reserved_pages=2,
        reserved_host_slots=64,
    )
    c._kvduo_prefill_future_pages[0] = 1
    assert batch.check_decode_mem()
    c._kvduo_common_available.assert_called_once_with(1)
    c._reclaim_for_physical_allocation.assert_not_called()
    assert c._kvduo_mandatory_hot_pages() == 1
    assert c._kvduo_reserved_prefill_pages() == 3
    assert c._kvduo_prefill_host_reservations[0][1] == 64


@pytest.mark.parametrize("shortage", ["host", "physical", "minimum", "prefill"])
def test_decode_shortage_uses_pressure_path_and_preserves_budget_rejection(shortage):
    batch, c, _, _ = decode_case(
        free_pages=0 if shortage == "physical" else 1,
        host_size=64,
        minimum=shortage == "minimum",
        reserved_pages=int(shortage == "prefill"),
        reserved_host_slots=64 * int(shortage == "host"),
    )
    assert not batch.check_decode_mem()
    assert c._kvduo_common_available.call_args_list[0].args == (0,)
    c._kvduo_host_capacity_ready.assert_called_once_with(64)
    if shortage == "host":
        c._reclaim_kvduo_radix_pages.assert_not_called()
        c._reclaim_for_physical_allocation.assert_not_called()
        c.reclaim_kvduo_full_pages.assert_not_called()
    else:
        c._reclaim_kvduo_radix_pages.assert_called_once()
        c._reclaim_for_physical_allocation.assert_called_once_with(
            64,
            reserve_hot_minimum=True,
            preferred_full_requests=((0, 128 if shortage == "minimum" else 64),),
            allow_emergency_hot_reclaim=False,
        )


@pytest.mark.parametrize("pool", ["logical", "swa"])
def test_decode_common_check_does_not_bypass_independent_scheduler_budgets(pool):
    batch, c, _, _ = decode_case(lengths=[256])
    allocator = batch.token_to_kv_pool_allocator.logical_attn_allocator
    pool_allocator = (
        allocator.full_attn_allocator
        if pool == "logical"
        else allocator.swa_attn_allocator
    )
    assert pool_allocator.alloc(pool_allocator.available_size()) is not None
    assert not batch.check_decode_mem()
    c._kvduo_common_available.assert_not_called()


def _tp_decode_budget_worker(rank, init_file):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        for scenario in (
            "ready",
            "radix",
            "host_blocked",
            "host_evictable",
            "physical_blocked",
        ):
            batch, c, physical, pool = decode_case(
                free_pages=rank * 2 if scenario in ("radix", "physical_blocked") else 1,
            )
            c.tp_world_size = 2
            c.tp_group = dist.group.WORLD
            if rank == 0 and scenario in ("host_blocked", "host_evictable"):
                locations = pool.alloc(128)
                cached_page(c, locations[:64], pinned=True)
                cached_page(c, locations[64:], pinned=scenario == "host_blocked")
            if scenario == "radix":
                batch.tree_cache.full_evictable_size.return_value = 256

                def evict(params):
                    assert params.num_tokens == 256
                    physical.free_pages += 1
                    return EvictResult(num_tokens_evicted=256)

                batch.tree_cache.evict.side_effect = evict
            with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce:
                result = c.ensure_kvduo_decode_capacity(
                    batch.reqs, tree_cache=batch.tree_cache
                )
            expected = (
                KVDuoPressureAction.ERROR
                if scenario in ("host_blocked", "physical_blocked")
                else KVDuoPressureAction.SUCCESS
            )
            assert result.action is expected
            if scenario == "ready":
                assert reduce.call_count == 1
                c._kvduo_host_capacity_ready.assert_not_called()
                c._reclaim_kvduo_radix_pages.assert_not_called()
                c._reclaim_for_physical_allocation.assert_not_called()
            else:
                assert reduce.call_count > 1
                c._kvduo_host_capacity_ready.assert_called_once_with(64)
                if scenario == "host_blocked":
                    assert reduce.call_count == 2
                    c._reclaim_kvduo_radix_pages.assert_not_called()
                    c._reclaim_for_physical_allocation.assert_not_called()
                    c.reclaim_kvduo_full_pages.assert_not_called()
                else:
                    c._reclaim_kvduo_radix_pages.assert_called_once()
                    c._reclaim_for_physical_allocation.assert_called_once()
            if scenario == "radix":
                batch.tree_cache.evict.assert_called_once()
                assert physical.free_pages == rank * 2 + 1
            if scenario == "host_evictable" and rank == 0:
                assert pool.available_size() == 64
                assert len(c.host_prefix_cache.records) == 1
                assert next(iter(c.host_prefix_cache.records.values())).pinned
            if scenario == "physical_blocked":
                c.reclaim_kvduo_full_pages.assert_called_once_with(
                    64, reserve_hot_minimum=True
                )
    finally:
        dist.destroy_process_group()


def test_two_tp_ranks_share_fast_path_and_pressure_fallback(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _tp_decode_budget_worker,
        args=(str(tmp_path / "gloo-init"),),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
