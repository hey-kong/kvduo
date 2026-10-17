"""History headroom is reserved per prompt, physical FULL space per chunk."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_prefill_admission import adder_and_req, ready_coordinator

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def setup(used=0, length=20):
    c, physical = ready_coordinator()
    physical.free_pages = 100
    adder, req = adder_and_req(c, logical_size=100)
    req.prefix_indices = torch.empty(0, dtype=torch.int64)
    req.full_untruncated_fill_ids = list(range(length))
    req.kv = SimpleNamespace(kv_allocated_len=0)
    inventory = c.token_to_kv_pool_allocator.logical_attn_allocator
    if used:
        assert inventory.alloc(used) is not None
    c.reclaim_kvduo_hot_pages = MagicMock(side_effect=AssertionError("HOT merge"))
    return c, physical, inventory, req, adder.tree_cache


def another(req, length):
    return SimpleNamespace(
        rid=str(length),
        req_pool_idx=None,
        prefix_indices=torch.empty(0, dtype=torch.int64),
        full_untruncated_fill_ids=list(range(length)),
        sampling_params=req.sampling_params,
        output_ids=[],
        kv=SimpleNamespace(kv_allocated_len=0),
        set_finish_with_abort=MagicMock(),
    )


def reserve(c, req, tree=None, end=2):
    return c.reserve_kvduo_prefill(
        c.kvduo_prefill_pages(len(req.prefix_indices), end),
        prefill_req=req,
        prefill_end=end,
        tree_cache=tree,
    )


@pytest.mark.parametrize("used,admitted", [(60, True), (62, False)])
@pytest.mark.parametrize("free_pages", [0, 4])
def test_history_and_physical_budgets_must_both_fit(used, admitted, free_pages):
    c, physical, _, req, tree = setup(used)
    physical.free_pages = free_pages
    c.reclaim_kvduo_full_pages.side_effect = AssertionError("active FULL reclaim")
    assert (reserve(c, req, tree) is not None) is (admitted and free_pages > 0)
    c.reclaim_kvduo_full_pages.assert_not_called()
    c.reclaim_kvduo_hot_pages.assert_not_called()
    req.set_finish_with_abort.assert_not_called()


def test_complete_prompt_not_current_chunk_must_fit_history_limit():
    c, _, _, req, tree = setup(length=82)
    assert reserve(c, req, tree) is None
    req.set_finish_with_abort.assert_called_once()
    assert "80%" in req.set_finish_with_abort.call_args.args[0]
    assert not c._kvduo_prefill_history
    c.reclaim_kvduo_full_pages.assert_not_called()


def test_outstanding_chunks_reserve_only_unallocated_history_and_cancel_releases_it():
    c, _, inventory, req, tree = setup(length=60)
    token = reserve(c, req, tree)
    assert token is not None
    second = another(req, 22)
    assert reserve(c, second, tree) is None
    assert inventory.alloc(10) is not None
    req.kv.kv_allocated_len = 10
    req.prefix_indices = torch.arange(10)
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    third = another(req, 20)
    third_token = reserve(c, third, tree)
    assert third_token is not None  # 10 allocated + 50 outstanding + 20 new = 80
    c.finish_kvduo_prefill_reservation(third_token, cancelled=True)
    assert len(c._kvduo_prefill_history) == 1
    c._end_kvduo_prefill_lifecycle(None, req)
    assert not c._kvduo_prefill_history
    assert reserve(c, second, tree) is not None


def test_admitted_chunk_continues_when_decode_uses_headroom_and_cancel_restores_hold():
    c, _, inventory, req, tree = setup(used=20, length=60)
    token = reserve(c, req, tree)
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    assert inventory.alloc(12) is not None
    req.kv.kv_allocated_len = 2  # chunk uses 2, concurrent decode uses 10
    req.prefix_indices = torch.arange(2)
    before = dict(c._kvduo_prefill_history)
    continuation = reserve(c, req, tree, end=4)
    assert continuation is not None
    c.finish_kvduo_prefill_reservation(continuation, cancelled=True)
    assert c._kvduo_prefill_history == before
    assert reserve(c, another(req, 2), tree) is None


def test_idle_radix_history_eviction_precedes_active_full_reclaim():
    c, _, inventory, req, tree = setup(length=20)
    cached = inventory.alloc(62)
    tree.evict.side_effect = lambda params: inventory.free(cached[: params.num_tokens])
    assert reserve(c, req, tree) is not None
    assert tree.evict.call_args.args[0].num_tokens == 2
    c.reclaim_kvduo_full_pages.assert_not_called()


@pytest.mark.parametrize(
    "pool,compression", [("c4_indexer_kv_pool", 4), ("c128_kv_pool", 128)]
)
def test_each_real_backing_capacity_limits_admission(pool, compression):
    c, _, _, req, tree = setup(length=20)
    inventory = PagedTokenToKVPoolAllocator(256, 2, torch.int64, "cpu", None, False)
    inventory.alloc(100)
    c.token_to_kv_pool_allocator.logical_attn_allocator = inventory
    c.is_dsv4_hisparse = True
    inventory._kvcache = SimpleNamespace(
        **{pool: SimpleNamespace(size=128 // compression)}
    )
    assert reserve(c, req, tree) is None  # 120 exceeds 80% of 128, but fits 80% of 256
    req.set_finish_with_abort.assert_not_called()


def test_expired_request_generation_cannot_reserve_history():
    c, _, _, req, tree = setup(length=60)
    token = reserve(c, req, tree)
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    req.kvduo_prefill_generation += 1
    assert reserve(c, another(req, 40), tree) is not None


def test_oversized_prompt_is_rejected_even_before_scheduler_logical_gate():
    c, _, _, req, _ = setup(length=102)
    from sglang.srt.managers.schedule_policy import AddReqResult

    adder, req = adder_and_req(c, logical_size=100)
    req.full_untruncated_fill_ids = list(range(102))
    assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    req.set_finish_with_abort.assert_called_once()
    assert "80%" in req.set_finish_with_abort.call_args.args[0]
    c.reclaim_kvduo_full_pages.assert_not_called()
