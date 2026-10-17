"""History headroom applies to new prompts, not continuation chunks or decode."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_prefill_admission import adder_and_req, ready_coordinator

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def case(free=22, hybrid=False):
    c, physical = ready_coordinator()
    physical.free_pages = 32
    _, req = adder_and_req(c)
    req.rid = "history"
    req.req_pool_idx = None
    req.set_finish_with_abort = MagicMock()
    pool = SimpleNamespace(size=40, page_size=2, free=free)
    pool.available_size = lambda: pool.free
    c.token_to_kv_pool_allocator.logical_attn_allocator = (
        SimpleNamespace(full_attn_allocator=pool) if hybrid else pool
    )
    return c, req, pool


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("free,allowed", [(22, True), (20, False)])
def test_full_prompt_projection_accepts_exactly_eighty_percent(hybrid, free, allowed):
    c, req, pool = case(free, hybrid)
    token = c.reserve_kvduo_prefill(2, prefill_req=req, prefill_end=6)
    assert (token is not None) == allowed
    assert bool(c._kvduo_prefill_history) == allowed
    req.set_finish_with_abort.assert_not_called()
    c.reclaim_kvduo_full_pages.assert_not_called()


def test_continuation_does_not_reapply_limit_after_decode_growth():
    c, req, pool = case()
    first = c.reserve_kvduo_prefill(2, prefill_req=req, prefill_end=6)
    c.finish_kvduo_prefill_reservation(first, cancelled=False)
    req.prefix_indices = torch.arange(6)
    pool.free = 10
    assert c.reserve_kvduo_prefill(2, prefill_req=req, prefill_end=10) is not None


def test_outstanding_prompt_holds_are_consumed_without_double_counting():
    c, req, pool = case(free=36)
    first = c.reserve_kvduo_prefill(2, prefill_req=req, prefill_end=6)
    assert first is not None
    _, other = adder_and_req(c)
    other.rid = "second"
    key = c._kvduo_prefill_key(other)
    assert c._kvduo_prefill_history_ready(other, key, None)
    # Actual allocation replaces two pages of the outstanding prompt hold.
    pool.free -= 4
    req.kv = SimpleNamespace(kv_allocated_len=6)
    assert c._kvduo_prefill_history_ready(other, key, None)
    pool.free -= 2
    assert not c._kvduo_prefill_history_ready(other, key, None)


def test_cancel_and_end_release_history_hold():
    c, req, pool = case()
    first = c.reserve_kvduo_prefill(2, prefill_req=req, prefill_end=6)
    c.finish_kvduo_prefill_reservation(first, cancelled=True)
    assert not c._kvduo_prefill_history
    second = c.reserve_kvduo_prefill(2, prefill_req=req, prefill_end=6)
    assert second is not None
    c.finish_kvduo_prefill_reservation(second, cancelled=False)
    assert c._kvduo_prefill_history
    c.end_kvduo_prefill(req)
    assert not c._kvduo_prefill_history


def test_insufficient_physical_space_does_not_leave_history_hold():
    c, req, pool = case()
    c.token_to_kv_pool_allocator.hisparse_attn_allocator.free_pages = 0
    assert c.reserve_kvduo_prefill(2, prefill_req=req, prefill_end=6) is None
    assert not c._kvduo_prefill_history


def test_history_gate_evicts_only_idle_cache_before_waiting():
    c, req, pool = case(free=20)
    cache = SimpleNamespace(
        evict=MagicMock(side_effect=lambda _: setattr(pool, "free", 22))
    )
    key = c._kvduo_prefill_key(req)
    assert c._kvduo_prefill_history_ready(req, key, cache)
    assert cache.evict.call_args.args[0].num_tokens == 2
    c.reclaim_kvduo_full_pages.assert_not_called()


def test_tp_minimum_free_space_controls_new_admission():
    c, req, pool = case(free=22)
    c._kvduo_common_available = MagicMock(return_value=10)
    key = c._kvduo_prefill_key(req)
    assert not c._kvduo_prefill_history_ready(req, key, None)


@pytest.mark.parametrize("prefix", [2, 38])
def test_prompt_that_cannot_fit_limit_is_rejected_not_queued_forever(prefix):
    c, req, pool = case(free=40)
    req.full_untruncated_fill_ids = list(range(40))
    req.prefix_indices = torch.arange(prefix)
    key = c._kvduo_prefill_key(req)
    assert not c._kvduo_prefill_history_ready(req, key, None)
    req.set_finish_with_abort.assert_called_once()
