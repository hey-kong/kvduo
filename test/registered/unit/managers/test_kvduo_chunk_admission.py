"""Chunk-sized HBM admission, pause/resume and final decode credit."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from test_kvduo_prefill_admission import adder_and_req, ready_coordinator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def test_chunk_admission_pauses_and_resumes_without_whole_prompt_hold():
    c, physical = ready_coordinator()
    adder, req = adder_and_req(c)
    req.sampling_params.max_new_tokens = 2
    req.rid = "chunks"
    req.req_pool_idx = None
    key = c._kvduo_prefill_key(req)
    # Only one chunk fits, far less than the seven-page remaining prompt.
    physical.free_pages = 2
    assert adder.add_chunked_req(req) is req
    assert req.extend_range.end == 6
    assert c._kvduo_reserved_prefill_pages() == 2
    token = adder.kvduo_prefill_reservations.pop()
    physical.alloc(4)
    c.kvduo_prefill_batch_allocated()
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    assert c._kvduo_reserved_prefill_pages() == 0
    assert key in c._kvduo_prefill_host_reservations
    req.prefix_indices = torch.arange(6)
    adder.can_run_list.clear()
    assert adder.add_chunked_req(req) is req
    assert adder.kvduo_prefill_blocked
    assert not adder.can_run_list
    assert req.extend_range.end == 6
    c.reclaim_kvduo_full_pages.assert_not_called()
    # Decode/request completion has released space at a later boundary.
    physical.free_pages = 2
    adder.kvduo_prefill_blocked = False
    assert adder.add_chunked_req(req) is req
    assert req.extend_range.end == 10
    assert adder.can_run_list == [req]
    c.finish_kvduo_prefill_reservation(
        adder.kvduo_prefill_reservations[-1], cancelled=True
    )
    assert c._kvduo_reserved_prefill_pages() == 0
    c.end_kvduo_prefill(req)
    assert key not in c._kvduo_prefill_host_reservations


@pytest.mark.parametrize("output,expected", [(1, 1), (2, 2)])
def test_last_chunk_holds_only_required_first_decode(output, expected):
    c, physical = ready_coordinator()
    adder, req = adder_and_req(c)
    req.prefix_indices = torch.arange(14)
    req.sampling_params.max_new_tokens = output
    physical.free_pages = expected
    assert adder.add_chunked_req(req) is None
    token = adder.kvduo_prefill_reservations[-1]
    assert c._kvduo_reserved_prefill_pages() == expected
    physical.alloc(2)
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    assert c._kvduo_reserved_prefill_pages() == expected - 1


def test_chunk_cannot_consume_next_decode_page_or_mandatory_hot():
    c, physical = ready_coordinator()
    _, req = adder_and_req(c)
    c._active_kvduo_reqs[0] = SimpleNamespace(
        kv_committed_len=6, finished=lambda: False
    )
    physical.free_pages = 2
    assert c.reserve_kvduo_prefill(2, prefill_req=req, prefill_end=6) is None
    assert not c._kvduo_prefill_host_reservations
    physical.free_pages = 3
    assert c.reserve_kvduo_prefill(2, prefill_req=req, prefill_end=6) is not None


def test_chunk_larger_than_entire_physical_pool_is_rejected_before_start():
    c, physical = ready_coordinator()
    _, req = adder_and_req(c)
    physical.size = 6 * c.page_size
    physical.free_pages = 6
    req.set_finish_with_abort = MagicMock()
    assert c.reserve_kvduo_prefill(2, prefill_req=req, prefill_end=6) is None
    req.set_finish_with_abort.assert_called_once()
    assert not c._kvduo_prefill_reservations


def test_request_reclaim_logs_actual_physical_blocks_after_free(caplog):
    from test_kvduo_full_reclaim import make_request_reclaim_coordinator

    c, physical, _ = make_request_reclaim_coordinator([3, 6])
    with caplog.at_level("INFO"):
        released = c.reclaim_kvduo_full_pages(c.page_size, reserve_hot_minimum=True)
    assert released == 6 * c.page_size
    assert "released_physical_blocks=6" in caplog.text
    assert physical.free_pages == 6
    caplog.clear()
    with caplog.at_level("INFO"):
        assert c.reclaim_kvduo_full_pages(0) == 0
    assert not caplog.records


@pytest.mark.parametrize("shortage", ["physical", "logical", "swa"])
def test_scheduler_blocked_chunk_does_not_start_waiting_prompt(monkeypatch, shortage):
    import sglang.srt.managers.scheduler as module

    c, physical = ready_coordinator()
    adder, req = adder_and_req(c, dsv4=shortage == "swa")
    if shortage == "physical":
        physical.free_pages = 0
    elif shortage == "logical":
        physical.free_pages = 100
        adder.rem_total_token_offset = 10000
    else:
        physical.free_pages = 100
        adder.rem_swa_token_offset = 10000
    req.init_next_round_input = MagicMock()
    running = SimpleNamespace(reqs=[], batch_is_full=False)
    s = module.Scheduler.__new__(module.Scheduler)
    s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
    s.enable_hierarchical_cache = s.enable_priority_preemption = False
    s.is_hybrid_swa = shortage == "swa"
    s.chunked_req = req
    s.waiting_queue = [MagicMock()]
    s.min_free_slots_delayer = None
    s.get_num_allocatable_reqs = lambda _: 2
    s.policy = SimpleNamespace(calc_priority=lambda *args: None)
    s.chunked_prefill_size = 4
    s.enable_dynamic_chunking = False
    s.page_size = adder.page_size
    s.tree_cache = adder.tree_cache
    s.token_to_kv_pool_allocator = adder.token_to_kv_pool_allocator
    s.new_token_ratio_tracker = SimpleNamespace(current=0.1)
    s.max_prefill_tokens = 100
    s.is_mixed_chunk = False
    s.priority_scheduling_preemption_threshold = 0
    s.max_prefill_bs = s.max_running_requests = 2
    s.dllm_config = None
    s.hisparse_coordinator = c
    monkeypatch.setattr(
        module, "get_memory", lambda: SimpleNamespace(enable_flexkv=False)
    )
    monkeypatch.setattr(
        module, "get_schedule", lambda: SimpleNamespace(prefill_max_requests=None)
    )
    monkeypatch.setattr(module, "PrefillAdder", lambda *args, **kwargs: adder)
    result, returned_running = s._get_new_batch_prefill_raw(None, running)
    assert result is None and returned_running is running
    assert s.chunked_req is req
    s.waiting_queue[0].init_next_round_input.assert_not_called()
    assert adder.kvduo_prefill_blocked and not adder.can_run_list


def test_next_decode_headroom_does_not_double_count_first_decode_hold():
    c, physical = ready_coordinator()
    req = SimpleNamespace(
        kv_committed_len=6,
        rid="decode",
        kvduo_prefill_generation=1,
        finished=lambda: False,
    )
    c._active_kvduo_reqs[0] = req
    assert c._kvduo_next_decode_pages() == 1
    c._kvduo_prefill_future_pages[("decode", 1)] = 1
    assert c._kvduo_next_decode_pages() == 0
    req.kvduo_prefill_generation = 2
    assert c._kvduo_next_decode_pages() == 1
