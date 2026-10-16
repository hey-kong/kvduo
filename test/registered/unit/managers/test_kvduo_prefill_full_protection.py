"""Prefill preserves active FULL and holds only admitted execution credit."""

from array import array
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache.sparsity.core.kvduo_state import KVDuoPressureAction
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_decode_budget_sync import decode_case
from test_kvduo_prefill_admission import adder_and_req, ready_coordinator

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_chunk_holds_are_released_between_chunks_and_protect_first_decode():
    c, physical = ready_coordinator()
    _, req = adder_and_req(c)
    physical.free_pages = 8
    req.sampling_params.max_new_tokens = 2
    req.rid = "chunked"
    key = c._kvduo_prefill_key(req)
    # Two tokens already cached; seven more FULL pages plus first decode fit.
    for start, end in [(2, 6), (6, 10), (10, 14), (14, 16)]:
        req.prefix_indices = torch.arange(start)
        pages = c.kvduo_prefill_pages(start, end)
        before = c._kvduo_prefill_future_pages.get(key, 0)
        token = c.reserve_kvduo_prefill(pages, prefill_req=req, prefill_end=end)
        assert token is not None
        c.protect_kvduo_prefill_plan(req, end)
        assert c._kvduo_prefill_plan_uncovered_pages() == 0
        assert c._kvduo_reserved_prefill_pages() == pages + int(end == 16)
        if start == 6:
            # Cancellation restores the previous hold without losing progress.
            c.finish_kvduo_prefill_reservation(token, cancelled=True)
            assert c._kvduo_prefill_future_pages.get(key, 0) == before
            assert c._kvduo_prefill_lifecycle[key] is req
            assert req.kvduo_prefill_generation == key[1]
            token = c.reserve_kvduo_prefill(pages, prefill_req=req, prefill_end=end)
            assert token is not None
        physical.alloc(pages * c.page_size)
        c.kvduo_prefill_batch_allocated()
        c.finish_kvduo_prefill_reservation(token, cancelled=False)
        assert c._kvduo_reserved_prefill_pages() == int(end == 16)
    assert physical.free_pages == c._kvduo_reserved_prefill_pages() == 1
    assert not c._kvduo_prefill_previous_future
    c._end_kvduo_prefill_lifecycle(None, req)
    assert c._kvduo_reserved_prefill_pages() == 0
    assert not hasattr(req, "kvduo_prefill_generation")


@pytest.mark.parametrize("dsv4", [False, True])
@pytest.mark.parametrize("success", [False, True])
def test_first_decode_uses_only_its_own_hold_and_restores_it_on_failure(dsv4, success):
    batch, c, physical, _ = decode_case(dsv4=dsv4, free_pages=2 if success else 1)
    req = batch.reqs[0]
    req.rid = "first-decode"
    key = c._kvduo_prefill_key(req)
    c._kvduo_prefill_future_pages[key] = 1
    c._kvduo_prefill_future_pages[("other", 99)] = 1
    assert batch.check_decode_mem() == success
    assert c._kvduo_prefill_future_pages[key] == 1
    assert c._kvduo_prefill_future_pages[("other", 99)] == 1
    assert physical.free_pages == (2 if success else 1)


def test_decode_exception_restores_reservation():
    batch, c, _, _ = decode_case()
    req = batch.reqs[0]
    key = c._kvduo_prefill_key(req)
    c._kvduo_prefill_future_pages[key] = 1
    c._ensure_kvduo_decode_capacity = MagicMock(side_effect=RuntimeError("failed"))
    with pytest.raises(RuntimeError, match="failed"):
        c.ensure_kvduo_decode_capacity(batch.reqs)
    assert c._kvduo_prefill_future_pages[key] == 1


@pytest.mark.parametrize("with_reservation", [False, True])
def test_host_restore_shortage_never_falls_back_to_active_full_or_ordinary_merge(
    with_reservation,
):
    c, physical = ready_coordinator()
    c._reclaim_for_physical_allocation = MethodType(
        HiSparseCoordinator._reclaim_for_physical_allocation, c
    )
    c.item_size_bytes = 1
    c.reclaim_kvduo_full_pages = MagicMock(return_value=100)
    c.reclaim_kvduo_hot_pages = MagicMock(return_value=100)
    token = None
    if with_reservation:
        physical.free_pages = 1
        token = c.reserve_kvduo_prefill(1)
        assert token is not None
        physical.free_pages = 0
    result = c._reclaim_for_physical_allocation(
        c.page_size,
        prefill_reservation=token,
        allow_hot_reclaim=False,
        allow_full_reclaim=False,
    )
    assert result.action is not KVDuoPressureAction.SUCCESS
    c.reclaim_kvduo_full_pages.assert_not_called()
    c.reclaim_kvduo_hot_pages.assert_not_called()


def test_impossible_whole_prompt_aborts_request_without_scheduler_exception():
    c, physical = ready_coordinator()
    _, req = adder_and_req(c)
    physical.size = 6 * c.page_size
    req.set_finish_with_abort = MagicMock()
    assert c.reserve_kvduo_prefill(2, prefill_req=req) is None
    req.set_finish_with_abort.assert_called_once()
    assert not c._kvduo_prefill_reservations
    assert not c._kvduo_prefill_future_pages
    assert not c._kvduo_prefill_host_reservations


def test_oversized_request_can_advance_as_abort_stub_on_next_scheduler_retry():
    from sglang.srt.managers.schedule_batch import Req, FINISH_ABORT
    from sglang.srt.sampling.sampling_params import SamplingParams

    c, physical = ready_coordinator()
    physical.size = 4 * c.page_size
    physical.free_pages = 4
    req = Req(
        "oversized",
        "",
        array("q", range(16)),
        SamplingParams(max_new_tokens=1),
        vocab_size=128,
    )
    req._refresh_fill_ids()
    with patch(
        "sglang.srt.managers.schedule_batch.get_parallel",
        return_value=SimpleNamespace(tp_rank=0),
    ):
        assert c.reserve_kvduo_prefill(2, prefill_req=req) is None
    assert isinstance(req.to_finish, FINISH_ABORT)
    req._refresh_fill_ids()
    assert len(req.full_untruncated_fill_ids) == 1
    token = c.reserve_kvduo_prefill(1, prefill_req=req)
    assert token is not None
    c.finish_kvduo_prefill_reservation(token, cancelled=True)
    assert not c._kvduo_prefill_future_pages


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("completed", [False, True])
def test_staging_returns_ready_requests_in_both_residency_modes(mixed, completed):
    c = HiSparseCoordinator.__new__(HiSparseCoordinator)
    c.enable_mixed_residency = mixed
    c.tp_world_size = 1
    req = SimpleNamespace(req_pool_idx=0, hisparse_staging=True)
    event = SimpleNamespace(query=lambda: completed)
    c.ack_staging_queue = [(None, event, req)]
    c._skip_first_backup = [False]
    c.alloc_device_buffer = MagicMock()
    assert c.collect_ready_reqs() == ([req] if completed else [])
    assert req.hisparse_staging is not completed
    assert c._skip_first_backup[0] is completed


@pytest.mark.parametrize("dsv4", [False, True])
@pytest.mark.parametrize("admission_first", [False, True])
@pytest.mark.parametrize("failure", [None, "guard", "allocation"])
def test_decode_preparation_consumes_only_batch_credit_after_success(
    dsv4, admission_first, failure
):
    batch, c, physical, _ = decode_case(
        dsv4=dsv4, free_pages=1 if failure == "guard" else 2
    )
    req = batch.reqs[0]
    req.rid = "A"
    req.decode_batch_idx = 0
    key = c._kvduo_prefill_key(req)
    other = ("B", 99)
    c._kvduo_prefill_future_pages.update({key: 1, other: 1})
    if admission_first:
        assert batch.check_decode_mem() == (failure != "guard")
        assert c._kvduo_prefill_future_pages == {key: 1, other: 1}

    batch.sampling_info = SimpleNamespace(
        penalizer_orchestrator=SimpleNamespace(is_required=False)
    )
    batch.model_config = SimpleNamespace(is_encoder_decoder=False)
    batch.seq_lens = torch.tensor([req.kv_committed_len])
    batch.seq_lens_cpu = batch.seq_lens.clone()
    batch.orig_seq_lens = batch.seq_lens.clone()
    batch.req_pool_indices = torch.tensor([0])
    batch.req_pool_indices_cpu = [0]
    c.map_last_loc_to_buffer = MagicMock()
    committed = req.kv_committed_len

    def allocate(current, token_per_req):
        assert current is batch and token_per_req == 1
        assert c._kvduo_prefill_future_pages == {other: 1}
        if not c._guard_kvduo_physical_allocation(c.page_size):
            raise RuntimeError("guard rejected")
        if failure == "allocation":
            raise RuntimeError("allocation failed")
        return physical.alloc(c.page_size)

    with patch(
        "sglang.srt.managers.schedule_batch.alloc_for_decode", side_effect=allocate
    ), patch(
        "sglang.srt.managers.schedule_batch.get_server_args",
        return_value=SimpleNamespace(enable_mamba_extra_buffer=lambda: False),
    ):
        if failure:
            with pytest.raises(RuntimeError, match=failure):
                batch.prepare_for_decode()
            assert c._kvduo_prefill_future_pages == {key: 1, other: 1}
            assert req.kv_committed_len == committed and req.decode_batch_idx == 0
        else:
            batch.prepare_for_decode()
            assert c._kvduo_prefill_future_pages == {other: 1}
            assert physical.free_pages == 1
            assert req.kv_committed_len == committed + 1 and req.decode_batch_idx == 1
            c.reclaim_kvduo_full_pages.assert_not_called()


@pytest.mark.parametrize("ratio,length", [(1, 16), (4, 67)])
@pytest.mark.parametrize("max_output,already_generated", [(1, 0), (2, 1), (2, 0)])
def test_prefill_exact_pool_boundary_charges_decode_only_if_still_needed(
    ratio, length, max_output, already_generated
):
    c, physical = ready_coordinator()
    c.compress_ratio = ratio
    c.is_dsv4_hisparse = ratio > 1
    physical.size = 8 * c.page_size
    physical.free_pages = 8
    req = SimpleNamespace(
        rid="boundary",
        prefix_indices=torch.empty(0, dtype=torch.int64),
        full_untruncated_fill_ids=list(range(length)),
        sampling_params=SimpleNamespace(max_new_tokens=max_output),
        output_ids=[0] * already_generated,
        set_finish_with_abort=MagicMock(),
    )
    token = c.reserve_kvduo_prefill(8, prefill_req=req)
    if max_output - already_generated > 1:
        assert token is None
        req.set_finish_with_abort.assert_called_once()
    else:
        assert token is not None
        req.set_finish_with_abort.assert_not_called()
        assert c._kvduo_reserved_prefill_pages() == 8
        assert c._kvduo_prefill_future_pages[c._kvduo_prefill_key(req)] == 0
        physical.alloc(8 * c.page_size)
        c.finish_kvduo_prefill_reservation(token, cancelled=False)
        assert physical.free_pages == c._kvduo_reserved_prefill_pages() == 0
