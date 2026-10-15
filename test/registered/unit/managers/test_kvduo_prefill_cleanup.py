"""Prefill finish and abort paths release generation-bound Host holds."""

from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import (  # noqa: E402
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
    Req,
    ReqKvInfo,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.batch_result_processor import (  # noqa: E402
    SchedulerBatchResultProcessor,
)
from sglang.srt.mem_cache.common import release_kv_cache  # noqa: E402
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.server_args import (  # noqa: E402
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from test_kvduo_host_capacity import cached_page, host_inventory  # noqa: E402
from test_kvduo_prefill_admission import ready_coordinator  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def cleanup_stack(*, ratio=1):
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", enable_kvduo=True, enable_hisparse=True)
    )
    c, physical = ready_coordinator(extra=4, requests=2)
    c.compress_ratio = ratio
    c.is_dsv4_hisparse = ratio > 1
    physical.free_pages = 64
    pool = host_inventory(c, 8)
    request_pool = ReqToTokenPool(1, 32 * ratio, "cpu", enable_memory_saver=False)
    cache = SimpleNamespace(
        kvduo_coordinator=c,
        req_to_token_pool=request_pool,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=2 * ratio),
        supports_mamba=lambda: False,
        cache_finished_req=MagicMock(),
        evictable_size=lambda: 0,
        evict=MagicMock(),
    )
    return c, pool, cache


def pending_prefill(c, pool, cache, *, mode="length", rid="incoming", finish=True):
    sampling = SamplingParams(max_new_tokens=1 if mode == "length" else 100)
    sampling.normalize(None)
    length = 8 * c.compress_ratio
    req = Req(
        rid,
        "",
        array("q", range(length)),
        sampling,
        eos_token_ids={7},
        vocab_size=128,
    )
    req._refresh_fill_ids()
    cache.req_to_token_pool.alloc([req])
    req.kv = ReqKvInfo(kv_allocated_len=length, swa_evicted_seqlen=0)
    req.kv_committed_len = length
    req.set_extend_range(0, length)
    identity = c._host_prefix_identity(req, 0)
    record = c.host_prefix_cache.records.get(identity)
    if record is None:
        record = cached_page(c, pool.alloc(2), identity=identity)
    c.host_prefix_cache.acquire(identity, req.rid, 0)
    req.kvduo_host_prefix_records.add(identity)
    c.token_to_kv_pool_allocator.hisparse_attn_allocator.free_pages = 8
    token = c.reserve_kvduo_prefill(1, prefill_req=req)
    assert token is not None
    if finish:
        c.finish_kvduo_prefill_reservation(token, cancelled=False)
    key = c._kvduo_prefill_key(req)
    assert c._kvduo_prefill_host_reservations[key][1] == 6
    assert record.pinned
    return req, key, record, token


def processor(c, cache):
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=True,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(enable_return_hidden_states=False),
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
        tree_cache=cache,
        hisparse_coordinator=c,
        req_to_token_pool=cache.req_to_token_pool,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=MagicMock(),
        draft_worker=None,
        model_worker=None,
        logprob_result_processor=None,
        output_streamer=MagicMock(),
        abort_request=MagicMock(),
    )


def prefill_result(req):
    batch = SimpleNamespace(
        reqs=[req],
        decoding_reqs=[],
        return_logprob=False,
        return_hidden_states=False,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
        spec_info=None,
        forward_mode=ForwardMode.EXTEND,
        prefill_stats=None,
        dp_cooperation_info=None,
    )
    result = SimpleNamespace(
        copy_done=None,
        routed_experts_output=None,
        indexer_topk_output=None,
        logits_output=SimpleNamespace(customized_info=None),
        next_token_ids=torch.tensor([7]),
        extend_input_len_per_req=[len(req.origin_input_ids)],
        extend_logprob_start_len_per_req=None,
        grammar_advanced=False,
        can_run_cuda_graph=False,
        skipped_output_comm=False,
    )
    return batch, result


def assert_prefill_released(c, cache, req, key, record):
    assert not c._kvduo_prefill_host_reservations
    assert key not in c._kvduo_prefill_lifecycle
    assert key not in c._kvduo_prefill_future_pages
    assert not record.pinned and not req.kvduo_host_prefix_records
    assert not hasattr(req, "kvduo_prefill_generation")
    assert req.req_pool_idx is None and req.kv is None
    assert cache.req_to_token_pool.available_size() == 1
    assert c._kvduo_host_capacity_ready(6)


@pytest.mark.parametrize("mode", ["length", "eos"])
@pytest.mark.parametrize("ratio", [1, 4])
def test_repeated_first_token_finish_releases_host_hold_and_match_pins(mode, ratio):
    c, pool, cache = cleanup_stack(ratio=ratio)
    c.admit_request_into_staging = MagicMock()
    for i in range(3):
        req, key, record, _ = pending_prefill(c, pool, cache, mode=mode, rid=f"req-{i}")
        before = c.req_device_buffer_tokens.clone()
        cache.cache_finished_req.side_effect = lambda *args, **kwargs: (
            assert_host_released_before_cache(c, record)
        )
        batch, result = prefill_result(req)
        processor(c, cache).process_batch_result_prefill(batch, result)
        reason = FINISH_LENGTH if mode == "length" else FINISH_MATCHED_TOKEN
        assert isinstance(req.finished_reason, reason)
        assert_prefill_released(c, cache, req, key, record)
        assert torch.equal(c.req_device_buffer_tokens, before)
    c.admit_request_into_staging.assert_not_called()


def assert_host_released_before_cache(c, record):
    assert not c._kvduo_prefill_host_reservations and not record.pinned


@pytest.mark.parametrize("inflight", [1, 2])
@pytest.mark.parametrize("ratio", [1, 4])
def test_chunk_abort_releases_host_before_slot_and_late_results_do_not_recreate_it(
    inflight,
    ratio,
):
    c, pool, cache = cleanup_stack(ratio=ratio)
    req, key, record, _ = pending_prefill(c, pool, cache)
    req.inflight_middle_chunks = inflight
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._pending_chunked_abort_req = scheduler.chunked_req = req
    scheduler.disaggregation_mode = None
    scheduler.enable_hicache_storage = False
    scheduler.tree_cache = cache
    scheduler.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
    scheduler.process_pending_chunked_abort()
    assert isinstance(req.finished_reason, FINISH_ABORT)
    assert scheduler.chunked_req is scheduler._pending_chunked_abort_req is None
    assert cache.cache_finished_req.call_args.kwargs["is_insert"] is False
    assert_prefill_released(c, cache, req, key, record)
    for _ in range(inflight):
        batch, result = prefill_result(req)
        processor(c, cache).process_batch_result_prefill(batch, result)
    assert req.inflight_middle_chunks == 0 and not req.output_ids
    assert_prefill_released(c, cache, req, key, record)
    cache.cache_finished_req.assert_called_once()


def test_request_end_cancels_pending_physical_reservation_before_scope_cleanup():
    c, pool, cache = cleanup_stack()
    with c.kvduo_prefill_reservation_scope():
        req, key, record, _ = pending_prefill(c, pool, cache, finish=False)
        release_kv_cache(req, cache, is_insert=False)
        assert_prefill_released(c, cache, req, key, record)
        assert not c._kvduo_prefill_reservations
        assert not c._kvduo_prefill_reservation_requests


def test_repeated_cleanup_of_old_request_does_not_release_reused_slots_host_hold():
    c, pool, cache = cleanup_stack()
    old, old_key, record, _ = pending_prefill(c, pool, cache, rid="same-rid")
    c.end_kvduo_prefill(old)
    release_kv_cache(old, cache)
    assert_prefill_released(c, cache, old, old_key, record)
    new, new_key, record, _ = pending_prefill(c, pool, cache, rid="same-rid")
    assert new_key != old_key and new.req_pool_idx == 1
    c.end_kvduo_prefill(old)
    assert c._kvduo_prefill_host_reservations[new_key][0] is new
    assert record.pinned and new_key in c._kvduo_prefill_lifecycle
    release_kv_cache(new, cache)
    assert_prefill_released(c, cache, new, new_key, record)


@pytest.mark.parametrize("attached", [False, True])
def test_kv_release_without_kvduo_retains_existing_behavior(attached):
    c, pool, cache = cleanup_stack()
    req, key, record, _ = pending_prefill(c, pool, cache)
    c.end_kvduo_prefill(req)
    if attached:
        c.enable_mixed_residency = False
    else:
        del cache.kvduo_coordinator
    release_kv_cache(req, cache)
    assert_prefill_released(c, cache, req, key, record)
