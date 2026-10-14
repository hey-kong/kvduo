"""KVDuo keeps SWA for active computation without retaining idle prefixes."""

from array import array
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.sparsity.core.kvduo_prefix_cache import (
    HostPrefixRecord,
    KVDuoHostPrefixCache,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_cache(*, kvduo=True, page=4, window=4, swa_size=32):
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=page))
    pool = ReqToTokenPool(4, 64, "cpu", enable_memory_saver=False)
    kv_pool = SWAKVPool(
        size=64,
        size_swa=swa_size,
        page_size=page,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=8,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        device="cpu",
    )
    allocator = SWATokenToKVPoolAllocator(
        size=64,
        size_swa=swa_size,
        page_size=page,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page,
            sliding_window_size=window,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
        )
    )
    cache.kvduo_coordinator = SimpleNamespace(enable_mixed_residency=kvduo)
    return cache, allocator, pool


def alloc(allocator, count):
    full = allocator.full_attn_allocator.alloc(count)
    swa = allocator.swa_attn_allocator.alloc(count)
    assert full is not None and swa is not None
    allocator.full_to_swa_index_mapping[full] = swa
    return full


def make_req(cache, tokens, *, slots=None, match=None, rid="request"):
    req = Req(rid, "", array("q", tokens), SamplingParams(max_new_tokens=8))
    cache.req_to_token_pool.alloc([req])
    req.kv = ReqKvInfo(kv_allocated_len=len(tokens), swa_evicted_seqlen=0)
    req.kv_committed_len = len(tokens)
    req._refresh_fill_ids()
    req.set_extend_range(0, len(tokens))
    req.last_node = (
        cache.root_node_handle() if match is None else match.last_device_node
    )
    req.cache_protected_len = 0 if match is None else len(match.device_indices)
    if match is not None:
        lock = cache.inc_lock_ref(req.last_node)
        req.swa_uuid_for_lock = lock.swa_uuid_for_lock
        req.skip_lock_node_ids = lock.skip_lock_node_ids
    if slots is not None:
        cache.req_to_token_pool.req_to_token[req.req_pool_idx, : len(slots)] = slots
    return req


def insert(cache, slots, tokens):
    cache.insert(InsertParams(key=RadixKey(array("q", tokens)), value=slots))
    return cache._match_prefix_without_kvduo(
        MatchPrefixParams(key=RadixKey(array("q", tokens)))
    )


def mapped(allocator, slots):
    return allocator.full_to_swa_index_mapping[slots]


def host_prefix_stack(gpu_prefix_len):
    cache, allocator, _ = make_cache()
    slots = alloc(allocator, 16)
    insert(cache, slots, range(16))
    if gpu_prefix_len:
        match = cache._match_prefix_without_kvduo(
            MatchPrefixParams(key=RadixKey(array("q", range(gpu_prefix_len))))
        )
        make_req(cache, range(16), match=match, rid="active-prefix")
    cache.release_kvduo_cached_swa()

    c = HiSparseCoordinator.__new__(HiSparseCoordinator)
    c.enable_mixed_residency = c.is_dsv4_hisparse = True
    c.page_size, c.compress_ratio, c.tp_world_size = 1, 4, 1
    c.device = "cpu"
    c.host_prefix_cache = KVDuoHostPrefixCache(32)
    c.token_to_kv_pool_allocator = SimpleNamespace(alloc_kvduo_restore=MagicMock())
    cache.kvduo_coordinator = c
    req = Req("incoming", "", array("q", range(20)), SamplingParams(max_new_tokens=8))
    req._refresh_fill_ids()
    for ordinal in range(4):
        identity = c._host_prefix_identity(req, ordinal)
        c.host_prefix_cache.insert(
            HostPrefixRecord(
                identity,
                domains=("main_kv",),
                host_locations={"main_kv": (ordinal,)},
                data_versions={"main_kv": (1,)},
                host_versions={"main_kv": (1,)},
            )
        )
    return cache, allocator, c, req


@pytest.mark.parametrize("gpu_prefix_len", [0, 8, 16])
def test_main_only_host_hit_cannot_bypass_swa_gate_or_change_prefill_boundary(
    gpu_prefix_len,
):
    cache, allocator, c, req = host_prefix_stack(gpu_prefix_len)
    # Exercise release of references from an earlier speculative match too.
    identity = next(iter(c.host_prefix_cache.records))
    c.host_prefix_cache.acquire(identity, req.rid, 0)
    req.kvduo_host_prefix_records = {identity}
    req.kvduo_residency_plan = object()
    req.init_next_round_input(cache)
    assert len(req.prefix_indices) == gpu_prefix_len
    assert req.host_hit_length == 0
    assert req.kvduo_residency_plan is None
    assert not req.kvduo_host_prefix_records
    assert all(
        record.fully_valid and not record.pinned
        for record in c.host_prefix_cache.records.values()
    )

    adder = PrefillAdder(
        page_size=cache.page_size,
        tree_cache=cache,
        token_to_kv_pool_allocator=allocator,
        running_batch=ScheduleBatch(reqs=[]),
        new_token_ratio=0,
        rem_input_tokens=64,
        rem_chunk_tokens=None,
    )
    assert adder.add_one_req(req, False, None) is AddReqResult.CONTINUE
    assert req.extend_range.start == gpu_prefix_len
    assert req.extend_range.end == 20
    # The first query must have every preceding SWA KV it can read. A main-only
    # Host extension to 16 would fail this assertion when the GPU hit is 0/8.
    preceding = req.prefix_indices[max(0, gpu_prefix_len - 3) : gpu_prefix_len]
    assert torch.all(mapped(allocator, preceding) > 0)
    c.token_to_kv_pool_allocator.alloc_kvduo_restore.assert_not_called()


def test_dsv4_load_back_rejects_stale_main_only_plan_before_pins_or_allocation():
    _, _, c, req = host_prefix_stack(0)
    req.kvduo_residency_plan = object()
    req.prefix_indices = torch.empty(0, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="required SWA and boundary state"):
        c.init_kvduo_load_back(req, 16)
    c.token_to_kv_pool_allocator.alloc_kvduo_restore.assert_not_called()
    assert all(not record.pinned for record in c.host_prefix_cache.records.values())


@pytest.mark.parametrize("finished", [True, False])
def test_host_restore_ownership_is_not_confused_with_shared_radix_prefix(finished):
    cache, allocator, _ = make_cache()
    tokens = list(range(16))
    original = alloc(allocator, 16)
    source = make_req(cache, tokens, slots=original, rid="source")
    cache.cache_finished_req(source, kv_len_to_handle=16)
    restored = alloc(allocator, 16)
    target = make_req(cache, tokens, slots=restored, rid="target")
    target.cache_protected_len = 16
    target.kvduo_shared_prefix_len = 0
    allocator.free_swa(restored[:12])
    target.swa_evicted_seqlen = 12
    if finished:
        cache.cache_finished_req(target, kv_len_to_handle=16)
        assert allocator.swa_available_size() == allocator.size_swa
    else:
        cache.cache_unfinished_req(target, chunked=True)
        assert not hasattr(target, "kvduo_shared_prefix_len")
        assert torch.all(mapped(allocator, target.prefix_indices[-4:]) > 0)
        cache.cache_finished_req(target, kv_len_to_handle=16)
    assert allocator.full_available_size() == allocator.size_full - 16
    assert allocator.swa_available_size() == allocator.size_swa


def _swa_host_match_tp_worker(rank, init_file):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        cache, _, c, req = host_prefix_stack(8 if rank == 0 else 0)
        c.tp_world_size, c.tp_group = 2, dist.group.WORLD
        req.init_next_round_input(cache)
        assert len(req.prefix_indices) == 0
        assert req.host_hit_length == 0
        assert req.kvduo_residency_plan is None
        assert all(not record.pinned for record in c.host_prefix_cache.records.values())
    finally:
        dist.destroy_process_group()


def test_tp_swa_validity_divergence_rematches_to_common_safe_gpu_boundary(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _swa_host_match_tp_worker,
        args=(str(tmp_path / "swa-host-match-gloo"),),
        nprocs=2,
        join=True,
        start_method="spawn",
    )


def test_completed_chunk_snapshot_preserves_inflight_chunk_and_full():
    cache, allocator, _ = make_cache()
    slots = alloc(allocator, 24)
    req = make_req(cache, range(24), slots=slots)
    full_available = allocator.full_available_size()
    batch = ScheduleBatch(reqs=[req], prefix_lens=[0], extend_lens=[12])
    batch.tree_cache = cache
    batch = batch.copy()
    assert batch.tree_cache is cache
    # The request already describes the next in-flight chunk ending at 24.
    ScheduleBatch.release_kvduo_prefill_swa(batch)
    assert req.kv.swa_evicted_seqlen == 8
    assert torch.all(mapped(allocator, slots[:8]) == 0)
    assert torch.all(mapped(allocator, slots[8:]) > 0)
    assert allocator.full_available_size() == full_available
    cache.maintain_kvduo_swa([(req, 24)])
    assert torch.all(mapped(allocator, slots[:20]) == 0)
    assert torch.all(mapped(allocator, slots[20:]) > 0)
    assert allocator.swa_available_size() == 28


@pytest.mark.parametrize("middle_chunk", [True, False])
@pytest.mark.parametrize("overlap", [True, False])
def test_result_handler_cleans_completed_prefill_after_copy_done(middle_chunk, overlap):
    cache, allocator, _ = make_cache()
    cache.kvduo_coordinator.end_kvduo_prefill = MagicMock()
    slots = alloc(allocator, 24)
    req = make_req(cache, range(24), slots=slots)
    req.sampling_params.max_new_tokens = 1
    req.inflight_middle_chunks = int(middle_chunk)
    batch = ScheduleBatch(reqs=[req], prefix_lens=[0], extend_lens=[24])
    batch.tree_cache = cache
    batch.forward_mode = ForwardMode.EXTEND
    if overlap:
        copied = batch.copy()
        # The next chunk may already have replaced the scheduling metadata.
        batch.prefix_lens = [24]
        batch.extend_lens = [8]
        batch = copied
        assert batch.tree_cache is cache
    result = SimpleNamespace(
        copy_done=MagicMock(),
        routed_experts_output=None,
        indexer_topk_output=None,
        logits_output=SimpleNamespace(),
        next_token_ids=torch.tensor([7]),
        extend_input_len_per_req=[24],
        extend_logprob_start_len_per_req=None,
        grammar_advanced=False,
        can_run_cuda_graph=False,
    )

    def before_completion():
        assert torch.all(mapped(allocator, slots) > 0)

    result.copy_done.synchronize.side_effect = before_completion
    processor = MagicMock(is_generation=True, tree_cache=cache)
    SchedulerBatchResultProcessor.process_batch_result_prefill(processor, batch, result)
    result.copy_done.synchronize.assert_called_once()
    if middle_chunk:
        assert req.kv.swa_evicted_seqlen == 0
        assert allocator.swa_available_size() == 8
        assert torch.all(mapped(allocator, slots) > 0)
    else:
        assert req.finished()
        assert req.kv is None
        assert allocator.swa_available_size() == 32


def test_reusing_freed_swa_pages_preserves_active_window_contents():
    cache, allocator, _ = make_cache()
    slots = alloc(allocator, 24)
    req = make_req(cache, range(24), slots=slots)
    swa = mapped(allocator, slots).clone()
    data = allocator.get_kvcache().get_key_buffer(1)
    data[swa] = torch.arange(24, dtype=data.dtype).view(24, 1, 1)
    window = data[swa[20:]].clone()
    cache.maintain_kvduo_swa([(req, 24)])
    new_slots = alloc(allocator, 24)
    data[mapped(allocator, new_slots)] = -1
    assert torch.equal(data[swa[20:]], window)


def test_idle_cached_swa_release_keeps_full_and_invalidates_prefix_match():
    cache, allocator, _ = make_cache()
    slots = alloc(allocator, 16)
    match = insert(cache, slots, range(16))
    full = cache.all_values_flatten().clone()
    available = allocator.full_available_size()
    cache.release_kvduo_cached_swa()
    assert cache.tree_core.node_by_id(match.last_device_node) is not None
    assert torch.equal(cache.all_values_flatten(), full)
    assert allocator.full_available_size() == available
    assert allocator.swa_available_size() == 32
    assert torch.all(mapped(allocator, slots) == 0)
    # Missing SWA must shorten the reusable prefix, rather than read freed KV.
    rematch = cache._match_prefix_without_kvduo(
        MatchPrefixParams(key=RadixKey(array("q", range(16))))
    )
    assert len(rematch.device_indices) == 0
    assert cache.swa_evictable_size() == cache.swa_protected_size() == 0
    cache.sanity_check()


def test_shared_prefix_lock_expires_independently_of_full_lock():
    cache, allocator, _ = make_cache()
    slots = alloc(allocator, 16)
    match = insert(cache, slots, range(16))
    req1 = make_req(cache, range(20), match=match, rid="first")
    req2 = make_req(cache, range(16), match=match, rid="second")
    assert cache.swa_protected_size() == 4
    node = cache.tree_core.node_by_id(match.last_device_node)
    assert node.component_data[ComponentType.SWA].lock_ref == 2
    # Request 1 has passed the shared window. Request 2 can still prefill/decode it.
    cache.maintain_kvduo_swa([(req1, 20)])
    assert req1.swa_prefix_lock_released
    assert not req2.swa_prefix_lock_released
    assert node.component_data[ComponentType.SWA].lock_ref == 1
    assert torch.all(mapped(allocator, slots[-4:]) > 0)
    assert torch.all(mapped(allocator, slots[:-4]) == 0)
    assert cache.full_protected_size() == 16
    # Ending request 2 drops only its remaining references.
    cache._dec_req_lock(req2)
    cache.release_kvduo_cached_swa()
    assert allocator.swa_available_size() == 32
    assert cache.full_protected_size() == 16
    cache._dec_req_lock(req1, skip_swa=True)
    cache.sanity_check()


def test_match_inside_existing_large_node_locks_only_a_window():
    cache, allocator, _ = make_cache()
    slots = alloc(allocator, 24)
    insert(cache, slots, range(24))
    match = cache._match_prefix_without_kvduo(
        MatchPrefixParams(key=RadixKey(array("q", range(16))))
    )
    node = cache.tree_core.node_by_id(match.last_device_node)
    assert len(node.key) == 16
    req = make_req(cache, range(16), match=match)
    assert len(node.key) == 4
    assert cache.swa_protected_size() == 4
    cache.maintain_kvduo_swa([(req, 16)])
    assert torch.all(mapped(allocator, slots[12:16]) > 0)
    assert torch.all(mapped(allocator, slots[:12]) == 0)
    assert cache.full_protected_size() == 16
    cache._dec_req_lock(req)
    cache.release_kvduo_cached_swa()
    cache.sanity_check()


def test_host_restored_suffix_is_request_owned_despite_full_protection():
    cache, allocator, _ = make_cache()
    slots = alloc(allocator, 24)
    req = make_req(cache, range(24), slots=slots)
    req.cache_protected_len = 20
    cache.maintain_kvduo_swa([(req, 24)])
    assert req.cache_protected_len == 20
    assert req.kv.swa_evicted_seqlen == 20
    assert torch.all(mapped(allocator, slots[:20]) == 0)
    assert torch.all(mapped(allocator, slots[20:]) > 0)


@pytest.mark.parametrize("page,window", [(4, 2), (1, 4)])
def test_window_floor_and_partial_page_remain_resident(page, window):
    cache, allocator, _ = make_cache(page=page, window=window)
    slots = alloc(allocator, 24)
    req = make_req(cache, range(23), slots=slots)
    req.swa_evict_floor = 5
    cache.maintain_kvduo_swa([(req, 23)])
    floor = (5 + page - 1) // page * page
    end = (23 - max(page, window)) // page * page
    assert torch.all(mapped(allocator, slots[:floor]) > 0)
    assert torch.all(mapped(allocator, slots[floor:end]) == 0)
    assert torch.all(mapped(allocator, slots[end:]) > 0)


def test_decode_checks_swa_after_window_cleanup_before_retraction():
    cache, allocator, _ = make_cache(swa_size=24)
    slots = alloc(allocator, 24)
    req = make_req(cache, range(24), slots=slots)
    req.output_ids = [1]
    req.decode_batch_idx = 1
    assert allocator.available_size() == 0
    coordinator = SimpleNamespace(
        enable_mixed_residency=True,
        ensure_kvduo_decode_capacity=MagicMock(),
        log_kvduo_decode_pressure=MagicMock(),
    )
    from sglang.srt.mem_cache.sparsity.core.kvduo_state import KVDuoPressureAction

    coordinator.ensure_kvduo_decode_capacity.return_value = SimpleNamespace(
        action=KVDuoPressureAction.SUCCESS
    )
    batch = ScheduleBatch(reqs=[req])
    batch.tree_cache = cache
    batch.hisparse_coordinator = coordinator
    batch.token_to_kv_pool_allocator = SimpleNamespace(logical_attn_allocator=allocator)
    batch.new_tokens_required_next_decode = lambda _: 4
    assert batch.check_decode_mem()
    assert allocator.swa_available_size() == 20
    coordinator.log_kvduo_decode_pressure.assert_not_called()
    assert alloc(allocator, 4) is not None


@pytest.mark.parametrize("is_insert", [True, False])
def test_request_finish_or_cancel_releases_all_swa_and_can_reuse_slot(is_insert):
    cache, allocator, pool = make_cache()
    slots = alloc(allocator, 24)
    req = make_req(cache, range(24), slots=slots)
    cache.maintain_kvduo_swa([(req, 24)])
    cache.cache_finished_req(req, is_insert=is_insert, kv_len_to_handle=24)
    assert allocator.swa_available_size() == 32
    assert allocator.full_available_size() == (40 if is_insert else 64)
    pool.free(req)
    fresh = alloc(allocator, 24)
    new_req = make_req(cache, range(24), slots=fresh, rid="replacement")
    assert new_req.kv.swa_evicted_seqlen == 0
    assert not new_req.swa_prefix_lock_released
    cache.maintain_kvduo_swa([(new_req, 24)])
    cache.cache_finished_req(new_req, is_insert=is_insert, kv_len_to_handle=24)
    assert allocator.swa_available_size() == 32
    assert allocator.full_available_size() == (40 if is_insert else 64)
    cache.sanity_check()


def test_deepseek_wrapper_trimming_leaves_main_sparse_mapping_untouched():
    from sglang.srt.mem_cache.allocator.hisparse import (
        DeepSeekV4HiSparseTokenToKVPoolAllocator,
    )

    cache, allocator, _ = make_cache()
    slots = alloc(allocator, 24)
    req = make_req(cache, range(24), slots=slots)
    wrapper = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
    wrapper.logical_attn_allocator = allocator
    wrapper.page_size = 4
    wrapper.full_to_hisparse_device_index_mapping = torch.arange(64)
    main_mapping = wrapper.full_to_hisparse_device_index_mapping.clone()
    cache.token_to_kv_pool_allocator = wrapper
    cache.maintain_kvduo_swa([(req, 24)])
    assert torch.equal(wrapper.full_to_hisparse_device_index_mapping, main_mapping)
    assert torch.all(mapped(allocator, slots[:20]) == 0)
    assert torch.all(mapped(allocator, slots[20:]) > 0)


def test_mixed_prefill_does_not_trim_decode_request_from_extend_lengths():
    cache, allocator, _ = make_cache()
    slots = alloc(allocator, 16)
    req = make_req(cache, range(16), slots=slots)
    batch = ScheduleBatch(reqs=[req], prefix_lens=[0], extend_lens=[16])
    batch.tree_cache = cache
    batch.decoding_reqs = [req]
    batch.release_kvduo_prefill_swa()
    assert torch.all(mapped(allocator, slots) > 0)
    assert req.kv.swa_evicted_seqlen == 0


def test_non_kvduo_retains_original_swa_policy():
    cache, allocator, _ = make_cache(kvduo=False)
    slots = alloc(allocator, 16)
    insert(cache, slots, range(16))
    req = make_req(cache, range(16), slots=slots)
    cache.maintain_kvduo_swa([(req, 16)])
    cache.release_kvduo_cached_swa()
    assert allocator.swa_available_size() == 16
    assert torch.all(mapped(allocator, slots) > 0)
