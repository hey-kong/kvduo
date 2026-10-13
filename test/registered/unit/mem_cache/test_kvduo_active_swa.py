"""KVDuo keeps SWA for active computation without retaining idle prefixes."""

from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
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


def test_completed_chunk_snapshot_preserves_inflight_chunk_and_full():
    cache, allocator, _ = make_cache()
    slots = alloc(allocator, 24)
    req = make_req(cache, range(24), slots=slots)
    full_available = allocator.full_available_size()
    batch = ScheduleBatch(reqs=[req], prefix_lens=[0], extend_lens=[12])
    batch.tree_cache = cache
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
def test_result_handler_cleans_completed_prefill_after_copy_done(middle_chunk):
    cache, allocator, _ = make_cache()
    cache.kvduo_coordinator.end_kvduo_prefill = MagicMock()
    slots = alloc(allocator, 24)
    req = make_req(cache, range(24), slots=slots)
    req.sampling_params.max_new_tokens = 1
    req.inflight_middle_chunks = int(middle_chunk)
    batch = ScheduleBatch(reqs=[req], prefix_lens=[0], extend_lens=[24])
    batch.tree_cache = cache
    batch.forward_mode = ForwardMode.EXTEND
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
        assert req.kv.swa_evicted_seqlen == 20
        assert allocator.swa_available_size() == 28
        assert torch.all(mapped(allocator, slots[20:]) > 0)
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
