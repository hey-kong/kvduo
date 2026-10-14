"""Complete DeepSeek-V4 Host checkpoints, including the preceding SWA window."""

import ctypes
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
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache import memory_pool_host
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.sparsity.core.kvduo_prefill_state import KVDuoPrefillHostCache
from sglang.srt.mem_cache.sparsity.core.kvduo_prefix_cache import KVDuoHostPrefixCache
from sglang.srt.mem_cache.sparsity.core.kvduo_state import KVDuoPressureAction
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def cpu_page_copy(
    src_layers, dst_layers, src_indices, dst_indices, item_size, num_layers
):
    # Substitute only CUDA's byte-copy kernel. Real pool layout, allocation,
    # page translation, ownership and restore admission run unchanged.
    for src, dst in zip(src_layers.tolist(), dst_layers.tolist()):
        for source, target in zip(src_indices.tolist(), dst_indices.tolist()):
            ctypes.memmove(
                dst + target * item_size, src + source * item_size, item_size
            )


def cpu_layer_copy(src, dst, src_indices, dst_indices, item_size):
    cpu_page_copy(
        torch.tensor([src.data_ptr()]),
        torch.tensor([dst.data_ptr()]),
        src_indices,
        dst_indices,
        item_size,
        1,
    )


def stack(monkeypatch, device="cpu"):
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    kv = DeepSeekV4TokenToKVPool(
        max_num_reqs=4,
        swa_size=2048,
        c4_size=1024,
        c128_size=64,
        c4_state_pool_size=136,
        c128_state_pool_size=640,
        page_size=256,
        swa_page_size=128,
        dtype=torch.float8_e4m3fn,
        c4_state_dtype=torch.bfloat16,
        c128_state_dtype=torch.bfloat16,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        indexer_head_dim=128,
        layer_num=2,
        device=device,
        enable_memory_saver=False,
        compression_ratios=[4, 128],
        enable_hisparse=True,
    )
    logical = SWATokenToKVPoolAllocator(
        8192, 2048, 256, kv.dtype, device, kv, need_sort=False
    )
    allocator = DeepSeekV4HiSparseTokenToKVPoolAllocator(logical)
    if device == "cpu":
        monkeypatch.setattr(
            memory_pool_host, "transfer_kv_all_layer_mla", cpu_page_copy, raising=False
        )
        monkeypatch.setattr(
            memory_pool_host, "transfer_kv_per_layer_mla", cpu_layer_copy, raising=False
        )
        for pool in (
            logical.full_attn_allocator,
            logical.swa_attn_allocator,
            allocator.hisparse_attn_allocator,
        ):

            def extend(
                prefix,
                prefix_cpu,
                seq,
                seq_cpu,
                last,
                count,
                num_new_pages=None,
                pool=pool,
            ):
                assert all(int(x) % pool.page_size == 0 for x in prefix_cpu)
                assert count % pool.page_size == 0
                return pool.alloc(count)

            monkeypatch.setattr(pool, "alloc_extend", extend)
    c = HiSparseCoordinator.__new__(HiSparseCoordinator)
    c.device = device
    c.enable_mixed_residency = c.is_dsv4_hisparse = True
    c.token_to_kv_pool_allocator = allocator
    c.mem_pool_device = kv.c4_kv_pool
    c.page_size, c.compress_ratio, c.tp_world_size = 64, 4, 1
    c.item_size_bytes = c.mem_pool_device.bytes_per_page_padded
    c.mem_pool_host = memory_pool_host.DeepSeekV4PagedHostPool(
        "test_c4",
        c.mem_pool_device.kv_buffer,
        c.item_size_bytes,
        32,
        64,
        pin_memory=device != "cpu",
    )
    c.host_prefix_cache = KVDuoHostPrefixCache(c.mem_pool_host.size)
    c.req_to_token_pool = ReqToTokenPool(4, 8192, device, enable_memory_saver=False)
    for name in ("full_data_version", "full_last_touch", "full_generation"):
        setattr(
            c, name, torch.zeros_like(allocator.full_to_hisparse_device_index_mapping)
        )
    c.full_host_version = torch.full_like(c.full_data_version, -1)
    c._init_kvduo_prefill_rollback_state(4)
    c._active_kvduo_reqs = {}
    c.write_staging_stream = (
        torch.cuda.Stream(device=device) if device == "cuda" else None
    )
    c._reclaim_for_physical_allocation = MagicMock(
        return_value=SimpleNamespace(action=KVDuoPressureAction.SUCCESS)
    )
    c.kvduo_prefill_host_cache = KVDuoPrefillHostCache.create(c, 2)
    c.req_to_host_pool = torch.full((4, 2048), -1, dtype=torch.int64, device=device)
    c.req_to_host_pool_allocated_len = torch.zeros(4, dtype=torch.int64)
    c._req_host_prefix_records = [set() for _ in range(4)]
    c.full_touch_clock = torch.zeros(1, dtype=torch.int64, device=device)
    return c


def request(c, rid="source"):
    req = Req(rid, "", array("q", range(1537)), SamplingParams(max_new_tokens=1))
    c.req_to_token_pool.alloc([req])
    req.kv = ReqKvInfo(kv_allocated_len=1536, swa_evicted_seqlen=0)
    req.kv_committed_len = 1536
    req._refresh_fill_ids()
    req.set_extend_range(0, 1537)
    return req


def prefill(c, req):
    allocator = c.token_to_kv_pool_allocator
    logical = allocator.logical_attn_allocator
    full = logical.full_attn_allocator.alloc(1536)
    swa = logical.swa_attn_allocator.alloc(1536)
    logical.set_full_to_swa_mapping(full, swa)
    compressed = c.mem_pool_device.translate_loc_from_full_to_compressed(full)
    physical = allocator.hisparse_attn_allocator.alloc(len(compressed))
    allocator.full_to_hisparse_device_index_mapping[compressed] = physical
    c.req_to_token_pool.req_to_token[req.req_pool_idx, :1536] = full
    c.full_data_version[compressed] = 17
    c.full_last_touch[compressed] = 9
    mirrors = (
        [c.mem_pool_host]
        + list(c.kvduo_prefill_host_cache.history.pools.values())
        + list(c.kvduo_prefill_host_cache.window.pools.values())
    )
    for color, pool in enumerate(mirrors, 1):
        buffers = getattr(
            pool, "device_buffers", getattr(pool, "device_page_views", ())
        )
        for layer, buffer in enumerate(buffers):
            raw = buffer.view(torch.uint8).reshape(buffer.shape[0], -1)
            raw[:] = (
                (torch.arange(len(raw), device=c.device) + color * 20 + layer) % 255
            ).to(torch.uint8)[:, None]
    key = c._kvduo_prefill_key(req)
    assert c._reserve_kvduo_prefill_host(req, key)
    return full


def complete_backup(c, req):
    state = c.kvduo_prefill_host_cache
    state.backup(req, 1536)
    for event, _, _ in state.pending:
        if event is not None:
            event.synchronize()
    state.poll()


def test_backup_preserves_all_prefix_windows_and_keeps_full_history(monkeypatch):
    c = stack(monkeypatch)
    req = request(c)
    prefill(c, req)
    state = c.kvduo_prefill_host_cache
    copies = {}
    for name, pool in {**state.history.pools, **state.window.pools}.items():
        copies[name] = MagicMock(wraps=pool.backup_from_device_all_layer)
        monkeypatch.setattr(pool, "backup_from_device_all_layer", copies[name])
    complete_backup(c, req)
    for name in state.window.pools:
        assert copies[name].call_args.args[2].numel() == 1536
    for name in state.history.pools:
        assert copies[name].call_args.args[2].numel() == 1536
    assert state.ready(req, 0, 1536)
    assert state.ready(req, 0, 1280)
    assert len(state.window.cache.records) == 6
    state.destroy()
    c.mem_pool_host.destroy()


def test_chunks_reuse_host_swa_and_trim_hbm_before_decode_transition(monkeypatch):
    c = stack(monkeypatch)
    req = request(c)
    full = prefill(c, req)
    state = c.kvduo_prefill_host_cache
    cache = swa_cache(c)
    req.last_node = cache.root_node_handle()
    req.inflight_middle_chunks = 1
    batch = ScheduleBatch(
        reqs=[req], tree_cache=cache, prefix_lens=[0], extend_lens=[1024]
    )
    batch.copy().release_kvduo_prefill_swa()
    swa = c.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(full)
    assert torch.all(swa[:768] == 0)
    assert torch.all(swa[768:] > 0)
    assert state.ready(req, 0, 512)
    assert state.ready(req, 0, 1024)
    copies = {}
    for name, pool in state.window.pools.items():
        copies[name] = MagicMock(wraps=pool.backup_from_device_all_layer)
        monkeypatch.setattr(pool, "backup_from_device_all_layer", copies[name])
    req.inflight_middle_chunks = 0
    batch.prefix_lens = [1024]
    batch.extend_lens = [512]
    batch.copy().release_kvduo_prefill_swa()
    for copy in copies.values():
        assert copy.call_args.args[2].numel() == 512
    swa = c.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(full)
    assert torch.all(swa[:-256] == 0)
    assert torch.all(swa[-256:] > 0)
    assert all(state.ready(req, 0, end) for end in range(256, 1537, 256))
    target = request(c, "shorter")
    target.origin_input_ids = array("q", range(1025))
    target._refresh_fill_ids()
    target.set_extend_range(0, 1025)
    assert host_match(c, target).host_hit_length == 1024
    state.destroy()
    c.mem_pool_host.destroy()


def test_prompt_larger_than_swa_pool_completes_chunked_prefill(monkeypatch):
    c = stack(monkeypatch)
    req = request(c)
    req.origin_input_ids = array("q", range(4096))
    req._refresh_fill_ids()
    req.set_extend_range(0, 4096)
    req.kv_committed_len = req.kv.kv_allocated_len = 0
    cache = swa_cache(c)
    req.last_node = cache.root_node_handle()
    allocator = c.token_to_kv_pool_allocator
    logical = allocator.logical_attn_allocator
    state = c.kvduo_prefill_host_cache
    assert c._reserve_kvduo_prefill_host(req, c._kvduo_prefill_key(req))
    assert allocator.swa_available_size() == 2048
    for start in range(0, 4096, 512):
        req.prefix_indices = c.req_to_token_pool.req_to_token[req.req_pool_idx, :start]
        # Exercise actual scheduler budgets and allocator sizes. No HOT/FULL
        # pressure is involved; the independent SWA pool must make progress.
        adder = PrefillAdder(256, cache, allocator, None, 1, 8192, 512)
        adder.add_chunked_req(req)
        assert adder.can_run_list == [req], f"SWA admission stalled at {start}"
        assert req.extend_range.length == 512
        full = logical.full_attn_allocator.alloc(512)
        swa = logical.swa_attn_allocator.alloc(512)
        assert full is not None and swa is not None
        logical.set_full_to_swa_mapping(full, swa)
        compressed = c.mem_pool_device.translate_loc_from_full_to_compressed(full)
        physical = allocator.hisparse_attn_allocator.alloc(len(compressed))
        assert physical is not None
        allocator.full_to_hisparse_device_index_mapping[compressed] = physical
        c.full_data_version[compressed] = 1
        c.req_to_token_pool.req_to_token[req.req_pool_idx, start : start + 512] = full
        req.kv_committed_len = req.kv.kv_allocated_len = start + 512
        req.inflight_middle_chunks = int(start + 512 < 4096)
        batch = ScheduleBatch(
            reqs=[req], tree_cache=cache, prefix_lens=[start], extend_lens=[512]
        )
        batch.copy().release_kvduo_prefill_swa()
        assert allocator.swa_available_size() == 1792
        assert state.ready(req, 0, start + 512)
    assert req.kv.swa_evicted_seqlen == 3840
    assert all(state.ready(req, 0, end) for end in range(256, 4097, 256))
    state.destroy()
    c.mem_pool_host.destroy()


def test_backup_does_not_download_synchronous_gpu_metadata(monkeypatch):
    c = stack(monkeypatch)
    req = request(c)
    prefill(c, req)
    # Any Tensor.cpu() here would serialize scheduler progress on CUDA.
    with monkeypatch.context() as checks:
        checks.setattr(
            torch.Tensor,
            "cpu",
            MagicMock(side_effect=AssertionError("blocking metadata download")),
        )
        complete_backup(c, req)
    assert c.kvduo_prefill_host_cache.ready(req, 0, 1536)
    c.kvduo_prefill_host_cache.destroy()
    c.mem_pool_host.destroy()


def test_aligned_repeat_reuses_previous_checkpoint_boundary(monkeypatch):
    c = stack(monkeypatch)
    source = request(c)
    prefill(c, source)
    complete_backup(c, source)
    target = request(c, "target")
    target.origin_input_ids = array("q", range(1536))
    target._refresh_fill_ids()
    target.set_extend_range(0, 1536)
    assert host_match(c, target).host_hit_length == 1280
    c.kvduo_prefill_host_cache.destroy()
    c.mem_pool_host.destroy()


def test_invalid_window_is_not_published_and_can_be_backed_up_again(monkeypatch):
    c = stack(monkeypatch)
    req = request(c)
    full = prefill(c, req)
    allocator = c.token_to_kv_pool_allocator
    swa = allocator.translate_loc_from_full_to_swa(full[-256:]).clone()
    allocator.logical_attn_allocator.full_to_swa_index_mapping[full[-256:]] = 0
    complete_backup(c, req)
    state = c.kvduo_prefill_host_cache
    assert not state.ready(req, 0, 1536)
    assert state.ready(req, 0, 1280)
    assert state.window.cache.used_slots == 1280 * len(state.window.pools)
    allocator.logical_attn_allocator.set_full_to_swa_mapping(full[-256:], swa)
    complete_backup(c, req)
    assert state.ready(req, 0, 1536)
    assert state.window.cache.used_slots == 1536 * len(state.window.pools)
    state.destroy()
    c.mem_pool_host.destroy()


def swa_cache(c):
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=c.req_to_token_pool,
            token_to_kv_pool_allocator=c.token_to_kv_pool_allocator,
            page_size=256,
            sliding_window_size=128,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
        )
    )
    cache.kvduo_coordinator = c
    return cache


def test_completed_chunk_frees_known_live_pages_without_gpu_unique(monkeypatch):
    c = stack(monkeypatch)
    req = request(c)
    full = prefill(c, req)
    cache = swa_cache(c)
    req.last_node = cache.root_node_handle()
    batch = ScheduleBatch(
        reqs=[req], tree_cache=cache, prefix_lens=[0], extend_lens=[1536]
    )
    with monkeypatch.context() as checks:
        checks.setattr(
            torch,
            "unique",
            MagicMock(side_effect=AssertionError("GPU unique would synchronize")),
        )
        batch.copy().release_kvduo_prefill_swa()
        cache.maintain_kvduo_swa([(req, 1536)])
    mapping = c.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(full)
    assert torch.all(mapping[:-256] == 0)
    assert torch.all(mapping[-256:] > 0)
    assert c.token_to_kv_pool_allocator.swa_available_size() == 2048 - 256
    assert req.kv.swa_evicted_seqlen == 1280
    assert c.kvduo_prefill_host_cache.ready(req, 0, 1280)
    assert c.kvduo_prefill_host_cache.ready(req, 0, 1536)
    c.kvduo_prefill_host_cache.destroy()
    c.mem_pool_host.destroy()


def test_restored_prefix_skips_absent_history_before_fixed_shape_swa_release(
    monkeypatch,
):
    c = stack(monkeypatch)
    source = request(c)
    full = prefill(c, source)
    complete_backup(c, source)
    c.end_kvduo_prefill(source)
    c.token_to_kv_pool_allocator.free(full)
    target = request(c, "target")
    result = host_match(c, target)
    target.prefix_indices = result.device_indices
    target.kvduo_residency_plan = result.kvduo_residency_plan
    restored = c.init_kvduo_load_back(target, result.host_hit_length)
    assert target.kvduo_swa_live_from == 1280
    c.req_to_token_pool.req_to_token[target.req_pool_idx, :1536] = restored
    cache = swa_cache(c)
    target.last_node = cache.root_node_handle()
    with monkeypatch.context() as checks:
        checks.setattr(
            torch,
            "unique",
            MagicMock(side_effect=AssertionError("GPU unique would synchronize")),
        )
        cache.maintain_kvduo_swa([(target, 1536)])
        assert c.token_to_kv_pool_allocator.swa_available_size() == 1792
        logical = c.token_to_kv_pool_allocator.logical_attn_allocator
        fresh = logical.full_attn_allocator.alloc(256)
        logical.set_full_to_swa_mapping(fresh, logical.swa_attn_allocator.alloc(256))
        c.req_to_token_pool.req_to_token[target.req_pool_idx, 1536:1792] = fresh
        cache.maintain_kvduo_swa([(target, 1792)])
    assert c.token_to_kv_pool_allocator.swa_available_size() == 1792
    assert torch.all(logical.translate_loc_from_full_to_swa(restored) == 0)
    assert torch.all(logical.translate_loc_from_full_to_swa(fresh) > 0)
    c.kvduo_prefill_host_cache.destroy()
    c.mem_pool_host.destroy()


def test_attaching_backed_prefix_does_not_read_gpu_clock(monkeypatch):
    c = stack(monkeypatch)
    req = request(c)
    prefill(c, req)
    complete_backup(c, req)
    with monkeypatch.context() as checks:
        checks.setattr(
            torch.Tensor,
            "__int__",
            MagicMock(side_effect=AssertionError("blocking GPU clock read")),
        )
        assert c._attach_cached_host_prefix(req, 384) == 384
    c.kvduo_prefill_host_cache.destroy()
    c.mem_pool_host.destroy()


def test_mixed_batch_does_not_mark_decode_swa_as_fresh_prefill(monkeypatch):
    c = stack(monkeypatch)
    req = request(c)
    decode = request(c, "decode")
    state = MagicMock()
    tree = SimpleNamespace(
        kvduo_active_swa=True,
        kvduo_coordinator=SimpleNamespace(kvduo_prefill_host_cache=state),
        maintain_kvduo_swa=MagicMock(),
    )
    batch = ScheduleBatch(
        reqs=[req, decode],
        tree_cache=tree,
        prefix_lens=[0, 1535],
        extend_lens=[1536, 1],
        decoding_reqs=[decode],
    )
    batch.release_kvduo_prefill_swa()
    assert req.kvduo_swa_live_from == 0
    assert not hasattr(decode, "kvduo_swa_live_from")
    state.backup_batch.assert_called_once_with([(req, 1536)])


def host_match(c, req):
    return c.augment_kvduo_prefix_match(
        req,
        MatchResult(
            device_indices=torch.empty(0, dtype=torch.int64, device=c.device),
            last_device_node=0,
            last_host_node=0,
            best_match_node=0,
        ),
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_host_checkpoint_restores_preceding_window_and_all_domains(monkeypatch, device):
    c = stack(monkeypatch, device)
    source = request(c)
    full = prefill(c, source)
    complete_backup(c, source)
    assert c._kvduo_prefill_host_reservations[c._kvduo_prefill_key(source)][1] == 0
    state = c.kvduo_prefill_host_cache
    assert state.ready(source, 0, 1536)
    assert set(state.history.pools) == {"c4_indexer", "c128"}
    assert set(state.window.pools) == {"swa", "c4_state", "c4_indexer_state"}
    # End the request and discard HBM, including SWA. Checkpoints remain Host-owned.
    c.end_kvduo_prefill(source)
    c.token_to_kv_pool_allocator.free(full)
    assert c.token_to_kv_pool_allocator.swa_available_size() == 2048
    assert not torch.any(
        c.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(full)
    )
    for pool in [
        c.mem_pool_host,
        *state.history.pools.values(),
        *state.window.pools.values(),
    ]:
        for buffer in getattr(
            pool, "device_buffers", getattr(pool, "device_page_views", ())
        ):
            buffer.zero_()
    target = request(c, "target")
    result = host_match(c, target)
    assert result.host_hit_length == 1536
    assert result.swa_host_hit_length == 256
    target.prefix_indices = result.device_indices
    target.kvduo_residency_plan = result.kvduo_residency_plan
    before = c.token_to_kv_pool_allocator.swa_available_size()
    restored = c.init_kvduo_load_back(target, result.host_hit_length)
    assert c.token_to_kv_pool_allocator.swa_available_size() == before - 256
    swa = c.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(restored)
    assert torch.all(swa[:-256] == 0)
    # The very first query at 1536 reads preceding keys 1409..1535.
    assert torch.all(swa[1536 - 127 :] > 0)
    for group, keys, locations in (
        (state.history, target.kvduo_prefill_state_plan.history, restored),
        (state.window, target.kvduo_prefill_state_plan.window, swa[-256:]),
    ):
        host = state._locations(group, keys, c.device)
        for pool in group.pools.values():
            buffers = getattr(
                pool, "device_buffers", getattr(pool, "device_page_views", ())
            )
            for actual, expected in zip(buffers, pool.data_refs):
                raw = actual.view(torch.uint8).reshape(actual.shape[0], -1)
                torch.testing.assert_close(
                    raw[pool._to_page_indices(locations)],
                    expected[pool._to_page_indices(host)],
                )
    assert torch.all(
        c.full_host_version[
            c.mem_pool_device.translate_loc_from_full_to_compressed(restored)
        ]
        == 17
    )
    main_keys = target.kvduo_prefill_state_plan.history
    main_host = torch.tensor(
        [
            slot
            for key in main_keys
            for slot in c.host_prefix_cache.records[key].host_locations["main_kv"]
        ],
        device=c.device,
    )
    main_device = c.mem_pool_device.translate_loc_from_full_to_hisparse_device(restored)
    for actual, expected in zip(
        c.mem_pool_host.device_buffers, c.mem_pool_host.data_refs
    ):
        torch.testing.assert_close(
            actual[c.mem_pool_host._to_page_indices(main_device)],
            expected[c.mem_pool_host._to_page_indices(main_host)],
        )
    c.end_kvduo_prefill(target)
    c.token_to_kv_pool_allocator.rollback_restore_allocation(restored, swa_tail_len=256)
    assert c.token_to_kv_pool_allocator.swa_available_size() == 2048
    state.clear()
    assert c.mem_pool_host.available_size() == c.mem_pool_host.size
    state.destroy()
    c.mem_pool_host.destroy()


@pytest.mark.parametrize(
    "missing", ["swa", "c4_state", "c4_indexer_state", "c4_indexer", "c128", "main_kv"]
)
def test_missing_checkpoint_domain_shrinks_prefix(monkeypatch, missing):
    c = stack(monkeypatch)
    source = request(c)
    prefill(c, source)
    complete_backup(c, source)
    state = c.kvduo_prefill_host_cache
    key = c._host_prefix_identity(source, 5)
    cache = (
        c.host_prefix_cache
        if missing == "main_kv"
        else (
            state.window.cache if missing in state.window.pools else state.history.cache
        )
    )
    cache.records[key].host_versions.pop(missing)
    assert not state.ready(source, 0, 1536)
    target = request(c, "target")
    result = host_match(c, target)
    assert result.host_hit_length == 1280
    assert state.ready(target, 0, 1280)
    state.destroy()
    c.mem_pool_host.destroy()


def test_overlap_copy_backs_up_before_swa_cleanup(monkeypatch):
    c = stack(monkeypatch)
    req = request(c)
    prefill(c, req)
    state = c.kvduo_prefill_host_cache
    cleanup = MagicMock(
        side_effect=lambda frontiers: state.ready(req, 0, frontiers[0][1])
    )
    tree = SimpleNamespace(
        kvduo_active_swa=True, kvduo_coordinator=c, maintain_kvduo_swa=cleanup
    )
    batch = ScheduleBatch(
        reqs=[req], tree_cache=tree, prefix_lens=[0], extend_lens=[1536]
    )
    batch.copy().release_kvduo_prefill_swa()
    cleanup.assert_called_once_with([(req, 1536)])
    assert state.ready(req, 0, 1536)
    state.destroy()
    c.mem_pool_host.destroy()


def test_pending_copy_and_recycled_request_generation_cannot_be_reused(monkeypatch):
    c = stack(monkeypatch)
    req = request(c)
    prefill(c, req)
    event = SimpleNamespace(query=lambda: False)
    monkeypatch.setattr(c.kvduo_prefill_host_cache, "_event", lambda: event)
    state = c.kvduo_prefill_host_cache
    state.backup(req, 1536)
    assert host_match(c, request(c, "early")).host_hit_length == 0
    c.end_kvduo_prefill(req)
    assert not state.sources
    assert all(r.dma_pins == 1 for r in c.host_prefix_cache.records.values())
    event.query = lambda: True
    state.poll()
    target = request(c, "reuse")
    result = host_match(c, target)
    assert result.host_hit_length == 1536
    target.prefix_indices = result.device_indices
    target.kvduo_residency_plan = result.kvduo_residency_plan
    target.kvduo_prefill_generation += 1
    with pytest.raises(RuntimeError, match="changed before restore"):
        c.init_kvduo_load_back(target, result.host_hit_length)
    c._reclaim_for_physical_allocation.assert_not_called()
    with pytest.raises(RuntimeError, match="missing or stale"):
        state.load(target, torch.arange(256, 1792))
    c.end_kvduo_prefill(target)
    assert not state.matches
    assert all(not r.pinned for r in state.window.cache.records.values())
    state.destroy()
    c.mem_pool_host.destroy()


def test_host_window_eviction_does_not_free_decode_main_backing(monkeypatch):
    c = stack(monkeypatch)
    source = request(c)
    prefill(c, source)
    complete_backup(c, source)
    state = c.kvduo_prefill_host_cache
    c.end_kvduo_prefill(source)
    main_slots = c.mem_pool_host.available_size()
    # Keep a matched checkpoint pinned while evicting other window snapshots.
    target = request(c, "target")
    assert state.plan(target, 0, 1536) is not None
    state.window.make_room(state.window.anchor.size)
    assert state.ready(target, 0, 1536)
    c.end_kvduo_prefill(target)
    assert state.window.make_room(state.window.anchor.size)
    assert not state.ready(source, 0, 1536)
    assert c.mem_pool_host.available_size() == main_slots
    assert len(c.host_prefix_cache.records) == 6
    assert host_match(c, request(c, "miss")).host_hit_length == 0
    state.destroy()
    c.mem_pool_host.destroy()


def test_restore_failure_releases_owned_window_without_publishing_it(monkeypatch):
    c = stack(monkeypatch)
    source = request(c)
    prefill(c, source)
    complete_backup(c, source)
    target = request(c, "target")
    result = host_match(c, target)
    target.prefix_indices = result.device_indices
    target.kvduo_residency_plan = result.kvduo_residency_plan
    before = (
        c.token_to_kv_pool_allocator.full_available_size(),
        c.token_to_kv_pool_allocator.swa_available_size(),
        c.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size(),
    )
    pool = c.kvduo_prefill_host_cache.window.pools["c4_state"]
    monkeypatch.setattr(
        pool,
        "load_to_device_per_layer",
        MagicMock(side_effect=RuntimeError("DMA failed")),
    )
    with pytest.raises(RuntimeError, match="DMA failed"):
        c.init_kvduo_load_back(target, result.host_hit_length)
    after = (
        c.token_to_kv_pool_allocator.full_available_size(),
        c.token_to_kv_pool_allocator.swa_available_size(),
        c.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size(),
    )
    assert before == after
    assert target.kvduo_restored_prefix_len == 0
    state = c.kvduo_prefill_host_cache
    c.end_kvduo_prefill(target)
    c.end_kvduo_prefill(source)
    state.poll()
    assert all(
        not r.pinned
        for group in (state.history, state.window)
        for r in group.cache.records.values()
    )
    assert all(not r.pinned for r in c.host_prefix_cache.records.values())
    state.clear()
    state.destroy()
    c.mem_pool_host.destroy()


def _tp_checkpoint_worker(rank, init_file, gpu_divergence):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=40),
    )
    patches = pytest.MonkeyPatch()
    try:
        c = stack(patches)
        source = request(c)
        full = prefill(c, source)
        c.kvduo_prefill_host_cache.backup(source, 1024)
        complete_backup(c, source)
        if not gpu_divergence:
            # Rank-local longest hits are 1280 and 1536, but 1280 itself is
            # invalid on rank 1. Only 1024 is a complete common checkpoint.
            key = c._host_prefix_identity(source, 5 if rank == 0 else 4)
            c.kvduo_prefill_host_cache.window.cache.records[key].host_versions.clear()
        c.tp_world_size, c.tp_group = 2, dist.group.WORLD
        target = request(c, "target")
        gpu = full[:256] if rank == 0 and gpu_divergence else full[:0]
        result = c.augment_kvduo_prefix_match(
            target,
            MatchResult(
                device_indices=gpu,
                last_device_node=0,
                last_host_node=0,
                best_match_node=0,
            ),
        )
        expected = 1536 if gpu_divergence else 1024
        assert len(gpu) + result.host_hit_length == expected
        assert result.swa_host_hit_length == 256
        target.prefix_indices = gpu
        target.kvduo_residency_plan = result.kvduo_residency_plan
        restored = c.init_kvduo_load_back(target, result.host_hit_length)
        assert len(restored) == expected - len(gpu)
        assert torch.all(
            c.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(restored[-128:])
            > 0
        )
        c.end_kvduo_prefill(target)
        c.end_kvduo_prefill(source)
        c.kvduo_prefill_host_cache.destroy()
        c.mem_pool_host.destroy()
    finally:
        patches.undo()
        dist.destroy_process_group()


@pytest.mark.parametrize("gpu_divergence", [False, True])
def test_tp_restore_uses_complete_common_checkpoint(tmp_path, gpu_divergence):
    import torch.multiprocessing as mp

    mp.start_processes(
        _tp_checkpoint_worker,
        args=(str(tmp_path / "checkpoint-gloo"), gpu_divergence),
        nprocs=2,
        join=True,
        start_method="spawn",
    )


@pytest.mark.parametrize("is_insert", [True, False])
def test_finished_prefill_keeps_host_checkpoint_and_frees_hbm_swa(
    monkeypatch, is_insert
):
    c = stack(monkeypatch)
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=c.req_to_token_pool,
            token_to_kv_pool_allocator=c.token_to_kv_pool_allocator,
            page_size=256,
            sliding_window_size=128,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
        )
    )
    cache.kvduo_coordinator = c
    source = request(c)
    source.last_node = cache.root_node_handle()
    prefill(c, source)
    complete_backup(c, source)
    release_kv_cache(source, cache)
    assert source.req_pool_idx is None
    assert c.token_to_kv_pool_allocator.swa_available_size() == 2048
    assert cache.full_evictable_size() == 1536
    target = request(c, "target")
    target.init_next_round_input(cache)
    assert len(target.prefix_indices) == 0
    assert target.host_hit_length == 1536
    restored = c.init_kvduo_load_back(target, target.host_hit_length)
    c.req_to_token_pool.req_to_token[target.req_pool_idx, :1536] = restored
    target.prefix_indices = restored
    target.cache_protected_len = 1536
    target.last_node = cache.root_node_handle()
    release_kv_cache(target, cache, is_insert=is_insert)
    assert c.token_to_kv_pool_allocator.swa_available_size() == 2048
    assert (
        c.token_to_kv_pool_allocator.logical_attn_allocator.full_available_size()
        == 8192 - 1536
    )
    state = c.kvduo_prefill_host_cache
    state.poll()
    assert len(state.window.cache.records) == 6
    assert all(not r.pinned for r in c.host_prefix_cache.records.values())
    assert all(
        not r.pinned
        for group in (state.history, state.window)
        for r in group.cache.records.values()
    )
    cache.reset()
    assert not c.host_prefix_cache.records
    assert c.mem_pool_host.available_size() == c.mem_pool_host.size
    state.destroy()
    c.mem_pool_host.destroy()


def test_finished_decode_saves_latest_window_without_duplicate_main_allocation(
    monkeypatch,
):
    c = stack(monkeypatch)
    source = request(c)
    full = prefill(c, source)
    state = c.kvduo_prefill_host_cache
    state.backup(source, 1024)
    assert state.ready(source, 0, 1024)
    assert not state.ready(source, 0, 1536)
    c._attach_cached_host_prefix(source, 256)
    c._release_kvduo_prefill_host(c._kvduo_prefill_key(source))
    host = c.mem_pool_host.alloc_paged_token_slots(
        c.req_to_host_pool,
        c.req_to_host_pool_allocated_len,
        source.req_pool_idx,
        0,
        384,
    )
    c.mem_pool_host.backup_from_device_all_layer(
        c.mem_pool_device,
        host[256:],
        c.mem_pool_device.translate_loc_from_full_to_hisparse_device(full[1024:]),
        "kernel",
    )
    compressed = c.mem_pool_device.translate_loc_from_full_to_compressed(full)
    c.full_host_version[compressed] = c.full_data_version[compressed]
    c._kvduo_host_valid_len = [0] * 4
    c._kvduo_host_valid_len[source.req_pool_idx] = 384
    c.req_to_full_lookup = torch.full((4, 2048), -1, dtype=torch.int64)
    c.req_to_full_lookup[source.req_pool_idx, :384] = compressed
    # Model-generated suffixes can also become complete Host checkpoints.
    source.origin_input_ids = array("q", range(1024))
    source.output_ids = list(range(1024, 1537))
    source._refresh_fill_ids()
    available = c.mem_pool_host.available_size()
    retained = state.backup_finished(source)
    assert c.mem_pool_host.available_size() == available
    assert len(retained) == 384
    assert state.ready(source, 0, 1536)
    c.end_kvduo_prefill(source)
    c._release_host_prefix_refs(source)
    c.token_to_kv_pool_allocator.free(full)
    assert c.token_to_kv_pool_allocator.swa_available_size() == 2048
    target = request(c, "target")
    assert host_match(c, target).host_hit_length == 1536
    c.end_kvduo_prefill(target)
    state.clear()
    state.destroy()
    c.mem_pool_host.destroy()


def test_incremental_backup_rescans_after_eviction_and_clears_generation(monkeypatch):
    c = stack(monkeypatch)
    req = request(c)
    prefill(c, req)
    state = c.kvduo_prefill_host_cache
    translate = MagicMock(
        wraps=c.token_to_kv_pool_allocator.translate_loc_from_full_to_swa
    )
    monkeypatch.setattr(
        c.token_to_kv_pool_allocator, "translate_loc_from_full_to_swa", translate
    )
    state.backup(req, 1024)
    state.backup(req, 1536)
    assert [call.args[0].numel() for call in translate.call_args_list] == [1024, 512]
    state.backup(req, 1536)
    assert translate.call_count == 2
    key = c._host_prefix_identity(req, 0)
    record = state.window.cache.remove(key)
    state.window.anchor.free(torch.tensor(record.host_locations["swa"]))
    state.backup(req, 1536)
    assert translate.call_args.args[0].numel() == 1536
    assert state.ready(req, 0, 256)
    c.end_kvduo_prefill(req)
    assert id(req) not in state.progress
    state.destroy()
    c.mem_pool_host.destroy()


def test_upload_buffer_is_reused_only_after_copy_completion(monkeypatch):
    c = stack(monkeypatch)
    req = request(c)
    prefill(c, req)
    state = c.kvduo_prefill_host_cache
    event = SimpleNamespace(query=lambda: False)
    monkeypatch.setattr(state, "_event", lambda: event)
    state.backup(req, 1024)
    first = state._upload_pending[0][1]
    snapshot = first.clone()
    state.backup(req, 1536)
    second = state._upload_pending[1][1]
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first, snapshot)
    event.query = lambda: True
    state.poll()
    assert not state._upload_pending
    buffer, sizes = state._prepare_upload(torch.arange(10), [], [], False)
    assert sizes == [10, 0, 0]
    assert buffer.data_ptr() == first.data_ptr()
    assert buffer[:10].tolist() == list(range(10))
    state.destroy()
    c.mem_pool_host.destroy()


def batch_sources(c):
    first = request(c)
    full = prefill(c, first)
    second = request(c, "shared")
    c.req_to_token_pool.req_to_token[second.req_pool_idx, :1536] = full
    assert c._reserve_kvduo_prefill_host(second, c._kvduo_prefill_key(second))
    return first, second


@pytest.mark.parametrize("covered", [True, False])
def test_batch_budget_preserves_fallback_admission(monkeypatch, covered):
    c = stack(monkeypatch)
    first, second = batch_sources(c)
    if not covered:
        c._release_kvduo_prefill_host(c._kvduo_prefill_key(second))
    common = MagicMock(wraps=c._kvduo_host_capacity_ready)
    monkeypatch.setattr(c, "_kvduo_host_capacity_ready", common)
    state = c.kvduo_prefill_host_cache
    state.backup_batch([(first, 1536), (second, 1536)])
    assert common.call_count == (1 if covered else 2)
    assert state.ready(first, 0, 1536)
    assert state.ready(second, 0, 1536)
    assert c.mem_pool_host.available_size() == c.mem_pool_host.size - 384
    state.destroy()
    c.mem_pool_host.destroy()


def _batch_budget_tp_worker(rank, path):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        for covered in (True, False):
            with pytest.MonkeyPatch.context() as patch:
                c = stack(patch)
                c.tp_world_size = 2
                c.tp_group = dist.group.WORLD
                first, second = batch_sources(c)
                if not covered and rank == 1:
                    c._release_kvduo_prefill_host(c._kvduo_prefill_key(second))
                common = MagicMock(wraps=c._kvduo_host_capacity_ready)
                patch.setattr(c, "_kvduo_host_capacity_ready", common)
                state = c.kvduo_prefill_host_cache
                state.backup_batch([(first, 1536), (second, 1536)])
                assert common.call_count == (1 if covered else 2)
                assert state.ready(first, 0, 1536) and state.ready(second, 0, 1536)
                state.destroy()
                c.mem_pool_host.destroy()
    finally:
        dist.destroy_process_group()


def test_batch_budget_fast_path_and_fallback_agree_across_ranks(tmp_path):
    import torch.multiprocessing as mp

    mp.start_processes(
        _batch_budget_tp_worker,
        args=(str(tmp_path / "batch-budget"),),
        nprocs=2,
        join=True,
        start_method="spawn",
    )


def test_eviction_during_backup_invalidates_incremental_progress(monkeypatch):
    c = stack(monkeypatch)
    req = request(c)
    prefill(c, req)
    state = c.kvduo_prefill_host_cache
    state.backup(req, 1024)
    key = c._host_prefix_identity(req, 0)
    pool = state.window.pools["swa"]
    copy = pool.backup_from_device_all_layer
    evicted = False

    def copy_and_evict(*args):
        nonlocal evicted
        copy(*args)
        if not evicted:
            record = state.window.cache.remove(key)
            state.window.anchor.free(torch.tensor(record.host_locations["swa"]))
            evicted = True

    monkeypatch.setattr(pool, "backup_from_device_all_layer", copy_and_evict)
    state.backup(req, 1536)
    assert not state.ready(req, 0, 256)
    state.backup(req, 1536)
    assert state.ready(req, 0, 256)
    state.destroy()
    c.mem_pool_host.destroy()
