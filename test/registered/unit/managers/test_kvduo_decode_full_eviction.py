from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache.base_prefix_cache import InsertResult
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_generic_dsa import make_stack
from test_kvduo_pressure_admission import make_full_coordinator

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_decode_coordinator(monkeypatch, *, ratio=1, mixed=False, prefill=4):
    import sglang.srt.managers.hisparse_coordinator as module

    c, physical, req = make_full_coordinator(pages=6, layers=3, ratio=ratio)
    count = 6 * c.page_size
    req.extend_range = Range(0, prefill * ratio)
    req.kv.kv_allocated_len = prefill * ratio
    c.alloc_device_buffer(req)
    req.kv.kv_allocated_len = count * ratio
    req.kv_committed_len = prefill * ratio
    physical.free_pages = 64
    physical.next_page = count + c.page_size
    c.reclaim_kvduo_full_pages = MagicMock(wraps=c.reclaim_kvduo_full_pages)
    c._has_pending_backup = False
    c._pending_kvduo_host_valid = []
    c.decode_producer_stream = MagicMock()
    c.decode_backup_stream = MagicMock()
    c._backup_done_event = MagicMock()
    current_stream = MagicMock()
    monkeypatch.setattr(module, "device_module", MagicMock())
    module.device_module.current_stream.return_value = current_stream
    module.device_module.stream.side_effect = lambda stream: nullcontext()
    c.wait_for_pending_backup = MethodType(
        HiSparseCoordinator.wait_for_pending_backup, c
    )
    c.token_to_kv_pool_allocator.get_last_loc_compressed = lambda locs: locs // ratio
    c.req_to_host_pool[0, :prefill] = torch.arange(prefill)
    c.req_to_host_pool_allocated_len[0] = prefill
    payload = torch.arange(3 * (count + c.page_size)).view(3, -1)
    host_payload = torch.full((3, count), -1, dtype=torch.int64)

    def allocate_host(bank, lengths, req_idx, start, size):
        locs = torch.arange(start, start + size)
        bank[req_idx, start : start + size] = locs
        lengths[req_idx] = max(int(lengths[req_idx]), start + size)
        return locs

    def backup(pool, host_locs, device_locs, *, io_backend):
        host_payload[:, host_locs] = payload[:, device_locs]

    c.mem_pool_host.alloc_paged_token_slots = MagicMock(side_effect=allocate_host)
    c.mem_pool_host.backup_from_device_all_layer = MagicMock(side_effect=backup)
    c._mixed_slots[0] = mixed
    if mixed:
        c._ensure_kvduo_hot_capacity_targets_batch(
            {(layer, 0): c.hot_page_size for layer in range(3)}
        )
    return c, physical, req, payload, host_payload, current_stream


def append(c, req, compressed_len):
    length = compressed_len * c.compress_ratio
    req.kv_committed_len = length
    c.map_last_loc_to_buffer(
        torch.tensor([length]),
        torch.tensor([length - 1]),
        torch.tensor([0]),
        torch.tensor([length]),
        torch.tensor([0]),
    )


@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("mixed", [False, True])
def test_decode_full_pages_remain_resident_until_request_finish(
    monkeypatch, ratio, mixed
):
    c, physical, req, payload, host_payload, current = make_decode_coordinator(
        monkeypatch, ratio=ratio, mixed=mixed
    )
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    original = mapping.clone()
    free_before = physical.free_pages
    for length in range(5, 10):
        append(c, req, length)
        c.ensure_kvduo_decode_capacity([req])
        c.prepare_kvduo_graph_replay(torch.tensor([0]))
    c.wait_for_pending_backup()
    assert torch.equal(mapping, original)
    assert torch.equal(host_payload[:, 4:8], payload[:, original[4:8]])
    assert physical.free_pages == free_before
    assert c._mixed_slots[0] is mixed
    expected = [4, 4, 4] if mixed else [0, 0, 0]
    assert c.kvduo_req_hot_capacity[:, 0].tolist() == expected
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_not_called()
    if mixed:
        # This fixture sets the mode without actually demoting FULL or creating
        # a Tail cursor. Replay may poll sealed-tail jobs but must free nothing.
        assert c.reclaim_kvduo_full_pages.call_count > 0
        assert all(
            call.kwargs.get("tail_cleanup") is True
            for call in c.reclaim_kvduo_full_pages.call_args_list
        )
    else:
        c.reclaim_kvduo_full_pages.assert_not_called()
    c.request_finished(req)
    assert mapping[:4].tolist() == original[:4].tolist()
    assert torch.all(mapping[4:] == 0)
    released = torch.cat(
        [
            call.args[0]
            for call in c.token_to_kv_pool_allocator.free_hisparse_indices.call_args_list
        ]
    )
    assert set(original[4:].tolist()).issubset(set(released.tolist()))
    assert released.unique().numel() == released.numel()
    assert physical.free_pages == free_before + 4 + (3 if mixed else 0)
    current.wait_stream.assert_called_with(c.decode_producer_stream)
    assert c.kvduo_req_hot_capacity[:, 0].tolist() == [0, 0, 0]
    assert req.kvduo_radix_insert_len == 4 * ratio
    assert not c._active_kvduo_reqs


@pytest.mark.parametrize("ratio", [1, 4])
def test_decode_can_fill_prefill_partial_page_before_finish(monkeypatch, ratio):
    c, _, req, _, host_payload, _ = make_decode_coordinator(
        monkeypatch, ratio=ratio, prefill=3
    )
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    original = mapping.clone()
    for length in range(4, 8):
        append(c, req, length)
        c.prepare_kvduo_graph_replay(torch.tensor([0]))
    c.wait_for_pending_backup()
    assert torch.equal(mapping, original)
    assert torch.all(host_payload[:, 2:6] >= 0)
    assert not c._mixed_slots[0]
    assert not torch.any(c.kvduo_req_hot_capacity)
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_not_called()
    c.request_finished(req)
    assert mapping[:2].tolist() == original[:2].tolist()
    assert torch.all(mapping[2:] == 0)
    assert req.kvduo_radix_insert_len == 2 * ratio


@pytest.mark.parametrize("prefill", [1, 63, 64, 65, 128])
def test_real_allocator_finish_and_radix_release_decode_pages_exactly_once(
    monkeypatch, prefill
):
    c, pool, allocator = make_stack(monkeypatch)
    c._active_kvduo_reqs.clear()
    c._mixed_slots = [False, False]
    physical = allocator.hisparse_attn_allocator
    physical.debug_mode = True
    logical = allocator.logical_attn_allocator
    logical.debug_mode = True
    allocated_len = 129
    indices = logical.alloc(192)[:allocated_len]
    locations = physical.alloc(192)[:allocated_len]
    allocator.full_to_hisparse_device_index_mapping[indices] = locations
    c.req_to_token_pool.req_to_token[0, :allocated_len] = indices.to(torch.int32)
    req = SimpleNamespace(
        req_pool_idx=0,
        cache_protected_len=0,
        hisparse_staging=False,
        kv=SimpleNamespace(kv_allocated_len=prefill),
        extend_range=Range(0, prefill),
        origin_input_ids=list(range(prefill)),
        output_ids=list(range(prefill, allocated_len)),
        extra_key=None,
        last_node=None,
    )
    c.alloc_device_buffer(req)
    req.kv.kv_allocated_len = allocated_len
    c.mem_pool_host.allocated_host_indices = lambda *args: torch.empty(
        0, dtype=torch.int64
    )
    c._retain_request_host_prefix = MagicMock(return_value=set())
    c._release_host_prefix_refs = MagicMock()
    c.request_finished(req)
    retained = prefill // pool.page_size * pool.page_size
    assert physical.available_size() == pool.size - retained
    cache = RadixCache.__new__(RadixCache)
    cache.disable = cache.disable_finished_insert = cache.is_eagle = False
    cache.page_size = pool.page_size
    cache.req_to_token_pool = c.req_to_token_pool
    cache.token_to_kv_pool_allocator = allocator
    cache.insert = MagicMock(return_value=InsertResult(prefix_len=0))
    cache.cache_finished_req(req, kv_len_to_handle=allocated_len)
    assert len(cache.insert.call_args.args[0].value) == retained
    assert physical.available_size() == pool.size - retained
    assert logical.available_size() == logical.size - retained
    assert physical.free_pages.unique().numel() == physical.free_pages.numel()
    assert not torch.any(
        allocator.full_to_hisparse_device_index_mapping[indices[retained:]]
    )


@pytest.mark.parametrize("prefill", [63, 64, 65, 128])
def test_complete_generic_finish_releases_prefill_and_decode_hbm(monkeypatch, prefill):
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.common import release_kv_cache
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    c, pool, allocator = make_stack(monkeypatch)
    c._active_kvduo_reqs.clear()
    c._mixed_slots = [False, False]
    physical = allocator.hisparse_attn_allocator
    logical = allocator.logical_attn_allocator
    physical.debug_mode = logical.debug_mode = True
    indices = logical.alloc(192)[:129]
    locations = physical.alloc(192)[:129]
    allocator.full_to_hisparse_device_index_mapping[indices] = locations
    c.req_to_token_pool.req_to_token[0, :129] = indices.to(torch.int32)
    c.req_to_token_pool.free = MagicMock()
    req = SimpleNamespace(
        req_pool_idx=0,
        cache_protected_len=0,
        hisparse_staging=False,
        kv=SimpleNamespace(kv_allocated_len=prefill),
        extend_range=Range(0, prefill),
        origin_input_ids=list(range(prefill)),
        output_ids=list(range(prefill, 129)),
        extra_key=None,
        last_node=None,
        kv_committed_len=129,
        effective_kv_committed_len=lambda: 129,
    )
    c.alloc_device_buffer(req)
    req.kv.kv_allocated_len = 129
    c.mem_pool_host.allocated_host_indices = lambda *args: torch.empty(
        0, dtype=torch.int64
    )
    c._retain_request_host_prefix = MagicMock(return_value=set())
    c._release_host_prefix_refs = MagicMock()
    c.request_finished(req)
    cache = RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=c.req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=pool.page_size,
        )
    )
    cache.kvduo_coordinator = c
    release_kv_cache(req, cache)
    assert physical.available_size() == pool.size
    assert logical.available_size() == logical.size
    assert cache.evictable_size() == 0
    assert not torch.any(allocator.full_to_hisparse_device_index_mapping[indices])
    assert physical.free_pages.unique().numel() == physical.free_pages.numel()
