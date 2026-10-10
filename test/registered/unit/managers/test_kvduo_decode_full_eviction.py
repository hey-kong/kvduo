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
def test_sealed_decode_full_pages_are_backed_up_and_evicted_without_pressure(
    monkeypatch, ratio, mixed
):
    c, physical, req, payload, host_payload, current = make_decode_coordinator(
        monkeypatch, ratio=ratio, mixed=mixed
    )
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    original = mapping.clone()
    host_indices = c.req_to_host_pool.clone()
    free_before = physical.free_pages
    for length in (5, 6, 7):
        append(c, req, length)
    assert c._has_pending_backup
    assert torch.equal(mapping, original)
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert mapping[:4].tolist() == original[:4].tolist()
    assert mapping[4:6].tolist() == [0, 0]
    assert mapping[6:].tolist() == original[6:].tolist()
    assert torch.equal(host_payload[:, 4:6], payload[:, original[4:6]])
    assert c.req_to_host_pool[0, :4].tolist() == host_indices[0, :4].tolist()
    assert c.req_to_host_pool[0, 4:6].tolist() == [4, 5]
    assert c.req_to_full_lookup[0, :7].tolist() == list(range(7))
    assert c.full_host_version[4:6].tolist() == c.full_data_version[4:6].tolist()
    assert req.kvduo_mixed_residency
    assert c.kvduo_req_hot_capacity[:, 0].tolist() == [4, 4, 4]
    assert physical.free_pages == free_before + 1 - (0 if mixed else 3)
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_called_once()
    assert (
        c.token_to_kv_pool_allocator.free_hisparse_indices.call_args.args[0].tolist()
        == original[4:6].tolist()
    )
    c._backup_done_event.wait.assert_called_once_with(current)
    current.wait_stream.assert_called_with(c.decode_producer_stream)
    c.reclaim_kvduo_full_pages.assert_not_called()
    assert not c._kvduo_decode_full_pending
    lru = c.lru_slots.clone()
    for length in (8, 9):
        append(c, req, length)
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert mapping[6:8].tolist() == [0, 0]
    assert mapping[8].item() == original[8].item()
    assert torch.equal(c.lru_slots, lru)
    assert c.token_to_kv_pool_allocator.free_hisparse_indices.call_count == 2


@pytest.mark.parametrize("ratio", [1, 4])
def test_prefill_partial_page_keeps_its_identity_when_decode_seals_it(
    monkeypatch, ratio
):
    c, _, req, _, host_payload, _ = make_decode_coordinator(
        monkeypatch, ratio=ratio, prefill=3
    )
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    original = mapping.clone()
    assert req.kvduo_decode_full_start == 4
    for length in (4, 5):
        append(c, req, length)
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert torch.equal(mapping, original)
    assert torch.all(host_payload[:, 2:4] >= 0)
    assert not c._mixed_slots[0]
    assert not torch.any(c.kvduo_req_hot_capacity)
    for length in (6, 7):
        append(c, req, length)
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert mapping[:4].tolist() == original[:4].tolist()
    assert mapping[4:6].tolist() == [0, 0]


def ready_page(monkeypatch):
    c, physical, req, _, _, _ = make_decode_coordinator(monkeypatch)
    for length in (5, 6, 7):
        append(c, req, length)
    c.wait_for_pending_backup()
    return c, physical, req


@pytest.mark.parametrize("blocked", ["version", "protected", "staging", "consensus"])
def test_unsafe_decode_page_is_retained_and_retried(monkeypatch, blocked):
    c, physical, req = ready_page(monkeypatch)
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    original = mapping.clone()
    if blocked == "version":
        c.full_data_version[4] += 1
    elif blocked == "protected":
        c._kvduo_pressure_protected = torch.tensor([5])
    elif blocked == "staging":
        req.hisparse_staging = True
    else:
        c._kvduo_consensus = MagicMock(return_value=False)
    free_before = physical.free_pages
    assert c._evict_kvduo_decode_full_pages() == 0
    assert torch.equal(mapping, original)
    assert physical.free_pages == free_before
    assert len(c._kvduo_decode_full_pending) == 1
    c.full_host_version.copy_(c.full_data_version)
    c._kvduo_pressure_protected = None
    req.hisparse_staging = False
    c._kvduo_consensus = MagicMock(return_value=True)
    assert c._evict_kvduo_decode_full_pages() == 2
    assert mapping[4:6].tolist() == [0, 0]
    assert physical.free_pages == free_before + 1
    assert not c._kvduo_decode_full_pending


def test_backup_completion_does_not_publish_for_reused_request_slot(monkeypatch):
    c, _, req, _, _, _ = make_decode_coordinator(monkeypatch)
    for length in (5, 6, 7):
        append(c, req, length)
    c._reset_kvduo_prefill_request(0)
    c.full_host_version[4:6] = 99
    c._kvduo_host_valid_len[0] = 0
    c.wait_for_pending_backup()
    assert c.full_host_version[4:6].tolist() == [99, 99]
    assert c._kvduo_host_valid_len[0] == 0
    assert not c._kvduo_decode_full_pending
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_not_called()


def test_pending_eviction_does_not_free_pages_reassigned_to_another_generation(
    monkeypatch,
):
    c, _, _ = ready_page(monkeypatch)
    c._kvduo_request_generation[0] += 1
    assert c._evict_kvduo_decode_full_pages() == 0
    assert not c._kvduo_decode_full_pending
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_not_called()


def test_page_already_demoted_by_pressure_is_not_freed_twice(monkeypatch):
    c, _, _ = ready_page(monkeypatch)
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    mapping[4:6] = 0
    c._kvduo_consensus = MagicMock(return_value=False)
    assert c._evict_kvduo_decode_full_pages() == 0
    assert len(c._kvduo_decode_full_pending) == 1
    c._kvduo_consensus.return_value = True
    assert c._evict_kvduo_decode_full_pages() == 0
    assert not c._kvduo_decode_full_pending
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_not_called()


@pytest.mark.parametrize("ratio", [1, 4])
def test_decode_admission_backs_up_sealed_page_before_allocating_the_next_tail(
    monkeypatch, ratio
):
    c, physical, req, _, _, _ = make_decode_coordinator(monkeypatch, ratio=ratio)
    for length in (5, 6):
        append(c, req, length)
    req.kv_committed_len = 7 * ratio - 1
    physical.free_pages = 0
    c.ensure_kvduo_decode_capacity([req])
    assert physical.free_pages == 1
    assert c._kvduo_mandatory_hot_pages() == 3
    c._reclaim_for_physical_allocation.assert_called_once_with(
        2, reserve_hot_minimum=True
    )
    assert c._kvduo_pending_hot_minimum == {0}
    append(c, req, 7)
    c.mem_pool_host.backup_from_device_all_layer.assert_called_once()


def test_c4_tail_stays_resident_until_next_compressed_write(monkeypatch):
    c, _, req, _, _, _ = make_decode_coordinator(monkeypatch, ratio=4)
    for length in (5, 6):
        append(c, req, length)
    for length in (24, 25, 26):
        req.kv_committed_len = length
        c.ensure_kvduo_decode_capacity([req])
    c.mem_pool_host.backup_from_device_all_layer.assert_not_called()
    assert torch.all(c.mem_pool_device.full_to_hisparse_device_index_mapping[4:6] > 0)
    req.kv_committed_len = 27
    c.ensure_kvduo_decode_capacity([req])
    c.mem_pool_host.backup_from_device_all_layer.assert_called_once()
    assert c.mem_pool_device.full_to_hisparse_device_index_mapping[4:6].tolist() == [
        0,
        0,
    ]


def test_hisparse_without_kvduo_does_not_evict_decode_full_pages(monkeypatch):
    c, _, _ = ready_page(monkeypatch)
    c.enable_mixed_residency = False
    assert c._evict_kvduo_decode_full_pages() == 0
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_not_called()
    assert len(c._kvduo_decode_full_pending) == 1


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
