"""CPU regressions exercising the production HOT allocation/replay boundary."""

import threading
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.sparsity.core.kvduo_prefix_cache import KVDuoHostPrefixCache


def make_host_pool(size=1024, page_size=2):
    pool = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
    pool.size = size
    pool.page_size = pool.slot_page_size = page_size
    pool.lock = threading.RLock()
    pool.clear()
    return pool


def make_coordinator(*, top_k=2, layers=1, requests=1, free_pages=64):
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.enable_mixed_residency = True
    coordinator.full_reclaim_policy = "request"
    coordinator.decode_emergency_reclaim = False
    coordinator.decode_producer_stream = None
    coordinator.device = "cpu"
    coordinator.page_size = 2
    coordinator.hot_storage_layers = 2
    coordinator.hot_page_size = 4
    coordinator.top_k = top_k
    coordinator.min_hot_pages = (top_k + 3) // 4
    coordinator.device_buffer_size = 32
    coordinator.padded_buffer_size = (
        coordinator.device_buffer_size + coordinator.page_size
    )
    coordinator.mem_pool_device = SimpleNamespace(
        layer_num=layers,
        flat_layer_slot_stride=10000,
        translate_loc_from_full_to_compressed=lambda indices: indices,
    )
    # These focused HOT fixtures have no FULL backing; pressure/admission
    # regressions bind the real FULL reclaimer to a complete backing fixture.
    coordinator.reclaim_kvduo_full_pages = MagicMock(return_value=0)
    coordinator._active_kvduo_reqs = {req_idx: object() for req_idx in range(requests)}
    coordinator._mixed_slots = [True] * requests
    coordinator._kvduo_pending_hot_minimum = set(range(requests))
    coordinator.kvduo_resolver_stats = torch.zeros(
        (layers, requests, 2), dtype=torch.int32
    )
    coordinator._kvduo_hot_pressure_protected = set()
    coordinator._kvduo_pressure_protected = None
    coordinator.is_dsv4_hisparse = False
    coordinator.compress_ratio = 1
    coordinator.mem_pool_host = make_host_pool()
    coordinator.host_prefix_cache = KVDuoHostPrefixCache(coordinator.mem_pool_host.size)
    coordinator.req_to_host_pool_allocated_len = torch.zeros(
        requests, dtype=torch.int64
    )

    coordinator.kvduo_max_relative_miss_increase = 0.10
    coordinator._init_kvduo_capacity_state(requests)
    coordinator.kvduo_observe = torch.ones(1, dtype=torch.int32)
    # Match production: LRU has HOT capacity, other banks have page padding.
    metadata_shape = (layers, requests, coordinator.padded_buffer_size)
    coordinator.kvduo_hot_growth_events = torch.zeros(
        (layers, requests), dtype=torch.int32
    )
    coordinator._kvduo_hot_growth_events_host = torch.empty(
        (layers, requests), dtype=torch.int32
    )
    coordinator.kvduo_req_hot_capacity = torch.zeros(
        (layers, requests), dtype=torch.int64
    )
    coordinator.kvduo_req_hot_capacity_gpu = torch.zeros(
        (layers, requests), dtype=torch.int32
    )
    coordinator.req_device_buffer_token_locs = torch.full(
        metadata_shape, -1, dtype=torch.int32
    )
    coordinator.req_device_buffer_tokens = torch.full(
        metadata_shape, -1, dtype=torch.int32
    )
    coordinator.lru_slots = torch.arange(32, dtype=torch.int16).repeat(
        layers, requests, 1
    )
    coordinator._kvduo_req_layer_pages = {}
    coordinator._kvduo_hot_page_owners = {}
    coordinator._reclaim_for_physical_allocation = MagicMock(
        return_value=SimpleNamespace(action=object())
    )
    coordinator._require_allocation_ready = MagicMock()
    physical = SimpleNamespace(free_pages=free_pages, next_page=2, released_starts=[])
    physical.size = 4096
    physical.device = "cpu"
    physical.page_size = coordinator.page_size

    def allocate(slots):
        pages = slots // coordinator.page_size
        if pages > physical.free_pages:
            return None
        physical.free_pages -= pages
        reused = physical.released_starts[:pages]
        del physical.released_starts[: len(reused)]
        starts = reused + list(
            range(
                physical.next_page,
                physical.next_page + (pages - len(reused)) * coordinator.page_size,
                coordinator.page_size,
            )
        )
        physical.next_page += (pages - len(reused)) * coordinator.page_size
        return torch.tensor(
            [
                slot
                for start in starts
                for slot in range(start, start + coordinator.page_size)
            ],
            dtype=torch.int64,
        )

    physical.alloc = MagicMock(side_effect=allocate)
    physical.available_size = MagicMock(
        side_effect=lambda: physical.free_pages * coordinator.page_size
    )

    def free(indices):
        physical.free_pages += indices.numel() // coordinator.page_size
        physical.released_starts.extend(
            indices.view(-1, coordinator.page_size)[:, 0].tolist()
        )

    coordinator.token_to_kv_pool_allocator = SimpleNamespace(
        hisparse_attn_allocator=physical,
        free_hisparse_indices=MagicMock(side_effect=free),
    )

    def release_page_ids(page_ids):
        indices = (
            page_ids[:, None] * coordinator.page_size
            + torch.arange(coordinator.page_size)
        ).flatten()
        coordinator.token_to_kv_pool_allocator.free_hisparse_indices(indices)

    physical._release_page_ids = MagicMock(side_effect=release_page_ids)
    return coordinator, physical


@pytest.mark.parametrize("top_k, expected_capacity", [(2, 4), (4, 4), (5, 8), (9, 12)])
def test_mixed_minimum_is_page_aligned_before_next_replay(top_k, expected_capacity):
    coordinator, physical = make_coordinator(top_k=top_k, layers=3, requests=2)

    coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1]))

    assert torch.all(coordinator.kvduo_req_hot_capacity == expected_capacity)
    assert torch.all(coordinator.kvduo_req_hot_capacity_gpu == expected_capacity)
    assert len(coordinator._kvduo_hot_page_owners) == 6 * expected_capacity // 4
    physical.alloc.assert_called_once_with(6 * expected_capacity // 2)
    assert coordinator._kvduo_pending_hot_minimum == set()


def test_full_resident_requests_keep_zero_hot_capacity():
    coordinator, physical = make_coordinator()
    coordinator._mixed_slots[0] = False
    coordinator._kvduo_pending_hot_minimum.clear()

    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))

    physical.alloc.assert_not_called()
    assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 0


@pytest.mark.parametrize("free_pages", [1, 2])
def test_mandatory_allocation_credits_its_minimum_and_protects_other_owners(free_pages):
    coordinator, physical = make_coordinator(requests=2, free_pages=free_pages)
    coordinator.item_size_bytes = 1
    coordinator._reclaim_for_physical_allocation = MethodType(
        HiSparseCoordinator._reclaim_for_physical_allocation, coordinator
    )
    coordinator._require_allocation_ready = (
        HiSparseCoordinator._require_allocation_ready
    )
    coordinator.reclaim_kvduo_hot_pages = MagicMock(return_value=0)

    if free_pages == 1:
        with pytest.raises(RuntimeError, match="physical allocation cannot proceed"):
            coordinator._ensure_kvduo_hot_capacity_targets_batch({(0, 0): 4})
        physical.alloc.assert_not_called()
        assert not coordinator._kvduo_req_layer_pages
    else:
        coordinator._ensure_kvduo_hot_capacity_targets_batch({(0, 0): 4})
        assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 4
        assert physical.free_pages == 1
        assert coordinator._kvduo_mandatory_hot_pages() == 1
        coordinator.reclaim_kvduo_full_pages.assert_not_called()


def test_capacity_event_grows_on_the_very_next_replay_without_miss_thresholds():
    coordinator, physical = make_coordinator()
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))
    coordinator._reclaim_for_physical_allocation.reset_mock()
    # Only one miss among many accesses still suffices if it needed an occupied
    # slot. Diagnostic counters cannot suppress the separate capacity event.
    coordinator.kvduo_resolver_stats[0, 0] = torch.tensor([1, 10000])
    coordinator.kvduo_hot_growth_events[0, 0] = 1
    events_address = coordinator.kvduo_hot_growth_events.data_ptr()
    capacity_address = coordinator.kvduo_req_hot_capacity_gpu.data_ptr()

    coordinator.prepare_kvduo_graph_replay(
        torch.tensor([99]), req_pool_indices_cpu=torch.tensor([0])
    )

    assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 8
    assert int(coordinator.kvduo_req_hot_capacity_gpu[0, 0]) == 8
    assert coordinator.kvduo_hot_growth_events.data_ptr() == events_address
    assert coordinator.kvduo_req_hot_capacity_gpu.data_ptr() == capacity_address
    assert not torch.any(coordinator.kvduo_hot_growth_events)
    coordinator._reclaim_for_physical_allocation.assert_not_called()
    physical.alloc.reset_mock()
    for _ in range(9):
        coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))
    physical.alloc.assert_not_called()


def test_host_misses_without_a_capacity_event_do_not_grow():
    coordinator, physical = make_coordinator()
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))
    physical.alloc.reset_mock()
    coordinator.kvduo_resolver_stats[0, 0] = torch.tensor([10000, 10000])

    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))

    physical.alloc.assert_not_called()
    assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 4


def test_each_event_adds_one_page_and_stops_at_eight_pages():
    coordinator, physical = make_coordinator()
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))
    capacities = []
    for _ in range(9):
        coordinator.kvduo_hot_growth_events[0, 0] = 1
        coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))
        capacities.append(int(coordinator.kvduo_req_hot_capacity[0, 0]))

    assert capacities == [8, 12, 16, 20, 24, 28, 32, 32, 32]
    assert physical.alloc.call_count == 8
    assert len(coordinator._kvduo_req_layer_pages[(0, 0)]) == 8


@pytest.mark.parametrize("free_pages", [0, 1, 2])
def test_growth_spends_available_pages_on_individual_request_layer_owners(free_pages):
    coordinator, physical = make_coordinator(layers=2, requests=2)
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    physical.alloc.reset_mock()
    coordinator._reclaim_for_physical_allocation.reset_mock()
    physical.free_pages = free_pages
    tokens = coordinator.req_device_buffer_tokens
    tokens[:, :, :4] = torch.tensor([10, 11, 12, 13])
    old_locs = coordinator.req_device_buffer_token_locs.clone()
    coordinator.kvduo_hot_growth_events.fill_(1)

    coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1]))

    expected = torch.full((2, 2), 4, dtype=torch.int64)
    expected.view(-1)[:free_pages] = 8
    assert torch.equal(coordinator.kvduo_req_hot_capacity, expected)
    assert torch.equal(coordinator.kvduo_req_hot_capacity_gpu, expected.int())
    assert physical.free_pages == 0
    coordinator._reclaim_for_physical_allocation.assert_not_called()
    assert torch.equal(tokens[:, :, :4], torch.tensor([10, 11, 12, 13]).expand(2, 2, 4))
    assert torch.equal(
        coordinator.req_device_buffer_token_locs[:, :, :4], old_locs[:, :, :4]
    )
    for layer_id, req_idx in torch.nonzero(expected == 8).tolist():
        assert coordinator.lru_slots[layer_id, req_idx, :8].tolist() == [
            4,
            5,
            6,
            7,
            0,
            1,
            2,
            3,
        ]
    # The failed attempt is consumed; free pages appearing later are not enough
    # to cause growth without another H2D capacity event.
    physical.free_pages = 4
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    assert torch.equal(coordinator.kvduo_req_hot_capacity, expected)


def test_only_the_request_layer_with_a_capacity_event_grows():
    coordinator, _ = make_coordinator(layers=3, requests=2)
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    coordinator.kvduo_hot_growth_events[2, 1] = 1

    coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1]))

    expected = torch.full((3, 2), 4, dtype=torch.int64)
    expected[2, 1] = 8
    assert torch.equal(coordinator.kvduo_req_hot_capacity, expected)


def test_multiple_h2d_capacity_events_coalesce_into_one_growth_step():
    coordinator, _ = make_coordinator()
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))
    coordinator.kvduo_hot_growth_events[0, 0] = 3

    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))

    assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 8


def test_mixed_minimum_failure_cannot_replay_with_zero_capacity():
    coordinator, _ = make_coordinator()
    coordinator._ensure_kvduo_hot_capacity_targets_batch = MagicMock()

    with pytest.raises(RuntimeError, match="mandatory HOT capacity"):
        coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))


def test_growth_events_from_finished_requests_are_consumed_without_allocation():
    coordinator, physical = make_coordinator(requests=2)
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    physical.alloc.reset_mock()
    coordinator._active_kvduo_reqs.pop(1)
    coordinator.kvduo_hot_growth_events[0, 1] = 1

    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))

    physical.alloc.assert_not_called()
    assert not torch.any(coordinator.kvduo_hot_growth_events)


def test_event_read_waits_for_the_preceding_producer_on_the_rank_device(monkeypatch):
    import sglang.srt.managers.hisparse_coordinator as hisparse_coordinator

    coordinator, _ = make_coordinator()
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))
    coordinator.device = torch.device("cuda:3")
    coordinator.decode_producer_stream = object()
    stream = SimpleNamespace(wait_stream=MagicMock())
    current_stream = MagicMock(return_value=stream)
    monkeypatch.setattr(
        hisparse_coordinator.device_module, "current_stream", current_stream
    )
    consume = coordinator._consume_kvduo_hot_growth_events

    def read_events():
        stream.wait_stream.assert_called_once_with(coordinator.decode_producer_stream)
        return consume()

    monkeypatch.setattr(coordinator, "_consume_kvduo_hot_growth_events", read_events)
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))

    current_stream.assert_called_once_with(torch.device("cuda:3"))


@pytest.mark.parametrize("dsv4", [False, True])
def test_resolver_ffi_receives_the_stable_growth_event_pointer(monkeypatch, dsv4):
    import sglang.kernels.ops.kvcache.hisparse as hisparse_ops

    module = SimpleNamespace(load_cache_to_device_buffer=MagicMock())
    monkeypatch.setattr(
        hisparse_ops, "_jit_sparse_module", lambda *args, **kwargs: module
    )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=0)
    )
    coordinator, _ = make_coordinator()
    resolver = (
        hisparse_ops.load_cache_to_device_buffer_dsv4_mla
        if dsv4
        else hisparse_ops.load_cache_to_device_buffer_mla
    )
    versions = torch.ones(16, dtype=torch.int64)
    resolver(
        top_k_tokens=torch.tensor([[0, 1]], dtype=torch.int32),
        device_buffer_tokens=coordinator.req_device_buffer_tokens[0],
        host_cache_locs=torch.arange(16, dtype=torch.int64).view(1, -1),
        device_buffer_locs=coordinator.req_device_buffer_token_locs[0],
        host_cache=torch.zeros((16, 1, 8)),
        device_buffer=torch.zeros((16, 1, 8)),
        top_k_device_locs=torch.zeros((1, 2), dtype=torch.int32),
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        lru_slots=coordinator.lru_slots[0],
        item_size_bytes=32,
        num_top_k=2,
        hot_buffer_size=32,
        num_real_reqs=torch.ones(1, dtype=torch.int32),
        req_hot_buffer_sizes=coordinator.kvduo_req_hot_capacity_gpu[0],
        capacity_ema=coordinator.kvduo_capacity_ema[0],
        capacity_steps=coordinator.kvduo_capacity_steps[0],
        req_to_logical_token=torch.arange(16, dtype=torch.int64).view(1, -1),
        full_to_device_loc=torch.zeros(16, dtype=torch.int64),
        full_last_touch=torch.zeros(16, dtype=torch.int64),
        full_data_version=versions,
        full_host_version=versions.clone(),
        swap_status=torch.zeros(1, dtype=torch.int32),
        resolver_stats=coordinator.kvduo_resolver_stats[0],
        touch_clock=torch.ones(1, dtype=torch.int64),
        hot_growth_events=coordinator.kvduo_hot_growth_events[0],
        observe_cache=coordinator.kvduo_observe,
    )

    args = module.load_cache_to_device_buffer.call_args.args
    assert args[20] == coordinator.kvduo_hot_growth_events[0].data_ptr()
    assert args[12] == coordinator.kvduo_capacity_ema[0].data_ptr()
    assert args[27] == coordinator.kvduo_capacity_steps[0].data_ptr()
    assert args[-4:] == (0, 0, 0, 0)  # This fixture keeps the original shared path.
    assert all(isinstance(arg, (int, bool, float)) for arg in args)
