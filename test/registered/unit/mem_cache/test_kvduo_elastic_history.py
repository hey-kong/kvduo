from types import SimpleNamespace

import pytest
import torch
from sglang.srt.mem_cache.kvduo_elastic_history import KVDuoElasticHistory, Region
from sglang.srt.mem_cache.kvduo_vmm_backing import VMMBacking


class Tensor:
    def __init__(self, address, rows, layers, row_bytes):
        self.address = address
        self.storage = torch.full((layers, rows, row_bytes), 7, dtype=torch.uint8)

    def data_ptr(self):
        return self.address

    def view(self, *shape):
        return self.storage.view(*shape)


class Allocator:
    page_size = 256

    def __init__(self, pages):
        self.free_pages = torch.tensor(pages, dtype=torch.int64)

    def merge_and_sort_free(self):
        self.free_pages = self.free_pages.sort().values

    def available_size(self):
        return self.free_pages.numel() * self.page_size


class Backend:
    def __init__(self):
        self.maps = {}
        self.handles = set()
        self.created = 0
        self.fail_create = False

    def create(self, size):
        if self.fail_create:
            raise MemoryError("CUDA allocation failed")
        self.created += 1
        self.handles.add(self.created)
        return self.created

    def map(self, address, size, handle):
        assert address not in self.maps
        self.maps[address] = handle

    def unmap(self, address, size):
        del self.maps[address]

    def release(self, handle):
        assert handle not in self.maps.values()
        self.handles.remove(handle)


@pytest.fixture
def layout(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(synchronize=lambda: None),
    )
    result = object.__new__(KVDuoElasticHistory)
    result.device = torch.device("cpu")
    result.block_bytes = 64
    result.initial_pages = result.current_pages = 64
    result.maximum_pages = 128
    result.chunk_bytes = 256
    result.reserve_bytes = 100
    result.growths = []
    result.regions = []
    for address, rows, layers, row_bytes in ((131072, 129, 2, 8), (196608, 129, 1, 8)):
        result.regions.append(
            Region(Tensor(address, rows, layers, row_bytes), rows, layers, row_bytes)
        )
    initial = result._blocks(0, 65)
    result.backing = VMMBacking(
        Backend(), block_bytes=64, budget_bytes=len(result._blocks(0, 129)) * 64
    )
    result.backing.prepare(
        result.backing.plan(destinations=sorted(initial)), accesses_complete=True
    ).commit()
    result.bind(Allocator(range(1, 129)))
    result.logical_allocator.free_pages = torch.tensor([63, 64])
    result._free_bytes = lambda: 100000
    return result


def grow(layout, **kwargs):
    return layout.grow(required_free_pages=10, agree=lambda value: True, **kwargs)


def test_growth_allocates_additional_hbm_and_preserves_old_rows(layout):
    before = layout.backing.allocated_bytes
    assert grow(layout) >= 8
    assert layout.backing.allocated_bytes > before
    assert not hasattr(layout, "physical_allocator")
    for region in layout.regions:
        assert torch.all(region.tensor.storage[:, :65] == 7)
        assert torch.all(region.tensor.storage[:, 65 : layout.current_pages + 1] == 0)
    assert layout.logical_allocator.available_size() >= 10 * 256


def test_trim_releases_driver_memory_only_after_all_rows_are_idle(layout):
    before = layout.backing.allocated_bytes
    grow(layout)
    logical = layout.logical_allocator
    logical.free_pages = logical.free_pages[logical.free_pages != 65]
    assert layout.trim(agree=lambda value: True) == 0
    logical.free_pages = torch.cat((logical.free_pages, torch.tensor([65])))
    assert layout.trim(agree=lambda value: True) > 0
    assert layout.current_pages == 64
    assert layout.backing.allocated_bytes == before
    assert len(layout.backing.backend.handles) * 64 == before


def test_no_shortage_never_allocates(layout):
    layout.logical_allocator.free_pages = torch.arange(1, 65)
    before = layout.backing.allocated_bytes
    assert grow(layout) == 0
    assert layout.backing.allocated_bytes == before


def test_reserve_and_tp_minimum_limit_growth(layout):
    before = dict(layout.backing.mappings)
    assert grow(layout, common_available=lambda value: min(value, 1)) == 0
    layout._free_bytes = lambda: 0
    assert grow(layout) == 0
    assert layout.backing.mappings == before


def test_driver_oom_retains_capacity_and_allocations(layout):
    before = layout.backing.allocated_bytes
    layout.backing.backend.fail_create = True
    assert grow(layout) == 0
    assert layout.current_pages == 64
    assert layout.backing.allocated_bytes == before


def test_tp_rejection_after_mapping_releases_new_handles(layout):
    before = layout.backing.allocated_bytes
    votes = iter([True, True, False])
    assert layout.grow(required_free_pages=10, agree=lambda value: next(votes)) == 0
    assert layout.current_pages == 64
    assert layout.backing.allocated_bytes == before


def test_metadata_oom_is_agreed_before_native_allocation(layout, monkeypatch):
    before = dict(layout.backing.mappings)

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("metadata")

    monkeypatch.setattr(torch, "cat", fail)
    votes = []
    assert (
        layout.grow(
            required_free_pages=10, agree=lambda value: votes.append(value) or True
        )
        == 0
    )
    assert votes[-1] is False
    assert layout.backing.mappings == before


def test_repeat_growth_trim_no_leak(layout):
    before = layout.backing.allocated_bytes
    for _ in range(8):
        assert grow(layout) > 0
        assert layout.trim(agree=lambda value: True) > 0
        assert layout.backing.allocated_bytes == before
        assert layout.current_pages == 64


def test_virtual_ceiling_does_not_publish_unmapped_pages(layout):
    assert layout.grow(required_free_pages=1000, agree=lambda value: True) == 0
    layout.logical_allocator.free_pages = torch.arange(1, 129)
    layout.reset_free_lists()
    assert layout.logical_allocator.free_pages.max() == 64


def test_mem_get_info_keeps_runtime_reserve(layout, monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (150, 1000))
    assert KVDuoElasticHistory._free_bytes(layout) == 50
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (50, 1000))
    assert KVDuoElasticHistory._free_bytes(layout) == 0


def test_no_headroom_does_not_wait_for_decode(layout):
    from unittest.mock import Mock

    boundary = Mock()
    layout._free_bytes = lambda: 0
    assert grow(layout, prepare_boundary=boundary) == 0
    boundary.assert_not_called()
