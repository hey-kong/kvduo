from types import SimpleNamespace

import pytest
import torch
from sglang.srt.mem_cache.kvduo_shared_hbm import SharedHBMBlocks
from sglang.srt.mem_cache.kvduo_shared_layout import KVDuoSharedLayout, Region


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
        self.created = 0

    def create(self, size):
        self.created += 1
        return self.created

    def map(self, address, size, handle):
        assert address not in self.maps
        self.maps[address] = handle

    def unmap(self, address, size):
        del self.maps[address]


@pytest.fixture
def layout(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(synchronize=lambda: None),
    )
    result = object.__new__(KVDuoSharedLayout)
    result.device = torch.device("cpu")
    result.block_bytes = 64
    result.initial_pages = result.current_pages = 64
    result.maximum_pages = 128
    result.chunk_bytes = 256
    result.loans = []
    result.regions = []
    for address, kind, rows, layers, row_bytes in (
        (65536, "full", 65, 2, 32),
        (131072, "history", 129, 2, 8),
        (196608, "history", 129, 1, 8),
    ):
        tensor = Tensor(address, rows, layers, row_bytes)
        result.regions.append(Region(tensor, kind, rows, layers, row_bytes))
    initial = result._blocks("full", 0, 65) | result._blocks("history", 0, 65)
    result.backing = SharedHBMBlocks(
        Backend(), block_bytes=64, budget_bytes=len(initial) * 64
    )
    result.backing.prepare(
        result.backing.plan(destinations=sorted(initial)), accesses_complete=True
    ).commit()
    result.bind(Allocator(range(1, 129)), Allocator(range(1, 65)))
    result.logical_allocator.free_pages = torch.tensor([63, 64])
    result.physical_allocator.free_pages = torch.arange(3, 65)
    return result


def grow(layout, **kwargs):
    return layout.grow(
        required_free_pages=10,
        protected_full_pages=20,
        agree=lambda value: True,
        **kwargs,
    )


def test_borrow_publishes_only_backed_logical_pages_and_preserves_live_full(layout):
    allocated = layout.backing.allocated_bytes
    before = set(layout.physical_allocator.free_pages.tolist())
    withheld = grow(layout)
    assert withheld > 0
    assert layout.backing.allocated_bytes == allocated
    assert layout.logical_allocator.available_size() // 256 >= 10
    assert layout.physical_allocator.available_size() // 256 >= 20
    assert len(before - set(layout.physical_allocator.free_pages.tolist())) == withheld
    loan = layout.loans[-1]
    assert layout.current_pages > 64
    for region in layout.regions:
        if region.kind == "history":
            assert torch.all(region.tensor.storage[:, loan.start : loan.end] == 0)
            assert torch.all(region.tensor.storage[:, : loan.start] == 7)
    assert layout._blocks("full", 0, 3) <= layout.backing.mappings.keys()
    assert (
        layout._blocks("history", 0, layout.current_pages + 1)
        <= layout.backing.mappings.keys()
    )


def test_return_requires_entire_loan_unused_and_restores_full_pages(layout):
    physical_before = set(layout.physical_allocator.free_pages.tolist())
    logical_before = set(layout.logical_allocator.free_pages.tolist())
    withheld = grow(layout)
    used = layout.loans[-1].start
    layout.logical_allocator.free_pages = layout.logical_allocator.free_pages[
        layout.logical_allocator.free_pages != used
    ]
    assert layout.trim(agree=lambda value: True) == 0
    layout.logical_allocator.free_pages = torch.cat(
        (layout.logical_allocator.free_pages, torch.tensor([used]))
    )
    assert layout.trim(agree=lambda value: True) == withheld
    assert layout.current_pages == 64
    assert set(layout.physical_allocator.free_pages.tolist()) == physical_before
    assert set(layout.logical_allocator.free_pages.tolist()) == logical_before
    assert layout._blocks("full", 0, 65) <= layout.backing.mappings.keys()


def test_no_loan_when_current_history_fits(layout):
    assert (
        layout.grow(
            required_free_pages=2, protected_full_pages=0, agree=lambda value: True
        )
        == 0
    )
    assert not layout.loans


def test_protected_budget_cannot_be_lent(layout):
    assert (
        layout.grow(
            required_free_pages=10, protected_full_pages=62, agree=lambda value: True
        )
        == 0
    )
    assert layout.current_pages == 64
    assert not layout.loans


def test_fragmented_free_pages_do_not_allow_live_blocks_to_be_unmapped(layout):
    layout.physical_allocator.free_pages = torch.arange(3, 65, 2)
    before = dict(layout.backing.mappings)
    assert grow(layout) == 0
    assert layout.backing.mappings == before


def test_tp_rejected_plan_changes_neither_allocator(layout):
    before = dict(layout.backing.mappings)
    logical = layout.logical_allocator.free_pages.clone()
    physical = layout.physical_allocator.free_pages.clone()
    calls = []

    def agree(value):
        calls.append(value)
        return len(calls) == 1

    assert (
        layout.grow(required_free_pages=10, protected_full_pages=20, agree=agree) == 0
    )
    assert layout.backing.mappings == before
    assert torch.equal(logical, layout.logical_allocator.free_pages)
    assert torch.equal(physical, layout.physical_allocator.free_pages)


def test_tp_free_counts_use_common_minimum(layout):
    grow(layout, common_available=lambda local: 0)
    assert layout.current_pages >= 74


def test_virtual_ceiling_is_not_exceeded(layout):
    layout.maximum_pages = 65
    assert grow(layout) == 0
    assert not layout.loans


def test_reset_preserves_unbacked_pages_and_allows_lifo_return(layout):
    grow(layout)
    loan = layout.loans[-1]
    layout.logical_allocator.free_pages = torch.arange(1, 129)
    layout.physical_allocator.free_pages = torch.arange(1, 65)
    layout.reset_free_lists()
    assert layout.logical_allocator.free_pages.max() == layout.current_pages
    assert set(loan.physical_pages).isdisjoint(
        layout.physical_allocator.free_pages.tolist()
    )
    assert layout.trim(agree=lambda value: True) == len(loan.physical_pages)


def test_repeated_borrow_return_has_no_handle_or_page_leak(layout):
    count = layout.backing._allocated
    for _ in range(8):
        borrowed = grow(layout)
        assert borrowed
        assert layout.trim(agree=lambda value: True) == borrowed
        assert layout.backing._allocated == count
        assert len(layout.physical_allocator.free_pages) == 62


def test_growth_metadata_oom_keeps_original_mappings(layout, monkeypatch):
    mappings = dict(layout.backing.mappings)
    full = layout.physical_allocator.free_pages.clone()
    logical = layout.logical_allocator.free_pages.clone()
    votes = []

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("metadata allocation")

    monkeypatch.setattr(torch, "cat", fail)
    assert (
        layout.grow(
            required_free_pages=10,
            protected_full_pages=20,
            agree=lambda value: votes.append(value) or True,
        )
        == 0
    )
    assert votes[-1] is False
    assert layout.backing.mappings == mappings
    assert torch.equal(layout.physical_allocator.free_pages, full)
    assert torch.equal(layout.logical_allocator.free_pages, logical)
    assert not layout.loans


def test_committed_and_unmapped_capacity_follow_borrow_and_return(layout):
    from sglang.srt.mem_cache.allocator.hisparse import (
        DeepSeekV4HiSparseTokenToKVPoolAllocator,
    )

    allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
    allocator.page_size = 256
    allocator._size_full = 128 * 256
    allocator.kvduo_shared_layout = layout
    assert allocator.committed_full_size == 64 * 256
    assert allocator.unmapped_full_size == 64 * 256
    assert grow(layout) > 0
    assert allocator.committed_full_size == layout.current_pages * 256
    assert allocator.committed_full_size > 64 * 256
    assert (
        allocator.committed_full_size + allocator.unmapped_full_size
        == allocator.size_full
    )
    assert layout.trim(agree=lambda value: True) > 0
    assert allocator.committed_full_size == 64 * 256
    assert allocator.unmapped_full_size == 64 * 256
    assert allocator.size_full == 128 * 256


def test_failed_coarse_loan_does_not_change_layout_before_minimum_retry(layout):
    layout.chunk_bytes = 4096
    original = dict(layout.backing.mappings)
    physical = layout.physical_allocator.free_pages.clone()
    assert (
        layout.grow(
            required_free_pages=10, protected_full_pages=52, agree=lambda value: True
        )
        == 0
    )
    assert layout.current_pages == 64
    assert layout.backing.mappings == original
    assert torch.equal(layout.physical_allocator.free_pages, physical)
    assert (
        layout.grow(
            required_free_pages=10,
            protected_full_pages=52,
            agree=lambda value: True,
            minimum_only=True,
        )
        > 0
    )
    assert layout.current_pages == 72
    assert layout.physical_allocator.available_size() // 256 >= 52


def test_minimum_donor_bound_is_read_only_and_respects_virtual_ceiling(layout):
    original = dict(layout.backing.mappings)
    assert layout.minimum_donor_pages(10, 2) > 0
    assert layout.minimum_donor_pages(2, 2) == 0
    assert layout.minimum_donor_pages(200, 2) == 0
    assert layout.backing.mappings == original
    assert layout.current_pages == 64
