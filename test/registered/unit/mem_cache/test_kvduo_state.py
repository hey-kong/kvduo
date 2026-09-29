import importlib.util
import sys
from pathlib import Path

import pytest

# Load this dependency-free metadata module directly.  These tests must remain
# executable in CPU-only tooling environments where importing the top-level
# sglang package would pull in torch/numpy.
MODULE_PATH = (
    Path(__file__).parents[4]
    / "python/sglang/srt/mem_cache/sparsity/core/kvduo_state.py"
)
SPEC = importlib.util.spec_from_file_location("kvduo_state_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
STATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STATE
SPEC.loader.exec_module(STATE)

KVDuoAllocationPlan = STATE.KVDuoAllocationPlan
KVDuoEntryVersionState = STATE.KVDuoEntryVersionState
KVDuoFullPageState = STATE.KVDuoFullPageState
KVDuoHotCapacity = STATE.KVDuoHotCapacity
KVDuoHotPageAllocator = STATE.KVDuoHotPageAllocator
KVDuoHotPageGeometry = STATE.KVDuoHotPageGeometry
KVDuoPhysicalPageKind = STATE.KVDuoPhysicalPageKind
KVDuoPagePinState = STATE.KVDuoPagePinState
KVDuoPageResidency = STATE.KVDuoPageResidency
KVDuoPressureResult = STATE.KVDuoPressureResult
KVDuoPressureAction = STATE.KVDuoPressureAction
KVDuoPressureReason = STATE.KVDuoPressureReason
KVDuoPressureStatus = STATE.KVDuoPressureStatus
KVDuoResourceBudget = STATE.KVDuoResourceBudget
KVDuoResourceKind = STATE.KVDuoResourceKind
KVDuoResidencyCatalog = STATE.KVDuoResidencyCatalog
KVDuoTailRotation = STATE.KVDuoTailRotation
execute_kvduo_pressure_plan = STATE.execute_kvduo_pressure_plan
plan_kvduo_allocation = STATE.plan_kvduo_allocation


def test_whole_hot_page_cross_layer_addressing_and_attention_read():
    geometry = KVDuoHotPageGeometry(21, 64, 64 * 11)
    page_start = 3 * 64
    slots = (0, 63, 64, 1343)
    addresses = [geometry.encode(page_start, slot) for slot in slots]
    assert geometry.slots_per_page == 1344
    assert addresses == [192, 255, 896, 14335]

    # Swap-in writes and sparse attention gathers the same flat backing.  In
    # particular slots 64 and 1343 are not copied into logical layer zero.
    backing = [None] * (21 * geometry.layer_slot_stride)
    values = ["kv0", "kv63", "kv64", "kv1343"]
    for address, value in zip(addresses, values):
        backing[address] = value
    assert [backing[address] for address in addresses] == values


def test_hot_geometry_bounds_and_explicit_page_states():
    geometry = KVDuoHotPageGeometry(2, 4, 32)
    with pytest.raises(IndexError):
        geometry.encode(0, geometry.slots_per_page)
    assert {kind.name for kind in KVDuoPhysicalPageKind} == {"FREE", "FULL", "HOT"}


def make_page(page_id, *, gpu=True, host=True):
    page = KVDuoFullPageState(page_id, ("shared", "shared", "mla"))
    if gpu:
        page.gpu_full_domains.update(page.domains)
    if host:
        for domain in page.domains:
            page.mark_host_copy_complete(domain)
    page.sealed = True
    page.stable = True
    return page


def test_page_deduplicates_shared_domains_and_requires_valid_host_copy():
    page = make_page("p0")
    assert page.domains == ("shared", "mla")
    assert page.residency is KVDuoPageResidency.GPU_FULL
    assert page.reclaimable

    page.record_model_write("mla", clock=4)
    assert not page.host_full_valid
    assert not page.reclaimable
    assert page.last_touch == 4

    page.stable = True
    page.mark_host_copy_complete("mla")
    assert page.host_full_valid
    assert page.reclaimable


def test_copy_and_restore_do_not_fabricate_touch():
    page = make_page("p0", host=False)
    page.record_attention_touch(7)
    page.mark_host_copy_complete("shared")
    page.mark_host_copy_complete("mla")
    assert page.last_touch == 7


def test_tail_and_inflight_pins_prevent_reclaim():
    page = make_page("p0")
    page.tail_protected = True
    assert not page.reclaimable
    page.tail_protected = False
    page.pins = KVDuoPagePinState(dma=1)
    assert not page.reclaimable
    page.pins.dma = 0
    assert page.reclaimable


def test_catalog_tracks_shared_request_ownership_and_prefix_boundary():
    catalog = KVDuoResidencyCatalog()
    for page_id in ("a", "b", "c", "d"):
        catalog.add_page(make_page(page_id))
    catalog.pages["c"].gpu_full_domains.clear()

    first = catalog.attach_request("r1", ("a", "b", "c", "d"))
    catalog.attach_request("r2", ("a", "b"))
    assert first.longest_contiguous_gpu_prefix(catalog.pages) == 2
    assert catalog.pages["a"].request_owners == {"r1", "r2"}

    catalog.detach_request("r1")
    assert catalog.pages["a"].request_owners == {"r2"}


def test_capacity_and_allocation_plan_use_physical_bytes():
    capacity = KVDuoHotCapacity(4096, 8192, 12288)
    assert capacity.shrinkable_bytes == 4096

    plan = KVDuoAllocationPlan(
        available_bytes=8192,
        new_full_bytes=4096,
        new_hot_bytes=4096,
        reserved_bytes=1024,
        turnover_bytes=2048,
    )
    assert plan.required_bytes == 11264
    assert plan.shortfall_bytes == 3072
    assert plan.status is KVDuoPressureStatus.RECLAIM_REQUIRED


def test_state_objects_reject_invalid_invariants():
    with pytest.raises(ValueError, match="minimum"):
        KVDuoHotCapacity(8, 4, 16)
    with pytest.raises(ValueError, match="negative"):
        KVDuoPagePinState(attention=-1).validate()
    with pytest.raises(ValueError, match="READY"):
        KVDuoPressureResult(KVDuoPressureStatus.READY, 8, 0, 1)


def test_pressure_plan_reclaims_exact_main_pool_shortfall_and_rechecks():
    available = [4096]
    reclaim_calls = []

    def reclaim(shortfall):
        reclaim_calls.append(shortfall)
        available[0] += shortfall
        return shortfall

    plan = KVDuoAllocationPlan(
        available_bytes=available[0],
        new_hot_bytes=4096,
        reserved_bytes=1024,
        turnover_bytes=2048,
        pending_commitment_bytes=1024,
    )
    result = execute_kvduo_pressure_plan(plan, reclaim, lambda: available[0])

    assert reclaim_calls == [4096]
    assert result.action is KVDuoPressureAction.SUCCESS
    assert result.reason is KVDuoPressureReason.NONE
    assert result.reclaimed_bytes == 4096


def test_independent_resource_shortage_never_reclaims_main_kv():
    reclaim_calls = []
    plan = KVDuoAllocationPlan(
        available_bytes=8192,
        new_hot_bytes=4096,
        independent_budgets=(KVDuoResourceBudget(KVDuoResourceKind.SWA, 1024, 2048),),
    )
    result = execute_kvduo_pressure_plan(plan, reclaim_calls.append, lambda: 8192)

    assert reclaim_calls == []
    assert result.action is KVDuoPressureAction.ERROR
    assert result.reason is KVDuoPressureReason.INVALID_STATE
    assert "SWA" in result.detail


def test_pressure_result_waits_only_with_explicit_dma_progress_source():
    plan = KVDuoAllocationPlan(available_bytes=0, new_hot_bytes=4096)
    without_progress = execute_kvduo_pressure_plan(plan, lambda _: 0, lambda: 0)
    with_progress = execute_kvduo_pressure_plan(
        plan, lambda _: 0, lambda: 0, dma_can_make_progress=True
    )

    assert without_progress.action is KVDuoPressureAction.ERROR
    assert without_progress.reason is KVDuoPressureReason.INSUFFICIENT_RECLAIMABLE_PAGES
    assert with_progress.action is KVDuoPressureAction.WAIT
    assert with_progress.reason is KVDuoPressureReason.DMA_PENDING


def test_planner_rejects_duplicate_or_main_independent_pool_snapshots():
    swa = KVDuoResourceBudget(KVDuoResourceKind.SWA, 8, 4)
    with pytest.raises(ValueError, match="only once"):
        plan_kvduo_allocation(available_main_kv_bytes=8, independent_budgets=(swa, swa))
    with pytest.raises(ValueError, match="Main KV"):
        plan_kvduo_allocation(
            available_main_kv_bytes=8,
            independent_budgets=(
                KVDuoResourceBudget(KVDuoResourceKind.MAIN_KV_PHYSICAL, 8, 4),
            ),
        )


def test_tail_rotation_seals_writes_back_and_protects_by_page():
    tail = KVDuoTailRotation(page_size=4, tail_protected_pages=2)
    for position in range(4):
        transition = tail.append(position, ("shared", "shared", "mla"), clock=position)
        assert tail.entry_readable(0, "shared", position)
        if position < 3:
            assert not tail.pages[0].gpu_full
    assert transition.sealed_page == 0
    assert tail.pages[0].gpu_full
    tail.mark_stable(0)

    transition = tail.append(4, ("shared", "mla"), clock=4)
    assert transition.new_page
    assert transition.protected_pages == (0, 1)
    assert transition.writeback_candidates == (0,)
    tail.submit_writeback(0)
    assert tail.pages[0].pins.dma == 1
    assert not tail.pages[0].reclaimable
    tail.complete_writeback(0)
    assert tail.pages[0].host_full_valid
    # It is still protected as the second newest page.
    assert not tail.pages[0].reclaimable

    for position in range(5, 9):
        tail.append(position, ("shared", "mla"), clock=position)
    assert tail.protected_pages == (1, 2)
    assert not tail.pages[0].tail_protected
    assert tail.pages[0].reclaimable


def test_tail_adapter_can_extend_protection_and_reject_holes():
    tail = KVDuoTailRotation(
        page_size=2, tail_protected_pages=2, adapter_extra_protected_pages=3
    )
    for position in range(6):
        tail.append(position, ("c4",), clock=position)
    assert tail.protected_pages == (0, 1, 2)
    with pytest.raises(ValueError, match="contiguously"):
        tail.append(8, ("c4",), clock=8)


def test_write_attention_and_version_lifecycle():
    entry = KVDuoEntryVersionState()
    entry.initialize_generation(generation=1, clock=3)
    first_ticket = entry.begin_writeback()
    entry.model_update(clock=5)
    assert not entry.host_valid
    assert not entry.complete_writeback(first_ticket)
    assert not entry.host_valid

    current_ticket = entry.begin_writeback()
    assert entry.complete_writeback(current_ticket)
    before = (entry.last_touch, entry.data_version, entry.host_version)
    entry.scan_or_score()
    entry.attach_or_restore()
    assert (entry.last_touch, entry.data_version, entry.host_version) == before
    entry.attention_complete(clock=8)
    assert entry.last_touch == 8


def test_address_reuse_does_not_inherit_touch_or_stale_writeback():
    entry = KVDuoEntryVersionState()
    entry.initialize_generation(generation=1, clock=100)
    stale = entry.begin_writeback()
    entry.initialize_generation(generation=2, clock=2)
    assert entry.last_touch == 2
    assert not entry.complete_writeback(stale)
    assert not entry.host_valid


def test_aggregate_touch_matches_per_layer_entry_reference():
    reference = {
        "layer0": [(1, 7), (4, 2)],
        "layer1": [(3, 5), (9, 6)],
    }
    per_entry_touch = [
        max(write, attention)
        for entries in reference.values()
        for write, attention in entries
    ]
    aggregate = KVDuoFullPageState("p", ("layer0", "layer1"))
    for touch in sorted(per_entry_touch):
        aggregate.record_attention_touch(touch)
    assert aggregate.last_touch == max(per_entry_touch) == 9


def test_hot_pages_are_request_domain_private_and_grow_only_on_host_miss():
    allocator = KVDuoHotPageAllocator(
        page_size=2, domain_aliases={"shared-layer-1": "shared-storage"}
    )
    physical_pages = iter((10, 11, 12))

    def allocate():
        return next(physical_pages)

    assert allocator.pages == {}
    assert allocator.lookup("r1", "shared-layer-1", 7) is None
    assert allocator.pages == {}  # lookup/full-page demotion allocates nothing

    assert allocator.insert_host_miss(
        "r1", "shared-layer-1", 7, clock=3, allocate_page=allocate
    ) == (10, 0)
    assert allocator.insert_host_miss(
        "r1", "shared-storage", 99, clock=4, allocate_page=allocate
    ) == (10, 1)
    # Entries from different original logical pages may share one hot page.
    assert allocator.pages[10].last_touch == 4

    assert allocator.insert_host_miss(
        "r2", "shared-storage", 7, clock=5, allocate_page=allocate
    ) == (11, 0)
    assert allocator.insert_host_miss(
        "r1", "other-layer", 7, clock=6, allocate_page=allocate
    ) == (12, 0)
    assert allocator.pages[10].request_id == "r1"
    assert allocator.pages[11].request_id == "r2"
    assert allocator.pages[12].domain_id == "other-layer"


def test_hot_hit_refreshes_slot_and_page_lru_without_allocating():
    allocator = KVDuoHotPageAllocator(page_size=4)
    calls = []

    def allocate():
        calls.append(True)
        return 20

    allocator.insert_host_miss("r", "layer", 1, clock=1, allocate_page=allocate)
    allocator.insert_host_miss(
        "r", "layer", 1, clock=9, allocate_page=lambda: pytest.fail("allocated")
    )
    assert len(calls) == 1
    assert allocator.pages[20].last_touch == 9
    assert allocator.pages[20].slots[0].last_touch == 9
