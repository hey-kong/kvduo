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
KVDuoFullPageState = STATE.KVDuoFullPageState
KVDuoHotCapacity = STATE.KVDuoHotCapacity
KVDuoPagePinState = STATE.KVDuoPagePinState
KVDuoPageResidency = STATE.KVDuoPageResidency
KVDuoPressureResult = STATE.KVDuoPressureResult
KVDuoPressureAction = STATE.KVDuoPressureAction
KVDuoPressureReason = STATE.KVDuoPressureReason
KVDuoPressureStatus = STATE.KVDuoPressureStatus
KVDuoResourceBudget = STATE.KVDuoResourceBudget
KVDuoResourceKind = STATE.KVDuoResourceKind
KVDuoResidencyCatalog = STATE.KVDuoResidencyCatalog
execute_kvduo_pressure_plan = STATE.execute_kvduo_pressure_plan
plan_kvduo_allocation = STATE.plan_kvduo_allocation


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
