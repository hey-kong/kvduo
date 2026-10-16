from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_kvduo_pressure_admission import make_decode_batch, make_full_coordinator


def attach(coordinator, *, borrowed=1):
    layout = SimpleNamespace(
        grow=MagicMock(return_value=borrowed),
        current_pages=64,
        maximum_pages=128,
        backing=SimpleNamespace(allocated_bytes=1024),
        loans=[],
        minimum_donor_pages=lambda *args: 1,
    )
    coordinator._kvduo_shared_layout = layout
    coordinator._kvduo_shared_trim_pending = False
    return layout


def test_request_reclaim_does_not_borrow_for_waiting_prefill():
    coordinator, _, _ = make_full_coordinator(full_reclaim_policy="request")
    layout = attach(coordinator)
    result = coordinator.reclaim_kvduo_full_pages(coordinator.page_size)
    # Six pages are returned intact; reclamation is not prefill admission.
    assert result == 6 * coordinator.page_size
    layout.grow.assert_not_called()


@pytest.mark.parametrize("release_all", [True, False])
def test_page_policy_and_finish_release_do_not_trigger_optional_loan(release_all):
    coordinator, _, _ = make_full_coordinator(
        full_reclaim_policy="request" if release_all else "page"
    )
    layout = attach(coordinator)
    coordinator.reclaim_kvduo_full_pages(coordinator.page_size, release_all=release_all)
    layout.grow.assert_not_called()


def test_decode_history_pressure_reclaims_before_original_logical_rejection():
    batch, _ = make_decode_batch([256], logical_size=0, free_pages=4)
    coordinator = batch.hisparse_coordinator
    attach(coordinator, borrowed=0)
    logical = coordinator.token_to_kv_pool_allocator.logical_attn_allocator

    def reclaim(*args, **kwargs):
        logical.full_available_size = lambda: 256
        logical.available_size = lambda: 256
        return 64

    coordinator.reclaim_kvduo_full_pages.side_effect = reclaim
    assert batch.check_decode_mem()
    coordinator.reclaim_kvduo_full_pages.assert_called_once_with(
        64, reserve_hot_minimum=True
    )


def test_swa_shortage_is_not_treated_as_borrowable_history():
    batch, _ = make_decode_batch([256], logical_size=0, swa_size=0, free_pages=4)
    layout = attach(batch.hisparse_coordinator)
    assert not batch.check_decode_mem()
    batch.hisparse_coordinator.reclaim_kvduo_full_pages.assert_not_called()
    layout.grow.assert_not_called()


def test_borrow_protects_only_the_explicit_execution_budget():
    batch, _ = make_decode_batch([256], logical_size=0, free_pages=4)
    coordinator = batch.hisparse_coordinator
    layout = attach(coordinator)
    coordinator._borrow_kvduo_history(256, 3)
    kwargs = layout.grow.call_args.kwargs
    assert kwargs["required_free_pages"] == 1
    assert kwargs["protected_full_pages"] == 3


def test_end_cleanup_only_runs_when_flagged_and_never_evicts_host():
    coordinator, _, _ = make_full_coordinator(full_reclaim_policy="request")
    layout = attach(coordinator)
    layout.loans = [object()]
    layout.trim = MagicMock(return_value=1)
    coordinator.kvduo_tree_cache = SimpleNamespace(
        full_evictable_size=lambda: 16, evict=MagicMock()
    )
    coordinator.maintain_kvduo_shared_backing()
    layout.trim.assert_not_called()
    coordinator._kvduo_shared_trim_pending = True
    coordinator.maintain_kvduo_shared_backing()
    layout.trim.assert_called_once()
    coordinator.kvduo_tree_cache.evict.assert_called_once()
    coordinator.mem_pool_host.free.assert_not_called()


@pytest.mark.parametrize("path", ["new", "chunk"])
@pytest.mark.parametrize(
    "case", ["ready", "borrow", "physical", "swa", "host", "fragmented"]
)
def test_prefill_only_borrows_after_other_admission_gates(path, case):
    import torch
    from test_kvduo_prefill_admission import (
        actual_budget_adder_and_req,
        ready_coordinator,
    )

    coordinator, physical = ready_coordinator()
    physical.free_pages = 64 if case != "physical" else 0
    adder, req = actual_budget_adder_and_req(
        coordinator,
        path=path,
        logical_size=1024 if case == "ready" else 0,
        swa_size=0 if case == "swa" else 1024,
    )
    coordinator.token_to_kv_pool_allocator = adder.token_to_kv_pool_allocator
    coordinator.is_dsv4_hisparse = True
    logical = (
        adder.token_to_kv_pool_allocator.logical_attn_allocator.full_attn_allocator
    )
    layout = attach(coordinator)
    layout.maximum_pages = 128
    layout.current_pages = 64
    coordinator._reserve_kvduo_prefill_host = MagicMock(return_value=case != "host")

    def grow(**kwargs):
        assert coordinator._reserve_kvduo_prefill_host.called
        assert isinstance(kwargs["required_free_pages"], int)
        assert physical.free_pages >= kwargs["protected_full_pages"]
        if case == "fragmented":
            return 0
        logical.free_pages = torch.arange(1, 65, dtype=torch.int64)
        physical.free_pages -= 4
        layout.current_pages += 64
        return 4

    layout.grow.side_effect = grow
    adder.rem_total_token_offset = 0.5
    before = physical.free_pages
    if path == "chunk":
        adder.add_chunked_req(req)
    else:
        adder.add_one_req(req, False, None)
    assert bool(adder.can_run_list) == (case in ("ready", "borrow"))
    if case in ("borrow", "fragmented"):
        assert layout.grow.called
    else:
        layout.grow.assert_not_called()
    if case != "borrow":
        assert physical.free_pages == before
        assert layout.current_pages == 64


def test_decode_borrows_existing_idle_backing_before_reclaim():
    batch, _ = make_decode_batch([256], logical_size=0, free_pages=4)
    coordinator = batch.hisparse_coordinator
    layout = attach(coordinator)
    logical = coordinator.token_to_kv_pool_allocator.logical_attn_allocator

    def grow(**kwargs):
        logical.full_available_size = lambda: 256
        return 1

    layout.grow.side_effect = grow
    coordinator.ensure_kvduo_decode_history(256)
    layout.grow.assert_called_once()
    coordinator.reclaim_kvduo_full_pages.assert_not_called()
    coordinator.reclaim_kvduo_hot_pages.assert_not_called()


def test_decode_with_sufficient_history_never_plans_a_loan():
    batch, _ = make_decode_batch([256], logical_size=1024, free_pages=4)
    layout = attach(batch.hisparse_coordinator)
    batch.hisparse_coordinator.ensure_kvduo_decode_history(256)
    layout.grow.assert_not_called()
    batch.hisparse_coordinator.reclaim_kvduo_full_pages.assert_not_called()


@pytest.mark.parametrize("minimum_succeeds", [True, False])
def test_prefill_coarse_loan_falls_back_without_consuming_execution_pages(
    minimum_succeeds,
):
    from test_kvduo_prefill_admission import (
        actual_budget_adder_and_req,
        ready_coordinator,
    )

    coordinator, physical = ready_coordinator()
    physical.free_pages = 64
    adder, req = actual_budget_adder_and_req(coordinator, logical_size=0)
    coordinator.token_to_kv_pool_allocator = adder.token_to_kv_pool_allocator
    coordinator.is_dsv4_hisparse = True
    logical = adder.token_to_kv_pool_allocator.logical_attn_allocator
    layout = attach(coordinator)
    coordinator._reserve_kvduo_prefill_host = MagicMock(return_value=True)

    def grow(**kwargs):
        if kwargs.get("minimum_only") and minimum_succeeds:
            logical.full_available_size = lambda: 1024
            physical.free_pages -= 1
            return 1
        return 0

    layout.grow.side_effect = grow
    adder.add_one_req(req, False, None)
    assert bool(adder.can_run_list) == minimum_succeeds
    assert layout.grow.call_count == 2
    assert not layout.grow.call_args_list[0].kwargs.get("minimum_only", False)
    assert layout.grow.call_args_list[1].kwargs["minimum_only"]
    assert physical.free_pages == 64 - int(minimum_succeeds)


def test_decode_history_uses_hot_merge_only_after_full_reclaim():
    batch, physical = make_decode_batch([256], logical_size=0, free_pages=0)
    coordinator = batch.hisparse_coordinator
    layout = attach(coordinator, borrowed=0)
    logical = coordinator.token_to_kv_pool_allocator.logical_attn_allocator
    calls = []
    coordinator.reclaim_kvduo_full_pages.side_effect = lambda *a, **k: (
        calls.append("full") or 0
    )

    def merge(*args):
        calls.append("hot")
        physical.free_pages = 8
        return 8 * coordinator.page_size

    coordinator.reclaim_kvduo_hot_pages.side_effect = merge

    def grow(**kwargs):
        if physical.free_pages >= 8:
            logical.full_available_size = lambda: 256
            return 1
        return 0

    layout.grow.side_effect = grow
    coordinator.ensure_kvduo_decode_history(256)
    assert calls == ["full", "hot"]
    assert logical.full_available_size() == 256


def test_exhausted_virtual_ceiling_does_not_reclaim_live_kv():
    batch, _ = make_decode_batch([256], logical_size=0, free_pages=0)
    coordinator = batch.hisparse_coordinator
    layout = attach(coordinator)
    layout.maximum_pages = layout.current_pages
    coordinator.ensure_kvduo_decode_history(256)
    layout.grow.assert_not_called()
    coordinator.reclaim_kvduo_full_pages.assert_not_called()
    coordinator.reclaim_kvduo_hot_pages.assert_not_called()


def test_first_chunk_does_not_borrow_backing_for_unexecuted_prompt():
    from test_kvduo_prefill_admission import (
        actual_budget_adder_and_req,
        ready_coordinator,
    )

    coordinator, physical = ready_coordinator()
    physical.free_pages = 64
    adder, req = actual_budget_adder_and_req(coordinator, logical_size=32)
    adder.rem_chunk_tokens = 16
    req.full_untruncated_fill_ids = list(range(128))
    coordinator.token_to_kv_pool_allocator = adder.token_to_kv_pool_allocator
    coordinator.is_dsv4_hisparse = True
    layout = attach(coordinator)
    coordinator._reserve_kvduo_prefill_host = MagicMock(return_value=True)
    adder.add_one_req(req, False, None)
    assert adder.can_run_list == [req]
    assert req.extend_range.length == 16
    layout.grow.assert_not_called()


@pytest.mark.parametrize("progress", [True, False])
def test_prefill_reclaims_committed_radix_history_before_borrow_at_virtual_limit(
    progress,
):
    import torch
    from test_kvduo_prefill_admission import (
        actual_budget_adder_and_req,
        ready_coordinator,
    )

    coordinator, physical = ready_coordinator()
    physical.free_pages = 600
    adder, _ = actual_budget_adder_and_req(coordinator, logical_size=6400)
    coordinator.token_to_kv_pool_allocator = adder.token_to_kv_pool_allocator
    logical = (
        adder.token_to_kv_pool_allocator.logical_attn_allocator.full_attn_allocator
    )
    page = logical.page_size
    allocated = logical.free_pages[100:400].clone()
    logical.free_pages = logical.free_pages[:100]
    assert len(allocated) == 300
    layout = attach(coordinator, borrowed=0)
    layout.maximum_pages = layout.current_pages = 400
    adder.tree_cache.full_evictable_size.return_value = 300 * page

    def evict(params):
        assert params.num_tokens == 101 * page
        if progress:
            logical.free_pages = torch.cat((logical.free_pages, allocated[:101]))

    adder.tree_cache.evict.side_effect = evict
    token = coordinator._reserve_kvduo_prefill_physical(
        1, history_tokens=201 * page, tree_cache=adder.tree_cache
    )
    assert (token is not None) == progress
    adder.tree_cache.evict.assert_called_once()
    assert physical.free_pages == 600
    if progress:
        layout.grow.assert_not_called()
        assert logical.available_size() == 201 * page
    else:
        # A stale evictable count cannot cause an endless eviction loop.
        assert layout.grow.call_count == 2
        assert logical.available_size() == 100 * page


@pytest.mark.parametrize("per_request_pages", [2, 6])
def test_decode_reclaim_includes_donors_and_retries_full_before_hot(per_request_pages):
    batch, physical = make_decode_batch([256], logical_size=0, free_pages=0)
    coordinator = batch.hisparse_coordinator
    logical = coordinator.token_to_kv_pool_allocator.logical_attn_allocator
    layout = attach(coordinator, borrowed=0)
    layout.minimum_donor_pages = lambda *args: 5
    targets = []

    def reclaim(tokens, **kwargs):
        targets.append(tokens // coordinator.page_size)
        physical.free_pages += per_request_pages
        return per_request_pages * coordinator.page_size

    def grow(**kwargs):
        if physical.free_pages >= kwargs["protected_full_pages"] + 5:
            physical.free_pages -= 5
            logical.full_available_size = lambda: 256
            return 5
        return 0

    coordinator.reclaim_kvduo_full_pages.side_effect = reclaim
    layout.grow.side_effect = grow
    coordinator.ensure_kvduo_decode_history(256)
    assert targets == ([6, 4, 2] if per_request_pages == 2 else [6])
    assert physical.free_pages == 1
    assert logical.full_available_size() == 256
    coordinator.reclaim_kvduo_hot_pages.assert_not_called()


def test_decode_retries_full_when_free_bytes_are_fragmented():
    batch, physical = make_decode_batch([256], logical_size=0, free_pages=10)
    coordinator = batch.hisparse_coordinator
    logical = coordinator.token_to_kv_pool_allocator.logical_attn_allocator
    layout = attach(coordinator, borrowed=0)
    layout.minimum_donor_pages = lambda *args: 5
    calls = []

    def reclaim(tokens, **kwargs):
        calls.append(tokens)
        physical.free_pages += 1
        return coordinator.page_size

    def grow(**kwargs):
        if len(calls) == 2:
            logical.full_available_size = lambda: 256
            return 5
        return 0

    coordinator.reclaim_kvduo_full_pages.side_effect = reclaim
    layout.grow.side_effect = grow
    coordinator.ensure_kvduo_decode_history(256)
    assert calls == [coordinator.page_size, coordinator.page_size]
    coordinator.reclaim_kvduo_hot_pages.assert_not_called()


def test_decode_reclaim_recomputes_new_hot_minimum_before_next_loan():
    batch, physical = make_decode_batch([256], logical_size=0, free_pages=0)
    coordinator = batch.hisparse_coordinator
    logical = coordinator.token_to_kv_pool_allocator.logical_attn_allocator
    layout = attach(coordinator, borrowed=0)
    layout.minimum_donor_pages = lambda *args: 5
    coordinator._kvduo_mandatory_hot_pages = MagicMock(return_value=0)
    targets = []

    def reclaim(tokens, **kwargs):
        targets.append(tokens // coordinator.page_size)
        coordinator._kvduo_mandatory_hot_pages.return_value = 1
        released = 6 if len(targets) == 1 else 1
        physical.free_pages += released
        return released * coordinator.page_size

    def grow(**kwargs):
        if physical.free_pages >= kwargs["protected_full_pages"] + 5:
            logical.full_available_size = lambda: 256
            physical.free_pages -= 5
            return 5
        return 0

    coordinator.reclaim_kvduo_full_pages.side_effect = reclaim
    layout.grow.side_effect = grow
    coordinator.ensure_kvduo_decode_history(256)
    assert targets == [6, 1]
    assert physical.free_pages == 2  # One decode page and one new HOT minimum.
    coordinator.reclaim_kvduo_hot_pages.assert_not_called()
