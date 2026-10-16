from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_kvduo_pressure_admission import make_decode_batch, make_full_coordinator


def attach(coordinator, *, borrowed=1):
    layout = SimpleNamespace(
        grow=MagicMock(return_value=borrowed),
        current_pages=100,
        backing=SimpleNamespace(allocated_bytes=1024),
        loans=[],
    )
    coordinator._kvduo_shared_layout = layout
    coordinator._kvduo_waiting_history_tokens = 64
    coordinator._kvduo_waiting_full_pages = 2
    coordinator._kvduo_decode_history_tokens = 0
    coordinator._kvduo_shared_trim_pending = False
    return layout


def test_request_reclaim_returns_net_physical_gain_after_loan():
    coordinator, _, _ = make_full_coordinator(full_reclaim_policy="request")
    layout = attach(coordinator)
    result = coordinator.reclaim_kvduo_full_pages(coordinator.page_size)
    # Eight pages, with the newest two protected: six released, one lent.
    assert result == 5 * coordinator.page_size
    kwargs = layout.grow.call_args.kwargs
    assert kwargs["required_free_pages"] == 32
    assert kwargs["protected_full_pages"] >= 3


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
    assert coordinator._kvduo_decode_history_tokens == 0


def test_swa_shortage_is_not_treated_as_borrowable_history():
    batch, _ = make_decode_batch([256], logical_size=0, swa_size=0, free_pages=4)
    layout = attach(batch.hisparse_coordinator)
    assert not batch.check_decode_mem()
    batch.hisparse_coordinator.reclaim_kvduo_full_pages.assert_not_called()
    layout.grow.assert_not_called()


def test_mandatory_decode_does_not_reserve_waiting_prefill_budget():
    coordinator, _, _ = make_full_coordinator(full_reclaim_policy="request")
    layout = attach(coordinator)
    layout.logical_allocator = SimpleNamespace(available_size=lambda: 100)
    coordinator._kvduo_decode_history_tokens = 2
    coordinator._kvduo_waiting_history_tokens = 1000
    coordinator._kvduo_waiting_full_pages = 500
    coordinator._borrow_kvduo_history_after_full_reclaim(1)
    kwargs = layout.grow.call_args.kwargs
    assert kwargs["required_free_pages"] == 1
    assert kwargs["protected_full_pages"] < 500


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
