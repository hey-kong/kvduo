from types import SimpleNamespace
from unittest.mock import MagicMock

from test_kvduo_pressure_admission import make_decode_batch, make_full_coordinator


def attach(coordinator):
    layout = SimpleNamespace(
        grow=MagicMock(return_value=1),
        current_pages=100,
        backing=SimpleNamespace(allocated_bytes=1024),
        growths=[],
        logical_allocator=SimpleNamespace(available_size=lambda: 0),
    )
    coordinator._kvduo_elastic_history = layout
    coordinator._kvduo_waiting_history_tokens = 64
    coordinator._kvduo_decode_history_tokens = 0
    coordinator._kvduo_history_trim_pending = False
    return layout


def test_full_reclaim_never_lends_its_physical_gain():
    coordinator, _, _ = make_full_coordinator(full_reclaim_policy="request")
    layout = attach(coordinator)
    assert (
        coordinator.reclaim_kvduo_full_pages(coordinator.page_size)
        == 6 * coordinator.page_size
    )
    assert layout.grow.call_args.kwargs["required_free_pages"] == 32
    assert "protected_full_pages" not in layout.grow.call_args.kwargs


def test_decode_history_growth_does_not_reclaim_full():
    batch, _ = make_decode_batch([256], logical_size=0, free_pages=4)
    coordinator = batch.hisparse_coordinator
    layout = attach(coordinator)
    logical = coordinator.token_to_kv_pool_allocator.logical_attn_allocator

    def grow(**kwargs):
        logical.full_available_size = lambda: 256
        logical.available_size = lambda: 256
        layout.logical_allocator.available_size = lambda: 256
        return 1

    layout.grow.side_effect = grow
    assert batch.check_decode_mem()
    coordinator.reclaim_kvduo_full_pages.assert_not_called()
    assert coordinator._kvduo_decode_history_tokens == 0


def test_swa_shortage_is_not_history_shortage():
    batch, _ = make_decode_batch([256], logical_size=0, swa_size=0, free_pages=4)
    layout = attach(batch.hisparse_coordinator)
    assert not batch.check_decode_mem()
    layout.grow.assert_not_called()


def test_mandatory_decode_growth_ignores_waiting_prefill():
    coordinator, _, _ = make_full_coordinator(full_reclaim_policy="request")
    layout = attach(coordinator)
    coordinator._kvduo_decode_history_tokens = 2
    coordinator._kvduo_waiting_history_tokens = 1000
    coordinator.grow_kvduo_history()
    assert layout.grow.call_args.kwargs["required_free_pages"] == 1


def test_end_cleanup_preserves_host():
    coordinator, _, _ = make_full_coordinator(full_reclaim_policy="request")
    layout = attach(coordinator)
    layout.growths = [object()]
    layout.trim = MagicMock(return_value=1)
    coordinator.kvduo_tree_cache = SimpleNamespace(
        full_evictable_size=lambda: 16, evict=MagicMock()
    )
    coordinator.maintain_kvduo_history_backing()
    layout.trim.assert_not_called()
    coordinator._kvduo_history_trim_pending = True
    coordinator.maintain_kvduo_history_backing()
    layout.trim.assert_called_once()
    coordinator.mem_pool_host.free.assert_not_called()
