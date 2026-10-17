"""Sealed Tail lifecycle after FULL -> HOT residency transition."""

from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.mem_cache.sparsity.core.kvduo_state import KVDuoResidencyMode
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_pressure_admission import make_full_coordinator

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def hot_request(ratio=1):
    c, p, req = make_full_coordinator(pages=10, ratio=ratio)
    req.kv.kv_allocated_len = 8 * c.page_size * ratio
    assert c.kvduo_residency_mode(0) is KVDuoResidencyMode.FULL_RESIDENT
    c.reclaim_kvduo_full_pages(c.page_size, reserve_hot_minimum=True)
    assert c.kvduo_residency_mode(0) is KVDuoResidencyMode.HOT_RESIDENT
    assert c._kvduo_hot_tail_starts[0][1] == 6 * c.page_size
    req.kv.kv_allocated_len = 9 * c.page_size * ratio
    assert not c._kvduo_tail_cleanup_pending
    return c, p, req


@pytest.mark.parametrize("ratio", [1, 4])
def test_tail_cleanup_releases_old_page_and_preserves_current_tail(ratio):
    c, p, _ = hot_request(ratio)
    before = c.mem_pool_device.full_to_hisparse_device_index_mapping.clone()
    free = p.free_pages
    c._cleanup_kvduo_old_tails()
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    assert mapping[12:14].tolist() == [0, 0]
    assert torch.equal(mapping[14:], before[14:])
    assert p.free_pages == free + 1
    assert not c._kvduo_tail_cleanup_pending
    c.reclaim_kvduo_full_pages = MagicMock(wraps=c.reclaim_kvduo_full_pages)
    c._cleanup_kvduo_old_tails()
    c.reclaim_kvduo_full_pages.assert_not_called()


@pytest.mark.parametrize("protected", ["host", "shared", "in_use", "partial"])
def test_old_tail_keeps_existing_page_protections(protected):
    c, p, req = hot_request()
    if protected == "host":
        c.full_host_version[12:14] = 0
    elif protected == "shared":
        req.cache_protected_len = 14
    elif protected == "in_use":
        c._kvduo_pressure_protected = torch.tensor([12])
    else:
        c.tail_protected_pages = 0
        req.kv.kv_allocated_len = 13
        c._kvduo_host_valid_len[0] = 12
    before = c.mem_pool_device.full_to_hisparse_device_index_mapping.clone()
    free = p.free_pages
    c._cleanup_kvduo_old_tails()
    assert torch.equal(c.mem_pool_device.full_to_hisparse_device_index_mapping, before)
    assert p.free_pages == free


@pytest.mark.parametrize("pending", ["backup", "rank"])
def test_cleanup_defers_without_waiting_until_all_ranks_ready(pending):
    c, p, _ = hot_request()
    c.reclaim_kvduo_full_pages = MagicMock(wraps=c.reclaim_kvduo_full_pages)
    c.decode_producer_stream = MagicMock()
    c.decode_producer_stream.query.return_value = pending != "producer"
    c._has_pending_backup = True
    c._backup_done_event = MagicMock()
    c._backup_done_event.query.return_value = pending != "backup"
    c._kvduo_consensus = MagicMock(return_value=pending != "rank")
    c._cleanup_kvduo_old_tails()
    c.reclaim_kvduo_full_pages.assert_not_called()
    assert c._kvduo_tail_cleanup_pending
    c._backup_done_event.wait.assert_not_called()


@pytest.mark.parametrize("stale", ["generation", "ended", "full"])
def test_cleanup_never_applies_old_job_to_reused_or_full_request(stale):
    c, p, _ = hot_request()
    c._kvduo_tail_cleanup_pending[0] = c._kvduo_request_generation[0]
    if stale == "generation":
        c._kvduo_request_generation[0] += 1
    elif stale == "ended":
        c._active_kvduo_reqs.clear()
    else:
        c._mixed_slots[0] = False
    c.reclaim_kvduo_full_pages = MagicMock()
    c._cleanup_kvduo_old_tails()
    c.reclaim_kvduo_full_pages.assert_not_called()
    assert not c._kvduo_tail_cleanup_pending


def test_request_reset_clears_cleanup_record():
    c, _, _ = hot_request()
    c._kvduo_tail_retry_pages[0] = {12}
    c._reset_kvduo_prefill_request(0)
    assert not c._kvduo_tail_retry_pages
    assert not c._kvduo_tail_cleanup_pending
    assert not c._kvduo_hot_tail_starts


def test_busy_decode_producer_is_stream_ordered_not_required_to_be_idle(monkeypatch):
    import sglang.srt.managers.hisparse_coordinator as module

    c, p, _ = hot_request()
    c.decode_producer_stream = MagicMock()
    c.decode_producer_stream.query.return_value = False
    device = MagicMock()
    monkeypatch.setattr(module, "device_module", device)
    c._cleanup_kvduo_old_tails()
    device.current_stream.return_value.wait_stream.assert_called_with(
        c.decode_producer_stream
    )
    c.decode_producer_stream.query.assert_not_called()
    assert c.mem_pool_device.full_to_hisparse_device_index_mapping[12:14].tolist() == [
        0,
        0,
    ]


@pytest.mark.parametrize("ratio", [1, 4])
def test_long_output_scans_each_expired_page_once(monkeypatch, ratio):
    import sglang.srt.managers.hisparse_coordinator as module

    c, _, req = make_full_coordinator(pages=108, ratio=ratio)
    req.kv.kv_allocated_len = 8 * c.page_size * ratio
    c.reclaim_kvduo_full_pages(c.page_size, reserve_hot_minimum=True)
    summarize = MagicMock(wraps=module.summarize_kvduo_full_pages)
    monkeypatch.setattr(module, "summarize_kvduo_full_pages", summarize)
    for step in range(100):
        req.kv.kv_allocated_len = (9 + step) * c.page_size * ratio
        c._cleanup_kvduo_old_tails()
    assert sum(call.args[0].size(0) for call in summarize.call_args_list) == 100
    assert c._kvduo_hot_tail_starts[0][1] == 106 * c.page_size
    assert not c._kvduo_tail_retry_pages
    assert not c._kvduo_tail_cleanup_pending


@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("blocked", ["version", "host_extent", "shared", "in_use"])
def test_blocked_page_retries_after_cursor_advances_without_new_turnover(
    ratio, blocked
):
    c, p, req = hot_request(ratio)
    if blocked == "version":
        c.full_host_version[12:14] = 0
    elif blocked == "host_extent":
        c._kvduo_host_valid_len[0] = 12
    elif blocked == "shared":
        req.cache_protected_len = 14 * ratio
    else:
        c._kvduo_pressure_protected = torch.tensor([12])
    free = p.free_pages
    c._cleanup_kvduo_old_tails()
    assert p.free_pages == free
    assert c._kvduo_hot_tail_starts[0][1] == 14
    assert c._kvduo_tail_retry_pages[0] == {12}
    assert c._kvduo_tail_cleanup_pending
    c.full_host_version[12:14] = c.full_data_version[12:14]
    c._kvduo_host_valid_len[0] = 20
    req.cache_protected_len = 0
    c._kvduo_pressure_protected = None
    c._cleanup_kvduo_old_tails()
    assert p.free_pages == free + 1
    assert not c._kvduo_tail_retry_pages
    assert not c._kvduo_tail_cleanup_pending


@pytest.mark.parametrize("ratio", [1, 4])
def test_blocked_hole_does_not_rescan_released_pages_behind_it(monkeypatch, ratio):
    import sglang.srt.managers.hisparse_coordinator as module

    c, p, req = hot_request(ratio)
    c.full_host_version[12:14] = 0
    c._cleanup_kvduo_old_tails()
    req.kv.kv_allocated_len = 10 * c.page_size * ratio
    c._cleanup_kvduo_old_tails()
    assert c._kvduo_hot_tail_starts[0][1] == 16
    assert c.mem_pool_device.full_to_hisparse_device_index_mapping[14:16].tolist() == [
        0,
        0,
    ]
    c.full_host_version[12:14] = 1
    summarize = MagicMock(wraps=module.summarize_kvduo_full_pages)
    monkeypatch.setattr(module, "summarize_kvduo_full_pages", summarize)
    c._cleanup_kvduo_old_tails()
    assert summarize.call_args.args[0].tolist() == [[12, 13]]
    assert not c._kvduo_tail_retry_pages


def test_absent_retry_page_is_retired_without_double_free():
    c, p, _ = hot_request()
    c.full_host_version[12:14] = 0
    c._cleanup_kvduo_old_tails()
    c.full_host_version[12:14] = 1
    c.reclaim_kvduo_full_pages(c.page_size, release_all=True)
    free = p.free_pages
    c._cleanup_kvduo_old_tails()
    assert p.free_pages == free
    assert not c._kvduo_tail_retry_pages
    assert not c._kvduo_tail_cleanup_pending


def test_failed_release_does_not_publish_advanced_cursor():
    c, _, _ = hot_request()
    before = dict(c._kvduo_hot_tail_starts)
    c.token_to_kv_pool_allocator.free_hisparse_indices.side_effect = RuntimeError(
        "release failed"
    )
    with pytest.raises(RuntimeError, match="release failed"):
        c._cleanup_kvduo_old_tails()
    assert c._kvduo_hot_tail_starts == before
    assert c._kvduo_tail_cleanup_pending


def test_reclaim_consensus_failure_does_not_advance_progress():
    c, p, _ = hot_request()
    before = dict(c._kvduo_hot_tail_starts)
    c._kvduo_consensus = MagicMock(side_effect=[True, False])
    free = p.free_pages
    c._cleanup_kvduo_old_tails()
    assert c._kvduo_hot_tail_starts == before
    assert c._kvduo_tail_cleanup_pending
    assert p.free_pages == free


def test_old_job_does_not_clear_new_generation_cursor_or_retries():
    c, _, _ = hot_request()
    c._kvduo_request_generation[0] += 1
    generation = c._kvduo_request_generation[0]
    c._kvduo_hot_tail_starts[0] = (generation, 18)
    c._kvduo_tail_retry_pages[0] = {16}
    c._cleanup_kvduo_old_tails()
    assert c._kvduo_hot_tail_starts[0] == (generation, 18)
    assert c._kvduo_tail_retry_pages[0] == {16}


@pytest.mark.parametrize("ratio", [1, 4])
def test_tail_cleanup_discovers_turnover_without_new_backup(ratio, caplog):
    c, p, req = hot_request(ratio)
    c._has_pending_backup = False
    free = p.free_pages
    with caplog.at_level("INFO"):
        c._cleanup_kvduo_old_tails()
        assert p.free_pages == free + 1
        req.kv.kv_allocated_len += c.page_size * ratio
        c._cleanup_kvduo_old_tails()
        assert p.free_pages == free + 2
    assert "FULL request reclaim completed" not in caplog.text
    assert (
        c.mem_pool_device.full_to_hisparse_device_index_mapping[12:16].tolist()
        == [0] * 4
    )


@pytest.mark.parametrize("ratio", [1, 4])
def test_decode_without_pressure_does_not_clean_expired_tail(ratio):
    c, p, req = hot_request(ratio)
    c._cleanup_kvduo_old_tails()
    free = p.free_pages
    req.kv.kv_allocated_len += c.page_size * ratio

    # Ordinary replay must not poll Tail cleanup before processing growth.
    class ReachedGrowth(Exception):
        pass

    def consume():
        assert p.free_pages == free
        raise ReachedGrowth

    c._consume_kvduo_hot_growth_events = consume
    with pytest.raises(ReachedGrowth):
        c.prepare_kvduo_graph_replay(torch.tensor([0]))


@pytest.mark.parametrize("path", ["prefill", "decode"])
@pytest.mark.parametrize("ratio", [1, 4])
def test_pressure_uses_old_tail_before_any_other_reclaim(path, ratio):
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.mem_cache.sparsity.core.kvduo_state import KVDuoPressureAction

    c, p, _ = hot_request(ratio)
    c._kvduo_pending_hot_minimum.clear()
    p.free_pages = 0
    c.reclaim_kvduo_full_pages = MagicMock(wraps=c.reclaim_kvduo_full_pages)
    if path == "prefill":
        assert c.reserve_kvduo_prefill(1) is not None
    else:
        result = HiSparseCoordinator._reclaim_for_physical_allocation(c, c.page_size)
        assert result.action is KVDuoPressureAction.SUCCESS
    assert p.free_pages == 1
    c.reclaim_kvduo_full_pages.assert_called_once()
    assert c.reclaim_kvduo_full_pages.call_args.kwargs["tail_cleanup"] is True
    c.reclaim_kvduo_hot_pages.assert_not_called()


def test_prefill_pressure_preserves_selected_prefix_and_does_not_merge():
    c, p, _ = hot_request()
    c._kvduo_pending_hot_minimum.clear()
    p.free_pages = 0
    c.reclaim_kvduo_full_pages = MagicMock(wraps=c.reclaim_kvduo_full_pages)
    assert c.reserve_kvduo_prefill(1, prefix_indices=torch.tensor([12, 13])) is None
    assert p.free_pages == 0
    assert c._kvduo_tail_retry_pages[0] == {12}
    assert all(
        call.kwargs.get("tail_cleanup")
        for call in c.reclaim_kvduo_full_pages.call_args_list
    )
    c.reclaim_kvduo_hot_pages.assert_not_called()


def test_sufficient_prefill_space_and_schedule_boundary_do_not_clean_tail():
    c, p, _ = hot_request()
    free = p.free_pages
    c.reclaim_kvduo_full_pages = MagicMock(wraps=c.reclaim_kvduo_full_pages)
    c.begin_kvduo_schedule_boundary()
    assert c.reserve_kvduo_prefill(1) is not None
    assert p.free_pages == free
    c.reclaim_kvduo_full_pages.assert_not_called()


def test_optional_growth_uses_tail_space_before_full_or_merge():
    from test_kvduo_hot_merge import make_merge_coordinator

    c, p = make_merge_coordinator({(0, 0): 2})

    def cleanup():
        p.free_pages += 1
        return c.page_size

    c._cleanup_kvduo_old_tails = MagicMock(side_effect=cleanup)
    c.reclaim_kvduo_hot_pages = MagicMock(
        side_effect=AssertionError("unexpected merge")
    )
    c.kvduo_hot_growth_events[0, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    c._cleanup_kvduo_old_tails.assert_called_once()
    c.reclaim_kvduo_full_pages.assert_not_called()
    c.reclaim_kvduo_hot_pages.assert_not_called()
    assert c.kvduo_req_hot_capacity[0, 0] == 12
