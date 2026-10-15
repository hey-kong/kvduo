"""Sealed Tail lifecycle after FULL -> HOT residency transition."""

from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.mem_cache.sparsity.core.kvduo_state import KVDuoResidencyMode
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_pressure_admission import make_full_coordinator

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def hot_request(ratio=1):
    c, p, req = make_full_coordinator(
        pages=10, ratio=ratio, full_reclaim_policy="request"
    )
    req.kv.kv_allocated_len = 8 * c.page_size * ratio
    assert c.kvduo_residency_mode(0) is KVDuoResidencyMode.FULL_RESIDENT
    c.reclaim_kvduo_full_pages(c.page_size, reserve_hot_minimum=True)
    assert c.kvduo_residency_mode(0) is KVDuoResidencyMode.HOT_RESIDENT
    assert c._kvduo_hot_tail_starts[0][1] == 6 * c.page_size
    req.kv.kv_allocated_len = 9 * c.page_size * ratio
    c._kvduo_tail_cleanup_pending[0] = c._kvduo_request_generation[0]
    return c, p, req


@pytest.mark.parametrize("ratio", [1, 4])
def test_old_tail_released_without_pressure_current_tail_preserved(ratio):
    c, p, _ = hot_request(ratio)
    before = c.mem_pool_device.full_to_hisparse_device_index_mapping.clone()
    free = p.free_pages
    c.begin_kvduo_schedule_boundary()
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    assert mapping[12:14].tolist() == [0, 0]
    assert torch.equal(mapping[14:], before[14:])
    assert p.free_pages == free + 1
    assert not c._kvduo_tail_cleanup_pending
    c.reclaim_kvduo_full_pages = MagicMock(wraps=c.reclaim_kvduo_full_pages)
    c.begin_kvduo_schedule_boundary()
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
    c.begin_kvduo_schedule_boundary()
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
    c.begin_kvduo_schedule_boundary()
    c.reclaim_kvduo_full_pages.assert_not_called()
    assert c._kvduo_tail_cleanup_pending
    c._backup_done_event.wait.assert_not_called()


@pytest.mark.parametrize("stale", ["generation", "ended", "full"])
def test_cleanup_never_applies_old_job_to_reused_or_full_request(stale):
    c, p, _ = hot_request()
    if stale == "generation":
        c._kvduo_request_generation[0] += 1
    elif stale == "ended":
        c._active_kvduo_reqs.clear()
    else:
        c._mixed_slots[0] = False
    c.reclaim_kvduo_full_pages = MagicMock()
    c.begin_kvduo_schedule_boundary()
    c.reclaim_kvduo_full_pages.assert_not_called()
    assert not c._kvduo_tail_cleanup_pending


def test_request_reset_clears_cleanup_record():
    c, _, _ = hot_request()
    c._reset_kvduo_prefill_request(0)
    assert not c._kvduo_tail_cleanup_pending
    assert not c._kvduo_hot_tail_starts


def test_busy_decode_producer_is_stream_ordered_not_required_to_be_idle(monkeypatch):
    import sglang.srt.managers.hisparse_coordinator as module

    c, p, _ = hot_request()
    c.decode_producer_stream = MagicMock()
    c.decode_producer_stream.query.return_value = False
    device = MagicMock()
    monkeypatch.setattr(module, "device_module", device)
    c.begin_kvduo_schedule_boundary()
    device.current_stream.return_value.wait_stream.assert_called_with(
        c.decode_producer_stream
    )
    c.decode_producer_stream.query.assert_not_called()
    assert c.mem_pool_device.full_to_hisparse_device_index_mapping[12:14].tolist() == [
        0,
        0,
    ]
