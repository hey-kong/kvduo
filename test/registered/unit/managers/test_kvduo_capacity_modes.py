"""Exercise FULL admission and independent optional-growth budgets."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.schedule_policy import AddReqResult
from test_kvduo_hot_growth import make_coordinator, make_host_pool
from test_kvduo_prefill_admission import (
    adder_and_req,
    ready_coordinator,
)
from test_kvduo_pressure_admission import make_full_coordinator


@pytest.mark.parametrize("dsv4", [False, True])
def test_waiting_host_prefix_chunk_defers_optional_growth_without_merge(dsv4):
    c, p = ready_coordinator(extra=3)
    adder, req = adder_and_req(c, dsv4=dsv4)
    page = adder.page_size
    adder.rem_chunk_tokens = page
    req.rid = "waiting-prefill"
    req.extra_key = None
    req.prefix_indices = torch.arange(page, dtype=torch.int64)
    req.host_hit_length = page
    req.full_untruncated_fill_ids = list(range(4 * page))
    c.mem_pool_host = make_host_pool(0, c.page_size)
    p.free_pages = 1
    before = c.kvduo_req_hot_capacity.clone()
    c.reclaim_kvduo_hot_pages = MagicMock(wraps=c.reclaim_kvduo_hot_pages)

    assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    assert c._kvduo_optional_protected_pages() == 2
    c.kvduo_hot_growth_events[0, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))

    assert torch.equal(c.kvduo_req_hot_capacity, before)
    assert p.free_pages == 1
    c.reclaim_kvduo_hot_pages.assert_not_called()
    c.reclaim_kvduo_full_pages.assert_not_called()


@pytest.mark.parametrize("ratio", [1, 4])
def test_prefill_completion_and_schedule_boundary_keep_full_residency(ratio):
    c, p, req = make_full_coordinator(pages=8, initial_pages=8, ratio=ratio)
    req.rid = "new"
    req.decode_batch_idx = 0
    mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
    before = mapping.clone()
    c.begin_kvduo_schedule_boundary()
    assert torch.equal(mapping, before)
    assert p.free_pages == 0 and not c._mixed_slots[0]
    assert req.decode_batch_idx == 0
    assert c.reclaim_kvduo_full_pages(c.page_size, reserve_hot_minimum=True) > 0
    assert c._mixed_slots[0]


def test_prefill_reserves_full_pages_without_future_mixed_minimum():
    c, p = ready_coordinator(extra=4)
    p.free_pages = 4
    req = SimpleNamespace(rid="new", prefix_indices=torch.empty(0, dtype=torch.int64))
    # Four free physical pages fit four FULL compute pages.
    token = c.reserve_kvduo_prefill(4, prefill_req=req)
    assert token is not None and c._kvduo_mandatory_hot_pages() == 0
    assert c._kvduo_optional_protected_pages() == 4
    c.finish_kvduo_prefill_reservation(token, cancelled=True)
    assert c._kvduo_optional_protected_pages() == 0


def test_plan_protection_does_not_double_count_reservation():
    c, p = ready_coordinator(extra=4)
    p.free_pages = 4
    req = SimpleNamespace(rid="new", prefix_indices=torch.empty(0, dtype=torch.int64))
    c.protect_kvduo_prefill_plan(req, 6)  # three compute pages
    token = c.reserve_kvduo_prefill(3, prefill_req=req)
    assert token is not None
    assert c._kvduo_optional_protected_pages() == 3
    c.kvduo_prefill_batch_allocated()
    c.consume_kvduo_prefill_reservation(token, 3)
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    assert c._kvduo_optional_protected_pages() == 0


@pytest.mark.parametrize("policy", ["page", "request"])
@pytest.mark.parametrize("donor_pages", [4, 8])
@pytest.mark.parametrize("reserved", [0, 2])
def test_optional_growth_reclaims_predecode_full_and_funds_donor_minima(
    policy, donor_pages, reserved
):
    from types import MethodType
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

    c, p = make_coordinator(layers=2, requests=2, free_pages=2)
    c._mixed_slots[1] = False
    c._kvduo_pending_hot_minimum = {0}
    c.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    assert p.free_pages == 0
    p.free_pages = reserved
    c._kvduo_next_prefill_pages = reserved
    full, _, donor = make_full_coordinator(pages=donor_pages, layers=2)
    donor.req_pool_idx = 1
    donor.decode_batch_idx = 0
    c._active_kvduo_reqs[1] = donor
    c.req_to_token_pool = SimpleNamespace(
        req_to_token=full.req_to_token_pool.req_to_token.repeat(2, 1)
    )
    for name in (
        "full_generation",
        "full_data_version",
        "full_host_version",
        "full_last_touch",
        "full_touch_clock",
    ):
        setattr(c, name, getattr(full, name))
    c.mem_pool_device.full_to_hisparse_device_index_mapping = (
        full.mem_pool_device.full_to_hisparse_device_index_mapping + 100
    )
    c.tail_protected_pages = 2
    c._kvduo_host_valid_len = [0, donor_pages * c.page_size]
    c.full_reclaim_policy = policy
    c.reclaim_kvduo_full_pages = MethodType(
        HiSparseCoordinator.reclaim_kvduo_full_pages, c
    )
    c.reclaim_kvduo_hot_pages = MagicMock(return_value=0)
    c.wait_for_pending_backup = MagicMock()
    c.item_size_bytes = 1
    c._reclaim_for_physical_allocation = MethodType(
        HiSparseCoordinator._reclaim_for_physical_allocation, c
    )
    c._require_allocation_ready = HiSparseCoordinator._require_allocation_ready
    c._active_kvduo_reqs[0] = SimpleNamespace(
        hisparse_staging=False,
        cache_protected_len=0,
        kv=SimpleNamespace(kv_allocated_len=0),
    )
    before = c.mem_pool_device.full_to_hisparse_device_index_mapping.clone()
    c.kvduo_hot_growth_events[0, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    if donor_pages == 4:
        assert torch.equal(
            c.mem_pool_device.full_to_hisparse_device_index_mapping, before
        )
        assert c.kvduo_req_hot_capacity.tolist() == [[4, 0], [4, 0]]
        c.reclaim_kvduo_hot_pages.assert_called_once_with(c.page_size)
    else:
        assert c.kvduo_req_hot_capacity.tolist() == [[8, 4], [4, 4]]
        assert not c._kvduo_pending_hot_minimum
        assert c._mixed_slots[1]
        assert donor.decode_batch_idx == 0
        assert torch.equal(
            c.mem_pool_device.full_to_hisparse_device_index_mapping[-4:], before[-4:]
        )
        c.reclaim_kvduo_hot_pages.assert_not_called()
        assert p.free_pages >= reserved
