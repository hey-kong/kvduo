"""Host restore uses its admitted quota without bypassing HOT rollback policy."""

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache, InitLoadBackParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.sparsity.core.kvduo_prefix_cache import (
    HostPrefixRecord,
    KVDuoHostPrefixCache,
    KVDuoPrefixPageView,
    plan_kvduo_prefix_restore,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_hot_growth import make_host_pool
from test_kvduo_prefill_rollback import adder_and_req, ready_coordinator

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def restore_setup(*, dsv4, requests=1, cache_type=RadixCache, extra=2):
    c, physical = ready_coordinator(extra=extra, requests=requests)
    physical.free = c.token_to_kv_pool_allocator.free_hisparse_indices.side_effect
    adder, req = adder_and_req(c, dsv4=dsv4, chunk=2 * (4 if dsv4 else 1))
    c.token_to_kv_pool_allocator = adder.token_to_kv_pool_allocator
    c.is_dsv4_hisparse = dsv4
    c.tp_world_size = 1
    c.item_size_bytes = 4
    c._reclaim_for_physical_allocation = MethodType(
        HiSparseCoordinator._reclaim_for_physical_allocation, c
    )
    for name in ("full_generation", "full_data_version", "full_last_touch"):
        setattr(c, name, torch.zeros(2048, dtype=torch.int64))
    c.full_host_version = torch.full((2048,), -1, dtype=torch.int64)
    c.host_prefix_cache = KVDuoHostPrefixCache(8)
    c.mem_pool_host = make_host_pool(8, c.page_size)
    c.mem_pool_host.load_to_device_per_layer = MagicMock()
    cached_locations = tuple(c.mem_pool_host.alloc(c.page_size).tolist())
    c.mem_pool_device.translate_loc_from_full_to_compressed = (
        lambda indices: indices[c.compress_ratio - 1 :: c.compress_ratio]
        // c.compress_ratio
    )
    c.mem_pool_device.translate_loc_from_full_to_hisparse_device = (
        c.mem_pool_device.translate_loc_from_full_to_compressed
    )

    page_tokens = adder.page_size
    req.rid = "host-restore"
    req.extra_key = None
    req.prefix_indices = torch.arange(page_tokens, dtype=torch.int64)
    req.cache_protected_len = page_tokens
    req.host_hit_length = page_tokens
    req.needs_host_load_back = lambda: req.host_hit_length > 0
    req.full_untruncated_fill_ids = list(range(3 * page_tokens))
    req.get_fill_ids = lambda: req.full_untruncated_fill_ids
    identity = c._host_prefix_identity(req, 1)
    c.host_prefix_cache.insert(
        HostPrefixRecord(
            identity=identity,
            domains=("main_kv",),
            host_locations={"main_kv": cached_locations},
            data_versions={"main_kv": (1, 1)},
            host_versions={"main_kv": (1, 1)},
            request_references={req.rid},
        )
    )
    req.kvduo_residency_plan = plan_kvduo_prefix_restore(
        [KVDuoPrefixPageView(identity, ("main_kv",))],
        c.host_prefix_cache,
        {"main_kv": c._physical_page_bytes()},
    )

    # Budget, admission, cache delegation, restore, and pressure use production
    # methods. Only device allocation/H2D are replaced with CPU page accounting.
    def allocate(prefix, prefix_cpu, target, target_cpu, last_loc, num_tokens):
        full = c.token_to_kv_pool_allocator.logical_attn_allocator
        full = full.full_attn_allocator if dsv4 else full
        indices = full.alloc(num_tokens)
        assert indices is not None
        assert physical.alloc(c.page_size) is not None
        return indices

    c.token_to_kv_pool_allocator.alloc_kvduo_restore = MagicMock(side_effect=allocate)
    c.token_to_kv_pool_allocator.rollback_restore_allocation = MagicMock()
    adder.tree_cache.kvduo_coordinator = c
    adder.tree_cache.init_load_back.side_effect = (
        lambda params: cache_type.init_load_back(adder.tree_cache, params)
    )
    return c, physical, adder, req


def restore_params(req, token):
    return InitLoadBackParams(
        best_match_node=req.best_match_node,
        host_hit_length=req.host_hit_length,
        req=req,
        kvduo_prefill_reservation=token,
    )


@pytest.mark.parametrize("dsv4", [False, True])
@pytest.mark.parametrize("cache_type", [BasePrefixCache, RadixCache, UnifiedRadixCache])
def test_real_prefill_reservation_funds_restore_and_keeps_fresh_tail_quota(
    dsv4, cache_type
):
    c, physical, adder, req = restore_setup(dsv4=dsv4, cache_type=cache_type, extra=3)
    assert physical.free_pages == 0
    assert adder.token_to_kv_pool_allocator.available_size() == 0

    assert adder.add_one_req(req, False, None) is AddReqResult.CONTINUE

    assert adder.can_run_list == [req]
    assert req.extend_range == Range(2 * adder.page_size, 3 * adder.page_size)
    assert physical.free_pages == 2
    assert c._kvduo_reserved_prefill_pages() == 1
    assert c._kvduo_mandatory_hot_pages() == 1
    assert c._kvduo_prefill_budget_cost == {(0, 0, 0): 1}
    assert c._kvduo_object_limited((0, 0))
    c.token_to_kv_pool_allocator.alloc_kvduo_restore.assert_called_once()
    c.mem_pool_host.load_to_device_per_layer.assert_called_once()
    c.reclaim_kvduo_full_pages.assert_called_once()  # Initial admission only.
    assert all(
        record.restore_pins == 0 for record in c.host_prefix_cache.records.values()
    )


@pytest.mark.parametrize("dsv4", [False, True])
@pytest.mark.parametrize("blocked_by", ["host", "logical", "physical"])
def test_waiting_chunk_protects_host_restore_and_fresh_tail(dsv4, blocked_by):
    c, physical, adder, req = restore_setup(dsv4=dsv4)
    reserve = MagicMock(wraps=c.reserve_kvduo_prefill)
    c.reserve_kvduo_prefill = reserve
    if blocked_by == "host":
        assert c.mem_pool_host.alloc(c.mem_pool_host.available_size()) is not None
    elif blocked_by == "logical":
        logical = adder.token_to_kv_pool_allocator.logical_attn_allocator
        logical = logical.full_attn_allocator if dsv4 else logical
        assert logical.alloc(logical.available_size()) is not None
    else:
        c.kvduo_prefill_steps.zero_()

    result = adder.add_one_req(req, False, None)

    assert result is (
        AddReqResult.NO_TOKEN if blocked_by == "logical" else AddReqResult.OTHER
    )
    assert c._kvduo_prefill_plan_uncovered_pages() == 2
    if blocked_by == "logical":
        reserve.assert_not_called()
    else:
        assert reserve.call_args.args[0] == 2
    assert physical.free_pages == 0
    assert not adder.can_run_list and not c._kvduo_prefill_limited_objects
    c.token_to_kv_pool_allocator.alloc_kvduo_restore.assert_not_called()


@pytest.mark.parametrize("dsv4", [False, True])
def test_waiting_plan_uses_the_aligned_chunk_end(dsv4):
    c, physical, adder, req = restore_setup(dsv4=dsv4)
    page = adder.page_size
    req.full_untruncated_fill_ids = list(range(6 * page))
    adder.rem_chunk_tokens = 3 * page + 1
    reserve = MagicMock(wraps=c.reserve_kvduo_prefill)
    c.reserve_kvduo_prefill = reserve

    assert adder.add_one_req(req, False, 2 * page) is AddReqResult.OTHER

    # One restored page and two freshly computed pages fit this execution.
    assert reserve.call_args.args[0] == 3
    assert c._kvduo_prefill_plan_uncovered_pages() == 3
    assert physical.free_pages == 0 and not adder.can_run_list
    c.token_to_kv_pool_allocator.alloc_kvduo_restore.assert_not_called()


def test_swa_limited_plan_updates_protection_before_host_admission():
    c, physical, adder, req = restore_setup(dsv4=True)
    page = adder.page_size
    req.full_untruncated_fill_ids = list(range(6 * page))
    adder.rem_chunk_tokens = 3 * page
    # Leave SWA room for one compute page plus the allocator's page slack.
    swa = adder.token_to_kv_pool_allocator.logical_attn_allocator.swa_attn_allocator
    assert swa.alloc(swa.available_size() - 2 * page) is not None
    reserve = MagicMock(wraps=c.reserve_kvduo_prefill)
    c.reserve_kvduo_prefill = reserve

    assert adder.add_one_req(req, False, None) is AddReqResult.OTHER

    assert reserve.call_args.args[0] == 2
    assert c._kvduo_prefill_plan_uncovered_pages() == 2
    assert physical.free_pages == 0 and not adder.can_run_list


@pytest.mark.parametrize("dsv4", [False, True])
@pytest.mark.parametrize("protected", ["fresh_tail", "other_prefill", "hot_minimum"])
def test_restore_cannot_spend_other_pending_allocations(dsv4, protected):
    c, physical, adder, req = restore_setup(dsv4=dsv4, requests=2)
    token = c.reserve_kvduo_prefill(2, [0])
    assert token is not None and physical.free_pages == 2
    if protected == "fresh_tail":
        physical.free_pages = 1  # Restore fits alone, but its tail no longer does.
    elif protected == "other_prefill":
        physical.free_pages += 1
        other = c.reserve_kvduo_prefill(1, [])
        assert other is not None
        physical.free_pages = 2  # Restore+tail fit, but not the other reservation.
    else:
        c._release_kvduo_layer_hot_pages(1)
        c._kvduo_pending_hot_minimum.add(1)
        # Restore+tail fit, but the mandatory HOT page does not.
        physical.free_pages = 2
    reserved = c._kvduo_reserved_prefill_pages()
    c.reclaim_kvduo_hot_pages = MagicMock(wraps=c.reclaim_kvduo_hot_pages)

    restored, _ = adder.tree_cache.init_load_back(restore_params(req, token))

    assert restored is None
    assert c._kvduo_reserved_prefill_pages() == reserved
    c.token_to_kv_pool_allocator.alloc_kvduo_restore.assert_not_called()
    c.reclaim_kvduo_hot_pages.assert_not_called()
    assert all(
        record.restore_pins == 0 for record in c.host_prefix_cache.records.values()
    )


@pytest.mark.parametrize("dsv4", [False, True])
@pytest.mark.parametrize("pass_reservation", [False, True])
def test_restore_shortfall_never_uses_unbudgeted_hand_merge(dsv4, pass_reservation):
    c, physical, adder, req = restore_setup(dsv4=dsv4, requests=2)
    physical.free_pages = 2
    c._ensure_kvduo_hot_capacity_targets_batch({(0, 1): 12})
    assert physical.free_pages == 0
    assert c.kvduo_prefill_steps[0, 1].item() == 0
    token = c.reserve_kvduo_prefill(2, [0])
    assert token is not None
    # An unrelated mandatory allocation consumes a page after admission. The
    # unobserved request can be shrunk by Hand, but cannot fund this prefill.
    physical.free_pages -= 1
    tokens_before = c.req_device_buffer_tokens.clone()
    hand_before = c._kvduo_hand_sequence
    c.reclaim_kvduo_hot_pages = MagicMock(wraps=c.reclaim_kvduo_hot_pages)

    restored, _ = adder.tree_cache.init_load_back(
        restore_params(req, token if pass_reservation else None)
    )

    assert restored is None and physical.free_pages == 1
    assert c.kvduo_req_hot_capacity[0, 1].item() == 12
    assert torch.equal(c.req_device_buffer_tokens, tokens_before)
    assert c._kvduo_hand_sequence == hand_before
    assert c._kvduo_prefill_budget_cost == {(0, 0, 0): 1}
    c.reclaim_kvduo_hot_pages.assert_not_called()
    c.token_to_kv_pool_allocator.alloc_kvduo_restore.assert_not_called()


@pytest.mark.parametrize("invalid", ["cancelled", "insufficient"])
def test_restore_rejects_invalid_reservation_without_reclaim(invalid):
    c, physical, adder, req = restore_setup(dsv4=True)
    token = c.reserve_kvduo_prefill(2, [0])
    assert token is not None
    if invalid == "cancelled":
        c.finish_kvduo_prefill_reservation(token, cancelled=True)
    else:
        c.consume_kvduo_prefill_reservation(token, 2)
    c.reclaim_kvduo_full_pages.reset_mock()

    restored, _ = adder.tree_cache.init_load_back(restore_params(req, token))

    assert restored is None and physical.free_pages == 2
    c.reclaim_kvduo_full_pages.assert_not_called()
    c.token_to_kv_pool_allocator.alloc_kvduo_restore.assert_not_called()


def test_failed_restore_cancels_prefill_quota_and_limits_without_spending_pages():
    c, physical, adder, req = restore_setup(dsv4=True, extra=3)
    c.token_to_kv_pool_allocator.alloc_kvduo_restore.return_value = None
    c.token_to_kv_pool_allocator.alloc_kvduo_restore.side_effect = None

    assert adder.add_one_req(req, False, None) is AddReqResult.OTHER

    c.token_to_kv_pool_allocator.alloc_kvduo_restore.assert_called_once()
    assert physical.free_pages == 3
    assert not c._kvduo_prefill_reservations
    assert not c._kvduo_prefill_limited_objects
    assert not c._kvduo_prefill_budget_cost
    assert not adder.can_run_list
    assert len(req.prefix_indices) == adder.page_size
    assert all(
        record.restore_pins == 0 for record in c.host_prefix_cache.records.values()
    )
