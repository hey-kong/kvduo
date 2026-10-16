from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.sparsity import parse_kvduo_config
from sglang.srt.mem_cache.sparsity.core.kvduo_state import KVDuoPressureAction
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_hot_merge import (
    assert_payloads_match_tokens,
    make_merge_coordinator,
)
from test_kvduo_pressure_admission import make_cache

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def pressure_batch(enabled, *, pages=4):
    c, physical = make_merge_coordinator({(0, 0): pages})
    c.decode_emergency_reclaim = enabled
    c.kvduo_hot_buffers[0, 0].frequency.zero_()
    logical = SimpleNamespace(
        available_size=lambda: 64,
        full_available_size=lambda: 64,
        swa_available_size=lambda: 64,
    )
    allocator = c.token_to_kv_pool_allocator
    allocator.logical_attn_allocator = logical
    allocator.page_size = c.page_size
    allocator.available_size = lambda: min(
        logical.available_size(), physical.available_size()
    )
    c.item_size_bytes = 1
    c._reclaim_for_physical_allocation = MethodType(
        HiSparseCoordinator._reclaim_for_physical_allocation, c
    )
    req = SimpleNamespace(req_pool_idx=0, kv_committed_len=2, hisparse_staging=False)
    c._active_kvduo_reqs[0] = req
    batch = ScheduleBatch(reqs=[req])
    batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    batch.token_to_kv_pool_allocator = allocator
    batch.hisparse_coordinator = c
    batch.tree_cache = make_cache(allocator)
    return batch, c, physical


@pytest.mark.parametrize(
    "raw,enabled",
    [
        (None, False),
        ("{}", False),
        (
            '{"top_k":512,"host_size":0,"swap_in_block_size":960}',
            False,
        ),
        ('{"decode_emergency_reclaim":false}', False),
        ('{"decode_emergency_reclaim":true}', True),
    ],
)
def test_default_config_disables_emergency_decode_recovery_and_allows_opt_in(
    raw, enabled
):
    config = parse_kvduo_config(SimpleNamespace(kvduo_config=raw))
    batch, c, physical = pressure_batch(config.decode_emergency_reclaim)
    assert batch.check_decode_mem()
    assert physical.free_pages == 3
    assert c.kvduo_req_hot_capacity[0, 0] == 4


def test_emergency_reclaim_is_not_used_when_normal_budgeted_merge_succeeds():
    batch, c, _ = pressure_batch(True)
    with patch.object(
        c,
        "_reclaim_kvduo_emergency_hot_pages",
        wraps=c._reclaim_kvduo_emergency_hot_pages,
    ) as emergency:
        assert batch.check_decode_mem()
    emergency.assert_not_called()


def test_full_reclaim_still_precedes_all_hot_reclaim():
    batch, c, physical = pressure_batch(True)

    def release_full(*args, **kwargs):
        physical.free_pages += 1
        return c.page_size

    c.reclaim_kvduo_full_pages.side_effect = release_full
    with (
        patch.object(c, "reclaim_kvduo_hot_pages") as normal,
        patch.object(c, "_reclaim_kvduo_emergency_hot_pages") as emergency,
    ):
        assert batch.check_decode_mem()
    c.reclaim_kvduo_full_pages.assert_called_once_with(2, reserve_hot_minimum=True)
    normal.assert_not_called()
    emergency.assert_not_called()
    assert c.kvduo_req_hot_capacity[0, 0] == 16


@pytest.mark.parametrize("enabled", [False, True])
def test_imminent_decode_allocator_guard_uses_the_same_emergency_option(enabled):
    _, c, physical = pressure_batch(enabled)
    assert c._guard_kvduo_physical_allocation(2)
    assert physical.free_pages == 3


def test_emergency_is_not_used_for_prefill_protection_without_decode_allocation():
    batch, c, physical = pressure_batch(True)
    batch.reqs[0].kv_committed_len = 1
    c._kvduo_prefill_reservations[0] = [3, set()]
    with patch.object(
        c,
        "_reclaim_kvduo_emergency_hot_pages",
        wraps=c._reclaim_kvduo_emergency_hot_pages,
    ) as emergency:
        assert batch.check_decode_mem()
    emergency.assert_not_called()
    assert physical.free_pages == 0


@pytest.mark.parametrize("allow_hot", [False, True])
def test_prefill_and_ordinary_allocation_do_not_implicitly_use_decode_emergency(
    allow_hot,
):
    _, c, physical = pressure_batch(True)
    result = c._reclaim_for_physical_allocation(
        2, reserve_hot_minimum=True, allow_hot_reclaim=allow_hot
    )
    assert result.action is (
        KVDuoPressureAction.SUCCESS if allow_hot else KVDuoPressureAction.ERROR
    )
    assert physical.free_pages == (3 if allow_hot else 0)


def test_emergency_cannot_override_no_hot_reclaim_or_spend_prefill_reservation():
    batch, c, physical = pressure_batch(True)
    result = c._reclaim_for_physical_allocation(
        2,
        reserve_hot_minimum=True,
        allow_hot_reclaim=False,
        allow_emergency_hot_reclaim=True,
    )
    assert result.action is KVDuoPressureAction.ERROR and physical.free_pages == 0
    c._kvduo_prefill_reservations[0] = [3, set()]
    assert not batch.check_decode_mem()
    # All three released pages are reserved; decode cannot consume them.
    assert physical.free_pages == 3
    assert c._kvduo_reserved_prefill_pages() == 3


@pytest.mark.parametrize(
    "protection", ["moving", "staging", "finished", "generation", "minimum"]
)
def test_emergency_keeps_all_existing_protection_and_minimum_rules(protection):
    _, c, physical = pressure_batch(True, pages=1 if protection == "minimum" else 4)
    if protection == "moving":
        c._kvduo_hot_pressure_protected.add((0, 0, 0))
    elif protection == "staging":
        c._active_kvduo_reqs[0].hisparse_staging = True
    elif protection == "finished":
        c._active_kvduo_reqs[0].finished = lambda: True
    elif protection == "generation":
        c._kvduo_hot_generation[0, 0] -= 1
    before = c.req_device_buffer_tokens.clone()
    assert c._reclaim_kvduo_emergency_hot_pages(2) == 0
    assert physical.free_pages == 0
    assert torch.equal(c.req_device_buffer_tokens, before)


def test_emergency_prefers_larger_releases_and_stops_without_shrinking_other_objects():
    c, physical = make_merge_coordinator({(0, 0): 8, (0, 1): 3, (0, 2): 4})
    assert c._reclaim_kvduo_emergency_hot_pages(2) == 12
    assert c.kvduo_req_hot_capacity[0].tolist() == [8, 12, 16]
    assert physical.free_pages == 6


def test_emergency_preserves_a_multi_page_topk_minimum():
    c, physical = make_merge_coordinator({(0, 0): 4}, top_k=6)
    expected = c.kvduo_hot_buffers[0, 0].tokens[:8].tolist()
    assert c._reclaim_kvduo_emergency_hot_pages(2) == 4
    assert physical.free_pages == 2
    assert c.kvduo_req_hot_capacity[0, 0] == 8
    assert c.kvduo_hot_buffers[0, 0].tokens.tolist() == expected
    assert_payloads_match_tokens(c, 0, 0)


def test_equal_release_counts_follow_cyclic_hand_order():
    c, physical = make_merge_coordinator({(0, 0): 4, (0, 1): 4, (0, 2): 4})
    c._kvduo_hand = 2
    assert c._reclaim_kvduo_emergency_hot_pages(2) == 4
    assert physical.free_pages == 2
    assert c.kvduo_req_hot_capacity[0].tolist() == [16, 16, 8]
    assert c._kvduo_hand == 0


def test_emergency_refuses_rank_disagreement_before_mutating_any_metadata():
    _, c, physical = pressure_batch(True)
    c._kvduo_consensus = MagicMock(return_value=False)
    before = c.req_device_buffer_tokens.clone()
    assert c._reclaim_kvduo_emergency_hot_pages(2) == 0
    assert physical.free_pages == 0
    assert torch.equal(c.req_device_buffer_tokens, before)


@pytest.mark.parametrize("resource", ["logical", "swa", "host"])
def test_independent_pool_shortages_are_reported_and_never_trigger_emergency(
    resource, caplog
):
    batch, c, physical = pressure_batch(True)
    logical = c.token_to_kv_pool_allocator.logical_attn_allocator
    if resource == "host":
        c.mem_pool_host.available_size = MagicMock(return_value=0)
    else:
        logical.available_size = lambda: 0
        setattr(
            logical,
            "full_available_size" if resource == "logical" else "swa_available_size",
            lambda: 0,
        )
    with patch.object(
        c,
        "_reclaim_kvduo_emergency_hot_pages",
        wraps=c._reclaim_kvduo_emergency_hot_pages,
    ) as emergency:
        assert not batch.check_decode_mem()
    emergency.assert_not_called()
    assert physical.free_pages == 0
    assert "decode admission failed before request retraction" in caplog.text
    assert "host_available=" in caplog.text and "swa_available=" in caplog.text
    assert (
        "Host pool" if resource == "host" else "LOGICAL_OR_SWA_SHORTAGE"
    ) in caplog.text


def test_retraction_subset_checks_do_not_repeat_initial_pressure_warning(caplog):
    batch, _, _ = pressure_batch(False, pages=1)
    assert not batch.check_decode_mem(selected_indices=[0])
    assert "decode admission failed before request retraction" not in caplog.text
