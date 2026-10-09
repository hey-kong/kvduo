"""Production prefill rollback policy, ownership, and admission regressions."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.sparsity.core.kvduo_prefill_rollback import (
    KVDuoPrefillObject,
    KVDuoPrefillPolicy,
    plan_kvduo_prefill_rollback,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_hot_merge import make_merge_coordinator


register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def obj(req=0, layer=0, **changes):
    return replace(KVDuoPrefillObject((layer, req), 2, 4, (1, 20), 65, 1), **changes)


def ready_coordinator(extra=3, requests=1):
    coordinator, physical = make_merge_coordinator(
        {(0, 0): 8, **{(0, req): 1 for req in range(1, requests)}}
    )
    coordinator.hot_visited[0, 0, 16:32] = True
    coordinator._merge_kvduo_hot_pages(0, 0)
    coordinator._ensure_kvduo_hot_capacity_targets_batch({(0, 0): (4 + extra) * 4})
    coordinator.kvduo_prefill_ema[0, 0] = torch.tensor([1, 19])
    coordinator.kvduo_prefill_steps[0, 0] = 65
    physical.free_pages = 0
    return coordinator, physical


def test_requests_use_recent_hand_order_not_loss_or_remaining_deficit():
    older = obj(0, processed=1, counts=(0, 100))
    recent = obj(1, processed=2, counts=(1, 20))
    assert plan_kvduo_prefill_rollback([older, recent], 1, KVDuoPrefillPolicy()) == (1,)
    assert plan_kvduo_prefill_rollback([older, recent], 3, KVDuoPrefillPolicy()) == (
        1,
        0,
    )
    assert plan_kvduo_prefill_rollback([older, recent], 5, KVDuoPrefillPolicy()) is None
    assert plan_kvduo_prefill_rollback([obj(1), obj(0)], 1, KVDuoPrefillPolicy()) == (
        1,
    )


def test_request_aggregates_same_expanded_layers_and_allows_layer_budget_sharing():
    layers = [
        obj(layer=0, counts=(8, 10)),
        obj(layer=1, counts=(0, 90)),
        obj(layer=2, base_pages=2, current_pages=2, counts=(999, 9999), steps=0),
    ]
    assert plan_kvduo_prefill_rollback(layers, 4, KVDuoPrefillPolicy()) == (0,)
    # Base-only misses cannot hide excessive extra-page contribution.
    layers[1] = replace(layers[1], counts=(0, 10))
    assert plan_kvduo_prefill_rollback(layers, 1, KVDuoPrefillPolicy()) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": 64},
        {"processed": 0},
        {"eligible": False},
        {"base_pages": 0},
        {"current_pages": 2},
        {"counts": (3, 20)},
        {"counts": (float("nan"), 20)},
        {"counts": (1, -1)},
    ],
)
def test_request_requires_safe_merged_base_and_mature_extra_evaluation(changes):
    assert (
        plan_kvduo_prefill_rollback([obj(**changes)], 1, KVDuoPrefillPolicy()) is None
    )
    assert (
        plan_kvduo_prefill_rollback(
            [obj(), obj(layer=1, eligible=False, current_pages=2)],
            1,
            KVDuoPrefillPolicy(),
        )
        is None
    )


def test_zero_miss_has_no_hidden_tolerance_and_new_requests_cannot_renew_fixed_budget():
    assert plan_kvduo_prefill_rollback(
        [obj(counts=(0, 0))], 1, KVDuoPrefillPolicy()
    ) == (0,)
    assert (
        plan_kvduo_prefill_rollback([obj(counts=(1e-12, 0))], 1, KVDuoPrefillPolicy())
        is None
    )
    refs = {(0, 0): 10, (0, 1): 10}
    assert (
        plan_kvduo_prefill_rollback(
            [obj(1, counts=(1, 1000))], 1, KVDuoPrefillPolicy(), refs, 1.5
        )
        is None
    )
    assert plan_kvduo_prefill_rollback([obj(1)], 1, KVDuoPrefillPolicy(), refs, 1) == (
        1,
    )
    assert (
        plan_kvduo_prefill_rollback(
            [obj(2, counts=(0, 10000))], 1, KVDuoPrefillPolicy(), refs
        )
        is None
    )


def test_recent_candidate_exceeding_remaining_round_budget_does_not_block_older_safe_request():
    objects = [
        obj(0, counts=(3, 100), processed=9),
        obj(1, counts=(0, 20), processed=1),
    ]
    assert plan_kvduo_prefill_rollback(
        objects, 1, KVDuoPrefillPolicy(), {(0, 0): 1, (0, 1): 1}
    ) == (1,)


@pytest.mark.parametrize(
    "prefix,end,ratio,expected",
    [
        (3, 4, 1, 0),
        (3, 5, 1, 1),
        (1, 3, 4, 0),
        (3, 4, 4, 1),
        (4, 9, 4, 0),
        (4, 12, 4, 1),
    ],
)
def test_physical_deficit_matches_allocator_rounding(prefix, end, ratio, expected):
    c, _ = ready_coordinator()
    c.compress_ratio = ratio
    assert c.kvduo_prefill_pages(prefix, end) == expected


def budget_allocator(coordinator, *, dsv4=False, logical_size=1024, swa_size=1024):
    # Use real budget methods and real logical/SWA page inventories. Only the
    # GPU backing construction is omitted; the physical pool is the same one
    # that production rollback returns pages to in this fixture.
    page_size = coordinator.page_size * (4 if dsv4 else 1)
    full = PagedTokenToKVPoolAllocator(
        logical_size, page_size, torch.int64, "cpu", None, False
    )
    if dsv4:
        logical = object.__new__(SWATokenToKVPoolAllocator)
        logical.full_attn_allocator = full
        logical.swa_attn_allocator = PagedTokenToKVPoolAllocator(
            swa_size, page_size, torch.int64, "cpu", None, False
        )
        allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
        coordinator.compress_ratio = 4
        allocator.hisparse_page_size = coordinator.page_size
    else:
        logical = full
        allocator = object.__new__(HiSparseTokenToKVPoolAllocator)
    allocator.logical_attn_allocator = logical
    allocator.hisparse_attn_allocator = (
        coordinator.token_to_kv_pool_allocator.hisparse_attn_allocator
    )
    allocator.compress_ratio = coordinator.compress_ratio
    allocator.page_size = page_size
    return allocator


def adder_and_req(
    coordinator, *, host_hit=0, chunk=4, dsv4=False, logical_size=1024, swa_size=1024
):
    adder = PrefillAdder.__new__(PrefillAdder)
    adder.dsa_prefill_cp_in_seq_split = False
    adder.prefill_max_requests = None
    adder.can_run_list = []
    adder.new_chunked_req = None
    adder.token_to_kv_pool_allocator = budget_allocator(
        coordinator, dsv4=dsv4, logical_size=logical_size, swa_size=swa_size
    )
    adder.page_size = adder.token_to_kv_pool_allocator.page_size
    adder.rem_chunk_tokens = chunk
    adder.rem_input_tokens = 100
    adder.rem_total_token_offset = adder.cur_rem_token_offset = 0
    adder._mamba_slot_cost = 0
    adder.is_all_swa = adder.is_hybrid_ssm_cache = False
    adder.is_hybrid_swa = dsv4
    adder.rem_swa_token_offset = 0
    adder.dllm_config = adder.prefill_delayer_single_pass = None
    adder.tree_cache = MagicMock(
        disable=False,
        evictable_size=MagicMock(return_value=0),
        full_evictable_size=MagicMock(return_value=0),
        swa_evictable_size=MagicMock(return_value=0),
        sliding_window_size=adder.page_size,
        is_tree_cache=MagicMock(return_value=False),
    )
    adder._req_inc_lock_ref = MagicMock()
    adder._update_prefill_budget = MagicMock()
    adder.budget_state = MagicMock(return_value=AddReqResult.CONTINUE)
    adder.kvduo_coordinator = coordinator
    adder.kvduo_prefill_reservations = []
    adder.running_batch = SimpleNamespace(
        reqs=[SimpleNamespace(req_pool_idx=0, finished=lambda: False)]
    )
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(ignore_eos=False, max_new_tokens=0),
        output_ids=[],
        full_untruncated_fill_ids=list(range(16)),
        prefix_indices=torch.tensor([0, 1], dtype=torch.int64),
        host_hit_length=host_hit,
        swa_host_hit_length=0,
        mamba_pool_idx=None,
        last_node=object(),
        best_match_node=object(),
        cache_protected_len=2,
        retracted_stain=False,
        kvduo_residency_plan=object(),
        needs_host_load_back=lambda: bool(host_hit),
    )
    req.set_extend_range = lambda start, end: setattr(
        req, "extend_range", Range(start, end)
    )
    return adder, req


def actual_budget_adder_and_req(
    coordinator, *, path="new", running_max_new=0, **kwargs
):
    fixture, req = adder_and_req(coordinator, dsv4=True, **kwargs)
    fixture.tree_cache.supports_mamba.return_value = False
    running_req = fixture.running_batch.reqs[0]
    running_req.sampling_params = SimpleNamespace(max_new_tokens=running_max_new)
    running_req.output_ids = []
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    adder = PrefillAdder(
        fixture.page_size,
        fixture.tree_cache,
        fixture.token_to_kv_pool_allocator,
        fixture.running_batch,
        new_token_ratio=1,
        rem_input_tokens=100,
        rem_chunk_tokens=16 if path == "chunk" else None,
        kvduo_coordinator=coordinator,
    )
    req.prefix_indices = torch.arange(8, dtype=torch.int64)
    req.full_untruncated_fill_ids = list(range(40 if path == "chunk" else 24))
    req.cache_protected_len = 8
    return adder, req


@pytest.mark.parametrize("path", ["new", "chunk"])
def test_dsv4_full_physical_pool_reaches_rollback_with_real_allocator_budgets(path):
    coordinator, physical = ready_coordinator()
    adder, req = actual_budget_adder_and_req(coordinator, path=path)
    allocator = adder.token_to_kv_pool_allocator
    assert allocator.logical_attn_allocator.full_available_size() == 1024
    assert allocator.swa_available_size() == 1024
    assert allocator.full_available_size() == 0
    assert allocator.available_size() == 0
    reserve = MagicMock(wraps=coordinator.reserve_kvduo_prefill)
    coordinator.reserve_kvduo_prefill = reserve
    if path == "new":
        assert adder.add_one_req(req, False, None) is AddReqResult.CONTINUE
    else:
        assert adder.add_chunked_req(req) is req
    reserve.assert_called_once_with(
        2, [0], tree_cache=adder.tree_cache, prefix_indices=req.prefix_indices
    )
    assert adder.can_run_list == [req]
    assert req.extend_range == Range(8, 24)
    assert physical.free_pages == 3
    assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 16
    assert coordinator._kvduo_reserved_prefill_pages() == 2
    assert adder.rem_total_tokens == adder.cur_rem_tokens == 1000
    # Actual allocator budgeting still bounds allocations by physical capacity.
    assert allocator.full_available_size() == 24
    assert adder.rem_swa_token_offset > 0


@pytest.mark.parametrize("path", ["new", "chunk"])
@pytest.mark.parametrize("pool", ["logical", "swa"])
def test_rollback_admission_preserves_logical_and_swa_budget_rejections(path, pool):
    coordinator, physical = ready_coordinator()
    adder, req = actual_budget_adder_and_req(
        coordinator,
        path=path,
        logical_size=0 if pool == "logical" else 1024,
        swa_size=0 if pool == "swa" else 1024,
    )
    coordinator.reserve_kvduo_prefill = MagicMock(
        wraps=coordinator.reserve_kvduo_prefill
    )
    if path == "new":
        assert adder.add_one_req(req, False, None) is AddReqResult.NO_TOKEN
    else:
        assert adder.add_chunked_req(req) is req
    coordinator.reserve_kvduo_prefill.assert_not_called()
    assert physical.free_pages == 0
    assert not adder.can_run_list and not hasattr(req, "extend_range")
    assert not coordinator._kvduo_prefill_limited_requests


@pytest.mark.parametrize("path", ["new", "chunk"])
@pytest.mark.parametrize("reason", ["immature", "miss_budget", "space"])
def test_real_allocator_budget_separation_does_not_bypass_rollback_conditions(
    path, reason
):
    coordinator, physical = ready_coordinator()
    adder, req = actual_budget_adder_and_req(coordinator, path=path)
    if reason == "immature":
        coordinator.kvduo_prefill_steps.zero_()
    elif reason == "miss_budget":
        coordinator.kvduo_prefill_ema[0, 0, 0] = 3
    else:
        req.full_untruncated_fill_ids = list(range(40))
        adder.rem_chunk_tokens = 32 if path == "chunk" else None
    reserve = MagicMock(wraps=coordinator.reserve_kvduo_prefill)
    coordinator.reserve_kvduo_prefill = reserve
    if path == "new":
        assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    else:
        assert adder.add_chunked_req(req) is req
    reserve.assert_called_once()
    assert adder.kvduo_prefill_blocked
    assert physical.free_pages == 0
    assert not adder.can_run_list and not hasattr(req, "extend_range")
    assert not coordinator._kvduo_prefill_reservations
    assert not coordinator._kvduo_prefill_limited_requests


@pytest.mark.parametrize("path", ["new", "chunk"])
@pytest.mark.parametrize("coordinator_state", ["absent", "disabled"])
def test_non_kvduo_hisparse_keeps_composite_physical_budget(path, coordinator_state):
    coordinator, physical = ready_coordinator()
    adder, req = actual_budget_adder_and_req(coordinator, path=path)
    coordinator.reserve_kvduo_prefill = MagicMock()
    if coordinator_state == "absent":
        adder.kvduo_coordinator = None
    else:
        coordinator.enable_mixed_residency = False
    assert adder.rem_total_tokens == adder.cur_rem_tokens == 0
    if path == "new":
        assert adder.add_one_req(req, False, None) is AddReqResult.NO_TOKEN
    else:
        assert adder.add_chunked_req(req) is req
    coordinator.reserve_kvduo_prefill.assert_not_called()
    assert physical.free_pages == 0
    assert not adder.can_run_list


def test_generic_kvduo_chunk_cannot_use_fallback_to_bypass_empty_logical_pool():
    coordinator, physical = ready_coordinator()
    adder, req = adder_and_req(coordinator, logical_size=0)
    coordinator.reserve_kvduo_prefill = MagicMock()
    assert adder.add_chunked_req(req) is req
    coordinator.reserve_kvduo_prefill.assert_not_called()
    assert not adder.can_run_list and not hasattr(req, "extend_range")
    assert physical.free_pages == 0


def test_logical_budget_is_rechecked_after_prefix_lock_before_rollback():
    coordinator, physical = ready_coordinator()
    adder, req = actual_budget_adder_and_req(coordinator, logical_size=0)
    adder.tree_cache.full_evictable_size.return_value = 32
    coordinator.reserve_kvduo_prefill = MagicMock()

    def lock(_):
        adder.tree_cache.full_evictable_size.return_value = 0

    adder.tree_cache.inc_lock_ref.side_effect = lock
    assert adder.add_one_req(req, False, None) is AddReqResult.NO_TOKEN
    adder.tree_cache.inc_lock_ref.assert_called_once()
    coordinator.reserve_kvduo_prefill.assert_not_called()
    assert physical.free_pages == 0
    assert not adder.can_run_list


@pytest.mark.parametrize("path", ["new", "chunk"])
def test_kvduo_keeps_logical_headroom_for_future_decode_before_rollback(path):
    coordinator, physical = ready_coordinator()
    adder, req = actual_budget_adder_and_req(
        coordinator, path=path, running_max_new=1024 if path == "chunk" else 0
    )
    coordinator.reserve_kvduo_prefill = MagicMock()
    assert (
        adder.token_to_kv_pool_allocator.logical_attn_allocator.full_available_size()
        == 1024
    )
    if path == "new":
        req.sampling_params.max_new_tokens = 1000  # extend + decode + page = 1024
        assert adder.add_one_req(req, False, None) is AddReqResult.NO_TOKEN
    else:
        assert adder.rem_total_token_offset == 1024
        assert adder.add_chunked_req(req) is req
    coordinator.reserve_kvduo_prefill.assert_not_called()
    assert not adder.can_run_list and not hasattr(req, "extend_range")
    assert physical.free_pages == 0


def test_multiple_prefills_charge_logical_swa_offsets_and_separate_physical_reservations():
    coordinator, physical = ready_coordinator()
    adder, first = actual_budget_adder_and_req(coordinator)
    _, second = actual_budget_adder_and_req(coordinator)
    second.full_untruncated_fill_ids = list(range(16))
    assert adder.add_one_req(first, False, None) is AddReqResult.CONTINUE
    assert adder.add_one_req(second, False, None) is AddReqResult.CONTINUE
    assert adder.can_run_list == [first, second]
    assert adder.rem_total_tokens == adder.cur_rem_tokens == 984
    assert adder.rem_swa_token_offset == 40
    assert coordinator._kvduo_reserved_prefill_pages() == physical.free_pages == 3
    assert len(adder.kvduo_prefill_reservations) == 2
    # A third prefill cannot use the already-reserved physical pages.
    _, third = actual_budget_adder_and_req(coordinator)
    third.full_untruncated_fill_ids = list(range(16))
    assert adder.add_one_req(third, False, None) is AddReqResult.OTHER
    assert adder.can_run_list == [first, second]
    assert coordinator._kvduo_reserved_prefill_pages() == 3


def test_prefill_waits_without_mutating_chunk_or_loading_host_when_no_complete_plan():
    coordinator, physical = ready_coordinator()
    coordinator.kvduo_prefill_steps.zero_()
    adder, req = adder_and_req(coordinator, host_hit=4)
    assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    assert not adder.can_run_list and not hasattr(req, "extend_range")
    adder.tree_cache.init_load_back.assert_not_called()
    assert physical.free_pages == 0
    assert adder.add_chunked_req(req) is req
    assert not adder.can_run_list and not hasattr(req, "extend_range")


def test_host_restore_and_chunk_share_exact_reservation_and_cancellation_releases_hold():
    coordinator, physical = ready_coordinator(extra=4)
    adder, req = adder_and_req(coordinator, host_hit=4)

    def restore(_):
        assert coordinator._kvduo_reserved_prefill_pages() == 4
        physical.free_pages -= 2
        return torch.tensor([2, 3, 4, 5]), object()

    adder.tree_cache.init_load_back.side_effect = restore
    assert adder.add_one_req(req, False, None) is AddReqResult.CONTINUE
    assert req.extend_range == Range(6, 10)
    assert coordinator._kvduo_reserved_prefill_pages() == 2
    assert physical.free_pages == 2
    token = adder.kvduo_prefill_reservations[0]
    coordinator.finish_kvduo_prefill_reservation(token, cancelled=False)
    assert coordinator._kvduo_reserved_prefill_pages() == 0
    coordinator, physical = ready_coordinator(extra=4)
    adder, req = adder_and_req(coordinator, host_hit=4)
    adder.tree_cache.init_load_back.return_value = (None, None)
    assert adder.add_one_req(req, False, None) is AddReqResult.OTHER
    assert not coordinator._kvduo_prefill_reservations
    assert not coordinator._kvduo_prefill_limited_requests


def merged_and_expanded(page_counts, expanded, *, layers=1):
    c, physical = make_merge_coordinator(page_counts, layers=layers)
    for owner in page_counts:
        c.hot_visited[owner][:8] = True
        c._merge_kvduo_hot_pages(*owner)
    c._ensure_kvduo_hot_capacity_targets_batch(
        {owner: pages * 4 for owner, pages in expanded.items()}
    )
    for owner in expanded:
        c.kvduo_prefill_ema[owner] = torch.tensor([1, 20])
        c.kvduo_prefill_steps[owner] = 65
    physical.free_pages = 0
    return c, physical


def test_rollback_cuda_entry_uses_unpadded_lru_capacity_and_passes_full_banks(
    monkeypatch,
):
    import sglang.kernels.ops.kvcache.hisparse as ops

    c, _ = make_merge_coordinator(
        {(layer, req): 4 for req in range(3) for layer in range(2)}, layers=2
    )
    assert c.req_device_buffer_tokens.shape == (2, 3, 34)
    assert c.req_device_buffer_token_locs.shape == c.hot_visited.shape == (2, 3, 34)
    assert c.lru_slots.shape == (2, 3, 32)
    owners = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    targets = torch.tensor([8, 12], dtype=torch.int32)
    module = MagicMock()
    load_module = MagicMock(return_value=module)
    monkeypatch.setattr(ops, "_jit_kvduo_rollback_module", load_module)
    # Exercise CUDA dispatch/argument wiring without requiring a GPU. Actual
    # padded-kernel execution is covered by the registered kernel regressions.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    banks = (
        c.req_device_buffer_tokens,
        c.req_device_buffer_token_locs,
        c.lru_slots,
        c.hot_visited,
        c.kvduo_req_hot_capacity_gpu,
    )
    ops.rollback_kvduo_hot_metadata(owners, targets, *banks)
    load_module.assert_called_once_with(32, 256)
    module.rollback.assert_called_once()
    assert all(
        actual is expected
        for actual, expected in zip(
            module.rollback.call_args.args, (owners, targets, *banks)
        )
    )


def test_multi_request_layer_rollback_with_coordinator_padding_publishes_before_free():
    page_counts = {(layer, req): 4 for req in range(3) for layer in range(2)}
    c, physical = merged_and_expanded(
        page_counts, {owner: 3 for owner in page_counts}, layers=2
    )
    banks = (
        c.req_device_buffer_tokens,
        c.req_device_buffer_token_locs,
        c.lru_slots,
        c.hot_visited,
        c.kvduo_req_hot_capacity_gpu,
    )
    assert banks[0].size(2) == c.device_buffer_size + c.page_size
    assert banks[2].size(2) == c.device_buffer_size
    c.req_device_buffer_tokens[:, :, 32:] = 999
    c.req_device_buffer_token_locs[:, :, 32:] = 999
    c.hot_visited.fill_(True)
    c.hot_visited[:, :, 1] = False
    c.lru_slots[:, :, :12] = torch.tensor(
        [9, 3, 10, 7, 0, 11, 6, 2, 8, 1, 5, 4], dtype=torch.int16
    )
    before = [bank.clone() for bank in banks]
    addresses = [(bank.data_ptr(), bank.shape, bank.stride()) for bank in banks]
    payloads = [bank.clone() for bank in c.mem_pool_device.kv_buffer]
    c._move_kvduo_hot_entries = MagicMock()
    c.token_to_kv_pool_allocator.free_hisparse_indices.reset_mock()
    free = c.token_to_kv_pool_allocator.free_hisparse_indices.side_effect

    def check_then_free(indices):
        # The most recently processed requests (2 and 1) each release two
        # pages; request 0, including its padding, must remain untouched.
        assert indices.numel() == 4 * c.page_size
        for bank, original in zip(banks, before):
            assert torch.equal(bank[:, 0], original[:, 0])
        for layer in range(2):
            for req in (1, 2):
                owner = (layer, req)
                assert c.kvduo_req_hot_capacity[owner].item() == 8
                assert c.kvduo_req_hot_capacity_gpu[owner].item() == 8
                assert c.lru_slots[owner][:8].tolist() == [3, 7, 0, 6, 2, 1, 5, 4]
                assert torch.all(c.lru_slots[owner][8:] == -1)
                for bank, original in zip(
                    (banks[0], banks[1], banks[3]), (before[0], before[1], before[3])
                ):
                    assert torch.equal(bank[owner][:8], original[owner][:8])
                assert torch.all(c.req_device_buffer_tokens[owner][8:] == -1)
                assert torch.all(c.req_device_buffer_token_locs[owner][8:] == -1)
                assert not c.hot_visited[owner][8:].any()
                assert c._kvduo_req_layer_pages[owner] == list(
                    c._kvduo_hot_base_pages[owner]
                )
        free(indices)

    c.token_to_kv_pool_allocator.free_hisparse_indices.side_effect = check_then_free
    reservation = c.reserve_kvduo_prefill(3, [0, 1, 2])
    assert reservation is not None and physical.free_pages == 4
    assert c._kvduo_reserved_prefill_pages() == 3
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_called_once()
    c._move_kvduo_hot_entries.assert_not_called()
    assert addresses == [(bank.data_ptr(), bank.shape, bank.stride()) for bank in banks]
    assert all(
        torch.equal(bank, original)
        for bank, original in zip(c.mem_pool_device.kv_buffer, payloads)
    )


def test_first_growth_starts_evaluation_without_reference_stage_later_growth_keeps_history():
    c, p = merged_and_expanded({(0, 0): 4}, {(0, 0): 3})
    assert c._kvduo_hot_base_pages[0, 0] == tuple(c._kvduo_req_layer_pages[0, 0][:2])
    pointers = [
        bank.data_ptr()
        for bank in (
            c.kvduo_hot_base_capacity,
            c.kvduo_prefill_ema,
            c.kvduo_prefill_steps,
            c.hot_visited,
        )
    ]
    p.free_pages = 10
    c._ensure_kvduo_hot_capacity_targets_batch({(0, 0): 16})
    assert c.kvduo_prefill_ema[0, 0].tolist() == [1, 20]
    assert c.kvduo_prefill_steps[0, 0].item() == 65
    assert pointers == [
        bank.data_ptr()
        for bank in (
            c.kvduo_hot_base_capacity,
            c.kvduo_prefill_ema,
            c.kvduo_prefill_steps,
            c.hot_visited,
        )
    ]
    c.req_device_buffer_tokens[0, 0, :16] = torch.arange(16, dtype=torch.int32)
    c.hot_visited[0, 0, :16] = True
    c._merge_kvduo_hot_pages(0, 0)  # no-release merge still resets the phase
    assert len(c._kvduo_hot_base_pages[0, 0]) == 4
    assert c.kvduo_prefill_steps[0, 0].item() == 0
    assert not c.kvduo_prefill_ema.any() and not c.hot_visited.any()
    c.kvduo_prefill_ema[0, 0] = torch.tensor([9, 10])
    c.kvduo_prefill_steps[0, 0] = 30
    c._ensure_kvduo_hot_capacity_targets_batch({(0, 0): 20})
    assert c.kvduo_prefill_steps[0, 0].item() == 0 and not c.kvduo_prefill_ema.any()


def test_whole_request_releases_all_extra_pages_and_limits_base_only_and_later_hot_layers():
    c, p = merged_and_expanded(
        {(0, 0): 4, (1, 0): 4, (2, 0): 2}, {(0, 0): 4, (1, 0): 3}, layers=4
    )
    c.kvduo_prefill_ema[0, 0] = torch.tensor([8, 10])
    c.kvduo_prefill_ema[1, 0] = torch.tensor([0, 90])
    c.hot_visited[:, 0, :8] = True
    c.hot_visited[0, 0, 8:16] = True
    c.lru_slots[0, 0, :16] = torch.tensor(
        [9, 3, 10, 7, 0, 11, 6, 12, 2, 8, 1, 13, 5, 14, 4, 15], dtype=torch.int16
    )
    before = c.req_device_buffer_tokens.clone()
    visited = c.hot_visited.clone()
    move = MagicMock()
    c._move_kvduo_hot_entries = move
    token = c.reserve_kvduo_prefill(1, [0])
    assert token is not None and p.free_pages == 3
    assert c.kvduo_req_hot_capacity[:, 0].tolist() == [8, 8, 8, 0]
    assert c.lru_slots[0, 0, :8].tolist() == [3, 7, 0, 6, 2, 1, 5, 4]
    assert torch.equal(c.req_device_buffer_tokens[:3, 0, :8], before[:3, 0, :8])
    assert torch.equal(c.hot_visited[:3, 0, :8], visited[:3, 0, :8])
    assert not c.hot_visited[:, :, 8:].any()
    assert c._kvduo_request_limited(0) and not c._kvduo_can_merge((2, 0))
    move.assert_not_called()
    assert c._kvduo_prefill_budget_cost[c._kvduo_request_key(0)] == 8
    assert c._kvduo_prefill_budget_reference == {(0, 0, 0): 10, (1, 0, 0): 90}
    # No established HOT cache in layer3: its later mandatory allocation is
    # allowed, but never above the minimum, and inherits the request limit.
    c._ensure_kvduo_hot_capacity_targets_batch({(3, 0): 16}, allow_reclaim=False)
    assert c.kvduo_req_hot_capacity[3, 0].item() == 4
    assert len(c._kvduo_hot_base_pages[3, 0]) == 1
    assert c._kvduo_request_limited(0)
    c._ensure_kvduo_hot_capacity_targets_batch({(2, 0): 12}, allow_reclaim=False)
    assert c.kvduo_req_hot_capacity[2, 0].item() == 8
    # The inherited minimum is a limit, not a Hand merge. After unlocking,
    # this new layer still needs its first ordinary merge before a later
    # whole-request rollback can use it as a valid merge base.
    c._release_kvduo_prefill_hold(c._kvduo_request_key(0))
    assert not next(
        obj for obj in c._kvduo_prefill_objects([0]) if obj.owner == (3, 0)
    ).base_valid


@pytest.mark.parametrize(
    "unsafe", ["immature", "no_base", "protected", "staging", "generation"]
)
def test_whole_request_safety_failure_never_partially_releases_other_layers(unsafe):
    c, p = merged_and_expanded({(0, 0): 4, (1, 0): 4}, {(0, 0): 3, (1, 0): 3}, layers=2)
    if unsafe == "immature":
        c.kvduo_prefill_steps[1, 0] = 64
    if unsafe == "no_base":
        c._kvduo_hot_base_pages.pop((1, 0))
    if unsafe == "protected":
        c._kvduo_hot_pressure_protected.add((1, 0, 0))
    if unsafe == "staging":
        c._active_kvduo_reqs[0] = SimpleNamespace(hisparse_staging=True)
    if unsafe == "generation":
        c._kvduo_hot_base_generation[1, 0] = -1
    before = c.req_device_buffer_tokens.clone()
    c.token_to_kv_pool_allocator.free_hisparse_indices.reset_mock()
    assert c.reserve_kvduo_prefill(1, [0]) is None
    assert torch.equal(c.req_device_buffer_tokens, before)
    assert p.free_pages == 0 and not c._kvduo_prefill_limited_requests
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_not_called()


def test_multiple_selected_requests_are_revalidated_independent_of_recency_order():
    c, p = merged_and_expanded({(0, 0): 4, (0, 1): 4}, {(0, 0): 3, (0, 1): 3})
    assert c._kvduo_hot_processed[0, 1] > c._kvduo_hot_processed[0, 0]
    assert c.reserve_kvduo_prefill(2, [0, 1]) is not None
    assert p.free_pages == 2 and len(c._kvduo_prefill_limited_requests) == 2


def test_fixed_round_budget_does_not_decay_or_expand_with_current_miss_observations():
    c, p = merged_and_expanded({(0, 0): 4, (0, 1): 4}, {(0, 0): 3, (0, 1): 3})
    c._kvduo_hot_processed[0, 0] = 20
    c.kvduo_prefill_ema[0] = torch.tensor([[5, 100], [1, 100]])
    token = c.reserve_kvduo_prefill(1, [0, 1])
    assert token is not None
    c.consume_kvduo_prefill_reservation(token, 1)
    p.free_pages = 0
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    assert c._kvduo_prefill_budget_cost == {(0, 0): 5}
    c.kvduo_prefill_ema[0, 0] = 0
    c.kvduo_prefill_ema[0, 1] = torch.tensor([17, 1000])
    assert c.reserve_kvduo_prefill(1, [0, 1]) is None
    assert p.free_pages == 0 and c._kvduo_prefill_budget_cost == {(0, 0): 5}
    c.kvduo_prefill_ema[0, 1] = torch.tensor([15, 1000])
    second = c.reserve_kvduo_prefill(1, [0, 1])
    assert second is not None and sum(c._kvduo_prefill_budget_cost.values()) == 20
    c._reset_kvduo_hot_metadata(0)
    assert c._kvduo_prefill_budget_reference == {(0, 1, 0): 100}
    assert c._kvduo_prefill_budget_cost == {(1, 0): 15}
    assert c._kvduo_hand_sequence == 2  # lifecycle reset never restarts Hand
    c.finish_kvduo_prefill_reservation(second, cancelled=True)
    assert (
        not c._kvduo_prefill_budget_reference and not c._kvduo_prefill_limited_requests
    )


def test_no_new_generation_or_evaluation_phase_can_use_old_round_reference():
    c, p = merged_and_expanded({(0, 0): 4, (0, 1): 4}, {(0, 0): 3, (0, 1): 3})
    c._kvduo_hot_processed[0, 0] = 20
    token = c.reserve_kvduo_prefill(1, [0, 1])
    c.consume_kvduo_prefill_reservation(token, 1)
    p.free_pages = 0
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    c.req_device_buffer_tokens[0, 1, :12] = torch.arange(12, dtype=torch.int32)
    c.hot_visited[0, 1, :12] = True
    c._merge_kvduo_hot_pages(0, 1)  # new base invalidates original round entry
    p.free_pages = 1
    c._ensure_kvduo_hot_capacity_targets_batch({(0, 1): 16})
    c.kvduo_prefill_ema[0, 1] = torch.tensor([0, 100])
    c.kvduo_prefill_steps[0, 1] = 65
    p.free_pages = 0
    assert c.reserve_kvduo_prefill(1, [0, 1]) is None
    c._reset_kvduo_hot_metadata(1)
    assert (0, 1, 0) not in c._kvduo_prefill_budget_reference
    assert c._kvduo_request_generation[1] == 1


def test_prefill_completion_keeps_limits_until_real_headroom_and_discards_old_events():
    c, p = ready_coordinator()
    token = c.reserve_kvduo_prefill(3, [0])
    c.consume_kvduo_prefill_reservation(token, 3)
    p.free_pages = 0
    c.finish_kvduo_prefill_reservation(token, cancelled=False)
    c.kvduo_hot_growth_events[0, 0] = 3
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c._kvduo_request_limited(0) and c.kvduo_req_hot_capacity[0, 0].item() == 16
    assert c.kvduo_hot_growth_events[0, 0].item() == 0
    p.free_pages = 1
    c.kvduo_hot_growth_events[0, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert not c._kvduo_request_limited(0)
    assert c.kvduo_req_hot_capacity[0, 0].item() == 16
    c.kvduo_hot_growth_events[0, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c.kvduo_req_hot_capacity[0, 0].item() == 20


def test_request_end_clears_all_generation_bound_records_but_not_global_hand():
    c, p = ready_coordinator()
    c._kvduo_hand = 0
    token = c.reserve_kvduo_prefill(1, [0])
    c._release_kvduo_layer_hot_pages(0)
    assert not c._kvduo_hot_base_pages and not c._kvduo_hot_generation
    assert not c._kvduo_hot_processed and not c._kvduo_prefill_budget_reference
    assert not c._kvduo_prefill_budget_cost and not c._kvduo_prefill_limited_requests
    assert c._kvduo_request_generation == [1] and c._kvduo_hand_sequence == 1
    assert c._kvduo_prefill_reservations[token][1] == set()
    c.finish_kvduo_prefill_reservation(token, cancelled=True)


def test_failed_batch_metadata_update_cancels_limits_and_never_frees_unsafe_pages(
    monkeypatch,
):
    import sglang.srt.managers.hisparse_coordinator as module

    c, p = ready_coordinator()
    before = c.req_device_buffer_tokens.clone()
    c.token_to_kv_pool_allocator.free_hisparse_indices.reset_mock()

    def fail(*args):
        assert c._kvduo_reserved_prefill_pages() == 1 and c._kvduo_request_limited(0)
        raise RuntimeError("metadata failed")

    monkeypatch.setattr(module, "rollback_kvduo_hot_metadata", fail)
    with pytest.raises(RuntimeError, match="metadata failed"):
        c.reserve_kvduo_prefill(1, [0])
    assert torch.equal(before, c.req_device_buffer_tokens) and p.free_pages == 0
    assert not c._kvduo_prefill_limited_requests and not c._kvduo_prefill_reservations
    assert not c._kvduo_prefill_budget_cost and not c._kvduo_prefill_budget_reference
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_not_called()
