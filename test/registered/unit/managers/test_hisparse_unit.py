"""Unit tests for HiSparse hierarchical sparse KV cache system.

Tests cover:
- CUDA kernel correctness (swap_in_selected_pages vs naive_load_topk oracle)
- Memory allocator lifecycle (alloc / free / available_size)
- Request lifecycle (staging path, direct-to-host path)
- Batch multi-request correctness
"""

import os
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

# ---------------------------------------------------------------------------
# Test configuration (small-scale for fast CI runs)
# ---------------------------------------------------------------------------
SIZE = 2048  # device buffer pool size (tokens)
PAGE_SIZE = 64  # page size (must be 64 for CUDA, 1 for ROCm)
TOP_K = 256  # top-k selection count
DEVICE_BUFFER_SIZE = 512  # device buffer per request
HOST_TO_DEVICE_RATIO = 2
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
KV_CACHE_DIM = 576  # MLA dim (DeepSeek-style)
LAYER_NUM = 2
MAX_NUM_REQS = 8
MAX_CONTEXT_LEN = 2048


class TestKVDuoPhysicalReclaim(unittest.TestCase):
    def test_prefix_identities_share_pages_and_compare_exactly(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        class CollidingToken(int):
            def __hash__(self):
                return 0

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.page_size = 2
        coordinator.compress_ratio = 1
        left = SimpleNamespace(
            extra_key="tenant", get_fill_ids=lambda: list(range(200))
        )
        identities = coordinator._host_prefix_identities(left, 99)

        self.assertEqual(sum(len(identity.tokens) for identity in identities), 200)
        self.assertIs(identities[-1].parent, identities[-2])
        self.assertIs(coordinator._host_prefix_identity(left, 99), identities[-1])

        # Matching a second request canonicalizes each parent as it advances,
        # so every dictionary equality checks only the newly appended page.
        right = SimpleNamespace(extra_key="tenant", get_fill_ids=MagicMock())
        records = {identity: identity for identity in identities}
        right_ids = []
        fill_ids = list(range(200))
        for ordinal in range(100):
            right_ids = coordinator._host_prefix_identities(right, ordinal, fill_ids)
            right_ids[ordinal] = records[right_ids[ordinal]]
        self.assertIs(right_ids[-1], identities[-1])
        right.get_fill_ids.assert_not_called()

        collision_a = SimpleNamespace(
            extra_key="tenant",
            get_fill_ids=lambda: [CollidingToken(1), CollidingToken(3)],
        )
        collision_b = SimpleNamespace(
            extra_key="tenant",
            get_fill_ids=lambda: [CollidingToken(2), CollidingToken(3)],
        )
        identity_a = coordinator._host_prefix_identity(collision_a, 0)
        identity_b = coordinator._host_prefix_identity(collision_b, 0)
        self.assertEqual(hash(identity_a), hash(identity_b))
        self.assertNotEqual(identity_a, identity_b)

    def test_full_gpu_match_does_not_build_host_identity(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
        from sglang.srt.mem_cache.base_prefix_cache import MatchResult
        from sglang.srt.mem_cache.sparsity.core.kvduo_prefix_cache import (
            KVDuoHostPrefixCache,
        )

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.enable_mixed_residency = True
        coordinator.page_size = 2
        coordinator.compress_ratio = 1
        coordinator.tp_world_size = 1
        coordinator.host_prefix_cache = KVDuoHostPrefixCache(0)
        req = SimpleNamespace(
            rid="gpu-only",
            extra_key="tenant",
            get_fill_ids=lambda: [1, 2, 3, 4, 5],
            _compute_max_prefix_len=lambda length: length - 1,
        )
        match = MatchResult(
            device_indices=torch.tensor([1, 2, 3, 4]),
            last_device_node=object(),
            last_host_node=object(),
            best_match_node=object(),
        )

        self.assertIs(coordinator.augment_kvduo_prefix_match(req, match), match)
        self.assertFalse(hasattr(req, "_kvduo_prefix_identity_state"))

    def test_long_prefix_lru_tie_break_is_fixed_width_and_stable(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
        from sglang.srt.mem_cache.sparsity.core.kvduo_prefix_cache import (
            HostPrefixRecord,
            KVDuoHostPrefixCache,
        )

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.page_size = 1
        coordinator.compress_ratio = 1
        token_sets = [list(range(10_000)), [-1, *range(1, 10_000)]]
        identities = []
        for tokens in token_sets:
            req = SimpleNamespace(extra_key="tenant")
            identities.append(
                coordinator._host_prefix_identity(req, len(tokens) - 1, tokens)
            )

        self.assertEqual(len(identities[0].stable_sort_digest), 16)
        self.assertEqual(len(identities[1].stable_sort_digest), 16)
        records = [
            HostPrefixRecord(
                identity=identity,
                domains=("main_kv",),
                host_locations={"main_kv": (index,)},
                data_versions={"main_kv": (0,)},
                host_versions={"main_kv": (0,)},
                cache_reference=True,
                last_access=7,
                tie_break_key=(9_999, identity.stable_sort_digest),
            )
            for index, identity in enumerate(identities)
        ]
        expected = min(records, key=lambda record: record.tie_break_key).identity
        cache = KVDuoHostPrefixCache(2)
        # Reverse insertion order so eviction must use the stable tie breaker.
        for record in reversed(records):
            cache.insert(record)

        self.assertIs(cache.evict_lru(1)[0].identity, expected)

    def test_generic_restore_requires_no_swa_pages(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.is_dsv4_hisparse = False
        coordinator.page_size = 2
        coordinator.compress_ratio = 1
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(page_size=2)

        requirements = coordinator._kvduo_restore_page_requirements(
            torch.tensor([2], dtype=torch.int64),
            torch.tensor([4], dtype=torch.int64),
        )

        self.assertEqual(requirements, (1, 0, 1))

    def test_regular_cache_reserves_swa_window_from_kvduo_host_match(self):
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache._sliding_window_size = 4
        cache.swa_reprefill_tail_tokens = MagicMock(return_value=0)
        coordinator = SimpleNamespace(is_dsv4_hisparse=True)
        key = RadixKey(token_ids=array("i", range(12)))

        self.assertEqual(cache._kvduo_max_prefix_len(key, coordinator), 8)

    def test_host_only_prefix_is_matched_and_restored_before_prefill(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
        from sglang.srt.mem_cache.base_prefix_cache import MatchResult
        from sglang.srt.mem_cache.sparsity.core.kvduo_prefix_cache import (
            HostPrefixRecord,
            KVDuoHostPrefixCache,
            KVDuoPrefixResidency,
        )
        from sglang.srt.mem_cache.sparsity.core.kvduo_state import (
            KVDuoPressureAction,
        )

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.enable_mixed_residency = True
        coordinator.top_k = 2
        coordinator.page_size = 2
        coordinator.compress_ratio = 1
        coordinator.device = "cpu"
        coordinator.is_dsv4_hisparse = True
        coordinator.tp_world_size = 1
        coordinator.item_size_bytes = 4
        coordinator.mem_pool_device = SimpleNamespace(layer_num=2)
        coordinator.full_generation = torch.zeros(32, dtype=torch.int64)
        coordinator.full_data_version = torch.zeros(32, dtype=torch.int64)
        coordinator.full_host_version = torch.full((32,), -1, dtype=torch.int64)
        coordinator.full_last_touch = torch.zeros(32, dtype=torch.int64)
        coordinator.host_prefix_cache = KVDuoHostPrefixCache(8)
        fill_call_count = 0

        def get_fill_ids():
            nonlocal fill_call_count
            fill_call_count += 1
            return [1, 2, 3, 4, 5, 6, 7]

        req = SimpleNamespace(
            rid="restore",
            extra_key="tenant-a",
            prefix_indices=torch.tensor([5, 6], dtype=torch.int64),
            get_fill_ids=get_fill_ids,
            _compute_max_prefix_len=lambda length: length - 1,
        )
        for ordinal, (locs, touches) in enumerate(
            (
                ((20, 21), (7, 8)),
                ((22, 23), (9, 10)),
                ((24, 25), (11, 12)),
            )
        ):
            identity = coordinator._host_prefix_identity(req, ordinal)
            coordinator.host_prefix_cache.insert(
                HostPrefixRecord(
                    identity=identity,
                    domains=("main_kv",),
                    host_locations={"main_kv": locs},
                    data_versions={"main_kv": (ordinal + 1,) * 2},
                    host_versions={"main_kv": (ordinal + 1,) * 2},
                    model_touches={"main_kv": touches},
                    cache_reference=True,
                )
            )

        gpu_match = MatchResult(
            device_indices=req.prefix_indices,
            last_device_node=object(),
            last_host_node=object(),
            best_match_node=object(),
        )
        fill_call_count = 0
        match = coordinator.augment_kvduo_prefix_match(req, gpu_match, max_prefix_len=4)
        self.assertEqual(fill_call_count, 1)
        self.assertEqual(match.host_hit_length, 2)
        self.assertEqual(match.full_kv_hit_length, 4)
        self.assertEqual(
            match.kvduo_residency_plan.execution_status,
            KVDuoPrefixResidency.RESTORE_REQUIRED,
        )
        req.kvduo_residency_plan = match.kvduo_residency_plan

        other_namespace = SimpleNamespace(
            rid="other",
            extra_key="tenant-b",
            prefix_indices=req.prefix_indices,
            get_fill_ids=req.get_fill_ids,
            _compute_max_prefix_len=req._compute_max_prefix_len,
        )
        isolated = coordinator.augment_kvduo_prefix_match(
            other_namespace, gpu_match, max_prefix_len=4
        )
        self.assertEqual(isolated.host_hit_length, 0)

        rank_mismatch = SimpleNamespace(
            rid="rank-mismatch",
            extra_key="tenant-a",
            prefix_indices=req.prefix_indices,
            get_fill_ids=req.get_fill_ids,
            _compute_max_prefix_len=req._compute_max_prefix_len,
        )
        coordinator.tp_world_size = 2
        coordinator.tp_group = object()

        heterogeneous_gpu = SimpleNamespace(
            rid="heterogeneous-gpu",
            extra_key="tenant-a",
            prefix_indices=req.prefix_indices,
            get_fill_ids=req.get_fill_ids,
            _compute_max_prefix_len=req._compute_max_prefix_len,
        )

        def remote_has_longer_gpu_prefix(value, **kwargs):
            # Local boundaries are [host_end=4, gpu_len=2, -gpu_len=-2]. The
            # remote rank has the same final host end but a 3-token GPU prefix.
            value[0] = min(int(value[0]), 4)
            value[1] = min(int(value[1]), 3)
            value[2] = min(int(value[2]), -3)

        with patch(
            "torch.distributed.all_reduce", side_effect=remote_has_longer_gpu_prefix
        ):
            synchronized = coordinator.augment_kvduo_prefix_match(
                heterogeneous_gpu, gpu_match, max_prefix_len=4
            )
        self.assertEqual(
            len(gpu_match.device_indices) + synchronized.host_hit_length, 4
        )
        coordinator.release_kvduo_match_refs(heterogeneous_gpu)

        # Preserve the two-token common GPU prefix instead of falling back to
        # an empty match; this can keep a long prompt within prefill budget.
        common_gpu_matcher = MagicMock(return_value=gpu_match)

        def remote_cannot_reach_longer_gpu_prefix(value, **kwargs):
            # The remote GPU prefix is longer than the shortest host-covered
            # prefix, so no positive restore can produce a common final end.
            if value.numel() == 3:
                value[0] = min(int(value[0]), 2)
                value[1] = min(int(value[1]), 3)
                value[2] = min(int(value[2]), -3)

        with patch(
            "torch.distributed.all_reduce",
            side_effect=remote_cannot_reach_longer_gpu_prefix,
        ):
            synchronized = coordinator.augment_kvduo_prefix_match(
                rank_mismatch,
                gpu_match,
                max_prefix_len=4,
                common_gpu_matcher=common_gpu_matcher,
            )
        self.assertIs(synchronized, gpu_match)
        self.assertEqual(len(synchronized.device_indices), 2)
        common_gpu_matcher.assert_called_once_with(2)

        nonmonotonic_req = SimpleNamespace(
            rid="nonmonotonic-swa",
            extra_key="tenant-a",
            prefix_indices=req.prefix_indices,
            get_fill_ids=req.get_fill_ids,
            _compute_max_prefix_len=req._compute_max_prefix_len,
        )
        empty_gpu_match = gpu_match._replace(
            device_indices=torch.empty(0, dtype=torch.int64)
        )
        nonmonotonic_matcher = MagicMock(side_effect=[gpu_match, empty_gpu_match])
        reduce_step = 0

        def remote_swa_rematch_misses(value, **kwargs):
            nonlocal reduce_step
            reduce_step += 1
            if reduce_step == 1:
                value[0] = min(int(value[0]), 2)
                value[1] = min(int(value[1]), 3)
                value[2] = min(int(value[2]), -3)
            elif reduce_step == 2:
                # Local bounded rematch still hits two tokens; remote SWA
                # validation temporarily loses the whole bounded prefix.
                value[0] = 0
                value[1] = min(int(value[1]), -2)

        with patch(
            "torch.distributed.all_reduce", side_effect=remote_swa_rematch_misses
        ):
            synchronized = coordinator.augment_kvduo_prefix_match(
                nonmonotonic_req,
                gpu_match,
                max_prefix_len=4,
                common_gpu_matcher=nonmonotonic_matcher,
            )
        self.assertEqual(len(synchronized.device_indices), 0)
        self.assertEqual(
            [call.args[0] for call in nonmonotonic_matcher.call_args_list], [2, 0]
        )

        def remote_miss(value, **kwargs):
            value.zero_()

        with patch("torch.distributed.all_reduce", side_effect=remote_miss) as reduce:
            synchronized = coordinator.augment_kvduo_prefix_match(
                rank_mismatch, gpu_match, max_prefix_len=4
            )
        self.assertEqual(synchronized.host_hit_length, 0)
        self.assertEqual(rank_mismatch.kvduo_host_prefix_records, set())
        reduce.assert_called_once()

        free_pages = {"full": 0, "swa": 0, "physical": 0}
        logical_allocator = SimpleNamespace(
            full_available_size=lambda: free_pages["full"] * 2,
            swa_available_size=lambda: free_pages["swa"] * 2,
        )
        physical_allocator = SimpleNamespace(
            available_size=lambda: free_pages["physical"] * 2
        )

        def alloc_extend(*args):
            free_pages["full"] -= 1
            free_pages["physical"] -= 1
            return torch.tensor([10, 11])

        def rollback_restore_allocation(indices):
            free_pages["full"] += 1
            free_pages["physical"] += 1

        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=2,
            logical_attn_allocator=logical_allocator,
            hisparse_attn_allocator=physical_allocator,
            alloc_kvduo_restore=MagicMock(side_effect=alloc_extend),
            rollback_restore_allocation=MagicMock(
                side_effect=rollback_restore_allocation
            ),
        )
        load = MagicMock()
        coordinator.mem_pool_device = SimpleNamespace(
            layer_num=2,
            translate_loc_from_full_to_compressed=lambda value: value,
            translate_loc_from_full_to_hisparse_device=lambda value: value,
        )
        coordinator.mem_pool_host = SimpleNamespace(load_to_device_per_layer=load)
        protected_during_reclaim = []

        def reclaim(slots):
            protected_during_reclaim.append(
                coordinator._kvduo_pressure_protected.clone()
            )
            return SimpleNamespace(action=KVDuoPressureAction.SUCCESS)

        coordinator._kvduo_pressure_protected = None
        coordinator._reclaim_for_physical_allocation = MagicMock(side_effect=reclaim)

        radix_reclaimer = MagicMock()

        def evict_radix(full_tokens, swa_tokens):
            # No active request can be reclaimed in this boundary case; the
            # schedulable capacity exists entirely in an evictable Radix page.
            free_pages["full"] = 4
            free_pages["swa"] = 0
            free_pages["physical"] = 1
            return full_tokens, swa_tokens

        radix_reclaimer.side_effect = evict_radix
        all_reduce_count = 0

        def fail_remote_allocation(value, **kwargs):
            nonlocal all_reduce_count
            all_reduce_count += 1
            # Logical preflight and physical reclaim pass on every rank. Only
            # the final allocation fails remotely.
            if all_reduce_count == 3:
                value.zero_()

        before_partial_alloc = dict(free_pages)
        with patch("torch.distributed.all_reduce", side_effect=fail_remote_allocation):
            deferred = coordinator.init_kvduo_load_back(
                req, match.host_hit_length, radix_reclaimer=radix_reclaimer
            )
        self.assertIsNone(deferred)
        radix_reclaimer.assert_called_once_with(2, 0)
        # Restore intentionally owns no SWA page. A successful local rank must
        # return its Full and C4 pages when another TP rank rejects allocation.
        self.assertEqual(free_pages, {"full": 4, "swa": 0, "physical": 1})
        self.assertNotEqual(free_pages, before_partial_alloc)
        rollback = coordinator.token_to_kv_pool_allocator.rollback_restore_allocation
        rollback.assert_called_once()
        self.assertTrue(
            torch.equal(
                rollback.call_args.args[0],
                torch.tensor([10, 11]),
            )
        )
        host_record = coordinator.host_prefix_cache.records[
            coordinator._host_prefix_identity(req, 1)
        ]
        self.assertIn(req.rid, host_record.request_references)
        self.assertNotIn(
            req.rid,
            coordinator.host_prefix_cache.records[
                coordinator._host_prefix_identity(req, 2)
            ].request_references,
        )
        self.assertEqual(host_record.restore_pins, 0)

        coordinator.tp_world_size = 1
        restored = coordinator.init_kvduo_load_back(
            req, match.host_hit_length, radix_reclaimer=radix_reclaimer
        )

        self.assertTrue(torch.equal(restored, torch.tensor([10, 11])))
        self.assertEqual(
            coordinator._reclaim_for_physical_allocation.call_args_list,
            [call(2), call(2)],
        )
        self.assertEqual(protected_during_reclaim[0].tolist(), [5, 6])
        self.assertIsNone(coordinator._kvduo_pressure_protected)
        self.assertEqual(load.call_count, 2)
        self.assertEqual(coordinator.full_data_version[10:12].tolist(), [2, 2])
        self.assertEqual(coordinator.full_last_touch[10:12].tolist(), [9, 10])
        self.assertEqual(
            coordinator.host_prefix_cache.records[
                coordinator._host_prefix_identity(req, 1)
            ].restore_pins,
            0,
        )

    def test_hot_pages_are_allocated_only_for_non_full_topk(self):
        """A full hit owns no hot page; the first host workset grows by a page."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.enable_mixed_residency = True
        coordinator.page_size = 4
        coordinator.device_buffer_size = 16
        coordinator.device = "cpu"
        coordinator.req_to_full_lookup = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int64
        )
        coordinator.mem_pool_device = SimpleNamespace(
            layer_num=2,
            full_to_hisparse_device_index_mapping=torch.tensor(
                [0, 11, 12, 0, 0, 0, 0, 0, 0], dtype=torch.int64
            ),
        )
        coordinator.req_device_buffer_size = torch.zeros(1, dtype=torch.int64)
        coordinator.req_device_buffer_size_gpu = torch.zeros(1, dtype=torch.int32)
        coordinator.kvduo_req_hot_capacity = torch.zeros((2, 1), dtype=torch.int64)
        coordinator.kvduo_req_hot_capacity_gpu = torch.zeros((2, 1), dtype=torch.int32)
        coordinator.req_to_device_buffer = torch.zeros((1, 16), dtype=torch.int64)
        coordinator.req_device_buffer_token_locs = torch.full(
            (2, 1, 16), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_tokens = torch.full(
            (2, 1, 16), -1, dtype=torch.int32
        )
        coordinator.lru_slots = (
            torch.arange(16, dtype=torch.int16).view(1, 1, -1).repeat(2, 1, 1)
        )
        physical = SimpleNamespace(alloc=MagicMock(return_value=torch.arange(21, 25)))
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            hisparse_attn_allocator=physical,
            free_hisparse_indices=MagicMock(),
        )
        coordinator._kvduo_hot_carriers = {}
        coordinator._kvduo_free_layer_pages = [set(), set()]
        coordinator._kvduo_req_layer_pages = {}
        coordinator._reclaim_for_physical_allocation = MagicMock(
            return_value=SimpleNamespace(action=object())
        )
        coordinator._require_allocation_ready = MagicMock()

        # Positions 0 and 1 are full hits, so no hot allocation is needed.
        coordinator._ensure_kvduo_hot_workset(
            torch.tensor([0]), torch.tensor([[0, 1, -1, -1]]), layer_id=1
        )
        physical.alloc.assert_not_called()

        # Positions 2 and 3 have no full mapping.  Two slots round to one page.
        coordinator._ensure_kvduo_hot_workset(
            torch.tensor([0]), torch.tensor([[0, 2, 3, -1]]), layer_id=1
        )
        physical.alloc.assert_called_once_with(4)
        self.assertEqual(int(coordinator.kvduo_req_hot_capacity[1, 0]), 4)
        self.assertTrue(
            torch.equal(
                coordinator.req_device_buffer_token_locs[1, 0, :4],
                torch.arange(21, 25, dtype=torch.int32),
            )
        )
        self.assertTrue(torch.all(coordinator.req_device_buffer_tokens[1, 0, :4] == -1))
        self.assertTrue(
            torch.all(coordinator.req_device_buffer_token_locs[0, 0, :4] == -1)
        )

        # Simulate the resolver publishing the first miss set.  A disjoint
        # later workset must retain these entries and demand a second page,
        # rather than treating one Top-k width as a fixed cache quota.
        coordinator.req_device_buffer_tokens[1, 0, :4] = torch.tensor([2, 3, 6, 7])
        physical.alloc.reset_mock()
        from sglang.srt.mem_cache.sparsity.core.kvduo_state import (
            KVDuoPressureAction,
        )

        coordinator._reclaim_for_physical_allocation.return_value = SimpleNamespace(
            action=KVDuoPressureAction.ERROR
        )
        coordinator._ensure_kvduo_hot_workset(
            torch.tensor([0]), torch.tensor([[4, 5, -1, -1]]), layer_id=1
        )
        physical.alloc.assert_not_called()
        self.assertEqual(int(coordinator.kvduo_req_hot_capacity[1, 0]), 4)

        coordinator._reclaim_for_physical_allocation.return_value = SimpleNamespace(
            action=KVDuoPressureAction.SUCCESS
        )
        physical.alloc.return_value = torch.arange(25, 29)
        coordinator._ensure_kvduo_hot_workset(
            torch.tensor([0]), torch.tensor([[4, 5, 6, 7]]), layer_id=1
        )
        physical.alloc.assert_called_once_with(4)
        self.assertEqual(int(coordinator.kvduo_req_hot_capacity[1, 0]), 8)
        self.assertEqual(
            coordinator.req_device_buffer_tokens[1, 0, :4].tolist(), [2, 3, 6, 7]
        )
        self.assertEqual(
            coordinator.lru_slots[1, 0, :8].tolist(), [4, 5, 6, 7, 0, 1, 2, 3]
        )

        # Shrink and reuse the released layer page. The reactivated range must
        # again be a complete permutation, despite stale values outside the
        # temporarily reduced active capacity.
        coordinator._evict_kvduo_hot_fragment(1, 0, 1)
        self.assertEqual(int(coordinator.kvduo_req_hot_capacity[1, 0]), 4)
        physical.alloc.reset_mock()
        coordinator._ensure_kvduo_hot_workset(
            torch.tensor([0]), torch.tensor([[4, 5, 6, 7]]), layer_id=1
        )
        physical.alloc.assert_not_called()
        self.assertEqual(
            coordinator.lru_slots[1, 0, :8].tolist(), [4, 5, 6, 7, 0, 1, 2, 3]
        )
        self.assertEqual(len(set(coordinator.lru_slots[1, 0, :8].tolist())), 8)

        # Releasing layer 1 does not touch layer 0 metadata.  Once all carrier
        # fragments are free, the carrier coalesces back to the legacy pool.
        coordinator._release_kvduo_layer_hot_pages(0)
        coordinator.token_to_kv_pool_allocator.free_hisparse_indices.assert_called_once()

    def test_graph_prepare_uses_miss_history_and_rejects_zero_capacity(self):
        """Replay preparation grows from counters and never admits mixed/zero."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.enable_mixed_residency = True
        coordinator.decode_producer_stream = None
        coordinator.device = "cpu"
        coordinator.top_k = 2
        coordinator.mem_pool_device = SimpleNamespace(layer_num=1)
        coordinator._active_kvduo_reqs = {0: object()}
        coordinator._mixed_slots = [True]
        coordinator.kvduo_stats_poll_interval = 1
        coordinator._kvduo_replay_count = 0
        coordinator.kvduo_resolver_stats = torch.tensor([[[3, 4]]], dtype=torch.int32)
        coordinator.kvduo_req_hot_capacity = torch.tensor([[4]], dtype=torch.int64)
        coordinator._ensure_kvduo_hot_capacity_targets = MagicMock(return_value=True)
        coordinator._ensure_kvduo_hot_capacity_targets_batch = MagicMock(
            return_value=True
        )

        coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))

        self.assertEqual(
            coordinator._ensure_kvduo_hot_capacity_targets.call_args.args[1],
            {0: 8},
        )
        self.assertTrue(torch.all(coordinator.kvduo_resolver_stats == 0))

        coordinator.kvduo_req_hot_capacity.zero_()
        with self.assertRaisesRegex(RuntimeError, "mandatory 2K"):
            coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))

    def test_graph_prepare_ignores_incidental_misses(self):
        """A sparse miss signal cannot cascade through every optional tier."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.enable_mixed_residency = True
        coordinator.decode_producer_stream = None
        coordinator.top_k = 2
        coordinator.mem_pool_device = SimpleNamespace(layer_num=1)
        owner = object()
        coordinator._active_kvduo_reqs = {0: owner}
        coordinator._mixed_slots = [True]
        coordinator.kvduo_stats_poll_interval = 1
        coordinator._kvduo_replay_count = 0
        coordinator.kvduo_resolver_stats = torch.tensor([[[1, 100]]], dtype=torch.int32)
        coordinator.kvduo_req_hot_capacity = torch.tensor([[4]], dtype=torch.int64)
        coordinator._ensure_kvduo_hot_capacity_targets = MagicMock()

        coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))

        coordinator._ensure_kvduo_hot_capacity_targets.assert_not_called()

    def test_graph_prepare_grows_hot_capacity_linearly_and_stops_at_16k(self):
        """Page allocation preserves the 4K -> 6K -> 8K progression."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.device = "cpu"
        coordinator.page_size = 2
        coordinator.top_k = 2
        coordinator.device_buffer_size = 32
        coordinator.mem_pool_device = SimpleNamespace(layer_num=1)
        coordinator._mixed_slots = [True]
        coordinator.kvduo_req_hot_capacity = torch.tensor([[4]], dtype=torch.int64)
        coordinator.kvduo_req_hot_capacity_gpu = torch.tensor([[4]], dtype=torch.int32)
        coordinator.req_device_buffer_token_locs = torch.full(
            (1, 1, 32), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_tokens = torch.full(
            (1, 1, 32), -1, dtype=torch.int32
        )
        coordinator.lru_slots = torch.arange(32, dtype=torch.int16).view(1, 1, -1)
        coordinator._kvduo_req_layer_pages = {(0, 0): [10, 12]}
        coordinator._kvduo_hot_carriers = {start: [0] for start in (10, 12)} | {
            start: [None] for start in range(14, 44, 2)
        }
        coordinator._kvduo_free_layer_pages = [set(range(14, 44, 2))]

        capacities = []
        physical_page_counts = []
        for target in (8, 12, 16):
            coordinator._ensure_kvduo_hot_capacity_targets(0, {0: target})
            capacities.append(int(coordinator.kvduo_req_hot_capacity[0, 0]))
            physical_page_counts.append(len(coordinator._kvduo_req_layer_pages[(0, 0)]))

        self.assertEqual(capacities, [8, 12, 16])
        self.assertEqual(physical_page_counts, [4, 6, 8])

        coordinator._ensure_kvduo_hot_capacity_targets(0, {0: 32})
        self.assertEqual(int(coordinator.kvduo_req_hot_capacity[0, 0]), 32)
        self.assertEqual(len(coordinator._kvduo_req_layer_pages[(0, 0)]), 16)
        with self.assertRaisesRegex(RuntimeError, "exceeds fixed 16K"):
            coordinator._ensure_kvduo_hot_capacity_targets(0, {0: 36})

    def test_optional_hot_growth_never_reclaims_full_pages(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.page_size = 4
        coordinator.top_k = 2
        coordinator.device_buffer_size = 32
        coordinator._mixed_slots = [True]
        coordinator.kvduo_req_hot_capacity = torch.tensor([[4]], dtype=torch.int64)
        coordinator._kvduo_free_layer_pages = [set()]
        physical = SimpleNamespace(available_size=MagicMock(return_value=0))
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            hisparse_attn_allocator=physical
        )
        coordinator._reclaim_for_physical_allocation = MagicMock()

        coordinator._ensure_kvduo_hot_capacity_targets(0, {0: 8}, allow_reclaim=False)

        coordinator._reclaim_for_physical_allocation.assert_not_called()
        self.assertEqual(int(coordinator.kvduo_req_hot_capacity[0, 0]), 4)

    def test_graph_prepare_rechecks_batch_requests_demoted_by_pressure(self):
        """A full batch peer demoted during growth receives 2K before replay."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.enable_mixed_residency = True
        coordinator.decode_producer_stream = None
        coordinator.device = "cpu"
        coordinator.top_k = 2
        coordinator.mem_pool_device = SimpleNamespace(layer_num=1)
        coordinator._active_kvduo_reqs = {0: object(), 1: object(), 2: object()}
        coordinator._mixed_slots = [True, False, False]
        coordinator.kvduo_stats_poll_interval = 1
        coordinator._kvduo_replay_count = 0
        coordinator.kvduo_resolver_stats = torch.zeros((1, 3, 2), dtype=torch.int32)
        coordinator.kvduo_req_hot_capacity = torch.tensor(
            [[0, 0, 0]], dtype=torch.int64
        )
        targets = []

        def allocate(_, requested_capacities):
            targets.append(dict(requested_capacities))
            if len(targets) == 1:
                coordinator.kvduo_req_hot_capacity[0, 0] = 4
                coordinator._mixed_slots[1] = True
            elif len(targets) == 2:
                coordinator.kvduo_req_hot_capacity[0, 1] = 4
                coordinator._mixed_slots[2] = True
            else:
                coordinator.kvduo_req_hot_capacity[0, 2] = 4
            return True

        coordinator._ensure_kvduo_hot_capacity_targets_batch = MagicMock(
            side_effect=lambda targets: allocate(
                None, {req_idx: target for (_, req_idx), target in targets.items()}
            )
        )

        coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1, 2]))

        self.assertEqual(targets, [{0: 4}, {1: 4}, {2: 4}])
        self.assertEqual(int(coordinator.kvduo_req_hot_capacity[0, 1]), 4)
        self.assertEqual(int(coordinator.kvduo_req_hot_capacity[0, 2]), 4)

    def test_graph_prepare_batches_two_requests_across_21_layers(self):
        """One fixed-point round uses one allocation and one page-id read."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.enable_mixed_residency = True
        coordinator.decode_producer_stream = None
        coordinator.device = "cpu"
        coordinator.page_size = 1
        coordinator.top_k = 1
        coordinator.device_buffer_size = 8
        coordinator.mem_pool_device = SimpleNamespace(
            layer_num=21, flat_layer_slot_stride=1000
        )
        coordinator.hot_storage_layers = 1
        coordinator.hot_page_size = 1
        coordinator.min_hot_pages = 1
        coordinator._active_kvduo_reqs = {0: object(), 1: object()}
        coordinator._mixed_slots = [True, True]
        coordinator._kvduo_pending_hot_minimum = {0, 1}
        coordinator.kvduo_stats_poll_interval = 8
        coordinator._kvduo_replay_count = 0
        coordinator.kvduo_resolver_stats = torch.zeros((21, 2, 2), dtype=torch.int32)
        coordinator.kvduo_req_hot_capacity = torch.zeros((21, 2), dtype=torch.int64)
        coordinator.kvduo_req_hot_capacity_gpu = torch.zeros((21, 2), dtype=torch.int32)
        coordinator.req_device_buffer_token_locs = torch.full(
            (21, 2, 8), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_tokens = torch.full(
            (21, 2, 8), -1, dtype=torch.int32
        )
        coordinator.lru_slots = torch.arange(8, dtype=torch.int16).repeat(21, 2, 1)
        coordinator._kvduo_req_layer_pages = {}
        coordinator._kvduo_hot_page_owners = {}
        coordinator._reclaim_for_physical_allocation = MagicMock(
            return_value=SimpleNamespace(action=object())
        )
        coordinator._require_allocation_ready = MagicMock()
        physical = SimpleNamespace(
            alloc=MagicMock(return_value=torch.arange(42)),
            available_size=MagicMock(return_value=42),
        )
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            hisparse_attn_allocator=physical,
            free_hisparse_indices=MagicMock(),
        )

        coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1]))

        physical.alloc.assert_called_once_with(42)
        self.assertTrue(torch.all(coordinator.kvduo_req_hot_capacity == 1))
        self.assertEqual(len(coordinator._kvduo_hot_page_owners), 42)
        self.assertEqual(
            set(coordinator._kvduo_hot_page_owners.values()),
            {(layer_id, req_idx) for layer_id in range(21) for req_idx in range(2)},
        )

    def test_batched_hot_allocation_failure_publishes_no_partial_state(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.page_size = 1
        coordinator.hot_page_size = 1
        coordinator.device_buffer_size = 8
        coordinator.kvduo_req_hot_capacity = torch.zeros((2, 2), dtype=torch.int64)
        coordinator._kvduo_req_layer_pages = {}
        coordinator._kvduo_hot_page_owners = {}
        coordinator.mem_pool_device = SimpleNamespace(layer_num=2)
        allocator = SimpleNamespace(
            available_size=MagicMock(return_value=4),
            alloc=MagicMock(return_value=None),
        )
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            hisparse_attn_allocator=allocator
        )
        coordinator._reclaim_for_physical_allocation = MagicMock(
            return_value=SimpleNamespace(action=object())
        )
        coordinator._require_allocation_ready = MagicMock()

        with self.assertRaisesRegex(RuntimeError, "whole-page HOT allocation failed"):
            coordinator._ensure_kvduo_hot_capacity_targets_batch(
                {
                    (layer_id, req_idx): 1
                    for layer_id in range(2)
                    for req_idx in range(2)
                }
            )

        self.assertEqual(coordinator._kvduo_hot_page_owners, {})
        self.assertEqual(coordinator._kvduo_req_layer_pages, {})
        self.assertTrue(torch.all(coordinator.kvduo_req_hot_capacity == 0))

    def test_batched_hot_post_allocation_exception_rolls_back(self):
        """A failure preparing the second pair frees pages and publishes nothing."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.page_size = 1
        coordinator.hot_page_size = 1
        coordinator.hot_storage_layers = 1
        coordinator.device_buffer_size = 8
        coordinator.device = "cpu"
        coordinator.mem_pool_device = SimpleNamespace(
            layer_num=2, flat_layer_slot_stride=8
        )
        coordinator.kvduo_req_hot_capacity = torch.zeros((2, 1), dtype=torch.int64)
        coordinator.kvduo_req_hot_capacity_gpu = torch.zeros((2, 1), dtype=torch.int32)
        coordinator.req_device_buffer_token_locs = torch.full(
            (2, 1, 8), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_tokens = torch.full(
            (2, 1, 8), -1, dtype=torch.int32
        )
        coordinator.lru_slots = torch.arange(8, dtype=torch.int16).repeat(2, 1, 1)
        coordinator._kvduo_req_layer_pages = {}
        coordinator._kvduo_hot_page_owners = {}
        physical = torch.arange(2)
        allocator = SimpleNamespace(
            available_size=MagicMock(return_value=2),
            alloc=MagicMock(return_value=physical),
        )
        free = MagicMock()
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            hisparse_attn_allocator=allocator, free_hisparse_indices=free
        )
        coordinator._reclaim_for_physical_allocation = MagicMock(
            return_value=SimpleNamespace(action=object())
        )
        coordinator._require_allocation_ready = MagicMock()
        publish = coordinator._publish_kvduo_hot_growth_plan
        publish_calls = 0

        def fail_during_second_publish(*args):
            nonlocal publish_calls
            publish_calls += 1
            if publish_calls == 2:
                raise RuntimeError("injected")
            return publish(*args)

        with patch.object(
            coordinator,
            "_publish_kvduo_hot_growth_plan",
            side_effect=fail_during_second_publish,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                coordinator._ensure_kvduo_hot_capacity_targets_batch(
                    {(0, 0): 1, (1, 0): 1}
                )

        free.assert_called_once_with(physical)
        self.assertEqual(coordinator._kvduo_hot_page_owners, {})
        self.assertEqual(coordinator._kvduo_req_layer_pages, {})
        self.assertTrue(torch.all(coordinator.kvduo_req_hot_capacity == 0))
        self.assertTrue(torch.all(coordinator.kvduo_req_hot_capacity_gpu == 0))
        self.assertTrue(torch.all(coordinator.req_device_buffer_token_locs == -1))
        self.assertTrue(torch.all(coordinator.req_device_buffer_tokens == -1))
        self.assertTrue(
            torch.equal(
                coordinator.lru_slots,
                torch.arange(8, dtype=torch.int16).repeat(2, 1, 1),
            )
        )

    def test_graph_prepare_polls_growth_every_eight_replays(self):
        """Miss statistics accumulate without affecting mandatory worksets."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.enable_mixed_residency = True
        coordinator.decode_producer_stream = None
        coordinator.device = "cpu"
        coordinator.top_k = 2
        coordinator.mem_pool_device = SimpleNamespace(layer_num=1)
        owner = object()
        coordinator._active_kvduo_reqs = {0: owner}
        coordinator._mixed_slots = [True]
        coordinator.kvduo_stats_poll_interval = 8
        coordinator._kvduo_replay_count = 0
        coordinator.kvduo_resolver_stats = torch.tensor([[[3, 4]]], dtype=torch.int32)
        coordinator.kvduo_req_hot_capacity = torch.tensor([[4]], dtype=torch.int64)
        coordinator._ensure_kvduo_hot_capacity_targets = MagicMock(return_value=True)

        for _ in range(7):
            coordinator.prepare_kvduo_graph_replay(
                torch.tensor([99]), req_pool_indices_cpu=torch.tensor([0])
            )
        coordinator._ensure_kvduo_hot_capacity_targets.assert_not_called()
        self.assertEqual(int(coordinator.kvduo_resolver_stats[0, 0, 0]), 3)

        coordinator.prepare_kvduo_graph_replay(
            torch.tensor([99]), req_pool_indices_cpu=torch.tensor([0])
        )
        self.assertEqual(
            coordinator._ensure_kvduo_hot_capacity_targets.call_args.args[1],
            {0: 8},
        )
        self.assertTrue(torch.all(coordinator.kvduo_resolver_stats == 0))

    def test_graph_prepare_skips_steady_state_capacity_scan(self):
        """Established hot tiers add no per-layer Python work per replay."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.enable_mixed_residency = True
        coordinator.decode_producer_stream = None
        coordinator.top_k = 2
        coordinator.mem_pool_device = SimpleNamespace(layer_num=64)
        coordinator._active_kvduo_reqs = {0: object()}
        coordinator._mixed_slots = [True]
        coordinator._kvduo_pending_hot_minimum = set()
        coordinator.kvduo_stats_poll_interval = 8
        coordinator._kvduo_replay_count = 0
        coordinator.kvduo_resolver_stats = torch.zeros((64, 1, 2), dtype=torch.int32)
        # Any layer/capacity scan would fail this test before reaching a mock.
        coordinator.kvduo_req_hot_capacity = None
        coordinator._ensure_kvduo_hot_capacity_targets = MagicMock()

        coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))

        coordinator._ensure_kvduo_hot_capacity_targets.assert_not_called()

    def test_reclaims_only_dedicated_physical_shortfall(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.enable_mixed_residency = True
        coordinator.item_size_bytes = 4
        coordinator.page_size = 1
        coordinator.mem_pool_device = SimpleNamespace(layer_num=2)
        physical = SimpleNamespace(available_size=MagicMock(side_effect=(3, 8)))
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            hisparse_attn_allocator=physical,
            # A deliberately unrelated composite value must never be consulted.
            available_size=MagicMock(return_value=0),
        )
        coordinator.reclaim_kvduo_full_pages = MagicMock(return_value=5)
        coordinator.reclaim_kvduo_hot_pages = MagicMock(return_value=0)

        coordinator._reclaim_for_physical_allocation(8)

        self.assertEqual(physical.available_size.call_count, 2)
        coordinator.token_to_kv_pool_allocator.available_size.assert_not_called()
        coordinator.reclaim_kvduo_full_pages.assert_called_once_with(5)
        coordinator.reclaim_kvduo_hot_pages.assert_called_once_with(3)

    def test_does_not_reclaim_when_physical_pool_fits_reserved_page(self):
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.enable_mixed_residency = True
        coordinator.item_size_bytes = 4
        coordinator.page_size = 1
        coordinator.mem_pool_device = SimpleNamespace(layer_num=2)
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            hisparse_attn_allocator=SimpleNamespace(
                available_size=MagicMock(return_value=65)
            )
        )
        coordinator.reclaim_kvduo_full_pages = MagicMock()
        coordinator.reclaim_kvduo_hot_pages = MagicMock()

        coordinator._reclaim_for_physical_allocation(65)

        coordinator.reclaim_kvduo_full_pages.assert_not_called()
        coordinator.reclaim_kvduo_hot_pages.assert_not_called()


def _make_req(rid="test-req-0", origin_input_ids=None, output_ids=None):
    """Create a minimal mock Req object with the fields HiSparseCoordinator uses."""
    if origin_input_ids is None:
        origin_input_ids = list(range(64))
    if output_ids is None:
        output_ids = []
    req = SimpleNamespace(
        rid=rid,
        origin_input_ids=origin_input_ids,
        output_ids=output_ids,
        fill_ids=origin_input_ids + output_ids,
        seqlen=len(origin_input_ids) + len(output_ids),
        req_pool_idx=None,
        kv=SimpleNamespace(kv_allocated_len=0),
        kv_committed_len=0,
        finished_reason=None,
        hisparse_staging=False,
        staging=False,
        inflight_middle_chunks=0,
    )
    req.finished = lambda: req.finished_reason is not None
    req.set_extend_range = lambda start, end: setattr(
        req, "extend_range", Range(start, end)
    )
    return req


class TestHiSparseUnit(unittest.TestCase):
    """Test class that builds a minimal HiSparse component stack."""

    # ==================================================================
    # Fixture
    # ==================================================================

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for HiSparse tests.")
        if is_npu() or is_xpu():
            raise unittest.SkipTest("HiSparse tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            raise unittest.SkipTest("CUDA/ROCm not available.")

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29599")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        cls.tp_group = torch.distributed.group.WORLD

        from sglang.srt.mem_cache.pool_host.common import (
            ALLOC_MEMORY_FUNCS,
            alloc_with_pin_memory,
        )

        cls._original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory

        if is_hip():
            from sglang.srt.layers.attention.dsa.utils import (
                aiter_can_use_preshuffle_paged_mqa,
            )

            global_page_size = 64 if aiter_can_use_preshuffle_paged_mqa() else 1
        else:
            global_page_size = PAGE_SIZE

        from sglang.srt.mem_cache.allocator.hisparse import (
            HiSparseTokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool

        cls.device_pool = HiSparseDSATokenToKVPool(
            size=SIZE,
            page_size=global_page_size,
            kv_lora_rank=KV_LORA_RANK,
            dtype=torch.bfloat16,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=LAYER_NUM,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=KV_CACHE_DIM,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )
        cls.allocator = HiSparseTokenToKVPoolAllocator(
            size=SIZE,
            page_size=global_page_size,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=cls.device_pool,
            need_sort=False,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )

        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        cls.req_to_token_pool = ReqToTokenPool(
            size=MAX_NUM_REQS,
            max_context_len=MAX_CONTEXT_LEN,
            device="cuda",
            enable_memory_saver=False,
        )

        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        cls.page_size = global_page_size
        cls.coordinator = HiSparseCoordinator(
            req_to_token_pool=cls.req_to_token_pool,
            token_to_kv_pool_allocator=cls.allocator,
            top_k=TOP_K,
            device_buffer_size=DEVICE_BUFFER_SIZE,
            device="cuda",
            tp_group=cls.tp_group,
            host_to_device_ratio=HOST_TO_DEVICE_RATIO,
        )

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS

        ALLOC_MEMORY_FUNCS["cuda"] = cls._original_alloc
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def setUp(self):
        """Reset shared allocator / coordinator state so tests are isolated.

        Without this, a mid-test assertion failure skips cleanup and leaks
        resources, causing unrelated failures in later tests.
        """
        self.allocator.clear()
        self.req_to_token_pool.clear()
        self.coordinator.mem_pool_host.clear()
        # Reset per-request coordinator bookkeeping
        self.coordinator.req_to_device_buffer.zero_()
        self.coordinator.req_device_buffer_size.zero_()
        self.coordinator.req_to_host_pool.fill_(-1)
        self.coordinator.req_to_host_pool_allocated_len.zero_()
        self.coordinator.req_device_buffer_tokens.fill_(-1)
        self.coordinator.req_device_buffer_token_locs.fill_(-1)
        self.coordinator.lru_slots[:] = self.coordinator._lru_init.view(1, 1, -1)
        self.coordinator.ack_staging_queue.clear()
        self.coordinator._has_pending_backup = False
        for i in range(len(self.coordinator._skip_first_backup)):
            self.coordinator._skip_first_backup[i] = False

    # ==================================================================
    # Low-level helpers
    # ==================================================================

    def _alloc_req_slot(self, req):
        """Allocate a req_pool_idx for the request."""
        indices = self.req_to_token_pool.alloc([req])
        self.assertIsNotNone(indices, "Failed to allocate req pool slot")
        return req.req_pool_idx

    def _free_req_slot(self, req):
        """Free the req_pool_idx."""
        if req.req_pool_idx is not None:
            self.req_to_token_pool.free(req)

    def _alloc_kv(self, req, fill_len, *, logical_only=False):
        """Allocate KV indices, write req_to_token_pool, update req fields.
        If logical_only=True, uses alloc_logical_only (PD-separated path).
        Returns kv_loc tensor."""
        device = self.allocator.device
        alloc_fn = (
            self.allocator.alloc_logical_only
            if logical_only
            else self.allocator.alloc_extend
        )
        kv_loc = alloc_fn(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=fill_len,
        )
        self.assertIsNotNone(kv_loc, "KV alloc failed")
        self.req_to_token_pool.write((req.req_pool_idx, slice(0, len(kv_loc))), kv_loc)
        req.kv.kv_allocated_len = fill_len
        req.kv_committed_len = fill_len
        req.full_untruncated_fill_ids = array("q", range(fill_len))
        req.extend_range = Range(0, fill_len)
        return kv_loc

    # ==================================================================
    # Mid-level helpers
    # ==================================================================

    @staticmethod
    def _kv_pattern(layer_id, token_id):
        """Deterministic KV value for (layer, token) — used by write & verify."""
        v = (layer_id * 10000 + token_id + 1) * 0.001
        return float(torch.tensor(v, dtype=torch.bfloat16))

    def _write_device_patterns(self, kv_loc, fill_len):
        """Write distinguishable patterns into device KV buffer for all layers.

        kv_loc contains *logical* indices; we must translate them to hisparse
        device indices before indexing kv_buffer (which is sized for the
        hisparse pool, not the larger logical space).
        """
        hisparse_locs = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        for lid in range(LAYER_NUM):
            for i in range(fill_len):
                self.device_pool.kv_buffer[lid][hisparse_locs[i]] = self._kv_pattern(
                    lid, i
                )

    def _populate_host_pool(self, req, fill_len):
        """Allocate host slots, write known patterns, register in coordinator.
        Returns host_indices (cuda tensor)."""
        host_pool = self.coordinator.mem_pool_host
        host_indices = host_pool.alloc(fill_len)
        self.assertIsNotNone(host_indices, "Host alloc failed")
        host_indices = host_indices.to(device="cuda")
        self.coordinator.req_to_host_pool[req.req_pool_idx, :fill_len] = host_indices
        self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx] = fill_len
        for lid in range(LAYER_NUM):
            for i in range(fill_len):
                host_pool.kv_buffer[lid][host_indices[i]] = self._kv_pattern(lid, i)
        return host_indices

    def _build_topk_tokens(self, fill_len, *, include_newest=False):
        """Build a 1-D [TOP_K] int32 cuda tensor of token positions.

        If include_newest=True, fill_len-1 is guaranteed as the last valid slot.
        Pads with -1 when fill_len (or fill_len-1) < TOP_K.

        For long-sequence tests (fill_len > DEVICE_BUFFER_SIZE) where the
        "newest token" reserved slot is not populated (it requires an actual
        decode step + map_last_loc_to_buffer), callers should pass
        ``fill_len - 1`` as the effective pool size so position fill_len-1 is
        never randomly selected.
        """
        n = min(fill_len, TOP_K)
        if include_newest and n > 1:
            tokens = torch.randperm(fill_len - 1, device="cuda")[: n - 1].to(
                torch.int32
            )
            tokens = torch.cat(
                [tokens, torch.tensor([fill_len - 1], dtype=torch.int32, device="cuda")]
            )
        else:
            tokens = torch.randperm(fill_len, device="cuda")[:n].to(torch.int32)
        if n < TOP_K:
            pad = torch.full((TOP_K - n,), -1, dtype=torch.int32, device="cuda")
            tokens = torch.cat([tokens, pad])
        return tokens

    def _make_batch_tensors(self, reqs, fill_lens):
        """Build (req_pool_indices [int64], seq_lens [int32]) on cuda."""
        rpi = torch.tensor(
            [r.req_pool_idx for r in reqs], dtype=torch.int64, device="cuda"
        )
        sls = torch.tensor(fill_lens, dtype=torch.int32, device="cuda")
        return rpi, sls

    def _assert_kv_correct(self, locs_row, tokens_row, layer_id, count, msg=""):
        """Assert device KV data at *locs_row[:count]* matches the written
        pattern for the corresponding *tokens_row[:count]* positions."""
        for i in range(count):
            tok = int(tokens_row[i].item())
            if tok < 0:
                continue
            expected = self._kv_pattern(layer_id, tok)
            actual = self.device_pool.kv_buffer[layer_id][locs_row[i].long()]
            self.assertTrue(
                torch.allclose(
                    actual.float(),
                    torch.full_like(actual.float(), expected),
                    atol=1e-2,
                ),
                f"{msg}layer {layer_id}, token {tok}: KV data mismatch",
            )

    def _assert_matches_naive(self, rpi, sls, batch, kernel_locs, layer_id, msg=""):
        """Assert kernel swap_in KV data matches naive_load_topk KV data."""
        naive_locs = self.coordinator.naive_load_topk(rpi, sls, batch, layer_id)
        for b in range(batch.shape[0]):
            for i in range(TOP_K):
                if batch[b, i] < 0:
                    continue
                naive_data = self.device_pool.kv_buffer[layer_id][
                    naive_locs[b, i].long()
                ]
                kernel_data = self.device_pool.kv_buffer[layer_id][
                    kernel_locs[b, i].long()
                ]
                self.assertTrue(
                    torch.allclose(naive_data.float(), kernel_data.float(), atol=1e-2),
                    f"{msg}layer {layer_id}, b{b} idx {i}: naive != kernel",
                )

    def _swap_in_selected_pages(
        self,
        rpi: torch.Tensor,
        sls: torch.Tensor,
        batch: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Wrapper that sets num_real_reqs before calling swap_in_selected_pages.

        In production, model_runner sets num_real_reqs before each forward
        pass.  Tests must replicate that to get correct kernel behaviour.
        """
        self.coordinator.num_real_reqs[0] = rpi.shape[0]
        return self.coordinator.swap_in_selected_pages(rpi, sls, batch, layer_id)

    def _cleanup_req(self, req, kv_loc, *, logical_only=False):
        """request_finished -> free KV -> free req slot."""
        self.coordinator.request_finished(req)
        if logical_only:
            self.allocator.logical_attn_allocator.free(kv_loc)
        else:
            self.allocator.free(kv_loc)
        self._free_req_slot(req)

    def _get_initial_sizes(self):
        """Snapshot allocator available sizes."""
        return (
            self.allocator.logical_attn_allocator.available_size(),
            self.allocator.hisparse_attn_allocator.available_size(),
            self.coordinator.mem_pool_host.available_size(),
        )

    def _assert_sizes_restored(self, initial_sizes, msg=""):
        """Assert allocator sizes match the snapshot."""
        logical, hisparse, host = self._get_initial_sizes()
        self.assertEqual(logical, initial_sizes[0], f"Logical leak {msg}")
        self.assertEqual(hisparse, initial_sizes[1], f"HiSparse leak {msg}")
        self.assertEqual(host, initial_sizes[2], f"Host leak {msg}")

    # ==================================================================
    # Test: Kernel correctness — short sequence (fast path)
    # ==================================================================
    def test_kernel_correctness_short_seq(self):
        """Short seq (len <= device_buffer_size): kernel fast path returns
        device buffer locs, matching naive_load_topk."""
        initial = self._get_initial_sizes()
        req = _make_req("short-seq", list(range(self.page_size)))
        self._alloc_req_slot(req)

        fill_len = self.page_size
        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)
        self.coordinator.alloc_device_buffer(req)

        tokens = self._build_topk_tokens(fill_len)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        for lid in range(LAYER_NUM):
            naive_locs = self.coordinator.naive_load_topk(rpi, sls, batch, lid)
            kernel_locs = self._swap_in_selected_pages(rpi, sls, batch, lid)
            valid = batch[0] >= 0
            self.assertTrue(
                torch.equal(naive_locs[0][valid].cpu(), kernel_locs[0][valid].cpu()),
                f"Layer {lid}: kernel locs != naive oracle",
            )

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "short_seq")

    # ==================================================================
    # Test: Kernel correctness — long sequence (cache miss + host DMA)
    # ==================================================================
    def test_kernel_correctness_long_seq(self):
        """Long seq (len > device_buffer_size): kernel loads from host,
        matching naive_load_topk for data correctness."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("long-seq", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        # Pass fill_len-1 so position fill_len-1 ("newest token") is never
        # randomly selected — its reserved device-buffer slot is only valid
        # after map_last_loc_to_buffer in a real decode step.
        tokens = self._build_topk_tokens(fill_len - 1)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        for lid in range(LAYER_NUM):
            naive_locs = self.coordinator.naive_load_topk(rpi, sls, batch, lid)
            kernel_locs = self._swap_in_selected_pages(rpi, sls, batch, lid)
            self.assertTrue(torch.all(naive_locs[0, :TOP_K] >= 0))
            self.assertTrue(torch.all(kernel_locs[0, :TOP_K] >= 0))
            # Verify both return correct KV data independently
            self._assert_kv_correct(naive_locs[0], tokens, lid, TOP_K, msg="Naive: ")
            self._assert_kv_correct(kernel_locs[0], tokens, lid, TOP_K, msg="Kernel: ")

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "long_seq")

    # ==================================================================
    # Test: Kernel LRU replacement across multiple decode steps
    # ==================================================================
    def test_kernel_lru_replacement(self):
        """Multi-step swap-in: second call hits cached tokens, only
        evicts/loads new misses."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size * 2
        req = _make_req("lru-test", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        rpi, sls = self._make_batch_tensors([req], [fill_len])

        # Step 1: load the first TOP_K positions from host (no newest token —
        # the reserved slot is only valid after map_last_loc_to_buffer which is
        # called during an actual decode step, not modelled here).
        tokens_s1 = torch.arange(TOP_K, dtype=torch.int32, device="cuda")
        locs1 = self._swap_in_selected_pages(
            rpi, sls, tokens_s1.unsqueeze(0), layer_id=0
        )
        self.assertTrue(torch.all(locs1[0, :TOP_K] >= 0))

        # Step 2: half overlap (hit) + half new (miss).
        # Choose new tokens from a range safely below fill_len.
        half = TOP_K // 2
        new_start = TOP_K  # first position not in step-1
        tokens_s2 = torch.cat(
            [
                tokens_s1[:half],  # hits
                torch.arange(
                    new_start, new_start + half, dtype=torch.int32, device="cuda"
                ),  # misses
            ]
        )
        locs2 = self._swap_in_selected_pages(
            rpi, sls, tokens_s2.unsqueeze(0), layer_id=0
        )
        self.assertTrue(torch.all(locs2[0, :TOP_K] >= 0))

        # Verify repeated (hit) tokens still have correct KV data
        self._assert_kv_correct(
            locs2[0], tokens_s2, layer_id=0, count=half, msg="LRU hit: "
        )
        # Also verify new (miss) tokens loaded correctly
        self._assert_kv_correct(
            locs2[0, half:],
            tokens_s2[half:],
            layer_id=0,
            count=half,
            msg="LRU miss: ",
        )

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "lru_replacement")

    # ==================================================================
    # Test: Allocator alloc/free lifecycle
    # ==================================================================
    def test_allocator_alloc_free_cycle(self):
        """alloc_extend / alloc_device_buffer / free restores available_size."""
        initial = self._get_initial_sizes()
        device = self.allocator.device
        fill_len = self.page_size * 2

        kv_loc = self.allocator.alloc_extend(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([fill_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([fill_len], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=fill_len,
        )
        self.assertIsNotNone(kv_loc)
        self.assertEqual(len(kv_loc), fill_len)

        mapping = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping > 0), "Mapping should be non-zero")
        self.assertLess(self.allocator.available_size(), initial[0])

        need_size = min(
            ((fill_len + self.page_size - 1) // self.page_size) * self.page_size,
            DEVICE_BUFFER_SIZE,
        )
        buf_idx = self.allocator.alloc_device_buffer(kv_loc, need_size)
        self.assertIsNotNone(buf_idx)
        mapping_after = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping_after == 0), "Mapping should be cleared")

        self.allocator.free_hisparse_indices(buf_idx)
        self.allocator.logical_attn_allocator.free(kv_loc)
        self._assert_sizes_restored(initial, "alloc_free_cycle")

    def test_allocator_page_size_one_alloc_free_cycle(self):
        """alloc() maps logical to hisparse indices for ROCm page_size=1."""
        if self.page_size != 1:
            self.skipTest("page_size=1 alloc path is ROCm-specific")

        initial = self._get_initial_sizes()
        need_size = 16

        kv_loc = self.allocator.alloc(need_size)
        self.assertIsNotNone(kv_loc)
        self.assertEqual(len(kv_loc), need_size)

        mapping = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping > 0), "Mapping should be non-zero")
        self.assertLess(self.allocator.available_size(), initial[0])

        self.allocator.free(kv_loc)
        mapping_after = self.allocator.full_to_hisparse_device_index_mapping[kv_loc]
        self.assertTrue(torch.all(mapping_after == 0), "Mapping should be cleared")
        self._assert_sizes_restored(initial, "page_size_one_alloc_free_cycle")

    def test_decode_remap_frees_stale_page_size_one_mapping(self):
        """map_last_loc_to_buffer frees the temporary alloc() hisparse slot."""
        if self.page_size != 1:
            self.skipTest("page_size=1 decode remap path is ROCm-specific")

        initial = self._get_initial_sizes()
        device = self.allocator.device
        fill_len = 2
        req = _make_req("decode-remap", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self.coordinator.alloc_device_buffer(req)
        self.coordinator._skip_first_backup[req.req_pool_idx] = True

        out_loc = self.allocator.alloc(1)
        self.assertIsNotNone(out_loc)
        stale_loc = self.allocator.full_to_hisparse_device_index_mapping[
            out_loc
        ].clone()
        self.assertTrue(torch.all(stale_loc > 0), "Temporary mapping should exist")

        seq_len = fill_len + 1
        self.req_to_token_pool.write((req.req_pool_idx, fill_len), out_loc)
        req.kv.kv_allocated_len = seq_len
        req.kv_committed_len = seq_len

        self.coordinator.map_last_loc_to_buffer(
            seq_lens=torch.tensor([seq_len], dtype=torch.int64, device=device),
            out_cache_loc=out_loc,
            req_pool_indices=torch.tensor(
                [req.req_pool_idx], dtype=torch.int64, device=device
            ),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([req.req_pool_idx], dtype=torch.int64),
        )

        remapped_loc = self.allocator.full_to_hisparse_device_index_mapping[out_loc]
        self.assertTrue(torch.all(remapped_loc > 0), "Remapped loc should exist")
        self.assertFalse(
            torch.equal(stale_loc, remapped_loc),
            "Decode loc should move from temporary mapping to device buffer",
        )
        self.assertEqual(
            self.allocator.hisparse_attn_allocator.available_size(),
            initial[1] - seq_len,
        )

        self.coordinator.request_finished(req)
        self.allocator.logical_attn_allocator.free(torch.cat([kv_loc, out_loc]))
        self._free_req_slot(req)
        self._assert_sizes_restored(initial, "decode_remap")

    # ==================================================================
    # Test: Staging (PD Colocate) path
    # ==================================================================
    def test_request_lifecycle_staging_path(self):
        """prefill -> staging DMA -> collect_ready -> swap-in -> finish."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size
        req = _make_req("staging-req", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)

        self.coordinator.admit_request_into_staging(req)
        self.assertTrue(req.hisparse_staging)

        torch.cuda.synchronize()
        ready = self.coordinator.collect_ready_reqs()
        self.assertEqual(len(ready), 1)
        self.assertFalse(req.hisparse_staging)
        self.assertTrue(self.coordinator._skip_first_backup[req.req_pool_idx])

        tokens = self._build_topk_tokens(fill_len)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        locs = self._swap_in_selected_pages(rpi, sls, batch, layer_id=0)
        valid_n = min(fill_len, TOP_K)
        self.assertTrue(torch.all(locs[0, :valid_n] >= 0))
        self._assert_kv_correct(
            locs[0], tokens, layer_id=0, count=valid_n, msg="Staging: "
        )
        self._assert_matches_naive(rpi, sls, batch, locs, layer_id=0, msg="Staging: ")

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "staging_path")

    # ==================================================================
    # Test: Single-node staging host page allocation
    # ==================================================================
    def test_single_node_staging_allocates_paged_host_slots(self):
        """Single-node staging should allocate host slots at page granularity."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size * 2 + 1
        rounded_len = (fill_len + self.page_size - 1) // self.page_size * self.page_size
        req = _make_req("single-node-staging-pages", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len)
        self._write_device_patterns(kv_loc, fill_len)

        self.coordinator.admit_request_into_staging(req)
        torch.cuda.synchronize()
        ready = self.coordinator.collect_ready_reqs()
        self.assertEqual(ready, [req])

        host_row = self.coordinator.req_to_host_pool[req.req_pool_idx, :rounded_len]
        self.assertTrue(torch.all(host_row >= 0))
        self.assertEqual(torch.unique(host_row).numel(), rounded_len)
        self.assertEqual(
            int(self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]),
            rounded_len,
        )

        available_size = self.coordinator.mem_pool_host.available_size()
        next_host_index = self.coordinator.mem_pool_host.alloc_paged_token_slots(
            self.coordinator.req_to_host_pool,
            self.coordinator.req_to_host_pool_allocated_len,
            req.req_pool_idx,
            fill_len,
            1,
        )
        # With page_size>1 the rounded-up staging allocation provides headroom,
        # so no new pages are needed.  With page_size=1 there is no headroom and
        # exactly one new page is allocated for the next token.
        expected_new_pages = 0 if fill_len < rounded_len else 1
        self.assertEqual(
            self.coordinator.mem_pool_host.available_size(),
            available_size - expected_new_pages,
        )
        self.assertTrue(torch.all(next_host_index >= 0))

        expected_total = rounded_len + expected_new_pages * self.page_size
        allocated_host_indices = self.coordinator.mem_pool_host.allocated_host_indices(
            self.coordinator.req_to_host_pool,
            req.req_pool_idx,
            int(self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]),
        )
        self.assertEqual(allocated_host_indices.numel(), expected_total)

        self._cleanup_req(req, kv_loc)
        self._assert_sizes_restored(initial, "single_node_staging_pages")

    # ==================================================================
    # Test: Direct-to-host (PD separated) path
    # ==================================================================
    def test_request_lifecycle_direct_path(self):
        """alloc_logical_only -> host write -> admit_direct -> swap-in -> finish."""
        initial = self._get_initial_sizes()
        fill_len = DEVICE_BUFFER_SIZE + self.page_size
        req = _make_req("direct-req", list(range(fill_len)))
        self._alloc_req_slot(req)

        kv_loc = self._alloc_kv(req, fill_len, logical_only=True)
        self._populate_host_pool(req, fill_len)
        self.coordinator.admit_request_direct(req)

        self.assertFalse(req.staging)
        self.assertTrue(self.coordinator._skip_first_backup[req.req_pool_idx])
        buf_tokens = self.coordinator.req_device_buffer_tokens[
            :, req.req_pool_idx, :DEVICE_BUFFER_SIZE
        ]
        self.assertTrue(torch.all(buf_tokens == -1))

        tokens = self._build_topk_tokens(fill_len - 1)
        batch = tokens.unsqueeze(0)
        rpi, sls = self._make_batch_tensors([req], [fill_len])

        locs = self._swap_in_selected_pages(rpi, sls, batch, layer_id=0)
        self.assertTrue(torch.all(locs[0, :TOP_K] >= 0))
        self._assert_kv_correct(
            locs[0], tokens, layer_id=0, count=TOP_K, msg="Direct: "
        )
        self._assert_matches_naive(rpi, sls, batch, locs, layer_id=0, msg="Direct: ")

        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "direct_path")

    # ==================================================================
    # Test: PD decode prealloc host page allocation
    # ==================================================================
    def test_pd_decode_prealloc_hisparse_host_slots(self):
        """PD decode prealloc should allocate RDMA targets through the host pool."""
        initial = self._get_initial_sizes()
        fill_len = self.page_size * 2 + 1
        req = _make_req("pd-decode-prealloc", list(range(fill_len)))

        from sglang.srt.disaggregation.decode import DecodePreallocQueue

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.req_to_token_pool = self.req_to_token_pool
        queue.token_to_kv_pool_allocator = self.allocator
        queue.token_to_kv_pool = self.allocator.get_kvcache()
        queue.tree_cache = SimpleNamespace(
            evictable_size=lambda: 0,
            protected_size=lambda: 0,
        )
        queue.scheduler = SimpleNamespace(
            enable_hisparse=True,
            hisparse_coordinator=self.coordinator,
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
        )

        host_indices = queue._pre_alloc(req)
        self.assertEqual(host_indices.numel(), fill_len)
        self.assertTrue(torch.all(host_indices >= 0))
        self.assertTrue(
            torch.equal(
                host_indices,
                self.coordinator.req_to_host_pool[req.req_pool_idx, :fill_len],
            )
        )
        self.assertEqual(req.kv.kv_allocated_len, fill_len)
        self.assertEqual(req.kv_committed_len, fill_len)
        self.assertEqual(req.extend_range.length, fill_len)

        rounded_len = (fill_len + self.page_size - 1) // self.page_size * self.page_size
        self.assertEqual(
            int(self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]),
            rounded_len,
        )
        allocated_host_indices = self.coordinator.mem_pool_host.allocated_host_indices(
            self.coordinator.req_to_host_pool,
            req.req_pool_idx,
            int(self.coordinator.req_to_host_pool_allocated_len[req.req_pool_idx]),
        )
        self.assertEqual(allocated_host_indices.numel(), rounded_len)

        kv_loc = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : req.kv.kv_allocated_len
        ].clone()
        self._cleanup_req(req, kv_loc, logical_only=True)
        self._assert_sizes_restored(initial, "pd_decode_prealloc_hisparse")

    # ==================================================================
    # Test: Batch multiple requests
    # ==================================================================
    def test_batch_multiple_requests(self):
        """Mix of short & long requests in batch: kernel correct + no leaks."""
        initial = self._get_initial_sizes()

        configs = [
            ("batch-short-0", self.page_size),
            ("batch-short-1", self.page_size),
            ("batch-long-0", DEVICE_BUFFER_SIZE + self.page_size),
            ("batch-long-1", DEVICE_BUFFER_SIZE + self.page_size * 2),
        ]

        reqs, kv_locs = [], []
        for rid, fl in configs:
            req = _make_req(rid, list(range(fl)))
            self._alloc_req_slot(req)
            is_long = fl > DEVICE_BUFFER_SIZE
            kv_loc = self._alloc_kv(req, fl, logical_only=is_long)
            if is_long:
                self._populate_host_pool(req, fl)
                self.coordinator.admit_request_direct(req)
            else:
                self._write_device_patterns(kv_loc, fl)
                self.coordinator.alloc_device_buffer(req)
            reqs.append(req)
            kv_locs.append(kv_loc)

        rpi, sls = self._make_batch_tensors(reqs, [c[1] for c in configs])
        top_k_batch = torch.stack(
            [
                # For long sequences pass fl-1 to exclude the "newest token" position
                # whose reserved device-buffer slot is not populated in unit tests.
                self._build_topk_tokens(fl - 1 if fl > DEVICE_BUFFER_SIZE else fl)
                for _, fl in configs
            ]
        )

        for lid in range(LAYER_NUM):
            locs = self._swap_in_selected_pages(rpi, sls, top_k_batch, lid)
            for i, (rid, fl) in enumerate(configs):
                vn = min(fl, TOP_K)
                self.assertTrue(
                    torch.all(locs[i, :vn] >= 0),
                    f"Req {rid}, layer {lid}: negative locs",
                )
                self._assert_kv_correct(
                    locs[i], top_k_batch[i], lid, vn, msg=f"{rid}: "
                )

        for i, req in enumerate(reqs):
            is_long = configs[i][1] > DEVICE_BUFFER_SIZE
            self._cleanup_req(req, kv_locs[i], logical_only=is_long)

        self._assert_sizes_restored(initial, "batch_multiple")


if __name__ == "__main__":
    unittest.main()
