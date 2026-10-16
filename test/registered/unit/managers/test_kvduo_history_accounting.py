from types import SimpleNamespace

import pytest
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)


@pytest.mark.parametrize(
    "committed,used", [(65536, 0), (65536, 256), (98304, 256), (65536, 256)]
)
def test_committed_history_stats_and_strict_busy_check(committed, used, monkeypatch):
    allocator = object.__new__(DeepSeekV4HiSparseTokenToKVPoolAllocator)
    allocator._size_full = 131072
    allocator.page_size = 256
    allocator.compress_ratio = 4
    allocator.kvduo_shared_layout = SimpleNamespace(current_pages=committed // 256)
    allocator.logical_attn_allocator = SimpleNamespace(
        full_available_size=lambda: committed - used,
        swa_available_size=lambda: 1024 - used,
    )
    # Physical capacity is deliberately smaller: logical accounting must not
    # confuse a physical shortage with leaked logical tokens either.
    allocator.hisparse_attn_allocator = SimpleNamespace(available_size=lambda: 100)
    tree = SimpleNamespace(
        full_evictable_size=lambda: 0,
        swa_evictable_size=lambda: 0,
        full_protected_size=lambda: 0,
        swa_protected_size=lambda: 0,
        session_held_full_tokens=lambda idx: 0,
        session_held_swa_tokens=lambda idx: 0,
    )
    req = SimpleNamespace(
        kv=SimpleNamespace(kv_allocated_len=used, swa_evicted_seqlen=0),
        cache_protected_len=0,
        req_pool_idx=0,
    )
    batch = SimpleNamespace(reqs=[req], is_empty=lambda: False)
    observer = SchedulerPoolStatsObserver(
        tree_cache=tree,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=None,
        session_controller=None,
        hisparse_coordinator=None,
        is_hybrid_swa=True,
        is_hybrid_ssm=False,
        enable_hisparse=True,
        full_tokens_per_layer=131072,
        swa_tokens_per_layer=1024,
        max_total_num_tokens=131072,
        get_last_batch=lambda: batch,
        get_running_batch=lambda: batch,
    )
    stats = observer.get_pool_stats()
    assert stats.full_num_used == used
    assert stats.full_token_usage == used / committed
    assert stats.full_committed_capacity == committed
    assert stats.full_virtual_capacity == 131072
    assert stats.full_unmapped_capacity == 131072 - committed
    assert allocator.size_full == 131072
    checker = SchedulerInvariantChecker(
        is_hybrid_swa=True,
        is_hybrid_ssm=False,
        disaggregation_mode=None,
        page_size=256,
        full_tokens_per_layer=131072,
        swa_tokens_per_layer=1024,
        max_total_num_tokens=131072,
        server_args=SimpleNamespace(dcp_size=1),
        tree_cache=tree,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=None,
        pool_stats_observer=observer,
        get_last_batch=lambda: batch,
        get_running_batch=lambda: batch,
    )
    monkeypatch.setenv("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY", "1")
    checker.self_check_during_busy()
    # A real lost committed page must still be reported.
    allocator.logical_attn_allocator.full_available_size = lambda: (
        committed - used - 256
    )
    with pytest.raises(AssertionError, match="Full Pool Mem Leak"):
        checker.self_check_during_busy()
