import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parents[4]
    / "python/sglang/srt/mem_cache/sparsity/core/kvduo_prefix_cache.py"
)
SPEC = importlib.util.spec_from_file_location(
    "kvduo_prefix_cache_under_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
STATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STATE
SPEC.loader.exec_module(STATE)

HostPrefixRecord = STATE.HostPrefixRecord
KVDuoHostPrefixCache = STATE.KVDuoHostPrefixCache
KVDuoPrefixPageView = STATE.KVDuoPrefixPageView
KVDuoPrefixResidency = STATE.KVDuoPrefixResidency
plan_kvduo_prefix_restore = STATE.plan_kvduo_prefix_restore
begin_kvduo_prefix_restore = STATE.begin_kvduo_prefix_restore
finish_kvduo_prefix_restore = STATE.finish_kvduo_prefix_restore


def record(identity, location, *, version=1, host_version=1, clock=0):
    return HostPrefixRecord(
        identity,
        ("shared", "shared"),
        {"shared": (location,)},
        {"shared": version},
        {"shared": host_version},
        cache_reference=True,
        last_access=clock,
        tie_break_key=(identity,),
    )


def test_request_slot_reuse_does_not_change_prefix_identity():
    cache = KVDuoHostPrefixCache(capacity_slots=2)
    cache.insert(record("prefix-a", 10))
    cache.acquire("prefix-a", "slot-0-generation-1", 1)
    cache.release("prefix-a", "slot-0-generation-1")
    cache.acquire("prefix-a", "slot-0-generation-2", 2)
    assert cache.records["prefix-a"].request_references == {"slot-0-generation-2"}
    assert cache.records["prefix-a"].host_locations["shared"] == (10,)


def test_shared_prefix_survives_one_request_finish_abort_or_retract():
    cache = KVDuoHostPrefixCache(capacity_slots=2)
    cache.insert(record("shared", 10))
    cache.acquire("shared", "r1", 1)
    cache.acquire("shared", "r2", 2)
    cache.release("shared", "r1")
    assert cache.records["shared"].request_references == {"r2"}
    assert cache.evict_lru(1) == ()
    cache.release("shared", "r2")
    assert cache.evict_lru(1)[0].identity == "shared"
    assert cache.used_slots == 0


def test_host_lru_is_bounded_and_skips_restore_dma_and_request_pins():
    cache = KVDuoHostPrefixCache(capacity_slots=3)
    for identity, location, clock in (("a", 1, 1), ("b", 2, 2), ("c", 3, 3)):
        cache.insert(record(identity, location, clock=clock))
    cache.records["a"].request_references.add("active")
    cache.records["b"].restore_pins = 1
    victims = cache.evict_lru(1)
    assert tuple(v.identity for v in victims) == ("c",)
    assert set(cache.records) == {"a", "b"}


def test_residency_plans_gpu_host_mixed_and_invalid_versions():
    cache = KVDuoHostPrefixCache(capacity_slots=4)
    cache.insert(record("a", 1))
    cache.insert(record("b", 2))
    domain_bytes = {"shared": 128}

    full = plan_kvduo_prefix_restore(
        [KVDuoPrefixPageView("a", ("shared",), frozenset({"shared"}))],
        cache,
        domain_bytes,
    )
    assert full.match_status is KVDuoPrefixResidency.FULL_GPU_HIT
    assert full.missing_bytes == 0

    host = plan_kvduo_prefix_restore(
        [KVDuoPrefixPageView("a", ("shared",))], cache, domain_bytes
    )
    assert host.match_status is KVDuoPrefixResidency.HOST_ONLY_HIT
    assert host.execution_status is KVDuoPrefixResidency.RESTORE_REQUIRED
    assert host.missing_bytes == 128

    mixed = plan_kvduo_prefix_restore(
        [
            KVDuoPrefixPageView("a", ("shared",), frozenset({"shared"})),
            KVDuoPrefixPageView("b", ("shared",)),
        ],
        cache,
        domain_bytes,
    )
    assert mixed.match_status is KVDuoPrefixResidency.PARTIAL_GPU_HIT
    assert mixed.restore_from_host == (("b", "shared"),)

    cache.records["b"].data_versions["shared"] = 2
    invalid = plan_kvduo_prefix_restore(
        [KVDuoPrefixPageView("b", ("shared",))], cache, domain_bytes
    )
    assert invalid.match_status is KVDuoPrefixResidency.INVALID_VERSION


def test_restore_planning_is_touch_neutral_and_deduplicates_shared_domains():
    cache = KVDuoHostPrefixCache(capacity_slots=2)
    cache.insert(record("a", 1, clock=9))
    before = cache.records["a"].last_access
    plan = plan_kvduo_prefix_restore(
        [KVDuoPrefixPageView("a", ("shared", "shared"))],
        cache,
        {"shared": 64},
    )
    assert plan.restore_from_host == (("a", "shared"),)
    assert plan.missing_bytes == 64
    assert cache.records["a"].last_access == before


def test_restore_commitment_prevents_eviction_until_h2d_finishes():
    cache = KVDuoHostPrefixCache(capacity_slots=1)
    cache.insert(record("a", 1))
    plan = plan_kvduo_prefix_restore(
        [KVDuoPrefixPageView("a", ("shared",))], cache, {"shared": 64}
    )
    commitment = begin_kvduo_prefix_restore(plan, cache)
    assert cache.records["a"].restore_pins == 1
    assert cache.evict_lru(1) == ()
    finish_kvduo_prefix_restore(commitment, cache)
    assert cache.evict_lru(1)[0].identity == "a"
