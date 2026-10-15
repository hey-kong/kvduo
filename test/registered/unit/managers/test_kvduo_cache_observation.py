"""Compile production resolver observation math and exercise slot generations."""

import ctypes
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import torch
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_hot_growth import make_coordinator

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture(scope="module")
def native_observation(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required")
    directory = tmp_path_factory.mktemp("kvduo-observation")
    source = directory / "observation.cpp"
    source.write_text("""
#include <sgl_kernel/kvduo_cache_observation.h>
extern "C" void capacity(const int* hits, int m, int d, int n, float* counts, int64_t* steps) {
  sglang::kvduo_update_capacity_ema(hits,m,d,n,counts,*steps);
}
extern "C" int segment(int capacity, int position, int k) {
  return sglang::kvduo_lru_segment(capacity,position,k);
}
extern "C" bool can_record_growth(bool full_lookup, bool observe, const int32_t* events,
                                  const int32_t* status, int64_t rid) {
  return sglang::kvduo_can_record_growth(full_lookup,observe,events,status,rid);
}
""")
    library = directory / "observation.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            f"-I{Path(__file__).parents[4]/'python/sglang/kernels/jit/include'}",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    native = ctypes.CDLL(str(library))
    native.capacity.argtypes = (
        [ctypes.POINTER(ctypes.c_int)]
        + [ctypes.c_int] * 3
        + [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int64)]
    )
    native.capacity.restype = None
    native.segment.argtypes = [ctypes.c_int] * 3
    native.segment.restype = ctypes.c_int
    native.can_record_growth.argtypes = [ctypes.c_bool] * 2 + [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int64,
    ]
    native.can_record_growth.restype = ctypes.c_bool
    return native


def test_standalone_hisparse_growth_guard_never_reads_null_kvduo_state(
    native_observation,
):
    # Real standalone resolver arguments, with default observation enabled and
    # a nonzero request slot: neither KVDuo bank exists in this mode.
    assert not native_observation.can_record_growth(False, True, None, None, 3)
    events = (ctypes.c_int32 * 4)()
    assert not native_observation.can_record_growth(False, True, events, None, 3)
    # Also fail closed for incomplete KVDuo state or synthetic observation.
    assert not native_observation.can_record_growth(True, True, events, None, 3)
    assert not native_observation.can_record_growth(True, True, None, None, 3)
    assert not native_observation.can_record_growth(True, False, events, None, 3)


def test_kvduo_growth_guard_keeps_valid_request_and_observation_checks(
    native_observation,
):
    events = (ctypes.c_int32 * 4)()
    status = (ctypes.c_int32 * 4)(1, 1, 1, 0)
    assert native_observation.can_record_growth(True, True, events, status, 3)
    assert not native_observation.can_record_growth(True, True, events, status, 0)
    assert not native_observation.can_record_growth(True, False, events, status, 3)
    assert not native_observation.can_record_growth(False, True, events, status, 3)
    assert not native_observation.can_record_growth(True, True, None, status, 3)


def test_production_lru_segments_use_pre_update_current_capacity(native_observation):
    assert [native_observation.segment(12, i, 4) for i in range(12)] == [2] * 4 + [
        1
    ] * 4 + [0] * 4
    hits = (ctypes.c_int * 8)(100, 30, 5, 1, 0, 0, 0, 0)
    counts = (ctypes.c_float * 9)(*([99] * 9))
    steps = ctypes.c_int64(0)
    native_observation.capacity(hits, 100, 236, 4, counts, ctypes.byref(steps))
    assert list(counts) == [100, 30, 5, 1, 0, 0, 0, 0, 100] and steps.value == 1
    zero = (ctypes.c_int * 8)(*([0] * 8))
    native_observation.capacity(zero, 0, 64, 4, counts, ctypes.byref(steps))
    assert (
        list(counts) == [v * 63 / 64 for v in [100, 30, 5, 1, 0, 0, 0, 0, 100]]
        and steps.value == 2
    )
    native_observation.capacity(zero, 0, 0, 4, counts, ctypes.byref(steps))
    assert steps.value == 2


def test_replay_reenables_observation_and_growth_keeps_capacity_history():
    c, _ = make_coordinator()
    c.kvduo_observe.zero_()
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c.kvduo_observe.item() == 1
    c.kvduo_capacity_ema.fill_(3)
    c.kvduo_capacity_steps.fill_(65)
    pointer = c.kvduo_capacity_ema.data_ptr()
    c.kvduo_hot_growth_events[0, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c.kvduo_capacity_ema[0, 0].tolist() == [3] * 9
    assert c.kvduo_capacity_steps.item() == 65
    assert c.kvduo_capacity_ema.data_ptr() == pointer


def test_growth_preserves_mru_segments_and_continues_production_ema(
    native_observation,
):
    c, _ = make_coordinator(layers=2, requests=2)
    owner = (1, 1)
    c._ensure_kvduo_hot_capacity_targets_batch({owner: 8})
    c.kvduo_capacity_ema[owner] = torch.tensor([8, 4, 9, 9, 0, 0, 0, 0, 2])
    c.kvduo_capacity_steps[owner] = 100
    old_lru = c.lru_slots[owner][:8].tolist()
    old_segments = [native_observation.segment(8, i, 4) for i in range(8)]
    shapes_and_pointers = [
        (bank.shape, bank.stride(), bank.data_ptr())
        for bank in (c.lru_slots, c.kvduo_capacity_ema, c.kvduo_capacity_steps)
    ]
    other_counts = c.kvduo_capacity_ema[:, 0].clone()

    c._ensure_kvduo_hot_capacity_targets_batch({owner: 16})

    assert c.lru_slots[owner][:16].tolist() == list(range(8, 16)) + old_lru
    assert old_segments == [native_observation.segment(16, i + 8, 4) for i in range(8)]
    assert c.kvduo_capacity_ema[owner].tolist() == [8, 4, 4, 4, 0, 0, 0, 0, 2]
    assert c.kvduo_capacity_steps[owner] == 100
    assert torch.equal(c.kvduo_capacity_ema[:, 0], other_counts)
    assert shapes_and_pointers == [
        (bank.shape, bank.stride(), bank.data_ptr())
        for bank in (c.lru_slots, c.kvduo_capacity_ema, c.kvduo_capacity_steps)
    ]
    counts = (ctypes.c_float * 9)(*c.kvduo_capacity_ema[owner].tolist())
    steps = ctypes.c_int64(int(c.kvduo_capacity_steps[owner]))
    native_observation.capacity(
        (ctypes.c_int * 8)(4, 0, 2, 0, 0, 0, 0, 0),
        0,
        6,
        4,
        counts,
        ctypes.byref(steps),
    )
    assert list(counts) == [
        8 - 4 / 64,
        4 * 63 / 64,
        4 - 2 / 64,
        4 * 63 / 64,
        0,
        0,
        0,
        0,
        2 * 63 / 64,
    ]
    assert steps.value == 101
    # A later expansion must use the latest last segment, including its EMA
    # updates, rather than the original seed or another more-recent segment.
    c.kvduo_capacity_ema[owner] = torch.tensor(list(counts))
    c.kvduo_capacity_steps[owner] = steps.value
    c._ensure_kvduo_hot_capacity_targets_batch({owner: 20})
    assert c.kvduo_capacity_ema[owner].tolist() == list(counts)[:4] + [
        counts[3],
        0,
        0,
        0,
        counts[8],
    ]
    assert c.kvduo_capacity_steps[owner] == 101


def test_first_hot_allocation_starts_fresh_observation():
    c, _ = make_coordinator(top_k=5)
    c.kvduo_capacity_ema.fill_(99)
    c.kvduo_capacity_steps.fill_(100)
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c.kvduo_req_hot_capacity.item() == 8
    assert not c.kvduo_capacity_ema.any() and not c.kvduo_capacity_steps.any()


def test_failed_growth_restores_capacity_history():
    c, p = make_coordinator(layers=2)
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    c.kvduo_capacity_ema.fill_(3)
    c.kvduo_capacity_ema[:, :, 1] = 9
    c.kvduo_capacity_steps.fill_(65)
    banks = (
        c.req_device_buffer_tokens,
        c.req_device_buffer_token_locs,
        c.lru_slots,
        c.kvduo_req_hot_capacity,
        c.kvduo_req_hot_capacity_gpu,
        c.kvduo_capacity_ema,
        c.kvduo_capacity_steps,
    )
    before = [bank.clone() for bank in banks]
    owners = dict(c._kvduo_hot_page_owners)
    free_pages = p.free_pages
    observe = c._observe_kvduo_hot_growth

    def fail_after_observing(owner, old, new):
        observe(owner, old, new)
        if owner == (1, 0):
            raise RuntimeError("publication failed")

    c._observe_kvduo_hot_growth = fail_after_observing
    with pytest.raises(RuntimeError, match="publication failed"):
        c._ensure_kvduo_hot_capacity_targets_batch({(0, 0): 8, (1, 0): 8})
    assert all(torch.equal(bank, old) for bank, old in zip(banks, before))
    assert c._kvduo_hot_page_owners == owners and p.free_pages == free_pages


def test_reset_waits_old_decode_then_clears_only_reused_request_generation():
    c, _ = make_coordinator(layers=2, requests=2)
    c.kvduo_capacity_ema.fill_(2)
    c._kvduo_hand = 3

    def old_decode():
        c.kvduo_capacity_ema[:, 0] = 100

    c._wait_kvduo_hot_merge_boundary = MagicMock(side_effect=old_decode)
    c._reset_kvduo_hot_metadata(0)
    c._wait_kvduo_hot_merge_boundary.assert_called_once()
    assert not c.kvduo_capacity_ema[:, 0].any() and torch.all(
        c.kvduo_capacity_ema[:, 1] == 2
    )
    assert c._kvduo_request_generation == [1, 0]
