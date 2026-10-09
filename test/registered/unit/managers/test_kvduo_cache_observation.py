"""Compile production resolver observation math and exercise slot generations."""

import ctypes
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import torch
from test_kvduo_hot_growth import make_coordinator


@pytest.fixture(scope="module")
def native_observation(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required")
    directory = tmp_path_factory.mktemp("kvduo-observation")
    source = directory / "observation.cpp"
    source.write_text(
        """
#include <sgl_kernel/kvduo_cache_observation.h>
extern "C" void observe(int e, int m, int d, float* counts, int64_t* steps) {
  sglang::kvduo_update_prefill_ema(e,m,d,counts,*steps);
}
extern "C" void capacity(const int* hits, int m, int d, int n, float* counts, int64_t* steps) {
  sglang::kvduo_update_capacity_ema(hits,m,d,n,counts,*steps);
}
extern "C" int segment(int capacity, int position, int k) {
  return sglang::kvduo_lru_segment(capacity,position,k);
}
"""
    )
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
    native.observe.argtypes = [ctypes.c_int] * 3 + [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int64),
    ]
    native.observe.restype = None
    native.capacity.argtypes = (
        [ctypes.POINTER(ctypes.c_int)]
        + [ctypes.c_int] * 3
        + [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int64)]
    )
    native.capacity.restype = None
    native.segment.argtypes = [ctypes.c_int] * 3
    native.segment.restype = ctypes.c_int
    return native


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


def test_production_ema_initializes_and_includes_valid_zero_count_steps(
    native_observation,
):
    counts = (ctypes.c_float * 2)(99, 99)
    steps = ctypes.c_int64(0)
    native_observation.observe(8, 20, 32, counts, ctypes.byref(steps))
    assert list(counts) == [8, 20] and steps.value == 1
    native_observation.observe(0, 0, 32, counts, ctypes.byref(steps))
    assert list(counts) == [8 * 63 / 64, 20 * 63 / 64] and steps.value == 2
    native_observation.observe(0, 0, 0, counts, ctypes.byref(steps))
    assert list(counts) == [8 * 63 / 64, 20 * 63 / 64] and steps.value == 2


def test_replay_reenables_real_observation_and_growth_resets_capacity_only():
    c, _ = make_coordinator()
    c.kvduo_observe.zero_()
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c.kvduo_observe.item() == 1
    c.kvduo_capacity_ema.fill_(3)
    c.kvduo_capacity_steps.fill_(65)
    c.kvduo_prefill_ema.fill_(7)
    c.kvduo_prefill_steps.fill_(66)
    pointer = c.kvduo_capacity_ema.data_ptr()
    c.kvduo_hot_growth_events[0, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert not c.kvduo_capacity_ema.any() and not c.kvduo_capacity_steps.any()
    assert c.kvduo_prefill_steps.item() == 66 and c.kvduo_prefill_ema[
        0, 0
    ].tolist() == [7, 7]
    assert c.kvduo_capacity_ema.data_ptr() == pointer


def test_reset_waits_old_decode_then_clears_only_reused_request_generation():
    c, _ = make_coordinator(layers=2, requests=2)
    c.kvduo_capacity_ema.fill_(2)
    c.kvduo_prefill_ema.fill_(3)
    c.kvduo_prefill_steps.fill_(65)
    c._kvduo_hand = 3
    c._kvduo_hand_sequence = 8

    def old_decode():
        c.kvduo_capacity_ema[:, 0] = 100
        c.kvduo_prefill_ema[:, 0] = 100

    c._wait_kvduo_hot_merge_boundary = MagicMock(side_effect=old_decode)
    c._reset_kvduo_hot_metadata(0)
    c._wait_kvduo_hot_merge_boundary.assert_called_once()
    assert not c.kvduo_capacity_ema[:, 0].any() and torch.all(
        c.kvduo_capacity_ema[:, 1] == 2
    )
    assert not c.kvduo_prefill_ema[:, 0].any() and torch.all(
        c.kvduo_prefill_ema[:, 1] == 3
    )
    assert c._kvduo_request_generation == [1, 0]
    assert c._kvduo_hand == 3 and c._kvduo_hand_sequence == 8
