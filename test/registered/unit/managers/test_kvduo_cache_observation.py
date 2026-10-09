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
extern "C" void hit(bool* v, bool real) { sglang::kvduo_record_hot_hit(*v,real); }
extern "C" void insert(bool* v) { sglang::kvduo_record_hot_insert(*v); }
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
    native.hit.argtypes = [ctypes.POINTER(ctypes.c_bool), ctypes.c_bool]
    native.hit.restype = None
    native.insert.argtypes = [ctypes.POINTER(ctypes.c_bool)]
    native.insert.restype = None
    return native


def test_production_visited_only_records_real_reuse_and_replacement_clears(
    native_observation,
):
    visited = ctypes.c_bool(False)
    native_observation.hit(ctypes.byref(visited), False)
    assert not visited.value
    native_observation.hit(ctypes.byref(visited), True)
    assert visited.value
    native_observation.insert(ctypes.byref(visited))
    assert not visited.value


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


def test_replay_reenables_real_observation_and_growth_preserves_visited():
    c, _ = make_coordinator()
    c.kvduo_observe.zero_()
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c.kvduo_observe.item() == 1
    pointer = c.hot_visited.data_ptr()
    c.hot_visited[0, 0, 0] = True
    c.kvduo_hot_growth_events[0, 0] = 1
    c.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert c.hot_visited[0, 0, 0] and not c.hot_visited[0, 0, 4:].any()
    assert c.hot_visited.data_ptr() == pointer


def test_reset_waits_old_decode_then_clears_only_reused_request_generation():
    c, _ = make_coordinator(layers=2, requests=2)
    c.hot_visited.fill_(True)
    c.kvduo_prefill_ema.fill_(3)
    c.kvduo_prefill_steps.fill_(65)
    c._kvduo_hand = 3
    c._kvduo_hand_sequence = 8

    def old_decode():
        c.hot_visited[:, 0] = True
        c.kvduo_prefill_ema[:, 0] = 100

    c._wait_kvduo_hot_merge_boundary = MagicMock(side_effect=old_decode)
    c._reset_kvduo_hot_metadata(0)
    c._wait_kvduo_hot_merge_boundary.assert_called_once()
    assert not c.hot_visited[:, 0].any() and c.hot_visited[:, 1].all()
    assert not c.kvduo_prefill_ema[:, 0].any() and torch.all(
        c.kvduo_prefill_ema[:, 1] == 3
    )
    assert c._kvduo_request_generation == [1, 0]
    assert c._kvduo_hand == 3 and c._kvduo_hand_sequence == 8
