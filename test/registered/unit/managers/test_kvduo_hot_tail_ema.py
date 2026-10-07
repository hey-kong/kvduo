"""Exercise the actual shared C++ EMA math and production HOT lifecycle on CPU."""

import ctypes
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from test_kvduo_hot_growth import make_coordinator


@pytest.fixture(scope="module")
def native_ema(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("Testing the production CUDA/CPU EMA helper needs a C++ compiler")
    directory = tmp_path_factory.mktemp("kvduo-hot-tail-ema")
    source = directory / "ema.cpp"
    source.write_text(
        """
#include <sgl_kernel/kvduo_hot_tail_ema.h>
extern "C" int releasable(int capacity, int k, int top_k) {
  return sglang::kvduo_hot_releasable_pages(capacity, k, top_k);
}
extern "C" void observe(int hits, int pages, float alpha, float* q, int64_t* t) {
  sglang::kvduo_update_hot_tail_ema(hits, pages, alpha, *q, *t);
}
extern "C" void observe_prefill(int d, int direct, int base, int extra, float alpha, float* counts, int64_t* t) {
  sglang::kvduo_update_prefill_hit_ema(d, direct, base, extra, alpha, counts, *t);
}
"""
    )
    include = Path(__file__).parents[4] / "python/sglang/kernels/jit/include"
    library = directory / "ema.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            "-fPIC",
            f"-I{include}",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    native = ctypes.CDLL(str(library))
    native.releasable.argtypes = [ctypes.c_int] * 3
    native.releasable.restype = ctypes.c_int
    native.observe.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int64),
    ]
    native.observe.restype = None
    native.observe_prefill.argtypes = [ctypes.c_int] * 4 + [
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int64),
    ]
    native.observe_prefill.restype = None
    return native


@pytest.mark.parametrize(
    "pages, top_k, k, released",
    [
        (1, 2, 4, 0),
        (2, 2, 4, 1),
        (3, 2, 4, 1),
        (4, 2, 4, 2),
        (5, 2, 4, 2),
        (7, 2, 4, 3),
        (8, 2, 4, 4),
        (4, 13, 4, 0),
        (5, 13, 4, 1),
        (6, 13, 4, 2),
        (8, 21, 4, 2),
        (8, 29, 4, 0),
        (8, 512, 1344, 4),
    ],
)
def test_actual_shrink_geometry_honors_half_rounding_and_topk_minimum(
    native_ema, pages, top_k, k, released
):
    assert native_ema.releasable(pages * k, k, top_k) == released


def observe(native, q, t, hits, pages, alpha=1 / 64):
    native.observe(hits, pages, alpha, ctypes.byref(q), ctypes.byref(t))


def test_first_observation_initializes_then_zero_hit_steps_decay(native_ema):
    q, t = ctypes.c_float(99), ctypes.c_int64(0)
    observe(native_ema, q, t, hits=6, pages=3)
    assert q.value == 2 and t.value == 1
    for step in range(1, 130):
        observe(native_ema, q, t, hits=0, pages=3)
        assert q.value == pytest.approx(2 * (63 / 64) ** step)
        assert t.value == step + 1


def test_growth_keeps_prior_and_next_step_uses_new_release_page_count(native_ema):
    q, t = ctypes.c_float(0), ctypes.c_int64(0)
    observe(native_ema, q, t, hits=4, pages=2)
    # Growth changes F from two to four: B=4 now means d=1, rather than 2.
    observe(native_ema, q, t, hits=4, pages=4)
    assert q.value == pytest.approx(127 / 64)
    assert t.value == 2


def test_minimum_capacity_freezes_score_and_count(native_ema):
    q, t = ctypes.c_float(3.5), ctypes.c_int64(140)
    observe(native_ema, q, t, hits=0, pages=0)
    assert q.value == 3.5 and t.value == 140


def test_alpha_one_uses_the_current_loss(native_ema):
    q, t = ctypes.c_float(12), ctypes.c_int64(5)
    observe(native_ema, q, t, hits=3, pages=2, alpha=1)
    assert q.value == 1.5 and t.value == 6


def test_actual_growth_keeps_ema_and_observation_steps():
    coordinator, _ = make_coordinator()
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))
    coordinator.kvduo_hot_tail_ema[0, 0] = 0.75
    coordinator.kvduo_hot_tail_observation_steps[0, 0] = 150
    coordinator.kvduo_hot_growth_events[0, 0] = 1
    addresses = (
        coordinator.kvduo_hot_tail_ema.data_ptr(),
        coordinator.kvduo_hot_tail_observation_steps.data_ptr(),
    )

    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))

    assert int(coordinator.kvduo_req_hot_capacity[0, 0]) == 8
    assert float(coordinator.kvduo_hot_tail_ema[0, 0]) == 0.75
    assert int(coordinator.kvduo_hot_tail_observation_steps[0, 0]) == 150
    assert addresses == (
        coordinator.kvduo_hot_tail_ema.data_ptr(),
        coordinator.kvduo_hot_tail_observation_steps.data_ptr(),
    )


def test_replay_resumes_observations_after_synthetic_capture():
    coordinator, _ = make_coordinator()
    coordinator.kvduo_hot_tail_ema_observe.zero_()
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0]))
    assert coordinator.kvduo_hot_tail_ema_observe.item() == 1


def test_actual_shrink_resets_only_its_request_layer():
    coordinator, _ = make_coordinator(layers=2, requests=2)
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    coordinator.kvduo_hot_growth_events[1, 0] = 1
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    coordinator.kvduo_hot_tail_ema.fill_(0.75)
    coordinator.kvduo_hot_tail_observation_steps.fill_(150)

    coordinator._merge_kvduo_hot_pages(1, 0)

    assert int(coordinator.kvduo_req_hot_capacity[1, 0]) == 4
    assert coordinator.kvduo_hot_tail_ema.tolist() == [[0.75, 0.75], [0, 0.75]]
    assert coordinator.kvduo_hot_tail_observation_steps.tolist() == [
        [150, 150],
        [0, 150],
    ]


def test_request_release_resets_all_its_layers_without_changing_other_requests():
    coordinator, _ = make_coordinator(layers=2, requests=2)
    coordinator.prepare_kvduo_graph_replay(torch.tensor([0, 1]))
    coordinator.kvduo_hot_tail_ema.fill_(0.75)
    coordinator.kvduo_hot_tail_observation_steps.fill_(150)

    coordinator._release_kvduo_layer_hot_pages(1)

    assert coordinator.kvduo_hot_tail_ema.tolist() == [[0.75, 0], [0.75, 0]]
    assert coordinator.kvduo_hot_tail_observation_steps.tolist() == [[150, 0], [150, 0]]


def test_reset_orders_old_decode_before_reusing_the_request_slot(monkeypatch):
    import sglang.srt.managers.hisparse_coordinator as coordinator_module

    coordinator, _ = make_coordinator()
    coordinator.device = torch.device("cuda:3")
    coordinator.decode_producer_stream = object()
    coordinator.kvduo_hot_tail_ema.fill_(1)
    coordinator.kvduo_hot_tail_observation_steps.fill_(140)

    def finish_old_decode(stream):
        assert stream is coordinator.decode_producer_stream
        assert int(coordinator.kvduo_hot_tail_observation_steps[0, 0]) == 140
        coordinator.kvduo_hot_tail_ema[0, 0] = 12
        coordinator.kvduo_hot_tail_observation_steps[0, 0] = 141

    stream = SimpleNamespace(wait_stream=MagicMock(side_effect=finish_old_decode))
    current = MagicMock(return_value=stream)
    monkeypatch.setattr(coordinator_module.device_module, "current_stream", current)
    coordinator._reset_kvduo_hot_tail_ema(0)

    current.assert_called_once_with(torch.device("cuda:3"))
    assert not torch.any(coordinator.kvduo_hot_tail_ema)
    assert not torch.any(coordinator.kvduo_hot_tail_observation_steps)


@pytest.mark.parametrize("alpha", [0, -1, 1.1, float("nan")])
def test_invalid_alpha_is_rejected_before_pool_construction(alpha):
    pool = SimpleNamespace(req_to_token=torch.zeros((1, 1)))
    with pytest.raises(ValueError, match="0 < alpha <= 1"):
        HiSparseCoordinator(
            pool,
            None,
            2,
            32,
            "cpu",
            None,
            tail_protected_pages=1,
            hot_tail_ema_alpha=alpha,
        )
