import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from sglang.kernels.ops.kvcache.kvduo_hand_plan import (
    allocate_kvduo_hand_workspace,
    prepare_kvduo_hand,
)
from sglang.srt.mem_cache.sparsity.core.kvduo_hot_merge import kvduo_shrink_target
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_hot_merge import (
    assert_payloads_match_tokens,
    make_merge_coordinator,
)

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def test_native_capacity_policy_matches_python_at_maturity_and_numeric_boundaries(
    tmp_path,
):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("Native policy check requires a C++ compiler")
    include = Path(__file__).resolve().parents[4] / "python/sglang/kernels/jit/include"
    source = tmp_path / "policy.cc"
    source.write_text(
        "#include <sgl_kernel/kvduo_capacity_plan.h>\n"
        'extern "C" int target(const float* h, float m, long long t, int n, int k, double rho) {\n'
        "return sglang::kvduo_capacity_target(h,m,t,n,k,rho);\n}\n"
    )
    library = tmp_path / "policy.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O3",
            "-shared",
            "-fPIC",
            "-I",
            str(include),
            str(source),
            "-o",
            str(library),
        ],
        check=True,
    )
    target = ctypes.CDLL(str(library)).target
    target.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_longlong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
    ]
    target.restype = ctypes.c_int
    rng = np.random.default_rng(43)
    for n in range(1, 9):
        for minimum in range(1, n + 1):
            for steps in (0, 64, 65, 1000):
                hits = rng.uniform(0, 4, 8).astype(np.float32)
                for special in (None, 0.0, np.nan, np.inf, -1.0):
                    values = hits.copy()
                    if special is not None:
                        values[-1 if n == 8 else n - 1] = special
                    for rho in (0.0, 0.1, 1.0, np.nan, -1.0):
                        miss = float(rng.uniform(0, 20))
                        actual = target(
                            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            np.float32(miss),
                            steps,
                            n,
                            minimum,
                            rho,
                        )
                        assert actual == kvduo_shrink_target(
                            values.tolist(),
                            float(np.float32(miss)),
                            steps,
                            n,
                            minimum,
                            rho,
                        )
    # Zero misses admit only an exactly zero discarded contribution.
    for tail in (0.0, np.nextafter(np.float32(0), np.float32(1)), 0.1):
        values = np.array([1, tail] + [0] * 6, dtype=np.float32)
        assert target(
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 0, 65, 2, 1, 0.1
        ) == (1 if tail == 0 else 2)
    assert (
        target(values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1, 65, 2, 3, 0.1)
        == -2
    )


@pytest.mark.parametrize("hand", [0, 1, 5, 11, 13])
@pytest.mark.parametrize("needed", [1, 3, 50])
def test_request_major_hand_prefix_retains_zero_release_visits_and_can_resume(
    hand, needed
):
    counts = torch.zeros((3, 4, 9), dtype=torch.float32)
    counts[:, :, :2] = 10
    steps = torch.full((3, 4), 65, dtype=torch.int64)
    steps[0, 0] = 64
    workspace = allocate_kvduo_hand_workspace(counts)
    workspace.pages_host.fill_(4)
    workspace.pages_host[1, 1] = 0
    snapshot = prepare_kvduo_hand(
        counts, steps, workspace, hand=hand, minimum=1, rho=0.1
    )
    reference = []
    for offset in range(12):
        req, layer = divmod((hand + offset) % 12, 3)
        if (layer, req) == (1, 1):
            continue
        reference.append(
            (offset, layer * 4 + req, 4, 4 if (layer, req) == (0, 0) else 2)
        )
    released = 0
    prefix = []
    for row in reference:
        prefix.append(row)
        released += row[2] - row[3]
        if released >= needed:
            break
    before = counts.clone(), steps.clone()
    assert snapshot.next(0, needed) == prefix
    cursor = prefix[-1][0] + 1
    assert snapshot.next(cursor, 100) == [row for row in reference if row[0] >= cursor]
    assert torch.equal(counts, before[0]) and torch.equal(steps, before[1])


@pytest.mark.parametrize("compact", [False, True])
def test_actual_merge_loop_preserves_hand_order_payloads_and_maturity(compact):
    c, p = make_merge_coordinator(
        {(layer, req): 3 for req in range(3) for layer in range(2)}, layers=2
    )
    c.kvduo_capacity_steps[1, 2] = 20
    c._kvduo_hot_pressure_protected.add((0, 0, 0))
    c._kvduo_hand = 5
    reclaim = c._reclaim_kvduo_hot_pages_gpu if compact else c.reclaim_kvduo_hot_pages
    assert reclaim(3) == 4
    assert c._kvduo_hand == 3
    assert int(c.kvduo_capacity_steps[1, 2]) == 20
    assert len(c._kvduo_req_layer_pages[1, 2]) == 3
    for owner in c._kvduo_req_layer_pages:
        assert_payloads_match_tokens(c, *owner)


def test_compact_loop_continues_after_an_actual_merge_cannot_release(monkeypatch):
    c, p = make_merge_coordinator({(0, req): 3 for req in range(3)})
    merge = c._merge_kvduo_hot_pages
    visited = []

    def unavailable(layer, req, target):
        visited.append(req)
        return 0 if req == 0 else merge(layer, req, target)

    monkeypatch.setattr(c, "_merge_kvduo_hot_pages", unavailable)
    assert c._reclaim_kvduo_hot_pages_gpu(2) == 2
    assert visited == [0, 1]
    assert c._kvduo_hand == 2 and p.free_pages == 1


@pytest.mark.parametrize("stale", ["generation", "staging", "tp"])
def test_compact_plan_revalidates_request_and_tp_before_any_merge(monkeypatch, stale):
    from sglang.kernels.ops.kvcache import kvduo_hand_plan as module

    c, p = make_merge_coordinator({(0, 0): 3})
    prepare = module.prepare_kvduo_hand

    def invalidate(*args, **kwargs):
        snapshot = prepare(*args, **kwargs)
        if stale == "generation":
            c._kvduo_request_generation[0] += 1
        elif stale == "staging":
            c._active_kvduo_reqs[0] = type("Req", (), {"hisparse_staging": True})()
        else:
            monkeypatch.setattr(c, "_kvduo_consensus", lambda plan: False)
        return snapshot

    monkeypatch.setattr(module, "prepare_kvduo_hand", invalidate)
    assert c._reclaim_kvduo_hot_pages_gpu(2) == 0
    assert p.free_pages == 0 and c._kvduo_hand == 0
