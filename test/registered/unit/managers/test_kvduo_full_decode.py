from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.kvcache.kvduo_full_decode import begin_kvduo_decode
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def coordinator():
    c = HiSparseCoordinator.__new__(HiSparseCoordinator)
    c.enable_mixed_residency = True
    c.is_dsv4_hisparse = True
    c._active_kvduo_reqs = {
        i: SimpleNamespace(hisparse_staging=False) for i in range(2)
    }
    c._mixed_slots = [False, False]
    c._kvduo_pending_hot_minimum = set()
    c.mem_pool_device = SimpleNamespace(layer_num=3)
    c.full_touch_clock = torch.tensor([10], dtype=torch.int64)
    c.kvduo_full_clock_base = torch.zeros(1, dtype=torch.int64)
    c.kvduo_observe = torch.zeros(1, dtype=torch.int32)
    c.num_real_reqs = torch.zeros(1, dtype=torch.int32)
    return c


def test_fully_resident_graph_replay_returns_to_mixed_and_handles_slot_reuse():
    c = coordinator()
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.kvduo_full_graph = True
    runner.model_runner = SimpleNamespace(hisparse_coordinator=c)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE, req_pool_indices_cpu=torch.tensor([1, 0])
    )
    assert runner._kvduo_full_replay(batch, 2)
    c.publish_kvduo_decode_controls(2, fully_resident=True)
    assert c.full_touch_clock.item() == 13 and c.kvduo_full_clock_base.item() == 10
    assert c.num_real_reqs.item() == 2 and c.kvduo_observe.item() == 1
    full_key = runner._make_graph_key(4, variant_label="kvduo_full")
    mixed_key = runner._make_graph_key(4)
    assert full_key != mixed_key
    c._mixed_slots[0] = True
    assert not runner._kvduo_full_replay(batch, 2)
    c.publish_kvduo_decode_controls(2)
    assert c.full_touch_clock.item() == 13
    c._active_kvduo_reqs.pop(0)
    assert not runner._kvduo_full_replay(batch, 2)
    c._mixed_slots[0] = False
    c._active_kvduo_reqs[0] = SimpleNamespace(hisparse_staging=True)
    assert not runner._kvduo_full_replay(batch, 2)
    c._active_kvduo_reqs[0].hisparse_staging = False
    c._kvduo_pending_hot_minimum.add(0)
    assert not runner._kvduo_full_replay(batch, 2)
    c._kvduo_pending_hot_minimum.clear()
    assert runner._kvduo_full_replay(batch, 2)
    batch.req_pool_indices_cpu = None
    assert not runner._kvduo_full_replay(batch, 2)


@pytest.mark.parametrize(
    "reason",
    [
        None,
        "hisparse",
        "generic",
        "speculative",
        "tbo",
        "dp",
        "cp",
        "pdmux",
        "pp",
        "lora",
        "compile",
        "torch",
        "debug",
        "v1",
    ],
)
def test_full_graph_capture_is_limited_to_supported_c4_kvduo(monkeypatch, reason):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    c = coordinator()
    runner.model_runner = SimpleNamespace(
        hisparse_coordinator=c, server_args=SimpleNamespace(enable_lora=False)
    )
    runner.attn_backend = SimpleNamespace(
        dsa_topk_backend=DSATopKBackend.SGL_KERNEL,
        debug_use_external_c4_sparse_indices=False,
    )
    runner.capture_forward_mode = ForwardMode.DECODE
    runner.captured_req_width = 1
    runner.backend = FullCudaGraphBackend.__new__(FullCudaGraphBackend)
    runner.enable_two_batch_overlap = runner.require_mlp_sync = (
        runner.enable_prefill_cp
    ) = runner.enable_pdmux = runner.enable_torch_compile = False
    runner.pp_size = 1
    runner.device = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    if reason == "hisparse":
        c.enable_mixed_residency = False
    if reason == "generic":
        c.is_dsv4_hisparse = False
    if reason == "speculative":
        runner.captured_req_width = 2
    if reason == "tbo":
        runner.enable_two_batch_overlap = True
    if reason == "dp":
        runner.require_mlp_sync = True
    if reason == "cp":
        runner.enable_prefill_cp = True
    if reason == "pdmux":
        runner.enable_pdmux = True
    if reason == "pp":
        runner.pp_size = 2
    if reason == "lora":
        runner.model_runner.server_args.enable_lora = True
    if reason == "compile":
        runner.enable_torch_compile = True
    if reason == "torch":
        runner.attn_backend.dsa_topk_backend = DSATopKBackend.TORCH
    if reason == "debug":
        runner.attn_backend.debug_use_external_c4_sparse_indices = True
    with envs.SGLANG_OPT_USE_TOPK_V2.override(
        reason != "v1"
    ), envs.SGLANG_TOPK_TRANSFORM_512_TORCH.override(False):
        assert runner._can_capture_kvduo_full_graph() == (reason is None)


def test_begin_decode_does_not_mutate_banks_on_bad_controls():
    c = coordinator()
    before = c.full_touch_clock.clone()
    with pytest.raises(ValueError):
        begin_kvduo_decode(
            c.full_touch_clock,
            c.kvduo_full_clock_base,
            c.kvduo_observe,
            c.num_real_reqs,
            batch_size=-1,
        )
    assert torch.equal(c.full_touch_clock, before)


def test_mixed_fused_rows_reserve_layer_clocks_at_replay_boundary():
    c = coordinator()
    c.publish_kvduo_decode_controls(2, fused_full_rows=True)
    assert c.kvduo_full_clock_base.item() == 10
    assert c.full_touch_clock.item() == 13
    c.publish_kvduo_decode_controls(2, fused_full_rows=True)
    assert c.kvduo_full_clock_base.item() == 13
    assert c.full_touch_clock.item() == 16
