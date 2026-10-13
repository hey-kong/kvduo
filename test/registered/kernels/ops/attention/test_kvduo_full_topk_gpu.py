import subprocess
import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import (
    _jit_topk_v2_module,
    plan_topk_v2,
    topk_transform_512_v2,
)
from sglang.kernels.ops.kvcache.kvduo_full_decode import (
    KVDuoFullTopK,
    begin_kvduo_decode,
)
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cuda()
    or torch.cuda.get_device_capability()[0] < 9,
    reason="FULL TopK fusion requires Hopper or newer",
)


@pytest.mark.parametrize(
    "batch,seq,k", [(4, 4096, 512), (4, 40003, 2048), (31, 70003, 512)]
)
@pytest.mark.parametrize("multi_stream", [False, True])
def test_fused_full_topk_replay_matches_addresses_page_ages_and_observations(
    batch, seq, k, multi_stream
):
    width = (seq + 63) // 64 * 64
    scores = torch.arange(width, device="cuda", dtype=torch.float32).repeat(batch, 1)
    lengths = torch.zeros(batch, dtype=torch.int32, device="cuda")
    lengths[:2] = torch.tensor([seq, 256], dtype=torch.int32, device="cuda")
    requests = torch.full((batch,), 9999, dtype=torch.int64, device="cuda")
    requests[:2] = torch.tensor([1, 0], device="cuda")
    lookup = torch.arange(2 * width, device="cuda", dtype=torch.int64).reshape(2, width)
    mapping = torch.arange(64, 64 + 2 * width, device="cuda", dtype=torch.int64)
    touches = torch.zeros_like(mapping)
    unused = torch.full((batch, k), -123, dtype=torch.int32, device="cuda")
    pages = torch.empty((batch, 0), dtype=torch.int32, device="cuda")
    raw = torch.empty_like(unused)
    outputs = torch.full((3, batch, k), -99, dtype=torch.int32, device="cuda")
    stats = torch.zeros((3, 2, 2), dtype=torch.int32, device="cuda")
    status = torch.zeros((3, 2), dtype=torch.int32, device="cuda")
    growth = torch.ones_like(status)
    ema = torch.full((3, 2, 2), 64.0, device="cuda")
    steps = torch.full((3, 2), 65, dtype=torch.int64, device="cuda")
    observe = torch.zeros(1, dtype=torch.int32, device="cuda")
    real = torch.zeros(1, dtype=torch.int32, device="cuda")
    clock = torch.tensor([10], dtype=torch.int64, device="cuda")
    base = torch.zeros_like(clock)
    metadata = plan_topk_v2(lengths)
    layer_stride = 2 * width + 128
    specs = [
        KVDuoFullTopK(
            outputs[layer],
            requests,
            lookup,
            mapping,
            touches,
            status[layer],
            stats[layer],
            growth[layer],
            observe,
            real,
            base,
            ema[layer],
            steps[layer],
            layer,
            layer_stride,
            64,
        )
        for layer in range(3)
    ]
    module = _jit_topk_v2_module()
    indexer_stream = torch.cuda.Stream() if multi_stream else None

    def select():
        if indexer_stream is None:
            run()
        else:
            stream = torch.cuda.current_stream()
            indexer_stream.wait_stream(stream)
            with torch.cuda.stream(indexer_stream):
                run()
            stream.wait_stream(indexer_stream)

    def run():
        module.topk_plan(lengths, metadata, 0)
        for spec in specs:
            topk_transform_512_v2(
                scores,
                lengths,
                pages,
                unused,
                64,
                metadata,
                raw,
                raw_only=True,
                full_metadata=spec,
            )

    # Match the model's two FULL-variant warmups, including its auxiliary stream.
    select()
    select()
    torch.cuda.synchronize()
    assert not stats.any() and torch.all(outputs == -1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select()
    banks = (
        scores,
        lengths,
        requests,
        lookup,
        mapping,
        touches,
        raw,
        outputs,
        stats,
        status,
        growth,
        ema,
        steps,
        clock,
        base,
    )
    identities = [(t.data_ptr(), t.shape, t.stride()) for t in banks]
    for step in range(3):
        scores.neg_()
        requests[:2] = torch.tensor([step % 2, 1 - step % 2], device="cuda")
        before_stats = stats.clone()
        before_ema = ema.clone()
        before_steps = steps.clone()
        before_clock = clock.item()
        begin_kvduo_decode(clock, base, observe, real, batch_size=2, full_layers=3)
        graph.replay()
        torch.cuda.synchronize()
        assert clock.item() == before_clock + 3 and base.item() == before_clock
        assert torch.all(outputs[:, 2:] == -1) and torch.all(unused == -123)
        assert not status.any() and not growth.any()
        for row, length in enumerate(lengths[:2].tolist()):
            ids = raw[row, : min(length, k)].long()
            expected = torch.topk(scores[row, :length], min(k, length)).indices
            assert torch.equal(ids.sort().values, expected.sort().values)
            rid = int(requests[row])
            logical = lookup[rid, ids]
            for layer in range(3):
                assert torch.equal(
                    outputs[layer, row, : ids.numel()].long(),
                    mapping[logical] + layer * layer_stride,
                )
                assert torch.all(outputs[layer, row, ids.numel() :] == -1)
                assert (stats - before_stats)[layer, rid].tolist() == [0, ids.numel()]
            ages = touches.reshape(-1, 64).amax(1)
            assert torch.all(ages[torch.unique(logical // 64)] == before_clock + 3)
        assert torch.allclose(ema, before_ema * (63 / 64), rtol=0, atol=0)
        assert torch.equal(steps, before_steps + 1)
        assert identities == [(t.data_ptr(), t.shape, t.stride()) for t in banks]


@pytest.mark.parametrize("replay", [False, True])
def test_full_topk_missing_mapping_fails_before_attention_in_an_isolated_process(
    replay,
):
    script = r"""
import os, sys, torch
from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_512_v2
from sglang.kernels.ops.kvcache.kvduo_full_decode import KVDuoFullTopK
scores=torch.zeros((1,4),device="cuda")
seq=torch.ones(1,dtype=torch.int32,device="cuda")
pages=torch.empty((1,0),dtype=torch.int32,device="cuda")
out=torch.empty((1,512),dtype=torch.int32,device="cuda")
raw=torch.empty_like(out)
resolved=torch.empty_like(out)
plan=plan_topk_v2(seq)
i32=lambda shape:torch.zeros(shape,dtype=torch.int32,device="cuda")
i64=lambda shape:torch.zeros(shape,dtype=torch.int64,device="cuda")
real=i32((1,))
spec=KVDuoFullTopK(resolved,i64((1,)),i64((1,4)),i64((1,)),i64((1,)),i32((1,)),i32((1,2)),i32((1,)),torch.ones(1,dtype=torch.int32,device="cuda"),real,i64((1,)),torch.zeros((1,2),device="cuda"),i64((1,)),0,64,64)
def run(): topk_transform_512_v2(scores,seq,pages,out,64,plan,raw,raw_only=True,full_metadata=spec)
run()
torch.cuda.synchronize()
if sys.argv[1] == "True":
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): run()
real.fill_(1)
try:
    graph.replay() if sys.argv[1] == "True" else run()
    torch.cuda.synchronize()
except RuntimeError as error:
    print("expected device assertion:",error,flush=True)
    os._exit(0)
os._exit(2)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(replay)],
        text=True,
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "KVDuo FULL TopK missing resident KV" in result.stdout + result.stderr
