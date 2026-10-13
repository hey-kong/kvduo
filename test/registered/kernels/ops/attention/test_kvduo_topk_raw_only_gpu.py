import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import (
    _jit_topk_v2_module,
    plan_topk_v2,
    topk_transform_512_v2,
)
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cuda()
    or torch.cuda.get_device_capability()[0] < 9,
    reason="TopK V2 requires Hopper or newer",
)


def make_case(batch, seq, k):
    width = (seq + 3) & ~3
    scores = torch.empty((batch, width + 4), device="cuda")[:, :width]
    scores.copy_(torch.linspace(-1, 1, width, device="cuda")[None, :])
    buckets = [seq, 0, min(k - 1, seq), min(k + 17, seq), min(12000, seq)]
    lengths = torch.tensor(
        [buckets[i % len(buckets)] for i in range(batch)],
        dtype=torch.int32,
        device="cuda",
    )
    pages = (
        torch.arange((width + 63) // 64, dtype=torch.int32, device="cuda")
        .flip(0)
        .repeat(batch, 1)
    )
    unused_pages = torch.empty((batch, 0), dtype=torch.int32, device="cuda")
    outputs = [
        torch.full((batch, k), -123, dtype=torch.int32, device="cuda") for _ in range(4)
    ]
    return scores, lengths, pages, unused_pages, outputs


def check(scores, lengths, pages, outputs):
    raw_only, untouched, mapped, reference_raw = outputs
    assert torch.all(untouched == -123)
    for row, length in enumerate(lengths.cpu().tolist()):
        actual = raw_only[row].cpu()
        baseline = reference_raw[row].cpu()
        valid = actual[actual >= 0]
        expected = torch.topk(
            scores[row, :length], min(length, actual.numel())
        ).indices.cpu()
        assert torch.equal(valid.sort().values.long(), expected.sort().values)
        assert torch.equal(actual.sort().values, baseline.sort().values)
        assert int((actual == -1).sum()) == actual.numel() - min(length, actual.numel())
    safe = reference_raw.long().clamp_min(0)
    transformed = pages.gather(1, safe // 64) * 64 + safe % 64
    expected_mapping = torch.where(reference_raw >= 0, transformed, -1)
    assert torch.equal(mapped.long(), expected_mapping.long())


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize(
    "batch,seq",
    [
        (4, 256),
        (4, 4096),
        (4, 12000),
        (4, 20000),
        (4, 40003),
        (31, 70003),
        (513, 70003),
    ],
)
def test_raw_only_matches_selection_without_reading_page_table(batch, seq, k):
    scores, lengths, pages, unused_pages, outputs = make_case(batch, seq, k)
    raw, untouched, mapped, reference = outputs
    metadata = plan_topk_v2(lengths)
    clock = torch.tensor([41], dtype=torch.int64, device="cuda")
    topk_transform_512_v2(
        scores,
        lengths,
        unused_pages,
        untouched,
        64,
        metadata,
        raw,
        raw_only=True,
        touch_clock=clock,
    )
    topk_transform_512_v2(scores, lengths, pages, mapped, 64, metadata, reference)
    torch.cuda.synchronize()
    assert clock.item() == 42
    check(scores, lengths, pages, outputs)


@pytest.mark.parametrize("batch", [4, 31])
@pytest.mark.parametrize("k", [512, 1024, 2048])
def test_raw_only_graph_replays_changing_lengths_and_scores(batch, k):
    scores, lengths, pages, unused_pages, outputs = make_case(batch, 70003, k)
    raw, untouched, mapped, reference = outputs
    metadata = plan_topk_v2(lengths)
    module = _jit_topk_v2_module()
    clock = torch.tensor([41], dtype=torch.int64, device="cuda")
    banks = (scores, lengths, pages, unused_pages, metadata, clock, *outputs)
    addresses = [(bank.data_ptr(), bank.shape, bank.stride()) for bank in banks]

    def run():
        module.topk_plan(lengths, metadata, 0)
        topk_transform_512_v2(
            scores,
            lengths,
            unused_pages,
            untouched,
            64,
            metadata,
            raw,
            raw_only=True,
            touch_clock=clock,
        )
        topk_transform_512_v2(scores, lengths, pages, mapped, 64, metadata, reference)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for step in range(3):
        scores.neg_()
        values = [0, min(k - 1, 70003), 4096, 12000, 20000, 40000, 70003]
        lengths.copy_(
            torch.tensor(
                [values[(row + step) % len(values)] for row in range(batch)],
                dtype=lengths.dtype,
                device="cuda",
            )
        )
        before = clock.item()
        graph.replay()
        torch.cuda.synchronize()
        assert clock.item() == before + 1
        check(scores, lengths, pages, outputs)
    assert addresses == [(bank.data_ptr(), bank.shape, bank.stride()) for bank in banks]
