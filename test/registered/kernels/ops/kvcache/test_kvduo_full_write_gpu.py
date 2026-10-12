import pytest
import torch

from sglang.kernels.ops.kvcache.kvduo_full_write import publish_kvduo_full_write
from sglang.srt.utils import is_cuda, is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not (is_cuda() or is_hip()),
    reason="FULL publication JIT tests require CUDA/ROCm",
)


@pytest.mark.parametrize("logical_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("request_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("items", [0, 32, 513])
def test_full_write_graph_matches_reference_with_strided_inputs_and_reused_locations(
    logical_dtype, request_dtype, position_dtype, items
):
    def strided(values, dtype):
        backing = torch.full((2 * items,), -99, dtype=dtype, device="cuda")
        result = backing[::2]
        result.copy_(values.to(device="cuda", dtype=dtype))
        return result

    logical = strided(torch.arange(items) + 3, logical_dtype)
    requests = strided(torch.arange(items) % 2, request_dtype)
    positions = strided(torch.arange(items) // 2, position_dtype)
    generation = torch.arange(items + 5, dtype=torch.int64, device="cuda") + 20
    data = torch.full_like(generation, 7)
    host = data.clone()
    touches = torch.full_like(generation, 40)
    clock = torch.tensor([41], dtype=torch.int64, device="cuda")
    lookup_backing = torch.full((4, items + 5), -1, dtype=torch.int64, device="cuda")
    lookup = lookup_backing[::2]
    banks = (generation, data, host, touches, clock, lookup_backing)
    addresses = [
        (bank.data_ptr(), bank.shape, bank.stride())
        for bank in (*banks, logical, requests, positions)
    ]

    def run():
        publish_kvduo_full_write(
            logical,
            generation,
            data,
            host,
            touches,
            clock,
            lookup=lookup,
            req_indices=requests,
            positions=positions,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        expected = [bank.cpu().clone() for bank in banks]
        g, d, h, t, c, l = expected
        ids, rows, pos = (
            logical.cpu().long(),
            requests.cpu().long(),
            positions.cpu().long(),
        )
        c += 1
        g[ids] += 1
        d[ids] = 1
        h[ids] = -1
        t[ids] = c[0]
        l[::2][rows, pos] = ids
        graph.replay()
        torch.cuda.synchronize()
        assert all(
            torch.equal(bank.cpu(), reference)
            for bank, reference in zip(banks, expected)
        )
    assert addresses == [
        (bank.data_ptr(), bank.shape, bank.stride())
        for bank in (*banks, logical, requests, positions)
    ]


@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("logical_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("request_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("items", [0, 32, 513])
def test_decode_full_write_graph_reads_alignment_on_each_replay(
    ratio, logical_dtype, request_dtype, length_dtype, items
):
    def strided(dtype):
        return torch.full((2 * items,), -99, dtype=dtype, device="cuda")[::2]

    locations = strided(logical_dtype)
    requests = strided(request_dtype)
    lengths = strided(length_dtype)
    generation = torch.arange(items + 5, dtype=torch.int64, device="cuda") + 20
    data = torch.full_like(generation, 7)
    host = data.clone()
    touches = torch.full_like(generation, 40)
    clock = torch.tensor([41], dtype=torch.int64, device="cuda")
    lookup_backing = torch.full((2 * items, 16), -1, dtype=torch.int64, device="cuda")
    lookup = lookup_backing[::2]
    banks = (generation, data, host, touches, clock, lookup_backing)
    addresses = [
        (bank.data_ptr(), bank.shape, bank.stride())
        for bank in (*banks, locations, requests, lengths)
    ]
    ordinal = torch.arange(items)
    logical_ids = ordinal + 3

    def inputs(mode, step):
        seq = (ordinal % 16 + 1) * ratio
        active = (
            torch.zeros(items, dtype=torch.bool)
            if mode == "none"
            else (
                torch.ones(items, dtype=torch.bool)
                if mode == "all"
                else (ordinal + step) % 3 != 0
            )
        )
        seq[~active] = 0 if ratio == 1 else seq[~active] + 1
        loc = logical_ids * ratio + ratio - 1
        row = (items - 1 - ordinal + step) % max(items, 1)
        loc[~active] = row[~active] = -99
        return loc, row, seq, active

    def update_inputs(mode, step):
        loc, row, seq, active = inputs(mode, step)
        locations.copy_(loc)
        requests.copy_(row)
        lengths.copy_(seq)
        return row, seq, active

    def run():
        publish_kvduo_full_write(
            locations,
            generation,
            data,
            host,
            touches,
            clock,
            lookup=lookup,
            req_indices=requests,
            positions=lengths,
            decode_compress_ratio=ratio,
        )

    update_inputs("mixed", 0)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for step, mode in enumerate(("none", "mixed", "all", "mixed", "none")):
        row, seq, active = update_inputs(mode, step)
        expected = [bank.cpu().clone() for bank in banks]
        g, d, h, t, c, l = expected
        if active.any():
            c += 1
            ids = logical_ids[active]
            g[ids] += 1
            d[ids] = 1
            h[ids] = -1
            t[ids] = c[0]
            l[::2][row[active], seq[active] // ratio - 1] = ids
        graph.replay()
        torch.cuda.synchronize()
        assert all(
            torch.equal(bank.cpu(), reference)
            for bank, reference in zip(banks, expected)
        )
    assert addresses == [
        (bank.data_ptr(), bank.shape, bank.stride())
        for bank in (*banks, locations, requests, lengths)
    ]
