"""Production frequency resolver, indirection and merge payload regressions."""

from types import SimpleNamespace

import pytest
import torch
from sglang.kernels.ops.kvcache.kvduo_frequency import merge, merge_target, resolve
from sglang.srt.mem_cache.sparsity.core.kvduo_frequency import (
    FrequencyBuffer,
    pack_frequency,
    plan_frequency_merge,
    unpack_frequency,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def fixture():
    device = "cuda"
    backing = torch.full((8192, 8), -999, dtype=torch.int32, device=device)
    host = torch.arange(1024 * 8, dtype=torch.int32).reshape(1024, 8).pin_memory()
    buffers = [
        FrequencyBuffer.empty(
            torch.arange(
                2048 + r * 1024, 2560 + r * 1024, dtype=torch.int32, device=device
            )
        )
        for r in range(2)
    ]
    mapping = torch.zeros(2048, dtype=torch.int64, device=device)
    mapping[1:513] = torch.arange(1, 513, device=device)
    backing[1:513] = host[1:513].to(device)
    c = SimpleNamespace(
        top_k=512,
        is_dsv4_hisparse=False,
        kvduo_hot_pointers=torch.tensor(
            [[b.pointers() for b in buffers]], dtype=torch.int64, device=device
        ),
        kvduo_req_hot_capacity_gpu=torch.full(
            (1, 2), 512, dtype=torch.int32, device=device
        ),
        num_real_reqs=torch.tensor([2], dtype=torch.int32, device=device),
        req_to_full_lookup=torch.arange(2048, dtype=torch.int64, device=device).view(
            2, 1024
        ),
        mem_pool_device=SimpleNamespace(
            full_to_hisparse_device_index_mapping=mapping,
            get_flat_key_buffer=lambda: backing,
            flat_layer_slot_stride=8192,
        ),
        full_last_touch=torch.zeros(2048, dtype=torch.int64, device=device),
        full_data_version=torch.ones(2048, dtype=torch.int64, device=device),
        full_host_version=torch.ones(2048, dtype=torch.int64, device=device),
        req_to_host_pool=torch.arange(1024, dtype=torch.int64, device=device).repeat(
            2, 1
        ),
        mem_pool_host=SimpleNamespace(kv_buffer=[host]),
        item_size_bytes=32,
        full_touch_clock=torch.ones(1, dtype=torch.int64, device=device),
        kvduo_observe=torch.ones(1, dtype=torch.int32, device=device),
        kvduo_swap_status=torch.zeros((1, 2), dtype=torch.int32, device=device),
        kvduo_resolver_stats=torch.zeros((1, 2, 2), dtype=torch.int32, device=device),
        kvduo_hot_growth_events=torch.zeros((1, 2), dtype=torch.int32, device=device),
        kvduo_frequency_scratch=torch.empty(
            (2, 4608), dtype=torch.int32, device=device
        ),
    )
    c.kvduo_hot_pointers[0, 0].zero_()
    c.kvduo_req_hot_capacity_gpu[0, 0] = 0
    selected = torch.stack((torch.arange(1, 513), torch.arange(512, 1024))).to(
        device=device, dtype=torch.int32
    )
    requests = torch.arange(2, dtype=torch.int64, device=device)
    lengths = torch.full((2,), 1024, dtype=torch.int64, device=device)
    output = torch.empty_like(selected)
    return c, buffers, selected, requests, lengths, output, backing, host


def test_graph_replays_real_host_misses_after_growth_and_merge():
    c, buffers, selected, requests, lengths, output, backing, host = fixture()
    run = lambda: resolve(c, 0, requests, lengths, selected, output)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    pointer_address = c.kvduo_hot_pointers.data_ptr()
    for iteration in range(3):
        if iteration == 1:
            buffers[1] = buffers[1].grown(
                torch.arange(3584, 3648, dtype=torch.int32, device="cuda")
            )
        elif iteration == 2:
            buffers[1].frequency.zero_()
            buffers[1], kept = merge(
                buffers[1],
                backing,
                k=64,
                minimum=8,
                paged=False,
                top_k=512,
                item_bytes=32,
                publication=(
                    c.kvduo_hot_pointers[0, 1],
                    c.kvduo_req_hot_capacity_gpu[0, 1:2],
                    c.kvduo_hot_growth_events[0, 1:2],
                ),
            )
            assert c.kvduo_hot_pointers[0, 1].tolist() == buffers[1].pointers()
            assert c.kvduo_req_hot_capacity_gpu[0, 1].item() == 512
        b = buffers[1]
        c.kvduo_hot_pointers[0, 1].copy_(
            torch.tensor(b.pointers(), dtype=torch.int64, device="cuda")
        )
        c.kvduo_req_hot_capacity_gpu[0, 1] = b.tokens.numel()
        # Reset before EVERY replay: these must be genuine H2D misses.
        b.tokens.fill_(-1)
        b.frequency.zero_()
        backing[b.locations.long()] = -999
        c.kvduo_resolver_stats.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert c.kvduo_resolver_stats[0, 1].tolist() == [512, 512]
        assert torch.equal(backing[output[1].long()].cpu(), host[512:])
        assert torch.equal(backing[output[0].long()].cpu(), host[1:513])
        assert unpack_frequency(b.frequency, b.tokens.numel()) == [0] * b.tokens.numel()
        assert c.kvduo_hot_pointers.data_ptr() == pointer_address
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        assert unpack_frequency(b.frequency, 512) == [3] * 512

    # The same graph must handle a FULL -> HOT transition without recapture.
    c.mem_pool_device.full_to_hisparse_device_index_mapping.zero_()
    c.kvduo_hot_pointers[0, 0].copy_(
        torch.tensor(buffers[0].pointers(), dtype=torch.int64, device="cuda")
    )
    c.kvduo_req_hot_capacity_gpu[0, 0] = 512
    c.kvduo_resolver_stats.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert c.kvduo_resolver_stats[0, 0].tolist() == [512, 512]
    assert torch.equal(backing[output[0].long()].cpu(), host[1:513])
    # Padding rows cannot update frequencies or perform transfers.
    c.num_real_reqs.fill_(1)
    before = buffers[1].frequency.clone()
    graph.replay()
    torch.cuda.synchronize()
    assert output[1].tolist() == [-1] * 512
    assert torch.equal(buffers[1].frequency, before)


def test_occupied_replacement_protects_current_hits_and_initializes_zero():
    c, buffers, selected, requests, lengths, output, backing, host = fixture()
    run = lambda: resolve(c, 0, requests, lengths, selected, output)
    run()
    selected[1, :256] = torch.arange(256, device="cuda")
    run()
    torch.cuda.synchronize()
    b = buffers[1]
    assert c.kvduo_hot_growth_events[0, 1].item() == 1
    assert torch.equal(backing[output[1].long()].cpu(), host[selected[1].cpu().long()])
    freq = unpack_frequency(b.frequency, b.tokens.numel())
    assert all(f == (0 if t < 256 else 1) for t, f in zip(b.tokens.tolist(), freq))
    # Duplicates increment the one resident entry, saturation never wraps.
    selected[1].fill_(0)
    run()
    torch.cuda.synchronize()
    assert (
        unpack_frequency(b.frequency, b.tokens.numel())[b.tokens.tolist().index(0)] == 3
    )
    selected[1].fill_(-1)
    before = b.frequency.clone()
    run()
    assert torch.equal(b.frequency, before)


@pytest.mark.parametrize("pages", [2, 9, 17])
@pytest.mark.parametrize("k", [64, 4992])
@pytest.mark.parametrize("minimum", [1, 2])
def test_gpu_merge_matches_frequency_reference_and_copies_payloads(pages, k, minimum):
    capacity = pages * k
    backing = torch.arange(
        (capacity + 64) * 8, dtype=torch.int32, device="cuda"
    ).reshape(-1, 8)
    b = FrequencyBuffer.empty(
        torch.arange(64, capacity + 64, dtype=torch.int32, device="cuda")
    )
    b.tokens.copy_(torch.arange(capacity, dtype=torch.int32, device="cuda"))
    freq = [(s * 7 + s // k) % 4 for s in range(capacity)]
    b.frequency.copy_(pack_frequency(freq, device="cuda"))
    expected = plan_frequency_merge(list(range(capacity)), freq, k, minimum)
    before = backing.clone()
    new, kept = merge(
        b, backing, k=k, minimum=minimum, paged=False, top_k=512, item_bytes=32
    )
    assert kept == expected.kept_pages
    assert new.tokens.tolist() == list(expected.tokens)
    assert unpack_frequency(new.frequency, new.tokens.numel()) == list(
        expected.frequency
    )
    valid = new.tokens >= 0
    assert torch.equal(
        backing[new.locations[valid].long()], before[new.tokens[valid].long() + 64]
    )


def test_paged_startup_warms_full_reclaim_resolver_and_frequency_merge():
    from sglang.kernels.ops.kvcache.kvduo_warmup import warmup_kvduo_reclaim

    # DSV4 C4 page layout: 576 value bytes + 8 scale bytes per token.
    backing = torch.zeros(
        (1, ((584 * 64 + 575) // 576) * 576), dtype=torch.uint8, device="cuda"
    )
    before = backing.clone()
    warmup_kvduo_reclaim(
        backing=backing, page_size=64, storage_layers=3, top_k=512, paged=True
    )
    assert torch.equal(backing, before)


def test_large_hot_capacity_uses_global_scratch_without_int16_slot_limit():
    c, buffers, selected, requests, lengths, output, _, host = fixture()
    backing = torch.full((50000, 8), -999, dtype=torch.int32, device="cuda")
    c.mem_pool_device.get_flat_key_buffer = lambda: backing
    b = FrequencyBuffer.empty(
        torch.arange(1024, 46080, dtype=torch.int32, device="cuda")
    )
    b.tokens[40000] = 512
    backing[b.locations[40000].long()] = host[512].to("cuda")
    c.kvduo_hot_pointers[0, 1].copy_(
        torch.tensor(b.pointers(), dtype=torch.int64, device="cuda")
    )
    c.kvduo_req_hot_capacity_gpu[0, 1] = b.tokens.numel()
    selected[0].fill_(-1)
    resolve(c, 0, requests, lengths, selected, output)
    torch.cuda.synchronize()
    assert output[1, 0] == b.locations[40000]
    assert unpack_frequency(b.frequency, b.tokens.numel())[40000] == 1
    assert torch.equal(backing[output[1].long()].cpu(), host[512:])


def test_c4_host_restore_and_merge_preserve_values_and_scales():
    from test_hisparse import DSV4_PAGE_BYTES, _read_dsv4_token, _write_dsv4_token

    c, buffers, selected, requests, lengths, output, _, _ = fixture()
    c.is_dsv4_hisparse = True
    backing = torch.zeros((64, DSV4_PAGE_BYTES), dtype=torch.uint8, device="cuda")
    host = torch.zeros((16, DSV4_PAGE_BYTES), dtype=torch.uint8, pin_memory=True)
    for i in range(512, 1024):
        _write_dsv4_token(host, i, seed=i % 239)
    c.mem_pool_device.get_flat_key_buffer = lambda: backing
    c.mem_pool_host.kv_buffer = [host]
    selected[0].fill_(-1)
    resolve(c, 0, requests, lengths, selected, output)
    torch.cuda.synchronize()
    snapshot = backing.cpu()
    for i in (0, 31, 63, 255, 511):
        assert torch.equal(
            _read_dsv4_token(snapshot, int(output[1, i])),
            _read_dsv4_token(host, i + 512),
        )
    b = buffers[1]
    frequencies = [3 if i % 63 == 0 else 0 for i in range(512)]
    b.frequency.copy_(pack_frequency(frequencies, device="cuda"))
    new, kept = merge(
        b, backing, k=64, minimum=1, paged=True, top_k=512, item_bytes=584
    )
    snapshot = backing.cpu()
    for token, loc in zip(new.tokens.tolist(), new.locations.tolist()):
        if token >= 0:
            assert torch.equal(
                _read_dsv4_token(snapshot, loc), _read_dsv4_token(host, token)
            )


@pytest.mark.parametrize("block_size", [256, 960])
def test_parallel_victim_selection_matches_stable_frequency_order(block_size):
    c, buffers, selected, requests, lengths, output, backing, host = fixture()
    c.swap_in_block_size = block_size
    b = FrequencyBuffer.empty(
        torch.arange(3072, 3648, dtype=torch.int32, device="cuda")
    )
    b.tokens[:512] = torch.arange(512, dtype=torch.int32, device="cuda")
    frequencies = [i % 3 if i < 48 else 3 for i in range(512)] + [0] * 64
    b.frequency.copy_(pack_frequency(frequencies, device="cuda"))
    backing[b.locations[:512].long()] = host[:512].to("cuda")
    c.kvduo_hot_pointers[0, 1].copy_(
        torch.tensor(b.pointers(), dtype=torch.int64, device="cuda")
    )
    c.kvduo_req_hot_capacity_gpu[0, 1] = 576
    selected[1] = torch.arange(256, 768, dtype=torch.int32, device="cuda")
    victims = list(range(512, 576)) + sorted(
        range(256), key=lambda i: (frequencies[i], i)
    )
    expected = list(range(512)) + [-1] * 64
    expected_freq = frequencies[:]
    for slot, token in zip(victims[:256], range(512, 768)):
        expected[slot] = token
        expected_freq[slot] = 0
    for _ in range(3):
        b.tokens.copy_(
            torch.tensor(list(range(512)) + [-1] * 64, dtype=torch.int32, device="cuda")
        )
        b.frequency.copy_(pack_frequency(frequencies, device="cuda"))
        c.kvduo_resolver_stats.zero_()
        resolve(c, 0, requests, lengths, selected, output)
        torch.cuda.synchronize()
        assert b.tokens.tolist() == expected
        assert unpack_frequency(b.frequency, 576) == expected_freq
        assert c.kvduo_resolver_stats[0, 1].tolist() == [256, 512]
        assert c.kvduo_hot_growth_events[0, 1].item() == 1
        assert torch.equal(backing[output[1].long()].cpu(), host[256:768])


def test_graph_replay_after_batched_publication_and_metadata_reuse():
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

    c, buffers, selected, requests, lengths, output, backing, host = fixture()
    c.device = "cuda"
    run = lambda: resolve(c, 0, requests, lengths, selected, output)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    table_address = c.kvduo_hot_pointers.data_ptr()
    first_pointers = None
    for start in (3584, 3648):
        buffers[1] = buffers[1].grown(
            torch.arange(start, start + 64, dtype=torch.int32, device="cuda")
        )
        if first_pointers is None:
            first_pointers = buffers[1].pointers()
        else:
            assert buffers[1].pointers() == first_pointers
        HiSparseCoordinator._publish_kvduo_frequency_buffers(
            c, [((0, r), b) for r, b in enumerate(buffers)]
        )
        buffers[1].tokens.fill_(-1)
        buffers[1].frequency.zero_()
        c.kvduo_resolver_stats.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert c.kvduo_hot_pointers.data_ptr() == table_address
        assert c.kvduo_req_hot_capacity_gpu[0, 1].item() == buffers[1].tokens.numel()
        assert c.kvduo_resolver_stats[0, 1].tolist() == [512, 512]
        assert torch.equal(backing[output[1].long()].cpu(), host[512:])


@pytest.mark.parametrize("block_size", [256, 960])
@pytest.mark.parametrize("padded", [False, True])
def test_mixed_batch_full_row_skips_hot_scratch_with_real_h2d(block_size, padded):
    c, buffers, selected, requests, lengths, output, backing, host = fixture()
    c.swap_in_block_size = block_size
    if padded:
        selected[0, 256:] = -1
        selected[0, :256] = 1  # Count duplicate FULL accesses individually.
    c.kvduo_frequency_scratch.fill_(12345)
    run = lambda: resolve(c, 0, requests, lengths, selected, output)
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        buffers[1].tokens.fill_(-1)
        buffers[1].frequency.zero_()
        backing[buffers[1].locations.long()] = -999
        c.kvduo_frequency_scratch.fill_(12345)
        c.kvduo_resolver_stats.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.all(c.kvduo_frequency_scratch[0] == 12345)
        assert c.kvduo_resolver_stats[0, 0].tolist() == [0, 256 if padded else 512]
        assert c.kvduo_resolver_stats[0, 1].tolist()[0] == 512
        assert torch.equal(backing[output[1].long()], host[512:1024].cuda())
        valid = selected[0] >= 0
        assert torch.equal(output[0, valid], selected[0, valid])
        assert torch.all(output[0, ~valid] == -1)


@pytest.mark.parametrize("k,pages", [(17, 41), (64, 257), (4992, 35)])
def test_parallel_merge_reuses_workspace_and_matches_stable_reference(k, pages):
    workspace = {}
    capacity = k * pages
    # Exercise both page-selection branches, packed-word boundary sharing,
    # no-release decay, full compaction, and repeated use of dirty scratch.
    for pattern in ("all", "mixed", "none", "mixed"):
        backing = torch.arange(
            (capacity + 64) * 4, dtype=torch.int32, device="cuda"
        ).view(-1, 4)
        b = FrequencyBuffer.empty(
            torch.arange(64, capacity + 64, dtype=torch.int32, device="cuda")
        )
        tokens = [s if pattern == "all" or s % 11 else -1 for s in range(capacity)]
        freq = [
            3 if pattern == "all" else 0 if pattern == "none" else (s * 7 + s // k) % 4
            for s in range(capacity)
        ]
        b.tokens.copy_(torch.tensor(tokens, dtype=torch.int32, device="cuda"))
        b.frequency.copy_(pack_frequency(freq, device="cuda"))
        expected = plan_frequency_merge(tokens, freq, k, 2)
        before = backing.clone()
        old_addresses = {name: value.data_ptr() for name, value in workspace.items()}
        new, kept = merge(
            b,
            backing,
            k=k,
            minimum=2,
            paged=False,
            top_k=512,
            item_bytes=16,
            workspace=workspace,
        )
        assert kept == expected.kept_pages
        assert new.tokens.tolist() == list(expected.tokens)
        assert unpack_frequency(new.frequency, new.tokens.numel()) == list(
            expected.frequency
        )
        valid = new.tokens >= 0
        assert torch.equal(
            backing[new.locations[valid].long()], before[new.tokens[valid].long() + 64]
        )
        assert all(
            workspace[name].data_ptr() == address
            for name, address in old_addresses.items()
        )


@pytest.mark.parametrize("count", [0, 1, 4, 5, 12])
def test_merge_target_counts_valid_threes_before_mutation(count):
    k, capacity = 4, 16
    buffer = FrequencyBuffer.empty(
        torch.arange(capacity, device="cuda", dtype=torch.int32)
    )
    buffer.tokens[:count] = torch.arange(count, device="cuda", dtype=torch.int32)
    buffer.frequency.copy_(pack_frequency([3] * capacity, device="cuda"))
    workspace = {}
    for _ in range(2):
        assert merge_target(
            buffer, k=k, minimum=1, top_k=512, paged=False, workspace=workspace
        ) == max(1, (count + k - 1) // k)
    assert unpack_frequency(buffer.frequency, capacity) == [3] * capacity
