import sys

import pytest
import torch

import sglang.kernels.ops.kvcache.hisparse as hisparse_ops
from sglang.kernels.ops.kvcache.hisparse import (
    allocate_hisparse_resolver_scratch,
    hisparse_resolver_scratch_words,
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
    transfer_cache_dsv4_mla,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd")
register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or is_npu()
    or is_xpu()
    or not (is_cuda() or is_hip()),
    reason="HiSparse JIT tests require CUDA/ROCm.",
)

DEVICE = "cuda"
DTYPE = torch.float32
KV_DIM = 8
HOT_BUFFER_SIZE = 4
PADDED_BUFFER_SIZE = HOT_BUFFER_SIZE + 1
HOST_CACHE_SIZE = 16
DEVICE_CACHE_SIZE = 16
ITEM_SIZE_BYTES = KV_DIM * torch.empty((), dtype=DTYPE).element_size()
DSV4_PAGE_SIZE = 64
DSV4_VALUE_BYTES = 576
DSV4_SCALE_BYTES = 8
DSV4_ITEM_BYTES = DSV4_VALUE_BYTES + DSV4_SCALE_BYTES
DSV4_PAGE_BYTES = ((DSV4_ITEM_BYTES * DSV4_PAGE_SIZE + 575) // 576) * 576
DSV4_SCALE_OFFSET = DSV4_VALUE_BYTES * DSV4_PAGE_SIZE


def _host_cache() -> torch.Tensor:
    host_cache = torch.empty(
        (HOST_CACHE_SIZE, 1, KV_DIM), dtype=DTYPE, device="cpu", pin_memory=True
    )
    host_cache.copy_(torch.arange(host_cache.numel(), dtype=DTYPE).view_as(host_cache))
    return host_cache


def test_mla_ffi_arguments_do_not_include_tensors(monkeypatch) -> None:
    """CUDA graph capture must not ask TVM FFI to convert Torch tensors."""

    class FakeModule:
        args = None

        def load_cache_to_device_buffer(self, *args) -> None:
            self.args = args

    module = FakeModule()
    monkeypatch.setattr(
        hisparse_ops, "_jit_sparse_module", lambda *args, **kwargs: module
    )

    top_k_tokens = torch.zeros((1, 1), dtype=torch.int32, device=DEVICE)
    load_cache_to_device_buffer_mla(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=torch.zeros((1, 2), dtype=torch.int32, device=DEVICE),
        host_cache_locs=torch.zeros((1, 1), dtype=torch.int64, device=DEVICE),
        device_buffer_locs=torch.zeros((1, 2), dtype=torch.int32, device=DEVICE),
        host_cache=_host_cache(),
        device_buffer=torch.zeros((1, 1, KV_DIM), dtype=DTYPE, device=DEVICE),
        top_k_device_locs=torch.zeros_like(top_k_tokens),
        req_pool_indices=torch.zeros(1, dtype=torch.int64, device=DEVICE),
        seq_lens=torch.ones(1, dtype=torch.int32, device=DEVICE),
        lru_slots=torch.zeros((1, 1), dtype=torch.int16, device=DEVICE),
        item_size_bytes=ITEM_SIZE_BYTES,
        num_top_k=1,
        hot_buffer_size=1,
        num_real_reqs=torch.ones(1, dtype=torch.int32, device=DEVICE),
    )

    assert module.args is not None
    assert module.args[4] != 0
    assert isinstance(module.args[4], int)
    assert not any(isinstance(arg, torch.Tensor) for arg in module.args)


def _dsv4_token_pattern(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    value = (
        (torch.arange(DSV4_VALUE_BYTES, dtype=torch.int16) + seed)
        .remainder(256)
        .to(torch.uint8)
    )
    scale = (
        (torch.arange(DSV4_SCALE_BYTES, dtype=torch.int16) + seed + 17)
        .remainder(256)
        .to(torch.uint8)
    )
    return value, scale


def _write_dsv4_token(cache: torch.Tensor, loc: int, seed: int) -> None:
    page = loc // DSV4_PAGE_SIZE
    offset = loc % DSV4_PAGE_SIZE
    value, scale = _dsv4_token_pattern(seed)
    cache[page, offset * DSV4_VALUE_BYTES : (offset + 1) * DSV4_VALUE_BYTES].copy_(
        value.to(cache.device)
    )
    scale_start = DSV4_SCALE_OFFSET + offset * DSV4_SCALE_BYTES
    cache[page, scale_start : scale_start + DSV4_SCALE_BYTES].copy_(
        scale.to(cache.device)
    )


def _read_dsv4_token(cache: torch.Tensor, loc: int) -> torch.Tensor:
    page = loc // DSV4_PAGE_SIZE
    offset = loc % DSV4_PAGE_SIZE
    value = cache[page, offset * DSV4_VALUE_BYTES : (offset + 1) * DSV4_VALUE_BYTES]
    scale_start = DSV4_SCALE_OFFSET + offset * DSV4_SCALE_BYTES
    scale = cache[page, scale_start : scale_start + DSV4_SCALE_BYTES]
    return torch.cat([value, scale])


def _dsv4_ptrs(cache: torch.Tensor) -> torch.Tensor:
    return torch.tensor([cache.data_ptr()], dtype=torch.uint64, device=DEVICE)


def _run_kernel(
    *,
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    lru_slots: torch.Tensor,
    seq_len: int | None = None,
    seq_lens: torch.Tensor | None = None,
    seq_lens_dtype: torch.dtype = torch.int32,
    req_pool_indices: torch.Tensor | None = None,
    num_real_reqs: int | None = None,
    output_fill_value: int = -1,
) -> torch.Tensor:
    batch_size = top_k_tokens.shape[0]
    if req_pool_indices is None:
        req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=DEVICE)
    if seq_lens is None:
        seq_lens = torch.full(
            (batch_size,), seq_len, dtype=seq_lens_dtype, device=DEVICE
        )
    if num_real_reqs is None:
        num_real_reqs = batch_size

    out = torch.full_like(top_k_tokens, output_fill_value)
    load_cache_to_device_buffer_mla(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=out,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        lru_slots=lru_slots,
        item_size_bytes=ITEM_SIZE_BYTES,
        num_top_k=top_k_tokens.shape[1],
        hot_buffer_size=HOT_BUFFER_SIZE,
        page_size=1,
        block_size=256,
        num_real_reqs=torch.tensor([num_real_reqs], dtype=torch.int32, device=DEVICE),
    )
    torch.cuda.synchronize()
    return out


def _make_state(
    device_buffer_locs_rows: list[list[int]],
    device_buffer_tokens_rows: list[list[int]],
    newest_tokens: list[int],
):
    host_cache = _host_cache()
    device_buffer = torch.full(
        (DEVICE_CACHE_SIZE, 1, KV_DIM), -1, dtype=DTYPE, device=DEVICE
    )
    device_buffer_locs = torch.tensor(
        device_buffer_locs_rows, dtype=torch.int32, device=DEVICE
    )
    device_buffer_tokens = torch.tensor(
        device_buffer_tokens_rows, dtype=torch.int32, device=DEVICE
    )
    lru_slots = (
        torch.arange(HOT_BUFFER_SIZE, dtype=torch.int16, device=DEVICE)
        .view(1, -1)
        .repeat(device_buffer_locs.shape[0], 1)
    )
    host_cache_locs = (
        torch.arange(HOST_CACHE_SIZE, dtype=torch.int64, device=DEVICE)
        .view(1, -1)
        .repeat(device_buffer_locs.shape[0], 1)
    )

    # Slots 0..3 participate in LRU; slot 4 is the reserved newest slot.
    for rid, newest_token in enumerate(newest_tokens):
        for slot, token in enumerate(device_buffer_tokens_rows[rid][:HOT_BUFFER_SIZE]):
            if token >= 0:
                device_buffer[device_buffer_locs[rid, slot]].copy_(
                    host_cache[token].to(DEVICE, non_blocking=True)
                )
        device_buffer[device_buffer_locs[rid, HOT_BUFFER_SIZE]].copy_(
            host_cache[newest_token].to(DEVICE, non_blocking=True)
        )
    torch.cuda.synchronize()

    return {
        "host_cache": host_cache,
        "device_buffer": device_buffer,
        "device_buffer_locs": device_buffer_locs,
        "device_buffer_tokens": device_buffer_tokens,
        "lru_slots": lru_slots,
        "host_cache_locs": host_cache_locs,
    }


@pytest.mark.skipif(is_hip(), reason="DSV4 paged-layout HiSparse test is CUDA-only.")
def test_transfer_cache_dsv4_mla_copies_paged_token() -> None:
    src_cache = torch.zeros((2, DSV4_PAGE_BYTES), dtype=torch.uint8, device=DEVICE)
    dst_cache = torch.zeros(
        (2, DSV4_PAGE_BYTES), dtype=torch.uint8, device="cpu", pin_memory=True
    )
    src_loc = DSV4_PAGE_SIZE + 6
    dst_loc = DSV4_PAGE_SIZE + 1
    _write_dsv4_token(src_cache, src_loc, seed=41)

    transfer_cache_dsv4_mla(
        src_ptrs=_dsv4_ptrs(src_cache),
        dst_ptrs=_dsv4_ptrs(dst_cache),
        src_indices=torch.tensor([src_loc], dtype=torch.int64, device=DEVICE),
        dst_indices=torch.tensor([dst_loc], dtype=torch.int64, device=DEVICE),
    )
    torch.cuda.synchronize()

    assert torch.equal(
        _read_dsv4_token(dst_cache, dst_loc).to(DEVICE),
        _read_dsv4_token(src_cache, src_loc),
    )


@pytest.mark.skipif(is_hip(), reason="DSV4 paged-layout HiSparse test is CUDA-only.")
def test_dsv4_swap_in_reads_paged_host_layout() -> None:
    host_cache = torch.zeros(
        (2, DSV4_PAGE_BYTES), dtype=torch.uint8, device="cpu", pin_memory=True
    )
    device_buffer = torch.zeros((2, DSV4_PAGE_BYTES), dtype=torch.uint8, device=DEVICE)
    host_loc = DSV4_PAGE_SIZE + 1
    swap_loc = DSV4_PAGE_SIZE + 12
    _write_dsv4_token(host_cache, host_loc, seed=41)

    top_k_tokens = torch.tensor([[3]], dtype=torch.int32, device=DEVICE)
    device_buffer_tokens = torch.full(
        (1, PADDED_BUFFER_SIZE), -1, dtype=torch.int32, device=DEVICE
    )
    host_cache_locs = torch.zeros((1, 8), dtype=torch.int64, device=DEVICE)
    host_cache_locs[0, 3] = host_loc
    device_buffer_locs = torch.tensor(
        [[swap_loc, swap_loc + 1, swap_loc + 2, swap_loc + 3, swap_loc + 4]],
        dtype=torch.int32,
        device=DEVICE,
    )
    lru_slots = torch.arange(HOT_BUFFER_SIZE, dtype=torch.int16, device=DEVICE).view(
        1, -1
    )
    out = torch.full_like(top_k_tokens, -1)

    load_cache_to_device_buffer_dsv4_mla(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=out,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([8], dtype=torch.int32, device=DEVICE),
        lru_slots=lru_slots,
        item_size_bytes=DSV4_ITEM_BYTES,
        num_top_k=1,
        hot_buffer_size=HOT_BUFFER_SIZE,
        page_size=1,
        block_size=256,
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device=DEVICE),
    )
    torch.cuda.synchronize()

    assert out.item() == swap_loc
    assert torch.equal(
        _read_dsv4_token(device_buffer, swap_loc),
        _read_dsv4_token(host_cache, host_loc).to(DEVICE),
    )


@pytest.mark.parametrize("dsv4", [False, True])
@pytest.mark.parametrize("global_scratch", [False, True])
def test_standalone_hisparse_512_topk_960_threads_with_h2d_and_graph(
    dsv4, global_scratch
):
    if dsv4 and is_hip():
        pytest.skip("DSV4 paged-layout HiSparse test is CUDA-only")
    top_k, capacity, requests = 512, 1024, 2
    page = DSV4_PAGE_SIZE if dsv4 else 1
    padded = capacity + page
    seq_len = 2 * capacity + 1  # newest token starts a page in either layout
    host_stride = (seq_len + page - 1) // page * page
    row_shape = (DSV4_PAGE_BYTES,) if dsv4 else (1, KV_DIM)
    dtype = torch.uint8 if dsv4 else DTYPE
    host_cache = torch.empty(
        (requests * host_stride // page, *row_shape), dtype=dtype, pin_memory=True
    )
    host_cache.copy_(
        torch.arange(host_cache.numel(), dtype=torch.int32)
        .remainder(251)
        .to(dtype)
        .view_as(host_cache)
    )
    device_buffer = torch.full(
        (requests * padded // page, *row_shape), -1, dtype=dtype, device=DEVICE
    )
    for rid in range(requests):
        dst = rid * padded // page
        src = rid * host_stride // page
        device_buffer[dst : dst + capacity // page].copy_(
            host_cache[src : src + capacity // page].to(DEVICE)
        )
        device_buffer[dst + capacity // page].copy_(
            host_cache[src + (seq_len - 1) // page].to(DEVICE)
        )
    tokens = torch.full((requests, padded), -1, dtype=torch.int32, device=DEVICE)
    tokens[:, :capacity] = torch.arange(capacity, dtype=torch.int32, device=DEVICE)
    locations = torch.arange(requests * padded, dtype=torch.int32, device=DEVICE).view(
        requests, padded
    )
    host_locations = (
        torch.arange(seq_len, dtype=torch.int64, device=DEVICE)[None, :]
        + torch.arange(requests, dtype=torch.int64, device=DEVICE)[:, None]
        * host_stride
    )
    lru = torch.arange(capacity, dtype=torch.int16, device=DEVICE).repeat(requests, 1)
    # Both HOT hits and occupied-slot H2D misses, plus direct newest-token use.
    selections = (
        torch.cat(
            [
                torch.arange(128),
                torch.arange(capacity, capacity + 383),
                torch.tensor([seq_len - 1]),
            ]
        )
        .to(dtype=torch.int32, device=DEVICE)
        .repeat(requests, 1)
    )
    output = torch.full_like(selections, -1)
    req_indices = torch.tensor([1, 0], dtype=torch.int64, device=DEVICE)
    lengths = torch.full((requests,), seq_len, dtype=torch.int32, device=DEVICE)
    real_reqs = torch.tensor([requests], dtype=torch.int32, device=DEVICE)
    capacities = torch.full((requests,), capacity, dtype=torch.int32, device=DEVICE)
    scratch = (
        torch.empty(
            (requests, hisparse_resolver_scratch_words(top_k, capacity)),
            dtype=torch.int32,
            device=DEVICE,
        )
        if global_scratch
        else None
    )
    loader = (
        load_cache_to_device_buffer_dsv4_mla
        if dsv4
        else load_cache_to_device_buffer_mla
    )
    banks = (tokens, lru, device_buffer)
    original = [bank.clone() for bank in banks]

    def dispatch():
        loader(
            top_k_tokens=selections,
            device_buffer_tokens=tokens,
            host_cache_locs=host_locations,
            device_buffer_locs=locations,
            host_cache=host_cache,
            device_buffer=device_buffer,
            top_k_device_locs=output,
            req_pool_indices=req_indices,
            seq_lens=lengths,
            lru_slots=lru,
            item_size_bytes=DSV4_ITEM_BYTES if dsv4 else ITEM_SIZE_BYTES,
            num_top_k=top_k,
            hot_buffer_size=capacity,
            page_size=1,
            block_size=960,
            num_real_reqs=real_reqs,
            req_hot_buffer_sizes=capacities,
            resolver_scratch=scratch,
            # Standalone HiSparse allocates no KVDuo status/event banks.
            swap_status=None,
            hot_growth_events=None,
        )

    def reset():
        for bank, old in zip(banks, original):
            bank.copy_(old)
        output.fill_(-1)

    def verify():
        torch.cuda.synchronize()  # Report kernel errors at the actual resolver.
        resolved = output.cpu()
        snapshot = device_buffer.cpu()
        for bid, rid in enumerate([1, 0]):
            if dsv4:
                actual = torch.stack(
                    [_read_dsv4_token(snapshot, loc) for loc in resolved[bid].tolist()]
                )
                expected = torch.stack(
                    [
                        _read_dsv4_token(host_cache, rid * host_stride + token)
                        for token in selections[bid].cpu().tolist()
                    ]
                )
            else:
                actual = snapshot[resolved[bid].long()]
                expected = host_cache[rid * host_stride + selections[bid].cpu().long()]
            assert torch.equal(actual, expected)
        assert torch.equal(
            lru.sort(dim=1).values.cpu(),
            torch.arange(capacity, dtype=torch.int16).repeat(requests, 1),
        )

    dispatch()  # Compile and exercise real H2D misses before graph capture.
    verify()
    reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dispatch()
    # Capture itself filled the misses. Restore them so replay also tests H2D.
    reset()
    graph.replay()
    verify()


def _long_case():
    # One-request baseline used by the stateful cases below:
    # req 0 LRU slots      : [0, 1, 2, 3]
    # req 0 cached tokens  : slot0->1, slot1->4, slot2->2, slot3->5
    # req 0 physical locs  : slot0->9, slot1->7, slot2->3, slot3->5
    # req 0 newest slot    : slot4/newest -> token 7 at physical loc 11
    return _make_state([[9, 7, 3, 5, 11]], [[1, 4, 2, 5, -1]], [7])


@pytest.mark.parametrize("seq_lens_dtype", [torch.int32, torch.int64])
def test_load_cache_to_device_buffer_fast_path(seq_lens_dtype: torch.dtype) -> None:
    host_cache = _host_cache()
    device_buffer = torch.arange(
        DEVICE_CACHE_SIZE * KV_DIM, dtype=DTYPE, device=DEVICE
    ).view(DEVICE_CACHE_SIZE, 1, KV_DIM)
    device_buffer_before = device_buffer.clone()
    device_buffer_locs = torch.tensor(
        [[13, 9, 5, 1, 15]], dtype=torch.int32, device=DEVICE
    )
    device_buffer_tokens = torch.tensor(
        [[10, 11, 12, 13, -1]], dtype=torch.int32, device=DEVICE
    )
    device_buffer_tokens_before = device_buffer_tokens.clone()
    lru_slots = torch.tensor([[0, 1, 2, 3]], dtype=torch.int16, device=DEVICE)
    lru_slots_before = lru_slots.clone()

    # Short-sequence layout:
    # token position 0 -> physical loc 13
    # token position 1 -> physical loc 9
    # token position 2 -> physical loc 5
    #
    # seq_len <= HOT_BUFFER_SIZE should skip host loads and LRU mutations,
    # so top_k_tokens acts like direct indexing into device_buffer_locs.
    out = _run_kernel(
        top_k_tokens=torch.tensor([[2, 0, 1]], dtype=torch.int32, device=DEVICE),
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=torch.arange(
            HOST_CACHE_SIZE, dtype=torch.int64, device=DEVICE
        ).view(1, -1),
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        lru_slots=lru_slots,
        seq_len=3,
        seq_lens_dtype=seq_lens_dtype,
    )

    assert torch.equal(out.cpu(), torch.tensor([[5, 13, 9]], dtype=torch.int32))
    assert torch.equal(device_buffer_tokens.cpu(), device_buffer_tokens_before.cpu())
    assert torch.equal(lru_slots.cpu(), lru_slots_before.cpu())
    assert torch.equal(device_buffer.cpu(), device_buffer_before.cpu())


def test_load_cache_to_device_buffer_fast_path_overwrites_stale_output() -> None:
    state = _make_state([[9, 7, 3, 5, 11]], [[0, 1, 2, 3, -1]], [4])

    out = _run_kernel(
        top_k_tokens=torch.tensor([[1, -1, 0, 0]], dtype=torch.int32, device=DEVICE),
        seq_len=2,
        output_fill_value=123456,
        **state,
    )

    assert torch.equal(out.cpu(), torch.tensor([[7, -1, -1, -1]], dtype=torch.int32))


def test_kvduo_full_resident_fast_path_skips_empty_hot_view() -> None:
    state = _long_case()
    top_k_tokens = torch.tensor([[4, 2, 1]], dtype=torch.int32, device=DEVICE)
    out = torch.full_like(top_k_tokens, -1)
    lru_before = state["lru_slots"].clone()
    full_to_device_loc = torch.zeros(16, dtype=torch.int64, device=DEVICE)
    full_to_device_loc[:8] = torch.tensor(
        [12, 13, 14, 15, 8, 9, 10, 11], dtype=torch.int64, device=DEVICE
    )
    versions = torch.ones(16, dtype=torch.int64, device=DEVICE)
    resolver_stats = torch.zeros((1, 2), dtype=torch.int32, device=DEVICE)

    load_cache_to_device_buffer_mla(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=state["device_buffer_tokens"],
        host_cache_locs=state["host_cache_locs"],
        device_buffer_locs=state["device_buffer_locs"],
        host_cache=state["host_cache"],
        device_buffer=state["device_buffer"],
        top_k_device_locs=out,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([8], dtype=torch.int32, device=DEVICE),
        lru_slots=state["lru_slots"],
        item_size_bytes=ITEM_SIZE_BYTES,
        num_top_k=3,
        hot_buffer_size=HOT_BUFFER_SIZE,
        page_size=1,
        block_size=256,
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device=DEVICE),
        req_hot_buffer_sizes=torch.zeros(1, dtype=torch.int32, device=DEVICE),
        capacity_ema=torch.zeros(
            (state["device_buffer_tokens"].size(0), 9),
            dtype=torch.float32,
            device=DEVICE,
        ),
        capacity_steps=torch.zeros(
            state["device_buffer_tokens"].size(0), dtype=torch.int64, device=DEVICE
        ),
        req_to_logical_token=torch.arange(8, dtype=torch.int64, device=DEVICE).view(
            1, -1
        ),
        full_to_device_loc=full_to_device_loc,
        full_last_touch=torch.zeros(16, dtype=torch.int64, device=DEVICE),
        full_data_version=versions,
        full_host_version=versions.clone(),
        swap_status=torch.zeros(1, dtype=torch.int32, device=DEVICE),
        resolver_stats=resolver_stats,
        touch_clock=torch.tensor([7], dtype=torch.int64, device=DEVICE),
    )
    torch.cuda.synchronize()

    assert torch.equal(out.cpu(), torch.tensor([[8, 14, 13]], dtype=torch.int32))
    assert torch.equal(state["lru_slots"].cpu(), lru_before.cpu())
    assert torch.equal(resolver_stats.cpu(), torch.tensor([[0, 3]], dtype=torch.int32))


@pytest.mark.parametrize(
    "cached, selected, capacity, expected_events",
    [
        ([-1, -1, -1, -1], [3, 6], 2, 0),
        ([1, -1, -1, -1], [1, 3], 2, 0),
        ([1, 4, -1, -1], [1, 4], 2, 0),
        ([1, 4, -1, -1], [3, -1], 2, 1),
        ([1, 4, -1, -1], [3, 6], 2, 1),
        ([1, -1, -1, -1], [3, 6], 2, 1),
        ([1, 4, 2, 5], [3, 6], 4, 0),
    ],
)
def test_kvduo_h2d_capacity_event_and_lru_fallback(
    cached, selected, capacity, expected_events
) -> None:
    state = _long_case()
    state["device_buffer_tokens"][0, :4] = torch.tensor(
        cached, dtype=torch.int32, device=DEVICE
    )
    # Empty slots precede occupied ones in the active LRU permutation.
    slots = sorted(range(capacity), key=lambda i: cached[i] >= 0)
    state["lru_slots"][0, :capacity] = torch.tensor(
        slots, dtype=torch.int16, device=DEVICE
    )
    top_k = torch.tensor([selected], dtype=torch.int32, device=DEVICE)
    out = torch.full_like(top_k, -1)
    events = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    versions = torch.ones(16, dtype=torch.int64, device=DEVICE)
    status = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    kwargs = dict(
        **state,
        top_k_tokens=top_k,
        top_k_device_locs=out,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([8], dtype=torch.int32, device=DEVICE),
        item_size_bytes=ITEM_SIZE_BYTES,
        num_top_k=2,
        hot_buffer_size=HOT_BUFFER_SIZE,
        page_size=1,
        block_size=256,
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device=DEVICE),
        req_hot_buffer_sizes=torch.tensor([capacity], dtype=torch.int32, device=DEVICE),
        capacity_ema=torch.zeros(
            (state["device_buffer_tokens"].size(0), 9),
            dtype=torch.float32,
            device=DEVICE,
        ),
        capacity_steps=torch.zeros(
            state["device_buffer_tokens"].size(0), dtype=torch.int64, device=DEVICE
        ),
        req_to_logical_token=torch.arange(8, dtype=torch.int64, device=DEVICE).view(
            1, -1
        ),
        full_to_device_loc=torch.zeros(16, dtype=torch.int64, device=DEVICE),
        full_last_touch=torch.zeros(16, dtype=torch.int64, device=DEVICE),
        full_data_version=versions,
        full_host_version=versions.clone(),
        swap_status=status,
        resolver_stats=torch.zeros((1, 2), dtype=torch.int32, device=DEVICE),
        hot_growth_events=events,
        touch_clock=torch.tensor([7], dtype=torch.int64, device=DEVICE),
    )
    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()

    assert events.item() == expected_events
    assert status.item() == 0
    for index, token in enumerate(selected):
        location = out[0, index].item()
        if token < 0:
            assert location == -1
        else:
            assert location >= 0
            host_location = state["host_cache_locs"][0, token].item()
            assert torch.equal(
                state["device_buffer"][location].cpu(),
                state["host_cache"][host_location],
            )
            if token in cached:
                assert (
                    location
                    == state["device_buffer_locs"][0, cached.index(token)].item()
                )
    if capacity == HOT_BUFFER_SIZE and expected_events == 0:
        assert out.cpu().tolist() == [[9, 7]]
        assert state["device_buffer_tokens"][0, :4].cpu().tolist() == [3, 6, 2, 5]

    # The same workset is now all HOT hits, including after an LRU fallback.
    events.zero_()
    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()
    assert events.item() == 0

    # Capture once, then replay a new disjoint selection and a HOT-hit selection.
    # The event must come from device execution on every replay, without Python
    # re-running the resolver or reallocating its metadata.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        load_cache_to_device_buffer_mla(**kwargs)
    events.zero_()
    top_k.copy_(torch.tensor([[0, 2]], dtype=torch.int32, device=DEVICE))
    graph.replay()
    torch.cuda.synchronize()
    assert events.item() == int(capacity < HOT_BUFFER_SIZE)
    events.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert events.item() == 0


def _kvduo_observation_case(selected=(2, 1), *, capacity=4, k=1, lru=(2, 0, 3, 1)):
    state = _long_case()
    state["lru_slots"][0] = torch.tensor(lru, dtype=torch.int16, device=DEVICE)
    top_k = torch.tensor([selected], dtype=torch.int32, device=DEVICE)
    versions = torch.ones(16, dtype=torch.int64, device=DEVICE)
    kwargs = dict(
        **state,
        top_k_tokens=top_k,
        top_k_device_locs=torch.full_like(top_k, -1),
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([8], dtype=torch.int32, device=DEVICE),
        item_size_bytes=ITEM_SIZE_BYTES,
        num_top_k=len(selected),
        hot_buffer_size=HOT_BUFFER_SIZE,
        page_size=k,
        block_size=256,
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device=DEVICE),
        req_hot_buffer_sizes=torch.tensor([capacity], dtype=torch.int32, device=DEVICE),
        capacity_ema=torch.zeros(
            (state["device_buffer_tokens"].size(0), 9),
            dtype=torch.float32,
            device=DEVICE,
        ),
        capacity_steps=torch.zeros(
            state["device_buffer_tokens"].size(0), dtype=torch.int64, device=DEVICE
        ),
        req_to_logical_token=torch.arange(8, dtype=torch.int64, device=DEVICE).view(
            1, -1
        ),
        full_to_device_loc=torch.zeros(16, dtype=torch.int64, device=DEVICE),
        full_last_touch=torch.zeros(16, dtype=torch.int64, device=DEVICE),
        full_data_version=versions,
        full_host_version=versions.clone(),
        swap_status=torch.zeros(1, dtype=torch.int32, device=DEVICE),
        resolver_stats=torch.zeros((1, 2), dtype=torch.int32, device=DEVICE),
        hot_growth_events=torch.zeros(1, dtype=torch.int32, device=DEVICE),
        touch_clock=torch.ones(1, dtype=torch.int64, device=DEVICE),
        observe_cache=torch.ones(1, dtype=torch.int32, device=DEVICE),
        hot_base_capacities=torch.tensor([2], dtype=torch.int32, device=DEVICE),
        prefill_ema=torch.zeros((1, 2), dtype=torch.float32, device=DEVICE),
        prefill_steps=torch.zeros(1, dtype=torch.int64, device=DEVICE),
    )
    return kwargs


def test_load_cache_to_device_buffer_hits_newest_and_updates_lru() -> None:
    state = _long_case()

    # Query [4, 2, 7]:
    # 4 hits slot1 -> loc 7
    # 2 hits slot2 -> loc 3
    # 7 is the newest token -> reserved newest loc 11
    #
    # Hits move to the MRU tail, so [0, 1, 2, 3] becomes [0, 3, 1, 2].
    out = _run_kernel(
        top_k_tokens=torch.tensor([[4, 2, 7]], dtype=torch.int32, device=DEVICE),
        seq_len=8,
        **state,
    )

    assert torch.equal(out.cpu(), torch.tensor([[7, 3, 11]], dtype=torch.int32))
    assert torch.equal(
        state["device_buffer_tokens"].cpu(),
        torch.tensor([[1, 4, 2, 5, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        state["lru_slots"].cpu(), torch.tensor([[0, 3, 1, 2]], dtype=torch.int16)
    )


def test_load_cache_to_device_buffer_miss_uses_updated_lru_slot() -> None:
    state = _long_case()

    # Step 1: touch tokens [4, 2], so LRU becomes [0, 3, 1, 2].
    # Step 2: query token 6, which is a miss.
    # The kernel should reuse the new LRU head slot0, whose physical loc is 9.
    # This round has no regular hits, so the freshly loaded miss slot ends up at the tail.
    _run_kernel(
        top_k_tokens=torch.tensor([[4, 2]], dtype=torch.int32, device=DEVICE),
        seq_len=8,
        **state,
    )
    out = _run_kernel(
        top_k_tokens=torch.tensor([[6]], dtype=torch.int32, device=DEVICE),
        seq_len=8,
        **state,
    )

    assert torch.equal(out.cpu(), torch.tensor([[9]], dtype=torch.int32))
    assert torch.equal(
        state["device_buffer_tokens"].cpu(),
        torch.tensor([[6, 4, 2, 5, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        state["lru_slots"].cpu(), torch.tensor([[3, 1, 2, 0]], dtype=torch.int16)
    )
    assert torch.equal(state["device_buffer"][9].cpu(), state["host_cache"][6])


def test_load_cache_to_device_buffer_multiple_misses_copy_all_slots() -> None:
    state = _make_state(
        [[9, 7, 3, 5, 11]],
        [[0, 1, 2, 3, -1]],
        [8],
    )

    out = _run_kernel(
        top_k_tokens=torch.tensor([[4, 5, 6, 7]], dtype=torch.int32, device=DEVICE),
        seq_len=9,
        **state,
    )

    assert torch.equal(out.cpu(), torch.tensor([[9, 7, 3, 5]], dtype=torch.int32))
    assert torch.equal(
        state["device_buffer_tokens"].cpu(),
        torch.tensor([[4, 5, 6, 7, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        state["lru_slots"].cpu(), torch.tensor([[0, 1, 2, 3]], dtype=torch.int16)
    )
    for token, loc in zip([4, 5, 6, 7], [9, 7, 3, 5]):
        assert torch.equal(
            state["device_buffer"][loc].cpu(), state["host_cache"][token]
        )


def test_load_cache_to_device_buffer_batched_with_padding() -> None:
    state = _make_state(
        [
            [9, 7, 3, 5, 11],
            [12, 10, 8, 6, 14],
            [15, 4, 2, 1, 13],
        ],
        [
            [1, 4, 2, 5, -1],
            [0, 1, 2, 3, -1],
            [9, 8, 7, 6, -1],
        ],
        [7, 4, 5],
    )
    padded_tokens_before = state["device_buffer_tokens"][2].clone()
    padded_lru_before = state["lru_slots"][2].clone()

    # req 0: long path
    #   cached tokens/locs : 1@9, 4@7, 2@3, 5@5, newest 7@11
    #   query [4, 6, 7]    : hit loc 7, miss into slot0/loc 9, newest loc 11
    #   LRU update         : remaining evictables [2, 3], then miss [0], then hit [1]
    #                      : [0, 1, 2, 3] -> [2, 3, 0, 1]
    #
    # req 1: fast path
    #   seq_len = 3 <= HOT_BUFFER_SIZE, so [2, 1, 0] maps directly to locs [8, 10, 12]
    #
    # req 2: padded block
    #   num_real_reqs = 2 means this row must be ignored entirely.
    out = _run_kernel(
        top_k_tokens=torch.tensor(
            [[4, 6, 7], [2, 1, 0], [9, 8, 7]], dtype=torch.int32, device=DEVICE
        ),
        seq_lens=torch.tensor([8, 3, 8], dtype=torch.int32, device=DEVICE),
        num_real_reqs=2,
        output_fill_value=123456,
        **state,
    )

    assert torch.equal(
        out.cpu(),
        torch.tensor([[7, 9, 11], [8, 10, 12], [-1, -1, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        state["device_buffer_tokens"][:2].cpu(),
        torch.tensor([[6, 4, 2, 5, -1], [0, 1, 2, 3, -1]], dtype=torch.int32),
    )
    assert torch.equal(
        state["lru_slots"][:2].cpu(),
        torch.tensor([[2, 3, 0, 1], [0, 1, 2, 3]], dtype=torch.int16),
    )
    assert torch.equal(
        state["device_buffer_tokens"][2].cpu(), padded_tokens_before.cpu()
    )
    assert torch.equal(state["lru_slots"][2].cpu(), padded_lru_before.cpu())
    assert torch.equal(state["device_buffer"][9].cpu(), state["host_cache"][6])


def test_load_cache_to_device_buffer_dsv4_mla_miss_copy_layout() -> None:
    # Both the host cache and the device buffer use the page-padded C4 layout,
    # matching DeepSeekV4PagedHostPool, the backup/write path, and the swap-in
    # kernel on both CUDA and ROCm. The miss copy must read the host source with
    # paged addressing (get_pointer_paged), not a linear per-item stride.
    num_pages = (HOST_CACHE_SIZE + DSV4_PAGE_SIZE - 1) // DSV4_PAGE_SIZE

    state = _long_case()
    host_cache = torch.zeros(
        (num_pages, DSV4_PAGE_BYTES),
        dtype=torch.uint8,
        device="cpu",
        pin_memory=True,
    )
    for token in range(HOST_CACHE_SIZE):
        _write_dsv4_token(host_cache, token, seed=token + 1)

    device_buffer = torch.full(
        (num_pages, DSV4_PAGE_BYTES),
        0xFF,
        dtype=torch.uint8,
        device=DEVICE,
    )
    out = torch.full((1, 1), -1, dtype=torch.int32, device=DEVICE)

    # Token 6 is a miss in _long_case(), so it should be copied into evict slot 0,
    # whose physical device loc is 9.
    load_cache_to_device_buffer_dsv4_mla(
        top_k_tokens=torch.tensor([[6]], dtype=torch.int32, device=DEVICE),
        device_buffer_tokens=state["device_buffer_tokens"],
        host_cache_locs=state["host_cache_locs"],
        device_buffer_locs=state["device_buffer_locs"],
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=out,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([8], dtype=torch.int32, device=DEVICE),
        lru_slots=state["lru_slots"],
        item_size_bytes=DSV4_ITEM_BYTES,
        num_top_k=1,
        hot_buffer_size=HOT_BUFFER_SIZE,
        page_size=DSV4_PAGE_SIZE,
        block_size=256,
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device=DEVICE),
    )
    torch.cuda.synchronize()

    assert torch.equal(out.cpu(), torch.tensor([[9]], dtype=torch.int32))

    # host_cache_locs[token=6] == 6 in _long_case(); evict slot 0 -> device loc 9.
    assert torch.equal(
        _read_dsv4_token(device_buffer, 9).cpu(),
        _read_dsv4_token(host_cache, 6),
    )


@pytest.mark.skipif(
    not is_hip(), reason="Covers a ROCm wavefront64 LRU writeback regression."
)
def test_load_cache_to_device_buffer_rocm_large_lru_writeback() -> None:
    top_k = 2048
    hot_buffer_size = 4096
    seq_len = 7299
    kv_dim = 4
    item_size_bytes = kv_dim * torch.empty((), dtype=DTYPE).element_size()

    top_k_tokens = torch.cat(
        [
            torch.arange(1000, 2000, dtype=torch.int32),
            torch.arange(5000, 6048, dtype=torch.int32),
        ]
    ).view(1, -1)
    device_buffer_tokens = torch.arange(hot_buffer_size, dtype=torch.int32).view(1, -1)
    device_buffer_locs = torch.arange(hot_buffer_size + 1, dtype=torch.int32).view(
        1, -1
    )
    lru_slots = torch.arange(hot_buffer_size, dtype=torch.int16).view(1, -1)
    host_cache_locs = torch.arange(seq_len, dtype=torch.int64).view(1, -1)

    top_k_tokens = top_k_tokens.to(DEVICE)
    device_buffer_tokens = device_buffer_tokens.to(DEVICE)
    device_buffer_locs = device_buffer_locs.to(DEVICE)
    lru_slots = lru_slots.to(DEVICE)
    host_cache_locs = host_cache_locs.to(DEVICE)

    host_cache = torch.empty((seq_len, 1, kv_dim), dtype=DTYPE, pin_memory=True)
    host_cache.zero_()
    device_buffer = torch.empty(
        (hot_buffer_size + 1, 1, kv_dim), dtype=DTYPE, device=DEVICE
    )
    out = torch.full_like(top_k_tokens, -1)

    load_cache_to_device_buffer_mla(
        top_k_tokens=top_k_tokens,
        device_buffer_tokens=device_buffer_tokens,
        host_cache_locs=host_cache_locs,
        device_buffer_locs=device_buffer_locs,
        host_cache=host_cache,
        device_buffer=device_buffer,
        top_k_device_locs=out,
        req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=DEVICE),
        lru_slots=lru_slots,
        item_size_bytes=item_size_bytes,
        num_top_k=top_k,
        hot_buffer_size=hot_buffer_size,
        page_size=1,
        block_size=1024,
        num_real_reqs=torch.tensor([1], dtype=torch.int32, device=DEVICE),
    )
    torch.cuda.synchronize()

    expected_lru = torch.cat(
        [
            torch.arange(2048, 4096, dtype=torch.int16),
            torch.arange(0, 1000, dtype=torch.int16),
            torch.arange(2000, 2048, dtype=torch.int16),
            torch.arange(1000, 2000, dtype=torch.int16),
        ]
    )
    assert torch.equal(lru_slots.cpu().view(-1), expected_lru)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))


@pytest.mark.parametrize("k", [1, 4, 63, 4992])
def test_kvduo_gpu_merge_plan_matches_reference_with_padding_and_multiple_owners(k):
    from sglang.srt.mem_cache.sparsity.core.kvduo_hot_merge import (
        plan_kvduo_hot_merge,
        prepare_kvduo_hot_merge,
    )

    generator = torch.Generator().manual_seed(821)
    slot_dtype = torch.int32 if 8 * k > 32767 else torch.int16
    tokens = torch.full((2, 2, 8 * k + 64), 999, dtype=torch.int32, device=DEVICE)
    locations = torch.full_like(tokens, 999)
    lru = torch.full((2, 2, 8 * k), -1, dtype=slot_dtype, device=DEVICE)
    addresses = [
        (bank.data_ptr(), bank.shape, bank.stride())
        for bank in (tokens, locations, lru)
    ]
    for case in range(12):
        n = 8 if case < 3 else int(torch.randint(1, 9, (), generator=generator))
        m = int(torch.randint(1, n + 1, (), generator=generator))
        capacity = n * k
        owner = (case % 2, (case // 2) % 2)
        ids = torch.arange(capacity, dtype=torch.int32)
        validity = torch.rand(capacity, generator=generator) < 0.6
        if case == 0:
            validity.fill_(False)
        if case == 1:
            validity.fill_(True)
        ids[~validity] = -1
        order = torch.randperm(capacity, generator=generator).to(slot_dtype)
        locs = torch.randperm(capacity, generator=generator).to(torch.int32) * 8192 + 17
        tokens[owner][:capacity] = ids.to(DEVICE)
        locations[owner][:capacity] = locs.to(DEVICE)
        lru[owner][:capacity] = order.to(DEVICE)
        originals = [bank.cpu().clone() for bank in (tokens, locations, lru)]
        expected = plan_kvduo_hot_merge(ids.tolist(), order.tolist(), k, 1, m)
        plan = prepare_kvduo_hot_merge(
            tokens[owner], locations[owner], lru[owner], capacity, k, 1, m
        )
        assert (
            plan.kept_pages == expected.kept_pages
            and plan.released_pages == expected.released_pages
        )
        assert plan.tokens[: m * k].cpu().tolist() == list(expected.tokens)
        assert plan.lru_slots[: m * k].cpu().tolist() == list(expected.lru_slots)
        assert plan.locations[: m * k].cpu().tolist() == [
            int(locs[p * k + offset])
            for p in expected.kept_pages
            for offset in range(k)
        ]
        assert plan.source_locations.cpu().tolist() == [
            int(locs[s]) for s in expected.source_slots
        ]
        assert plan.destination_locations.cpu().tolist() == [
            int(locs[s]) for s in expected.destination_slots
        ]
        assert all(
            torch.all(bank[m * k :] == -1)
            for bank in (plan.tokens, plan.locations, plan.lru_slots)
        )
        assert all(
            torch.equal(bank.cpu(), saved)
            for bank, saved in zip((tokens, locations, lru), originals)
        )
        assert addresses == [
            (bank.data_ptr(), bank.shape, bank.stride())
            for bank in (tokens, locations, lru)
        ]


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate_lru",
        "negative_lru",
        "out_of_range_lru",
        "duplicate_location",
        "negative_location",
    ],
)
def test_kvduo_gpu_merge_rejects_corrupt_metadata_without_modifying_inputs(corruption):
    from sglang.srt.mem_cache.sparsity.core.kvduo_hot_merge import (
        prepare_kvduo_hot_merge,
    )

    tokens = torch.arange(34, dtype=torch.int32, device=DEVICE)
    locs = tokens.clone() + 100
    lru = torch.arange(32, dtype=torch.int16, device=DEVICE)
    if corruption == "duplicate_lru":
        lru[1] = 0
    elif corruption == "negative_lru":
        lru[1] = -1
    elif corruption == "out_of_range_lru":
        lru[1] = 16
    elif corruption == "duplicate_location":
        locs[1] = locs[0]
    else:
        locs[1] = -1
    before = [bank.clone() for bank in (tokens, locs, lru)]
    with pytest.raises((ValueError, RuntimeError)):
        prepare_kvduo_hot_merge(tokens, locs, lru, 16, 4, 1, 2)
    torch.cuda.synchronize()
    assert all(
        torch.equal(bank, saved) for bank, saved in zip((tokens, locs, lru), before)
    )


def test_kvduo_merge_copies_flat_c4_values_and_scales_before_freeing_pages():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.enable_mixed_residency = True
    coordinator.is_dsv4_hisparse = True
    coordinator.device = DEVICE
    coordinator.decode_producer_stream = torch.cuda.current_stream()
    coordinator._has_pending_backup = False
    coordinator.page_size = DSV4_PAGE_SIZE
    coordinator.hot_page_size = 2 * DSV4_PAGE_SIZE
    coordinator.top_k = 2
    coordinator.min_hot_pages = 1
    coordinator._kvduo_hot_pressure_protected = set()
    k = coordinator.hot_page_size
    stride = 8 * DSV4_PAGE_SIZE
    pages = [64, 128, 192, 256]
    locs = [
        start + layer * stride + offset
        for start in pages
        for layer in range(2)
        for offset in range(64)
    ]
    backing = (
        torch.arange(16 * DSV4_PAGE_BYTES, dtype=torch.int32, device=DEVICE)
        .remainder(251)
        .to(torch.uint8)
        .view(16, DSV4_PAGE_BYTES)
    )
    coordinator.mem_pool_device = SimpleNamespace(get_flat_key_buffer=lambda: backing)
    coordinator._kvduo_req_layer_pages = {(0, 0): list(pages)}
    coordinator._kvduo_hot_page_owners = {start: (0, 0) for start in pages}
    coordinator.kvduo_req_hot_capacity = torch.tensor([[4 * k]], dtype=torch.int64)
    coordinator.kvduo_req_hot_capacity_gpu = torch.tensor(
        [[4 * k]], dtype=torch.int32, device=DEVICE
    )
    coordinator.mem_pool_device.layer_num = 1
    coordinator._active_kvduo_reqs = {0: object()}
    coordinator._init_kvduo_prefill_rollback_state(1)
    coordinator.kvduo_hot_growth_events = torch.ones(
        (1, 1), dtype=torch.int32, device=DEVICE
    )
    coordinator._kvduo_hot_generation[0, 0] = 0
    coordinator.kvduo_capacity_steps.fill_(65)
    coordinator.kvduo_capacity_ema[..., :2] = 100
    coordinator.req_device_buffer_tokens = torch.full(
        (1, 1, 8 * k), -1, dtype=torch.int32, device=DEVICE
    )
    coordinator.req_device_buffer_tokens[0, 0, : 4 * k] = torch.arange(
        10, 10 + 4 * k, dtype=torch.int32, device=DEVICE
    )
    coordinator.req_device_buffer_token_locs = torch.full_like(
        coordinator.req_device_buffer_tokens, -1
    )
    coordinator.req_device_buffer_token_locs[0, 0, : 4 * k] = torch.tensor(
        locs, dtype=torch.int32, device=DEVICE
    )
    survivors = [1] + list(range(k, 2 * k)) + list(range(3 * k, 4 * k - 1))
    lru = [slot for slot in range(4 * k) if slot not in set(survivors)] + survivors
    coordinator.lru_slots = torch.full(
        (1, 1, 8 * k), -1, dtype=torch.int16, device=DEVICE
    )
    coordinator.lru_slots[0, 0, : 4 * k] = torch.tensor(
        lru, dtype=torch.int16, device=DEVICE
    )
    expected = backing.cpu().clone()
    src, dst = locs[1], locs[-1]  # one survivor moves across storage layers
    for offset, width in ((0, DSV4_VALUE_BYTES), (DSV4_SCALE_OFFSET, DSV4_SCALE_BYTES)):
        src_offset = offset + (src % 64) * width
        dst_offset = offset + (dst % 64) * width
        expected[dst // 64, dst_offset : dst_offset + width] = expected[
            src // 64, src_offset : src_offset + width
        ].clone()

    def free_after_copy(indices):
        assert torch.equal(backing.cpu(), expected)
        assert coordinator._kvduo_req_layer_pages[(0, 0)] == [128, 256]
        assert indices.cpu().tolist() == list(range(64, 128)) + list(range(192, 256))

    free = MagicMock(side_effect=free_after_copy)
    coordinator.token_to_kv_pool_allocator = SimpleNamespace(free_hisparse_indices=free)
    coordinator.kvduo_capacity_ema[..., :2] = 100
    assert coordinator._merge_kvduo_hot_pages(0, 0) == 2 * DSV4_PAGE_SIZE
    assert not coordinator.kvduo_hot_growth_events.any()
    free.assert_called_once()
    assert coordinator.lru_slots[0, 0, : 2 * k].cpu().tolist() == [2 * k - 1] + list(
        range(2 * k - 1)
    )
    assert coordinator.kvduo_prefill_steps.item() == 0


def test_kvduo_resolver_graph_replays_after_actual_merge_without_recapture():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

    kwargs = _kvduo_observation_case()
    load_cache_to_device_buffer_mla(**kwargs)
    kwargs["observe_cache"].zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        load_cache_to_device_buffer_mla(**kwargs)
    pointers = [
        kwargs[key].data_ptr()
        for key in (
            "device_buffer_tokens",
            "device_buffer_locs",
            "lru_slots",
            "req_hot_buffer_sizes",
        )
    ]
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.enable_mixed_residency = True
    coordinator.is_dsv4_hisparse = False
    coordinator.device = DEVICE
    coordinator.decode_producer_stream = torch.cuda.current_stream()
    coordinator._has_pending_backup = False
    coordinator.page_size = coordinator.hot_page_size = 1
    coordinator.top_k = coordinator.min_hot_pages = 2
    coordinator._kvduo_hot_pressure_protected = set()
    coordinator.mem_pool_device = SimpleNamespace(
        kv_buffer=[kwargs["device_buffer"]],
        get_flat_key_buffer=lambda: kwargs["device_buffer"],
    )
    pages = kwargs["device_buffer_locs"][0, :4].cpu().tolist()
    coordinator._kvduo_req_layer_pages = {(0, 0): pages}
    coordinator._kvduo_hot_page_owners = {start: (0, 0) for start in pages}
    coordinator.kvduo_req_hot_capacity = torch.tensor([[4]], dtype=torch.int64)
    coordinator.kvduo_req_hot_capacity_gpu = kwargs["req_hot_buffer_sizes"].unsqueeze(0)
    coordinator.mem_pool_device.layer_num = 1
    coordinator._active_kvduo_reqs = {0: object()}
    coordinator._init_kvduo_prefill_rollback_state(1)
    coordinator.kvduo_hot_growth_events = torch.ones(
        (1, 1), dtype=torch.int32, device=DEVICE
    )
    coordinator.kvduo_hot_base_capacity = kwargs["hot_base_capacities"].unsqueeze(0)
    coordinator.kvduo_prefill_ema = kwargs["prefill_ema"].unsqueeze(0)
    coordinator.kvduo_prefill_steps = kwargs["prefill_steps"].unsqueeze(0)
    coordinator._kvduo_hot_generation[0, 0] = 0
    coordinator.req_device_buffer_tokens = kwargs["device_buffer_tokens"].unsqueeze(0)
    coordinator.req_device_buffer_token_locs = kwargs["device_buffer_locs"].unsqueeze(0)
    coordinator.lru_slots = kwargs["lru_slots"].unsqueeze(0)
    coordinator.kvduo_capacity_ema = kwargs["capacity_ema"].unsqueeze(0)
    coordinator.kvduo_capacity_steps = kwargs["capacity_steps"].unsqueeze(0)
    coordinator.kvduo_capacity_steps.fill_(65)
    coordinator.token_to_kv_pool_allocator = SimpleNamespace(
        free_hisparse_indices=MagicMock()
    )
    coordinator.kvduo_capacity_ema[..., :2] = 100
    assert coordinator._merge_kvduo_hot_pages(0, 0) == 2
    assert not coordinator.kvduo_hot_growth_events.any()
    assert pointers == [
        kwargs[key].data_ptr()
        for key in (
            "device_buffer_tokens",
            "device_buffer_locs",
            "lru_slots",
            "req_hot_buffer_sizes",
        )
    ]
    kwargs["observe_cache"].fill_(1)
    kwargs["top_k_tokens"][0] = torch.tensor([5, 4], dtype=torch.int32, device=DEVICE)
    graph.replay()
    torch.cuda.synchronize()
    assert kwargs["req_hot_buffer_sizes"].item() == 2
    assert coordinator.kvduo_prefill_steps.item() == 1
    assert coordinator.kvduo_prefill_ema.cpu().tolist() == [[[0, 2]]]
    for index, token in enumerate([5, 4]):
        location = kwargs["top_k_device_locs"][0, index].item()
        assert torch.equal(
            kwargs["device_buffer"][location].cpu(), kwargs["host_cache"][token]
        )


@pytest.mark.parametrize(
    "kind",
    [
        "mixed",
        "full",
        "tail",
        "miss",
        "partial",
        "invalid",
        "padding",
        "capture",
        "failed",
    ],
)
def test_kvduo_segment_and_extra_ema_use_actual_pre_update_hot_reuse_and_misses(kind):
    kwargs = _kvduo_observation_case()
    expected = [1, 0]  # slot2 extra, slot0 base, independent of LRU rank.
    observed = True
    if kind in {"full", "tail"}:
        if kind == "full":
            kwargs["req_hot_buffer_sizes"].zero_()
            kwargs["full_to_device_loc"][[1, 2]] = torch.tensor([9, 3], device=DEVICE)
        else:
            kwargs["top_k_tokens"][0] = torch.tensor([7, -1], device=DEVICE)
            kwargs["full_to_device_loc"][7] = 11
        expected = [0, 0]
    elif kind == "miss":
        kwargs["top_k_tokens"][0] = torch.tensor([3, 6], device=DEVICE)
        expected = [0, 2]
    elif kind == "partial":
        kwargs["top_k_tokens"][0] = torch.tensor([-1, 2], device=DEVICE)
        expected = [1, 0]
    elif kind in {"invalid", "padding", "capture", "failed"}:
        observed = False
        expected = [0, 0]
        if kind == "invalid":
            kwargs["top_k_tokens"].fill_(-1)
        if kind == "padding":
            kwargs["num_real_reqs"].zero_()
        if kind == "capture":
            kwargs["observe_cache"].zero_()
        if kind == "failed":
            kwargs["full_host_version"][2] = 0
    capacity = kwargs["req_hot_buffer_sizes"].item()
    selected = set(kwargs["top_k_tokens"][0].cpu().tolist())
    old_tokens = kwargs["device_buffer_tokens"][0].cpu().tolist()
    old_order = kwargs["lru_slots"][0].cpu().tolist()[:capacity]
    full = kwargs["full_to_device_loc"].cpu().tolist()
    segments = [0] * 8
    if observed:
        for rank, slot in enumerate(old_order):
            token = old_tokens[slot]
            if token >= 0 and token in selected and full[token] <= 0:
                segments[(capacity - 1 - rank) // kwargs["page_size"]] += 1
    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()
    assert kwargs["prefill_ema"].cpu().tolist() == [expected]
    assert kwargs["prefill_steps"].item() == int(observed)

    assert kwargs["capacity_ema"].cpu().tolist() == [
        segments + [expected[1] if capacity and observed else 0]
    ]
    assert kwargs["capacity_steps"].item() == int(observed and capacity > 0)


def test_kvduo_ema_decays_on_valid_zero_counts_and_marks_reuse_after_h2d():
    kwargs = _kvduo_observation_case(selected=(3, 6))
    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()
    assert kwargs["prefill_ema"].cpu().tolist() == [[0, 2]]
    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()
    assert kwargs["prefill_ema"].cpu().tolist() == [[1 / 64, 2 * 63 / 64]]
    assert kwargs["prefill_steps"].item() == 2


@pytest.mark.parametrize("kind", ["valid", "invalid", "failed", "padding", "capture"])
def test_mixed_full_only_resolver_keeps_hot_metadata_and_observation_rules(kind):
    kwargs = _kvduo_observation_case(selected=(1, 2))
    kwargs["full_to_device_loc"][[1, 2]] = torch.tensor([12, 13], device=DEVICE)
    kwargs["capacity_ema"].fill_(64)
    kwargs["capacity_steps"].fill_(65)
    kwargs["prefill_ema"][0] = torch.tensor([8, 16], device=DEVICE)
    kwargs["prefill_steps"].fill_(67)
    kwargs["hot_growth_events"].fill_(1)
    kwargs["touch_clock"].fill_(7)
    expected_accesses = 2
    if kind == "invalid":
        kwargs["top_k_tokens"].fill_(-1)
        expected_accesses = 0
    elif kind == "failed":
        kwargs["req_to_logical_token"][0, 2] = -1
        expected_accesses = 1
    elif kind == "padding":
        kwargs["num_real_reqs"].zero_()
        expected_accesses = 0
    elif kind == "capture":
        kwargs["observe_cache"].zero_()
    hot_banks = [
        kwargs[key]
        for key in ("device_buffer_tokens", "device_buffer_locs", "lru_slots")
    ]
    originals = [bank.clone() for bank in hot_banks]
    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()
    assert all(torch.equal(bank, saved) for bank, saved in zip(hot_banks, originals))
    assert kwargs["resolver_stats"].tolist() == [[0, expected_accesses]]
    observed = kind == "valid"
    assert kwargs["capacity_steps"].item() == 65 + observed
    assert kwargs["prefill_steps"].item() == 67 + observed
    expected_capacity = [63] * 9 if observed else [64] * 9
    assert kwargs["capacity_ema"].tolist() == [expected_capacity]
    assert kwargs["prefill_ema"].tolist() == [
        [8 * (63 / 64 if observed else 1), 16 * (63 / 64 if observed else 1)]
    ]
    assert kwargs["hot_growth_events"].item() == int(kind in {"padding", "capture"})
    if kind in {"valid", "capture"}:
        assert kwargs["top_k_device_locs"].tolist() == [[12, 13]]
        assert kwargs["full_last_touch"][[1, 2]].tolist() == [7, 7]
    elif kind == "failed":
        assert kwargs["top_k_device_locs"].tolist() == [[12, -1]]
        assert kwargs["swap_status"].item() == 1
    else:
        assert kwargs["top_k_device_locs"].tolist() == [[-1, -1]]


def test_mixed_full_fast_path_graph_switches_to_hot_and_h2d_without_recapture():
    kwargs = _kvduo_observation_case(selected=(1, 2))
    kwargs["full_to_device_loc"][[1, 2]] = torch.tensor([12, 13], device=DEVICE)
    full_payload = kwargs["device_buffer"][12].clone()
    banks = [value for value in kwargs.values() if isinstance(value, torch.Tensor)]
    addresses = [(bank.data_ptr(), bank.shape, bank.stride()) for bank in banks]
    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()
    kwargs["observe_cache"].zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        load_cache_to_device_buffer_mla(**kwargs)
    kwargs["observe_cache"].fill_(1)
    kwargs["capacity_ema"].zero_()
    kwargs["capacity_steps"].zero_()
    kwargs["prefill_ema"].zero_()
    kwargs["prefill_steps"].zero_()
    kwargs["full_to_device_loc"][2] = 0
    graph.replay()
    torch.cuda.synchronize()
    assert kwargs["top_k_device_locs"].tolist() == [[12, 1]]
    assert kwargs["prefill_ema"].tolist() == [[1, 0]]
    kwargs["top_k_tokens"][0, 1] = 6
    graph.replay()
    torch.cuda.synchronize()
    h2d_location = kwargs["top_k_device_locs"][0, 1].item()
    assert torch.equal(
        kwargs["device_buffer"][h2d_location].cpu(), kwargs["host_cache"][6]
    )
    assert kwargs["hot_growth_events"].item() == 0  # Already at maximum capacity.
    old_lru = kwargs["lru_slots"].clone()
    old_capacity_ema = kwargs["capacity_ema"].clone()
    old_prefill_ema = kwargs["prefill_ema"].clone()
    kwargs["full_to_device_loc"][6] = 14
    graph.replay()
    torch.cuda.synchronize()
    assert kwargs["top_k_device_locs"].tolist() == [[12, 14]]
    assert torch.equal(kwargs["lru_slots"], old_lru)
    assert torch.equal(kwargs["capacity_ema"], old_capacity_ema * (63 / 64))
    assert torch.equal(kwargs["prefill_ema"], old_prefill_ema * (63 / 64))
    assert kwargs["capacity_steps"].item() == kwargs["prefill_steps"].item() == 3
    assert torch.equal(kwargs["device_buffer"][12], full_payload)
    assert addresses == [(bank.data_ptr(), bank.shape, bank.stride()) for bank in banks]


@pytest.mark.parametrize("needed", [1, 129, 521])
@pytest.mark.parametrize("costs", [[0, 0, 0, 0], [4, 7, 2, 9]])
@pytest.mark.parametrize("release_all", [False, True])
def test_full_reclaim_gpu_selection_scans_multiple_tiles_and_charges_interleaved_owners(
    needed, costs, release_all
):
    from sglang.kernels.ops.kvcache.kvduo_full_reclaim import select_kvduo_full_pages

    generator = torch.Generator().manual_seed(782)
    owners = [0] * 130 + [1] * 131 + [2] * 128 + [3] * 132
    scores = torch.randint(0, 20, (len(owners),), generator=generator)
    valid = (torch.rand(len(owners), generator=generator) > 0.2).int()
    scores[-2:] = torch.iinfo(torch.int64).max
    valid[-2:] = torch.tensor([0, 1])
    expected = select_kvduo_full_pages(
        scores, valid, owners, costs, needed, release_all=release_all
    )
    assert (
        select_kvduo_full_pages(
            scores.to(DEVICE),
            valid.to(DEVICE),
            owners,
            costs,
            needed,
            release_all=release_all,
        )
        == expected
    )


@pytest.mark.parametrize("page_size", [1, 64, 129])
@pytest.mark.parametrize("release_all", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_full_reclaim_gpu_summary_and_selection_match_cpu_with_multiple_requests(
    page_size, release_all, index_dtype
):
    from sglang.kernels.ops.kvcache.kvduo_full_reclaim import (
        select_kvduo_full_pages,
        summarize_kvduo_full_pages,
    )

    pages = torch.arange(12 * page_size, dtype=index_dtype).view(12, page_size)
    mapping = pages.flatten().long() + 10
    versions = torch.ones_like(mapping)
    host_versions = versions.clone()
    mapping[page_size] = 0
    host_versions[3 * page_size] = 0
    touches = torch.arange(page_size).repeat(12)
    touches[5 * page_size : 6 * page_size] = torch.iinfo(torch.int64).max
    protected = torch.tensor([9 * page_size], dtype=index_dtype)
    cpu = summarize_kvduo_full_pages(
        pages, mapping, versions, host_versions, touches, protected
    )
    gpu = summarize_kvduo_full_pages(
        *(
            bank.to(DEVICE)
            for bank in (pages, mapping, versions, host_versions, touches, protected)
        )
    )
    assert all(
        torch.equal(actual.cpu(), expected) for actual, expected in zip(gpu, cpu)
    )
    owners, costs = [0] * 4 + [1] * 4 + [2] * 4, [1, 2, 1]
    expected = select_kvduo_full_pages(
        cpu[1], cpu[2], owners, costs, 2, release_all=release_all
    )
    assert (
        select_kvduo_full_pages(
            gpu[1], gpu[2], owners, costs, 2, release_all=release_all
        )
        == expected
    )
    pages[0, 0] = -1
    pages[-1, -1] = mapping.numel()
    _, _, valid = summarize_kvduo_full_pages(
        *(
            bank.to(DEVICE)
            for bank in (pages, mapping, versions, host_versions, touches)
        )
    )
    assert valid[[0, -1]].tolist() == [0, 0]


def _generic_kvduo_stack(monkeypatch):
    if is_hip():
        pytest.skip("Actual DSA pool fixture uses CUDA's 64-token page layout")
    from types import SimpleNamespace

    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
    from sglang.srt.mem_cache.pool_host.common import (
        ALLOC_MEMORY_FUNCS,
        alloc_with_pin_memory,
    )

    pool = HiSparseDSATokenToKVPool(
        size=1024,
        page_size=64,
        dtype=torch.bfloat16,
        kv_lora_rank=12,
        qk_rope_head_dim=4,
        kv_cache_dim=16,
        layer_num=3,
        device=DEVICE,
        index_head_dim=128,
        enable_memory_saver=False,
        enable_kvduo=True,
    )
    allocator = HiSparseTokenToKVPoolAllocator(
        1024, 64, pool.dtype, DEVICE, pool, False
    )
    monkeypatch.setitem(ALLOC_MEMORY_FUNCS, DEVICE, alloc_with_pin_memory)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda **kw: 1)
    c = HiSparseCoordinator(
        SimpleNamespace(
            req_to_token=torch.zeros((2, 128), dtype=torch.int32, device=DEVICE),
            max_context_len=128,
        ),
        allocator,
        top_k=4,
        device_buffer_size=8,
        device=DEVICE,
        tp_group=None,
        tail_protected_pages=2,
    )
    c._active_kvduo_reqs = {0: SimpleNamespace(hisparse_staging=False)}
    c._mixed_slots[0] = True
    c.num_real_reqs.fill_(1)
    c._ensure_kvduo_hot_capacity_targets_batch({(1, 0): c.hot_page_size})
    c.kvduo_capacity_steps[1, 0] = 65
    c.kvduo_capacity_ema[1, 0] = 0
    c._merge_kvduo_hot_pages(1, 0)  # Establish a real merge base, even with no release.
    c._ensure_kvduo_hot_capacity_targets_batch({(1, 0): 2 * c.hot_page_size})
    return c, pool, allocator


def test_generic_kvduo_resolver_reads_full_hot_and_host_across_layers_after_graph_merge(
    monkeypatch,
):
    c, pool, allocator = _generic_kvduo_stack(monkeypatch)
    k = c.hot_page_size
    full = allocator.hisparse_attn_allocator.alloc(64)
    allocator.full_to_hisparse_device_index_mapping[1] = full[0]
    c.req_to_full_lookup[0, :4] = torch.arange(1, 5, dtype=torch.int64, device=DEVICE)
    c.req_to_host_pool[0, :4] = torch.arange(4, dtype=torch.int64, device=DEVICE)
    c.full_host_version[1:5] = 0
    base_slot, extra_slot = 2 * 64, k
    c.req_device_buffer_tokens[1, 0, base_slot] = 1
    c.req_device_buffer_tokens[1, 0, extra_slot] = 2
    base_loc = c.req_device_buffer_token_locs[1, 0, base_slot].long()
    extra_loc = c.req_device_buffer_token_locs[1, 0, extra_slot].long()
    backing = pool.get_flat_key_buffer()
    pool.kv_buffer[1][full[0]] = 10
    backing[base_loc] = 20  # Logical layer 1's KV stored in storage layer 2.
    backing[extra_loc] = 30  # Logical layer 1's KV stored in storage layer 0.
    c.mem_pool_host.kv_buffer[1, 3] = 40
    args = (
        torch.tensor([0], dtype=torch.int64, device=DEVICE),
        torch.tensor([4], dtype=torch.int32, device=DEVICE),
        torch.tensor([[0, 1, 2, 3]], dtype=torch.int32, device=DEVICE),
        1,
    )
    output = c.swap_in_selected_pages(*args)
    torch.cuda.synchronize()
    assert output[0, 0].item() == pool.flat_layer_slot_stride + full[0].item()
    assert output[0, 1].item() == base_loc.item()
    assert output[0, 2].item() == extra_loc.item()
    assert backing[output[0].long(), 0, 0].tolist() == [10, 20, 30, 40]
    assert c.kvduo_prefill_ema[1, 0].tolist() == [1, 1]
    assert c.kvduo_prefill_steps[1, 0].item() == 1
    pointers = [
        bank.data_ptr()
        for bank in (
            backing,
            c.req_device_buffer_tokens,
            c.req_device_buffer_token_locs,
            c.lru_slots,
            c.kvduo_req_hot_capacity_gpu,
        )
    ]
    c.kvduo_observe.zero_()
    c.swap_in_selected_pages(*args)  # Warm up assertion and resolver kernels.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        c.swap_in_selected_pages(*args)
    c.kvduo_observe.fill_(1)
    before = allocator.hisparse_attn_allocator.available_size()
    c.kvduo_capacity_steps[1, 0] = 65
    c.kvduo_capacity_ema[1, 0] = 0
    assert c._merge_kvduo_hot_pages(1, 0) == 64
    assert allocator.hisparse_attn_allocator.available_size() == before + 64
    graph.replay()
    torch.cuda.synchronize()
    assert backing[output[0].long(), 0, 0].tolist() == [10, 20, 30, 40]
    # MRU retention includes the freshly inserted item; compaction is not an access.
    assert c.kvduo_prefill_ema[1, 0].tolist() == [0, 0]
    assert c.kvduo_prefill_steps[1, 0].item() == 1
    assert pointers == [
        bank.data_ptr()
        for bank in (
            backing,
            c.req_device_buffer_tokens,
            c.req_device_buffer_token_locs,
            c.lru_slots,
            c.kvduo_req_hot_capacity_gpu,
        )
    ]
    c.destroy()


def test_generic_kvduo_fast_rollback_keeps_base_payload_and_filters_lru(monkeypatch):
    c, pool, allocator = _generic_kvduo_stack(monkeypatch)
    k = c.hot_page_size
    c.req_device_buffer_tokens[1, 0, : 2 * k] = torch.arange(2 * k, device=DEVICE)
    locs = c.req_device_buffer_token_locs[1, 0, : 2 * k].long()
    pool.get_flat_key_buffer()[locs] = torch.arange(
        2 * k, device=DEVICE, dtype=torch.bfloat16
    )[:, None, None]
    c.lru_slots[1, 0, : 2 * k] = torch.arange(2 * k, device=DEVICE).flip(0)
    expected = pool.get_flat_key_buffer().clone()
    base = c._kvduo_hot_base_pages[1, 0]
    before = allocator.hisparse_attn_allocator.available_size()
    c._rollback_kvduo_hot_batch(((1, 0),))
    assert c._kvduo_req_layer_pages[1, 0] == list(base)
    assert allocator.hisparse_attn_allocator.available_size() == before + 64
    assert torch.equal(pool.get_flat_key_buffer(), expected)
    assert c.lru_slots[1, 0, :k].tolist() == list(reversed(range(k)))
    assert torch.all(c.req_device_buffer_tokens[1, 0, k:] == -1)
    assert torch.all(c.req_device_buffer_token_locs[1, 0, k:] == -1)
    c.destroy()


def test_generic_resolver_preserves_lru_slots_above_int16_range():
    kwargs = _kvduo_observation_case(selected=(1, 2))
    capacity = 33000
    tokens = torch.full((1, capacity + 1), -1, dtype=torch.int32, device=DEVICE)
    tokens[0, capacity - 2 :] = torch.tensor([1, 2, -1], device=DEVICE)
    kwargs.update(
        device_buffer_tokens=tokens,
        device_buffer_locs=torch.full_like(tokens, 5),
        capacity_ema=torch.zeros(
            (tokens.size(0), 9), dtype=torch.float32, device=DEVICE
        ),
        capacity_steps=torch.zeros(tokens.size(0), dtype=torch.int64, device=DEVICE),
        lru_slots=torch.arange(capacity, dtype=torch.int32, device=DEVICE).unsqueeze(0),
        hot_buffer_size=capacity,
        req_hot_buffer_sizes=torch.tensor([capacity], dtype=torch.int32, device=DEVICE),
        hot_base_capacities=torch.tensor(
            [capacity - 1], dtype=torch.int32, device=DEVICE
        ),
        resolver_scratch=allocate_hisparse_resolver_scratch(
            1, 2, capacity, 256, torch.device(DEVICE)
        ),
    )
    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()
    assert kwargs["lru_slots"][0, -2:].tolist() == [capacity - 2, capacity - 1]
    assert kwargs["prefill_ema"].tolist() == [[1, 0]]


@pytest.mark.parametrize("capacity", [8, 512, 39936])
@pytest.mark.parametrize("page_size", [2, 64])
@pytest.mark.parametrize("reset_stats", [False, True])
def test_kvduo_batched_no_copy_rollback_preserves_graph_addresses_and_stable_lru(
    capacity, page_size, reset_stats
):
    from sglang.kernels.ops.kvcache.hisparse import rollback_kvduo_hot_metadata

    # Match the coordinator's padded banks and unpadded LRU. Selected owners
    # cross both request and layer boundaries, exposing incorrect row strides.
    padded_capacity = capacity + page_size
    tokens = torch.arange(
        2 * 3 * padded_capacity, dtype=torch.int32, device=DEVICE
    ).view(2, 3, padded_capacity)
    locations = (tokens + 100).clone()
    slot_dtype = torch.int32 if capacity > 32767 else torch.int16
    order = torch.arange(capacity, dtype=slot_dtype, device=DEVICE).flip(0)
    lru = torch.stack([order.roll(i) for i in range(6)]).view(2, 3, capacity)
    capacities = torch.full((2, 3), capacity, dtype=torch.int32, device=DEVICE)
    statistics = (
        (
            torch.arange(1, 55, dtype=torch.float32, device=DEVICE).view(2, 3, 9),
            torch.full((2, 3), 73, dtype=torch.int64, device=DEVICE),
            torch.full((2, 3), 2, dtype=torch.int32, device=DEVICE),
            torch.arange(1, 13, dtype=torch.float32, device=DEVICE).view(2, 3, 2),
            torch.full((2, 3), 65, dtype=torch.int64, device=DEVICE),
        )
        if reset_stats
        else ()
    )
    banks = (tokens, locations, lru, capacities, *statistics)
    before = [bank.clone() for bank in banks]
    addresses = [(bank.data_ptr(), bank.shape, bank.stride()) for bank in banks]
    selected = {(0, 1): capacity // 2, (1, 0): capacity // 4, (1, 2): capacity * 3 // 4}
    rollback_kvduo_hot_metadata(
        torch.tensor(list(selected), dtype=torch.int64, device=DEVICE),
        torch.tensor(list(selected.values()), dtype=torch.int32, device=DEVICE),
        tokens,
        locations,
        lru,
        capacities,
        *statistics,
    )
    torch.cuda.synchronize()
    for layer in range(2):
        for req in range(3):
            owner = (layer, req)
            if owner not in selected:
                for bank, original in zip(banks, before):
                    assert torch.equal(bank[owner], original[owner])
                continue
            target = selected[owner]
            old_order = before[2][owner]
            assert torch.equal(lru[owner][:target], old_order[old_order < target])
            assert torch.all(lru[owner][target:] == -1)
            for bank, original in zip((tokens, locations), (before[0], before[1])):
                assert torch.equal(bank[owner][:target], original[owner][:target])
            assert torch.all(tokens[owner][target:] == -1)
            assert torch.all(locations[owner][target:] == -1)
            assert capacities[owner].item() == target
            assert all(torch.all(bank[owner] == 0) for bank in statistics)
    assert addresses == [(bank.data_ptr(), bank.shape, bank.stride()) for bank in banks]


@pytest.mark.parametrize("active_capacity", [0, 78 * 64, 8 * 78 * 64])
@pytest.mark.parametrize("global_scratch", [True, False])
@pytest.mark.parametrize("full_only", [True, False])
def test_large_resolver_full_one_page_and_max_capacity_replay_with_persistent_scratch(
    active_capacity, global_scratch, full_only
):
    # Exact configuration from the resource regression: 210712 bytes at
    # compile-time max capacity, even when only one HOT page is currently live.
    topk, capacity, base, requests = 2048, 39936, 4992, 2
    scratch = allocate_hisparse_resolver_scratch(
        requests, topk, capacity, 960, torch.device(DEVICE)
    )
    if not global_scratch and scratch is not None:
        pytest.skip("Device cannot run this fixture's shared-memory variant")
    words = hisparse_resolver_scratch_words(topk, capacity)
    if global_scratch:
        # Padding exposes using the wrong row stride between simultaneous CTAs.
        scratch = torch.full((requests, words + 4), 7, dtype=torch.int32, device=DEVICE)
    else:
        scratch = None
    host = torch.empty((requests * topk, 1, KV_DIM), dtype=DTYPE, pin_memory=True)
    host.copy_(torch.arange(requests * topk, dtype=DTYPE).view(-1, 1, 1))
    tokens = torch.full((requests, capacity + 64), -1, dtype=torch.int32, device=DEVICE)
    locations = torch.full_like(tokens, -1)
    for req in range(requests):
        locations[req, :capacity] = (
            64 + req * capacity + torch.arange(capacity, device=DEVICE)
        )
    logical = torch.arange(
        1, requests * topk + 1, dtype=torch.int64, device=DEVICE
    ).view(requests, topk)
    full_offset = requests * capacity + 64
    device = torch.zeros(
        (full_offset + requests * topk + 1, 1, KV_DIM), dtype=DTYPE, device=DEVICE
    )
    full_mapping = torch.zeros(requests * topk + 1, dtype=torch.int64, device=DEVICE)
    expected = host[:, 0, 0].reshape(requests, topk).to(DEVICE)
    if active_capacity == 0 or full_only:
        full_mapping[1:] = full_offset + logical.flatten()
        device[full_mapping[1:]] = host.to(DEVICE)
    else:
        for req in range(requests):
            tokens[req, active_capacity - 1] = 0
            device[locations[req, active_capacity - 1]] = 9000 + req
            full_mapping[logical[req, 1]] = full_offset + req + 1
            device[full_offset + req + 1] = 8000 + req
            expected[req, :2] = torch.tensor([9000 + req, 8000 + req], device=DEVICE)
    versions = torch.ones_like(full_mapping)
    selections = torch.arange(topk, dtype=torch.int32, device=DEVICE).repeat(3, 1)
    kwargs = dict(
        top_k_tokens=selections,
        device_buffer_tokens=tokens,
        host_cache_locs=torch.arange(
            requests * topk, dtype=torch.int64, device=DEVICE
        ).view(requests, topk),
        device_buffer_locs=locations,
        host_cache=host,
        device_buffer=device,
        top_k_device_locs=torch.full_like(selections, -1),
        req_pool_indices=torch.tensor([1, 0, 0], dtype=torch.int64, device=DEVICE),
        seq_lens=torch.full((3,), topk, dtype=torch.int32, device=DEVICE),
        lru_slots=torch.arange(capacity, dtype=torch.int32, device=DEVICE).repeat(
            requests, 1
        ),
        item_size_bytes=ITEM_SIZE_BYTES,
        num_top_k=topk,
        hot_buffer_size=capacity,
        page_size=base,
        block_size=960,
        num_real_reqs=torch.tensor([2], dtype=torch.int32, device=DEVICE),
        req_hot_buffer_sizes=torch.full(
            (requests,), active_capacity, dtype=torch.int32, device=DEVICE
        ),
        capacity_ema=torch.zeros(
            (tokens.size(0), 9), dtype=torch.float32, device=DEVICE
        ),
        capacity_steps=torch.zeros(tokens.size(0), dtype=torch.int64, device=DEVICE),
        req_to_logical_token=logical,
        full_to_device_loc=full_mapping,
        full_last_touch=torch.zeros_like(full_mapping),
        full_data_version=versions,
        full_host_version=versions.clone(),
        swap_status=torch.zeros(requests, dtype=torch.int32, device=DEVICE),
        resolver_stats=torch.zeros((requests, 2), dtype=torch.int32, device=DEVICE),
        hot_growth_events=torch.zeros(requests, dtype=torch.int32, device=DEVICE),
        touch_clock=torch.ones(1, dtype=torch.int64, device=DEVICE),
        observe_cache=torch.ones(1, dtype=torch.int32, device=DEVICE),
        hot_base_capacities=torch.full(
            (requests,), base, dtype=torch.int32, device=DEVICE
        ),
        prefill_ema=torch.zeros((requests, 2), dtype=torch.float32, device=DEVICE),
        prefill_steps=torch.zeros(requests, dtype=torch.int64, device=DEVICE),
        resolver_scratch=scratch,
    )
    banks = [value for value in kwargs.values() if isinstance(value, torch.Tensor)]
    addresses = [(bank.data_ptr(), bank.shape, bank.stride()) for bank in banks]
    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()
    output = kwargs["top_k_device_locs"]
    assert torch.equal(device[output[:2].long(), 0, 0], expected.flip(0))
    assert torch.all(output[2] == -1)
    extra_hits = int(active_capacity > base and not full_only)
    misses = topk - 2 if active_capacity and not full_only else 0
    assert kwargs["prefill_ema"].tolist() == [[extra_hits, misses]] * requests
    if scratch is not None:
        assert torch.all(scratch[:, words:] == 7)
    kwargs["observe_cache"].zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        load_cache_to_device_buffer_mla(**kwargs)
    kwargs["observe_cache"].fill_(1)
    kwargs["prefill_ema"].zero_()
    kwargs["prefill_steps"].zero_()
    if scratch is not None:
        scratch.fill_(31)  # Scratch from older objects must never affect replay.
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(device[output[:2].long(), 0, 0], expected.flip(0))
    assert kwargs["prefill_ema"].tolist() == [[extra_hits, 0]] * requests
    assert kwargs["prefill_steps"].tolist() == [1] * requests
    assert addresses == [(bank.data_ptr(), bank.shape, bank.stride()) for bank in banks]
    if scratch is not None:
        assert torch.all(scratch[:, words:] == 31)


@pytest.mark.parametrize("bad_bank", ["locations", "lru"])
def test_kvduo_rollback_rejects_incompatible_bank_shapes_before_mutation(bad_bank):
    from sglang.kernels.ops.kvcache.hisparse import rollback_kvduo_hot_metadata

    tokens = torch.ones((2, 3, 10), dtype=torch.int32, device=DEVICE)
    locations = tokens.clone()
    lru = torch.arange(8, dtype=torch.int16, device=DEVICE).repeat(2, 3, 1)
    if bad_bank == "locations":
        locations = torch.ones((2, 3, 9), dtype=torch.int32, device=DEVICE)
    else:
        lru = torch.arange(12, dtype=torch.int16, device=DEVICE).repeat(2, 3, 1)
    capacities = torch.full((2, 3), 8, dtype=torch.int32, device=DEVICE)
    banks = (tokens, locations, lru, capacities)
    before = [bank.clone() for bank in banks]
    with pytest.raises(RuntimeError, match="Size mismatch|Padded metadata"):
        rollback_kvduo_hot_metadata(
            torch.tensor([[1, 2]], dtype=torch.int64, device=DEVICE),
            torch.tensor([4], dtype=torch.int32, device=DEVICE),
            *banks,
        )
    assert all(torch.equal(bank, original) for bank, original in zip(banks, before))


def test_kvduo_resolver_graph_replays_after_no_copy_rollback_without_recapture():
    from sglang.kernels.ops.kvcache.hisparse import rollback_kvduo_hot_metadata

    kwargs = _kvduo_observation_case()
    kwargs.update(
        hot_base_capacities=torch.tensor([2], dtype=torch.int32, device=DEVICE),
        prefill_ema=torch.zeros((1, 2), dtype=torch.float32, device=DEVICE),
        prefill_steps=torch.zeros(1, dtype=torch.int64, device=DEVICE),
    )
    load_cache_to_device_buffer_mla(**kwargs)
    kwargs["observe_cache"].zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        load_cache_to_device_buffer_mla(**kwargs)
    keys = (
        "device_buffer_tokens",
        "device_buffer_locs",
        "lru_slots",
        "req_hot_buffer_sizes",
        "hot_base_capacities",
        "prefill_ema",
        "prefill_steps",
        "capacity_ema",
        "capacity_steps",
        "hot_growth_events",
    )
    addresses = [
        (kwargs[key].data_ptr(), kwargs[key].shape, kwargs[key].stride())
        for key in keys
    ]
    rollback_kvduo_hot_metadata(
        torch.tensor([[0, 0]], dtype=torch.int64, device=DEVICE),
        torch.tensor([2], dtype=torch.int32, device=DEVICE),
        kwargs["device_buffer_tokens"].unsqueeze(0),
        kwargs["device_buffer_locs"].unsqueeze(0),
        kwargs["lru_slots"].unsqueeze(0),
        kwargs["req_hot_buffer_sizes"].unsqueeze(0),
        kwargs["capacity_ema"].unsqueeze(0),
        kwargs["capacity_steps"].unsqueeze(0),
        kwargs["hot_growth_events"].unsqueeze(0),
        kwargs["prefill_ema"].unsqueeze(0),
        kwargs["prefill_steps"].unsqueeze(0),
    )
    assert torch.all(kwargs["device_buffer_tokens"][:, 2:] == -1)
    assert torch.all(kwargs["device_buffer_locs"][:, 2:] == -1)
    assert all(
        torch.all(kwargs[key] == 0)
        for key in (
            "capacity_ema",
            "capacity_steps",
            "hot_growth_events",
            "prefill_ema",
            "prefill_steps",
        )
    )
    kwargs["observe_cache"].fill_(1)
    kwargs["top_k_tokens"].copy_(
        torch.tensor([[1, 6]], dtype=torch.int32, device=DEVICE)
    )
    graph.replay()
    torch.cuda.synchronize()
    assert kwargs["top_k_device_locs"].cpu().tolist() == [[9, 7]]
    assert kwargs["prefill_ema"].cpu().tolist() == [[0, 1]]
    assert kwargs["prefill_steps"].item() == 1
    assert addresses == [
        (kwargs[key].data_ptr(), kwargs[key].shape, kwargs[key].stride())
        for key in keys
    ]
    assert torch.equal(kwargs["device_buffer"][7].cpu(), kwargs["host_cache"][6])


def test_kvduo_graph_observes_new_capacity_without_recapture_or_synthetic_reuse():
    kwargs = _kvduo_observation_case(selected=(1, 4), capacity=2, lru=(0, 1, 2, 3))
    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()
    kwargs["observe_cache"].zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        load_cache_to_device_buffer_mla(**kwargs)
    keys = (
        "prefill_ema",
        "prefill_steps",
        "capacity_ema",
        "capacity_steps",
        "req_hot_buffer_sizes",
        "lru_slots",
    )
    addresses = [
        (kwargs[key].data_ptr(), kwargs[key].shape, kwargs[key].stride())
        for key in keys
    ]
    kwargs["req_hot_buffer_sizes"].fill_(4)
    kwargs["device_buffer_tokens"][0, 2:4] = -1
    kwargs["lru_slots"][0] = torch.tensor(
        [2, 3, 0, 1], dtype=torch.int16, device=DEVICE
    )
    kwargs["prefill_ema"].zero_()
    kwargs["prefill_steps"].zero_()
    kwargs["observe_cache"].fill_(1)
    kwargs["top_k_tokens"][0] = torch.tensor([3, 6], dtype=torch.int32, device=DEVICE)
    graph.replay()
    torch.cuda.synchronize()
    assert kwargs["prefill_ema"].cpu().tolist() == [[0, 2]]
    graph.replay()
    torch.cuda.synchronize()
    assert kwargs["prefill_ema"].cpu().tolist() == [[2 / 64, 2 * 63 / 64]]
    assert kwargs["prefill_steps"].item() == 2
    assert addresses == [
        (kwargs[key].data_ptr(), kwargs[key].shape, kwargs[key].stride())
        for key in keys
    ]
