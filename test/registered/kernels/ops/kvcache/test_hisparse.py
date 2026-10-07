import sys

import pytest
import torch

import sglang.kernels.ops.kvcache.hisparse as hisparse_ops
from sglang.kernels.ops.kvcache.hisparse import (
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
        hot_page_last_touch=torch.zeros(
            (1, HOT_BUFFER_SIZE), dtype=torch.int64, device=DEVICE
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
        hot_page_last_touch=torch.zeros((1, 4), dtype=torch.int64, device=DEVICE),
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


def _kvduo_hot_tail_ema_case(
    selected=(2, 1), *, capacity=4, k=1, lru=(2, 0, 3, 1), q=99, steps=0, alpha=1 / 64
):
    state = _long_case()
    state["lru_slots"][0] = torch.tensor(lru, dtype=torch.int16, device=DEVICE)
    top_k = torch.tensor([selected], dtype=torch.int32, device=DEVICE)
    versions = torch.ones(16, dtype=torch.int64, device=DEVICE)
    score = torch.tensor([q], dtype=torch.float32, device=DEVICE)
    observations = torch.tensor([steps], dtype=torch.int64, device=DEVICE)
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
        hot_page_last_touch=torch.zeros((1, 4), dtype=torch.int64, device=DEVICE),
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
        hot_tail_ema=score,
        hot_tail_observation_steps=observations,
        hot_tail_ema_alpha=alpha,
        hot_tail_ema_observe=torch.ones(1, dtype=torch.int32, device=DEVICE),
    )
    return kwargs


def test_kvduo_tail_ema_counts_old_lru_rank_and_decays_after_hits_become_mru():
    kwargs = _kvduo_hot_tail_ema_case()
    # Old LRU [2,0,3,1] truncates slots 2 and 0; both selected HOT tokens hit
    # that region. Neither the physical slot number nor the new rank is valid
    # for measuring this step: both entries become MRU after resolution.
    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()
    assert kwargs["hot_tail_ema"].item() == 1  # B=2, F=2
    assert kwargs["hot_tail_observation_steps"].item() == 1
    assert kwargs["lru_slots"].cpu().tolist() == [[3, 1, 2, 0]]

    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()
    assert kwargs["hot_tail_ema"].item() == pytest.approx(63 / 64)
    assert kwargs["hot_tail_observation_steps"].item() == 2


@pytest.mark.parametrize(
    "selected, capacity, k, lru, expected",
    [
        ((2, 1), 3, 1, (2, 0, 1, 3), 1),  # ceil(3/2)=2, F=1
        ((2, 1), 4, 2, (2, 0, 3, 1), 2),  # two slots/page, F=1
        ((2, 1, 4), 4, 1, (2, 0, 3, 1), 1),  # n_min=3, F=1
    ],
)
def test_kvduo_tail_ema_uses_current_shrink_geometry(
    selected, capacity, k, lru, expected
):
    kwargs = _kvduo_hot_tail_ema_case(selected, capacity=capacity, k=k, lru=lru)
    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()
    assert kwargs["hot_tail_ema"].item() == expected
    assert kwargs["hot_tail_observation_steps"].item() == 1


@pytest.mark.parametrize("kind", ["miss", "full", "tail"])
def test_kvduo_tail_ema_excludes_h2d_full_and_tail_direct_hits(kind):
    selected = (3, 6) if kind == "miss" else ((2, 1) if kind == "full" else (7, -1))
    kwargs = _kvduo_hot_tail_ema_case(selected, q=2, steps=10)
    if kind == "full":
        # These logical positions ALSO have old tail HOT copies. FULL takes
        # precedence and must not contribute to the HOT-tail loss.
        kwargs["full_to_device_loc"][2] = 3
        kwargs["full_to_device_loc"][1] = 9
    elif kind == "tail":
        kwargs["full_to_device_loc"][7] = 11
    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()
    assert kwargs["hot_tail_ema"].item() == pytest.approx(2 * 63 / 64)
    assert kwargs["hot_tail_observation_steps"].item() == 11


@pytest.mark.parametrize("kind", ["minimum", "padded", "capture", "invalid_source"])
def test_kvduo_tail_ema_ignores_non_observations(kind):
    kwargs = _kvduo_hot_tail_ema_case(q=2, steps=10)
    if kind == "minimum":
        kwargs["req_hot_buffer_sizes"].fill_(2)
        kwargs["lru_slots"][0] = torch.tensor(
            [0, 1, 2, 3], dtype=torch.int16, device=DEVICE
        )
    elif kind == "padded":
        kwargs["num_real_reqs"].zero_()
    elif kind == "capture":
        kwargs["hot_tail_ema_observe"].zero_()
    else:
        kwargs["top_k_tokens"][0] = torch.tensor(
            [3, 6], dtype=torch.int32, device=DEVICE
        )
        kwargs["full_host_version"][3] = 0
    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()
    assert kwargs["hot_tail_ema"].item() == 2
    assert kwargs["hot_tail_observation_steps"].item() == 10


def test_kvduo_tail_ema_counts_only_real_rows_in_a_padded_batch():
    kwargs = _kvduo_hot_tail_ema_case()
    kwargs["top_k_tokens"] = kwargs["top_k_tokens"].repeat(2, 1)
    kwargs["top_k_device_locs"] = torch.full_like(kwargs["top_k_tokens"], -1)
    kwargs["req_pool_indices"] = kwargs["req_pool_indices"].repeat(2)
    kwargs["seq_lens"] = kwargs["seq_lens"].repeat(2)
    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()
    assert kwargs["hot_tail_ema"].item() == 1
    assert kwargs["hot_tail_observation_steps"].item() == 1
    assert kwargs["top_k_device_locs"][1].cpu().tolist() == [-1, -1]


def test_kvduo_tail_ema_survives_capture_growth_reset_and_request_slot_reuse():
    kwargs = _kvduo_hot_tail_ema_case(
        (1, 4), capacity=2, lru=(0, 1, 2, 3), q=2, steps=10
    )
    load_cache_to_device_buffer_mla(**kwargs)  # warm the JIT before capture
    graph = torch.cuda.CUDAGraph()
    kwargs["hot_tail_ema_observe"].zero_()
    with torch.cuda.graph(graph):
        load_cache_to_device_buffer_mla(**kwargs)
    assert kwargs["hot_tail_observation_steps"].item() == 10

    # Grow without clearing history. The same graph now sees n=4,F=2 and the
    # changed OLD LRU; growth puts new empty slots first. There is no Python
    # per-replay EMA calculation, and the first grown observation has B=0.
    kwargs["req_hot_buffer_sizes"].fill_(4)
    kwargs["hot_tail_ema_observe"].fill_(1)
    kwargs["device_buffer_tokens"][0, 2:4] = -1
    kwargs["lru_slots"][0] = torch.tensor(
        [2, 3, 0, 1], dtype=torch.int16, device=DEVICE
    )
    graph.replay()
    torch.cuda.synchronize()
    assert kwargs["hot_tail_ema"].item() == pytest.approx(2 * 63 / 64)
    assert kwargs["hot_tail_observation_steps"].item() == 11

    # Fill the new slots through actual H2D. The old tokens now occupy the LRU
    # tail [0,1], while the loaded tokens at [2,3] are MRU.
    kwargs["top_k_tokens"][0] = torch.tensor([2, 5], dtype=torch.int32, device=DEVICE)
    graph.replay()
    torch.cuda.synchronize()
    assert kwargs["lru_slots"].cpu().tolist() == [[0, 1, 2, 3]]

    # The replay stream completes before resetting for a merge or new owner.
    # Scores start from the next observation instead of blending old history.
    kwargs["hot_tail_ema"].zero_()
    kwargs["hot_tail_observation_steps"].zero_()
    kwargs["top_k_tokens"][0] = torch.tensor([1, 4], dtype=torch.int32, device=DEVICE)
    graph.replay()
    torch.cuda.synchronize()
    assert kwargs["hot_tail_ema"].item() == 1
    assert kwargs["hot_tail_observation_steps"].item() == 1


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
    coordinator.kvduo_hot_tail_ema = torch.tensor([[2.0]], device=DEVICE)
    coordinator.kvduo_hot_tail_observation_steps = torch.tensor(
        [[65]], dtype=torch.int64, device=DEVICE
    )
    coordinator.hot_page_last_touch = torch.ones(
        (1, 1, 8), dtype=torch.int64, device=DEVICE
    )
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
    assert coordinator._merge_kvduo_hot_pages(0, 0) == 2 * DSV4_PAGE_SIZE
    free.assert_called_once()
    assert coordinator.lru_slots[0, 0, : 2 * k].cpu().tolist() == [2 * k - 1] + list(
        range(2 * k - 1)
    )
    assert coordinator.kvduo_hot_tail_ema.item() == 0
    assert coordinator.kvduo_hot_tail_observation_steps.item() == 0


def test_kvduo_resolver_graph_replays_after_actual_merge_without_recapture():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

    kwargs = _kvduo_hot_tail_ema_case()
    load_cache_to_device_buffer_mla(**kwargs)
    kwargs["hot_tail_ema_observe"].zero_()
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
    coordinator.mem_pool_device = SimpleNamespace(kv_buffer=[kwargs["device_buffer"]])
    pages = kwargs["device_buffer_locs"][0, :4].cpu().tolist()
    coordinator._kvduo_req_layer_pages = {(0, 0): pages}
    coordinator._kvduo_hot_page_owners = {start: (0, 0) for start in pages}
    coordinator.kvduo_req_hot_capacity = torch.tensor([[4]], dtype=torch.int64)
    coordinator.kvduo_req_hot_capacity_gpu = kwargs["req_hot_buffer_sizes"].unsqueeze(0)
    coordinator.kvduo_hot_tail_ema = kwargs["hot_tail_ema"].unsqueeze(0)
    coordinator.kvduo_hot_tail_observation_steps = kwargs[
        "hot_tail_observation_steps"
    ].unsqueeze(0)
    coordinator.req_device_buffer_tokens = kwargs["device_buffer_tokens"].unsqueeze(0)
    coordinator.req_device_buffer_token_locs = kwargs["device_buffer_locs"].unsqueeze(0)
    coordinator.lru_slots = kwargs["lru_slots"].unsqueeze(0)
    coordinator.hot_page_last_touch = kwargs["hot_page_last_touch"].unsqueeze(0)
    coordinator.token_to_kv_pool_allocator = SimpleNamespace(
        free_hisparse_indices=MagicMock()
    )
    assert coordinator._merge_kvduo_hot_pages(0, 0) == 2
    assert pointers == [
        kwargs[key].data_ptr()
        for key in (
            "device_buffer_tokens",
            "device_buffer_locs",
            "lru_slots",
            "req_hot_buffer_sizes",
        )
    ]
    kwargs["hot_tail_ema_observe"].fill_(1)
    kwargs["top_k_tokens"][0] = torch.tensor([5, 4], dtype=torch.int32, device=DEVICE)
    graph.replay()
    torch.cuda.synchronize()
    assert kwargs["req_hot_buffer_sizes"].item() == 2
    assert kwargs["hot_tail_observation_steps"].item() == 0  # F=0 after merge
    for index, token in enumerate([5, 4]):
        location = kwargs["top_k_device_locs"][0, index].item()
        assert torch.equal(
            kwargs["device_buffer"][location].cpu(), kwargs["host_cache"][token]
        )


@pytest.mark.parametrize(
    "kind", ["mixed", "full", "miss", "invalid", "padding", "capture", "failed"]
)
def test_kvduo_prefill_contribution_counts_actual_slot_group_and_effective_accesses(
    kind,
):
    kwargs = _kvduo_hot_tail_ema_case(selected=(2, 1))
    counts = torch.zeros((1, 4), dtype=torch.float32, device=DEVICE)
    steps = torch.zeros(1, dtype=torch.int64, device=DEVICE)
    kwargs.update(
        hot_base_capacities=torch.tensor([2], dtype=torch.int32, device=DEVICE),
        prefill_hit_ema=counts,
        prefill_hit_steps=steps,
    )
    expected = [2, 0, 1, 1]  # slot2 extra, slot0 base, despite old LRU rank.
    if kind == "full":
        kwargs["req_hot_buffer_sizes"].zero_()
        kwargs["full_to_device_loc"][1] = 9
        kwargs["full_to_device_loc"][2] = 3
        expected = [2, 2, 0, 0]
    elif kind == "miss":
        kwargs["top_k_tokens"].copy_(
            torch.tensor([[3, 6]], dtype=torch.int32, device=DEVICE)
        )
        expected = [2, 0, 0, 0]
    elif kind == "invalid":
        kwargs["top_k_tokens"].copy_(
            torch.tensor([[-1, 2]], dtype=torch.int32, device=DEVICE)
        )
        expected = [1, 0, 0, 1]
    elif kind == "padding":
        kwargs["num_real_reqs"].zero_()
        expected = [0, 0, 0, 0]
    elif kind == "capture":
        kwargs["hot_tail_ema_observe"] = torch.zeros(
            1, dtype=torch.int32, device=DEVICE
        )
        expected = [0, 0, 0, 0]
    elif kind == "failed":
        kwargs["top_k_tokens"].copy_(
            torch.tensor([[3, 6]], dtype=torch.int32, device=DEVICE)
        )
        kwargs["full_host_version"].zero_()
        expected = [0, 0, 0, 0]
    load_cache_to_device_buffer_mla(**kwargs)
    torch.cuda.synchronize()
    assert counts.cpu().tolist() == [expected]
    assert steps.item() == int(kind not in {"padding", "capture", "failed"})


def test_kvduo_batched_no_copy_rollback_preserves_graph_addresses_and_stable_lru():
    from sglang.kernels.ops.kvcache.hisparse import rollback_kvduo_hot_metadata

    # Two selected owners and one untouched owner; physical slot ids differ
    # from LRU rank, so truncating the permutation would be incorrect.
    tokens = torch.arange(24, dtype=torch.int32, device=DEVICE).view(1, 3, 8)
    locations = (tokens + 100).clone()
    lru = torch.tensor(
        [
            [
                [7, 3, 0, 6, 1, 5, 2, 4],
                [6, 5, 4, 3, 2, 1, 0, 7],
                [0, 1, 2, 3, 4, 5, 6, 7],
            ]
        ],
        dtype=torch.int16,
        device=DEVICE,
    )
    touch = torch.ones((1, 3, 4), dtype=torch.int64, device=DEVICE)
    capacities = torch.full((1, 3), 8, dtype=torch.int32, device=DEVICE)
    addresses = [t.data_ptr() for t in (tokens, locations, lru, touch, capacities)]
    rollback_kvduo_hot_metadata(
        torch.tensor([[0, 0], [0, 1]], dtype=torch.int64, device=DEVICE),
        torch.tensor([4, 6], dtype=torch.int32, device=DEVICE),
        tokens,
        locations,
        lru,
        touch,
        capacities,
        2,
    )
    torch.cuda.synchronize()
    assert lru.cpu().tolist() == [
        [[3, 0, 1, 2, -1, -1, -1, -1], [5, 4, 3, 2, 1, 0, -1, -1], list(range(8))]
    ]
    assert capacities.cpu().tolist() == [[4, 6, 8]]
    assert touch.cpu().tolist() == [[[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]]
    assert addresses == [
        t.data_ptr() for t in (tokens, locations, lru, touch, capacities)
    ]
    assert tokens[0, 0].cpu().tolist() == [0, 1, 2, 3, -1, -1, -1, -1]
    assert locations[0, 1].cpu().tolist() == [108, 109, 110, 111, 112, 113, -1, -1]


def test_kvduo_resolver_graph_replays_after_no_copy_rollback_without_recapture():
    from sglang.kernels.ops.kvcache.hisparse import rollback_kvduo_hot_metadata

    kwargs = _kvduo_hot_tail_ema_case()
    kwargs.update(
        hot_base_capacities=torch.tensor([2], dtype=torch.int32, device=DEVICE),
        prefill_hit_ema=torch.zeros((1, 4), dtype=torch.float32, device=DEVICE),
        prefill_hit_steps=torch.zeros(1, dtype=torch.int64, device=DEVICE),
    )
    load_cache_to_device_buffer_mla(**kwargs)
    kwargs["hot_tail_ema_observe"].zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        load_cache_to_device_buffer_mla(**kwargs)
    keys = (
        "device_buffer_tokens",
        "device_buffer_locs",
        "lru_slots",
        "req_hot_buffer_sizes",
        "hot_base_capacities",
        "prefill_hit_ema",
        "prefill_hit_steps",
    )
    pointers = [kwargs[key].data_ptr() for key in keys]
    rollback_kvduo_hot_metadata(
        torch.tensor([[0, 0]], dtype=torch.int64, device=DEVICE),
        torch.tensor([2], dtype=torch.int32, device=DEVICE),
        kwargs["device_buffer_tokens"][:, :4].unsqueeze(0),
        kwargs["device_buffer_locs"][:, :4].unsqueeze(0),
        kwargs["lru_slots"].unsqueeze(0),
        kwargs["hot_page_last_touch"].unsqueeze(0),
        kwargs["req_hot_buffer_sizes"].unsqueeze(0),
        1,
    )
    kwargs["prefill_hit_ema"].zero_()
    kwargs["prefill_hit_steps"].zero_()
    kwargs["hot_tail_ema_observe"].fill_(1)
    kwargs["top_k_tokens"].copy_(
        torch.tensor([[1, 6]], dtype=torch.int32, device=DEVICE)
    )
    graph.replay()
    torch.cuda.synchronize()
    assert kwargs["top_k_device_locs"].cpu().tolist() == [[9, 7]]
    assert kwargs["prefill_hit_ema"].cpu().tolist() == [[2, 0, 1, 0]]
    assert kwargs["prefill_hit_steps"].item() == 1
    assert pointers == [kwargs[key].data_ptr() for key in keys]
    assert torch.equal(kwargs["device_buffer"][7].cpu(), kwargs["host_cache"][6])
