"""Sparse prefill must gather physical C4 slots, not logical token IDs."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    SparsePrefillChunkCache,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def make_plan(device, mapped):
    cache = SimpleNamespace(
        seq_lens=torch.tensor([32, 16], dtype=torch.int32, device=device),
        max_seq_len=32,
        num_reqs=2,
        query_start_loc=torch.tensor([0, 2, 3], device=device),
        swa_offsets=torch.tensor([0, 4, 8], device=device),
        c4_flat_token_ids=None,
    )
    if mapped:
        table = torch.tensor([[8, 9], [8, 9], [10, 1000000]], device=device)
        mapping = torch.zeros(44, dtype=torch.int64, device=device)
        mapping[32:40] = torch.tensor([8, 9, 10, 11, 4, 5, 6, 7], device=device)
        mapping[40:44] = torch.arange(12, 16, device=device)
    else:
        table = torch.tensor([[2, 1], [2, 1], [3, 0]], device=device)
        mapping = None
    SparsePrefillChunkCache.ensure_c4(cache, table, 4, mapping)
    return cache, table, mapping


@pytest.mark.parametrize("mapped", [False, True])
def test_sparse_prefill_c4_address_space_and_short_request_padding(mapped):
    cache, table, mapping = make_plan("cpu", mapped)
    expected = [8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
    expected += [0] * 4 if mapped else [0, 1, 2, 3]
    assert cache.c4_flat_token_ids.tolist() == expected
    assert cache.c4_flat_token_ids.dtype == torch.int32
    assert cache.c4_compressed_base.tolist() == [0, 8]
    assert cache.c4_swa_base.tolist() == [16, 20]
    first = cache.c4_flat_token_ids
    SparsePrefillChunkCache.ensure_c4(cache, table, 4, mapping)
    assert cache.c4_flat_token_ids is first


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sparse_prefill_mapped_addresses_dequantize_physical_payload():
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
        dequantize_k_cache_paged,
        dequantize_k_cache_paged_ref,
    )

    cache, _, _ = make_plan("cuda", True)
    # Logical slots begin at 32, beyond this physical pool's 16 slots.
    raw = torch.zeros((4, 2880), dtype=torch.uint8, device="cuda")
    for page in range(4):
        for offset in range(4):
            value = page * 4 + offset
            raw[page, offset * 576 : offset * 576 + 448] = (
                torch.full((448,), float(value), device="cuda")
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
            )
            raw[page, 2304 + offset * 8 : 2304 + offset * 8 + 7] = 127
    actual = dequantize_k_cache_paged(raw, cache.c4_flat_token_ids, 4)
    expected = dequantize_k_cache_paged_ref(raw, cache.c4_flat_token_ids, 4)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        actual[:, 0, 0], cache.c4_flat_token_ids.to(torch.bfloat16), rtol=0, atol=0
    )


def test_backend_sparse_prefill_passes_c4_mapping(monkeypatch):
    import ast
    import sys
    from pathlib import Path

    path = (
        Path(__file__).parents[4]
        / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4AttnBackend"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_forward_prefill_sparse"
    )
    gathered = []
    namespace = {
        "dequantize_k_cache_paged": lambda buf, ids, **kwargs: gathered.append(
            ids.clone()
        )
    }
    exec("from __future__ import annotations\n" + ast.unparse(method), namespace)
    monkeypatch.setitem(
        sys.modules,
        "sgl_kernel.flash_mla",
        SimpleNamespace(flash_mla_sparse_fwd=lambda **kw: (kw["kv"], None, None)),
    )
    cache, table, mapping = make_plan("cpu", True)
    cache.c4_flat_token_ids = None
    cache.num_qo_tokens = 3
    cache.swa_token_ids = torch.tensor([0, 1])
    cache.swa_page_size = 4
    cache.ensure_c4 = lambda *a, **kw: SparsePrefillChunkCache.ensure_c4(
        cache, *a, **kw
    )
    cache.combine_c4_layer = lambda **kw: (
        torch.zeros((3, 1), dtype=torch.int32),
        torch.ones(3, dtype=torch.int32),
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(sparse_prefill_cache=cache),
        sparse_prefill_workspace=SimpleNamespace(
            get=lambda n: torch.zeros((n, 1, 512), dtype=torch.bfloat16)
        ),
        softmax_scale=1,
        head_dim_v=512,
    )
    pool = SimpleNamespace(
        get_extra_key_page_size=lambda layer: 4,
        get_extra_key_buffer=lambda layer: None,
        get_swa_key_buffer_radix=lambda layer: None,
        c4_kv_pool=SimpleNamespace(full_to_hisparse_device_index_mapping=mapping),
    )
    namespace["_forward_prefill_sparse"](
        backend,
        q=torch.zeros((3, 1, 1, 512)),
        layer_id=0,
        compress_ratio=4,
        forward_batch=None,
        token_to_kv_pool=pool,
        core_attn_metadata=SimpleNamespace(
            page_table=table,
            c4_sparse_raw_indices=torch.zeros((3, 512), dtype=torch.int32),
        ),
        attn_sink=torch.zeros(1),
    )
    assert gathered[0].tolist() == [
        8,
        9,
        10,
        11,
        4,
        5,
        6,
        7,
        12,
        13,
        14,
        15,
        0,
        0,
        0,
        0,
    ]
