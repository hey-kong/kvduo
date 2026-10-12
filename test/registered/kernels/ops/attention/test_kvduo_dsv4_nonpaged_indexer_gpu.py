"""Compare KVDuo paged/nonpaged prefill through the real gather and DeepGEMM."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsv4 import indexer as indexer_module
from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin, FP8_DTYPE
from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4TokenToKVPool,
    HiSparseC4DevicePool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability()[0] not in (9, 10),
    reason="Requires CUDA SM90/SM100 and DeepGEMM.",
)


class Indexer(C4IndexerBackendMixin):
    pass


@pytest.mark.parametrize("prefix", [0, 512, 516])
@torch.inference_mode()
def test_kvduo_nonpaged_matches_paged_with_logical_prefix_and_partial_chunk(
    monkeypatch, prefix
):
    deep_gemm = pytest.importorskip("deep_gemm")
    queries, heads, dim, k = 8192, 64, 128, 512
    end = prefix + queries
    count = end // 4
    page_count = (count + 63) // 64
    logical_pages = torch.arange(page_count, 0, -1, dtype=torch.int32, device="cuda")
    positions = torch.arange(count, device="cuda")
    logical_locs = logical_pages[positions // 64].long() * 64 + positions % 64
    keys = torch.zeros((count, dim), device="cuda")
    keys[:, 0] = positions // 256
    keys[:, 1] = (positions // 16) % 16
    keys[:, 2] = positions % 16
    query = torch.zeros((queries, heads, dim), device="cuda")
    query[..., :3] = torch.tensor([256, 16, 1], device="cuda")
    query = query.to(FP8_DTYPE)
    weights = torch.full((queries, heads, 1), 1 / heads, device="cuda")
    lengths = (torch.arange(prefix + 1, end + 1, device="cuda") // 4).clamp_min(1).int()
    page_table = logical_pages[None].repeat(queries, 1)
    output = torch.full((queries, k), -1, dtype=torch.int32, device="cuda")
    raw = torch.full_like(output, -1)

    with (
        get_context().override_server_args(enable_deepseek_v4_fp4_indexer=False),
        envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.override(False),
        envs.SGLANG_OPT_USE_TOPK_V2.override(False),
        envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(False),
        envs.SGLANG_OPT_USE_AITER_INDEXER.override(False),
        envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
        envs.SGLANG_TOPK_TRANSFORM_512_TORCH.override(False),
        envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS.override(8192),
        get_parallel().override(attn_cp_size=1),
    ):
        index_pool = DeepSeekV4IndexerPool(
            (page_count + 1) * 64, 64, FP8_DTYPE, dim, 1, "cuda", False
        )
        index_pool.set_index_k_scale_buffer(
            0, logical_locs, keys.to(FP8_DTYPE), torch.ones((count, 1), device="cuda")
        )
        full_pool = HiSparseC4DevicePool.__new__(HiSparseC4DevicePool)
        mapping = torch.zeros((page_count + 2) * 64, dtype=torch.int64, device="cuda")
        mapping[logical_locs] = positions + 64
        full_pool.register_mapping(mapping)
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool.c4_indexer_kv_pool = index_pool
        pool.c4_kv_pool = full_pool
        pool.layer_mapping = {7: (4, 0, full_pool)}
        pool.wait_layer_transfer = MagicMock()
        metadata = PagedIndexerMetadata(256, page_table, lengths)
        core = SimpleNamespace(
            positions=torch.arange(prefix, end, device="cuda"),
            page_table=page_table,
            c4_sparse_page_indices=output,
            c4_sparse_raw_indices=raw,
        )
        backend = Indexer()
        backend.token_to_kv_pool = pool
        backend.forward_metadata = SimpleNamespace(
            indexer_metadata=metadata, core_metadata=core
        )
        backend.hisparse_coordinator = SimpleNamespace(enable_mixed_residency=True)
        backend.debug_use_external_c4_sparse_indices = False
        backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
        backend._forward_prepare_normal = MagicMock(return_value=(query, weights))
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            _original_forward_mode=None,
            tbo_parent_token_range=None,
            seq_lens=torch.tensor([end], device="cuda"),
            seq_lens_cpu=[end],
            extend_seq_lens=torch.tensor([queries], device="cuda"),
            extend_seq_lens_cpu=[queries],
            extend_start_loc=torch.tensor([0], device="cuda"),
            extend_num_tokens=queries,
        )
        c4_indexer = SimpleNamespace(use_fp4_indexer=False, layer_id=7)
        captured_logits = []
        topk = indexer_module.topk_transform_512

        def capture_logits(*args):
            captured_logits.append(args[0])
            topk(*args)

        monkeypatch.setattr(indexer_module, "topk_transform_512", capture_logits)
        monkeypatch.setattr(indexer_module, "get_global_indexer_capturer", lambda: None)
        paged = MagicMock(wraps=deep_gemm.fp8_paged_mqa_logits)
        nonpaged = MagicMock(wraps=deep_gemm.fp8_mqa_logits)
        monkeypatch.setattr(deep_gemm, "fp8_paged_mqa_logits", paged)
        monkeypatch.setattr(deep_gemm, "fp8_mqa_logits", nonpaged)
        results = []
        for enabled in (False, True):
            core.c4_sparse_page_indices = output
            with envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER.override(enabled):
                backend.forward_c4_indexer(
                    torch.empty((queries, 1), device="cuda"),
                    torch.empty((queries, 1), device="cuda"),
                    c4_indexer,
                    batch,
                )
            torch.cuda.synchronize()
            results.append(
                (
                    captured_logits[-1][:, :count].clone(),
                    raw.clone(),
                    core.c4_sparse_page_indices.clone(),
                )
            )
        paged.assert_called_once()
        nonpaged.assert_called_once()
        plan = metadata.nonpaged_plan
        assert plan is not None and plan.seq_len_sum == count
        assert torch.equal(plan.page_table[0], logical_pages)
        assert torch.equal(plan.ke, lengths)

    valid = positions[None] < lengths[:, None]
    torch.testing.assert_close(
        results[0][0][valid], results[1][0][valid], rtol=0, atol=0.01
    )
    torch.testing.assert_close(
        results[1][0][valid],
        positions[None].expand(queries, count)[valid].float(),
        rtol=0,
        atol=0.01,
    )
    assert torch.equal(
        torch.sort(results[0][1], dim=1).values, torch.sort(results[1][1], dim=1).values
    )
    assert torch.equal(
        torch.sort(results[0][2], dim=1).values, torch.sort(results[1][2], dim=1).values
    )
    for row in (0, 127, 4095, queries - 1):
        visible = int(lengths[row])
        expected = torch.cat(
            [
                torch.full((k - min(k, visible),), -1, device="cuda"),
                torch.arange(max(0, visible - k), visible, device="cuda"),
            ]
        )
        assert torch.equal(torch.sort(results[1][1][row]).values.long(), expected)
