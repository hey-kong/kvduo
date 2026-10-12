"""KVDuo prefill gathers indexer data with logical, not FULL physical, pages."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.kernels.ops.attention.dsa import index_buf_accessor
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4 import indexer as indexer_module
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4TokenToKVPool,
    HiSparseC4DevicePool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_dsv4_topk import indexer_case

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "prefix,queries,pages",
    [(0, 12, [17]), (512, 8, [9, 3, 17]), (520, 8, [9, 3, 17]), (512, 7, [9, 3, 17])],
)
def test_kvduo_nonpaged_prefill_uses_logical_prefix_and_chunk_pages(
    indexer_case, monkeypatch, prefix, queries, pages
):
    case = indexer_case(mode=ForwardMode.EXTEND, queries=queries)
    del case.backend._get_nonpaged_indexer_plan
    monkeypatch.setattr(indexer_module, "is_cuda", lambda: True)
    monkeypatch.setattr(indexer_module, "is_hip", lambda: False)
    monkeypatch.setattr(indexer_module, "is_in_tc_piecewise_cuda_graph", lambda: False)
    monkeypatch.setattr(indexer_module, "is_in_breakable_cuda_graph", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    end = prefix + queries
    lengths = (torch.arange(prefix + 1, end + 1) // 4).clamp_min(1).int()
    logical_pages = torch.tensor([pages], dtype=torch.int32).repeat(queries, 1)
    case.metadata.page_table = case.core.page_table = logical_pages
    case.metadata.c4_seq_lens = lengths
    case.core.positions = torch.arange(prefix, end)
    case.batch.__dict__.update(
        batch_size=1,
        _original_forward_mode=None,
        tbo_parent_token_range=None,
        seq_lens=torch.tensor([end]),
        seq_lens_cpu=[end],
        extend_seq_lens=torch.tensor([queries]),
        extend_seq_lens_cpu=[queries],
        extend_start_loc=torch.tensor([0]),
        extend_num_tokens=queries,
    )
    with get_context().override_server_args(enable_deepseek_v4_fp4_indexer=False):
        pool = DeepSeekV4IndexerPool(
            20 * 64, 64, torch.float8_e4m3fn, 128, 1, "cpu", False
        )
    buffer = pool.index_k_with_scale_buffer[0]
    key_bytes = (
        (torch.arange(buffer.shape[0] * 64 * 128) % 112).byte().view(-1, 64, 128)
    )
    scale_values = torch.arange(buffer.shape[0] * 64).float().add(1).view(-1, 64)
    buffer[:, : 64 * 128] = key_bytes.flatten(1)
    buffer[:, 64 * 128 :] = scale_values.view(torch.uint8)

    def gather(
        source_pool, source, *, page_indices, seq_len_tensor, seq_len_sum, max_seq_len
    ):
        assert source_pool is pool and source is buffer
        assert seq_len_tensor.tolist() == [end // 4]
        assert seq_len_sum == max_seq_len == end // 4
        positions = torch.arange(seq_len_sum)
        page_ids = page_indices[0, positions // 64].long()
        offsets = positions % 64
        keys = source[:, : 64 * 128].view(-1, 64, 128)[page_ids, offsets]
        scales = source[:, 64 * 128 :].view(torch.float32)[page_ids, offsets, None]
        return keys, scales.view(torch.uint8)

    gather_kernel = MagicMock(side_effect=gather)
    monkeypatch.setattr(index_buf_accessor.GetKAndS, "execute", gather_kernel)
    full_pool = HiSparseC4DevicePool.__new__(HiSparseC4DevicePool)
    mapping = torch.arange(pool.size + 64, 64, -1, dtype=torch.int64)
    full_pool.register_mapping(mapping)
    parent = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    parent.c4_indexer_kv_pool = pool
    parent.c4_kv_pool = full_pool
    parent.layer_mapping = {7: (4, 0, full_pool)}
    parent.wait_layer_transfer = MagicMock()
    case.backend.token_to_kv_pool = parent
    deep_gemm = sys.modules["deep_gemm"]
    deep_gemm.fp8_mqa_logits = MagicMock(return_value=case.scores)

    with (
        envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER.override(True),
        envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS.override(8),
        get_parallel().override(attn_cp_size=1),
    ):
        case.run()

    if queries < 8:
        assert case.metadata.nonpaged_plan is None
        gather_kernel.assert_not_called()
        deep_gemm.fp8_mqa_logits.assert_not_called()
        deep_gemm.fp8_paged_mqa_logits.assert_called_once()
    else:
        plan = case.metadata.nonpaged_plan
        assert plan.seq_len_sum == end // 4
        assert plan.max_seqlen_k == ((end // 4 + 63) // 64) * 64
        assert plan.page_table.tolist() == [pages]
        assert torch.equal(plan.ke, lengths)
        assert not plan.ks.any()
        gather_kernel.assert_called_once()
        deep_gemm.fp8_mqa_logits.assert_called_once()
        deep_gemm.fp8_paged_mqa_logits.assert_not_called()
        gathered_k, gathered_scale = deep_gemm.fp8_mqa_logits.call_args.args[1]
        positions = torch.arange(end // 4)
        logical_ids = logical_pages[0, positions // 64].long()
        assert torch.equal(
            gathered_k.view(torch.uint8), key_bytes[logical_ids, positions % 64]
        )
        assert torch.equal(gathered_scale, scale_values[logical_ids, positions % 64])
    for row, length in enumerate(lengths.tolist()):
        positions = torch.arange(length)
        logical_ids = logical_pages[row, positions // 64].long() * 64 + positions % 64
        assert torch.equal(case.raw[row, :length].long(), positions)
        assert torch.equal(
            case.core.c4_sparse_page_indices[row, :length].long(), mapping[logical_ids]
        )
    case.backend.hisparse_coordinator.swap_in_selected_pages.assert_not_called()
    case.v2.assert_not_called()
