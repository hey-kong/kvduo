"""Exercise the real C4 indexer dispatch and raw-index handoff on CPU."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsv4 import indexer as indexer_module
from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class Indexer(C4IndexerBackendMixin):
    pass


def emit_topk(scores, lengths, pages, output, page_size, raw=None, raw_only=False):
    if not raw_only:
        output.fill_(-1)
    if raw is not None:
        raw.fill_(-1)
    k = output.shape[1]
    for row, length in enumerate(lengths.tolist()):
        positions = (
            torch.topk(scores[row, :length], k).indices
            if length > k
            else torch.arange(length)
        )
        count = positions.numel()
        if not raw_only:
            output[row, :count] = (
                pages[row, positions // page_size] * page_size + positions % page_size
            )
        if raw is not None:
            raw[row, :count] = positions


@pytest.fixture
def indexer_case(monkeypatch):
    monkeypatch.setattr(indexer_module, "get_global_indexer_capturer", lambda: None)

    def build(k=512, *, mode=ForwardMode.DECODE, coordinator="kvduo", queries=4):
        lengths = torch.tensor([2048, 256, 0, 0], dtype=torch.int32)
        pages = torch.arange(32, dtype=torch.int32).flip(0).repeat(4, 1)
        scores = torch.arange(2048, dtype=torch.float32).repeat(queries, 1)
        deep_gemm = SimpleNamespace(
            get_num_sms=lambda: 1,
            get_paged_mqa_logits_metadata=lambda *args: torch.zeros(
                (4, 2), dtype=torch.int32
            ),
            fp8_paged_mqa_logits=MagicMock(return_value=scores),
        )
        monkeypatch.setitem(sys.modules, "deep_gemm", deep_gemm)
        monkeypatch.setattr(
            "sglang.kernels.ops.attention.dsv4.plan_topk_v2",
            lambda seq: torch.zeros((seq.numel() + 1, 2), dtype=torch.int32),
        )
        metadata = PagedIndexerMetadata(256, pages, lengths)
        output = torch.full((queries, k), -123, dtype=torch.int32)
        raw = torch.full_like(output, -456)
        core = SimpleNamespace(
            positions=torch.arange(queries),
            page_table=pages,
            c4_sparse_page_indices=output,
            c4_sparse_raw_indices=raw if mode.is_extend() else None,
        )
        backend = Indexer()
        backend.forward_metadata = SimpleNamespace(
            indexer_metadata=metadata, core_metadata=core
        )
        backend.token_to_kv_pool = SimpleNamespace(
            get_index_k_with_scale_buffer=lambda **_: torch.zeros(
                (1, 64 * 132), dtype=torch.uint8
            ),
            layer_mapping={7: SimpleNamespace(compress_layer_id=2)},
            c4_kv_pool=SimpleNamespace(
                translate_loc_to_hisparse_device=lambda locations: locations + 100
            ),
        )
        backend.debug_use_external_c4_sparse_indices = False
        backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
        backend._get_nonpaged_indexer_plan = MagicMock(return_value=None)
        backend._forward_prepare_normal = MagicMock(
            return_value=(torch.zeros((queries, 1, 128)), torch.ones((queries, 1, 1)))
        )
        backend.hisparse_coordinator = (
            SimpleNamespace(
                enable_mixed_residency=coordinator == "kvduo",
                full_touch_clock=torch.zeros(1, dtype=torch.int64),
                raw_indices_buffer=raw,
                swap_in_selected_pages=MagicMock(
                    side_effect=lambda **kw: torch.where(
                        kw["top_k_result"] >= 0, kw["top_k_result"] + 10_000, -1
                    )
                ),
            )
            if coordinator is not None
            else None
        )
        batch = SimpleNamespace(
            forward_mode=mode,
            req_pool_indices=torch.tensor([1, 0, 9, 9][:queries], dtype=torch.int64),
        )
        c4_indexer = SimpleNamespace(use_fp4_indexer=False, layer_id=7)
        v1 = MagicMock(side_effect=emit_topk)
        v2 = MagicMock(
            side_effect=lambda scores, seq, pages, out, page_size, plan, raw, raw_only=False, touch_clock=None: emit_topk(
                scores, seq, pages, out, page_size, raw, raw_only
            )
        )
        monkeypatch.setattr(indexer_module, "topk_transform_512", v1)
        monkeypatch.setattr(indexer_module, "topk_transform_512_v2", v2)

        def run():
            backend.forward_c4_indexer(
                torch.zeros((queries, 1)),
                torch.zeros((queries, 1)),
                c4_indexer,
                batch,
            )

        return SimpleNamespace(
            backend=backend,
            metadata=metadata,
            core=core,
            batch=batch,
            scores=scores,
            raw=raw,
            output=output,
            v1=v1,
            v2=v2,
            run=run,
        )

    with (
        envs.SGLANG_OPT_USE_TOPK_V2.override(True),
        envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.override(False),
        envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(False),
        envs.SGLANG_OPT_USE_AITER_INDEXER.override(False),
        envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
        envs.SGLANG_TOPK_TRANSFORM_512_TORCH.override(False),
    ):
        yield build


@pytest.mark.parametrize("k", [512, 1024])
@pytest.mark.parametrize("capture", [False, True])
def test_kvduo_decode_v2_passes_raw_positions_to_resolver(
    indexer_case, monkeypatch, k, capture
):
    case = indexer_case(k)
    capturer = MagicMock() if capture else None
    monkeypatch.setattr(indexer_module, "get_global_indexer_capturer", lambda: capturer)
    raw_ptr = case.raw.data_ptr()
    plan_ptr = case.metadata.topk_metadata.data_ptr()
    for _ in range(2):
        case.run()
        args = case.v2.call_args.args
        assert case.v2.call_args.kwargs == {
            "raw_only": True,
            "touch_clock": case.backend.hisparse_coordinator.full_touch_clock,
        }
        assert torch.all(case.output == -123)
        assert args[1] is case.metadata.c4_seq_lens
        assert args[5] is case.metadata.topk_metadata
        selected = args[6]
        assert selected.shape == (4, k)
        assert torch.all(selected[0] >= 0) and torch.all(selected[0] < 2048)
        assert selected[1, :256].tolist() == list(range(256))
        assert torch.all(selected[1, 256:] == -1)
        assert torch.all(selected[2:] == -1)
        resolver = case.backend.hisparse_coordinator.swap_in_selected_pages
        call = resolver.call_args.kwargs
        assert call["top_k_result"] is selected
        assert call["req_pool_indices"] is case.batch.req_pool_indices
        assert call["layer_id"] == 2
        assert call["touch_clock_advanced"] is True
        assert torch.equal(
            case.core.c4_sparse_page_indices,
            torch.where(selected >= 0, selected + 10_000, -1),
        )
        assert case.raw.data_ptr() == raw_ptr
        assert case.metadata.topk_metadata.data_ptr() == plan_ptr
        if capturer is not None:
            assert capturer.capture.call_args.args == (2, selected)
        else:
            assert selected.data_ptr() == raw_ptr
        case.scores.neg_()
    case.v1.assert_not_called()
    assert case.v2.call_count == 2


@pytest.mark.parametrize("reason", ["disabled", "hisparse", "trimmed", "missing_plan"])
def test_kvduo_v2_ineligible_decode_keeps_v1(indexer_case, reason):
    with envs.SGLANG_OPT_USE_TOPK_V2.override(reason != "disabled"):
        case = indexer_case(
            coordinator="hisparse" if reason == "hisparse" else "kvduo",
            queries=2 if reason == "trimmed" else 4,
        )
        if reason == "missing_plan":
            case.metadata.topk_metadata = torch.empty((0, 2), dtype=torch.int32)
        case.run()
    case.v1.assert_called_once()
    case.v2.assert_not_called()
    assert case.backend.hisparse_coordinator.swap_in_selected_pages.call_count == 1
    assert (
        "touch_clock_advanced"
        not in case.backend.hisparse_coordinator.swap_in_selected_pages.call_args.kwargs
    )


def test_ordinary_decode_keeps_v2_without_raw_output(indexer_case):
    case = indexer_case(coordinator=None)
    case.run()
    case.v1.assert_not_called()
    case.v2.assert_called_once()
    assert case.v2.call_args.args[6] is None
    assert case.v2.call_args.kwargs == {"raw_only": False}
    assert case.core.c4_sparse_page_indices is case.output


@pytest.mark.parametrize("coordinator", [None, "kvduo"])
def test_prefill_raw_output_keeps_existing_v1_path(indexer_case, coordinator):
    case = indexer_case(mode=ForwardMode.EXTEND, coordinator=coordinator)
    case.run()
    case.v1.assert_called_once()
    case.v2.assert_not_called()
    assert case.v1.call_args.args[5] is case.raw
    if coordinator is not None:
        case.backend.hisparse_coordinator.swap_in_selected_pages.assert_not_called()


@pytest.mark.parametrize("raw_only", [False, True])
def test_topk_v2_wrapper_passes_output_mode_without_device_reads(monkeypatch, raw_only):
    from sglang.kernels.ops.attention.dsv4 import topk as ops

    args = (
        torch.zeros((2, 1024)),
        torch.tensor([1024, 0], dtype=torch.int32),
        torch.empty((2, 0), dtype=torch.int32),
        torch.full((2, 512), -99, dtype=torch.int32),
        64,
        torch.zeros((3, 2), dtype=torch.int32),
        torch.empty((2, 512), dtype=torch.int32),
    )
    module = SimpleNamespace(topk_transform=MagicMock())
    monkeypatch.setattr(ops, "_jit_topk_v2_module", lambda: module)
    for operation in ("item", "cpu", "tolist"):
        monkeypatch.setattr(
            torch.Tensor,
            operation,
            MagicMock(side_effect=AssertionError("device read")),
        )
    ops.topk_transform_512_v2(*args, raw_only=raw_only)
    module.topk_transform.assert_called_once_with(*args, raw_only, None)


def test_topk_v2_raw_only_rejects_missing_or_aliased_output_before_loading(monkeypatch):
    from sglang.kernels.ops.attention.dsv4 import topk as ops

    load = MagicMock()
    monkeypatch.setattr(ops, "_jit_topk_v2_module", load)
    out = torch.empty((1, 512), dtype=torch.int32)
    args = (
        torch.zeros((1, 1024)),
        torch.tensor([1024]),
        torch.zeros((1, 16)),
        out,
        64,
        torch.zeros((2, 2)),
    )
    with pytest.raises(ValueError, match="requires a raw output"):
        ops.topk_transform_512_v2(*args, raw_only=True)
    with pytest.raises(ValueError, match="must not alias"):
        ops.topk_transform_512_v2(*args, out, raw_only=True)
    backing = torch.empty(out.numel() + 1, dtype=out.dtype)
    overlapping_args = (*args[:3], backing[:-1].view_as(out), *args[4:])
    with pytest.raises(ValueError, match="must not alias"):
        ops.topk_transform_512_v2(
            *overlapping_args, backing[1:].view_as(out), raw_only=True
        )


@pytest.mark.parametrize(
    "raw_only,clock",
    [
        (False, torch.zeros(1, dtype=torch.int64)),
        (True, torch.zeros(1)),
        (True, torch.zeros(2, dtype=torch.int64)),
    ],
)
def test_topk_v2_rejects_invalid_clock_before_launch(monkeypatch, raw_only, clock):
    from sglang.kernels.ops.attention.dsv4 import topk as ops

    load = MagicMock(side_effect=AssertionError("invalid clock reached JIT"))
    monkeypatch.setattr(ops, "_jit_topk_v2_module", load)
    out = torch.empty((1, 512), dtype=torch.int32)
    with pytest.raises(ValueError, match="clock"):
        ops.topk_transform_512_v2(
            torch.zeros((1, 1024)),
            torch.ones(1, dtype=torch.int32),
            torch.empty((1, 0), dtype=torch.int32),
            out,
            64,
            torch.zeros((2, 2), dtype=torch.int32),
            torch.empty_like(out),
            raw_only=raw_only,
            touch_clock=clock,
        )
    load.assert_not_called()
    load.assert_not_called()
