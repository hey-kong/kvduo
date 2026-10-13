from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import tvm_ffi

from sglang.kernels.ops.attention.dsv4 import topk
from sglang.kernels.ops.kvcache import kvduo_full_reclaim
from sglang.kernels.ops.kvcache.kvduo_full_decode import KVDuoFullTopK
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def full_case():
    scores = torch.zeros((2, 256), dtype=torch.float32)
    raw = torch.empty((2, 512), dtype=torch.int32)
    spec = KVDuoFullTopK(
        torch.empty_like(raw),
        torch.arange(2, dtype=torch.int64),
        torch.zeros((2, 136), dtype=torch.int64)[:, :128],
        torch.ones(256, dtype=torch.int64),
        torch.zeros(256, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int32),
        torch.zeros((2, 2), dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        torch.full((1,), 2, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros((2, 2), dtype=torch.float32),
        torch.zeros(2, dtype=torch.int64),
        1,
        512,
        64,
    )
    return scores, raw, spec


def launch(scores, raw, spec):
    topk.topk_transform_512_v2(
        scores,
        torch.full((2,), 128, dtype=torch.int32),
        torch.empty((2, 0), dtype=torch.int32),
        torch.empty_like(raw),
        64,
        torch.zeros((3, 2), dtype=torch.int32),
        raw,
        raw_only=True,
        full_metadata=spec,
    )


def test_real_ffi_transports_full_metadata_as_pointers_and_retains_lookup_stride(
    monkeypatch,
):
    scores, raw, spec = full_case()
    observed = []
    callback = tvm_ffi.convert(lambda *args: observed.append(args))
    monkeypatch.setattr(
        topk,
        "_jit_topk_v2_module",
        lambda: SimpleNamespace(topk_transform_full=callback),
    )
    for _ in range(2):
        with torch.inference_mode():
            launch(scores, raw, spec)
        args = observed[-1]
        assert len(args) == 27
        assert all(isinstance(args[i], tvm_ffi.Tensor) for i in (0, 1, 2, 3, 5, 6))
        assert all(isinstance(arg, int) for arg in args[7:])
        assert args[7:20] == tuple(
            getattr(spec, field).data_ptr()
            for field in (
                "output",
                "requests",
                "lookup",
                "mapping",
                "touches",
                "status",
                "stats",
                "growth",
                "observe",
                "real_requests",
                "clock_base",
                "prefill_ema",
                "prefill_steps",
            )
        )
        assert args[20:] == (2, 128, 136, 256, 1, 512, 64)


@pytest.mark.parametrize(
    "name,invalid",
    [
        ("requests", lambda s: s.requests.int()),
        ("mapping", lambda s: s.mapping.int()),
        ("touches", lambda s: s.touches[:-1]),
        ("status", lambda s: s.status[:1]),
        ("stats", lambda s: torch.empty((2, 2), dtype=torch.int32).T),
        ("growth", lambda s: torch.empty(2, dtype=torch.int32, device="meta")),
        ("observe", lambda s: torch.ones(1, dtype=torch.int64)),
        ("clock_base", lambda s: torch.ones(2, dtype=torch.int64)),
        ("prefill_ema", lambda s: s.prefill_ema.double()),
        ("prefill_steps", lambda s: s.prefill_steps.int()),
        ("lookup", lambda s: torch.zeros((2, 256), dtype=torch.int64)[:, ::2]),
    ],
)
def test_invalid_full_metadata_fails_before_jit_load_or_pointer_launch(
    monkeypatch, name, invalid
):
    scores, raw, spec = full_case()
    loader = MagicMock(side_effect=AssertionError("unsafe native launch"))
    monkeypatch.setattr(topk, "_jit_topk_v2_module", loader)
    with pytest.raises(ValueError, match=name):
        launch(scores, raw, replace(spec, **{name: invalid(spec)}))
    loader.assert_not_called()


@pytest.mark.parametrize("protected", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_full_summary_dispatch_passes_only_logical_pages_as_a_tensor(
    monkeypatch, protected, dtype
):
    pages = torch.arange(128, dtype=dtype).reshape(2, 64)
    banks = [torch.arange(128, dtype=torch.int64) for _ in range(4)]
    protection = torch.tensor([17], dtype=dtype) if protected else None
    native = MagicMock()
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(
        kvduo_full_reclaim,
        "_jit_full_reclaim_module",
        lambda: SimpleNamespace(summarize_i32=native, summarize_i64=native),
    )
    outputs = kvduo_full_reclaim.summarize_kvduo_full_pages(pages, *banks, protection)
    args = native.call_args.args
    assert args[0] is pages
    assert all(isinstance(value, int) for value in args[1:])
    assert args[1:5] == tuple(bank.data_ptr() for bank in banks)
    assert bool(args[5]) == protected
    assert args[6:9] == tuple(output.data_ptr() for output in outputs)
    assert args[9] == 128
