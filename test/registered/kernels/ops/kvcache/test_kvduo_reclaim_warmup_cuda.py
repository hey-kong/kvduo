import pytest
import torch

from sglang.kernels.ops.kvcache import kvduo_warmup as ops
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "paged,layers,dtype,width",
    [
        (True, 4, torch.uint8, 37440),
        (False, 4, torch.bfloat16, 576),
        (False, 78, torch.uint8, 656),
    ],
)
def test_real_startup_warmup_executes_moves_and_preserves_backing(
    monkeypatch, paged, layers, dtype, width
):
    # C4 packed page bytes: ceil(64 * (576 + 8) / 576) * 576 = 37440.
    backing = torch.full((3, width), 17, dtype=dtype, device="cuda")
    before = backing.clone()
    calls = []
    original = ops.execute_kvduo_hot_merge

    def execute(*args, **kwargs):
        assert args[3].data_ptr() != backing.data_ptr()
        calls.append(kwargs["paged"])
        payload, workspace = args[3:5]
        before_payload = payload.clone()
        original(*args, **kwargs)
        workspace.completion.synchronize()
        moved = int(workspace.summary_host[1])
        assert moved > 0
        src, dst = workspace.sources[:moved], workspace.destinations[:moved]
        if kwargs["paged"]:
            offsets = torch.cat(
                (
                    torch.arange(576, device="cuda"),
                    torch.arange(8, device="cuda") + 576 * 64,
                )
            )

            def entries(data, locs):
                columns = offsets[None, :] + torch.where(
                    offsets[None, :] < 576,
                    (locs % 64)[:, None] * 576,
                    (locs % 64)[:, None] * 8,
                )
                return data[(locs // 64)[:, None], columns]

            assert torch.equal(entries(payload, dst), entries(before_payload, src))
        else:
            assert torch.equal(payload[dst], before_payload[src])

    monkeypatch.setattr(ops, "execute_kvduo_hot_merge", execute)
    ops.warmup_kvduo_reclaim(
        backing=backing,
        page_size=64,
        storage_layers=layers,
        maximum=8 * 64 * layers,
        padded=8 * 64 * layers + 64,
        paged=paged,
    )
    assert calls == [paged]
    assert torch.equal(backing, before)
