import runpy
from pathlib import Path

import pytest
import torch

from sglang.kernels.ops.kvcache.kvduo_hand_plan import (
    allocate_kvduo_hand_workspace,
    prepare_kvduo_hand,
)
from sglang.srt.utils import is_cuda, is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not (is_cuda() or is_hip()),
    reason="Hand planning requires CUDA/ROCm",
)


@pytest.mark.parametrize("layers,requests", [(2, 3), (7, 70), (80, 257)])
@pytest.mark.parametrize("needed", [1, 31, 100000])
def test_gpu_hand_targets_and_compacted_prefix_match_host_policy(
    layers, requests, needed
):
    generator = torch.Generator().manual_seed(37)
    counts = torch.rand((layers, requests, 9), generator=generator) * 4
    counts[:, :, 8] *= 12
    counts[0, 0, 0] = float("nan")
    counts[0, 1, 1] = float("inf")
    counts[0, 2, 2] = -1
    steps = torch.randint(20, 100, (layers, requests), generator=generator)
    pages = torch.randint(
        0, 9, (layers, requests), generator=generator, dtype=torch.int32
    )
    host = allocate_kvduo_hand_workspace(counts)
    host.pages_host.copy_(pages)
    device_counts, device_steps = counts.cuda(), steps.cuda()
    workspace = allocate_kvduo_hand_workspace(device_counts)
    workspace.pages_host.copy_(pages)
    tensors = (
        device_counts,
        device_steps,
        workspace.pages,
        workspace.targets,
        workspace.packet,
    )
    addresses = [(t.data_ptr(), t.shape, t.stride()) for t in tensors]
    for hand in (0, layers * requests - 1, 17):
        reference = prepare_kvduo_hand(
            counts, steps, host, hand=hand, minimum=1, rho=0.1
        )
        actual = prepare_kvduo_hand(
            device_counts, device_steps, workspace, hand=hand, minimum=1, rho=0.1
        )
        assert torch.equal(workspace.targets.cpu(), host.targets)
        expected = reference.next(0, needed)
        assert actual.next(0, needed) == expected
        if expected:
            cursor = expected[-1][0] + 1
            assert actual.next(cursor, needed) == reference.next(cursor, needed)
        assert addresses == [(t.data_ptr(), t.shape, t.stride()) for t in tensors]
        torch.testing.assert_close(
            device_counts.cpu(), counts, rtol=0, atol=0, equal_nan=True
        )
        assert torch.equal(device_steps.cpu(), steps)


def test_gpu_hand_keeps_zero_release_visits_before_stopping_at_sufficient_space(
    monkeypatch,
):
    directory = Path(__file__).parents[3] / "unit/managers"
    monkeypatch.syspath_prepend(str(directory))
    helpers = runpy.run_path(str(directory / "test_kvduo_hot_merge.py"))
    c, p = helpers["make_merge_coordinator"](
        {(layer, req): 3 for req in range(3) for layer in range(2)}, layers=2
    )
    c.device = "cuda"
    for name in (
        "req_device_buffer_tokens",
        "req_device_buffer_token_locs",
        "lru_slots",
        "kvduo_req_hot_capacity_gpu",
        "kvduo_capacity_ema",
        "kvduo_capacity_steps",
        "kvduo_hot_growth_events",
    ):
        setattr(c, name, getattr(c, name).cuda())
    backing = c.mem_pool_device.get_flat_key_buffer().cuda()
    c.mem_pool_device.get_flat_key_buffer = lambda: backing
    c.mem_pool_device.kv_buffer = [backing] * 2
    c.kvduo_capacity_steps[1, 2] = 20
    c._kvduo_hot_pressure_protected.add((0, 0, 0))
    c._kvduo_hand = 5
    assert c.reclaim_kvduo_hot_pages(3) == 4
    assert c._kvduo_hand == 3
    assert c._kvduo_hot_processed == {(1, 2): 1, (1, 0): 2, (0, 1): 3}
    assert int(c.kvduo_capacity_steps[1, 2]) == 20
    assert len(c._kvduo_req_layer_pages[1, 2]) == 3
    for layer, req in c._kvduo_req_layer_pages:
        size = int(c.kvduo_req_hot_capacity[layer, req])
        tokens = c.req_device_buffer_tokens[layer, req, :size]
        locs = c.req_device_buffer_token_locs[layer, req, :size].long()
        valid = tokens >= 0
        assert torch.equal(
            backing[locs[valid]],
            tokens[valid].long()[:, None] * 10 + torch.arange(4, device="cuda"),
        )
