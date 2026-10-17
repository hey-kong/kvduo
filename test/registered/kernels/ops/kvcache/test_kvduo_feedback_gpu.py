import runpy
from pathlib import Path

import pytest
import torch
from sglang.kernels.ops.kvcache.kvduo_control import pack_kvduo_feedback
from sglang.srt.utils import is_cuda, is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=6, stage="stage-b", runner_config="1-gpu-small-amd")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not (is_cuda() or is_hip()),
    reason="Feedback packing requires CUDA/ROCm",
)


@pytest.mark.parametrize("active", [0, 1, 3, 70])
def test_compact_feedback_reads_each_event_before_resetting_its_source(active):
    events = torch.arange(5 * 70, device="cuda", dtype=torch.int32).view(5, 70)
    saved = events.clone()
    stats = torch.full((5, 70, 2), 99, device="cuda", dtype=torch.int32)
    requests = torch.arange(active).flip(0).cuda()
    positions = torch.full((70,), -1, device="cuda", dtype=torch.int32)
    positions[requests] = torch.arange(active, device="cuda", dtype=torch.int32)
    packet = torch.full((5 * 70,), -99, device="cuda", dtype=torch.int32)
    pack_kvduo_feedback(events, stats, positions, packet, active)
    torch.cuda.synchronize()
    assert torch.equal(packet[: 5 * active].view(5, active), saved[:, requests])
    assert torch.all(packet[5 * active :] == -99)
    assert not events.any() and not stats.any()


def test_coordinator_feedback_rebinds_slots_and_rejects_old_generation(monkeypatch):
    from sglang.srt.managers import hisparse_coordinator as module

    make = runpy.run_path(
        str(Path(__file__).parents[3] / "unit/managers/test_kvduo_hot_growth.py")
    )["make_coordinator"]
    c, _ = make(layers=2, requests=4)
    c.kvduo_hot_growth_events = c.kvduo_hot_growth_events.cuda()
    c.kvduo_resolver_stats = c.kvduo_resolver_stats.cuda()
    c._active_kvduo_reqs = {1: object(), 3: object()}
    c._mixed_slots = [True] * 4
    c.kvduo_hot_growth_events[0, 0] = 1  # Inactive old request is reset, not forwarded.
    c.kvduo_hot_growth_events[0, 1] = 1
    c.kvduo_hot_growth_events[1, 3] = 2
    assert c._consume_kvduo_hot_growth_events() == {(0, 1): 1, (1, 3): 2}
    addresses = (
        c._kvduo_feedback_positions.data_ptr(),
        c._kvduo_feedback_packet.data_ptr(),
    )
    c._kvduo_request_generation[1] += 1
    c.kvduo_hot_growth_events[1, 1] = 1
    assert c._consume_kvduo_hot_growth_events() == {(1, 1): 1}
    pack = module.pack_kvduo_feedback

    def change_generation(*args):
        pack(*args)
        c._kvduo_request_generation[1] += 1

    monkeypatch.setattr(module, "pack_kvduo_feedback", change_generation)
    c.kvduo_hot_growth_events[0, 1] = 1
    assert c._consume_kvduo_hot_growth_events() == {}
    assert addresses == (
        c._kvduo_feedback_positions.data_ptr(),
        c._kvduo_feedback_packet.data_ptr(),
    )


def test_publication_graph_replay_preserves_addresses_and_unmodified_owners():
    from sglang.kernels.ops.kvcache.kvduo_control import publish_kvduo_buffers

    pointers = torch.full((3, 5, 4), 99, dtype=torch.int64, device="cuda")
    capacities = torch.full((3, 5), 99, dtype=torch.int32, device="cuda")
    packet = torch.tensor([[1, 64, 10, 11, 12, 13], [14, 0, 0, 0, 0, 0]], device="cuda")
    publish_kvduo_buffers(packet, pointers, capacities)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        publish_kvduo_buffers(packet, pointers, capacities)
    addresses = pointers.data_ptr(), capacities.data_ptr()
    for capacity in (128, 256, 0):
        packet[0, 1] = capacity
        graph.replay()
        torch.cuda.synchronize()
        assert pointers[0, 1].tolist() == [10, 11, 12, 13]
        assert capacities[0, 1].item() == capacity
        assert not pointers[2, 4].any()
        assert capacities[2, 4] == 0
        assert (pointers.view(-1, 4)[[0, 2, 7, 13]] == 99).all()
        assert addresses == (pointers.data_ptr(), capacities.data_ptr())
