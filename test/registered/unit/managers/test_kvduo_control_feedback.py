import runpy
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.kernels.ops.kvcache.kvduo_control import pack_kvduo_feedback
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def fixture():
    return runpy.run_path(str(Path(__file__).with_name("test_kvduo_hot_growth.py")))[
        "make_coordinator"
    ]


def test_feedback_preserves_active_events_and_clears_all_source_statistics():
    events = torch.arange(21, dtype=torch.int32).reshape(3, 7)
    original = events.clone()
    stats = torch.full((3, 7, 2), 17, dtype=torch.int32)
    positions = torch.tensor([-1, 2, -1, 0, -1, 1, -1], dtype=torch.int32)
    packet = torch.full((21,), -99, dtype=torch.int32)
    pack_kvduo_feedback(events, stats, positions, packet, 3)
    assert torch.equal(packet[:9].view(3, 3), original[:, [3, 5, 1]])
    assert torch.all(packet[9:] == -99)
    assert not events.any() and not stats.any()


def _tp_worker(rank, path):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        world_size=2,
        rank=rank,
        timeout=timedelta(seconds=20),
    )
    try:
        c, allocator = fixture()(layers=2, requests=70, free_pages=8 + rank * 3)
        c.tp_world_size = 2
        c.tp_group = dist.group.WORLD
        for scenario in ("same", "different", "protected", "all_bits", "empty"):
            applicants = {(0, 0): 1, (0, 63): 1, (1, 69): 1}
            protected = 2
            if scenario == "different":
                applicants = {(0, rank): 1}
            if scenario == "protected":
                protected = 2 + rank
            if scenario == "all_bits":
                applicants = {(0, i): 1 for i in range(63)}
            if scenario == "empty":
                applicants = {}
            with patch.object(
                dist, "all_reduce", wraps=dist.all_reduce
            ) as reduce, patch.object(
                dist, "all_gather_object", wraps=dist.all_gather_object
            ) as gather:
                free, consistent = c._kvduo_optional_growth_resources(
                    protected, applicants
                )
            assert free == 8
            assert consistent == (scenario not in ("different", "protected"))
            assert reduce.call_count == 1 and gather.call_count == 0
    finally:
        dist.destroy_process_group()


def test_two_tp_ranks_check_exact_applicants_and_free_pages_once(tmp_path):
    torch.multiprocessing.start_processes(
        _tp_worker,
        args=(str(tmp_path / "gloo-init"),),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
