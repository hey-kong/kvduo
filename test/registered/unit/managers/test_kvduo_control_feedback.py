import runpy
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from sglang.kernels.ops.kvcache.kvduo_control import pack_kvduo_feedback
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


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
        packet_address = None
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
            address = c._kvduo_growth_resource_packet.data_ptr()
            assert packet_address is None or packet_address == address
            packet_address = address
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


def _replay_tp_worker(rank, path):
    c, allocator = fixture()(layers=2, requests=2, free_pages=32)
    requests = torch.tensor([0, 1])
    # Start with identical, already materialized mandatory HOT capacities.
    c.prepare_kvduo_graph_replay(requests)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        world_size=2,
        rank=rank,
        timeout=timedelta(seconds=10),
    )
    try:
        c.tp_world_size = 2
        c.tp_group = dist.group.WORLD
        for scenario in ("rank0_only", "rank1_only", "empty", "matching"):
            before = c.kvduo_req_hot_capacity.clone()
            if scenario == "matching" or scenario == f"rank{rank}_only":
                c.kvduo_hot_growth_events[0, 0] = 1
            with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce:
                c.prepare_kvduo_graph_replay(requests)
            # The next collective must match on both ranks even if one had no
            # local event. A skipped replay collective deadlocks here.
            dist.barrier()
            assert reduce.call_count == 1
            expected = before.clone()
            if scenario == "matching":
                expected[0, 0] += c.hot_page_size
            assert torch.equal(c.kvduo_req_hot_capacity, expected)
            assert not c.kvduo_hot_growth_events.any()
    finally:
        dist.destroy_process_group()


def test_replay_no_local_growth_still_participates_in_tp_consensus(tmp_path):
    torch.multiprocessing.start_processes(
        _replay_tp_worker,
        args=(str(tmp_path / "gloo-replay"),),
        nprocs=2,
        join=True,
        start_method="spawn",
    )


def _prefill_tp_worker(rank, path):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        world_size=2,
        rank=rank,
        timeout=timedelta(seconds=20),
    )
    try:
        c, allocator = fixture()(layers=1, requests=1, free_pages=8 + rank)
        c.tp_world_size = 2
        c.tp_group = dist.group.WORLD
        c._kvduo_mandatory_hot_pages = lambda: 0
        c._kvduo_next_decode_pages = lambda: 0
        c._reclaim_kvduo_radix_pages = lambda *args: None
        for scenario, need in (("fit", 8), ("short", 9), ("different", 2 + rank)):
            c._kvduo_prefill_reservations.clear()
            with patch.object(
                dist, "all_reduce", wraps=dist.all_reduce
            ) as reduce, patch.object(
                dist, "all_gather_object", wraps=dist.all_gather_object
            ) as gather:
                token = c._reserve_kvduo_prefill_physical(need)
            assert (token is not None) == (scenario == "fit")
            assert reduce.call_count == 1 and gather.call_count == 0
            if token is None:
                assert not c._kvduo_prefill_reservations
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_prefill_agrees_on_required_and_common_capacity_in_one_collective(tmp_path):
    torch.multiprocessing.start_processes(
        _prefill_tp_worker,
        args=(str(tmp_path / "gloo-prefill"),),
        nprocs=2,
        join=True,
        start_method="spawn",
    )


def test_buffer_publication_updates_only_selected_owners():
    from sglang.kernels.ops.kvcache.kvduo_control import publish_kvduo_buffers

    pointers = torch.full((2, 3, 4), 99, dtype=torch.int64)
    capacities = torch.full((2, 3), 99, dtype=torch.int32)
    packet = torch.tensor([[1, 64, 10, 11, 12, 13], [5, 0, 0, 0, 0, 0]])
    publish_kvduo_buffers(packet, pointers, capacities)
    assert pointers[0, 1].tolist() == [10, 11, 12, 13]
    assert capacities[0, 1] == 64 and capacities[1, 2] == 0
    assert pointers[1, 2].tolist() == [0] * 4
    assert (pointers.view(-1, 4)[[0, 2, 3, 4]] == 99).all()
