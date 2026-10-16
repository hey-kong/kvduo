"""CUDA allocation/release and Graph replay across history capacity changes.

pytest -q test/registered/kernels/ops/memory/test_kvduo_elastic_history_cuda.py
torchrun --standalone --nproc-per-node=4 <this file>
"""

import gc
import os

import pytest
import torch
import torch.distributed as dist
from sglang.srt.mem_cache.kvduo_elastic_history import KVDuoElasticHistory
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu")


class Allocator:
    page_size = 256

    def __init__(self, pages, device):
        self.free_pages = torch.arange(1, pages + 1, dtype=torch.int64, device=device)

    def merge_and_sort_free(self):
        self.free_pages = self.free_pages.sort().values

    def available_size(self):
        return self.free_pages.numel() * self.page_size


def exercise_history(device_id, group=None):
    def agree(value):
        if group is None:
            return True
        values = [None] * dist.get_world_size(group)
        dist.all_gather_object(values, value, group=group)
        return all(v == values[0] for v in values)

    def common(value):
        if group is None:
            return value
        t = torch.tensor(value, dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
        return int(t.item())

    with torch.cuda.device(device_id):
        layout = KVDuoElasticHistory(
            device=f"cuda:{device_id}",
            initial_pages=512,
            maximum_pages=1024,
            chunk_mb=4,
            reserve_bytes=64 << 20,
        )
        graph = None
        try:
            indexer = layout.allocate(rows=1025, layers=2, row_bytes=8448)
            history = layout.allocate(rows=1025, layers=1, row_bytes=1728)
            full = torch.full((1 << 20,), 19, dtype=torch.uint8, device=device_id)
            layout.initialize()
            allocator = Allocator(1024, device_id)
            layout.bind(allocator)
            allocator.free_pages = allocator.free_pages[:2]
            initial_bytes = layout.backing.allocated_bytes
            pointers = (indexer.data_ptr(), history.data_ptr())
            for cycle in range(3):
                assert (
                    layout.grow(
                        required_free_pages=100, agree=agree, common_available=common
                    )
                    > 0
                )
                assert layout.backing.allocated_bytes > initial_bytes
                indexer.view(2, 1025, -1)[:, 513].fill_(cycle + 7)
                history[513].fill_(cycle + 11)
                if graph is None:
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        output = (
                            indexer.view(2, 1025, -1)[:, 513].sum() + history[513].sum()
                        )
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = (
                            indexer.view(2, 1025, -1)[:, 513].sum() + history[513].sum()
                        )
                graph.replay()
                torch.cuda.synchronize()
                assert output.item() == 2 * 8448 * (cycle + 7) + 1728 * (cycle + 11)
                assert (indexer.data_ptr(), history.data_ptr()) == pointers
                assert torch.all(full == 19)
                allocator.free_pages = allocator.free_pages[allocator.free_pages != 513]
                assert layout.trim(agree=agree) == 0
                allocator.free_pages = torch.cat(
                    (allocator.free_pages, torch.tensor([513], device=device_id))
                )
                assert layout.trim(agree=agree) > 0
                assert layout.current_pages == 512
                assert layout.backing.allocated_bytes == initial_bytes
            # One rank without usable headroom prevents growth everywhere.
            native_free = layout._free_bytes
            layout._free_bytes = lambda: 0 if device_id == 0 else native_free()
            assert (
                layout.grow(
                    required_free_pages=100, agree=agree, common_available=common
                )
                == 0
            )
        finally:
            graph = None
            gc.collect()
            layout.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_elastic_history_graph_replay():
    exercise_history(0)


if __name__ == "__main__":
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    try:
        exercise_history(rank, dist.group.WORLD)
    finally:
        dist.destroy_process_group()
