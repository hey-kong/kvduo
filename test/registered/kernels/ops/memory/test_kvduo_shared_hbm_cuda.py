"""CUDA VMM borrow/return and capture-stability gate for KVDuo shared backing.

Single GPU: pytest -q test/registered/kernels/ops/memory/test_kvduo_shared_hbm_cuda.py
TP consensus: torchrun --standalone --nproc-per-node=4 <this file>
"""

import gc
import os

import pytest
import torch
from sglang.srt.mem_cache.kv_vmm_backing import KvVmmArena, query_granularity
from sglang.srt.mem_cache.kvduo_shared_hbm import CudaMappingBackend, SharedHBMBlocks
from sglang.srt.mem_cache.kvduo_shared_layout import KVDuoSharedLayout
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu")


def exercise_shared_backing(device_id, *, group=None):
    """No model or serving allocator state is touched by this storage test."""
    import torch.distributed as dist

    with torch.cuda.device(device_id):
        granularity = query_granularity(device_id)
        arena = KvVmmArena(device_id, reserve_bytes=4 * (granularity + (32 << 20)))
        backing = SharedHBMBlocks(
            CudaMappingBackend(device_id),
            block_bytes=granularity,
            budget_bytes=2 * granularity,
        )
        graph = None
        try:
            with torch.cuda.use_mem_pool(arena.pool):
                full, full_other, history = [
                    torch.empty(granularity, dtype=torch.uint8, device=device_id)
                    for _ in range(3)
                ]
            initial_pointers = tuple(x.data_ptr() for x in (full, full_other, history))
            backing.prepare(
                backing.plan(destinations=initial_pointers[:2]), accesses_complete=True
            ).commit()
            full.fill_(7)
            full_other.fill_(9)
            torch.cuda.synchronize(device_id)

            # Simulate a rank rejecting admission after others prepared aliases.
            # All ranks abort, preserving the original FULL data and mappings.
            plan = backing.plan(
                donors=(full.data_ptr(),), destinations=(history.data_ptr(),)
            )
            transaction = backing.prepare(plan, accesses_complete=True)
            if group is not None:
                ok = torch.tensor([int(dist.get_rank() != 0)], dtype=torch.int32)
                dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
                assert ok.item() == 0
            transaction.abort()
            assert torch.all(full == 7).item()

            backing.prepare(plan, accesses_complete=True).commit()
            history.fill_(11)
            output = torch.empty_like(history)
            torch.cuda.synchronize(device_id)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                torch.add(history, 1, out=output)
            graph.replay()
            torch.cuda.synchronize(device_id)
            assert torch.all(output == 12).item()

            for value in (13, 17, 21):
                # Complete old graph accesses before returning history backing.
                done = torch.cuda.Event()
                done.record()
                done.synchronize()
                backing.prepare(
                    backing.plan(
                        donors=(history.data_ptr(),), destinations=(full.data_ptr(),)
                    ),
                    accesses_complete=done.query(),
                ).commit()
                full.fill_(value)
                done.record()
                done.synchronize()
                transaction = backing.prepare(
                    backing.plan(
                        donors=(full.data_ptr(),), destinations=(history.data_ptr(),)
                    ),
                    accesses_complete=done.query(),
                )
                if group is not None:
                    ok = torch.ones(1, dtype=torch.int32)
                    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
                    assert ok.item() == 1
                transaction.commit()
                assert (
                    tuple(x.data_ptr() for x in (full, full_other, history))
                    == initial_pointers
                )
                graph.replay()
                torch.cuda.synchronize(device_id)
                assert torch.all(output == value + 1).item()
                assert torch.all(full_other == 9).item()
                assert backing.allocated_bytes == 2 * granularity
        finally:
            torch.cuda.synchronize(device_id)
            del graph
            gc.collect()
            backing.close(accesses_complete=True)
            arena.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA VMM requires a GPU")
def test_shared_backing_graph_borrow_return():
    exercise_shared_backing(torch.cuda.current_device())


def exercise_model_layout(device_id, *, group=None):
    import torch.distributed as dist

    class Allocator:
        page_size = 256

        def __init__(self, pages):
            self.free_pages = torch.tensor(pages, dtype=torch.int64, device=device_id)

        def merge_and_sort_free(self):
            self.free_pages = self.free_pages.sort().values

        def available_size(self):
            return self.free_pages.numel() * self.page_size

    def agree(value):
        if group is None:
            return True
        values = [None] * dist.get_world_size(group)
        dist.all_gather_object(values, value, group=group)
        return all(item == values[0] for item in values)

    def common(value):
        if group is None:
            return value
        tensor = torch.tensor([value], dtype=torch.int64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=group)
        return tensor.item()

    with torch.cuda.device(device_id):
        layout = KVDuoSharedLayout(
            device=f"cuda:{device_id}",
            initial_pages=512,
            maximum_pages=1024,
            chunk_mb=4,
        )
        graph = None
        try:
            full = layout.allocate(rows=513, layers=2, row_bytes=37440, kind="full")
            indexer = layout.allocate(
                rows=1025, layers=2, row_bytes=8448, kind="history"
            )
            c128 = layout.allocate(rows=1025, layers=1, row_bytes=1728, kind="history")
            layout.initialize()
            logical, physical = (
                Allocator([511, 512]),
                Allocator(range(8 + device_id, 513)),
            )
            layout.bind(logical, physical)
            pointer = indexer.data_ptr()
            initial_full = set(physical.free_pages.cpu().tolist())
            full.view(2, 513, -1)[:, :8].fill_(29)
            borrowed = layout.grow(
                required_free_pages=32,
                protected_full_pages=32,
                agree=agree,
                common_available=common,
            )
            assert borrowed > 0
            loan = layout.loans[-1]
            sample = indexer.view(2, 1025, -1)[:, loan.start]
            sample.fill_(31)
            c128[loan.start].fill_(37)
            output = torch.empty_like(sample)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                torch.add(sample, 1, out=output)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.all(output == 32).item()
            assert torch.all(c128[loan.start] == 37).item()
            assert torch.all(full.view(2, 513, -1)[:, :8] == 29).item()
            # Mark history live: no return is allowed until its page is freed.
            logical.free_pages = logical.free_pages[logical.free_pages != loan.start]
            assert layout.trim(agree=agree) == 0
            logical.free_pages = torch.cat(
                (logical.free_pages, torch.tensor([loan.start], device=device_id))
            )
            assert layout.trim(agree=agree) == borrowed
            assert set(physical.free_pages.cpu().tolist()) == initial_full
            assert (
                layout.grow(
                    required_free_pages=32,
                    protected_full_pages=32,
                    agree=agree,
                    common_available=common,
                )
                > 0
            )
            assert indexer.data_ptr() == pointer
            sample.fill_(41)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.all(output == 42).item()
        finally:
            torch.cuda.synchronize()
            del graph
            gc.collect()
            layout.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA VMM requires a GPU")
def test_model_layout_borrow_and_graph_replay():
    exercise_model_layout(torch.cuda.current_device())


if __name__ == "__main__":
    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    try:
        exercise_shared_backing(rank, group=dist.group.WORLD)
        exercise_model_layout(rank, group=dist.group.WORLD)
        if dist.get_rank() == 0:
            print(
                "KVDuo shared HBM: TP prepare/abort, borrow/return and Graph replay passed"
            )
    finally:
        dist.destroy_process_group()
