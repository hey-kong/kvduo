from unittest.mock import MagicMock

import pytest
import torch

import sglang.kernels.ops.kvcache.kvduo_hot_merge as ops
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.sparsity.core.kvduo_hot_merge import free_kvduo_whole_pages
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_hot_merge import make_merge_coordinator

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("status", [0, 1, 2, "cuda_failure"])
def test_gpu_merge_has_one_completion_wait_and_never_releases_failed_work(
    monkeypatch, status
):
    c, physical = make_merge_coordinator({(0, 0): 4})
    event = MagicMock()
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: "stream")
    workspace = ops.allocate_kvduo_merge_workspace(
        c.req_device_buffer_tokens[0, 0], c.lru_slots[0, 0]
    )
    c._kvduo_merge_workspace = workspace
    module = MagicMock()
    monkeypatch.setattr(ops, "_jit_kvduo_merge_module", lambda *args: module)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(
        torch.Tensor, "cpu", lambda *args: pytest.fail("unexpected GPU planning read")
    )
    c._move_kvduo_hot_entries = MagicMock(side_effect=AssertionError("legacy move"))
    c._reset_kvduo_capacity_stats = MagicMock(side_effect=AssertionError("extra reset"))
    order = []

    def execute(*args):
        order.append("execute")
        workspace.summary[:] = torch.tensor(
            [status if isinstance(status, int) else 0, 0, 3], dtype=torch.int32
        )

    def wait():
        order.append("complete")
        if status == "cuda_failure":
            raise RuntimeError("device execution failed")

    def release(page_ids):
        assert order == ["execute", "complete"]
        assert c._kvduo_req_layer_pages[0, 0] == [2, 4]
        order.append("free")

    module.execute_linear.side_effect = execute
    event.synchronize.side_effect = wait
    physical._release_page_ids.side_effect = release
    original = list(c._kvduo_req_layer_pages[0, 0])
    if status:
        with pytest.raises((ValueError, RuntimeError)):
            c._merge_kvduo_hot_pages(0, 0, 2)
        assert c._kvduo_req_layer_pages[0, 0] == original
        assert c.kvduo_req_hot_capacity[0, 0] == 16
        physical._release_page_ids.assert_not_called()
    else:
        assert c._merge_kvduo_hot_pages(0, 0, 2) == 4
        assert order == ["execute", "complete", "free"]
        assert c.kvduo_req_hot_capacity[0, 0] == 8
    module.execute_linear.assert_called_once()
    assert event.record.call_count == (1 if status else 2)
    event.record.assert_called_with("stream")
    event.synchronize.assert_called_once()
    c._move_kvduo_hot_entries.assert_not_called()
    c._reset_kvduo_capacity_stats.assert_not_called()
    c.token_to_kv_pool_allocator.free_hisparse_indices.assert_not_called()


@pytest.mark.parametrize("need_sort", [False, True])
def test_known_whole_page_return_avoids_unique_and_reuses_real_allocator(
    monkeypatch, need_sort
):
    allocator = PagedTokenToKVPoolAllocator(
        16, 2, torch.float32, "cpu", None, need_sort
    )
    allocated = allocator.alloc(16)
    assert allocated is not None and allocator.available_size() == 0
    allocator.free_group_begin()
    monkeypatch.setattr(torch, "unique", lambda *args: pytest.fail("unexpected unique"))
    monkeypatch.setattr(allocator, "free", lambda *args: pytest.fail("legacy free"))
    free_kvduo_whole_pages(allocator, [2, 6, 10], 2)
    assert allocator.available_size() == 6
    assert allocator.is_not_in_free_group and not allocator.free_group
    allocator.merge_and_sort_free()
    assert set(allocator.alloc(6).tolist()) == {2, 3, 6, 7, 10, 11}
    assert allocator.available_size() == 0


def test_workspace_reuse_orders_previous_page_upload_on_the_next_stream(monkeypatch):
    c, _ = make_merge_coordinator({(0, 0): 4})
    event, stream = MagicMock(), MagicMock()
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    workspace = ops.allocate_kvduo_merge_workspace(
        c.req_device_buffer_tokens[0, 0], c.lru_slots[0, 0]
    )
    workspace.free_pending = True
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    module = MagicMock()
    monkeypatch.setattr(ops, "_jit_kvduo_merge_module", lambda *args: module)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    owner = (0, 0)
    ops.execute_kvduo_hot_merge(
        c.req_device_buffer_tokens[owner],
        c.req_device_buffer_token_locs[owner],
        c.lru_slots[owner],
        c.mem_pool_device.get_flat_key_buffer(),
        workspace,
        capacity=16,
        k=4,
        top_k=2,
        target_pages=2,
        paged=False,
        published_capacity=c.kvduo_req_hot_capacity_gpu[owner],
        capacity_ema=c.kvduo_capacity_ema[owner],
        capacity_steps=c.kvduo_capacity_steps[owner],
        growth_events=c.kvduo_hot_growth_events[owner],
    )
    stream.wait_event.assert_called_once_with(event)
    event.record.assert_called_once_with(stream)
    event.synchronize.assert_not_called()
    assert not workspace.free_pending


@pytest.mark.parametrize("starts", [[0], [3], [2, 2], [-2]])
def test_invalid_whole_page_return_never_mutates_allocator(starts):
    allocator = MagicMock(page_size=2)
    with pytest.raises(ValueError):
        free_kvduo_whole_pages(allocator, starts, 2)
    allocator._release_page_ids.assert_not_called()
