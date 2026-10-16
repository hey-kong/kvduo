"""Dependency-free regressions for KVDuo's whole-page integration contract."""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[4]
COORDINATOR = ROOT / "python/sglang/srt/managers/hisparse_coordinator.py"
BACKEND = ROOT / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"
SCHEDULER = ROOT / "python/sglang/srt/managers/scheduler.py"


def _method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    klass = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in klass.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    return ast.get_source_segment(source, method)


def test_attention_selects_kv_tensor_in_the_index_address_space():
    forward = _method_source(BACKEND, "DeepseekV4AttnBackend", "forward")
    assert "forward_batch.forward_mode.is_decode()" in forward
    assert "c4_kv_pool.get_flat_key_buffer()" in forward

    sparse_prefill = _method_source(
        BACKEND, "DeepseekV4AttnBackend", "_forward_prefill_sparse"
    )
    assert "get_extra_key_buffer(layer_id)" in sparse_prefill
    assert "get_flat_key_buffer" not in sparse_prefill


def test_owner_keys_keep_asymmetric_requests_and_layers_separate():
    from test_kvduo_hot_merge import make_merge_coordinator

    c, _ = make_merge_coordinator({(3, 17): 3, (17, 3): 3}, layers=18)
    untouched = c._kvduo_req_layer_pages[17, 3].copy()
    c.kvduo_hot_buffers[3, 17].frequency.zero_()
    assert c._merge_kvduo_hot_pages(3, 17) == 2 * c.page_size
    assert len(c._kvduo_req_layer_pages[3, 17]) == 1
    assert c._kvduo_req_layer_pages[17, 3] == untouched
    assert all(c._kvduo_hot_page_owners[page] == (17, 3) for page in untouched)


def test_merge_returns_whole_allocator_pages_after_payload_copy_completion():
    from unittest.mock import MagicMock

    import torch
    from test_kvduo_hot_merge import (
        assert_payloads_match_tokens,
        make_merge_coordinator,
    )

    c, physical = make_merge_coordinator({(0, 0): 4})
    survivors = [0, 1, 4, 5, 8, 9, 12, 13]
    c.lru_slots[0, 0, :16] = torch.tensor(
        [slot for slot in range(16) if slot not in survivors] + survivors,
        dtype=torch.int16,
    )
    free = physical._release_page_ids

    def free_after_copy(page_ids):
        assert_payloads_match_tokens(c, 0, 0)
        assert c._kvduo_req_layer_pages[0, 0] == [2, 4]
        assert page_ids.tolist() == [3, 4]
        free(page_ids)

    physical._release_page_ids = MagicMock(side_effect=free_after_copy)
    assert c._merge_kvduo_hot_pages(0, 0) == 2 * c.page_size
    physical._release_page_ids.assert_called_once()
    assert physical.free_pages == 2


def test_batched_addresses_match_the_per_slot_reference_for_many_requests():
    from test_kvduo_hot_growth import make_coordinator

    c, physical = make_coordinator(layers=3, requests=30)
    c.hot_storage_layers = 3
    c.hot_page_size = 6
    c._ensure_kvduo_hot_capacity_targets_batch({(2, 17): 12, (1, 29): 6})
    physical.alloc.assert_called_once_with(6)
    for owner, capacity in [((2, 17), 12), ((1, 29), 6)]:
        pages = c._kvduo_req_layer_pages[owner]
        expected = [
            p + layer * c.mem_pool_device.flat_layer_slot_stride + offset
            for p in pages
            for layer in range(3)
            for offset in range(2)
        ]
        assert c.kvduo_hot_buffers[owner].locations.tolist() == expected
        assert c.kvduo_hot_buffers[owner].tokens.numel() == capacity


def test_graph_prepare_batches_a_whole_capacity_round():
    prepare = _method_source(
        COORDINATOR, "HiSparseCoordinator", "prepare_kvduo_graph_replay"
    )
    assert "_ensure_kvduo_hot_capacity_targets_batch(targets)" in prepare
    assert "(layer_id, req_idx): self.min_hot_pages" in prepare

    batch = _method_source(
        COORDINATOR,
        "HiSparseCoordinator",
        "_materialize_kvduo_hot_growth_batch",
    )
    # One allocation and one page-number D2H read serve all request/layer pairs.
    assert batch.count("allocator.alloc(physical_slots)") == 1
    assert batch.count("cpu().tolist()") == 1
    assert "for layer, req, current, target, grow in requests" in batch


def test_cuda_streams_are_bound_to_the_tensor_parallel_device():
    source = COORDINATOR.read_text()
    init = _method_source(COORDINATOR, "HiSparseCoordinator", "__init__")
    staging = _method_source(
        COORDINATOR, "HiSparseCoordinator", "admit_request_into_staging"
    )

    # CUDA's current device is thread-local.  A scheduler worker starts on
    # logical device zero, so implicit stream selection would make every TP
    # process establish an otherwise unused context on TP0's GPU.
    assert "device = req_to_token_pool.req_to_token.device" in init
    assert "device_module.Stream()" not in init
    assert init.count("device_module.Stream(device=device)") == 2
    assert "device_module.current_stream()" not in source
    assert "device_module.current_stream(self.device)" in source
    assert "start_event.record(schedule_stream)" in staging
    assert "finish_event.record(self.write_staging_stream)" in staging
    assert ".record()" not in source


def test_kvduo_cpu_group_never_reduces_cuda_control_tensors():
    source = COORDINATOR.read_text()
    tree = ast.parse(source)
    klass = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HiSparseCoordinator"
    )
    restore = next(
        node
        for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name == "init_kvduo_load_back"
    )

    ready_tensors = []
    for node in ast.walk(restore):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "ready"
            for target in node.targets
        ):
            continue
        assert isinstance(node.value, ast.Call)
        ready_tensors.append(node.value)

    assert len(ready_tensors) == 3
    assert all(
        not any(keyword.arg == "device" for keyword in call.keywords)
        for call in ready_tensors
    )
