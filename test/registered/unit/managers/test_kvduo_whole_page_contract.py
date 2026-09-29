"""Dependency-free regressions for KVDuo's whole-page integration contract."""

import ast
from pathlib import Path


ROOT = Path(__file__).parents[4]
COORDINATOR = ROOT / "python/sglang/srt/managers/hisparse_coordinator.py"
BACKEND = ROOT / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"


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
    source = COORDINATOR.read_text()
    owners = {(3, 17): [64], (17, 3): [128]}
    assert owners.pop((3, 17)) == [64]
    assert owners == {(17, 3): [128]}

    assert "owner = (req_idx, layer_id)" not in source
    assert "owner = (layer_id, req_idx)" in source
    assert "for (layer_id, req_idx), pages" in source
    assert "get((layer_id, req_idx))" in source


def test_tail_and_non_tail_2k_to_1k_shrink_use_whole_page_stride():
    evict = _method_source(
        COORDINATOR, "HiSparseCoordinator", "_evict_kvduo_hot_fragment"
    )
    assert "page_index * self.hot_page_size" in evict
    assert "last_index * self.hot_page_size" in evict
    assert evict.count("slot + self.hot_page_size") >= 3
    assert "new_capacity = last_slot" in evict
    assert "victim_start + self.page_size" in evict

    k = 8
    # Evict page zero: the surviving page's K locations move into its slot.
    pages = [100, 200]
    locs = list(range(2 * k))
    pages[0], locs[:k] = pages[-1], locs[k:]
    del pages[-1]
    del locs[k:]
    assert pages == [200]
    assert locs == list(range(k, 2 * k))

    # Evict the tail: the first page and all of its K locations remain.
    pages = [100, 200]
    locs = list(range(2 * k))
    del pages[-1]
    del locs[k:]
    assert pages == [100]
    assert locs == list(range(k))

    # A freed physical page has no owner and can be assigned to another pair.
    page_owners = {100: (3, 17), 200: (3, 17)}
    assert page_owners.pop(100) == (3, 17)
    page_owners[100] = (5, 29)
    assert page_owners == {100: (5, 29), 200: (3, 17)}


def test_batched_addresses_match_the_per_slot_reference_for_many_requests():
    storage_layers, page_size, stride = 3, 4, 64
    starts = [8, 20, 32]
    reference = [
        storage_layer * stride + start + offset
        for start in starts
        for storage_layer in range(storage_layers)
        for offset in range(page_size)
    ]
    broadcast_shape_order = [
        start + storage_offset + offset
        for start in starts
        for storage_offset in range(0, storage_layers * stride, stride)
        for offset in range(page_size)
    ]
    assert broadcast_shape_order == reference
    # Request 17 receives two pages, request 29 the next page, in allocator order.
    k = storage_layers * page_size
    assert broadcast_shape_order[: 2 * k] == reference[: 2 * k]
    assert broadcast_shape_order[2 * k :] == reference[2 * k :]

    growth = _method_source(
        COORDINATOR, "HiSparseCoordinator", "_materialize_kvduo_hot_growth_batch"
    )
    assert "int(page[0])" not in growth
    assert 'to(device="cpu").tolist()' in growth
    assert "page_starts_device[:, None, None]" in growth
    assert "storage_offsets[None, :, None]" in growth
    assert "within_page[None, None, :]" in growth
    assert "next_page" in growth


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
    assert batch.count('to(device="cpu").tolist()') == 1
    assert "for layer_id, req_idx, current, target, grow in requests" in batch


def test_cuda_streams_are_bound_to_the_tensor_parallel_device():
    source = COORDINATOR.read_text()
    init = _method_source(COORDINATOR, "HiSparseCoordinator", "__init__")
    staging = _method_source(
        COORDINATOR, "HiSparseCoordinator", "admit_request_into_staging"
    )
    stats = _method_source(
        COORDINATOR, "HiSparseCoordinator", "_schedule_kvduo_stats_snapshot"
    )

    # CUDA's current device is thread-local.  A scheduler worker starts on
    # logical device zero, so implicit stream selection would make every TP
    # process establish an otherwise unused context on TP0's GPU.
    assert "device = req_to_token_pool.req_to_token.device" in init
    assert "device_module.Stream()" not in init
    assert init.count("device_module.Stream(device=device)") == 3
    assert "device_module.current_stream()" not in source
    assert "device_module.current_stream(self.device)" in source
    assert "start_event.record(schedule_stream)" in staging
    assert "finish_event.record(self.write_staging_stream)" in staging
    assert "_kvduo_stats_snapshot_ready_event.record(schedule_stream)" in stats
    assert "_kvduo_stats_event.record(self._kvduo_stats_stream)" in stats
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
        if isinstance(node, ast.FunctionDef)
        and node.name == "init_kvduo_load_back"
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
