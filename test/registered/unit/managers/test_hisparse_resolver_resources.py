"""Resolver resource planning uses the production C++ layout and device limits."""

import ctypes
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.kernels.ops.kvcache import hisparse as ops
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_fused_validation_has_a_distinct_jit_cache_key(monkeypatch):
    load = MagicMock()
    monkeypatch.setattr(ops, "load_jit", load)
    for validate in (False, True):
        ops._jit_sparse_module.__wrapped__(
            576, 256, 512, 1536, True, False, False, False, validate
        )
    calls = load.call_args_list
    assert calls[0].args != calls[1].args
    assert calls[0].kwargs["cuda_wrappers"] != calls[1].kwargs["cuda_wrappers"]


def test_full_touch_aggregation_has_a_distinct_jit_cache_key(monkeypatch):
    load = MagicMock()
    monkeypatch.setattr(ops, "load_jit", load)
    for size in (1, 64):
        ops._jit_sparse_module.__wrapped__(
            576, 256, 512, 1536, True, False, False, True, True, size
        )
    calls = load.call_args_list
    assert calls[0].args != calls[1].args
    assert calls[0].kwargs["cuda_wrappers"] != calls[1].kwargs["cuda_wrappers"]


@pytest.mark.parametrize(
    "loader",
    [ops.load_cache_to_device_buffer_mla, ops.load_cache_to_device_buffer_dsv4_mla],
)
@pytest.mark.parametrize("validate", [False, True])
def test_resolver_wrappers_select_validation_without_extra_tensor_operations(
    monkeypatch, loader, validate
):
    module = SimpleNamespace(load_cache_to_device_buffer=MagicMock())
    compile_module = MagicMock(return_value=module)
    monkeypatch.setattr(ops, "_jit_sparse_module", compile_module)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=123)
    )
    versions = torch.ones(8, dtype=torch.int64)
    clock = torch.zeros(1, dtype=torch.int64)
    status = torch.zeros(1, dtype=torch.int32)
    for operation in ("all", "_assert_async"):
        monkeypatch.setattr(
            torch, operation, MagicMock(side_effect=AssertionError("unfused check"))
        )
    loader(
        top_k_tokens=torch.zeros((1, 1), dtype=torch.int32),
        device_buffer_tokens=torch.full((1, 3), -1, dtype=torch.int32),
        host_cache_locs=torch.zeros((1, 8), dtype=torch.int64),
        device_buffer_locs=torch.ones((1, 3), dtype=torch.int32),
        host_cache=torch.zeros((8, 32)),
        device_buffer=torch.zeros((8, 32)),
        top_k_device_locs=torch.zeros((1, 1), dtype=torch.int32),
        req_pool_indices=torch.zeros(1, dtype=torch.int64),
        seq_lens=torch.ones(1, dtype=torch.int32),
        lru_slots=torch.tensor([[0, 1]], dtype=torch.int16),
        item_size_bytes=32,
        num_top_k=1,
        hot_buffer_size=2,
        num_real_reqs=torch.ones(1, dtype=torch.int32),
        req_to_logical_token=torch.arange(8, dtype=torch.int64).view(1, -1),
        full_to_device_loc=versions,
        full_last_touch=versions.clone(),
        full_data_version=versions,
        full_host_version=versions.clone(),
        swap_status=status,
        resolver_stats=torch.zeros((1, 2), dtype=torch.int32),
        touch_clock=clock,
        validate_status=validate,
    )
    assert compile_module.call_args.kwargs["validate_status"] == validate
    assert module.load_cache_to_device_buffer.call_count == 1
    assert not any(
        isinstance(arg, torch.Tensor)
        for arg in module.load_cache_to_device_buffer.call_args.args
    )


LAYOUTS = [
    (2048, 39936, 32),
    (2048, 39936, 64),
    (2048, 31232, 32),
    (4, 1536, 32),
    (2, 5, 32),
]


@pytest.fixture(scope="module")
def native_layouts(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required")
    directory = tmp_path_factory.mktemp("hisparse-layouts")
    source = directory / "layouts.cpp"
    source.write_text(
        "#include <sgl_kernel/hisparse_resolver_layout.h>\n"
        + "\n".join(
            f'extern "C" size_t layout_{i}() {{ return '
            f"sglang::HisparseResolverLayout<{topk},{capacity},{warp}>::BYTES; }}"
            for i, (topk, capacity, warp) in enumerate(LAYOUTS)
        )
        + "\n".join(
            f'extern "C" size_t compact_{i}() {{ return '
            f"sglang::HisparseResolverLayout<{topk},{capacity},{warp}>::COMPACT_BYTES; }}"
            for i, (topk, capacity, warp) in enumerate(LAYOUTS)
        )
    )
    library = directory / "layouts.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            f"-I{Path(__file__).parents[4] / 'python/sglang/kernels/jit/include'}",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    native = ctypes.CDLL(str(library))
    sizes = []
    for i in range(len(LAYOUTS)):
        function = getattr(native, f"layout_{i}")
        function.restype = ctypes.c_size_t
        sizes.append(function())
    compact_sizes = []
    for i in range(len(LAYOUTS)):
        function = getattr(native, f"compact_{i}")
        function.restype = ctypes.c_size_t
        compact_sizes.append(function())
    return sizes, compact_sizes


@pytest.mark.parametrize("case", range(len(LAYOUTS)))
def test_python_workspace_covers_actual_compiled_layout(native_layouts, case):
    topk, capacity, warp = LAYOUTS[case]
    size = native_layouts[0][case]
    words = ops.hisparse_resolver_scratch_words(topk, capacity, warp_size=warp)
    assert size <= words * 4 < size + 4
    assert native_layouts[0][0] == 210712


@pytest.mark.parametrize("case", range(len(LAYOUTS)))
def test_compact_workspace_matches_cpp_and_excludes_capacity_sized_lru(
    native_layouts, case
):
    topk, capacity, warp = LAYOUTS[case]
    size = native_layouts[1][case]
    words = ops.hisparse_resolver_scratch_words(
        topk, capacity, warp_size=warp, use_lru_scratch=True
    )
    assert size <= words * 4 < size + 4
    slot_bytes = 4 if capacity > 32767 else 2
    assert size == native_layouts[0][case] - capacity * slot_bytes
    assert native_layouts[1][0] == 50968


@pytest.mark.parametrize(
    "shared_limit,global_scratch", [(49152, True), (163840, True), (233472, False)]
)
def test_init_allocates_global_workspace_when_maximum_capacity_exceeds_gpu_limit(
    monkeypatch, shared_limit, global_scratch
):
    properties = SimpleNamespace(
        shared_memory_per_block=49152,
        shared_memory_per_block_optin=shared_limit,
        max_threads_per_block=1024,
    )
    monkeypatch.setattr(torch.version, "hip", None)
    get_properties = MagicMock(return_value=properties)
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_properties)
    workspace = torch.empty((3, 52678), dtype=torch.int32)
    allocate = MagicMock(return_value=workspace)
    monkeypatch.setattr(torch, "empty", allocate)

    result = ops.allocate_hisparse_resolver_scratch(
        3, 2048, 39936, 960, torch.device("cuda:2")
    )

    get_properties.assert_called_once_with(torch.device("cuda:2"))
    if global_scratch:
        assert result is workspace
        allocate.assert_called_once_with(
            (3, 52678), dtype=torch.int32, device=torch.device("cuda:2")
        )
    else:
        assert result is None
        allocate.assert_not_called()


@pytest.mark.parametrize("block_size", [0, 31, 1536])
def test_init_rejects_invalid_block_before_allocation(monkeypatch, block_size):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(max_threads_per_block=1024),
    )
    allocate = MagicMock()
    monkeypatch.setattr(torch, "empty", allocate)
    with pytest.raises(ValueError, match="block_size"):
        ops.allocate_hisparse_resolver_scratch(
            1, 2048, 39936, block_size, torch.device("cuda:1")
        )
    allocate.assert_not_called()


def test_small_cache_keeps_shared_path_and_rocm_large_cache_uses_global(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(shared_memory_per_block=65536),
    )
    monkeypatch.setattr(torch.version, "hip", "test")
    allocate = MagicMock()
    monkeypatch.setattr(torch, "empty", allocate)
    assert (
        ops.allocate_hisparse_resolver_scratch(2, 4, 1536, 256, torch.device("cuda:0"))
        is None
    )
    ops.allocate_hisparse_resolver_scratch(2, 2048, 39936, 960, torch.device("cuda:0"))
    allocate.assert_called_once_with(
        (2, 51430), dtype=torch.int32, device=torch.device("cuda:0")
    )


@pytest.mark.parametrize(
    "loader",
    [ops.load_cache_to_device_buffer_mla, ops.load_cache_to_device_buffer_dsv4_mla],
)
@pytest.mark.parametrize("global_scratch", [False, True])
def test_wrapper_passes_persistent_workspace_address_and_stride_without_tensor_ffi(
    monkeypatch, loader, global_scratch
):
    module = SimpleNamespace(load_cache_to_device_buffer=MagicMock())
    compile_module = MagicMock(return_value=module)
    monkeypatch.setattr(ops, "_jit_sparse_module", compile_module)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=123)
    )
    monkeypatch.setattr(torch.version, "hip", None)
    workspace = (
        torch.empty((2, ops.hisparse_resolver_scratch_words(1, 2)), dtype=torch.int32)
        if global_scratch
        else None
    )
    loader(
        top_k_tokens=torch.zeros((1, 1), dtype=torch.int32),
        device_buffer_tokens=torch.zeros((2, 3), dtype=torch.int32),
        host_cache_locs=torch.zeros((2, 4), dtype=torch.int64),
        device_buffer_locs=torch.zeros((2, 3), dtype=torch.int32),
        host_cache=torch.zeros((4, 1, 8)),
        device_buffer=torch.zeros((4, 1, 8)),
        top_k_device_locs=torch.zeros((1, 1), dtype=torch.int32),
        req_pool_indices=torch.ones(1, dtype=torch.int64),
        seq_lens=torch.ones(1, dtype=torch.int32),
        lru_slots=torch.zeros((2, 2), dtype=torch.int16),
        item_size_bytes=32,
        num_top_k=1,
        hot_buffer_size=2,
        resolver_scratch=workspace,
    )
    arguments = module.load_cache_to_device_buffer.call_args.args
    assert arguments[-4:-2] == (
        (workspace.data_ptr(), workspace.stride(0)) if global_scratch else (0, 0)
    )
    assert arguments[-2:] == (0, 0)
    assert compile_module.call_args.kwargs["use_global_scratch"] is global_scratch
    assert not any(isinstance(arg, torch.Tensor) for arg in arguments)


@pytest.mark.parametrize(
    "shared_limit,global_scratch", [(49152, True), (163840, False), (233472, False)]
)
def test_compact_workspace_uses_small_shared_layout_or_the_complete_global_fallback(
    monkeypatch, shared_limit, global_scratch
):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            shared_memory_per_block=49152,
            shared_memory_per_block_optin=shared_limit,
            max_threads_per_block=1024,
        ),
    )
    workspace = torch.empty((3, 52678), dtype=torch.int32)
    allocate = MagicMock(return_value=workspace)
    monkeypatch.setattr(torch, "empty", allocate)
    result = ops.allocate_hisparse_resolver_scratch(
        3,
        2048,
        39936,
        960,
        torch.device("cuda:2"),
        use_lru_scratch=True,
    )
    if global_scratch:
        assert result is workspace
        allocate.assert_called_once_with(
            (3, 52678), dtype=torch.int32, device=torch.device("cuda:2")
        )
    else:
        assert result is None
        allocate.assert_not_called()


@pytest.mark.parametrize("capacity", [1536, 39936])
def test_compact_mla_wrapper_passes_separate_lru_storage_without_tensor_ffi(
    monkeypatch, capacity
):
    module = SimpleNamespace(load_cache_to_device_buffer=MagicMock())
    compile_module = MagicMock(return_value=module)
    monkeypatch.setattr(ops, "_jit_sparse_module", compile_module)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=123)
    )
    slots = torch.empty(
        (2, capacity), dtype=torch.int16 if capacity < 32768 else torch.int32
    )
    scratch = torch.empty_like(slots)
    full = torch.ones(4, dtype=torch.int64)
    ops.load_cache_to_device_buffer_mla(
        top_k_tokens=torch.zeros((1, 1), dtype=torch.int32),
        device_buffer_tokens=torch.zeros((2, capacity + 64), dtype=torch.int32),
        host_cache_locs=torch.zeros((2, 4), dtype=torch.int64),
        device_buffer_locs=torch.zeros((2, capacity + 64), dtype=torch.int32),
        host_cache=torch.zeros((4, 1, 8)),
        device_buffer=torch.zeros((4, 1, 8)),
        top_k_device_locs=torch.zeros((1, 1), dtype=torch.int32),
        req_pool_indices=torch.ones(1, dtype=torch.int64),
        seq_lens=torch.ones(1, dtype=torch.int32),
        lru_slots=slots,
        item_size_bytes=32,
        num_top_k=1,
        hot_buffer_size=capacity,
        req_hot_buffer_sizes=torch.ones(2, dtype=torch.int32),
        req_to_logical_token=torch.zeros((2, 4), dtype=torch.int64),
        full_to_device_loc=full,
        full_last_touch=full.clone(),
        full_data_version=full.clone(),
        full_host_version=full.clone(),
        swap_status=torch.zeros(2, dtype=torch.int32),
        resolver_stats=torch.zeros((2, 2), dtype=torch.int32),
        touch_clock=torch.zeros(1, dtype=torch.int64),
        lru_scratch=scratch,
    )
    arguments = module.load_cache_to_device_buffer.call_args.args
    assert arguments[-4:] == (0, 0, scratch.data_ptr(), scratch.stride(0))
    assert compile_module.call_args.kwargs["use_lru_scratch"]
    assert not compile_module.call_args.kwargs["use_global_scratch"]
    assert not any(isinstance(arg, torch.Tensor) for arg in arguments)
