"""Startup ordering and isolation of KVDuo pressure-kernel warmup."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.kernels.ops.kvcache import kvduo_warmup as ops
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def startup_method():
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/model_executor/model_runner.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "maybe_init_hisparse_coordinator"
    )
    namespace = {"get_parallel": lambda: SimpleNamespace(enable_dp_attention=False)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[fn.name]


@pytest.mark.parametrize("enabled,kvduo", [(False, False), (True, False), (True, True)])
def test_only_kvduo_waits_for_warmup_before_startup_returns(
    monkeypatch, enabled, kvduo
):
    import sglang.srt.managers.hisparse_coordinator as coordinator
    import sglang.srt.mem_cache.sparsity as sparsity

    cfg = SimpleNamespace(
        top_k=512,
        device_buffer_size=8192,
        host_to_device_ratio=2,
        swap_in_block_size=960,
        host_size=0,
        tail_protected_pages=2,
        full_reclaim_policy="request",
        decode_emergency_reclaim=False,
        max_relative_miss_increase=0.1,
    )
    monkeypatch.setattr(sparsity, "parse_hisparse_config", lambda _: cfg)
    monkeypatch.setattr(sparsity, "parse_kvduo_config", lambda _: cfg)
    instance = MagicMock()
    factory = MagicMock(return_value=instance)
    monkeypatch.setattr(coordinator, "HiSparseCoordinator", factory)
    runner = SimpleNamespace(
        enable_hisparse=enabled,
        server_args=SimpleNamespace(enable_kvduo=kvduo),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace()),
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(),
        device="cuda",
        tp_group=SimpleNamespace(cpu_group=object()),
    )
    startup_method()(runner)
    assert factory.call_count == int(enabled)
    assert instance.warmup_reclamation.call_count == int(enabled and kvduo)
    if enabled and kvduo:
        instance.warmup_reclamation.side_effect = RuntimeError("warmup failed")
        with pytest.raises(RuntimeError, match="warmup failed"):
            startup_method()(runner)


@pytest.mark.parametrize("enabled,cuda", [(False, True), (True, False), (True, True)])
def test_coordinator_passes_actual_geometry_without_mutating_banks(
    monkeypatch, enabled, cuda
):
    warm = MagicMock()
    monkeypatch.setattr(ops, "warmup_kvduo_reclaim", warm)
    backing = object()
    c = SimpleNamespace(
        enable_mixed_residency=enabled,
        req_device_buffer_tokens=SimpleNamespace(is_cuda=cuda, size=lambda _: 40000),
        mem_pool_device=SimpleNamespace(get_flat_key_buffer=lambda: backing),
        page_size=64,
        hot_storage_layers=78,
        device_buffer_size=39936,
        is_dsv4_hisparse=True,
    )
    HiSparseCoordinator.warmup_reclamation(c)
    if enabled and cuda:
        warm.assert_called_once_with(
            backing=backing,
            page_size=64,
            storage_layers=78,
            maximum=39936,
            padded=40000,
            paged=True,
        )
    else:
        warm.assert_not_called()


def test_cpu_public_entry_does_not_load_cuda_modules(monkeypatch):
    monkeypatch.setattr(ops, "_warmup", MagicMock(side_effect=AssertionError("CUDA")))
    ops.warmup_kvduo_reclaim(
        backing=torch.zeros(1, 2),
        page_size=64,
        storage_layers=1,
        maximum=512,
        padded=576,
        paged=False,
    )


def test_warmup_fixture_forces_moves_and_never_uses_real_backing(monkeypatch):
    backing = torch.full((3, 16), 37, dtype=torch.uint8)
    workspace = SimpleNamespace(summary_host=torch.tensor([0, 1, 1]))
    monkeypatch.setattr(ops, "allocate_kvduo_merge_workspace", lambda *args: workspace)
    monkeypatch.setattr(ops, "finish_kvduo_hot_merge", lambda *args: ((0,), (1,)))

    def execute(tokens, locations, lru, payload, scratch, **kwargs):
        from sglang.srt.mem_cache.sparsity.core.kvduo_hot_merge import (
            plan_kvduo_hot_merge,
        )

        k = kwargs["k"]
        plan = plan_kvduo_hot_merge(
            tokens[: 2 * k].tolist(), lru[: 2 * k].tolist(), k, 1, 1
        )
        assert len(plan.source_slots) > 0
        assert locations[: 2 * k].unique().numel() == 2 * k
        assert int(locations[: 2 * k].max()) < payload.size(0)
        assert payload.data_ptr() != backing.data_ptr()
        payload.fill_(99)

    monkeypatch.setattr(ops, "execute_kvduo_hot_merge", execute)
    ops._warmup(backing, 64, 2, 1024, 1088, False)
    assert torch.all(backing == 37)
