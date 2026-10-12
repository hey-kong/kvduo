"""FULL publication preserves identity, version and recency across slot reuse."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.kernels.ops.kvcache import kvduo_full_write as ops
from sglang.test.ci.ci_register import register_cpu_ci
from test_kvduo_decode_full_eviction import make_decode_coordinator

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("items", [0, 1, 32, 513])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_full_write_reuses_locations_with_new_identity_and_invalid_host_version(
    items, dtype
):
    generation = torch.arange(items + 5, dtype=torch.int64) + 10
    data = torch.full_like(generation, 7)
    host = data.clone()
    touches = torch.full_like(generation, 40)
    clock = torch.tensor([41], dtype=torch.int64)
    backing = torch.full((2 * items,), -99, dtype=dtype)
    logical = backing[::2]
    logical.copy_(torch.arange(items, dtype=dtype) + 3)
    rows = torch.zeros(items, dtype=torch.int32)
    positions = torch.arange(items, dtype=torch.int64)
    lookup_backing = torch.full((4, items + 5), -1, dtype=torch.int64)
    lookup = lookup_backing[::2]
    original_generation = generation.clone()
    addresses = [
        (bank.data_ptr(), bank.shape, bank.stride())
        for bank in (generation, data, host, touches, clock, lookup)
    ]
    for step in (1, 2):
        ops.publish_kvduo_full_write(
            logical,
            generation,
            data,
            host,
            touches,
            clock,
            lookup=lookup,
            req_indices=rows,
            positions=positions,
        )
        assert clock.item() == 41 + step
        assert torch.equal(generation[logical], original_generation[logical] + step)
        assert torch.all(data[logical] == 1)
        assert torch.all(host[logical] == -1)
        assert torch.all(touches[logical] == 41 + step)
        assert torch.equal(lookup[0, :items], logical.long())
        assert torch.all(lookup_backing[1::2] == -1)
        assert torch.equal(generation[:3], original_generation[:3])
        assert torch.all(touches[:3] == 40)
    assert addresses == [
        (bank.data_ptr(), bank.shape, bank.stride())
        for bank in (generation, data, host, touches, clock, lookup)
    ]


@pytest.mark.parametrize("ratio", [1, 4])
def test_actual_decode_append_publishes_lookup_and_versions_in_one_call(
    monkeypatch, ratio
):
    import sglang.srt.managers.hisparse_coordinator as module

    c, _, req, _, _, _ = make_decode_coordinator(monkeypatch, ratio=ratio)
    publish = MagicMock(wraps=ops.publish_kvduo_full_write)
    monkeypatch.setattr(module, "publish_kvduo_full_write", publish)
    c._backup_kvduo_sealed_pages = MagicMock()
    start_clock = c.full_touch_clock.item()
    locations = torch.tensor([6 * ratio - 1], dtype=torch.int64)
    lengths = torch.tensor([6 * ratio], dtype=torch.int64)
    request = torch.tensor([req.req_pool_idx], dtype=torch.int64)
    c.map_last_loc_to_buffer(lengths, locations, request, lengths, request)
    publish.assert_called_once()
    assert publish.call_args.kwargs["lookup"] is c.req_to_full_lookup
    logical = (6 * ratio - 1) // ratio
    assert c.req_to_full_lookup[0, 5].item() == logical
    assert c.full_data_version[logical].item() == 1
    assert c.full_host_version[logical].item() == -1
    assert c.full_last_touch[logical].item() == start_clock + 1


def test_full_write_without_lookup_and_partial_metadata_validation():
    banks = [torch.zeros(5, dtype=torch.int64) for _ in range(4)]
    clock = torch.tensor(10, dtype=torch.int64)
    ops.publish_kvduo_full_write(torch.tensor([2]), *banks, clock)
    assert [bank[2].item() for bank in banks] == [1, 1, -1, 11]
    before = [bank.clone() for bank in (*banks, clock)]
    with pytest.raises(ValueError, match="required together"):
        ops.publish_kvduo_full_write(
            torch.tensor([2]),
            *banks,
            clock,
            req_indices=torch.tensor([0]),
        )
    assert all(torch.equal(old, bank) for old, bank in zip(before, (*banks, clock)))


def test_full_write_ffi_uses_only_scalar_addresses_and_preserves_strides(monkeypatch):
    class CudaAddressTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    logical = torch.tensor([3, -1, 5, -1], dtype=torch.int32)[::2].as_subclass(
        CudaAddressTensor
    )
    banks = [torch.zeros(8, dtype=torch.int64) for _ in range(4)]
    clock = torch.tensor([10], dtype=torch.int64)
    lookup = torch.zeros((4, 6), dtype=torch.int64)[::2]
    rows = torch.tensor([0, -1, 1, -1], dtype=torch.int64)[::2]
    positions = torch.tensor([1, -1, 4, -1], dtype=torch.int32)[::2]
    module = SimpleNamespace(publish=MagicMock())
    monkeypatch.setattr(ops, "_jit_full_write_module", lambda: module)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=123)
    )
    ops.publish_kvduo_full_write(
        logical,
        *banks,
        clock,
        lookup=lookup,
        req_indices=rows,
        positions=positions,
    )
    arguments = module.publish.call_args.args
    assert not any(isinstance(arg, torch.Tensor) for arg in arguments)
    assert arguments[:9] == tuple(
        bank.data_ptr() for bank in (logical, *banks, clock, lookup, rows, positions)
    )
    assert arguments[10:] == (2, 8, 2, 12, 2, 6, 2, 2, False, True, False)
