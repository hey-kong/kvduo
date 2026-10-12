"""FULL publication preserves identity, version and recency across slot reuse."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.kernels.ops.kvcache import kvduo_full_write as ops
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
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
    assert publish.call_args.kwargs["positions"] is lengths
    assert publish.call_args.kwargs["decode_compress_ratio"] == ratio
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
    monkeypatch.setattr(ops, "_jit_full_write_module", lambda ratio=0: module)
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


@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("items", [0, 1, 32, 513])
def test_decode_publication_matches_allocator_alignment_and_skips_inactive_rows(
    ratio, items
):
    ids = torch.arange(items, dtype=torch.int64) + 3
    locations = ids * ratio + ratio - 1
    lengths = (torch.arange(items, dtype=torch.int32) % 8 + 1) * ratio
    lengths[::3] = 0 if ratio == 1 else lengths[::3] + 1
    rows = torch.arange(items - 1, -1, -1, dtype=torch.int64)
    active = (lengths > 0) & (lengths % ratio == 0)
    # Invalid indices in inactive rows must never reach metadata indexing.
    locations[~active] = rows[~active] = -99
    generation = torch.arange(items + 5, dtype=torch.int64) + 10
    data = torch.full_like(generation, 7)
    host = data.clone()
    touches = torch.full_like(generation, 40)
    clock = torch.tensor([41], dtype=torch.int64)
    lookup = torch.full((items, 8), -1, dtype=torch.int64)
    banks = (generation, data, host, touches, clock, lookup)
    expected = [bank.clone() for bank in banks]
    g, d, h, t, c, l = expected
    if active.any():
        compressed = (
            DeepSeekV4HiSparseTokenToKVPoolAllocator.get_last_loc_compressed(
                SimpleNamespace(compress_ratio=ratio), locations[active]
            )
            if ratio == 4
            else locations[active]
        )
        c += 1
        g[compressed] += 1
        d[compressed] = 1
        h[compressed] = -1
        t[compressed] = c[0]
        l[rows[active], lengths[active] // ratio - 1] = compressed
    ops.publish_kvduo_full_write(
        locations,
        generation,
        data,
        host,
        touches,
        clock,
        lookup=lookup,
        req_indices=rows,
        positions=lengths,
        decode_compress_ratio=ratio,
    )
    assert all(torch.equal(bank, ref) for bank, ref in zip(banks, expected))


@pytest.mark.parametrize("mixed", [False, True])
def test_actual_c4_decode_only_publishes_aligned_requests(monkeypatch, mixed):
    c, _, _, _, _, _ = make_decode_coordinator(monkeypatch, ratio=4, mixed=mixed)
    c.req_to_full_lookup = torch.full((4, 12), -1, dtype=torch.int64)
    calls = []
    c._backup_kvduo_sealed_pages = MagicMock(
        side_effect=lambda *args: calls.append("backup")
    )
    publish = ops.publish_kvduo_full_write

    def record_publish(*args, **kwargs):
        calls.append("publish")
        publish(*args, **kwargs)

    import sglang.srt.managers.hisparse_coordinator as module

    monkeypatch.setattr(module, "publish_kvduo_full_write", record_publish)
    original = [
        bank.clone()
        for bank in (
            c.full_generation,
            c.full_data_version,
            c.full_host_version,
            c.full_last_touch,
        )
    ]
    clock = c.full_touch_clock.item()
    request_ids = torch.tensor([3, 1, 2, 0])
    lengths = torch.tensor([21, 24, 23, 28])
    locations = torch.tensor([-99, 23, -99, 27])
    c.map_last_loc_to_buffer(lengths, locations, request_ids, lengths, request_ids)
    assert calls == ["backup", "publish"]
    expected_lookup = torch.full_like(c.req_to_full_lookup, -1)
    expected_lookup[1, 5] = 5
    expected_lookup[0, 6] = 6
    assert torch.equal(c.req_to_full_lookup, expected_lookup)
    assert c.full_touch_clock.item() == clock + 1
    updated = torch.tensor([5, 6])
    original[0][updated] += 1
    original[1][updated] = 1
    original[2][updated] = -1
    original[3][updated] = clock + 1
    assert all(
        torch.equal(bank, ref)
        for bank, ref in zip(
            (
                c.full_generation,
                c.full_data_version,
                c.full_host_version,
                c.full_last_touch,
            ),
            original,
        )
    )
    inactive_lengths = torch.tensor([21, 25, 29, 33])
    c.map_last_loc_to_buffer(
        inactive_lengths,
        torch.full_like(locations, -99),
        request_ids,
        inactive_lengths,
        request_ids,
    )
    assert c.full_touch_clock.item() == clock + 1
    assert torch.equal(c.req_to_full_lookup, expected_lookup)


@pytest.mark.parametrize("ratio", [1, 4])
def test_cuda_decode_dispatch_never_reads_device_values_on_host(monkeypatch, ratio):
    class CudaAddressTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

        def __getitem__(self, key):
            raise AssertionError("dynamic device indexing")

    c, _, _, _, _, _ = make_decode_coordinator(monkeypatch, ratio=ratio)
    c._backup_kvduo_sealed_pages = MagicMock()
    locations = torch.tensor([19, 23, 27], dtype=torch.int32).as_subclass(
        CudaAddressTensor
    )
    lengths = torch.tensor(
        [20, 0 if ratio == 1 else 23, 28], dtype=torch.int64
    ).as_subclass(CudaAddressTensor)
    rows = torch.tensor([0, -99, 0], dtype=torch.int32).as_subclass(CudaAddressTensor)
    kernel = SimpleNamespace(publish=MagicMock())
    loader = MagicMock(return_value=kernel)
    monkeypatch.setattr(ops, "_jit_full_write_module", loader)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda _: SimpleNamespace(cuda_stream=123)
    )
    for name in ("any", "nonzero", "masked_select"):
        monkeypatch.setattr(
            torch, name, MagicMock(side_effect=AssertionError("device reduction"))
        )
    for name in ("item", "cpu", "tolist", "__bool__"):
        monkeypatch.setattr(
            torch.Tensor, name, MagicMock(side_effect=AssertionError("device read"))
        )
    c.map_last_loc_to_buffer(lengths, locations, rows, lengths, rows)
    loader.assert_called_once_with(ratio)
    args = kernel.publish.call_args.args
    assert not any(isinstance(arg, torch.Tensor) for arg in args)
    assert args[0] == locations.data_ptr()
    assert args[7:9] == (rows.data_ptr(), lengths.data_ptr())
    assert args[10] == locations.numel()


@pytest.mark.parametrize("ratio", [2, 8])
def test_invalid_decode_compression_does_not_modify_metadata(ratio):
    banks = [torch.zeros(5, dtype=torch.int64) for _ in range(4)]
    clock = torch.tensor([10], dtype=torch.int64)
    with pytest.raises(ValueError, match="decode compression"):
        ops.publish_kvduo_full_write(
            torch.tensor([2]), *banks, clock, decode_compress_ratio=ratio
        )
    assert clock.item() == 10
    assert not any(torch.any(bank) for bank in banks)


def test_decode_publication_requires_mapping_before_modifying_metadata():
    banks = [torch.zeros(5, dtype=torch.int64) for _ in range(4)]
    clock = torch.tensor([10], dtype=torch.int64)
    with pytest.raises(ValueError, match="requires request mapping"):
        ops.publish_kvduo_full_write(
            torch.tensor([2]), *banks, clock, decode_compress_ratio=4
        )
    assert clock.item() == 10


def test_jit_variants_have_distinct_cache_keys(monkeypatch):
    load = MagicMock()
    monkeypatch.setattr(ops, "load_jit", load)
    for ratio in (0, 1, 4):
        ops._jit_full_write_module.__wrapped__(ratio)
    calls = load.call_args_list
    assert len({call.args for call in calls}) == 3
    for ratio, call in zip((0, 1, 4), calls):
        assert call.args == ("kvduo_full_write", f"decode_{ratio}")
        assert call.kwargs["cuda_wrappers"] == [
            ("publish", f"kvduo_publish_full_write<256, {ratio}>")
        ]
