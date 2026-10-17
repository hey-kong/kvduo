"""Graph-stable indirection for variable-capacity frequency HOT buffers."""

import functools

import torch
from sglang.kernels.jit.utils import load_jit, make_cpp_args


@functools.cache
def module(top_k, paged, block_size=256):
    args = make_cpp_args(top_k, paged, block_size)
    return load_jit(
        "kvduo_frequency",
        *args,
        cuda_files=["kvduo_frequency.cuh"],
        cuda_wrappers=[
            ("resolve", f"resolve_frequency<{args}>"),
            ("merge", f"merge_frequency<{make_cpp_args(paged)}>"),
        ],
    )


def resolve(coordinator, layer, requests, lengths, selected, output):
    c = coordinator
    if selected.dtype != torch.int32 or selected.stride(-1) != 1:
        raise ValueError("KVDuo top-k requires contiguous int32 rows")
    launch_resolve(
        c.top_k,
        c.is_dsv4_hisparse,
        selected,
        requests.to(torch.int64),
        lengths.to(torch.int64),
        c.kvduo_hot_pointers[layer],
        c.kvduo_req_hot_capacity_gpu[layer],
        c.num_real_reqs,
        c.req_to_full_lookup,
        c.mem_pool_device.full_to_hisparse_device_index_mapping,
        c.full_last_touch,
        c.full_data_version,
        c.full_host_version,
        c.req_to_host_pool,
        c.mem_pool_host.kv_buffer[layer],
        c.mem_pool_device.get_flat_key_buffer(),
        c.item_size_bytes,
        layer * c.mem_pool_device.flat_layer_slot_stride,
        c.full_touch_clock,
        c.kvduo_observe,
        output,
        c.kvduo_swap_status[layer],
        c.kvduo_resolver_stats[layer],
        c.kvduo_hot_growth_events[layer],
        c.kvduo_frequency_scratch,
        block_size=getattr(c, "swap_in_block_size", 256),
    )


def merge(
    buffer,
    backing,
    *,
    k,
    minimum,
    paged,
    top_k,
    item_bytes,
    publication=None,
    workspace=None,
):
    from sglang.srt.mem_cache.sparsity.core.kvduo_frequency import FrequencyBuffer

    capacity = buffer.tokens.numel()
    if k <= 0 or capacity % k or not 1 <= minimum <= capacity // k:
        raise ValueError("Invalid KVDuo frequency merge geometry")
    if (
        buffer.locations.shape != buffer.tokens.shape
        or buffer.frequency.numel() != (capacity + 15) // 16
        or buffer.tokens.dtype != torch.int32
        or buffer.locations.dtype != torch.int32
        or buffer.frequency.dtype != torch.int32
        or item_bytes <= 0
    ):
        raise ValueError("Invalid KVDuo frequency merge metadata")
    if publication is not None:
        if len(publication) != 3 or any(
            t.device != backing.device
            or t.dtype != dtype
            or tuple(t.shape) != shape
            or not t.is_contiguous()
            for t, dtype, shape in zip(
                publication, (torch.int64, torch.int32, torch.int32), ((4,), (1,), (1,))
            )
        ):
            raise ValueError("Invalid KVDuo merge publication layout")
    pages = capacity // k
    # The plan initializes all output token/location/frequency entries.
    storage = (
        torch.empty_like(buffer.tokens),
        torch.empty_like(buffer.locations),
        torch.empty_like(buffer.frequency),
        torch.zeros_like(buffer.selected),
    )
    new = FrequencyBuffer._view(storage, capacity)
    # One coordinator stream serializes merges; retain only scratch, never live metadata.
    if workspace is None:
        workspace = {}

    def reserve(name, size, dtype):
        value = workspace.get(name)
        if value is None or value.numel() < size or value.device != backing.device:
            value = torch.empty(size, dtype=dtype, device=backing.device)
            workspace[name] = value
        return value[:size]

    sources = reserve("sources", capacity, torch.int64)
    destinations = reserve("destinations", capacity, torch.int64)
    work = reserve("work", capacity + 2 * pages, torch.int32)
    summary = reserve("summary", pages + 2, torch.int32)
    host_summary = workspace.get("host_summary")
    if host_summary is None or host_summary.numel() < pages + 2:
        host_summary = torch.empty(pages + 2, dtype=torch.int32, pin_memory=True)
        workspace["host_summary"] = host_summary
    module(top_k, paged).merge(
        *(
            t.data_ptr()
            for t in (
                buffer.tokens,
                buffer.locations,
                buffer.frequency,
                new.tokens,
                new.locations,
                new.frequency,
                sources,
                destinations,
                work,
                summary,
                backing,
            )
        ),
        k,
        minimum,
        item_bytes,
        capacity,
        *(
            (t.data_ptr() for t in publication)
            if publication is not None
            else (0, 0, 0)
        ),
        new.selected.data_ptr(),
        torch.cuda.current_stream(backing.device).cuda_stream,
    )
    host_summary[: pages + 2].copy_(summary)
    target, moved, *keep = host_summary[: pages + 2].tolist()
    if (
        not minimum <= target <= pages
        or sum(keep) != target
        or any(v not in (0, 1) for v in keep)
    ):
        raise RuntimeError("Invalid KVDuo frequency merge result")
    capacity = target * k
    new.tokens = new.tokens[:capacity]
    new.locations = new.locations[:capacity]
    new.frequency = new.frequency[: (capacity + 15) // 16]
    new.selected = new.selected[:capacity]
    return new, tuple(i for i, retained in enumerate(keep) if retained)


def launch_resolve(
    top_k,
    paged,
    selected,
    requests,
    lengths,
    pointers,
    capacities,
    real,
    lookup,
    mapping,
    touches,
    versions,
    host_versions,
    host_locs,
    host_data,
    backing,
    item_bytes,
    location_base,
    clock,
    observe,
    output,
    status,
    stats,
    growth,
    scratch,
    *,
    block_size=256,
):
    # Raw-pointer FFI avoids Torch/TensorView conversion inside CUDA Graph and
    # inference-mode contexts. Validate geometry before crossing that boundary.
    device = selected.device
    batch, owners = selected.size(0), pointers.size(0)

    def check(t, dtype, shape, *, rows=False):
        if (
            t.dtype != dtype
            or t.device != device
            or tuple(t.shape) != tuple(shape)
            or (t.stride(-1) != 1 if rows else not t.is_contiguous())
        ):
            raise ValueError("Invalid KVDuo frequency metadata layout")

    for t in (selected, output):
        check(t, torch.int32, (batch, top_k), rows=True)
    for t in (requests, lengths):
        check(t, torch.int64, (batch,))
    check(pointers, torch.int64, (owners, 4))
    for t in (capacities, status, growth):
        check(t, torch.int32, (owners,))
    for t in (real, observe):
        check(t, torch.int32, (1,))
    check(clock, torch.int64, (1,))
    for t in (mapping, touches, versions, host_versions):
        check(t, torch.int64, (mapping.numel(),))
    for t in (lookup, host_locs):
        check(t, torch.int64, (owners, t.size(1)))
    check(stats, torch.int32, (owners, 2))
    check(scratch, torch.int32, (owners, 9 * top_k))
    if (
        item_bytes <= 0
        or backing.device != device
        or not backing.is_contiguous()
        or not host_data.is_contiguous()
    ):
        raise ValueError("Invalid KVDuo payload layout")
    tensors = (
        selected,
        requests,
        lengths,
        pointers,
        capacities,
        real,
        lookup,
        mapping,
        touches,
        versions,
        host_versions,
        host_locs,
        host_data,
        backing,
    )
    rest = (clock, observe, output, status, stats, growth, scratch)
    if not 32 <= block_size <= 1024 or block_size % 32:
        raise ValueError("KVDuo resolver block size must be a warp multiple")
    module(top_k, paged, block_size).resolve(
        *(t.data_ptr() for t in tensors),
        item_bytes,
        location_base,
        *(t.data_ptr() for t in rest),
        batch,
        owners,
        selected.stride(0),
        output.stride(0),
        lookup.stride(0),
        mapping.numel(),
        host_locs.stride(0),
        torch.cuda.current_stream(device).cuda_stream,
    )
