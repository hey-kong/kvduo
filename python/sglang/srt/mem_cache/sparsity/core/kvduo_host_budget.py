"""Host payload sizing independent of KVDuo's HBM and logical capacities."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class KVDuoHostBudget:
    full_pages: int
    swa_pages: int
    pool_bytes: dict[str, int]
    budget_bytes: int

    @property
    def allocated_bytes(self):
        return sum(self.pool_bytes.values())


def plan_host_budget(host_ratio, full_pages, swa_pages, full_bytes, window_bytes):
    """Scale device history/window capacities by a Host/device page ratio."""
    if (
        not isinstance(host_ratio, (int, float))
        or isinstance(host_ratio, bool)
        or not math.isfinite(host_ratio)
        or host_ratio <= 0
    ):
        raise ValueError("KVDuo host_ratio must be a finite positive number")
    if full_pages <= 0 or sum(full_bytes.values()) <= 0:
        raise ValueError("KVDuo requires a nonempty Host KV pool")
    n = int(full_pages * host_ratio)
    windows = int(swa_pages * host_ratio) if window_bytes else 0
    if n <= 0 or (window_bytes and windows <= 0):
        raise ValueError("KVDuo host_ratio cannot fit one page in each Host pool")
    pool_bytes = {
        **{name: n * size for name, size in full_bytes.items()},
        **{name: windows * size for name, size in window_bytes.items()},
    }
    return KVDuoHostBudget(n, windows, pool_bytes, sum(pool_bytes.values()))


def plan_dsv4_host_budget(allocator, host_ratio):
    kv = allocator.get_kvcache()
    if getattr(kv, "_unified_kv", False):
        raise ValueError("KVDuo Host SWA requires the paged DeepSeek-V4 layout")
    main = allocator.hisparse_kvcache
    full_pages = main.size // main.page_size
    swa_pages = (kv.swa_size + kv.swa_page_size - 1) // kv.swa_page_size
    full_bytes = {"c4": len(main.kv_buffer) * main.bytes_per_page_padded}
    if kv.c4_indexer_kv_pool is not None:
        buffers = kv.c4_indexer_kv_pool.index_k_with_scale_buffer
        full_bytes["c4_indexer"] = sum(buf[0].nbytes for buf in buffers)
    if kv.c128_kv_pool is not None and kv.c128_kv_pool.kv_buffer:
        pool = kv.c128_kv_pool
        full_bytes["c128"] = len(pool.kv_buffer) * pool.bytes_per_page_padded
    window_bytes = {
        "swa": len(kv.swa_kv_pool.kv_buffer) * kv.swa_kv_pool.bytes_per_page_padded
    }
    c4_layers = [
        layer
        for layer in range(kv.start_layer, kv.end_layer)
        if kv.layer_mapping[layer].compress_ratio == 4
    ]
    for name, pools in (
        ("c4_state", kv.compress_state_pools),
        ("c4_indexer_state", kv.indexer_compress_state_pools),
    ):
        if c4_layers:
            window_bytes[name] = sum(
                pools[layer].ring_size * pools[layer].kv_score_buffer.kv_score[0].nbytes
                for layer in c4_layers
            )
    return plan_host_budget(host_ratio, full_pages, swa_pages, full_bytes, window_bytes)
