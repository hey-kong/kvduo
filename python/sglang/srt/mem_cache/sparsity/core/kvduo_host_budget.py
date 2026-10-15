"""Host payload sizing independent of KVDuo's HBM and logical capacities."""

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


def plan_host_budget(host_size, full_pages, swa_pages, full_bytes, window_bytes):
    """Scale history/window pools together; round down within a fixed GB budget."""
    if full_pages <= 0 or sum(full_bytes.values()) <= 0:
        raise ValueError("KVDuo requires a nonempty Host KV pool")
    if window_bytes and swa_pages <= 0:
        raise ValueError("KVDuo requires a nonempty Host SWA pool")

    def counts(n):
        return max(1, n * swa_pages // full_pages) if window_bytes else 0

    def sizes(n):
        return {
            **{name: n * size for name, size in full_bytes.items()},
            **{name: counts(n) * size for name, size in window_bytes.items()},
        }

    if host_size == 0:
        n = 2 * full_pages
        budget_bytes = sum(sizes(n).values())
    else:
        budget_bytes = int(host_size * 1e9)
        lo, hi = 0, budget_bytes // sum(full_bytes.values())
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if sum(sizes(mid).values()) <= budget_bytes:
                lo = mid
            else:
                hi = mid - 1
        n = lo
        if not n:
            raise ValueError("KVDuo host_size cannot fit one page in each Host pool")

    # PP stages can have different bytes per page. Use the same capacities,
    # bounded by the smallest stage, as fixed-size HiCache does.
    from sglang.srt.mem_cache.pool_host.base import sync_fixed_hicache_size

    n = sync_fixed_hicache_size(n, host_size)
    windows = sync_fixed_hicache_size(counts(n), host_size)
    pool_bytes = {
        **{name: n * size for name, size in full_bytes.items()},
        **{name: windows * size for name, size in window_bytes.items()},
    }
    return KVDuoHostBudget(n, windows, pool_bytes, budget_bytes)


def plan_dsv4_host_budget(allocator, host_size):
    kv = allocator.get_kvcache()
    if getattr(kv, "_unified_kv", False):
        raise ValueError("KVDuo Host SWA requires the paged DeepSeek-V4 layout")
    main = allocator.hisparse_kvcache
    full_pages = (allocator.size_full + allocator.page_size - 1) // allocator.page_size
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
    return plan_host_budget(host_size, full_pages, swa_pages, full_bytes, window_bytes)
