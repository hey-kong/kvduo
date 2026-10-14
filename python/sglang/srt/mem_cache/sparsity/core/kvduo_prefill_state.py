"""Versioned Host checkpoints for KVDuo prefix reuse."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from sglang.srt.mem_cache.sparsity.core.kvduo_prefix_cache import (
    HostPrefixRecord,
    KVDuoHostPrefixCache,
)


@dataclass(frozen=True)
class KVDuoPrefillStatePlan:
    prefix_len: int
    target_len: int
    history: tuple
    window: tuple
    owner: tuple
    swa_tokens: int


@dataclass
class _HostPublication:
    cache: KVDuoHostPrefixCache
    record: HostPrefixRecord
    metadata: torch.Tensor | None = None
    offset: int = 0
    valid: torch.Tensor | None = None


class _HostStateGroup:
    def __init__(self, pools, page_size):
        self.pools = pools
        self.anchor = next(iter(pools.values()))
        self.page_size = page_size
        self.cache = KVDuoHostPrefixCache(self.anchor.size * len(pools))

    def make_room(self, slots):
        missing = max(0, slots - self.anchor.available_size())
        for record in self.cache.evict_lru(missing * len(self.pools)):
            self.anchor.free(torch.tensor(next(iter(record.host_locations.values()))))
        return self.anchor.available_size() >= slots

    def reserve(self, identities, version):
        missing = [key for key in identities if key not in self.cache.records]
        existing = [
            self.cache.records[key] for key in identities if key in self.cache.records
        ]
        for record in existing:
            record.dma_pins += 1
        try:
            if not self.make_room(len(missing) * self.page_size):
                return None
            locations = self.anchor.alloc(len(missing) * self.page_size)
        finally:
            for record in existing:
                record.dma_pins -= 1
        if locations is None:
            return None
        records = []
        for key, rows in zip(missing, locations.split(self.page_size)):
            locs = tuple(rows.tolist())
            record = HostPrefixRecord(
                identity=key,
                domains=tuple(self.pools),
                host_locations={name: locs for name in self.pools},
                data_versions={name: version for name in self.pools},
                host_versions={},
                cache_reference=True,
                dma_pins=1,
                tie_break_key=(key.stable_sort_digest,),
            )
            self.cache.insert(record)
            records.append(record)
        return records


class KVDuoPrefillHostCache:
    """Bounded, versioned history and SWA-window records independent of slots."""

    def __init__(self, coordinator, history_pools, window_pools, page_size):
        self.coordinator = coordinator
        self.page_size = page_size
        self.history = _HostStateGroup(history_pools, page_size)
        self.window = _HostStateGroup(window_pools, page_size)
        self.pending = []
        self.sources = {}
        self.matches = {}
        self.clock = 0

    @classmethod
    def create(cls, coordinator, ratio):
        from sglang.srt.mem_cache.memory_pool_host import (
            DeepSeekV4PagedHostPool,
            DeepSeekV4StateHostPool,
        )
        from sglang.srt.utils import is_pin_memory_available

        allocator = coordinator.token_to_kv_pool_allocator
        kv = allocator.get_kvcache()
        if getattr(kv, "_unified_kv", False):
            raise ValueError("KVDuo Host SWA requires the paged DeepSeek-V4 layout")
        page = allocator.page_size
        full_pages = coordinator.mem_pool_host.num_host_pages
        pin_memory = is_pin_memory_available(coordinator.device)
        swa_pages = max(1, int(kv.swa_size * ratio) // kv.swa_page_size)
        history = {}
        if kv.c4_indexer_kv_pool is not None:
            buffers = kv.c4_indexer_kv_pool.index_k_with_scale_buffer
            history["c4_indexer"] = DeepSeekV4PagedHostPool(
                "kvduo_c4_indexer",
                buffers,
                buffers[0][0].nbytes,
                full_pages,
                page,
                layout="layer_first",
                pin_memory=pin_memory,
            )
        if kv.c128_kv_pool is not None and kv.c128_kv_pool.kv_buffer:
            history["c128"] = DeepSeekV4PagedHostPool(
                "kvduo_c128",
                kv.c128_kv_pool.kv_buffer,
                kv.c128_kv_pool.bytes_per_page_padded,
                full_pages,
                page,
                layout="layer_first",
                pin_memory=pin_memory,
            )
        window = {
            "swa": DeepSeekV4PagedHostPool(
                "kvduo_swa",
                kv.swa_kv_pool.kv_buffer,
                kv.swa_kv_pool.bytes_per_page_padded,
                swa_pages,
                kv.swa_page_size,
                layout="layer_first",
                pin_memory=pin_memory,
            )
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
                window[name] = DeepSeekV4StateHostPool(
                    f"kvduo_{name}",
                    [pools[layer] for layer in c4_layers],
                    swa_pages,
                    kv.swa_page_size,
                    layout="layer_first",
                    pin_memory=pin_memory,
                )
        # Like HiCache, C128's request-scoped state is unused at a 256-aligned
        # boundary. Its compressed history still needs the mirror above.
        if page % 128:
            raise ValueError("KVDuo prefix checkpoints require 128-aligned pages")
        return cls(coordinator, history, window, page)

    def poll(self):
        remaining = []
        for event, publish, unpin in self.pending:
            if event is not None and not event.query():
                remaining.append((event, publish, unpin))
                continue
            for publication in publish:
                cache, record = publication.cache, publication.record
                if cache.records.get(record.identity) is record:
                    if publication.valid is not None and not bool(
                        publication.valid.item()
                    ):
                        cache.remove(record.identity)
                        self.window.anchor.free(
                            torch.tensor(
                                record.host_locations["swa"], dtype=torch.int64
                            )
                        )
                        continue
                    if publication.metadata is not None:
                        start = publication.offset
                        end = start + len(record.host_locations["main_kv"])
                        record.data_versions["main_kv"] = tuple(
                            publication.metadata[0, start:end].tolist()
                        )
                        record.model_touches["main_kv"] = tuple(
                            publication.metadata[1, start:end].tolist()
                        )
                    record.host_versions = dict(record.data_versions)
            for record in unpin:
                record.dma_pins -= 1
        self.pending = remaining

    def _owner(self, req, kind):
        return (kind, self.coordinator._kvduo_prefill_key(req))

    def _hold_source(self, req, cache, identities):
        owner = self._owner(req, "source")
        entry = self.sources.setdefault(id(req), (owner, set()))
        if entry[0] != owner:
            raise RuntimeError(
                "KVDuo prefill source generation changed without cleanup"
            )
        refs = entry[1]
        for identity in identities:
            cache.acquire(identity, owner, self.clock)
            refs.add((cache, identity))

    def owns_pending_main(self, req, identity):
        entry = self.sources.get(id(req))
        return (
            entry is not None
            and (self.coordinator.host_prefix_cache, identity) in entry[1]
        )

    def backup_finished(self, req):
        c = self.coordinator
        retained = c._retain_request_host_prefix(req)
        end = min(
            req.effective_kv_committed_len(),
            c._kvduo_host_valid_len[req.req_pool_idx] * c.compress_ratio,
        )
        self.backup(req, end, main_ready=True)
        return retained

    def backup(self, req, completed_len, *, main_ready=False):
        """Snapshot closed KV pages before their SWA slots are recycled."""
        self.poll()
        c = self.coordinator
        end = completed_len // self.page_size * self.page_size
        if not end:
            return
        self.clock += 1
        keys = c._host_prefix_identities(req, end // self.page_size - 1)
        keys = tuple(keys[: end // self.page_size])
        full = c.req_to_token_pool.req_to_token[req.req_pool_idx, :end].to(
            dtype=torch.int64, copy=True
        )
        missing = [
            i for i, key in enumerate(keys) if key not in c.host_prefix_cache.records
        ]
        self._hold_source(
            req,
            c.host_prefix_cache,
            [key for key in keys if key in c.host_prefix_cache.records],
        )
        slots = len(missing) * c.page_size
        prefill_key = c._kvduo_prefill_key(req)
        if main_ready and missing:
            raise RuntimeError("Finished KVDuo checkpoint has incomplete main backing")
        if not main_ready and not c._kvduo_host_capacity_ready(
            slots, prefill_key=prefill_key
        ):
            return
        main_host = c.mem_pool_host.alloc(slots)
        if main_host is None:
            return
        publish = []
        unpin = []
        main_full = (
            torch.cat(
                [full[i * self.page_size : (i + 1) * self.page_size] for i in missing]
            )
            if missing
            else full[:0]
        )
        compressed = c.mem_pool_device.translate_loc_from_full_to_compressed(main_full)
        main_metadata = (
            torch.empty(
                (2, slots), dtype=torch.int64, device="cpu", pin_memory=full.is_cuda
            )
            if slots
            else None
        )
        main_clock = c.host_prefix_cache.next_clock()
        for index, (ordinal, rows) in enumerate(
            zip(missing, main_host.split(c.page_size))
        ):
            record = HostPrefixRecord(
                identity=keys[ordinal],
                domains=("main_kv",),
                host_locations={"main_kv": tuple(rows.tolist())},
                data_versions={"main_kv": ()},
                host_versions={},
                model_touches={"main_kv": ()},
                cache_reference=True,
                dma_pins=1,
                last_access=main_clock,
                tie_break_key=(ordinal, keys[ordinal].stable_sort_digest),
            )
            c.host_prefix_cache.insert(record)
            publish.append(
                _HostPublication(
                    c.host_prefix_cache, record, main_metadata, index * c.page_size
                )
            )
            unpin.append(record)
        reservation = c._kvduo_prefill_host_reservations.get(prefill_key)
        if reservation is not None:
            c._kvduo_prefill_host_reservations[prefill_key] = (
                req,
                max(0, reservation[1] - slots),
                reservation[2],
            )
        self._hold_source(req, c.host_prefix_cache, keys)
        history = self.history.reserve(keys, self.clock)
        if history is not None:
            self._hold_source(req, self.history.cache, keys)
        # Retain every available prefix boundary within the bounded Host pool.
        # reserve() copies only missing records, so later chunks reuse backups.
        window_pages = min(len(keys), self.window.anchor.size // self.page_size)
        window_start = len(keys) - window_pages
        window_keys = keys[window_start:] if window_pages else ()
        swa = c.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            full[window_start * self.page_size :]
        )
        window_metadata = torch.empty(
            (window_pages,), dtype=torch.int8, device="cpu", pin_memory=full.is_cuda
        )
        window = self.window.reserve(window_keys, self.clock)
        if full.is_cuda:
            source_ready = torch.cuda.Event()
            source_ready.record(torch.cuda.current_stream(c.device))
            context = torch.cuda.stream(c.write_staging_stream)
        else:
            context = nullcontext()
        done = None
        try:
            with context:
                if full.is_cuda:
                    c.write_staging_stream.wait_event(source_ready)
                    for tensor in (full, main_full, compressed, swa):
                        tensor.record_stream(c.write_staging_stream)
                try:
                    if slots:
                        main_metadata.copy_(
                            torch.stack(
                                (
                                    c.full_data_version[compressed],
                                    c.full_last_touch[compressed],
                                )
                            ),
                            non_blocking=True,
                        )
                        c.mem_pool_host.backup_from_device_all_layer(
                            c.mem_pool_device,
                            main_host.to(c.device),
                            c.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                                main_full
                            ),
                            "kernel",
                        )
                        c.full_host_version[compressed] = c.full_data_version[
                            compressed
                        ]
                    if window:
                        window_metadata.copy_(
                            torch.all(swa.reshape(-1, self.page_size) > 0, dim=1).to(
                                torch.int8
                            ),
                            non_blocking=True,
                        )
                    for group, records, source, source_keys in (
                        (self.history, history, full, keys),
                        (self.window, window, swa.clamp_min(0), window_keys),
                    ):
                        if not records:
                            continue
                        ordinal_by_key = {key: i for i, key in enumerate(source_keys)}
                        device_locs = torch.cat(
                            [
                                source[
                                    ordinal_by_key[r.identity]
                                    * self.page_size : (ordinal_by_key[r.identity] + 1)
                                    * self.page_size
                                ]
                                for r in records
                            ]
                        )
                        host_locs = self._locations(
                            group, [r.identity for r in records], c.device
                        )
                        for pool in group.pools.values():
                            pool.backup_from_device_all_layer(
                                None, host_locs, device_locs, "kernel"
                            )
                        publish.extend(
                            _HostPublication(
                                group.cache,
                                r,
                                valid=(
                                    window_metadata[ordinal_by_key[r.identity]]
                                    if group is self.window
                                    else None
                                ),
                            )
                            for r in records
                        )
                        unpin.extend(records)
                        for record in records:
                            record.last_access = self.clock
                except Exception:
                    unpin = list(
                        {
                            id(r): r for r in unpin + (history or []) + (window or [])
                        }.values()
                    )
                    done = self._event()
                    self.pending.append((done, (), unpin))
                    raise
                done = self._event()
                self.pending.append((done, publish, unpin))
        finally:
            # GPU consumers and frees follow even a partially queued snapshot.
            if full.is_cuda and done is not None:
                torch.cuda.current_stream(c.device).wait_event(done)
        self.poll()

    def release_match(self, req):
        entry = self.matches.pop(id(req), None)
        if entry is not None:
            owner, refs = entry
            for cache, key in refs:
                if key in cache.records:
                    cache.release(key, owner)
        req.kvduo_prefill_state_plan = None

    def end_request(self, req):
        self.release_match(req)
        entry = self.sources.pop(id(req), None)
        if entry is not None:
            owner, refs = entry
            for cache, key in refs:
                if key in cache.records:
                    cache.release(key, owner)

    def ready_boundaries(self, req, prefix_len, target_len):
        """Probe history once; window readiness can have holes after eviction."""
        c = self.coordinator
        keys = c._host_prefix_identities(req, target_len // self.page_size - 1)
        first = prefix_len // self.page_size
        window_pages = self.window_tokens // self.page_size
        window_missing = [0]
        history_valid = True
        ready = []
        for ordinal in range(target_len // self.page_size):
            key = keys[ordinal]
            if ordinal >= first:
                history = self.history.cache.records.get(key)
                main = c.host_prefix_cache.records.get(key)
                history_valid = (
                    history_valid
                    and history is not None
                    and main is not None
                    and history.fully_valid
                    and main.fully_valid
                )
            window = self.window.cache.records.get(key)
            window_missing.append(
                window_missing[-1] + int(window is None or not window.fully_valid)
            )
            start = max(first, ordinal + 1 - window_pages)
            ready.append(
                ordinal >= first
                and history_valid
                and window_missing[-1] == window_missing[start]
            )
        return ready

    def ready(self, req, prefix_len, target_len):
        return (
            target_len > prefix_len
            and target_len % self.page_size == 0
            and self.ready_boundaries(req, prefix_len, target_len)[-1]
        )

    @property
    def window_tokens(self):
        window = (
            self.coordinator.token_to_kv_pool_allocator.get_kvcache().sliding_window
        )
        return (window + self.page_size - 1) // self.page_size * self.page_size

    def plan(self, req, prefix_len, target_len):
        if not self.ready(req, prefix_len, target_len):
            return None
        self.release_match(req)
        c = self.coordinator
        keys = c._host_prefix_identities(req, target_len // self.page_size - 1)
        tokens = min(target_len - prefix_len, self.window_tokens)
        history = tuple(
            keys[prefix_len // self.page_size : target_len // self.page_size]
        )
        window = tuple(
            keys[(target_len - tokens) // self.page_size : target_len // self.page_size]
        )
        owner = self._owner(req, "match")
        refs = set()
        for cache, identities in (
            (self.history.cache, history),
            (self.window.cache, window),
        ):
            for key in identities:
                cache.acquire(key, owner, self.clock)
                refs.add((cache, key))
        self.matches[id(req)] = (owner, refs)
        plan = KVDuoPrefillStatePlan(
            prefix_len, target_len, history, window, owner, tokens
        )
        req.kvduo_prefill_state_plan = plan
        return plan

    @staticmethod
    def _locations(group, keys, device):
        name = next(iter(group.pools))
        return torch.tensor(
            [
                slot
                for key in keys
                for slot in group.cache.records[key].host_locations[name]
            ],
            dtype=torch.int64,
            device=device,
        )

    def _event(self):
        if str(self.coordinator.device).startswith("cuda"):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.coordinator.device))
            return event
        return None

    def pin_main_restore(self, plan):
        records = [
            self.coordinator.host_prefix_cache.records[key] for key in plan.history
        ]
        for record in records:
            record.dma_pins += 1
        return records

    def defer_unpin(self, records):
        if records:
            self.pending.append((self._event(), (), records))

    def load(self, req, restored):
        plan = getattr(req, "kvduo_prefill_state_plan", None)
        if (
            plan is None
            or plan.owner != self._owner(req, "match")
            or not self.ready(req, plan.prefix_len, plan.target_len)
        ):
            raise RuntimeError("KVDuo prefill checkpoint is missing or stale")
        c = self.coordinator
        records = [
            group.cache.records[key]
            for group, keys in (
                (self.history, plan.history),
                (self.window, plan.window),
            )
            for key in keys
        ]
        for record in records:
            record.dma_pins += 1
        try:
            for group, keys, device_locs in (
                (self.history, plan.history, restored),
                (
                    self.window,
                    plan.window,
                    c.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        restored[-plan.swa_tokens :]
                    ),
                ),
            ):
                host = self._locations(group, keys, c.device)
                for pool in group.pools.values():
                    for layer in range(pool.layer_num):
                        pool.load_to_device_per_layer(
                            None, host, device_locs, layer, "kernel"
                        )
        finally:
            self.defer_unpin(records)
        self._hold_source(req, self.history.cache, plan.history)

    def clear(self):
        if (
            self.sources
            or self.matches
            or getattr(self.coordinator, "_active_kvduo_reqs", None)
        ):
            raise RuntimeError("Cannot clear KVDuo Host checkpoints in use")
        for event, _, _ in self.pending:
            if event is not None:
                event.synchronize()
        self.poll()
        for group in (self.history, self.window):
            group.cache.clear()
            group.anchor.clear()
        self.coordinator.host_prefix_cache.clear()
        self.coordinator.mem_pool_host.clear()

    def destroy(self):
        for event, _, _ in self.pending:
            if event is not None:
                event.synchronize()
        self.poll()
        for group in (self.history, self.window):
            for pool in group.pools.values():
                pool.destroy()
