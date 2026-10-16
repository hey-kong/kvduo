"""Saturating-frequency HOT metadata and page compaction (no entry LRU)."""

from dataclasses import dataclass, field

import torch


def pack_frequency(values, *, device=None):
    values = list(values)
    words = [0] * ((len(values) + 15) // 16)
    for slot, value in enumerate(values):
        if not 0 <= value <= 3:
            raise ValueError("HOT frequency must fit two bits")
        words[slot // 16] |= int(value) << (2 * (slot % 16))
    return torch.tensor(words, dtype=torch.int64, device=device).to(torch.int32)


def unpack_frequency(words, size):
    data = words.cpu().tolist()
    return [(data[i // 16] >> (2 * (i % 16))) & 3 for i in range(size)]


@dataclass
class FrequencyBuffer:
    tokens: torch.Tensor
    locations: torch.Tensor
    frequency: torch.Tensor
    selected: torch.Tensor

    _storage: tuple | None = field(default=None, repr=False)

    @classmethod
    def empty(cls, locations, *, reserve=None):
        n = locations.numel()
        capacity = max(n, reserve or n)
        storage = (
            torch.full((capacity,), -1, dtype=torch.int32, device=locations.device),
            torch.empty(capacity, dtype=torch.int32, device=locations.device),
            torch.zeros(
                (capacity + 15) // 16, dtype=torch.int32, device=locations.device
            ),
            torch.zeros(capacity, dtype=torch.uint8, device=locations.device),
        )
        storage[1][:n].copy_(locations)
        return cls._view(storage, n)

    @classmethod
    def _view(cls, storage, n):
        return cls(
            storage[0][:n],
            storage[1][:n],
            storage[2][: (n + 15) // 16],
            storage[3][:n],
            storage,
        )

    def pointers(self):
        return [
            x.data_ptr()
            for x in (self.tokens, self.locations, self.frequency, self.selected)
        ]

    def grown(self, locations):
        n = self.tokens.numel()
        target = n + locations.numel()
        storage = self._storage or (
            self.tokens,
            self.locations,
            self.frequency,
            self.selected,
        )
        if storage[0].numel() < target:
            result = self.empty(
                self.locations, reserve=max(target, storage[0].numel() * 2)
            )
            result.tokens.copy_(self.tokens)
            result.frequency.copy_(self.frequency)
            storage = result._storage
        result = self._view(storage, target)
        result.locations[n:].copy_(locations)
        result.tokens[n:].fill_(-1)
        result.selected[n:].zero_()
        # Clear only newly visible bits, including after a failed publication.
        if n % 16:
            result.frequency[n // 16].bitwise_and_((1 << (2 * (n % 16))) - 1)
        result.frequency[(n + 15) // 16 :].zero_()
        return result


@dataclass(frozen=True)
class FrequencyMergePlan:
    kept_pages: tuple
    released_pages: tuple
    tokens: tuple
    frequency: tuple
    sources: tuple
    destinations: tuple


def plan_frequency_merge(tokens, frequency, page_entries, minimum_pages):
    """Retain pre-decay threes; choose physical pages to minimize copying."""
    n = len(tokens)
    if page_entries <= 0 or n % page_entries or len(frequency) != n:
        raise ValueError("Invalid HOT metadata geometry")
    pages = n // page_entries
    if not 1 <= minimum_pages <= pages or any(not 0 <= f <= 3 for f in frequency):
        raise ValueError("Invalid HOT capacity/frequency")
    survivors = [
        i for i, token in enumerate(tokens) if token >= 0 and frequency[i] == 3
    ]
    target = max(minimum_pages, (len(survivors) + page_entries - 1) // page_entries)
    counts = [0] * pages
    for slot in survivors:
        counts[slot // page_entries] += 1
    kept = tuple(sorted(sorted(range(pages), key=lambda p: (-counts[p], p))[:target]))
    retained = [
        p * page_entries + offset for p in kept for offset in range(page_entries)
    ]
    slots = {old: new for new, old in enumerate(retained)}
    live = set(survivors)
    available = [slot for slot in retained if slot not in live]
    sources = tuple(slot for slot in survivors if slot not in slots)
    destinations = tuple(available[: len(sources)])
    moved = dict(zip(sources, destinations))
    new_tokens = [-1] * len(retained)
    new_frequency = [0] * len(retained)
    for slot in survivors:
        new = slots[moved.get(slot, slot)]
        new_tokens[new] = tokens[slot]
        new_frequency[new] = 1
    return FrequencyMergePlan(
        kept,
        tuple(p for p in range(pages) if p not in kept),
        tuple(new_tokens),
        tuple(new_frequency),
        sources,
        destinations,
    )
