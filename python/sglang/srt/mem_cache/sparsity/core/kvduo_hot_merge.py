"""Return owned whole HOT pages to the shared physical allocator."""


def free_kvduo_whole_pages(allocator, starts, page_size):
    """Return known unique whole-page ownership without filtering token indices."""
    import torch

    if not starts:
        return
    if (
        page_size <= 0
        or len(set(starts)) != len(starts)
        or any(start <= 0 or start % page_size for start in starts)
        or getattr(allocator, "page_size", page_size) != page_size
    ):
        raise ValueError("KVDuo reclaim requires unique aligned non-dummy pages")
    page_ids = torch.tensor(
        [start // page_size for start in starts],
        dtype=torch.int64,
        device=allocator.device,
    )
    # Match the immediate physical-page return used by free_hisparse_indices.
    allocator.is_not_in_free_group = True
    allocator._release_page_ids(page_ids)
    if getattr(allocator, "debug_mode", False):
        allocator._debug_check_no_duplicate_pages()
