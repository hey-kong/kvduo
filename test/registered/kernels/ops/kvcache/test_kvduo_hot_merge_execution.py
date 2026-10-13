import pytest
import torch

from sglang.kernels.ops.kvcache.kvduo_hot_merge import (
    allocate_kvduo_merge_workspace,
    execute_kvduo_hot_merge,
    finish_kvduo_hot_merge,
)
from sglang.srt.mem_cache.sparsity.core.kvduo_hot_merge import plan_kvduo_hot_merge
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU merge execution requires CUDA"
)


def merge_banks(tokens, locations, lru):
    shape = tokens.shape[:2]
    return dict(
        published_capacity=torch.zeros(shape, dtype=torch.int32, device="cuda"),
        capacity_ema=torch.ones((*shape, 9), dtype=torch.float32, device="cuda"),
        capacity_steps=torch.full(shape, 65, dtype=torch.int64, device="cuda"),
        growth_events=torch.ones(shape, dtype=torch.int32, device="cuda"),
        base_capacity=torch.zeros(shape, dtype=torch.int32, device="cuda"),
        prefill_ema=torch.ones((*shape, 2), dtype=torch.float32, device="cuda"),
        prefill_steps=torch.full(shape, 65, dtype=torch.int64, device="cuda"),
    )


@pytest.mark.parametrize("k", [4, 4992])
@pytest.mark.parametrize("dtype,width", [(torch.uint8, 8), (torch.bfloat16, 3)])
def test_gpu_execution_reuses_fixed_workspace_and_preserves_payload_and_lru(
    k, dtype, width
):
    maximum = 8 * k
    tokens = torch.full((2, 2, maximum + 64), -1, dtype=torch.int32, device="cuda")
    locations = torch.full_like(tokens, -1)
    slot_dtype = torch.int32 if maximum > 32767 else torch.int16
    lru = torch.full((2, 2, maximum), -1, dtype=slot_dtype, device="cuda")
    backing = (
        torch.arange((4 * maximum + 1) * width, device="cuda")
        .remainder(251)
        .to(dtype)
        .view(-1, width)
    )
    banks = merge_banks(tokens, locations, lru)
    workspace = allocate_kvduo_merge_workspace(tokens[0, 0], lru[0, 0])
    original_addresses = [
        (t.data_ptr(), t.shape, t.stride()) for t in (tokens, locations, lru)
    ]
    scratch_ptrs = [
        t.data_ptr() for t in (workspace.scratch, workspace.sources, workspace.summary)
    ]
    generator = torch.Generator().manual_seed(329)
    for case, owner in enumerate(((0, 0), (1, 1), (0, 1), (1, 0))):
        n, m = ((8, 4), (4, 2), (3, 1), (2, 1))[case]
        ids = torch.arange(n * k, dtype=torch.int32)
        if case == 2:
            ids.fill_(-1)  # zero moves still publishes a valid empty cache
        order = torch.randperm(n * k, generator=generator).to(slot_dtype)
        locs = torch.randperm(n * k, generator=generator).int() + 1 + case * maximum
        tokens[owner][: n * k] = ids.to("cuda")
        locations[owner][: n * k] = locs.to("cuda")
        lru[owner][: n * k] = order.to("cuda")
        expected = plan_kvduo_hot_merge(ids.tolist(), order.tolist(), k, 1, m)
        expected_backing = backing.cpu().clone()
        for source, destination in zip(
            expected.source_slots, expected.destination_slots
        ):
            expected_backing[locs[destination]] = expected_backing[locs[source]].clone()
        snapshots = [t.cpu().clone() for t in (tokens, locations, lru)]
        owner_banks = {name: bank[owner] for name, bank in banks.items()}
        execute_kvduo_hot_merge(
            tokens[owner],
            locations[owner],
            lru[owner],
            backing,
            workspace,
            capacity=n * k,
            k=k,
            top_k=1,
            target_pages=m,
            paged=False,
            **owner_banks,
        )
        kept, released = finish_kvduo_hot_merge(workspace, n, m, k)
        assert (kept, released) == (expected.kept_pages, expected.released_pages)
        assert torch.equal(backing.cpu(), expected_backing)
        assert tokens[owner][: m * k].tolist() == list(expected.tokens)
        assert lru[owner][: m * k].tolist() == list(expected.lru_slots)
        assert locations[owner][: m * k].tolist() == [
            int(locs[page * k + offset])
            for page in expected.kept_pages
            for offset in range(k)
        ]
        assert all(torch.all(t[owner][m * k :] == -1) for t in (tokens, locations, lru))
        for name, bank in owner_banks.items():
            assert torch.all(
                bank
                == (m * k if name in ("published_capacity", "base_capacity") else 0)
            )
        for layer in range(2):
            for req in range(2):
                if (layer, req) != owner:
                    assert all(
                        torch.equal(t[layer, req].cpu(), saved[layer, req])
                        for t, saved in zip((tokens, locations, lru), snapshots)
                    )
        assert original_addresses == [
            (t.data_ptr(), t.shape, t.stride()) for t in (tokens, locations, lru)
        ]
        assert scratch_ptrs == [
            t.data_ptr()
            for t in (workspace.scratch, workspace.sources, workspace.summary)
        ]


@pytest.mark.parametrize(
    "corruption", ["lru", "duplicate", "negative", "out_of_backing"]
)
def test_invalid_gpu_plan_prevents_payload_and_publication(corruption):
    tokens = torch.arange(34, dtype=torch.int32, device="cuda").view(1, 1, -1)
    locations = tokens.clone() + 1
    lru = torch.arange(32, dtype=torch.int16, device="cuda").view(1, 1, -1)
    backing = torch.arange(128 * 8, dtype=torch.float32, device="cuda").view(128, 8)
    if corruption == "lru":
        lru[0, 0, 1] = 0
    elif corruption == "duplicate":
        locations[0, 0, 1] = locations[0, 0, 0]
    elif corruption == "negative":
        locations[0, 0, 1] = -1
    else:
        locations[0, 0, 1] = backing.size(0)
    banks = merge_banks(tokens, locations, lru)
    snapshots = [t.clone() for t in (tokens, locations, lru, backing, *banks.values())]
    workspace = allocate_kvduo_merge_workspace(tokens[0, 0], lru[0, 0])
    execute_kvduo_hot_merge(
        tokens[0, 0],
        locations[0, 0],
        lru[0, 0],
        backing,
        workspace,
        capacity=16,
        k=4,
        top_k=1,
        target_pages=2,
        paged=False,
        **{name: bank[0, 0] for name, bank in banks.items()},
    )
    with pytest.raises((ValueError, RuntimeError)):
        finish_kvduo_hot_merge(workspace, 4, 2, 4)
    assert all(
        torch.equal(t, saved)
        for t, saved in zip(
            (tokens, locations, lru, backing, *banks.values()), snapshots
        )
    )
