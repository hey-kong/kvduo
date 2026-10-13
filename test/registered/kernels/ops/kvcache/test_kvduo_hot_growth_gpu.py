import runpy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from sglang.kernels.ops.kvcache.kvduo_hot_growth import publish_kvduo_hot_growth
from sglang.srt.utils import is_cuda, is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=8, stage="stage-b", runner_config="1-gpu-small-amd")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not (is_cuda() or is_hip()),
    reason="HOT publication requires CUDA/ROCm",
)


@pytest.mark.parametrize(
    "P,L,current_pages,new_pages",
    [(2, 2, 0, 2), (2, 2, 1, 3), (64, 78, 0, 1), (64, 78, 7, 8)],
)
@pytest.mark.parametrize("reset_prefill", [False, True])
def test_growth_preserves_lru_prior_and_padding(
    P, L, current_pages, new_pages, reset_prefill
):
    K = P * L
    maximum, current, target = 8 * K, current_pages * K, new_pages * K
    slot_dtype = torch.int32 if maximum > 32767 else torch.int16
    shape = (2, 3, maximum + P)
    tokens = (
        torch.arange(shape[-1], device="cuda", dtype=torch.int32).expand(shape).clone()
    )
    locations = tokens + 100
    lru = (
        torch.arange(maximum, device="cuda", dtype=slot_dtype)
        .flip(0)
        .expand(2, 3, maximum)
        .clone()
    )
    lru[1, 2, :current] = torch.arange(current, device="cuda", dtype=slot_dtype).flip(0)
    old_lru = lru[1, 2, :target].clone()
    capacity = torch.full((2, 3), current, dtype=torch.int32, device="cuda")
    ema = (
        torch.arange(9, dtype=torch.float32, device="cuda").expand(2, 3, 9).clone() + 2
    )
    steps = torch.full((2, 3), 65, dtype=torch.int64, device="cuda")
    prefill = torch.full((2, 3, 2), 7.0, device="cuda")
    prefill_steps = steps.clone()
    banks = (tokens, locations, lru, capacity, ema, steps, prefill, prefill_steps)
    before = [bank.clone() for bank in banks]
    addresses = [(bank.data_ptr(), bank.shape, bank.stride()) for bank in banks]
    pages = torch.empty(
        (new_pages - current_pages, P), device="cuda", dtype=torch.int64
    )[:, 0]
    pages.copy_(
        torch.arange(1, new_pages - current_pages + 1, device="cuda", dtype=torch.int64)
        * P
    )
    publish_kvduo_hot_growth(
        pages,
        tokens[1, 2],
        locations[1, 2],
        lru[1, 2],
        old_lru,
        capacity[1, 2].reshape(1),
        ema[1, 2],
        steps[1, 2].reshape(1),
        prefill[1, 2],
        prefill_steps[1, 2].reshape(1),
        current=current,
        target=target,
        page_size=P,
        storage_layers=L,
        layer_stride=1_000_000,
        reset_prefill=reset_prefill,
    )
    torch.cuda.synchronize()
    for bank, saved in zip(banks, before):
        assert torch.equal(bank[0], saved[0]) and torch.equal(bank[1, :2], saved[1, :2])
    assert capacity[1, 2].item() == target
    for bank, saved in zip((tokens, locations), before[:2]):
        assert torch.equal(bank[1, 2, :current], saved[1, 2, :current])
        assert torch.equal(bank[1, 2, target:], saved[1, 2, target:])
    assert torch.all(tokens[1, 2, current:target] == -1)
    assert torch.equal(lru[1, 2, target:], before[2][1, 2, target:])
    assert torch.equal(
        lru[1, 2, : target - current],
        torch.arange(current, target, device="cuda", dtype=slot_dtype),
    )
    assert torch.equal(lru[1, 2, target - current : target], old_lru[:current])
    expected_locs = torch.cat(
        [
            torch.arange(P) + int(page) + layer * 1_000_000
            for page in pages.cpu()
            for layer in range(L)
        ]
    )
    assert torch.equal(locations[1, 2, current:target].cpu().long(), expected_locs)
    if current:
        expected_ema = before[4][1, 2].clone()
        expected_ema[current_pages:new_pages] = expected_ema[current_pages - 1]
        assert torch.equal(ema[1, 2], expected_ema) and steps[1, 2].item() == 65
    else:
        assert not ema[1, 2].any() and steps[1, 2].item() == 0
    assert prefill_steps[1, 2].item() == (0 if reset_prefill else 65)
    assert torch.all(prefill[1, 2] == (0 if reset_prefill else 7))
    assert addresses == [(bank.data_ptr(), bank.shape, bank.stride()) for bank in banks]


@pytest.mark.parametrize("fail", [False, True])
def test_coordinator_growth_transaction_and_failure_restore(monkeypatch, fail):
    from sglang.srt.managers import hisparse_coordinator as module

    make_coordinator = runpy.run_path(
        str(Path(__file__).parents[3] / "unit/managers/test_kvduo_hot_growth.py")
    )["make_coordinator"]
    c, allocator = make_coordinator(layers=2, requests=2)
    names = (
        "req_device_buffer_tokens",
        "req_device_buffer_token_locs",
        "lru_slots",
        "kvduo_req_hot_capacity_gpu",
        "kvduo_capacity_ema",
        "kvduo_capacity_steps",
        "kvduo_prefill_ema",
        "kvduo_prefill_steps",
    )
    for name in names:
        setattr(c, name, getattr(c, name).cuda())
    c.device = torch.device("cuda")
    allocate = allocator.alloc.side_effect
    allocator.alloc.side_effect = lambda slots: allocate(slots).cuda()
    c._ensure_kvduo_hot_capacity_targets_batch({(0, 0): 4, (1, 1): 4})
    for owner in ((0, 0), (1, 1)):
        c._establish_kvduo_hot_base(owner, c._kvduo_req_layer_pages[owner])
        c._kvduo_prefill_budget_reference[c._kvduo_owner_key(owner)] = 2.0
    c.kvduo_capacity_ema.fill_(3)
    c.kvduo_capacity_steps.fill_(65)
    c.kvduo_prefill_ema.fill_(7)
    c.kvduo_prefill_steps.fill_(66)
    c.lru_slots[0, 0, :4] = torch.tensor(
        [2, 0, 3, 1], device="cuda", dtype=c.lru_slots.dtype
    )
    banks = [getattr(c, name) for name in names] + [c.kvduo_req_hot_capacity]
    before = [bank.clone() for bank in banks]
    pages = dict(c._kvduo_hot_page_owners)
    references = dict(c._kvduo_prefill_budget_reference)
    free = allocator.free_pages
    publish = module.publish_kvduo_hot_growth
    count = 0

    def injected(*args, **kwargs):
        nonlocal count
        count += 1
        publish(*args, **kwargs)
        if fail and count == 2:
            torch.cuda.synchronize()
            raise RuntimeError("publication failed")

    monkeypatch.setattr(module, "publish_kvduo_hot_growth", injected)
    c._observe_kvduo_hot_growth = MagicMock(wraps=c._observe_kvduo_hot_growth)
    if fail:
        with pytest.raises(RuntimeError, match="publication failed"):
            c._ensure_kvduo_hot_capacity_targets_batch({(0, 0): 8, (1, 1): 8})
        torch.cuda.synchronize()
        assert all(torch.equal(bank, saved) for bank, saved in zip(banks, before))
        assert c._kvduo_hot_page_owners == pages and allocator.free_pages == free
        assert c._kvduo_prefill_budget_reference == references
    else:
        c._ensure_kvduo_hot_capacity_targets_batch({(0, 0): 8, (1, 1): 8})
        torch.cuda.synchronize()
        assert c.lru_slots[0, 0, :8].tolist() == [4, 5, 6, 7, 2, 0, 3, 1]
        assert c.kvduo_capacity_steps[0, 0].item() == 65
        assert c.kvduo_capacity_ema[0, 0, 1].item() == 3
        assert c.kvduo_prefill_steps[0, 0].item() == 0
        assert not c._kvduo_prefill_budget_reference
        assert c._observe_kvduo_hot_growth.call_args.kwargs == {"publish_stats": False}
