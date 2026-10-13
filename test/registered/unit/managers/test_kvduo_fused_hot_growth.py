import runpy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers import hisparse_coordinator as module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class DevicePages(torch.Tensor):
    """Select the CUDA control path while executing the publication CPU oracle."""

    @property
    def is_cuda(self):
        return True


@pytest.mark.parametrize("fail", [False, True])
def test_fused_growth_control_path_commits_or_restores_all_owners(monkeypatch, fail):
    make_coordinator = runpy.run_path(
        str(Path(__file__).with_name("test_kvduo_hot_growth.py"))
    )["make_coordinator"]
    c, allocator = make_coordinator(layers=2, requests=2)
    c._ensure_kvduo_hot_capacity_targets_batch({(0, 0): 4, (1, 1): 4})
    for owner in ((0, 0), (1, 1)):
        c._establish_kvduo_hot_base(owner, c._kvduo_req_layer_pages[owner])
        c._kvduo_prefill_budget_reference[c._kvduo_owner_key(owner)] = 2.0
        c.req_device_buffer_tokens[owner][:4] = torch.tensor([8, 9, 10, 11])
        c.lru_slots[owner][:4] = torch.tensor([2, 0, 3, 1], dtype=torch.int16)
    c.kvduo_capacity_ema.fill_(3)
    c.kvduo_capacity_steps.fill_(65)
    c.kvduo_prefill_ema.fill_(7)
    c.kvduo_prefill_steps.fill_(66)
    banks = (
        c.req_device_buffer_tokens,
        c.req_device_buffer_token_locs,
        c.lru_slots,
        c.kvduo_req_hot_capacity,
        c.kvduo_req_hot_capacity_gpu,
        c.kvduo_capacity_ema,
        c.kvduo_capacity_steps,
        c.kvduo_prefill_ema,
        c.kvduo_prefill_steps,
    )
    before = [bank.clone() for bank in banks]
    free, owners = allocator.free_pages, dict(c._kvduo_hot_page_owners)
    references = dict(c._kvduo_prefill_budget_reference)
    allocate = allocator.alloc.side_effect
    allocator.alloc.side_effect = lambda slots: allocate(slots).as_subclass(DevicePages)
    publish = module.publish_kvduo_hot_growth
    calls = []

    def publication(*args, **kwargs):
        calls.append(kwargs)
        publish(*args, **kwargs)
        if fail and len(calls) == 2:
            raise RuntimeError("fused publication failed")

    monkeypatch.setattr(module, "publish_kvduo_hot_growth", publication)
    observe = MagicMock(wraps=c._observe_kvduo_hot_growth)
    c._observe_kvduo_hot_growth = observe
    if fail:
        with pytest.raises(RuntimeError, match="fused publication failed"):
            c._ensure_kvduo_hot_capacity_targets_batch({(0, 0): 8, (1, 1): 8})
        assert all(torch.equal(bank, old) for bank, old in zip(banks, before))
        assert c._kvduo_hot_page_owners == owners and allocator.free_pages == free
        assert c._kvduo_prefill_budget_reference == references
    else:
        c._ensure_kvduo_hot_capacity_targets_batch({(0, 0): 8, (1, 1): 8})
        for owner in ((0, 0), (1, 1)):
            assert c.lru_slots[owner][:8].tolist() == [4, 5, 6, 7, 2, 0, 3, 1]
            assert c.req_device_buffer_tokens[owner][:4].tolist() == [8, 9, 10, 11]
            assert torch.all(c.req_device_buffer_tokens[owner][4:8] == -1)
            assert c.kvduo_capacity_steps[owner].item() == 65
            assert c.kvduo_capacity_ema[owner].tolist() == [3] * 9
            assert c.kvduo_prefill_steps[owner].item() == 0
            assert c.kvduo_req_hot_capacity[owner].item() == 8
            assert c.kvduo_req_hot_capacity_gpu[owner].item() == 8
        assert not c._kvduo_prefill_budget_reference
        assert all(
            call.kwargs == {"publish_stats": False} for call in observe.call_args_list
        )
    assert len(calls) == 2 and all(call["reset_prefill"] for call in calls)


def test_publication_rejects_aliased_lru_snapshot_before_mutating_metadata():
    make_coordinator = runpy.run_path(
        str(Path(__file__).with_name("test_kvduo_hot_growth.py"))
    )["make_coordinator"]
    c, _ = make_coordinator()
    c._ensure_kvduo_hot_capacity_targets_batch({(0, 0): 4})
    owner = (0, 0)
    banks = (
        c.req_device_buffer_tokens,
        c.req_device_buffer_token_locs,
        c.lru_slots,
        c.kvduo_req_hot_capacity_gpu,
        c.kvduo_capacity_ema,
        c.kvduo_capacity_steps,
        c.kvduo_prefill_ema,
        c.kvduo_prefill_steps,
    )
    before = [bank.clone() for bank in banks]
    with pytest.raises(ValueError, match="independent LRU snapshot"):
        module.publish_kvduo_hot_growth(
            torch.tensor([4], dtype=torch.int64),
            c.req_device_buffer_tokens[owner],
            c.req_device_buffer_token_locs[owner],
            c.lru_slots[owner],
            c.lru_slots[owner][:8],
            c.kvduo_req_hot_capacity_gpu[owner].reshape(1),
            c.kvduo_capacity_ema[owner],
            c.kvduo_capacity_steps[owner].reshape(1),
            c.kvduo_prefill_ema[owner],
            c.kvduo_prefill_steps[owner].reshape(1),
            current=4,
            target=8,
            page_size=2,
            storage_layers=2,
            layer_stride=10000,
            reset_prefill=False,
        )
    assert all(torch.equal(bank, old) for bank, old in zip(banks, before))
