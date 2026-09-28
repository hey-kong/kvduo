import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.schedule_policy import (  # noqa: E402
    AddReqResult,
    PrefillAdder,
)
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.srt.utils.common import Range  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_host_restore_waits_for_one_page_of_chunk_budget():
    adder = PrefillAdder.__new__(PrefillAdder)
    adder.dsa_prefill_cp_in_seq_split = False
    adder.prefill_max_requests = None
    adder.can_run_list = []
    adder.page_size = 4
    adder.rem_chunk_tokens = 3
    adder.rem_input_tokens = 100
    adder.rem_total_token_offset = 0
    adder.cur_rem_token_offset = 0
    adder._mamba_slot_cost = 0
    adder.is_all_swa = False
    adder.is_hybrid_swa = False
    adder.is_hybrid_ssm_cache = False
    adder.dllm_config = None
    adder.prefill_delayer_single_pass = None
    adder.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 100)
    adder.tree_cache = MagicMock(
        disable=False,
        evictable_size=MagicMock(return_value=0),
        is_tree_cache=MagicMock(return_value=False),
    )
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(ignore_eos=False, max_new_tokens=0),
        output_ids=[],
        full_untruncated_fill_ids=list(range(10)),
        prefix_indices=torch.tensor([1, 2], dtype=torch.int64),
        host_hit_length=4,
        swa_host_hit_length=0,
        mamba_pool_idx=None,
        last_node=object(),
    )

    result = adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)

    assert result is AddReqResult.OTHER
    adder.tree_cache.init_load_back.assert_not_called()
    assert adder.token_to_kv_pool_allocator.available_size() == 100


def test_kvduo_keeps_chunked_prefill_enabled():
    args = SimpleNamespace(
        enable_kvduo=True,
        enable_hisparse=False,
        kvduo_config='{"top_k": 512}',
        chunked_prefill_size=4096,
    )

    ServerArgs._handle_kvduo_aliases(args)

    assert args.chunked_prefill_size == 4096
    assert args.enable_hisparse is True
    assert json.loads(args.hisparse_config) == {
        "top_k": 512,
        "device_buffer_size": 8192,
        "host_to_device_ratio": 2,
        "swap_in_block_size": 960,
    }


def test_kvduo_stashes_chunk_without_publishing_to_radix_tree():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tree_cache = MagicMock()
    scheduler.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.int32)
    )
    req = SimpleNamespace(
        req_pool_idx=0,
        extend_range=Range(2, 4),
        prefix_indices=torch.tensor([10, 11], dtype=torch.int64),
        cache_protected_len=2,
    )

    with patch(
        "sglang.srt.managers.scheduler.get_memory",
        return_value=SimpleNamespace(enable_kvduo=True),
    ):
        scheduler.stash_chunked_request(req)

    assert req.prefix_indices.tolist() == [10, 11, 12, 13]
    assert req.prefix_indices.dtype == torch.int64
    assert req.cache_protected_len == 2
    scheduler.tree_cache.cache_unfinished_req.assert_not_called()


def test_regular_chunk_stash_does_not_require_scheduler_kvduo_attribute():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.tree_cache = MagicMock()
    req = MagicMock()

    with (
        patch(
            "sglang.srt.managers.scheduler.get_memory",
            return_value=SimpleNamespace(enable_kvduo=False),
        ),
        patch(
            "sglang.srt.managers.scheduler.maybe_cache_unfinished_req"
        ) as cache_unfinished,
    ):
        scheduler.stash_chunked_request(req)

    cache_unfinished.assert_called_once_with(req, scheduler.tree_cache, chunked=True)
