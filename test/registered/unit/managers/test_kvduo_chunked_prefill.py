import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.srt.utils.common import Range  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
