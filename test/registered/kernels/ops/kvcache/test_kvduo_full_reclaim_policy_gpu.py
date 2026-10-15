import pytest
import torch

from sglang.kernels.ops.kvcache.kvduo_full_reclaim import select_kvduo_full_requests
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_cuda(),
    reason="Request FULL selection requires CUDA",
)


@pytest.mark.parametrize("needed", [1, 129, 10000])
@pytest.mark.parametrize("release_all", [False, True])
@pytest.mark.parametrize("charged", [False, True])
@pytest.mark.parametrize(
    "mixed", [[False] * 9, [True, False, True, False, True, False, True, False, True]]
)
def test_request_selection_matches_cpu_across_tiles_and_interleaved_owners(
    needed, release_all, charged, mixed
):
    generator = torch.Generator().manual_seed(7712)
    owners = torch.randint(0, 9, (1025,), generator=generator).tolist()
    costs = [0, 10, 120, 5, 3, 20, 200, 0, 90] if charged else [0] * 9
    ages = [7, 3, 8, 5, 0, 4, 6, 2, 1]
    for _ in range(3):
        valid = (torch.rand(len(owners), generator=generator) > 0.25).int()
        expected = select_kvduo_full_requests(
            valid,
            owners,
            costs,
            ages,
            needed,
            release_all=release_all,
            mixed_owners=mixed,
        )
        assert (
            select_kvduo_full_requests(
                valid.cuda(),
                owners,
                costs,
                ages,
                needed,
                release_all=release_all,
                mixed_owners=mixed,
            )
            == expected
        )


@pytest.mark.parametrize("ages", [[3, 2, 1], [1, 1, 1]])
def test_equal_net_gains_prefer_decode_age_then_stable_owner(ages):
    owners = [0] * 130 + [1] * 131 + [2] * 132
    costs = [1, 2, 3]
    valid = torch.ones(len(owners), dtype=torch.int32)
    expected = select_kvduo_full_requests(valid, owners, costs, ages, 1)
    assert select_kvduo_full_requests(valid.cuda(), owners, costs, ages, 1) == expected


@pytest.mark.parametrize("release_all", [False, True])
@pytest.mark.parametrize("no_valid_pages", [False, True])
def test_request_selection_without_positive_net_gain_returns_no_pages(
    release_all, no_valid_pages
):
    valid = torch.full(
        (512,), 0 if no_valid_pages else 1, dtype=torch.int32, device="cuda"
    )
    assert (
        select_kvduo_full_requests(
            valid,
            [0] * 256 + [1] * 256,
            [256, 256],
            [2, 1],
            1,
            release_all=release_all,
        )
        == []
    )


def test_request_selection_download_is_bounded_by_whole_owner_overshoot(monkeypatch):
    valid = torch.ones(8192, dtype=torch.int32, device="cuda")
    owners = [i // 1024 for i in range(valid.numel())]
    downloads = []
    original = torch.Tensor.cpu

    def download(tensor, *args, **kwargs):
        downloads.append(tensor.numel())
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", download)
    assert select_kvduo_full_requests(
        valid, owners, [1] * 8, list(range(8)), 1
    ) == list(range(1024))
    assert downloads == [1033]


@pytest.mark.parametrize("costs,expected", [([0, 0], [0, 1, 2]), ([1, 0], [3, 4])])
def test_net_gain_precedes_residency_and_residency_breaks_gain_ties(costs, expected):
    assert (
        select_kvduo_full_requests(
            torch.ones(5, dtype=torch.int32, device="cuda"),
            [0, 0, 0, 1, 1],
            costs,
            [0, 99],
            1,
            mixed_owners=[False, True],
        )
        == expected
    )
