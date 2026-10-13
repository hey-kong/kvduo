import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_cuda(),
    reason="Resolver device-assert tests require CUDA",
)


@pytest.mark.parametrize("graph_replay", [False, True])
@pytest.mark.parametrize("failure", ["missing_full", "missing_host", "stale_host"])
def test_fused_validation_rejects_bad_sources_in_isolated_process(
    failure, graph_replay
):
    # A device assertion poisons its CUDA context; isolate each negative case.
    script = r"""
import os
import sys
import torch
sys.path.insert(0, sys.argv[1])
from test_hisparse import _kvduo_observation_case
from sglang.kernels.ops.kvcache.hisparse import load_cache_to_device_buffer_mla
case = _kvduo_observation_case(selected=(1, 2))
case["validate_status"] = True
failure, replay = sys.argv[2], sys.argv[3] == "True"
if failure == "missing_full":
    case["full_to_device_loc"][[1, 2]] = torch.tensor([12, 13], device="cuda")
load_cache_to_device_buffer_mla(**case)
torch.cuda.synchronize()
if replay:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        load_cache_to_device_buffer_mla(**case)
if failure == "missing_full":
    case["req_to_logical_token"][0, 2] = -1
elif failure == "missing_host":
    case["host_cache_locs"][0, 2] = -1
else:
    case["full_host_version"][2] = 0
try:
    if replay:
        graph.replay()
    else:
        load_cache_to_device_buffer_mla(**case)
    torch.cuda.synchronize()
except RuntimeError as error:
    print("EXPECTED_DEVICE_ERROR:", error, flush=True)
    sys.stderr.flush()
    os._exit(0)
sys.exit("Invalid resolver output was accepted")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).parent),
            failure,
            str(graph_replay),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXPECTED_DEVICE_ERROR:" in result.stdout
    assert (
        "KVDuo resolver produced an incomplete address table"
        in result.stdout + result.stderr
    )
