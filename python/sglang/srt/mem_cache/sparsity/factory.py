import json
import logging
from typing import Optional

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_dsa import DeepSeekDSAAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    DSABackendAdaptor,
    FlashAttentionAdaptor,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
    KVDuoConfig,
    SparseConfig,
    SparseCoordinator,
)

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 2048
DEFAULT_DEVICE_BUFFER_SIZE = 4096
DEFAULT_HOST_TO_DEVICE_RATIO = 2
DEFAULT_SWAP_IN_BLOCK_SIZE = 960

_global_sparse_coordinator: Optional[SparseCoordinator] = None

_ALGORITHM_REGISTRY = {
    "quest": lambda config, device, **kw: QuestAlgorithm(config, device, **kw),
    "deepseek_dsa": lambda config, device, **kw: DeepSeekDSAAlgorithm(
        config, device, **kw
    ),
}


def _create_sparse_algorithm(
    config: SparseConfig,
    device: torch.device,
    **kwargs,
) -> BaseSparseAlgorithm:
    algorithm_name = config.algorithm.lower()
    factory = _ALGORITHM_REGISTRY.get(algorithm_name)

    if factory is None:
        raise ValueError(f"Unknown sparse algorithm: {algorithm_name}")

    return factory(config, device, **kwargs)


def _create_backend_adaptor(
    backend: str,
    device: torch.device,
    sparse_algorithm: BaseSparseAlgorithm,
    req_to_token_pool,
):
    """Create backend adaptor."""
    if isinstance(sparse_algorithm, DeepSeekDSAAlgorithm):
        return DSABackendAdaptor(device, req_to_token_pool)

    if backend in ["fa3", "flashattention"]:
        return FlashAttentionAdaptor(device)

    raise ValueError(f"Unknown attention backend: {backend}")


def _parse_sparse_config(server_args) -> SparseConfig:
    """Parse hierarchical sparse config from JSON string.

    Required fields with defaults: top_k (2048), device_buffer_size (2*top_k),
    host_to_device_ratio (2), swap_in_block_size (960).
    Optional fields (default None): algorithm, backend, min_sparse_prompt_len,
    page_size. All remaining fields go to sparse_extra_config.
    """
    extra_config_str = server_args.hisparse_config
    if extra_config_str is not None:
        try:
            extra_config = json.loads(extra_config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse hisparse_config: {e}") from e
    else:
        extra_config = {}

    top_k = extra_config.pop("top_k", 2048)
    device_buffer_size = extra_config.pop("device_buffer_size", 2 * top_k)
    host_to_device_ratio = extra_config.pop("host_to_device_ratio", 2)
    swap_in_block_size = extra_config.pop("swap_in_block_size", 960)

    if device_buffer_size < top_k:
        raise ValueError(
            f"device_buffer_size ({device_buffer_size}) must be no smaller than top_k ({top_k})"
        )
    if not isinstance(swap_in_block_size, int) or isinstance(swap_in_block_size, bool):
        raise ValueError(
            f"swap_in_block_size must be an integer, got {swap_in_block_size!r}"
        )
    if swap_in_block_size <= 0 or swap_in_block_size > 1024:
        raise ValueError(
            f"swap_in_block_size ({swap_in_block_size}) must be in the range [1, 1024]"
        )

    algorithm = extra_config.pop("algorithm", None)
    backend = extra_config.pop("backend", None)
    min_sparse_prompt_len = extra_config.pop("min_sparse_prompt_len", None)
    page_size = extra_config.pop("page_size", None)

    return SparseConfig(
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        host_to_device_ratio=host_to_device_ratio,
        swap_in_block_size=swap_in_block_size,
        algorithm=algorithm,
        backend=backend,
        page_size=page_size,
        min_sparse_prompt_len=min_sparse_prompt_len,
        sparse_extra_config=extra_config,
    )


def parse_hisparse_config(server_args) -> SparseConfig:
    """Parse hisparse config from server_args, returning defaults if no config provided."""
    return _parse_sparse_config(server_args)


def parse_kvduo_config(server_args) -> KVDuoConfig:
    """Parse KVDuo JSON without mutation.

    ``swap_in_block_size`` is the resolver CUDA thread-block size (threads),
    not the number of KV entries transferred by one host-to-device operation.
    """
    raw = server_args.kvduo_config
    if raw is None:
        values = {}
    else:
        try:
            values = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse kvduo_config: {e}") from e
        if not isinstance(values, dict):
            raise ValueError("kvduo_config must be a JSON object")

    known = {
        "top_k",
        "min_device_buffer_size",
        "host_to_device_ratio",
        "swap_in_block_size",
        "tail_protected_pages",
        "N",
    }
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown kvduo_config field(s): {sorted(unknown)}")
    if "N" in values and "tail_protected_pages" in values:
        raise ValueError("Specify only one of N and tail_protected_pages")

    top_k = values.get("top_k", DEFAULT_TOP_K)
    minimum = values.get("min_device_buffer_size", DEFAULT_DEVICE_BUFFER_SIZE)
    ratio = values.get("host_to_device_ratio", DEFAULT_HOST_TO_DEVICE_RATIO)
    block_size = values.get("swap_in_block_size", DEFAULT_SWAP_IN_BLOCK_SIZE)
    tail_pages = values.get("tail_protected_pages", values.get("N", 2))
    fields = {
        "top_k": top_k,
        "min_device_buffer_size": minimum,
        "host_to_device_ratio": ratio,
        "swap_in_block_size": block_size,
        "tail_protected_pages": tail_pages,
    }
    for name, value in fields.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if minimum < top_k:
        raise ValueError(
            f"min_device_buffer_size ({minimum}) must be no smaller than top_k ({top_k})"
        )
    if block_size > 1024:
        raise ValueError(
            f"swap_in_block_size ({block_size}) must be in the range [1, 1024]"
        )

    return KVDuoConfig(
        top_k=top_k,
        min_device_buffer_size=minimum,
        host_to_device_ratio=ratio,
        swap_in_block_size=block_size,
        tail_protected_pages=tail_pages,
    )


def create_sparse_coordinator(
    device: torch.device,
    req_to_token_pool,
    token_to_kv_pool,
    start_layer: int,
    end_layer: int,
    server_args,
    **kwargs,
) -> SparseCoordinator:
    config = _parse_sparse_config(server_args)
    algorithm = _create_sparse_algorithm(config, device, **kwargs)
    backend_adaptor = _create_backend_adaptor(
        config.backend, device, algorithm, req_to_token_pool
    )

    coordinator = SparseCoordinator(
        config=config,
        algorithm=algorithm,
        backend_adaptor=backend_adaptor,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        start_layer=start_layer,
        end_layer=end_layer,
        device=device,
    )
    register_sparse_coordinator(coordinator)
    return coordinator


def register_sparse_coordinator(coordinator: SparseCoordinator) -> None:
    global _global_sparse_coordinator
    _global_sparse_coordinator = coordinator


def get_sparse_coordinator() -> Optional[SparseCoordinator]:
    return _global_sparse_coordinator
