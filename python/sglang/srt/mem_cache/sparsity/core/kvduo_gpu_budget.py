"""Physical-cache targets and separately budgeted logical backing for KVDuo."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Logical mappings and FULL version/touch/generation tensors; charged in
# addition to the model-specific indexer/history storage. Dynamic HOT frequency
# buffers are separate allocations, not part of this per-logical-token charge.
LOGICAL_METADATA_BYTES = 64


@dataclass(frozen=True)
class KVDuoGPUBudget:
    physical_tokens: int
    logical_tokens: int
    target_physical_tokens: int
    allocated_bytes: int


def plan_gpu_budget(
    budget,
    extra_budget,
    physical_bytes,
    logical_bytes,
    baseline_bytes,
    logical_ratio,
    page_size,
    logical_tokens=0,
):
    target = int(budget // baseline_bytes) // page_size * page_size
    limit = budget + extra_budget
    # Fix the address-space target before sizing physical storage. Reducing
    # the physical pool must not silently reduce indexer/history capacity.
    logical = logical_tokens or target * logical_ratio
    logical = (logical + page_size - 1) // page_size * page_size
    physical = min(
        target, logical, int((limit - logical * logical_bytes) // physical_bytes)
    )
    physical = physical // page_size * page_size
    if physical <= 0 or logical <= 0:
        raise ValueError(
            "KVDuo GPU budget cannot fit the logical KV backing target and one physical page; "
            "reduce logical_capacity_tokens/logical_capacity_ratio or increase available memory"
        )
    desired_physical = min(target, logical)
    required_extra = max(
        0, int(desired_physical * physical_bytes + logical * logical_bytes - budget)
    )
    if physical < desired_physical:
        logger.info(
            "KVDuo physical KV capacity reduced: target=%d, allocated=%d, "
            "logical=%d, extra_required_bytes=%d, extra_available_bytes=%d, "
            "additional_headroom_needed_bytes=%d. Logical backing is preserved; "
            "indexer/history backing takes priority over physical capacity. "
            "HOT frequency metadata is allocated separately outside this budget.",
            desired_physical,
            physical,
            logical,
            required_extra,
            extra_budget,
            max(0, required_extra - extra_budget),
        )
    used = int(physical * physical_bytes + logical * logical_bytes)
    return KVDuoGPUBudget(physical, logical, target, used)
