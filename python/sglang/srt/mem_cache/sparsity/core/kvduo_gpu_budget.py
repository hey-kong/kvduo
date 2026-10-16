"""Physical-cache targets and separately budgeted logical backing for KVDuo."""

from dataclasses import dataclass

# Logical mappings and FULL version/touch/generation tensors; charged in
# addition to the model-specific indexer/history storage.
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
    used = int(physical * physical_bytes + logical * logical_bytes)
    return KVDuoGPUBudget(physical, logical, target, used)
