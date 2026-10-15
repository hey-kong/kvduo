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
):
    target = int(budget // baseline_bytes) // page_size * page_size
    limit = budget + extra_budget
    physical = min(
        target, int(limit // (physical_bytes + logical_ratio * logical_bytes))
    )
    physical = physical // page_size * page_size
    logical = physical * logical_ratio
    if physical <= 0:
        raise ValueError("KVDuo GPU budget cannot fit physical and logical KV backing")
    used = int(physical * physical_bytes + logical * logical_bytes)
    return KVDuoGPUBudget(physical, logical, target, used)
