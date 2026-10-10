#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <type_traits>
#ifdef USE_ROCM
#include <hipcub/hipcub.hpp>
namespace rollback_cub = hipcub;
#else
#include <cub/block/block_scan.cuh>
namespace rollback_cub = cub;
#endif

namespace {

struct KVDuoRollbackStrides {
  int64_t layer;
  int64_t request;
};

struct KVDuoRollbackStats {
  float* capacity_ema = nullptr;
  int64_t* capacity_steps = nullptr;
  int32_t* growth_events = nullptr;
  float* prefill_ema = nullptr;
  int64_t* prefill_steps = nullptr;
};

template <int CAPACITY>
using KVDuoRollbackSlot = std::conditional_t<(CAPACITY > INT16_MAX), int32_t, int16_t>;

template <int CAPACITY, int BLOCK_SIZE>
__global__ void kvduo_prefill_rollback_kernel(
    const int64_t* owners,
    const int32_t* targets,
    int32_t* tokens,
    int32_t* locations,
    KVDuoRollbackSlot<CAPACITY>* lru,
    int32_t* capacities,
    int requests,
    int64_t padded_capacity,
    KVDuoRollbackStrides token_strides,
    KVDuoRollbackStrides location_strides,
    KVDuoRollbackStrides lru_strides,
    KVDuoRollbackStats stats) {
  constexpr int ITEMS = (CAPACITY + BLOCK_SIZE - 1) / BLOCK_SIZE;
  using SlotT = KVDuoRollbackSlot<CAPACITY>;
  using Scan = rollback_cub::BlockScan<int, BLOCK_SIZE>;
  __shared__ typename Scan::TempStorage scratch;
  const int tid = threadIdx.x;
  const int64_t layer = owners[blockIdx.x * 2];
  const int64_t request = owners[blockIdx.x * 2 + 1];
  const int64_t owner = layer * requests + request;
  tokens += layer * token_strides.layer + request * token_strides.request;
  locations += layer * location_strides.layer + request * location_strides.request;
  lru += layer * lru_strides.layer + request * lru_strides.request;
  const int target = targets[blockIdx.x];
  const int old_capacity = capacities[owner];
  SlotT slots[ITEMS];
  int flags[ITEMS], offsets[ITEMS];
  // Blocked ordering gives a stable scan across the full old permutation.
  for (int i = 0; i < ITEMS; ++i) {
    const int rank = tid * ITEMS + i;
    slots[i] = rank < old_capacity ? lru[rank] : -1;
    flags[i] = slots[i] >= 0 && slots[i] < target;
  }
  Scan(scratch).ExclusiveSum(flags, offsets);
  // Every source slot is already in this CTA's register/local snapshot.
  // Wait for all readers before writing directly into the same LRU bank;
  // no capacity-sized shared output buffer is needed, even for int32 slots.
  __syncthreads();
  for (int i = 0; i < ITEMS; ++i) {
    if (flags[i]) lru[offsets[i]] = slots[i];
  }
  __syncthreads();
  for (int rank = target + tid; rank < CAPACITY; rank += BLOCK_SIZE) {
    lru[rank] = -1;
  }
  // These banks include coordinator padding that is absent from LRU.
  for (int64_t slot = target + tid; slot < padded_capacity; slot += BLOCK_SIZE) {
    tokens[slot] = -1;
    locations[slot] = -1;
  }
  if (stats.capacity_ema != nullptr) {
    if (tid < 9) stats.capacity_ema[owner * 9 + tid] = 0;
    if (tid < 2) stats.prefill_ema[owner * 2 + tid] = 0;
    if (tid == 0) {
      stats.capacity_steps[owner] = 0;
      stats.growth_events[owner] = 0;
      stats.prefill_steps[owner] = 0;
    }
  }
  __syncthreads();
  if (tid == 0) capacities[owner] = target;
}

template <int CAPACITY, int BLOCK_SIZE>
void kvduo_prefill_rollback(
    tvm::ffi::TensorView owners,
    tvm::ffi::TensorView targets,
    tvm::ffi::TensorView tokens,
    tvm::ffi::TensorView locations,
    tvm::ffi::TensorView lru,
    tvm::ffi::TensorView capacities,
    tvm::ffi::Optional<tvm::ffi::TensorView> capacity_ema,
    tvm::ffi::Optional<tvm::ffi::TensorView> capacity_steps,
    tvm::ffi::Optional<tvm::ffi::TensorView> growth_events,
    tvm::ffi::Optional<tvm::ffi::TensorView> prefill_ema,
    tvm::ffi::Optional<tvm::ffi::TensorView> prefill_steps) {
  using namespace host;
  using SlotT = KVDuoRollbackSlot<CAPACITY>;
  auto n = SymbolicSize{"objects"};
  auto layers = SymbolicSize{"layers"};
  auto requests = SymbolicSize{"requests"};
  auto padded_capacity = SymbolicSize{"padded_capacity"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA, kDLROCM>();
  TensorMatcher({n, 2}).with_dtype<int64_t>().with_device(device).verify(owners);
  TensorMatcher({n}).with_dtype<int32_t>().with_device(device).verify(targets);
  TensorMatcher({layers, requests, padded_capacity})
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(tokens)
      .verify(locations);
  TensorMatcher({layers, requests, CAPACITY}).with_dtype<SlotT>().with_device(device).verify(lru);
  TensorMatcher({layers, requests}).with_dtype<int32_t>().with_device(device).verify(capacities);
  RuntimeCheck(padded_capacity.unwrap() >= CAPACITY, "Padded metadata cannot be smaller than LRU capacity");
  const bool reset = capacity_ema.has_value();
  RuntimeCheck(
      reset == capacity_steps.has_value() && reset == growth_events.has_value() && reset == prefill_ema.has_value() &&
          reset == prefill_steps.has_value(),
      "Rollback statistics must be supplied together");
  KVDuoRollbackStats stats;
  if (reset) {
    TensorMatcher({layers, requests, 9}).with_dtype<float>().with_device(device).verify(capacity_ema.value());
    TensorMatcher({layers, requests})
        .with_dtype<int64_t>()
        .with_device(device)
        .verify(capacity_steps.value())
        .verify(prefill_steps.value());
    TensorMatcher({layers, requests}).with_dtype<int32_t>().with_device(device).verify(growth_events.value());
    TensorMatcher({layers, requests, 2}).with_dtype<float>().with_device(device).verify(prefill_ema.value());
    stats = KVDuoRollbackStats{
        static_cast<float*>(capacity_ema.value().data_ptr()),
        static_cast<int64_t*>(capacity_steps.value().data_ptr()),
        static_cast<int32_t*>(growth_events.value().data_ptr()),
        static_cast<float*>(prefill_ema.value().data_ptr()),
        static_cast<int64_t*>(prefill_steps.value().data_ptr())};
  }
  LaunchKernel(n.unwrap(), BLOCK_SIZE, device.unwrap())(
      kvduo_prefill_rollback_kernel<CAPACITY, BLOCK_SIZE>,
      static_cast<const int64_t*>(owners.data_ptr()),
      static_cast<const int32_t*>(targets.data_ptr()),
      static_cast<int32_t*>(tokens.data_ptr()),
      static_cast<int32_t*>(locations.data_ptr()),
      static_cast<SlotT*>(lru.data_ptr()),
      static_cast<int32_t*>(capacities.data_ptr()),
      static_cast<int>(requests.unwrap()),
      padded_capacity.unwrap(),
      KVDuoRollbackStrides{tokens.stride(0), tokens.stride(1)},
      KVDuoRollbackStrides{locations.stride(0), locations.stride(1)},
      KVDuoRollbackStrides{lru.stride(0), lru.stride(1)},
      stats);
}

}  // namespace
