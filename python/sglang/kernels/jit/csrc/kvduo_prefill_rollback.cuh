#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#ifdef USE_ROCM
#include <hipcub/hipcub.hpp>
namespace rollback_cub = hipcub;
#else
#include <cub/block/block_scan.cuh>
namespace rollback_cub = cub;
#endif

namespace {

template <int CAPACITY, int BLOCK_SIZE>
__global__ void kvduo_prefill_rollback_kernel(
    const int64_t* owners,
    const int32_t* targets,
    int32_t* tokens,
    int32_t* locations,
    int16_t* lru,
    int64_t* page_touch,
    int32_t* capacities,
    int requests,
    int pages,
    int k) {
  constexpr int ITEMS = (CAPACITY + BLOCK_SIZE - 1) / BLOCK_SIZE;
  using Scan = rollback_cub::BlockScan<int, BLOCK_SIZE>;
  __shared__ typename Scan::TempStorage scratch;
  extern __shared__ int16_t compacted[];
  const int tid = threadIdx.x;
  const int owner = owners[blockIdx.x * 2] * requests + owners[blockIdx.x * 2 + 1];
  const int target = targets[blockIdx.x];
  const int old_capacity = capacities[owner];
  int16_t slots[ITEMS];
  int flags[ITEMS], offsets[ITEMS];
  // Blocked ordering gives a stable scan across the full old permutation.
  for (int i = 0; i < ITEMS; ++i) {
    const int rank = tid * ITEMS + i;
    slots[i] = rank < old_capacity ? lru[owner * CAPACITY + rank] : -1;
    flags[i] = slots[i] >= 0 && slots[i] < target;
  }
  Scan(scratch).ExclusiveSum(flags, offsets);
  for (int i = 0; i < ITEMS; ++i) {
    if (flags[i]) compacted[offsets[i]] = slots[i];
  }
  __syncthreads();
  for (int rank = tid; rank < CAPACITY; rank += BLOCK_SIZE) {
    lru[owner * CAPACITY + rank] = rank < target ? compacted[rank] : -1;
    if (rank >= target) {
      tokens[owner * CAPACITY + rank] = -1;
      locations[owner * CAPACITY + rank] = -1;
    }
  }
  for (int page = target / k + tid; page < pages; page += BLOCK_SIZE) {
    page_touch[owner * pages + page] = 0;
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
    tvm::ffi::TensorView page_touch,
    tvm::ffi::TensorView capacities,
    int64_t slots_per_page) {
  using namespace host;
  auto n = SymbolicSize{"objects"};
  auto layers = SymbolicSize{"layers"};
  auto requests = SymbolicSize{"requests"};
  auto pages = SymbolicSize{"pages"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA, kDLROCM>();
  TensorMatcher({n, 2}).with_dtype<int64_t>().with_device(device).verify(owners);
  TensorMatcher({n}).with_dtype<int32_t>().with_device(device).verify(targets);
  TensorMatcher({layers, requests, CAPACITY})
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(tokens)
      .verify(locations);
  TensorMatcher({layers, requests, CAPACITY}).with_dtype<int16_t>().with_device(device).verify(lru);
  TensorMatcher({layers, requests, pages}).with_dtype<int64_t>().with_device(device).verify(page_touch);
  TensorMatcher({layers, requests}).with_dtype<int32_t>().with_device(device).verify(capacities);
  constexpr size_t smem_bytes = CAPACITY * sizeof(int16_t);
#ifndef USE_ROCM
  if constexpr (smem_bytes > 46u * 1024u) {
    cudaFuncSetAttribute(
        kvduo_prefill_rollback_kernel<CAPACITY, BLOCK_SIZE>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  }
#endif
  LaunchKernel(n.unwrap(), BLOCK_SIZE, device.unwrap(), smem_bytes)(
      kvduo_prefill_rollback_kernel<CAPACITY, BLOCK_SIZE>,
      static_cast<const int64_t*>(owners.data_ptr()),
      static_cast<const int32_t*>(targets.data_ptr()),
      static_cast<int32_t*>(tokens.data_ptr()),
      static_cast<int32_t*>(locations.data_ptr()),
      static_cast<int16_t*>(lru.data_ptr()),
      static_cast<int64_t*>(page_touch.data_ptr()),
      static_cast<int32_t*>(capacities.data_ptr()),
      static_cast<int>(requests.unwrap()),
      static_cast<int>(pages.unwrap()),
      static_cast<int>(slots_per_page));
}

}  // namespace
