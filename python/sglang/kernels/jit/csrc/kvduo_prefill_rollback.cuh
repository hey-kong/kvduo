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

struct KVDuoRollbackStrides {
  int64_t layer;
  int64_t request;
};

template <int CAPACITY, int BLOCK_SIZE>
__global__ void kvduo_prefill_rollback_kernel(
    const int64_t* owners,
    const int32_t* targets,
    int32_t* tokens,
    int32_t* locations,
    int16_t* lru,
    bool* visited,
    int32_t* capacities,
    int requests,
    int64_t padded_capacity,
    KVDuoRollbackStrides token_strides,
    KVDuoRollbackStrides location_strides,
    KVDuoRollbackStrides lru_strides,
    KVDuoRollbackStrides visited_strides) {
  constexpr int ITEMS = (CAPACITY + BLOCK_SIZE - 1) / BLOCK_SIZE;
  using Scan = rollback_cub::BlockScan<int, BLOCK_SIZE>;
  __shared__ typename Scan::TempStorage scratch;
  extern __shared__ int16_t compacted[];
  const int tid = threadIdx.x;
  const int64_t layer = owners[blockIdx.x * 2];
  const int64_t request = owners[blockIdx.x * 2 + 1];
  const int64_t owner = layer * requests + request;
  tokens += layer * token_strides.layer + request * token_strides.request;
  locations += layer * location_strides.layer + request * location_strides.request;
  lru += layer * lru_strides.layer + request * lru_strides.request;
  visited += layer * visited_strides.layer + request * visited_strides.request;
  const int target = targets[blockIdx.x];
  const int old_capacity = capacities[owner];
  int16_t slots[ITEMS];
  int flags[ITEMS], offsets[ITEMS];
  // Blocked ordering gives a stable scan across the full old permutation.
  for (int i = 0; i < ITEMS; ++i) {
    const int rank = tid * ITEMS + i;
    slots[i] = rank < old_capacity ? lru[rank] : -1;
    flags[i] = slots[i] >= 0 && slots[i] < target;
  }
  Scan(scratch).ExclusiveSum(flags, offsets);
  for (int i = 0; i < ITEMS; ++i) {
    if (flags[i]) compacted[offsets[i]] = slots[i];
  }
  __syncthreads();
  for (int rank = tid; rank < CAPACITY; rank += BLOCK_SIZE) {
    lru[rank] = rank < target ? compacted[rank] : -1;
  }
  // These banks include coordinator padding that is absent from LRU.
  for (int64_t slot = target + tid; slot < padded_capacity; slot += BLOCK_SIZE) {
    tokens[slot] = -1;
    locations[slot] = -1;
    visited[slot] = false;
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
    tvm::ffi::TensorView visited,
    tvm::ffi::TensorView capacities) {
  using namespace host;
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
  TensorMatcher({layers, requests, CAPACITY}).with_dtype<int16_t>().with_device(device).verify(lru);
  // bool is integral in C++; TensorMatcher's generic integral trait maps it
  // to uint8. PyTorch exports a distinct DLPack bool8 dtype.
  TensorMatcher({layers, requests, padded_capacity})
      .with_dtype(DLDataType{kDLBool, 8, 1})
      .with_device(device)
      .verify(visited);
  TensorMatcher({layers, requests}).with_dtype<int32_t>().with_device(device).verify(capacities);
  RuntimeCheck(padded_capacity.unwrap() >= CAPACITY, "Padded metadata cannot be smaller than LRU capacity");
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
      static_cast<bool*>(visited.data_ptr()),
      static_cast<int32_t*>(capacities.data_ptr()),
      static_cast<int>(requests.unwrap()),
      padded_capacity.unwrap(),
      KVDuoRollbackStrides{tokens.stride(0), tokens.stride(1)},
      KVDuoRollbackStrides{locations.stride(0), locations.stride(1)},
      KVDuoRollbackStrides{lru.stride(0), lru.stride(1)},
      KVDuoRollbackStrides{visited.stride(0), visited.stride(1)});
}

}  // namespace
