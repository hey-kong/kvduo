#include <sgl_kernel/kvduo_capacity_plan.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#ifdef USE_ROCM
#include <hipcub/hipcub.hpp>
namespace hand_cub = hipcub;
#else
#include <cub/block/block_scan.cuh>
namespace hand_cub = cub;
#endif

namespace {

template <int BLOCK>
__global__ void kvduo_hand_targets_kernel(
    const float* counts,
    const int64_t* steps,
    const int32_t* pages,
    int32_t* targets,
    int64_t layers,
    int64_t requests,
    int64_t hand,
    int minimum,
    double rho) {
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;
  const int64_t objects = layers * requests;
  if (offset >= objects) return;
  const int64_t position = (hand + offset) % objects;
  const int64_t owner = (position % layers) * requests + position / layers;
  const int n = pages[owner];
  targets[offset] =
      n == 0 ? -1
             : sglang::kvduo_capacity_target(counts + owner * 9, counts[owner * 9 + 8], steps[owner], n, minimum, rho);
}

template <int BLOCK>
__global__ void kvduo_hand_packet_kernel(
    const int32_t* pages,
    const int32_t* targets,
    int64_t* packet,
    int64_t layers,
    int64_t requests,
    int64_t hand,
    int64_t cursor,
    int64_t needed) {
  using Scan = hand_cub::BlockScan<int, BLOCK>;
  __shared__ typename Scan::TempStorage scratch;
  __shared__ int previous_count, previous_free, cutoff;
  const int tid = threadIdx.x;
  const int64_t objects = layers * requests;
  if (tid == 0) {
    packet[0] = packet[1] = 0;
    previous_count = previous_free = 0;
  }
  __syncthreads();
  for (int64_t start = cursor; start < objects; start += BLOCK) {
    const int64_t offset = start + tid;
    const int64_t position = (hand + offset) % objects;
    const int64_t owner = (position % layers) * requests + position / layers;
    const int target = offset < objects ? targets[offset] : -1;
    if (target == -2) atomicExch(reinterpret_cast<unsigned long long*>(packet + 1), 1ull);
    const int eligible = target >= 0;
    const int n = eligible ? pages[owner] : 0;
    int rank, total_count;
    Scan(scratch).ExclusiveSum(eligible, rank, total_count);
    __syncthreads();
    int released_prefix, total_released;
    Scan(scratch).InclusiveSum(eligible ? n - target : 0, released_prefix, total_released);
    if (tid == 0) cutoff = BLOCK - 1;
    __syncthreads();
    const bool enough = static_cast<int64_t>(previous_free) + released_prefix >= needed;
    if (enough) atomicMin(&cutoff, tid);
    const bool stop = __syncthreads_or(enough);
    if (eligible && tid <= cutoff) {
      const int64_t index = previous_count + rank;
      int64_t* row = packet + 2 + index * 4;
      row[0] = offset;
      row[1] = owner;
      row[2] = n;
      row[3] = target;
      atomicMax(reinterpret_cast<unsigned long long*>(packet), static_cast<unsigned long long>(index + 1));
    }
    __syncthreads();
    if (stop) return;
    if (tid == 0) {
      previous_count += total_count;
      previous_free += total_released;
    }
    __syncthreads();
  }
}

template <int BLOCK>
void kvduo_hand_targets(
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView steps,
    tvm::ffi::TensorView pages,
    tvm::ffi::TensorView targets,
    int64_t hand,
    int64_t minimum,
    double rho) {
  using namespace host;
  auto L = SymbolicSize{"layers"};
  auto R = SymbolicSize{"requests"};
  auto N = SymbolicSize{"objects"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA, kDLROCM>();
  TensorMatcher({L, R, 9}).with_dtype<float>().with_device(device).verify(counts);
  TensorMatcher({L, R}).with_dtype<int64_t>().with_device(device).verify(steps);
  TensorMatcher({L, R}).with_dtype<int32_t>().with_device(device).verify(pages);
  TensorMatcher({N}).with_dtype<int32_t>().with_device(device).verify(targets);
  RuntimeCheck(
      N.unwrap() == L.unwrap() * R.unwrap() && minimum >= 1 && minimum <= 8 && hand >= 0 && hand < N.unwrap(),
      "Invalid Hand geometry");
  LaunchKernel((N.unwrap() + BLOCK - 1) / BLOCK, BLOCK, device.unwrap())(
      kvduo_hand_targets_kernel<BLOCK>,
      static_cast<const float*>(counts.data_ptr()),
      static_cast<const int64_t*>(steps.data_ptr()),
      static_cast<const int32_t*>(pages.data_ptr()),
      static_cast<int32_t*>(targets.data_ptr()),
      L.unwrap(),
      R.unwrap(),
      hand,
      minimum,
      rho);
}

template <int BLOCK>
void kvduo_hand_packet(
    tvm::ffi::TensorView pages,
    tvm::ffi::TensorView targets,
    tvm::ffi::TensorView packet,
    int64_t hand,
    int64_t cursor,
    int64_t needed) {
  using namespace host;
  auto L = SymbolicSize{"layers"};
  auto R = SymbolicSize{"requests"};
  auto N = SymbolicSize{"objects"};
  auto P = SymbolicSize{"packet"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA, kDLROCM>();
  TensorMatcher({L, R}).with_dtype<int32_t>().with_device(device).verify(pages);
  TensorMatcher({N}).with_dtype<int32_t>().with_device(device).verify(targets);
  TensorMatcher({P}).with_dtype<int64_t>().with_device(device).verify(packet);
  RuntimeCheck(
      N.unwrap() == L.unwrap() * R.unwrap() && P.unwrap() >= 2 + 4 * N.unwrap() && hand >= 0 && hand < N.unwrap() &&
          cursor >= 0 && cursor <= N.unwrap() && needed > 0,
      "Invalid Hand packet geometry");
  LaunchKernel(1, BLOCK, device.unwrap())(
      kvduo_hand_packet_kernel<BLOCK>,
      static_cast<const int32_t*>(pages.data_ptr()),
      static_cast<const int32_t*>(targets.data_ptr()),
      static_cast<int64_t*>(packet.data_ptr()),
      L.unwrap(),
      R.unwrap(),
      hand,
      cursor,
      needed);
}

}  // namespace
