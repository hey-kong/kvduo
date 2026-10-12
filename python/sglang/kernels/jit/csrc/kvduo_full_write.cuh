#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace {

template <int BLOCK_SIZE, typename LogicalT, typename RequestT, typename PositionT>
__global__ void kvduo_publish_full_write_kernel(
    const LogicalT* logical_locs,
    int64_t* generation,
    int64_t* data_versions,
    int64_t* host_versions,
    int64_t* touches,
    int64_t* clock,
    int64_t* lookup,
    const RequestT* req_indices,
    const PositionT* positions,
    int64_t items,
    int64_t logical_size,
    int64_t logical_stride,
    int64_t lookup_stride,
    int64_t requests,
    int64_t context,
    int64_t request_stride,
    int64_t position_stride) {
  __shared__ int64_t stamp;
  if (threadIdx.x == 0) {
    stamp = clock[0] + 1;
    clock[0] = stamp;
  }
  __syncthreads();
  for (int64_t i = threadIdx.x; i < items; i += BLOCK_SIZE) {
    const int64_t logical = logical_locs[i * logical_stride];
    assert(logical >= 0 && logical < logical_size);
    generation[logical] += 1;
    data_versions[logical] = 1;
    host_versions[logical] = -1;
    touches[logical] = stamp;
    if (lookup != nullptr) {
      const int64_t request = req_indices[i * request_stride];
      const int64_t position = positions[i * position_stride];
      assert(request >= 0 && request < requests && position >= 0 && position < context);
      lookup[request * lookup_stride + position] = logical;
    }
  }
}

template <int BLOCK_SIZE>
void kvduo_publish_full_write(
    uint64_t logical_locs,
    uint64_t generation,
    uint64_t data_versions,
    uint64_t host_versions,
    uint64_t touches,
    uint64_t clock,
    uint64_t lookup,
    uint64_t req_indices,
    uint64_t positions,
    uint64_t stream,
    int64_t items,
    int64_t logical_size,
    int64_t logical_stride,
    int64_t lookup_stride,
    int64_t requests,
    int64_t context,
    int64_t request_stride,
    int64_t position_stride,
    bool logical_i64,
    bool request_i64,
    bool position_i64) {
  using namespace host;
  auto launch = [&](const auto* logical_ptr, const auto* request_ptr, const auto* position_ptr) {
    using LogicalT = std::remove_cv_t<std::remove_pointer_t<decltype(logical_ptr)>>;
    using RequestT = std::remove_cv_t<std::remove_pointer_t<decltype(request_ptr)>>;
    using PositionT = std::remove_cv_t<std::remove_pointer_t<decltype(position_ptr)>>;
    LaunchKernel(1, BLOCK_SIZE, reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream)))(
        kvduo_publish_full_write_kernel<BLOCK_SIZE, LogicalT, RequestT, PositionT>,
        logical_ptr,
        reinterpret_cast<int64_t*>(generation),
        reinterpret_cast<int64_t*>(data_versions),
        reinterpret_cast<int64_t*>(host_versions),
        reinterpret_cast<int64_t*>(touches),
        reinterpret_cast<int64_t*>(clock),
        reinterpret_cast<int64_t*>(lookup),
        request_ptr,
        position_ptr,
        items,
        logical_size,
        logical_stride,
        lookup_stride,
        requests,
        context,
        request_stride,
        position_stride);
  };
  auto select_position = [&](const auto* logical_ptr, const auto* request_ptr) {
    if (position_i64) {
      launch(logical_ptr, request_ptr, reinterpret_cast<const int64_t*>(positions));
    } else {
      launch(logical_ptr, request_ptr, reinterpret_cast<const int32_t*>(positions));
    }
  };
  auto select_request = [&](const auto* logical_ptr) {
    if (request_i64) {
      select_position(logical_ptr, reinterpret_cast<const int64_t*>(req_indices));
    } else {
      select_position(logical_ptr, reinterpret_cast<const int32_t*>(req_indices));
    }
  };
  if (logical_i64) {
    select_request(reinterpret_cast<const int64_t*>(logical_locs));
  } else {
    select_request(reinterpret_cast<const int32_t*>(logical_locs));
  }
}

}  // namespace
