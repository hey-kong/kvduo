#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

namespace {

template <int BLOCK_SIZE, typename SlotT>
__global__ void kvduo_publish_hot_growth_kernel(
    const int64_t* pages,
    int32_t* tokens,
    int32_t* locations,
    SlotT* lru,
    const SlotT* previous_lru,
    int32_t* capacity,
    float* capacity_ema,
    int64_t* capacity_steps,
    int64_t current,
    int64_t target,
    int64_t page_size,
    int64_t K,
    int64_t layer_stride,
    int64_t page_stride) {
  const int tid = threadIdx.x;
  const int64_t grow = target - current;
  for (int64_t offset = tid; offset < grow; offset += BLOCK_SIZE) {
    const int64_t in_page = offset % K;
    locations[current + offset] =
        pages[(offset / K) * page_stride] + (in_page / page_size) * layer_stride + in_page % page_size;
    tokens[current + offset] = -1;
  }
  for (int64_t rank = tid; rank < target; rank += BLOCK_SIZE) {
    lru[rank] = rank < grow ? static_cast<SlotT>(current + rank) : previous_lru[rank - grow];
  }
  if (current == 0) {
    if (tid < 9) capacity_ema[tid] = 0;
    if (tid == 0) capacity_steps[0] = 0;
  } else if (tid >= current / K && tid < target / K) {
    capacity_ema[tid] = capacity_ema[current / K - 1];
  }
  __syncthreads();
  if (tid == 0) capacity[0] = target;
}

template <int BLOCK_SIZE>
void kvduo_publish_hot_growth(
    tvm::ffi::TensorView pages,
    tvm::ffi::TensorView tokens,
    tvm::ffi::TensorView locations,
    tvm::ffi::TensorView lru,
    tvm::ffi::TensorView previous_lru,
    tvm::ffi::TensorView capacity,
    tvm::ffi::TensorView capacity_ema,
    tvm::ffi::TensorView capacity_steps,
    int64_t current,
    int64_t target,
    int64_t page_size,
    int64_t storage_layers,
    int64_t layer_stride) {
  using namespace host;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA, kDLROCM>();
  const int64_t K = page_size * storage_layers;
  RuntimeCheck(page_size > 0 && storage_layers > 0 && current >= 0 && target > current, "Invalid HOT growth geometry");
  RuntimeCheck(current % K == 0 && target % K == 0 && target <= 8 * K, "HOT growth must use whole pages");
  auto padded = SymbolicSize{"padded_capacity"};
  auto max_capacity = SymbolicSize{"max_capacity"};
  auto previous = SymbolicSize{"previous_capacity"};
  auto page_stride = SymbolicSize{"page_stride"};
  TensorMatcher({(target - current) / K})
      .with_strides({page_stride})
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(pages);
  TensorMatcher({padded}).with_dtype<int32_t>().with_device(device).verify(tokens).verify(locations);
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(capacity);
  TensorMatcher({9}).with_dtype<float>().with_device(device).verify(capacity_ema);
  TensorMatcher({1}).with_dtype<int64_t>().with_device(device).verify(capacity_steps);
  auto launch = [&]<typename SlotT>() {
    TensorMatcher({max_capacity}).with_dtype<SlotT>().with_device(device).verify(lru);
    TensorMatcher({previous}).with_dtype<SlotT>().with_device(device).verify(previous_lru);
    RuntimeCheck(
        target <= max_capacity.unwrap() && target <= padded.unwrap() && current <= previous.unwrap(),
        "HOT growth exceeds metadata capacity");
    RuntimeCheck(sizeof(SlotT) != 2 || target <= INT16_MAX, "HOT slots exceed int16 range");
    LaunchKernel(1, BLOCK_SIZE, device.unwrap())(
        kvduo_publish_hot_growth_kernel<BLOCK_SIZE, SlotT>,
        static_cast<const int64_t*>(pages.data_ptr()),
        static_cast<int32_t*>(tokens.data_ptr()),
        static_cast<int32_t*>(locations.data_ptr()),
        static_cast<SlotT*>(lru.data_ptr()),
        static_cast<const SlotT*>(previous_lru.data_ptr()),
        static_cast<int32_t*>(capacity.data_ptr()),
        static_cast<float*>(capacity_ema.data_ptr()),
        static_cast<int64_t*>(capacity_steps.data_ptr()),
        current,
        target,
        page_size,
        K,
        layer_stride,
        page_stride.unwrap());
  };
  if (lru.dtype().code == kDLInt && lru.dtype().bits == 16) {
    launch.template operator()<int16_t>();
  } else {
    launch.template operator()<int32_t>();
  }
}

}  // namespace
