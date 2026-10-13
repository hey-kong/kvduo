#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

namespace {

__global__ void kvduo_begin_decode_kernel(
    int64_t* clock, int64_t* base, int32_t* observe, int32_t* requests, int64_t batch_size, int64_t layers) {
  base[0] = clock[0];
  clock[0] += layers;
  observe[0] = 1;
  requests[0] = batch_size;
}

void kvduo_begin_decode(
    tvm::ffi::TensorView clock,
    tvm::ffi::TensorView base,
    tvm::ffi::TensorView observe,
    tvm::ffi::TensorView requests,
    int64_t batch_size,
    int64_t layers) {
  using namespace host;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA, kDLROCM>();
  TensorMatcher({1}).with_dtype<int64_t>().with_device(device).verify(clock).verify(base);
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(observe).verify(requests);
  RuntimeCheck(batch_size >= 0 && layers >= 0, "Invalid decode controls");
  LaunchKernel(1, 1, device.unwrap())(
      kvduo_begin_decode_kernel,
      static_cast<int64_t*>(clock.data_ptr()),
      static_cast<int64_t*>(base.data_ptr()),
      static_cast<int32_t*>(observe.data_ptr()),
      static_cast<int32_t*>(requests.data_ptr()),
      batch_size,
      layers);
}

}  // namespace
