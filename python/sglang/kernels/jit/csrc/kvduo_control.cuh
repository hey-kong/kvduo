#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cassert>

namespace {

template <int BLOCK>
__global__ void kvduo_pack_feedback_kernel(
    int32_t* events,
    int32_t* stats,
    const int32_t* positions,
    int32_t* packet,
    int64_t layers,
    int64_t requests,
    int64_t active) {
  const int64_t object = static_cast<int64_t>(blockIdx.x) * BLOCK + threadIdx.x;
  if (object >= layers * requests) return;
  const int64_t layer = object / requests;
  const int64_t request = object % requests;
  const int32_t position = positions[request];
  // The same thread owns the read and reset; a separate gather would race a
  // parallel clearing thread on the source object.
  if (position >= 0 && position < active) packet[layer * active + position] = events[object];
  events[object] = 0;
  stats[object * 2] = 0;
  stats[object * 2 + 1] = 0;
}

template <int BLOCK>
void kvduo_pack_feedback(
    tvm::ffi::TensorView events,
    tvm::ffi::TensorView stats,
    tvm::ffi::TensorView positions,
    tvm::ffi::TensorView packet,
    int64_t active) {
  using namespace host;
  auto L = SymbolicSize{"layers"};
  auto R = SymbolicSize{"requests"};
  auto N = SymbolicSize{"packet_capacity"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA, kDLROCM>();
  TensorMatcher({L, R}).with_dtype<int32_t>().with_device(device).verify(events);
  TensorMatcher({L, R, 2}).with_dtype<int32_t>().with_device(device).verify(stats);
  TensorMatcher({R}).with_dtype<int32_t>().with_device(device).verify(positions);
  TensorMatcher({N}).with_dtype<int32_t>().with_device(device).verify(packet);
  RuntimeCheck(active >= 0 && active <= R.unwrap() && N.unwrap() >= L.unwrap() * active, "Invalid feedback capacity");
  const int64_t count = L.unwrap() * R.unwrap();
  if (count == 0) return;
  LaunchKernel((count + BLOCK - 1) / BLOCK, BLOCK, device.unwrap())(
      kvduo_pack_feedback_kernel<BLOCK>,
      static_cast<int32_t*>(events.data_ptr()),
      static_cast<int32_t*>(stats.data_ptr()),
      static_cast<const int32_t*>(positions.data_ptr()),
      static_cast<int32_t*>(packet.data_ptr()),
      L.unwrap(),
      R.unwrap(),
      active);
}

__global__ void kvduo_publish_buffers_kernel(
    const int64_t* packet, int64_t* pointers, int32_t* capacities, int64_t updates, int64_t owners) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= updates) return;
  const int64_t* row = packet + i * 6;
  const int64_t owner = row[0];
  if (owner < 0 || owner >= owners) {
    __assert_fail("Invalid KVDuo publication owner", __FILE__, __LINE__, __func__);
    return;
  }
  for (int j = 0; j < 4; ++j)
    pointers[owner * 4 + j] = row[j + 2];
  capacities[owner] = row[1];
}

void kvduo_publish_buffers(
    uint64_t packet, uint64_t pointers, uint64_t capacities, int64_t updates, int64_t owners, uint64_t stream) {
  using namespace host;
  RuntimeCheck(updates >= 0 && owners > 0, "Invalid KVDuo publication extent");
  if (!updates) return;
  LaunchKernel((updates + 127) / 128, 128, reinterpret_cast<cudaStream_t>(stream))(
      kvduo_publish_buffers_kernel,
      reinterpret_cast<const int64_t*>(packet),
      reinterpret_cast<int64_t*>(pointers),
      reinterpret_cast<int32_t*>(capacities),
      updates,
      owners);
}

}  // namespace
