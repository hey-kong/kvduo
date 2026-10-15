#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <limits>
#ifdef USE_ROCM
#include <hipcub/hipcub.hpp>
namespace full_cub = hipcub;
#else
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
namespace full_cub = cub;
#endif

namespace {

struct FullTouchMax {
  __device__ int64_t operator()(int64_t left, int64_t right) const {
    return left > right ? left : right;
  }
};

template <int BLOCK_SIZE, typename LogicalT>
__global__ void kvduo_full_page_summary_kernel(
    const LogicalT* logical_pages,
    const int64_t* mapping,
    const int64_t* data_versions,
    const int64_t* host_versions,
    const int64_t* touches,
    const int32_t* protected_pages,
    int64_t* physical_pages,
    int64_t* scores,
    int32_t* valid_pages,
    int64_t page_size,
    int64_t logical_size) {
  using TouchReduce = full_cub::BlockReduce<int64_t, BLOCK_SIZE>;
  using ValidReduce = full_cub::BlockReduce<int, BLOCK_SIZE>;
  __shared__ union {
    typename TouchReduce::TempStorage touch;
    typename ValidReduce::TempStorage valid;
  } storage;
  const int page = blockIdx.x;
  const int tid = threadIdx.x;
  int valid = protected_pages == nullptr || protected_pages[page] == 0;
  int64_t touch = std::numeric_limits<int64_t>::min();
  for (int64_t offset = tid; offset < page_size; offset += BLOCK_SIZE) {
    const int64_t index = page * page_size + offset;
    const int64_t logical = logical_pages[index];
    int64_t physical = 0;
    if (logical >= 0 && logical < logical_size) {
      physical = mapping[logical];
      valid &= physical > 0 && host_versions[logical] == data_versions[logical];
      touch = FullTouchMax{}(touch, touches[logical]);
    } else {
      valid = 0;
    }
    physical_pages[index] = physical;
  }
  const int64_t page_touch = TouchReduce(storage.touch).Reduce(touch, FullTouchMax{});
  __syncthreads();
  const int invalid_threads = ValidReduce(storage.valid).Sum(1 - valid);
  if (tid == 0) {
    scores[page] = page_touch;
    valid_pages[page] = invalid_threads == 0;
  }
}

template <int BLOCK_SIZE>
__global__ void kvduo_full_page_select_kernel(
    const int64_t* order,
    const int32_t* valid,
    const int64_t* owners,
    const int64_t* costs,
    int32_t* workspace,
    int64_t* packet,
    int pages,
    int owner_count,
    int bound,
    int64_t needed,
    bool release_all,
    bool charge_minimum) {
  using NetScan = full_cub::BlockScan<int64_t, BLOCK_SIZE>;
  using IndexScan = full_cub::BlockScan<int, BLOCK_SIZE>;
  __shared__ union {
    typename NetScan::TempStorage net;
    typename IndexScan::TempStorage index;
  } storage;
  __shared__ int cutoff, selected;
  __shared__ int64_t net_carry;
  const int tid = threadIdx.x;
  int* counts = workspace;
  int* first = counts + owner_count;
  if (tid == 0) {
    cutoff = pages;
    selected = 0;
    net_carry = 0;
  }
  if (charge_minimum) {
    for (int owner = tid; owner < owner_count; owner += BLOCK_SIZE) {
      counts[owner] = 0;
      first[owner] = pages;
    }
    __syncthreads();
    for (int page = tid; page < pages; page += BLOCK_SIZE) {
      if (valid[page]) atomicAdd(counts + owners[page], 1);
    }
    __syncthreads();
    for (int rank = tid; rank < pages; rank += BLOCK_SIZE) {
      const int page = order[rank];
      const int owner = owners[page];
      if (valid[page] && counts[owner] > costs[owner]) atomicMin(first + owner, rank);
    }
  }
  __syncthreads();
  for (int base = 0; base < pages; base += BLOCK_SIZE) {
    const int rank = base + tid;
    const int page = rank < pages ? order[rank] : 0;
    const int owner = rank < pages ? owners[page] : 0;
    const bool eligible = rank < pages && valid[page] && (!charge_minimum || counts[owner] > costs[owner]);
    const int64_t delta = eligible ? 1 - (charge_minimum && first[owner] == rank ? costs[owner] : 0) : 0;
    int64_t net, tile_net;
    NetScan(storage.net).InclusiveSum(delta, net, tile_net);
    net += net_carry;
    __syncthreads();
    if (!release_all && eligible && net >= needed) atomicMin(&cutoff, rank);
    __syncthreads();
    const int keep = eligible && rank <= cutoff;
    int offset, tile_selected;
    IndexScan(storage.index).ExclusiveSum(keep, offset, tile_selected);
    offset += selected;
    __syncthreads();
    if (keep && offset < bound) packet[1 + offset] = page;
    if (tid == 0) {
      selected += tile_selected;
      net_carry += tile_net;
    }
    __syncthreads();
    if (cutoff < pages) break;
  }
  if (tid == 0) packet[0] = selected;
}

template <int BLOCK_SIZE>
void kvduo_full_page_select(
    tvm::ffi::TensorView order,
    tvm::ffi::TensorView valid,
    tvm::ffi::TensorView owners,
    tvm::ffi::TensorView costs,
    tvm::ffi::TensorView workspace,
    tvm::ffi::TensorView packet,
    int64_t needed,
    bool release_all,
    bool charge_minimum) {
  using namespace host;
  auto pages = SymbolicSize{"pages"};
  auto owner_count = SymbolicSize{"owners"};
  auto words = SymbolicSize{"words"};
  auto packet_size = SymbolicSize{"packet"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA, kDLROCM>();
  TensorMatcher({pages}).with_dtype<int64_t>().with_device(device).verify(order).verify(owners);
  TensorMatcher({pages}).with_dtype<int32_t>().with_device(device).verify(valid);
  TensorMatcher({owner_count}).with_dtype<int64_t>().with_device(device).verify(costs);
  TensorMatcher({words}).with_dtype<int32_t>().with_device(device).verify(workspace);
  TensorMatcher({packet_size}).with_dtype<int64_t>().with_device(device).verify(packet);
  RuntimeCheck(pages.unwrap() > 0 && pages.unwrap() <= INT32_MAX && owner_count.unwrap() > 0, "Invalid FULL plan size");
  RuntimeCheck(
      words.unwrap() >= 2 * owner_count.unwrap() && packet_size.unwrap() > 1, "Insufficient FULL plan workspace");
  RuntimeCheck(needed > 0, "FULL plan requires a positive shortage");
  LaunchKernel(1, BLOCK_SIZE, device.unwrap())(
      kvduo_full_page_select_kernel<BLOCK_SIZE>,
      static_cast<const int64_t*>(order.data_ptr()),
      static_cast<const int32_t*>(valid.data_ptr()),
      static_cast<const int64_t*>(owners.data_ptr()),
      static_cast<const int64_t*>(costs.data_ptr()),
      static_cast<int32_t*>(workspace.data_ptr()),
      static_cast<int64_t*>(packet.data_ptr()),
      static_cast<int>(pages.unwrap()),
      static_cast<int>(owner_count.unwrap()),
      static_cast<int>(packet_size.unwrap() - 1),
      needed,
      release_all,
      charge_minimum);
}

template <int BLOCK_SIZE>
__global__ void kvduo_full_request_select_kernel(
    const int32_t* valid,
    const int64_t* owners,
    const int64_t* costs,
    const int64_t* admission,
    const int32_t* mixed,
    int32_t* workspace,
    int64_t* packet,
    int pages,
    int owner_count,
    int bound,
    int64_t needed,
    bool release_all) {
  using Scan = full_cub::BlockScan<int, BLOCK_SIZE>;
  __shared__ typename Scan::TempStorage storage;
  __shared__ int selected_pages;
  const int tid = threadIdx.x;
  int* counts = workspace;
  int* chosen = counts + owner_count;
  for (int owner = tid; owner < owner_count; owner += BLOCK_SIZE) {
    counts[owner] = chosen[owner] = 0;
  }
  if (tid == 0) selected_pages = 0;
  __syncthreads();
  for (int page = tid; page < pages; page += BLOCK_SIZE) {
    if (valid[page]) atomicAdd(counts + owners[page], 1);
  }
  __syncthreads();
  if (tid == 0) {
    if (release_all) {
      for (int owner = 0; owner < owner_count; ++owner)
        chosen[owner] = counts[owner] > costs[owner];
    } else {
      int64_t net = 0;
      while (net < needed) {
        int best = -1;
        int64_t best_gain = 0;
        for (int owner = 0; owner < owner_count; ++owner) {
          const int64_t gain = counts[owner] - costs[owner];
          if (chosen[owner] || gain <= 0) continue;
          if (best < 0 || mixed[owner] > mixed[best] ||
              (mixed[owner] == mixed[best] &&
               (gain > best_gain || (gain == best_gain && admission[owner] < admission[best])))) {
            best = owner;
            best_gain = gain;
          }
        }
        if (best < 0) break;
        chosen[best] = 1;
        net += best_gain;
      }
    }
  }
  __syncthreads();
  for (int base = 0; base < pages; base += BLOCK_SIZE) {
    const int page = base + tid;
    const int keep = page < pages && valid[page] && chosen[owners[page]];
    int offset, total;
    Scan(storage).ExclusiveSum(keep, offset, total);
    offset += selected_pages;
    __syncthreads();
    if (keep && offset < bound) packet[1 + offset] = page;
    if (tid == 0) selected_pages += total;
    __syncthreads();
  }
  if (tid == 0) packet[0] = selected_pages;
}

template <int BLOCK_SIZE>
void kvduo_full_request_select(
    tvm::ffi::TensorView valid,
    tvm::ffi::TensorView owners,
    tvm::ffi::TensorView costs,
    tvm::ffi::TensorView admission,
    tvm::ffi::TensorView mixed,
    tvm::ffi::TensorView workspace,
    tvm::ffi::TensorView packet,
    int64_t needed,
    bool release_all) {
  using namespace host;
  auto pages = SymbolicSize{"pages"};
  auto owner_count = SymbolicSize{"owners"};
  auto words = SymbolicSize{"words"};
  auto packet_size = SymbolicSize{"packet"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA, kDLROCM>();
  TensorMatcher({pages}).with_dtype<int64_t>().with_device(device).verify(owners);
  TensorMatcher({pages}).with_dtype<int32_t>().with_device(device).verify(valid);
  TensorMatcher({owner_count}).with_dtype<int64_t>().with_device(device).verify(costs).verify(admission);
  TensorMatcher({owner_count}).with_dtype<int32_t>().with_device(device).verify(mixed);
  TensorMatcher({words}).with_dtype<int32_t>().with_device(device).verify(workspace);
  TensorMatcher({packet_size}).with_dtype<int64_t>().with_device(device).verify(packet);
  RuntimeCheck(pages.unwrap() > 0 && pages.unwrap() <= INT32_MAX && owner_count.unwrap() > 0, "Invalid FULL plan size");
  RuntimeCheck(
      words.unwrap() >= 2 * owner_count.unwrap() && packet_size.unwrap() > 1, "Insufficient FULL plan workspace");
  RuntimeCheck(needed > 0, "FULL plan requires a positive shortage");
  LaunchKernel(1, BLOCK_SIZE, device.unwrap())(
      kvduo_full_request_select_kernel<BLOCK_SIZE>,
      static_cast<const int32_t*>(valid.data_ptr()),
      static_cast<const int64_t*>(owners.data_ptr()),
      static_cast<const int64_t*>(costs.data_ptr()),
      static_cast<const int64_t*>(admission.data_ptr()),
      static_cast<const int32_t*>(mixed.data_ptr()),
      static_cast<int32_t*>(workspace.data_ptr()),
      static_cast<int64_t*>(packet.data_ptr()),
      static_cast<int>(pages.unwrap()),
      static_cast<int>(owner_count.unwrap()),
      static_cast<int>(packet_size.unwrap() - 1),
      needed,
      release_all);
}

template <int BLOCK_SIZE, typename LogicalT>
void kvduo_full_page_summary(
    tvm::ffi::TensorView logical_pages,
    uint64_t mapping,
    uint64_t data_versions,
    uint64_t host_versions,
    uint64_t touches,
    uint64_t protected_pages,
    uint64_t physical_pages,
    uint64_t scores,
    uint64_t valid_pages,
    int64_t logical_size) {
  using namespace host;
  auto pages = SymbolicSize{"pages"};
  auto page_size = SymbolicSize{"page_size"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA, kDLROCM>();
  TensorMatcher({pages, page_size}).with_dtype<LogicalT>().with_device(device).verify(logical_pages);
  RuntimeCheck(page_size.unwrap() > 0 && logical_size > 0, "Invalid FULL page geometry");
  if (pages.unwrap() == 0) return;
  RuntimeCheck(
      mapping && data_versions && host_versions && touches && physical_pages && scores && valid_pages,
      "FULL summary pointers must be non-null");
  LaunchKernel(pages.unwrap(), BLOCK_SIZE, device.unwrap())(
      kvduo_full_page_summary_kernel<BLOCK_SIZE, LogicalT>,
      static_cast<const LogicalT*>(logical_pages.data_ptr()),
      reinterpret_cast<const int64_t*>(mapping),
      reinterpret_cast<const int64_t*>(data_versions),
      reinterpret_cast<const int64_t*>(host_versions),
      reinterpret_cast<const int64_t*>(touches),
      reinterpret_cast<const int32_t*>(protected_pages),
      reinterpret_cast<int64_t*>(physical_pages),
      reinterpret_cast<int64_t*>(scores),
      reinterpret_cast<int32_t*>(valid_pages),
      page_size.unwrap(),
      logical_size);
}

}  // namespace
